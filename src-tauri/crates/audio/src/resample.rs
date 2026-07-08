//! Anti-aliased sample-rate conversion from the capture device rate to 16 kHz.
//!
//! The previous resampler ([`capture`](crate::capture)'s old `resample`) was linear
//! interpolation with no low-pass filter, so downsampling (e.g. 48 kHz → 16 kHz) folded
//! everything above the 8 kHz output Nyquist back into the speech band — broadband
//! aliasing distortion and a direct WER tax, worst on fricatives and accented speech.
//!
//! This replaces it with rubato's band-limited sinc resampler, which low-passes below the
//! output Nyquist before decimating (rubato scales the sinc cutoff by the ratio when
//! downsampling, which is exactly the anti-aliasing we need).
//!
//! The resampler is **stateful** — it carries sinc filter history across calls, so it must
//! persist for the life of one capture stream and be fed every device callback. Feeding
//! independent per-callback resamplers would inject discontinuities at every block edge.

use rubato::{
    calculate_cutoff, Resampler as _, SincFixedIn, SincInterpolationParameters,
    SincInterpolationType, WindowFunction,
};

/// Fixed input frames per rubato `process` call. Device callbacks are variable-size, so we
/// buffer up to this and process in fixed chunks. Buffering latency ≈ CHUNK/source_rate
/// (~11 ms at 48 kHz) — small relative to the ~0.5–1 s ASR lookahead downstream.
const CHUNK: usize = 512;
/// Sinc filter length. 128 balances stopband attenuation against the output delay it adds
/// (~SINC_LEN/2 output frames, a few ms at 16 kHz). Higher = sharper but laggier.
const SINC_LEN: usize = 128;

/// Streaming resampler: device-rate mono `f32` → 16 kHz mono `f32`. A pass-through when the
/// device already runs at the target rate (no filter, no delay).
pub struct Resampler {
    /// `None` ⇒ pass-through (source == target, or construction failed).
    inner: Option<SincFixedIn<f32>>,
    /// Pending input samples not yet consumed into a full `CHUNK`.
    in_buf: Vec<f32>,
    /// Reused single-channel input container handed to `process` (avoids per-call alloc).
    scratch: Vec<Vec<f32>>,
}

impl Resampler {
    /// Build a resampler for `source_rate` → `target_rate` (both in Hz).
    pub fn new(source_rate: u32, target_rate: u32) -> Self {
        if source_rate == 0 || source_rate == target_rate {
            return Self::passthrough();
        }
        let window = WindowFunction::Blackman2;
        let params = SincInterpolationParameters {
            sinc_len: SINC_LEN,
            // Window-matched cutoff (fraction of Nyquist) that maximizes stopband
            // rejection for this sinc_len/window; rubato scales it by the ratio to
            // band-limit below the OUTPUT Nyquist when downsampling.
            f_cutoff: calculate_cutoff::<f32>(SINC_LEN, window),
            oversampling_factor: 256,
            interpolation: SincInterpolationType::Quadratic,
            window,
        };
        let ratio = target_rate as f64 / source_rate as f64;
        // max_resample_ratio_relative = 1.0: the ratio is fixed (we never retune it).
        match SincFixedIn::<f32>::new(ratio, 1.0, params, CHUNK, 1) {
            Ok(inner) => Self {
                inner: Some(inner),
                in_buf: Vec::with_capacity(CHUNK * 2),
                scratch: vec![Vec::with_capacity(CHUNK)],
            },
            Err(e) => {
                // Should not happen for sane rates; fail safe to pass-through rather than
                // kill capture. The stream would then be at the wrong rate, so log loudly.
                log::error!(
                    "[AUDIO] resampler init failed ({source_rate}->{target_rate}): {e}; \
                     falling back to pass-through (audio will be at the WRONG rate)"
                );
                Self::passthrough()
            }
        }
    }

    fn passthrough() -> Self {
        Self {
            inner: None,
            in_buf: Vec::new(),
            scratch: Vec::new(),
        }
    }

    /// Feed device-rate mono `f32`; returns whatever target-rate output is ready. May be
    /// empty while the resampler buffers toward the next full `CHUNK` — that is normal, the
    /// caller simply forwards nothing that callback.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let Some(rs) = self.inner.as_mut() else {
            return input.to_vec();
        };
        self.in_buf.extend_from_slice(input);
        let mut out = Vec::new();
        while self.in_buf.len() >= CHUNK {
            self.scratch[0].clear();
            self.scratch[0].extend(self.in_buf.drain(..CHUNK));
            match rs.process(&self.scratch, None) {
                Ok(mut res) => {
                    if !res.is_empty() {
                        out.append(&mut res[0]);
                    }
                }
                Err(e) => {
                    // Drop this chunk but keep the stream alive — never take down capture.
                    log::error!("[AUDIO] resample failed: {e}");
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn passthrough_when_rates_equal() {
        let mut r = Resampler::new(16_000, 16_000);
        let input: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = r.process(&input);
        assert_eq!(out.len(), input.len());
        assert_eq!(out, input);
    }

    #[test]
    fn downsamples_48k_to_16k() {
        let mut r = Resampler::new(48_000, 16_000);
        // 1 s of a 1 kHz tone at 48 kHz — well inside the 8 kHz output passband.
        let n = 48_000usize;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / 48_000.0).sin() * 0.5)
            .collect();

        // Feed in device-callback-sized chunks (variable, not aligned to CHUNK).
        let mut out = Vec::new();
        for chunk in input.chunks(480) {
            out.extend(r.process(chunk));
        }

        // ~1/3 the input frames, minus the tail still buffered (< CHUNK). Not exact
        // because of buffering + the resampler's intrinsic output delay.
        assert!(
            out.len() > 15_000 && out.len() < 16_100,
            "unexpected output length {}",
            out.len()
        );
        assert!(out.iter().all(|s| s.is_finite()), "non-finite samples");
        // The tone must survive (not be low-passed/aliased into silence).
        let rms = (out.iter().map(|s| s * s).sum::<f32>() / out.len() as f32).sqrt();
        assert!(rms > 0.2, "signal lost, rms {rms}");
    }

    #[test]
    fn rejects_supra_nyquist_tone_instead_of_aliasing() {
        // 12 kHz at 48 kHz is above the 8 kHz output Nyquist. A correct anti-aliased
        // resampler filters it out (low output energy). The OLD linear resampler would
        // alias it to |12k - 16k| = 4 kHz — a strong spurious in-band tone. This test is
        // what distinguishes the fix from the bug.
        let mut r = Resampler::new(48_000, 16_000);
        let n = 48_000usize;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 12_000.0 * i as f32 / 48_000.0).sin() * 0.5)
            .collect();

        let mut out = Vec::new();
        for chunk in input.chunks(480) {
            out.extend(r.process(chunk));
        }

        let rms = (out.iter().map(|s| s * s).sum::<f32>() / out.len() as f32).sqrt();
        // Input RMS ≈ 0.354; a stopband tone must come out heavily attenuated.
        assert!(rms < 0.1, "supra-Nyquist tone not rejected (aliasing?), rms {rms}");
    }

    #[test]
    fn buffers_until_a_full_chunk() {
        let mut r = Resampler::new(48_000, 16_000);
        // Fewer than CHUNK input frames → nothing ready yet.
        assert!(r.process(&vec![0.0; 100]).is_empty());
    }
}
