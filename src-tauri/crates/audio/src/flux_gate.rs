//! Sub-band spectral-flux gate (Phase 1, Bullet 1.1).
//!
//! ## Why `SubbandFluxGate` (an approximation) and not `rustfft`
//!
//! The canonical spectral-flux measure is computed from an FFT magnitude
//! spectrum: `flux(t) = Σ_k max(|X(t,k)| - |X(t-1,k)|, 0)`. A full FFT gives
//! fine frequency resolution but (1) requires the `rustfft` crate, which is
//! **not** on the approved-dependency list, (2) must operate on windowed blocks
//! and allocates scratch buffers per transform, and (3) is overkill for a *gate*
//! that only needs a coarse yes/no "did the spectral shape lurch like an
//! instrument onset?" decision.
//!
//! This module instead approximates the spectrum with a small bank of RBJ
//! constant-0 dB band-pass biquad filters (6 log-spaced bands across the 16 kHz
//! speech range). Per analysis frame we accumulate each band's output energy,
//! normalise the bands into a spectral *distribution* (so the measure is
//! volume-independent — a loud steady voice must not trip the gate), and compute
//! the half-wave-rectified flux of that distribution against the previous frame.
//! A sharp broadband attack (worship band, clapping bleeding in from a stage mic)
//! shifts the distribution abruptly → high flux. Rolling speech formants shift it
//! gently → low flux. The filterbank is allocation-free in the hot path and cheap
//! enough (`O(samples × bands)`) for the realtime-adjacent fan-out thread.
//!
//! The action taken on a gated frame (drop vs. attenuate) is the *caller's*
//! responsibility — this component only reports `flux` and a `gated` verdict so
//! it stays composable in the Phase 1 gate chain. Per the companion's Example A,
//! a flux-gated (musical) frame is intended to be suppressed (not forwarded).

use std::f32::consts::PI;

/// Center frequencies (Hz) of the sub-band analysis filterbank, log-spaced
/// across the 16 kHz speech band. Bands above Nyquist for the configured sample
/// rate are skipped at construction time.
const BAND_CENTERS_HZ: [f32; 6] = [250.0, 500.0, 1000.0, 2000.0, 4000.0, 6000.0];

/// Quality factor for each band-pass filter. `Q = 1.0` gives moderately broad,
/// overlapping bands — appropriate for a coarse spectral-shape estimate.
const DEFAULT_BAND_Q: f32 = 1.0;

/// Default normalized-flux threshold above which a frame is judged "musical".
/// Flux is in `0.0..≈1.0` (fraction of spectral mass that shifted bands), so
/// `0.5` means roughly half the spectral energy abruptly changed bands.
const DEFAULT_FLUX_THRESHOLD: f32 = 0.5;

/// Total band energy below which the frame is treated as silence: there is no
/// spectral onset to measure, so flux is reported as `0.0` and the flux
/// continuity is broken (the next audible frame starts fresh).
const SILENCE_ENERGY_EPSILON: f32 = 1e-6;

/// Scale factor to map `i16` PCM into the `-1.0..=1.0` float domain.
const I16_SCALE: f32 = 32_768.0;

/// A single direct-form-I biquad section.
#[derive(Clone)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    /// RBJ "constant 0 dB peak gain" band-pass at `center_hz` for sample rate
    /// `sample_rate` and quality factor `q`. Coefficients are pre-normalised by
    /// `a0` so `process` needs no per-sample division.
    fn bandpass(sample_rate: f32, center_hz: f32, q: f32) -> Self {
        let w0 = 2.0 * PI * center_hz / sample_rate;
        let (sin_w0, cos_w0) = w0.sin_cos();
        let alpha = sin_w0 / (2.0 * q);
        let a0 = 1.0 + alpha;

        Self {
            b0: alpha / a0,
            b1: 0.0,
            b2: -alpha / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x0: f32) -> f32 {
        let y0 = self.b0 * x0 + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x0;
        self.y2 = self.y1;
        self.y1 = y0;
        y0
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Configuration for [`SubbandFluxGate`].
#[derive(Debug, Clone)]
pub struct SubbandFluxConfig {
    /// Sample rate of the incoming PCM (Hz). The live pipeline is 16 kHz.
    pub sample_rate: u32,
    /// Quality factor for each band-pass filter.
    pub band_q: f32,
    /// Normalized-flux value at/above which [`FluxResult::gated`] is `true`.
    pub flux_threshold: f32,
}

impl Default for SubbandFluxConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            band_q: DEFAULT_BAND_Q,
            flux_threshold: DEFAULT_FLUX_THRESHOLD,
        }
    }
}

/// Result of analysing one frame through the [`SubbandFluxGate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluxResult {
    /// Half-wave-rectified flux of the normalized spectral distribution versus
    /// the previous frame, in `0.0..≈1.0`. Higher = a sharper spectral lurch.
    pub flux: f32,
    /// `true` when `flux >= flux_threshold` — i.e. the frame looks like an
    /// instrument onset / stage bleed rather than rolling speech.
    pub gated: bool,
}

/// Coarse spectral-flux gate built from a band-pass filterbank.
///
/// Feed it consecutive analysis frames via [`SubbandFluxGate::process`]; each
/// call compares the current frame's spectral shape to the previous one. State
/// (filter histories, previous distribution, scratch energy buffer) is owned and
/// reused — `process` performs no heap allocation.
pub struct SubbandFluxGate {
    bands: Vec<Biquad>,
    /// Scratch per-band energy accumulator, reused every `process` call.
    energy: Vec<f32>,
    /// Previous frame's normalized spectral distribution.
    prev_dist: Vec<f32>,
    /// Whether `prev_dist` holds a valid (audible) previous frame.
    have_prev: bool,
    threshold: f32,
}

impl SubbandFluxGate {
    /// Build a gate for the given configuration. Bands whose center frequency is
    /// at or above Nyquist for `config.sample_rate` are dropped.
    pub fn new(config: SubbandFluxConfig) -> Self {
        let fs = config.sample_rate as f32;
        let nyquist = fs / 2.0;
        let bands: Vec<Biquad> = BAND_CENTERS_HZ
            .iter()
            .copied()
            .filter(|&f| f < nyquist)
            .map(|f| Biquad::bandpass(fs, f, config.band_q))
            .collect();
        let n = bands.len();
        Self {
            bands,
            energy: vec![0.0; n],
            prev_dist: vec![0.0; n],
            have_prev: false,
            threshold: config.flux_threshold,
        }
    }

    /// Analyse one frame of PCM samples and return its flux and gate verdict.
    ///
    /// Consecutive calls form the flux time-series; the very first call (and the
    /// first audible call after a silent gap) reports `flux = 0.0`.
    pub fn process(&mut self, samples: &[i16]) -> FluxResult {
        for e in self.energy.iter_mut() {
            *e = 0.0;
        }

        // Parallel filterbank: each input sample is run through every band-pass
        // section and its squared output accumulated into that band's energy.
        for &s in samples {
            let x = s as f32 / I16_SCALE;
            for b in 0..self.bands.len() {
                let y = self.bands[b].process(x);
                self.energy[b] += y * y;
            }
        }

        let total: f32 = self.energy.iter().sum();
        if total < SILENCE_ENERGY_EPSILON {
            // Silence carries no spectral onset; break flux continuity so the
            // next audible frame is not compared across the gap.
            self.have_prev = false;
            return FluxResult {
                flux: 0.0,
                gated: false,
            };
        }

        let mut flux = 0.0;
        if self.have_prev {
            for b in 0..self.energy.len() {
                let p = self.energy[b] / total;
                let increase = p - self.prev_dist[b];
                if increase > 0.0 {
                    flux += increase;
                }
            }
        }

        for b in 0..self.energy.len() {
            self.prev_dist[b] = self.energy[b] / total;
        }
        self.have_prev = true;

        FluxResult {
            flux,
            gated: flux >= self.threshold,
        }
    }

    /// Clear all filter and flux history (e.g. when (re)starting capture).
    pub fn reset(&mut self) {
        for b in self.bands.iter_mut() {
            b.reset();
        }
        for e in self.energy.iter_mut() {
            *e = 0.0;
        }
        for p in self.prev_dist.iter_mut() {
            *p = 0.0;
        }
        self.have_prev = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate an `n`-sample sine window at `freq` Hz, amplitude 0.5 full-scale.
    fn sine(freq: f32, sample_rate: f32, n: usize) -> Vec<i16> {
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate;
                let v = (2.0 * PI * freq * t).sin() * 0.5;
                (v * 32_767.0) as i16
            })
            .collect()
    }

    fn gate() -> SubbandFluxGate {
        SubbandFluxGate::new(SubbandFluxConfig::default())
    }

    #[test]
    fn first_frame_reports_zero_flux_and_passes() {
        let mut g = gate();
        let r = g.process(&sine(1000.0, 16_000.0, 1024));
        assert_eq!(r.flux, 0.0);
        assert!(!r.gated);
    }

    #[test]
    fn silence_passes_with_zero_flux() {
        let mut g = gate();
        let r = g.process(&vec![0i16; 1024]);
        assert_eq!(r.flux, 0.0);
        assert!(!r.gated);
    }

    #[test]
    fn empty_frame_is_treated_as_silence() {
        let mut g = gate();
        let r = g.process(&[]);
        assert_eq!(r.flux, 0.0);
        assert!(!r.gated);
    }

    #[test]
    fn steady_tone_has_low_flux_and_passes() {
        let mut g = gate();
        // Warm up + establish previous distribution, then measure a steady frame.
        g.process(&sine(1000.0, 16_000.0, 1024));
        g.process(&sine(1000.0, 16_000.0, 1024));
        let r = g.process(&sine(1000.0, 16_000.0, 1024));
        assert!(r.flux < 0.5, "steady tone flux too high: {}", r.flux);
        assert!(!r.gated);
    }

    #[test]
    fn abrupt_spectral_shift_is_gated() {
        let mut g = gate();
        // Establish a low-band (250 Hz) spectral shape.
        g.process(&sine(250.0, 16_000.0, 1024));
        g.process(&sine(250.0, 16_000.0, 1024));
        // Lurch to the top band (6 kHz): spectral mass jumps bands → high flux.
        let r = g.process(&sine(6000.0, 16_000.0, 1024));
        assert!(r.flux >= 0.5, "shift flux too low: {}", r.flux);
        assert!(r.gated);
    }

    #[test]
    fn reset_clears_flux_history() {
        let mut g = gate();
        g.process(&sine(250.0, 16_000.0, 1024));
        g.reset();
        // After reset there is no previous frame, so flux is zero again.
        let r = g.process(&sine(6000.0, 16_000.0, 1024));
        assert_eq!(r.flux, 0.0);
        assert!(!r.gated);
    }
}
