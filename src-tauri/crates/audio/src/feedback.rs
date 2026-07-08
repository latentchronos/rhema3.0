//! PA feedback-loop detector (Phase 1, Bullet 1.4).
//!
//! Public-address feedback is a rapidly growing, near-pure sinusoid sitting on a
//! room resonance, almost always in the 500 Hz–8 kHz range. It is characterised
//! by a single frequency dominating the spectrum and *staying there* — speech
//! energy wanders frame to frame, feedback does not. This detector flags a frame
//! when a single in-band tone holds >60 % of the signal energy for ≥5 consecutive
//! frames (~100 ms at 20 ms/frame), then **zeroes the frame in place** (the
//! caller still forwards it, so the STT stream stays continuous) and emits a
//! single `warn!` on the rising edge. It raises no Tauri event — this is an
//! internal audio-quality signal, not user-facing.
//!
//! ## Why a Goertzel bank, not an FFT
//! Per the approved Decision #4 we add no FFT dependency. Goertzel is the
//! textbook FFT-free way to measure energy at specific frequencies (DTMF
//! decoders use it) and is exact, allocation-free, and `O(samples)` per probe.
//! We run a dense bank of probes spaced at the analysis window's natural bin
//! width (50 Hz for a 320-sample / 20 ms window at 16 kHz) across the danger
//! band, so any in-band tone lands within half a bin of a probe.
//!
//! ## Dominance measure (the part that resists false positives)
//! For each frame we find the dominant probe and sum it with its two neighbours
//! (a tone half a bin off a probe splits into the adjacent probe). Dominance is
//!
//! ```text
//! dominance = 2 · cluster_power / (N · total_time_energy)
//! ```
//!
//! which by Parseval is the *fraction of the whole signal's energy* held by that
//! in-band cluster (the factor 2 accounts for the unprobed negative-frequency
//! mirror; it is `1.0` for a pure on-probe in-band tone). Dividing by the **total
//! time-domain energy** — not merely the in-band spectral energy — is what makes
//! a strong *out-of-band* tone unable to trigger: its energy inflates the
//! denominator while contributing almost nothing to the in-band cluster.
//!
//! As flagged in Decision #4, this coarse approach leans on the **sustained ≥5
//! frames + stable dominant frequency** criterion to separate feedback from
//! transient speech peaks, and the exact dominance threshold still wants
//! real-audio validation.

/// Lowest probe / feedback frequency considered (Hz).
const DEFAULT_MIN_FREQ_HZ: f32 = 500.0;
/// Highest probe / feedback frequency considered (Hz).
const DEFAULT_MAX_FREQ_HZ: f32 = 8000.0;
/// Probe spacing (Hz) — matches the 50 Hz bin width of a 320-sample window.
const DEFAULT_PROBE_SPACING_HZ: f32 = 50.0;
/// Fraction of total energy a single in-band tone cluster must hold to count as
/// a feedback candidate.
const DEFAULT_DOMINANCE_THRESHOLD: f32 = 0.60;
/// Consecutive candidate frames required before declaring feedback (~100 ms).
const DEFAULT_SUSTAIN_FRAMES: u32 = 5;
/// Total normalized time energy below which a frame is treated as silence.
const SILENCE_ENERGY_EPSILON: f32 = 1e-6;
/// Scale factor to map `i16` PCM into the `-1.0..=1.0` float domain.
const I16_SCALE: f32 = 32_768.0;

/// Configuration for [`FeedbackDetector`].
#[derive(Debug, Clone)]
pub struct FeedbackConfig {
    pub sample_rate: u32,
    pub min_freq_hz: f32,
    pub max_freq_hz: f32,
    pub probe_spacing_hz: f32,
    pub dominance_threshold: f32,
    pub sustain_frames: u32,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            min_freq_hz: DEFAULT_MIN_FREQ_HZ,
            max_freq_hz: DEFAULT_MAX_FREQ_HZ,
            probe_spacing_hz: DEFAULT_PROBE_SPACING_HZ,
            dominance_threshold: DEFAULT_DOMINANCE_THRESHOLD,
            sustain_frames: DEFAULT_SUSTAIN_FRAMES,
        }
    }
}

/// Result of analysing one frame through the [`FeedbackDetector`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeedbackResult {
    /// `true` when feedback is currently confirmed (and the frame was zeroed).
    pub detected: bool,
    /// Frequency (Hz) of the dominant in-band probe this frame.
    pub frequency_hz: f32,
    /// Fraction of total energy held by the dominant in-band cluster (`0.0..=1.0`).
    pub dominance: f32,
}

/// Stateful feedback detector. Construct once per capture session and feed it
/// every ~20 ms frame in order via [`FeedbackDetector::process`].
pub struct FeedbackDetector {
    /// `2·cos(ω)` Goertzel coefficient per probe.
    coeffs: Vec<f32>,
    /// Probe center frequency (Hz), parallel to `coeffs`.
    freqs: Vec<f32>,
    /// Per-probe power scratch, reused every frame.
    powers: Vec<f32>,
    dominance_threshold: f32,
    sustain_frames: u32,
    sustain_count: u32,
    last_dominant_index: Option<usize>,
    /// Rising-edge latch so we `warn!` once per feedback episode.
    warned: bool,
}

impl FeedbackDetector {
    pub fn new(config: FeedbackConfig) -> Self {
        let fs = config.sample_rate as f32;
        let mut freqs = Vec::new();
        let mut f = config.min_freq_hz;
        while f <= config.max_freq_hz + 1e-3 {
            freqs.push(f);
            f += config.probe_spacing_hz;
        }
        let coeffs: Vec<f32> = freqs
            .iter()
            .map(|&f| 2.0 * (2.0 * std::f32::consts::PI * f / fs).cos())
            .collect();
        let n = freqs.len();
        Self {
            coeffs,
            freqs,
            powers: vec![0.0; n],
            dominance_threshold: config.dominance_threshold,
            sustain_frames: config.sustain_frames,
            sustain_count: 0,
            last_dominant_index: None,
            warned: false,
        }
    }

    /// Analyse a frame; on confirmed feedback, zero `samples` in place and return
    /// `detected = true`. The caller is expected to forward the (now-silent)
    /// frame regardless, to keep the STT stream continuous.
    pub fn process(&mut self, samples: &mut [i16]) -> FeedbackResult {
        if samples.is_empty() {
            self.clear_run();
            return FeedbackResult {
                detected: false,
                frequency_hz: 0.0,
                dominance: 0.0,
            };
        }

        let n = samples.len();
        let mut time_energy = 0.0_f32;
        for &s in samples.iter() {
            let x = s as f32 / I16_SCALE;
            time_energy += x * x;
        }

        if time_energy < SILENCE_ENERGY_EPSILON {
            self.clear_run();
            return FeedbackResult {
                detected: false,
                frequency_hz: 0.0,
                dominance: 0.0,
            };
        }

        // Goertzel power at every probe; track the dominant.
        let mut dom_index = 0;
        let mut dom_power = 0.0_f32;
        for i in 0..self.coeffs.len() {
            let p = goertzel_power(samples, self.coeffs[i]);
            self.powers[i] = p;
            if p > dom_power {
                dom_power = p;
                dom_index = i;
            }
        }

        // Dominant cluster = peak probe + immediate neighbours.
        let mut cluster = self.powers[dom_index];
        if dom_index > 0 {
            cluster += self.powers[dom_index - 1];
        }
        if dom_index + 1 < self.powers.len() {
            cluster += self.powers[dom_index + 1];
        }
        let dominance = (2.0 * cluster / (n as f32 * time_energy)).min(1.0);
        let frequency_hz = self.freqs[dom_index];

        let is_candidate = dominance >= self.dominance_threshold;
        if is_candidate {
            let continues = self
                .last_dominant_index
                .map(|last| (dom_index as i64 - last as i64).abs() <= 1)
                .unwrap_or(false);
            if continues {
                self.sustain_count += 1;
            } else {
                self.sustain_count = 1;
            }
            self.last_dominant_index = Some(dom_index);
        } else {
            self.clear_run();
        }

        let detected = self.sustain_count >= self.sustain_frames;
        if detected {
            if !self.warned {
                log::warn!("audio_gate: feedback detected at {}Hz", frequency_hz.round());
                self.warned = true;
            }
            for s in samples.iter_mut() {
                *s = 0;
            }
        } else {
            self.warned = false;
        }

        FeedbackResult {
            detected,
            frequency_hz,
            dominance,
        }
    }

    /// Reset all detection state (e.g. when (re)starting capture).
    pub fn reset(&mut self) {
        self.clear_run();
    }

    fn clear_run(&mut self) {
        self.sustain_count = 0;
        self.last_dominant_index = None;
        self.warned = false;
    }
}

/// Goertzel magnitude-squared `|X(f)|²` at the frequency whose coefficient is
/// `coeff = 2·cos(2π f / fs)`, over a fresh (per-frame) filter state.
fn goertzel_power(samples: &[i16], coeff: f32) -> f32 {
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    for &x in samples {
        let xn = x as f32 / I16_SCALE;
        let s0 = xn + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    s1 * s1 + s2 * s2 - coeff * s1 * s2
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn tone(freq: f32, amplitude: f32, n: usize) -> Vec<i16> {
        (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                ((2.0 * PI * freq * t).sin() * amplitude * 32_767.0)
                    .round()
                    .clamp(i16::MIN as f32, i16::MAX as f32) as i16
            })
            .collect()
    }

    fn mix(freqs: &[f32], amplitude: f32, n: usize) -> Vec<i16> {
        (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                let v: f32 = freqs.iter().map(|&f| (2.0 * PI * f * t).sin()).sum();
                (v * amplitude * 32_767.0)
                    .round()
                    .clamp(i16::MIN as f32, i16::MAX as f32) as i16
            })
            .collect()
    }

    fn detector() -> FeedbackDetector {
        FeedbackDetector::new(FeedbackConfig::default())
    }

    #[test]
    fn sustained_tone_is_detected_and_zeroed() {
        let mut d = detector();
        for _ in 0..4 {
            let mut f = tone(2300.0, 0.5, 320);
            assert!(!d.process(&mut f).detected);
        }
        let mut f5 = tone(2300.0, 0.5, 320);
        let r = d.process(&mut f5);
        assert!(r.detected, "not detected on 5th frame (dominance {})", r.dominance);
        assert!((r.frequency_hz - 2300.0).abs() <= 50.0, "freq {}", r.frequency_hz);
        assert!(f5.iter().all(|&s| s == 0), "detected frame was not zeroed");
    }

    #[test]
    fn tone_below_sustain_threshold_is_not_detected() {
        let mut d = detector();
        let mut last = tone(2300.0, 0.5, 320);
        for _ in 0..4 {
            last = tone(2300.0, 0.5, 320);
            assert!(!d.process(&mut last).detected);
        }
        // Fourth frame must be left untouched (not zeroed).
        assert!(last.iter().any(|&s| s != 0));
    }

    #[test]
    fn out_of_band_tone_is_not_detected() {
        // 200 Hz is below the 500 Hz band; its energy must not trigger feedback.
        let mut d = detector();
        for _ in 0..8 {
            let mut f = tone(200.0, 0.7, 320);
            assert!(!d.process(&mut f).detected);
        }
    }

    #[test]
    fn broadband_mix_is_not_detected() {
        let mut d = detector();
        let parts = [600.0, 1500.0, 3000.0, 5000.0, 7000.0];
        for _ in 0..8 {
            let mut f = mix(&parts, 0.15, 320);
            let r = d.process(&mut f);
            assert!(!r.detected, "broadband flagged (dominance {})", r.dominance);
        }
    }

    #[test]
    fn drifting_frequency_never_sustains() {
        let mut d = detector();
        for i in 0..10 {
            let freq = if i % 2 == 0 { 1000.0 } else { 5000.0 };
            let mut f = tone(freq, 0.5, 320);
            assert!(!d.process(&mut f).detected, "drift sustained at frame {}", i);
        }
    }

    #[test]
    fn silence_resets_the_run() {
        let mut d = detector();
        for _ in 0..5 {
            let mut f = tone(2300.0, 0.5, 320);
            d.process(&mut f);
        }
        let mut sil = vec![0i16; 320];
        assert!(!d.process(&mut sil).detected);
        // Run restarted: four more frames are not yet enough.
        for _ in 0..4 {
            let mut f = tone(2300.0, 0.5, 320);
            assert!(!d.process(&mut f).detected);
        }
    }

    #[test]
    fn empty_frame_is_not_detected() {
        let mut d = detector();
        let r = d.process(&mut []);
        assert!(!r.detected);
    }
}
