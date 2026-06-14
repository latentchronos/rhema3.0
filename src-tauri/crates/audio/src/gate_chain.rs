//! Phase 1 gate chain (integration of Bullets 1.1–1.4).
//!
//! Wires the four hardware-gating components into one ordered pipeline and
//! handles re-blocking of the variable-length capture frames into the fixed
//! 20 ms (320-sample at 16 kHz) analysis windows the gates assume. This is the
//! single object the app's fan-out thread drives; it owns all per-session gate
//! state plus a small carry-over accumulator.
//!
//! ## Pipeline order (per the Phase 1 companion)
//! For each complete 320-sample window:
//! 1. [`SubbandFluxGate`] and [`EnergyVarianceGate`] decide whether the window
//!    is musical / dynamic bleed. If either gates it, the window is **dropped**
//!    (not forwarded) — it never reaches STT.
//! 2. Surviving windows go through the [`RmsAgc`] adaptive normalizer (which
//!    supersedes the static capture gain — that gain now acts as a pre-AGC trim).
//! 3. [`FeedbackDetector`] runs last; on a confirmed PA-feedback tone it zeroes
//!    the window in place but the window is still **forwarded** (zeroed, not
//!    dropped) to keep the STT stream continuous.
//!
//! ## A note on ordering and feedback
//! Because flux/variance run before the feedback detector, a *loud* feedback
//! squeal (high variance) is dropped at step 1 and never reaches the detector —
//! which is fine, the tone is removed either way. The feedback detector's real
//! value is catching feedback **early, while it is still quiet and growing**
//! (low variance, high tonality): such windows pass steps 1–2 and are flagged,
//! zeroed, and logged before the squeal becomes loud.
//!
//! Output samples are the concatenation of the surviving windows (AGC-applied,
//! feedback-zeroed where detected). Leftover (< one window) samples are retained
//! for the next call. The caller reads its level meter on the *raw* frame before
//! invoking this chain, so metering stays pre-AGC.

use crate::agc::{AgcConfig, RmsAgc};
use crate::feedback::{FeedbackConfig, FeedbackDetector};
use crate::flux_gate::{SubbandFluxConfig, SubbandFluxGate};
use crate::variance_gate::{EnergyVarianceGate, VarianceConfig};

/// Analysis window length in samples (20 ms at 16 kHz).
const DEFAULT_WINDOW_SAMPLES: usize = 320;

/// Configuration for [`GateChain`], composing each gate's configuration.
#[derive(Debug, Clone, Default)]
pub struct GateChainConfig {
    pub flux: SubbandFluxConfig,
    pub variance: VarianceConfig,
    pub agc: AgcConfig,
    pub feedback: FeedbackConfig,
    pub window_samples: usize,
    /// When `false` (the default, until the flux/variance thresholds are tuned
    /// against real audio), the flux/variance gates run in **observe mode**:
    /// their verdicts are computed and counted but windows are **not dropped**,
    /// so legitimate speech is never discarded. AGC and feedback detection still
    /// apply. Set `true` once thresholds are validated to enable real dropping.
    pub suppress_active: bool,
}

impl GateChainConfig {
    fn window_or_default(&self) -> usize {
        if self.window_samples == 0 {
            DEFAULT_WINDOW_SAMPLES
        } else {
            self.window_samples
        }
    }
}

/// Outcome of running one capture frame through the [`GateChain`].
#[derive(Debug, Clone, PartialEq)]
pub struct GateOutcome {
    /// Processed samples to forward downstream (may be empty if every window was
    /// suppressed, or if not enough samples have accumulated for a window yet).
    pub samples: Vec<i16>,
    /// Number of complete windows processed this call.
    pub windows_total: u32,
    /// Number of windows the flux/variance gates flagged as bleed. These are
    /// dropped only when `suppress_active` is set; otherwise this is the count
    /// that *would* be dropped (observe mode).
    pub windows_suppressed: u32,
    /// Whether feedback was confirmed in any window this call.
    pub feedback_active: bool,
    /// Highest spectral-flux value observed across this call's windows — for
    /// threshold tuning against real audio.
    pub peak_flux: f32,
    /// Highest energy-variance value observed across this call's windows — for
    /// threshold tuning against real audio.
    pub peak_variance: f32,
}

/// The ordered Phase 1 gate pipeline with frame re-blocking. Construct one per
/// capture session and drive it with each [`process`](GateChain::process) call.
pub struct GateChain {
    flux: SubbandFluxGate,
    variance: EnergyVarianceGate,
    agc: RmsAgc,
    feedback: FeedbackDetector,
    window_samples: usize,
    suppress_active: bool,
    accumulator: Vec<i16>,
}

impl GateChain {
    pub fn new(config: GateChainConfig) -> Self {
        let window_samples = config.window_or_default();
        Self {
            flux: SubbandFluxGate::new(config.flux),
            variance: EnergyVarianceGate::new(config.variance),
            agc: RmsAgc::new(config.agc),
            feedback: FeedbackDetector::new(config.feedback),
            window_samples,
            suppress_active: config.suppress_active,
            accumulator: Vec::new(),
        }
    }

    /// Append a capture frame's samples, process every complete window now
    /// available, and return the forwardable output plus per-call diagnostics.
    pub fn process(&mut self, frame_samples: &[i16]) -> GateOutcome {
        self.accumulator.extend_from_slice(frame_samples);

        let mut samples = Vec::new();
        let mut windows_total = 0;
        let mut windows_suppressed = 0;
        let mut feedback_active = false;
        let mut peak_flux = 0.0_f32;
        let mut peak_variance = 0.0_f32;

        while self.accumulator.len() >= self.window_samples {
            let mut window: Vec<i16> = self.accumulator.drain(..self.window_samples).collect();
            windows_total += 1;

            let flux = self.flux.process(&window);
            let variance = self.variance.process(&window);
            if flux.flux > peak_flux {
                peak_flux = flux.flux;
            }
            if variance.variance > peak_variance {
                peak_variance = variance.variance;
            }

            if flux.gated || variance.gated {
                windows_suppressed += 1;
                if self.suppress_active {
                    continue; // musical / dynamic bleed — do not forward
                }
                // Observe mode: counted but forwarded so speech is never lost.
            }

            self.agc.process(&mut window);
            let feedback = self.feedback.process(&mut window);
            if feedback.detected {
                feedback_active = true;
            }
            // Forward the window (zeroed in place if feedback was detected).
            samples.extend_from_slice(&window);
        }

        GateOutcome {
            samples,
            windows_total,
            windows_suppressed,
            feedback_active,
            peak_flux,
            peak_variance,
        }
    }

    /// Reset all gate state and discard any buffered samples.
    pub fn reset(&mut self) {
        self.flux.reset();
        self.agc.reset();
        self.feedback.reset();
        self.accumulator.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Quiet broadband mix (speech-like): low variance, low flux, no dominant
    /// tone — should pass every gate.
    fn quiet_broadband(n: usize) -> Vec<i16> {
        let parts = [600.0, 1500.0, 3000.0, 5000.0];
        (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                let v: f32 = parts.iter().map(|&f| (2.0 * PI * f * t).sin()).sum();
                (v * 0.05 * 32_767.0)
                    .round()
                    .clamp(i16::MIN as f32, i16::MAX as f32) as i16
            })
            .collect()
    }

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

    fn chain() -> GateChain {
        GateChain::new(GateChainConfig::default())
    }

    fn active_chain() -> GateChain {
        GateChain::new(GateChainConfig {
            suppress_active: true,
            ..GateChainConfig::default()
        })
    }

    #[test]
    fn quiet_speech_like_signal_passes() {
        let mut c = chain();
        let out = c.process(&quiet_broadband(320 * 3));
        assert_eq!(out.windows_total, 3);
        assert_eq!(out.windows_suppressed, 0);
        assert!(!out.feedback_active);
        assert!(!out.samples.is_empty());
    }

    #[test]
    fn loud_dynamic_signal_is_suppressed_when_active() {
        let mut c = active_chain();
        // Full-scale tone → variance ≈ 0.5 (> the 0.45 default) → dropped.
        let out = c.process(&tone(1000.0, 1.0, 320 * 3));
        assert!(out.windows_suppressed > 0);
        assert!(out.samples.is_empty());
    }

    #[test]
    fn observe_mode_counts_but_forwards() {
        let mut c = chain(); // default = observe
        let out = c.process(&tone(1000.0, 1.0, 320 * 3));
        // Flagged as bleed, but forwarded rather than dropped (speech-safe default).
        assert!(out.windows_suppressed > 0);
        assert!(!out.samples.is_empty());
        assert!(out.peak_variance > 0.0);
    }

    #[test]
    fn reblocking_accumulates_across_partial_frames() {
        let mut c = chain();
        // 200-sample sub-frames: window emerges only once ≥320 have accumulated.
        let a = c.process(&quiet_broadband(200));
        assert_eq!(a.windows_total, 0); // 200 buffered
        let b = c.process(&quiet_broadband(200));
        assert_eq!(b.windows_total, 1); // 400 → one window, 80 left
        let d = c.process(&quiet_broadband(200));
        assert_eq!(d.windows_total, 0); // 280 buffered
        let e = c.process(&quiet_broadband(200));
        assert_eq!(e.windows_total, 1); // 480 → one window
    }

    #[test]
    fn quiet_sustained_tone_is_feedback_zeroed_but_forwarded() {
        let mut c = chain();
        // Amplitude 0.1 → variance 0.005 (< 0.01) so it passes the variance gate
        // and reaches the feedback detector; 6 windows ≥ the 5-frame sustain.
        let out = c.process(&tone(2300.0, 0.1, 320 * 6));
        assert_eq!(out.windows_suppressed, 0, "tone was dropped before feedback stage");
        assert!(out.feedback_active, "feedback not detected");
        assert_eq!(out.samples.len(), 320 * 6, "frames were dropped, not forwarded");
        assert!(out.samples.iter().any(|&s| s == 0), "no zeroed (muted) samples");
    }

    #[test]
    fn reset_clears_accumulator() {
        let mut c = chain();
        c.process(&quiet_broadband(200)); // 200 buffered
        c.reset();
        // After reset the earlier 200 are gone, so 200 more yields no window.
        let out = c.process(&quiet_broadband(200));
        assert_eq!(out.windows_total, 0);
    }
}
