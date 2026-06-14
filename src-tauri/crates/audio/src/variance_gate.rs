//! Local energy-variance gate (Phase 1, Bullet 1.2).
//!
//! Computes the sample variance of the PCM signal over a rolling 20 ms window
//! (320 samples at 16 kHz):
//!
//! ```text
//! variance = E[x²] - E[x]²
//! ```
//!
//! This is *additive* to the existing RMS-based VAD. The VAD gates on low energy
//! (silence); this gate flags windows whose energy is unusually high/dynamic —
//! worship instruments, clapping, PA transients bleeding into a stage mic —
//! which steady speech does not produce. Working in the normalized `-1.0..=1.0`
//! domain keeps the variance scale-stable and equal to the signal *power* for
//! the (near-zero-mean) audio we see in practice.
//!
//! The `- E[x]²` term matters in exactly one case: a loud constant **DC offset**
//! has high power but zero variance, so it is *not* flagged — which is correct,
//! since a DC bias is a wiring artifact, not dynamic content. That property is
//! pinned by a unit test.
//!
//! Like the [`crate::flux_gate`], this component only *reports* a verdict; the
//! caller decides whether to drop the frame. It carries no cross-frame state.

/// Analysis window length in samples (20 ms at 16 kHz).
const VARIANCE_WINDOW_SAMPLES: usize = 320;

/// Default variance value (normalized power units, `0.0..≈1.0`) at/above which a
/// window is judged "dynamic" and the frame is gated.
///
/// Tuned against real captured audio (2026-06-12): speech `peak_var` topped out
/// near 0.31 while sustained instrumental music ran 0.5–0.89, so 0.45 separates
/// them with margin for loud speech. NOTE this discriminates by signal *power*,
/// so it relies on worship music being louder than speech at the input; if a
/// deployment has them at similar levels, retune via [`VarianceConfig`].
const DEFAULT_VARIANCE_THRESHOLD: f32 = 0.45;

/// Scale factor to map `i16` PCM into the `-1.0..=1.0` float domain.
const I16_SCALE: f64 = 32_768.0;

/// Configuration for [`EnergyVarianceGate`].
#[derive(Debug, Clone)]
pub struct VarianceConfig {
    /// Variance value at/above which [`VarianceResult::gated`] is `true`.
    pub variance_threshold: f32,
}

impl Default for VarianceConfig {
    fn default() -> Self {
        Self {
            variance_threshold: DEFAULT_VARIANCE_THRESHOLD,
        }
    }
}

/// Result of analysing one frame through the [`EnergyVarianceGate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarianceResult {
    /// The maximum per-window variance found in the frame, in normalized power
    /// units (`0.0..≈1.0`).
    pub variance: f32,
    /// `true` when `variance >= variance_threshold` — i.e. at least one 20 ms
    /// window looked dynamic rather than speech-like.
    pub gated: bool,
}

/// Stateless gate that flags frames containing a high-variance (dynamic) window.
pub struct EnergyVarianceGate {
    threshold: f32,
}

impl EnergyVarianceGate {
    pub fn new(config: VarianceConfig) -> Self {
        Self {
            threshold: config.variance_threshold,
        }
    }

    /// Analyse a frame and return the worst (highest) 20 ms-window variance plus
    /// the gate verdict.
    ///
    /// Frames of at least one window length are split into non-overlapping
    /// 320-sample windows; any trailing partial window (`< 320` samples) is
    /// ignored, because integration aligns frames to window boundaries and a
    /// short partial yields a noisy estimate. Frames shorter than one window are
    /// assessed whole (so small inputs still produce a value).
    pub fn process(&self, samples: &[i16]) -> VarianceResult {
        let variance = if samples.len() < VARIANCE_WINDOW_SAMPLES {
            window_variance(samples)
        } else {
            let full = samples.len() - (samples.len() % VARIANCE_WINDOW_SAMPLES);
            let mut max_var = 0.0_f32;
            for window in samples[..full].chunks(VARIANCE_WINDOW_SAMPLES) {
                let v = window_variance(window);
                if v > max_var {
                    max_var = v;
                }
            }
            max_var
        };

        VarianceResult {
            variance,
            gated: variance >= self.threshold,
        }
    }
}

/// Compute `E[x²] - E[x]²` over a window in the normalized float domain.
/// Returns `0.0` for an empty window. Tiny negative results from floating-point
/// rounding are clamped to `0.0`.
fn window_variance(samples: &[i16]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let n = samples.len() as f64;
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for &s in samples {
        let x = s as f64 / I16_SCALE;
        sum += x;
        sum_sq += x * x;
    }
    let mean = sum / n;
    let variance = (sum_sq / n) - mean * mean;
    variance.max(0.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Generate an `n`-sample sine window at `freq` Hz with the given amplitude
    /// (fraction of full scale).
    fn sine(freq: f32, amplitude: f32, sample_rate: f32, n: usize) -> Vec<i16> {
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate;
                let v = (2.0 * PI * freq * t).sin() * amplitude;
                (v * 32_767.0) as i16
            })
            .collect()
    }

    fn gate() -> EnergyVarianceGate {
        EnergyVarianceGate::new(VarianceConfig::default())
    }

    /// Gate with an explicit threshold, so gating-mechanism tests are decoupled
    /// from the tuned default.
    fn gate_with(threshold: f32) -> EnergyVarianceGate {
        EnergyVarianceGate::new(VarianceConfig {
            variance_threshold: threshold,
        })
    }

    #[test]
    fn silence_has_zero_variance_and_passes() {
        let r = gate().process(&vec![0i16; 320]);
        assert!(r.variance < 1e-6);
        assert!(!r.gated);
    }

    #[test]
    fn empty_frame_passes() {
        let r = gate().process(&[]);
        assert_eq!(r.variance, 0.0);
        assert!(!r.gated);
    }

    #[test]
    fn loud_dynamic_tone_is_gated() {
        // 1 kHz at 16 kHz over 320 samples = 20 whole cycles → mean ≈ 0,
        // so variance ≈ A²/2 = 0.5²/2 = 0.125.
        let r = gate_with(0.01).process(&sine(1000.0, 0.5, 16_000.0, 320));
        assert!(r.variance > 0.10, "variance lower than expected: {}", r.variance);
        assert!(r.gated);
    }

    #[test]
    fn quiet_tone_passes() {
        // Amplitude 0.05 → variance ≈ 0.00125, below the 0.01 threshold.
        let r = gate().process(&sine(1000.0, 0.05, 16_000.0, 320));
        assert!(r.variance < 0.01);
        assert!(!r.gated);
    }

    #[test]
    fn dc_offset_has_zero_variance_and_passes() {
        // A loud constant (DC bias) has high power but zero variance — the
        // `- E[x]²` term must cancel it. This is the gate's defining property.
        let r = gate().process(&vec![16_383i16; 320]);
        assert!(r.variance < 1e-6, "DC offset produced nonzero variance: {}", r.variance);
        assert!(!r.gated);
    }

    #[test]
    fn variance_grows_with_amplitude() {
        let quiet = gate().process(&sine(1000.0, 0.05, 16_000.0, 320)).variance;
        let loud = gate().process(&sine(1000.0, 0.5, 16_000.0, 320)).variance;
        assert!(loud > quiet);
    }

    #[test]
    fn multi_window_frame_uses_worst_window() {
        // 1024 samples = three full 320-windows (+64 ignored). A loud frame gates.
        let r = gate_with(0.01).process(&sine(1000.0, 0.5, 16_000.0, 1024));
        assert!(r.gated);
    }
}
