//! RMS adaptive normalizer / software AGC (Phase 1, Bullet 1.3).
//!
//! Replaces the static gain coefficient in the capture path with a running gain
//! that tracks the signal level and keeps the output RMS inside a target window.
//! A preacher who steps back from the mic is brought back up; a sudden shout is
//! eased down — without the pumping artifacts of a hard limiter.
//!
//! ## Target window
//! The companion specifies an i16 RMS window of **1036–2068** (used directly by
//! its Example D: `2.3x = 1036 / 450`). Note: the companion *also* labels this
//! window "−18 to −12 dBFS", but those labels are inconsistent with the RMS
//! values (`1036/32767 ≈ −30 dBFS`). The RMS values are authoritative here
//! because the worked example depends on them; the dBFS labels are not used.
//!
//! ## Attack / release
//! Per Example D, the gain is moved with an asymmetric one-pole envelope:
//! - **Rising** gain (input got quieter → boost): slow, `attack_ms = 200`. This
//!   avoids pumping up room noise / echo during speech gaps.
//! - **Falling** gain (input got louder → cut): fast, `release_ms = 50`. This
//!   eases down loud transients quickly before they can clip.
//!
//! The per-frame smoothing coefficient is `exp(-frame_seconds / tau)`, the exact
//! one-pole response — no approximation. Within a frame the gain is interpolated
//! linearly from the previous value to the new value so level changes are
//! click-free. The only hard clamp is the unavoidable i16 saturation on output;
//! because the target window sits far below full scale, it is rarely reached.
//!
//! The component carries state (`current_gain`) and mutates samples in place;
//! it performs no heap allocation in [`RmsAgc::process`].

/// Lower bound of the target output-RMS window (i16 scale).
const TARGET_RMS_FLOOR: f32 = 1036.0;
/// Upper bound of the target output-RMS window (i16 scale).
const TARGET_RMS_CEIL: f32 = 2068.0;
/// Slow time constant (ms) used when the gain is *rising* (boosting).
const DEFAULT_ATTACK_MS: f32 = 200.0;
/// Fast time constant (ms) used when the gain is *falling* (cutting).
const DEFAULT_RELEASE_MS: f32 = 50.0;
/// Maximum gain; bounds how much near-silent input is amplified.
const DEFAULT_MAX_GAIN: f32 = 8.0;
/// Minimum gain; keeps very loud input from being driven to silence.
const DEFAULT_MIN_GAIN: f32 = 0.05;
/// Input RMS (i16 scale) below which the gain is held — silence/noise must not
/// be boosted toward the target window.
const DEFAULT_SILENCE_FLOOR_RMS: f32 = 50.0;

/// Configuration for [`RmsAgc`].
#[derive(Debug, Clone)]
pub struct AgcConfig {
    pub sample_rate: u32,
    pub target_rms_floor: f32,
    pub target_rms_ceil: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub max_gain: f32,
    pub min_gain: f32,
    pub silence_floor_rms: f32,
}

impl Default for AgcConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            target_rms_floor: TARGET_RMS_FLOOR,
            target_rms_ceil: TARGET_RMS_CEIL,
            attack_ms: DEFAULT_ATTACK_MS,
            release_ms: DEFAULT_RELEASE_MS,
            max_gain: DEFAULT_MAX_GAIN,
            min_gain: DEFAULT_MIN_GAIN,
            silence_floor_rms: DEFAULT_SILENCE_FLOOR_RMS,
        }
    }
}

/// Stateful software AGC. Construct once per capture session and feed it every
/// frame in order via [`RmsAgc::process`].
pub struct RmsAgc {
    sample_rate: f32,
    target_rms_floor: f32,
    target_rms_ceil: f32,
    attack_s: f32,
    release_s: f32,
    max_gain: f32,
    min_gain: f32,
    silence_floor_rms: f32,
    current_gain: f32,
}

impl RmsAgc {
    pub fn new(config: AgcConfig) -> Self {
        Self {
            sample_rate: config.sample_rate as f32,
            target_rms_floor: config.target_rms_floor,
            target_rms_ceil: config.target_rms_ceil,
            attack_s: config.attack_ms / 1000.0,
            release_s: config.release_ms / 1000.0,
            max_gain: config.max_gain,
            min_gain: config.min_gain,
            silence_floor_rms: config.silence_floor_rms,
            current_gain: 1.0,
        }
    }

    /// The gain currently applied (value at the end of the last processed frame).
    pub fn current_gain(&self) -> f32 {
        self.current_gain
    }

    /// Apply adaptive gain to `samples` in place and return the gain reached at
    /// the end of the frame.
    pub fn process(&mut self, samples: &mut [i16]) -> f32 {
        if samples.is_empty() {
            return self.current_gain;
        }

        let n = samples.len();
        let mut sum_sq = 0.0_f64;
        for &s in samples.iter() {
            sum_sq += (s as f64) * (s as f64);
        }
        let rms_in = (sum_sq / n as f64).sqrt() as f32;

        let g_start = self.current_gain;

        // Decide the gain we'd like to reach for this frame.
        let g_target = if rms_in < self.silence_floor_rms {
            g_start // silence/noise: hold, do not boost
        } else {
            let out_rms = rms_in * g_start;
            if out_rms < self.target_rms_floor {
                (self.target_rms_floor / rms_in).clamp(self.min_gain, self.max_gain)
            } else if out_rms > self.target_rms_ceil {
                (self.target_rms_ceil / rms_in).clamp(self.min_gain, self.max_gain)
            } else {
                g_start // inside the window: hold (dead zone, no pumping)
            }
        };

        // Slow when rising (attack), fast when falling (release).
        let tau = if g_target > g_start {
            self.attack_s
        } else {
            self.release_s
        };
        let dt = n as f32 / self.sample_rate;
        let alpha = (-dt / tau).exp();
        let g_end = g_target + (g_start - g_target) * alpha;

        // Apply the gain interpolated linearly across the frame (click-free),
        // saturating to i16 on output.
        let denom = if n > 1 { (n - 1) as f32 } else { 1.0 };
        for (i, s) in samples.iter_mut().enumerate() {
            let g = g_start + (g_end - g_start) * (i as f32 / denom);
            let scaled = (*s as f32 * g).round();
            *s = scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        }

        self.current_gain = g_end;
        g_end
    }

    /// Reset the running gain to unity (e.g. when (re)starting capture).
    pub fn reset(&mut self) {
        self.current_gain = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Generate a 1 kHz sine frame whose RMS is approximately `target_rms`
    /// (i16 scale). 1 kHz at 16 kHz over 320 samples is 20 whole cycles, so the
    /// realized RMS equals `amplitude / sqrt(2)` exactly.
    fn sine_rms(target_rms: f32, n: usize) -> Vec<i16> {
        let amp = target_rms * 2.0_f32.sqrt();
        (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                ((2.0 * PI * 1000.0 * t).sin() * amp)
                    .round()
                    .clamp(i16::MIN as f32, i16::MAX as f32) as i16
            })
            .collect()
    }

    fn rms(samples: &[i16]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let ss: f64 = samples.iter().map(|&v| (v as f64) * (v as f64)).sum();
        (ss / samples.len() as f64).sqrt() as f32
    }

    fn agc() -> RmsAgc {
        RmsAgc::new(AgcConfig::default())
    }

    #[test]
    fn quiet_input_boosts_slowly() {
        let mut a = agc();
        let mut frame = sine_rms(450.0, 320);
        a.process(&mut frame);
        // After a single 20 ms frame the slow attack should have moved only a
        // little toward the ~2.3x asymptote.
        assert!(a.current_gain() > 1.0 && a.current_gain() < 1.2,
            "one-frame attack moved too far/little: {}", a.current_gain());
    }

    #[test]
    fn quiet_input_converges_toward_example_d_gain() {
        let mut a = agc();
        for _ in 0..100 {
            let mut frame = sine_rms(450.0, 320);
            a.process(&mut frame);
        }
        // Example D: rms 450 → gain ~2.3x (= 1036/450).
        assert!(a.current_gain() > 2.1 && a.current_gain() < 2.4,
            "did not converge near 2.3x: {}", a.current_gain());
    }

    #[test]
    fn sustained_quiet_input_reaches_target_window() {
        let mut a = agc();
        for _ in 0..100 {
            let mut frame = sine_rms(450.0, 320);
            a.process(&mut frame);
        }
        let mut frame = sine_rms(450.0, 320);
        a.process(&mut frame);
        let out = rms(&frame);
        assert!(out >= 1000.0 && out <= TARGET_RMS_CEIL,
            "output rms outside target window: {}", out);
    }

    #[test]
    fn loud_input_is_cut_quickly() {
        let mut a = agc();
        for _ in 0..5 {
            let mut frame = sine_rms(8000.0, 320);
            a.process(&mut frame);
        }
        // Fast release should have driven the gain well below unity within a few
        // frames. Target is 2068/8000 ≈ 0.26; the analytic value after 5 frames
        // (release τ=50 ms, frame=20 ms → α=e^-0.4) is ≈0.36 — already a steep
        // drop from unity in 100 ms.
        assert!(a.current_gain() < 0.40, "release too slow: {}", a.current_gain());
    }

    #[test]
    fn in_window_input_holds_unity_gain() {
        let mut a = agc();
        for _ in 0..20 {
            let mut frame = sine_rms(1500.0, 320); // inside 1036..2068
            a.process(&mut frame);
        }
        assert!((a.current_gain() - 1.0).abs() < 1e-3,
            "gain drifted in dead zone: {}", a.current_gain());
    }

    #[test]
    fn silence_holds_gain() {
        let mut a = agc();
        for _ in 0..20 {
            let mut frame = vec![0i16; 320];
            a.process(&mut frame);
        }
        assert_eq!(a.current_gain(), 1.0);
    }

    #[test]
    fn near_silent_input_is_clamped_to_max_gain() {
        let mut a = agc();
        // rms 60 is above the silence floor but would demand 1036/60 ≈ 17x.
        for _ in 0..200 {
            let mut frame = sine_rms(60.0, 320);
            a.process(&mut frame);
        }
        assert!(a.current_gain() <= DEFAULT_MAX_GAIN + 1e-3,
            "exceeded max gain: {}", a.current_gain());
        assert!(a.current_gain() > 7.0, "did not approach max gain: {}", a.current_gain());
    }

    #[test]
    fn empty_frame_keeps_gain() {
        let mut a = agc();
        assert_eq!(a.process(&mut []), 1.0);
    }

    #[test]
    fn reset_restores_unity_gain() {
        let mut a = agc();
        let mut frame = sine_rms(8000.0, 320);
        a.process(&mut frame);
        a.reset();
        assert_eq!(a.current_gain(), 1.0);
    }
}
