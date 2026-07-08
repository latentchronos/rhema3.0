//! Endpoint state machine over Silero speech probabilities.
//!
//! This is the `VADIterator` logic from Silero's Python reference (`utils_vad.py`):
//! hysteresis (`neg_threshold = threshold − 0.15`), a silence grace window before an end
//! is declared, and speech padding on both edges. The Rust `voice_activity_detector`
//! crate omits the hysteresis, so we reimplement it here — it is what keeps brief dips in a
//! sustained phrase from prematurely ending speech during a sermon.
//!
//! Pure and deterministic: feed it per-frame probabilities, get back `is_speech` plus a
//! `SpeechStart`/`SpeechEnd` event at the transitions. Unit-tested without the model.

use crate::{FRAME_SAMPLES, SAMPLE_RATE};

/// Endpointing configuration. Defaults are Silero's, except `min_silence_ms` is raised to
/// suit sermon cadence (a preacher pauses mid-thought; 100 ms would chop phrases).
#[derive(Clone, Copy, Debug)]
pub struct VadConfig {
    /// Speech begins when probability rises to/above this. Silero default 0.5.
    pub threshold: f32,
    /// Speech may end only when probability falls below this (`threshold − 0.15`). The gap
    /// between the two thresholds is the hysteresis band that debounces brief dips.
    pub neg_threshold: f32,
    /// Trailing sub-threshold audio required before declaring the phrase over.
    pub min_silence_ms: u32,
    /// Padding added before a start / after an end so onsets and tails aren't clipped.
    pub speech_pad_ms: u32,
    pub sample_rate: u32,
    /// Samples advanced per [`VadGate::process`] call (one Silero frame).
    pub frame_samples: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        let threshold = 0.5;
        Self {
            threshold,
            neg_threshold: threshold - 0.15,
            min_silence_ms: 400,
            speech_pad_ms: 100,
            sample_rate: SAMPLE_RATE,
            frame_samples: FRAME_SAMPLES,
        }
    }
}

/// A speech-boundary transition, timestamped in stream-relative milliseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VadEvent {
    SpeechStart { t_ms: u64 },
    SpeechEnd { t_ms: u64 },
}

/// The outcome of feeding one frame's probability to the gate.
#[derive(Clone, Copy, Debug)]
pub struct VadFrameResult {
    /// The raw Silero probability for this frame.
    pub prob: f32,
    /// Whether the gate currently considers audio to be speech (post-update).
    pub is_speech: bool,
    /// A boundary transition, if one occurred on this frame.
    pub event: Option<VadEvent>,
}

/// Endpoint state machine. One per capture stream; `reset` on stream restart / new service.
pub struct VadGate {
    cfg: VadConfig,
    triggered: bool,
    /// Sample index at which the current silence run began (0 = not in silence).
    temp_end: u64,
    /// Total samples advanced since construction / reset.
    current_sample: u64,
    min_silence_samples: u64,
    speech_pad_samples: u64,
}

impl VadGate {
    pub fn new(cfg: VadConfig) -> Self {
        let sr = cfg.sample_rate as u64;
        Self {
            min_silence_samples: sr * cfg.min_silence_ms as u64 / 1000,
            speech_pad_samples: sr * cfg.speech_pad_ms as u64 / 1000,
            cfg,
            triggered: false,
            temp_end: 0,
            current_sample: 0,
        }
    }

    /// Advance by one frame and fold in its probability. Mirrors `VADIterator.__call__`.
    pub fn process(&mut self, prob: f32) -> VadFrameResult {
        let frame = self.cfg.frame_samples as u64;
        self.current_sample += frame;
        let mut event = None;

        // A rebound above threshold cancels a pending end.
        if prob >= self.cfg.threshold && self.temp_end != 0 {
            self.temp_end = 0;
        }

        if prob >= self.cfg.threshold && !self.triggered {
            self.triggered = true;
            let start = self
                .current_sample
                .saturating_sub(self.speech_pad_samples + frame);
            event = Some(VadEvent::SpeechStart {
                t_ms: self.samples_to_ms(start),
            });
        } else if prob < self.cfg.neg_threshold && self.triggered {
            if self.temp_end == 0 {
                self.temp_end = self.current_sample;
            }
            if self.current_sample - self.temp_end >= self.min_silence_samples {
                let end = self.temp_end + self.speech_pad_samples.saturating_sub(frame);
                self.temp_end = 0;
                self.triggered = false;
                event = Some(VadEvent::SpeechEnd {
                    t_ms: self.samples_to_ms(end),
                });
            }
        }

        VadFrameResult {
            prob,
            is_speech: self.triggered,
            event,
        }
    }

    /// Whether the gate currently regards audio as speech.
    pub fn is_speech(&self) -> bool {
        self.triggered
    }

    /// Reset to the pre-speech state (new stream / service).
    pub fn reset(&mut self) {
        self.triggered = false;
        self.temp_end = 0;
        self.current_sample = 0;
    }

    fn samples_to_ms(&self, samples: u64) -> u64 {
        samples * 1000 / self.cfg.sample_rate as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A config with tiny, frame-aligned windows so tests are exact and readable.
    /// frame = 100 samples @ 1000 Hz = 100 ms/frame; min_silence = 200 ms = 2 frames.
    fn test_cfg() -> VadConfig {
        VadConfig {
            threshold: 0.5,
            neg_threshold: 0.35,
            min_silence_ms: 200,
            speech_pad_ms: 0, // no padding → boundaries land on frame edges
            sample_rate: 1000,
            frame_samples: 100,
        }
    }

    #[test]
    fn fires_start_on_first_supra_threshold_frame() {
        let mut g = VadGate::new(test_cfg());
        let r = g.process(0.2);
        assert!(!r.is_speech);
        assert!(r.event.is_none());

        let r = g.process(0.9);
        assert!(r.is_speech);
        assert!(matches!(r.event, Some(VadEvent::SpeechStart { .. })));
    }

    #[test]
    fn stays_in_speech_through_a_brief_dip() {
        let mut g = VadGate::new(test_cfg());
        g.process(0.9); // start
                        // A single sub-threshold frame within the hysteresis/grace window
                        // must NOT end speech.
        let r = g.process(0.1);
        assert!(r.is_speech, "brief dip prematurely ended speech");
        assert!(r.event.is_none());
    }

    #[test]
    fn ends_after_min_silence_of_low_probability() {
        let mut g = VadGate::new(test_cfg());
        g.process(0.9); // start
        let r1 = g.process(0.1); // temp_end set here (grace begins)
        assert!(r1.is_speech && r1.event.is_none());
        // min_silence = 2 frames; end fires once current-temp_end >= 200 samples.
        let r2 = g.process(0.1);
        assert!(r2.is_speech && r2.event.is_none(), "ended too early");
        let r3 = g.process(0.1);
        assert!(matches!(r3.event, Some(VadEvent::SpeechEnd { .. })));
        assert!(!r3.is_speech);
    }

    #[test]
    fn rebound_cancels_pending_end() {
        let mut g = VadGate::new(test_cfg());
        g.process(0.9); // start
        g.process(0.1); // grace begins
        let r = g.process(0.9); // rebound before min_silence elapses
        assert!(r.is_speech);
        assert!(r.event.is_none());
        // Now a fresh silence run must restart the full grace window.
        g.process(0.1);
        g.process(0.1);
        let r = g.process(0.1);
        assert!(matches!(r.event, Some(VadEvent::SpeechEnd { .. })));
    }

    #[test]
    fn reset_returns_to_idle() {
        let mut g = VadGate::new(test_cfg());
        g.process(0.9);
        assert!(g.is_speech());
        g.reset();
        assert!(!g.is_speech());
    }
}
