//! Real-time voice-activity detection front-end for Rhema v2 (Phase 2).
//!
//! Replaces the three overlapping energy heuristics (the RMS `Vad` state machine, the
//! observe-mode flux/variance gates, and the STT engine's internal RMS segmentation) with
//! a single neural front-end: **Silero VAD v5** (2 MB, ONNX) as the source of truth for
//! speech/non-speech, plus a proper endpoint state machine.
//!
//! ## Pieces
//! * [`Reblocker`] — re-blocks variable-size capture buffers into the exact fixed frames
//!   Silero requires. Pure; always available.
//! * [`VadGate`] — the endpoint state machine over Silero probabilities: hysteresis
//!   (`neg_threshold = threshold − 0.15`), a silence grace window, and speech padding.
//!   This is the `VADIterator` logic the Rust `voice_activity_detector` crate omits.
//!   Pure; always available and unit-tested without the model.
//! * [`SileroVad`] (feature `silero`) — the ONNX session. Carries the recurrent state and
//!   the 64-sample context buffer the Python reference prepends (also omitted by the Rust
//!   crate), so cross-frame continuity matches upstream.
//!
//! Typical wiring: `capture f32 → Reblocker → SileroVad::process → VadGate::process`.
//!
//! ## Model asset
//! [`SileroVad`] needs `silero_vad.onnx` (v5, opset 16, ~2 MB). Like the GGUFs it lives in
//! the gitignored `model/` dir (not committed). Fetch it from the upstream repo:
//! `https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx`.

pub mod error;
pub mod gate;
pub mod reblock;
#[cfg(feature = "silero")]
pub mod silero;

pub use error::VadError;
pub use gate::{VadConfig, VadEvent, VadFrameResult, VadGate};
pub use reblock::Reblocker;
#[cfg(feature = "silero")]
pub use silero::SileroVad;

/// Silero v5 requires exactly this many samples per inference at 16 kHz (32 ms).
pub const FRAME_SAMPLES: usize = 512;
/// Samples of the previous frame Silero v5 prepends as context (16 kHz).
pub const CONTEXT_SAMPLES: usize = 64;
/// The only sample rate this front-end targets (STT is 16 kHz mono).
pub const SAMPLE_RATE: u32 = 16_000;
