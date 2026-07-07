/// Errors from the VAD front-end.
#[derive(Debug, thiserror::Error)]
pub enum VadError {
    /// The Silero ONNX session failed to load or run.
    #[error("silero: {0}")]
    Silero(String),
    /// A frame handed to the model was the wrong length (Silero v5 is strict).
    #[error("frame must be {expected} samples at 16 kHz, got {got}")]
    FrameSize { expected: usize, got: usize },
}
