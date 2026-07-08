//! Silero VAD v5 ONNX session.
//!
//! Loads `silero_vad.onnx` (v5, opset 16) and runs one 16 kHz frame per call, returning a
//! speech probability in `[0, 1]`. Stateful across calls: it carries the recurrent `state`
//! tensor `(2, 1, 128)` and a 64-sample `context` buffer that is prepended to each frame
//! (input length 64 + 512 = 576) — matching the Python reference. The Rust
//! `voice_activity_detector` crate drops that context and relies on the recurrent state
//! alone; we keep it for parity with upstream.
//!
//! Model I/O (verified from the graph): inputs `input` `[1, 576]` f32, `state` `[2,1,128]`
//! f32, `sr` scalar i64; outputs `output` (prob) and `stateN` (next state).

use std::path::Path;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::error::VadError;
use crate::{CONTEXT_SAMPLES, FRAME_SAMPLES, SAMPLE_RATE};

const STATE_LEN: usize = 2 * 1 * 128;

/// A loaded Silero VAD session plus its per-stream recurrent state.
pub struct SileroVad {
    session: Session,
    /// Recurrent hidden state `(2, 1, 128)`, flattened. Zeroed at reset.
    state: Vec<f32>,
    /// Last [`CONTEXT_SAMPLES`] samples of the previous frame, prepended to the next.
    context: Vec<f32>,
    sr: i64,
}

impl SileroVad {
    /// Load the Silero v5 ONNX model. Single-threaded session — the model is tiny and
    /// per-frame cost is sub-millisecond, so extra threads only add contention (this
    /// matches both the Python reference and the Rust crate).
    pub fn load(model_path: &Path) -> Result<Self, VadError> {
        let session = Session::builder()
            .map_err(|e| VadError::Silero(format!("session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| VadError::Silero(format!("optimization level: {e}")))?
            .with_intra_threads(1)
            .map_err(|e| VadError::Silero(format!("intra threads: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| VadError::Silero(format!("load {}: {e}", model_path.display())))?;

        log::info!("[VAD] Silero loaded: {}", model_path.display());
        Ok(Self {
            session,
            state: vec![0.0; STATE_LEN],
            context: vec![0.0; CONTEXT_SAMPLES],
            sr: SAMPLE_RATE as i64,
        })
    }

    /// Run one frame (exactly [`FRAME_SAMPLES`] at 16 kHz) and return its speech
    /// probability. Advances the recurrent state and context.
    pub fn process(&mut self, frame: &[f32]) -> Result<f32, VadError> {
        if frame.len() != FRAME_SAMPLES {
            return Err(VadError::FrameSize {
                expected: FRAME_SAMPLES,
                got: frame.len(),
            });
        }

        // input = context (64) ++ frame (512) = 576.
        let mut input = Vec::with_capacity(CONTEXT_SAMPLES + FRAME_SAMPLES);
        input.extend_from_slice(&self.context);
        input.extend_from_slice(frame);

        let input_tensor = Tensor::from_array((
            vec![1i64, input.len() as i64],
            input.clone(),
        ))
        .map_err(|e| VadError::Silero(format!("input tensor: {e}")))?;
        let state_tensor = Tensor::from_array((vec![2i64, 1, 128], self.state.clone()))
            .map_err(|e| VadError::Silero(format!("state tensor: {e}")))?;
        // `sr` is a rank-0 (scalar) int64 in the graph.
        let sr_tensor = Tensor::from_array((Vec::<i64>::new(), vec![self.sr]))
            .map_err(|e| VadError::Silero(format!("sr tensor: {e}")))?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input" => input_tensor,
                "state" => state_tensor,
                "sr" => sr_tensor,
            ])
            .map_err(|e| VadError::Silero(format!("run: {e}")))?;

        let (_, prob_data) = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(|e| VadError::Silero(format!("extract output: {e}")))?;
        let prob = prob_data.first().copied().unwrap_or(0.0);

        let (_, state_data) = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .map_err(|e| VadError::Silero(format!("extract stateN: {e}")))?;
        self.state = state_data.to_vec();

        // Next context = the last CONTEXT_SAMPLES of this input (i.e. the tail of `frame`).
        let tail = input.len() - CONTEXT_SAMPLES;
        self.context.clear();
        self.context.extend_from_slice(&input[tail..]);

        Ok(prob)
    }

    /// Reset recurrent state + context (new stream / new service / long gap).
    pub fn reset(&mut self) {
        self.state = vec![0.0; STATE_LEN];
        self.context = vec![0.0; CONTEXT_SAMPLES];
    }
}
