//! On-device speech-to-text via `transcribe-cpp` (the ggml/GGUF runtime used by Handy).
//!
//! Feature-gated behind `local-stt` so the default cloud build never pulls in the
//! native ggml dependency. Loads a GGUF model (e.g. Parakeet or Cohere Transcribe)
//! from a local path and transcribes fully offline.
//!
//! v1 uses the offline `Session::run` on fixed audio windows — the exact API path
//! validated by the standalone spike (load → session → run → text). It transcribes
//! ~34x faster than real-time on CPU, so windowed batching gives usable latency.
//! The lower-latency streaming API (`Session::stream` / `feed` / `get_text`) is the
//! next optimization and slots in behind this same [`SttEngine`] impl.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use crossbeam_channel::Receiver;
use tokio::sync::mpsc;
use transcribe_cpp::{Model, RunOptions, Session};

use crate::engine::SttEngine;
use crate::error::SttError;
use crate::types::TranscriptEvent;

const SAMPLE_RATE: usize = 16_000;
/// Seconds of audio to accumulate before running one offline transcription pass.
/// Balances latency against giving the model enough acoustic context.
const WINDOW_SECS: usize = 3;
const WINDOW_SAMPLES: usize = WINDOW_SECS * SAMPLE_RATE;

/// On-device STT engine backed by a local GGUF model file.
pub struct LocalSttClient {
    model_path: String,
}

impl LocalSttClient {
    /// `model_path` is an absolute path to a `.gguf` model (Parakeet or Cohere Transcribe).
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
        }
    }
}

#[async_trait]
impl SttEngine for LocalSttClient {
    async fn connect(
        &self,
        audio_rx: Receiver<Vec<i16>>,
        event_tx: mpsc::Sender<TranscriptEvent>,
        keep_running: Arc<AtomicBool>,
    ) -> Result<(), SttError> {
        let model_path = self.model_path.clone();
        // ggml inference is blocking and CPU-heavy: run the whole loop off the async pool.
        tokio::task::spawn_blocking(move || run_loop(model_path, audio_rx, event_tx, keep_running))
            .await
            .map_err(|e| SttError::ConnectionFailed(format!("local stt task panicked: {e}")))?
    }
}

fn run_loop(
    model_path: String,
    audio_rx: Receiver<Vec<i16>>,
    event_tx: mpsc::Sender<TranscriptEvent>,
    keep_running: Arc<AtomicBool>,
) -> Result<(), SttError> {
    let _ = transcribe_cpp::init_backends_default();

    let model = Model::load(std::path::Path::new(&model_path))
        .map_err(|e| SttError::ConnectionFailed(format!("load model {model_path}: {e}")))?;
    let mut session = model
        .session()
        .map_err(|e| SttError::ConnectionFailed(format!("create session: {e}")))?;

    let _ = event_tx.blocking_send(TranscriptEvent::Connected);
    log::info!(
        "[LocalSTT] loaded {} — transcribing in {}s windows",
        model_path,
        WINDOW_SECS
    );

    let mut buf: Vec<i16> = Vec::with_capacity(WINDOW_SAMPLES);
    loop {
        if !keep_running.load(Ordering::SeqCst) {
            break;
        }
        match audio_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(samples) => {
                buf.extend_from_slice(&samples);
                if buf.len() >= WINDOW_SAMPLES {
                    transcribe_window(&mut session, &buf, &event_tx);
                    buf.clear();
                }
            }
            // Silence gap: flush trailing audio so the tail of an utterance isn't dropped.
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                if !buf.is_empty() {
                    transcribe_window(&mut session, &buf, &event_tx);
                    buf.clear();
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }

    let _ = event_tx.blocking_send(TranscriptEvent::Disconnected);
    log::info!("[LocalSTT] engine stopped");
    Ok(())
}

fn transcribe_window(session: &mut Session, pcm_i16: &[i16], event_tx: &mpsc::Sender<TranscriptEvent>) {
    // transcribe-cpp expects 16 kHz mono f32 in [-1, 1]; capture is already 16 kHz mono i16.
    let pcm: Vec<f32> = pcm_i16.iter().map(|&s| s as f32 / 32768.0).collect();
    match session.run(&pcm, &RunOptions::default()) {
        Ok(result) => {
            let text = result.text.trim().to_string();
            if !text.is_empty() {
                let _ = event_tx.blocking_send(TranscriptEvent::Final {
                    transcript: text,
                    words: Vec::new(),
                    confidence: 1.0,
                    speech_final: true,
                });
            }
        }
        Err(e) => {
            log::error!("[LocalSTT] transcription failed: {e}");
            let _ = event_tx.blocking_send(TranscriptEvent::Error(format!("local stt: {e}")));
        }
    }
}
