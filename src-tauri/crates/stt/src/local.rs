//! On-device speech-to-text via `transcribe-cpp` (the ggml/GGUF runtime used by Handy).
//!
//! Feature-gated behind `local-stt` so the default cloud build never pulls in the
//! native ggml dependency. Loads a GGUF model (Parakeet or Cohere Transcribe) from a
//! local path and transcribes fully offline.
//!
//! Uses the crate's **streaming** API (`Session::stream` → `feed` → `text`) — the same
//! low-latency path Handy drives — feeding 100 ms chunks and emitting the model's
//! `tentative` (interim) text as [`TranscriptEvent::Partial`] and newly `committed`
//! text as [`TranscriptEvent::Final`]. This mirrors Deepgram's partial/final semantics
//! that the frontend already renders.
//!
//! NOTE: build in release (`--release`) for live use — the RNN-T predictor/joint are
//! CPU loops that are an order of magnitude slower in a debug build.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use crossbeam_channel::Receiver;
use tokio::sync::mpsc;
use transcribe_cpp::{CommitPolicy, Model, RunOptions, StreamOptions};

use crate::engine::SttEngine;
use crate::error::SttError;
use crate::types::TranscriptEvent;

const SAMPLE_RATE: usize = 16_000;
/// Feed the model in 100 ms chunks (1600 samples @ 16 kHz) — the cadence used by the
/// crate's streaming example. Small enough to feel live, large enough to be efficient.
const CHUNK_SAMPLES: usize = SAMPLE_RATE / 10;

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

    // Auto commit policy: the model decides when a token prefix is stable enough to
    // move from `tentative` → `committed`, giving smooth incremental output.
    let opts = StreamOptions {
        commit_policy: CommitPolicy::Auto,
        ..Default::default()
    };
    let mut stream = session
        .stream(&RunOptions::default(), &opts)
        .map_err(|e| SttError::ConnectionFailed(format!("open stream: {e}")))?;

    let _ = event_tx.blocking_send(TranscriptEvent::Connected);
    log::info!("[LocalSTT] streaming {} (100ms chunks)", model_path);

    // f32 staging buffer; we feed exactly CHUNK_SAMPLES at a time.
    let mut pending: Vec<f32> = Vec::with_capacity(CHUNK_SAMPLES * 4);
    // Track what we've already emitted so we only send deltas.
    let mut emitted_committed = String::new();
    let mut last_partial = String::new();

    loop {
        if !keep_running.load(Ordering::SeqCst) {
            break;
        }
        match audio_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(samples) => {
                extend_f32(&mut pending, &samples);
                // Drain anything else already queued so we never fall behind real-time.
                while let Ok(more) = audio_rx.try_recv() {
                    extend_f32(&mut pending, &more);
                }
                feed_full_chunks(&mut stream, &mut pending, &event_tx, &mut emitted_committed, &mut last_partial);
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // Idle gap: flush a short remainder so trailing words aren't stuck.
                if !pending.is_empty() {
                    if let Err(e) = stream.feed(&pending) {
                        log::error!("[LocalSTT] feed failed: {e}");
                    }
                    pending.clear();
                    emit(&stream, &event_tx, &mut emitted_committed, &mut last_partial);
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }

    // Flush + finalize so the last partial becomes committed text.
    if !pending.is_empty() {
        let _ = stream.feed(&pending);
    }
    if let Err(e) = stream.finalize() {
        log::error!("[LocalSTT] finalize failed: {e}");
    }
    emit(&stream, &event_tx, &mut emitted_committed, &mut last_partial);

    let _ = event_tx.blocking_send(TranscriptEvent::Disconnected);
    log::info!("[LocalSTT] engine stopped");
    Ok(())
}

/// Append i16 PCM as f32 in [-1, 1] (capture is already 16 kHz mono).
fn extend_f32(dst: &mut Vec<f32>, src: &[i16]) {
    dst.extend(src.iter().map(|&s| s as f32 / 32768.0));
}

/// Feed every complete 100 ms chunk currently staged, retaining the remainder.
fn feed_full_chunks(
    stream: &mut transcribe_cpp::Stream,
    pending: &mut Vec<f32>,
    event_tx: &mpsc::Sender<TranscriptEvent>,
    emitted_committed: &mut String,
    last_partial: &mut String,
) {
    let mut off = 0;
    while pending.len() - off >= CHUNK_SAMPLES {
        if let Err(e) = stream.feed(&pending[off..off + CHUNK_SAMPLES]) {
            log::error!("[LocalSTT] feed failed: {e}");
        }
        off += CHUNK_SAMPLES;
    }
    if off > 0 {
        pending.drain(0..off);
        emit(stream, event_tx, emitted_committed, last_partial);
    }
}

/// Emit committed deltas as Final and the tentative tail as Partial (deduplicated).
fn emit(
    stream: &transcribe_cpp::Stream,
    event_tx: &mpsc::Sender<TranscriptEvent>,
    emitted_committed: &mut String,
    last_partial: &mut String,
) {
    let text = stream.text();

    if text.committed != *emitted_committed {
        let delta = text
            .committed
            .strip_prefix(emitted_committed.as_str())
            .unwrap_or(&text.committed)
            .trim()
            .to_string();
        *emitted_committed = text.committed.clone();
        if !delta.is_empty() {
            let _ = event_tx.blocking_send(TranscriptEvent::Final {
                transcript: delta,
                words: Vec::new(),
                confidence: 1.0,
                speech_final: true,
            });
        }
    }

    let tentative = text.tentative.trim();
    if !tentative.is_empty() && tentative != *last_partial {
        *last_partial = tentative.to_string();
        let _ = event_tx.blocking_send(TranscriptEvent::Partial {
            transcript: tentative.to_string(),
            words: Vec::new(),
        });
    }
}
