//! On-device speech-to-text via `transcribe-cpp` (the ggml/GGUF runtime used by Handy).
//!
//! Feature-gated behind `local-stt` so the default cloud build never pulls in the
//! native ggml dependency. Loads a GGUF model (Parakeet or Cohere Transcribe) from a
//! local path and transcribes fully offline.
//!
//! ## Why utterance-segmented offline `run()` (not the streaming API)
//! The Parakeet "unified" GGUF is exported with *unlimited* attention context, so
//! transcribe-cpp's cache-aware `ParakeetStream` mode is rejected, and its buffered
//! streaming reprocesses all history each step (O(N²) — unusable for long sermons).
//! Offline `Session::run`, by contrast, is ~34x real-time and accurate. So we do what
//! Handy effectively does for this model: segment audio into utterances by silence and
//! transcribe each utterance offline — emitting periodic previews as `Partial` and a
//! clean per-utterance `Final`. Bounded compute, no O(N²), a few seconds of latency.
//!
//! Build with `--features local-stt`. The STT crates are optimized even in dev builds
//! (see the per-package `opt-level` profile in the workspace manifest).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use crossbeam_channel::Receiver;
use tokio::sync::mpsc;
use transcribe_cpp::{Model, RunOptions, Session, SessionOptions};

use crate::engine::SttEngine;
use crate::error::SttError;
use crate::types::TranscriptEvent;

const SR: usize = 16_000;
/// Below this RMS a chunk is treated as silence. Low enough to catch quiet speech
/// while still gating true silence / room noise.
const SILENCE_RMS: f32 = 0.006;
/// Trailing silence after speech that ends an utterance (~0.5s).
const SILENCE_HANG: usize = SR / 2;
/// Re-transcribe for a live preview after this much new audio (~0.8s).
const PARTIAL_EVERY: usize = SR * 4 / 5;
/// Hard commit window: force-finalize a continuous utterance this long even without
/// a pause (~5s). This is the single most important latency/cost knob — it bounds
/// BOTH how long continuous speech can go before a `Final` appears AND the largest
/// buffer any single `run()` ever sees. Every partial re-transcribes the current
/// buffer from scratch, so an unbounded window makes per-run cost grow without limit
/// (the 6.8s stalls we saw at ~14s buffers). Keep this small.
const MAX_UTTERANCE: usize = SR * 5;
/// Require at least this much speech before finalizing (avoids empty blips, ~0.3s).
const MIN_SPEECH: usize = SR * 3 / 10;
/// While waiting for speech, keep at most this much trailing audio so leading
/// silence never accumulates unbounded (~0.5s of pre-roll context).
const PREROLL: usize = SR / 2;

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

    // Pin ggml's CPU thread count. Left at the library default (all cores) it
    // oversubscribes against audio capture + detection + the Tauri UI, which is what
    // blew one decode's joint step up to ~5s. Leave a couple of cores free.
    let n_threads = std::env::var("RHEMA_STT_THREADS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            let cores = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4) as i32;
            (cores - 2).clamp(2, 8)
        });
    let mut session = model
        .session_with(&SessionOptions {
            n_threads,
            ..Default::default()
        })
        .map_err(|e| SttError::ConnectionFailed(format!("create session: {e}")))?;

    let _ = event_tx.blocking_send(TranscriptEvent::Connected);
    log::info!(
        "[LocalSTT] utterance mode ({n_threads} threads, {}s window), model {}",
        MAX_UTTERANCE / SR,
        model_path
    );

    let mut utter: Vec<f32> = Vec::with_capacity(MAX_UTTERANCE);
    let mut had_speech = false;
    let mut silence: usize = 0; // trailing silence samples since last speech
    let mut since_partial: usize = 0; // new samples since last preview run

    loop {
        if !keep_running.load(Ordering::SeqCst) {
            break;
        }
        match audio_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(samples) => {
                let mut batch = to_f32(&samples);
                while let Ok(more) = audio_rx.try_recv() {
                    batch.extend(to_f32(&more));
                }
                let voiced = rms(&batch) > SILENCE_RMS;
                if voiced {
                    had_speech = true;
                    silence = 0;
                } else {
                    silence += batch.len();
                }
                utter.extend_from_slice(&batch);
                since_partial += batch.len();

                if had_speech
                    && utter.len() >= MIN_SPEECH
                    && (silence >= SILENCE_HANG || utter.len() >= MAX_UTTERANCE)
                {
                    // Utterance boundary → transcribe and commit.
                    if let Some(text) = transcribe(&mut session, &utter) {
                        send_final(&event_tx, text);
                    }
                    utter.clear();
                    had_speech = false;
                    silence = 0;
                    since_partial = 0;
                } else if had_speech && since_partial >= PARTIAL_EVERY {
                    // Live preview of the utterance so far.
                    if let Some(text) = transcribe(&mut session, &utter) {
                        send_partial(&event_tx, text);
                    }
                    since_partial = 0;
                } else if !had_speech && utter.len() > PREROLL {
                    // Drop accumulated pre-speech silence, keep a little pre-roll context.
                    let drop = utter.len() - PREROLL;
                    utter.drain(0..drop);
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // Idle: if we have buffered speech, flush it as a final utterance.
                if had_speech && utter.len() >= MIN_SPEECH {
                    if let Some(text) = transcribe(&mut session, &utter) {
                        send_final(&event_tx, text);
                    }
                    utter.clear();
                    had_speech = false;
                    silence = 0;
                    since_partial = 0;
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }

    // Flush any trailing speech on stop.
    if had_speech && utter.len() >= MIN_SPEECH {
        if let Some(text) = transcribe(&mut session, &utter) {
            send_final(&event_tx, text);
        }
    }

    let _ = event_tx.blocking_send(TranscriptEvent::Disconnected);
    log::info!("[LocalSTT] engine stopped");
    Ok(())
}

/// Run offline transcription over `pcm`, returning trimmed non-empty text.
fn transcribe(session: &mut Session, pcm: &[f32]) -> Option<String> {
    match session.run(pcm, &RunOptions::default()) {
        Ok(r) => {
            let t = r.text.trim().to_string();
            if t.is_empty() {
                None
            } else {
                Some(t)
            }
        }
        Err(e) => {
            log::error!("[LocalSTT] transcription failed: {e}");
            None
        }
    }
}

fn send_final(event_tx: &mpsc::Sender<TranscriptEvent>, transcript: String) {
    let _ = event_tx.blocking_send(TranscriptEvent::Final {
        transcript,
        words: Vec::new(),
        confidence: 1.0,
        speech_final: true,
    });
}

fn send_partial(event_tx: &mpsc::Sender<TranscriptEvent>, transcript: String) {
    let _ = event_tx.blocking_send(TranscriptEvent::Partial {
        transcript,
        words: Vec::new(),
    });
}

/// i16 PCM → f32 in [-1, 1] (capture is already 16 kHz mono).
fn to_f32(src: &[i16]) -> Vec<f32> {
    src.iter().map(|&s| s as f32 / 32768.0).collect()
}

fn rms(pcm: &[f32]) -> f32 {
    if pcm.is_empty() {
        return 0.0;
    }
    (pcm.iter().map(|&s| s * s).sum::<f32>() / pcm.len() as f32).sqrt()
}
