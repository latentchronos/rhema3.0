//! On-device speech-to-text via `transcribe-cpp` (the ggml/GGUF runtime used by Handy).
//!
//! Feature-gated behind `local-stt` so the default cloud build never pulls in the
//! native ggml dependency. Loads a GGUF model (Parakeet or Cohere Transcribe) from a
//! local path and transcribes fully offline.
//!
//! ## Two engines, selected by `RHEMA_STT_STREAM`
//!
//! **Offline utterance mode (default).** For the `parakeet-unified` GGUF: its
//! cache-aware `ParakeetStream` mode is rejected and its buffered streaming re-encodes
//! the whole window each chunk (4–16x real-time — unusable). Offline `Session::run` is
//! ~34x real-time and accurate, so we segment audio into utterances by silence and
//! transcribe each offline — periodic previews as `Partial`, clean per-utterance
//! `Final`. Downside: each partial re-runs the whole buffer, so the live line refreshes
//! in ~0.35s steps and can rewrite earlier words.
//!
//! **True streaming mode (`RHEMA_STT_STREAM=1`).** For a *cache-aware streaming* GGUF
//! (`nemotron-3.5-asr-streaming-0.6b`): we feed audio to `Session::stream` and read its
//! append-only `committed` + volatile `tentative` snapshot. Words commit ~0.3s apart,
//! flicker-free, at ~1x real-time on ~1 core — the Handy-grade experience. This model
//! is multilingual and *requires* a locale language (`RHEMA_STT_LANG`, default `en-US`)
//! and an attention-right lookahead from its menu {0,3,6,13} (`RHEMA_STT_ATT_RIGHT`,
//! default 3 — the smoothest low-latency point; higher = more accurate but chunkier).
//! Verified end-to-end with a standalone spike before shipping.
//!
//! Build with `--features local-stt`. The STT crates are optimized even in dev builds
//! (see the per-package `opt-level` profile in the workspace manifest).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use crossbeam_channel::Receiver;
use tokio::sync::mpsc;
use transcribe_cpp::{
    CommitPolicy, Model, ParakeetStreamOptions, RunOptions, Session, SessionOptions,
    StreamExtension, StreamOptions, StreamState,
};

use crate::engine::SttEngine;
use crate::error::SttError;
use crate::types::TranscriptEvent;

const SR: usize = 16_000;
/// Below this RMS a chunk is treated as silence. Low enough to catch quiet speech
/// while still gating true silence / room noise.
const SILENCE_RMS: f32 = 0.006;
/// Trailing silence after speech that ends an utterance (~0.5s).
const SILENCE_HANG: usize = SR / 2;
/// Re-transcribe for a live preview after this much new audio (~0.35s). This is the
/// live-transcript refresh rate AND the first-word latency (the first partial fires
/// this long after speech onset). Kept small because decode has huge headroom on a
/// bounded window — a full 5s buffer decodes in ~160ms (~30x real-time), so even a
/// ~0.35s cadence runs <50% duty and never backs up the audio channel.
const PARTIAL_EVERY: usize = SR * 35 / 100;
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
        let streaming = std::env::var("RHEMA_STT_STREAM").as_deref() == Ok("1");
        // ggml inference is blocking and CPU-heavy: run the whole loop off the async pool.
        tokio::task::spawn_blocking(move || {
            if streaming {
                run_stream_loop(model_path, audio_rx, event_tx, keep_running)
            } else {
                run_loop(model_path, audio_rx, event_tx, keep_running)
            }
        })
        .await
        .map_err(|e| SttError::ConnectionFailed(format!("local stt task panicked: {e}")))?
    }
}

/// CPU threads for ggml. Left at the library default (all cores) it oversubscribes
/// against audio capture + detection + the Tauri UI. Leave a couple of cores free.
/// Overridable via `RHEMA_STT_THREADS`.
fn resolve_threads() -> i32 {
    std::env::var("RHEMA_STT_THREADS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            let cores = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4) as i32;
            (cores - 2).clamp(2, 8)
        })
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

    let n_threads = resolve_threads();
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

/// True-streaming engine for a cache-aware streaming GGUF (nemotron-3.5-asr-streaming).
///
/// Feeds audio to `Session::stream` and reads its append-only `committed` prefix +
/// volatile `tentative` tail. Committed sentences are emitted as `Final` (stable, drives
/// detection); the in-progress remainder + tentative is the live `Partial`. Because
/// `committed` never rewrites, the live line grows word-by-word without flicker.
fn run_stream_loop(
    model_path: String,
    audio_rx: Receiver<Vec<i16>>,
    event_tx: mpsc::Sender<TranscriptEvent>,
    keep_running: Arc<AtomicBool>,
) -> Result<(), SttError> {
    let _ = transcribe_cpp::init_backends_default();

    let model = Model::load(std::path::Path::new(&model_path))
        .map_err(|e| SttError::ConnectionFailed(format!("load model {model_path}: {e}")))?;
    let n_threads = resolve_threads();
    let mut session = model
        .session_with(&SessionOptions {
            n_threads,
            ..Default::default()
        })
        .map_err(|e| SttError::ConnectionFailed(format!("create session: {e}")))?;

    // This model is multilingual and emits only blanks without a locale language.
    let lang = std::env::var("RHEMA_STT_LANG").unwrap_or_else(|_| "en-US".to_string());
    // Attention-right lookahead, from the model's menu {0,3,6,13}. 3 = smoothest
    // low-latency; higher trades latency for accuracy.
    let att_right = std::env::var("RHEMA_STT_ATT_RIGHT")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(3);
    let run = RunOptions {
        language: Some(lang.clone()),
        ..Default::default()
    };
    let sopts = StreamOptions {
        commit_policy: CommitPolicy::Auto,
        family: Some(StreamExtension::ParakeetStream(ParakeetStreamOptions {
            att_context_right: Some(att_right),
        })),
        ..Default::default()
    };
    let mut stream = session
        .stream(&run, &sopts)
        .map_err(|e| SttError::ConnectionFailed(format!("stream begin (lang={lang}, att_right={att_right}): {e}")))?;

    let _ = event_tx.blocking_send(TranscriptEvent::Connected);
    log::info!("[LocalSTT] streaming mode ({n_threads} threads, lang={lang}, att_right={att_right}), model {model_path}");

    // Byte offset into the committed prefix already emitted as `Final`.
    let mut final_upto = 0usize;

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
                let update = match stream.feed(&batch) {
                    Ok(u) => u,
                    Err(e) => {
                        log::error!("[LocalSTT] stream feed failed: {e}");
                        break;
                    }
                };
                if stream.state() == StreamState::Failed {
                    log::error!("[LocalSTT] stream entered Failed state");
                    break;
                }
                if update.committed_changed || update.tentative_changed {
                    let txt = stream.text();
                    emit_stream_text(&event_tx, &txt.committed, &txt.tentative, &mut final_upto, false);
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }

    // Flush: commit whatever's buffered and emit the tail as a Final.
    if stream.finalize().is_ok() {
        let txt = stream.text();
        emit_stream_text(&event_tx, &txt.committed, &txt.tentative, &mut final_upto, true);
    }

    let _ = event_tx.blocking_send(TranscriptEvent::Disconnected);
    log::info!("[LocalSTT] streaming engine stopped");
    Ok(())
}

/// Turn a streaming snapshot into events: newly-completed committed sentences become
/// `Final`s; the remaining committed tail + `tentative` is the live `Partial`. On
/// `flush`, the whole remainder is emitted as a final `Final`. `final_upto` tracks how
/// much of `committed` has already been finalized (committed is append-only).
fn emit_stream_text(
    event_tx: &mpsc::Sender<TranscriptEvent>,
    committed: &str,
    tentative: &str,
    final_upto: &mut usize,
    flush: bool,
) {
    // Offsets are tracked against the RAW committed string (append-only, stable), so
    // tag-stripping — which changes byte lengths — is applied only to emitted text.
    let upto = (*final_upto).min(committed.len());

    // Emit each completed sentence in the newly-committed region as its own Final.
    let mut cut = upto;
    while let Some(end) = next_sentence_end(committed, cut) {
        let seg = strip_tags(&committed[cut..end]);
        let seg = seg.trim();
        if !seg.is_empty() {
            send_final(event_tx, seg.to_string());
        }
        cut = end;
    }
    *final_upto = cut;

    let remainder = strip_tags(&format!("{}{}", &committed[cut..], tentative));
    let remainder = remainder.trim();
    if flush {
        if !remainder.is_empty() {
            send_final(event_tx, remainder.to_string());
        }
    } else if !remainder.is_empty() {
        send_partial(event_tx, remainder.to_string());
    }
}

/// Byte index just past the next sentence-ending punctuation at or after `from`.
fn next_sentence_end(s: &str, from: usize) -> Option<usize> {
    s[from..]
        .find(['.', '?', '!'])
        .map(|rel| from + rel + 1)
}

/// Drop end-of-utterance / end-of-burst control tokens the model may surface.
fn strip_tags(s: &str) -> String {
    s.replace("<EOU>", "").replace("<EOB>", "")
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
