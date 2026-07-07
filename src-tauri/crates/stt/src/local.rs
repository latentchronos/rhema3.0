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
//! default 13 — ~1s lookahead. On CPU without a GPU this is the floor that stays ahead
//! of real time with the Q5 model AND reads most accurately; 6 and 3 drift behind here).
//! Verified end-to-end with a standalone spike before shipping.
//!
//! Build with `--features local-stt`. The STT crates are optimized even in dev builds
//! (see the per-package `opt-level` profile in the workspace manifest).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use crossbeam_channel::Receiver;
use tokio::sync::mpsc;
use transcribe_cpp::{
    CommitPolicy, Model, ParakeetStreamOptions, RunOptions, Session, SessionOptions,
    StreamExtension, StreamOptions, StreamState, TimestampKind,
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
/// Re-transcribe for a live preview after this much new audio (~0.2s). This is the
/// live-transcript refresh rate AND the first-word latency (the first partial fires
/// this long after speech onset). Kept small because offline decode has huge headroom
/// on this weak CPU — a bounded ~1.6s window decodes in a few ms at ~30x real-time, so
/// even a ~0.2s cadence runs well under 100% duty and never backs up the audio channel.
/// This is the key perceived-latency knob for the offline engine (streaming can't go
/// below its att_right lookahead, but offline is bounded only by this + decode).
const PARTIAL_EVERY: usize = SR / 5;
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
/// Partial previews re-encode only the last ~1.6s of the utterance, not the whole
/// (growing) thing. `session.run` re-runs the *encoder* over its entire input every
/// call, and that cost grows with utterance length — a partial near the 5s cap was
/// re-encoding 5s of audio (seconds of CPU, hidden from the decoder-only timing log),
/// which fell behind real time and backed audio up. Capping the preview window keeps
/// per-partial cost flat so the engine stays ahead of the mic. The Final still encodes
/// the whole utterance for an accurate committed line.
const PARTIAL_WINDOW: usize = SR * 8 / 5;
/// Streaming only: flush a run of committed-but-not-yet-finalized text as a `Final` once
/// it exceeds this many chars, even without sentence punctuation. The streaming model
/// rarely emits periods, so otherwise the un-finalized remainder grows without bound and
/// every `Partial` re-sends the whole thing — making the verse detector re-scan the entire
/// transcript on every tick (the flood of repeated `Found`/`suppressed` lines). Committed
/// text is append-only and stable, so finalizing it early is safe; it just means the
/// detector sees each chunk once instead of on every update.
const COMMIT_FLUSH_CHARS: usize = 80;

// ── Streaming auto-recovery (Rhema v2, Phase 1) ─────────────────────────────
// A live sermon cannot tolerate "transcription stopped and won't come back". When the
// native stream faults (`StreamState::Failed` or a `feed` error) we recreate ONLY the
// stream — the model + session stay loaded, so recovery costs ~ms, not a multi-second
// reload. This mirrors the Deepgram client's reconnect loop (deepgram.rs) on the local
// path. Bounded by a restart budget so a genuinely-dead model still surfaces an error
// instead of spinning forever.
/// Max consecutive stream (re)creation failures before giving up and emitting an Error.
const MAX_STREAM_RESTARTS: u32 = 5;
/// Backoff between a stream fault and recreating it. Small — the model is already warm.
const STREAM_RESTART_BACKOFF: Duration = Duration::from_millis(250);
/// If a stream ran healthy at least this long before faulting, the restart budget resets
/// (an isolated blip should not count toward the give-up threshold of a later fault run).
const STREAM_HEALTHY_RESET: Duration = Duration::from_secs(30);
/// Recent audio retained to re-feed on recovery, warming the fresh stream's encoder cache
/// so the first post-recovery words aren't cold-start garbage (~1s). Replayed audio is
/// NOT re-emitted (final_upto is advanced past it), so recovery never duplicates text.
const PREROLL_RECOVER: usize = SR;

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
        let streaming = std::env::var("RHEMA_STT_STREAM").as_deref() == Ok("1");
        // When RHEMA_STT_MODEL is unset, fall back to the config we settled on for this
        // hardware: streaming → nemotron Q5 (accurate + keeps pace at att_right=13),
        // offline → parakeet Q8 (accurate, ~30x-realtime decode). Lets the app run with
        // just RHEMA_STT_PROVIDER=local [RHEMA_STT_STREAM=1] and no model path.
        let model_path = {
            let p = self.model_path.clone();
            if p.trim().is_empty() {
                default_model_path(streaming)
            } else {
                p
            }
        };
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

/// Default GGUF path when `RHEMA_STT_MODEL` is unset — the two configs we validated on
/// this CPU. Resolved relative to the repo's `model/` dir (this crate lives at
/// `src-tauri/crates/stt`, so the repo root is three levels up). Overridable any time by
/// setting `RHEMA_STT_MODEL` to an absolute path.
fn default_model_path(streaming: bool) -> String {
    let file = if streaming {
        "nemotron-3.5-asr-streaming-0.6b-Q5_K_M.gguf"
    } else {
        "parakeet-unified-en-0.6b-Q8_0.gguf"
    };
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../model")
        .join(file)
        .to_string_lossy()
        .into_owned()
}

/// Diagnostic: how far behind real time the engine is running. Audio is produced at
/// 1x, so `(wall time since first sample) - (audio seconds consumed)` is roughly the
/// unprocessed backlog. Also reports the channel depth (`queued` frames) — the most
/// direct signal: near the channel capacity means we're maxed out and lagging.
/// Zero behaviour change; logs at most every 2s.
struct BacklogMeter {
    start: Option<Instant>,
    consumed: u64,
    last_report: Instant,
}

impl BacklogMeter {
    fn new() -> Self {
        Self {
            start: None,
            consumed: 0,
            last_report: Instant::now(),
        }
    }

    fn record(&mut self, samples: usize, queued: usize) {
        let start = *self.start.get_or_insert_with(Instant::now);
        self.consumed += samples as u64;
        if self.last_report.elapsed() >= Duration::from_secs(2) {
            let produced = start.elapsed().as_secs_f32();
            let consumed_s = self.consumed as f32 / SR as f32;
            log::info!(
                "[LocalSTT] backlog ~{:.1}s behind real-time (queue {queued} frames)",
                (produced - consumed_s).max(0.0),
            );
            self.last_report = Instant::now();
        }
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
    let mut meter = BacklogMeter::new();

    loop {
        if !keep_running.load(Ordering::SeqCst) {
            break;
        }
        match audio_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(samples) => {
                // Frames still waiting after this one = the current backlog depth.
                let queued = audio_rx.len();
                let mut batch = to_f32(&samples);
                while let Ok(more) = audio_rx.try_recv() {
                    batch.extend(to_f32(&more));
                }
                meter.record(batch.len(), queued);
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
                    if let Some((text, words, conf)) = transcribe(&mut session, &utter, true) {
                        send_final(&event_tx, text, words, conf);
                    }
                    utter.clear();
                    had_speech = false;
                    silence = 0;
                    since_partial = 0;
                } else if had_speech && since_partial >= PARTIAL_EVERY {
                    // Live preview: re-encode only the recent window, not the whole
                    // utterance, so per-partial cost stays flat and the engine keeps up.
                    let start = utter.len().saturating_sub(PARTIAL_WINDOW);
                    if let Some((text, _, _)) = transcribe(&mut session, &utter[start..], false) {
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
                    if let Some((text, words, conf)) = transcribe(&mut session, &utter, true) {
                        send_final(&event_tx, text, words, conf);
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
        if let Some((text, words, conf)) = transcribe(&mut session, &utter, true) {
            send_final(&event_tx, text, words, conf);
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
    // Attention-right lookahead, from the model's menu {0,3,6,13}. Each step is ~80ms of
    // built-in delay before a word can commit. On this CPU (i5-8265U, no GPU) 13 (~1s) is
    // the only value that stays ahead of real time with Q5 AND reads most accurately (more
    // right-context per word): measured queue rock-solid 17-33. 6 (~480ms) drifts — queue
    // climbs to ~130-170 even with Q4 — and 3 runs away entirely. So 13 is the floor here,
    // not a compromise; lower values are laggier in practice because they fall behind.
    let att_right = std::env::var("RHEMA_STT_ATT_RIGHT")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(13);
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
    // Discard the audio that piled up in the channel while the model was loading (several
    // seconds — capture starts before the model is ready). The streaming model at low
    // att_right runs near 1x real-time with little headroom: it can keep pace but cannot
    // *catch up*, so without this it would carry this entire startup backlog as permanent
    // latency (the ~100-frame standing queue we measured). This drops only the pre-ready
    // warm-up window — transcription is not live until this point — never mid-speech audio.
    let mut warmup = 0usize;
    while let Ok(chunk) = audio_rx.try_recv() {
        warmup += chunk.len();
    }
    if warmup > 0 {
        log::info!(
            "[LocalSTT] discarded {:.1}s of audio buffered during model load (warm-up)",
            warmup as f32 / SR as f32
        );
    }

    // Cross-restart state. `final_upto` is a byte offset into the CURRENT stream's
    // committed prefix already emitted as `Final`; it resets to 0 whenever we recreate the
    // stream (a fresh stream starts with empty committed text). `preroll` retains the most
    // recent audio so a recovered stream can warm its encoder cache.
    let mut final_upto = 0usize;
    let mut meter = BacklogMeter::new();
    let mut preroll: VecDeque<f32> = VecDeque::with_capacity(PREROLL_RECOVER);
    let mut restart_attempts: u32 = 0;
    let mut connected_announced = false;

    // Outer lifecycle loop: each iteration owns exactly one native stream. A stream FAULT
    // (`feed` error or `StreamState::Failed`) breaks the inner feed loop and falls through
    // to recreate the stream; a CLEAN stop (user stop / audio channel closed) finalizes and
    // exits. The model + session persist across iterations, so recovery is ~ms, not a
    // multi-second reload. This is the local-path analogue of the Deepgram reconnect loop.
    'session: loop {
        if !keep_running.load(Ordering::SeqCst) {
            break;
        }

        // (Re)create the stream. Cheap relative to model load — the session stays warm.
        let mut stream = match session.stream(&run, &sopts) {
            Ok(s) => s,
            Err(e) => {
                restart_attempts += 1;
                log::error!(
                    "[LocalSTT] stream begin failed (attempt {restart_attempts}/{MAX_STREAM_RESTARTS}, lang={lang}, att_right={att_right}): {e}"
                );
                if restart_attempts >= MAX_STREAM_RESTARTS {
                    let _ = event_tx.blocking_send(TranscriptEvent::Error(format!(
                        "local STT stream could not start after {MAX_STREAM_RESTARTS} attempts: {e}"
                    )));
                    break 'session;
                }
                std::thread::sleep(STREAM_RESTART_BACKOFF);
                continue 'session;
            }
        };
        let stream_started = Instant::now();

        if !connected_announced {
            connected_announced = true;
            let _ = event_tx.blocking_send(TranscriptEvent::Connected);
            log::info!(
                "[LocalSTT] streaming mode ({n_threads} threads, lang={lang}, att_right={att_right}), model {model_path}"
            );
        } else {
            // Recovery path. A fresh stream's committed prefix is empty, so the emitted
            // offset resets. Warm the new stream's encoder cache with the recent pre-roll,
            // but do NOT re-emit it — advance `final_upto` past whatever the replay commits
            // so recovery stays loss-tolerant (it may drop the ~1s in flight at the fault)
            // rather than duplicate-prone. An informational Error tells the UI we recovered.
            final_upto = 0;
            let _ = event_tx.blocking_send(TranscriptEvent::Error(
                "local STT recovered from a stream fault".to_string(),
            ));
            if !preroll.is_empty() {
                let pr: Vec<f32> = preroll.iter().copied().collect();
                log::warn!(
                    "[LocalSTT] stream recovered; replaying {:.1}s pre-roll to warm cache",
                    pr.len() as f32 / SR as f32
                );
                if stream.feed(&pr).is_ok() {
                    // Skip re-emitting the replayed transcript.
                    final_upto = stream.text().committed.len();
                }
            }
        }

        // Inner feed loop. Yields `true` on a stream fault (→ recreate) or `false` on a
        // clean stop (→ finalize + exit the lifecycle).
        let faulted = 'feed: loop {
            if !keep_running.load(Ordering::SeqCst) {
                break 'feed false;
            }
            match audio_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(samples) => {
                    let queued = audio_rx.len();
                    let mut batch = to_f32(&samples);
                    while let Ok(more) = audio_rx.try_recv() {
                        batch.extend(to_f32(&more));
                    }
                    meter.record(batch.len(), queued);
                    push_preroll(&mut preroll, &batch);
                    let update = match stream.feed(&batch) {
                        Ok(u) => u,
                        Err(e) => {
                            log::error!("[LocalSTT] stream feed failed: {e}");
                            break 'feed true;
                        }
                    };
                    if stream.state() == StreamState::Failed {
                        log::error!("[LocalSTT] stream entered Failed state");
                        break 'feed true;
                    }
                    if update.committed_changed || update.tentative_changed {
                        let txt = stream.text();
                        emit_stream_text(
                            &event_tx,
                            &txt.committed,
                            &txt.tentative,
                            &mut final_upto,
                            false,
                        );
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break 'feed false,
            }
        };

        if !faulted {
            // Clean stop: flush and emit the tail as a Final, then exit the lifecycle.
            if stream.finalize().is_ok() {
                let txt = stream.text();
                emit_stream_text(&event_tx, &txt.committed, &txt.tentative, &mut final_upto, true);
            }
            break 'session;
        }

        // Fault path: an isolated blip after a long healthy run resets the restart budget
        // so it does not count toward a later fault run's give-up threshold.
        if stream_started.elapsed() >= STREAM_HEALTHY_RESET {
            restart_attempts = 0;
        }
        restart_attempts += 1;
        if restart_attempts >= MAX_STREAM_RESTARTS {
            log::error!(
                "[LocalSTT] stream faulted {restart_attempts} times consecutively; giving up"
            );
            let _ = event_tx.blocking_send(TranscriptEvent::Error(format!(
                "local STT stream failed repeatedly ({restart_attempts}x); transcription stopped"
            )));
            break 'session;
        }
        log::warn!(
            "[LocalSTT] recreating stream after fault (attempt {restart_attempts}/{MAX_STREAM_RESTARTS})"
        );
        std::thread::sleep(STREAM_RESTART_BACKOFF);
        // `stream` drops here at end of iteration, releasing its &mut borrow of `session`.
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

    // Emit each completed phrase in the newly-committed region as its own Final, cutting at
    // the model's own end-of-utterance token *or* sentence punctuation (whichever comes
    // first). Streaming Finals don't yet carry per-word timing/confidence — the streaming
    // snapshot's words would need committed-token alignment (deferred to the confidence-
    // gated projection step); until then these finalize with unknown confidence (1.0).
    let mut cut = upto;
    while let Some(end) = next_boundary(committed, cut) {
        let seg = strip_tags(&committed[cut..end]);
        let seg = seg.trim();
        if !seg.is_empty() {
            send_final(event_tx, seg.to_string(), Vec::new(), 1.0);
        }
        cut = end;
    }

    // If a long run of committed text carries no sentence punctuation, flush it as a Final
    // anyway (broken at the last word boundary so no word is split). This keeps the live
    // Partial — and the detector that re-scans it every tick — from ever carrying an
    // unbounded, ever-growing remainder. Committed text is stable, so this is lossless.
    while committed.len().saturating_sub(cut) > COMMIT_FLUSH_CHARS {
        match committed[cut..].rfind(' ') {
            Some(rel) if rel > 0 => {
                let boundary = cut + rel + 1;
                let seg = strip_tags(&committed[cut..boundary]);
                let seg = seg.trim();
                if !seg.is_empty() {
                    send_final(event_tx, seg.to_string(), Vec::new(), 1.0);
                }
                cut = boundary;
            }
            // One unbroken token longer than the cap — nothing safe to split on yet.
            _ => break,
        }
    }
    *final_upto = cut;

    let remainder = strip_tags(&format!("{}{}", &committed[cut..], tentative));
    let remainder = remainder.trim();
    if flush {
        if !remainder.is_empty() {
            send_final(event_tx, remainder.to_string(), Vec::new(), 1.0);
        }
    } else if !remainder.is_empty() {
        send_partial(event_tx, remainder.to_string());
    }
}

/// Byte index just past the next hard phrase boundary at or after `from`: either an
/// end-of-utterance / end-of-burst control token (`<EOU>`/`<EOB>`) the streaming model
/// emits, or sentence-ending punctuation — whichever comes first. Using the model's own
/// `<EOU>` as a boundary (this model rarely emits periods) is what lets a committed phrase
/// finalize promptly, instead of waiting for the `COMMIT_FLUSH_CHARS` char-count safety net
/// that previously did the endpointing. The tag itself is stripped from emitted text by
/// [`strip_tags`]; here we only use its position as the cut point.
fn next_boundary(s: &str, from: usize) -> Option<usize> {
    let hay = &s[from..];
    [
        hay.find(['.', '?', '!']).map(|rel| from + rel + 1),
        hay.find("<EOU>").map(|rel| from + rel + "<EOU>".len()),
        hay.find("<EOB>").map(|rel| from + rel + "<EOB>".len()),
    ]
    .into_iter()
    .flatten()
    .min()
}

/// Drop end-of-utterance / end-of-burst control tokens the model may surface.
fn strip_tags(s: &str) -> String {
    s.replace("<EOU>", "").replace("<EOB>", "")
}

/// Run offline transcription over `pcm`. When `want_words` is set, request word
/// timestamps and return them alongside a real confidence (from token `p`); previews pass
/// `false` to stay cheap (text only). Returns `(text, words, confidence)`.
fn transcribe(
    session: &mut Session,
    pcm: &[f32],
    want_words: bool,
) -> Option<(String, Vec<crate::types::Word>, f64)> {
    let opts = RunOptions {
        timestamps: if want_words {
            TimestampKind::Word
        } else {
            TimestampKind::None
        },
        ..Default::default()
    };
    match session.run(pcm, &opts) {
        Ok(r) => {
            let t = r.text.trim().to_string();
            if t.is_empty() {
                return None;
            }
            let (words, confidence) = if want_words {
                map_words(&r.words, &r.tokens)
            } else {
                (Vec::new(), 1.0)
            };
            Some((t, words, confidence))
        }
        Err(e) => {
            log::error!("[LocalSTT] transcription failed: {e}");
            None
        }
    }
}

/// Map a run's word + token rows into stt [`Word`](crate::types::Word) events and an
/// overall utterance confidence. Per-word confidence is the mean of that word's tokens'
/// `p`; the family emits `NaN` for `p` when it has no per-token probability, so those are
/// skipped, and a word with no finite `p` reports `1.0` (unknown → don't penalize).
/// Overall confidence is the mean of per-word confidences, or `1.0` when the model
/// provides no probabilities at all — so downstream gating reads "no confidence signal"
/// as "trust it", never as "reject".
fn map_words(
    words: &[transcribe_cpp::Word],
    tokens: &[transcribe_cpp::Token],
) -> (Vec<crate::types::Word>, f64) {
    let mut out = Vec::with_capacity(words.len());
    let mut conf_sum = 0.0f64;
    let mut conf_n = 0usize;
    for w in words {
        let start = w.first_token.max(0) as usize;
        let n = w.n_tokens.max(0) as usize;
        let toks = tokens.get(start..start.saturating_add(n)).unwrap_or(&[]);
        let (psum, pn) = toks.iter().fold((0.0f64, 0usize), |(s, c), t| {
            if t.p.is_finite() {
                (s + t.p as f64, c + 1)
            } else {
                (s, c)
            }
        });
        let wconf = if pn > 0 {
            let m = psum / pn as f64;
            conf_sum += m;
            conf_n += 1;
            m
        } else {
            1.0
        };
        out.push(crate::types::Word {
            text: w.text.clone(),
            start: w.t0_ms as f64 / 1000.0,
            end: w.t1_ms as f64 / 1000.0,
            confidence: wconf,
            punctuated_word: None,
        });
    }
    let overall = if conf_n > 0 {
        conf_sum / conf_n as f64
    } else {
        1.0
    };
    (out, overall)
}

fn send_final(
    event_tx: &mpsc::Sender<TranscriptEvent>,
    transcript: String,
    words: Vec<crate::types::Word>,
    confidence: f64,
) {
    let _ = event_tx.blocking_send(TranscriptEvent::Final {
        transcript,
        words,
        confidence,
        speech_final: true,
    });
}

fn send_partial(event_tx: &mpsc::Sender<TranscriptEvent>, transcript: String) {
    // Non-blocking: a partial is a disposable preview. If the event consumer is briefly
    // behind, dropping one preview is far better than stalling audio intake (a blocked
    // send lets the audio queue build the standing backlog that makes the transcript
    // trail real speech). Finals still use blocking_send so they're never lost.
    let _ = event_tx.try_send(TranscriptEvent::Partial {
        transcript,
        words: Vec::new(),
    });
}

/// Append `batch` to the recovery pre-roll ring, evicting the oldest samples once it
/// exceeds `PREROLL_RECOVER`. Cheap: at most one drain of the front per call.
fn push_preroll(ring: &mut VecDeque<f32>, batch: &[f32]) {
    ring.extend(batch.iter().copied());
    while ring.len() > PREROLL_RECOVER {
        ring.pop_front();
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(p: f32) -> transcribe_cpp::Token {
        transcribe_cpp::Token {
            p,
            ..Default::default()
        }
    }

    fn word(text: &str, t0: i64, t1: i64, first_token: i32, n_tokens: i32) -> transcribe_cpp::Word {
        transcribe_cpp::Word {
            t0_ms: t0,
            t1_ms: t1,
            first_token,
            n_tokens,
            text: text.to_string(),
            seg_index: 0,
        }
    }

    #[test]
    fn map_words_maps_timing_and_averages_token_confidence() {
        let tokens = vec![tok(0.9), tok(0.8), tok(0.6)];
        let words = vec![word("hello", 0, 500, 0, 2), word("world", 500, 1000, 2, 1)];
        let (mapped, overall) = map_words(&words, &tokens);

        assert_eq!(mapped.len(), 2);
        assert_eq!(mapped[0].text, "hello");
        assert!((mapped[0].start - 0.0).abs() < 1e-9);
        assert!((mapped[0].end - 0.5).abs() < 1e-9);
        assert!((mapped[0].confidence - 0.85).abs() < 1e-6); // mean(0.9, 0.8)
        assert!((mapped[1].confidence - 0.6).abs() < 1e-6);
        assert!((overall - (0.85 + 0.6) / 2.0).abs() < 1e-6);
    }

    #[test]
    fn map_words_treats_missing_probability_as_unknown_not_zero() {
        // The family emits NaN for `p` when it has no per-token confidence.
        let tokens = vec![tok(f32::NAN), tok(f32::NAN)];
        let words = vec![word("x", 0, 100, 0, 2)];
        let (mapped, overall) = map_words(&words, &tokens);
        assert_eq!(mapped[0].confidence, 1.0);
        assert_eq!(overall, 1.0); // no finite p anywhere → trust, don't reject
    }

    #[test]
    fn next_boundary_cuts_at_eou_before_punctuation() {
        let s = "turn to john <EOU> three sixteen.";
        let end = next_boundary(s, 0).unwrap();
        assert_eq!(&s[..end], "turn to john <EOU>");
        assert_eq!(strip_tags(&s[..end]).trim(), "turn to john");
    }

    #[test]
    fn next_boundary_cuts_at_punctuation_when_no_tag() {
        let s = "hello world. and more";
        let end = next_boundary(s, 0).unwrap();
        assert_eq!(&s[..end], "hello world.");
    }

    #[test]
    fn next_boundary_none_when_no_boundary() {
        assert!(next_boundary("no boundary yet", 0).is_none());
    }
}
