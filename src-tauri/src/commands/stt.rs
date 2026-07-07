use std::sync::atomic::Ordering;
use std::sync::Mutex;
use std::time::Instant;

use tauri::{AppHandle, Emitter, Manager, State};

use rhema_broadcast::{SuggestedVerse, VerseDisplay};
use rhema_detection::{CursorMode, CursorState, VersePosition, VerseRef};

use crate::suggestion::SuggestionEngine;

use crate::events::{
    AudioLevelPayload, TranscriptPayload, EVENT_AUDIO_LEVEL, EVENT_TRANSCRIPT_FINAL,
    EVENT_TRANSCRIPT_PARTIAL,
};
use crate::epoch::EpochLock;
use crate::state::AppState;
use rhema_audio::{
    AudioConfig, AudioFrame, GateChain, GateChainConfig, Vad, VadConfig, VadTransition,
};
use rhema_stt::{DeepgramClient, SttConfig, TranscriptEvent};

/// A speech-boundary transition, provider-agnostic so the fanout thread emits the same
/// `stt_speech_started` / `stt_speech_ended` UI events regardless of which detector runs.
#[derive(Clone, Copy)]
enum SpeechTransition {
    Started,
    Ended,
}

/// The speech indicator behind the fanout thread. Either the legacy energy VAD (default,
/// and the `neural-vad`-off build) or the Silero neural VAD (when built with the feature
/// and its model loads). Both only drive UI events — neither gates the audio stream.
enum SpeechDetector {
    /// Energy RMS VAD; `None` when the user disabled the speech indicator.
    Energy(Option<Vad>),
    #[cfg(feature = "neural-vad")]
    Neural {
        vad: rhema_vad::SileroVad,
        reblock: rhema_vad::Reblocker,
        gate: rhema_vad::VadGate,
    },
}

impl SpeechDetector {
    /// Fold one (gated) capture frame in and return any speech-boundary transitions.
    fn process(&mut self, frame: &AudioFrame) -> Vec<SpeechTransition> {
        match self {
            SpeechDetector::Energy(vad) => match vad.as_mut().and_then(|v| v.process(frame).transition) {
                Some(VadTransition::SpeechStarted) => vec![SpeechTransition::Started],
                Some(VadTransition::SpeechEnded) => vec![SpeechTransition::Ended],
                None => vec![],
            },
            #[cfg(feature = "neural-vad")]
            SpeechDetector::Neural { vad, reblock, gate } => {
                // Silero needs 16 kHz f32 in fixed 512-sample frames; the capture stream is
                // i16 at 16 kHz, so convert and re-block. On a per-frame inference error we
                // log and skip — the audio itself is still forwarded to STT upstream.
                let samples: Vec<f32> = frame.samples.iter().map(|&s| s as f32 / 32768.0).collect();
                let mut out = Vec::new();
                for chunk in reblock.push(&samples) {
                    match vad.process(&chunk) {
                        Ok(prob) => {
                            if let Some(event) = gate.process(prob).event {
                                out.push(match event {
                                    rhema_vad::VadEvent::SpeechStart { .. } => SpeechTransition::Started,
                                    rhema_vad::VadEvent::SpeechEnd { .. } => SpeechTransition::Ended,
                                });
                            }
                        }
                        Err(e) => log::warn!("[VAD] silero process failed: {e}"),
                    }
                }
                out
            }
        }
    }
}

/// Build the speech indicator for a capture session. Silero when the `neural-vad` feature
/// is built AND its model loads; otherwise (feature off, load failure, or the user
/// disabled the indicator) the energy VAD — so capture is never blocked on the model.
fn build_speech_detector(use_vad: bool) -> SpeechDetector {
    if !use_vad {
        return SpeechDetector::Energy(None);
    }
    #[cfg(feature = "neural-vad")]
    {
        let model = std::env::var("RHEMA_VAD_MODEL").unwrap_or_else(|_| default_vad_model_path());
        match rhema_vad::SileroVad::load(std::path::Path::new(&model)) {
            Ok(vad) => {
                log::info!("[VAD] neural (Silero) speech indicator enabled: {model}");
                return SpeechDetector::Neural {
                    vad,
                    reblock: rhema_vad::Reblocker::new(),
                    gate: rhema_vad::VadGate::new(rhema_vad::VadConfig::default()),
                };
            }
            Err(e) => log::warn!("[VAD] Silero load failed ({e}); falling back to energy VAD"),
        }
    }
    SpeechDetector::Energy(Some(Vad::new(VadConfig::default())))
}

/// Default Silero model path: the gitignored repo `model/` dir (one level up from the app
/// crate). Overridable with `RHEMA_VAD_MODEL`.
#[cfg(feature = "neural-vad")]
fn default_vad_model_path() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../model/silero_vad.onnx")
        .to_string_lossy()
        .into_owned()
}

/// Start the full audio-capture-to-transcription pipeline.
///
/// 1. Opens the microphone via cpal (on a dedicated thread so the non-Send
///    `AudioCapture` never crosses thread boundaries).
/// 2. Connects to Deepgram via WebSocket.
/// 3. Fans audio out to both the level meter (emits `audio_level` events) and Deepgram.
/// 4. Receives transcripts and emits `transcript_partial` / `transcript_final` events.
/// 5. On final transcripts, runs the detection pipeline and emits `verse_detected` events.
#[tauri::command]
pub async fn start_transcription(
    app: AppHandle,
    state: State<'_, Mutex<AppState>>,
    api_key: String,
    device_id: Option<String>,
    gain: Option<f32>,
    channel_index: Option<u16>,
    vad_enabled: Option<bool>,
    command_wake_word: Option<String>,
) -> Result<(), String> {
    // ── 1. Guard: already running? ──────────────────────────────────────
    let (stt_active, audio_active, session_active) = {
        let app_state = state.lock().map_err(|e| e.to_string())?;
        if app_state.stt_active.load(Ordering::Relaxed) {
            return Err("Transcription is already running".into());
        }
        (
            app_state.stt_active.clone(),
            app_state.audio_active.clone(),
            app_state.session_active.clone(),
        )
    };

    // STT provider selection. On-device transcription is opt-in: build with
    // `--features local-stt` and set RHEMA_STT_PROVIDER=local. Otherwise Deepgram (cloud).
    let use_local = cfg!(feature = "local-stt")
        && std::env::var("RHEMA_STT_PROVIDER").as_deref() == Ok("local");

    // Resolve API key: use provided key, or fall back to DEEPGRAM_API_KEY env var
    let resolved_api_key = if api_key.is_empty() {
        std::env::var("DEEPGRAM_API_KEY").unwrap_or_default()
    } else {
        api_key
    };

    // The local engine needs no API key; only the cloud path requires one.
    if !use_local && resolved_api_key.is_empty() {
        return Err(
            "No Deepgram API key provided. Set it in Settings or via DEEPGRAM_API_KEY env var."
                .into(),
        );
    }

    log::info!(
        "Starting transcription: api_key={}..., device_id={:?}, gain={:?}, channel_index={:?}, vad_enabled={:?}, command_wake_word={:?}",
        &resolved_api_key[..8.min(resolved_api_key.len())],
        device_id,
        gain,
        channel_index,
        vad_enabled,
        command_wake_word
    );

    stt_active.store(true, Ordering::SeqCst);
    audio_active.store(true, Ordering::SeqCst);

    // ── 2. Prepare channels ─────────────────────────────────────────────
    // STT audio channel carries Vec<i16> (the samples from each AudioFrame).
    // Sized to absorb a full transcription pause without dropping speech: the local
    // engine can't drain while a blocking decode runs, and at ~20ms gated frames a
    // 64-slot buffer holds only ~1.3s — shorter than a worst-case decode, so audio
    // was being discarded mid-utterance (the `drop channel=deepgram` flood). 1024
    // slots (~20s) is trivial memory and drops nothing; the ~30x-real-time engine
    // drains straight back to empty after each decode, so it adds no steady-state lag.
    let (deepgram_tx, deepgram_rx) = crossbeam_channel::bounded::<Vec<i16>>(1024);

    // ── 3. Spawn the audio-capture + fan-out thread ─────────────────────
    // cpal's `Stream` (inside `AudioCapture`) is !Send, so we must create
    // and drop it on the same thread. This thread:
    //   a) starts the cpal capture
    //   b) reads AudioFrames
    //   c) computes levels → emits audio_level events
    //   d) forwards samples to Deepgram via crossbeam
    let gain_val = gain.unwrap_or(1.0).clamp(0.0, 2.0);
    let selected_channel = channel_index;
    let use_vad = vad_enabled.unwrap_or(false);
    // Way 5: capture wake word (None or empty = disabled, behavior identical to today).
    let wake_word: Option<String> = command_wake_word
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_lowercase());
    let fan_active = stt_active.clone();
    let fan_app = app.clone();

    std::thread::Builder::new()
        .name("audio-fanout".into())
        .spawn(move || {
            let config = AudioConfig {
                device_id,
                sample_rate: 16_000,
                gain: gain_val,
                channel_index: selected_channel,
                vad_enabled: use_vad,
            };

            let (audio_tx, audio_rx) = crossbeam_channel::bounded::<AudioFrame>(64);

            // Start capture on THIS thread — AudioCapture stays here.
            let capture = match rhema_audio::capture::start(config, audio_tx) {
                Ok(c) => c,
                Err(e) => {
                    log::error!("Failed to start audio capture: {e}");
                    fan_active.store(false, Ordering::SeqCst);
                    return;
                }
            };

            log::info!("Audio capture started on fanout thread");

            let mut frame_count: u64 = 0;
            // Speech indicator: Silero neural VAD when built `--features neural-vad` and its
            // model loads, else the energy VAD (identical to before). Conservative wiring —
            // it drives the UI speech-start/end events and NEVER gates the audio stream
            // (every frame is still forwarded to STT), honouring never-drop-the-sermon.
            let mut speech = build_speech_detector(use_vad);
            // Phase 1 hardware gating runs always, independent of the VAD toggle.
            // Kept SPEECH-SAFE (observe mode: gates measure + log but never drop)
            // until the level-invariant discriminator + per-venue calibration land
            // (see the gate-polish backlog). Prime directive: never drop the sermon.
            // AGC leveling and feedback detection still apply.
            let mut gate_chain = GateChain::new(GateChainConfig::default());

            loop {
                if !fan_active.load(Ordering::SeqCst) {
                    break;
                }

                match audio_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                    Ok(frame) => {
                        frame_count += 1;

                        // (a) Compute audio levels at ~15 Hz
                        //     At 16 kHz with ~1024-sample frames, every 4th frame is ~15 Hz.
                        if frame_count % 4 == 0 {
                            let level = rhema_audio::meter::compute_level(&frame.samples);
                            let _ = fan_app.emit(
                                EVENT_AUDIO_LEVEL,
                                AudioLevelPayload {
                                    rms: level.rms,
                                    peak: level.peak,
                                },
                            );
                        }

                        // (b) Hardware gate chain: re-block to 320-sample windows →
                        //     flux → variance (drop bleed) → AGC (level) → feedback
                        //     (zero-but-forward). The meter above read the PRE-AGC input.
                        let gated = gate_chain.process(&frame.samples);
                        if gated.windows_suppressed > 0 || gated.feedback_active {
                            log::info!(
                                "audio_gate: flagged {}/{} windows (peak_flux={:.3} peak_var={:.4}) feedback={} (observe: audio forwarded)",
                                gated.windows_suppressed,
                                gated.windows_total,
                                gated.peak_flux,
                                gated.peak_variance,
                                gated.feedback_active
                            );
                        }
                        if gated.samples.is_empty() {
                            continue;
                        }
                        let frame = AudioFrame {
                            samples: gated.samples,
                            timestamp_ms: frame.timestamp_ms,
                        };

                        // (c) Forward audio to Deepgram. The local VAD is used
                        //     ONLY for the UI speech-indicator events — it does
                        //     NOT gate the audio stream. Deepgram does its own
                        //     server-side VAD/endpointing (vad_events/endpointing
                        //     in the connection URL), so gating locally before it
                        //     is redundant and could starve the transcriber of
                        //     speech (e.g. when the gate's AGC under-boosts quiet
                        //     onsets). Always forwarding keeps transcription
                        //     reliable while preserving the speech indicators.
                        for transition in speech.process(&frame) {
                            match transition {
                                SpeechTransition::Started => {
                                    log::info!("[VAD] speech started");
                                    let _ = fan_app.emit("stt_speech_started", ());
                                }
                                SpeechTransition::Ended => {
                                    log::info!("[VAD] speech ended");
                                    let _ = fan_app.emit("stt_speech_ended", ());
                                }
                            }
                        }
                        if deepgram_tx.try_send(frame.samples).is_err() {
                            rhema_detection::metrics::log_channel_drop("deepgram");
                        }
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                }
            }

            // Dropping `capture` stops the cpal stream.
            capture.stop();
            log::info!("Audio capture stopped on fanout thread");
        })
        .map_err(|e| {
            stt_active.store(false, Ordering::SeqCst);
            audio_active.store(false, Ordering::SeqCst);
            format!("Failed to spawn audio fanout thread: {e}")
        })?;

    // ── 4. Spawn the Deepgram connection on the tokio runtime ───────────
    let stt_config = SttConfig {
        api_key: resolved_api_key,
        model: "nova-3".to_string(),
        sample_rate: 16_000,
        encoding: "linear16".to_string(),
        language: None,
    };

    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<TranscriptEvent>(64);
    let conn_active = stt_active.clone();

    // ── 4. Spawn the selected STT engine ────────────────────────────────
    // Both engines publish to the same `event_tx`, so Task B and the detection
    // pipeline below are provider-agnostic.
    if use_local {
        #[cfg(feature = "local-stt")]
        {
            use rhema_stt::SttEngine;
            let model_path = std::env::var("RHEMA_STT_MODEL").unwrap_or_default();
            log::info!("[STT] provider=local, model={model_path}");
            let engine = rhema_stt::LocalSttClient::new(model_path);
            let local_active = conn_active.clone();
            let local_rx = deepgram_rx.clone();
            tauri::async_runtime::spawn(async move {
                if let Err(e) = engine.connect(local_rx, event_tx, local_active.clone()).await {
                    log::error!("[STT] local engine failed: {e}");
                }
                local_active.store(false, Ordering::SeqCst);
                log::info!("[STT] local engine task exited");
            });
        }
        #[cfg(not(feature = "local-stt"))]
        unreachable!("RHEMA_STT_PROVIDER=local requires the `local-stt` build feature");
    } else {
    let client = DeepgramClient::new(stt_config.clone());

    // Task A: run the Deepgram WebSocket connection.
    // On max reconnect failure, falls back to REST mode (hybrid).
    let rest_event_tx = event_tx.clone();
    let rest_config = stt_config.clone();
    tauri::async_runtime::spawn(async move {
        let result = client.connect(deepgram_rx.clone(), event_tx, conn_active.clone()).await;
        if let Err(e) = result {
            log::error!("Deepgram WebSocket failed: {e}");

            // ── Hybrid mode: fall back to REST transcription ──
            {
                log::warn!("[STT] Connection unstable, switching to Hybrid mode (REST fallback)");
                let _ = rest_event_tx
                    .send(TranscriptEvent::Error(
                        "Connection unstable, switching to Hybrid mode".into(),
                    ))
                    .await;

                let rest_client = rhema_stt::DeepgramRestClient::new(rest_config);
                let mut audio_buffer: Vec<i16> = Vec::new();
                let flush_interval = std::time::Duration::from_secs(5);
                let mut last_flush = std::time::Instant::now();

                loop {
                    if !conn_active.load(Ordering::SeqCst) {
                        break;
                    }

                    match deepgram_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                        Ok(samples) => {
                            audio_buffer.extend(samples);

                            // Flush every 5 seconds of accumulated audio
                            if last_flush.elapsed() >= flush_interval && !audio_buffer.is_empty() {
                                match rest_client.transcribe(&audio_buffer).await {
                                    Ok(events) => {
                                        for evt in events {
                                            let _ = rest_event_tx.send(evt).await;
                                        }
                                    }
                                    Err(e) => {
                                        log::error!("[STT-REST] Transcription failed: {e}");
                                    }
                                }
                                audio_buffer.clear();
                                last_flush = std::time::Instant::now();
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                            // Flush if we have audio and enough time has passed
                            if last_flush.elapsed() >= flush_interval && !audio_buffer.is_empty() {
                                match rest_client.transcribe(&audio_buffer).await {
                                    Ok(events) => {
                                        for evt in events {
                                            let _ = rest_event_tx.send(evt).await;
                                        }
                                    }
                                    Err(e) => {
                                        log::error!("[STT-REST] Transcription failed: {e}");
                                    }
                                }
                                audio_buffer.clear();
                                last_flush = std::time::Instant::now();
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                    }
                }
            }
        }
        conn_active.store(false, Ordering::SeqCst);
        log::info!("Deepgram connection task exited");
    });
    }

    // Task B: consume TranscriptEvents, emit to frontend, run detection
    let evt_active = stt_active.clone();
    let event_app = app.clone();

    // Background semantic detection channel — non-blocking, drops if busy
    let (semantic_tx, mut semantic_rx) = tokio::sync::mpsc::channel::<String>(4);

    // Spawn semantic detection worker (runs ONNX inference without blocking transcript)
    let sem_app = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(text) = semantic_rx.recv().await {
            run_semantic_detection(&sem_app, &text);
        }
    });

    // Background quotation matching channel — fast but separate thread
    let (quotation_tx, mut quotation_rx) = tokio::sync::mpsc::channel::<String>(8);

    let quot_app = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(text) = quotation_rx.recv().await {
            run_quotation_matching(&quot_app, &text);
        }
    });

    // Detection worker: runs ALL heavy detection (direct, translation, reading
    // mode, quotation/semantic routing) off the transcript-event hot path.
    //
    // Two channels so fast speech can't lose authoritative results:
    //  • partials → small (8) channel, `try_send` (drop-if-busy) → coalesced
    //    under an interim flood (interims are redundant previews).
    //  • finals/utterance-end → large (256) channel → effectively never dropped.
    // The worker drains finals with PRIORITY (`biased` select), so a partial
    // flood can never starve a committed final.
    let (partial_tx, mut partial_rx) = tokio::sync::mpsc::channel::<String>(8);
    let (final_tx, mut final_rx) = tokio::sync::mpsc::channel::<FinalJob>(256);
    let det_app = app.clone();
    let det_session = session_active.clone();
    // Clone the (already-normalised) wake word into the detection worker.
    let det_wake = wake_word.clone();
    tauri::async_runtime::spawn(async move {
        // Sentence buffer accumulates is_final fragments into complete sentences.
        // Flushes on sentence-ending punctuation or speech_final signal.
        let mut sentence_buf = rhema_detection::SentenceBuffer::new();

        // Pace-driven adaptive timeout (Task 2.3): track inter-word gap with an
        // EMA and feed it back into the sentence buffer's flush timeout.
        let mut pace = rhema_detection::pace::PaceEstimator::new(4.0);
        let mut last_final_at: Option<std::time::Instant> = None;
        // Periodic tick to activate the previously dead check_timeout() path.
        let mut tick = tokio::time::interval(std::time::Duration::from_millis(250));

        loop {
            tokio::select! {
                biased;

                // Finals first — always processed, never starved by partials.
                final_job = final_rx.recv() => {
                    let Some(job) = final_job else { break };
                    // Session gate (§2.4): ignore all detection/commands until the
                    // operator has started the service. Transcript still displays
                    // (that path is ungated in the consumer).
                    if !det_session.load(Ordering::SeqCst) {
                        continue;
                    }
                    match job {
                        FinalJob::Final { transcript, speech_final, n_words, span_secs } => {
                            // ── Pace-driven adaptive timeout (Task 2.3) ──────────
                            // Measure wall-clock gap since the last final, feed the
                            // EMA estimator, then push the smoothed gap back into
                            // the sentence buffer so its flush timeout tracks the
                            // speaker's natural rhythm.
                            let now = std::time::Instant::now();
                            let dt = last_final_at.map(|t| now.duration_since(t).as_secs_f64()).unwrap_or(0.0);
                            last_final_at = Some(now);
                            pace.observe(n_words, span_secs, dt);
                            sentence_buf.set_adaptive_timeout(pace.gap_secs());
                            // Instantaneous gap this fragment (valid only when ≥2 words
                            // and span ≥ 0.4 s, otherwise 0.0 — mirrors PaceEstimator's
                            // validity check so the metric is meaningful).
                            let inst = if n_words >= 2 && span_secs >= 0.4 {
                                span_secs / (n_words as f64 - 1.0)
                            } else {
                                0.0
                            };
                            rhema_detection::metrics::log_pace(inst, pace.gap_secs().unwrap_or(0.0));
                            // ─────────────────────────────────────────────────────
                            if !transcript.is_empty() {
                                // Translation commands: "read in NIV", "switch to ESV"
                                check_translation_command(&det_app, &transcript);
                                // Direct detection: instant (regex), every is_final
                                let direct_found = run_direct_detection(&det_app, &transcript, true);
                                // Reading mode: does transcript match expected verse?
                                check_reading_mode(&det_app, &transcript, direct_found);
                                // Intent layer (Gap 2): a short voice control command
                                // ("next verse", "clear screen") drives an action; a
                                // genuinely-ambiguous command attempt is escalated to
                                // the Stage-2 fallback. Skipped when direct already hit.
                                if !direct_found {
                                    if rhema_detection::is_isolated_command_context(&transcript) {
                                        check_voice_command(&det_app, &transcript, det_wake.as_deref());
                                    }
                                    if quotation_tx.try_send(transcript.clone()).is_err() {
                                        rhema_detection::metrics::log_channel_drop("quotation");
                                    }
                                    if let Some(sentence) = sentence_buf.append(&transcript) {
                                        if semantic_tx.try_send(sentence).is_err() {
                                            rhema_detection::metrics::log_channel_drop("semantic");
                                        }
                                    }
                                } else {
                                    sentence_buf.force_flush();
                                }
                            }
                            if speech_final {
                                if let Some(sentence) = sentence_buf.force_flush() {
                                    if semantic_tx.try_send(sentence).is_err() {
                                        rhema_detection::metrics::log_channel_drop("semantic");
                                    }
                                }
                            }
                        }
                        FinalJob::UtteranceEnd => {
                            if let Some(sentence) = sentence_buf.force_flush() {
                                if semantic_tx.try_send(sentence).is_err() {
                                    rhema_detection::metrics::log_channel_drop("semantic");
                                }
                            }
                        }
                    }
                }

                // Partials only when no final is pending. Coalesced under flood.
                partial = partial_rx.recv() => {
                    let Some(transcript) = partial else { break };
                    if !det_session.load(Ordering::SeqCst) {
                        continue;
                    }
                    // Direct detection on partials — instant preview for verbose forms
                    // like "Psalm chapter 2 verse 3". Tagged is_final=false: it primes the
                    // merger/context and shows in the panel, but must not move the screen
                    // (projection is committed-only — RHEMA_V2_ARCHITECTURE §4).
                    run_direct_detection(&det_app, &transcript, false);
                }

                // Periodic tick: activates the sentence buffer's timeout flush
                // (previously dead code — check_timeout() was never called).
                // biased ordering keeps finals highest-priority; the tick fires
                // only when neither final nor partial is immediately ready.
                _ = tick.tick() => {
                    if !det_session.load(Ordering::SeqCst) {
                        continue;
                    }
                    if let Some(sentence) = sentence_buf.check_timeout() {
                        if semantic_tx.try_send(sentence).is_err() {
                            rhema_detection::metrics::log_channel_drop("semantic");
                        }
                    }
                }
            }
        }
        log::info!("Detection worker task exited");
    });

    tauri::async_runtime::spawn(async move {
        // Throttle interim (partial) UI emits: at extreme WPM Deepgram floods
        // interims; the UI only needs the latest, so cap partial emits to ~20/s.
        // Finals are always emitted unthrottled.
        let mut last_partial_emit = std::time::Instant::now();
        let partial_emit_min_gap = std::time::Duration::from_millis(50);

        while let Some(event) = event_rx.recv().await {
            if !evt_active.load(Ordering::SeqCst) {
                break;
            }

            // Hot path: only emit transcript events and hand text to the
            // detection worker. No heavy work here, so the bounded transcript
            // channel drains fast and the WebSocket reader never backpressures.
            match event {
                TranscriptEvent::Partial { transcript, .. } => {
                    if !transcript.is_empty() {
                        if last_partial_emit.elapsed() >= partial_emit_min_gap {
                            let _ = event_app.emit(
                                EVENT_TRANSCRIPT_PARTIAL,
                                TranscriptPayload {
                                    text: transcript.clone(),
                                    is_final: false,
                                    confidence: 0.0,
                                },
                            );
                            last_partial_emit = std::time::Instant::now();
                        }
                        if partial_tx.try_send(transcript).is_err() {
                            rhema_detection::metrics::log_channel_drop("partial");
                        }
                    }
                }
                TranscriptEvent::Final {
                    transcript,
                    confidence,
                    speech_final,
                    words,
                } => {
                    if !transcript.is_empty() {
                        // Emit as permanent transcript segment (every is_final)
                        let _ = event_app.emit(
                            EVENT_TRANSCRIPT_FINAL,
                            TranscriptPayload {
                                text: transcript.clone(),
                                is_final: true,
                                confidence,
                            },
                        );
                    }
                    let (n_words, span_secs) = word_span(&words);
                    if final_tx.try_send(FinalJob::Final {
                        transcript,
                        speech_final,
                        n_words,
                        span_secs,
                    }).is_err() {
                        rhema_detection::metrics::log_channel_drop("final");
                    }
                }
                TranscriptEvent::UtteranceEnd => {
                    if final_tx.try_send(FinalJob::UtteranceEnd).is_err() {
                        rhema_detection::metrics::log_channel_drop("final");
                    }
                }
                TranscriptEvent::SpeechStarted => {
                    let _ = event_app.emit("stt_speech_started", ());
                }
                TranscriptEvent::Error(msg) => {
                    log::error!("[STT] Error: {msg}");
                    let _ = event_app.emit("stt_error", msg);
                }
                TranscriptEvent::Connected => {
                    log::info!("[STT] Connected");
                    let _ = event_app.emit("stt_connected", ());
                }
                TranscriptEvent::Disconnected => {
                    log::warn!("[STT] Disconnected");
                    let _ = event_app.emit("stt_disconnected", ());
                }
            }
        }

        log::info!("Transcript event consumer task exited");
    });

    Ok(())
}

/// Run direct (regex/pattern) detection only. Instant, no ONNX.
/// Uses SEPARATE Mutex<DirectDetector> and Mutex<DetectionMerger> so it
/// never blocks on the semantic worker, and cooldown state persists across calls.
/// Returns true if high-confidence results were found (>= 0.90).
/// Single egress point for `verse_detections` events (Gap C).
///
/// Runs the suppression cache (Gap B) — dropping verses that echo a
/// recently-displayed one — then emits the survivors. Source `"contextual"`
/// (reading mode) is exempt from suppression. Fails open: if `AppState` is
/// momentarily locked (e.g. by the semantic worker), results are emitted
/// unfiltered rather than blocking the detection path.
fn emit_detections(
    app: &AppHandle,
    source: &str,
    results: Vec<super::detection::DetectionResult>,
    epoch_at_detection: u64,
) {
    if results.is_empty() {
        return;
    }
    // Epoch lock (Bullet 3.2): operator manual actions win. Discard detections
    // whose epoch is stale or that arrive inside the operator lock window.
    if app.state::<EpochLock>().is_locked_out(epoch_at_detection) {
        log::info!(
            "epoch_lock: discarded {} {} detection(s) (epoch_at_detection={})",
            results.len(),
            source,
            epoch_at_detection
        );
        return;
    }
    let managed: State<'_, Mutex<AppState>> = app.state();
    // Bind to a local (not a block tail) so the guard/Result temporary drop at
    // this statement's `;` — the lock is released before we emit.
    let kept = match managed.try_lock() {
        Ok(mut state) => state.suppression_cache.filter(source, results),
        Err(_) => results,
    };
    if kept.is_empty() {
        return;
    }
    let _ = app.emit("verse_detections", &kept);
    // Phase 4 (Bullet 4.2): also route the surviving detections to the Operator
    // channel (confidence + raw detections are operator-only). Additive — the
    // verse_detections event above is retained for existing consumers.
    crate::channels::route_detections(app, &kept);
}

/// Compute the word count and time span from a slice of Deepgram word-timing objects.
///
/// Returns `(n_words, span_secs)` where `span_secs` is
/// `last_word.end - first_word.start`. For fewer than 2 words the span is 0.0
/// (not enough endpoints to measure a gap). Pure / no side-effects — unit-testable.
fn word_span(words: &[rhema_stt::types::Word]) -> (usize, f64) {
    let n = words.len();
    let span = if n >= 2 { words[n - 1].end - words[0].start } else { 0.0 };
    (n, span)
}

/// An authoritative (is_final / utterance-end) detection job. Routed on its own
/// channel, separate from the interim flood, so finals are **never dropped** even
/// when fast speech saturates the partial channel (fast-speech hardening). The
/// detection worker drains this channel with priority over partials.
enum FinalJob {
    Final {
        transcript: String,
        speech_final: bool,
        /// Number of words in this Deepgram-final fragment (from word-timing data).
        n_words: usize,
        /// `last_word.end - first_word.start` in seconds; 0.0 when fewer than 2 words.
        span_secs: f64,
    },
    UtteranceEnd,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_span_empty() {
        let (n, span) = word_span(&[]);
        assert_eq!(n, 0);
        assert_eq!(span, 0.0);
    }

    #[test]
    fn word_span_single() {
        let w = rhema_stt::types::Word {
            text: "hello".into(),
            start: 1.0,
            end: 1.5,
            confidence: 1.0,
            punctuated_word: None,
        };
        let (n, span) = word_span(&[w]);
        assert_eq!(n, 1);
        assert_eq!(span, 0.0);
    }

    #[test]
    fn word_span_three_words() {
        let make = |start: f64, end: f64| rhema_stt::types::Word {
            text: "x".into(),
            start,
            end,
            confidence: 1.0,
            punctuated_word: None,
        };
        let words = vec![make(1.0, 1.5), make(2.0, 2.5), make(3.5, 4.0)];
        let (n, span) = word_span(&words);
        assert_eq!(n, 3);
        assert!((span - 3.0).abs() < 1e-9, "expected span=3.0 got {span}");
    }
}

/// Intent layer, live wiring (Gap 2 / Bullet V5). Parse a short voice utterance
/// into a structured [`rhema_detection::NavCommand`] (absolute jump, relative
/// step, or clear) and emit it to the frontend (`voice_command`); escalate an
/// ambiguous command attempt to the Stage-2 fallback. Only short, command-like
/// utterances are considered, so ordinary preaching never triggers navigation.
fn check_voice_command(app: &AppHandle, transcript: &str, wake: Option<&str>) {
    use rhema_detection::{is_control_command, parse_nav_command_with_wake};

    let nav = parse_nav_command_with_wake(transcript, wake);
    // Metrics: log every command attempt and whether it parsed (observe-only).
    let parsed_str: Option<String> = nav.as_ref().map(|cmd| format!("{cmd:?}"));
    rhema_detection::metrics::log_command_attempt(transcript, parsed_str.as_deref());

    if let Some(cmd) = nav {
        // Emit the structured command (jump / step / clear) for the frontend to
        // route through the navigation cursor (Bullet V5).
        log::info!("voice_command: {cmd:?} (from '{transcript}')");
        let _ = app.emit("voice_command", cmd);
    } else if transcript.split_whitespace().count() <= 4 && is_control_command(transcript) {
        // Command-shaped but unrecognized → genuinely ambiguous; escalate to the
        // Stage-2 fallback (placeholder until 5.4 wires the real Claude call).
        let managed: State<'_, Mutex<AppState>> = app.state();
        let locked = managed.try_lock();
        if let Ok(state) = locked {
            state.detection_pipeline.queue_stage2(transcript);
        }
    }
}

/// Drain the Stage-2 (LLM fallback) channel and run the real multi-provider
/// classification (Bullet L5). For each ambiguous transcript, read the current
/// provider config; if one is set, ask the provider whether the utterance refers
/// to scripture and, when it does, resolve the returned reference through direct
/// detection so it surfaces like any other detection. No-op when unconfigured.
pub async fn run_stage2_worker(
    app: AppHandle,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<String>,
) {
    use rhema_api::llm::{self, LlmConfig, Stage2Request};

    while let Some(transcript) = rx.recv().await {
        let config: Option<LlmConfig> = app
            .state::<Mutex<Option<LlmConfig>>>()
            .lock()
            .ok()
            .and_then(|guard| guard.clone());
        let Some(config) = config else {
            continue; // no provider configured → Stage-2 disabled
        };
        if !config.is_usable() {
            continue;
        }

        let req = Stage2Request {
            transcript: transcript.clone(),
            context: None,
        };
        match llm::classify(&config, &req).await {
            Ok(res) if res.is_scripture => {
                if let Some(reference) = res.reference {
                    log::info!(
                        "stage2: '{transcript}' → {reference} (confidence {:.2})",
                        res.confidence
                    );
                    // Stage-2 resolves an ambiguous (already-final) utterance to a
                    // concrete reference — authoritative, may project.
                    run_direct_detection(&app, &reference, true);
                }
            }
            Ok(_) => log::debug!("stage2: '{transcript}' not scripture"),
            Err(e) => log::warn!("stage2 classify error: {e}"),
        }
    }
}

/// Run direct (regex/automaton) scripture detection over `transcript`.
///
/// `is_final` marks whether the text is authoritative (committed/final) or an unstable
/// interim/partial. Both run detection — partials give the operator an instant preview and
/// prime the merger/sermon-context — but only `is_final` detections are allowed to drive
/// projection (preview selection + auto-queue) on the frontend. Partial detections are
/// tagged `is_final = false` so the frontend shows them in the panel/operator console
/// without moving the screen. See RHEMA_V2_ARCHITECTURE §4 (committed-only projection).
fn run_direct_detection(app: &AppHandle, transcript: &str, is_final: bool) -> bool {
    use rhema_detection::{DetectionMerger, DirectDetector};

    let epoch_at_detection = app.state::<EpochLock>().current();

    let detector_state: State<'_, Mutex<DirectDetector>> = app.state();
    let mut detector = match detector_state.lock() {
        Ok(d) => d,
        Err(e) => {
            log::error!("Failed to lock DirectDetector: {e}");
            return false;
        }
    };
    let direct_results = detector.detect(transcript);
    drop(detector); // Release immediately

    if direct_results.is_empty() {
        return false;
    }

    // Check if any result has high confidence before merging
    let has_high_confidence = direct_results.iter().any(|d| d.confidence >= 0.90);

    // Merge using the managed merger (persists cooldown state across calls,
    // preventing duplicate emissions when running on both partials and finals)
    let merger_state: State<'_, Mutex<DetectionMerger>> = app.state();
    let mut merger = match merger_state.lock() {
        Ok(m) => m,
        Err(e) => {
            log::error!("Failed to lock DetectionMerger: {e}");
            return false;
        }
    };
    let merged = merger.merge(direct_results, vec![]);
    drop(merger);
    if merged.is_empty() {
        return false;
    }

    // Resolve verse info from DB (needs AppState, but only briefly for DB lookup)
    let app_managed: State<'_, Mutex<AppState>> = app.state();
    let mut app_state = match app_managed.try_lock() {
        Ok(s) => s,
        Err(_) => {
            // AppState locked by semantic worker — emit results without verse text
            let results: Vec<super::detection::DetectionResult> = merged
                .iter()
                .map(|m| {
                    let vr = &m.detection.verse_ref;
                    super::detection::DetectionResult {
                        verse_ref: format!("{} {}:{}", vr.book_name, vr.chapter, vr.verse_start),
                        verse_text: String::new(),
                        book_name: vr.book_name.clone(),
                        book_number: vr.book_number,
                        chapter: vr.chapter,
                        verse: vr.verse_start,
                        confidence: m.detection.confidence,
                        source: "direct".to_string(),
                        auto_queued: m.auto_queued,
                        raw_score: m.decision.raw_score,
                        minimum_threshold: m.decision.minimum_threshold,
                        auto_queue_threshold: m.decision.auto_queue_threshold,
                        decision: m.decision.decision.to_string(),
                        explanation: m.decision.explanation.clone(),
                        transcript_snippet: m.detection.transcript_snippet.clone(),
                        is_final,
                    }
                })
                .collect();
            for r in &results {
                log::info!(
                    "[DET-DIRECT] Found: {} ({:.0}%) (no DB)",
                    r.verse_ref,
                    r.confidence * 100.0
                );
            }
            emit_detections(app, "direct", results, epoch_at_detection);
            return has_high_confidence;
        }
    };
    let mut results: Vec<super::detection::DetectionResult> = merged
        .iter()
        .map(|m| super::detection::to_result(&app_state, m))
        .collect();
    // to_result defaults is_final=true; tag with the actual finality of this transcript
    // so partial-derived detections can't drive projection on the frontend.
    for r in &mut results {
        r.is_final = is_final;
    }

    // Update sermon context with direct detection results
    for m in &merged {
        app_state
            .sermon_context
            .update(&m.detection.verse_ref, m.detection.confidence, "direct");
    }

    for r in &results {
        log::info!(
            "[DET-DIRECT] Found: {} ({:.0}%)",
            r.verse_ref,
            r.confidence * 100.0
        );
    }
    drop(app_state);
    emit_detections(app, "direct", results, epoch_at_detection);
    has_high_confidence
}

/// Phase 5 (Bullet 5.3): ingest the sentence into the topic vector and, if a
/// strong unshown thematically-related verse exists, surface ONE suggestion to
/// the Operator channel. Runs in the background semantic worker (off the live
/// transcript path). No-op when the semantic model/index is not loaded.
fn run_topic_suggestion(app: &AppHandle, transcript: &str) {
    // --- Under the AppState lock: ingest + topic search + priming boost. ---
    let candidate: Option<(VerseRef, f32, String)> = {
        let managed: State<'_, Mutex<AppState>> = app.state();
        let mut state = match managed.lock() {
            Ok(s) => s,
            Err(_) => return,
        };
        if !state.detection_pipeline.semantic.is_ready() {
            return;
        }
        // Ingest this sentence's embedding into the time-decay topic vector.
        if let Some(emb) = state.detection_pipeline.semantic.embed_text(transcript) {
            state.sermon_context.ingest_embedding(emb);
        }
        let topic = match state.sermon_context.topic_vector() {
            Some(t) => t,
            None => return,
        };
        let hits = state.detection_pipeline.semantic.search_vector(&topic, 5);
        if hits.is_empty() {
            return;
        }
        let Some(db) = state.bible_db.as_ref() else {
            return;
        };
        // Resolve verse ids → references (keep verse text for the payload).
        let mut texts: Vec<(VerseRef, String)> = Vec::new();
        let mut scored: Vec<(VerseRef, f32)> = Vec::new();
        for (id, sim) in hits {
            if let Ok(Some(v)) = db.get_verse_by_id(id) {
                let vref = VerseRef {
                    book_number: v.book_number,
                    book_name: v.book_name.clone(),
                    chapter: v.chapter,
                    verse_start: v.verse,
                    verse_end: None,
                };
                scored.push((vref.clone(), sim as f32));
                texts.push((vref, v.text));
            }
        }
        if scored.is_empty() {
            return;
        }
        // Pre-service priming boost (Bullet 5.2), then take the best candidate.
        state.priming_index.apply_boost(&mut scored);
        let (best_ref, best_score) = scored.into_iter().next().unwrap();
        let text = texts
            .iter()
            .find(|(r, _)| {
                r.book_number == best_ref.book_number
                    && r.chapter == best_ref.chapter
                    && r.verse_start == best_ref.verse_start
            })
            .map(|(_, t)| t.clone())
            .unwrap_or_default();
        Some((best_ref, best_score, text))
    };

    let (best_ref, score, text) = match candidate {
        Some(c) => c,
        None => return,
    };

    // --- Gate via the SuggestionEngine (separate lock, dropped before emit). ---
    let now = Instant::now();
    {
        let engine_state: State<'_, Mutex<SuggestionEngine>> = app.state();
        let mut engine = match engine_state.lock() {
            Ok(e) => e,
            Err(_) => return,
        };
        if !engine.should_suggest(
            &best_ref.book_name,
            best_ref.chapter,
            best_ref.verse_start,
            score,
            now,
        ) {
            return;
        }
        engine.note_suggested(now);
    }

    // --- Emit to the Operator channel ONLY. ---
    let reference = format!(
        "{} {}:{}",
        best_ref.book_name, best_ref.chapter, best_ref.verse_start
    );
    log::info!("suggestion: proposing {} ({:.0}%)", reference, score * 100.0);
    let suggestion = SuggestedVerse {
        verse: VerseDisplay {
            book: best_ref.book_name.clone(),
            chapter: best_ref.chapter.max(0) as u16,
            verse_start: best_ref.verse_start.max(0) as u16,
            verse_end: None,
            reference,
            text,
            translation: String::new(),
        },
        score,
        reason: "Thematically related to current sermon context".to_string(),
    };
    crate::channels::route_suggestion(app, suggestion);
}

/// Run semantic (ONNX embedding) detection. Slow, runs in background worker.
fn run_semantic_detection(app: &AppHandle, transcript: &str) {
    // Phase 5: feed the topic vector and evaluate a proactive suggestion first
    // (ingests every sentence this worker sees, regardless of detections below).
    run_topic_suggestion(app, transcript);

    let epoch_at_detection = app.state::<EpochLock>().current();
    log::info!(
        "[DET-SEMANTIC] Running on: {:?}",
        &transcript[..transcript.len().min(80)]
    );
    let managed: State<'_, Mutex<AppState>> = app.state();
    let mut app_state = match managed.lock() {
        Ok(s) => s,
        Err(e) => {
            log::error!("Failed to lock AppState for semantic detection: {e}");
            return;
        }
    };
    let mut detections = app_state.detection_pipeline.process_semantic(transcript);
    if detections.is_empty() {
        log::info!("[DET-SEMANTIC] No detections");
        return;
    }

    // Apply context boosting: same-book/chapter detections get higher confidence
    for m in &mut detections {
        let boost = app_state.sermon_context.confidence_boost(
            m.detection.verse_ref.book_number,
            m.detection.verse_ref.chapter,
        );
        if boost > 0.0 {
            m.detection.confidence = (m.detection.confidence + boost).min(1.0);
        }
    }

    // Update sermon context with the top detection
    if let Some(top) = detections.first() {
        app_state.sermon_context.update(
            &top.detection.verse_ref,
            top.detection.confidence,
            "semantic",
        );
    }

    let results: Vec<super::detection::DetectionResult> = detections
        .iter()
        .map(|m| super::detection::to_result(&app_state, m))
        .collect();
    for r in &results {
        log::info!(
            "[DET-SEMANTIC] Found: {} ({:.0}% {}) auto_q={}",
            r.verse_ref,
            r.confidence * 100.0,
            r.source,
            r.auto_queued
        );
    }
    drop(app_state);
    emit_detections(app, "semantic", results, epoch_at_detection);
}

/// Check reading mode: if active, test transcript against expected verse.
/// If direct detection just found a new verse, start/restart reading mode.
fn check_reading_mode(app: &AppHandle, transcript: &str, direct_found: bool) {
    use rhema_detection::ReadingMode;

    let epoch_at_detection = app.state::<EpochLock>().current();

    // If direct detection found a verse, consider starting/restarting reading mode.
    // BUT: if reading mode is already active on a book/chapter, do NOT restart
    // on a different book — false positives from bare numbers (e.g., "verse 5"
    // getting matched as "Job 3:5") would hijack the reading session.
    if direct_found {
        let verse_info = {
            let detector_state: State<'_, Mutex<rhema_detection::DirectDetector>> = app.state();
            let detector = match detector_state.lock() {
                Ok(d) => d,
                Err(_) => return,
            };
            detector.recent_detections.front().cloned()
        };

        if let Some(recent) = verse_info {
            // Get the confidence of the detection to distinguish explicit refs from false positives
            let detection_confidence = {
                let detector_state: State<'_, Mutex<rhema_detection::DirectDetector>> = app.state();
                detector_state
                    .lock()
                    .ok()
                    .and_then(|d| d.recent_detections.front().map(|_| 0.95)) // Direct detections are always high confidence
                    .unwrap_or(0.0)
            };

            let should_start = {
                let rm_managed: &Mutex<ReadingMode> = app.state::<Mutex<ReadingMode>>().inner();
                match rm_managed.lock() {
                    Ok(rm) => {
                        if !rm.is_active() && !rm.has_verses() {
                            true // Not active, no verses loaded — start fresh
                        } else if !rm.is_active() && rm.has_verses() {
                            // Paused — restart on any new explicit reference
                            true
                        } else if rm.current_book() == recent.book_number
                            && rm.current_chapter() == recent.chapter
                        {
                            false // Same book+chapter — already tracking this
                        } else if rm.current_book() != recent.book_number
                            && detection_confidence >= 0.90
                        {
                            // Different book with high confidence — explicit new reference
                            // (e.g., "John 1:1" after reading Exodus). Restart.
                            true
                        } else if rm.current_book() == recent.book_number {
                            // Same book, different chapter — natural progression
                            true
                        } else {
                            // Different book, low confidence — likely false positive
                            false
                        }
                    }
                    Err(_) => false,
                }
            };

            if should_start {
                let chapter_data = {
                    let app_managed: State<'_, Mutex<crate::state::AppState>> = app.state();
                    let app_state = match app_managed.try_lock() {
                        Ok(s) => s,
                        Err(_) => return,
                    };
                    match &app_state.bible_db {
                        Some(db) => db
                            .get_chapter(
                                app_state.active_translation_id,
                                recent.book_number,
                                recent.chapter,
                            )
                            .ok(),
                        None => None,
                    }
                };

                if let Some(chapter_verses) = chapter_data {
                    let verses: Vec<(i32, String)> = chapter_verses
                        .into_iter()
                        .map(|v| (v.verse, v.text))
                        .collect();

                    let rm_managed: &Mutex<ReadingMode> = app.state::<Mutex<ReadingMode>>().inner();
                    if let Ok(mut rm) = rm_managed.lock() {
                        rm.start(
                            recent.book_number,
                            &recent.book_name,
                            recent.chapter,
                            recent.verse_start,
                            verses,
                        );
                    }
                }
            }
        }
    }

    // Check reading mode for verse advancement
    let rm_managed: &Mutex<ReadingMode> = app.state::<Mutex<ReadingMode>>().inner();
    let advance = {
        let mut rm = match rm_managed.lock() {
            Ok(rm) => rm,
            Err(_) => return,
        };
        if !rm.is_active() {
            return;
        }
        rm.check_transcript(transcript)
    };

    if let Some(advance) = advance {
        let _ = app.emit("reading_mode_verse", &advance);

        // Phase 3 bridge: keep the formal navigation cursor in sync with the
        // reading-mode advance, so a later manual next/previous-verse continues
        // from the read position. Backend-internal (not via set_cursor_position),
        // so it does NOT trip the §6.8 "manual nav exits reading mode" rule.
        if (1..=66).contains(&advance.book_number) {
            let managed: State<'_, Mutex<AppState>> = app.state();
            let locked = managed.lock();
            if let Ok(mut st) = locked {
                let translation = match &st.cursor {
                    Some(c) => c.position().translation.clone(),
                    None => st
                        .bible_db
                        .as_ref()
                        .and_then(|db| db.list_translations().ok())
                        .and_then(|ts| {
                            ts.into_iter()
                                .find(|t| t.id == st.active_translation_id)
                                .map(|t| t.abbreviation)
                        })
                        .unwrap_or_else(|| "KJV".to_string()),
                };
                if let Ok(pos) = VersePosition::new(
                    advance.book_number as u8,
                    advance.chapter as u16,
                    advance.verse as u16,
                    translation,
                    None,
                ) {
                    match &mut st.cursor {
                        Some(c) => c.navigate_to(pos, CursorMode::Reading),
                        None => st.cursor = Some(CursorState::new(pos, CursorMode::Reading)),
                    }
                }
            }
        }

        // Also emit as a verse_detection so it appears in the detections panel
        let confidence = advance.confidence;
        let auto_queued = true;
        let (raw_score, minimum_threshold, auto_queue_threshold, decision, explanation) =
            super::detection::live_detection_metadata("contextual", confidence, auto_queued);
        let result = super::detection::DetectionResult {
            verse_ref: advance.reference.clone(),
            verse_text: advance.verse_text.clone(),
            book_name: advance.book_name.clone(),
            book_number: advance.book_number,
            chapter: advance.chapter,
            verse: advance.verse,
            confidence,
            source: "contextual".to_string(),
            auto_queued,
            raw_score,
            minimum_threshold,
            auto_queue_threshold,
            decision,
            explanation,
            transcript_snippet: String::new(),
            // Reading-mode advance fires on committed/final transcripts only.
            is_final: true,
        };
        emit_detections(app, "contextual", vec![result], epoch_at_detection);
    }
}

/// Check for voice translation commands like "read in NIV", "switch to ESV".
fn check_translation_command(app: &AppHandle, transcript: &str) {
    let detector_state: State<'_, Mutex<rhema_detection::DirectDetector>> = app.state();
    let detector = match detector_state.lock() {
        Ok(d) => d,
        Err(_) => return,
    };

    if let Some(abbrev) = detector.detect_translation_command(transcript) {
        drop(detector);

        // Find the translation ID for this abbreviation
        let managed: State<'_, Mutex<AppState>> = app.state();
        let mut app_state = match managed.try_lock() {
            Ok(s) => s,
            Err(_) => return,
        };

        if let Some(ref db) = app_state.bible_db {
            if let Ok(translations) = db.list_translations() {
                if let Some(t) = translations.iter().find(|t| t.abbreviation == abbrev) {
                    app_state.active_translation_id = t.id;
                    log::info!("[STT] Voice command: switched to {} (id={})", abbrev, t.id);
                    drop(app_state);

                    // Emit event so frontend updates
                    #[derive(serde::Serialize, Clone)]
                    struct TranslationSwitch {
                        abbreviation: String,
                        translation_id: i64,
                    }
                    let _ = app.emit(
                        "translation_command",
                        TranslationSwitch {
                            abbreviation: abbrev,
                            translation_id: t.id,
                        },
                    );
                }
            }
        }
    }
}

/// Run quotation matching against all loaded Bible translations.
fn run_quotation_matching(app: &AppHandle, transcript: &str) {
    let epoch_at_detection = app.state::<EpochLock>().current();
    // When reading mode is active, suppress quotation matching entirely.
    // The reader is actively reading a passage — quotation matches for
    // OTHER books would hijack the display away from what's being read.
    {
        use rhema_detection::ReadingMode;
        let rm_managed: &Mutex<ReadingMode> = app.state::<Mutex<ReadingMode>>().inner();
        if let Ok(rm) = rm_managed.lock() {
            if rm.is_active() || rm.has_verses() {
                return; // Reading mode owns the display
            }
        }
    }

    let managed: State<'_, Mutex<AppState>> = app.state();
    let app_state = match managed.try_lock() {
        Ok(s) => s,
        Err(_) => return, // AppState busy
    };

    if !app_state.quotation_matcher.is_ready() {
        return;
    }

    let detections = app_state.quotation_matcher.match_transcript(transcript);
    if detections.is_empty() {
        return;
    }

    let results: Vec<super::detection::DetectionResult> = detections
        .iter()
        .map(|d| {
            let vr = &d.verse_ref;
            // Try to resolve verse text from DB
            let verse_text = if let Some(ref db) = app_state.bible_db {
                db.get_verse(
                    app_state.active_translation_id,
                    vr.book_number,
                    vr.chapter,
                    vr.verse_start,
                )
                .ok()
                .flatten()
                .map(|v| v.text)
                .unwrap_or_default()
            } else {
                String::new()
            };

            let auto_queued = d.confidence >= 0.85;
            let (raw_score, minimum_threshold, auto_queue_threshold, decision, explanation) =
                super::detection::live_detection_metadata("quotation", d.confidence, auto_queued);

            super::detection::DetectionResult {
                verse_ref: format!("{} {}:{}", vr.book_name, vr.chapter, vr.verse_start),
                verse_text,
                book_name: vr.book_name.clone(),
                book_number: vr.book_number,
                chapter: vr.chapter,
                verse: vr.verse_start,
                confidence: d.confidence,
                source: "quotation".to_string(),
                auto_queued,
                raw_score,
                minimum_threshold,
                auto_queue_threshold,
                decision,
                explanation,
                transcript_snippet: d.transcript_snippet.clone(),
                // Quotation matching runs on committed/final transcripts only.
                is_final: true,
            }
        })
        .collect();

    for r in &results {
        log::info!(
            "[DET-QUOTATION] Found: {} ({:.0}%) auto_q={}",
            r.verse_ref,
            r.confidence * 100.0,
            r.auto_queued
        );
    }

    drop(app_state);
    emit_detections(app, "quotation", results, epoch_at_detection);
}

/// Stop the transcription pipeline (audio capture + Deepgram).
#[tauri::command]
pub fn stop_transcription(state: State<'_, Mutex<AppState>>) -> Result<(), String> {
    let app_state = state.lock().map_err(|e| e.to_string())?;
    // Idempotent: always reset, even if a dropped connection already cleared
    // the flag, so the user can always recover the UI without killing the app.
    app_state.stt_active.store(false, Ordering::SeqCst);
    app_state.audio_active.store(false, Ordering::SeqCst);
    log::info!("Transcription stop requested");
    Ok(())
}
