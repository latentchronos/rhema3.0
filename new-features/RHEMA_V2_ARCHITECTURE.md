# Rhema v2 — Production Architecture Blueprint

**Status:** approved design, phased implementation in progress.
**Scope:** transform Rhema into a production-grade *Real-Time Sermon Intelligence & Presentation OS*.

**Design principle (non-negotiable, in priority order):**
Reliability → Latency → Accuracy → Maintainability → Local capability.
Every trade-off resolves in that order.

**Guiding constraint from the architecture audit (3 rounds, source-level, vs NeMo /
parakeet-rs / Silero VAD / Sortformer / ONNX Runtime):** we are **not** rebuilding the
ASR engine. Rhema already runs cache-aware streaming (Nemotron/Parakeet) with an
append-only committed/tentative contract that is *ahead* of parakeet-rs. The real work is
(1) hardening the front-end (resampling, VAD), (2) wiring signals the runtime already
produces but we discard (word timestamps, per-token confidence, `<EOU>`), and (3) making
the pipeline survive failure mid-service. ~60% keep/modify, ~30% add, ~10% remove.

---

## 0. The pipeline, annotated

```
cpal callback  ──emit──►  AudioFrame{f32,16k,t_capture}          [rhema-audio]
   native rate, f32, downmix
        │
        ▼  anti-aliased resample (rubato) → f32 16k
Audio Processing  ── high-pass 80Hz, optional slow AGC ──►        [rhema-audio]
        │
        ▼  re-block to 512-sample frames
Silero VAD  ──VadEvent + is_speech──►                             [rhema-vad]
   gates silence, emits SpeechStart/End, feeds endpoint
        │  (voiced audio only)
        ▼
Streaming ASR  ── StreamingAsr::accept_audio ──►                  [rhema-asr]
   cache-aware Nemotron/Parakeet-TDT (ggml default, ONNX-CUDA opt)
        │  AsrUpdate{committed_delta, tentative, words[p,t0,t1], eou}
        ▼
Transcript State Manager  ──►                                    [rhema-transcript]
   committed (append-only) + tentative + word/conf/time
   endpoint = <EOU> ∨ VAD-silence ∨ max-duration
        │  TranscriptEvent{Partial|Committed|Final}
        ▼
Intelligence  ── committed→project, tentative→prime ──►          [rhema-detection]
   normalize → intent → scripture → command
        │  ScriptureDetected / IntentDetected / Command
        ▼
Projection Engine  ── operator gate, epoch lock, suppression ──► [rhema-presentation]
        │  ProjectionCommand
        ▼
UI / Broadcast (Tauri emit, OBS, output windows)
        ▲
Supervisor  ── watches every stage, restarts on failure ────────[rhema-runtime]
```

**Two independent clocks:** the VAD frame clock (512 samples / 32 ms, fixed) and the ASR
feed clock (fixed chunk, model-dependent). They must never be conflated again — that
conflation is why today one variable batch does capture, gating, and feed all at once.

---

## 1. Current Architecture Migration

| Existing component | Current design | v2 design | Action |
|---|---|---|---|
| `capture.rs` | cpal, **linear resampler**, i16 | cpal f32, **rubato polyphase**, f32 frames | **Modify** |
| `resample()` | linear, no anti-alias | FIR/polyphase w/ LP @7.8k | **Replace** |
| `vad.rs` (energy SM) | RMS state machine, UI-only | — | **Remove** (superseded by Silero) |
| `gate_chain.rs` + flux/variance | observe-mode spectral gates | keep feedback+AGC; drop flux/variance gating | **Modify** |
| `feedback.rs`, `agc.rs` | PA-feedback zero, RMS-AGC | keep; AGC slow-adapt, post-VAD | **Keep/Modify** |
| `stt::SttEngine` trait | cloud/local seam | generalize → `StreamingAsr` | **Modify** |
| `stt::local.rs` | dual offline+stream, no recovery | streaming-default, cache reset, **auto-recover** | **Modify** (major) |
| `stt::deepgram.rs` | WS + reconnect + keepalive | keep as a `StreamingAsr` impl | **Keep** |
| offline utterance mode | RMS-segment + re-encode | demote to explicit fallback engine | **Keep (demote)** |
| `strip_tags`/`COMMIT_FLUSH_CHARS` | discard `<EOU>`, char-count flush | `<EOU>`-driven endpoint | **Replace** |
| `send_final` conf=1.0, words=[] | discards timing/conf | map `Token.p`, `Word.t0/t1` | **Modify** |
| transcript+detection worker (stt.rs) | in the Tauri command | move to `rhema-transcript` | **Modify** (extract) |
| `detection/*` | strong, mostly done | keep; feed committed-only for projection | **Keep** |
| `normalizer.rs` | **dead (tests-only)** | wire on committed text | **Modify** (activate) |
| `pipeline.rs` Stage-2 | LLM placeholder | wire real `rhema-api` call | **Modify** |
| app `channels/epoch/suppression/suggestion` | operator gating, good | consolidate → `rhema-presentation` | **Keep (relocate)** |
| `broadcast`, `bible`, `notes`, `api` | fine | unchanged | **Keep** |
| — | (none) | Silero VAD | **Add** `rhema-vad` |
| — | (none) | supervisor/watchdog | **Add** `rhema-runtime` |

**Explicitly preserve:** the `SttEngine` trait seam, committed/tentative, Deepgram
reliability engineering, the whole detection intelligence layer, operator gate + epoch
lock + suppression, backlog meter, drop-to-latest partials, never-drop-the-sermon posture.

---

## 2. Module Boundaries (crates)

```
src-tauri/crates/
├── audio/          rhema-audio       capture, resample, filter, meter, feedback/AGC
├── vad/            rhema-vad         Silero VAD + re-blocker + endpoint hints
├── asr/            rhema-asr         StreamingAsr trait + engines (ggml/deepgram/onnx)
├── transcript/     rhema-transcript  committed/tentative state, endpoint, sentence buffer
├── detection/      rhema-detection   normalize→intent→scripture→command (existing, kept)
├── presentation/   rhema-presentation projection engine, epoch, suppression, suggestion
├── runtime/        rhema-runtime     Supervisor, health, recovery, event wiring
├── bible/ broadcast/ api/ notes/     (unchanged)
```

### rhema-audio
- **Responsibility:** own the cpal stream; produce clean 16 kHz mono f32 frames.
- **In:** `AudioConfig{device_id, channel_index, gain, target_rate:16000}`. **Out:** `crossbeam::Sender<AudioFrame>`.
- **Types:** `AudioFrame { samples: Vec<f32>, t_capture_ms: u64 }`, `Resampler` (rubato SincFixedIn, LP < 7.8kHz).
- **Pattern:** cpal callback thread → bounded(64) channel. Keep the `!Send` stream on its own thread.

### rhema-vad
- **Responsibility:** the single source of speech/non-speech truth and speech-boundary events. Replaces `vad.rs` + flux/variance gates.
- **Types:**
  ```rust
  pub struct SileroVad { session: Session, state: Array3<f32>/*(2,1,128)*/, context: [f32;64] }
  pub struct Reblocker { buf: Vec<f32> } // yields exact 512-sample frames
  pub enum VadEvent { SpeechStart{t_ms:u64}, SpeechEnd{t_ms:u64} }
  pub struct VadFrameResult { pub prob: f32, pub is_speech: bool, pub event: Option<VadEvent> }
  pub struct VadConfig { threshold:0.5, neg_threshold:0.35, min_silence_ms:400, speech_pad_ms:100 }
  impl SileroVad { fn process(&mut self, frame512:&[f32])->VadFrameResult; fn reset(&mut self); }
  ```
- **Critical:** the Rust `voice_activity_detector` crate **omits** the 64-sample context prepend and the `−0.15` hysteresis. Reimplement both (context ring + VADIterator grace-window). `min_silence_ms=400` (not Silero's 100) for sermon cadence.

### rhema-asr
- **Responsibility:** audio → committed/tentative text + word metadata, behind one trait. Owns the encoder cache lifecycle.
- **Trait (see §5):** `StreamingAsr { init, accept_audio, snapshot, flush, reset, health }`.
- **Engines:** `GgmlStreamingEngine` (default), `DeepgramEngine`, `OfflineGgmlEngine` (fallback), future `OnnxCudaEngine`.

### rhema-transcript
- **Responsibility:** the stability authority. Convert `AsrUpdate`s into the three-tier `Partial/Committed/Final` contract and detect endpoints. Extracts logic currently in `stt.rs` + `local.rs::emit_stream_text`.
- **Types (see §4):** `TranscriptState`, `TranscriptEvent{Partial|Committed|Final}`, `EndReason`, `EndpointDetector`.

### rhema-detection (kept)
- **Rule:** `Committed`/`Final` may project; `Partial` may only prime/preview. Existing modules unchanged except the input contract.

### rhema-presentation
- **Responsibility:** the only thing allowed to change the screen. Owns operator gate, epoch lock, suppression, suggestion, cursor→verse projection. Consolidates app `channels.rs`/`epoch.rs`/`suppression.rs`/`suggestion.rs`/`nav_lookup.rs`.

### rhema-runtime
- **Responsibility:** wire the pipeline, own the supervisor, drive recovery. New home for the orchestration currently in `commands/stt.rs`.

**Communication pattern:** typed crossbeam/mpsc channels between stages (NOT a single dynamic event bus — avoid over-engineering). Audio path = crossbeam; event path = tokio mpsc. One Tauri-emit bridge at the edge.

---

## 3. Event System

Typed channels, explicit producer→consumer.

| Event | Data | Emitted when | Producer → Consumer |
|---|---|---|---|
| `AudioFrame` | `{samples:Vec<f32>, t_capture_ms}` | every cpal callback | audio → vad |
| `SpeechStart` | `{t_ms}` | VAD crosses threshold | vad → transcript, UI |
| `SpeechEnd` | `{t_ms}` | VAD silence > min_silence | vad → transcript, UI |
| `AsrUpdate` | `{committed_delta, committed_words, tentative, eou, audio_committed_ms}` | each ASR feed with change | asr → transcript |
| `PartialTranscript` | `{text}` | tentative changed | transcript → UI (throttled ~20/s) |
| `CommittedTranscript` | `{text, words[p,t0,t1], conf}` | committed prefix grows | transcript → detection, UI |
| `FinalTranscript` | `{text, words, reason:EndReason}` | endpoint fires | transcript → detection, UI |
| `IntentDetected` | `{class, text}` | on committed/final | detection → presentation |
| `ScriptureDetected` | `{ref, confidence, source, span}` | reference matched | detection → presentation |
| `Command` | `{action, confidence}` | control utterance | detection → presentation |
| `ProjectionCommand` | `{op, payload, epoch}` | operator-gated decision | presentation → UI/broadcast |
| `AsrError` | `{kind, detail}` | engine fault | asr → supervisor |
| `StageHealth` | `{stage, health}` | health transition | any stage → supervisor → UI |

**Consumption rules:** `PartialTranscript` → UI + priming index only. `CommittedTranscript`/`FinalTranscript` → the only inputs to scripture/command detection that can reach `ProjectionCommand`. The trust boundary is encoded in the event graph itself.

---

## 4. Transcript Management Architecture

**Tentative** — "I think the pastor said this."
- Source: `AsrUpdate.tentative` + un-finalized committed tail. Stability: none; may rewrite freely.
- Consumers: live UI line; priming index only. Never touches cursor/projection/commands.
- Update freq: throttled ~20/s for UI.

**Committed** — "this will not change."
- Source: `AsrUpdate.committed_delta` (append-only, monotonic). `final_upto` tracks emitted length.
- **Confidence gate:** segment conf = min/mean of `Word.conf`. Below `PROJECT_CONF=0.55` → display but require corroboration (direct + semantic agree) before projecting.
- Consumers: scripture/intent/command detection; may project.

**Final** — "the utterance is complete."
- Fires on `EndpointDetector`:
  ```
  endpoint = <EOU> token                       (primary, model-emitted)
           ∨ SpeechEnd + hang (VAD silence ≥400ms)
           ∨ committed run ≥ MAX_UTTERANCE (~8s safety)
           ∨ sentence punctuation
  ```
  Replaces the `COMMIT_FLUSH_CHARS` char-count hack.
- Consumers: force `SentenceBuffer` flush → semantic detection, reading-mode alignment, pace estimator.

**Starting thresholds (tune per venue):** `PROJECT_CONF=0.55`, `min_silence_ms=400`, `MAX_UTTERANCE=8s`, partial UI throttle `50ms`, semantic on final only.

---

## 5. Streaming ASR Integration Design

```rust
#[async_trait]
pub trait StreamingAsr: Send {
    fn init(&mut self, cfg: &AsrConfig) -> Result<(), AsrError>;
    fn accept_audio(&mut self, pcm: &[f32]) -> Result<AsrUpdate, AsrError>;
    fn snapshot(&self) -> TranscriptSnapshot;
    fn flush(&mut self) -> Result<AsrUpdate, AsrError>;
    fn reset(&mut self);                 // recreate stream + zero cache (recover / epoch)
    fn health(&self) -> AsrHealth;       // Healthy | Degraded | Failed
}
pub struct AsrUpdate {
    pub committed_delta: String, pub committed_words: Vec<Word>,
    pub tentative: String, pub eou: bool, pub audio_committed_ms: i64,
}
pub struct Word { pub text:String, pub t0_ms:i64, pub t1_ms:i64, pub conf:f32 }
```

Design notes:
- **`init` vs `reset`:** `init` loads model + opens stream (expensive, once). `reset` recreates only the stream/cache (cheap, on recover or epoch boundary).
- **`accept_audio` returns a delta**, not full text — committed is append-only, so avoid re-sending the whole transcript each tick.
- **Cache ownership:** ggml engine keeps the cache hidden in the native `Stream` handle (simpler; no need to thread tensors). A future ONNX engine threads `cache_last_channel/time/len` itself; the trait hides which.
- **`health()` + `AsrError`:** engines self-report `Failed`. The supervisor decides restart, not the engine.
- **Engine selection stays config-driven** (`RHEMA_STT_PROVIDER`): `ggml-stream` (default), `ggml-offline` (fallback), `deepgram`, `onnx-cuda`.

---

## 6. Latency Budget

End-to-end = mic → verse on screen. Two profiles (att_right ≈ 80 ms/unit).

| Stage | Minimum (i5, no VNNI) | Professional (GPU/VNNI) | Source |
|---|---|---|---|
| Mic capture buffer | 10–20 ms | 10–20 ms | cpal frame size |
| Resample (rubato) | ~1 ms | ~1 ms | FIR compute |
| VAD (512-frame) | 32 ms + <1 ms | 32 ms + <1 ms | fixed Silero window |
| **ASR lookahead** | **~1040 ms (att=13)** | **~240–480 ms (att=3–6)** | cache-aware right context |
| Decode | few ms | few ms | TDT greedy |
| Transcript update | <5 ms | <5 ms | string ops |
| Intent/scripture (direct) | <2 ms | <2 ms | automaton |
| Semantic (off-path) | 50–400 ms async | 50–400 ms async | ONNX embed — not on projection path |
| Projection + render | <16 ms | <16 ms | one frame |
| **Committed→screen** | **~1.1–1.2 s** | **~0.4–0.6 s** | dominated by ASR lookahead |

~90% of latency is ASR right-context lookahead. The only lever that moves the number is
dropping `att_right`, which needs VNNI-or-GPU headroom. ≤600 ms reads as "live"; ~1 s (i5)
is usable but visibly trails — the reliability floor, not the goal.

---

## 7. ONNX Runtime Optimization Strategy

Applies only to the future `OnnxCudaEngine` (ggml stays default).

**Now (when first built):**
- Fixed encode window → static shapes → keep `with_memory_pattern(true)` + arena reuse valid.
- Session built once, `Arc<Mutex>`, `GraphOptimizationLevel::Level3`, sequential exec (parallel OFF), `intra_threads = physical cores`, `inter_threads = 1`.
- EP list with default `fail_silently` = free CPU fallback; append `CPUExecutionProvider` after every GPU EP.
- Thread `cache_last_channel/time/len` across runs.

**Later (measured):**
- **IoBinding — bind *persistent state* + *pre-allocated outputs* once.** Do NOT bind the per-chunk audio input (changes every run — ORT's documented anti-pattern).
- `with_optimized_model_path` to cut cold-start. `with_cuda_graph(true)` on CUDA.
- **INT8 dynamic quant gated on a VNNI check at startup.** No VNNI → keep Q5 ggml.
- Spinning threads ON for latency, but measure against capture/UI contention.

---

## 8. Reliability Architecture

```rust
pub struct RestartPolicy { max_attempts:u32, backoff:Duration, reset_budget_after:Duration }
pub enum StageHealth { Up, Restarting{attempt:u32}, Down }
```

**ASR failure** (the #1 gap today — local streaming just `break`s and dies):
```
StreamState::Failed / feed Err
   → emit AsrError, StageHealth{Degraded}
   → engine.reset()  (recreate stream, zero cache, keep model loaded)
   → replay recent pre-roll for encoder-cache warmth (no re-emit)
   → StageHealth{Up}; after N failures → fall back to OfflineGgmlEngine
```
Model stays loaded; only the stream is recreated (~ms, not multi-second reload). This is
the Deepgram reconnect pattern ported to local.

**Audio device failure:** cpal `err_fn` → re-enumerate → re-`start` on default device → preserve `keep_running` → surface "mic reconnecting".

**Model loading failure:** try fallback model path → offline engine → Deepgram cloud (if key) → operator alert. Never crash.

**GPU failure:** ORT EP register fails → `fail_silently` degrades to CPU → emit "running on CPU"; `att_right` auto-raises.

**Memory exhaustion:** bound every buffer (mostly done). Watchdog: backlog > ~5 s → drop-to-latest + `StageHealth{Degraded}`. Expose the backlog meter to the operator UI.

**Health as a first-class UI signal:** per-stage status row (Mic / VAD / ASR / Detection). Silent degradation is the enemy in a live service — make every recovery visible.

---

## 9. Hardware Deployment Profiles

| Profile | Hardware | Runtime | Model | att_right | Committed latency | Notes |
|---|---|---|---|---|---|---|
| **Minimum** | i5-8265U class, no VNNI | ggml CPU | Nemotron-stream **Q5** | 13 (~1 s) | ~1.1 s | Reliability floor. Works, visibly trails. |
| **Recommended** | VNNI CPU (Ice Lake+/Zen4+) / Apple Silicon | ggml CPU (or ORT-INT8 on VNNI) | Nemotron/Parakeet-TDT **Q8/INT8** | 6 (~480 ms) | ~0.5–0.6 s | Target for real deployments. |
| **Professional** | NVIDIA GPU (RTX 3060+) | **ORT-CUDA** | Parakeet-TDT fp16 | 3 (~240 ms) | ~0.35 s | Sub-conversational; enables optional Sortformer. |

Recommend the *Recommended* tier as the baseline purchase — a modern mini-PC with a VNNI
CPU beats a 2018 laptop by more than any software change. Ship *Minimum* so it runs
anywhere; document the latency expectation honestly.

---

## 10. Implementation Roadmap

> **Progress (branch `RhemaV2`, not yet merged to `main`):** Phase 1 complete; Phase 2
> largely complete. Status + commit per task below. All verified by unit tests + two
> runtime smokes (STT native paths, Silero ONNX); live mic→UI validation is the user's.

### Phase 1 — Production Foundation — ✅ COMPLETE

| Task | Status | Files | Impact |
|---|---|---|---|
| **ASR auto-recovery on `Failed`** | ✅ `5977a14` | `stt/local.rs` | Stops mid-service death — stream recreates instead of dying. |
| Anti-aliased resampler | ✅ `fae7ac7` | `audio/capture.rs` (rubato) | Kills the aliasing WER tax; supra-Nyquist rejection tested. |
| Wire `Token.p` + `Word.t0/t1` | ✅ `9fc0caa` | `stt/local.rs` | Un-blinds the pace estimator (offline); confirmed live. |
| `<EOU>` endpoint, drop char-flush | ✅ `9fc0caa` | `stt/local.rs` → `rhema-transcript` | Model-driven phrase boundaries. |
| Enforce committed-only projection | ✅ `a209543` | `commands/stt.rs` + frontend (`is_final`) | Kills false triggers from unstable partials. |
| Stream default = streaming | ⏸ per-deployment | `stt/local.rs` | Left a config decision (`att_right=13` floor on the i5). |

### Phase 2 — Streaming Intelligence — largely complete

| Task | Status | Files | Impact |
|---|---|---|---|
| `rhema-vad` (Silero + context + hysteresis) | ✅ `43649f8`,`c7b8360`,`34cc882` | new crate + capture wiring (`neural-vad`) | Neural VAD front-end; conservative (non-gating) first wiring, verified loading live. |
| Retire energy VAD + flux/variance gating | ⏳ deferred | `audio/vad.rs`, `gate_chain.rs` | Aggressive consolidation — deferred pending real-audio validation. |
| Extract `rhema-transcript` state manager | ✅ `15c9b69` | new crate (streaming committer) | Testable committed/tentative + endpoint contract. |
| Confidence-gated scripture projection | ✅ `da820db`,`5f54b8d` | `commands/stt.rs` + streaming confidence | Low-conf finals stay review-only; **live on the streaming engine** (real per-segment `p`). |
| Activate `normalize_transcript` | ⏸ intentionally NOT wired | `detection/normalizer.rs` | Unconditional homophones (e.g. `roaming→Romans`) corrupt normal speech — needs a context-gated variant first. |
| Wire real Stage-2 LLM | ✅ already shipped | `rhema-api/llm/*`, `commands/stt.rs::run_stage2_worker` | Multi-provider (Anthropic/OpenAI/Gemini). Dead placeholder removed (`5fdeb1f`). |
| Fixed ASR feed cadence + supervisor health UI | ⏳ not started | rhema-runtime | Deterministic latency, visible per-stage health. |

### Phase 3 — Advanced

| Task | Files | Difficulty | Depends on | Impact |
|---|---|---|---|---|
| `OnnxCudaEngine` (att=3–6) | new `asr` engine | High | rhema-asr trait | Sub-500 ms on GPU/VNNI. |
| Parakeet-TDT decode eval | asr config | Med | — | Possibly lower att on same CPU. |
| Runtime ITN/Pnc | asr `RunOptions` | Med | — | "three sixteen"→"3:16" at model. |
| Target-speaker gating (close-mic + VAD; TS-VAD if room-mic) | rhema-vad/audio | Low–High | rhema-vad | Ignore congregation/worship. |
| Sortformer diarization | opt module | High | ONNX path | Only for genuine multi-speaker segments — not the sermon core. |

**Sequencing rule:** Phase 1 ships as a unit — it is the difference between a demo and a
system trustworthy on a Sunday. Do not start Phase 2 crate-splitting until Phase 1's
recovery + committed-only projection are proven in a real service.

**Two hard pushbacks:** (1) don't split crates before ASR auto-recovery lands —
reorganizing code that still dies mid-service is polishing the wrong thing. (2) don't add
diarization to the core path — the close mic is the source separation.
