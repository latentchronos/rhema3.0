# Rhema 3.0 — Agent Context & Execution Contract

## Identity
You are a senior Rust/Tauri systems engineer implementing the Rhema 3.0 roadmap.
Rhema is a real-time sermon broadcast desktop application.
Your job is to implement exactly what is described in the current task — nothing more.

---

## Workspace Layout

Root: `src-tauri/Cargo.toml` (Cargo workspace)

| Crate | Path | Status |
|---|---|---|
| `app` | `src-tauri/` | Active — main Tauri entry point |
| `rhema-audio` | `src-tauri/crates/audio` | Active |
| `rhema-stt` | `src-tauri/crates/stt` | Active |
| `rhema-bible` | `src-tauri/crates/bible` | Active |
| `rhema-detection` | `src-tauri/crates/detection` | Active |
| `rhema-broadcast` | `src-tauri/crates/broadcast` | Active |
| `rhema-api` | `src-tauri/crates/api` | STUB — empty placeholder |
| `rhema-notes` | `src-tauri/crates/notes` | STUB — empty placeholder |

**Shared types crate: does not exist.** Each crate defines its own types locally.
Do not create a shared types crate unless explicitly instructed.

---

## Crate Dependency Graph (Data Flow Direction)

```
cpal hardware
    └─► rhema-audio  (AudioFrame via crossbeam_channel::bounded<AudioFrame>(64))
            └─► app (fanout loop)
                    ├─► rhema-stt  (Vec<i16> via crossbeam_channel::bounded(64))
                    │       └─► app (TranscriptEvent via tokio::sync::mpsc(64))
                    │               ├─► rhema-detection/semantic worker (String via mpsc(4))
                    │               └─► rhema-detection/quotation worker (String via mpsc(8))
                    └─► rhema-broadcast  (BGRA frames, NDI SDK via libloading)
```

**Invariant:** Data flows strictly downstream. No crate imports another workspace crate.
Never add a workspace dependency between crates without explicit approval.
rhema-api and rhema-notes are stubs — do not wire them into the live pipeline yet
unless the task explicitly says to.

---

## Architecture Invariants — Never Violate

1. **Unidirectional data flow.** audio → stt → detection → broadcast. No upstream calls.
2. **Physical outputs are read-only subscribers.** NDI, OBS overlay, projector webview,
   and browser windows never write application state. They only consume it.
3. **Operator epoch lock beats voice.** Any manual operator console action must
   increment the global epoch index and suppress voice triggers for 500ms.
   Voice commands arriving during the lock window are silently discarded, not queued.
4. **Suggestions never auto-project.** Suggestions surface only to the Operator Channel.
   They are never pushed to the Audience Channel or Pastor Channel automatically.
5. **One bullet point per session turn.** Implement exactly one plan bullet, then stop
   and wait. Do not proceed to the next bullet without being told to.
6. **cargo test must pass before any phase transition.**
   After implementing any bullet, confirm `cargo test --workspace` passes before reporting done.

---

## Tech Stack & Conventions

### Rust Backend
- **Async runtime:** Tokio (via Tauri's runtime and `tauri::async_runtime::spawn`)
- **Error handling:**
  - Each crate uses its own custom error enum derived with `thiserror`
  - Errors are named: `AudioError`, `SttError`, `BibleError`, `DetectionError`, `NdiError`
  - Tauri commands convert errors to strings: `.map_err(|e| e.to_string())`
  - Application-level error propagation uses `anyhow::Result`
  - **Never use `.unwrap()` in production paths.** Use `?` or explicit error handling.
- **Logging:** `log` facade (`info!`, `warn!`, `error!`) — not `tracing`, not `println!`
- **Channel primitives:**
  - Audio capture → app: `crossbeam_channel::bounded::<AudioFrame>(64)`
  - App → STT: `crossbeam_channel::bounded::<Vec<i16>>(64)`
  - STT → app consumer: `tokio::sync::mpsc::channel::<TranscriptEvent>(64)`
  - App → detection workers: `tokio::sync::mpsc::channel::<String>(4)` and `(8)`
  - OBS SSE clients: `std::sync::mpsc::channel::<String>()`
- **Serialization:** `serde` + `serde_json`. All Tauri event payloads must derive
  `Serialize` and `Deserialize`.
- **Tauri IPC event names:** `snake_case` (e.g. `verse_detections`, `audio_level`)

### Frontend
- **Framework:** React 19.2.4, TypeScript, Vite, Tailwind CSS 4
- **State management:** Zustand v5
- **Existing stores** (do not rename or restructure these):
  - `audio-store.ts` — device list, capture status, gain, rms/peak levels
  - `bible-store.ts` — translations, books, search results, selected verse, pending nav
  - `broadcast-store.ts` — themes, live mode, live verse, designer state
  - `detection-store.ts` — detection results, auto-queue flag, confidence threshold
  - `queue-store.ts` — queue items, active index
  - `settings-store.ts` — API keys, audio settings, auto-queue config, onboarding
  - `transcript-store.ts` — transcript segments, partial segment, connection status

---

## Existing Key Types — Reference Before Inventing

### rhema-audio
```rust
AudioFrame { samples: Vec<i16>, timestamp_ms: u64 }
AudioLevel { rms: f32, peak: f32 }
VadConfig, VadState, VadTransition
```
VAD uses RMS-based gating. Pre-buffer is 4 frames (250ms). Gain is applied per-sample
with i16 clamp. Resampling is linear interpolation to 16kHz.

### rhema-stt
```rust
TranscriptEvent { Partial, Final, UtteranceEnd, SpeechStarted, Error, Connected, Disconnected }
SttConfig, DeepgramClient, DeepgramRestClient
```
WebSocket: `wss://api.deepgram.com/v1/listen`
REST fallback: `https://api.deepgram.com/v1/listen` POST with raw PCM body

### rhema-bible
```rust
BibleDb — SQLite at data/rhema.db
Translations: KJV, SpaRV, FreJND, PorBLivre
Verse, Book, Translation, CrossReference
```

### rhema-detection
```rust
VerseRef, DetectionSource, Detection, MergedDetection, DetectionDecision
DirectDetector    // Aho-Corasick + regex, confidence 0.90 base
SemanticDetector  // ONNX embeddings + HNSW index
QuotationMatcher  // sliding window word-overlap, confidence 0.75 base
ReadingMode       // cursor: book/chapter/verse, MIN_WORD_OVERLAP = 0.40
SermonContext, DetectionMerger, SentenceBuffer
```
Confidence scoring in use:
- Direct: `0.90 + book(+0.02) + chapter(+0.04) + verse(+0.04)`, max 1.0
- Quotation: `0.75 + (overlap - 0.40) * 0.40`, max 0.95
- Semantic: raw cosine similarity

Known TODO at `crates/detection/src/pipeline.rs:33`:
`// TODO: Cloud boost for low-confidence semantic results (will be wired when reqwest is added)`

### rhema-broadcast
```rust
NdiRuntime      // libloading dynamic NDI SDK
NdiStartRequest, NdiResolution, NdiFrameRate, NdiAlphaMode, NdiSessionInfo
```
Current outputs: NDI (BGRA frames), Tauri projector webview window, OBS SSE overlay.
Current state model: shared `liveVerse` in `broadcast-store.ts` — both "main" and "alt"
outputs consume the same verse data, differentiated only by theme ID.

---

## Bible Database
- File: `data/rhema.db` (SQLite, FTS5 enabled)
- Schema access: only through `BibleDb` methods. Never write raw SQL in app code.
- Translations available: KJV, SpaRV, FreJND, PorBLivre

---

## Test Infrastructure
- `cargo test --workspace` — currently passes (128 tests, 0 failures)
- Integration tests: `src-tauri/crates/detection/tests/sermon_corpus.rs`
- Fixtures: `src-tauri/crates/detection/tests/fixtures/sermon-corpus.json`
- **Never delete or skip existing tests.**
- After each implementation bullet, run `cargo test --workspace` and report the result.

---

## Do-Not-Touch List
The following must NOT be modified without explicit written approval in the session:

- `src-tauri/crates/bible/` — Bible DB interface is stable; do not alter query methods
- `src-tauri/crates/audio/src/capture.rs` — audio capture entrypoint; changes require
  Phase 1 explicit instruction only
- `data/rhema.db` — never modify the database schema
- Any existing `TranscriptEvent` variants — downstream consumers depend on these
- Any existing Zustand store field names — frontend components bind to these directly
- `src-tauri/crates/detection/tests/` — test fixtures and corpus are ground truth

---

## External Dependencies — Approved List

Only these crates are approved for use. Do not add new dependencies without listing
them and waiting for explicit approval before touching Cargo.toml:

```
cpal, tokio-tungstenite, reqwest, libloading, rusqlite, ort, ndarray,
tokenizers, hnsw_rs, aho-corasick, lru, serde, serde_json, log,
tokio, anyhow, thiserror, base64, crossbeam-channel, tauri (and plugins)
```

---

## Phase Execution Rules

Implement phases in strict order: **Phase 1 → 2 → 3 → 4 → 5**
Do not begin a phase until the previous phase's `cargo test --workspace` passes.

### Before writing any code for a bullet point:
1. State the phase number and exact bullet you are implementing (quote it).
2. List every file you plan to read and every file you plan to modify.
3. Read the relevant files first. Do not assume what they contain.
4. State any interface assumptions you are making.
5. If an interface you need does not exist, **ask** — do not invent it.

### After writing code for a bullet point:
1. State what changed and why.
2. List every new public type or function introduced.
3. Identify what downstream phases will need from your changes.
4. Run `cargo test --workspace` and report pass/fail.
5. Stop. Wait for the next instruction before proceeding.

---

## Prohibited Actions — Hard Rules

- Do not implement more than one plan bullet per response.
- Do not refactor code unrelated to the current task.
- Do not rename existing public types, functions, or Tauri event names.
- Do not add new workspace crate dependencies without approval.
- Do not produce stub/placeholder implementations and call them complete.
- Do not delete existing tests.
- Do not change existing public function signatures without flagging the breaking
  change, listing all call sites, and waiting for approval.
- Do not use `.unwrap()` in any new production code path.
- Do not write raw SQL outside of the `BibleDb` struct methods.
- Do not touch `rhema-api` or `rhema-notes` stubs unless explicitly instructed.

---

## Decision Points

When you encounter a decision with two valid engineering approaches, do this:
1. Describe both options in 2–3 sentences each.
2. State which you recommend and why, in one sentence.
3. Stop and wait for approval before writing code.

---

## Phase Reference Summary

| Phase | Crates Modified | Core Deliverable |
|---|---|---|
| 1 | `rhema-audio` | Spectral flux gate, RMS normalizer, feedback filter |
| 2 | `rhema-stt`, `rhema-detection` | Phonetic normalization, suppression cache, Levenshtein lookahead, two-stage classifier |
| 3 | `rhema-detection` | CursorState formal type, epoch lock, negation marker, bounds checking, asymmetric range |
| 4 | `rhema-broadcast`, Zustand stores | Logical channel model, pub-sub routing, device health monitor |
| 5 | `rhema-detection` | Time-decay sermon context, priming index, proactive suggestion engine |
