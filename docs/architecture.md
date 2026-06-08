# Rhema Architecture

Rhema is a Tauri v2 desktop application for real-time Bible verse detection during live sermons and broadcasts. The frontend is a React/TypeScript operator console. The backend is a Rust workspace that owns audio capture, speech-to-text integration, Bible data access, verse detection, and NDI broadcast output.

## System Story

At runtime, Rhema follows this path:

```text
microphone or mixer input
  -> Rust audio capture
  -> Deepgram speech-to-text
  -> transcript events
  -> verse detection strategies
  -> Bible database lookup
  -> React operator UI
  -> queue / live verse selection
  -> broadcast output window
  -> optional NDI frame output
```

The operator works mostly inside `Dashboard`, which is made of the transport bar, transcript panel, preview panel, live output panel, queue panel, search panel, and detections panel. Rust emits live events into this UI, and the UI sends command calls back to Rust through Tauri `invoke`. OBS Browser Source output is served by a local-only HTTP/SSE server when enabled.

## Main Technologies

Frontend:

- React 19 for UI composition.
- TypeScript for typed frontend models and command/event payloads.
- Vite 7 for development and production builds.
- Tailwind CSS v4 for styling.
- shadcn/ui-style local components, Radix-style primitives, and lucide icons for interface controls.
- Zustand for local UI state.
- Fuse.js for client-side fuzzy/contextual search.
- Fabric.js for theme designer canvas editing.
- Canvas 2D for final verse rendering.

Backend:

- Tauri v2 for desktop shell, command bridge, secondary windows, plugins, and app lifecycle.
- Rust for native code, audio, async STT, detection, SQLite access, and NDI FFI.
- Rust standard networking for the local OBS browser-source server.
- Tokio for async tasks.
- crossbeam-channel for audio and worker fan-out.
- SQLite through `rusqlite` in the Bible crate.
- ONNX Runtime through `ort` for local embeddings when the model files exist.
- Deepgram WebSocket and REST fallback for speech-to-text.
- NDI SDK through dynamic loading in the broadcast crate.

Setup/data tooling:

- Bun for package management and TypeScript data scripts.
- Python for BibleGateway download helpers and embedding precompute paths.
- ONNX tooling for model export/quantization.

## Rust Workspace

The Tauri backend lives in `src-tauri/`. The root package wires commands, app state, startup loading, and plugins. The domain code is split into crates.

### `rhema-audio`

Purpose:

- Enumerate audio input devices.
- Capture PCM audio with `cpal`.
- Compute level meter data.
- Provide VAD code, although live STT currently forwards all frames and relies on Deepgram's built-in silence handling.

Primary paths:

- `src-tauri/crates/audio/src/device.rs`
- `src-tauri/crates/audio/src/capture.rs`
- `src-tauri/crates/audio/src/meter.rs`
- `src-tauri/crates/audio/src/vad.rs`

### `rhema-stt`

Purpose:

- Stream audio to Deepgram over WebSocket.
- Emit transcript events for partials, finals, utterance end, connection state, and errors.
- Fall back to REST transcription if the streaming connection fails.

Primary paths:

- `src-tauri/crates/stt/src/deepgram.rs`
- `src-tauri/crates/stt/src/rest.rs`
- `src-tauri/crates/stt/src/types.rs`

### `rhema-bible`

Purpose:

- Open the local SQLite Bible database.
- List translations and books.
- Fetch verses and chapters.
- Search verses through SQLite FTS5.
- Load cross-references.
- Load verses for quotation matching.

Primary paths:

- `src-tauri/crates/bible/src/db.rs`
- `src-tauri/crates/bible/src/search.rs`
- `src-tauri/crates/bible/src/lookup.rs`
- `src-tauri/crates/bible/src/crossref.rs`

### `rhema-detection`

Purpose:

- Detect explicit references such as `John 3:16`.
- Match quoted Bible text.
- Run semantic detection through local embeddings when available.
- Track sermon context.
- Manage reading mode.
- Merge results and control duplicate/cooldown behavior.

Primary paths:

- `src-tauri/crates/detection/src/direct/`
- `src-tauri/crates/detection/src/semantic/`
- `src-tauri/crates/detection/src/quotation.rs`
- `src-tauri/crates/detection/src/reading_mode.rs`
- `src-tauri/crates/detection/src/context.rs`
- `src-tauri/crates/detection/src/merger.rs`
- `src-tauri/crates/detection/src/pipeline.rs`

Architecture note:

- `HnswVectorIndex` uses HNSW for approximate candidate retrieval and exact dot-product reranking for final similarity scores.

### `rhema-broadcast`

Purpose:

- Manage NDI runtime sessions.
- Dynamically load the NDI SDK.
- Send RGBA video frames to active NDI outputs.

Primary paths:

- `src-tauri/crates/broadcast/src/lib.rs`
- `src-tauri/crates/broadcast/src/ndi.rs`

### `rhema-api`

Purpose:

- Placeholder for an API layer crate.
- The current Tauri command layer lives directly under `src-tauri/src/commands/`.

### `rhema-notes`

Purpose:

- Placeholder crate for future notes functionality.

## Tauri App Startup

Startup is handled in `src-tauri/src/lib.rs`.

At launch, the app:

1. Loads environment variables from `.env` paths.
2. Installs Tauri plugins for logging, global shortcuts, and local store.
3. Manages shared state:
   - `AppState`
   - `NdiRuntime`
   - `DirectDetector`
   - `DetectionMerger`
   - `ReadingMode`
4. Registers Tauri commands.
5. Locates and opens `data/rhema.db` in development or bundled resource paths in production.
6. Builds the quotation matching index from loaded English verses.
7. Attempts to load the ONNX embedding model and precomputed embeddings.
8. Enables semantic detection only if model, tokenizer, and embeddings are present.

If the Bible database is missing, the app logs a warning and the Bible-dependent commands fail cleanly with "Bible database not loaded".

If the ONNX model or embeddings are missing, semantic search is disabled while direct and quotation detection can still work.

## Shared Backend State

`AppState` lives in `src-tauri/src/state.rs`.

It stores:

- optional `BibleDb`
- `DetectionPipeline`
- `SermonContext`
- `QuotationMatcher`
- active translation ID
- audio/STT atomic flags
- optional Deepgram API key field

Separate managed mutexes also exist for direct detection, merge cooldown state, reading mode, and NDI runtime. This is important because live partial detection can proceed without waiting for slower semantic detection work.

## Command Layer

Tauri commands live in `src-tauri/src/commands/`.

### Audio Commands

`get_audio_devices` returns available input devices from `rhema-audio`.

### STT Commands

`start_transcription` starts the live audio-to-transcript pipeline.

Runtime flow:

1. Guard against duplicate transcription sessions.
2. Resolve the Deepgram API key from settings argument or `DEEPGRAM_API_KEY`.
3. Set STT/audio active flags.
4. Spawn an `audio-fanout` thread.
5. Capture audio at 16 kHz.
6. Emit `audio_level` events for the level meter.
7. Forward audio samples to Deepgram.
8. Spawn the Deepgram WebSocket task.
9. Fall back to REST mode if streaming fails.
10. Consume transcript events.
11. Emit transcript events to the frontend.
12. Run detection strategies.

`stop_transcription` clears the active flags so the capture and STT tasks exit.

### Detection Commands

`detect_verses` runs the full pipeline manually on supplied text.

`detection_status` reports whether direct, semantic, and cloud detection are available.

`semantic_search` runs manual semantic search if model/index assets are loaded.

`quotation_search` runs quotation matching over the loaded quote index.

`toggle_paraphrase_detection` enables/disables synonym expansion mode.

`reading_mode_status` and `stop_reading_mode` expose reading mode state.

### Bible Commands

Bible commands wrap the SQLite DB:

- `list_translations`
- `list_books`
- `get_chapter`
- `get_verse`
- `search_verses`
- `get_cross_references`
- `get_active_translation`
- `set_active_translation`
- `get_translation_verses_for_search`

### Broadcast Commands

Broadcast commands manage secondary output windows and NDI:

- `list_monitors`
- `ensure_broadcast_window`
- `open_broadcast_window`
- `close_broadcast_window`
- `start_ndi`
- `stop_ndi`
- `get_ndi_status`
- `push_ndi_frame`
- `start_obs_overlay`
- `stop_obs_overlay`
- `get_obs_overlay_status`
- `push_obs_overlay`

There are two output IDs today:

- `main` maps to the `broadcast` window.
- `alt` maps to the `broadcast-alt` window.

## Event Flow

Rust emits these primary events:

```text
audio_level
transcript_partial
transcript_final
verse_detections
reading_mode_verse
translation_command
stt_speech_started
stt_error
stt_connected
stt_disconnected
```

The shared event constants currently cover:

- `audio_level`
- `transcript_partial`
- `transcript_final`

Some detection/STT events are string literals in the command code. A later cleanup could centralize all event names to reduce drift.

The React broadcast store emits to secondary windows:

```text
broadcast:verse-update
```

The broadcast output window listens for:

```text
broadcast:verse-update
broadcast:ndi-config
```

It also emits:

```text
broadcast:output-ready
```

## Live Detection Flow

The live STT pipeline uses different detection strategies at different moments.

Partial transcripts:

- emit `transcript_partial`
- run direct detection only

Final transcripts:

- emit `transcript_final`
- check translation voice commands such as "read in NIV"
- run direct detection
- check reading mode
- run quotation matching if direct detection did not find a high-confidence result
- buffer sentence text for semantic detection if direct detection did not handle it

Speech-final or utterance-end signals:

- force flush the sentence buffer
- queue semantic detection for the flushed sentence

Background workers:

- semantic detection worker runs ONNX embedding and vector search without blocking transcript handling
- quotation worker runs quote matching separately

Result emission:

- direct, semantic, quotation, and reading mode detections are converted into frontend `DetectionResult` payloads
- payloads are emitted through `verse_detections`

## Detection Strategies

Direct reference detection:

- uses book names, abbreviations, parser logic, fuzzy helpers, and automaton-style matching
- cheap enough to run on partial transcript fragments
- high-confidence direct results can suppress slower semantic work

Quotation matching:

- indexes loaded Bible verse text
- tries to match transcript snippets against known verse wording
- suppressed while reading mode owns the display

Semantic detection:

- uses `OnnxEmbedder` when model assets exist
- searches precomputed verse embeddings
- can use synonym/ensemble expansion when enabled
- depends on `HnswVectorIndex`, which uses HNSW candidate retrieval with exact reranking

Reading mode:

- starts from a detected direct reference and loads the current chapter
- tracks expected verse progression
- emits contextual detections as the speaker reads through the passage

Merger/context:

- `DetectionMerger` ranks and deduplicates detections
- cooldown state prevents repeated emissions
- `SermonContext` can boost semantic results in the same book/chapter context
- merged detections now include raw score, source-specific thresholds, decision state, and an explanation for the operator

## Frontend Architecture

The root app renders `Dashboard`.

Dashboard panels:

- `TransportBar`: controls transcription and settings.
- `TranscriptPanel`: shows partial and final transcripts.
- `PreviewPanel`: selected verse preview.
- `LiveOutputPanel`: currently live output state.
- `QueuePanel`: queued verses for operator control.
- `SearchPanel`: manual Bible/contextual search.
- `DetectionsPanel`: recent detected verses with confidence/source badges.

State is held in Zustand stores:

- `audio-store.ts`
- `transcript-store.ts`
- `detection-store.ts`
- `settings-store.ts`
- `bible-store.ts`
- `queue-store.ts`
- `broadcast-store.ts`

Hooks are the bridge between UI components, Tauri commands, and stores:

- `use-audio.ts`
- `use-transcription.ts`
- `use-detection.ts`
- `use-bible.ts`
- `use-broadcast.ts`
- `use-tauri-event.ts`

## Broadcast Rendering Flow

Operator flow:

1. A verse is selected manually, from detection, or from the queue.
2. `toVerseRenderData` converts the selected verse into broadcast render data.
3. `broadcast-store` sets `liveVerse`.
4. `broadcast-store` emits `broadcast:verse-update` to `broadcast` and `broadcast-alt`.
5. `broadcast-output.tsx` receives the event in the secondary window.
6. `renderVerse` draws the theme and verse to Canvas 2D.
7. If NDI is active, the output window reads RGBA pixels from canvas.
8. The frame is base64-encoded and sent to Rust through `push_ndi_frame`.
9. `rhema-broadcast` sends the frame through the NDI SDK.

OBS flow:

1. The operator starts the OBS overlay server.
2. OBS loads `http://127.0.0.1:4763/?output=main` or `?output=alt` as a Browser Source.
3. `broadcast-store` pushes each live verse/theme update to Rust through `push_obs_overlay`.
4. The local server sends updates to connected OBS pages through Server-Sent Events.

The broadcast output window also sends periodic keepalive frames when NDI is active.

## Data And Asset Pipeline

The `data/` folder contains setup scripts:

- download public Bible sources and cross-references
- optionally download BibleGateway translations
- build `data/rhema.db`
- export verses for embedding precompute
- export/quantize the Qwen3 embedding model to ONNX
- precompute verse embeddings
- download the NDI SDK

Runtime assets expected by the app:

- `data/rhema.db`
- optional `models/qwen3-embedding-0.6b/...`
- optional `models/qwen3-embedding-0.6b-int8/...`
- optional `embeddings/kjv-qwen3-0.6b.bin`
- optional `embeddings/kjv-qwen3-0.6b-ids.bin`
- optional `sdk/ndi/...`

## Current Architecture Strengths

- Clear Rust crate boundaries for audio, STT, Bible, detection, and broadcast.
- Direct detection is separated from slower semantic detection.
- Detection results expose source, threshold, and review/auto-queue rationale to the frontend.
- OBS Browser Source output is available without NDI hardware or plugins.
- Semantic detection is optional, so missing model files do not break the app.
- SQLite is appropriate for local Bible data and FTS search.
- Tauri secondary windows are a natural fit for broadcast output.
- NDI SDK is loaded dynamically, which keeps NDI optional.
- Frontend state is simple and local through Zustand.

## Current Architecture Risks

- Several event names are string literals instead of centralized constants.
- Direct detection state is separate from the full `DetectionPipeline`, so future changes must preserve cooldown/context behavior carefully.
- Confidence thresholds are currently fixed in code; a later settings UI could expose them safely per source.
- Semantic detection holds `AppState` while processing results; heavy work should remain outside UI-critical paths.
- OBS output currently uses a simple browser overlay rather than full theme-renderer parity.
- ProPresenter is not abstracted yet.
- Tauri CSP is currently disabled in `tauri.conf.json`.
- HNSW graph persistence is not implemented yet, so startup still rebuilds the in-memory graph from embedding files.

## Phase Dependencies

Recommended order after this document:

1. Real HNSW vector search and exact reranking.
2. Explainable confidence and threshold policy.
3. Broadcast output abstraction before adding OBS.
4. Regression corpus before aggressive detection changes.
5. Audio/VAD changes with measurable transcript/detection outcomes.
6. ProPresenter and Tauri security hardening.

This document should be updated whenever a phase changes architectural boundaries, events, commands, or shared data flow.
