# Rhema 3.0 — Phase-by-Phase Implementation Plan

> Binding spec: the `markdowns/PHASE*.md` companions and `markdowns/GEMINI.md` execution
> contract. Background/vision context is [`ARCHITECTURE.md`](ARCHITECTURE.md) (the consolidated
> spec — the old v1/v2 `.docx` files were merged into it). Where the vision spec and a phase
> companion conflict, defer to the companion. [`TRACEABILITY.md`](TRACEABILITY.md) maps every
> spec stage to its phase status (in-roadmap / simplified / deferred).

---

## Scope

Rhema 3.0 is a real-time sermon-intelligence desktop app (Rust/Tauri backend, React/TS
frontend) with an existing working pipeline `audio → stt → detection → broadcast`
(baseline: 128 tests passing). This plan layers a 5-phase hardening roadmap onto that
pipeline. Each phase touches a distinct layer.

| Phase | Layer / Crate | Core deliverable | Solves |
|---|---|---|---|
| 1 | `rhema-audio` | Subband/spectral flux gate, energy-variance gate, RMS adaptive AGC, feedback detector | Stage bleed, compression trap, PA feedback squeal |
| 2 | `rhema-stt` + `rhema-detection` + app consumer | Phonetic normalizer, Levenshtein token-skip, two-stage classifier (+ stage-2 channel stub), 45s suppression cache | Homophones, echo/cyclic loop, "Amen!" splice, rigid command mapping |
| 3 | `rhema-detection` | Formal `CursorState`, epoch lock, negation filter, coordinate bounds, asymmetric range | Cold-state, cross-boundary arithmetic, mid-sentence pivot, race condition |
| 4 | `rhema-broadcast` + Zustand stores | Logical channel model, pub-sub routing, device health monitor | Sync drift, device divergence, screen-bound state |
| 5 | `rhema-detection` + app consumer + `rhema-api` | Time-decay topic vector, priming index, proactive suggestion engine, wire stage-2 to Claude | Semantic drift, thematic ambiguity, outline boosting |

---

## Non-Negotiable Execution Rules (from GEMINI.md)

1. Strict phase order **1 → 2 → 3 → 4 → 5**. No phase starts until the previous phase's
   `cargo test --workspace` passes.
2. **One plan bullet per turn**, then stop and wait for approval.
3. Unidirectional data flow; no new cross-crate workspace deps and no new external crates
   without explicit approval.
4. Each bullet requires a **Reasoning Block before code** and a **self-verification
   checklist after**.
5. Honor the **do-not-touch list**: `rhema-bible/`, `audio/src/capture.rs` entrypoint
   (except under explicit Phase 1 instruction), `data/rhema.db` schema, existing
   `TranscriptEvent` variants, existing Zustand store field names,
   `detection/tests/` fixtures.
6. **Decision points** (two valid approaches) are surfaced and approved, not chosen
   silently.
7. Never delete or skip existing tests. Never use `.unwrap()` in new production paths.
   No raw SQL outside `BibleDb`.

---

## Phase 0 — Recon & Baseline (prerequisite, no production code)

Replace assumptions with ground truth before any phase begins.

- [ ] Verify workspace matches GEMINI.md crate layout. NOTE (RhemaV2): `rhema-api` is no
      longer a stub — it's the multi-provider Stage-2 LLM client (`rhema-api/src/llm/`); and
      the workspace now also has `rhema-vad` + `rhema-transcript`.
- [ ] Run `cargo test --workspace` → confirm the **128-test baseline is green** before touching anything.
- [ ] Read the real entry points each companion references and record interface gaps as explicit "ask"/"decision" items:
  - `audio/src/capture.rs` pipeline ordering (resample → gain → RMS → VAD → forward)
  - `detection/src/pipeline.rs` (incl. the `:33` cloud-boost TODO), `detector/direct.rs`
  - the app-level detection consumer loop (suppression cache + suggestion engine land here)
  - existing `SermonContext`, `ReadingMode`, `DirectDetector`, `SemanticDetector::search_query`
  - `broadcast-store.ts` (`liveVerse`, `isLive`, `activeThemeId`, `altActiveThemeId`)
- [ ] Resolve the **first decision point**: FFT via `rustfft` vs a `SubbandFluxGate` approximation.

---

## Phase 1 — Audio Ingestion & Hardware Gating (`rhema-audio` only)

Insertion points: between resample and gain, and between gain and VAD. VAD pre-buffer
must remain untouched. `AudioFrame` struct unchanged. Feedback-zeroed frames are
forwarded (not dropped) to preserve STT stream continuity.

- [ ] **Bullet 1.1** — Spectral-flux gate (decide `rustfft` vs `SubbandFluxGate` approximation first).
- [ ] **Bullet 1.2** — Local energy-variance gate (320-sample / 20ms window).
- [ ] **Bullet 1.3** — RMS adaptive normalizer / software AGC (200ms attack, 50ms release; replaces static gain; smooth interpolation, no hard limiter).
- [ ] **Bullet 1.4** — Feedback-loop detector (dominant bin >60%, sustained ≥5 frames, 500Hz–8kHz → zero samples but still forward; `warn!` with frequency, no Tauri event).

Exit gate: `cargo test --workspace` green; touched nothing outside `rhema-audio`; every
gate trigger has a log statement; all numeric constants named.

---

## Phase 2 — Speech Normalization & Intent Pipeline (`rhema-stt`, `rhema-detection`, app consumer)

- [ ] **Bullet 2.1** — `rhema-detection/src/normalizer.rs`: `normalize_transcript(&str) -> String`, data-driven rule table, all 8 homophone patterns + ≥3 unit tests.
- [ ] **Bullet 2.2** — Levenshtein token-skip lookahead (6-token window, named exclamation list) inside `DirectDetector` pre-parse, running on normalized transcript.
- [ ] **Bullet 2.3** — Two-stage classifier in `pipeline.rs` (`ControlCommand | ExplicitScriptureRequest | Ambiguous`) + Stage-2 `tokio::sync::mpsc` channel to a logging placeholder (no real Claude yet — Phase 5 depends on this channel existing).
- [ ] **Bullet 2.4** — 45s TTL semantic suppression cache: `VecDeque<(VerseRef, Instant)>` in the **app-level consumer**, keyed on VerseRef, TTL enforced on every check, written when `liveVerse` is set.

Exit gate: tests green; no touch to `rhema-audio`/`rhema-broadcast`.

---

## Phase 3 — Navigation State Machine (`rhema-detection` only)

- [ ] **Bullet 3.1** — Formal `CursorState` / `VersePosition` / `CursorMode` with invariant enforcement; history `VecDeque` cap 50, forward stack cleared on non-redo navigation.
- [ ] **Bullet 3.2** — Epoch lock: `Arc<AtomicU64>` (`SeqCst`) + `lock_until: Arc<Mutex<Option<Instant>>>` in `AppState`; `epoch_at_detection` stamp; discard on stale epoch OR within 500ms window.
- [ ] **Bullet 3.3** — Negation-marker filter (runs after normalizer + token-skip, before merger; discards entities preceding the last marker, locks onto terminal reference).
- [ ] **Bullet 3.4** — Coordinate bounds: all 5 cases (fwd/back cross-chapter, both Bible boundaries, translation omission with ≤3 retries) — via `BibleDb`, never hardcoded counts.
- [ ] **Bullet 3.5** — Asymmetric range rule (forward = `range.end + 1`, backward = `range.start - 1`, always collapse to `CursorMode::Single`).

Exit gate: tests cover epoch discard, negation filter, Genesis 1:31→2:1, Rev 22:21 block,
range collapse; no touch to audio/stt/broadcast.

---

## Phase 4 — Channel Routing Layer (`rhema-broadcast` + Zustand stores)

- [ ] **Bullet 4.1** — Three channel structs (Audience/Pastor/Operator) with `Serialize`; Audience excludes suggestions/queue/confidence.
- [ ] **Bullet 4.2** — Pub-sub wiring: `emit` (all windows) for Audience, `emit_to` for Pastor/Operator; `RoutingMode { Locked, Preview, Independent }`.
- [ ] **Bullet 4.3** — Zustand additive migration: new channel/routing/deviceHealth fields **alongside** existing ones (don't delete), populated only by Tauri event listeners.
- [ ] **Bullet 4.4** — Device health monitor: 2s ping; disconnect preserves channel state; reconnect resyncs current state. Operator-console routing-mode UI toggle.

Exit gate: tests green; no touch to audio/stt/detection; existing `broadcast-store.ts`
fields retained.

---

## Phase 5 — Sermon Intelligence & Suggestion (`rhema-detection`, app consumer, `rhema-api`)

- [ ] **Bullet 5.1** — Exponential time-decay topic vector on `SermonContext` (75/25 two-block, 90s half-life, background window cap 500, lazy compute).
- [ ] **Bullet 5.2** — `PrimingIndex::build_from_notes` via `DirectDetector`, +0.25 boost capped at 1.0, re-sort; `set_sermon_notes` Tauri command storing index in `AppState`.
- [ ] **Bullet 5.3** — `SuggestionEngine` in app consumer: 300s cooldown, threshold 0.72, skip `display_history` + `dismissed_verses`, emit via **Operator channel only**.
- [ ] **Bullet 5.4** — Wire the Phase-2 Stage-2 channel to the real Claude call **iff `rhema-api` is implemented** — else keep the placeholder and note it (requires `reqwest` approval).

Exit gate: tests cover decay window shift, priming boost, cooldown suppression,
dismissal.

---

## Per-Bullet Working Rhythm

For every bullet:

1. **Reasoning Block** (per the phase companion template) before any code.
2. List files to read and files to modify; read them first — do not assume contents.
3. State interface assumptions; if a needed interface does not exist, **ask** — do not invent.
4. Implement the minimal correct change.
5. State what changed, new public types/functions, and what downstream phases need.
6. Run `cargo test --workspace`; report pass/fail.
7. Complete the phase's self-verification checklist.
8. **Stop. Wait for the next instruction.**
