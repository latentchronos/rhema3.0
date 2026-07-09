# Rhema 3.0 — Comprehension Layer: Design & Implementation Plan

**Date:** 2026-07-09
**Branch:** `RhemaV2`
**Status:** Design approved (sequencing + first-spec decisions locked); ready for per-bullet implementation
**Source architecture:** reproduced in full as **Appendix A** (the "Comprehension Layer: Full Architecture (Merged)" document)

---

## 0. Reading guide

This document is self-contained. It embeds the original architecture (Appendix A), records the
design decisions we made, folds in the 2026 technology research (Appendix B, with sources), and
lays out a detailed, bullet-by-bullet implementation plan across all sub-projects.

Section references like "(§11)" point at **Appendix A** unless stated otherwise.

The implementation follows the existing project contract used across Phases 1–5 and the V/L tracks:
**one bullet per turn, a reasoning block, `cargo test --workspace` green with 0 warnings (plus
`tsc`/`vitest` clean for any frontend touch), update the progress memory, then stop for approval.**

---

## 1. The idea in one paragraph

Rhema already detects Scripture in a live transcript (direct / semantic / quotation matching) and
projects it. On top of detection we add a **comprehension layer**: a small local LLM run as a
**continuous observer** that periodically asks *"has my understanding of what the preacher is doing
changed?"* — tracking a discourse state (teaching, storytelling, applying, praying, exhorting,
announcements) and, **independently and in parallel**, grounding the discourse in zero, one, or many
Bible passages. Comprehension never projects; it only **enriches detection's confidence** (evidence
fusion), and **detection remains the sole projection authority** (§11). The whole point is sermons
that don't explicitly cite verses — e.g. a pastor retelling the Prodigal Son — where detection alone
sits below threshold but accumulated context can legitimately raise confidence enough to project.

---

## 2. Codebase reality — what already exists vs. what is genuinely new

The architecture in Appendix A reads as greenfield, but much of its machinery already exists in the
tree. **The fastest way to get this wrong is to rebuild what is already there.**

### Already exists (reuse, do not rebuild)

| Appendix-A concept | Where it lives today |
|---|---|
| **Bible Retrieval / local RAG (§12)** — embed transcript, nearest-neighbour verse search | `rhema-detection` semantic detector (ONNX embeddings) + `bible/src/search.rs`. §12 is a *wiring* job, not a new subsystem. |
| **Detection = projection authority (§11)** | Already true by construction — projection flows through the detection pipeline + confidence threshold. §11 is a rule we already obey. |
| **Sermon working memory (§5, §14C)** | `SermonContext` (`detection/src/context.rs`) already holds a time-decay **topic vector**, priming index, suppression cache. The observer's "current understanding" extends this. |
| **An LLM adapter seam (§16)** | `LlmProvider` trait in `api/src/llm/mod.rs` — but it is **cloud-only** and returns the narrow `{is_scripture, reference, confidence}`. It is a *starting point*, not the comprehension adapter. |
| **Multi-provider cloud client** | `api/src/llm/{anthropic,openai,gemini}.rs` + `classify()` dispatcher, wired via `commands/stt.rs::run_stage2_worker`. |
| **Channel routing / broadcast** | `rhema-broadcast` + `src/channels.rs` (Phase 4) — the path comprehension state will emit through to the UI. |
| **Cursor / nav state machine** | `detection/src/cursor.rs`, `voice_nav.rs` (Phase 3 + Track V). |

### Genuinely new (the real work)

1. **A local, in-process LLM backend (§18).** No `candle`/`llama.cpp`/`mistral.rs` in the workspace
   today — the only LLM is the *cloud* Stage-2 client. This is the single biggest new piece and the
   highest technical risk (CPU-only i5-8265U).
2. **The Observer / Comprehension Scheduler (§5).** Owns the sliding window, refresh interval, the
   "has my understanding changed?" framing, and `NO_CHANGE`/`STATE_CHANGED`. Pure app-side logic.
3. **The comprehension state machine (§6, §7).** Dominant intent + supporting activities + an
   open-ended passage list, recording a transition only on genuine change.
4. **Evidence fusion (§11).** Adjusts detection confidence from comprehension agreement, retrieval,
   state, and local adaptation — *before* the threshold check. Detection still owns projection.
5. **Comprehension storage + memory management (§13, §17, §19).** Separate `detections` and
   `context_states` logs, a rolling "sermon arc" delta summary, and a session-only-vs-persistent
   SQLite backend toggle.
6. **Local adaptation (§14B)** — later layer.

---

## 3. Design decisions (locked)

These were discussed and approved before writing this doc.

### D1. Engine-first (model-free), then local Qwen directly — with an early build-spike

*(Decision refined 2026-07-09: build directly on the local Qwen model; the cloud path is dropped from
the critical path.)*

Build the Observer, state machine, storage, and fusion as **model-free logic validated against a
scripted mock model** (no network, no llama.cpp) — that is Spec #1 and it needs **no real backend at
all**. Then wire the **local Qwen (llama.cpp) backend directly**. The cloud client is **not** on the
critical path: the two-trait design still supports a cloud `ComprehensionModel` if ever wanted, but
the shipped product is local-only per §18, so we build what we ship.

**De-risk the scary part early.** The single highest-risk task is getting `llama.cpp` to build/link
and run Qwen on the i5 (the same class of native-build friction already hit with `local-stt` + BLAS
linking). So **before** committing to it, do a throwaway **build-spike (B0)**: confirm llama.cpp
compiles on the target, loads Qwen3-4B Q4_K_M, and answers one prompt inside the latency budget. Run
it **in parallel with the engine work (Phase C)** so it never blocks Spec #1 and the native-build risk
is retired first.

**The discipline this keeps:** because we build directly on the 4B, we feel its real limits from day
one — so **every prompt, schema, and memory budget is sized for the 4B** (short fixed prompts, strict
schema, single-delta rolling summary, grammar-shaped output). No smart cloud model can mask a design
that is too demanding for what actually ships.

### D2. A new, model-free `rhema-comprehension` crate

All comprehension intelligence lives in a new crate that is **DB-free and model-free**, exactly
mirroring how `rhema-detection` is testable without a network or a model. It owns: the state machine,
the observer scheduler, the incremental-prompt builder, the rolling summary, the fusion math, and the
`ComprehensionModel` **adapter trait**. Every *concrete* backend (the local llama.cpp adapter; an
optional cloud one) lives **outside** it. This preserves the single most valuable property the detection crate already has:
the intelligence is fully unit-testable against a **mock model**.

### D3. Two traits, not one

`LlmProvider` (existing) and `ComprehensionModel` (new) are **different jobs**:

- `LlmProvider::classify` — *fast, per-ambiguous-utterance* scripture check, narrow output. **Stays as-is.**
- `ComprehensionModel` — *periodic observer* call: takes a prompt + a JSON schema, returns validated
  structured output, plus `capabilities()`, `health()`, `init`/`shutdown`. Richer output shape (§8).

They **share the reqwest plumbing** (one cloud provider struct can implement both), so HTTP code is
not duplicated — but the engine depends **only** on `ComprehensionModel`, never on a concrete
provider. That is what keeps the local swap a config change.

### D4. The observer loop = Σ/Δ delta prompting

Adopt the 2026 "stable state Σ + emit only the delta Δ" formalization (RGMem; Appendix B §3). Hold a
stable state object Σ (`intent, running_summary, active_topic, open_threads, passages`); each tick
feed **compact Σ + only the new transcript delta**, and have the model emit **only the Δ** — commit
it only if non-empty. This maps 1:1 onto `NO_CHANGE`/`STATE_CHANGED` (§5.4) and keeps prompts short
enough for a 4B model. This is the concrete realization of §5.3 "verification over rediscovery."

### D5. Fixed interval **plus** optional topic-shift early trigger

§5.2 triggers the observer on a fixed, configurable interval (the CPU-budget ceiling). The 2026
research (Appendix B §3) prefers triggering on **topic shift**. We take both: fixed interval as the
ceiling, **plus** an optional early trigger when the **existing `SermonContext` topic vector** (Phase
5.1) moves sharply (a cheap cosine jump — free code we already ship). Best of both, no new model
calls wasted, CPU load still bounded. Configurable; the topic-shift trigger is opt-in.

### D6. `Capabilities` discovery at construction

Following `rig`/`genai` (Appendix B §2): `ComprehensionModel` carries a typed `Capabilities`
(`max_context_tokens`, `supports_structured_output`, `supports_streaming`, `name`) read **once at
load**, not per call. The engine uses `max_context_tokens` to size the §17 injected memory — a 4B
local model gets a trimmed single-delta summary; a large cloud model gets a richer rollup. The engine
does not hardcode this.

### D7. Grammar-constrained JSON for the local backend

The local backend (§18) enforces the §8 schema with llama.cpp's native **GBNF grammar** generated
from our JSON schema (Appendix B §3 / Local-inference brief). This turns "ground it in a passage"
into a **grammar-enforced** `Book Chapter:Verse` shape and constrains the intent field to our exact
enum — a guarantee, not a hope. Any non-grammar backend (the `MockModel`, or an optional cloud
adapter) instead relies on the tolerant JSON extraction in `schema.rs`.

---

## 4. Recommended technology stack (2026 research — sources in Appendix B)

### Language allocation (matches the "use each where it's strongest" ask)

- **Rust** — everything in the engine, state machine, fusion, storage, adapter trait, and all
  wiring (sub-projects A, C, D, E, F, G). Stateful, latency-sensitive, must live in-process next to
  detection. No reason to leave Rust.
- **C / C++ via llama.cpp** — sub-project B only. We **do not** write an inference kernel; we bind to
  `llama.cpp` (mature C/C++ runtime, best CPU quantized inference + native GBNF). Your C/C++
  background is for building/tuning/troubleshooting the native layer, not writing it from scratch.
  Embedding `llama-cpp-2` needs `clang` at build time (same class of native-build gotcha as the
  existing `local-stt` BLAS linking).
- **Python** — offline tooling only, **never** a shipped runtime dependency: one-off scripts to
  build/quantize the embedding index, run `llama-bench`, or evaluate model/quant choices on recorded
  sermon clips. It must never enter the app's runtime.

### Concrete stack

| Concern | Recommendation | Notes |
|---|---|---|
| **Local inference backend** | `llama-cpp-2` (utilityai/llama-cpp-rs), embedded **in-process** | No sidecar/port/IPC for a once-a-minute call. Needs `clang`. Ollama = prototype only, not shipped. |
| **Local model** | **Qwen3-4B-Instruct @ Q4_K_M** primary; **Qwen3-1.7B** fast fallback | ~2.5 GB; est. ~6–10 tok/s on the 8265U → ~100-token JSON worst-cases ~12 s, inside a once-a-minute budget. **Benchmark on the real machine before locking the default.** |
| **Structured output (local)** | llama.cpp native **GBNF** from JSON schema; `llguidance` as a compile-flag upgrade | Constrains intent enum + passage pattern. |
| **Quantization** | **Q4_K_M** default; Q5_K_M if labels wobble; Q8 only for debugging | Classification-shaped tasks are quant-resilient. |
| **Adapter-trait shape** | Copy `rig`'s `CompletionModel` + `Capabilities` and `genai`'s native-protocol/capability negotiation | Copy the *shape*; no dependency required. |
| **Verse index (later, §12 upgrade)** | **usearch** (C++ FFI, mmap HNSW) or **hannoy** (pure-Rust, LMDB) at multi-translation scale; keep brute-force cosine as oracle | Current cosine-over-verses is fine at today's scale; upgrade when adding translations. |
| **Embeddings (later)** | consider **fastembed-rs** for dense/sparse + **cross-encoder reranking** (helps paraphrase grounding) | Reranking materially improves verse disambiguation. |
| **Deterministic fast-path (optional)** | port **openbibleinfo/Bible-Passage-Reference-Parser** as a pre-LLM detector + eval corpus (4.7M reference strings) | Cheap wins + a labeled test set. |

**Do not adopt as core inference in 2026:** Kalosm/Floneum (next-gen "Fusor" backend is
pre-production), `candle` as a runtime (low-level; trails llama.cpp on CPU), `rustformers/llm`
(unmaintained), `sqlite-vss` (unmaintained). Borrow Kalosm's *structured-generation technique* only.

---

## 5. Sub-project decomposition & build order

This document is ~6 sub-systems. They are decomposed into independently shippable pieces. The key
unlock (D1): **the engine (C) is built and fully validated against a scripted mock model** — no real
backend — so the observer architecture is proven independently of any model. The local backend (B) is
then wired directly, with its highest-risk part retired early via the **B0 build-spike (run in
parallel with C)**.

| # | Sub-project | Spec | Depends on | Risk |
|---|---|---|---|---|
| **A** | Comprehension adapter trait (§16) — folded into C | #1 | — | low |
| **C** | Observer engine + state machine (§5–§8), model-free crate, mock-validated | #1 | A | medium |
| **B** | Local LLM backend (§18) — llama.cpp/Qwen, CPU-first; **early build-spike B0** | #2 | A | **high** |
| **I** | App integration — wire the Observer in, settings, emit state to UI (backend-agnostic) | #3 | C, B | low |
| **D** | Comprehension storage (§13/§17/§19) | #4 | C | medium |
| **E** | Evidence fusion (§11) into the confidence path | #5 | C, B, detection | medium (correctness) |
| **F** | Comprehension UI (§13) — timeline + detection join | #6 | D | low |
| **G** | Local adaptation (§14B) | #7 | D, E | low, later |

The old "cloud wiring" phase is dropped from the critical path — a cloud adapter stays an optional
extra the trait supports, not a required step. Do the **B0 build-spike as early as possible, in
parallel with Phase C**, so the native-build risk is retired before it can block anything.

---

## 6. Detailed design — Spec #1: the Comprehension Engine (`rhema-comprehension`)

The first and foundational spec. Everything here is **model-free and network-free**, validated
against a `MockModel`.

### 6.1 Crate layout

```
src-tauri/crates/comprehension/
  Cargo.toml            # deps: serde, serde_json, thiserror  (NO reqwest, NO model libs)
  src/
    lib.rs              # re-exports
    types.rs            # DominantIntent, SupportingActivity, PassageRef, ComprehensionState, StateTransition
    schema.rs           # OutputSchema, Decision, JSON contract (§8), tolerant parse/validate
    model.rs            # ComprehensionModel trait, Capabilities, ModelHealth, ModelError, MockModel
    window.rs           # TranscriptWindow — time-bounded sliding buffer (§3)
    summary.rs          # RollingSummary (Σ) — per-state micro-records + delta condense (§17)
    observer.rs         # Observer — scheduler, should_evaluate, build_prompt (Σ/Δ), apply_decision
    fusion.rs           # (Spec #4) EvidenceFusion — pure confidence math (§11)
```

Added to `[workspace] members`. Consumed by the app (`src-tauri/src/`), never by `rhema-detection`
(keeps the dependency graph one-directional: app → {detection, comprehension}).

### 6.2 Core types (`types.rs`, `schema.rs`)

```rust
pub enum DominantIntent { Idle, Teaching, StoryTelling, Applying, Praying, Exhorting, Announcements }

pub enum SupportingActivity { QuotingScripture, Illustrating, Exhorting, Praying, /* … */ }

pub struct PassageRef { pub reference: String /* canonical "Luke 15:11-32" */ }

pub struct ComprehensionState {
    pub intent: DominantIntent,
    pub supporting: Vec<SupportingActivity>,   // §6 — logged, not competing top-level states
    pub passages: Vec<PassageRef>,             // §7 — OPEN-ENDED: zero, one, or many; may be empty (§9)
    pub topic: Option<String>,                 // §8 — OPTIONAL, never invented to fill the field
    pub confidence: f32,
}

pub enum Decision {                            // §5.4 — exactly two outcomes
    NoChange,
    StateChanged(ComprehensionState),
}

pub struct StateTransition {                   // recorded only on a genuine change (§7)
    pub from: Option<ComprehensionState>,
    pub to: ComprehensionState,
    pub at_ms: u64,                            // caller-supplied timestamp (time is injected, see §6.6)
}
```

`schema.rs` owns the `OutputSchema` description (used to build the observer prompt and, for the local
backend, to generate the GBNF grammar) and a **tolerant** parser mirroring the existing `parse_stage2_json` (extract the
outermost `{…}`, tolerate prose/fences). Zero passages is a valid parse (§9), never an error.

### 6.3 The adapter trait (`model.rs`) — §16 + D3/D6

```rust
pub struct Capabilities {
    pub name: String,
    pub max_context_tokens: usize,       // drives §17 memory sizing
    pub supports_structured_output: bool,// grammar / json-schema at the boundary
    pub supports_streaming: bool,
}

pub enum ModelHealth { Ready, Loading, Error(String) }

#[allow(async_fn_in_trait)]
pub trait ComprehensionModel {
    fn capabilities(&self) -> &Capabilities;                 // read once at construction (D6)
    async fn health(&self) -> ModelHealth;
    async fn infer(&self, prompt: &str, schema: &OutputSchema) -> Result<Decision, ModelError>;
    // init/shutdown are the constructor / Drop of the concrete adapter.
}

/// Test double: returns a scripted sequence of Decisions. Enables full engine testing
/// with no network and no model — the crate's core value (mirrors detection's MockBible).
pub struct MockModel { /* scripted responses + fixed Capabilities */ }
```

### 6.4 Transcript window (`window.rs`) — §3

Time-bounded sliding buffer: push new segments to the front, evict by **age** (default ~60 s), never
content-bounded (§3 — even a fast speaker adds only a modest token count in a fixed window). Provides
the current window text and a `delta_since(marker)` for Σ/Δ prompting (D4). Pauses **extend**, never
clear, the buffer (§3).

### 6.5 Rolling summary (`summary.rs`) — §17

Holds Σ across ticks and keeps memory bounded over a 4-hour service:
- **Per-state micro-record** — each resolved state written as a compact `context_states` entry
  (intent, passage list, supporting activities, start/end).
- **Rolling delta summary** — periodically condense micro-records since the last rollup into one
  short incremental update; passage lists compress ("covered Luke 15 and Luke 19 during the story").
- **Sized to `Capabilities::max_context_tokens`** (D6) — small local model gets the latest single
  delta with a trimmed passage list; a large-context model gets a richer rollup.

### 6.6 Observer (`observer.rs`) — §5, the heart

Time is **injected** (caller passes `now_ms`/`Instant`), matching `SuppressionCache::filter_at` and
`SermonContext::ingest_embedding_at`, so the whole scheduler is deterministically unit-testable.

```rust
pub struct ObserverConfig {
    pub interval_ms: u64,                 // §5.2 fixed ceiling (default conservative, ~60s — §15)
    pub topic_shift_trigger: bool,        // D5 opt-in early trigger
    pub topic_shift_threshold: f32,       // cosine-distance jump that counts as a shift
}

pub struct Observer {
    current: Option<ComprehensionState>,  // Σ
    summary: RollingSummary,
    window: TranscriptWindow,
    config: ObserverConfig,
    last_eval_ms: u64,
}

impl Observer {
    pub fn ingest_segment(&mut self, text: &str, at_ms: u64);
    pub fn should_evaluate(&self, now_ms: u64, topic_shift: Option<f32>) -> bool; // interval OR shift
    pub fn build_prompt(&self) -> String;                 // Σ (compact) + Δ (new transcript only)
    pub fn apply_decision(&mut self, d: Decision, now_ms: u64) -> Option<StateTransition>;
    //   NoChange      → current continues, no record, returns None (§5.4)
    //   StateChanged  → record transition, reset passage list going forward, returns Some (§7)
    pub fn clear_session(&mut self);                      // reset Σ + window + summary
}
```

The **application** owns the async loop (background task): on the interval tick (or topic-shift
signal) it calls `should_evaluate` → `build_prompt` → `model.infer` → `apply_decision`, then emits any
transition. The Observer itself never touches the model or the clock — it is pure logic.

### 6.7 Testing strategy (Spec #1)

100% against `MockModel`, no network, no model:
- serde round-trips for every enum/struct; schema parse of `NO_CHANGE`, `STATE_CHANGED`, empty
  passages, and rejection of malformed JSON.
- window eviction/overlap/bound; summary append/condense/size-cap-to-capabilities.
- observer: no eval before interval; eval after interval; topic-shift early trigger; `NoChange` keeps
  state and records nothing; `StateChanged` records a transition and resets passages.
- **End-to-end**: a scripted `MockModel` drives a simulated sermon → a sequence of decisions → an
  asserted state timeline (e.g. `IDLE → STORY_TELLING(Luke 15) → APPLYING`), proving the loop with no
  model dependency.

---

## 7. Detailed implementation plan (all sub-projects, bullet-by-bullet)

Each bullet is one turn under the project contract (reasoning block; `cargo test --workspace` green /
0 warnings; frontend touches also `tsc` + `vitest` clean; update the progress memory; stop for
approval). Bullets are sized to be individually reviewable, matching the Phase-1–5 / V / L cadence.

### Phase C — Comprehension Engine (Spec #1) · `rhema-comprehension`, model-free

- **C1 — Crate scaffold + types.** New `crates/comprehension` (serde/serde_json/thiserror only),
  add to workspace. `types.rs`: `DominantIntent`, `SupportingActivity`, `PassageRef`,
  `ComprehensionState`, `StateTransition` with serde. Tests: serde round-trips, enum coverage.
- **C2 — Output schema + JSON contract (§8).** `schema.rs`: `OutputSchema`, `Decision`, tolerant
  parse/validate (extract outermost `{…}`; tolerate prose/fences; empty passages OK; malformed →
  error). Tests: parse both decisions, empty-passage validity, garbage rejection.
- **C3 — Adapter trait + MockModel (§16/D3/D6).** `model.rs`: `ComprehensionModel`, `Capabilities`,
  `ModelHealth`, `ModelError`, `MockModel` (scripted). Tests: mock returns scripted decision;
  capabilities read once.
- **C4 — Transcript window (§3).** `window.rs`: time-bounded sliding buffer, age eviction,
  `delta_since`, pause-extends-not-clears. Time-injected. Tests: eviction, overlap, delta.
- **C5 — Rolling summary Σ (§17).** `summary.rs`: per-state micro-records + delta condense + size to
  `Capabilities::max_context_tokens`. Tests: append, condense, cap.
- **C6 — Observer scheduler (§5, D4/D5).** `observer.rs`: `ingest_segment`, `should_evaluate`
  (interval OR topic-shift), `build_prompt` (Σ/Δ), `apply_decision` (NoChange vs StateChanged →
  transition + passage reset), `clear_session`. Time-injected. Tests: interval gating, topic-shift
  trigger, NoChange keeps state, StateChanged records + resets.
- **C7 — End-to-end engine test.** Scripted `MockModel` drives a simulated sermon → asserted state
  timeline. No network, no model. Closes Spec #1.

### Phase B — Local LLM backend (Spec #2) · §18 · **the risky part — spike it EARLY**

Do **B0 in parallel with Phase C** so the native-build risk is retired before it can block anything.

- **B0 — Build-spike (throwaway).** Confirm `llama-cpp-2`/`llama.cpp` **compiles and links on the
  i5** (expect the `local-stt`/BLAS class of friction), **loads Qwen3-4B Q4_K_M**, and answers one
  hard-coded prompt **inside the latency budget** (`llama-bench` + a one-shot infer). Not wired to
  anything — a go/no-go on the hard part. Deliverable: a documented tok/s number + the exact build
  incantation for this machine. (If it fights us badly, fall back to the 1.7B or reconsider the
  Ollama-sidecar shape — decision recorded here.)
- **B1 — `llama-cpp-2` adapter behind `ComprehensionModel`.** New adapter crate/module;
  feature-flagged (like `local-stt`); bundle a pinned GGUF; `clang` in the build. Load once at
  startup (like Whisper). Tests: load + one infer on a tiny model (integration, feature-gated).
- **B2 — GBNF grammar from `OutputSchema` (D7).** Generate a grammar constraining the intent enum +
  passage `Book Chapter:Verse` pattern; attach as the sampler. Tests: grammar rejects off-enum
  output; passage shape enforced.
- **B3 — Capability reporting + health (D6).** Report `max_context_tokens`, structured-output = true;
  `health()` reflects load state. Verify the engine sizes Σ to the reported context.
- **B4 — Benchmark + model picker.** Firm up B0's numbers; decide 4B vs 1.7B default; runtime model
  picker mirroring the existing STT model dropdown. Document tok/s on the real machine.

*(A cloud `ComprehensionModel` adapter is intentionally NOT on the critical path. The two-trait design
supports one if ever wanted — reuse `api/src/llm` — but the shipped product is local-only per §18.)*

### Phase I — App integration (Spec #3) · backend-agnostic

Wire the (now real, local) observer into the running app. Needs Phase C **and** Phase B.

- **I1 — Wire the Observer into the app.** New `commands/comprehension.rs` + a background task
  (analogous to `run_stage2_worker`) fed by **final** transcript segments; managed `Mutex<Observer>`;
  interval from settings; gated by `session_active` (reuse the Phase-5 gate). Emits transitions. Runs
  against whatever `ComprehensionModel` is loaded (local Qwen today).
- **I2 — Settings UI.** Comprehension **enable** toggle + **refresh interval** picker (10/15/20/30/
  45/60 s, default conservative per §15) + the local model picker (from B4). `tsc`/`vitest` clean.
- **I3 — Emit state to frontend.** Route comprehension transitions through the existing broadcast/
  channel layer; a minimal read-only timeline for dev validation (full UI is Phase F).

### Phase D — Comprehension storage (Spec #4) · §13/§17/§19

- **D1 — Storage module + backend toggle (§19).** SQLite via `rusqlite`; **one schema, two
  backends** — session-only (`:memory:`) vs persistent (file), chosen at startup from a small
  always-persisted prefs file. Two tables: `detections` (point-in-time) and `context_states`
  (duration). Tests: open both backends, insert/read each log.
- **D2 — Rolling sermon-arc persistence + retention (§17/§19).** Persist the rolling delta summary;
  retention setting (keep N days/services, manual clear/export, soft size ceiling). Tests: retention
  prune, export.
- **D3 — Join-at-display query (§13).** Given a detection timestamp, return the `context_state`
  active then (with its passage list + supporting activities). Tests: join correctness across
  overlapping states.
- **D4 — At-rest encryption (optional, §19).** Encrypt the persistent DB file (church-sensitive
  content). Feature-gated; session-only mode unaffected.

### Phase E — Evidence fusion (Spec #5) · §11 (correctness-sensitive)

- **E1 — `fusion.rs` pure math (§11).** `EvidenceFusion` combines base detection confidence +
  comprehension agreement + retrieval + current state + local adaptation → adjusted confidence.
  Documents the **weighting strategy** and the authoritative/corroborative/advisory classification
  (answers §21). Pure, unit-tested, no I/O.
- **E2 — Wire fusion into the confidence path.** Insert fusion **before** the detection threshold
  check; **detection retains projection authority** (§11.1) — comprehension only strengthens/weakens.
  Tests: fusion never bypasses the threshold; projection still flows through detection.
- **E3 — Low-confidence narrative scenario (§11.3).** Reproduce the Prodigal Son case: detection 0.41
  → comprehension `Storytelling`+Luke 15 → fused confidence crosses threshold → projection via
  detection. Integration test with recorded/synthetic input.

*(Phase B — the local LLM backend — is documented above, immediately after Phase C, per the
local-first sequencing. Run its **B0 spike early**, in parallel with Phase C.)*

### Phase F — Comprehension UI (Spec #6) · §13

- **F1 — Comprehension Timeline panel.** Duration-oriented: start/end, dominant intent, supporting
  activities, grounded passages, confidence, optional topic, `NO_CHANGE`/`STATE_CHANGED`. `tsc`/
  `vitest` clean.
- **F2 — Recent-Detection join (§13).** Each recent detection shows the active comprehension state at
  its timestamp as surrounding context (uses D3's query).

### Phase G — Local adaptation (Spec #7) · §14B (later)

- **G1 — Lightweight per-install stats store.** Recurring wording, frequent confirmed passages,
  operator corrections — additive statistics, never model retraining. **Adapt slowly** (§14): require
  accumulated consistent evidence before influencing confidence.
- **G2 — Feed adaptation as fusion evidence.** Wire G1 into `EvidenceFusion` as one more
  advisory input; reset/disable controls (§21).

### Optional / opportunistic

- **Deterministic BCV fast-path.** Port `openbibleinfo/Bible-Passage-Reference-Parser` as a pre-LLM
  reference detector + eval corpus (Appendix B §4). Slots into detection, independent of comprehension.
- **Verse-index upgrade (§12).** Move brute-force cosine → `usearch`/`hannoy` when translations are
  added; add `fastembed-rs` reranking for paraphrase grounding. Independent, do when scale demands.

---

## 8. Open questions (§21) — proposed answers

The architecture asks these to be answered explicitly. Proposed positions (finalized in the relevant
spec):

1. **ASR errors lower detection but comprehension is strong** → fusion may raise confidence, but the
   threshold and detection authority still gate projection (E1/E2). Comprehension is *corroborative*,
   never *authoritative*.
2. **When may comprehension influence detection?** → always as fusion evidence, never as a bypass
   (§11.1). Only *before* the threshold check.
3. **Weighting strategy** → detection = authoritative base; comprehension agreement + retrieval =
   corroborative multipliers within a bounded range; local adaptation = small advisory nudge, slow to
   accrue (§14). Exact weights tuned in E1 with recorded data.
4. **Conflicting evidence** → conflicts *lower* confidence rather than letting the loudest source win
   (§10 — make it hard for multiple wrong sources to agree).
5. **When does local adaptation begin influencing confidence?** → only after accumulated consistent
   evidence (§14 "adapt slowly"); one isolated event never moves it.
6. **Reset/disable adaptation** → explicit user control + per-session clear (G2).
7. **Disagreement logs** → stored in the `context_states`/`detections` cold logs (§13/§17), analysed
   offline (product learning §14A).
8. **Authoritative / corroborative / advisory** → detection authoritative; comprehension + retrieval
   corroborative; local adaptation advisory (matches answer 3).

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **CPU inference too slow on the i5** (highest) | Engine (C) proven against a `MockModel` before B; the **B0 spike** retires the build/latency risk early (parallel with C); once-a-minute cadence gives ~12 s headroom; 1.7B fallback; `llama-bench` before locking default (B4). |
| **Design over-trusts the model** | D1 discipline — we build directly on the local 4B, so its real limits are felt from day one (short prompts, strict schema, single-delta summary); no cloud model masks them. |
| **`llama-cpp-2` native build friction** (`clang`/link) | Same class as the existing `local-stt` BLAS gotcha; feature-gate it; document the build in the crate. |
| **Comprehension hallucinates a state/passage** | Detection is the hallucination firewall (§17) — verse detections never depend on the LLM; worst case is a mislabeled context tag. `NO_CHANGE` + optional/nullable tags reduce fabrication surface. |
| **State oscillation / thrash** | Σ/Δ verification-over-rediscovery (D4), `NO_CHANGE` bias, dominant-intent (not seconds-counted) selection (§6). |
| **Memory growth over a 4-hour service** | Rolling delta summary + cold-storage logs (§17); window is time-bounded, not content-bounded (§3). |

---

## 10. What "done" looks like per spec

- **Spec #1 (C):** `rhema-comprehension` crate compiles, `cargo test --workspace` green / 0 warnings,
  the end-to-end mock-driven state-timeline test passes. No network, no model.
- **Spec #2 (B):** the **B0 spike** proved llama.cpp builds/runs Qwen on the i5 within budget; the
  local `ComprehensionModel` adapter loads Qwen and returns grammar-valid JSON; benchmarked on target.
- **Spec #3 (I):** the observer runs **live in the app on the local model**; transitions appear in a
  dev timeline; session-gated; interval + model configurable in settings.
- **Spec #4 (D):** both storage backends work; retention + join-at-display verified.
- **Spec #5 (E):** the Prodigal Son scenario projects via fused confidence, through detection.
- **Spec #6 (F):** timeline + detection-join UI.
- **Spec #7 (G):** slow local adaptation feeds fusion.

---

# Appendix A — Comprehension Layer: Full Architecture (Merged)

*Reproduced in full (source-of-truth architecture). This merges the original Comprehension Layer
architecture with the later addendum (Evidence Fusion, Scripture Grounding, Local Adaptation &
Comprehension Integration); the addendum is treated as part of the core architecture.*

## 1. The Core Idea

Rhema already has a working detection pipeline: direct verse matching, semantic matching, and
quotation detection, running against a continuous live transcript. On top of detection sits a
**second, higher-level layer** — comprehension — that understands *what the speaker is doing* at any
given moment: quoting a verse, retelling a Bible story, applying a lesson, or just speaking generally.
This lets the system reason about context (e.g. "he's in the middle of retelling the Prodigal Son")
even when no explicit verse number is spoken.

Detection answers: *"Does this text match a verse?"* Comprehension answers two things in parallel:
*"What is the speaker doing right now?"* and *"Can this be grounded in Scripture?"* (Section 9). These
are treated as distinct jobs, not one merged process.

## 2. Detection vs. Comprehension — Running in Parallel

The two processes run **asynchronously and concurrently**, not one blocking the other:
- **Detection** is fast, operates on small units (a sentence or phrase), and produces an instant
  confidence score. It stays fast because Auto Mode depends on immediate response.
- **Comprehension** is slower and deliberately works on a wider window of speech, because
  understanding "is this a story or a quote" requires more context than one sentence.

Neither waits on the other. They both read from the same live transcript stream, on different clocks.

## 3. The Rolling Window Is Context, Not the Thing Being Classified

The ~60-second rolling transcript window is **evidence**, not a unit that gets labeled.
- New segments are added to the front; old ones fall off the back — a *sliding* window, not a
  *tumbling* one; content overlaps between passes. A tumbling window would risk severing a story
  mid-thought.
- Analysis is triggered on the **configured refresh interval** (5.2); pauses extend rather than clear
  the buffer, since a pause doesn't mean a change of topic.
- The buffer is **time-bounded, not content-bounded** — even a fast speaker only adds a modest number
  of extra tokens within a fixed window. It never grows unbounded.

**Why overlapping windows matter:** consecutive windows overlap heavily by design (0–60s, 10–70s,
20–80s…). That overlap should **not** be removed — without it the model loses continuity. The real
optimization is not "send less transcript"; it's **"don't make the model rediscover what it already
knows."** Each pass is framed as: *here is the current understanding — here is the newly updated
transcript — has anything actually changed?*

## 4. The Observer-Based Mindset

The comprehension model should be treated as a **continuous observer** — like a human listener
following a sermon — not a classifier that repeatedly labels windows in isolation. A human doesn't
forget the last minute every time a new sentence is spoken; they carry understanding forward and
revise only when there's real reason. Every pass asks *"has my understanding changed?"* rather than
*"what is happening from scratch?"*

**Context vs. State:** *Context* is the evidence available at decision time (rolling window, previous
state, rolling summary, previously-identified passages and supporting activities) — never the stored
result. *State* is the model's current conclusion about what the pastor is primarily doing
(`STORY_TELLING`, `TEACHING`, `APPLYING`, `PRAYING`, `ANNOUNCEMENTS`, `EXHORTING`). State selection is
evidence-based: transcript → evidence extraction → evaluation → state decision.

## 5. Observer Architecture — Scheduling, Reuse, Separation of Responsibilities

**5.1** The "observer" is **not** the LLM — it's an application component that decides **when** the
LLM is consulted. Pipeline: Microphone → STT → Rolling Transcript Buffer → Comprehension Scheduler
(Observer) → LLM → Updated Comprehension State.

**5.2 Configurable refresh interval** — a user-facing setting trading responsiveness vs CPU load.
Capable hardware → shorter (e.g. 10s); slower CPUs → longer (30–60s). Default ~10s is reasonable, but
a default nearer 60s is favored for CPU-only compatibility. **This interval only affects
comprehension** — detection runs continuously and independently.

**5.3 Previous understanding is always reused** — never reason from scratch if a previous
understanding exists. Frame as *"here is what I currently understand — does new evidence change it?"*

**5.4 Only two outcomes** — **No change** (existing understanding still valid; no new record) or
**State changed** (record a transition, begin tracking the new state with fresh supporting activities
and passage list).

**5.5 Separation of responsibilities** — the **observer** maintains the buffer, the previous
understanding, the interval, invokes the LLM, and updates the state machine; it does **not** decide
the state. The **LLM** interprets evidence, decides whether understanding holds, identifies a new
dominant intent only if warranted, returns structured output, and never invents transitions.

**5.6 Verification over rediscovery** — always ask *"is the current understanding still correct, or
should it be revised?"* This reduces reasoning, stabilizes state, minimizes oscillation.

## 6. Dominant Intent vs. Supporting Activities

A pastor frequently does several things at once. **Dominant intent** = what he's *mainly* trying to
accomplish, judged by communicative purpose, **not** seconds spent (quote John 3:16 then explain it
for a minute = teaching; read one verse mid-David-and-Goliath = still storytelling). **Supporting
activities** = secondary actions alongside it (e.g. `QUOTING_SCRIPTURE` inside `TEACHING`) — logged,
not competing top-level states. Momentary references no longer flip the state.

## 7. The State Model

Modes, e.g.: `IDLE → STORY_TELLING → APPLYING → PRAYING → EXHORTING → ANNOUNCEMENTS`. Each entry
captures dominant intent, start/end, supporting activities, and passages — and only gets a new entry
on a genuine state change. **A state can hold one or many passages**: an **open, variable-length
list** (zero, one, several) that grows while the dominant intent is unchanged. A transition doesn't
require exactly one passage to "resolve"; a genuinely new intent starts a new entry and resets the
list.

## 8. Tag Design Principles

A tag must be **Observable** (inferable from transcript evidence), **Optional** (never mandatory;
omit/null if absent), **Stable** (no oscillation), have a clear **Purpose** (answer one architectural
question), and add **No duplication**. **Topic specifically must not be mandatory** — many states
(prayer, announcement, bare reference) have no meaningful topic; forcing one increases hallucination.

Example — no change:
```json
{ "decision": "NO_CHANGE" }
```
Example — shifted:
```json
{
  "decision": "STATE_CHANGED",
  "new_state": {
    "state": "TEACHING",
    "supporting_activities": ["QUOTING_SCRIPTURE"],
    "passages": ["John 3:16"],
    "topic": null,
    "confidence": 0.87
  }
}
```

## 9. Scripture Grounding — A Standing Responsibility

A **permanent** responsibility, not a byproduct. Every evaluation has two independent, parallel
objectives: (1) determine the dominant communicative intent; (2) independently determine whether the
discourse can be grounded in one or more passages. Grounding is always attempted regardless of the
active state and may produce zero, one, or many passages. **Zero passages is valid and must never be
treated as failure** — never invent passages to populate metadata. Examples: Storytelling + Luke 15;
Teaching + John 3; Prayer + no passage; Applying + Matthew 18 + Colossians 3.

## 10. Evidence Rather Than Guessing

Engineering objective: **make it increasingly difficult for multiple independent sources of evidence
to all agree on the same wrong passage.** Combine independent evidence sources, each contributing
evidence not authority: direct verse detection, semantic matching, quotation detection, Bible
retrieval (§12), comprehension reasoning, current sermon state, previous understanding, local
adaptation (§14), operator confirmation (future).

## 11. Evidence Fusion and the Projection Authority

**11.1 Detection remains the projection authority.** Projection is always owned by Detection.
Comprehension never directly projects — it enriches detection with contextual evidence. Projection
flows: Detection → Confidence Evaluation → Threshold Check → Projection.

**11.2 Complementary evidence providers.** Detection → Base Confidence → Evidence Fusion (Direct
Match · Semantic Match · Quotation Match · Bible Retrieval · Comprehension Agreement · Current State ·
Local Adaptation · future) → Adjusted Confidence → Threshold Check → Projection.

**11.3 Low-confidence narrative sermons.** Prodigal Son example: detection reports Luke 15 at 0.41
(no projection); after ~a minute the observer concludes Storytelling + Luke 15 at 0.94; detection's
own confidence climbs to 0.89 as more transcript arrives; fusion may legitimately raise overall
confidence enough for the threshold to succeed. Projection still occurs through Detection.

**11.4 Why the LLM may suggest an incorrect passage.** It reasons semantically, not
deterministically — it may read an illustration as a biblical narrative. Expected, not failure. The
observer resolves this by accumulating transcript, preserving previous understanding, verifying rather
than rediscovering, and revising as evidence changes.

## 12. Bible Retrieval (Local RAG)

The LLM should not rely solely on pretrained biblical knowledge. **Install-time:** installed Bible
versions are embedded and indexed locally. **Runtime:** the current transcript is embedded;
nearest-neighbour search returns candidates; the LLM receives candidates as evidence and reasons over
them instead of searching Scripture itself. Multiple translations (KJV/NKJV/NIV/ESV/AMP/GNB…) are
recommended; embeddings generated once; retrieval stays in the millisecond range on ordinary desktop
hardware. **Retrieval proposes; the LLM evaluates** over transcript, previous understanding, summary,
intent, supporting activities, retrieval and detection candidates.

## 13. How Detection and Comprehension Are Stored — Separately

A **detection** is a point-in-time event (one verse at a timestamp); **comprehension** is a
duration/state (a story spanning start→end, with dominant intent, supporting activities, and an open
passage list). Storing separately keeps the data honest and avoids snapshotting an ongoing state into
an instant. They are **joined only at display time**: each detection looks up whichever state was
active at its timestamp. **Recent Detection** shows timestamp, passage, confidence, methods,
projection status, active state. **Comprehension Timeline** shows start/end, dominant intent,
supporting activities, grounded passages, confidence, optional topic, `NO_CHANGE`/`STATE_CHANGED`.

## 14. Three Independent Memory Layers

**A. Product Learning** — developers analyse disagreement logs, improve algorithms, ship versions;
every installation benefits. **B. Local Adaptation** — per-installation; learns recurring local
patterns (pastor wording, recurring expressions, confirmed passages, operator corrections); never
retrains the LLM — contributes confidence evidence only. **C. Current Service Memory** — the
observer's rolling understanding for the current service (working memory, not learning).

**Local adaptation principles:** global improvement and local adaptation are independent; local
adaptation never modifies models — it stores lightweight statistics that become additional evidence.
**Adapt slowly:** accumulate consistent evidence before influencing confidence; never adapt after one
isolated event (protects against transcription mistakes, guest speakers, unusual sermons, mic
problems).

## 15. Techniques for Keeping It Low-Latency

**Quantization** (e.g. 4-bit) shrinks footprint and speeds inference with minimal accuracy loss for a
classification task. **Constrained/structured output** (§8) reduces generation time. **Short,
consistent prompts** — same fixed instructions, only current understanding + new transcript change.
**The `NO_CHANGE` decision** — the single most important latency/stability lever. **A configurable
refresh interval** (§5.2) — the most direct responsiveness-vs-CPU lever, without touching detection.

**Interval philosophy:** conservative default (~60s) prioritises CPU-only compatibility; stronger
hardware reduces it. Projection responsiveness is unaffected (driven by Detection). **Long-running
services:** running local models 4+ hours is fine — models don't lose accuracy from uptime; the real
concerns are CPU/GPU utilisation, temperature, scheduling, responsiveness — which the configurable
interval manages.

## 16. Model-Agnostic Architecture — The Adapter Pattern

The comprehension layer must **never be tightly coupled to a specific LLM**. All model communication
goes through a common **Adapter Pattern** (Dependency Inversion: the Comprehension Engine depends on
an "LLM Adapter" interface, never a concrete model). **Two-layer split:** the **Comprehension Engine
(observer)** owns the intelligence (prompts, sliding buffer, previous understanding, interval, state)
and never talks to a model directly; the **Adapter Layer** owns everything model-specific (loading,
inference, prompt/response formats, memory, backend — llama.cpp/Ollama/vLLM/MLX/HF). Every adapter
exposes the same operations (initialize, run inference, check health, report capabilities, shut
down). Provider/model live in **configuration** — switching models is a config change, not a code
change. **Capability discovery** lets each adapter report structured-output support, context size,
streaming, memory — which also drives how much memory the engine injects (§17). **Structured output
stays a hard requirement**, enforced at the adapter boundary regardless of model. **User-facing model
selection** (pick provider/model, load, see status/memory/CPU-vs-GPU) is exposed; ideally no restart.

**Staged hardware strategy:** Phase 1 (current i5, CPU-only) small efficient local model (Qwen3 4B
class via llama.cpp) with a longer interval; Phase 2 (better CPU/RAM) larger models + shorter
interval; Phase 3 (GPU) larger models still — all config swaps, no architectural change. Qwen3
(1.7B–4B) is today's default, not a hardcoded dependency.

## 17. Managing Memory Over Long Services (4+ Hours) Without Hallucination

The rolling transcript buffer (§3) does **not** grow. The real growth risk is the **"current
understanding" memory** carried forward — the growing passage list and supporting-activity history of
a long-running state. Borrowing from Claude Code's long-session approach: **Microcompaction** (a hot
tail stays visible; older material becomes referenced cold storage), **Auto-compaction** (summarize
history into a compact working state near the limit), **Delta summarization** (give the model the
previous summary plus only what's new; produce a short incremental update).

**Applied to Rhema:** **Hot tail** = the rolling transcript buffer. **Per-state micro-record** = each
resolved state written as a small structured `context_states` entry (passage list + supporting
activities). **Rolling delta summary ("sermon arc")** = periodically condense micro-records into one
short incremental update (passage lists compress: "covered Luke 15 and Luke 19 during the story");
this rolling summary — not the full history — is fed back as "current understanding," keeping size
roughly constant regardless of service length. **Cold storage** = the full timestamped `detections`
and `context_states` logs remain available for review but are never re-fed wholesale; recall is a
lookup, not something the model holds.

**Why this also protects against hallucination:** small structured deltas give less surface to invent
than a growing transcript wall; `NO_CHANGE` (§5.4) and optional/nullable tags (§8) prevent fabricated
states/passages/topics; and the deterministic detection pipeline stays completely separate — if
comprehension hallucinates, the worst case is a mislabeled context tag; the verse detections remain
accurate because they never depended on the LLM. **That separation is the real hallucination
firewall.** Since each adapter reports `max_context` (§16), the rolling-summary size scales to the
active model.

## 18. Where the Model Runs

Whichever model is active, inference runs **locally**, inside the same desktop app — never a cloud
dependency. The selected adapter loads its model once at startup (the same pattern as Whisper STT and
the embedding model) and keeps it resident, called by the observer when the interval triggers.
Everything — transcription, detection, comprehension — stays self-contained, no internet dependency,
no per-use cost, regardless of the configured model.

## 19. Data Storage & Retention — Session-Only vs. Persistent

A settings toggle lets each user decide. **One storage layer, two backends:** **Session-only
("RAM-like")** — the database opens purely in memory; nothing touches disk; on close the OS releases
it, no cleanup. **Persistent** — the same structure opens as a file on disk; data survives restarts
("look back on a past service"). One storage module, one startup switch. The setting lives in a
small, always-persisted **preferences file** separate from session data. **Even persistent mode
shouldn't grow forever:** keep-for-N-days/services (log-rotation spirit), manual clear/export, soft
size ceiling. Three tiers: **off**, **persistent with retention**, **persistent unlimited** — all
config over the same module. Mental model: a browser's private/incognito mode. If persistent,
**encrypting the DB at rest** is worth considering for sermon content/timing sensitivity.

## 20. Future Developer Feature (Not Yet Implemented): Evidence Inspector

Architecturally prepare an optional Evidence Inspector — but **do not implement yet**. Could surface:
detection candidates, retrieval candidates, comprehension output, local-adaptation contribution,
confidence-fusion breakdown, final projection decision. For debugging/testing/tuning.

## 21. Open Questions Developers Should Explicitly Answer

How to behave when ASR errors reduce detection confidence but comprehension has strong context; under
what conditions comprehension may influence detection confidence; the fusion weighting strategy; how
conflicting evidence is handled; when local adaptation begins influencing confidence; how it is
reset/disabled; how disagreement logs are stored/analysed; which sources are authoritative /
corroborative / advisory. *(Proposed answers: §8 of the main design doc above.)*

## 22. Summary of the Full Flow

1. Live transcript streams into a rolling buffer. 2. **Detection** (fast path) matches verses in real
time, independently — feeds Auto Mode. 3. **The observer** (a Rhema component, not the LLM) waits for
the configured interval, then packages current understanding + newly-updated transcript and invokes
the LLM through the **Adapter Layer**, optionally enriched with **Bible retrieval candidates** (§12).
4. The LLM confirms or revises: `NO_CHANGE`, or `STATE_CHANGED` with a fresh dominant intent,
supporting activities, an independently-grounded open-ended passage list (possibly empty), optional
topic, confidence. 5. `NO_CHANGE` → continue, no record; `STATE_CHANGED` → record and reset the
passage list. 6. Comprehension feeds **evidence fusion** (§11) alongside direct/semantic/quotation
detection, retrieval, current state, and local adaptation — adjusting confidence, not replacing it;
**Detection remains the sole projection authority**. 7. Detection and comprehension are logged
separately, joined only at display; older micro-records condense into a rolling "sermon arc" so memory
stays bounded. 8. Auto Mode projects the specific passage detection matched (state = supporting
context); Manual Mode surfaces the full state so the user can choose among related passages. 9. **Local
adaptation** (§14) accumulates slow per-install statistics that feed back as evidence — separate from
**product learning**. 10. All data is stored per the user's retention setting. 11. Both the configured
model (adapter pattern) and the interval can be tuned as hardware improves — the rest is untouched.

---

# Appendix B — 2026 Technology Research (with sources)

## B.1 Local LLM inference for a Rust/Tauri desktop app (CPU-only i5-8265U)

**Task profile drives the choice:** one call / 10–60 s, short structured-JSON output,
latency-tolerant, must bundle the runtime, no cloud. This is the *easy* end of local inference — no
server-grade engine needed; optimize for binary simplicity, robust JSON, and RAM fit.

- **Backend — `llama-cpp-2`** (utilityai/llama-cpp-rs), embedded in-process. Tracks upstream
  llama.cpp, exposes the native grammar sampler (constrained JSON free). Needs `clang` at build time;
  `unsafe` under the hood (fine for a controlled, low-frequency call site). Preferred over a
  llama-server sidecar (more moving parts: Tauri `externalBin`, WebView network policy, lifecycle) and
  over **Ollama** (great for prototyping, too heavy to bundle/pin). **mistral.rs** = viable pure-Rust
  plan B (no clang) but trails llama.cpp on CPU; **candle** = too low-level; `rustformers/llm` =
  unmaintained.
- **Model — Qwen3-4B-Instruct @ Q4_K_M** primary (best accuracy-per-token in class, native JSON/tool
  behavior, non-thinking mode for terse classification); **Gemma 3 4B** most RAM-efficient alt;
  **Phi-4-mini** ~12 tok/s CPU; **Qwen3-1.7B** fast fallback. Est. **~6–10 tok/s** for 4B-Q4 on the
  8265U (memory-bandwidth-bound) → a ~100-token reply worst-cases ~12 s. **Benchmark with
  `llama-bench` on the real target before locking the default.**
- **Structured output — GBNF** (llama.cpp native, JSON-schema→grammar) is lowest-friction and
  in-process; **llguidance** (~50 µs/token) is a compile-flag upgrade for complex schemas; **XGrammar**
  is server-oriented; **Outlines** is Python. Constrain the intent field to the enum and passage to a
  `Book Chapter:Verse` pattern.
- **Quant — Q4_K_M** default (~1–3.5% quality loss, ~2.5 GB for 4B); **Q5_K_M** if labels wobble;
  **IQ4_XS** only if RAM-tight (needs a good imatrix); **Q8** overkill. Classification tasks are
  quant-resilient.

Sources: utilityai/llama-cpp-rs (github, crates.io, docs.rs) · Tauri sidecar docs (v2.tauri.app) ·
EricLBuehler/mistral.rs (PERFORMANCE.md) · guidance-ai/llguidance · JSONSchemaBench (arXiv
2501.10868) · llama.cpp grammars (ggml-org) · Qwen3 report (arXiv 2505.09388) · popularai "best
CPU-only local LLM 2026" · bmdpat "GGUF quantization … 2026" · myaihardware llama.cpp benchmarks.

## B.2 Adapter / provider abstraction (the trait to copy)

- **`rig`** (0xPlaygrounds/rig, ~7.6k★) — reference design: `CompletionModel` / `EmbeddingModel` /
  `VectorStoreIndex` traits; providers implement the trait, everything above is generic. Already has
  `rig-llama-cpp` + Ollama/LM Studio backends (streaming, tools, structured output). Adopt directly or
  copy the decomposition.
- **`genai`** (jeremychone/rust-genai) — leaner client; uses each provider's **native protocol** when
  available, falls back to OpenAI-compat — the clean way to model **capability discovery** (normalize
  a request, let adapters advertise features).
- **`llm-connector`** — newer, explicit "Protocol vs Provider separation."

**Shape to copy:** a `CompletionModel`-style trait + a `Capabilities` struct (`supports_tools`,
`supports_grammar`, `supports_streaming`, `max_context`, `native_json_schema`), normalized
request/response, per-backend adapters (native protocol or OpenAI-compat shim). **Capability discovery
at construction, not per call.** (Rhema's existing Stage-2 layer is a validation/refactor target, not
greenfield — compare against `rig`'s `CompletionModel` before adding a dependency.)

## B.3 Streaming / incremental comprehension (avoid re-reasoning)

The 2025–2026 literature converged on **anchored/delta summarization** over re-summarizing each tick:
- **Dynamic sliding window driven by topic shift, not fixed increments** ("agenda-aware real-time
  meeting summarization") → re-run on topic shift, not only a timer. (Maps onto **D5**.)
- **Anchored/incremental summarization** — update the state object with new tokens rather than
  rewriting (LangMem running-summary; RGMem; ContextWeaver).
- **Explicit (Σ, Δ) state** (RGMem) — keep a stable "what I know" Σ; the LLM emits only the **delta**
  Δ each step; commit only if non-empty. Cleanest formalization of "has my understanding changed?"
  (Maps onto **D4**.)
- **Write-side adjudication / staleness** (CUPMEM, STALE) — treat each new chunk as a candidate update
  (keep/revise/block); avoids state thrash.
- **Sliding-window "label only the last utterance"** — full window as context, classify only the
  newest utterance — cheapest per-tick discourse tagging.

**Concrete design (adopted):** persistent `{intent, running_summary, active_topic, open_threads,
passages}`; each tick feed compact state + only the new transcript delta → ask for a delta (mode
change? topic shift? new claim to ground?) constrained to JSON; commit only non-empty; re-ground in
Scripture only when a new claim/quote appears.

Sources: Springer "Dynamic agenda-aware real-time meeting summarization" · RGMem / ContextWeaver
(arXiv 2510.16392) · LangMem summarization docs · CUPMEM/STALE (arXiv 2605.06527) · conversation
threads via LLMs (arXiv 2510.22844).

## B.4 Local vector search / RAG (§12 upgrade, later)

At multi-translation scale (100k–1M vectors), keep brute-force cosine as an oracle but move the index
to ANN:
- **usearch** — single-header C++ HNSW, first-class Rust bindings, SIMD, mmap on-disk, concurrent
  readers; sub-ms at this scale. Top pick if you accept C++ FFI (you already ship ONNX).
- **hannoy** — pure-Rust, LMDB-backed DiskANN-inspired HNSW; now Meilisearch's default (replaced
  arroy); best if you want no C++ FFI.
- **arroy** — legacy (behind hannoy). **sqlite-vec** — same-DB convenience but pre-1.0 and brute-force
  by default. **LanceDB** — embedded IVF-PQ + **BM25 hybrid** (helps paraphrase grounding), heavier.
  **qdrant** — a service, wrong shape. **sqlite-vss** — unmaintained, avoid.
- **Embeddings — fastembed-rs** (dense/sparse + **cross-encoder reranking**) materially improves verse
  disambiguation.

## B.5 Sermon / Scripture open source to learn from

- **openbibleinfo/Bible-Passage-Reference-Parser** — robust BCV parser + **4.7M reference strings**;
  use as a deterministic pre-LLM fast-path detector and a labeled eval corpus.
- **OpenLP** — reference for Bible import formats and reference-vs-phrase search UX.
- **seven1m/open-bibles** — libre translations in OSIS/Zefania/USFX — source-of-truth corpus.
- **openscriptures** + **biblenerd/awesome-bible-developer-resources** — linked-data models + index.
- Commercial peers (Pewbeam / Loghema / SmartVerses / Kairos) advertise sub-2s detection across
  quotations *and loose paraphrases* — paraphrase grounding (hybrid lexical+semantic) is the
  differentiator our comprehension layer targets.

## B.6 Kalosm / Floneum

Capable one-stack pure-Rust (local LLM + embeddings + audio + structured generation) but **not yet the
safe foundation for a shipping core in 2026** (next-gen "Fusor" backend explicitly pre-production;
inherits candle's CPU perf). **Recommendation:** keep inference on llama.cpp behind the adapter trait,
embeddings on fastembed/ONNX — but **borrow Kalosm's structured-generation technique** (grammar/regex
DFA-constrained decoding), which we get natively via llama.cpp GBNF anyway.
