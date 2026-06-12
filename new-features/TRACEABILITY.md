# Rhema 3.0 — Traceability Map (Spec ⇄ Roadmap)

This file is the **bridge** between the vision spec ([`ARCHITECTURE.md`](ARCHITECTURE.md))
and the buildable roadmap ([`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) + the
`markdowns/PHASE*_COMPANION.md` files). It exists so that nothing in the vision is *silently*
dropped: every stage is marked **Implemented-in-roadmap**, **Simplified**, or **Deferred**,
with the reason.

**Status legend**

| Status | Meaning |
|---|---|
| ✅ In roadmap | A Phase bullet implements this directly |
| 🟡 Simplified | Shipped in a reduced form; the full spec version is a later-Gen upgrade |
| ⏳ Deferred | Described in the spec, **not** in the 5-phase roadmap; future Gen |

---

## Stage → Phase coverage

| Spec (ARCHITECTURE.md) | Capability | Status | Phase / bullet | Notes |
|---|---|---|---|---|
| §2.4 | Spectral flux + energy-variance gate (stage bleed) | ✅ | P1 · 1.1, 1.2 | `rhema-audio` |
| §2.4 | RMS adaptive normalizer (compression trap) | ✅ | P1 · 1.3 | replaces static gain |
| §2.4 | PA feedback detector | ✅ | P1 · 1.4 | band **500 Hz–8 kHz** (reconciled) |
| §3.4 / §7.2 | Phonetic normalisation (homophones) | ✅ | P2 · 2.1 | canonical map = §7.2 (merged list) |
| §2.4 / §4.5 | Levenshtein token-skip (Amen splice) | ✅ | P2 · 2.2 | 6-token window |
| §4.3 | Two-stage classifier (Stage-1 local) | 🟡 | P2 · 2.3 | **pattern classifier, not ML** — ML is Gen-3 |
| §4.3 | Stage-2 LLM fallback channel | ✅ | P2 · 2.3 → P5 · 5.4 | channel placeholder in P2, real Claude call in P5 |
| §2.4 / §6.10 | 45 s semantic suppression cache (echo loop) | ✅ | P2 · 2.4 | app-level consumer |
| §6.1 | Formal CursorState | ✅ | P3 · 3.1 | `active_output` **removed** (reconciled → channel layer) |
| §5.3 | Epoch lock (operator beats voice) | ✅ | P3 · 3.2 | 500 ms window, `SeqCst` |
| §4.5 | Negation-marker filter (mid-sentence pivot) | ✅ | P3 · 3.3 | |
| §6.4–6.7 | Coordinate bounds (cross-chapter, Bible edges, omissions) | ✅ | P3 · 3.4 | via `BibleDb`, never hardcoded |
| §6.9 | Asymmetric range collapse | ✅ | P3 · 3.5 | |
| §11.2 | Three logical channels (Audience/Pastor/Operator) | ✅ | P4 · 4.1 | Audience excludes suggestions/queue/confidence |
| §11.4 / §11.6 | Pub-sub routing + `RoutingMode` | ✅ | P4 · 4.2 | `emit` vs `emit_to` |
| §11 | Zustand additive channel/routing/health fields | ✅ | P4 · 4.3 | existing fields retained |
| §11.9 | Device health monitor (2 s ping, resync) | ✅ | P4 · 4.4 | |
| §9.3 | Time-decay topic vector (SermonContext) | ✅ | P5 · 5.1 | 75/25, 90 s half-life, per-sentence ingest |
| §7.3 / §9 | Pre-service priming index (+0.25 boost) | ✅ | P5 · 5.2 | `set_sermon_notes` Tauri command |
| §9.4 | Proactive suggestion engine | ✅ | P5 · 5.3 | 300 s cooldown, 0.72 threshold, Operator channel only |
| §4.3 / §12.2 | Wire Stage-2 to real Claude | ✅ | P5 · 5.4 | iff `rhema-api` implemented (needs `reqwest` approval) |

---

## Deferred / simplified — explicitly out of the 5-phase roadmap

These are real spec items that the roadmap does **not** build. Listed so they are decisions,
not accidents. Each maps to a future Generation (§0.2).

| Spec | Capability | Status | Target | Why deferred |
|---|---|---|---|---|
| §4.2–4.3 | **Nigerian English / Pidgin** classification | ⏳ | Gen 3 | Lives in the ML Stage-1 classifier; the v1 pattern classifier doesn't cover broad pidgin. **Highest-value deferral to revisit** given the vision emphasis. |
| §4.3 | **ML intent classifier** (distilled BERT/MobileBERT) | ⏳ | Gen 3 | v1 ships deterministic patterns behind the same interface; ML needs a labelled training set. |
| §5.2 | **Compound command AST** (sequential/conditional) | ⏳ | Gen 3 | Phase 3 ships only the negation filter, not the full AST executor. |
| §4.5 | Filler-word strip, interrogative forms, ordinal adjacency, book disambiguation | ⏳ | Gen 3 | Intent-layer polish beyond negation. |
| §6.12 | **Relational delta mapping** ("the verse above this") | ⏳ | Gen 3 | Folded in from the EdgeCases doc; not in any phase. |
| §7.3 | Contextual search Categories A–H (narrative, deictic, cross-testament, exclusion, ranked) | ⏳ | Gen 4 | Phase 5 ships topic-vector + threshold suggestion only. |
| §7.3 | Top-3 active-theme intersection for broad queries | ⏳ | Gen 4 | Folded in from EdgeCases doc. |
| §8 | **Translation engine** (peek, compare, dual-language, live registry) | ⏳ | Gen 3/4 | No phase implements it; current build switches the 4 DB translations via `BibleDb`. |
| §9.2 | Content importance scoring (project threshold 0.80) | ⏳ | Gen 4 | Only the suggestion sliver of content-understanding is in P5. |
| §9.1 | Full content categorisation (teaching point / quote / prayer / announcement / title) | ⏳ | Gen 4/5 | |
| §10.1–10.3 | **Manual / Auto mode** + 0.92/0.85 thresholds + accuracy unlock | ⏳ | Gen 4 | No mode switching in the roadmap; build is effectively Manual. |
| §10.5 | **Slide / presentation generation** | ⏳ | Gen 6 | |
| §3.4 / §10.4 | Prayer mode toggle, force-project, session pause overrides | ⏳ | Gen 3 | Session-start gating + epoch lock are in; the rest of the override panel is not. |
| §11.11 (OS-4/6) | OBS+NDI as channel subscribers; web/mobile/remote outputs | ⏳ | post-P4 | OBS/NDI exist as outputs today; full channel-subscriber model + remote is later. |

---

## Open integration thread

- **The "app-level detection consumer" file is unnamed.** Phase 2 (suppression cache) and
  Phase 5 (suggestion engine) both attach to "the app detection consumer loop
  (`detection_consumer.rs` *or wherever it lives*)". **Phase 0 recon must pin down the exact
  file** before Phase 2 starts — it is the shared anchor for both phases. Until then it is the
  one genuinely loose joint between phases.

---

## How to keep this in sync going forward

1. When a Phase bullet ships, flip its row to ✅ and (if it implemented a deferred item) move
   the row up from the deferred table.
2. When the spec changes, add/adjust the matching row here in the same commit.
3. `ARCHITECTURE.md` = *what & why*. The phase companions = *how & when*. **This file is the
   only place the two are reconciled** — if they disagree, fix it here and in both sources.
