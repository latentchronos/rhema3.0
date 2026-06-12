# Rhema 3.0 — Feature & Architecture Docs

This folder holds the architecture vision and the 5-phase hardening roadmap that layers onto
the existing `audio → stt → detection → broadcast` pipeline. The documents are now a single,
internally consistent set — read them in this order.

## Source of truth (read in order)

| # | File | Tier | What it is |
|---|---|---|---|
| 1 | [`ARCHITECTURE.md`](ARCHITECTURE.md) | Vision | **What & why.** The consolidated system spec (Stages 1–9, channel model, Gen 1–7 vision). Canonical. |
| 2 | [`TRACEABILITY.md`](TRACEABILITY.md) | Bridge | **Spec ⇄ roadmap map.** Every spec stage marked ✅ in-roadmap / 🟡 simplified / ⏳ deferred. The one place the two tiers are reconciled. |
| 3 | [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Roadmap | **When.** The 5-phase, bullet-by-bullet build plan with exit gates. |
| 4 | [`markdowns/GEMINI.md`](markdowns/GEMINI.md) | Contract | **Codebase ground truth + execution rules.** Crate layout, existing types, do-not-touch list, invariants. |
| 5 | `markdowns/PHASE1–5_COMPANION.md` | Companion | **How.** Per-phase domain knowledge, reasoning-block templates, and self-verification checklists. |

> **Precedence rule:** for anything being *built right now*, the phase companions + GEMINI.md
> win over `ARCHITECTURE.md`. The spec is the destination; the companions are the road.

## Phase map (one line each)

| Phase | Crate(s) | Deliverable |
|---|---|---|
| 1 | `rhema-audio` | Spectral/variance gate, RMS AGC, feedback detector |
| 2 | `rhema-stt` + `rhema-detection` + app | Phonetic normalizer, token-skip, two-stage classifier, suppression cache |
| 3 | `rhema-detection` | CursorState, epoch lock, negation filter, coordinate bounds, range collapse |
| 4 | `rhema-broadcast` + Zustand | Channel model, pub-sub routing, device health monitor |
| 5 | `rhema-detection` + app + `rhema-api` | Time-decay topic vector, priming index, suggestion engine, wire Stage-2 → Claude |

## Doc-sync history

This set was consolidated from an earlier mix of Word documents. The following were folded
into `ARCHITECTURE.md` / `TRACEABILITY.md` and then removed (content fully preserved here):

- `Rhema3_0_Architecture_Specification.docx` (v1) and `..._Specification01.docx` (v2) → `ARCHITECTURE.md`
- `Rhema_Output_Architecture_Review.docx` + `Rhema_Multi_Output_Routing_Architecture_v2.docx` → `ARCHITECTURE.md` §11
- `Rhema_Master_Architecture_EdgeCases.docx` → distributed into the relevant `ARCHITECTURE.md` edge-case tables (§6.12, §7.3 unique items)

Conflicts resolved during the sync (all marked **⟲ Reconciled** in `ARCHITECTURE.md`):
- v1 screen-bound output model dropped in favour of the v2 channel model
- `active_output` removed from the §6.1 cursor (it contradicted "devices never own navigation state")
- output rollout renamed to "Output Rollout Stages OS-1…6" to stop colliding with implementation Phases 1–5
- PA-feedback band fixed to a single value (500 Hz–8 kHz)
- one canonical phonetic-normalisation table (the two divergent homophone lists merged)
- topic-vector cadence set to per-sentence ingest + lazy compute (not the illustrative "every 30 s")
- Stage-1 classifier documented as a deliberate v1 pattern-based simplification of the ML target
