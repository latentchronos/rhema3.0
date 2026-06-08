# Production Readiness Roadmap

This roadmap tracks the planned improvements that move Rhema from a promising live sermon detection app toward a production-grade church broadcast platform.

## Phase 1 - Progress Tracking And Roadmap Foundation

Status: completed locally.

Purpose:

- Add a local-only progress convention for developers and AI agents.
- Keep continuation state out of GitHub.
- Create a public roadmap for the major workstreams.

Primary files:

- `.agent-progress/progress.md`
- `docs/local-agent-progress.md`
- `docs/production-readiness-roadmap.md`

## Phase 2 - Architecture Documentation

Status: completed.

Purpose:

- Document the current system end to end.
- Explain how audio, STT, detection, Bible lookup, queueing, and broadcast output connect.
- Give contributors a safe map before deeper changes begin.

Expected outputs:

- `docs/architecture.md`
- module map for Rust crates
- frontend state-flow overview
- event and command flow overview

## Phase 3 - Real HNSW Semantic Search With Exact Reranking

Status: completed.

Purpose:

- Replace the current brute-force vector search hidden behind `HnswVectorIndex`.
- Use real approximate nearest-neighbor lookup for faster semantic search.
- Rerank the final candidate set with exact cosine similarity for better accuracy.

Expected outputs:

- real HNSW-backed vector index
- flat vector index retained for tests/debugging if useful
- benchmark or regression checks for recall and latency
- `docs/detection-vector-search.md`

## Phase 4 - Explainable, Threshold-Driven Detection Confidence

Status: completed.

Purpose:

- Make detection confidence operationally useful for live operators.
- Explain why a verse matched.
- Add source-specific thresholds and safer auto-queue behavior.

Expected outputs:

- richer detection metadata
- threshold settings per source
- clearer detection panel and queue indicators
- `docs/detection-confidence.md`

## Phase 5 - OBS Browser/WebSocket Output

Status: completed.

Purpose:

- Add a broadcast path for churches that use OBS instead of NDI.
- Keep NDI while making output targets pluggable.

Expected outputs:

- broadcast output abstraction
- OBS browser source support
- optional OBS WebSocket connection controls
- `docs/broadcast-outputs.md`

## Phase 6 - Sermon Regression Corpus

Status: completed.

Purpose:

- Measure detection quality with repeatable fixtures instead of intuition.
- Track precision, recall, false positives, and latency.

Expected outputs:

- transcript fixture format
- expected detection format
- Rust or script-based evaluation runner
- small committed sample corpus
- `docs/sermon-regression-corpus.md`

## Phase 7 - Improved VAD And Multi-Channel Audio

Status: completed.

Purpose:

- Improve speech segmentation quality.
- Support production setups with multiple microphones or mixer channels.

Expected outputs:

- stronger VAD option or better-tuned existing VAD
- channel-aware capture configuration
- UI support for source/channel selection
- `docs/audio-pipeline.md`

## Phase 8 - ProPresenter Integration, Tauri Security, And Contributor Docs Polish

Status: completed for docs/security hardening. Live ProPresenter API push remains a follow-up behind the documented adapter contract.

Purpose:

- Add another common church presentation output.
- Harden Tauri permissions and content security.
- Finish contributor-facing architecture docs.

Expected outputs:

- ProPresenter output adapter
- stricter Tauri CSP and capabilities review
- `docs/propresenter-integration.md`
- `docs/security.md`
- contributor docs updates
- `docs/contributor-architecture.md`
