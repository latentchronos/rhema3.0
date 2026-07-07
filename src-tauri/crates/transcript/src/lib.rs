//! Transcript state management for Rhema v2 (Phase 2).
//!
//! Owns the logic that turns a streaming ASR engine's append-only `committed` prefix +
//! volatile `tentative` tail into an ordered stream of stable `Final` and disposable
//! `Partial` segments — the committed/tentative contract, extracted out of the STT engine
//! so it is engine-agnostic and unit-testable in isolation.
//!
//! This is the seed of the broader transcript state manager described in
//! `RHEMA_V2_ARCHITECTURE.md` §4. Today it covers the streaming committer (endpoint policy
//! + finalize offset tracking); confidence gating and the app-side sentence buffer remain
//! their own pieces for now.

pub mod stream;

pub use stream::{StreamCommitter, TranscriptSegment};
