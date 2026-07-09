//! Rhema comprehension layer — the model-free Observer engine and discourse
//! state machine. Mirrors `rhema-detection`: no DB, no model, no network.
//! All model communication goes through the [`ComprehensionModel`] adapter
//! trait; time is always injected (`now_ms: u64`) so the logic is deterministic.

pub mod types;

pub use types::{
    ComprehensionState, DominantIntent, PassageRef, StateTransition, SupportingActivity,
};
