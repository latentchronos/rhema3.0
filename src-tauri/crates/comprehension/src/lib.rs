//! Rhema comprehension layer — the model-free Observer engine and discourse
//! state machine. Mirrors `rhema-detection`: no DB, no model, no network.
//! All model communication goes through the [`ComprehensionModel`] adapter
//! trait; time is always injected (`now_ms: u64`) so the logic is deterministic.

pub mod model;
pub mod observer;
pub mod schema;
pub mod summary;
pub mod types;
pub mod window;

pub use model::{Capabilities, ComprehensionModel, ModelError, ModelHealth, MockModel};
pub use observer::{Observer, ObserverConfig};
pub use schema::{parse_decision, Decision, OutputSchema, SchemaError};
pub use summary::{MicroRecord, RollingSummary};
pub use types::{
    ComprehensionState, DominantIntent, PassageRef, StateTransition, SupportingActivity,
};
pub use window::TranscriptWindow;
