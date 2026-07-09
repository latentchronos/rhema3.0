//! The model-agnostic adapter boundary (§16). The engine depends only on
//! [`ComprehensionModel`]; concrete backends (local llama.cpp, an optional
//! cloud one) live outside this crate. [`MockModel`] is the test double that
//! makes the whole engine testable with no network and no model.

use std::collections::VecDeque;
use std::sync::Mutex;

use thiserror::Error;

use crate::schema::{Decision, OutputSchema};

/// What a backend reports about itself, read once at construction (§16, D6).
/// `max_context_tokens` drives how much rolling memory the engine injects (§17).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    pub name: String,
    pub max_context_tokens: usize,
    pub supports_structured_output: bool,
    pub supports_streaming: bool,
}

/// Backend health (§16: check health).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelHealth {
    Ready,
    Loading,
    Error(String),
}

/// Errors a backend may return from inference.
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("model inference failed: {0}")]
    Inference(String),
    #[error("model not ready")]
    NotReady,
}

/// A comprehension backend (§16). Every adapter exposes the same operations;
/// the engine never knows which model is underneath. `async_fn_in_trait` is
/// allowed here exactly as the existing `LlmProvider` trait does it.
#[allow(async_fn_in_trait)]
pub trait ComprehensionModel {
    /// Capabilities, read once at construction (D6) — not per call.
    fn capabilities(&self) -> &Capabilities;
    /// Current backend health.
    async fn health(&self) -> ModelHealth;
    /// Run one observer evaluation, returning a validated [`Decision`].
    async fn infer(&self, prompt: &str, schema: &OutputSchema) -> Result<Decision, ModelError>;
}

/// Test double: replays a scripted sequence of [`Decision`]s, then yields
/// `NO_CHANGE` once exhausted. Enables full engine testing with no real model.
pub struct MockModel {
    caps: Capabilities,
    scripted: Mutex<VecDeque<Decision>>,
}

impl MockModel {
    pub fn new(scripted: Vec<Decision>) -> Self {
        let caps = Capabilities {
            name: "mock".to_string(),
            max_context_tokens: 4096,
            supports_structured_output: true,
            supports_streaming: false,
        };
        Self::with_capabilities(caps, scripted)
    }

    pub fn with_capabilities(caps: Capabilities, scripted: Vec<Decision>) -> Self {
        Self {
            caps,
            scripted: Mutex::new(scripted.into()),
        }
    }
}

impl ComprehensionModel for MockModel {
    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    async fn health(&self) -> ModelHealth {
        ModelHealth::Ready
    }

    async fn infer(&self, _prompt: &str, _schema: &OutputSchema) -> Result<Decision, ModelError> {
        let next = self.scripted.lock().unwrap().pop_front();
        Ok(next.unwrap_or(Decision::NoChange))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ComprehensionState, DominantIntent};

    fn state(intent: DominantIntent) -> ComprehensionState {
        ComprehensionState {
            intent,
            supporting_activities: vec![],
            passages: vec![],
            topic: None,
            confidence: 0.9,
        }
    }

    #[tokio::test]
    async fn mock_returns_scripted_decisions_in_order() {
        let model = MockModel::new(vec![
            Decision::StateChanged { new_state: state(DominantIntent::StoryTelling) },
            Decision::NoChange,
        ]);
        let s = OutputSchema::default();
        let first = model.infer("p", &s).await.unwrap();
        assert!(matches!(first, Decision::StateChanged { .. }));
        let second = model.infer("p", &s).await.unwrap();
        assert_eq!(second, Decision::NoChange);
    }

    #[tokio::test]
    async fn mock_defaults_to_no_change_when_exhausted() {
        let model = MockModel::new(vec![]);
        let out = model.infer("p", &OutputSchema::default()).await.unwrap();
        assert_eq!(out, Decision::NoChange);
    }

    #[tokio::test]
    async fn mock_reports_ready_health() {
        let model = MockModel::new(vec![]);
        assert_eq!(model.health().await, ModelHealth::Ready);
    }

    #[test]
    fn mock_capabilities_are_readable() {
        let model = MockModel::new(vec![]);
        let caps = model.capabilities();
        assert_eq!(caps.name, "mock");
        assert!(caps.supports_structured_output);
    }

    #[test]
    fn mock_accepts_custom_capabilities() {
        let caps = Capabilities {
            name: "tiny".to_string(),
            max_context_tokens: 2048,
            supports_structured_output: true,
            supports_streaming: false,
        };
        let model = MockModel::with_capabilities(caps.clone(), vec![]);
        assert_eq!(model.capabilities(), &caps);
    }
}
