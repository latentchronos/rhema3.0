//! Core comprehension data model (§6–§8). Serde shapes are the §8 JSON contract.

use serde::{Deserialize, Serialize};

/// The pastor's dominant communicative intent right now (§4, §6). One conclusion,
/// not the raw evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DominantIntent {
    Idle,
    Teaching,
    StoryTelling,
    Applying,
    Praying,
    Exhorting,
    Announcements,
}

/// A secondary action happening alongside the dominant intent (§6) — logged as
/// supporting detail, never a competing top-level state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportingActivity {
    QuotingScripture,
    Illustrating,
    Exhorting,
    Praying,
    Teaching,
    Applying,
}

/// A canonical passage reference (e.g. `"Luke 15:11-32"`). Serializes as a bare
/// JSON string so a passage list is `["John 3:16"]` per the §8 contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PassageRef(pub String);

impl PassageRef {
    pub fn new(reference: impl Into<String>) -> Self {
        Self(reference.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// The observer's current conclusion about what the pastor is doing (§4, §7).
/// Holds an OPEN-ENDED passage list (zero, one, or many — §7, §9) and an
/// OPTIONAL topic (§8: never invented to fill the field).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComprehensionState {
    /// Dominant intent. JSON key is `state` per the §8 contract.
    #[serde(rename = "state")]
    pub intent: DominantIntent,
    #[serde(default)]
    pub supporting_activities: Vec<SupportingActivity>,
    #[serde(default)]
    pub passages: Vec<PassageRef>,
    /// Optional — omitted from output when `None`; parses from `null` on input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topic: Option<String>,
    pub confidence: f32,
}

/// A recorded transition between states (§7) — only created on a genuine change.
/// `at_ms` is a caller-supplied timestamp (time is injected).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateTransition {
    pub from: Option<ComprehensionState>,
    pub to: ComprehensionState,
    pub at_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intent_serializes_screaming_snake_case() {
        let json = serde_json::to_string(&DominantIntent::StoryTelling).unwrap();
        assert_eq!(json, "\"STORY_TELLING\"");
        let back: DominantIntent = serde_json::from_str("\"STORY_TELLING\"").unwrap();
        assert_eq!(back, DominantIntent::StoryTelling);
    }

    #[test]
    fn supporting_activity_serializes_screaming_snake_case() {
        let json = serde_json::to_string(&SupportingActivity::QuotingScripture).unwrap();
        assert_eq!(json, "\"QUOTING_SCRIPTURE\"");
    }

    #[test]
    fn passage_ref_is_a_bare_string() {
        let p = PassageRef::new("Luke 15:11-32");
        assert_eq!(serde_json::to_string(&p).unwrap(), "\"Luke 15:11-32\"");
        assert_eq!(p.as_str(), "Luke 15:11-32");
    }

    #[test]
    fn state_uses_state_key_and_omits_none_topic() {
        let s = ComprehensionState {
            intent: DominantIntent::Teaching,
            supporting_activities: vec![SupportingActivity::QuotingScripture],
            passages: vec![PassageRef::new("John 3:16")],
            topic: None,
            confidence: 0.87,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"state\":\"TEACHING\""), "got: {json}");
        assert!(json.contains("\"supporting_activities\":[\"QUOTING_SCRIPTURE\"]"), "got: {json}");
        assert!(json.contains("\"passages\":[\"John 3:16\"]"), "got: {json}");
        assert!(!json.contains("topic"), "None topic must be omitted, got: {json}");
    }

    #[test]
    fn state_parses_with_null_topic_and_empty_passages() {
        let s: ComprehensionState = serde_json::from_str(
            r#"{"state":"PRAYING","supporting_activities":[],"passages":[],"topic":null,"confidence":0.5}"#,
        )
        .unwrap();
        assert_eq!(s.intent, DominantIntent::Praying);
        assert!(s.passages.is_empty());
        assert_eq!(s.topic, None);
    }

    #[test]
    fn state_transition_round_trips() {
        let to = ComprehensionState {
            intent: DominantIntent::StoryTelling,
            supporting_activities: vec![],
            passages: vec![PassageRef::new("Luke 15")],
            topic: Some("prodigal son".to_string()),
            confidence: 0.94,
        };
        let t = StateTransition { from: None, to: to.clone(), at_ms: 60_000 };
        let json = serde_json::to_string(&t).unwrap();
        let back: StateTransition = serde_json::from_str(&json).unwrap();
        assert_eq!(back, t);
    }
}
