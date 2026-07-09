use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::types::ComprehensionState;

/// The two-outcome comprehension decision (§5.4). Internally tagged by the
/// `decision` key to match the §8 JSON contract exactly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "decision")]
pub enum Decision {
    /// The existing understanding still holds — no new record (§5.4).
    #[serde(rename = "NO_CHANGE")]
    NoChange,
    /// The understanding has genuinely shifted — record a transition (§5.4, §7).
    #[serde(rename = "STATE_CHANGED")]
    StateChanged { new_state: ComprehensionState },
}

/// Describes the structured output the observer requests. For the model-free
/// engine this only carries the system instruction; the local backend (Phase B)
/// will additionally derive a GBNF grammar from it (§16, D7).
#[derive(Debug, Clone, Copy, Default)]
pub struct OutputSchema;

impl OutputSchema {
    pub fn instruction(&self) -> &'static str {
        SCHEMA_INSTRUCTION
    }
}

/// Fixed system instruction (§15: short, consistent prompts). Only the current
/// understanding and the new transcript change between calls (§5.3).
pub const SCHEMA_INSTRUCTION: &str = "You are a sermon comprehension observer. \
Given the current understanding and newly added transcript, decide whether your understanding has changed. \
Respond with ONLY a JSON object. Either {\"decision\":\"NO_CHANGE\"}, or \
{\"decision\":\"STATE_CHANGED\",\"new_state\":{\"state\":<INTENT>,\"supporting_activities\":[...],\"passages\":[...],\"topic\":<string or null>,\"confidence\":<0..1>}}. \
INTENT is one of IDLE, TEACHING, STORY_TELLING, APPLYING, PRAYING, EXHORTING, ANNOUNCEMENTS. \
Ground in scripture independently: passages may be empty. Never invent a topic. Output no text other than the JSON object.";

/// Errors from parsing a model reply into a [`Decision`].
#[derive(Debug, Error)]
pub enum SchemaError {
    #[error("no JSON object found in model reply")]
    NoJson,
    #[error("failed to parse decision JSON: {0}")]
    Parse(String),
}

/// Parse a model's text reply into a [`Decision`]. Tolerant of surrounding prose
/// or ```code fences``` by extracting the outermost `{ … }` span (mirrors the
/// existing `parse_stage2_json` in `rhema-api`).
pub fn parse_decision(text: &str) -> Result<Decision, SchemaError> {
    let json = extract_json_object(text).ok_or(SchemaError::NoJson)?;
    serde_json::from_str::<Decision>(json).map_err(|e| SchemaError::Parse(e.to_string()))
}

/// The substring from the first `{` to the last `}` (inclusive), if any.
fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end > start {
        Some(&text[start..=end])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DominantIntent, PassageRef, SupportingActivity};

    #[test]
    fn no_change_serializes_to_contract() {
        let json = serde_json::to_string(&Decision::NoChange).unwrap();
        assert_eq!(json, r#"{"decision":"NO_CHANGE"}"#);
    }

    #[test]
    fn state_changed_nests_under_new_state() {
        let d = Decision::StateChanged {
            new_state: ComprehensionState {
                intent: DominantIntent::Teaching,
                supporting_activities: vec![SupportingActivity::QuotingScripture],
                passages: vec![PassageRef::new("John 3:16")],
                topic: None,
                confidence: 0.87,
            },
        };
        let json = serde_json::to_string(&d).unwrap();
        assert!(json.contains(r#""decision":"STATE_CHANGED""#), "got: {json}");
        assert!(json.contains(r#""new_state":{"#), "got: {json}");
        assert!(json.contains(r#""state":"TEACHING""#), "got: {json}");
    }

    #[test]
    fn parses_clean_no_change() {
        let d = parse_decision(r#"{"decision":"NO_CHANGE"}"#).unwrap();
        assert_eq!(d, Decision::NoChange);
    }

    #[test]
    fn parses_state_changed_example_from_spec() {
        let text = r#"{
            "decision": "STATE_CHANGED",
            "new_state": {
                "state": "TEACHING",
                "supporting_activities": ["QUOTING_SCRIPTURE"],
                "passages": ["John 3:16"],
                "topic": null,
                "confidence": 0.87
            }
        }"#;
        let d = parse_decision(text).unwrap();
        match d {
            Decision::StateChanged { new_state } => {
                assert_eq!(new_state.intent, DominantIntent::Teaching);
                assert_eq!(new_state.passages, vec![PassageRef::new("John 3:16")]);
                assert!((new_state.confidence - 0.87).abs() < 1e-6);
            }
            other => panic!("expected STATE_CHANGED, got {other:?}"),
        }
    }

    #[test]
    fn tolerates_prose_and_code_fences() {
        let text = "Sure, here you go:\n```json\n{\"decision\":\"NO_CHANGE\"}\n```\n";
        assert_eq!(parse_decision(text).unwrap(), Decision::NoChange);
    }

    #[test]
    fn empty_passages_is_valid_not_an_error() {
        let text = r#"{"decision":"STATE_CHANGED","new_state":{"state":"PRAYING","supporting_activities":[],"passages":[],"confidence":0.6}}"#;
        let d = parse_decision(text).unwrap();
        match d {
            Decision::StateChanged { new_state } => assert!(new_state.passages.is_empty()),
            other => panic!("expected STATE_CHANGED, got {other:?}"),
        }
    }

    #[test]
    fn garbage_is_an_error() {
        assert!(matches!(parse_decision("no json here"), Err(SchemaError::NoJson)));
        assert!(matches!(parse_decision("{not valid json}"), Err(SchemaError::Parse(_))));
    }

    #[test]
    fn output_schema_instruction_is_nonempty() {
        assert!(!OutputSchema::default().instruction().is_empty());
    }
}
