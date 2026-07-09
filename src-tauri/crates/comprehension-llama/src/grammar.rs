//! Grammar constraint for the §8 `Decision` contract (D7). Rather than hand-roll
//! GBNF (which is brittle against a model's tokenizer — merged tokens overshoot
//! rule boundaries and abort llama.cpp's grammar engine), we hand llama.cpp its
//! own JSON-Schema→GBNF converter (`json_schema_to_grammar`) a schema of
//! `Decision`. The generated grammar uses the whitespace handling llama.cpp's
//! engine expects, and still constrains `decision`/`state` to their enums so the
//! reply is guaranteed to be a valid decision.

/// JSON Schema for [`rhema_comprehension::Decision`]. `json_schema_to_grammar`
/// turns this into the GBNF attached to the sampler. Intent and supporting
/// activities are enum-constrained; passages/topic are free (grounding accuracy
/// is handled by retrieval/fusion, not the grammar).
pub const DECISION_SCHEMA: &str = r#"{
  "oneOf": [
    {
      "type": "object",
      "properties": { "decision": { "enum": ["NO_CHANGE"] } },
      "required": ["decision"]
    },
    {
      "type": "object",
      "properties": {
        "decision": { "enum": ["STATE_CHANGED"] },
        "new_state": {
          "type": "object",
          "properties": {
            "state": { "enum": ["IDLE","TEACHING","STORY_TELLING","APPLYING","PRAYING","EXHORTING","ANNOUNCEMENTS"] },
            "supporting_activities": { "type": "array", "items": { "enum": ["QUOTING_SCRIPTURE","ILLUSTRATING","EXHORTING","PRAYING","TEACHING","APPLYING"] } },
            "passages": { "type": "array", "items": { "type": "string" } },
            "topic": { "type": ["string","null"] },
            "confidence": { "type": "number" }
          },
          "required": ["state","supporting_activities","passages","confidence"]
        }
      },
      "required": ["decision","new_state"]
    }
  ]
}"#;

/// Root rule name produced by `json_schema_to_grammar`.
pub const DECISION_ROOT: &str = "root";
