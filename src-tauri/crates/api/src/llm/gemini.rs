//! Google Gemini Stage-2 adapter (Bullet L4).
//!
//! `POST {base}/v1beta/models/{model}:generateContent?key=…` with body
//! `{systemInstruction, contents, generationConfig}`; the reply's
//! `candidates[0].content.parts[].text` is fed to the shared
//! [`parse_stage2_json`]. Pure `build_body`/`extract_text`; only
//! [`GeminiProvider::classify`] does I/O.

use serde_json::Value;

use super::{
    build_http_client, parse_stage2_json, LlmConfig, LlmError, LlmProvider, Stage2Request,
    Stage2Result, STAGE2_MAX_TOKENS, STAGE2_SYSTEM_PROMPT,
};

pub struct GeminiProvider {
    client: reqwest::Client,
    config: LlmConfig,
}

impl GeminiProvider {
    pub fn new(config: LlmConfig) -> Result<Self, LlmError> {
        Ok(Self {
            client: build_http_client()?,
            config,
        })
    }
}

/// Build the JSON request body for `generateContent`.
fn build_body(req: &Stage2Request) -> Value {
    serde_json::json!({
        "systemInstruction": { "parts": [{ "text": STAGE2_SYSTEM_PROMPT }] },
        "contents": [{ "role": "user", "parts": [{ "text": req.user_message() }] }],
        "generationConfig": { "maxOutputTokens": STAGE2_MAX_TOKENS, "temperature": 0 },
    })
}

/// Extract and concatenate the text parts from the first candidate.
fn extract_text(json: &Value) -> Result<String, LlmError> {
    json.get("candidates")
        .and_then(Value::as_array)
        .and_then(|cands| cands.first())
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<String>()
        })
        .filter(|s| !s.is_empty())
        .ok_or_else(|| LlmError::Parse("no candidates[0].content.parts text in response".to_string()))
}

impl LlmProvider for GeminiProvider {
    async fn classify(&self, req: &Stage2Request) -> Result<Stage2Result, LlmError> {
        if !self.config.is_usable() {
            return Err(LlmError::NoApiKey);
        }
        let url = format!(
            "{}/v1beta/models/{}:generateContent",
            self.config.base_url(),
            self.config.model()
        );
        let resp = self
            .client
            .post(&url)
            .query(&[("key", self.config.api_key.trim())])
            .json(&build_body(req))
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = resp.status();
        let body = resp.text().await.map_err(|e| LlmError::Http(e.to_string()))?;
        if !status.is_success() {
            return Err(LlmError::Status {
                status: status.as_u16(),
                body,
            });
        }
        let json: Value =
            serde_json::from_str(&body).map_err(|e| LlmError::Parse(e.to_string()))?;
        let text = extract_text(&json)?;
        parse_stage2_json(&text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_has_system_instruction_and_contents() {
        let req = Stage2Request {
            transcript: "blessed are the meek".to_string(),
            context: None,
        };
        let body = build_body(&req);
        assert!(body["systemInstruction"]["parts"][0]["text"]
            .as_str()
            .unwrap()
            .contains("scripture"));
        assert_eq!(body["contents"][0]["role"], "user");
        assert!(body["contents"][0]["parts"][0]["text"]
            .as_str()
            .unwrap()
            .contains("blessed are the meek"));
        assert_eq!(body["generationConfig"]["maxOutputTokens"], STAGE2_MAX_TOKENS);
    }

    #[test]
    fn extracts_text_from_candidates() {
        let resp = serde_json::json!({
            "candidates": [{
                "content": { "parts": [{ "text": "{\"is_scripture\": true, \"reference\": \"Matthew 5:5\", \"confidence\": 0.88}" }] }
            }],
        });
        let text = extract_text(&resp).unwrap();
        let parsed = parse_stage2_json(&text).unwrap();
        assert!(parsed.is_scripture);
        assert_eq!(parsed.reference.as_deref(), Some("Matthew 5:5"));
    }

    #[test]
    fn extract_text_errors_when_missing() {
        assert!(extract_text(&serde_json::json!({"candidates": []})).is_err());
        assert!(extract_text(&serde_json::json!({"promptFeedback": {}})).is_err());
    }
}
