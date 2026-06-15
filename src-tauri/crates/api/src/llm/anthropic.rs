//! Anthropic (Claude) Stage-2 adapter (Bullet L2).
//!
//! `POST {base}/v1/messages` with `x-api-key` + `anthropic-version`, body
//! `{model, max_tokens, system, messages}`; the reply's `content[].text` is fed
//! to the shared [`parse_stage2_json`]. Request-shaping and response-parsing are
//! pure (and unit-tested); only [`AnthropicProvider::classify`] does I/O.

use serde_json::Value;

use super::{
    build_http_client, parse_stage2_json, LlmConfig, LlmError, LlmProvider, Stage2Request,
    Stage2Result, STAGE2_MAX_TOKENS, STAGE2_SYSTEM_PROMPT,
};

/// The Anthropic Messages API version header value.
const ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    client: reqwest::Client,
    config: LlmConfig,
}

impl AnthropicProvider {
    pub fn new(config: LlmConfig) -> Result<Self, LlmError> {
        Ok(Self {
            client: build_http_client()?,
            config,
        })
    }
}

/// Build the JSON request body for the Messages API.
fn build_body(config: &LlmConfig, req: &Stage2Request) -> Value {
    serde_json::json!({
        "model": config.model(),
        "max_tokens": STAGE2_MAX_TOKENS,
        "system": STAGE2_SYSTEM_PROMPT,
        "messages": [{ "role": "user", "content": req.user_message() }],
    })
}

/// Extract the assistant text from a Messages API response (`content[].text`).
fn extract_text(json: &Value) -> Result<String, LlmError> {
    json.get("content")
        .and_then(Value::as_array)
        .and_then(|blocks| {
            blocks
                .iter()
                .find_map(|b| b.get("text").and_then(Value::as_str))
        })
        .map(str::to_string)
        .ok_or_else(|| LlmError::Parse("no text content in Anthropic response".to_string()))
}

impl LlmProvider for AnthropicProvider {
    async fn classify(&self, req: &Stage2Request) -> Result<Stage2Result, LlmError> {
        if !self.config.is_usable() {
            return Err(LlmError::NoApiKey);
        }
        let url = format!("{}/v1/messages", self.config.base_url());
        let resp = self
            .client
            .post(&url)
            .header("x-api-key", self.config.api_key.trim())
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&build_body(&self.config, req))
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
    use crate::llm::ProviderKind;

    #[test]
    fn body_has_model_system_and_message() {
        let cfg = LlmConfig::new(ProviderKind::Anthropic, "sk-ant-x");
        let req = Stage2Request {
            transcript: "turn to John three".to_string(),
            context: None,
        };
        let body = build_body(&cfg, &req);
        assert_eq!(body["model"], "claude-haiku-4-5-20251001");
        assert_eq!(body["max_tokens"], STAGE2_MAX_TOKENS);
        assert!(body["system"].as_str().unwrap().contains("scripture"));
        assert_eq!(body["messages"][0]["role"], "user");
        assert!(body["messages"][0]["content"]
            .as_str()
            .unwrap()
            .contains("turn to John three"));
    }

    #[test]
    fn extracts_text_from_content_blocks() {
        let resp = serde_json::json!({
            "content": [{ "type": "text", "text": "{\"is_scripture\": true, \"reference\": \"John 3:16\", \"confidence\": 0.95}" }],
        });
        let text = extract_text(&resp).unwrap();
        let parsed = parse_stage2_json(&text).unwrap();
        assert!(parsed.is_scripture);
        assert_eq!(parsed.reference.as_deref(), Some("John 3:16"));
    }

    #[test]
    fn extract_text_errors_when_missing() {
        assert!(extract_text(&serde_json::json!({"content": []})).is_err());
        assert!(extract_text(&serde_json::json!({"error": "bad"})).is_err());
    }
}
