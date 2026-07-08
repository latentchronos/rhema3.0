//! OpenAI-compatible Stage-2 adapter (Bullet L3).
//!
//! `POST {base}/v1/chat/completions` with `Authorization: Bearer`, body
//! `{model, max_tokens, temperature, messages:[system, user]}`; the reply's
//! `choices[0].message.content` is fed to the shared [`parse_stage2_json`].
//!
//! This one adapter covers OpenAI/ChatGPT *and* the many providers that expose
//! an OpenAI-compatible endpoint — DeepSeek, Qwen (DashScope compat mode),
//! Moonshot, Groq, Together, OpenRouter, local Ollama — by pointing `base_url`
//! at the provider. `response_format` is intentionally omitted so endpoints that
//! don't support it still work; the tolerant parser handles prose-wrapped JSON.

use serde_json::Value;

use super::{
    build_http_client, parse_stage2_json, LlmConfig, LlmError, LlmProvider, Stage2Request,
    Stage2Result, STAGE2_MAX_TOKENS, STAGE2_SYSTEM_PROMPT,
};

pub struct OpenAiProvider {
    client: reqwest::Client,
    config: LlmConfig,
}

impl OpenAiProvider {
    pub fn new(config: LlmConfig) -> Result<Self, LlmError> {
        Ok(Self {
            client: build_http_client()?,
            config,
        })
    }
}

/// Build the JSON request body for the Chat Completions API.
fn build_body(config: &LlmConfig, req: &Stage2Request) -> Value {
    serde_json::json!({
        "model": config.model(),
        "max_tokens": STAGE2_MAX_TOKENS,
        "temperature": 0,
        "messages": [
            { "role": "system", "content": STAGE2_SYSTEM_PROMPT },
            { "role": "user", "content": req.user_message() },
        ],
    })
}

/// Extract the assistant text from a Chat Completions response.
fn extract_text(json: &Value) -> Result<String, LlmError> {
    json.get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| LlmError::Parse("no choices[0].message.content in response".to_string()))
}

impl LlmProvider for OpenAiProvider {
    async fn classify(&self, req: &Stage2Request) -> Result<Stage2Result, LlmError> {
        if !self.config.is_usable() {
            return Err(LlmError::NoApiKey);
        }
        let url = format!("{}/v1/chat/completions", self.config.base_url());
        let resp = self
            .client
            .post(&url)
            .bearer_auth(self.config.api_key.trim())
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
    fn body_has_system_and_user_messages() {
        let cfg = LlmConfig::new(ProviderKind::OpenAiCompatible, "sk-x");
        let req = Stage2Request {
            transcript: "the lord is my shepherd".to_string(),
            context: Some("comfort".to_string()),
        };
        let body = build_body(&cfg, &req);
        assert_eq!(body["model"], "gpt-4o-mini");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["role"], "user");
        assert!(body["messages"][1]["content"]
            .as_str()
            .unwrap()
            .contains("the lord is my shepherd"));
        assert!(body["messages"][1]["content"]
            .as_str()
            .unwrap()
            .contains("comfort"));
    }

    #[test]
    fn extracts_text_from_choices() {
        let resp = serde_json::json!({
            "choices": [{
                "message": { "role": "assistant", "content": "{\"is_scripture\": false, \"reference\": null, \"confidence\": 0.1}" }
            }],
        });
        let text = extract_text(&resp).unwrap();
        let parsed = parse_stage2_json(&text).unwrap();
        assert!(!parsed.is_scripture);
        assert_eq!(parsed.reference, None);
    }

    #[test]
    fn extract_text_errors_when_missing() {
        assert!(extract_text(&serde_json::json!({"choices": []})).is_err());
        assert!(extract_text(&serde_json::json!({"error": {"message": "bad"}})).is_err());
    }
}
