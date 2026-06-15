//! Provider-agnostic Stage-2 LLM client (Track L).
//!
//! One key field, any provider. Three wire formats cover the whole field:
//! - **Anthropic** (Claude) — `POST /v1/messages`, `x-api-key`.
//! - **OpenAI-compatible** — `POST {base}/v1/chat/completions`, `Bearer`. Covers
//!   OpenAI/ChatGPT *and* most others that expose an OpenAI-compatible endpoint:
//!   DeepSeek, Qwen, Moonshot, Groq, Together, OpenRouter, local Ollama, …
//!   (point `base_url` at the provider).
//! - **Google Gemini** — `POST /v1beta/models/{model}:generateContent?key=…`.
//!
//! L1 (this bullet) defines the shared types, key auto-detection, the shared
//! prompt + response parser, and the [`LlmProvider`] trait. The three adapters
//! and the dispatcher are L2–L5. Everything here is network-free and unit-tested.

use serde::{Deserialize, Serialize};
use thiserror::Error;

mod anthropic;
mod gemini;
mod openai;
pub use anthropic::AnthropicProvider;
pub use gemini::GeminiProvider;
pub use openai::OpenAiProvider;

/// Dispatch a Stage-2 classification to the provider named by `config.kind`
/// (Bullet L5). Builds the right adapter and runs one call.
pub async fn classify(config: &LlmConfig, req: &Stage2Request) -> Result<Stage2Result, LlmError> {
    if !config.is_usable() {
        return Err(LlmError::NoApiKey);
    }
    match config.kind {
        ProviderKind::Anthropic => AnthropicProvider::new(config.clone())?.classify(req).await,
        ProviderKind::OpenAiCompatible => OpenAiProvider::new(config.clone())?.classify(req).await,
        ProviderKind::Gemini => GeminiProvider::new(config.clone())?.classify(req).await,
    }
}

/// Output-token cap for Stage-2 replies (a small JSON object — keep it tight).
pub(crate) const STAGE2_MAX_TOKENS: u32 = 256;

/// HTTP request timeout for a Stage-2 call. Real-time path: fail fast rather
/// than stall the detection loop.
pub(crate) const STAGE2_TIMEOUT_SECS: u64 = 15;

/// Build a reqwest client with the Stage-2 timeout, or an [`LlmError::Http`].
pub(crate) fn build_http_client() -> Result<reqwest::Client, LlmError> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(STAGE2_TIMEOUT_SECS))
        .build()
        .map_err(|e| LlmError::Http(e.to_string()))
}

/// Which provider's wire format a key/config targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    /// Anthropic (Claude).
    Anthropic,
    /// OpenAI and any OpenAI-compatible endpoint (DeepSeek, Qwen, Groq, …).
    OpenAiCompatible,
    /// Google Gemini.
    Gemini,
}

impl ProviderKind {
    /// Best-effort detection from an API key's prefix:
    /// - `sk-ant-…` → [`Anthropic`](ProviderKind::Anthropic)
    /// - `AIza…` → [`Gemini`](ProviderKind::Gemini)
    /// - `sk-…` → [`OpenAiCompatible`](ProviderKind::OpenAiCompatible) (OpenAI and
    ///   most OpenAI-compatible providers all use `sk-…`, so the base URL — not
    ///   the key — distinguishes them; the UI lets the user pick/override)
    ///
    /// Returns `None` for an unrecognized prefix so the UI can prompt for the
    /// provider explicitly.
    pub fn detect_from_key(key: &str) -> Option<ProviderKind> {
        let k = key.trim();
        if k.starts_with("sk-ant-") {
            Some(ProviderKind::Anthropic)
        } else if k.starts_with("AIza") {
            Some(ProviderKind::Gemini)
        } else if k.starts_with("sk-") {
            Some(ProviderKind::OpenAiCompatible)
        } else {
            None
        }
    }

    /// Default model id when the user hasn't chosen one. These favor the fast,
    /// inexpensive tier — Stage-2 runs per ambiguous utterance in real time.
    pub fn default_model(self) -> &'static str {
        match self {
            ProviderKind::Anthropic => "claude-haiku-4-5-20251001",
            ProviderKind::OpenAiCompatible => "gpt-4o-mini",
            ProviderKind::Gemini => "gemini-1.5-flash",
        }
    }

    /// Default API base URL (no trailing slash). Custom OpenAI-compatible
    /// providers override this with their own base (e.g. `https://api.deepseek.com`).
    pub fn default_base_url(self) -> &'static str {
        match self {
            ProviderKind::Anthropic => "https://api.anthropic.com",
            ProviderKind::OpenAiCompatible => "https://api.openai.com",
            ProviderKind::Gemini => "https://generativelanguage.googleapis.com",
        }
    }
}

/// Resolved provider configuration: the user's key plus optional base-URL/model
/// overrides. Pasting a key + (optionally) picking a provider is all that's
/// required; everything else falls back to provider defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub kind: ProviderKind,
    pub api_key: String,
    /// Override base URL (for custom OpenAI-compatible providers). `None`/empty →
    /// provider default.
    #[serde(default)]
    pub base_url: Option<String>,
    /// Override model id. `None`/empty → provider default.
    #[serde(default)]
    pub model: Option<String>,
}

impl LlmConfig {
    pub fn new(kind: ProviderKind, api_key: impl Into<String>) -> Self {
        Self {
            kind,
            api_key: api_key.into(),
            base_url: None,
            model: None,
        }
    }

    /// Resolved base URL (override or provider default), trailing slash trimmed.
    pub fn base_url(&self) -> String {
        self.base_url
            .as_deref()
            .map(|s| s.trim().trim_end_matches('/'))
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| self.kind.default_base_url().to_string())
    }

    /// Resolved model id (override or provider default).
    pub fn model(&self) -> String {
        self.model
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| self.kind.default_model().to_string())
    }

    /// Whether a usable key is present.
    pub fn is_usable(&self) -> bool {
        !self.api_key.trim().is_empty()
    }
}

/// Input to a Stage-2 classification: the ambiguous utterance plus optional
/// context (e.g. the current sermon topic) to help the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage2Request {
    pub transcript: String,
    #[serde(default)]
    pub context: Option<String>,
}

impl Stage2Request {
    /// Build the user-message text sent to the provider.
    pub fn user_message(&self) -> String {
        match &self.context {
            Some(ctx) if !ctx.trim().is_empty() => {
                format!("Sermon context: {ctx}\n\nTranscript: {}", self.transcript)
            }
            _ => format!("Transcript: {}", self.transcript),
        }
    }
}

/// Structured Stage-2 result. The model is instructed to return JSON matching
/// this shape; [`parse_stage2_json`] extracts it from the raw reply.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stage2Result {
    /// Whether the utterance refers to a specific scripture.
    pub is_scripture: bool,
    /// Canonical reference if found (e.g. `"John 3:16"`), else `None`.
    #[serde(default)]
    pub reference: Option<String>,
    /// Model confidence, 0.0–1.0.
    pub confidence: f64,
}

/// Errors from a Stage-2 LLM call.
#[derive(Debug, Error)]
pub enum LlmError {
    #[error("no api key configured")]
    NoApiKey,
    #[error("http error: {0}")]
    Http(String),
    #[error("provider returned status {status}: {body}")]
    Status { status: u16, body: String },
    #[error("failed to parse provider response: {0}")]
    Parse(String),
}

/// Shared system instruction for Stage-2 classification across all providers.
pub const STAGE2_SYSTEM_PROMPT: &str = "You are a scripture-reference detector for a live sermon system. \
Given a short transcript snippet (optionally with sermon context), decide whether the speaker is referring to a specific Bible passage. \
Respond with ONLY a compact JSON object of the form {\"is_scripture\": <bool>, \"reference\": <string or null>, \"confidence\": <number 0..1>}. \
\"reference\" must be a canonical reference like \"John 3:16\" or \"Romans 8:1-4\", or null when not scripture. Output no text other than the JSON object.";

/// Parse a model's text reply (expected to be JSON) into a [`Stage2Result`].
/// Tolerant of surrounding prose or ```code fences``` by extracting the first
/// balanced-ish `{ … }` span (the outermost braces).
pub fn parse_stage2_json(text: &str) -> Result<Stage2Result, LlmError> {
    let json = extract_json_object(text)
        .ok_or_else(|| LlmError::Parse("no JSON object found in response".to_string()))?;
    serde_json::from_str::<Stage2Result>(json).map_err(|e| LlmError::Parse(e.to_string()))
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

/// A Stage-2 LLM backend. Each provider (Anthropic, OpenAI-compatible, Gemini)
/// implements this in L2–L4; the dispatcher (L5) picks one by [`LlmConfig::kind`].
#[allow(async_fn_in_trait)]
pub trait LlmProvider {
    /// Classify an ambiguous utterance into a [`Stage2Result`].
    async fn classify(&self, req: &Stage2Request) -> Result<Stage2Result, LlmError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_provider_from_key_prefix() {
        assert_eq!(
            ProviderKind::detect_from_key("sk-ant-api03-abc"),
            Some(ProviderKind::Anthropic)
        );
        assert_eq!(
            ProviderKind::detect_from_key("AIzaSyD-xyz"),
            Some(ProviderKind::Gemini)
        );
        assert_eq!(
            ProviderKind::detect_from_key("sk-proj-openai123"),
            Some(ProviderKind::OpenAiCompatible)
        );
        // DeepSeek/Groq/etc also use sk- → OpenAI-compatible (base URL distinguishes).
        assert_eq!(
            ProviderKind::detect_from_key("sk-deepseek-123"),
            Some(ProviderKind::OpenAiCompatible)
        );
        assert_eq!(ProviderKind::detect_from_key("not-a-key"), None);
        assert_eq!(ProviderKind::detect_from_key("  sk-ant-x "), Some(ProviderKind::Anthropic));
    }

    #[test]
    fn config_resolves_defaults_and_overrides() {
        let c = LlmConfig::new(ProviderKind::Anthropic, "sk-ant-x");
        assert_eq!(c.base_url(), "https://api.anthropic.com");
        assert_eq!(c.model(), "claude-haiku-4-5-20251001");
        assert!(c.is_usable());

        let mut d = LlmConfig::new(ProviderKind::OpenAiCompatible, "sk-x");
        d.base_url = Some("https://api.deepseek.com/".to_string()); // trailing slash trimmed
        d.model = Some("deepseek-chat".to_string());
        assert_eq!(d.base_url(), "https://api.deepseek.com");
        assert_eq!(d.model(), "deepseek-chat");

        // Empty overrides fall back to defaults.
        let mut e = LlmConfig::new(ProviderKind::Gemini, "AIza-x");
        e.base_url = Some("  ".to_string());
        e.model = Some(String::new());
        assert_eq!(e.base_url(), "https://generativelanguage.googleapis.com");
        assert_eq!(e.model(), "gemini-1.5-flash");
    }

    #[test]
    fn not_usable_without_key() {
        assert!(!LlmConfig::new(ProviderKind::Anthropic, "   ").is_usable());
    }

    #[test]
    fn user_message_includes_context_when_present() {
        let req = Stage2Request {
            transcript: "turn with me to the gospel".to_string(),
            context: Some("grace and forgiveness".to_string()),
        };
        let msg = req.user_message();
        assert!(msg.contains("Sermon context: grace and forgiveness"));
        assert!(msg.contains("Transcript: turn with me to the gospel"));

        let bare = Stage2Request {
            transcript: "hello".to_string(),
            context: None,
        };
        assert_eq!(bare.user_message(), "Transcript: hello");
    }

    #[test]
    fn parses_clean_json() {
        let r = parse_stage2_json(r#"{"is_scripture": true, "reference": "John 3:16", "confidence": 0.9}"#)
            .unwrap();
        assert!(r.is_scripture);
        assert_eq!(r.reference.as_deref(), Some("John 3:16"));
        assert!((r.confidence - 0.9).abs() < 1e-9);
    }

    #[test]
    fn parses_json_wrapped_in_prose_and_fences() {
        let text = "Sure! Here is the result:\n```json\n{\"is_scripture\": false, \"reference\": null, \"confidence\": 0.2}\n```\n";
        let r = parse_stage2_json(text).unwrap();
        assert!(!r.is_scripture);
        assert_eq!(r.reference, None);
    }

    #[test]
    fn parse_errors_on_garbage() {
        assert!(parse_stage2_json("no json here").is_err());
        assert!(parse_stage2_json("{not valid json}").is_err());
    }
}
