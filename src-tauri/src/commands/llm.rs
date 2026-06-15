//! Stage-2 LLM provider configuration commands (Bullet L5/L6).
//!
//! The provider config (key + optional base-URL/model) lives in a dedicated
//! managed `Mutex<Option<LlmConfig>>` so the Stage-2 worker and these commands
//! can share it without touching the hot `AppState` lock. The key is held only
//! in memory; the frontend persists it (mirroring the Deepgram key) and pushes
//! it back on startup. Status never returns the key.

use std::sync::Mutex;

use rhema_api::llm::{LlmConfig, ProviderKind};
use serde::Serialize;
use tauri::State;

/// Trim a possibly-empty override down to `None`.
fn norm(value: Option<String>) -> Option<String> {
    value
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Set / update the Stage-2 LLM provider config. `provider` may be omitted to
/// auto-detect from the key prefix (`"anthropic"`, `"open_ai_compatible"`,
/// `"gemini"` when given). Returns the resolved provider so the UI can confirm.
#[tauri::command]
pub fn set_llm_config(
    provider: Option<ProviderKind>,
    api_key: String,
    base_url: Option<String>,
    model: Option<String>,
    llm: State<'_, Mutex<Option<LlmConfig>>>,
) -> Result<ProviderKind, String> {
    if api_key.trim().is_empty() {
        return Err("API key is empty".to_string());
    }
    let kind = match provider {
        Some(k) => k,
        None => ProviderKind::detect_from_key(&api_key).ok_or_else(|| {
            "Could not detect the provider from the key prefix; please choose one.".to_string()
        })?,
    };
    let config = LlmConfig {
        kind,
        api_key: api_key.trim().to_string(),
        base_url: norm(base_url),
        model: norm(model),
    };
    *llm.lock().map_err(|e| e.to_string())? = Some(config);
    log::info!("llm: configured provider {kind:?}");
    Ok(kind)
}

/// Provider status for the UI (never includes the key).
#[derive(Serialize)]
pub struct LlmStatus {
    pub configured: bool,
    pub provider: Option<ProviderKind>,
    pub model: Option<String>,
    pub base_url: Option<String>,
}

/// Report whether a provider is configured, and the resolved model/base URL.
#[tauri::command]
pub fn llm_status(llm: State<'_, Mutex<Option<LlmConfig>>>) -> Result<LlmStatus, String> {
    let guard = llm.lock().map_err(|e| e.to_string())?;
    Ok(match &*guard {
        Some(c) => LlmStatus {
            configured: true,
            provider: Some(c.kind),
            model: Some(c.model()),
            base_url: Some(c.base_url()),
        },
        None => LlmStatus {
            configured: false,
            provider: None,
            model: None,
            base_url: None,
        },
    })
}

/// Clear the configured provider (Stage-2 falls back to no-op).
#[tauri::command]
pub fn clear_llm_config(llm: State<'_, Mutex<Option<LlmConfig>>>) -> Result<(), String> {
    *llm.lock().map_err(|e| e.to_string())? = None;
    log::info!("llm: cleared provider config");
    Ok(())
}
