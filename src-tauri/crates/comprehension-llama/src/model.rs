//! The shipped local backend: [`LlamaComprehensionModel`] implements
//! `rhema_comprehension::ComprehensionModel` over `llama-cpp-2` (llama.cpp),
//! running a local GGUF (Qwen3-1.7B on this hardware) entirely on CPU.
//!
//! Everything here is behind the `llama` feature so the default build compiles
//! no native code. The model is loaded once; a fresh context is created per
//! inference (cheap relative to the model load, and it sidesteps the
//! self-referential model/context borrow).

use std::num::NonZeroU32;
use std::path::Path;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use rhema_comprehension::{
    parse_decision, Capabilities, ComprehensionModel, Decision, ModelError, ModelHealth,
    OutputSchema,
};

use crate::grammar;

/// Runtime knobs for the local model. Model path is separate (see [`LlamaComprehensionModel::load`]).
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    /// Context window (KV-cache size). Must hold prompt + generated tokens.
    pub n_ctx: u32,
    /// CPU threads (physical cores on the i5-8265U = 4).
    pub n_threads: i32,
    /// Hard cap on generated tokens per call.
    pub max_gen_tokens: usize,
    /// Human-readable model name reported in [`Capabilities`].
    pub name: String,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            n_ctx: 2048,
            // Deliberately HALF the 4 physical cores on the target i5-8265U so a
            // comprehension inference can't pin every core and starve the realtime
            // STT decode (the I6 lag fix). Override with RHEMA_COMPREHENSION_THREADS
            // on hardware with more headroom.
            n_threads: 2,
            max_gen_tokens: 160,
            name: "qwen3-1.7b-q4_k_m".to_string(),
        }
    }
}

impl LlamaConfig {
    /// The [`Capabilities`] a model loaded with this config reports (§16, D6).
    /// Pure (no model needed): `max_context_tokens` drives the engine's §17
    /// rolling-summary sizing. This backend always enforces structured output
    /// (via the grammar) and does not stream.
    pub fn capabilities(&self) -> Capabilities {
        Capabilities {
            name: self.name.clone(),
            max_context_tokens: self.n_ctx as usize,
            supports_structured_output: true,
            supports_streaming: false,
        }
    }
}

/// A local llama.cpp-backed comprehension model. Load once, keep resident.
pub struct LlamaComprehensionModel {
    backend: LlamaBackend,
    model: LlamaModel,
    cfg: LlamaConfig,
    caps: Capabilities,
    /// GBNF derived from the §8 `Decision` schema, built once at load.
    gbnf: String,
}

impl LlamaComprehensionModel {
    /// Load a GGUF model from disk on CPU. Called once at startup, like Whisper.
    pub fn load(path: &Path, cfg: LlamaConfig) -> Result<Self, ModelError> {
        let backend = LlamaBackend::init().map_err(|e| ModelError::Inference(e.to_string()))?;
        let model = LlamaModel::load_from_file(&backend, path, &LlamaModelParams::default())
            .map_err(|e| ModelError::Inference(format!("load model: {e}")))?;
        let caps = cfg.capabilities();
        let gbnf = llama_cpp_2::json_schema_to_grammar(grammar::DECISION_SCHEMA)
            .map_err(|e| ModelError::Inference(format!("build grammar: {e}")))?;
        Ok(Self {
            backend,
            model,
            cfg,
            caps,
            gbnf,
        })
    }

    /// Format a system + user turn in Qwen3's ChatML layout so the instruct
    /// model actually follows instructions (a raw prompt does not).
    fn build_chatml(system: &str, user: &str) -> String {
        format!(
            "<|im_start|>system\n{system}<|im_end|>\n\
             <|im_start|>user\n{user}<|im_end|>\n\
             <|im_start|>assistant\n"
        )
    }

    /// Config-driven load from the environment. Reads the GGUF path from
    /// `RHEMA_COMPREHENSION_MODEL` (absolute path), with optional overrides
    /// `RHEMA_COMPREHENSION_NCTX` / `RHEMA_COMPREHENSION_THREADS`. Returns
    /// `Ok(None)` when the path var is unset (comprehension simply disabled).
    /// Phase I feeds the same values from user settings instead.
    pub fn from_env() -> Result<Option<Self>, ModelError> {
        let path = match std::env::var("RHEMA_COMPREHENSION_MODEL") {
            Ok(p) if !p.trim().is_empty() => p,
            _ => return Ok(None),
        };
        let mut cfg = LlamaConfig::default();
        if let Some(v) = std::env::var("RHEMA_COMPREHENSION_NCTX")
            .ok()
            .and_then(|s| s.parse().ok())
        {
            cfg.n_ctx = v;
        }
        if let Some(v) = std::env::var("RHEMA_COMPREHENSION_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
        {
            cfg.n_threads = v;
        }
        Self::load(Path::new(&path), cfg).map(Some)
    }

    /// Free (unconstrained) completion — used for generic prompts and testing.
    pub fn complete(&self, system: &str, user: &str) -> Result<String, ModelError> {
        self.generate(system, user, None)
    }

    /// Synchronous, CPU-blocking inference — the real work behind [`ComprehensionModel::infer`].
    /// Exposed so the app can run it inside `tokio::task::spawn_blocking`: the
    /// grammar-constrained `generate()` blocks its thread for seconds, which would
    /// otherwise stall the async runtime (the I6 lag fix). Returns a guaranteed-valid
    /// [`Decision`] (the grammar enforces the shape).
    pub fn infer_blocking(
        &self,
        prompt: &str,
        schema: &OutputSchema,
    ) -> Result<Decision, ModelError> {
        // `/no_think` is Qwen3's documented soft switch to skip its reasoning
        // mode — we want a terse, direct classification.
        let system = format!("{} /no_think", schema.instruction());
        let raw = self.generate(&system, prompt, Some(&self.gbnf))?;
        parse_decision(&raw).map_err(|e| ModelError::Inference(format!("parse decision: {e}")))
    }

    /// Run one blocking generation over a system+user turn and return the raw
    /// assistant text. `grammar` (GBNF) optionally constrains the output. This
    /// is the shared engine behind [`ComprehensionModel::infer`].
    fn generate(
        &self,
        system: &str,
        user: &str,
        grammar: Option<&str>,
    ) -> Result<String, ModelError> {
        let prompt = Self::build_chatml(system, user);

        // Cap BOTH generation and prompt-eval (batch) threads: the prompt-eval
        // pass is the most CPU-intensive part, so leaving batch threads at the
        // llama.cpp default would still saturate every core and starve STT.
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.cfg.n_ctx))
            .with_n_threads(self.cfg.n_threads)
            .with_n_threads_batch(self.cfg.n_threads);
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| ModelError::Inference(format!("new_context: {e}")))?;

        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| ModelError::Inference(format!("tokenize: {e}")))?;
        let n_prompt = tokens.len();
        if n_prompt + self.cfg.max_gen_tokens >= self.cfg.n_ctx as usize {
            return Err(ModelError::Inference(format!(
                "prompt too long: {n_prompt} tokens + {} gen exceeds n_ctx {}",
                self.cfg.max_gen_tokens, self.cfg.n_ctx
            )));
        }

        let mut batch = LlamaBatch::new(self.cfg.n_ctx as usize, 1);
        for (i, tok) in tokens.iter().enumerate() {
            batch
                .add(*tok, i as i32, &[0], i == n_prompt - 1)
                .map_err(|e| ModelError::Inference(format!("batch add: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| ModelError::Inference(format!("decode prompt: {e}")))?;

        // Grammar sampler (if any) masks invalid tokens; greedy then picks the
        // best allowed one. `sampler.accept` (below) advances the grammar state.
        let mut sampler = match grammar {
            Some(g) => {
                let gr = LlamaSampler::grammar(&self.model, g, grammar::DECISION_ROOT)
                    .map_err(|e| ModelError::Inference(format!("grammar: {e}")))?;
                LlamaSampler::chain_simple([gr, LlamaSampler::greedy()])
            }
            None => LlamaSampler::greedy(),
        };
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut out = String::new();
        let mut pos = n_prompt as i32;
        // First sample reads the prompt's last-token logits (batch index
        // n_tokens-1); thereafter each single-token decode puts logits at 0.
        let mut sample_idx = batch.n_tokens() - 1;
        for _ in 0..self.cfg.max_gen_tokens {
            // `sample` already calls `accept` internally (llama_sampler_sample),
            // so we must NOT accept again — a double-accept overshoots a stateful
            // grammar sampler and empties its stack.
            let next = sampler.sample(&ctx, sample_idx);
            if next == self.model.token_eos() {
                break;
            }
            out.push_str(
                &self
                    .model
                    .token_to_piece(next, &mut decoder, false, None)
                    .map_err(|e| ModelError::Inference(format!("detokenize: {e}")))?,
            );
            // With a grammar the object is guaranteed valid; stop as soon as it
            // parses, so we never sample past completion (which empties the
            // grammar stack and aborts llama.cpp).
            if grammar.is_some() && parse_decision(&out).is_ok() {
                break;
            }
            batch.clear();
            batch
                .add(next, pos, &[0], true)
                .map_err(|e| ModelError::Inference(format!("batch add: {e}")))?;
            ctx.decode(&mut batch)
                .map_err(|e| ModelError::Inference(format!("decode: {e}")))?;
            sample_idx = batch.n_tokens() - 1;
            pos += 1;
        }
        Ok(out)
    }
}

impl ComprehensionModel for LlamaComprehensionModel {
    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    async fn health(&self) -> ModelHealth {
        // An existing instance is, by construction, a loaded model (load is
        // synchronous and fails with `Err` rather than yielding a half-ready
        // instance). Async load-state tracking (Loading/Error before the model
        // exists) is the app's concern in Phase I, which can wrap `load` in a
        // background task and surface those states itself.
        ModelHealth::Ready
    }

    async fn infer(&self, prompt: &str, schema: &OutputSchema) -> Result<Decision, ModelError> {
        // Delegates to the synchronous path. In the app this is instead called
        // via `spawn_blocking` (see commands/comprehension.rs) so the CPU-heavy
        // generation never stalls the async runtime.
        self.infer_blocking(prompt, schema)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_capabilities_reflect_n_ctx() {
        let cfg = LlamaConfig {
            n_ctx: 1536,
            n_threads: 2,
            max_gen_tokens: 100,
            name: "custom".to_string(),
        };
        let caps = cfg.capabilities();
        assert_eq!(caps.max_context_tokens, 1536);
        assert_eq!(caps.name, "custom");
        assert!(caps.supports_structured_output);
        assert!(!caps.supports_streaming);
    }

    #[test]
    fn default_config_is_the_1_7b() {
        let caps = LlamaConfig::default().capabilities();
        assert_eq!(caps.name, "qwen3-1.7b-q4_k_m");
        assert_eq!(caps.max_context_tokens, 2048);
    }

    #[test]
    fn default_threads_leave_headroom_for_realtime_stt() {
        // Half the 4 physical cores on the target: a comprehension inference must
        // not pin every core and starve the STT decode (I6 lag fix). If this ever
        // regresses to a higher default, revisit the CPU-contention analysis.
        assert_eq!(LlamaConfig::default().n_threads, 2);
    }
}
