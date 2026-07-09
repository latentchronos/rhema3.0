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
            n_threads: 4,
            max_gen_tokens: 160,
            name: "qwen3-1.7b-q4_k_m".to_string(),
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
        let caps = Capabilities {
            name: cfg.name.clone(),
            max_context_tokens: cfg.n_ctx as usize,
            supports_structured_output: true,
            supports_streaming: false,
        };
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

    /// Free (unconstrained) completion — used for generic prompts and testing.
    pub fn complete(&self, system: &str, user: &str) -> Result<String, ModelError> {
        self.generate(system, user, None)
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

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.cfg.n_ctx))
            .with_n_threads(self.cfg.n_threads);
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
        ModelHealth::Ready
    }

    async fn infer(&self, prompt: &str, schema: &OutputSchema) -> Result<Decision, ModelError> {
        // `/no_think` is Qwen3's documented soft switch to skip its reasoning
        // mode — we want a terse, direct classification.
        let system = format!("{} /no_think", schema.instruction());
        // Grammar-constrained: the reply is guaranteed to be a valid Decision.
        let raw = self.generate(&system, prompt, Some(&self.gbnf))?;
        parse_decision(&raw).map_err(|e| ModelError::Inference(format!("parse decision: {e}")))
    }
}
