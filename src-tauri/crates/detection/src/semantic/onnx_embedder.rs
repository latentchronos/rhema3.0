//! Real ONNX Runtime embedder using `ort` and HuggingFace `tokenizers`.
//!
//! This module is only compiled when the `onnx` feature is enabled.

#[cfg(feature = "onnx")]
use std::path::Path;
#[cfg(feature = "onnx")]
use std::sync::Mutex;

#[cfg(feature = "onnx")]
use ort::session::Session;
#[cfg(feature = "onnx")]
use ort::value::Tensor;
#[cfg(feature = "onnx")]
use tokenizers::Tokenizer;

#[cfg(feature = "onnx")]
use crate::error::DetectionError;
#[cfg(feature = "onnx")]
use super::embedder::TextEmbedder;

/// ONNX-based text embedder.
///
/// Loads a transformer model exported to ONNX format and a corresponding
/// HuggingFace tokenizer.  Inference produces a fixed-dimension dense
/// vector via mean pooling over the last hidden state.
///
/// The inner `Session` requires `&mut self` for `run`, so we wrap it in a
/// `Mutex` to satisfy the `&self` signature of the `TextEmbedder` trait.
#[cfg(feature = "onnx")]
pub struct OnnxEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    dim: usize,
    prompt_prefix: String,
    has_position_ids: bool,
}

// Safety: Tokenizer is Send but not Sync by default.  We never share
// mutable references across threads — the tokenizer is only used behind
// the session mutex — so the Send + Sync bound required by TextEmbedder
// is safe.
#[cfg(feature = "onnx")]
unsafe impl Sync for OnnxEmbedder {}

/// L2-normalise a vector in place (no-op on a zero-norm vector).
#[cfg(feature = "onnx")]
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Masked mean-pool one sequence's token hidden states. `hidden` is the
/// `[seq_len, dim]` row-major slice for a single sequence; tokens with
/// `mask[tok] == 0` (padding) are excluded. Shared by `embed_impl` and
/// `embed_batch` so both produce identical vectors.
#[cfg(feature = "onnx")]
fn mean_pool(hidden: &[f32], mask: &[i64], seq_len: usize, dim: usize) -> Vec<f32> {
    let mut pooled = vec![0.0f32; dim];
    let mut mask_sum = 0.0f32;
    for tok in 0..seq_len {
        if mask[tok] > 0 {
            let offset = tok * dim;
            for d in 0..dim {
                pooled[d] += hidden[offset + d];
            }
            mask_sum += 1.0;
        }
    }
    if mask_sum > 0.0 {
        for d in 0..dim {
            pooled[d] /= mask_sum;
        }
    }
    pooled
}

#[cfg(feature = "onnx")]
impl OnnxEmbedder {
    /// Maximum number of tokens the model will accept.
    /// Bible verses are short (~20 tokens avg). 128 is plenty and 4x faster
    /// than 512 because the model doesn't process unnecessary padding tokens.
    /// MUST match the Python precompute script (data/precompute-embeddings-onnx.py MAX_LENGTH).
    const MAX_TOKENS: usize = 128;

    /// Load an ONNX model and its tokenizer from disk.
    ///
    /// `model_path` should point to a `.onnx` file and `tokenizer_path`
    /// to a `tokenizer.json` file (HuggingFace format).
    pub fn load(model_path: &Path, tokenizer_path: &Path) -> Result<Self, DetectionError> {
        // Determine thread counts: use half of available CPUs for intra-op
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let intra_threads = (num_cpus / 2).max(1);

        log::info!(
            "OnnxEmbedder: configuring session with {} intra-op threads (of {} CPUs), graph optimization ALL",
            intra_threads, num_cpus
        );

        let session = Session::builder()
            .map_err(|e| DetectionError::Internal(format!("ort session builder: {e}")))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| DetectionError::Internal(format!("ort optimization level: {e}")))?
            .with_intra_threads(intra_threads)
            .map_err(|e| DetectionError::Internal(format!("ort intra threads: {e}")))?
            .with_inter_threads(2)
            .map_err(|e| DetectionError::Internal(format!("ort inter threads: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| DetectionError::Internal(format!("ort load model: {e}")))?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| DetectionError::Internal(format!("tokenizer load: {e}")))?;

        // Ensure the tokenizer pads and truncates to our max length.
        let pad_id = tokenizer
            .get_vocab(true)
            .get("[PAD]")
            .copied()
            .unwrap_or(0);
        let pad_token = tokenizer
            .id_to_token(pad_id)
            .unwrap_or_else(|| "[PAD]".to_string());

        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(Self::MAX_TOKENS),
            pad_id,
            pad_token,
            ..Default::default()
        }));

        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: Self::MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| DetectionError::Internal(format!("tokenizer truncation: {e}")))?;

        let has_position_ids = session.inputs().iter().any(|i| i.name() == "position_ids");

        // Log all model inputs for diagnostics
        for input in session.inputs() {
            log::info!(
                "ONNX model input: name='{}', type={:?}",
                input.name(),
                input.dtype()
            );
        }
        for output in session.outputs() {
            log::info!(
                "ONNX model output: name='{}', type={:?}",
                output.name(),
                output.dtype()
            );
        }

        // Determine embedding dimension from the model's output shape.
        // The output is typically Tensor { shape: [batch, seq_len, dim], .. }.
        let dim = session
            .outputs()
            .first()
            .and_then(|outlet| {
                if let ort::value::ValueType::Tensor { ref shape, .. } = *outlet.dtype() {
                    // Shape derefs to &[i64]; grab the last dimension.
                    shape.last().copied()
                } else {
                    None
                }
            })
            .unwrap_or(-1);

        if dim <= 0 {
            return Err(DetectionError::Internal(
                "cannot determine embedding dimension from model output shape".into(),
            ));
        }

        log::info!(
            "OnnxEmbedder loaded: dim={}, model={}",
            dim,
            model_path.display()
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            dim: dim as usize,
            // No prefix — matches the Python precompute script which embeds
            // documents with no prefix. Symmetric mode gives highest similarity.
            prompt_prefix: String::new(),
            has_position_ids,
        })
    }

    /// Override the prompt prefix prepended to every input text.
    ///
    /// Some models (e.g. E5) expect `"query: "` for queries and
    /// `"passage: "` for documents.
    pub fn set_prompt_prefix(&mut self, prefix: impl Into<String>) {
        self.prompt_prefix = prefix.into();
    }

    /// Embed a single text string.
    ///
    /// Steps:
    /// 1. Prepend the prompt prefix.
    /// 2. Tokenize (pad / truncate to `MAX_TOKENS`).
    /// 3. Build `input_ids` and `attention_mask` tensors.
    /// 4. Run ONNX inference.
    /// 5. Mean-pool the last hidden state over the attention mask.
    /// 6. L2-normalise the resulting vector.
    fn embed_impl(&self, text: &str) -> Result<Vec<f32>, DetectionError> {
        let embed_start = std::time::Instant::now();
        let prefixed = format!("{}{}", self.prompt_prefix, text);

        let encoding = self
            .tokenizer
            .encode(prefixed, true)
            .map_err(|e| DetectionError::Internal(format!("tokenize: {e}")))?;

        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let seq_len = ids.len();

        // Build owned tensors with shape [1, seq_len].
        let shape = vec![1i64, seq_len as i64];

        let input_ids_data: Vec<i64> = ids.iter().map(|&v| v as i64).collect();
        let input_ids_tensor = Tensor::from_array((shape.clone(), input_ids_data))
            .map_err(|e| DetectionError::Internal(format!("input_ids tensor: {e}")))?;

        let attention_mask_data: Vec<i64> = mask.iter().map(|&v| v as i64).collect();
        let attention_mask_tensor = Tensor::from_array((shape.clone(), attention_mask_data))
            .map_err(|e| DetectionError::Internal(format!("attention_mask tensor: {e}")))?;

        // Qwen3 needs position_ids. For models that don't have this input, it's ignored.
        let position_ids_data: Vec<i64> = (0..seq_len as i64).collect();
        let position_ids_tensor = Tensor::from_array((shape, position_ids_data))
            .map_err(|e| DetectionError::Internal(format!("position_ids tensor: {e}")))?;

        // Qwen3 needs position_ids; BERT-style models don't
        let inputs = if self.has_position_ids {
            ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "position_ids" => position_ids_tensor,
            ]
        } else {
            ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ]
        };

        let mut session = self
            .session
            .lock()
            .map_err(|e| DetectionError::Internal(format!("session lock: {e}")))?;

        let outputs = session
            .run(inputs)
            .map_err(|e| DetectionError::Internal(format!("ort run: {e}")))?;

        // Prefer `sentence_embedding` (pre-pooled by sentence-transformers, shape [1, dim]).
        // Fall back to `token_embeddings`/`last_hidden_state` with manual pooling.
        let output_value = if outputs.contains_key("sentence_embedding") {
            &outputs["sentence_embedding"]
        } else if outputs.contains_key("last_hidden_state") {
            &outputs["last_hidden_state"]
        } else {
            &outputs[0usize]
        };

        let (out_shape, data) = output_value
            .try_extract_tensor::<f32>()
            .map_err(|e| DetectionError::Internal(format!("extract tensor: {e}")))?;

        let out_dims: &[i64] = &*out_shape;

        let mut result = if out_dims.len() == 2 {
            // sentence_embedding: shape [1, dim] — already pooled by sentence-transformers
            let dim = out_dims[1] as usize;
            data[..dim].to_vec()
        } else if out_dims.len() == 3 {
            // token_embeddings: shape [1, seq_len, dim] — mean pooling over attention mask
            // MUST match the Python precompute script (data/precompute-embeddings-onnx.py)
            // which uses mean pooling. Using last-token pooling here would put queries
            // in a different vector space than the pre-computed verse embeddings.
            let seq_len = out_dims[1] as usize;
            let dim = out_dims[2] as usize;
            let mask_i64: Vec<i64> = mask.iter().map(|&v| v as i64).collect();
            mean_pool(&data, &mask_i64, seq_len, dim)
        } else {
            return Err(DetectionError::Internal(format!(
                "unexpected tensor rank: {:?}",
                out_dims
            )));
        };

        // L2 normalise (safe to re-normalize even if already normalized)
        l2_normalize(&mut result);

        let elapsed = embed_start.elapsed();
        log::info!(
            "[ONNX] embed() took {:?} for {} chars",
            elapsed,
            text.len()
        );

        Ok(result)
    }

    /// Embed many texts in a single ONNX run. Because the tokenizer pads every
    /// sequence to a fixed length (`MAX_TOKENS`), each batch row is identical to
    /// `embed_impl` of that text (same masked mean-pool + L2) — verified by the
    /// `embed_batch_matches_single` equivalence test. Used by the offline verse
    /// precompute to amortize per-call overhead across the batch (~10x fewer runs).
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, DetectionError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{}", self.prompt_prefix, t))
            .collect();

        let encodings = self
            .tokenizer
            .encode_batch(prefixed, true)
            .map_err(|e| DetectionError::Internal(format!("tokenize batch: {e}")))?;

        let batch = encodings.len();
        // Fixed padding (MAX_TOKENS) → every row has the same length.
        let seq_len = encodings[0].get_ids().len();

        let mut input_ids_data: Vec<i64> = Vec::with_capacity(batch * seq_len);
        let mut attention_mask_data: Vec<i64> = Vec::with_capacity(batch * seq_len);
        let mut position_ids_data: Vec<i64> = Vec::with_capacity(batch * seq_len);
        let mut masks: Vec<Vec<i64>> = Vec::with_capacity(batch);
        for enc in &encodings {
            let ids = enc.get_ids();
            if ids.len() != seq_len {
                return Err(DetectionError::Internal(format!(
                    "batch row length {} != {seq_len} (fixed padding expected)",
                    ids.len()
                )));
            }
            let mask_i64: Vec<i64> = enc.get_attention_mask().iter().map(|&v| v as i64).collect();
            for &v in ids {
                input_ids_data.push(v as i64);
            }
            attention_mask_data.extend_from_slice(&mask_i64);
            for p in 0..seq_len as i64 {
                position_ids_data.push(p);
            }
            masks.push(mask_i64);
        }

        let shape = vec![batch as i64, seq_len as i64];
        let input_ids_tensor = Tensor::from_array((shape.clone(), input_ids_data))
            .map_err(|e| DetectionError::Internal(format!("input_ids tensor: {e}")))?;
        let attention_mask_tensor = Tensor::from_array((shape.clone(), attention_mask_data))
            .map_err(|e| DetectionError::Internal(format!("attention_mask tensor: {e}")))?;
        let position_ids_tensor = Tensor::from_array((shape, position_ids_data))
            .map_err(|e| DetectionError::Internal(format!("position_ids tensor: {e}")))?;

        let inputs = if self.has_position_ids {
            ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "position_ids" => position_ids_tensor,
            ]
        } else {
            ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ]
        };

        let mut session = self
            .session
            .lock()
            .map_err(|e| DetectionError::Internal(format!("session lock: {e}")))?;
        let outputs = session
            .run(inputs)
            .map_err(|e| DetectionError::Internal(format!("ort run: {e}")))?;

        let output_value = if outputs.contains_key("sentence_embedding") {
            &outputs["sentence_embedding"]
        } else if outputs.contains_key("last_hidden_state") {
            &outputs["last_hidden_state"]
        } else {
            &outputs[0usize]
        };
        let (out_shape, data) = output_value
            .try_extract_tensor::<f32>()
            .map_err(|e| DetectionError::Internal(format!("extract tensor: {e}")))?;
        let out_dims: &[i64] = &*out_shape;

        let mut results = Vec::with_capacity(batch);
        if out_dims.len() == 2 {
            // [batch, dim] — pre-pooled sentence embeddings.
            let dim = out_dims[1] as usize;
            for b in 0..batch {
                let mut row = data[b * dim..(b + 1) * dim].to_vec();
                l2_normalize(&mut row);
                results.push(row);
            }
        } else if out_dims.len() == 3 {
            // [batch, seq_len, dim] — masked mean pool per row (same as embed_impl).
            let out_seq = out_dims[1] as usize;
            let dim = out_dims[2] as usize;
            let row_stride = out_seq * dim;
            for b in 0..batch {
                let hidden = &data[b * row_stride..(b + 1) * row_stride];
                let mut pooled = mean_pool(hidden, &masks[b], out_seq, dim);
                l2_normalize(&mut pooled);
                results.push(pooled);
            }
        } else {
            return Err(DetectionError::Internal(format!(
                "unexpected tensor rank: {:?}",
                out_dims
            )));
        }

        Ok(results)
    }
}

#[cfg(all(test, feature = "onnx"))]
mod tests {
    use super::OnnxEmbedder;
    use std::path::Path;

    /// Load the embedder from env-provided model/tokenizer paths, or `None` to skip.
    fn load_from_env() -> Option<OnnxEmbedder> {
        let model = std::env::var("RHEMA_ONNX_MODEL").ok()?;
        let tokenizer = std::env::var("RHEMA_ONNX_TOKENIZER").ok()?;
        OnnxEmbedder::load(Path::new(&model), Path::new(&tokenizer)).ok()
    }

    /// `embed_batch` must return vectors numerically identical to per-text `embed_impl`
    /// — batching is a throughput optimization, it must not move any vector. Skips
    /// silently unless RHEMA_ONNX_MODEL + RHEMA_ONNX_TOKENIZER point at a real model.
    #[test]
    fn embed_batch_matches_single() {
        let Some(embedder) = load_from_env() else {
            eprintln!(
                "skipping embed_batch_matches_single: set RHEMA_ONNX_MODEL + RHEMA_ONNX_TOKENIZER"
            );
            return;
        };
        let texts = [
            "For God so loved the world",
            "In the beginning was the Word",
            "The Lord is my shepherd, I shall not want",
        ];
        let singles: Vec<Vec<f32>> = texts
            .iter()
            .map(|t| embedder.embed_impl(t).expect("single embed"))
            .collect();
        let batched = embedder.embed_batch(&texts).expect("batch embed");

        assert_eq!(batched.len(), texts.len());
        for (i, (b, s)) in batched.iter().zip(singles.iter()).enumerate() {
            assert_eq!(b.len(), s.len(), "row {i}: dim mismatch");
            let max_diff = b
                .iter()
                .zip(s.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(max_diff < 1e-4, "row {i}: max diff {max_diff} exceeds 1e-4");
        }
    }
}

#[cfg(feature = "onnx")]
impl TextEmbedder for OnnxEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, DetectionError> {
        self.embed_impl(text)
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}
