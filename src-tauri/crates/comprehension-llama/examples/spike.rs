//! Throwaway B0 spike: prove llama.cpp builds + loads Qwen3-1.7B + generates,
//! and measure tok/s on the target. NOT shipped. Run:
//!   LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu \
//!   cargo run -p rhema-comprehension-llama --example spike --features llama --release -- ../model/Qwen3-1.7B-Q4_K_M.gguf
use std::path::Path;
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: spike <model.gguf>");
    let backend = LlamaBackend::init()?;
    let model = LlamaModel::load_from_file(&backend, Path::new(&path), &LlamaModelParams::default())?;

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(1024))
        .with_n_threads(4);
    let mut ctx = model.new_context(&backend, ctx_params)?;

    // `/no_think` biases Qwen3 out of its reasoning mode toward a direct reply.
    let prompt = "You are a sermon comprehension observer. /no_think New transcript: \
'a certain man had two sons and the younger asked for his inheritance'. \
Reply ONLY with JSON like {\"decision\":\"STATE_CHANGED\",\"new_state\":{\"state\":\"STORY_TELLING\",\"supporting_activities\":[],\"passages\":[\"Luke 15\"],\"confidence\":0.9}}.";

    let tokens = model.str_to_token(prompt, AddBos::Always)?;
    let n_prompt = tokens.len();
    let mut batch = LlamaBatch::new(512, 1);
    for (i, tok) in tokens.iter().enumerate() {
        batch.add(*tok, i as i32, &[0], i == tokens.len() - 1)?;
    }
    ctx.decode(&mut batch)?;

    let mut sampler = LlamaSampler::greedy();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut out = String::new();
    let mut n_gen = 0i32;
    let start = Instant::now();
    let mut pos = n_prompt as i32;
    // Sample from the last token that had logits computed (the prompt's last
    // token first, then the single freshly-decoded token thereafter).
    let mut sample_idx = batch.n_tokens() - 1;
    for _ in 0..96 {
        let next = sampler.sample(&ctx, sample_idx);
        sampler.accept(next);
        if next == model.token_eos() {
            break;
        }
        out.push_str(&model.token_to_piece(next, &mut decoder, false, None)?);
        batch.clear();
        batch.add(next, pos, &[0], true)?;
        ctx.decode(&mut batch)?;
        sample_idx = batch.n_tokens() - 1;
        pos += 1;
        n_gen += 1;
    }
    let secs = start.elapsed().as_secs_f64();
    eprintln!("\n--- SPIKE RESULT ---");
    eprintln!(
        "prompt_tokens={n_prompt} generated_tokens={n_gen} secs={secs:.2} tok_s={:.2}",
        n_gen as f64 / secs
    );
    println!("OUTPUT:\n{out}");
    Ok(())
}
