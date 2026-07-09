//! Manual smoke test for the local llama backend (needs the model on disk).
//! Skips gracefully unless `RHEMA_COMPREHENSION_MODEL` points at a GGUF. Run:
//!   LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu \
//!   BINDGEN_EXTRA_CLANG_ARGS=-I/usr/lib/gcc/x86_64-linux-gnu/15/include \
//!   RHEMA_COMPREHENSION_MODEL=../model/Qwen3-1.7B-Q4_K_M.gguf \
//!   cargo test -p rhema-comprehension-llama --features llama --release -- --nocapture
#![cfg(feature = "llama")]

use std::path::Path;

use rhema_comprehension::{ComprehensionModel, OutputSchema};
use rhema_comprehension_llama::{LlamaComprehensionModel, LlamaConfig};

fn model_path() -> Option<String> {
    match std::env::var("RHEMA_COMPREHENSION_MODEL") {
        Ok(p) => Some(p),
        Err(_) => {
            eprintln!("SKIP: set RHEMA_COMPREHENSION_MODEL to the GGUF path to run this test");
            None
        }
    }
}

#[test]
fn loads_and_completes() {
    let Some(path) = model_path() else { return };
    let model =
        LlamaComprehensionModel::load(Path::new(&path), LlamaConfig::default()).expect("load model");
    let raw = model
        .complete("You are a helpful assistant. /no_think", "Say hello in one word.")
        .expect("complete");
    eprintln!("RAW COMPLETION: {raw:?}");
    assert!(!raw.trim().is_empty(), "model produced no output");
}

#[test]
fn infer_runs_end_to_end() {
    let Some(path) = model_path() else { return };
    let model =
        LlamaComprehensionModel::load(Path::new(&path), LlamaConfig::default()).expect("load model");
    let schema = OutputSchema::default();
    let prompt = "CURRENT_UNDERSTANDING:\nnull\n\nSERMON_ARC:\n(nothing yet)\n\n\
                  NEW_TRANSCRIPT:\na certain man had two sons and the younger asked for his inheritance";
    // The GBNF grammar (B2) guarantees the output parses into a Decision.
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let result = rt.block_on(model.infer(prompt, &schema));
    eprintln!("INFER RESULT: {result:?}");
    assert!(result.is_ok(), "grammar must guarantee a parseable Decision: {result:?}");
}
