//! Local llama.cpp backend for the comprehension observer. All model code is
//! behind the `llama` feature so the default build compiles no native code.
#![cfg_attr(not(feature = "llama"), allow(unused))]

#[cfg(feature = "llama")]
mod model;

#[cfg(feature = "llama")]
pub use model::{LlamaComprehensionModel, LlamaConfig};
