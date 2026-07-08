pub mod deepgram;
pub mod engine;
pub mod error;
pub mod keyterms;
pub mod rest;
pub mod types;

#[cfg(feature = "local-stt")]
pub mod local;

pub use deepgram::DeepgramClient;
pub use engine::SttEngine;
pub use error::SttError;
pub use keyterms::bible_keyterms;
pub use types::{SttConfig, TranscriptEvent, Word};

pub use rest::DeepgramRestClient;

#[cfg(feature = "local-stt")]
pub use local::LocalSttClient;
