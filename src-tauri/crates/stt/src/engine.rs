use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use async_trait::async_trait;
use crossbeam_channel::Receiver;
use tokio::sync::mpsc;

use crate::error::SttError;
use crate::types::TranscriptEvent;

/// A speech-to-text engine: consumes 16 kHz mono `i16` PCM and emits transcript events.
///
/// This is the adapter seam. Both the Deepgram cloud client ([`crate::DeepgramClient`])
/// and the on-device local client ([`crate::local::LocalSttClient`], behind the
/// `local-stt` feature) implement it, so `commands::stt` can pick an engine at
/// runtime without touching the audio pipeline or the detection side.
///
/// Contract (identical to the pre-existing `DeepgramClient::connect`):
/// - `audio_rx`: 16 kHz mono `i16` PCM frames from the capture fan-out.
/// - `event_tx`: where the engine publishes [`TranscriptEvent`]s.
/// - `keep_running`: liveness flag; `false` means the user has stopped — the engine
///   should return `Ok(())` promptly.
#[async_trait]
pub trait SttEngine: Send + Sync {
    async fn connect(
        &self,
        audio_rx: Receiver<Vec<i16>>,
        event_tx: mpsc::Sender<TranscriptEvent>,
        keep_running: Arc<AtomicBool>,
    ) -> Result<(), SttError>;
}
