//! Background comprehension observer worker. Fed coalesced final sentences from
//! the STT detection worker; gated by `session_active`; emits state transitions.
//!
//! All model/inference code that touches llama.cpp is behind the
//! `local-comprehension` feature. Without the feature (or with no model
//! configured) the worker simply drains the feed channel and no events fire, so
//! the default build stays native-free and green.
use std::sync::atomic::Ordering;
use std::sync::Mutex;

use tauri::{AppHandle, Emitter, Manager};

use rhema_comprehension::Observer;

use crate::events::EVENT_COMPREHENSION_STATE;
use crate::state::AppState;

/// Managed holder for the sentence-feed sender (filled at setup). The STT
/// detection worker taps coalesced sentences into this channel.
// The sender is only *read* by the STT feed added in Task I2; until then it is
// write-only at setup, so silence the dead-field lint (the allow no-ops once I2 lands).
#[allow(dead_code)]
pub struct ComprehensionFeed(pub Mutex<Option<tokio::sync::mpsc::Sender<String>>>);

/// Millisecond monotonic clock for the Observer's window math (a fixed baseline
/// captured at worker start; only deltas matter).
fn now_ms(base: std::time::Instant) -> u64 {
    base.elapsed().as_millis() as u64
}

pub async fn run_comprehension_worker(
    app: AppHandle,
    mut rx: tokio::sync::mpsc::Receiver<String>,
) {
    let base = std::time::Instant::now();

    // Load the local model once (feature-gated). Without the feature, or with no
    // model configured, there is no model and the worker just drains the channel.
    #[cfg(feature = "local-comprehension")]
    let model: Option<rhema_comprehension_llama::LlamaComprehensionModel> =
        match rhema_comprehension_llama::LlamaComprehensionModel::from_env() {
            Ok(m) => m,
            Err(e) => {
                log::warn!("[comprehension] model load failed: {e}");
                None
            }
        };

    // Shared session gate — same flag detection uses (§2.4).
    let session_active = {
        let st = app.state::<Mutex<AppState>>();
        let guard = st.lock().unwrap();
        guard.session_active.clone()
    };

    while let Some(sentence) = rx.recv().await {
        if !session_active.load(Ordering::SeqCst) {
            continue;
        }
        let t = now_ms(base);

        // Ingest + decide-to-evaluate under the Observer lock (brief).
        let (should, prompt) = {
            let obs_state = app.state::<Mutex<Observer>>();
            let mut obs = obs_state.lock().unwrap();
            obs.ingest_segment(&sentence, t);
            // topic_shift wired in Task I5; None for now.
            if obs.should_evaluate(t, None) {
                (true, obs.build_prompt())
            } else {
                (false, String::new())
            }
        };
        if !should {
            continue;
        }

        // Inference happens OUTSIDE the lock (it can take seconds).
        #[cfg(feature = "local-comprehension")]
        let decision = match &model {
            Some(m) => {
                use rhema_comprehension::{ComprehensionModel, OutputSchema};
                match m.infer(&prompt, &OutputSchema::default()).await {
                    Ok(d) => Some(d),
                    Err(e) => {
                        log::warn!("[comprehension] infer error: {e}");
                        None
                    }
                }
            }
            None => None,
        };
        #[cfg(not(feature = "local-comprehension"))]
        let decision: Option<rhema_comprehension::Decision> = {
            let _ = &prompt;
            None
        };

        if let Some(decision) = decision {
            let transition = {
                let obs_state = app.state::<Mutex<Observer>>();
                let mut obs = obs_state.lock().unwrap();
                obs.apply_decision(decision, t)
            };
            if let Some(transition) = transition {
                let _ = app.emit(EVENT_COMPREHENSION_STATE, &transition);
            }
        }
    }
}
