//! Background comprehension observer worker. Fed coalesced final sentences from
//! the STT detection worker; gated by `session_active`; emits state transitions.
//!
//! All model/inference code that touches llama.cpp is behind the
//! `local-comprehension` feature. Without the feature (or with no model
//! configured) the worker simply drains the feed channel and no events fire, so
//! the default build stays native-free and green.
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

use rhema_comprehension::Observer;

use crate::events::EVENT_COMPREHENSION_STATE;
use crate::state::AppState;

/// Managed holder for the sentence-feed sender (filled at setup). The STT
/// detection worker taps coalesced sentences into this channel.
pub struct ComprehensionFeed(pub Mutex<Option<tokio::sync::mpsc::Sender<String>>>);

/// Live STT queued-frame backlog (= `deepgram_tx.len()`), published by the STT fanout
/// thread and read by the comprehension worker's STT-backlog gate. `0` when idle. Only the
/// depth is shared — never a `Sender` clone, which would keep the audio channel alive and
/// break the engine's `Disconnected` detection.
pub struct SttBacklogGauge(pub Arc<AtomicUsize>);

impl Default for SttBacklogGauge {
    fn default() -> Self {
        Self(Arc::new(AtomicUsize::new(0)))
    }
}

/// Runtime-tunable comprehension controls pushed from Settings (I4). The enable
/// gate is read by the worker each tick (the interval lives on the managed
/// `Observer` instead, since the worker already locks it every tick).
pub struct ComprehensionRuntime {
    pub enabled: AtomicBool,
}

impl Default for ComprehensionRuntime {
    fn default() -> Self {
        // On by default; without the `local-comprehension` feature (or a model)
        // this is inert anyway — the worker just drains the feed.
        Self { enabled: AtomicBool::new(true) }
    }
}

/// Where the user's chosen comprehension model path is persisted so it survives
/// restarts and is readable by the backend at startup (the model loads once, like
/// Whisper — §18). A plain UTF-8 file in the app config dir; absent = use the
/// `RHEMA_COMPREHENSION_MODEL` env default.
fn model_pref_path(app: &AppHandle) -> Option<PathBuf> {
    app.path()
        .app_config_dir()
        .ok()
        .map(|d| d.join("comprehension_model.txt"))
}

/// The persisted model path, if one is set and still exists on disk. Only read
/// by the feature-gated model loader at startup.
#[cfg(feature = "local-comprehension")]
fn persisted_model_path(app: &AppHandle) -> Option<PathBuf> {
    let pref = model_pref_path(app)?;
    let raw = std::fs::read_to_string(&pref).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let p = PathBuf::from(trimmed);
    p.exists().then_some(p)
}

/// Millisecond monotonic clock for the Observer's window math (a fixed baseline
/// captured at worker start; only deltas matter).
fn now_ms(base: std::time::Instant) -> u64 {
    base.elapsed().as_millis() as u64
}

/// Cosine *distance* (`1 - cosine_similarity`) between two topic vectors, in
/// `[0, 2]`. Used to detect a sharp topic shift (I5). Guards defensively: a
/// dimension mismatch or a zero-norm vector returns `0.0` (= no shift), so a
/// degenerate reading never fires a spurious early evaluation.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    1.0 - dot / (na.sqrt() * nb.sqrt())
}

/// Whether a due comprehension tick should run now, given STT load (STT-backlog gate).
/// `allow(dead_code)`: wired into `run_comprehension_worker` in the next bullet (the gate
/// consumer); until then only the unit tests exercise it.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunDecision {
    /// STT is caught up (or the starvation valve fired) — run the inference tick.
    Run,
    /// STT is behind — defer this tick; it is re-offered on the next final sentence.
    Skip,
    /// Not due this tick — do nothing.
    Idle,
}

/// Pure gate decision for a comprehension tick (I6 freeze fix). `due` is
/// `Observer::should_evaluate`; `backlog` is the queued STT frame depth; `max` is the
/// skip threshold (inclusive — `backlog <= max` runs); `deferred_since` is the timestamp
/// of the first consecutive skip (`None` = not deferring); `force_ms` is the starvation
/// valve ceiling. Returns `(decision, next_deferred_since)`.
///
/// A `Skip` loses nothing: `should_evaluate` is a pure query and `last_eval_ms` only
/// advances in `apply_decision`, so the tick is re-checked against a fresh backlog
/// reading on the next sentence and resumes the moment STT drains below `max`.
#[allow(dead_code)] // consumed by run_comprehension_worker in the next bullet
fn decide_run(
    due: bool,
    backlog: usize,
    max: usize,
    now_ms: u64,
    deferred_since: Option<u64>,
    force_ms: u64,
) -> (RunDecision, Option<u64>) {
    if !due {
        return (RunDecision::Idle, None);
    }
    if backlog <= max {
        return (RunDecision::Run, None);
    }
    match deferred_since {
        None => (RunDecision::Skip, Some(now_ms)),
        Some(since) => {
            if now_ms.saturating_sub(since) >= force_ms {
                (RunDecision::Run, None) // starvation valve: force a run, reset timer
            } else {
                (RunDecision::Skip, Some(since)) // keep deferring; preserve original timer
            }
        }
    }
}

/// Load the local model once at worker startup: prefer the user's persisted
/// choice (Settings, §I4), else fall back to `from_env()` (the
/// `RHEMA_COMPREHENSION_MODEL` default). A persisted path uses `LlamaConfig`
/// defaults; the env path also honours `_NCTX`/`_THREADS` overrides. Returns
/// `None` when neither is configured (comprehension simply stays idle).
#[cfg(feature = "local-comprehension")]
fn load_configured_model(
    app: &AppHandle,
) -> Option<rhema_comprehension_llama::LlamaComprehensionModel> {
    use rhema_comprehension_llama::{LlamaComprehensionModel, LlamaConfig};
    if let Some(p) = persisted_model_path(app) {
        match LlamaComprehensionModel::load(&p, LlamaConfig::default()) {
            Ok(m) => {
                log::info!("[comprehension] loaded model from settings: {}", p.display());
                return Some(m);
            }
            Err(e) => {
                log::warn!(
                    "[comprehension] settings model load failed ({}): {e}; trying env default",
                    p.display()
                );
            }
        }
    }
    match LlamaComprehensionModel::from_env() {
        Ok(m) => m,
        Err(e) => {
            log::warn!("[comprehension] model load failed: {e}");
            None
        }
    }
}

pub async fn run_comprehension_worker(
    app: AppHandle,
    mut rx: tokio::sync::mpsc::Receiver<String>,
) {
    let base = std::time::Instant::now();

    // Load the local model once (feature-gated). Without the feature, or with no
    // model configured, there is no model and the worker just drains the channel.
    // Wrapped in an Arc so each inference can be handed to a blocking thread.
    #[cfg(feature = "local-comprehension")]
    let model = load_configured_model(&app).map(std::sync::Arc::new);

    // Shared session gate — same flag detection uses (§2.4).
    let session_active = {
        let st = app.state::<Mutex<AppState>>();
        let guard = st.lock().unwrap();
        guard.session_active.clone()
    };

    // Previous topic centroid for the optional topic-shift early trigger (I5, D5).
    // Only consulted when the Observer's config has `topic_shift_trigger` enabled.
    let mut prev_topic: Option<Vec<f32>> = None;

    while let Some(sentence) = rx.recv().await {
        if !session_active.load(Ordering::SeqCst) {
            continue;
        }
        // Enable gate (I4): the user can turn comprehension off without a restart.
        if !app
            .state::<ComprehensionRuntime>()
            .enabled
            .load(Ordering::Relaxed)
        {
            continue;
        }
        let t = now_ms(base);

        // Topic-shift early trigger (I5, D5). Read the current topic centroid
        // (reusing the Phase-5.1 vector detection already maintains) and measure
        // how far it moved. Fail open: if AppState is contended, skip the shift
        // signal this tick (never block detection — mirrors stt.rs:1104). The
        // Observer only acts on this when `topic_shift_trigger` is enabled.
        let topic_shift: Option<f32> = {
            let st = app.state::<Mutex<AppState>>();
            let cur = st
                .try_lock()
                .ok()
                .and_then(|mut g| g.sermon_context.topic_vector());
            match cur {
                Some(c) => {
                    let shift = prev_topic.as_ref().map(|p| cosine_distance(p, &c));
                    prev_topic = Some(c);
                    shift
                }
                None => None,
            }
        };

        // Ingest + decide-to-evaluate under the Observer lock (brief).
        let (should, prompt) = {
            let obs_state = app.state::<Mutex<Observer>>();
            let mut obs = obs_state.lock().unwrap();
            obs.ingest_segment(&sentence, t);
            if obs.should_evaluate(t, topic_shift) {
                (true, obs.build_prompt())
            } else {
                (false, String::new())
            }
        };
        if !should {
            continue;
        }

        // Inference happens OUTSIDE the lock (it can take seconds) AND on a
        // blocking thread (spawn_blocking) so it never stalls the async runtime.
        // Combined with the capped thread count in LlamaConfig, this keeps the
        // realtime STT decode from being starved during a comprehension call (I6
        // lag fix).
        #[cfg(feature = "local-comprehension")]
        let decision = match &model {
            Some(m) => {
                let m = std::sync::Arc::clone(m);
                match tokio::task::spawn_blocking(move || {
                    m.infer_blocking(&prompt, &rhema_comprehension::OutputSchema::default())
                })
                .await
                {
                    Ok(Ok(d)) => Some(d),
                    Ok(Err(e)) => {
                        log::warn!("[comprehension] infer error: {e}");
                        None
                    }
                    Err(e) => {
                        log::warn!("[comprehension] infer task failed: {e}");
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

/// A local comprehension GGUF the user may select (I4).
#[derive(Serialize)]
pub struct ComprehensionModelInfo {
    pub path: String,
    pub label: String,
}

/// Push the enable gate + refresh interval to the running observer (I4). Applies
/// live: the enable flag is read each tick and the interval takes effect on the
/// next evaluation. Model selection is separate (`set_comprehension_model`) since
/// the model loads once at startup.
#[tauri::command]
pub fn set_comprehension_config(
    enabled: bool,
    interval_ms: u64,
    observer: State<'_, Mutex<Observer>>,
    runtime: State<'_, ComprehensionRuntime>,
) -> Result<(), String> {
    runtime.enabled.store(enabled, Ordering::Relaxed);
    if let Ok(mut obs) = observer.lock() {
        obs.set_interval(interval_ms);
    }
    log::info!("comprehension: enabled={enabled} interval_ms={interval_ms}");
    Ok(())
}

/// Persist the user's chosen comprehension model path (I4). Takes effect on the
/// next app launch — the model loads once at startup (§18). `None`/empty clears
/// the choice, reverting to the `RHEMA_COMPREHENSION_MODEL` env default.
#[tauri::command]
pub fn set_comprehension_model(app: AppHandle, path: Option<String>) -> Result<(), String> {
    let pref = model_pref_path(&app).ok_or("no app config dir")?;
    let chosen = path.map(|p| p.trim().to_string()).filter(|p| !p.is_empty());
    match chosen {
        Some(p) => {
            if let Some(parent) = pref.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            std::fs::write(&pref, p.as_bytes()).map_err(|e| e.to_string())?;
            log::info!("comprehension: model set to {p} (applies on next launch)");
        }
        None => {
            let _ = std::fs::remove_file(&pref);
            log::info!("comprehension: model choice cleared (env default on next launch)");
        }
    }
    Ok(())
}

/// List local comprehension GGUFs the user may select (I4). Scans the gitignored
/// `model/` dir, excluding on-device STT models (which share the dir). Returns
/// `[]` without the `local-comprehension` feature so the Settings dropdown hides.
#[tauri::command]
pub fn list_comprehension_models() -> Vec<ComprehensionModelInfo> {
    #[cfg(not(feature = "local-comprehension"))]
    {
        Vec::new()
    }
    #[cfg(feature = "local-comprehension")]
    {
        // STT GGUFs live in the same dir; skip them so this picker shows only
        // comprehension LLMs (heuristic on well-known STT family markers).
        const STT_MARKERS: [&str; 4] = ["nemotron", "parakeet", "cohere", "streaming"];
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../model");
        let mut out: Vec<ComprehensionModelInfo> = match std::fs::read_dir(&dir) {
            Ok(entries) => entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("gguf"))
                .filter_map(|p| {
                    let file = p.file_name().and_then(|f| f.to_str())?.to_string();
                    let lower = file.to_lowercase();
                    if STT_MARKERS.iter().any(|m| lower.contains(m)) {
                        return None;
                    }
                    Some(ComprehensionModelInfo {
                        label: file.trim_end_matches(".gguf").to_string(),
                        path: p.to_string_lossy().into_owned(),
                    })
                })
                .collect(),
            Err(e) => {
                log::warn!(
                    "[comprehension] list models: cannot read {}: {e}",
                    dir.display()
                );
                Vec::new()
            }
        };
        out.sort_by(|a, b| a.label.cmp(&b.label));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{cosine_distance, decide_run, RunDecision};

    #[test]
    fn not_due_is_idle_and_clears_timer() {
        assert_eq!(
            decide_run(false, 999, 256, 1000, Some(500), 180_000),
            (RunDecision::Idle, None)
        );
    }

    #[test]
    fn due_and_caught_up_runs_and_clears_timer() {
        assert_eq!(
            decide_run(true, 24, 256, 1000, None, 180_000),
            (RunDecision::Run, None)
        );
    }

    #[test]
    fn backlog_equal_to_max_runs_inclusive() {
        assert_eq!(
            decide_run(true, 256, 256, 1000, None, 180_000),
            (RunDecision::Run, None)
        );
    }

    #[test]
    fn due_and_behind_first_time_skips_and_sets_timer() {
        assert_eq!(
            decide_run(true, 492, 256, 1000, None, 180_000),
            (RunDecision::Skip, Some(1000))
        );
    }

    #[test]
    fn due_and_behind_within_ceiling_keeps_original_timer() {
        // deferred at 1000, now 60_000, ceiling 180_000 -> still skipping, timer unchanged
        assert_eq!(
            decide_run(true, 492, 256, 60_000, Some(1000), 180_000),
            (RunDecision::Skip, Some(1000))
        );
    }

    #[test]
    fn due_and_behind_past_ceiling_forces_run_and_clears_timer() {
        // deferred at 1000, now 181_001, ceiling 180_000 -> valve fires
        assert_eq!(
            decide_run(true, 492, 256, 181_001, Some(1000), 180_000),
            (RunDecision::Run, None)
        );
    }

    #[test]
    fn identical_vectors_have_zero_distance() {
        let a = [1.0, 2.0, 3.0];
        assert!(cosine_distance(&a, &a).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_vectors_have_distance_one() {
        assert!((cosine_distance(&[1.0, 0.0], &[0.0, 1.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn opposite_vectors_have_distance_two() {
        assert!((cosine_distance(&[1.0, 0.0], &[-1.0, 0.0]) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn dimension_mismatch_is_no_shift() {
        assert_eq!(cosine_distance(&[1.0, 2.0], &[1.0]), 0.0);
    }

    #[test]
    fn zero_norm_is_no_shift() {
        assert_eq!(cosine_distance(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }
}
