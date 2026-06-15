mod channels;
mod commands;
mod epoch;
mod events;
mod nav_lookup;
mod state;
mod suggestion;
mod suppression;

use std::sync::Mutex;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Load .env file — try src-tauri/.env first, then project root ../.env
    dotenvy::dotenv().ok();
    dotenvy::from_filename("../.env").ok();
    tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::new()
                .level(tauri_plugin_log::log::LevelFilter::Info)
                .build(),
        )
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_store::Builder::new().build())
        .manage(Mutex::new(state::AppState::new()))
        .manage(epoch::EpochLock::default())
        .manage(Mutex::new(rhema_broadcast::ndi::NdiRuntime::default()))
        .manage(Mutex::new(rhema_detection::DirectDetector::new()))
        .manage(Mutex::new(rhema_detection::DetectionMerger::new()))
        .manage(Mutex::new(rhema_detection::ReadingMode::new()))
        .manage(Mutex::new(commands::obs::ObsOverlayServer::default()))
        .manage(Mutex::new(channels::ChannelState::default()))
        .manage(Mutex::new(suggestion::SuggestionEngine::new()))
        .manage(Mutex::new(Option::<rhema_api::llm::LlmConfig>::None))
        .invoke_handler(tauri::generate_handler![
            commands::bible::list_translations,
            commands::bible::list_books,
            commands::bible::get_chapter,
            commands::bible::get_verse,
            commands::bible::search_verses,
            commands::bible::get_translation_verses_for_search,
            commands::bible::get_cross_references,
            commands::bible::get_active_translation,
            commands::bible::set_active_translation,
            commands::detection::detect_verses,
            commands::detection::acquire_operator_lock,
            commands::detection::detection_status,
            commands::detection::semantic_search,
            commands::detection::toggle_paraphrase_detection,
            commands::detection::quotation_search,
            commands::detection::reading_mode_status,
            commands::detection::stop_reading_mode,
            commands::detection::set_sermon_notes,
            commands::detection::dismiss_suggestion,
            commands::detection::set_cursor_position,
            commands::detection::next_verse,
            commands::detection::previous_verse,
            commands::detection::go_to_reference,
            commands::detection::step_verses,
            commands::detection::start_session,
            commands::detection::end_session,
            commands::detection::session_status,
            commands::audio::get_audio_devices,
            commands::stt::start_transcription,
            commands::stt::stop_transcription,
            commands::broadcast::list_monitors,
            commands::broadcast::ensure_broadcast_window,
            commands::broadcast::open_broadcast_window,
            commands::broadcast::close_broadcast_window,
            commands::broadcast::start_ndi,
            commands::broadcast::stop_ndi,
            commands::broadcast::get_ndi_status,
            commands::broadcast::push_ndi_frame,
            commands::obs::start_obs_overlay,
            commands::obs::stop_obs_overlay,
            commands::obs::get_obs_overlay_status,
            commands::obs::push_obs_overlay,
            channels::set_routing_mode,
            channels::commit_live_verse,
            commands::llm::set_llm_config,
            commands::llm::llm_status,
            commands::llm::clear_llm_config,
        ])
        .setup(|app| {
            use tauri::Manager;

            // Drain the Stage-2 (LLM fallback) channel (Bullet L5). The detection
            // pipeline queues ambiguous transcripts; the worker classifies each
            // via the configured multi-provider LLM and routes scripture hits
            // back through direct detection. No-op until a provider is configured.
            {
                let stage2_rx = app
                    .state::<Mutex<state::AppState>>()
                    .lock()
                    .unwrap()
                    .detection_pipeline
                    .take_stage2_receiver();
                if let Some(rx) = stage2_rx {
                    let handle = app.handle().clone();
                    tauri::async_runtime::spawn(commands::stt::run_stage2_worker(handle, rx));
                }
            }

            // Phase 4 (Bullet 4.4): start the 2s device-health monitor. It
            // probes each output endpoint, updates the operator channel's
            // device_health on change, and resyncs the audience channel on
            // reconnect. Channel content state is never reset by a device drop.
            channels::spawn_device_health_monitor(app.handle().clone());

            // Try resource dir first (production), then dev fallback
            let db_path = app
                .path()
                .resource_dir()
                .map(|p| p.join("rhema.db"))
                .ok()
                .filter(|p| p.exists())
                .unwrap_or_else(|| {
                    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("../data/rhema.db")
                });

            if db_path.exists() {
                let bible_db = rhema_bible::BibleDb::open(&db_path)
                    .expect("Failed to open Bible database");

                // Build quotation matching index from all English verses
                log::info!("Building quotation matching index...");
                let quotation_matcher = match bible_db.load_all_verses_for_quotation(Some("en")) {
                    Ok(verses) => {
                        log::info!("Loaded {} English verses for quotation index", verses.len());
                        rhema_detection::QuotationMatcher::build(verses)
                    }
                    Err(e) => {
                        log::warn!("Failed to load verses for quotation index: {e}");
                        rhema_detection::QuotationMatcher::new()
                    }
                };

                let managed_state = app.state::<Mutex<state::AppState>>();
                let mut state = managed_state.lock().unwrap();
                state.bible_db = Some(bible_db);
                state.quotation_matcher = quotation_matcher;
                drop(state);
                log::info!("Bible database loaded from {:?}", db_path);
            } else {
                log::warn!("Bible database not found at {:?}", db_path);
            }

            // Try to load ONNX embedding model and pre-computed verse index
            // Prefer INT8 quantized model (~571MB) over FP32 (~2.4GB)
            let base_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
            let model_path = {
                let int8 = base_dir.join("models/qwen3-embedding-0.6b-int8/model_quantized.onnx");
                let fp32 = base_dir.join("models/qwen3-embedding-0.6b/model.onnx");
                if int8.exists() {
                    log::info!("Using INT8 quantized ONNX model");
                    int8
                } else if fp32.exists() {
                    log::info!("Using FP32 ONNX model (INT8 not found)");
                    fp32
                } else {
                    fp32
                }
            };
            let tokenizer_path = base_dir.join("models/qwen3-embedding-0.6b/tokenizer.json");
            let embeddings_path = base_dir.join("embeddings/kjv-qwen3-0.6b.bin");
            let ids_path = base_dir.join("embeddings/kjv-qwen3-0.6b-ids.bin");

            if model_path.exists() && tokenizer_path.exists() {
                use rhema_detection::semantic::embedder::TextEmbedder;
                use rhema_detection::semantic::index::VectorIndex;
                match rhema_detection::OnnxEmbedder::load(&model_path, &tokenizer_path) {
                    Ok(embedder) => {
                        log::info!("ONNX embedding model loaded");
                        let managed_state = app.state::<Mutex<state::AppState>>();
                        let mut state = managed_state.lock().unwrap();

                        // If pre-computed embeddings exist, load the vector index
                        if embeddings_path.exists() && ids_path.exists() {
                            let dim = embedder.dimension();
                            match rhema_detection::HnswVectorIndex::load(&embeddings_path, &ids_path, dim) {
                                Ok(index) => {
                                    log::info!("Verse embeddings loaded ({} vectors)", index.len());
                                    state.detection_pipeline.semantic =
                                        rhema_detection::SemanticDetector::new(
                                            Box::new(embedder),
                                            Box::new(index),
                                        );
                                }
                                Err(e) => {
                                    log::warn!("Failed to load verse embeddings: {e}");
                                }
                            }
                        } else {
                            log::info!("No pre-computed verse embeddings found. Run 'bun run export:verses' then the precompute binary.");
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to load ONNX model: {e}");
                    }
                }
            } else {
                log::info!("ONNX model not found. Semantic search disabled. Run 'bun run download:model' to download.");
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
