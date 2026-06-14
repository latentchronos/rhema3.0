use std::sync::Mutex;
use tauri::State;

use crate::epoch::EpochLock;
use crate::state::AppState;
use rhema_detection::{MergedDetection, ReadingMode};
use serde::Serialize;

/// Serializable detection result for the frontend
#[derive(Clone, Serialize)]
pub struct DetectionResult {
    pub verse_ref: String,
    pub verse_text: String,
    pub book_name: String,
    pub book_number: i32,
    pub chapter: i32,
    pub verse: i32,
    pub confidence: f64,
    pub source: String,
    pub auto_queued: bool,
    pub raw_score: f64,
    pub minimum_threshold: f64,
    pub auto_queue_threshold: f64,
    pub decision: String,
    pub explanation: String,
    pub transcript_snippet: String,
}

fn source_to_string(source: &rhema_detection::DetectionSource) -> String {
    match source {
        rhema_detection::DetectionSource::DirectReference => "direct".to_string(),
        rhema_detection::DetectionSource::Contextual => "contextual".to_string(),
        rhema_detection::DetectionSource::QuotationMatch { .. } => "quotation".to_string(),
        rhema_detection::DetectionSource::SemanticLocal { .. } => "semantic_local".to_string(),
        rhema_detection::DetectionSource::SemanticCloud { .. } => "semantic_cloud".to_string(),
    }
}

pub fn to_result(state: &AppState, merged: &MergedDetection) -> DetectionResult {
    let vr = &merged.detection.verse_ref;
    let vid = merged.detection.verse_id;

    // Resolve verse info: try verse_id first (semantic), then book/chapter/verse (direct)
    let (reference, verse_text, book_name, book_number, chapter, verse) =
        if let (Some(id), Some(ref db)) = (vid, &state.bible_db) {
            // Semantic detection: resolve via DB primary key
            if let Ok(Some(v)) = db.get_verse_by_id(id) {
                let r = format!("{} {}:{}", v.book_name, v.chapter, v.verse);
                (r, v.text, v.book_name, v.book_number, v.chapter, v.verse)
            } else {
                let r = format!("{} {}:{}", vr.book_name, vr.chapter, vr.verse_start);
                (
                    r,
                    String::new(),
                    vr.book_name.clone(),
                    vr.book_number,
                    vr.chapter,
                    vr.verse_start,
                )
            }
        } else if let Some(ref db) = state.bible_db {
            // Direct detection: resolve via book/chapter/verse
            if vr.book_number > 0 && vr.chapter > 0 && vr.verse_start > 0 {
                if let Ok(Some(v)) = db.get_verse(
                    state.active_translation_id,
                    vr.book_number,
                    vr.chapter,
                    vr.verse_start,
                ) {
                    let r = format!("{} {}:{}", v.book_name, v.chapter, v.verse);
                    (r, v.text, v.book_name, v.book_number, v.chapter, v.verse)
                } else {
                    let r = format!("{} {}:{}", vr.book_name, vr.chapter, vr.verse_start);
                    (
                        r,
                        String::new(),
                        vr.book_name.clone(),
                        vr.book_number,
                        vr.chapter,
                        vr.verse_start,
                    )
                }
            } else {
                let r = format!("{} {}:{}", vr.book_name, vr.chapter, vr.verse_start);
                (
                    r,
                    String::new(),
                    vr.book_name.clone(),
                    vr.book_number,
                    vr.chapter,
                    vr.verse_start,
                )
            }
        } else {
            let r = format!("{} {}:{}", vr.book_name, vr.chapter, vr.verse_start);
            (
                r,
                String::new(),
                vr.book_name.clone(),
                vr.book_number,
                vr.chapter,
                vr.verse_start,
            )
        };

    DetectionResult {
        verse_ref: reference,
        verse_text,
        book_name,
        book_number,
        chapter,
        verse,
        confidence: merged.detection.confidence,
        source: source_to_string(&merged.detection.source),
        auto_queued: merged.auto_queued,
        raw_score: merged.decision.raw_score,
        minimum_threshold: merged.decision.minimum_threshold,
        auto_queue_threshold: merged.decision.auto_queue_threshold,
        decision: merged.decision.decision.to_string(),
        explanation: merged.decision.explanation.clone(),
        transcript_snippet: merged.detection.transcript_snippet.clone(),
    }
}

pub fn live_detection_metadata(
    source: &str,
    confidence: f64,
    auto_queued: bool,
) -> (f64, f64, f64, String, String) {
    let (minimum_threshold, auto_queue_threshold, label) = match source {
        "direct" => (0.45, 0.90, "direct reference"),
        "contextual" => (0.45, 0.80, "reading context"),
        "quotation" => (0.45, 0.85, "quotation match"),
        "semantic_cloud" => (0.55, 0.90, "cloud semantic search"),
        _ => (0.50, 0.92, "local semantic search"),
    };
    let decision = if auto_queued {
        "auto_queued"
    } else {
        "review_required"
    };
    let explanation = if auto_queued {
        format!(
            "{label} confidence {:.0}% met the {:.0}% auto-queue threshold.",
            confidence * 100.0,
            auto_queue_threshold * 100.0,
        )
    } else {
        format!(
            "{label} confidence {:.0}% is below the {:.0}% auto-queue threshold; operator review required.",
            confidence * 100.0,
            auto_queue_threshold * 100.0,
        )
    };

    (
        confidence,
        minimum_threshold,
        auto_queue_threshold,
        decision.to_string(),
        explanation,
    )
}

/// Register an operator manual action (Phase 3, Bullet 3.2): bump the epoch
/// lock so in-flight voice detections are discarded for the lock window.
///
/// The frontend should call this whenever the operator manually picks,
/// projects, or navigates a verse, so the operator's choice wins over a voice
/// detection that arrives a few hundred ms later. Returns the new epoch.
#[tauri::command]
pub fn acquire_operator_lock(epoch: State<'_, EpochLock>) -> u64 {
    epoch.acquire()
}

/// Run the detection pipeline on a piece of transcript text
#[tauri::command]
pub fn detect_verses(
    state: State<'_, Mutex<AppState>>,
    text: String,
) -> Result<Vec<DetectionResult>, String> {
    let mut app_state = state.lock().map_err(|e| e.to_string())?;
    let merged = app_state.detection_pipeline.process(&text);
    let results: Vec<DetectionResult> = merged.iter().map(|m| to_result(&app_state, m)).collect();
    Ok(results)
}

/// Check if semantic search is available
#[tauri::command]
pub fn detection_status(
    state: State<'_, Mutex<AppState>>,
) -> Result<DetectionStatusResult, String> {
    let app_state = state.lock().map_err(|e| e.to_string())?;
    Ok(DetectionStatusResult {
        has_direct: true,
        has_semantic: app_state.detection_pipeline.has_semantic(),
        has_cloud: app_state.detection_pipeline.has_cloud(),
        paraphrase_enabled: app_state.detection_pipeline.use_synonyms(),
    })
}

/// Toggle paraphrase detection (synonym expansion) on/off
#[tauri::command]
pub fn toggle_paraphrase_detection(
    state: State<'_, Mutex<AppState>>,
    enabled: bool,
) -> Result<bool, String> {
    let mut app_state = state.lock().map_err(|e| e.to_string())?;
    app_state.detection_pipeline.set_use_synonyms(enabled);
    log::info!("[DET] Paraphrase detection (synonyms) set to: {enabled}");
    Ok(enabled)
}

#[derive(Serialize)]
pub struct DetectionStatusResult {
    pub has_direct: bool,
    pub has_semantic: bool,
    pub has_cloud: bool,
    pub paraphrase_enabled: bool,
}

#[derive(Serialize)]
pub struct SemanticSearchResult {
    pub verse_ref: String,
    pub verse_text: String,
    pub book_name: String,
    pub book_number: i32,
    pub chapter: i32,
    pub verse: i32,
    pub similarity: f64,
}

#[tauri::command]
pub fn semantic_search(
    state: State<'_, Mutex<AppState>>,
    query: String,
    limit: Option<usize>,
) -> Result<Vec<SemanticSearchResult>, String> {
    let k = limit.unwrap_or(10);
    let mut app_state = state.lock().map_err(|e| e.to_string())?;

    if !app_state.detection_pipeline.has_semantic() {
        return Err("Semantic search not available — model or embeddings not loaded".into());
    }

    let hits = app_state
        .detection_pipeline
        .semantic
        .search_query(&query, k);

    let mut results: Vec<SemanticSearchResult> = hits
        .into_iter()
        .filter_map(|(verse_id, similarity)| {
            if let Some(ref db) = app_state.bible_db {
                if let Ok(Some(v)) = db.get_verse_by_id(verse_id) {
                    return Some(SemanticSearchResult {
                        verse_ref: format!("{} {}:{}", v.book_name, v.chapter, v.verse),
                        verse_text: v.text,
                        book_name: v.book_name,
                        book_number: v.book_number,
                        chapter: v.chapter,
                        verse: v.verse,
                        similarity,
                    });
                }
            }
            None
        })
        .collect();

    // Ensure highest similarity is always first
    results.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(results)
}

/// Search for verses using quotation matching (word overlap).
/// Used by the context search tab alongside semantic search.
#[tauri::command]
pub fn quotation_search(
    state: State<'_, Mutex<AppState>>,
    query: String,
    limit: Option<usize>,
) -> Result<Vec<QuotationSearchResult>, String> {
    let k = limit.unwrap_or(10);
    let app_state = state.lock().map_err(|e| e.to_string())?;

    if !app_state.quotation_matcher.is_ready() {
        return Ok(vec![]);
    }

    let detections = app_state.quotation_matcher.match_transcript(&query);

    let results: Vec<QuotationSearchResult> = detections
        .into_iter()
        .take(k)
        .map(|d| {
            let vr = &d.verse_ref;
            let verse_text = if let Some(ref db) = app_state.bible_db {
                db.get_verse(
                    app_state.active_translation_id,
                    vr.book_number,
                    vr.chapter,
                    vr.verse_start,
                )
                .ok()
                .flatten()
                .map(|v| v.text)
                .unwrap_or_default()
            } else {
                String::new()
            };

            QuotationSearchResult {
                verse_ref: format!("{} {}:{}", vr.book_name, vr.chapter, vr.verse_start),
                verse_text,
                book_name: vr.book_name.clone(),
                book_number: vr.book_number,
                chapter: vr.chapter,
                verse: vr.verse_start,
                similarity: d.confidence,
            }
        })
        .collect();

    Ok(results)
}

#[derive(Serialize)]
pub struct QuotationSearchResult {
    pub verse_ref: String,
    pub verse_text: String,
    pub book_name: String,
    pub book_number: i32,
    pub chapter: i32,
    pub verse: i32,
    pub similarity: f64,
}

/// Get reading mode status
#[tauri::command]
pub fn reading_mode_status(
    state: State<'_, Mutex<ReadingMode>>,
) -> Result<ReadingModeStatus, String> {
    let rm = state.lock().map_err(|e| e.to_string())?;
    Ok(ReadingModeStatus {
        active: rm.is_active(),
        current_verse: rm.current_verse(),
    })
}

#[derive(Serialize)]
pub struct ReadingModeStatus {
    pub active: bool,
    pub current_verse: Option<i32>,
}

/// Stop reading mode
#[tauri::command]
pub fn stop_reading_mode(state: State<'_, Mutex<ReadingMode>>) -> Result<(), String> {
    let mut rm = state.lock().map_err(|e| e.to_string())?;
    rm.deactivate();
    Ok(())
}
