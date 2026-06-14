use std::sync::Mutex;
use tauri::State;

use crate::epoch::EpochLock;
use crate::state::AppState;
use rhema_detection::{
    CursorMode, CursorState, MergedDetection, NavOutcome, PrimingIndex, ReadingMode, VersePosition,
};
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

/// A verse resolved by a navigation command, for the frontend to project.
#[derive(Clone, Serialize)]
pub struct NavVerse {
    pub book_number: i32,
    pub book_name: String,
    pub chapter: i32,
    pub verse: i32,
    pub text: String,
    pub reference: String,
}

/// Seed / update the formal navigation cursor (Phase 3 app-side follow-up).
/// Called by the frontend whenever a verse goes live, so next/previous-verse
/// navigate from the current live position. No-op when already on that verse
/// (so re-committing after a nav step doesn't reset history) or when the verse
/// has no resolvable book number (book_number < 1, e.g. a name-only suggestion).
#[tauri::command]
pub fn set_cursor_position(
    book_number: i32,
    chapter: i32,
    verse: i32,
    translation: String,
    state: State<'_, Mutex<AppState>>,
    reading: State<'_, Mutex<ReadingMode>>,
) -> Result<(), String> {
    if !(1..=66).contains(&book_number) {
        return Ok(());
    }
    let book = book_number as u8;
    let chapter = chapter.max(1) as u16;
    let verse = verse.max(1) as u16;

    // Manual navigation exits reading mode (ARCHITECTURE §6.8).
    exit_reading_mode(&reading);

    let mut app_state = state.lock().map_err(|e| e.to_string())?;
    if let Some(cursor) = &app_state.cursor {
        let p = cursor.position();
        if p.book == book && p.chapter == chapter && p.verse == verse && p.translation == translation
        {
            return Ok(());
        }
    }
    let position = match VersePosition::new(book, chapter, verse, translation, None) {
        Ok(p) => p,
        Err(_) => return Ok(()),
    };
    match &mut app_state.cursor {
        Some(cursor) => cursor.navigate_to(position, CursorMode::Single),
        None => app_state.cursor = Some(CursorState::new(position, CursorMode::Single)),
    }
    Ok(())
}

/// Advance the cursor to the next verse (Phase 3.4/3.5 bounds via BibleDb).
/// Returns the resolved verse for the frontend to project, or `None` on a Bible
/// boundary / cold cursor / lookup failure.
#[tauri::command]
pub fn next_verse(
    state: State<'_, Mutex<AppState>>,
    reading: State<'_, Mutex<ReadingMode>>,
) -> Result<Option<NavVerse>, String> {
    navigate(state, reading, true)
}

/// Step the cursor to the previous verse. See [`next_verse`].
#[tauri::command]
pub fn previous_verse(
    state: State<'_, Mutex<AppState>>,
    reading: State<'_, Mutex<ReadingMode>>,
) -> Result<Option<NavVerse>, String> {
    navigate(state, reading, false)
}

/// Deactivate reading mode if active — manual navigation exits it (§6.8).
fn exit_reading_mode(reading: &Mutex<ReadingMode>) {
    if let Ok(mut rm) = reading.lock() {
        if rm.is_active() {
            rm.deactivate();
        }
    }
}

fn navigate(
    state: State<'_, Mutex<AppState>>,
    reading: State<'_, Mutex<ReadingMode>>,
    forward: bool,
) -> Result<Option<NavVerse>, String> {
    use crate::nav_lookup::BibleDbVerseLookup;

    // Manual navigation exits reading mode (ARCHITECTURE §6.8).
    exit_reading_mode(&reading);

    let mut app_state = state.lock().map_err(|e| e.to_string())?;
    let AppState {
        cursor, bible_db, ..
    } = &mut *app_state;
    let (Some(cursor), Some(db)) = (cursor.as_mut(), bible_db.as_ref()) else {
        return Ok(None);
    };

    let lookup = BibleDbVerseLookup::new(db);
    let outcome = if forward {
        cursor.next_verse(&lookup)
    } else {
        cursor.previous_verse(&lookup)
    };
    if !matches!(outcome, NavOutcome::Moved) {
        return Ok(None);
    }

    let p = cursor.position().clone();
    let tid = db
        .list_translations()
        .map_err(|e| e.to_string())?
        .into_iter()
        .find(|t| t.abbreviation.eq_ignore_ascii_case(&p.translation))
        .map(|t| t.id);
    let Some(tid) = tid else {
        return Ok(None);
    };
    let resolved = db
        .get_verse(tid, p.book as i32, p.chapter as i32, p.verse as i32)
        .map_err(|e| e.to_string())?;
    Ok(resolved.map(|v| NavVerse {
        book_number: v.book_number,
        book_name: v.book_name.clone(),
        chapter: v.chapter,
        verse: v.verse,
        text: v.text,
        reference: format!("{} {}:{}", v.book_name, v.chapter, v.verse),
    }))
}

/// Set the pastor's pre-service sermon notes (Phase 5, Bullet 5.2). Parses the
/// notes for explicit verse references and stores a priming index in `AppState`;
/// those verses receive a +0.25 boost in later suggestion ranking. Returns the
/// number of primed verse coordinates (for an operator-facing confirmation).
/// Rebuilds the index on every call (pastor may edit notes mid-service).
#[tauri::command]
pub fn set_sermon_notes(notes: String, state: State<'_, Mutex<AppState>>) -> Result<usize, String> {
    let index = PrimingIndex::build_from_notes(&notes);
    let count = index.len();
    let mut app_state = state.lock().map_err(|e| e.to_string())?;
    app_state.priming_index = index;
    log::info!("priming: set_sermon_notes indexed {count} primed verse(s)");
    Ok(count)
}

/// Operator dismisses a proactive suggestion (Phase 5, Bullet 5.3). The verse is
/// suppressed from further suggestions for the rest of the service.
#[tauri::command]
pub fn dismiss_suggestion(
    book: String,
    chapter: i32,
    verse: i32,
    engine: State<'_, Mutex<crate::suggestion::SuggestionEngine>>,
) -> Result<(), String> {
    engine
        .lock()
        .map_err(|e| e.to_string())?
        .dismiss(&book, chapter, verse);
    log::info!("suggestion: dismissed {book} {chapter}:{verse} for the service");
    Ok(())
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
