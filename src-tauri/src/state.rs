use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use rhema_bible::BibleDb;
use rhema_detection::{CursorState, DetectionPipeline, PrimingIndex, QuotationMatcher, SermonContext};

use crate::suppression::SuppressionCache;

pub struct AppState {
    pub bible_db: Option<BibleDb>,
    pub detection_pipeline: DetectionPipeline,
    pub sermon_context: SermonContext,
    pub quotation_matcher: QuotationMatcher,
    /// 45s TTL cache that suppresses cyclic verse echoes (Phase 2, Bullet 2.4).
    pub suppression_cache: SuppressionCache,
    /// Pre-service priming index from the pastor's notes (Phase 5, Bullet 5.2).
    pub priming_index: PrimingIndex,
    /// Formal navigation cursor (Phase 3); `None` until a verse is set live.
    pub cursor: Option<CursorState>,
    pub active_translation_id: i64,
    pub audio_active: Arc<AtomicBool>,
    pub stt_active: Arc<AtomicBool>,
    #[allow(dead_code)]
    pub deepgram_api_key: Option<String>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            bible_db: None,
            detection_pipeline: DetectionPipeline::new(),
            sermon_context: SermonContext::new(),
            quotation_matcher: QuotationMatcher::new(),
            suppression_cache: SuppressionCache::new(),
            priming_index: PrimingIndex::default(),
            cursor: None,
            active_translation_id: 1, // Default to first translation (KJV)
            audio_active: Arc::new(AtomicBool::new(false)),
            stt_active: Arc::new(AtomicBool::new(false)),
            deepgram_api_key: None,
        }
    }
}
