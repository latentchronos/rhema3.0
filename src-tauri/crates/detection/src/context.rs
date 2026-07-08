use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::types::VerseRef;

/// How long context remains valid (3 minutes, matching Logos AI).
const CONTEXT_TIMEOUT_SECS: u64 = 180;

/// Active-window span for the topic vector (Bullet 5.1). A sentence exactly this
/// old sits on the 0.75/0.25 block boundary — chosen so the two-block weighting
/// approximates an exponential decay with a 90s half-life (a 90s-old sentence
/// carries ~half the weight of a current one).
const ACTIVE_WINDOW_SECS: u64 = 90;

/// Weight on the active (last-90s) block of the topic vector.
const ACTIVE_WEIGHT: f32 = 0.75;

/// Weight on the background (older-than-90s) block of the topic vector.
const BACKGROUND_WEIGHT: f32 = 0.25;

/// Maximum number of embeddings retained in the background block. When exceeded,
/// the oldest are dropped — the topic vector degrades gracefully (§9.3).
const BACKGROUND_WINDOW_CAP: usize = 500;

/// Confidence boost for detections in the same book as the current context.
pub const SAME_BOOK_BOOST: f64 = 0.05;

/// Confidence boost for detections in the same chapter as the current context.
pub const SAME_CHAPTER_BOOST: f64 = 0.10;

/// A timestamped detection entry in the session history.
#[derive(Debug, Clone)]
pub struct SessionEntry {
    pub timestamp_ms: u64,
    pub verse_ref: VerseRef,
    pub confidence: f64,
    pub source: String,
}

/// Tracks the sermon context — current book/chapter focus, session history,
/// and provides confidence boosting for contextually relevant detections.
///
/// Logos AI maintains context for approximately 3 minutes of sermon audio.
/// Context is refreshed on each new explicit reference.
pub struct SermonContext {
    /// The currently focused book number (from most recent detection).
    current_book: Option<i32>,
    /// The currently focused chapter (from most recent detection).
    current_chapter: Option<i32>,
    /// When the context was last updated.
    last_update: Option<Instant>,
    /// History of all detected verses this session.
    session_history: Vec<SessionEntry>,
    /// Sentence embeddings from the last [`ACTIVE_WINDOW_SECS`] (Bullet 5.1).
    active_window: VecDeque<(Vec<f32>, Instant)>,
    /// Sentence embeddings older than the active window (capped FIFO).
    background_window: VecDeque<(Vec<f32>, Instant)>,
    /// Lazily computed topic vector, recomputed when `topic_dirty` is set.
    topic_vector: Option<Vec<f32>>,
    /// True when the windows have changed since the topic vector was computed.
    topic_dirty: bool,
}

impl SermonContext {
    pub fn new() -> Self {
        Self {
            current_book: None,
            current_chapter: None,
            last_update: None,
            session_history: Vec::new(),
            active_window: VecDeque::new(),
            background_window: VecDeque::new(),
            topic_vector: None,
            topic_dirty: false,
        }
    }

    /// Check if context is still valid (within timeout).
    pub fn is_valid(&self) -> bool {
        match self.last_update {
            Some(ts) => ts.elapsed().as_secs() < CONTEXT_TIMEOUT_SECS,
            None => false,
        }
    }

    /// Update context with a new detection.
    pub fn update(&mut self, verse_ref: &VerseRef, confidence: f64, source: &str) {
        if verse_ref.book_number > 0 {
            self.current_book = Some(verse_ref.book_number);
        }
        if verse_ref.chapter > 0 {
            self.current_chapter = Some(verse_ref.chapter);
        }
        self.last_update = Some(Instant::now());

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.session_history.push(SessionEntry {
            timestamp_ms: now,
            verse_ref: verse_ref.clone(),
            confidence,
            source: source.to_string(),
        });
    }

    /// Get the current book number in focus (if context is valid).
    pub fn current_book(&self) -> Option<i32> {
        if self.is_valid() {
            self.current_book
        } else {
            None
        }
    }

    /// Get the current chapter in focus (if context is valid).
    pub fn current_chapter(&self) -> Option<i32> {
        if self.is_valid() {
            self.current_chapter
        } else {
            None
        }
    }

    /// Calculate confidence boost for a detection based on current context.
    ///
    /// - Same book: +0.05
    /// - Same book AND chapter: +0.10 (replaces book boost, not additive)
    pub fn confidence_boost(&self, book_number: i32, chapter: i32) -> f64 {
        if !self.is_valid() {
            return 0.0;
        }

        if let Some(ctx_book) = self.current_book {
            if ctx_book == book_number {
                if let Some(ctx_chapter) = self.current_chapter {
                    if ctx_chapter == chapter {
                        return SAME_CHAPTER_BOOST;
                    }
                }
                return SAME_BOOK_BOOST;
            }
        }

        0.0
    }

    /// Ingest one sentence embedding into the topic-vector windows (Bullet 5.1).
    /// Embeddings are supplied pre-computed (by the app consumer via the semantic
    /// model) — `SermonContext` stays model-free.
    pub fn ingest_embedding(&mut self, embedding: Vec<f32>) {
        self.ingest_embedding_at(embedding, Instant::now());
    }

    /// Time-injected core of [`ingest_embedding`] for deterministic tests.
    pub fn ingest_embedding_at(&mut self, embedding: Vec<f32>, now: Instant) {
        self.active_window.push_back((embedding, now));

        // Drain active entries older than the 90s window into the background block.
        if let Some(cutoff) = now.checked_sub(Duration::from_secs(ACTIVE_WINDOW_SECS)) {
            while let Some((_, ts)) = self.active_window.front() {
                if *ts < cutoff {
                    if let Some(entry) = self.active_window.pop_front() {
                        self.background_window.push_back(entry);
                    }
                } else {
                    break;
                }
            }
        }

        // Cap the background block, dropping the oldest entries.
        while self.background_window.len() > BACKGROUND_WINDOW_CAP {
            self.background_window.pop_front();
        }

        self.topic_dirty = true;
    }

    /// The current topic vector (lazy 75/25 two-block compute, Bullet 5.1).
    /// `None` when no sentences have been ingested. Cached until the windows change.
    pub fn topic_vector(&mut self) -> Option<Vec<f32>> {
        if !self.topic_dirty {
            if let Some(v) = &self.topic_vector {
                return Some(v.clone());
            }
        }
        let computed = self.compute_topic_vector();
        self.topic_vector = computed.clone();
        self.topic_dirty = false;
        computed
    }

    fn compute_topic_vector(&self) -> Option<Vec<f32>> {
        let active_mean = mean_embedding(self.active_window.iter().map(|(e, _)| e))?;
        let bg_mean = match mean_embedding(self.background_window.iter().map(|(e, _)| e)) {
            Some(m) if m.len() == active_mean.len() => m,
            // Empty or dimension-mismatched background → active block only.
            _ => return Some(active_mean),
        };
        let topic = active_mean
            .iter()
            .zip(bg_mean.iter())
            .map(|(a, b)| ACTIVE_WEIGHT * a + BACKGROUND_WEIGHT * b)
            .collect();
        Some(topic)
    }

    /// Number of embeddings in the active block (test/inspection helper).
    pub fn active_window_len(&self) -> usize {
        self.active_window.len()
    }

    /// Number of embeddings in the background block (test/inspection helper).
    pub fn background_window_len(&self) -> usize {
        self.background_window.len()
    }

    /// Get the full session history.
    pub fn history(&self) -> &[SessionEntry] {
        &self.session_history
    }

    /// Clear session history (e.g., when starting a new service).
    pub fn clear_session(&mut self) {
        self.session_history.clear();
        self.current_book = None;
        self.current_chapter = None;
        self.last_update = None;
        self.active_window.clear();
        self.background_window.clear();
        self.topic_vector = None;
        self.topic_dirty = false;
    }

    /// Find the most recent detection for a given book (for "back in Genesis" pattern).
    pub fn last_in_book(&self, book_number: i32) -> Option<&VerseRef> {
        self.session_history
            .iter()
            .rev()
            .find(|e| e.verse_ref.book_number == book_number)
            .map(|e| &e.verse_ref)
    }
}

impl Default for SermonContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Element-wise mean of a set of equal-length embeddings. Returns `None` for an
/// empty set; embeddings whose length differs from the first are skipped.
fn mean_embedding<'a, I: Iterator<Item = &'a Vec<f32>>>(embeddings: I) -> Option<Vec<f32>> {
    let mut sum: Vec<f32> = Vec::new();
    let mut count = 0usize;
    for e in embeddings {
        if sum.is_empty() {
            sum = vec![0.0; e.len()];
        }
        if e.len() != sum.len() {
            continue;
        }
        for (s, v) in sum.iter_mut().zip(e.iter()) {
            *s += v;
        }
        count += 1;
    }
    if count == 0 {
        return None;
    }
    let inv = 1.0 / count as f32;
    for s in sum.iter_mut() {
        *s *= inv;
    }
    Some(sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ref(book: i32, chapter: i32, verse: i32) -> VerseRef {
        VerseRef {
            book_number: book,
            book_name: "Test".to_string(),
            chapter,
            verse_start: verse,
            verse_end: None,
        }
    }

    #[test]
    fn test_new_context_not_valid() {
        let ctx = SermonContext::new();
        assert!(!ctx.is_valid());
        assert_eq!(ctx.confidence_boost(1, 1), 0.0);
    }

    #[test]
    fn test_update_makes_valid() {
        let mut ctx = SermonContext::new();
        ctx.update(&make_ref(45, 8, 28), 0.95, "direct");
        assert!(ctx.is_valid());
        assert_eq!(ctx.current_book(), Some(45));
        assert_eq!(ctx.current_chapter(), Some(8));
    }

    #[test]
    fn test_same_book_boost() {
        let mut ctx = SermonContext::new();
        ctx.update(&make_ref(45, 8, 28), 0.95, "direct"); // Romans 8:28
        // Same book (Romans), different chapter
        assert!((ctx.confidence_boost(45, 3) - SAME_BOOK_BOOST).abs() < f64::EPSILON);
    }

    #[test]
    fn test_same_chapter_boost() {
        let mut ctx = SermonContext::new();
        ctx.update(&make_ref(45, 8, 28), 0.95, "direct"); // Romans 8:28
        // Same book AND chapter
        assert!((ctx.confidence_boost(45, 8) - SAME_CHAPTER_BOOST).abs() < f64::EPSILON);
    }

    #[test]
    fn test_different_book_no_boost() {
        let mut ctx = SermonContext::new();
        ctx.update(&make_ref(45, 8, 28), 0.95, "direct"); // Romans
        // Different book (John)
        assert_eq!(ctx.confidence_boost(43, 3), 0.0);
    }

    #[test]
    fn test_session_history() {
        let mut ctx = SermonContext::new();
        ctx.update(&make_ref(45, 8, 28), 0.95, "direct");
        ctx.update(&make_ref(45, 8, 29), 0.88, "contextual");
        assert_eq!(ctx.history().len(), 2);
    }

    #[test]
    fn test_clear_session() {
        let mut ctx = SermonContext::new();
        ctx.update(&make_ref(45, 8, 28), 0.95, "direct");
        ctx.clear_session();
        assert!(!ctx.is_valid());
        assert!(ctx.history().is_empty());
    }

    #[test]
    fn test_last_in_book() {
        let mut ctx = SermonContext::new();
        ctx.update(&make_ref(45, 8, 28), 0.95, "direct");  // Romans
        ctx.update(&make_ref(43, 3, 16), 0.90, "direct");  // John
        ctx.update(&make_ref(45, 9, 1), 0.85, "direct");   // Romans again

        let last_romans = ctx.last_in_book(45).unwrap();
        assert_eq!(last_romans.chapter, 9);
        assert_eq!(last_romans.verse_start, 1);
    }

    // --- Bullet 5.1: time-decay topic vector ---

    fn approx(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "expected {b:?}, got {a:?}");
        }
    }

    #[test]
    fn test_empty_topic_vector_is_none() {
        let mut ctx = SermonContext::new();
        assert!(ctx.topic_vector().is_none());
    }

    #[test]
    fn test_topic_vector_active_only_is_mean() {
        let mut ctx = SermonContext::new();
        let base = Instant::now();
        ctx.ingest_embedding_at(vec![1.0, 0.0], base);
        ctx.ingest_embedding_at(vec![0.0, 1.0], base + Duration::from_secs(1));
        // Both within the 90s active window, background empty → plain mean.
        approx(&ctx.topic_vector().unwrap(), &[0.5, 0.5]);
        assert_eq!(ctx.active_window_len(), 2);
        assert_eq!(ctx.background_window_len(), 0);
    }

    #[test]
    fn test_time_decay_window_shift() {
        let mut ctx = SermonContext::new();
        let base = Instant::now();
        // "grace" at t0
        ctx.ingest_embedding_at(vec![1.0, 0.0], base);
        // "judgment" 100s later → grace (age 100s > 90s) drains to background
        ctx.ingest_embedding_at(vec![0.0, 1.0], base + Duration::from_secs(100));

        assert_eq!(ctx.active_window_len(), 1);
        assert_eq!(ctx.background_window_len(), 1);
        // 0.75*[0,1] + 0.25*[1,0] = [0.25, 0.75] → shifted toward judgment
        approx(&ctx.topic_vector().unwrap(), &[0.25, 0.75]);
    }

    #[test]
    fn test_background_window_capped_at_500() {
        let mut ctx = SermonContext::new();
        let base = Instant::now();
        // Each ingest is 100s after the previous, so every prior active entry
        // drains to background. After 600 ingests the cap holds at 500.
        for i in 0..600u64 {
            ctx.ingest_embedding_at(vec![1.0], base + Duration::from_secs(i * 100));
        }
        assert_eq!(ctx.background_window_len(), BACKGROUND_WINDOW_CAP);
        assert_eq!(ctx.active_window_len(), 1);
    }

    #[test]
    fn test_topic_vector_lazy_recompute_on_ingest() {
        let mut ctx = SermonContext::new();
        let base = Instant::now();
        ctx.ingest_embedding_at(vec![1.0, 0.0], base);
        approx(&ctx.topic_vector().unwrap(), &[1.0, 0.0]);
        // New sentence marks dirty → next call reflects it.
        ctx.ingest_embedding_at(vec![0.0, 1.0], base + Duration::from_secs(1));
        approx(&ctx.topic_vector().unwrap(), &[0.5, 0.5]);
    }

    #[test]
    fn test_clear_session_resets_topic_state() {
        let mut ctx = SermonContext::new();
        ctx.ingest_embedding_at(vec![1.0, 0.0], Instant::now());
        ctx.clear_session();
        assert_eq!(ctx.active_window_len(), 0);
        assert_eq!(ctx.background_window_len(), 0);
        assert!(ctx.topic_vector().is_none());
    }
}
