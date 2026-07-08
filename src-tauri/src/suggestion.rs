//! Proactive suggestion engine (Phase 5, Bullet 5.3).
//!
//! When the sermon's topic vector finds a strong, unshown, thematically related
//! verse, the engine surfaces it as a suggestion **to the Operator channel only**
//! (ARCHITECTURE §9.4). The hard rules — never auto-project, max one per 300s,
//! skip verses already displayed, honour operator dismissals, fire only above the
//! 0.72 cosine threshold — are enforced here.
//!
//! This is the pure gating core; the live topic-vector search and the
//! Operator-channel emission live in the app consumer (`commands/stt.rs` +
//! `channels.rs`). Verses are keyed on `(book_name.lowercased, chapter, verse)`
//! so display-history (from go-live `VerseDisplay`, which carries the book name)
//! and dismissals line up with BibleDb-resolved suggestion candidates.

use std::collections::HashSet;
use std::time::{Duration, Instant};

/// Minimum gap between surfaced suggestions (ARCHITECTURE §9.4).
pub const SUGGESTION_COOLDOWN_SECS: u64 = 300;

/// Minimum cosine similarity for a suggestion to fire (ARCHITECTURE §9.4).
pub const SUGGESTION_THRESHOLD: f32 = 0.72;

type VerseKey = (String, i32, i32);

fn key(book: &str, chapter: i32, verse: i32) -> VerseKey {
    (book.to_lowercase(), chapter, verse)
}

/// Tracks suggestion cooldown, dismissals, and display history for a service.
pub struct SuggestionEngine {
    last_suggestion_at: Option<Instant>,
    dismissed: HashSet<VerseKey>,
    display_history: HashSet<VerseKey>,
    cooldown: Duration,
    threshold: f32,
}

impl Default for SuggestionEngine {
    fn default() -> Self {
        Self {
            last_suggestion_at: None,
            dismissed: HashSet::new(),
            display_history: HashSet::new(),
            cooldown: Duration::from_secs(SUGGESTION_COOLDOWN_SECS),
            threshold: SUGGESTION_THRESHOLD,
        }
    }
}

impl SuggestionEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that a verse went live this service (skips it from suggestions).
    pub fn record_displayed(&mut self, book: &str, chapter: i32, verse: i32) {
        self.display_history.insert(key(book, chapter, verse));
    }

    /// Operator dismissed a suggestion — suppress that verse for the service.
    pub fn dismiss(&mut self, book: &str, chapter: i32, verse: i32) {
        self.dismissed.insert(key(book, chapter, verse));
    }

    /// Whether a candidate may be suggested right now. Pure (time injected).
    pub fn should_suggest(
        &self,
        book: &str,
        chapter: i32,
        verse: i32,
        similarity: f32,
        now: Instant,
    ) -> bool {
        if similarity <= self.threshold {
            return false;
        }
        let k = key(book, chapter, verse);
        if self.display_history.contains(&k) || self.dismissed.contains(&k) {
            return false;
        }
        match self.last_suggestion_at {
            Some(last) => now.saturating_duration_since(last) >= self.cooldown,
            None => true,
        }
    }

    /// Mark that a suggestion was surfaced at `now` (starts the cooldown).
    pub fn note_suggested(&mut self, now: Instant) {
        self.last_suggestion_at = Some(now);
    }

    /// Reset all per-service state (new service). Called by `start_session`.
    pub fn clear_session(&mut self) {
        self.last_suggestion_at = None;
        self.dismissed.clear();
        self.display_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_high_similarity_suggests() {
        let engine = SuggestionEngine::new();
        let now = Instant::now();
        assert!(engine.should_suggest("Ephesians", 2, 8, 0.84, now));
    }

    #[test]
    fn below_threshold_does_not_suggest() {
        let engine = SuggestionEngine::new();
        let now = Instant::now();
        assert!(!engine.should_suggest("Ephesians", 2, 8, 0.60, now));
        // Exactly at the threshold is not "> 0.72".
        assert!(!engine.should_suggest("Ephesians", 2, 8, 0.72, now));
    }

    #[test]
    fn cooldown_suppresses_then_allows() {
        let mut engine = SuggestionEngine::new();
        let t0 = Instant::now();
        assert!(engine.should_suggest("Romans", 5, 8, 0.79, t0));
        engine.note_suggested(t0);

        // 180s later — inside the 300s cooldown → suppressed.
        let t180 = t0 + Duration::from_secs(180);
        assert!(!engine.should_suggest("Romans", 5, 8, 0.79, t180));

        // 300s later — cooldown elapsed → allowed again.
        let t300 = t0 + Duration::from_secs(300);
        assert!(engine.should_suggest("Romans", 5, 8, 0.79, t300));
    }

    #[test]
    fn dismissed_verse_never_suggests() {
        let mut engine = SuggestionEngine::new();
        engine.dismiss("Romans", 5, 8);
        let now = Instant::now();
        assert!(!engine.should_suggest("Romans", 5, 8, 0.95, now));
        // Case-insensitive on the book name.
        assert!(!engine.should_suggest("romans", 5, 8, 0.95, now));
    }

    #[test]
    fn displayed_verse_never_suggests() {
        let mut engine = SuggestionEngine::new();
        engine.record_displayed("Ephesians", 2, 8);
        let now = Instant::now();
        assert!(!engine.should_suggest("Ephesians", 2, 8, 0.91, now));
    }

    #[test]
    fn clear_session_resets_state() {
        let mut engine = SuggestionEngine::new();
        let t0 = Instant::now();
        engine.note_suggested(t0);
        engine.dismiss("Romans", 5, 8);
        engine.record_displayed("Ephesians", 2, 8);
        engine.clear_session();
        // Cooldown cleared and both sets emptied.
        assert!(engine.should_suggest("Romans", 5, 8, 0.95, t0 + Duration::from_secs(1)));
        assert!(engine.should_suggest("Ephesians", 2, 8, 0.95, t0 + Duration::from_secs(1)));
    }
}
