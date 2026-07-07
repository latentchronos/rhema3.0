//! Semantic suppression cache (Phase 2, Bullet 2.4 — Gap B).
//!
//! Breaks the cyclic self-trigger loop: a verse is displayed, the pastor reads
//! it aloud, STT transcribes it, detection re-fires the same verse, and it
//! re-displays forever. This cache remembers recently-emitted verses for a TTL
//! and drops any detection that echoes one of them.
//!
//! Lives in the app (not `rhema-detection`) because it is detection-egress
//! policy. Per the approved Gap B decision it is **emission-keyed** (we record
//! every emitted verse, approximating "what is on screen") and **source-aware**:
//! reading-mode (`"contextual"`) emissions are exempt — they are deliberate
//! navigation (including intentional re-reads), never an echo to suppress.
//!
//! Keyed on `(book_number, chapter, verse)` — translation-agnostic, since an
//! echo of the same verse in any translation is still an echo.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::commands::detection::DetectionResult;

/// How long a displayed verse suppresses its own echo.
const SUPPRESSION_TTL: Duration = Duration::from_secs(45);
/// Hard cap on cached entries (bounds memory; far above any real burst).
const SUPPRESSION_CAP: usize = 64;
/// Detection sources exempt from suppression — legitimate navigation, not echo.
const EXEMPT_SOURCES: &[&str] = &["contextual"];

/// `(book_number, chapter, verse)` — translation-agnostic verse identity.
type VerseKey = (i32, i32, i32);

/// A TTL cache of recently-emitted verses, used to suppress cyclic echoes.
#[derive(Default)]
pub struct SuppressionCache {
    entries: VecDeque<(VerseKey, Instant)>,
}

impl SuppressionCache {
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
        }
    }

    /// Filter a batch of detection results, dropping any that echo a verse
    /// emitted within the TTL. Survivors are recorded so they suppress future
    /// echoes. Results whose `source` is exempt are always kept (and still
    /// recorded, so a displayed reading-mode verse suppresses later echoes from
    /// other sources).
    pub fn filter(&mut self, source: &str, results: Vec<DetectionResult>) -> Vec<DetectionResult> {
        self.filter_at(source, results, Instant::now())
    }

    /// Time-injected core of [`filter`], for deterministic tests.
    fn filter_at(
        &mut self,
        source: &str,
        results: Vec<DetectionResult>,
        now: Instant,
    ) -> Vec<DetectionResult> {
        self.expire(now);
        let suppress = !EXEMPT_SOURCES.contains(&source);

        let mut kept = Vec::with_capacity(results.len());
        for r in results {
            let key: VerseKey = (r.book_number, r.chapter, r.verse);
            if suppress && self.contains(&key) {
                log::info!(
                    "suppression_cache: cyclic hit suppressed {} {}:{} (source {})",
                    r.book_name,
                    r.chapter,
                    r.verse,
                    source
                );
                continue;
            }
            kept.push(r);
        }

        for r in &kept {
            self.record((r.book_number, r.chapter, r.verse), now);
        }
        kept
    }

    /// Drop entries older than the TTL (entries are time-ordered, oldest front).
    fn expire(&mut self, now: Instant) {
        while let Some(&(_, inserted)) = self.entries.front() {
            if now.duration_since(inserted) >= SUPPRESSION_TTL {
                self.entries.pop_front();
            } else {
                break;
            }
        }
    }

    fn contains(&self, key: &VerseKey) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
    }

    /// Record (or refresh) a verse, keeping newest at the back and bounding size.
    fn record(&mut self, key: VerseKey, now: Instant) {
        if let Some(pos) = self.entries.iter().position(|(k, _)| *k == key) {
            self.entries.remove(pos);
        }
        self.entries.push_back((key, now));
        while self.entries.len() > SUPPRESSION_CAP {
            self.entries.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(book: i32, chapter: i32, verse: i32) -> DetectionResult {
        DetectionResult {
            verse_ref: format!("B{book} {chapter}:{verse}"),
            verse_text: String::new(),
            book_name: format!("Book{book}"),
            book_number: book,
            chapter,
            verse,
            confidence: 0.9,
            source: String::new(),
            auto_queued: true,
            raw_score: 0.0,
            minimum_threshold: 0.0,
            auto_queue_threshold: 0.0,
            decision: String::new(),
            explanation: String::new(),
            transcript_snippet: String::new(),
            is_final: true,
        }
    }

    #[test]
    fn first_emission_passes_then_echo_suppressed() {
        let mut c = SuppressionCache::new();
        let t0 = Instant::now();
        let kept1 = c.filter_at("direct", vec![result(45, 8, 1)], t0);
        assert_eq!(kept1.len(), 1);
        let kept2 = c.filter_at("direct", vec![result(45, 8, 1)], t0 + Duration::from_secs(10));
        assert!(kept2.is_empty(), "echo within TTL was not suppressed");
    }

    #[test]
    fn different_verse_not_suppressed() {
        let mut c = SuppressionCache::new();
        let t0 = Instant::now();
        c.filter_at("direct", vec![result(45, 8, 1)], t0);
        let kept = c.filter_at("direct", vec![result(45, 8, 2)], t0);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn contextual_source_is_exempt() {
        let mut c = SuppressionCache::new();
        let t0 = Instant::now();
        c.filter_at("direct", vec![result(43, 3, 16)], t0);
        // Reading-mode re-emitting the displayed verse must not be suppressed.
        let kept =
            c.filter_at("contextual", vec![result(43, 3, 16)], t0 + Duration::from_secs(5));
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn expires_after_ttl() {
        let mut c = SuppressionCache::new();
        let t0 = Instant::now();
        c.filter_at("direct", vec![result(45, 8, 1)], t0);
        let later = t0 + SUPPRESSION_TTL + Duration::from_secs(1);
        let kept = c.filter_at("direct", vec![result(45, 8, 1)], later);
        assert_eq!(kept.len(), 1, "verse should be allowed again after TTL");
    }

    #[test]
    fn capacity_is_bounded() {
        let mut c = SuppressionCache::new();
        let t0 = Instant::now();
        for v in 0..(SUPPRESSION_CAP as i32 + 10) {
            c.filter_at("direct", vec![result(1, 1, v)], t0);
        }
        assert!(c.entries.len() <= SUPPRESSION_CAP);
    }
}
