//! Pre-service priming index (Phase 5, Bullet 5.2).
//!
//! Before a service the pastor pastes their sermon notes. Any explicit verse
//! references in those notes are extracted and given a +0.25 scalar boost in
//! later semantic-search rankings, so the verses the pastor planned to use
//! surface first (ARCHITECTURE §7.3 / §9).
//!
//! Verses are keyed on the coordinate tuple `(book_number, chapter, verse)`
//! rather than the full [`VerseRef`] — `VerseRef` is `PartialEq`-only and carries
//! `book_name`/`verse_end`, which make set membership brittle. Coordinate keying
//! is robust to book-name spelling and range boundaries, matching the Phase-2
//! suppression-cache convention.

use std::collections::HashSet;

use crate::direct::detector::DirectDetector;
use crate::types::VerseRef;

/// The scalar boost applied to a primed verse's search score.
pub const PRIMING_BOOST: f32 = 0.25;

/// Coordinate key: `(book_number, chapter, verse)`.
type VerseKey = (i32, i32, i32);

/// Set of verse coordinates explicitly referenced in the pastor's notes.
#[derive(Debug, Clone, Default)]
pub struct PrimingIndex {
    primed: HashSet<VerseKey>,
}

impl PrimingIndex {
    /// Build a priming index from free-text sermon notes.
    ///
    /// Parses the notes with a **fresh** [`DirectDetector`] (not the live one —
    /// `detect` mutates the detector's sermon context, which must not be polluted
    /// by notes). Verse ranges are expanded to individual coordinates.
    pub fn build_from_notes(notes: &str) -> Self {
        let mut detector = DirectDetector::new();
        let detections = detector.detect(notes);

        let mut primed = HashSet::new();
        for detection in &detections {
            let r = &detection.verse_ref;
            if r.book_number <= 0 || r.chapter <= 0 || r.verse_start <= 0 {
                continue;
            }
            let end = r.verse_end.unwrap_or(r.verse_start).max(r.verse_start);
            for verse in r.verse_start..=end {
                primed.insert((r.book_number, r.chapter, verse));
            }
        }
        Self { primed }
    }

    /// Number of primed verse coordinates.
    pub fn len(&self) -> usize {
        self.primed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.primed.is_empty()
    }

    /// Whether a specific verse coordinate was primed by the notes.
    pub fn is_primed(&self, book_number: i32, chapter: i32, verse: i32) -> bool {
        self.primed.contains(&(book_number, chapter, verse))
    }

    /// Apply the priming boost to ranked search results in place, then re-sort
    /// by score descending. Primed verses gain [`PRIMING_BOOST`], capped at 1.0.
    /// A result's coordinate is keyed on its `verse_start`.
    pub fn apply_boost(&self, results: &mut [(VerseRef, f32)]) {
        if self.primed.is_empty() {
            return;
        }
        for (verse_ref, score) in results.iter_mut() {
            if self.is_primed(verse_ref.book_number, verse_ref.chapter, verse_ref.verse_start) {
                *score = (*score + PRIMING_BOOST).min(1.0);
            }
        }
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vref(book: i32, chapter: i32, verse: i32) -> VerseRef {
        VerseRef {
            book_number: book,
            book_name: "Test".to_string(),
            chapter,
            verse_start: verse,
            verse_end: None,
        }
    }

    #[test]
    fn build_extracts_and_expands_ranges() {
        let index = PrimingIndex::build_from_notes("Romans 8:1-4");
        // Romans = book 45; the range expands to four coordinates.
        assert!(index.is_primed(45, 8, 1));
        assert!(index.is_primed(45, 8, 2));
        assert!(index.is_primed(45, 8, 3));
        assert!(index.is_primed(45, 8, 4));
        assert!(!index.is_primed(45, 8, 5));
        assert_eq!(index.len(), 4);
    }

    #[test]
    fn build_extracts_multiple_refs() {
        let index = PrimingIndex::build_from_notes("Romans 8:1, John 3:16");
        assert!(index.is_primed(45, 8, 1)); // Romans
        assert!(index.is_primed(43, 3, 16)); // John
    }

    #[test]
    fn apply_boost_caps_at_one_and_resorts() {
        let index = PrimingIndex::build_from_notes("Romans 8:1, John 3:16");
        let mut results = vec![
            (vref(45, 8, 1), 0.81),  // Romans 8:1 — primed
            (vref(58, 12, 1), 0.79), // Hebrews 12:1 — not primed
            (vref(43, 3, 16), 0.76), // John 3:16 — primed
        ];
        index.apply_boost(&mut results);

        // Both primed verses capped at 1.0 and sorted to the top.
        assert!((results[0].1 - 1.0).abs() < 1e-6);
        assert!((results[1].1 - 1.0).abs() < 1e-6);
        // Hebrews unboosted, sorted last.
        assert_eq!(results[2].0.book_number, 58);
        assert!((results[2].1 - 0.79).abs() < 1e-6);
    }

    #[test]
    fn boost_does_not_exceed_one_from_high_base() {
        let index = PrimingIndex::build_from_notes("Romans 8:1");
        let mut results = vec![(vref(45, 8, 1), 0.9)];
        index.apply_boost(&mut results);
        assert!((results[0].1 - 1.0).abs() < 1e-6, "0.9 + 0.25 must cap at 1.0");
    }

    #[test]
    fn empty_index_leaves_results_unchanged() {
        let index = PrimingIndex::default();
        let mut results = vec![(vref(45, 8, 1), 0.5)];
        index.apply_boost(&mut results);
        assert!((results[0].1 - 0.5).abs() < 1e-6);
    }
}
