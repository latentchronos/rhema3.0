//! Formal navigation cursor state machine (Phase 3, Bullet 3.1).
//!
//! Replaces the informal position tracking in [`crate::reading_mode`] with a
//! `CursorState` that enforces its invariants at the type level: positions are
//! validated on construction, the undo `history` is bounded, and the redo
//! `forward` stack is cleared on any fresh (non-redo) navigation.
//!
//! Scope of this bullet: the types + structural invariants only. DB-backed
//! maximum-chapter/verse bounds and translation-specific verse omission are
//! Bullet 3.4 (they need `BibleDb`, which this crate does not depend on); the
//! epoch lock that resolves operator-vs-voice races is Bullet 3.2.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Maximum entries retained in the undo history; the oldest is dropped silently.
pub const HISTORY_CAP: usize = 50;

/// Errors from constructing or mutating cursor state.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CursorError {
    #[error("book {0} out of range (must be 1..=66)")]
    BookOutOfRange(u8),
    #[error("chapter {0} invalid (must be >= 1)")]
    InvalidChapter(u16),
    #[error("verse {0} invalid (must be >= 1)")]
    InvalidVerse(u16),
    #[error("range {0}-{1} invalid (start >= 1 and end >= start)")]
    InvalidRange(u16, u16),
}

/// How the current verse is being displayed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CursorMode {
    /// Normal single-verse display.
    Single,
    /// A verse range, e.g. Romans 8:1-4.
    Range,
    /// Reading-mode auto-progression is active.
    Reading,
}

/// A validated display position. Construct via [`VersePosition::new`] — the
/// fields are public for reads, but the constructor is the only way to build a
/// valid one, so callers cannot fabricate an out-of-range position.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersePosition {
    /// Book number, 1..=66 (OT 1–39, NT 40–66).
    pub book: u8,
    pub chapter: u16,
    pub verse: u16,
    pub translation: String,
    /// For range display: `Some((start_verse, end_verse))` with `end >= start`.
    pub range: Option<(u16, u16)>,
}

impl VersePosition {
    /// Build a validated position. Enforces book ∈ 1..=66, chapter ≥ 1,
    /// verse ≥ 1, and a well-formed range. (Upper bounds per book/chapter and
    /// translation omission are checked against `BibleDb` in Bullet 3.4.)
    pub fn new(
        book: u8,
        chapter: u16,
        verse: u16,
        translation: impl Into<String>,
        range: Option<(u16, u16)>,
    ) -> Result<Self, CursorError> {
        if !(1..=66).contains(&book) {
            return Err(CursorError::BookOutOfRange(book));
        }
        if chapter < 1 {
            return Err(CursorError::InvalidChapter(chapter));
        }
        if verse < 1 {
            return Err(CursorError::InvalidVerse(verse));
        }
        if let Some((start, end)) = range {
            if start < 1 || end < start {
                return Err(CursorError::InvalidRange(start, end));
            }
        }
        Ok(Self {
            book,
            chapter,
            verse,
            translation: translation.into(),
            range,
        })
    }

    /// The natural display mode for this position (range present → `Range`).
    fn natural_mode(&self) -> CursorMode {
        if self.range.is_some() {
            CursorMode::Range
        } else {
            CursorMode::Single
        }
    }
}

/// The navigation state machine: current position plus bounded undo/redo stacks.
///
/// Internal stacks are private so the cap-50 and clear-on-non-redo invariants
/// can only be maintained through the navigation methods.
#[derive(Debug, Clone)]
pub struct CursorState {
    position: VersePosition,
    history: VecDeque<VersePosition>,
    forward: VecDeque<VersePosition>,
    mode: CursorMode,
    /// Epoch at which this position was set; the lock protocol is Bullet 3.2.
    epoch: u64,
}

impl CursorState {
    /// Start a cursor at `position` in `mode`, with empty history/forward.
    pub fn new(position: VersePosition, mode: CursorMode) -> Self {
        Self {
            position,
            history: VecDeque::new(),
            forward: VecDeque::new(),
            mode,
            epoch: 0,
        }
    }

    pub fn position(&self) -> &VersePosition {
        &self.position
    }

    pub fn mode(&self) -> CursorMode {
        self.mode
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Stamp the epoch (used by the Bullet 3.2 lock protocol).
    pub fn set_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
    }

    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    pub fn forward_len(&self) -> usize {
        self.forward.len()
    }

    /// Fresh navigation to a new position: pushes the current position onto the
    /// undo history (dropping the oldest beyond the cap) and **clears the redo
    /// (forward) stack** — a non-redo move invalidates any pending redo.
    pub fn navigate_to(&mut self, position: VersePosition, mode: CursorMode) {
        let current = std::mem::replace(&mut self.position, position);
        self.push_history(current);
        self.mode = mode;
        self.forward.clear();
    }

    /// Undo: step back to the previous position, moving the current one onto the
    /// redo stack. Returns `false` if there is nothing to undo.
    pub fn back(&mut self) -> bool {
        match self.history.pop_back() {
            Some(previous) => {
                let current = std::mem::replace(&mut self.position, previous);
                self.forward.push_back(current);
                self.mode = self.position.natural_mode();
                true
            }
            None => false,
        }
    }

    /// Redo: re-apply the most recently undone position. Pushes the current
    /// position onto history but does **not** clear the forward stack. Returns
    /// `false` if there is nothing to redo.
    pub fn redo(&mut self) -> bool {
        match self.forward.pop_back() {
            Some(next) => {
                let current = std::mem::replace(&mut self.position, next);
                self.push_history(current);
                self.mode = self.position.natural_mode();
                true
            }
            None => false,
        }
    }

    /// Move to the next existing verse, bounds-checked via `lookup` (Bullet 3.4):
    /// cross-chapter / cross-book arithmetic, both Bible boundaries, and
    /// translation-omitted verses (skips up to [`OMISSION_MAX_RETRIES`]).
    pub fn next_verse(&mut self, lookup: &dyn VerseLookup) -> NavOutcome {
        let p = self.position.clone();
        // Asymmetric range rule (Bullet 3.5): forward navigation calculates from
        // the END of a displayed range; the result collapses to a single verse.
        let from_verse = p.range.map_or(p.verse, |(_, end)| end);
        self.apply_step(
            compute_next(lookup, &p.translation, p.book, p.chapter, from_verse),
            "forward",
        )
    }

    /// Move to the previous existing verse, bounds-checked via `lookup`.
    pub fn previous_verse(&mut self, lookup: &dyn VerseLookup) -> NavOutcome {
        let p = self.position.clone();
        // Asymmetric range rule (Bullet 3.5): backward navigation calculates from
        // the START of a displayed range; the result collapses to a single verse.
        let from_verse = p.range.map_or(p.verse, |(start, _)| start);
        self.apply_step(
            compute_prev(lookup, &p.translation, p.book, p.chapter, from_verse),
            "backward",
        )
    }

    fn apply_step(&mut self, step: Result<(u8, u16, u16), NavError>, dir: &str) -> NavOutcome {
        match step {
            Ok((book, chapter, verse)) => {
                let translation = self.position.translation.clone();
                match VersePosition::new(book, chapter, verse, translation, None) {
                    Ok(pos) => {
                        self.navigate_to(pos, CursorMode::Single);
                        NavOutcome::Moved
                    }
                    Err(_) => NavOutcome::LookupFailed,
                }
            }
            Err(NavError::AtBibleBoundary) => {
                log::warn!("cursor: at Bible boundary, cannot go {dir}");
                NavOutcome::AtBibleBoundary
            }
            Err(NavError::LookupFailed) => {
                log::error!("cursor: verse lookup failed going {dir} (omission retries exhausted)");
                NavOutcome::LookupFailed
            }
        }
    }

    fn push_history(&mut self, position: VersePosition) {
        self.history.push_back(position);
        while self.history.len() > HISTORY_CAP {
            self.history.pop_front();
        }
    }
}

/// Maximum consecutive omitted verses to skip when navigating (Bullet 3.4).
pub const OMISSION_MAX_RETRIES: u16 = 3;

/// Result of a bounds-checked navigation step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavOutcome {
    /// The cursor moved to a new verse.
    Moved,
    /// Already at the first/last verse of the Bible — no change.
    AtBibleBoundary,
    /// Could not resolve an existing verse (omission retries exhausted).
    LookupFailed,
}

/// Verse-existence oracle for navigation bounds. The app implements this over
/// `BibleDb`; tests use a mock. Keeps `rhema-detection` free of a database
/// dependency while honouring "consult the DB, never hardcode verse counts".
pub trait VerseLookup {
    /// Whether `verse` exists in `book`/`chapter` for `translation`.
    fn verse_exists(&self, translation: &str, book: u8, chapter: u16, verse: u16) -> bool;
    /// Highest verse number in `book`/`chapter`, or `None` if it does not exist.
    fn last_verse(&self, translation: &str, book: u8, chapter: u16) -> Option<u16>;
    /// Highest chapter number in `book`, or `None` if the book does not exist.
    fn last_chapter(&self, translation: &str, book: u8) -> Option<u16>;
}

enum NavError {
    AtBibleBoundary,
    LookupFailed,
}

fn compute_next(
    lookup: &dyn VerseLookup,
    t: &str,
    book: u8,
    chapter: u16,
    verse: u16,
) -> Result<(u8, u16, u16), NavError> {
    // 1. Next verse within the current chapter (skipping omissions).
    if let Some(last) = lookup.last_verse(t, book, chapter) {
        if verse < last {
            return match first_existing_up(lookup, t, book, chapter, verse + 1, last) {
                Some(v) => Ok((book, chapter, v)),
                None => Err(NavError::LookupFailed),
            };
        }
    }
    // 2. First verse of the next chapter.
    if let Some(last) = lookup.last_verse(t, book, chapter + 1) {
        if let Some(v) = first_existing_up(lookup, t, book, chapter + 1, 1, last) {
            return Ok((book, chapter + 1, v));
        }
    }
    // 3. First verse of the next book.
    if book < 66 {
        let nb = book + 1;
        if let Some(last) = lookup.last_verse(t, nb, 1) {
            if let Some(v) = first_existing_up(lookup, t, nb, 1, 1, last) {
                return Ok((nb, 1, v));
            }
        }
    }
    // 4. End of the Bible.
    Err(NavError::AtBibleBoundary)
}

fn compute_prev(
    lookup: &dyn VerseLookup,
    t: &str,
    book: u8,
    chapter: u16,
    verse: u16,
) -> Result<(u8, u16, u16), NavError> {
    // 1. Previous verse within the current chapter.
    if verse > 1 {
        if let Some(v) = first_existing_down(lookup, t, book, chapter, verse - 1) {
            return Ok((book, chapter, v));
        }
    }
    // 2. Last verse of the previous chapter.
    if chapter > 1 {
        let pc = chapter - 1;
        if let Some(last) = lookup.last_verse(t, book, pc) {
            if let Some(v) = first_existing_down(lookup, t, book, pc, last) {
                return Ok((book, pc, v));
            }
        }
    }
    // 3. Last verse of the last chapter of the previous book.
    if book > 1 {
        let pb = book - 1;
        if let Some(lc) = lookup.last_chapter(t, pb) {
            if let Some(last) = lookup.last_verse(t, pb, lc) {
                if let Some(v) = first_existing_down(lookup, t, pb, lc, last) {
                    return Ok((pb, lc, v));
                }
            }
        }
    }
    // 4. Start of the Bible.
    Err(NavError::AtBibleBoundary)
}

/// First existing verse at or above `start` (≤ `last`), skipping up to
/// [`OMISSION_MAX_RETRIES`] omitted verses.
fn first_existing_up(
    lookup: &dyn VerseLookup,
    t: &str,
    book: u8,
    chapter: u16,
    start: u16,
    last: u16,
) -> Option<u16> {
    let mut cand = start;
    let mut tries = 0;
    while cand <= last && tries <= OMISSION_MAX_RETRIES {
        if lookup.verse_exists(t, book, chapter, cand) {
            return Some(cand);
        }
        cand += 1;
        tries += 1;
    }
    None
}

/// First existing verse at or below `start` (≥ 1), skipping up to
/// [`OMISSION_MAX_RETRIES`] omitted verses.
fn first_existing_down(
    lookup: &dyn VerseLookup,
    t: &str,
    book: u8,
    chapter: u16,
    start: u16,
) -> Option<u16> {
    let mut cand = start;
    let mut tries = 0;
    loop {
        if lookup.verse_exists(t, book, chapter, cand) {
            return Some(cand);
        }
        if cand == 1 || tries >= OMISSION_MAX_RETRIES {
            return None;
        }
        cand -= 1;
        tries += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pos(verse: u16) -> VersePosition {
        VersePosition::new(45, 8, verse, "KJV", None).unwrap()
    }

    use std::collections::HashMap;

    struct MockBible {
        chapters: HashMap<(u8, u16), Vec<u16>>,
        last_chapters: HashMap<u8, u16>,
    }

    impl VerseLookup for MockBible {
        fn verse_exists(&self, _t: &str, book: u8, chapter: u16, verse: u16) -> bool {
            self.chapters
                .get(&(book, chapter))
                .map_or(false, |vs| vs.contains(&verse))
        }
        fn last_verse(&self, _t: &str, book: u8, chapter: u16) -> Option<u16> {
            self.chapters
                .get(&(book, chapter))
                .and_then(|vs| vs.iter().max().copied())
        }
        fn last_chapter(&self, _t: &str, book: u8) -> Option<u16> {
            self.last_chapters.get(&book).copied()
        }
    }

    fn mock_bible() -> MockBible {
        let mut chapters = HashMap::new();
        chapters.insert((1, 1), (1..=31).collect()); // Genesis 1
        chapters.insert((1, 2), (1..=25).collect()); // Genesis 2
        chapters.insert((66, 22), (1..=21).collect()); // Revelation 22
        chapters.insert((41, 7), (1..=37).filter(|v| *v != 16).collect()); // Mark 7, v16 omitted
        chapters.insert((45, 7), (1..=25).collect()); // Romans 7
        chapters.insert((45, 8), (1..=39).collect()); // Romans 8
        let mut last_chapters = HashMap::new();
        last_chapters.insert(1, 50);
        last_chapters.insert(41, 16);
        last_chapters.insert(66, 22);
        MockBible {
            chapters,
            last_chapters,
        }
    }

    fn cursor_at(book: u8, chapter: u16, verse: u16) -> CursorState {
        CursorState::new(
            VersePosition::new(book, chapter, verse, "KJV", None).unwrap(),
            CursorMode::Single,
        )
    }

    #[test]
    fn next_crosses_chapter_forward() {
        let mut c = cursor_at(1, 1, 31); // Genesis 1:31 (last verse of ch 1)
        assert_eq!(c.next_verse(&mock_bible()), NavOutcome::Moved);
        let p = c.position();
        assert_eq!((p.book, p.chapter, p.verse), (1, 2, 1)); // Genesis 2:1
    }

    #[test]
    fn previous_crosses_chapter_backward() {
        let mut c = cursor_at(1, 2, 1); // Genesis 2:1
        assert_eq!(c.previous_verse(&mock_bible()), NavOutcome::Moved);
        let p = c.position();
        assert_eq!((p.book, p.chapter, p.verse), (1, 1, 31)); // Genesis 1:31
    }

    #[test]
    fn previous_blocked_at_bible_start() {
        let mut c = cursor_at(1, 1, 1); // Genesis 1:1
        assert_eq!(c.previous_verse(&mock_bible()), NavOutcome::AtBibleBoundary);
        let p = c.position();
        assert_eq!((p.book, p.chapter, p.verse), (1, 1, 1)); // unchanged
    }

    #[test]
    fn next_blocked_at_bible_end() {
        let mut c = cursor_at(66, 22, 21); // Revelation 22:21
        assert_eq!(c.next_verse(&mock_bible()), NavOutcome::AtBibleBoundary);
        let p = c.position();
        assert_eq!((p.book, p.chapter, p.verse), (66, 22, 21)); // unchanged
    }

    #[test]
    fn next_skips_omitted_verse() {
        let mut c = cursor_at(41, 7, 15); // Mark 7:15; 7:16 omitted in this translation
        assert_eq!(c.next_verse(&mock_bible()), NavOutcome::Moved);
        let p = c.position();
        assert_eq!((p.book, p.chapter, p.verse), (41, 7, 17)); // skipped to 7:17
    }

    fn cursor_at_range(book: u8, chapter: u16, start: u16, end: u16) -> CursorState {
        CursorState::new(
            VersePosition::new(book, chapter, start, "KJV", Some((start, end))).unwrap(),
            CursorMode::Range,
        )
    }

    #[test]
    fn range_forward_calculates_from_end_and_collapses() {
        let mut c = cursor_at_range(45, 8, 1, 4); // Romans 8:1-4
        assert_eq!(c.next_verse(&mock_bible()), NavOutcome::Moved);
        let p = c.position();
        assert_eq!((p.book, p.chapter, p.verse), (45, 8, 5)); // from end (4) → 8:5
        assert_eq!(p.range, None);
        assert_eq!(c.mode(), CursorMode::Single);
    }

    #[test]
    fn range_backward_calculates_from_start_and_collapses() {
        let mut c = cursor_at_range(45, 8, 1, 4); // Romans 8:1-4
        assert_eq!(c.previous_verse(&mock_bible()), NavOutcome::Moved);
        let p = c.position();
        assert_eq!((p.book, p.chapter, p.verse), (45, 7, 25)); // from start (1) → Romans 7:25
        assert_eq!(p.range, None);
        assert_eq!(c.mode(), CursorMode::Single);
    }

    #[test]
    fn new_validates_book_range() {
        assert_eq!(
            VersePosition::new(0, 1, 1, "KJV", None),
            Err(CursorError::BookOutOfRange(0))
        );
        assert_eq!(
            VersePosition::new(67, 1, 1, "KJV", None),
            Err(CursorError::BookOutOfRange(67))
        );
        assert!(VersePosition::new(1, 1, 1, "KJV", None).is_ok());
        assert!(VersePosition::new(66, 1, 1, "KJV", None).is_ok());
    }

    #[test]
    fn new_validates_chapter_and_verse() {
        assert_eq!(
            VersePosition::new(1, 0, 1, "KJV", None),
            Err(CursorError::InvalidChapter(0))
        );
        assert_eq!(
            VersePosition::new(1, 1, 0, "KJV", None),
            Err(CursorError::InvalidVerse(0))
        );
    }

    #[test]
    fn new_validates_range() {
        assert_eq!(
            VersePosition::new(45, 8, 1, "KJV", Some((0, 4))),
            Err(CursorError::InvalidRange(0, 4))
        );
        assert_eq!(
            VersePosition::new(45, 8, 1, "KJV", Some((5, 3))),
            Err(CursorError::InvalidRange(5, 3))
        );
        assert!(VersePosition::new(45, 8, 1, "KJV", Some((1, 4))).is_ok());
    }

    #[test]
    fn navigate_pushes_history_and_clears_forward() {
        let mut c = CursorState::new(pos(1), CursorMode::Single);
        c.navigate_to(pos(2), CursorMode::Single); // history: [1]
        c.back(); // pos 1, forward: [2]
        assert_eq!(c.forward_len(), 1);
        c.navigate_to(pos(9), CursorMode::Single); // fresh nav → forward cleared
        assert_eq!(c.forward_len(), 0);
        assert!(!c.redo(), "redo must do nothing after forward cleared");
        assert_eq!(c.position().verse, 9);
    }

    #[test]
    fn back_and_redo_walk_the_chain() {
        let mut c = CursorState::new(pos(1), CursorMode::Single);
        c.navigate_to(pos(2), CursorMode::Single);
        c.navigate_to(pos(3), CursorMode::Single);
        assert_eq!(c.position().verse, 3);

        assert!(c.back());
        assert_eq!(c.position().verse, 2);
        assert!(c.back());
        assert_eq!(c.position().verse, 1);
        assert!(!c.back(), "no more history");
        assert_eq!(c.position().verse, 1);

        assert!(c.redo());
        assert_eq!(c.position().verse, 2);
        assert!(c.redo());
        assert_eq!(c.position().verse, 3);
        assert!(!c.redo(), "no more forward");
    }

    #[test]
    fn history_is_capped_at_50() {
        let mut c = CursorState::new(pos(1), CursorMode::Single);
        for v in 2..=70 {
            c.navigate_to(pos(v), CursorMode::Single);
        }
        assert_eq!(c.history_len(), HISTORY_CAP);
    }

    #[test]
    fn back_derives_mode_from_range() {
        let mut c = CursorState::new(
            VersePosition::new(45, 8, 1, "KJV", Some((1, 4))).unwrap(),
            CursorMode::Range,
        );
        c.navigate_to(pos(5), CursorMode::Single); // single
        assert_eq!(c.mode(), CursorMode::Single);
        c.back(); // back to the range position → mode derived as Range
        assert_eq!(c.mode(), CursorMode::Range);
    }
}
