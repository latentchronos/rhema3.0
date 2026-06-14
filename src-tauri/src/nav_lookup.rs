//! BibleDb-backed [`VerseLookup`] adapter (Phase 3.4 app-side follow-up).
//!
//! `rhema-detection::CursorState` navigates via the `VerseLookup` trait so the
//! crate stays DB-free (dependency inversion). This adapter implements that trait
//! over the app's `BibleDb`, letting the real navigation commands drive the
//! formal cursor with real per-translation verse bounds.
//!
//! `rhema-bible` is on the do-not-touch list, so `last_chapter` is computed by
//! probing `get_chapter` (chapters are contiguous from 1, so the first empty
//! chapter ends the book) rather than by adding a new DB query method.

use rhema_bible::BibleDb;
use rhema_detection::VerseLookup;

/// Upper bound for the `last_chapter` probe (Psalms, the longest book, has 150).
const MAX_PROBE_CHAPTERS: u16 = 150;

pub struct BibleDbVerseLookup<'a> {
    db: &'a BibleDb,
}

impl<'a> BibleDbVerseLookup<'a> {
    pub fn new(db: &'a BibleDb) -> Self {
        Self { db }
    }

    /// Resolve a translation abbreviation (e.g. "KJV") to its DB id.
    fn translation_id(&self, translation: &str) -> Option<i64> {
        self.db
            .list_translations()
            .ok()?
            .into_iter()
            .find(|t| t.abbreviation.eq_ignore_ascii_case(translation))
            .map(|t| t.id)
    }
}

impl VerseLookup for BibleDbVerseLookup<'_> {
    fn verse_exists(&self, translation: &str, book: u8, chapter: u16, verse: u16) -> bool {
        let Some(tid) = self.translation_id(translation) else {
            return false;
        };
        matches!(
            self.db
                .get_verse(tid, book as i32, chapter as i32, verse as i32),
            Ok(Some(_))
        )
    }

    fn last_verse(&self, translation: &str, book: u8, chapter: u16) -> Option<u16> {
        let tid = self.translation_id(translation)?;
        let verses = self.db.get_chapter(tid, book as i32, chapter as i32).ok()?;
        verses
            .iter()
            .map(|v| v.verse)
            .max()
            .and_then(|m| u16::try_from(m).ok())
    }

    fn last_chapter(&self, translation: &str, book: u8) -> Option<u16> {
        let tid = self.translation_id(translation)?;
        let mut last = None;
        // Chapters are contiguous from 1; stop at the first empty one.
        for chapter in 1..=MAX_PROBE_CHAPTERS {
            let present = self
                .db
                .get_chapter(tid, book as i32, chapter as i32)
                .map(|v| !v.is_empty())
                .unwrap_or(false);
            if present {
                last = Some(chapter);
            } else {
                break;
            }
        }
        last
    }
}
