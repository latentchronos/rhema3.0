//! Deterministic phonetic correction for scripture proper nouns.
//!
//! Uses Double Metaphone phonetic codes (via `rphonetic`) combined with
//! Levenshtein edit distance to find near-misses of canonical scripture names,
//! while a stoplist prevents false-positives on common English words.

use rphonetic::{DoubleMetaphone, Encoder};

/// Canonical scripture vocabulary: 66 Bible book base names + commonly-mangled
/// proper nouns. Deduplicated single-word forms only.
static DICTIONARY: &[&str] = &[
    // Old Testament
    "Genesis",
    "Exodus",
    "Leviticus",
    "Numbers",
    "Deuteronomy",
    "Joshua",
    "Judges",
    "Ruth",
    "Samuel",
    "Kings",
    "Chronicles",
    "Ezra",
    "Nehemiah",
    "Esther",
    "Job",
    "Psalms",
    "Proverbs",
    "Ecclesiastes",
    "Isaiah",
    "Jeremiah",
    "Lamentations",
    "Ezekiel",
    "Daniel",
    "Hosea",
    "Joel",
    "Amos",
    "Obadiah",
    "Jonah",
    "Micah",
    "Nahum",
    "Habakkuk",
    "Zephaniah",
    "Haggai",
    "Zechariah",
    "Malachi",
    // New Testament
    "Matthew",
    "Mark",
    "Luke",
    "John",
    "Acts",
    "Romans",
    "Corinthians",
    "Galatians",
    "Ephesians",
    "Philippians",
    "Colossians",
    "Thessalonians",
    "Timothy",
    "Titus",
    "Philemon",
    "Hebrews",
    "James",
    "Peter",
    "Jude",
    "Revelation",
    // Commonly-mangled proper nouns
    "Hezekiah",
    "Zerubbabel",
    "Melchizedek",
    "Nebuchadnezzar",
    "Zacchaeus",
    "Nicodemus",
];

/// Scripture words that are also common English words — never correct these.
static STOPLIST: &[&str] = &[
    "mark", "job", "acts", "ruth", "amos", "hosea", "joel", "james", "jude",
    "numbers", "judges", "kings", "romans", "titus", "lord",
];

/// Common English function words we never want to "correct".
static COMMON_ENGLISH: &[&str] = &[
    "the", "and", "to", "of", "a", "in", "is", "he", "that", "for", "this", "with",
];

/// Precomputed dictionary entry with canonical name and phonetic code.
struct DictEntry {
    canonical: &'static str,
    lower: String,
    phonetic: String,
}

/// Deterministic phonetic corrector for scripture proper nouns.
///
/// Build once with `ScriptureCorrector::new()` (or `Default::default()`);
/// then call `correct_token` for each transcript token.
pub struct ScriptureCorrector {
    entries: Vec<DictEntry>,
    encoder: DoubleMetaphone,
}

impl ScriptureCorrector {
    /// Build the corrector: pre-compute lowercased names and Double Metaphone
    /// codes for every dictionary entry.
    pub fn new() -> Self {
        let encoder = DoubleMetaphone::default();
        let entries = DICTIONARY
            .iter()
            .map(|&canonical| {
                let lower = canonical.to_lowercase();
                let phonetic = encoder.encode(&lower);
                DictEntry { canonical, lower, phonetic }
            })
            .collect();
        Self { entries, encoder }
    }

    /// Returns the canonical scripture spelling when `token` is a clear near-miss
    /// of a scripture word AND `context_allows` is true. Otherwise `None`.
    ///
    /// Matching criteria (all must hold):
    /// - `context_allows` is `true`
    /// - token is not already correct (exact case-insensitive match → `None`)
    /// - token is not in STOPLIST or COMMON_ENGLISH
    /// - Levenshtein distance to the best dictionary word ≤ 2
    /// - Double Metaphone primary code of the token equals that of the dictionary word
    pub fn correct_token(&self, token: &str, context_allows: bool) -> Option<String> {
        if !context_allows {
            return None;
        }

        let lower = token.to_lowercase();

        // Stoplist and common English guard
        if STOPLIST.contains(&lower.as_str()) || COMMON_ENGLISH.contains(&lower.as_str()) {
            return None;
        }

        let token_phonetic = self.encoder.encode(&lower);

        let mut best_dist: Option<usize> = None;
        let mut best_canonical: Option<&'static str> = None;

        for entry in &self.entries {
            // Exact match (case-insensitive) → nothing to correct
            if entry.lower == lower {
                return None;
            }

            // Quick length pre-filter: skip if lengths differ by more than 2
            let len_diff = if lower.len() > entry.lower.len() {
                lower.len() - entry.lower.len()
            } else {
                entry.lower.len() - lower.len()
            };
            if len_diff > 2 {
                continue;
            }

            // Edit distance gate
            let dist = crate::textutil::levenshtein(&lower, &entry.lower);
            if dist > 2 {
                continue;
            }

            // Phonetic gate: primary codes must match
            if token_phonetic != entry.phonetic {
                continue;
            }

            // Keep the best (smallest edit distance) candidate; ties → first seen
            match best_dist {
                None => {
                    best_dist = Some(dist);
                    best_canonical = Some(entry.canonical);
                }
                Some(prev) if dist < prev => {
                    best_dist = Some(dist);
                    best_canonical = Some(entry.canonical);
                }
                _ => {}
            }
        }

        best_canonical.map(|s| s.to_string())
    }
}

impl Default for ScriptureCorrector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrects_obvious_near_miss() {
        let c = ScriptureCorrector::new();
        assert_eq!(c.correct_token("habakuk", true).as_deref(), Some("Habakkuk"));
        assert_eq!(c.correct_token("hezzakiah", true).as_deref(), Some("Hezekiah"));
    }

    #[test]
    fn leaves_clean_and_ambiguous_words_alone() {
        let c = ScriptureCorrector::new();
        assert_eq!(c.correct_token("mark", true), None);     // stoplist
        assert_eq!(c.correct_token("the", true), None);      // common English
        assert_eq!(c.correct_token("habakuk", false), None); // context gate closed
    }

    #[test]
    fn exact_match_returns_none() {
        let c = ScriptureCorrector::new();
        // Already correct spellings should not be "corrected"
        assert_eq!(c.correct_token("Habakkuk", true), None);
        assert_eq!(c.correct_token("Genesis", true), None);
        assert_eq!(c.correct_token("Revelation", true), None);
    }

    #[test]
    fn unrelated_word_returns_none() {
        let c = ScriptureCorrector::new();
        assert_eq!(c.correct_token("programming", true), None);
        assert_eq!(c.correct_token("elephant", true), None);
    }
}
