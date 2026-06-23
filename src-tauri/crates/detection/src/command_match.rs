//! Accent-generalizing nearest-slot command/number word recovery.
//!
//! Maps a garbled STT token to one of a tiny reserved vocabulary using a
//! combined distance: min(levenshtein(raw), levenshtein(phonetic_codes)).
//! Only accepts a clear, close winner — ambiguous or distant tokens are rejected.

use std::sync::OnceLock;
use rphonetic::{DoubleMetaphone, Encoder};
use crate::textutil::levenshtein;

/// Maximum score for a match to be accepted.
const CEILING: usize = 2;
/// Minimum gap between best and second-best to accept a winner.
const MARGIN: usize = 1;

// ---------------------------------------------------------------------------
// Slot vocabularies
// ---------------------------------------------------------------------------

static COMMAND_SLOTS: &[&str] = &[
    "next", "previous", "forward", "back", "verse", "verses",
    "chapter", "chapters",
];

/// Known accent mis-hearings of "verse" (Nigerian English, observed in testing).
/// Mapped with PRECEDENCE over the generic nearest-slot matcher, because some of
/// these are phonetically closer to the wrong slot (e.g. "pass" -> "back").
const VERSE_CONFUSIONS: &[&str] = &["pass", "class", "face", "bus", "fence", "phase", "vase"];

static NUMBER_SLOTS: &[&str] = &[
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred",
];

// ---------------------------------------------------------------------------
// Precomputed phonetic codes
// ---------------------------------------------------------------------------

struct SlotCodes {
    slots: &'static [&'static str],
    /// Parallel vec of Double-Metaphone primary codes (lowercased for distance).
    codes: Vec<String>,
}

impl SlotCodes {
    fn build(slots: &'static [&'static str]) -> Self {
        let dm = DoubleMetaphone::default();
        let codes = slots
            .iter()
            .map(|s| dm.encode(s).to_lowercase())
            .collect();
        SlotCodes { slots, codes }
    }
}

static COMMAND_CODES: OnceLock<SlotCodes> = OnceLock::new();
static NUMBER_CODES: OnceLock<SlotCodes> = OnceLock::new();

fn command_codes() -> &'static SlotCodes {
    COMMAND_CODES.get_or_init(|| SlotCodes::build(COMMAND_SLOTS))
}

fn number_codes() -> &'static SlotCodes {
    NUMBER_CODES.get_or_init(|| SlotCodes::build(NUMBER_SLOTS))
}

// ---------------------------------------------------------------------------
// Core algorithm
// ---------------------------------------------------------------------------

/// Find the nearest slot for `token` within the given precomputed `SlotCodes`.
///
/// Returns `Some(slot)` only if the best match is within CEILING and beats the
/// second-best by at least MARGIN. Returns `None` otherwise.
///
/// When two slots share the same combined score (e.g. same phonetic code),
/// raw Levenshtein breaks the tie: the one with smaller raw distance is "best"
/// and the other is "second-best" with raw-distance difference counted.
fn nearest_slot(token: &str, sc: &'static SlotCodes) -> Option<&'static str> {
    let lower = token.to_lowercase();

    // Exact match fast path.
    if let Some(&slot) = sc.slots.iter().find(|&&s| s == lower.as_str()) {
        return Some(slot);
    }

    let dm = DoubleMetaphone::default();
    let token_code = dm.encode(&lower).to_lowercase();

    // Collect (combined_score, raw_score, idx) for all slots.
    let mut scored: Vec<(usize, usize, usize)> = sc
        .slots
        .iter()
        .zip(sc.codes.iter())
        .enumerate()
        .map(|(i, (slot, slot_code))| {
            let raw = levenshtein(&lower, slot);
            let phonetic = levenshtein(&token_code, slot_code);
            let combined = raw.min(phonetic);
            (combined, raw, i)
        })
        .collect();

    // Sort by (combined, raw) so ties on combined are broken by raw distance.
    scored.sort_unstable_by_key(|&(c, r, _)| (c, r));

    let (best_combined, best_raw, best_idx) = scored[0];
    let (second_combined, second_raw, _) = scored[1];

    // For the margin check we use the effective comparison score at each rank.
    // If combined scores differ, use combined difference.
    // If combined scores tie, use raw difference (tiebreaker).
    let margin = if second_combined > best_combined {
        second_combined - best_combined
    } else {
        // Combined tied — check if raw breaks the tie clearly enough.
        second_raw.saturating_sub(best_raw)
    };

    if best_combined <= CEILING && margin >= MARGIN {
        Some(sc.slots[best_idx])
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Map a possibly-garbled STT token to the nearest command vocabulary word,
/// or `None` if it is too far or too ambiguous.
///
/// Precedence order:
/// 1. Exact match against COMMAND_SLOTS (handled inside `nearest_slot`).
/// 2. VERSE_CONFUSIONS table — direct accent-confusion override.
/// 3. Nearest-slot fuzzy matcher.
pub fn canonical_command_word(token: &str) -> Option<&'static str> {
    let lower = token.to_lowercase();

    // (1) Exact match — fast path inside nearest_slot handles this, but we
    //     need to check it here too so VERSE_CONFUSIONS doesn't shadow exact
    //     slots. Check slots directly before the confusion table.
    if command_codes().slots.iter().any(|&s| s == lower.as_str()) {
        return nearest_slot(token, command_codes());
    }

    // (2) Verse-confusion override: known mis-hearings of "verse" in Nigerian
    //     English take precedence over the generic fuzzy matcher, because some
    //     of these tokens ("pass", "bus") are phonetically close to other slots
    //     ("back", "verse") and the confusion table reflects observed STT output.
    if VERSE_CONFUSIONS.contains(&lower.as_str()) {
        return Some("verse");
    }

    // (3) Short-token guard: all command slots are at least 4 characters long.
    //     A token shorter than 4 chars is too short to reliably fuzzy-match
    //     (e.g. "hey" phonetically collides with "back" at edit-distance 2).
    //     The VERSE_CONFUSIONS table above explicitly handles the one exception
    //     ("bus" → "verse").
    if lower.len() < 4 {
        return None;
    }

    // (4) Generic nearest-slot fuzzy matcher.
    nearest_slot(token, command_codes())
}

/// Map a possibly-garbled STT token to the nearest number word,
/// or `None` if it is too far or too ambiguous.
pub fn canonical_number_word(token: &str) -> Option<&'static str> {
    nearest_slot(token, number_codes())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_words_pass_through_all_categories() {
        assert_eq!(canonical_command_word("next"), Some("next"));
        assert_eq!(canonical_command_word("chapter"), Some("chapter"));
        // "clear" was removed from COMMAND_SLOTS (exact-only via voice_nav raw check).
        assert_eq!(canonical_command_word("clear"), None);
        assert_eq!(canonical_number_word("three"), Some("three"));
    }

    #[test]
    fn nearest_slot_generalizes_to_unseen_garble() {
        assert_eq!(canonical_command_word("nest"), Some("next"));
        assert_eq!(canonical_command_word("phase"), Some("verse"));
        assert_eq!(canonical_command_word("chaptah"), Some("chapter"));
        assert_eq!(canonical_command_word("provious"), Some("previous"));
        assert_eq!(canonical_number_word("tree"), Some("three"));
        assert_eq!(canonical_number_word("tirty"), Some("thirty"));
    }

    #[test]
    fn ambiguous_or_unrelated_is_rejected() {
        assert_eq!(canonical_command_word("worship"), None);
        assert_eq!(canonical_command_word("hallelujah"), None);
        assert_eq!(canonical_command_word("verter"), None);
    }

    #[test]
    fn verse_confusions_map_to_verse() {
        assert_eq!(canonical_command_word("pass"), Some("verse"));
        assert_eq!(canonical_command_word("class"), Some("verse"));
        assert_eq!(canonical_command_word("face"), Some("verse"));
    }

    #[test]
    fn common_words_no_longer_map_to_clear_family() {
        assert_eq!(canonical_command_word("hey"), None);
        assert_eq!(canonical_command_word("good"), None);
    }

}
