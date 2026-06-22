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
    "chapter", "chapters", "clear", "blank", "hide",
];

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
pub fn canonical_command_word(token: &str) -> Option<&'static str> {
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
        assert_eq!(canonical_command_word("clear"), Some("clear"));
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

    // ---------- diagnostic helpers (not part of the contract) ----------

    #[test]
    fn debug_scores() {
        let dm = DoubleMetaphone::default();
        let token = "phase";
        let lower = token.to_lowercase();
        let token_code = dm.encode(&lower).to_lowercase();
        for slot in COMMAND_SLOTS {
            let slot_code = dm.encode(slot).to_lowercase();
            let raw = levenshtein(&lower, slot);
            let phonetic = levenshtein(&token_code, &slot_code);
            let combined = raw.min(phonetic);
            println!(
                "  {token:12} vs {slot:10}  raw={raw}  ph_tok={token_code} ph_slot={slot_code}  ph_dist={phonetic}  combined={combined}"
            );
        }
    }

    #[test]
    fn debug_scores_chaptah() {
        let dm = DoubleMetaphone::default();
        let token = "chaptah";
        let lower = token.to_lowercase();
        let token_code = dm.encode(&lower).to_lowercase();
        println!("token_code for {token}: {token_code}");
        for slot in COMMAND_SLOTS {
            let slot_code = dm.encode(slot).to_lowercase();
            let raw = levenshtein(&lower, slot);
            let phonetic = levenshtein(&token_code, &slot_code);
            let combined = raw.min(phonetic);
            println!(
                "  {token:12} vs {slot:10}  raw={raw}  ph_tok={token_code} ph_slot={slot_code}  ph_dist={phonetic}  combined={combined}"
            );
        }
    }

    #[test]
    fn debug_scores_verter() {
        let dm = DoubleMetaphone::default();
        let token = "verter";
        let lower = token.to_lowercase();
        let token_code = dm.encode(&lower).to_lowercase();
        for slot in COMMAND_SLOTS {
            let slot_code = dm.encode(slot).to_lowercase();
            let raw = levenshtein(&lower, slot);
            let phonetic = levenshtein(&token_code, &slot_code);
            let combined = raw.min(phonetic);
            println!(
                "  {token:12} vs {slot:10}  raw={raw}  ph_tok={token_code} ph_slot={slot_code}  ph_dist={phonetic}  combined={combined}"
            );
        }
    }
}
