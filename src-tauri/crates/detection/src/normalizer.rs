//! Phonetic transcript normalization (Phase 2, Bullet 2.1).
//!
//! STT output is treated as potentially corrupted input. This is a pure,
//! stateless pre-processing pass applied to the raw transcript *before* entity
//! extraction. It implements the canonical normalization map from
//! `new-features/ARCHITECTURE.md` §7.2, with two deliberate scope decisions:
//!
//! 1. **No number/coordinate conversion.** `direct/parser.rs` already parses
//!    spoken numbers ("three sixteen"), two-number coordinates ("3 16"),
//!    "chapter N verse M", compound numbers, and ranges. Re-doing that here
//!    would be redundant and could fight the parser. So this layer only fixes
//!    what the parser *cannot*: book names (the automaton matches canonical
//!    spellings only) plus translation and numbered-book-ordinal homophones.
//!
//! 2. **Risky context-dependent rules are excluded.** §7.2 lists homophones
//!    like `"envy" → NIV`, `"easy" → ESV`, and `"to/too" → 2`. As *blind*
//!    substitutions these corrupt extremely common English ("do not envy",
//!    "that's easy", "turn **to** Romans"). They require a context-gated pass
//!    (only inside a translation command / between a book and "verse") and are
//!    intentionally left out of this v1 to preserve the safety property: normal
//!    speech must pass through unchanged. Tracked as a follow-up.
//!
//! Mangled forms of long book names (e.g. "doo-ter-onomy" for Deuteronomy) are
//! better handled by the existing `direct/fuzzy.rs` approximate matcher than by
//! exact substitution, so only clearly-spelled homophones live here.
//!
//! The rule set is data-driven (static arrays), so it is testable and
//! extensible without touching logic.

/// Ordered homophone substitution rules: `(lowercased input token sequence,
/// canonical replacement)`. Multi-token patterns (e.g. spelled-out acronyms)
/// are supported; at each position the longest matching pattern wins.
const HOMOPHONE_RULES: &[(&[&str], &str)] = &[
    // ── Book-name homophones (§7.2) ─────────────────────────────────────
    (&["roaming"], "Romans"),
    (&["corintians"], "Corinthians"),
    (&["corinithians"], "Corinthians"),
    (&["efesians"], "Ephesians"),
    (&["phillipians"], "Philippians"),
    (&["soms"], "Psalms"),
    (&["salms"], "Psalms"),
    (&["izaya"], "Isaiah"),
    (&["deuteronomy"], "Deuteronomy"), // canonicalises casing; mangled forms → fuzzy.rs
    (&["revelations"], "Revelation"),  // STT pluralises
    // ── Translation homophones (safe subset only) ───────────────────────
    (&["niv"], "NIV"),
    (&["en", "eye", "vee"], "NIV"),
    (&["nkjv"], "NKJV"),
    (&["en", "kay", "jay", "vee"], "NKJV"),
    (&["esv"], "ESV"),
    (&["kjv"], "KJV"),
    (&["authorized", "version"], "KJV"),
];

/// Lowercased names of the books that have numbered editions, used to gate the
/// ordinal-prefix rule so "first/second/third" is only rewritten when it really
/// is a book prefix (not normal speech like "the first thing").
const NUMBERED_BOOKS: &[&str] = &[
    "samuel",
    "kings",
    "chronicles",
    "corinthians",
    "thessalonians",
    "timothy",
    "peter",
    "john",
];

/// Normalize a raw STT transcript for entity extraction.
///
/// Fixes book-name and translation homophones and rewrites ordinal numbered-book
/// prefixes ("first Corinthians" → "1 Corinthians"). Leaves everything else —
/// including all numbers and coordinates — untouched for the downstream parser.
pub fn normalize_transcript(input: &str) -> String {
    let tokens: Vec<&str> = input.split_whitespace().collect();
    if tokens.is_empty() {
        return String::new();
    }
    let after_homophones = apply_homophones(&tokens);
    let after_ordinals = apply_ordinal_prefixes(&after_homophones);
    after_ordinals.join(" ")
}

/// Apply the homophone substitution table, preferring the longest match at each
/// position so multi-token patterns ("en eye vee") beat single tokens.
fn apply_homophones(tokens: &[&str]) -> Vec<String> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        if let Some((len, replacement)) = longest_match(tokens, i) {
            out.push(replacement.to_string());
            i += len;
        } else {
            out.push(tokens[i].to_string());
            i += 1;
        }
    }
    out
}

/// Find the longest homophone rule whose pattern matches `tokens` starting at
/// `i` (case-insensitive). Returns `(pattern_len, replacement)`.
fn longest_match(tokens: &[&str], i: usize) -> Option<(usize, &'static str)> {
    let mut best: Option<(usize, &'static str)> = None;
    for (pattern, replacement) in HOMOPHONE_RULES {
        if i + pattern.len() > tokens.len() {
            continue;
        }
        let matches = pattern
            .iter()
            .enumerate()
            .all(|(k, &p)| tokens[i + k].eq_ignore_ascii_case(p));
        if matches && best.map_or(true, |(best_len, _)| pattern.len() > best_len) {
            best = Some((pattern.len(), replacement));
        }
    }
    best
}

/// Rewrite "first/second/third" → "1/2/3" only when immediately followed by a
/// numbered book name, so ordinary speech ("the first thing") is untouched.
fn apply_ordinal_prefixes(tokens: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        let ordinal = match tokens[i].to_ascii_lowercase().as_str() {
            "first" => Some("1"),
            "second" => Some("2"),
            "third" => Some("3"),
            _ => None,
        };
        if let Some(digit) = ordinal {
            let next_is_numbered_book = tokens
                .get(i + 1)
                .map(|t| NUMBERED_BOOKS.contains(&t.to_ascii_lowercase().as_str()))
                .unwrap_or(false);
            if next_is_numbered_book {
                out.push(digit.to_string());
                i += 1;
                continue;
            }
        }
        out.push(tokens[i].clone());
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixes_single_book_homophone_without_touching_to() {
        // "roaming" → "Romans", and the common word "to" must survive intact.
        assert_eq!(
            normalize_transcript("turn to roaming chapter eight"),
            "turn to Romans chapter eight"
        );
    }

    #[test]
    fn fixes_multiple_book_homophones() {
        assert_eq!(
            normalize_transcript("efesians and phillipians"),
            "Ephesians and Philippians"
        );
    }

    #[test]
    fn fixes_psalms_variants() {
        assert_eq!(normalize_transcript("soms one nineteen"), "Psalms one nineteen");
        assert_eq!(normalize_transcript("salms twenty three"), "Psalms twenty three");
    }

    #[test]
    fn revelations_is_singularised() {
        assert_eq!(
            normalize_transcript("revelations twenty two"),
            "Revelation twenty two"
        );
    }

    #[test]
    fn izaya_and_corinthians_spellings() {
        assert_eq!(normalize_transcript("izaya fifty three"), "Isaiah fifty three");
        assert_eq!(normalize_transcript("corintians thirteen"), "Corinthians thirteen");
        assert_eq!(normalize_transcript("corinithians thirteen"), "Corinthians thirteen");
    }

    #[test]
    fn multiword_translation_homophone() {
        assert_eq!(normalize_transcript("switch to en eye vee"), "switch to NIV");
        assert_eq!(normalize_transcript("read it in nkjv"), "read it in NKJV");
        assert_eq!(
            normalize_transcript("the authorized version please"),
            "the KJV please"
        );
    }

    #[test]
    fn ordinal_prefix_on_numbered_book() {
        // The ordinal pass adds the prefix; numbers are left for the downstream
        // parser. Correctly-spelled book names keep their original casing (the
        // automaton matches case-insensitively) — only homophones are recased,
        // which is why "corintians" → "Corinthians" but "timothy" stays as-is.
        assert_eq!(
            normalize_transcript("first corintians thirteen"),
            "1 Corinthians thirteen"
        );
        assert_eq!(
            normalize_transcript("second timothy two twelve"),
            "2 timothy two twelve"
        );
    }

    #[test]
    fn ordinal_not_rewritten_in_normal_speech() {
        // "first" before a non-book word must be preserved.
        assert_eq!(
            normalize_transcript("the first thing he said"),
            "the first thing he said"
        );
    }

    #[test]
    fn clean_speech_passes_through_unchanged() {
        // The safety property: no false substitutions on ordinary text.
        let s = "for God so loved the world that he gave his only son";
        assert_eq!(normalize_transcript(s), s);
    }

    #[test]
    fn matching_is_case_insensitive() {
        assert_eq!(normalize_transcript("ROAMING"), "Romans");
    }

    #[test]
    fn empty_input_is_empty() {
        assert_eq!(normalize_transcript(""), "");
        assert_eq!(normalize_transcript("   "), "");
    }
}
