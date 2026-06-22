//! Voice-navigation utterance parsing (Track V).
//!
//! Bullet V1: number-word parsing. A spoken number reaches us from Deepgram in
//! several shapes — digits (`"7"`, `"117"`), digit-ordinals (`"3rd"`, `"21st"`),
//! spelled cardinals (`"seven"`, `"twenty three"`), ordinals (`"seventh"`,
//! `"twenty-first"`), or the article `"a"`/`"an"` (→ 1, as in "back a chapter").
//! [`parse_number`] folds any of these into a `u16`. Pure and DB-free; the
//! command grammar (V2) is built on top.
//!
//! Bullet V2: [`parse_nav_command`] turns a short utterance into a
//! [`NavCommand`] — an absolute jump (verse / chapter / chapter+verse), a
//! relative step by N in either direction over verses or chapters, or clear.
//! Word order disambiguates: a unit keyword *followed by* a number is an
//! absolute target (`"verse 7"`), a number *followed by* a unit is a relative
//! count (`"three verses"`). Conservative by construction: after stripping a
//! small filler set, every remaining token must be a recognized structural
//! word, so ordinary speech ("the next verse says…") is rejected.

use serde::{Deserialize, Serialize};

/// Map a single spelled cardinal/ordinal word to its value. Tens (twenty..ninety)
/// and units/teens (zero..nineteen) only; hundreds and digits are handled by the
/// caller. Returns `None` for any word that is not a number word.
fn word_value(tok: &str) -> Option<u32> {
    Some(match tok {
        "zero" => 0,
        "one" | "first" | "a" | "an" => 1,
        "two" | "second" => 2,
        "three" | "third" => 3,
        "four" | "fourth" => 4,
        "five" | "fifth" => 5,
        "six" | "sixth" => 6,
        "seven" | "seventh" => 7,
        "eight" | "eighth" => 8,
        "nine" | "ninth" => 9,
        "ten" | "tenth" => 10,
        "eleven" | "eleventh" => 11,
        "twelve" | "twelfth" => 12,
        "thirteen" | "thirteenth" => 13,
        "fourteen" | "fourteenth" => 14,
        "fifteen" | "fifteenth" => 15,
        "sixteen" | "sixteenth" => 16,
        "seventeen" | "seventeenth" => 17,
        "eighteen" | "eighteenth" => 18,
        "nineteen" | "nineteenth" => 19,
        "twenty" | "twentieth" => 20,
        "thirty" | "thirtieth" => 30,
        "forty" | "fortieth" => 40,
        "fifty" | "fiftieth" => 50,
        "sixty" | "sixtieth" => 60,
        "seventy" | "seventieth" => 70,
        "eighty" | "eightieth" => 80,
        "ninety" | "ninetieth" => 90,
        _ => return None,
    })
}

/// Parse a digit token, tolerating an ordinal suffix (`"21st"`, `"3rd"`, `"4th"`).
fn digit_value(tok: &str) -> Option<u32> {
    if let Ok(d) = tok.parse::<u32>() {
        return Some(d);
    }
    for suffix in ["st", "nd", "rd", "th"] {
        if let Some(stripped) = tok.strip_suffix(suffix) {
            if let Ok(d) = stripped.parse::<u32>() {
                return Some(d);
            }
        }
    }
    None
}

/// Fold a run of words into a single number in `0..=999`.
///
/// Accepts digits, digit-ordinals, spelled cardinals/ordinals, hyphenated
/// compounds (`"twenty-three"`), `"hundred"` as a ×100 multiplier, and the
/// article `"a"`/`"an"` (→ 1). Returns `None` if the phrase is empty, contains a
/// token that is not part of a number, or exceeds 999 (no Bible chapter or verse
/// approaches that).
pub fn parse_number(phrase: &str) -> Option<u16> {
    let normalized = phrase.to_lowercase().replace('-', " ");
    let mut current: u32 = 0;
    let mut saw_any = false;

    for tok in normalized.split_whitespace() {
        if let Some(d) = digit_value(tok) {
            current = current.checked_add(d)?;
        } else if tok == "hundred" || tok == "hundredth" {
            // "hundred"/"a hundred"/"one hundred" all start the count at 1.
            if current == 0 {
                current = 1;
            }
            current = current.checked_mul(100)?;
        } else if let Some(v) = word_value(tok) {
            current = current.checked_add(v)?;
        } else {
            return None;
        }
        saw_any = true;
        if current > 999 {
            return None;
        }
    }

    if !saw_any {
        return None;
    }
    u16::try_from(current).ok()
}

/// Whether a navigation step moves toward later or earlier scripture.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum NavDirection {
    Forward,
    Backward,
}

/// The granularity a relative step moves by.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum NavUnit {
    Verse,
    Chapter,
}

/// A navigation/display intent parsed from a short operator utterance.
///
/// Serializes to a tagged object for the frontend, e.g.
/// `{"kind":"jump_verse","verse":7}` or
/// `{"kind":"step","unit":"verse","direction":"backward","count":3}`.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NavCommand {
    /// Relative move by `count` units (≥ 1) in `direction`.
    Step {
        unit: NavUnit,
        direction: NavDirection,
        count: u16,
    },
    /// Absolute: a verse within the current chapter.
    JumpVerse { verse: u16 },
    /// Absolute: a chapter within the current book (lands on verse 1).
    JumpChapter { chapter: u16 },
    /// Absolute: a chapter+verse within the current book.
    JumpChapterVerse { chapter: u16, verse: u16 },
    /// Clear / blank the live output.
    Clear,
}

/// Words that carry no navigation meaning and are dropped before parsing. NOT
/// included: `a`/`an` (they mean the number 1), direction words, and unit words.
const NAV_FILLER: &[&str] = &[
    "go", "to", "the", "jump", "please", "show", "take", "me", "us", "now", "lets", "let", "ok",
    "okay", "screen", "can", "you", "would", "could", "will", "just",
];

fn is_forward(tok: &str) -> bool {
    matches!(tok, "next" | "forward")
}

fn is_backward(tok: &str) -> bool {
    matches!(tok, "previous" | "prev" | "back" | "backward")
}

fn is_unit_word(tok: &str) -> bool {
    matches!(tok, "verse" | "verses" | "chapter" | "chapters")
}

/// Whether a single token reads as a number (`"7"`, `"seven"`, `"a"`, `"3rd"`).
fn is_number_token(tok: &str) -> bool {
    parse_number(tok).is_some()
}

/// Whether a non-consumed token is structurally allowed to remain — a direction,
/// a unit, or a number. Anything else means the utterance is not a clean command.
fn is_structural(tok: &str) -> bool {
    is_forward(tok) || is_backward(tok) || is_unit_word(tok) || is_number_token(tok)
}

/// Words that look like direction targets but are NOT navigation units.
/// Checked against raw tokens (before fuzzy remap) so garbled forms can’t hide them.
const FALSE_FRIENDS: &[&str] = &[
    "week", "weeks", "time", "times", "point", "points", "year", "years",
    "sunday", "morning", "day", "thing", "things",
];

/// Way 2 (isolation gating): a final transcript is a command CANDIDATE only when
/// the speaker finished the utterance (Deepgram `speech_final`) AND it is short
/// and standalone — not a command-shaped phrase embedded in flowing speech.
pub fn is_isolated_command_context(transcript: &str, speech_final: bool) -> bool {
    if !speech_final {
        return false;
    }
    let words = transcript.split_whitespace().count();
    (1..=8).contains(&words)
}

/// Parse a short utterance into a [`NavCommand`], or `None` if it is not a clean
/// navigation/display command. Translation-agnostic: it only produces intent;
/// existence (does the verse/chapter exist in the active version) is checked
/// later at the `VerseLookup` layer.
pub fn parse_nav_command(text: &str) -> Option<NavCommand> {
    // Step 1: build raw tokens — lowercase, apostrophes/hyphens → space,
    // split on non-alphanumeric, drop empties. Keep ALL tokens (no filler drop yet).
    let normalized = text.to_lowercase().replace(['\'', '\u{2019}', '-'], " ");
    let all_toks: Vec<&str> = normalized
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();

    if all_toks.is_empty() || all_toks.len() > 10 {
        return None;
    }

    // Step 2: false-friend guard on RAW tokens (before fuzzy remap).
    // If a direction word is immediately followed by a false-friend noun → reject.
    for i in 0..all_toks.len().saturating_sub(1) {
        if (is_forward(all_toks[i]) || is_backward(all_toks[i]))
            && FALSE_FRIENDS.contains(&all_toks[i + 1])
        {
            return None;
        }
    }

    // Step 3: filter filler → toks_raw.
    let toks_raw: Vec<&str> = all_toks
        .iter()
        .copied()
        .filter(|s| !NAV_FILLER.contains(s))
        .collect();

    if toks_raw.is_empty() || toks_raw.len() > 10 {
        return None;
    }

    // Step 4: fuzzy-map each token via the Task 5.1 canonical matchers.
    // Only remap tokens that are NOT already structural (direction/unit/number),
    // and that are at least 3 characters long (prevents false mapping of short
    // common words like "we", "ll" that are close in edit distance to nav words).
    let mapped: Vec<String> = toks_raw.iter().map(|t| {
        if is_structural(t) || t.len() < 3 {
            // Token is already valid or too short to remap reliably — pass through.
            t.to_string()
        } else {
            crate::command_match::canonical_command_word(t)
                .or_else(|| crate::command_match::canonical_number_word(t))
                .map(|c| c.to_string())
                .unwrap_or_else(|| t.to_string())
        }
    }).collect();
    let toks: Vec<&str> = mapped.iter().map(|s| s.as_str()).collect();

    // Clear wins outright — "hide verse", "clear the screen", etc.
    if toks.iter().any(|t| matches!(*t, "clear" | "blank" | "hide")) {
        return Some(NavCommand::Clear);
    }

    // Pass 1: absolute targets are a unit keyword immediately followed by a
    // number run ("chapter 3 verse 7"). Consume those tokens.
    let mut consumed = vec![false; toks.len()];
    let mut chapter_target: Option<u16> = None;
    let mut verse_target: Option<u16> = None;
    let mut i = 0;
    while i < toks.len() {
        let is_chapter = matches!(toks[i], "chapter" | "chapters");
        let is_verse = matches!(toks[i], "verse" | "verses");
        if is_chapter || is_verse {
            let mut j = i + 1;
            while j < toks.len() && is_number_token(toks[j]) {
                j += 1;
            }
            if j > i + 1 {
                let num = parse_number(&toks[i + 1..j].join(" "))?;
                if is_chapter {
                    if chapter_target.is_some() {
                        return None; // two chapter targets — incoherent
                    }
                    chapter_target = Some(num);
                } else {
                    if verse_target.is_some() {
                        return None;
                    }
                    verse_target = Some(num);
                }
                for c in consumed.iter_mut().take(j).skip(i) {
                    *c = true;
                }
                i = j;
                continue;
            }
        }
        i += 1;
    }

    let has_fwd = toks.iter().any(|t| is_forward(t));
    let has_back = toks.iter().any(|t| is_backward(t));
    if has_fwd && has_back {
        return None; // contradictory directions
    }

    // Absolute jump: at least one keyword-then-number target was found. Any
    // leftover token must be a connective (direction/unit), else reject.
    if chapter_target.is_some() || verse_target.is_some() {
        for (k, t) in toks.iter().enumerate() {
            if !consumed[k] && !(is_forward(t) || is_backward(t) || is_unit_word(t)) {
                return None;
            }
        }
        return Some(match (chapter_target, verse_target) {
            (Some(chapter), Some(verse)) => NavCommand::JumpChapterVerse { chapter, verse },
            (Some(chapter), None) => NavCommand::JumpChapter { chapter },
            (None, Some(verse)) => NavCommand::JumpVerse { verse },
            (None, None) => unreachable!(),
        });
    }

    // Relative step: needs a direction. Unit defaults to verse; count defaults
    // to 1 (the free-floating number, e.g. "back three verses").
    let direction = if has_fwd {
        NavDirection::Forward
    } else if has_back {
        NavDirection::Backward
    } else {
        return None;
    };

    // Step 5: require an explicit unit word for relative steps.
    // A bare direction with no "verse"/"chapter" is rejected (closes the
    // "next fires accidentally" hole).
    if !toks.iter().any(|t| is_unit_word(t)) {
        return None;
    }

    let unit = if toks.iter().any(|t| matches!(*t, "chapter" | "chapters")) {
        NavUnit::Chapter
    } else {
        NavUnit::Verse
    };

    let nums: Vec<&str> = toks.iter().copied().filter(|t| is_number_token(t)).collect();
    let count = if nums.is_empty() {
        1
    } else {
        parse_number(&nums.join(" "))?
    };
    if count == 0 {
        return None;
    }

    // Full-match guard: every token must be structural, or it is not a command.
    if !toks.iter().all(|t| is_structural(t)) {
        return None;
    }

    Some(NavCommand::Step {
        unit,
        direction,
        count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_digits() {
        assert_eq!(parse_number("7"), Some(7));
        assert_eq!(parse_number("117"), Some(117));
        assert_eq!(parse_number("0"), Some(0));
    }

    #[test]
    fn parses_digit_ordinals() {
        assert_eq!(parse_number("3rd"), Some(3));
        assert_eq!(parse_number("21st"), Some(21));
        assert_eq!(parse_number("2nd"), Some(2));
        assert_eq!(parse_number("150th"), Some(150));
    }

    #[test]
    fn parses_cardinals() {
        assert_eq!(parse_number("seven"), Some(7));
        assert_eq!(parse_number("twelve"), Some(12));
        assert_eq!(parse_number("twenty"), Some(20));
    }

    #[test]
    fn parses_compound_tens() {
        assert_eq!(parse_number("twenty three"), Some(23));
        assert_eq!(parse_number("twenty-three"), Some(23));
        assert_eq!(parse_number("ninety nine"), Some(99));
    }

    #[test]
    fn parses_ordinals() {
        assert_eq!(parse_number("seventh"), Some(7));
        assert_eq!(parse_number("twenty-first"), Some(21));
        assert_eq!(parse_number("twentieth"), Some(20));
    }

    #[test]
    fn parses_hundreds() {
        assert_eq!(parse_number("one hundred nineteen"), Some(119));
        assert_eq!(parse_number("a hundred"), Some(100));
        assert_eq!(parse_number("hundred"), Some(100));
        assert_eq!(parse_number("one hundred seventy six"), Some(176));
    }

    #[test]
    fn article_is_one() {
        assert_eq!(parse_number("a"), Some(1));
        assert_eq!(parse_number("an"), Some(1));
    }

    #[test]
    fn rejects_non_numbers() {
        assert_eq!(parse_number("psalm"), None);
        assert_eq!(parse_number("verse seven"), None); // "verse" is not a number word
        assert_eq!(parse_number(""), None);
        assert_eq!(parse_number("   "), None);
    }

    #[test]
    fn rejects_out_of_range() {
        assert_eq!(parse_number("one thousand"), None); // "thousand" unknown
        assert_eq!(parse_number("9999"), None);
    }

    // ---- parse_nav_command (V2) ----

    fn step(unit: NavUnit, direction: NavDirection, count: u16) -> NavCommand {
        NavCommand::Step {
            unit,
            direction,
            count,
        }
    }

    #[test]
    fn relative_verse_steps_default_one() {
        use NavDirection::*;
        use NavUnit::*;
        // These have an explicit unit word — still work.
        assert_eq!(parse_nav_command("next verse"), Some(step(Verse, Forward, 1)));
        assert_eq!(
            parse_nav_command("previous verse"),
            Some(step(Verse, Backward, 1))
        );
        // These have no unit word — rejected by require-unit rule.
        assert_eq!(parse_nav_command("go forward"), None);
        assert_eq!(parse_nav_command("go back"), None);
        assert_eq!(parse_nav_command("previous one"), None);
    }

    #[test]
    fn bare_direction_without_unit_is_rejected() {
        assert_eq!(parse_nav_command("next"), None);
        assert_eq!(parse_nav_command("back"), None);
    }

    #[test]
    fn direction_plus_unit_still_works_via_fuzzy() {
        use NavDirection::*;
        use NavUnit::*;
        assert_eq!(parse_nav_command("next verse"), Some(step(Verse, Forward, 1)));
        assert_eq!(parse_nav_command("nest phase"), Some(step(Verse, Forward, 1))); // accent-garbled
    }

    #[test]
    fn false_friends_rejected() {
        assert_eq!(parse_nav_command("next week"), None);
        assert_eq!(parse_nav_command("next point"), None);
    }

    #[test]
    fn relative_steps_by_count_both_directions() {
        use NavDirection::*;
        use NavUnit::*;
        assert_eq!(
            parse_nav_command("forward two verses"),
            Some(step(Verse, Forward, 2))
        );
        assert_eq!(
            parse_nav_command("back three verses"),
            Some(step(Verse, Backward, 3))
        );
        assert_eq!(
            parse_nav_command("next 2 verses"),
            Some(step(Verse, Forward, 2))
        );
    }

    #[test]
    fn relative_chapter_steps_both_directions() {
        use NavDirection::*;
        use NavUnit::*;
        assert_eq!(
            parse_nav_command("next chapter"),
            Some(step(Chapter, Forward, 1))
        );
        assert_eq!(
            parse_nav_command("back a chapter"),
            Some(step(Chapter, Backward, 1))
        );
        assert_eq!(
            parse_nav_command("forward two chapters"),
            Some(step(Chapter, Forward, 2))
        );
        assert_eq!(
            parse_nav_command("back two chapters"),
            Some(step(Chapter, Backward, 2))
        );
    }

    #[test]
    fn absolute_verse_jump() {
        assert_eq!(
            parse_nav_command("go to verse 7"),
            Some(NavCommand::JumpVerse { verse: 7 })
        );
        assert_eq!(
            parse_nav_command("verse seven"),
            Some(NavCommand::JumpVerse { verse: 7 })
        );
        assert_eq!(
            parse_nav_command("jump to verse twenty three"),
            Some(NavCommand::JumpVerse { verse: 23 })
        );
    }

    #[test]
    fn absolute_chapter_jump() {
        assert_eq!(
            parse_nav_command("chapter 3"),
            Some(NavCommand::JumpChapter { chapter: 3 })
        );
        assert_eq!(
            parse_nav_command("go to chapter three"),
            Some(NavCommand::JumpChapter { chapter: 3 })
        );
    }

    #[test]
    fn absolute_chapter_and_verse_jump() {
        assert_eq!(
            parse_nav_command("chapter 3 verse 7"),
            Some(NavCommand::JumpChapterVerse {
                chapter: 3,
                verse: 7
            })
        );
        assert_eq!(
            parse_nav_command("go to chapter twenty three verse seventeen"),
            Some(NavCommand::JumpChapterVerse {
                chapter: 23,
                verse: 17
            })
        );
    }

    #[test]
    fn keyword_then_number_beats_direction() {
        // "go back to verse 7" means jump to verse 7, not "back 7".
        assert_eq!(
            parse_nav_command("go back to verse 7"),
            Some(NavCommand::JumpVerse { verse: 7 })
        );
    }

    #[test]
    fn clear_variants() {
        assert_eq!(parse_nav_command("clear"), Some(NavCommand::Clear));
        assert_eq!(parse_nav_command("clear the screen"), Some(NavCommand::Clear));
        assert_eq!(parse_nav_command("blank"), Some(NavCommand::Clear));
        assert_eq!(parse_nav_command("hide verse"), Some(NavCommand::Clear));
    }

    #[test]
    fn rejects_ordinary_speech() {
        assert_eq!(parse_nav_command("the next verse says something"), None);
        assert_eq!(parse_nav_command("we'll be back after worship"), None);
        assert_eq!(parse_nav_command("let us turn our hearts"), None);
        assert_eq!(parse_nav_command(""), None);
    }

    #[test]
    fn rejects_contradictions_and_bare_numbers() {
        assert_eq!(parse_nav_command("next previous"), None); // both directions
        assert_eq!(parse_nav_command("seven"), None); // no unit/direction keyword
        assert_eq!(parse_nav_command("go to 7"), None); // bare number, no keyword
    }

    #[test]
    fn serializes_to_tagged_json() {
        let v = serde_json::to_value(NavCommand::JumpVerse { verse: 7 }).unwrap();
        assert_eq!(v["kind"], "jump_verse");
        assert_eq!(v["verse"], 7);
        let s = serde_json::to_value(step(NavUnit::Verse, NavDirection::Backward, 3)).unwrap();
        assert_eq!(s["kind"], "step");
        assert_eq!(s["unit"], "verse");
        assert_eq!(s["direction"], "backward");
        assert_eq!(s["count"], 3);
    }

    #[test]
    fn isolation_gate_only_fires_on_short_completed_utterances() {
        assert!(is_isolated_command_context("next verse", true));
        assert!(is_isolated_command_context("go to chapter three verse seven", true));
        assert!(!is_isolated_command_context("next verse", false)); // not completed (embedded)
        assert!(!is_isolated_command_context("", true));            // empty
        assert!(!is_isolated_command_context(
            "and so the next verse really shows us that god is faithful to us", true)); // too long (narration)
    }
}
