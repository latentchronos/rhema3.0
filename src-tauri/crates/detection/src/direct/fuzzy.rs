use super::books::BOOKS;

/// A fuzzy match of a Bible book name found in text.
#[derive(Debug, Clone)]
pub struct FuzzyMatch {
    pub book_name: String,
    pub book_number: i32,
    pub start: usize,
    pub end: usize,
    pub distance: usize,
}

/// Determine the maximum allowed edit distance for a book name.
/// Names longer than 8 characters allow distance 3; otherwise distance 2.
fn max_distance_for(name: &str) -> usize {
    if name.len() > 8 {
        3
    } else {
        2
    }
}

/// Find Bible book names in text using fuzzy (Levenshtein) matching.
///
/// Each whitespace-delimited token (and each consecutive pair of tokens for
/// multi-word book names like "1 Corinthians" or "Song of Solomon") is
/// compared against all 66 canonical book names. A match is returned when
/// the edit distance is within the allowed threshold.
pub fn fuzzy_find_books(text: &str) -> Vec<FuzzyMatch> {
    let mut matches: Vec<FuzzyMatch> = Vec::new();
    let text_lower = text.to_lowercase();

    // Collect word spans: (start_byte, end_byte, word_lowercase)
    let words: Vec<(usize, usize, String)> = {
        let mut v = Vec::new();
        let mut i = 0;
        let bytes = text_lower.as_bytes();
        while i < bytes.len() {
            // Skip whitespace
            if bytes[i].is_ascii_whitespace() {
                i += 1;
                continue;
            }
            let start = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            let word = text_lower[start..i].to_string();
            v.push((start, i, word));
        }
        v
    };

    // Try single-word matches and multi-word windows (up to 4 tokens for
    // names like "Song of Solomon").
    for window_size in 1..=4 {
        for wi in 0..words.len() {
            if wi + window_size > words.len() {
                break;
            }
            let span_start = words[wi].0;
            let span_end = words[wi + window_size - 1].1;
            let candidate = &text_lower[span_start..span_end];

            for book in BOOKS {
                let book_lower = book.name.to_lowercase();
                let max_dist = max_distance_for(&book_lower);

                // Quick length filter: if lengths differ by more than max_dist, skip
                let len_diff = if candidate.len() > book_lower.len() {
                    candidate.len() - book_lower.len()
                } else {
                    book_lower.len() - candidate.len()
                };
                if len_diff > max_dist {
                    continue;
                }

                let dist = crate::textutil::levenshtein(candidate, &book_lower);
                if dist > 0 && dist <= max_dist {
                    // Avoid overlapping with an already-found match at the same position
                    let dominated = matches.iter().any(|m| {
                        m.start == span_start && m.end == span_end && m.distance <= dist
                    });
                    if !dominated {
                        matches.push(FuzzyMatch {
                            book_name: book.name.to_string(),
                            book_number: book.number,
                            start: span_start,
                            end: span_end,
                            distance: dist,
                        });
                    }
                }
            }
        }
    }

    // De-duplicate: for overlapping spans keep the lowest-distance match.
    matches.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| a.distance.cmp(&b.distance)));
    let mut result: Vec<FuzzyMatch> = Vec::new();
    let mut last_end: usize = 0;
    for m in matches {
        if m.start >= last_end {
            last_end = m.end;
            result.push(m);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(crate::textutil::levenshtein("hello", "hello"), 0);
    }

    #[test]
    fn test_levenshtein_empty() {
        assert_eq!(crate::textutil::levenshtein("", "abc"), 3);
        assert_eq!(crate::textutil::levenshtein("abc", ""), 3);
    }

    #[test]
    fn test_levenshtein_substitution() {
        // filipians vs philippians: f→ph (1 sub), i→i ok, l→l ok, i→i ok, p→p ok, i→i ok, a→a ok, n→n ok, s→s ok
        // Actually: "filipians" (9 chars) vs "philippians" (11 chars)
        let d = crate::textutil::levenshtein("filipians", "philippians");
        assert!(d <= 3, "distance was {d}");
    }

    #[test]
    fn test_fuzzy_filipians() {
        let matches = fuzzy_find_books("in Filipians chapter 4");
        assert!(!matches.is_empty());
        assert_eq!(matches[0].book_name, "Philippians");
    }

    #[test]
    fn test_fuzzy_revelations() {
        // Common misspelling: "Revelations" (extra 's' but that's just 1 edit from "Revelation")
        let matches = fuzzy_find_books("Revelations 21:1");
        assert!(!matches.is_empty());
        assert_eq!(matches[0].book_name, "Revelation");
    }

    #[test]
    fn test_fuzzy_no_false_positive() {
        // A totally unrelated word should not match any book
        let matches = fuzzy_find_books("programming is fun");
        assert!(matches.is_empty());
    }

    #[test]
    fn test_fuzzy_hebrews_misspelled() {
        let matches = fuzzy_find_books("in Hebrws chapter 11");
        assert!(!matches.is_empty());
        assert_eq!(matches[0].book_name, "Hebrews");
    }
}
