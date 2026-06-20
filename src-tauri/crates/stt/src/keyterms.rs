/// Reserved voice-command words, boosted so Deepgram transcribes them
/// correctly across accents at the source (moderate boost, sermon-context safe).
pub fn command_keyterms() -> Vec<String> {
    ["verse", "chapter", "next", "previous", "forward", "back", "clear"]
        .iter().map(|s| s.to_string()).collect()
}

/// Returns Bible book names, spoken numbered forms, and high-value theological terms
/// for use as Deepgram keyword boosting.
/// Written abbreviations (Jn, Ps, etc.) are intentionally excluded — nobody speaks them
/// and they waste the 100-term cap. Total: 66 books + 18 spoken + 4 theological = 88.
pub fn bible_keyterms() -> Vec<String> {
    let mut terms: Vec<String> = Vec::new();

    // 66 Bible book names
    let books = [
        "Genesis",
        "Exodus",
        "Leviticus",
        "Numbers",
        "Deuteronomy",
        "Joshua",
        "Judges",
        "Ruth",
        "1 Samuel",
        "2 Samuel",
        "1 Kings",
        "2 Kings",
        "1 Chronicles",
        "2 Chronicles",
        "Ezra",
        "Nehemiah",
        "Esther",
        "Job",
        "Psalms",
        "Proverbs",
        "Ecclesiastes",
        "Song of Solomon",
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
        "Matthew",
        "Mark",
        "Luke",
        "John",
        "Acts",
        "Romans",
        "1 Corinthians",
        "2 Corinthians",
        "Galatians",
        "Ephesians",
        "Philippians",
        "Colossians",
        "1 Thessalonians",
        "2 Thessalonians",
        "1 Timothy",
        "2 Timothy",
        "Titus",
        "Philemon",
        "Hebrews",
        "James",
        "1 Peter",
        "2 Peter",
        "1 John",
        "2 John",
        "3 John",
        "Jude",
        "Revelation",
    ];
    terms.extend(books.iter().map(|s| s.to_string()));

    // Spoken forms
    let spoken = [
        "First Samuel",
        "Second Samuel",
        "First Kings",
        "Second Kings",
        "First Chronicles",
        "Second Chronicles",
        "First Corinthians",
        "Second Corinthians",
        "First Thessalonians",
        "Second Thessalonians",
        "First Timothy",
        "Second Timothy",
        "First Peter",
        "Second Peter",
        "First John",
        "Second John",
        "Third John",
        "Song of Songs",
    ];
    terms.extend(spoken.iter().map(|s| s.to_string()));

    // Theological terms — only distinctive, rarely-confused words and proper nouns.
    // Common English words (grace, mercy, salvation, etc.) are intentionally excluded
    // to avoid false insertions. Trimmed to 4 to fit within the Deepgram 100-term cap
    // (budget: 5 core + 7 command + 88 bible = 100; removed "justification" and "eschatology").
    let theological = [
        "propitiation",
        "sanctification",
        "Melchizedek",
        "Nebuchadnezzar",
    ];
    terms.extend(theological.iter().map(|s| s.to_string()));

    terms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spoken_numbered_books_survive_the_cap() {
        let terms = bible_keyterms();
        assert!(terms.iter().any(|t| t == "First John"));
        assert!(terms.iter().any(|t| t == "Second Corinthians"));
    }

    #[test]
    fn drops_unspoken_written_abbreviations() {
        let terms = bible_keyterms();
        assert!(!terms.iter().any(|t| t == "Jn"));   // nobody says "Jn"
        assert!(!terms.iter().any(|t| t == "Ps"));
    }

    #[test]
    fn total_stays_within_budget() {
        // 66 books + 18 spoken + 4 theological = 88; leave headroom for 5 core + 7 command in deepgram.rs
        assert!(bible_keyterms().len() <= 95);
    }

    #[test]
    fn command_words_are_boosted() {
        let terms = command_keyterms();
        for w in ["verse", "chapter", "next", "previous", "forward", "clear"] {
            assert!(terms.iter().any(|t| t == w), "missing {w}");
        }
    }

    #[test]
    fn full_keyterm_budget_fits_under_cap() {
        // 5 core (added in deepgram.rs) + command + bible must not exceed Deepgram's 100 cap.
        assert!(5 + command_keyterms().len() + bible_keyterms().len() <= 100);
    }
}
