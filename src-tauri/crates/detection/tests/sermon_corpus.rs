use rhema_detection::DetectionPipeline;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Corpus {
    cases: Vec<CorpusCase>,
}

#[derive(Debug, Deserialize)]
struct CorpusCase {
    id: String,
    transcript: String,
    expected: Vec<ExpectedReference>,
    max_false_positives: usize,
}

#[derive(Debug, Deserialize)]
struct ExpectedReference {
    book: String,
    chapter: i32,
    verse: i32,
}

#[test]
fn sermon_corpus_direct_detection_regression() {
    let corpus: Corpus = serde_json::from_str(include_str!("fixtures/sermon-corpus.json"))
        .expect("sermon corpus fixture should parse");
    let mut pipeline = DetectionPipeline::new();

    for case in corpus.cases {
        let detections = pipeline.process_direct(&case.transcript);
        let matched = detections
            .iter()
            .map(|d| &d.detection.verse_ref)
            .collect::<Vec<_>>();

        for expected in &case.expected {
            assert!(
                matched.iter().any(|actual| {
                    actual.book_name == expected.book
                        && actual.chapter == expected.chapter
                        && actual.verse_start == expected.verse
                }),
                "case {} did not detect {} {}:{}; got {:?}",
                case.id,
                expected.book,
                expected.chapter,
                expected.verse,
                matched
            );
        }

        let false_positive_count = matched.len().saturating_sub(case.expected.len());
        assert!(
            false_positive_count <= case.max_false_positives,
            "case {} had {} false positives; allowed {}",
            case.id,
            false_positive_count,
            case.max_false_positives
        );
    }
}
