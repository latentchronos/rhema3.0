# Sermon Regression Corpus

The sermon regression corpus gives Rhema repeatable detection-quality checks.

## Location

Fixtures live in:

```text
src-tauri/crates/detection/tests/fixtures/sermon-corpus.json
```

The runner lives in:

```text
src-tauri/crates/detection/tests/sermon_corpus.rs
```

## Fixture Format

Each case has:

- `id`: stable test identifier
- `transcript`: transcript text to detect against
- `expected`: expected verse references
- `max_false_positives`: allowed extra detections

Example:

```json
{
  "id": "explicit-john-316",
  "transcript": "Let us look at John chapter three verse sixteen.",
  "expected": [{ "book": "John", "chapter": 3, "verse": 16 }],
  "max_false_positives": 0
}
```

## Current Scope

The first corpus runner targets direct detection because it is deterministic and does not require model files, API keys, audio, or the Bible database.

## Future Work

- Add noisy STT variants.
- Add quotation-matching cases.
- Add semantic-search cases once model fixtures are available.
- Report precision, recall, false-positive count, and latency as metrics.
- Keep private church transcripts out of the public repo unless permission is explicit.
