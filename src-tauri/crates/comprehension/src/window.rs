//! Time-bounded sliding transcript window (§3). Evidence, not the thing being
//! classified. Time-bounded, not content-bounded; a pause extends rather than
//! clears the buffer (eviction runs only on ingest).

use std::collections::VecDeque;

struct Segment {
    text: String,
    at_ms: u64,
}

pub struct TranscriptWindow {
    max_age_ms: u64,
    /// Oldest at the front, newest at the back.
    segments: VecDeque<Segment>,
    /// Segments strictly newer than this are the unconsumed delta (Δ) for the
    /// next observer evaluation (D4).
    last_delta_ms: u64,
}

impl TranscriptWindow {
    pub fn new(max_age_ms: u64) -> Self {
        Self {
            max_age_ms,
            segments: VecDeque::new(),
            last_delta_ms: 0,
        }
    }

    /// Append the newest segment and drop anything older than `at_ms - max_age`.
    pub fn ingest(&mut self, text: &str, at_ms: u64) {
        self.segments.push_back(Segment {
            text: text.to_string(),
            at_ms,
        });
        let cutoff = at_ms.saturating_sub(self.max_age_ms);
        while let Some(front) = self.segments.front() {
            if front.at_ms < cutoff {
                self.segments.pop_front();
            } else {
                break;
            }
        }
    }

    /// All retained segments, joined by a single space (the ~60s context window).
    pub fn window_text(&self) -> String {
        self.join(|_| true)
    }

    /// Only segments newer than the last consumed marker (the Δ for Σ/Δ prompting).
    pub fn delta_text(&self) -> String {
        let marker = self.last_delta_ms;
        self.join(|s| s.at_ms > marker)
    }

    /// Mark the delta as consumed up to `now_ms` (called after an evaluation).
    pub fn mark_delta_consumed(&mut self, now_ms: u64) {
        self.last_delta_ms = now_ms;
    }

    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn clear(&mut self) {
        self.segments.clear();
        self.last_delta_ms = 0;
    }

    fn join(&self, keep: impl Fn(&Segment) -> bool) -> String {
        self.segments
            .iter()
            .filter(|s| keep(s))
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingest_and_read_window_text() {
        let mut w = TranscriptWindow::new(60_000);
        w.ingest("hello", 1_000);
        w.ingest("world", 2_000);
        assert_eq!(w.window_text(), "hello world");
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn evicts_segments_older_than_max_age() {
        let mut w = TranscriptWindow::new(10_000);
        w.ingest("old", 1_000);
        w.ingest("mid", 8_000);
        // now_ms 12_000: cutoff = 2_000, so "old" (1_000) falls off.
        w.ingest("new", 12_000);
        assert_eq!(w.window_text(), "mid new");
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn pause_does_not_clear_the_buffer() {
        let mut w = TranscriptWindow::new(10_000);
        w.ingest("kept", 1_000);
        // No further ingest for a long time (a pause). Eviction only runs on
        // ingest, so the buffer is preserved.
        assert_eq!(w.window_text(), "kept");
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn delta_is_only_material_after_the_marker() {
        let mut w = TranscriptWindow::new(60_000);
        w.ingest("a", 1_000);
        w.ingest("b", 2_000);
        assert_eq!(w.delta_text(), "a b"); // marker starts at 0
        w.mark_delta_consumed(2_000);
        assert_eq!(w.delta_text(), ""); // nothing newer than 2_000 yet
        w.ingest("c", 3_000);
        assert_eq!(w.delta_text(), "c");
    }

    #[test]
    fn clear_empties_everything() {
        let mut w = TranscriptWindow::new(60_000);
        w.ingest("x", 1_000);
        w.clear();
        assert!(w.is_empty());
        assert_eq!(w.delta_text(), "");
    }
}
