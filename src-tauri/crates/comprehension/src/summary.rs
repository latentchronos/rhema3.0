//! Rolling "sermon arc" summary (§17) — the Σ carried between observer ticks.
//! Keeps memory bounded over a multi-hour service: recent states stay as full
//! micro-records; older ones condense into a compact head string. The record
//! cap is sized to the active model's context (D6).

use crate::types::{DominantIntent, PassageRef};

/// One resolved state, written when a transition closes it (§17 per-state record).
#[derive(Debug, Clone, PartialEq)]
pub struct MicroRecord {
    pub intent: DominantIntent,
    pub passages: Vec<PassageRef>,
    pub start_ms: u64,
    pub end_ms: u64,
}

impl MicroRecord {
    /// Compact one-line rendering, e.g. `STORY_TELLING [Luke 15, Luke 19]`.
    pub fn render(&self) -> String {
        let intent = intent_label(self.intent);
        if self.passages.is_empty() {
            intent
        } else {
            let passages = self
                .passages
                .iter()
                .map(PassageRef::as_str)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{intent} [{passages}]")
        }
    }
}

/// The rolling summary: a condensed head plus the most recent full records.
#[derive(Debug, Clone)]
pub struct RollingSummary {
    head: String,
    records: Vec<MicroRecord>,
    max_records: usize,
}

impl RollingSummary {
    pub fn new(max_records: usize) -> Self {
        Self {
            head: String::new(),
            records: Vec::new(),
            max_records: max_records.max(1),
        }
    }

    /// Size the retained-record cap to the model's context window (§17, D6):
    /// roughly one record per 1k tokens, clamped to a sane [1, 8].
    pub fn sized_for(max_context_tokens: usize) -> Self {
        Self::new((max_context_tokens / 1000).clamp(1, 8))
    }

    pub fn max_records(&self) -> usize {
        self.max_records
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty() && self.head.is_empty()
    }

    /// Append a resolved state; condense the oldest into the head if over cap.
    pub fn record(&mut self, rec: MicroRecord) {
        self.records.push(rec);
        while self.records.len() > self.max_records {
            let oldest = self.records.remove(0);
            let condensed = oldest.render();
            if self.head.is_empty() {
                self.head = condensed;
            } else {
                self.head = format!("{}; {}", self.head, condensed);
            }
        }
    }

    /// Render the whole arc as one compact line (fed back in as Σ).
    pub fn render(&self) -> String {
        if self.is_empty() {
            return "(nothing yet)".to_string();
        }
        let mut parts: Vec<String> = Vec::new();
        if !self.head.is_empty() {
            parts.push(self.head.clone());
        }
        for r in &self.records {
            parts.push(r.render());
        }
        parts.join("; ")
    }

    pub fn clear(&mut self) {
        self.head.clear();
        self.records.clear();
    }
}

/// The SCREAMING_SNAKE_CASE label for an intent, reusing the serde contract so
/// there is a single source of truth for the names.
fn intent_label(intent: DominantIntent) -> String {
    serde_json::to_string(&intent)
        .unwrap_or_default()
        .trim_matches('"')
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(intent: DominantIntent, passages: &[&str], start: u64, end: u64) -> MicroRecord {
        MicroRecord {
            intent,
            passages: passages.iter().map(|p| PassageRef::new(*p)).collect(),
            start_ms: start,
            end_ms: end,
        }
    }

    #[test]
    fn micro_record_renders_intent_and_passages() {
        let r = rec(DominantIntent::StoryTelling, &["Luke 15", "Luke 19"], 0, 60_000);
        assert_eq!(r.render(), "STORY_TELLING [Luke 15, Luke 19]");
    }

    #[test]
    fn micro_record_without_passages_is_just_the_intent() {
        let r = rec(DominantIntent::Praying, &[], 0, 1);
        assert_eq!(r.render(), "PRAYING");
    }

    #[test]
    fn sized_for_maps_context_to_bounded_record_count() {
        assert_eq!(RollingSummary::sized_for(1_000).max_records(), 1);
        assert_eq!(RollingSummary::sized_for(8_000).max_records(), 8);
        assert_eq!(RollingSummary::sized_for(100_000).max_records(), 8); // clamped
        assert_eq!(RollingSummary::sized_for(0).max_records(), 1); // floor
    }

    #[test]
    fn empty_summary_renders_placeholder() {
        assert_eq!(RollingSummary::new(4).render(), "(nothing yet)");
    }

    #[test]
    fn condenses_oldest_records_beyond_the_cap_into_the_head() {
        let mut s = RollingSummary::new(1);
        s.record(rec(DominantIntent::StoryTelling, &["Luke 15"], 0, 60_000));
        s.record(rec(DominantIntent::Teaching, &["John 3:16"], 60_000, 120_000));
        // Only 1 full record retained; the older one is folded into the head.
        assert_eq!(s.len(), 1);
        let rendered = s.render();
        assert!(rendered.contains("STORY_TELLING [Luke 15]"), "got: {rendered}");
        assert!(rendered.contains("TEACHING [John 3:16]"), "got: {rendered}");
    }

    #[test]
    fn clear_resets_records_and_head() {
        let mut s = RollingSummary::new(1);
        s.record(rec(DominantIntent::StoryTelling, &["Luke 15"], 0, 1));
        s.record(rec(DominantIntent::Teaching, &["John 3"], 1, 2));
        s.clear();
        assert_eq!(s.len(), 0);
        assert_eq!(s.render(), "(nothing yet)");
    }
}
