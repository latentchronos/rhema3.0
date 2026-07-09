//! The Observer (§5): an application-side scheduler + state machine. It decides
//! WHEN to evaluate (fixed interval, plus an optional topic-shift early trigger —
//! D5), frames each call as Σ (current understanding) + Δ (new transcript) — D4,
//! and applies the two-outcome decision (§5.4). It never talks to a model and
//! never reads a clock: the app passes `now_ms` and calls a `ComprehensionModel`.

use crate::schema::Decision;
use crate::summary::{MicroRecord, RollingSummary};
use crate::types::{ComprehensionState, StateTransition};
use crate::window::TranscriptWindow;

/// Scheduling policy (§5.2, D5).
#[derive(Debug, Clone)]
pub struct ObserverConfig {
    /// Fixed refresh interval — the CPU-budget ceiling (§5.2).
    pub interval_ms: u64,
    /// Opt-in: also evaluate early on a sharp topic shift (D5).
    pub topic_shift_trigger: bool,
    /// Cosine-distance jump that counts as a shift (only if enabled).
    pub topic_shift_threshold: f32,
}

impl Default for ObserverConfig {
    fn default() -> Self {
        Self {
            interval_ms: 60_000,
            topic_shift_trigger: false,
            topic_shift_threshold: 0.35,
        }
    }
}

pub struct Observer {
    current: Option<ComprehensionState>,
    current_started_ms: u64,
    summary: RollingSummary,
    window: TranscriptWindow,
    config: ObserverConfig,
    last_eval_ms: u64,
    window_age_ms: u64,
    summary_max_records: usize,
}

impl Observer {
    pub fn new(config: ObserverConfig, window_age_ms: u64, summary_max_records: usize) -> Self {
        Self {
            current: None,
            current_started_ms: 0,
            summary: RollingSummary::new(summary_max_records),
            window: TranscriptWindow::new(window_age_ms),
            config,
            last_eval_ms: 0,
            window_age_ms,
            summary_max_records,
        }
    }

    /// Feed a final transcript segment into the rolling window.
    pub fn ingest_segment(&mut self, text: &str, at_ms: u64) {
        self.window.ingest(text, at_ms);
    }

    /// Should the observer consult the model now? True when the interval has
    /// elapsed, or (if enabled) a topic shift exceeds the threshold. Never fires
    /// on an empty window.
    pub fn should_evaluate(&self, now_ms: u64, topic_shift: Option<f32>) -> bool {
        if self.window.is_empty() {
            return false;
        }
        let interval_due = now_ms.saturating_sub(self.last_eval_ms) >= self.config.interval_ms;
        let shift_due = self.config.topic_shift_trigger
            && topic_shift.is_some_and(|s| s >= self.config.topic_shift_threshold);
        interval_due || shift_due
    }

    /// Build the Σ/Δ prompt body: current understanding + sermon arc + new
    /// transcript only (D4). The schema instruction is attached by the caller.
    pub fn build_prompt(&self) -> String {
        let current = match &self.current {
            Some(s) => serde_json::to_string(s).unwrap_or_else(|_| "null".to_string()),
            None => "null".to_string(),
        };
        let arc = self.summary.render();
        let delta = self.window.delta_text();
        format!(
            "CURRENT_UNDERSTANDING:\n{current}\n\nSERMON_ARC:\n{arc}\n\nNEW_TRANSCRIPT:\n{delta}"
        )
    }

    /// Apply the model's decision (§5.4). Marks this tick as evaluated and the
    /// delta as consumed. `NO_CHANGE` returns `None`; `STATE_CHANGED` closes the
    /// previous state into the rolling summary, becomes current, and returns the
    /// recorded transition.
    pub fn apply_decision(&mut self, decision: Decision, now_ms: u64) -> Option<StateTransition> {
        self.last_eval_ms = now_ms;
        self.window.mark_delta_consumed(now_ms);
        match decision {
            Decision::NoChange => None,
            Decision::StateChanged { new_state } => {
                if let Some(prev) = &self.current {
                    self.summary.record(MicroRecord {
                        intent: prev.intent,
                        passages: prev.passages.clone(),
                        start_ms: self.current_started_ms,
                        end_ms: now_ms,
                    });
                }
                let from = self.current.take();
                self.current = Some(new_state.clone());
                self.current_started_ms = now_ms;
                Some(StateTransition {
                    from,
                    to: new_state,
                    at_ms: now_ms,
                })
            }
        }
    }

    pub fn current(&self) -> Option<&ComprehensionState> {
        self.current.as_ref()
    }

    /// Reset all working memory for a new service (§14C).
    pub fn clear_session(&mut self) {
        self.current = None;
        self.current_started_ms = 0;
        self.last_eval_ms = 0;
        self.window = TranscriptWindow::new(self.window_age_ms);
        self.summary = RollingSummary::new(self.summary_max_records);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DominantIntent, PassageRef};

    fn observer() -> Observer {
        Observer::new(ObserverConfig::default(), 60_000, 4)
    }

    fn state(intent: DominantIntent, passages: &[&str]) -> ComprehensionState {
        ComprehensionState {
            intent,
            supporting_activities: vec![],
            passages: passages.iter().map(|p| PassageRef::new(*p)).collect(),
            topic: None,
            confidence: 0.9,
        }
    }

    #[test]
    fn does_not_evaluate_with_an_empty_window() {
        let o = observer();
        assert!(!o.should_evaluate(999_999, None));
    }

    #[test]
    fn does_not_evaluate_before_the_interval() {
        let mut o = observer();
        o.ingest_segment("hello", 1_000);
        // last_eval_ms starts at 0; interval is 60_000.
        assert!(!o.should_evaluate(30_000, None));
    }

    #[test]
    fn evaluates_once_the_interval_elapses() {
        let mut o = observer();
        o.ingest_segment("hello", 1_000);
        assert!(o.should_evaluate(60_000, None));
    }

    #[test]
    fn topic_shift_triggers_early_when_enabled() {
        let cfg = ObserverConfig {
            interval_ms: 60_000,
            topic_shift_trigger: true,
            topic_shift_threshold: 0.35,
        };
        let mut o = Observer::new(cfg, 60_000, 4);
        o.ingest_segment("hello", 1_000);
        assert!(o.should_evaluate(5_000, Some(0.5))); // big shift, early
        assert!(!o.should_evaluate(5_000, Some(0.1))); // small shift, wait
    }

    #[test]
    fn topic_shift_ignored_when_disabled() {
        let mut o = observer(); // topic_shift_trigger = false
        o.ingest_segment("hello", 1_000);
        assert!(!o.should_evaluate(5_000, Some(0.99)));
    }

    #[test]
    fn build_prompt_includes_current_arc_and_delta() {
        let mut o = observer();
        o.ingest_segment("the father ran to him", 1_000);
        let prompt = o.build_prompt();
        assert!(prompt.contains("CURRENT_UNDERSTANDING"), "got: {prompt}");
        assert!(prompt.contains("null"), "no current state yet, got: {prompt}");
        assert!(prompt.contains("the father ran to him"), "got: {prompt}");
    }

    #[test]
    fn no_change_keeps_state_and_records_nothing() {
        let mut o = observer();
        o.ingest_segment("hello", 1_000);
        let t = o.apply_decision(Decision::NoChange, 60_000);
        assert!(t.is_none());
        assert!(o.current().is_none());
    }

    #[test]
    fn state_changed_records_a_transition_and_becomes_current() {
        let mut o = observer();
        o.ingest_segment("the prodigal son", 1_000);
        let new = state(DominantIntent::StoryTelling, &["Luke 15"]);
        let t = o
            .apply_decision(Decision::StateChanged { new_state: new.clone() }, 60_000)
            .expect("a transition");
        assert_eq!(t.from, None);
        assert_eq!(t.to, new);
        assert_eq!(t.at_ms, 60_000);
        assert_eq!(o.current(), Some(&new));
    }

    #[test]
    fn second_transition_carries_previous_as_from_and_advances_the_clock() {
        let mut o = observer();
        o.ingest_segment("story", 1_000);
        let first = state(DominantIntent::StoryTelling, &["Luke 15"]);
        o.apply_decision(Decision::StateChanged { new_state: first.clone() }, 60_000);

        // Interval must elapse again relative to the last evaluation (60_000).
        assert!(!o.should_evaluate(90_000, None));
        assert!(o.should_evaluate(120_000, None));

        o.ingest_segment("now apply it", 110_000);
        let second = state(DominantIntent::Applying, &[]);
        let t = o
            .apply_decision(Decision::StateChanged { new_state: second.clone() }, 120_000)
            .expect("a transition");
        assert_eq!(t.from, Some(first));
        assert_eq!(t.to, second);
    }

    #[test]
    fn clear_session_resets_state_window_and_clock() {
        let mut o = observer();
        o.ingest_segment("story", 1_000);
        o.apply_decision(
            Decision::StateChanged { new_state: state(DominantIntent::StoryTelling, &["Luke 15"]) },
            60_000,
        );
        o.clear_session();
        assert!(o.current().is_none());
        assert!(!o.should_evaluate(60_000, None)); // window empty again
    }
}
