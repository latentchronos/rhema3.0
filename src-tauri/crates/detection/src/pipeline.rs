use crate::direct::detector::DirectDetector;
use crate::merger::{DetectionMerger, MergedDetection};
use crate::semantic::cloud::CloudBooster;
use crate::semantic::detector::SemanticDetector;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

/// Stage-1 intent classification bucket for an utterance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntentClass {
    /// Operational command (navigation/display) — handled locally, no LLM.
    ControlCommand,
    /// A concrete scripture reference was detected (confidence ≥ threshold).
    ExplicitScriptureRequest,
    /// Neither — natural language that needs the Stage-2 LLM fallback.
    Ambiguous,
}

/// Minimum direct-detection confidence for an utterance to count as an explicit
/// scripture request in Stage 1.
const EXPLICIT_REQUEST_CONFIDENCE: f64 = 0.70;

/// Multi-word control-command patterns, matched as case-insensitive substrings.
const CONTROL_COMMAND_PATTERNS: &[&str] = &[
    "next verse",
    "previous verse",
    "go back",
    "go forward",
    "clear screen",
    "clear the screen",
    "blank screen",
    "blank the screen",
    "hide verse",
    "show next",
    "scroll down",
    "scroll up",
];

/// Bare control words, only treated as a command inside a short utterance so
/// ordinary speech ("we'll be back after worship") is not misclassified.
const CONTROL_WORDS: &[&str] = &["next", "previous", "back", "clear", "hide", "blank"];
const CONTROL_WORD_MAX_WORDS: usize = 3;

/// The main detection pipeline that runs on each transcript segment.
///
/// Orchestrates direct reference detection, semantic search, cloud boost,
/// and merging into a single call. Consumers should create one pipeline
/// and reuse it across transcript segments so that the merger's cooldown
/// state is preserved.
pub struct DetectionPipeline {
    pub direct: DirectDetector,
    pub semantic: SemanticDetector,
    pub cloud: CloudBooster,
    pub merger: DetectionMerger,
    /// Sender for Stage-2 (LLM fallback) — ambiguous transcripts are queued here.
    stage2_tx: UnboundedSender<String>,
    /// Receiver, taken once by the app which spawns [`run_stage2_placeholder`].
    stage2_rx: Option<UnboundedReceiver<String>>,
}

impl DetectionPipeline {
    pub fn new() -> Self {
        let (stage2_tx, stage2_rx) = unbounded_channel();
        Self {
            direct: DirectDetector::new(),
            semantic: SemanticDetector::stub(),
            cloud: CloudBooster::new(),
            merger: DetectionMerger::new(),
            stage2_tx,
            stage2_rx: Some(stage2_rx),
        }
    }

    /// Process a transcript segment and return merged detections.
    ///
    /// 1. Run direct reference detection (pattern / automaton based).
    /// 2. Run semantic detection (returns empty if no model loaded).
    /// 3. TODO: Cloud boost for low-confidence semantic results
    ///    (will be wired when reqwest is added).
    /// 4. Merge and rank all results.
    /// Run the full pipeline (direct + semantic + merge). Used by `detect_verses` command.
    pub fn process(&mut self, text: &str) -> Vec<MergedDetection> {
        let direct_results = self.direct.detect(text);

        // Skip semantic on short fragments (no signal in < 5 words)
        let semantic_results = if text.split_whitespace().count() >= 5 {
            self.semantic.detect(text)
        } else {
            vec![]
        };

        let merged = self.merger.merge(direct_results, semantic_results);
        self.route_stage2(text, &merged);
        merged
    }

    /// Run only direct (regex/pattern) detection. Instant, no ONNX inference.
    /// Used during live transcription on every is_final fragment.
    pub fn process_direct(&mut self, text: &str) -> Vec<MergedDetection> {
        let direct_results = self.direct.detect(text);
        self.merger.merge(direct_results, vec![])
    }

    /// Run only semantic (ONNX embedding) detection. Slow, 50-400ms.
    /// Used on speech_final only, in a background task.
    pub fn process_semantic(&mut self, text: &str) -> Vec<MergedDetection> {
        if text.split_whitespace().count() < 5 {
            return vec![];
        }
        let semantic_results = self.semantic.detect(text);
        self.merger.merge(vec![], semantic_results)
    }

    /// Check if semantic search is available (model loaded + index populated).
    pub fn has_semantic(&self) -> bool {
        self.semantic.is_ready()
    }

    /// Check if cloud boost is available (API key configured).
    pub fn has_cloud(&self) -> bool {
        self.cloud.is_enabled()
    }

    /// Enable or disable synonym expansion (paraphrase detection mode).
    pub fn set_use_synonyms(&mut self, enabled: bool) {
        self.semantic.set_use_synonyms(enabled);
    }

    /// Returns whether synonym expansion is currently enabled.
    pub fn use_synonyms(&self) -> bool {
        self.semantic.use_synonyms()
    }

    /// Stage-1 classification: bucket an utterance into one of [`IntentClass`].
    ///
    /// Precedence is scripture-first: a concrete reference (any detection with
    /// confidence ≥ [`EXPLICIT_REQUEST_CONFIDENCE`]) is an explicit request even
    /// if phrased as a command; otherwise a control pattern → ControlCommand;
    /// otherwise Ambiguous (which routes to the Stage-2 fallback).
    pub fn classify(&self, text: &str, detections: &[MergedDetection]) -> IntentClass {
        if detections
            .iter()
            .any(|d| d.detection.confidence >= EXPLICIT_REQUEST_CONFIDENCE)
        {
            return IntentClass::ExplicitScriptureRequest;
        }
        if is_control_command(text) {
            return IntentClass::ControlCommand;
        }
        IntentClass::Ambiguous
    }

    /// Take the Stage-2 receiver (once). The app drains it with
    /// [`run_stage2_placeholder`] inside the Tauri async runtime.
    pub fn take_stage2_receiver(&mut self) -> Option<UnboundedReceiver<String>> {
        self.stage2_rx.take()
    }

    /// Classify `text` and, if ambiguous, queue it for the Stage-2 fallback.
    fn route_stage2(&self, text: &str, detections: &[MergedDetection]) {
        if self.classify(text, detections) == IntentClass::Ambiguous {
            // Best-effort: if the receiver/task has gone away, drop it silently.
            let _ = self.stage2_tx.send(text.to_string());
        }
    }
}

/// Whether `text` looks like an operational control command (Stage 1).
fn is_control_command(text: &str) -> bool {
    let lower = text.to_lowercase();
    if CONTROL_COMMAND_PATTERNS.iter().any(|p| lower.contains(p)) {
        return true;
    }
    let words: Vec<&str> = lower.split_whitespace().collect();
    words.len() <= CONTROL_WORD_MAX_WORDS && words.iter().any(|w| CONTROL_WORDS.contains(w))
}

/// Stage-2 (LLM fallback) placeholder consumer. Logs each ambiguous transcript
/// queued by the pipeline so the channel and its consumer are wired end-to-end.
///
/// # Phase 5, Bullet 5.4 — deliberately still a placeholder
///
/// The roadmap wires this to a real Claude call *iff* `rhema-api` is implemented.
/// It is currently an empty stub crate, so per the bullet the placeholder is
/// retained. Turning it into the real Stage-2 fallback requires, in order:
///
/// 1. **Implement `rhema-api`** as the Anthropic HTTP client. The project has no
///    official Anthropic Rust SDK available, so this is raw HTTP via `reqwest`
///    (`reqwest` is on the approved dependency list but is NOT yet a dependency
///    of `rhema-api`/`rhema-detection`/`app` — adding it to `rhema-api` needs
///    explicit approval, per the GEMINI dependency rule).
/// 2. **Endpoint / headers:** `POST https://api.anthropic.com/v1/messages`,
///    headers `x-api-key: <key>` and `anthropic-version: 2023-06-01`.
///    Model `claude-opus-4-8`; adaptive thinking; structured-output JSON
///    (`output_config.format`) returning `{intent, entities}`; latency budget
///    200–500ms (ARCHITECTURE §4.3 Stage-2). The API key currently lives in the
///    frontend `settings-store`; it must be surfaced to the backend (e.g. via
///    `AppState`) for this call.
/// 3. **Prompt:** the ambiguous utterance + current sermon context + cursor
///    position, as described in ARCHITECTURE §4.3.
/// 4. **Route the result** back through the existing detection/suggestion paths
///    (which are already operator-gated — Operator channel, epoch lock,
///    suppression); Stage-2 classifies only and never auto-projects.
///
/// Also note (carried from Phase 2): the LIVE stt path classifies via
/// `Mutex<DirectDetector>` directly, so `classify()` → `route_stage2()` only runs
/// through `DetectionPipeline::process`. Feeding the live flow into Stage-2 is a
/// separate wiring step that must land alongside the real Claude call.
pub async fn run_stage2_placeholder(mut rx: UnboundedReceiver<String>) {
    while let Some(transcript) = rx.recv().await {
        log::info!("stage2_fallback: queued '{}' (placeholder — see run_stage2_placeholder docs for real-Claude wiring)", transcript);
    }
}

impl Default for DetectionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_explicit_scripture_request() {
        let mut pipeline = DetectionPipeline::new();
        let detections = pipeline.process("John 3:16");
        assert!(!detections.is_empty());
        assert_eq!(
            pipeline.classify("John 3:16", &detections),
            IntentClass::ExplicitScriptureRequest
        );
    }

    #[test]
    fn classify_control_command() {
        let pipeline = DetectionPipeline::new();
        assert_eq!(pipeline.classify("next verse", &[]), IntentClass::ControlCommand);
        assert_eq!(pipeline.classify("clear the screen", &[]), IntentClass::ControlCommand);
        assert_eq!(pipeline.classify("go back", &[]), IntentClass::ControlCommand);
    }

    #[test]
    fn classify_ambiguous() {
        let pipeline = DetectionPipeline::new();
        assert_eq!(
            pipeline.classify("he is talking about love and forgiveness", &[]),
            IntentClass::Ambiguous
        );
    }

    #[test]
    fn control_word_only_in_short_utterance() {
        let pipeline = DetectionPipeline::new();
        // Bare command word in a terse utterance → control.
        assert_eq!(pipeline.classify("next", &[]), IntentClass::ControlCommand);
        // Same word buried in a sentence → not a command.
        assert_eq!(
            pipeline.classify("we will be back after the offering", &[]),
            IntentClass::Ambiguous
        );
    }

    #[test]
    fn ambiguous_transcript_routed_to_stage2_channel() {
        let mut pipeline = DetectionPipeline::new();
        let text = "he is talking about love and forgiveness";
        let _ = pipeline.process(text);
        let mut rx = pipeline.take_stage2_receiver().expect("receiver present");
        assert_eq!(rx.try_recv().unwrap(), text);
    }

    #[test]
    fn explicit_request_not_routed_to_stage2() {
        let mut pipeline = DetectionPipeline::new();
        let _ = pipeline.process("John 3:16");
        let mut rx = pipeline.take_stage2_receiver().expect("receiver present");
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_pipeline_direct_only() {
        let mut pipeline = DetectionPipeline::new();
        let results = pipeline.process("Jesus said in John 3:16 that God loved the world");
        assert!(!results.is_empty());
        assert_eq!(results[0].detection.verse_ref.book_name, "John");
        assert_eq!(results[0].detection.verse_ref.chapter, 3);
        assert_eq!(results[0].detection.verse_ref.verse_start, 16);
    }

    #[test]
    fn test_pipeline_no_match() {
        let mut pipeline = DetectionPipeline::new();
        let results = pipeline.process("The weather is nice today");
        assert!(results.is_empty());
    }

    #[test]
    fn test_pipeline_multiple_references() {
        let mut pipeline = DetectionPipeline::new();
        let results =
            pipeline.process("Compare John 3:16 with Romans 5:8 for understanding God's love");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_pipeline_semantic_not_ready_by_default() {
        let pipeline = DetectionPipeline::new();
        assert!(!pipeline.has_semantic());
    }

    #[test]
    fn test_pipeline_cloud_not_ready_by_default() {
        let pipeline = DetectionPipeline::new();
        assert!(!pipeline.has_cloud());
    }

    #[test]
    fn test_pipeline_auto_queue_for_direct() {
        let mut pipeline = DetectionPipeline::new();
        let results = pipeline.process("John 3:16");
        assert!(!results.is_empty());
        // Direct references have confidence >= 0.90 which is above the
        // default auto_queue_threshold (0.80), so should be auto-queued.
        assert!(results[0].auto_queued);
    }
}
