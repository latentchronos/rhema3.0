use std::time::Instant;

use crate::types::{Detection, DetectionSource};

/// Default confidence threshold — detections below this are dropped.
const DEFAULT_CONFIDENCE_THRESHOLD: f64 = 0.45;

/// Default auto-queue threshold — detections above this are auto-queued.
const DEFAULT_AUTO_QUEUE_THRESHOLD: f64 = 0.80;

/// Default cooldown in milliseconds between auto-displayed results.
const DEFAULT_COOLDOWN_MS: u64 = 2500;

/// Operator-facing decision for a merged detection.
#[derive(Debug, Clone)]
pub struct DetectionDecision {
    pub raw_score: f64,
    pub minimum_threshold: f64,
    pub auto_queue_threshold: f64,
    pub decision: &'static str,
    pub explanation: String,
}

/// A detection after merging, with an auto-queue flag and review metadata.
#[derive(Debug, Clone)]
pub struct MergedDetection {
    pub detection: Detection,
    pub auto_queued: bool,
    pub decision: DetectionDecision,
}

/// Merges results from direct reference detection and semantic search
/// into a single ranked list.
///
/// # Dedup strategy
/// When both direct and semantic detectors match the same verse
/// (same `book_number` + `chapter` + `verse_start`), the direct detection
/// is kept because it has higher trust (confidence >= 0.90).
///
/// # Auto-queue
/// High-confidence results are marked `auto_queued = true` so the UI
/// can display them immediately. A cooldown timer prevents flooding
/// the user with too many auto-displayed results.
pub struct DetectionMerger {
    confidence_threshold: f64,
    auto_queue_threshold: f64,
    cooldown_ms: u64,
    last_auto_display: Option<Instant>,
}

impl DetectionMerger {
    pub fn new() -> Self {
        Self {
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
            auto_queue_threshold: DEFAULT_AUTO_QUEUE_THRESHOLD,
            cooldown_ms: DEFAULT_COOLDOWN_MS,
            last_auto_display: None,
        }
    }

    /// Merge direct and semantic detections into a ranked list.
    ///
    /// 1. Combine all detections.
    /// 2. Dedup: if direct and semantic found the same verse, keep direct.
    /// 3. Sort by confidence descending.
    /// 4. Drop anything below `confidence_threshold`.
    /// 5. Mark `auto_queued = true` for items above `auto_queue_threshold`.
    /// 6. Apply cooldown: if last auto-display was < cooldown_ms ago,
    ///    don't auto-queue.
    pub fn merge(
        &mut self,
        direct: Vec<Detection>,
        semantic: Vec<Detection>,
    ) -> Vec<MergedDetection> {
        // 1. Combine
        let mut all: Vec<Detection> = Vec::with_capacity(direct.len() + semantic.len());
        all.extend(direct);

        // 2. Dedup: only add semantic detections whose verse is not already
        //    present from the direct pass.
        for s in semantic {
            let dominated = all.iter().any(|d| {
                matches!(d.source, DetectionSource::DirectReference)
                    && d.verse_ref.book_number == s.verse_ref.book_number
                    && d.verse_ref.chapter == s.verse_ref.chapter
                    && d.verse_ref.verse_start == s.verse_ref.verse_start
            });
            if !dominated {
                all.push(s);
            }
        }

        // 3. Sort by confidence descending
        all.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 4. Drop below threshold. Each source can require a stricter
        // threshold than the global baseline.
        all.retain(|d| {
            let policy = source_policy(&d.source);
            d.confidence >= self.confidence_threshold.max(policy.minimum_threshold)
        });

        // 5 & 6. Build merged list with auto-queue decisions
        let now = Instant::now();
        let cooldown_ok = match self.last_auto_display {
            Some(last) => now.duration_since(last).as_millis() as u64 >= self.cooldown_ms,
            None => true,
        };

        let mut results = Vec::with_capacity(all.len());
        for detection in all {
            let policy = source_policy(&detection.source);
            let minimum_threshold = self.confidence_threshold.max(policy.minimum_threshold);
            let auto_queue_threshold = self.auto_queue_threshold.max(policy.auto_queue_threshold);
            let score = raw_score(&detection);
            let meets_auto_threshold = detection.confidence >= auto_queue_threshold;
            let auto_queued = meets_auto_threshold && cooldown_ok;
            if auto_queued {
                self.last_auto_display = Some(now);
            }
            let decision = DetectionDecision {
                raw_score: score,
                minimum_threshold,
                auto_queue_threshold,
                decision: if auto_queued {
                    "auto_queued"
                } else {
                    "review_required"
                },
                explanation: decision_explanation(
                    &detection.source,
                    detection.confidence,
                    auto_queue_threshold,
                    auto_queued,
                    meets_auto_threshold,
                    cooldown_ok,
                ),
            };
            results.push(MergedDetection {
                detection,
                auto_queued,
                decision,
            });
        }

        results
    }

    /// Apply context boosting to a list of detections.
    ///
    /// Boosts confidence for detections in the same book/chapter as
    /// the current sermon context. Call this BEFORE `merge()`.
    pub fn apply_context_boost(
        detections: &mut [Detection],
        context: &crate::context::SermonContext,
    ) {
        for detection in detections.iter_mut() {
            let boost = context
                .confidence_boost(detection.verse_ref.book_number, detection.verse_ref.chapter);
            if boost > 0.0 {
                detection.confidence = (detection.confidence + boost).min(1.0);
            }
        }
    }

    /// Update the minimum confidence threshold.
    pub fn set_confidence_threshold(&mut self, threshold: f64) {
        self.confidence_threshold = threshold;
    }

    /// Update the auto-queue threshold.
    pub fn set_auto_queue_threshold(&mut self, threshold: f64) {
        self.auto_queue_threshold = threshold;
    }

    /// Update the cooldown between auto-displayed results.
    pub fn set_cooldown_ms(&mut self, ms: u64) {
        self.cooldown_ms = ms;
    }
}

#[derive(Debug, Clone, Copy)]
struct SourcePolicy {
    minimum_threshold: f64,
    auto_queue_threshold: f64,
}

fn source_policy(source: &DetectionSource) -> SourcePolicy {
    match source {
        DetectionSource::DirectReference => SourcePolicy {
            minimum_threshold: 0.45,
            auto_queue_threshold: 0.90,
        },
        DetectionSource::Contextual => SourcePolicy {
            minimum_threshold: 0.45,
            auto_queue_threshold: 0.80,
        },
        DetectionSource::QuotationMatch { .. } => SourcePolicy {
            minimum_threshold: 0.45,
            auto_queue_threshold: 0.85,
        },
        DetectionSource::SemanticLocal { .. } => SourcePolicy {
            minimum_threshold: 0.50,
            auto_queue_threshold: 0.92,
        },
        DetectionSource::SemanticCloud { .. } => SourcePolicy {
            minimum_threshold: 0.55,
            auto_queue_threshold: 0.90,
        },
    }
}

fn raw_score(detection: &Detection) -> f64 {
    match detection.source {
        DetectionSource::QuotationMatch { similarity }
        | DetectionSource::SemanticLocal { similarity }
        | DetectionSource::SemanticCloud { similarity } => similarity,
        DetectionSource::DirectReference | DetectionSource::Contextual => detection.confidence,
    }
}

fn source_label(source: &DetectionSource) -> &'static str {
    match source {
        DetectionSource::DirectReference => "direct reference",
        DetectionSource::Contextual => "reading context",
        DetectionSource::QuotationMatch { .. } => "quotation match",
        DetectionSource::SemanticLocal { .. } => "local semantic search",
        DetectionSource::SemanticCloud { .. } => "cloud semantic search",
    }
}

fn decision_explanation(
    source: &DetectionSource,
    confidence: f64,
    auto_queue_threshold: f64,
    auto_queued: bool,
    meets_auto_threshold: bool,
    cooldown_ok: bool,
) -> String {
    let label = source_label(source);
    if auto_queued {
        return format!(
            "{label} confidence {:.0}% met the {:.0}% auto-queue threshold.",
            confidence * 100.0,
            auto_queue_threshold * 100.0,
        );
    }

    if !meets_auto_threshold {
        return format!(
            "{label} confidence {:.0}% is below the {:.0}% auto-queue threshold; operator review required.",
            confidence * 100.0,
            auto_queue_threshold * 100.0,
        );
    }

    if !cooldown_ok {
        return format!(
            "{label} confidence {:.0}% met the auto-queue threshold, but cooldown prevented another automatic queue action.",
            confidence * 100.0,
        );
    }

    format!("{label} requires operator review.")
}

impl Default for DetectionMerger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DetectionSource, VerseRef};

    fn make_detection(
        book_number: i32,
        book_name: &str,
        chapter: i32,
        verse_start: i32,
        confidence: f64,
        source: DetectionSource,
    ) -> Detection {
        Detection {
            verse_ref: VerseRef {
                book_number,
                book_name: book_name.to_string(),
                chapter,
                verse_start,
                verse_end: None,
            },
            verse_id: None,
            confidence,
            source,
            transcript_snippet: format!("{} {}:{}", book_name, chapter, verse_start),
            detected_at: 0,
        }
    }

    #[test]
    fn test_merger_dedup_keeps_direct() {
        let mut merger = DetectionMerger::new();

        let direct = vec![make_detection(
            43,
            "John",
            3,
            16,
            0.96,
            DetectionSource::DirectReference,
        )];
        let semantic = vec![make_detection(
            43,
            "John",
            3,
            16,
            0.72,
            DetectionSource::SemanticLocal { similarity: 0.72 },
        )];

        let results = merger.merge(direct, semantic);
        assert_eq!(results.len(), 1);
        assert!(matches!(
            results[0].detection.source,
            DetectionSource::DirectReference
        ));
        assert!((results[0].detection.confidence - 0.96).abs() < f64::EPSILON);
    }

    #[test]
    fn test_merger_keeps_distinct_verses() {
        let mut merger = DetectionMerger::new();

        let direct = vec![make_detection(
            43,
            "John",
            3,
            16,
            0.96,
            DetectionSource::DirectReference,
        )];
        let semantic = vec![make_detection(
            45,
            "Romans",
            8,
            28,
            0.65,
            DetectionSource::SemanticLocal { similarity: 0.65 },
        )];

        let results = merger.merge(direct, semantic);
        assert_eq!(results.len(), 2);
        // Sorted by confidence descending
        assert_eq!(results[0].detection.verse_ref.book_name, "John");
        assert_eq!(results[1].detection.verse_ref.book_name, "Romans");
    }

    #[test]
    fn test_merger_drops_below_threshold() {
        let mut merger = DetectionMerger::new();

        let direct = vec![];
        let semantic = vec![
            make_detection(
                43,
                "John",
                3,
                16,
                0.50,
                DetectionSource::SemanticLocal { similarity: 0.50 },
            ),
            make_detection(
                45,
                "Romans",
                8,
                28,
                0.20, // below 0.35 threshold
                DetectionSource::SemanticLocal { similarity: 0.20 },
            ),
        ];

        let results = merger.merge(direct, semantic);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].detection.verse_ref.book_name, "John");
    }

    #[test]
    fn test_merger_auto_queue() {
        let mut merger = DetectionMerger::new();

        let direct = vec![make_detection(
            43,
            "John",
            3,
            16,
            0.96,
            DetectionSource::DirectReference,
        )];

        let results = merger.merge(direct, vec![]);
        assert_eq!(results.len(), 1);
        // 0.96 >= 0.80 auto_queue_threshold and no cooldown yet
        assert!(results[0].auto_queued);
    }

    #[test]
    fn test_merger_auto_queue_below_threshold() {
        let mut merger = DetectionMerger::new();

        let semantic = vec![make_detection(
            43,
            "John",
            3,
            16,
            0.50,
            DetectionSource::SemanticLocal { similarity: 0.50 },
        )];

        let results = merger.merge(vec![], semantic);
        assert_eq!(results.len(), 1);
        // 0.50 < semantic-local 0.92 auto_queue_threshold
        assert!(!results[0].auto_queued);
    }

    #[test]
    fn test_semantic_high_confidence_auto_queue_uses_source_policy() {
        let mut merger = DetectionMerger::new();

        let semantic = vec![make_detection(
            43,
            "John",
            3,
            16,
            0.93,
            DetectionSource::SemanticLocal { similarity: 0.93 },
        )];

        let results = merger.merge(vec![], semantic);
        assert_eq!(results.len(), 1);
        assert!(results[0].auto_queued);
        assert_eq!(results[0].decision.decision, "auto_queued");
        assert!((results[0].decision.auto_queue_threshold - 0.92).abs() < f64::EPSILON);
    }

    #[test]
    fn test_semantic_below_source_minimum_is_dropped() {
        let mut merger = DetectionMerger::new();

        let semantic = vec![make_detection(
            43,
            "John",
            3,
            16,
            0.48,
            DetectionSource::SemanticLocal { similarity: 0.48 },
        )];

        let results = merger.merge(vec![], semantic);
        assert!(results.is_empty());
    }

    #[test]
    fn test_merger_sort_order() {
        let mut merger = DetectionMerger::new();

        let direct = vec![make_detection(
            43,
            "John",
            3,
            16,
            0.90,
            DetectionSource::DirectReference,
        )];
        let semantic = vec![
            make_detection(
                45,
                "Romans",
                8,
                28,
                0.95,
                DetectionSource::SemanticLocal { similarity: 0.95 },
            ),
            make_detection(
                1,
                "Genesis",
                1,
                1,
                0.60,
                DetectionSource::SemanticLocal { similarity: 0.60 },
            ),
        ];

        let results = merger.merge(direct, semantic);
        assert_eq!(results.len(), 3);
        // Highest confidence first
        assert!((results[0].detection.confidence - 0.95).abs() < f64::EPSILON);
        assert!((results[1].detection.confidence - 0.90).abs() < f64::EPSILON);
        assert!((results[2].detection.confidence - 0.60).abs() < f64::EPSILON);
    }

    #[test]
    fn test_merger_empty_inputs() {
        let mut merger = DetectionMerger::new();
        let results = merger.merge(vec![], vec![]);
        assert!(results.is_empty());
    }
}
