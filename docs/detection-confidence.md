# Detection Confidence

Rhema detection results are designed for live operators, not just machines. A detection should say what verse was found, how it was found, how confident the system is, and whether the operator should trust it automatically.

## Result Fields

Frontend detection payloads include:

- `confidence`: final confidence after detection-specific scoring and any merger/context behavior.
- `raw_score`: source-native score before operator display. For semantic and quotation sources this is the similarity score.
- `minimum_threshold`: minimum score needed for the detection to appear.
- `auto_queue_threshold`: score needed for automatic queueing.
- `decision`: `auto_queued` or `review_required`.
- `explanation`: short operator-facing reason for the decision.
- `source`: `direct`, `contextual`, `quotation`, `semantic_local`, or `semantic_cloud`.

## Source Policy

Current built-in thresholds:

| Source           | Minimum | Auto-Queue | Reason                                                                                       |
| ---------------- | ------: | ---------: | -------------------------------------------------------------------------------------------- |
| Direct reference |     45% |        90% | Explicit references are high-trust once parsed.                                              |
| Reading context  |     45% |        80% | Reading mode is constrained to the active passage.                                           |
| Quotation match  |     45% |        85% | Strong literal overlap is useful but can still need review.                                  |
| Local semantic   |     50% |        92% | Semantic guesses are valuable but should be conservative live.                               |
| Cloud semantic   |     55% |        90% | Cloud confirmation can be trusted more than local semantic alone, but still should be gated. |

The global merger thresholds still exist, but source-specific thresholds can require stricter behavior.

## Operator Behavior

`auto_queued` means the detection met its source-specific threshold and was not blocked by cooldown.

`review_required` means the detection is visible but should be manually approved before going live or into the queue. A common example is a semantic match that is plausible but below the high auto-queue threshold.

Cooldown can also turn a high-confidence result into `review_required` to prevent rapid automatic queue flooding.

## UI

The detections panel shows:

- source badge
- Auto/Review badge
- verse reference
- confidence percentage
- raw/minimum/auto threshold values
- explanation text

The goal is a fast operator answer to: "Why did Rhema suggest this, and should I trust it?"

## Future Work

- Add a settings UI for per-source thresholds.
- Persist threshold profiles for different service types.
- Include conflict explanations when multiple sources disagree.
- Feed these decisions into the Phase 6 sermon regression metrics.
