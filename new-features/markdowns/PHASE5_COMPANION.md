# Phase 5 Companion — Sermon Intelligence & Semantic Suggestion

## Session Persona
You are a senior ML systems engineer and Rust developer who has shipped
embedding-based recommendation systems in production. You understand that
an AI suggestion appearing unbidden on a congregation's projector screen
is a catastrophic UX failure — the system must be conservative to the point
of near-silence. You also understand topic vector mathematics well enough to
know that naive averaging of embeddings destroys temporal relevance, and that
a decaying window is not the same as a sliding average. You implement the
math correctly or you flag it and ask.

---

## Mandatory Reasoning Block
Before writing any code for any bullet in this phase, produce this block in full.

```
=== REASONING BLOCK ===
Bullet being implemented: [quote it exactly]
ML concern: [what is the mathematical operation this introduces?
             write the formula before writing any code]
State management: [where does this state live? how long does it persist?
                   what resets it?]
Operator safety check: [what prevents this from auto-projecting to the congregation?
                        name the specific gate]
Integration point: [which existing struct/function does this attach to?]
Test case: [what input transcript sequence produces what suggestion output?]
=== END REASONING BLOCK ===
```

---

## Domain Knowledge: Phase 5 Concepts

### Exponential Time-Decay Window on Topic Vector

**The problem with naive averaging:**
If the pastor preaches on "grace" for 20 minutes, then switches to "judgment"
for 5 minutes, a naive average of the transcript embeddings still strongly
represents "grace." Suggestions remain stuck on grace-related verses even
though the sermon has clearly moved on.

**The solution: exponential time-decay**

The `SermonContext` maintains a topic vector `v` that is a weighted sum of
sentence embeddings, where recent sentences have exponentially higher weight.

```
decay_factor(age_seconds) = exp(-λ * age_seconds)

where λ = ln(2) / half_life_seconds
      half_life = 90 seconds (configurable)
      
This means a sentence from 90 seconds ago has 50% the weight of a current sentence.
A sentence from 180 seconds ago has 25% the weight.
A sentence from 5 minutes ago has ~6% the weight.
```

**The 75/25 implementation:**

Rather than computing continuous decay over the full history (expensive),
use a two-block approximation:

```
active_block:     transcript from last 90 seconds → weight = 0.75
background_block: transcript from 90s–∞ ago      → weight = 0.25

topic_vector = 0.75 * mean(active_embeddings) + 0.25 * mean(background_embeddings)
```

This is the "focus factor" described in the spec.

**Data structure:**
```rust
pub struct SermonContext {
    // Each entry: (embedding: Vec<f32>, timestamp: Instant)
    active_window: VecDeque<(Vec<f32>, Instant)>,
    background_window: VecDeque<(Vec<f32>, Instant)>,
    // Cached topic vector, recomputed when windows change
    topic_vector: Option<Vec<f32>>,
    // Flag: topic_vector needs recomputation
    dirty: bool,
}
```

**Update logic (called on each new transcript sentence):**
1. Compute sentence embedding via the existing `SemanticDetector` ONNX model
2. Push (embedding, Instant::now()) to `active_window`
3. Drain entries from `active_window` older than 90s → push to `background_window`
4. Set `dirty = true`

**Topic vector computation (lazy, called when suggestion needed):**
```rust
fn compute_topic_vector(&mut self) -> Option<Vec<f32>> {
    if self.active_window.is_empty() { return None; }
    
    let active_mean = mean_embedding(&self.active_window.iter()
        .map(|(e, _)| e).collect::<Vec<_>>());
    
    let topic = if self.background_window.is_empty() {
        active_mean
    } else {
        let bg_mean = mean_embedding(&self.background_window.iter()
            .map(|(e, _)| e).collect::<Vec<_>>());
        // 75/25 weighted sum
        active_mean.iter().zip(bg_mean.iter())
            .map(|(a, b)| 0.75 * a + 0.25 * b)
            .collect()
    };
    
    Some(topic)
}
```

**Memory management:** Cap `background_window` at 500 entries. When full,
drop the oldest. The semantic meaning degrades gracefully — we don't need
a perfect representation of the entire sermon history.

### Pre-Service Priming Index

**Purpose:** The pastor uploads or pastes their sermon notes before the service.
Notes contain explicit verse references (e.g. "Romans 8:1-4", "John 3:16").
These references receive a +0.25 scalar boost in semantic search rankings.

**Implementation:**
```rust
pub struct PrimingIndex {
    // Verse references explicitly mentioned in pastor's notes
    primed_verses: HashSet<VerseRef>,
}

impl PrimingIndex {
    // Parse the notes text, extract VerseRefs using DirectDetector
    pub fn build_from_notes(notes: &str, detector: &DirectDetector) -> Self
    
    // Apply boost to a ranked list of search results
    pub fn apply_boost(&self, results: &mut Vec<(VerseRef, f32)>) {
        for (verse_ref, score) in results.iter_mut() {
            if self.primed_verses.contains(verse_ref) {
                *score = (*score + 0.25).min(1.0);
            }
        }
        // Re-sort after boost
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    }
}
```

The priming index is built once before the service starts and stored in `AppState`.
It is rebuilt if the pastor updates their notes mid-service.

### Proactive Suggestion Engine

**The hard rules (non-negotiable):**
1. Suggestions are NEVER auto-projected to `AudienceChannel`
2. Maximum 1 suggestion per 5 minutes (300 seconds)
3. Never suggest a verse already in `display_history` (verses shown this service)
4. Operator dismissals suppress that specific verse for the rest of the service
5. Suggestions surface ONLY to `OperatorChannel`

**Trigger condition:**
- A suggestion fires when `SemanticDetector.search_query(topic_vector)` returns
  a result with cosine similarity > 0.72 that:
  - Is NOT in the display history
  - Is NOT dismissed
  - The last suggestion was > 300 seconds ago

**Data structure:**
```rust
pub struct SuggestionEngine {
    last_suggestion_at: Option<Instant>,
    dismissed_verses: HashSet<VerseRef>,
    // Verses displayed this service (populated from AudienceChannel updates)
    display_history: HashSet<VerseRef>,
    cooldown: Duration, // 300 seconds
    similarity_threshold: f32, // 0.72
}
```

**Output:**
```rust
pub struct SuggestedVerse {
    pub verse_ref: VerseRef,
    pub confidence: f32,
    pub reason: String, // e.g. "Thematically related to current sermon context"
}
```

This is emitted via `OperatorChannel` only — never via `AudienceChannel`.

---

## Concrete Examples — Sermon Intelligence

### Example A: Time-Decay Window Shift
```
t=0-20min:   Pastor preaches on "grace and forgiveness"
             active_window: 15 sentences about grace
             background_window: empty
             topic_vector: strongly represents "grace"
             Suggestion engine: proposes Ephesians 2:8 (grace-related)

t=21min:     active_window sentences from t=0-11min drain to background
             active_window: now contains 5 sentences about "judgment"  
             background_window: 10 sentences about grace
             topic_vector = 0.75 * judgment_mean + 0.25 * grace_mean
             → topic has SHIFTED toward judgment
             Suggestion engine: now proposes Romans 2:5 (judgment-related)
```

### Example B: Priming Boost
```
Pastor notes contain: "Romans 8:1-4, John 3:16, Galatians 5:22"
PrimingIndex.primed_verses = {Romans 8:1, Romans 8:2, Romans 8:3, Romans 8:4,
                               John 3:16, Galatians 5:22}

Semantic search returns:
  [(Romans 8:1, 0.81), (Hebrews 12:1, 0.79), (John 3:16, 0.76)]

After priming boost:
  Romans 8:1: 0.81 + 0.25 = 1.0 (capped)
  John 3:16:  0.76 + 0.25 = 1.0 (capped)
  Hebrews 12:1: 0.79 (no boost, not in notes)

Re-ranked: [(Romans 8:1, 1.0), (John 3:16, 1.0), (Hebrews 12:1, 0.79)]
```

### Example C: Suggestion Engine Rules
```
t=0min:     Last suggestion: none. Display history: {}. Dismissed: {}
            Topic vector search → Ephesians 2:8 (similarity 0.84) ✓
            Emit suggestion to OperatorChannel: "Ephesians 2:8 (84%)"
            last_suggestion_at = now

t=3min:     Topic vector search → Romans 5:8 (similarity 0.79)
            Time since last suggestion: 180s < 300s cooldown → SUPPRESS
            No suggestion emitted

t=8min:     Cooldown expired (>300s). Romans 5:8 (similarity 0.79) ✓
            Operator dismisses it → dismissed_verses.insert(Romans 5:8)
            
t=9min:     Topic vector search → Romans 5:8 again
            In dismissed_verses → SUPPRESS forever this service

t=10min:    Operator displays Ephesians 2:8 (from queue, not from suggestion)
            display_history.insert(Ephesians 2:8)
            
t=15min:    Topic vector search → Ephesians 2:8 (similarity 0.91)
            In display_history → SUPPRESS
```

---

## Integration with Existing Code

The existing `SermonContext` struct in `rhema-detection` is the attachment point
for the time-decay window. Read its current implementation before modifying it.

The existing `SemanticDetector` has `search_query` — use this for topic vector
search. Pass the `topic_vector` as the query embedding.

The `SuggestionEngine` belongs in the app-level detection consumer (same place as
the suppression cache from Phase 2), not inside `rhema-detection`.

The `PrimingIndex` needs a new Tauri command: `set_sermon_notes(notes: String)`
that parses the notes and stores the priming index in `AppState`.

---

## Post-Code Self-Verification Checklist

- [ ] Is the time-decay formula using the 75/25 two-block approximation (not full decay)?
- [ ] Does `active_window` drain entries older than 90s to `background_window` on each update?
- [ ] Is `background_window` capped at 500 entries?
- [ ] Is the topic vector computed lazily (only when a suggestion is needed)?
- [ ] Does `PrimingIndex::build_from_notes` use `DirectDetector` for parsing?
- [ ] Does the priming boost cap at 1.0 (not exceed it)?
- [ ] Is the suggestion cooldown exactly 300 seconds?
- [ ] Does the suggestion engine check display_history before proposing?
- [ ] Does the suggestion engine check dismissed_verses before proposing?
- [ ] Are suggestions emitted ONLY via `OperatorChannel` — never `AudienceChannel`?
- [ ] Is there a Tauri command `set_sermon_notes` that rebuilds the priming index?
- [ ] Does `cargo test --workspace` pass?
- [ ] Do tests cover: decay window shift, priming boost, cooldown suppression, dismissal?
- [ ] Is the Stage 2 LLM channel from Phase 2 now wired to the real Claude API call?
  (if rhema-api is now implemented — otherwise leave the placeholder and note it)
