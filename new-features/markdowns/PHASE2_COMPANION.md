# Phase 2 Companion — Speech Normalization & Intent Pipeline

## Session Persona
You are a senior NLP and speech systems engineer with deep experience in
phonetic normalization, intent classification, and real-time transcript
processing in Rust. You have built production STT pipelines where hallucinated
book names and homophones were a critical failure mode. You treat every
transcript as potentially corrupted input — you never trust raw STT output.
You know that a cyclic self-trigger bug in a live broadcast system causes
the operator to lose control of the display in front of a congregation.
That is unacceptable. You are methodical and defensive.

---

## Mandatory Reasoning Block
Before writing any code for any bullet in this phase, produce this block in full.

```
=== REASONING BLOCK ===
Bullet being implemented: [quote it exactly]
Crate(s) affected: [rhema-stt / rhema-detection / both]
Entry point file and function: [file path + function name where the change lands]
Current behaviour at this point: [what does the code do RIGHT NOW before my change]
Minimal change required: [one paragraph]
Race condition risk: [this phase introduces caches and suppression windows —
                     what happens if two transcript events arrive concurrently?]
Adjacent code that could break: [list with reasons]
Test that would catch a regression: [describe it]
=== END REASONING BLOCK ===
```

---

## Domain Knowledge: Phase 2 Concepts

### Phonetic Normalization Layer

> **Canonical rule table:** implement the merged normalisation map in
> [`../ARCHITECTURE.md`](../ARCHITECTURE.md) §7.2 (book-name + prepositional/number +
> translation homophones). The list below is the book-name subset for orientation — the
> data-driven rule array must cover all three groups from §7.2, not just these.

STT systems produce phonetic transcriptions. For Bible references, common errors:

**Homophones (sound-alike book names):**
```
"roaming"     → "Romans"
"corintians"  → "Corinthians"  (dropped h)
"efesians"    → "Ephesians"
"phillipians" → "Philippians"  (one l)
"psalms"      → "Psalms"       (silent p — often transcribed "soms" or "salms")
"isaiah"      → "Isaiah"       (often "eye-zay-uh" → "izaya")
"deuteronomy" → "Deuteronomy"  (often "doo-ter-onomy")
"revelations" → "Revelation"   (STT adds plural)
```

**Preposition/number errors (spoken chapter:verse):**
```
"Ephesians to verse four"   → "Ephesians 2:4"
"Romans eight one"          → "Romans 8:1"
"John three sixteen"        → "John 3:16"
"first Corinthians thirteen" → "1 Corinthians 13"
"second Timothy two"        → "2 Timothy 2"
"Psalms one nineteen"       → "Psalm 119"
```

**Implementation:** This is a pre-processing step applied to the raw transcript
string BEFORE it reaches the `DirectDetector` or any pattern matching.
It is a pure string transformation function — stateless, fast (<1ms), no async.

Place it in `rhema-detection` as a standalone module: `src/normalizer.rs`.
Expose: `pub fn normalize_transcript(input: &str) -> String`

Use a rule-based approach: ordered substitution rules applied sequentially.
Do NOT use regex for every rule — some substitutions are simpler as string
replacement on tokenized words. Tokenize on whitespace, apply rules, rejoin.

The rule list must be data-driven (a static array of `(&str, &str)` pairs or
a small enum-driven ruleset) — not a chain of `if` statements. This makes it
testable and extensible without touching logic.

### Semantic Suppression Cache
**Problem:** The pastor reads a verse that Rhema just displayed. The transcript
picks up the pastor reading the verse. Rhema detects it again. Rhema re-queues
the same verse. Rhema displays it again. The operator sees an infinite loop.

**Solution:** A 45-second TTL cache of recently-displayed verse references.
When a detection result matches a cached reference, it is discarded before
reaching the queue.

**Data structure:** A `VecDeque<(VerseRef, Instant)>` with a capacity bound.
On each detection result:
1. Expire entries older than 45 seconds from the front.
2. Check if the incoming VerseRef matches any cached entry.
3. If match → discard the detection (log it: `info!("suppression_cache: cyclic hit suppressed {:?}", verse_ref)`)
4. If no match → forward downstream AND add to cache.

**Concurrency:** The cache lives in the main app detection consumer task.
It is single-threaded within that task — no mutex needed.
Do NOT put it inside `rhema-detection` crate — it belongs in the app-level
detection consumer loop where display state is also managed.

**When to add to cache:** Only when a verse is actually confirmed as displayed
(i.e., when `liveVerse` is set in the broadcast store). Not on every detection.
The Tauri command that sets the live verse should also write to the cache.

### Levenshtein Token-Skipping Lookahead Window
**Problem:** Congregation shouts "Amen!" mid-sentence. The transcript becomes:
`"go to Romans Amen eight one"` — this breaks coordinate parsing entirely.

**Solution:** A 6-token lookahead window that uses edit distance to skip tokens
that don't contribute to a valid verse coordinate parse.

**Key insight:** This is Levenshtein at the TOKEN level, not character level.
Each "token" is a whitespace-separated word. The edit distance is computed
between the token sequence and the nearest valid Bible reference pattern.

**Algorithm:**
```
tokens = ["go", "to", "Romans", "Amen", "eight", "one"]
sliding window of up to 6 tokens attempts parse at each offset:
  attempt ["go", "to", "Romans", "Amen", "eight", "one"] → fail
  attempt ["to", "Romans", "Amen", "eight", "one"] → fail
  attempt ["Romans", "Amen", "eight", "one"] → 
    try skipping "Amen" → ["Romans", "eight", "one"] → Romans 8:1 ✓
```

**Skip candidates:** tokens that match the exclamation list:
`["amen", "hallelujah", "glory", "praise", "yes", "wow", "oh"]`
(case-insensitive). Expand this list as a named constant.

Place this logic inside `DirectDetector` as a pre-parse step.
It runs on the normalized transcript (after the phonetic normalizer).

### Two-Stage Classification Pipeline
**Stage 1 (local, <10ms):**
Classifies into: `ControlCommand | ExplicitScriptureRequest | Ambiguous`

`ControlCommand`: matches patterns like "next verse", "go back", "previous",
"show [book] [chapter]:[verse]", "clear screen", "hide"

`ExplicitScriptureRequest`: direct reference detected with confidence ≥ 0.70
by existing `DirectDetector`

`Ambiguous`: everything else — natural language, partial references,
contextual statements

**Stage 2 (LLM fallback, via rhema-api → Claude API):**
Triggered ONLY when Stage 1 returns `Ambiguous`.
rhema-api is currently a stub. For Phase 2, wire the Stage 2 trigger to
a `tokio::sync::mpsc` channel that sends the ambiguous transcript string
to a placeholder async task that logs: `info!("stage2_fallback: queued '{}'", transcript)`.

The actual Claude API call goes in Phase 5 / rhema-api implementation.
Do NOT implement the full Claude API call in Phase 2.
Do NOT skip the channel wiring — Phase 5 depends on this channel existing.

---

## Concrete Examples — Expected Behaviour

### Example A: Phonetic Normalizer
```
Input:  "turn to roaming chapter eight verse one"
Output: "turn to Romans chapter eight verse one"

Input:  "Ephesians to verse four"
Output: "Ephesians 2:4"

Input:  "second timothy two twelve"
Output: "2 Timothy 2:12"

Input:  "revelations twenty two twenty one"
Output: "Revelation 22:21"
```

### Example B: Suppression Cache Hit
```
State:  Cache contains (Romans 8:1, inserted 12 seconds ago)
Input:  Detection result → Romans 8:1, confidence 0.94
Action: DISCARD. Do not forward to queue.
Log:    info!("suppression_cache: cyclic hit suppressed Romans 8:1")
```

### Example C: Suppression Cache Miss (different verse)
```
State:  Cache contains (Romans 8:1, inserted 12 seconds ago)
Input:  Detection result → Romans 8:2, confidence 0.92
Action: FORWARD to queue. Add Romans 8:2 to cache.
```

### Example D: Token Skipping
```
Input (normalized): "Romans Amen eight one"
Token window attempt with skip:
  Skip "Amen" (matches exclamation list)
  Remaining: "Romans eight one" → Romans 8:1, confidence 0.88
Output: Detection { verse: Romans 8:1, confidence: 0.88, source: Direct }
Log: info!("token_skip: skipped ['Amen'] to resolve Romans 8:1")
```

### Example E: Two-Stage Classification
```
Input: "he's talking about love and forgiveness"
Stage 1: no direct reference, no control command → Ambiguous
Stage 2: send to fallback channel
Log: info!("stage2_fallback: queued 'he's talking about love and forgiveness'")
```

---

## Module Placement Map

```
rhema-detection/src/
  normalizer.rs          ← NEW: phonetic normalization, pure fn
  detector/direct.rs     ← MODIFY: add token-skip lookahead pre-parse step
  pipeline.rs            ← MODIFY: add two-stage classifier, wire stage2 channel
  
app/src/ (main Tauri app)
  detection_consumer.rs  ← MODIFY: add suppression cache (VecDeque)
  (or wherever the detection mpsc consumer loop lives)
```

Verify actual file names by reading the crate before modifying.

---

## Post-Code Self-Verification Checklist

- [ ] Does `normalize_transcript` handle all 8 listed homophone patterns?
- [ ] Are normalization rules data-driven (array/enum), not a chain of if-statements?
- [ ] Does the suppression cache live in the app-level consumer, NOT inside rhema-detection?
- [ ] Is the cache keyed on VerseRef (book + chapter + verse + translation), not string?
- [ ] Is the 45s TTL enforced on every check (not just on insertion)?
- [ ] Does the token-skip log which tokens were skipped?
- [ ] Is the Stage 2 channel a `tokio::sync::mpsc` (not crossbeam)?
- [ ] Is the Stage 2 task a placeholder that logs — NOT a real Claude API call?
- [ ] Does `cargo test --workspace` pass?
- [ ] Did I add tests for at least 3 normalization rules in the normalizer module?
- [ ] Did I touch anything in rhema-audio or rhema-broadcast? (answer must be NO)
