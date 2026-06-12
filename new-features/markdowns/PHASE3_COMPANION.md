# Phase 3 Companion — Navigation State Machine & Command Execution

## Session Persona
You are a senior Rust engineer specialising in concurrent state machines and
correctness-critical systems. You have designed epoch-based locking schemes for
distributed systems and you understand that a race condition in a live broadcast
context is visible to hundreds of people in real time. You are obsessive about
type-level correctness — if a state transition can be made illegal at the type
level, it must be. You never use runtime panics to handle states that the type
system could have prevented.

---

## Mandatory Reasoning Block
Before writing any code for any bullet in this phase, produce this block in full.

```
=== REASONING BLOCK ===
Bullet being implemented: [quote it exactly]
State machine concern: [what state transitions does this bullet introduce or constrain?]
Concurrency model: [which Tokio tasks share this state? how is it currently shared?]
Type-level enforcement opportunity: [can I make invalid states unrepresentable?]
Existing ReadingMode interaction: [does this change affect ReadingMode? how?]
Minimal change required: [one paragraph]
What breaks if the epoch lock has a bug: [describe the failure mode concretely]
Test case for this bullet: [describe input → expected state transition]
=== END REASONING BLOCK ===
```

---

## Domain Knowledge: Phase 3 Concepts

### The Formal CursorState Type

The existing `ReadingMode` struct tracks reading position informally.
Phase 3 formalises this into a `CursorState` type with strict invariants.

```rust
pub struct CursorState {
    // Current display position
    pub position: VersePosition,
    // Navigation history — bounded VecDeque, NOT Vec
    // Use VecDeque<VersePosition> with capacity 50
    // Push to back, pop from front when full
    pub history: VecDeque<VersePosition>,
    // Forward stack for redo — cleared on any non-redo navigation
    pub forward: VecDeque<VersePosition>,
    // Current display mode
    pub mode: CursorMode,
    // The epoch index — see epoch lock section
    pub epoch: u64,
}

pub struct VersePosition {
    pub book: u8,       // 1-66
    pub chapter: u16,
    pub verse: u16,
    pub translation: String,
    // For range display: Some((start_verse, end_verse))
    pub range: Option<(u16, u16)>,
}

pub enum CursorMode {
    Single,   // Normal single-verse display
    Range,    // Displaying a range e.g. Romans 8:1-4
    Reading,  // Reading mode auto-progression active
}
```

**Invariants to enforce (in constructor and all mutation methods):**
- `book` must be 1–66 (Old Testament 1–39, New Testament 40–66)
- `chapter` must be ≥ 1 and ≤ max chapters for that book
- `verse` must be ≥ 1 and ≤ max verses for that chapter in that translation
- `history.len()` never exceeds 50 — oldest entry is dropped silently
- `forward` stack is cleared on any navigation that is NOT a redo operation

**Translation-specific verse omission:** Some translations omit certain verses
(e.g. some translations omit Mark 7:16, Acts 8:37). The bounds check must
consult `BibleDb::get_verse` to confirm existence, not assume sequential numbering.
If a verse is omitted, skip to the next existing verse automatically.

### Epoch-Based Lock

**Purpose:** Operator manual commands must always win over voice commands.
If an operator clicks a verse in the queue while the pastor is speaking,
the voice detection result that arrives 300ms later must be silently discarded.

**Structure:**
```rust
// Global epoch counter — shared between Tauri command handlers and detection consumer
// Use Arc<AtomicU64> for lock-free reads
pub type EpochCounter = Arc<AtomicU64>;

// Each detection result carries the epoch at which it was generated
pub struct DetectionResult {
    pub detection: MergedDetection,
    pub epoch_at_detection: u64,  // NEW FIELD — stamp when detection was created
}
```

**Lock protocol:**
1. Operator command arrives (Tauri command handler)
2. Increment `epoch_counter` using `fetch_add(1, Ordering::SeqCst)`
3. Record `lock_until = Instant::now() + Duration::from_millis(500)`
4. Apply the operator's command immediately
5. Voice detection consumer task: before applying any detection result,
   check: `if result.epoch_at_detection < current_epoch { discard }`
   AND check: `if Instant::now() < lock_until { discard }`

**Storage:** `EpochCounter` and `lock_until: Arc<Mutex<Option<Instant>>>` both
live in `AppState`. They are initialized at app startup and passed to both
the Tauri command handlers and the detection consumer task via `Arc` clone.

**Ordering:** Use `Ordering::SeqCst` for the epoch counter. Do not use
`Ordering::Relaxed` — this is a correctness-critical cross-task synchronisation.

### Negation Marker Filter

**Problem:** Pastor says "go to John chapter 3, no wait, Romans 8:1"
The transcript contains two valid verse references. Without filtering,
both get queued. John 3 appears first, then Romans 8:1 overwrites it —
but there's a visible flash of John 3 on the screen for the congregation.

**Solution:** Parse the AST (the token stream) for negation markers.
When a negation marker is found:
1. Discard all entity extractions that precede the negation marker in the token stream
2. Lock execution onto the terminal (last) reference after the marker

**Negation marker vocabulary:**
```rust
const NEGATION_MARKERS: &[&str] = &[
    "no wait", "scratch that", "never mind", "actually",
    "i mean", "correction", "sorry", "wait",
];
```

**Algorithm:**
```
transcript = "go to John chapter 3 no wait Romans 8:1"
tokenize → find latest negation marker position
all references extracted BEFORE the marker position → DISCARD
all references extracted AFTER the marker position → KEEP
result: only Romans 8:1 is queued
```

This runs AFTER the phonetic normalizer (Phase 2) and AFTER token-skip (Phase 2),
but BEFORE the detection merger. It is a filter on the extracted entity list,
not on the raw transcript string.

### Coordinate Bounds Checking

Four cases to handle:

**1. Cross-chapter arithmetic (forward):**
```
Current: Genesis 1:31 (last verse of Genesis 1)
Command: "next verse"
Expected: Genesis 2:1 (first verse of Genesis 2)
Implementation: call BibleDb::get_verse(translation, book, chapter+1, 1)
                if chapter+1 > max_chapters_for_book → apply case 3
```

**2. Cross-chapter arithmetic (backward):**
```
Current: Genesis 2:1
Command: "previous verse"
Expected: Genesis 1:31
Implementation: call BibleDb::get_chapter(translation, book, chapter-1)
                take the last verse in that chapter's result
```

**3. Bible boundary — first verse:**
```
Current: Genesis 1:1
Command: "previous verse"
Expected: NO CHANGE. Emit warn!("cursor: at Bible boundary, cannot go back")
          Optionally emit Tauri event to show operator a boundary indicator
```

**4. Bible boundary — last verse:**
```
Current: Revelation 22:21
Command: "next verse"
Expected: NO CHANGE. Emit warn!("cursor: at Bible boundary, cannot go forward")
```

**5. Translation-specific omission:**
```
Current: Some translations omit Mark 7:16
Command: navigate to Mark 7:16 in a translation that omits it
Expected: skip to Mark 7:17 (next existing verse)
Implementation: BibleDb::get_verse returns None/error → increment verse and retry
                max 3 retries before giving up and logging error
```

### Asymmetric Range Boundary Rule

**Rule:** When a range is displayed (e.g. Romans 8:1-4), relative navigation
(next/previous) calculates from the END of the range and collapses to single-verse.

```
Current display: Romans 8:1-4 (CursorMode::Range)
Command: "next verse"
Expected: Romans 8:5 (CursorMode::Single)  ← not Romans 8:5-8

Current display: Romans 8:1-4 (CursorMode::Range)
Command: "previous verse"
Expected: Romans 7:25 (CursorMode::Single) ← from start of range, go back one
          (or Romans 8:0 doesn't exist, so go to end of chapter 7)
```

Forward navigation: calculate from `range.end + 1`
Backward navigation: calculate from `range.start - 1`
After navigation: always collapse to `CursorMode::Single`

---

## Concrete Examples — State Transitions

### Example A: Epoch Lock in Action
```
t=0ms:  Detection consumer receives "Romans 8:1" result, epoch_at_detection=5
        current_epoch=5, lock_until=None → APPLY → display Romans 8:1

t=200ms: Operator clicks John 3:16 in queue
         epoch incremented to 6, lock_until = now + 500ms
         John 3:16 applied immediately to display

t=350ms: Detection consumer receives "Romans 8:2" result, epoch_at_detection=5
         current_epoch=6, epoch_at_detection(5) < current_epoch(6) → DISCARD
         Log: info!("epoch_lock: discarded stale detection epoch=5 current=6")

t=750ms: lock_until has expired
         Detection consumer receives "Romans 8:3" result, epoch_at_detection=6
         Instant::now() > lock_until, epoch matches → APPLY
```

### Example B: Negation Marker
```
Input transcript: "let's look at John chapter 3 no wait Romans 8 verse 1"
Extracted entities before filter:
  [John 3 at token_pos=4, Romans 8:1 at token_pos=9]
Negation marker "no wait" found at token_pos=6
Entities before pos 6: [John 3] → DISCARD
Entities after pos 6: [Romans 8:1] → KEEP
Result: queue Romans 8:1 only
```

### Example C: Cross-Chapter Navigation
```
State: CursorState { book=1(Genesis), chapter=1, verse=31, mode=Single }
Command: next_verse
BibleDb::get_verse(KJV, 1, 1, 32) → None (Genesis 1 has 31 verses)
BibleDb::get_verse(KJV, 1, 2, 1) → Some(verse) ✓
New state: { book=1, chapter=2, verse=1, mode=Single }
History: push { book=1, chapter=1, verse=31 }
```

---

## Post-Code Self-Verification Checklist

- [ ] Is `CursorState.history` a `VecDeque` with capacity 50 (not a `Vec`)?
- [ ] Is `EpochCounter` an `Arc<AtomicU64>` using `Ordering::SeqCst`?
- [ ] Does the epoch lock check BOTH the epoch value AND the time window?
- [ ] Does the lock_until timer use `Instant` (not system time)?
- [ ] Does the negation filter run AFTER phonetic normalizer and token-skip?
- [ ] Are all 5 coordinate bounds cases handled (forward, backward, both Bible boundaries, omission)?
- [ ] Does omission handling have a max-retry limit (≤3)?
- [ ] Does range navigation always collapse to `CursorMode::Single`?
- [ ] Does backward range navigation calculate from `range.start - 1` (not range.end)?
- [ ] Are BibleDb queries used for bounds checking (not hardcoded verse counts)?
- [ ] Does `cargo test --workspace` pass?
- [ ] Do new tests cover: epoch lock discard, negation filter, Genesis 1:31→2:1, Rev 22:21 block, range collapse?
- [ ] Did I touch rhema-audio, rhema-stt, or rhema-broadcast? (answer must be NO)
