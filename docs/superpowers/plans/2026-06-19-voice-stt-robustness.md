# Voice / STT Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Disposable doc:** This plan captures a design discussion (June 2026). Once all six phases are merged, delete this file if it is cluttering the repo — the design intent lives in code + tests by then.

**Goal:** Make voice transcription, command recognition, and scripture detection stay correct as speech gets faster and as accent (Nigerian English) degrades the transcript — without sacrificing the deterministic, low-false-activation behavior that makes the system safe in front of a congregation.

**Architecture:** Six independent, individually-shippable phases. Adaptation lives client-side (we never re-tune Deepgram mid-stream). Precision comes from grammar + framing, not from a mandatory wake word. Everything new is bounded (clamped), reversible (undo), or observe-only (logging) so no phase can regress live behavior.

**Tech Stack:** Rust (Tauri backend, `detection` + `stt` + `audio` crates), Deepgram nova-3 streaming STT, ONNX Runtime (`ort`) for local semantic embeddings, React/Zustand frontend. **Reuse** the existing hand-rolled `levenshtein` in `direct/fuzzy.rs` (extract to a shared util — do NOT add a new edit-distance dependency). Add only `rphonetic` (Double Metaphone) as an additive phonetic signal.

## Global Constraints

- **Never drop the sermon.** No change may cause audio frames or `is_final` transcripts to be dropped. Cosmetic data (partials) may still be dropped under load; authoritative finals may not. (Matches existing "prime directive" in `stt.rs`.)
- **CPU is the default and must work standalone.** GPU execution providers are optional config only — never required, never the default.
- **No new ML models.** Phonetic/fuzzy work is deterministic string algorithms only.
- **Deterministic command path stays primary.** The LLM (Stage-2) is only ever a last-resort fallback, never the primary navigation path.
- **Bounded by construction.** Every adaptive value is clamped to a hard `[min, max]`; every new gate is observe-only until proven, mirroring the existing audio-gate "observe mode."
- **TDD, DRY, YAGNI, frequent commits.** Each task ends with a passing test and a commit.
- Build/test invariant: `cargo test -p rhema-detection`, `cargo test -p rhema-stt`, `cargo build` must stay green.

---

## File Structure (what each phase touches)

| Phase | Primary files | Responsibility |
|-------|---------------|----------------|
| 1 Instrumentation | `crates/detection/src/metrics.rs` (new), `src/commands/stt.rs` | Log pace, command fires, channel drops — evidence before tuning |
| 2 Adaptive segmentation | `crates/detection/src/sentence_buffer.rs`, `crates/detection/src/pace.rs` (new), `src/commands/stt.rs` | Self-tuning flush + max-word cap; wire the (currently dead) timeout |
| 3 Keyterm curation | `crates/stt/src/keyterms.rs`, `crates/stt/src/deepgram.rs` | Fix the 100-cap eviction; drop un-spoken abbreviations; boost command words at source (3.3) |
| 4 Phonetic correction | `crates/detection/src/scripture_phonetic.rs` (new), `crates/detection/src/normalizer.rs` | Post-transcription near-miss correction for scripture vocab |
| 5 Command robustness + precision | `crates/detection/src/voice_nav.rs`, `crates/detection/src/command_match.rs` (new), `crates/detection/src/textutil.rs` (shared levenshtein), `crates/detection/src/pipeline.rs`, `src/commands/detection.rs`, frontend | Require unit word; isolation gate; false-friend list; **nearest-slot accent match (all categories + numbers)**; undo; optional wake word |
| 6 Semantic throughput | `src/commands/stt.rs`, `crates/detection/src/semantic.rs` | CPU worker pool + batching; GPU optional |

---

# Phase 1 — Instrumentation (do this first)

**Why first:** You cannot tune pace constants (Phase 2) or build the accent-confusion table (Phase 5) without seeing what Deepgram actually returns. This phase adds observe-only logging and changes no behavior.

### Task 1.1: Pace + command-fire + drop logging

**Files:**
- Create: `src-tauri/crates/detection/src/metrics.rs`
- Modify: `src-tauri/crates/detection/src/lib.rs` (add `pub mod metrics;`)
- Test: in `metrics.rs` `#[cfg(test)]`

**Interfaces:**
- Produces: `pub fn log_command_attempt(raw: &str, parsed: Option<&str>)`, `pub fn log_pace(gap_secs: f64, smoothed_gap_secs: f64)`, `pub fn log_channel_drop(channel: &'static str)`.

- [ ] **Step 1: Write the failing test**
```rust
#[test]
fn formats_command_attempt_line() {
    let line = render_command_attempt("nest verse", Some("step:forward:verse:1"));
    assert!(line.contains("nest verse"));
    assert!(line.contains("step:forward:verse:1"));
    assert!(line.contains("MATCH"));
    let miss = render_command_attempt("next phase", None);
    assert!(miss.contains("MISS"));
}
```
- [ ] **Step 2: Run, expect fail** — `cargo test -p rhema-detection render_command_attempt` → FAIL (fn missing).
- [ ] **Step 3: Implement** `render_command_attempt(raw, parsed) -> String` (pure formatter) plus the `log_*` wrappers that call `log::info!` with a stable `voice_metrics:` prefix so transcripts are greppable. Keep formatting pure and tested; the `log_*` wrappers just call the formatter.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `feat(metrics): observe-only voice metrics logging`.

### Task 1.2: Wire the logging into the hot path

**Files:**
- Modify: `src-tauri/src/commands/stt.rs` — in `check_voice_command` (~L552) call `log_command_attempt`; at each `try_send` that can drop (partial/semantic/quotation) call `log_channel_drop` when the send errs.

- [ ] **Step 1:** Add the calls (no test — integration logging). Verify `cargo build` succeeds.
- [ ] **Step 2: Commit** — `feat(stt): emit voice metrics from detection hot path`.

**Manual validation:** Run the app, say "next verse" 10× in your own accent, `grep voice_metrics: <logfile>`. The MISS lines are the raw material for the Phase 5 confusion *supplement* (the nearest-slot matcher is the foundation; logs only seed the table that catches what sound-distance misses).

### Task 1.3: Deepgram dialect/model spike (source-level accent fix)

**Why:** The most general accent fix is at the source. Before compensating downstream, measure whether a Deepgram `language`/region/model option improves African (Nigerian) English transcription of the command words.

**Files:**
- Modify: `src-tauri/crates/stt/src/deepgram.rs` — DONE (commit `feat(stt): env-var model/language override + A/B spike procedure for accent`). `build_url` now reads `RHEMA_DG_MODEL` and `RHEMA_DG_LANGUAGE` env vars at runtime, defaulting to config values when unset. Default behavior is byte-for-byte unchanged. A `log::info!` line reports effective model + language on every connection.

#### Deepgram English model/language options (researched June 2026)

**Models available for streaming `/v1/listen`:**
| Model name | Notes |
|---|---|
| `nova-3` | Current recommended general-purpose ASR — best WER, supports most English dialect codes |
| `nova-3-medical` | Domain-specific; adds `en-CA`, `en-IE` vs base nova-3 |
| `nova-2` | Legacy; kept for unsupported language fallback; avoid for new work |
| `flux` | Conversational / turn-detection model; language code is just `en` |

**English dialect `language=` codes supported by nova-3:**
| Code | Dialect |
|---|---|
| `en` | English (auto, no region bias — recommended for mixed/unlisted accents) |
| `en-US` | American English |
| `en-GB` | British English |
| `en-AU` | Australian English |
| `en-IN` | Indian English (closest available to West African accent patterns) |
| `en-NZ` | New Zealand English |

**No dedicated African / Nigerian English code exists.** Deepgram does not offer `en-NG`, `en-GH`, `en-ZA` (streaming), or any explicit West African dialect code for nova-3. The two candidates most worth A/B-testing for Nigerian-accented English are:
1. `language=en` (unset region — the model picks the most likely dialect per utterance)
2. `language=en-IN` (Indian English; phonologically closer to West African English than en-US or en-GB due to shared non-rhotic features, vowel quality, and rhythm)

The current production setting (`language` unset → Deepgram defaults to `en`) is already the safe baseline.

**Recommended A/B pairs to test:**
- **Pair A (baseline):** `model=nova-3`, no `RHEMA_DG_LANGUAGE` set
- **Pair B (Indian proxy):** `model=nova-3`, `RHEMA_DG_LANGUAGE=en-IN`

Doc references: https://developers.deepgram.com/docs/models-languages-overview

- [x] **Step 1 — DONE:** Mechanism implemented in `deepgram.rs`. Env vars `RHEMA_DG_MODEL` / `RHEMA_DG_LANGUAGE` override model and language at runtime, default-off. `cargo build` passes.

- [ ] **Step 2 — PENDING USER ACTION: A/B measurement**

  **Fixed test phrase set (speak each phrase clearly, 10 times per run):**
  ```
  1.  next verse
  2.  previous verse
  3.  next chapter
  4.  previous chapter
  5.  chapter three verse seven
  6.  chapter one verse one
  7.  go to Genesis chapter two
  8.  clear screen
  9.  go forward
  10. go back
  11. verse fifteen
  12. John chapter three verse sixteen
  ```

  **Run A — Baseline (default settings):**
  ```bash
  # Start the app normally (no env overrides):
  cargo tauri dev   # or however you normally launch

  # In another terminal, tail the log and speak all 12 phrases 10x each:
  grep "voice_metrics:" <logfile> | tee /tmp/run_A.log

  # Count misses:
  grep "MISS" /tmp/run_A.log | wc -l
  # Record total attempts:
  wc -l /tmp/run_A.log
  ```
  > Log file location: check `~/.local/share/rhema3/rhema3.log` or the console output from `cargo tauri dev`.

  **Run B — Indian-English proxy:**
  ```bash
  RHEMA_DG_LANGUAGE=en-IN cargo tauri dev

  # Confirm the log shows:
  #   Deepgram effective model=nova-3 language=Some("en-IN") ...
  # Then speak the same 12 phrases 10x each:
  grep "voice_metrics:" <logfile> | tee /tmp/run_B.log
  grep "MISS" /tmp/run_B.log | wc -l
  wc -l /tmp/run_B.log
  ```

  **Optional Run C — Explicit en (no dialect bias):**
  ```bash
  RHEMA_DG_LANGUAGE=en cargo tauri dev
  grep "voice_metrics:" <logfile> | tee /tmp/run_C.log
  grep "MISS" /tmp/run_C.log | wc -l
  ```

  Record MISS count and total attempts for each run here:
  | Run | Env | MISS count | Total attempts | MISS rate |
  |-----|-----|-----------|----------------|-----------|
  | A | (baseline) | — | — | — |
  | B | `RHEMA_DG_LANGUAGE=en-IN` | — | — | — |
  | C | `RHEMA_DG_LANGUAGE=en` | — | — | — |

  Also spot-check scripture detection: say 3 scripture references ("Genesis chapter two", "John three sixteen", "Psalm twenty-three") per run and confirm they still display correctly (no regression).

- [ ] **Step 3 — PENDING USER ACTION: Decision gate**

  **Keep the setting if ALL of:**
  - MISS rate drops by ≥ 10 percentage points vs baseline (Run A), AND
  - Scripture detection shows no regression (references still resolve), AND
  - No new false activations on non-command sermon phrases.

  **Revert if none of the above is satisfied.** If reverting, set env var to unset (remove from launch config) and add a one-line note here. Either way, commit the outcome and this updated plan.

  **If a winning setting is found:** add `RHEMA_DG_LANGUAGE=<value>` to your `.env.local` / Tauri env config so it persists across launches. Then proceed to Phases 3.3 + 5 for downstream command robustness.

> **Future scope (NOT in this plan — YAGNI until needed):** per-pastor adaptive learning. Since commands come from a single speaker, the system could log misses + the operator's/pastor's corrections and grow a *personal* alias set over a few services, adapting to that speaker's idiolect. Revisit only if Phases 3.3 + 5 prove insufficient in real use.

---

# Phase 2 — Adaptive segmentation

**Why:** Fast continuous speech piles into run-on blobs; slow speech gets chopped. Today's 3 s timeout is dead code (`check_timeout` is never called). Build the self-tuning version and wire it on.

### Task 2.1: Pace estimator (time-aware EMA of inter-word gap)

**Files:**
- Create: `src-tauri/crates/detection/src/pace.rs`
- Modify: `src-tauri/crates/detection/src/lib.rs` (add `pub mod pace;`)
- Test: in `pace.rs`

**Interfaces:**
- Produces: `pub struct PaceEstimator` with `pub fn new(tau_secs: f64) -> Self`, `pub fn observe(&mut self, n_words: usize, span_secs: f64, dt_secs: f64)`, `pub fn gap_secs(&self) -> Option<f64>` (None until warm). `tau_secs` default 4.0.

- [ ] **Step 1: Write the failing test**
```rust
#[test]
fn warms_up_then_tracks_pace_change() {
    let mut p = PaceEstimator::new(4.0);
    assert_eq!(p.gap_secs(), None);                  // cold: not enough data
    // fast speaker: 5 words over 1.0s -> 0.25s/gap, feed a few samples
    for _ in 0..5 { p.observe(5, 1.0, 0.5); }
    let fast = p.gap_secs().unwrap();
    assert!(fast < 0.4, "fast gap should be small, got {fast}");
    // speaker slows: 3 words over 4.5s -> ~2.25s/gap
    for _ in 0..8 { p.observe(3, 4.5, 1.5); }
    let slow = p.gap_secs().unwrap();
    assert!(slow > 1.0, "should glide upward to slow, got {slow}");
}

#[test]
fn rejects_invalid_samples() {
    let mut p = PaceEstimator::new(4.0);
    p.observe(1, 0.0, 0.5);   // single word, no span -> ignored
    p.observe(0, 1.0, 0.5);   // no words -> ignored
    assert_eq!(p.gap_secs(), None);
}
```
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement.** Sample = `span_secs / (n_words - 1)`. Ignore samples where `n_words < 2` or `span_secs < 0.4`. Time-aware EMA: `alpha = 1.0 - (-dt_secs / tau).exp(); gap = alpha*sample + (1-alpha)*gap`. Warm-up: return `None` until at least 3 valid samples seen.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `feat(pace): time-aware EMA inter-word-gap estimator`.

### Task 2.2: Make SentenceBuffer adaptive + add max-word cap

**Files:**
- Modify: `src-tauri/crates/detection/src/sentence_buffer.rs`
- Test: extend its `#[cfg(test)]`

**Interfaces:**
- Consumes: `pace::PaceEstimator`.
- Produces: `SentenceBuffer::new()` unchanged in signature; new `pub fn set_adaptive_timeout(&mut self, gap_secs: Option<f64>)` → sets `flush_timeout_ms = clamp(2.5*gap, 0.8s, 5.0s)*1000`, falling back to 3000 ms when `None` (warm-up). New word-count cap: `append` returns `Some(flush)` once accumulated word count ≥ `MAX_WORDS` (20).

- [ ] **Step 1: Write failing tests**
```rust
#[test]
fn flushes_at_word_cap() {
    let mut buf = SentenceBuffer::new();
    let words = "a ".repeat(25);            // 25 words, no punctuation
    let out = buf.append(words.trim());
    assert!(out.is_some(), "should flush at word cap even with no punctuation");
}

#[test]
fn adaptive_timeout_is_clamped() {
    let mut buf = SentenceBuffer::new();
    buf.set_adaptive_timeout(Some(0.1));    // absurdly fast
    assert_eq!(buf.flush_timeout_ms(), 800);   // floor
    buf.set_adaptive_timeout(Some(10.0));   // absurdly slow
    assert_eq!(buf.flush_timeout_ms(), 5000);  // ceiling
    buf.set_adaptive_timeout(None);
    assert_eq!(buf.flush_timeout_ms(), 3000);  // warm-up default
}
```
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement.** Add `MAX_WORDS: usize = 20`; count words on each append (`split_whitespace().count()`), flush if `>= MAX_WORDS`. Add `set_adaptive_timeout` with the clamp and a `flush_timeout_ms()` getter. Keep punctuation flush first (natural boundary always wins).
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `feat(sentence-buffer): adaptive clamped timeout + word-count cap`.

### Task 2.3: Wire pace + periodic timeout into the detection worker

**Files:**
- Modify: `src-tauri/src/commands/stt.rs` — the detection-worker `tokio::select!` loop (~L339-407) and the Deepgram event consumer.

- [ ] **Step 1:** Construct one `PaceEstimator` in the worker. On each `Final`, compute `n_words`, `span_secs` (from the words' first/last timestamps — they're already in `TranscriptEvent::Final`), and `dt_secs` (time since last final via `Instant`), call `pace.observe(...)`, then `sentence_buf.set_adaptive_timeout(pace.gap_secs())` and `log_pace(...)`.
- [ ] **Step 2:** Add a `tokio::time::interval(Duration::from_millis(250))` branch to the `select!` that calls `sentence_buf.check_timeout()` and, on `Some`, routes the sentence to semantic detection — this is the dead-code fix that finally activates the timeout.
- [ ] **Step 3:** `cargo build`; manual: fast speech yields ≤20-word chunks, slow speech is not chopped (watch `voice_metrics:` pace lines).
- [ ] **Step 4: Commit** — `feat(stt): self-tuning sentence segmentation wired into worker`.

---

# Phase 3 — Keyterm curation (fixes the silent cap eviction)

**Why:** `core(5) + books(66) + abbreviations(48) = 119 > 100`, so spoken numbered books ("First John") and theological terms are **never sent**. Written abbreviations ("Jn", "Ps") are spoken by no one yet evict the useful terms.

### Task 3.1: Drop un-spoken abbreviations; order by spoken value

**Files:**
- Modify: `src-tauri/crates/stt/src/keyterms.rs`
- Test: in `keyterms.rs` `#[cfg(test)]`

- [ ] **Step 1: Write failing tests**
```rust
#[test]
fn spoken_numbered_books_survive_the_cap() {
    let terms = bible_keyterms();
    assert!(terms.iter().any(|t| t == "First John"));
    assert!(terms.iter().any(|t| t == "Second Corinthians"));
}
#[test]
fn drops_unspoken_written_abbreviations() {
    let terms = bible_keyterms();
    assert!(!terms.iter().any(|t| t == "Jn"));   // nobody says "Jn"
    assert!(!terms.iter().any(|t| t == "Ps"));
}
#[test]
fn total_stays_within_budget() {
    // leave headroom for the 5 core terms added in deepgram.rs
    assert!(bible_keyterms().len() <= 95);
}
```
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement.** Remove the `abbreviations` array entirely. Keep: 66 book names + spoken numbered forms + a trimmed, distinct theological set. Verify the total ≤ 95 so all of it (plus 5 core) fits under Deepgram's 100.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `fix(keyterms): stop evicting spoken book-forms with un-spoken abbreviations`.

### Task 3.2: Confirm Deepgram still sends everything

**Files:** Modify: `src-tauri/crates/stt/src/deepgram.rs` (only if the log line at L83 needs the count assertion).

- [ ] **Step 1:** No logic change expected (the dedup/cap loop already handles ≤100). Add a debug-build `debug_assert!(all_keyterms.len() <= 100)`.
- [ ] **Step 2:** `cargo build`; **Commit** — `chore(deepgram): assert keyterm budget`.

### Task 3.3: Boost the command words at the source (accent help, moderate)

**Why:** Fixing accent garble downstream (Phase 5) is good; making Deepgram hear the command words correctly in the first place is better and helps *every* accent. Add the reserved command words as keyterms — moderate, not hard.

**Files:**
- Modify: `src-tauri/crates/stt/src/keyterms.rs` (add a small `command_keyterms()` set) and `deepgram.rs` (include them, still under the 100 cap).
- Test: in `keyterms.rs`

- [ ] **Step 1: Write failing test**
```rust
#[test]
fn command_words_are_boosted() {
    let terms = command_keyterms();
    for w in ["verse", "chapter", "next", "previous", "forward", "clear"] {
        assert!(terms.iter().any(|t| t == w), "missing {w}");
    }
}
```
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** `pub fn command_keyterms() -> Vec<String>` with the reserved command words; include it in `deepgram.rs` ahead of theological terms but after book-forms, keeping total ≤ 100. (These words are sermon-context-expected, so moderate boosting will not cause meaningful false insertions; do NOT use heavy weights.)
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `feat(keyterms): boost reserved command words at the source for accent robustness`.

---

# Phase 4 — Phonetic correction for scripture vocabulary (no model)

**Why:** Recover mangled proper nouns ("Habakuk" → "Habakkuk") *after* transcription, never biasing the recognizer. Complements (does not replace) `normalizer.rs`.

### Task 4.1: Add deterministic phonetic deps + matcher

**Files:**
- Modify: `src-tauri/crates/detection/Cargo.toml` (add `rphonetic` ONLY — reuse the existing `levenshtein`)
- Create: `src-tauri/crates/detection/src/textutil.rs` (extract `levenshtein` here from `direct/fuzzy.rs`, re-export so `fuzzy.rs` still uses it — DRY)
- Create: `src-tauri/crates/detection/src/scripture_phonetic.rs`
- Modify: `src-tauri/crates/detection/src/lib.rs` (`pub mod textutil;`, `pub mod scripture_phonetic;`)
- Test: in the new files

**Interfaces:**
- Produces: `pub struct ScriptureCorrector` with `pub fn new() -> Self` (precomputes Double Metaphone codes for the scripture dictionary once) and `pub fn correct_token(&self, token: &str, context_allows: bool) -> Option<String>` — returns a canonical spelling only when the token is NOT a common English word, phonetically matches a scripture word, edit distance ≤ 2, and `context_allows` is true.

- [ ] **Step 1: Write failing tests**
```rust
#[test]
fn corrects_obvious_near_miss() {
    let c = ScriptureCorrector::new();
    assert_eq!(c.correct_token("habakuk", true).as_deref(), Some("Habakkuk"));
    assert_eq!(c.correct_token("hezzakiah", true).as_deref(), Some("Hezekiah"));
}
#[test]
fn leaves_clean_and_ambiguous_words_alone() {
    let c = ScriptureCorrector::new();
    assert_eq!(c.correct_token("mark", true), None);   // common word / ambiguous -> stop-list
    assert_eq!(c.correct_token("the", true), None);
    assert_eq!(c.correct_token("habakuk", false), None); // context gate closed
}
```
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement.** Static dictionary (book base names + commonly-mangled OT/NT names). A `STOPLIST` of scripture words that are also common English words ("mark", "job", "acts", "ruth", "hosea"? — keep conservative). Match logic: skip if in a common-English set or STOPLIST; compute Double Metaphone; find dictionary entries with equal primary code; among those require `textutil::levenshtein ≤ 2` (the shared util); return the closest. `context_allows=false` short-circuits to `None`.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `feat(detection): deterministic phonetic correction for scripture vocab`.

### Task 4.2: Apply correction ahead of normalization, context-gated

**Files:**
- Modify: `src-tauri/crates/detection/src/normalizer.rs` (call corrector token-wise before homophone pass) OR the direct-detection entry in `src/commands/stt.rs` — whichever owns the pre-detection text. Prefer `normalizer.rs` so all detection paths benefit.
- Test: in `normalizer.rs`

- [ ] **Step 1: Write failing test** — `normalize_transcript("turn to habakuk three")` yields a string containing `"Habakkuk"`; `normalize_transcript("please mark this day")` is unchanged (context gate / stop-list).
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement.** Define `context_allows` = a number token nearby OR "book of" precedes OR reading-mode active (pass a flag in). Run `correct_token` per token before the existing homophone substitution.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `feat(normalizer): context-gated phonetic correction pass`.

---

# Phase 5 — Command robustness + precision (the core design)

Implements: Way 1 (require unit word), Way 2 (isolation gate), Way 3 (false-friend list), accent-aware fuzzy command match, Way 4 (undo backstop), Way 5 (optional wake word, off by default). Phases 5.1–5.4 are coupled — require-unit-word only works for Nigerian English because the fuzzy matcher recovers the mangled unit word.

### Task 5.1: Accent-generalizing command-word matcher (nearest-slot, NOT a lookup table)

**Design intent (read first):** This must generalize to accents we have never logged, across ALL command categories (directions, units, clear, numbers) — not just the few confusions observed during testing. The foundation is **nearest-slot classification over the tiny reserved vocabulary**: given a token known to be part of a short isolated command attempt, return the reserved word it is *closest* to (by combined edit-distance + phonetic code) **only when that nearest word wins by a clear margin** over the runner-up. A curated confusion table is a *supplement* (catches non-phonetic quirks), never the foundation. Safety comes from the grammar + isolation gate in Tasks 5.2/5.3 — the matcher is only ever invoked on utterances already believed to be commands, so generous per-word matching cannot cause false activations.

**Files:**
- Create: `src-tauri/crates/detection/src/command_match.rs`
- Modify: `lib.rs`
- Consumes: `textutil::levenshtein` (shared, from Task 4.1), `rphonetic` Double Metaphone.
- Test: in the new file

**Interfaces:**
- Produces:
  - `pub fn canonical_command_word(token: &str) -> Option<&'static str>` — maps a raw/garbled token to one reserved command word (`next`, `previous`, `forward`, `back`, `verse`, `verses`, `chapter`, `chapters`, `clear`, `blank`, `hide`) or `None`.
  - `pub fn canonical_number_word(token: &str) -> Option<&'static str>` — same nearest-slot logic over the number-word set (`one`..`twenty`, tens, `hundred`, ordinals) so "tree"→"three", "tirty"→"thirty".
  - Internal `fn nearest_slot(token, slots) -> Option<&'static str>` shared by both: score each slot by `levenshtein` **and** Double-Metaphone-code equality; accept the best slot only if `best_score` is within an absolute ceiling AND beats the second-best by a margin (e.g. ≥ 1). Returns `None` when nothing is clearly nearest (ambiguous → reject).

- [ ] **Step 1: Write failing tests**
```rust
#[test]
fn exact_words_pass_through_all_categories() {
    assert_eq!(canonical_command_word("next"), Some("next"));
    assert_eq!(canonical_command_word("chapter"), Some("chapter"));
    assert_eq!(canonical_command_word("clear"), Some("clear"));
    assert_eq!(canonical_number_word("three"), Some("three"));
}
#[test]
fn nearest_slot_generalizes_to_unseen_garble() {
    // NOT in any hardcoded table — must be recovered by nearest-slot distance.
    assert_eq!(canonical_command_word("nest"), Some("next"));      // edit dist 1
    assert_eq!(canonical_command_word("phase"), Some("verse"));    // nearest among ~11 slots
    assert_eq!(canonical_command_word("chaptah"), Some("chapter"));
    assert_eq!(canonical_command_word("provious"), Some("previous"));
    assert_eq!(canonical_number_word("tree"), Some("three"));
    assert_eq!(canonical_number_word("tirty"), Some("thirty"));
}
#[test]
fn ambiguous_or_unrelated_is_rejected() {
    assert_eq!(canonical_command_word("worship"), None);   // not near any slot
    assert_eq!(canonical_command_word("hallelujah"), None);
    // a token roughly equidistant from two slots must reject, not guess:
    assert_eq!(canonical_command_word("verter"), None);    // no clear winner
}
```
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement.** Order: (1) exact match; (2) `nearest_slot` over the reserved set using `levenshtein` + Double Metaphone primary-code equality, with the margin rule above; (3) a small `CONFUSIONS: &[(&str,&str)]` *supplement* seeded from Phase-1 logs, consulted only when nearest-slot returns `None`. Keep all sets tiny (the safety property is "few targets, clear-margin nearest-neighbour").
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `feat(command-match): accent-generalizing nearest-slot command/number recovery`.

### Task 5.2: Require explicit unit word; apply fuzzy matcher; false-friend list (Ways 1+3)

**Files:**
- Modify: `src-tauri/crates/detection/src/voice_nav.rs` (`parse_nav_command`, `is_forward/is_backward/is_unit_word`)
- Test: extend its `#[cfg(test)]` (several existing tests change intentionally)

- [ ] **Step 1: Write failing tests**
```rust
#[test]
fn bare_direction_without_unit_is_rejected() {
    assert_eq!(parse_nav_command("next"), None);       // was Step(forward,verse,1)
    assert_eq!(parse_nav_command("back"), None);
}
#[test]
fn direction_plus_unit_still_works_via_fuzzy() {
    use NavDirection::*; use NavUnit::*;
    assert_eq!(parse_nav_command("next verse"), Some(step(Verse, Forward, 1)));
    assert_eq!(parse_nav_command("nest phase"),  Some(step(Verse, Forward, 1))); // accent
}
#[test]
fn false_friends_rejected() {
    assert_eq!(parse_nav_command("next week"), None);
    assert_eq!(parse_nav_command("next point"), None);
}
```
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement.** Normalize each token through `canonical_command_word` (and, in `parse_number`/its callers, through `canonical_number_word`) before classification, so accent-garbled directions, units, AND number words are all recovered. **Relative step now requires a unit word present** (remove the "default to Verse" path; return `None` if no unit). Add a `FALSE_FRIENDS: &[&str]` continuation set (`week, point, time, year, sunday, morning, …`); reject when a direction word is immediately followed by a false-friend. Update the existing tests that asserted bare "go forward"/"go back" → Step (those now require "go forward a verse" etc. — adjust intentionally and note in commit).
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `feat(voice-nav): require unit word + accent fuzzy + false-friend guard (Ways 1/3)`.

### Task 5.3: Isolation gating (Way 2)

**Files:**
- Modify: `src-tauri/src/commands/stt.rs` (`check_voice_command` call site) — only treat a final as a command candidate when it arrived as its own isolated utterance.

**Interfaces:**
- Consumes: the leading/trailing silence around the final. Deepgram already segments on silence; the simplest robust signal is "this final is the whole utterance" — i.e. it was delivered with `speech_final == true` and the buffered sentence equals just this fragment. Use that as the isolation proxy (no new timing math required).

- [ ] **Step 1:** Gate `check_voice_command` so it runs only on finals flagged isolated (standalone short utterance), not on fragments embedded mid-sentence. Add a unit test at the helper level if extracted into `voice_nav` (e.g. `is_isolated_command_context(transcript, speech_final)`).
- [ ] **Step 2:** `cargo build` + the helper test passes.
- [ ] **Step 3: Commit** — `feat(stt): isolation-gate command recognition (Way 2)`.

### Task 5.4: Undo backstop wired to cursor history (Way 4)

**Files:**
- Modify: `src-tauri/src/commands/detection.rs` (new `#[tauri::command] undo_navigation` calling `CursorState::back()`)
- Modify: `crates/detection/src/voice_nav.rs` (recognize "undo" / "undo that" / "go back" → a new `NavCommand::Undo` variant) — note "go back" currently maps to a backward step; decide precedence (recommend: "undo"/"undo that" → Undo; keep "go back" as step unless followed by "that").
- Modify: frontend `use-broadcast.ts` `applyVoiceNavCommand` to handle `undo`.
- Test: `voice_nav.rs` for the new variant; `cursor.rs` already tests `back()`.

- [ ] **Step 1: Write failing test** — `parse_nav_command("undo that")` → `Some(NavCommand::Undo)`.
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** the `Undo` variant + parser branch + Tauri command (`back()` then emit the resulting position) + frontend case.
- [ ] **Step 4: Run, expect pass; `cargo build`.**
- [ ] **Step 5: Commit** — `feat: voice undo backstop wired to cursor history (Way 4)`.

### Task 5.5: Optional wake word (Way 5, OFF by default)

**Files:**
- Modify: `src-tauri/crates/detection/src/voice_nav.rs` (accept optional wake prefix) or a thin wrapper in `pipeline.rs`
- Modify: settings persistence (locate the frontend settings store + the backend config; wire a `command_wake_word: Option<String>` defaulting to `None`/off)
- Test: `voice_nav.rs`

- [ ] **Step 1: Write failing test** — with wake word `Some("rhema")`: `"rhema next verse"` parses; `"next verse"` (no prefix) → `None`. With wake word `None` (default): `"next verse"` parses as today.
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement.** Add `parse_nav_command_with_wake(text, wake: Option<&str>)`; when `Some`, require and strip the prefix before parsing; keep `parse_nav_command` as `..._with_wake(text, None)` for back-compat. Surface the toggle in settings UI (off by default).
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `feat(voice-nav): optional wake-word prefix, disabled by default (Way 5)`.

---

# Phase 6 — Semantic throughput (CPU pool, GPU optional)

**Why:** The single ONNX worker behind a depth-4 drop channel ([stt.rs:306](../../../src-tauri/src/commands/stt.rs)) is the first thing to degrade under sustained fast speech. Raise the ceiling on CPU; allow GPU as opt-in.

### Task 6.1: CPU worker pool for semantic detection

**Files:**
- Modify: `src-tauri/src/commands/stt.rs` (spawn N semantic workers draining one shared channel; N = `min(3, available_parallelism - reserved)`).
- Modify: `crates/detection/src/semantic.rs` if the detector holds non-`Sync` state (clone per worker or wrap appropriately).

- [ ] **Step 1:** Replace the single `tokio::spawn` semantic consumer with a pool draining the same receiver (use a shared `Arc<Mutex<Receiver>>` or a fan-out). Keep the drop-if-busy channel behavior — just more consumers. Size from `std::thread::available_parallelism()` minus the audio + detection reservations.
- [ ] **Step 2:** `cargo build`; manual load test: sustained fast speech no longer starves semantic detection (watch Phase-1 drop counters fall).
- [ ] **Step 3: Commit** — `perf(stt): CPU worker pool for semantic detection`.

### Task 6.2: Optional GPU execution provider

**Files:**
- Modify: ONNX session construction in `crates/detection/src/semantic.rs`; add config `semantic_execution_provider: "cpu" | "cuda" | "directml" | "coreml"` defaulting to `"cpu"`.

- [ ] **Step 1:** Read the provider from config; default `"cpu"`. When non-cpu, attempt the matching `ort` execution provider and **fall back to CPU with a warning** if unavailable (never hard-fail).
- [ ] **Step 2:** `cargo build` (CPU path only in CI).
- [ ] **Step 3: Commit** — `feat(semantic): optional GPU execution provider, CPU default + fallback`.

---

## Self-Review notes

- **Spec coverage:** Adaptive segmentation (P2), keyterm fix (P3.1/3.2), source-level command boosting (P3.3), phonetic correction no-model (P4), GPU-optional/CPU-default (P6), command-degradation + **accent generalization across all categories** (P1.3 source spike → P5.1 nearest-slot for directions/units/clear/numbers → P5.2), Ways 1–5 (P5.2/5.3/5.4/5.5), instrumentation-first (P1). All present.
- **Accent generalization is by nearest-slot, not a lookup table** (P5.1): unseen accents are recovered by clear-margin nearest-neighbour over the ~15-word reserved vocab + number words; the logged confusion table is only a supplement. Safety = grammar + isolation gate (P5.2/5.3) means the forgiving matcher only ever runs on believed-command utterances.
- **Coupling called out:** P5.1 reuses `textutil::levenshtein` extracted in P4.1. P5.2 depends on P5.1 (fuzzy unit/number recovery) and on P1.3/P3.3 reducing source garble. P2.3 depends on P2.1+P2.2. Build phases in order; within a phase, tasks are ordered.
- **Per-pastor adaptive learning is explicitly deferred** (noted under P1.3) — YAGNI until real-use data shows P3.3+P5 are insufficient.
- **Behavior changes that break existing tests (intentional):** P5.2 changes `parse_nav_command("next")`, `"go forward"`, `"go back"` (bare/unitless) to `None`. Update those tests deliberately; do not "fix" by reverting the requirement.
- **Bounded-by-construction check:** P2 clamps timeout to [0.8s,5s]; P4/P5 fuzzy thresholds are tight (edit distance ≤1–2) and stop-listed; P5.3 gates on isolation; P5.5 defaults off; P6.2 falls back to CPU. No phase can silently regress live behavior.
- **Open item to resolve during P5.5:** locate the concrete settings persistence path (frontend store + backend config) — grep showed config touchpoints in `lib.rs`, `commands/stt.rs`, `commands/llm.rs` but no single `settings.rs`; confirm before wiring the toggle.
