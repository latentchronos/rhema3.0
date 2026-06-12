# Rhema 3.0 — System Architecture & Engineering Specification

**Real-Time Sermon Intelligence & Presentation Operating System**
UGWU TECH / Ugwu Group — Godsvictory Ugwu
Canonical architecture spec (consolidated, markdown). Supersedes the former
`Rhema3_0_Architecture_Specification.docx` (v1), `..._Specification01.docx` (v2),
`Rhema_Output_Architecture_Review.docx`, `Rhema_Multi_Output_Routing_Architecture_v2.docx`,
and `Rhema_Master_Architecture_EdgeCases.docx`. All of those were folded into this file.

> **Reading order:** This file is the *vision / what & why*. The *how & when* lives in
> [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md), [`markdowns/GEMINI.md`](markdowns/GEMINI.md),
> and the `markdowns/PHASE*_COMPANION.md` files. Where this spec describes more than the
> 5-phase roadmap builds, [`TRACEABILITY.md`](TRACEABILITY.md) records exactly what is
> in-scope, simplified, or deferred. **For anything being built right now, the phase
> companions win over this document.**
>
> Points marked **⟲ Reconciled** were changed during the doc-sync to remove a
> contradiction that existed across the old `.docx` set; the rationale is given inline.

---

## 0. Overview & Vision

Rhema 3.0 is a real-time sermon intelligence and presentation operating system. It listens
to a live church service, understands what is being said and why, extracts scripture
references and key teaching content, and projects the right information at the right moment
with minimal human intervention.

Rhema is **not** a manual presentation tool. The operator does not hunt for verses or build
slides during a service. The AI is the primary actor at all times. The operator supervises
in Manual mode and monitors in Auto mode.

> **CORE PRINCIPLE** — Separate *what* is being said, *why* it is being said, and *what
> action* should result. Never merge these three concerns.

### 0.1 What Rhema Is Not
- Not a replacement for ProPresenter, EasyWorship, or similar manual presentation software
- Not a voice-controlled remote for a traditional slide deck
- Not a transcription tool
- Not a system where the operator voices commands

### 0.2 Long-Term Evolution Path

| Generation | Capability |
|---|---|
| Gen 1 | Verse Detection — detect and project explicit Bible references |
| Gen 2 | Intent Detection — is a reference a projection request or just content? |
| Gen 3 | Scripture + Commands — full command classification and navigation |
| Gen 4 | Sermon Understanding — extract teaching points, quotes, prayer points |
| Gen 5 | Automatic Teaching Point Extraction |
| Gen 6 | Automatic Slide Generation |
| Gen 7 | Real-Time Sermon Intelligence Operating System (full vision) |

> The 5-phase hardening roadmap in `IMPLEMENTATION_PLAN.md` lands Rhema solidly across
> **Gen 1–3** and seeds **Gen 4** (topic vector + proactive suggestion). Gen 5–7 capabilities
> described later in this spec (slide generation, full content understanding, the ML intent
> classifier, multi-language) are **deferred** — see [`TRACEABILITY.md`](TRACEABILITY.md).

---

## 1. High-Level System Architecture

The system is a sequential pipeline with two completely separate input modalities that never
interfere with each other.

```
INPUT MODALITY 1: PASTOR'S MIC (voice)
  Audio Pipeline
    |
    Speech-to-Text (Deepgram)
    |
    Intent Boundary Layer
    _____|_____________________________|____________________________
    |                       |                      |               |
 [CONTROL_COMMAND]  [SCRIPTURE_REQUEST]  [PROJECTABLE_CONTENT]  [NON_PROJECTABLE]
    |                       |                      |
 Command Executor    Scripture Engine    Content Understanding Engine
    |                       |                      |
    _________________ Projection Engine ___________________________

INPUT MODALITY 2: OPERATOR CONSOLE (manual)
  Direct system events ───────────────────────────> Command Executor
  (bypasses entire NLP pipeline, always authoritative)
```

> **KEY DECISION** — Only the pastor's microphone feeds the intelligence pipeline. The
> operator has no voice channel. Operator actions go directly to the Command Executor as
> typed system events — no classification needed, no ambiguity. (Mirrors the
> "operator epoch lock beats voice" invariant in `GEMINI.md`.)

---

## 2. Stage 1 — Audio Pipeline

### 2.1 Audio Ingestion Contract
This is a formal specification. Every upstream decision in the audio pipeline references
these values.

| Parameter | Specification |
|---|---|
| Sample Rate | 16 kHz (Deepgram optimal, ASR standard) |
| Bit Depth | 16-bit PCM |
| Channels | Mono (post-routing from mixer — pastor's isolated channel only) |
| Chunk / Buffer Size | 20 ms (320 samples at 16 kHz — matches VAD window) |
| Target Input Level | −18 dBFS to −12 dBFS (RMS adaptive normalisation enforced) |
| End-to-End Latency Budget | ≤ 80 ms (mic → ADC → USB/Dante → cpal buffer → VAD → Deepgram WS) |
| Fallback (no mixer) | ML source separation before Deepgram (Demucs or SepFormer) |

### 2.2 Hardware-First Voice Isolation Strategy
The highest-leverage fix for audio quality requires no ML — it is hardware routing. Modern
digital mixers (Behringer X32, Allen & Heath SQ, Yamaha QL, PreSonus StudioLive) expose each
microphone on its own dedicated channel. The pastor's mic is already isolated in the mixer's
digital routing matrix before it is ever mixed with band, worship team, or ambient room audio.

- Connect the mixer to the Rhema machine via USB audio interface (multi-channel) or Dante/AES67
- Route only the pastor's mic channel to Deepgram — zero ML, zero added latency, perfect isolation
- Operator configures the active channel number in the Rhema console per-church

> **RESULT** — Zero ML source separation required. Zero latency added. Zero audio quality
> degradation. The separation happened in hardware before Rhema ever saw the signal.

### 2.3 Software Fallback — When No Mixer is Available
- ML source separation: Demucs or SepFormer (separate voice from instruments)
- Classical DSP fallback: spectral subtraction / Wiener filtering (stationary noise only)
- Mic strategy: directional or lavalier mic on pastor as minimum requirement

### 2.4 Audio Pipeline Edge Cases & Resolutions

| Problem | Resolution |
|---|---|
| **High-Gain Stage Bleed** — vocal mics capture nearby instruments / choir monitors / transients → false STT tokens | Inline **Spectral Flux + Local Energy Variance Gate** in the ingestion layer. Assess spectral change across a rolling 20 ms window; distinguish continuous melodic patterns from dynamic human speech; suppress downstream routing when energy shapes reveal musical content. *(Phase 1, bullets 1.1–1.2)* |
| **Dynamic Range Compression Trap** — FOH compression squeezes peaks, inflates noise, degrades phonetics | Enforce pre-fader, pre-processing direct output. Apply software **RMS adaptive normalisation** to lock input between −18 dBFS and −12 dBFS. *(Phase 1, bullet 1.3)* |
| **Acoustic Echo Feedback Loop** — system projects a verse, pastor reads it, mic re-ingests it → recursive detection | Rolling **45-second semantic suppression cache** with unique reference identifiers. Any transcript matching cached output is dropped at the transcription gateway. *(Phase 2, bullet 2.4)* |
| **Congregational Amen Splice** — "Amen!"/"Hallelujah!" interrupt scripture coordinates mid-utterance | **Levenshtein Token-Skipping Lookahead Window.** When a canonical book + chapter is initiated, scan up to 6 leading token indices and discard liturgical stop-words to stitch coordinates back together. *(Phase 2, bullet 2.2)* |
| **PA Feedback Squeal** — tonal feedback transcribed as phantom words | **Frequency-domain check** in the gate layer. Pure tonal signals (a near-sine wave) are gated before Deepgram. Detection band: **500 Hz – 8 kHz** (see ⟲ note below). *(Phase 1, bullet 1.4)* |

> **⟲ Reconciled — feedback band.** The old docs gave two different upper bounds (4 kHz in
> the Phase-1 prose, 8 kHz everywhere else). The **canonical band is 500 Hz – 8 kHz**. The
> Phase 1 companion prose has been corrected to match.

**Multiple Microphone Scenarios**

| Scenario | Resolution |
|---|---|
| Multiple pastors / guests on different channels | Monitor multiple channels simultaneously, merged into one stream. Quick channel switch from operator console. |
| Wireless mic dropout | VAD layer catches silence / RF noise bursts and gates the stream. |
| Two mics on pulpit (lapel + handheld) | Multi-channel mix of only those two channels into Rhema's input. |
| Monitor aux feed (wrong source) | Document: Rhema requires a pre-fader direct output / clean aux send, never a monitor mix. |
| Dante/AES67 latency spike | Dedicated VLAN for audio traffic. Standard pro-install practice. |
| USB audio driver issues on Linux | Recommend JACK with low buffer size for production. Document in setup guide. |

**Service Session Gating** — Sound check, announcements, and pre-service audio pollute
context and trigger false detections. Resolution: explicit **session start flag** in the
operator console. Rhema does not process detections or commands until the operator explicitly
starts the session; session-start timestamps gate all detection logic.

---

## 3. Stage 2 — Speech-to-Text

### 3.1 Primary Engine
Deepgram is the primary STT engine, operating in WebSocket streaming mode for real-time
transcription. Example output:

```json
{
  "text": "Turn with me to Romans chapter 8",
  "timestamp": "00:12:43",
  "speaker": "Pastor",
  "words": [
    { "word": "Romans",  "confidence": 0.94 },
    { "word": "chapter", "confidence": 0.98 },
    { "word": "8",       "confidence": 0.97 }
  ]
}
```

### 3.2 Interim vs. Final Transcripts
Deepgram returns both interim (partial, low-latency) and final (complete, higher accuracy)
transcripts.

> **DEFERRED COMMAND PATTERN** — Detect intent on interim transcripts for speed; wait for the
> final transcript to confirm the full entity before executing. Low latency for intent
> awareness, high accuracy for entity extraction.

### 3.3 Confidence Score Handling
Deepgram returns word-level confidence on every transcript. These scores must be used.
- A verse reference whose book name has confidence < 0.5 is flagged uncertain, not acted on
- Command confidence is weighted by the minimum word confidence in key entity positions (book, chapter, verse)
- Entire utterances below a global confidence threshold (congregation noise, applause) are discarded without parsing

### 3.4 STT Edge Cases & Resolutions

| Problem | Resolution |
|---|---|
| Congregation noise hallucination | Global utterance confidence threshold. Below threshold → discard entirely without parsing. |
| Speaker overlap | Be skeptical of transcripts with mixed semantic content. A verse ref surrounded by unrelated words signals overlap corruption. |
| Deepgram WS disconnect mid-service | Prominent visual indicator on operator console when streaming drops to REST fallback. Audio alert option. REST has higher latency — operator must know. |
| Mic handling transients | Sudden amplitude spike with no preceding speech context is gated by VAD before Deepgram. |
| Homophone confusion (Romans/roaming, Corinthians/corintians) | **Phonetic normalisation** — see canonical map in §7.2. *(Phase 2, bullet 2.1)* |
| Number homophones (fourteen/forty, sixteen/sixty, first/verse) | Same phonetic normalisation layer handles numeric tokens. |
| Prepositional homophone ("Ephesians 2:4" → "Ephesians to verse four") | Pre-parse coordinate translation rule swapping homophonic words for canonical integer tokens before entity extraction. |
| Open mic during prayer | Prayer mode toggle from operator console — light-touch session pause with one-tap resume. *(deferred — see TRACEABILITY)* |

---

## 4. Stage 3 — Intent Boundary Layer

The most critical component in the system. It classifies every utterance before any action
is taken. **Intent classification is primary — not the source person.** Pastor voice and
operator console are different modalities, but all pastor utterances are always classified here.

### 4.1 The Four Classification Buckets

| Classification | Definition |
|---|---|
| CONTROL_COMMAND | Operational commands at the system. Navigation, display, translation switch, mode changes. |
| SCRIPTURE_REQUEST | Request to display a specific scripture. Explicit (verse ref) or implicit ("turn with me to…"). |
| PROJECTABLE_CONTENT | Sermon content with projection value — teaching points, quotes, prayer points, announcements. |
| NON_PROJECTABLE_CONTENT | Ordinary preaching, exposition, illustrations, asides not meant for projection. |

### 4.2 Classification Examples

| Utterance | Classification |
|---|---|
| "Next verse" / "Clear the screen" / "Switch to NIV" | CONTROL_COMMAND |
| "Turn with me to John 3:16" / "Romans 8:28 — show it" | SCRIPTURE_REQUEST |
| "Project am — Ephesians 2:8" | SCRIPTURE_REQUEST (Nigerian English) |
| "Na John 3:16 we dey talk about, show am" | SCRIPTURE_REQUEST (Nigerian English) |
| "Transformation begins with renewing the mind" | PROJECTABLE_CONTENT |
| "Lord help us walk faithfully before you" | PROJECTABLE_CONTENT (prayer point) |
| "The conference starts next Friday" | PROJECTABLE_CONTENT (announcement) |
| "Can everyone hear me?" / "Today we are studying Romans 8" | NON_PROJECTABLE_CONTENT |

> **⟲ Note — Nigerian English / Pidgin.** The vision treats pidgin as first-class. In the
> current roadmap, Stage 1 is implemented as a **pattern/rule classifier** (not the ML model
> below), so broad pidgin coverage is **deferred to the Gen-3 ML classifier**. Tracked in
> [`TRACEABILITY.md`](TRACEABILITY.md) so it is not silently lost.

### 4.3 Two-Stage Classification Architecture

**Stage 1 — Fast local classifier (primary path)** — *vision target:*
- Small fine-tuned transformer (distilled BERT / MobileBERT equivalent)
- Trained on labelled church-command utterances incl. Nigerian English variants
- Runs entirely on-device, no API call, target < 10 ms
- Returns intent label + entities + confidence; handles ~90% of utterances

> **⟲ Reconciled — Stage 1 in the roadmap.** Phase 2 implements Stage 1 as a **deterministic
> pattern classifier** returning `ControlCommand | ExplicitScriptureRequest | Ambiguous`, not
> the ML model. This is a deliberate v1 simplification: it ships now, needs no training data,
> and keeps the < 10 ms budget. The ML classifier is a **Gen-3 upgrade** behind the same
> interface. See `PHASE2_COMPANION.md` and `TRACEABILITY.md`.

**Stage 2 — LLM fallback (when Stage 1 confidence < 0.7)**
- Triggered only on genuinely ambiguous utterances
- Structured prompt: current sermon context + cursor position + ambiguous utterance
- Returns structured JSON (intent + entities); acceptable latency 200–500 ms
- **Claude** as the LLM endpoint *(Phase 2 wires the channel to a placeholder; Phase 5 wires the real call)*

> **PRECEDENT** — Local wake-word + cloud NLU is exactly how Alexa and Google Assistant ship.
> The pattern is proven.

### 4.4 Latency Requirements by Intent Category

| Category | Latency | Reason |
|---|---|---|
| CONTROL_COMMAND (show/hide/navigate) | < 50 ms (local only) | Visible immediately; delay is perceptible. |
| SCRIPTURE_REQUEST | < 50 ms (local only) | Same. |
| PROJECTABLE_CONTENT | < 500 ms (LLM acceptable) | Goes to queue for operator review. |
| Queue management commands | < 500 ms | Human confirms anyway. |
| Contextual search | < 500 ms | Result presented before projection decision. |

### 4.5 Intent Layer Edge Cases

| Problem | Resolution |
|---|---|
| **Mid-sentence self-correction** ("Go to — no wait — Romans 8:1") | Inline **negation-marker filter** in the command AST parser. On a marker ("no wait", "scratch that", "I mean"), drop the earlier entity's execution token and lock onto the terminal reference. *(Phase 3, bullet 3.3)* |
| **Commands embedded in preaching** ("the Bible says in John 3:16 — show that to them") | Detect verse ref AND command in one utterance; the command ("show that") triggers immediate projection rather than queued detection. *(partially scoped — see TRACEABILITY)* |
| **Filler-word stripping** ("Uh, go to, uh, next verse") | Filler filter before classification — strip "uh", "um", "you know", "like". |
| **Interrogative command forms** ("Can we show John 3:16?") | Train "Can we… / Would it be possible to… / Should we…" + display verb + ref as SCRIPTURE_REQUEST. |
| **Ordinal book-name ambiguity** ("First John 3" vs "in the first John chapter") | **Token-Proximity Adjacency** parser — an ordinal is a book-title modifier only if directly adjacent to the book name with no intervening tokens. |
| **Book-name disambiguation** ("Go to John", "Kings") | With no number qualifier and multiple matches, default to the first (1 Kings) and notify the operator: "Defaulted to 1 Kings. Tap to change." Never ask mid-service. |

---

## 5. Stage 4 — Command Executor

Handles all CONTROL_COMMAND intents. Also receives all direct operator console events, which
bypass the NLP pipeline entirely and arrive as typed system events.

### 5.1 Command Categories
- **Navigation:** next/previous verse, jump to reference, first/last verse of chapter, skip N verses
- **Display:** show, hide, clear screen, blank screen
- **Translation:** switch translation, translation peek, compare translations
- **Mode:** start/stop reading mode, start/end session
- **Queue:** add to queue, remove from queue, clear queue
- **Output:** route to main screen, alt screen, both

### 5.2 Compound Commands
Commands can be chained in one utterance, parsed as an Abstract Syntax Tree (AST).

| Compound Command | Execution |
|---|---|
| "Clear the screen and go to Romans 8:1" | HIDE → NAVIGATE Romans 8:1 → SHOW |
| "Go back one verse and add it to queue" | NAVIGATE −1 → QUEUE ADD |
| "Show Romans 8:1 on both screens" | SHOW on MAIN + SHOW on ALT (parallel) |
| "Find the verse about faith and show it" | SEARCH semantic:faith → SHOW result (depends on first) |
| "Go to Romans 8 and read from there" | NAVIGATE Romans 8:1 → ACTIVATE reading mode |

Sequential compounds execute in order. Conditional compounds wait for the first action to
resolve. The AST executor has a configurable timeout that prompts the operator if a
dependency takes too long. *(Full AST executor is Gen-3+ — see TRACEABILITY; Phase 3 ships only the negation filter.)*

### 5.3 Epoch-Lock Conflict Resolution
Manual operator actions and in-flight voice detections can hit the Command Executor at the
same millisecond. Resolution: **Epoch-Based Lock.** Any operator console action increments
the global epoch index and engages a strict 500 ms window lock. Incoming voice-triggered
commands with older timestamps/epochs are dropped safely. **Operator action always wins.**
*(Phase 3, bullet 3.2 — matches the "operator epoch lock beats voice" invariant in GEMINI.md.)*

---

## 6. Stage 5 — Navigation State Machine

### 6.1 Formal Cursor State Object
Every navigation decision is gated by this state. It must be typed and formally defined before
unit tests can be written.

```
CursorState {
  position: {
    book:        String | null,
    chapter:     u32    | null,
    verse:       u32    | null,
    translation: String | null
  },
  range_end:           u32 | null,        // null = single verse display
  history_stack:       Vec<Position>,     // max depth: 50, oldest dropped
  forward_stack:       Vec<Position>,     // for redo after going back
  navigation_mode:     'locked' | 'preview' | 'independent',
  reading_mode: {
    active:       bool,
    sub_position: Position | null
  },
  session_start_verse: Position | null    // pinned separately, never dropped
}
```

> **⟲ Reconciled — `active_output` removed from the cursor.** The old v2 spec's CursorState
> carried `active_output: 'main' | 'alt' | 'both'`. That binds navigation state to physical
> screens, which directly contradicts §11 ("Channels own state. Devices never own navigation
> state."). **Output targeting is no longer a cursor field** — it is governed by the channel
> layer (§11) and the channel-level `RoutingMode` (Phase 4). The cursor owns *where in the
> Bible we are*; the channel layer owns *who sees what*. This matches the `CursorState` in
> `PHASE3_COMPANION.md`, which already dropped `active_output`.
>
> `navigation_mode` is retained here as the *logical* lock/preview/independent intent, and is
> surfaced to outputs as the channel `RoutingMode` (§11.6 / Phase 4). It is defined once
> (here) and consumed by the routing layer — not duplicated as authoritative state in two places.

### 6.2 Navigation Modes

| Mode | Behaviour |
|---|---|
| Locked | Audience and Pastor channels always show the same verse. Navigate together. **Default.** |
| Preview | Pastor channel is one verse ahead. Congregation sees current; Pastor sees next. "Next" advances audience to what pastor previewed; pastor advances one more. |
| Independent | Audience and Pastor channels navigate separately, each with its own cursor. |

### 6.3 Cold State — No Position Loaded

| Command | Response |
|---|---|
| "Next verse" | "Nothing is currently displayed. Please call a specific verse first." |
| "Go back" | "No previous verse in history." |
| "Jump to verse 5" | "Please specify the book and chapter." |
| "Last verse of this chapter" | "No chapter is currently loaded. Please specify." |

*(Phase 3 — cold-state guards.)*

### 6.4 Forward Navigation Edge Cases

| Scenario | Resolution |
|---|---|
| Last verse of a chapter (Genesis 1:31) → "next verse" | Auto-advance to Genesis 2:1. Never fail or wrap. |
| Last verse of last chapter of a book (Malachi 4:6) → "next verse" | Advance to Matthew 1:1 OR ask "Move to Matthew?" — configurable. |
| Revelation 22:21 → "next verse" | "You are at the last verse of the Bible." |
| "Skip 3 verses forward" from Genesis 1:30 | Land at Genesis 2:2 (cross-chapter arithmetic). Never fail on boundary. |
| "Go to chapter 159 of Genesis" | "Genesis only has 50 chapters." |
| "Skip to verse 50" when chapter has 28 | "This chapter only has 28 verses." |

### 6.5 Backward Navigation Edge Cases

| Scenario | Resolution |
|---|---|
| Genesis 1:1 → "previous verse" | "You are at the first verse of the Bible." |
| First verse of any chapter → "previous verse" | Go to last verse of previous chapter (look up count from translation map). |
| "Go back 5 verses" from Genesis 1:3 | Cannot reach chapter 0. Stop at Genesis 1:1 with explanation. |
| "Go back 2 chapters" from chapter 1 | Cannot go to chapter −1. Stop at chapter 1. |
| Genesis 1:1 → "go back" (history-based) | "No previous verse in history." |

*(Phase 3, bullet 3.4 — all bounds via `BibleDb`, never hardcoded counts.)*

### 6.6 Book-Level Navigation Edge Cases

| Scenario | Resolution |
|---|---|
| "Next book" from Revelation | "Revelation is the last book." |
| "Previous book" from Genesis | "Genesis is the first book." |
| "Go back to Genesis" when already in Genesis | "Already in Genesis." |
| "Go to the next book" with no position loaded | "Please specify a starting book and chapter." |

### 6.7 Translation Boundary Conflicts
Some translations omit certain verses (e.g. John 5:4, Acts 8:37). **The Bible structure map
must be per-translation, not global.**
- Navigating to an omitted verse → skip to next present verse, or inform the operator
- Switching translation mid-reading → re-fetch chapter in the new translation, keep verse position
- Fewer verses in a chapter (combined verses) → handle verse-number mismatch gracefully
- Apocrypha / deuterocanonical books → structure map is per-translation; "Next book" from Malachi may differ

*(Phase 3, bullet 3.4 — translation-specific omission, ≤ 3 retries.)*

### 6.8 History Stack Rules
- Empty stack → "go back" fails gracefully
- Max depth 50 — oldest entries dropped when exceeded
- Going back then forward: forward navigation clears the forward stack (browser model)
- Reading mode has its own sub-position cursor separate from navigation history. Any manual
  navigation exits reading mode and notifies: "Reading mode was active on [book]. It has been paused."
- Session-start verse is pinned separately and never dropped

### 6.9 Verse Range Behaviour
When a range is displayed (e.g. Romans 8:1–4):
- "Next verse" → the verse after the **end** of the range (verse 5), collapsing to single-verse
- "Previous verse" → the verse before the **start** of the range
- The cursor tracks `range_end` for all arithmetic

> **Asymmetric range rule** *(Phase 3, bullet 3.5)*: forward = `range.end + 1`,
> backward = `range.start − 1`, always collapse to `CursorMode::Single`.

### 6.10 The Same Verse Again
System at John 3:16, pastor says "John 3:16" again → show a brief operator confirmation rather
than re-rendering, unless the operator explicitly says "show it again." The 45 s suppression
cache (§2.4 / Phase 2) prevents duplicate detection.

### 6.11 Queue vs. Live Position Conflict
"Next verse" **always** means numerically from the live cursor position. Queue advancement
requires explicit queue commands. These are two separate navigational contexts that never
share a command.

### 6.12 Spatial & Relational Lookups *(folded in from EdgeCases doc)*
Speakers use visual landmarks — "the verse right above this," "the following chapter."
Resolution: a **Relational Delta Mapping** table in the state router that pairs directional
keywords with deterministic ±1 offset transformations. *(Not in the 5-phase roadmap — Gen-3
navigation polish; tracked in TRACEABILITY.)*

---

## 7. Stage 6 — Scripture Understanding Engine

### 7.1 Detection Methods

| Detection Type | Method |
|---|---|
| Explicit reference (John 3:16) | Regex + NER + phonetic normalisation |
| Abbreviated reference (Jn 3:16, Ps 23) | Abbreviation expansion map + NER |
| Spoken numbers (Ephesians two eight) | Number-word → integer before extraction |
| Semantic / partial quote ("The Lord is my shepherd") | Embedding vector search against pre-embedded corpus |
| Partial memory ("something about running the race") | Semantic search → Hebrews 12:1 |
| Named narrative (the prodigal son) | Named-narrative lookup table + semantic fallback |

### 7.2 Phonetic Normalisation Map (Canonical)
Applied before all entity extraction. **This is the single canonical normalisation table** —
it merges the book-name list from the Phase 2 companion with the translation/prepositional
entries from the original spec map.

> **⟲ Reconciled — one normalisation table.** The old docs had two different "the 8
> homophones" lists (Phase 2 companion = book names only; spec §7.2 = a book/translation/
> prepositional mix). They are merged below. The Phase 2 companion's data-driven rule table
> should implement **this** set.

**Book-name homophones**

| Deepgram output | Normalised to |
|---|---|
| "roaming" | "Romans" |
| "corintians" / "corinithians" | "Corinthians" |
| "efesians" | "Ephesians" |
| "phillipians" | "Philippians" |
| "soms" / "salms" | "Psalms" |
| "izaya" | "Isaiah" |
| "doo-ter-onomy" (mangled forms) | "Deuteronomy" |
| "revelations" | "Revelation" (STT adds plural) |
| "philemon" | disambiguate: Philemon vs Philippians (check context) |

**Prepositional / number homophones**

| Deepgram output | Normalised to |
|---|---|
| "to verse four" / "too four" | "2:4" |
| "Ephesians to verse four" | "Ephesians 2:4" |
| "Romans eight one" | "Romans 8:1" |
| "first Corinthians thirteen" | "1 Corinthians 13" |
| "Psalms one nineteen" | "Psalm 119" |

**Translation homophones**

| Deepgram output | Normalised to |
|---|---|
| "niv" / "en eye vee" / "envy" | "NIV" |
| "nkjv" / "en kay jay vee" | "NKJV" |
| "esv" / "easy" | "ESV" |
| "authorized version" | "KJV" |

### 7.3 Contextual Reference Finding — Categories A–H
The semantic search supports a wide grammar of contextual queries. **Most of these are Gen-4+
and not in the 5-phase roadmap** (Phase 5 ships topic-vector + priming + threshold-gated
suggestion only). Retained as forward spec:

- **A — Direct thematic** ("Find the verse about faith") → semantic search, boost books already referenced
- **B — Narrative / event** ("the prodigal son") → named-narrative lookup → Luke 15:11–32
- **C — Character-filtered** ("What does Jesus say about prayer") → red-letter filter + topic
- **D — Pronoun / deictic** ("the passage about that") → last 10–15 s of transcript as query (60 s window)
- **E — Cross-testament** ("the OT prophecy behind this verse") → cross-reference + testament filter
- **F — Exclusion** ("a verse about love NOT in 1 Corinthians 13") → semantic search w/ exclusion filter
- **G — Ranked** ("the most important verse about salvation") → rank by cross-ref frequency + relevance
- **H — Confession of ignorance** ("I don't remember where that verse is") → quotation match first, semantic fallback; non-existent phrase → "No strong match found. This phrase may not be in the Bible."

> **Thematic scope ambiguity** *(folded in from EdgeCases doc)*: broad theological terms span
> thousands of verses. Resolution: intersect the general semantic query with the **top 3 active
> themes** from the sermon-context metadata. *(Gen-4; tracked in TRACEABILITY.)*

---

## 8. Stage 7 — Translation Engine

> **Scope:** The full translation engine (peek mode, compare, dual-language, live registry)
> is **deferred** — no phase companion implements it. The current build ships the four DB
> translations (KJV, SpaRV, FreJND, PorBLivre) with switching via existing `BibleDb`. Retained
> as forward spec. See [`TRACEABILITY.md`](TRACEABILITY.md).

### 8.1 Live Translation Registry
A queryable runtime state (not static config) that updates when translations are added/removed.
Each entry: canonical ID, full name, abbreviations, aliases, phonetic variants, style tag
(literal/dynamic/paraphrase), per-translation verse-coverage map, load status.

### 8.2 Translation Command Simulations

| Command | Response |
|---|---|
| "Switch to NIV" | Switch; confirm "Switched to New International Version" |
| "Read that in KJV" | Re-fetch current verse in KJV without switching default |
| "What does the Amplified say" | Translation peek — display without switching default |
| "Switch to the Message" (not loaded) | "The Message is not loaded. Available: KJV, NIV, NKJV, ESV" |
| "Switch to the Robertson translation" (not real) | "I don't recognise that translation. Available: …" |
| "Show John 3:16 in both KJV and NIV" | Side-by-side display — not a switch |

### 8.3 Translation Peek Mode
"Let's see what NIV says here" → temporary single-verse switch that does not change the
default. The next verse call returns to the default automatically unless a full switch is issued.

### 8.4 Translation Edge Cases
- Translation added mid-service → registry file-watch / poll, refresh automatically
- New translation covers only NT, cursor at Genesis → warn, remain on current translation
- Two-language church → dual-translation display (one per line or split screen)
- "What does the original say" → recognise Hebrew/Greek request; if Nestle-Aland/BHS not loaded, say so

---

## 9. Stage 8 — Content Understanding Engine

> **Scope:** Transforms Rhema from verse detector into sermon-intelligence system. Processes
> all PROJECTABLE_CONTENT. **Most of this is Gen-4/5/6 and deferred.** Phase 5 ships only the
> SermonContext topic vector, priming index, and proactive suggestion engine (§9.3–9.4). The
> content scoring (§9.2) and slide generation (§10.5) are **not** in the 5-phase roadmap.

### 9.1 Content Categories
KEY_TEACHING_POINT, MEMORABLE_QUOTE, PRAYER_POINT, ANNOUNCEMENT, SERMON_TITLE.

### 9.2 Content Importance Scoring *(deferred)*
Every PROJECTABLE_CONTENT utterance is scored; default projection threshold 0.80, configurable
per-church. ("Can everyone hear me?" → 0.05 discard; "Faith without obedience eventually dies"
→ 0.92 project.)

### 9.3 Sermon Intelligence Context (SermonContext) *(Phase 5, bullet 5.1)*
A continuously updated state built from all final transcripts, all displayed verses, detected
topics, and a topic vector representing the sermon's semantic centre of gravity.

The topic vector uses an **Exponential Time-Decay Window**: 75 % weight on the active 90-second
sliding window, 25 % background bias from earlier content. half-life = 90 s.

> **⟲ Reconciled — update cadence.** The spec said "updated every 30 seconds"; the Phase 5
> companion updates **on each new transcript sentence** with **lazy (on-demand) recompute** of
> the vector. The canonical behaviour is the Phase 5 one: ingest per-sentence, compute the
> vector only when a suggestion is evaluated. The "30 s" figure was an illustrative cadence,
> not a contract.

### 9.4 Proactive Suggestion Engine *(Phase 5, bullet 5.3)*
When the SermonContext finds a strong, unshown thematic cluster, it surfaces a suggestion **in
the operator console only — never auto-projected.**

Rules (non-negotiable):
- **Never auto-projects** — always requires operator tap; surfaces only to the Operator Channel
- Max one suggestion per **5 minutes (300 s)**
- Excludes all verses already in the session display history
- Operator dismiss suppresses that verse for the rest of the session
- Fires only when cosine similarity > **0.72** AND topic has moved AND a high-relevance unshown verse exists

---

## 10. Stage 9 — Projection Engine & Operating Modes

> **Scope:** Manual/Auto modes and the accuracy-unlock strategy are **deferred** — no phase
> companion implements mode switching or the 0.92/0.85 thresholds. Retained as forward spec.
> The current build is effectively Manual (operator-supervised queue).

> **IDENTITY NOTE** — Both modes are AI-primary. "Manual" means AI-supervised, not
> human-operated. This separates Rhema from all existing church presentation software.

### 10.1 / 10.2 The Two Modes & Thresholds

| Mode | Default Threshold | Below Threshold | Above Threshold |
|---|---|---|---|
| Manual (default) | 0.92 | Sits in queue — operator must approve | Auto-promotes to screen without approval |
| Auto | 0.85 | Surfaces as suggestion only — nothing projects | Projects immediately without approval |

### 10.3 Mode Default Strategy
Manual is the default for all new installs — a deliberate trust-building strategy. A church
service is a zero-mistake environment. Auto mode is **earned** through demonstrated accuracy
(e.g. operator approval rate > 94 % across 10 services).

### 10.4 Operator Console — Always-Available Overrides
Regardless of mode, always available, always authoritative, always instant:
force-project, pull-down, block queued item, switch translation, start/end session, switch
active channel (pastor mic routing), toggle prayer mode.

### 10.5 Presentation Generation *(deferred — Gen 6)*
Raw PROJECTABLE_CONTENT is formatted into presentation-ready slides (summarisation, layout,
theme consistency, typography scaling) before projection.

---

## 11. Multi-Output & Routing Architecture

> **This section is the consolidated, canonical output model.** It folds in and supersedes
> `Rhema_Output_Architecture_Review.docx` and `Rhema_Multi_Output_Routing_Architecture_v2.docx`.
> It implements the v2 **channel** model and discards the v1 **screen-bound** model entirely.

### 11.0 Architectural Decision: Channels, Not Screens
Design around **channels**, not screens. A physical screen is a rendering endpoint. A channel
is a content stream with a defined audience and a defined set of rules.

Many output architectures fail by assigning state to *devices*: a projector tracks its own
verse, a monitor tracks its own verse, an OBS scene tracks its own verse — one updates, another
doesn't, and you get divergence. The fix: make the **channel** the owner of truth and the
**device** a read-only subscriber.

> **The Loophole Killer:** One content state, many device endpoints.
> **Channels own truth. Devices display truth.**

What should *not* be in each channel matters as much as what is:
- **Audience Channel:** NO confidence scores, AI suggestions, queue items, internal state, or next-verse previews
- **Pastor Channel:** NO predicted future teaching points in v1
- **Operator Channel:** NO engineering internals (embedding scores, model diagnostics) during a live service

### 11.1 Design Principles
1. Single source of truth
2. Channels own state
3. Devices never own navigation state
4. One content state may be rendered by many devices
5. Operator authority always overrides automation
6. Routing is independent from content generation
7. Device failures must not corrupt channel state

### 11.2 Channel Model

| Channel | Content |
|---|---|
| **Audience** | Scripture, projectable content, announcements |
| **Pastor (User)** | Current verse, next verse (preview mode), translation indicator, service timer |
| **Operator** | Queue, confidence scores, suggestions, session controls, routing controls |

These are content streams, not physical screens. Physical devices subscribe to channels.

### 11.3 Device Model
A device is a rendering endpoint. It subscribes to a channel and does not own content state.

| Channel | Example Devices |
|---|---|
| Audience | Projector, Rear TV, OBS scene, NDI output |
| Pastor | Stage TV, Floor monitor, Tablet, NDI |
| Operator | Control workstation, Secondary admin display |

### 11.4 Routing Layer
The routing layer maps channels to devices. **Routing changes do not modify content state.**

```
Projection Engine
    ├── Audience Channel ── Projector, Rear TV, OBS, NDI
    ├── Pastor Channel  ── Stage TV, Floor Monitor, Tablet
    └── Operator Channel ── Control Console
```

> **Core principle:** Channels own state. Devices own presentation. Routing owns delivery.

### 11.5 State Ownership Rules
Audience Channel owns audience content state. Pastor Channel owns pastor preview state.
Operator Channel owns supervision state. **No physical device stores authoritative navigation
state.** *(This is exactly why `active_output` was removed from the §6.1 cursor — see ⟲ note there.)*

### 11.6 Navigation Mode Impact on Multi-Output
The `navigation_mode` from §6.1 is surfaced to the routing layer as the channel `RoutingMode`:

| Mode | Behaviour |
|---|---|
| Locked | Audience and Pastor channels move together. |
| Preview | Pastor channel shows current + upcoming verse; congregation sees current. |
| Independent | Channels may diverge intentionally, only through explicit operator control. |

*(Phase 4, bullet 4.2 — `RoutingMode { Locked, Preview, Independent }`.)*

### 11.7 Synchronisation Guarantees
A channel update is published once. All subscribed devices receive the same state.
Device-specific rendering may differ visually but never semantically.

### 11.8 Device Lifecycle
Registered → Connected → Active → Degraded → Disconnected → Removed.
**Channel state survives device failures at all stages.**

### 11.9 Health Monitoring & Failure Recovery *(Phase 4, bullet 4.4)*
Every endpoint periodically reports health (2 s ping). Failures generate operator
notifications — never silently degrade — and never corrupt channel state.

| Failure | Recovery |
|---|---|
| Projector offline | Audience Channel stays active; other audience devices unaffected. |
| Pastor monitor offline | Pastor Channel stays active. |
| OBS offline | Streaming endpoint marked unhealthy; local projection unaffected. |
| Device reconnect | Device requests latest channel state and resyncs. No operator action needed. |

### 11.10 Resolved Edge Cases
- Multiple stage monitors → all subscribe to the same Pastor Channel
- Pastor reads from congregation screen / phone → projected channel state remains authoritative
- OBS / NDI divergence → impossible; both subscribe to the same Audience Channel state
- Device disconnect during service → channel stays alive; only endpoint health changes
- Future scaling → new displays just subscribe; no architectural redesign

### 11.11 Output Rollout Stages

> **⟲ Reconciled — renamed to avoid collision.** The old output docs called these "Phase 1–6",
> colliding with the unrelated implementation **Phase 1–5** (audio→stt→nav→channels→intelligence).
> These are now **Output Rollout Stages** to keep the two numbering systems distinct. Most of
> this rollout lands inside implementation **Phase 4**.

| Output Stage | Deliverable | Maps to |
|---|---|---|
| OS-1 | Core channel model: Audience, Pastor, Operator operational | Phase 4, bullet 4.1 |
| OS-2 | Routing layer + device registration | Phase 4, bullet 4.2 |
| OS-3 | Health monitoring + failure recovery | Phase 4, bullet 4.4 |
| OS-4 | OBS + NDI integration | (existing broadcast outputs) |
| OS-5 | Advanced channel management (independent nav, preview mode) | Phase 4, bullet 4.2 |
| OS-6 | Future expansion: web, mobile, remote outputs | deferred |

### 11.12 What Should NOT Be Added Prematurely
Avoid in v1: predicted future teaching points during a sermon, AI-generated pastor guidance
injected into the preaching flow, complex multi-translation divergence per device, engineering
diagnostics on service screens.

---

## 12. Research Areas & Implementation Stack

### 12.1 Core Research Areas
Audio (DSP, VAD, noise reduction, voice isolation, spectral analysis, beamforming, source
separation) · Speech (ASR, streaming transcription, confidence scoring) · Language (NLP, intent
classification, NER) · Retrieval (embeddings, vector DBs, semantic search, RAG) · Understanding
(information extraction, topic modelling, ranking) · Generation (LLMs, summarisation, slide
generation) · Systems (event-driven architecture, real-time systems, state machines, agentic
workflows).

### 12.2 Recommended Technology Decisions

| Component | Recommendation |
|---|---|
| STT Engine | Deepgram (WebSocket streaming, word-level confidence, diarisation) |
| Audio Capture | cpal (cross-platform, low-latency, JACK-compatible on Linux) |
| Local Intent Classifier | Distilled BERT / MobileBERT *(Gen-3; v1 ships pattern classifier)* |
| LLM Fallback | Claude API (structured JSON output, sermon context in prompt) |
| Verse Embeddings | Pre-embedded verse corpus, vector search at query time (ONNX + HNSW) |
| Source Separation (fallback) | Demucs or SepFormer |
| Hardware Audio Routing | Direct output / clean aux send from digital mixer via USB or Dante/AES67 |
| Production Audio on Linux | JACK with low buffer size |

> The concrete crate/store/file ground truth for the current codebase lives in
> [`markdowns/GEMINI.md`](markdowns/GEMINI.md) — defer to it for anything implementation-level.

---

## 13. Final Goal
Rhema 3.0 listens to a live sermon. It understands what matters. It extracts scripture
references and key teaching moments. It generates presentation-ready content. It projects the
right information at the right time with minimal human intervention. At that point, Rhema is no
longer a projection tool — it is a **Real-Time Sermon Intelligence & Presentation Operating System.**

---

*UGWU TECH — Rhema 3.0 — Architecture Specification (Consolidated, markdown canonical)*
