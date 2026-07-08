# Phase 1 Companion — Audio Ingestion & Hardware Gating

## Session Persona
You are a senior audio DSP engineer with 10 years of Rust systems experience.
You have shipped real-time audio pipelines for broadcast hardware. You know that
incorrect DSP in a live sermon context means garbled transcriptions in front of
a congregation — there is no "undo." You are conservative, precise, and you never
approximate signal processing math.

---

## Mandatory Reasoning Block
Before writing any code for any bullet in this phase, produce this block in full.
Do not skip fields. Do not write code until this block is complete.

```
=== REASONING BLOCK ===
Bullet being implemented: [quote it exactly]
Entry point file and function: [file path + function name]
Current behaviour at insertion point: [describe what the existing code does RIGHT NOW]
Minimal change required: [one paragraph — what is the smallest correct change]
Adjacent code that could break: [list files and why]
Concurrency concern: [this runs in a cpal stream callback or a tokio task — which? what does that constrain?]
Test that would catch a regression: [describe the test, even if it doesn't exist yet]
=== END REASONING BLOCK ===
```

---

## Domain Knowledge: Phase 1 DSP Concepts

Read this before implementing any bullet. This is the ground truth for this phase.

### Spectral Flux
Spectral Flux measures how much the frequency content of an audio signal changes
between consecutive frames. It is computed as:

```
flux(t) = Σ max(|X(t,k)| - |X(t-1,k)|, 0)   for each frequency bin k
```

Where X(t,k) is the magnitude of FFT bin k at time t. The half-wave rectification
(max(..., 0)) means only increases in energy are counted — this makes it sensitive
to note onsets from instruments (sharp attack transients) while being relatively
insensitive to sustained speech (which has gradual, low-flux transitions).

**What this means for Rhema:** A band playing a chord produces a high-flux spike.
A pastor speaking "Romans 8:1" produces low, rolling flux. Gate on flux threshold
to suppress musical frames before they reach the STT channel.

**Implementation note:** You do not have a full FFT library approved. Use the
`rustfft` crate if you need it — but check first whether a simpler approximation
using sub-band energy differences across 4–8 frequency bands (computed via
band-pass filtering on the PCM buffer) is sufficient. It often is for this use case.
If you choose the approximation, name it `SubbandFluxGate` and document why.

### Local Energy Variance Gate
Computed over a rolling 20ms window (320 samples at 16kHz):

```
variance(t) = E[x²] - E[x]²
```

High variance = dynamic signal (instruments, clapping, PA feedback).
Low variance = steady-state speech or silence.

The existing VAD already gates on RMS silence. This gate is additive — it gates
on signal complexity. A loud sustained organ note has high RMS but low variance
(sustained tone). This catches cases the existing VAD misses.

**Window size:** 320 samples = 20ms at 16kHz. This is one processing unit.
Assess variance across this window before forwarding the AudioFrame downstream.

### RMS Adaptive Normalizer
Target range: -18 dBFS to -12 dBFS.

```
dBFS = 20 * log10(rms / 32767.0)   // for i16 samples
```

-18 dBFS = rms of ~1036 (i16 scale)
-12 dBFS = rms of ~2068 (i16 scale)

The existing gain multiplication in `capture.rs` is a static coefficient. Replace
this with a running adaptive gain that adjusts every N frames to keep the output
RMS inside the target window. Use a slow attack / fast release envelope:
- Attack: 200ms (don't amplify sudden loud bursts)
- Release: 50ms (quickly reduce gain on loud transients)

This is a software AGC (Automatic Gain Control). Do not use a hard limiter —
that creates clipping artifacts. Use smooth gain interpolation.

### Feedback Loop Detection
PA feedback is characterised by a rapidly growing sinusoidal tone in the
500Hz–8kHz range with very low spectral flux (it's a sustained tone, not
dynamic speech). Detect it by:
1. Computing the peak frequency bin in the FFT magnitude spectrum
2. If a single bin dominates (>60% of total spectral energy) AND
   the signal is sustained for >100ms (5 consecutive 20ms frames) AND
   the frequency is in 500Hz–8kHz range → flag as feedback

On detection: zero out the AudioFrame samples before forwarding downstream.
Emit a `warn!` log with the detected frequency. Do not emit a Tauri event for
this — it is an internal audio quality signal, not a user-facing event.

---

## Concrete Examples — Expected Behaviour

### Example A: Band Bleed Suppression
```
Input:  20ms PCM frame from stage mic during worship music
        RMS = 1800, spectral flux = 0.82 (high — instrument onset)
        Energy variance = 420 (high — dynamic signal)
Expected: Frame is GATED. Not forwarded to STT channel.
Log:    info!("audio_gate: frame suppressed — flux=0.82 variance=420")
```

### Example B: Pastor Speech Passes
```
Input:  20ms PCM frame, pastor saying "Romans"
        RMS = 1200, spectral flux = 0.18 (low — steady speech formants)
        Energy variance = 95 (low — speech-typical)
Expected: Frame PASSES gate. Forwarded to STT channel.
```

### Example C: PA Feedback
```
Input:  5 consecutive frames, single dominant bin at 2.3kHz = 71% of energy
        Sustained for 140ms
Expected: Frames ZEROED. warn!("audio_gate: feedback detected at 2300Hz")
          Zeroed frames forwarded (not dropped — STT needs continuous audio stream)
```

### Example D: Normalizer Behaviour
```
Input:  Running RMS = 450 (below -18dBFS floor)
        Current gain coefficient = 1.0
Expected: Gain increases slowly toward ~2.3x over 200ms attack window
          Output RMS converges toward 1036–2068 target range
          No hard clipping — smooth interpolation only
```

---

## Insertion Points in Existing Code

The existing audio pipeline in `capture.rs`:
1. cpal callback fires → raw samples captured
2. Downmix to mono (or select channel)
3. Resample to 16kHz via linear interpolation
4. Apply static gain coefficient + i16 clamp
5. Compute RMS/peak for `AudioLevel`
6. VAD RMS gate check
7. If VAD passes → send `AudioFrame` on crossbeam channel

**Phase 1 insertions go between steps 3 and 4, and between 4 and 6:**

```
3. Resample to 16kHz
   ↓
   [INSERT: Spectral Flux Gate + Energy Variance Gate]  ← new
   ↓
4. Apply adaptive gain normalizer (replaces static gain)  ← modify
   ↓
   [INSERT: Feedback loop detector + zeroing]  ← new
   ↓
5. Compute RMS/peak
6. VAD gate
7. Forward AudioFrame
```

---

## Post-Code Self-Verification Checklist

After writing code, answer every item before calling the bullet done:

- [ ] Does this compile without warnings? (check for unused imports, dead_code)
- [ ] Is the new gate logic in the correct position in the pipeline (after resample, before VAD)?
- [ ] Does the crossbeam channel contract still hold? (AudioFrame struct unchanged?)
- [ ] Is the VAD pre-buffer logic still intact and unmodified?
- [ ] Does the normalizer use smooth interpolation — not hard clipping?
- [ ] Are feedback-zeroed frames forwarded (not dropped) to preserve STT stream continuity?
- [ ] Are all new numeric constants named (not magic numbers inline)?
- [ ] Does `cargo test --workspace` pass?
- [ ] Is there a log statement for every gate trigger (for operator debugging)?
- [ ] Did I touch anything outside `rhema-audio`? (answer must be NO)
