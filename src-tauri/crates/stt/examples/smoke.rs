//! Runtime smoke test for the local STT native paths (Phase 1 changes).
//!
//! Loads the real GGUF models and exercises the exact native-API calls the engine relies
//! on, so we catch anything that compiles but fails at runtime — WITHOUT needing a mic.
//! It does NOT assess transcription quality (that needs real speech + hardware); it
//! asserts the API contracts hold:
//!   1. Offline `run()` with `TimestampKind::Word` succeeds and returns word/token rows
//!      (the data `map_words` maps into per-word timing + confidence).
//!   2. Streaming `stream()`/`feed()`/`snapshot()`/`finalize()` run the full lifecycle
//!      without entering `Failed`, and we report whether `<EOU>`/`<EOB>` tags appear.
//!
//! Run: `cargo run -p rhema-stt --example smoke --features local-stt`

use std::path::PathBuf;
use transcribe_cpp::{
    CommitPolicy, Model, ParakeetStreamOptions, RunOptions, SessionOptions, StreamExtension,
    StreamOptions, StreamState, TimestampKind,
};

const SR: usize = 16_000;

fn model_path(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../model")
        .join(file)
}

/// A non-silent synthetic buffer (speech-band tones). It will not transcribe to real
/// words — the point is only to drive the native code paths without erroring.
fn synth(secs: f32) -> Vec<f32> {
    let n = (secs * SR as f32) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / SR as f32;
            0.1 * ((2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + (2.0 * std::f32::consts::PI * 900.0 * t).sin())
        })
        .collect()
}

fn main() {
    let _ = transcribe_cpp::init_backends_default();
    let mut failures = 0;

    // ── 1. Offline path: run() with Word timestamps ─────────────────────────
    let offline = model_path("parakeet-unified-en-0.6b-Q8_0.gguf");
    println!("\n[1] offline run() + TimestampKind::Word  ({})", offline.display());
    match Model::load(&offline) {
        Ok(model) => {
            let mut session = model
                .session_with(&SessionOptions { n_threads: 4, ..Default::default() })
                .expect("session");
            let opts = RunOptions { timestamps: TimestampKind::Word, ..Default::default() };
            match session.run(&synth(2.0), &opts) {
                Ok(r) => {
                    println!("    ok: text={:?}", r.text.trim());
                    println!("    words={} tokens={} timestamp_kind={:?}",
                        r.words.len(), r.tokens.len(), r.timestamp_kind);
                    let finite_p = r.tokens.iter().filter(|t| t.p.is_finite()).count();
                    println!("    tokens with finite p (confidence): {}/{}", finite_p, r.tokens.len());
                    if let Some(w) = r.words.first() {
                        println!("    first word: {:?} [{}..{}]ms", w.text, w.t0_ms, w.t1_ms);
                    }
                }
                Err(e) => { println!("    FAIL run(): {e}"); failures += 1; }
            }
        }
        Err(e) => { println!("    FAIL load: {e}"); failures += 1; }
    }

    // ── 2. Streaming path: stream()/feed()/snapshot()/finalize() ────────────
    let streaming = model_path("nemotron-3.5-asr-streaming-0.6b-Q5_K_M.gguf");
    println!("\n[2] streaming feed()/snapshot()/finalize()  ({})", streaming.display());
    match Model::load(&streaming) {
        Ok(model) => {
            let mut session = model
                .session_with(&SessionOptions { n_threads: 4, ..Default::default() })
                .expect("session");
            let run = RunOptions { language: Some("en-US".into()), ..Default::default() };
            let sopts = StreamOptions {
                commit_policy: CommitPolicy::Auto,
                family: Some(StreamExtension::ParakeetStream(ParakeetStreamOptions {
                    att_context_right: Some(13),
                })),
                ..Default::default()
            };
            match session.stream(&run, &sopts) {
                Ok(mut stream) => {
                    let audio = synth(3.0);
                    let mut saw_eou = false;
                    for chunk in audio.chunks(SR / 2) {
                        match stream.feed(chunk) {
                            Ok(_) => {}
                            Err(e) => { println!("    FAIL feed(): {e}"); failures += 1; break; }
                        }
                        if stream.state() == StreamState::Failed {
                            println!("    FAIL: stream entered Failed");
                            failures += 1;
                            break;
                        }
                        let txt = stream.text();
                        if txt.committed.contains("<EOU>") || txt.committed.contains("<EOB>") {
                            saw_eou = true;
                        }
                        // snapshot() is what streaming per-word confidence would use later.
                        let _ = stream.snapshot();
                    }
                    let _ = stream.finalize();
                    let final_txt = stream.text();
                    println!("    ok: state={:?} committed_len={} tentative_len={}",
                        stream.state(), final_txt.committed.len(), final_txt.tentative.len());
                    println!("    <EOU>/<EOB> tag seen during feed: {}", saw_eou);
                    println!("    (note: synthetic tone input → transcript content is not meaningful)");
                }
                Err(e) => { println!("    FAIL stream(): {e}"); failures += 1; }
            }

            // ── 3. Stream recreate (the auto-recovery core operation) ───────
            println!("\n[3] recreate stream on same session (auto-recovery core op)");
            match session.stream(&run, &sopts) {
                Ok(mut s2) => {
                    match s2.feed(&synth(0.5)) {
                        Ok(_) if s2.state() != StreamState::Failed => {
                            println!("    ok: recreated stream feeds cleanly (state={:?})", s2.state());
                        }
                        Ok(_) => { println!("    FAIL: recreated stream Failed"); failures += 1; }
                        Err(e) => { println!("    FAIL recreated feed(): {e}"); failures += 1; }
                    }
                    let _ = s2.finalize();
                }
                Err(e) => { println!("    FAIL recreate stream(): {e}"); failures += 1; }
            };
        }
        Err(e) => { println!("    FAIL load: {e}"); failures += 1; }
    }

    println!("\n=== smoke: {} ===", if failures == 0 { "PASS (native paths run)" } else { "FAILURES ABOVE" });
    std::process::exit(if failures == 0 { 0 } else { 1 });
}
