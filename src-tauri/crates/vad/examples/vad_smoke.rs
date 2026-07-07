//! Runtime smoke for the Silero ONNX path — no mic needed.
//!
//! Loads `model/silero_vad.onnx`, confirms the I/O contract works at runtime, and drives a
//! silence→tone→silence sequence through `SileroVad` + `VadGate` so we see probabilities
//! move and a start/end fire. Tone is not speech, so exact probabilities aren't meaningful
//! — the point is that the native path runs and the gate transitions plausibly.
//!
//! Run: `cargo run -p rhema-vad --example vad_smoke --features silero`

use std::path::PathBuf;

use rhema_vad::{Reblocker, SileroVad, VadConfig, VadEvent, VadGate, FRAME_SAMPLES, SAMPLE_RATE};

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../model/silero_vad.onnx")
}

/// `secs` of either silence (amp 0) or a 300 Hz tone.
fn segment(secs: f32, tone: bool) -> Vec<f32> {
    let n = (secs * SAMPLE_RATE as f32) as usize;
    (0..n)
        .map(|i| {
            if tone {
                0.3 * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / SAMPLE_RATE as f32).sin()
            } else {
                0.0
            }
        })
        .collect()
}

fn main() {
    let path = model_path();
    println!("[vad smoke] loading {}", path.display());
    let mut vad = match SileroVad::load(&path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("FAIL load: {e}");
            std::process::exit(1);
        }
    };

    let mut reblock = Reblocker::new();
    let mut gate = VadGate::new(VadConfig::default());

    let mut audio = Vec::new();
    audio.extend(segment(0.5, false));
    audio.extend(segment(1.5, true));
    audio.extend(segment(1.0, false));

    let (mut frames, mut min_p, mut max_p, mut starts, mut ends) = (0u32, 1.0f32, 0.0f32, 0u32, 0u32);
    // Feed in odd-sized chunks so the re-blocker actually does its job.
    for chunk in audio.chunks(377) {
        for frame in reblock.push(chunk) {
            let prob = match vad.process(&frame) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("FAIL process: {e}");
                    std::process::exit(1);
                }
            };
            if !(0.0..=1.0).contains(&prob) {
                eprintln!("FAIL: probability out of range: {prob}");
                std::process::exit(1);
            }
            min_p = min_p.min(prob);
            max_p = max_p.max(prob);
            frames += 1;
            match gate.process(prob).event {
                Some(VadEvent::SpeechStart { t_ms }) => {
                    starts += 1;
                    println!("    SpeechStart @ {t_ms} ms (prob {prob:.3})");
                }
                Some(VadEvent::SpeechEnd { t_ms }) => {
                    ends += 1;
                    println!("    SpeechEnd   @ {t_ms} ms (prob {prob:.3})");
                }
                None => {}
            }
        }
    }

    println!(
        "[vad smoke] frames={frames} (={} samples) prob range [{min_p:.3}, {max_p:.3}] starts={starts} ends={ends}",
        frames as usize * FRAME_SAMPLES
    );
    // The native path ran and probabilities are in range: that's the contract we assert
    // here. (Tone input means we don't require a specific start/end count.)
    println!("=== vad smoke: PASS (native path runs, probs in range) ===");
}
