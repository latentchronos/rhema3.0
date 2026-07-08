//! Re-block variable-size capture buffers into the exact fixed frames Silero requires.
//!
//! cpal delivers callbacks of arbitrary (device-dependent) length; Silero v5 demands
//! exactly [`FRAME_SAMPLES`](crate::FRAME_SAMPLES) per inference. This accumulates input
//! and yields complete frames, retaining any remainder for the next call.

use crate::FRAME_SAMPLES;

/// Fixed-size frame accumulator. Feed arbitrary-length audio with [`push`](Reblocker::push);
/// receive zero or more complete frames.
pub struct Reblocker {
    buf: Vec<f32>,
    frame: usize,
}

impl Reblocker {
    /// A re-blocker yielding [`FRAME_SAMPLES`](crate::FRAME_SAMPLES)-sample frames.
    pub fn new() -> Self {
        Self::with_frame(FRAME_SAMPLES)
    }

    /// A re-blocker yielding `frame`-sample frames (for tests / non-default models).
    pub fn with_frame(frame: usize) -> Self {
        Self {
            buf: Vec::with_capacity(frame * 2),
            frame,
        }
    }

    /// Append `input` and return every complete frame now available (possibly none).
    pub fn push(&mut self, input: &[f32]) -> Vec<Vec<f32>> {
        self.buf.extend_from_slice(input);
        let mut frames = Vec::new();
        while self.buf.len() >= self.frame {
            frames.push(self.buf.drain(..self.frame).collect());
        }
        frames
    }

    /// Samples buffered but not yet emitted as a frame.
    pub fn pending(&self) -> usize {
        self.buf.len()
    }

    /// Drop any buffered remainder (e.g. on stop / stream reset).
    pub fn reset(&mut self) {
        self.buf.clear();
    }
}

impl Default for Reblocker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_nothing_until_a_full_frame() {
        let mut r = Reblocker::with_frame(512);
        assert!(r.push(&vec![0.0; 100]).is_empty());
        assert_eq!(r.pending(), 100);
    }

    #[test]
    fn emits_one_frame_and_keeps_remainder() {
        let mut r = Reblocker::with_frame(512);
        let frames = r.push(&vec![0.0; 600]); // 600 → one 512 frame, 88 left
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].len(), 512);
        assert_eq!(r.pending(), 88);
    }

    #[test]
    fn emits_multiple_frames_across_pushes() {
        let mut r = Reblocker::with_frame(512);
        assert_eq!(r.push(&vec![0.0; 500]).len(), 0); // 500 buffered
        let frames = r.push(&vec![0.0; 600]); // 1100 total → two frames, 76 left
        assert_eq!(frames.len(), 2);
        assert_eq!(r.pending(), 76);
    }

    #[test]
    fn reset_drops_remainder() {
        let mut r = Reblocker::with_frame(512);
        r.push(&vec![0.0; 100]);
        r.reset();
        assert_eq!(r.pending(), 0);
    }
}
