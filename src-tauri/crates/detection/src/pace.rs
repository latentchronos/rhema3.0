/// Time-aware exponential moving average estimator for inter-word gap.
///
/// Feeds on Deepgram-final fragments and tracks the speaker's current pace as
/// the smoothed average gap (in seconds) between consecutive words.
pub struct PaceEstimator {
    tau_secs: f64,
    gap: Option<f64>,
    valid_count: usize,
}

impl PaceEstimator {
    /// Create a new estimator.
    ///
    /// `tau_secs` is the EMA time constant in seconds; callers typically pass `4.0`.
    pub fn new(tau_secs: f64) -> Self {
        Self {
            tau_secs,
            gap: None,
            valid_count: 0,
        }
    }

    /// Feed one Deepgram-final sample.
    ///
    /// - `n_words`   – number of words in the fragment
    /// - `span_secs` – `last_word.end - first_word.start` in seconds
    /// - `dt_secs`   – wall-clock seconds elapsed since the previous `observe` call
    ///
    /// A sample is valid only when `n_words >= 2` AND `span_secs >= 0.4`.
    /// Invalid samples are silently ignored.
    pub fn observe(&mut self, n_words: usize, span_secs: f64, dt_secs: f64) {
        if n_words < 2 || span_secs < 0.4 {
            return;
        }

        let gap_sample = span_secs / (n_words as f64 - 1.0);

        match self.gap {
            None => {
                // First valid sample: seed directly.
                self.gap = Some(gap_sample);
            }
            Some(prior) => {
                let alpha = 1.0 - (-dt_secs / self.tau_secs).exp();
                self.gap = Some(alpha * gap_sample + (1.0 - alpha) * prior);
            }
        }

        self.valid_count += 1;
    }

    /// Current smoothed average inter-word gap in seconds.
    ///
    /// Returns `None` until at least **3 valid samples** have been observed.
    pub fn gap_secs(&self) -> Option<f64> {
        if self.valid_count >= 3 {
            self.gap
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warms_up_then_tracks_pace_change() {
        let mut p = PaceEstimator::new(4.0);
        assert_eq!(p.gap_secs(), None); // cold: not enough data
        // fast speaker: 5 words over 1.0s -> 0.25s/gap
        for _ in 0..5 {
            p.observe(5, 1.0, 0.5);
        }
        let fast = p.gap_secs().unwrap();
        assert!(fast < 0.4, "fast gap should be small, got {fast}");
        // speaker slows: 3 words over 4.5s -> ~2.25s/gap
        for _ in 0..8 {
            p.observe(3, 4.5, 1.5);
        }
        let slow = p.gap_secs().unwrap();
        assert!(slow > 1.0, "should glide upward to slow, got {slow}");
    }

    #[test]
    fn rejects_invalid_samples() {
        let mut p = PaceEstimator::new(4.0);
        p.observe(1, 0.0, 0.5); // single word, no span -> ignored
        p.observe(0, 1.0, 0.5); // no words -> ignored
        assert_eq!(p.gap_secs(), None);
    }
}
