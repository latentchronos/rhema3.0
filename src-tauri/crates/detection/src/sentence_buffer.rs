use std::time::Instant;

/// Default timeout: flush the buffer if no new text arrives within 3 seconds.
const DEFAULT_FLUSH_TIMEOUT_MS: u128 = 3000;

/// Maximum word count before forcing a flush (prevents run-on chunks from fast speakers).
const MAX_WORDS: usize = 20;

/// Accumulates transcript fragments into complete sentences before
/// passing them to the detection pipeline.
///
/// Deepgram sends many small `is_final` chunks per utterance. Running
/// detection on each fragment produces worse results than running on
/// complete sentences. This buffer:
///
/// 1. Accumulates `is_final` text fragments.
/// 2. Flushes when sentence-ending punctuation (`.!?`) is detected.
/// 3. Flushes on timeout if no punctuation arrives (fallback).
/// 4. Flushes on `speech_final` signal (Deepgram utterance boundary).
pub struct SentenceBuffer {
    buffer: String,
    last_append: Option<Instant>,
    flush_timeout_ms: u128,
}

impl SentenceBuffer {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            last_append: None,
            flush_timeout_ms: DEFAULT_FLUSH_TIMEOUT_MS,
        }
    }

    /// Append a transcript fragment. Returns the flushed sentence if a
    /// sentence boundary was detected, or None if still accumulating.
    pub fn append(&mut self, text: &str) -> Option<String> {
        if text.is_empty() {
            return None;
        }

        if !self.buffer.is_empty() {
            self.buffer.push(' ');
        }
        self.buffer.push_str(text);
        self.last_append = Some(Instant::now());

        // Check for sentence-ending punctuation (natural boundary — wins over word cap)
        let trimmed = self.buffer.trim_end();
        if trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?') {
            return Some(self.flush());
        }

        // Safety cap: flush if the buffer has grown beyond MAX_WORDS
        if self.buffer.split_whitespace().count() >= MAX_WORDS {
            return Some(self.flush());
        }

        None
    }

    /// Update the flush timeout based on the speaker's observed inter-word gap.
    ///
    /// `gap_secs`: measured silence between words. `None` resets to the warm-up default.
    /// The computed timeout is `2.5 × gap_secs`, clamped to [800 ms, 5000 ms].
    pub fn set_adaptive_timeout(&mut self, gap_secs: Option<f64>) {
        self.flush_timeout_ms = match gap_secs {
            Some(g) => (2.5 * g * 1000.0).clamp(800.0, 5000.0) as u128,
            None => DEFAULT_FLUSH_TIMEOUT_MS,
        };
    }

    /// Returns the current flush timeout in milliseconds (exposed for testing).
    pub fn flush_timeout_ms(&self) -> u128 {
        self.flush_timeout_ms
    }

    /// Check if the buffer should be flushed due to timeout.
    /// Call this periodically (e.g., on each event loop iteration).
    /// Returns the buffered text if timeout exceeded, None otherwise.
    pub fn check_timeout(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }
        if let Some(last) = self.last_append {
            if last.elapsed().as_millis() >= self.flush_timeout_ms {
                return Some(self.flush());
            }
        }
        None
    }

    /// Force-flush the buffer (e.g., on speech_final signal).
    /// Returns the buffered text, or None if empty.
    pub fn force_flush(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }
        Some(self.flush())
    }

    /// Returns true if the buffer has accumulated text.
    pub fn has_content(&self) -> bool {
        !self.buffer.is_empty()
    }

    fn flush(&mut self) -> String {
        let text = self.buffer.trim().to_string();
        self.buffer.clear();
        self.last_append = None;
        text
    }
}

impl Default for SentenceBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_ending_flushes() {
        let mut buf = SentenceBuffer::new();
        assert!(buf.append("God so loved").is_none());
        assert!(buf.append("the world.").is_some());
    }

    #[test]
    fn test_accumulates_fragments() {
        let mut buf = SentenceBuffer::new();
        buf.append("God so");
        buf.append("loved the");
        let result = buf.append("world.");
        assert_eq!(result.unwrap(), "God so loved the world.");
    }

    #[test]
    fn test_force_flush() {
        let mut buf = SentenceBuffer::new();
        buf.append("God so loved the world");
        let result = buf.force_flush();
        assert_eq!(result.unwrap(), "God so loved the world");
        assert!(!buf.has_content());
    }

    #[test]
    fn test_empty_buffer() {
        let mut buf = SentenceBuffer::new();
        assert!(buf.force_flush().is_none());
        assert!(buf.check_timeout().is_none());
    }

    #[test]
    fn test_question_mark_flushes() {
        let mut buf = SentenceBuffer::new();
        let result = buf.append("What does John 3:16 say?");
        assert!(result.is_some());
    }

    #[test]
    fn test_exclamation_flushes() {
        let mut buf = SentenceBuffer::new();
        buf.append("Praise the Lord");
        let result = buf.append("for He is good!");
        assert!(result.is_some());
    }

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
}
