//! The streaming committer: `committed` + `tentative` snapshots → `Final`/`Partial` segments.
//!
//! A cache-aware streaming engine exposes an append-only `committed` prefix and a volatile
//! `tentative` tail. This converts successive snapshots into an ordered segment stream:
//! completed phrases become `Final`s (stable — safe to drive detection/projection), and the
//! in-progress remainder becomes a `Partial` (disposable preview). It tracks how far into
//! `committed` has already been finalized so each phrase is emitted exactly once.
//!
//! Endpoint policy (what makes a `Final`), earliest wins:
//! 1. the model's own end-of-utterance / end-of-burst token (`<EOU>` / `<EOB>`), or
//! 2. sentence-ending punctuation, or
//! 3. a char-count safety flush (`flush_chars`) for long runs with neither — so the live
//!    `Partial` (and any detector re-scanning it) never carries an unbounded remainder.
//!
//! Extracted verbatim from the STT engine's former `emit_stream_text`; behaviour is
//! unchanged, but it now returns segments instead of sending events, so callers own I/O.

/// A piece of transcript ready to emit. Byte-for-byte the text a caller should surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptSegment {
    /// Stable, will not change — safe to finalize downstream (detection/projection).
    Final(String),
    /// The current in-progress hypothesis — disposable preview only.
    Partial(String),
}

/// Default char-count safety flush: force-finalize a committed run this long even without
/// an `<EOU>` or punctuation, broken at a word boundary. The streaming model rarely emits
/// periods, so without this the un-finalized remainder — and every `Partial` re-sending it
/// — would grow without bound.
pub const DEFAULT_FLUSH_CHARS: usize = 80;

/// Tracks finalize progress across successive streaming snapshots. One per stream; `reset`
/// on stream recreate (a fresh stream's `committed` restarts empty).
pub struct StreamCommitter {
    /// Byte offset into the current stream's `committed` prefix already finalized.
    final_upto: usize,
    flush_chars: usize,
}

impl StreamCommitter {
    pub fn new() -> Self {
        Self {
            final_upto: 0,
            flush_chars: DEFAULT_FLUSH_CHARS,
        }
    }

    /// A committer with a custom char-count safety-flush threshold.
    pub fn with_flush_chars(flush_chars: usize) -> Self {
        Self {
            final_upto: 0,
            flush_chars,
        }
    }

    /// Reset the finalize offset (new/recreated stream: `committed` starts empty again).
    pub fn reset(&mut self) {
        self.final_upto = 0;
    }

    /// Skip finalizing everything up to `committed_len` — used after replaying pre-roll on
    /// recovery, so the replayed (already-lost) transcript isn't re-emitted.
    pub fn skip_to(&mut self, committed_len: usize) {
        self.final_upto = committed_len;
    }

    /// Convert the current snapshot into segments to emit, in order.
    ///
    /// `flush = true` (stream end) emits the whole remainder as a `Final` instead of a
    /// `Partial`. Offsets track the RAW `committed` string (append-only, stable); tag
    /// stripping — which changes byte lengths — is applied only to emitted text.
    pub fn advance(
        &mut self,
        committed: &str,
        tentative: &str,
        flush: bool,
    ) -> Vec<TranscriptSegment> {
        let mut out = Vec::new();
        let mut cut = self.final_upto.min(committed.len());

        // Each completed phrase in the newly-committed region → its own Final.
        while let Some(end) = next_boundary(committed, cut) {
            let seg = strip_tags(&committed[cut..end]);
            let seg = seg.trim();
            if !seg.is_empty() {
                out.push(TranscriptSegment::Final(seg.to_string()));
            }
            cut = end;
        }

        // Long committed run with no boundary → flush at the last word boundary. Committed
        // text is stable, so this is lossless; it just bounds the live remainder.
        while committed.len().saturating_sub(cut) > self.flush_chars {
            match committed[cut..].rfind(' ') {
                Some(rel) if rel > 0 => {
                    let boundary = cut + rel + 1;
                    let seg = strip_tags(&committed[cut..boundary]);
                    let seg = seg.trim();
                    if !seg.is_empty() {
                        out.push(TranscriptSegment::Final(seg.to_string()));
                    }
                    cut = boundary;
                }
                // One unbroken token longer than the cap — nothing safe to split on yet.
                _ => break,
            }
        }
        self.final_upto = cut;

        let remainder = strip_tags(&format!("{}{}", &committed[cut..], tentative));
        let remainder = remainder.trim();
        if !remainder.is_empty() {
            out.push(if flush {
                TranscriptSegment::Final(remainder.to_string())
            } else {
                TranscriptSegment::Partial(remainder.to_string())
            });
        }
        out
    }
}

impl Default for StreamCommitter {
    fn default() -> Self {
        Self::new()
    }
}

/// Byte index just past the next hard phrase boundary at or after `from`: an end-of-
/// utterance / end-of-burst token (`<EOU>`/`<EOB>`) or sentence-ending punctuation,
/// whichever comes first. The tag itself is stripped from emitted text by [`strip_tags`];
/// here only its position matters.
fn next_boundary(s: &str, from: usize) -> Option<usize> {
    let hay = &s[from..];
    [
        hay.find(['.', '?', '!']).map(|rel| from + rel + 1),
        hay.find("<EOU>").map(|rel| from + rel + "<EOU>".len()),
        hay.find("<EOB>").map(|rel| from + rel + "<EOB>".len()),
    ]
    .into_iter()
    .flatten()
    .min()
}

/// Drop end-of-utterance / end-of-burst control tokens from surfaced text.
fn strip_tags(s: &str) -> String {
    s.replace("<EOU>", "").replace("<EOB>", "")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finals(segs: &[TranscriptSegment]) -> Vec<&str> {
        segs.iter()
            .filter_map(|s| match s {
                TranscriptSegment::Final(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    fn partial(segs: &[TranscriptSegment]) -> Option<&str> {
        segs.iter().find_map(|s| match s {
            TranscriptSegment::Partial(t) => Some(t.as_str()),
            _ => None,
        })
    }

    #[test]
    fn next_boundary_cuts_at_eou_before_punctuation() {
        let s = "turn to john <EOU> three sixteen.";
        let end = next_boundary(s, 0).unwrap();
        assert_eq!(&s[..end], "turn to john <EOU>");
        assert_eq!(strip_tags(&s[..end]).trim(), "turn to john");
    }

    #[test]
    fn next_boundary_cuts_at_punctuation_when_no_tag() {
        let s = "hello world. and more";
        assert_eq!(&s[..next_boundary(s, 0).unwrap()], "hello world.");
    }

    #[test]
    fn next_boundary_none_when_no_boundary() {
        assert!(next_boundary("no boundary yet", 0).is_none());
    }

    #[test]
    fn eou_committed_phrase_becomes_a_final_and_remainder_is_partial() {
        let mut c = StreamCommitter::new();
        let segs = c.advance("turn to john <EOU> and then", "", false);
        assert_eq!(finals(&segs), vec!["turn to john"]);
        assert_eq!(partial(&segs), Some("and then"));
    }

    #[test]
    fn committed_is_finalized_exactly_once_across_snapshots() {
        let mut c = StreamCommitter::new();
        // First snapshot finalizes the EOU phrase.
        let first = c.advance("first phrase. ", "", false);
        assert_eq!(finals(&first), vec!["first phrase."]);
        // Next snapshot grows committed; the already-finalized phrase is NOT re-emitted.
        let second = c.advance("first phrase. second phrase.", "", false);
        assert_eq!(finals(&second), vec!["second phrase."]);
    }

    #[test]
    fn tentative_is_partial_not_final_until_committed() {
        let mut c = StreamCommitter::new();
        let segs = c.advance("", "live words", false);
        assert!(finals(&segs).is_empty());
        assert_eq!(partial(&segs), Some("live words"));
    }

    #[test]
    fn flush_emits_remainder_as_final() {
        let mut c = StreamCommitter::new();
        // committed/tentative are concatenated raw (spacing lives in the tokens), so the
        // committed prefix carries its own trailing space — as real streaming output does.
        let segs = c.advance("no punctuation here ", "tail", true);
        assert_eq!(finals(&segs), vec!["no punctuation here tail"]);
        assert!(partial(&segs).is_none());
    }

    #[test]
    fn char_count_safety_flush_breaks_long_run_at_word_boundary() {
        let mut c = StreamCommitter::with_flush_chars(10);
        // 20 chars, no boundary → flushed at a word boundary once over the 10-char cap.
        let segs = c.advance("alpha beta gamma delta", "", false);
        // Everything up to the last safe word boundary is finalized; the tail is Partial.
        assert!(!finals(&segs).is_empty(), "expected a safety-flush Final");
        assert!(finals(&segs).iter().all(|f| !f.contains("  ")));
    }

    #[test]
    fn skip_to_suppresses_replayed_transcript() {
        let mut c = StreamCommitter::new();
        // Simulate recovery: the fresh stream already committed some replayed text we don't
        // want re-emitted.
        c.skip_to("replayed. ".len());
        let segs = c.advance("replayed. new content.", "", false);
        assert_eq!(finals(&segs), vec!["new content."]);
    }
}
