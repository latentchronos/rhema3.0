/// Observe-only voice metrics logging.
/// All functions in this module emit `log::info!` lines with the stable prefix
/// `voice_metrics:` so they can be grepped from structured logs.

/// Pure formatter — no side effects.
/// Returns a string with prefix `voice_metrics:` that encodes the raw transcript
/// and whether the command was matched (`MATCH`) or missed (`MISS`).
pub fn render_command_attempt(raw: &str, parsed: Option<&str>) -> String {
    match parsed {
        Some(p) => format!("voice_metrics: command raw=\"{}\" parsed=\"{}\" MATCH", raw, p),
        None => format!("voice_metrics: command raw=\"{}\" MISS", raw),
    }
}

/// Emits one `log::info!` line for a command attempt.
pub fn log_command_attempt(raw: &str, parsed: Option<&str>) {
    log::info!("{}", render_command_attempt(raw, parsed));
}

/// Emits one `log::info!` line with current and smoothed inter-utterance gap.
pub fn log_pace(gap_secs: f64, smoothed_gap_secs: f64) {
    log::info!("voice_metrics: pace gap={:.3} smoothed={:.3}", gap_secs, smoothed_gap_secs);
}

/// Emits one `log::info!` line when a channel send is dropped.
pub fn log_channel_drop(channel: &'static str) {
    log::info!("voice_metrics: drop channel={}", channel);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_command_attempt_line() {
        let line = render_command_attempt("nest verse", Some("step:forward:verse:1"));
        assert!(line.contains("nest verse"));
        assert!(line.contains("step:forward:verse:1"));
        assert!(line.contains("MATCH"));
        let miss = render_command_attempt("next phase", None);
        assert!(miss.contains("MISS"));
    }
}
