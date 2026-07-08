//! Logical channel model for the multi-output routing layer (Phase 4).
//!
//! ARCHITECTURE §11 establishes the core rule: **channels own state, devices
//! display it**. A *channel* is a content stream with a defined audience and a
//! defined set of rules; a physical device (projector, NDI, OBS overlay, pastor
//! monitor, operator console) is a read-only subscriber. Binding state to
//! devices instead of channels is what causes the sync-drift / divergence bugs
//! the spec calls the "loophole" (§11.0).
//!
//! This module defines the three logical channels as distinct, serializable
//! payloads. Bullet 4.1 is types only — the pub-sub `emit`/`emit_to` wiring and
//! [`RoutingMode`](https://) are Bullet 4.2, and device-health monitoring is
//! Bullet 4.4. Those land additively on top of these structs.
//!
//! ## Cross-crate boundary
//! `rhema-broadcast` never imports `rhema-detection` (unidirectional data flow:
//! detection → broadcast, never the reverse). So the verse/detection payloads
//! below are plain data types defined locally. The `app` crate — which depends
//! on both — maps `rhema-detection` types into these at the emit site (4.2).

use serde::{Deserialize, Serialize};

/// A single verse rendered for display on any channel.
///
/// Plain presentation data: the coordinates plus the resolved text. It carries
/// no confidence, source, or detection metadata — those are operator-only
/// concerns that live on [`DetectionResult`] / [`SuggestedVerse`], never on the
/// verse payload an audience device might render.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct VerseDisplay {
    /// Book name as displayed (e.g. "Romans").
    pub book: String,
    pub chapter: u16,
    pub verse_start: u16,
    /// End verse for a range display; `None` for a single verse.
    pub verse_end: Option<u16>,
    /// Pre-formatted human reference (e.g. "Romans 8:1-4").
    pub reference: String,
    /// Resolved verse text in the active translation.
    pub text: String,
    /// Translation id (e.g. "KJV").
    pub translation: String,
}

/// One item awaiting operator action. Operator-channel only.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct QueueItem {
    /// Stable id for selection/removal from the queue.
    pub id: String,
    pub verse: VerseDisplay,
}

/// A raw detection surfaced to the operator, with its confidence and source.
///
/// `confidence` is deliberately on this type (and thus only reachable through
/// [`OperatorChannel`]) — it must never appear on the audience surface.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DetectionResult {
    pub verse: VerseDisplay,
    /// Detector confidence in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Originating detector (e.g. "direct", "semantic", "quotation").
    pub source: String,
}

/// A proactive suggestion from the sermon-intelligence engine (Phase 5).
///
/// Operator-channel only — suggestions never auto-project (GEMINI invariant #4).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SuggestedVerse {
    pub verse: VerseDisplay,
    /// Relevance score in `[0.0, 1.0]`.
    pub score: f32,
    /// Short human-readable rationale for the suggestion.
    pub reason: String,
}

/// How the audience and pastor channels relate (ARCHITECTURE §6.2 / §11.6).
///
/// The `navigation_mode` from the cursor (§6.1) is surfaced to the routing layer
/// as this `RoutingMode`. Wire format is snake_case (`"locked"`/`"preview"`/
/// `"independent"`) to match the Zustand store fields wired in Bullet 4.3.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum RoutingMode {
    /// Audience and Pastor channels always show the same verse. Default.
    #[default]
    Locked,
    /// Pastor channel shows the NEXT verse; audience shows the CURRENT one.
    Preview,
    /// Audience and Pastor channels navigate separately.
    Independent,
}

/// Connection state of a physical output endpoint (§11.8 device lifecycle).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeviceConnection {
    Connected,
    Disconnected,
    Reconnecting,
}

/// Health snapshot for one output endpoint (Bullet 4.4).
///
/// Serializable by construction: the last state-change time is a UNIX-epoch
/// millisecond count (`u64`), not an `Instant` (which cannot serialize). On
/// disconnect, `last_known_verse` records the verse that was live so the
/// operator can see channel state was preserved across the failure (§11.9).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DeviceStatus {
    /// Stable endpoint id, e.g. "ndi-main", "projector-alt", "obs".
    pub label: String,
    /// Endpoint family, e.g. "ndi", "projector", "obs".
    pub kind: String,
    pub connection: DeviceConnection,
    /// UNIX-epoch milliseconds of the last connection-state change.
    pub last_change_ms: u64,
    /// Verse that was live when this endpoint last disconnected; `None` while
    /// connected. Demonstrates channel state survives device failure.
    pub last_known_verse: Option<VerseDisplay>,
}

/// **Audience channel** — what the congregation sees.
///
/// Committed content ONLY. By construction it carries *no* suggestions, queue,
/// confidence scores, detections, or next-verse previews (ARCHITECTURE §11.0,
/// "the loophole killer"). The negative-contract test in this module
/// machine-enforces that exclusion so a future field can't silently leak
/// operator-only data onto an audience device.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct AudienceChannel {
    /// The verse currently live to the congregation; `None` when blank.
    pub active_verse: Option<VerseDisplay>,
    /// Theme id the audience devices render with.
    pub theme_id: String,
}

impl AudienceChannel {
    /// A blank audience channel (nothing live).
    pub fn new() -> Self {
        Self::default()
    }
}

/// **Pastor channel** — the stage/monitor view for the preacher.
///
/// Current verse plus an optional next-verse preview (Preview routing mode),
/// the active translation indicator, and a service timer. Decoupled from the
/// audience channel — it can sit on a different verse. The `RoutingMode` field
/// is added in Bullet 4.2.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct PastorChannel {
    /// Verse the pastor is currently on.
    pub current_verse: Option<VerseDisplay>,
    /// Upcoming verse shown only in Preview mode; `None` otherwise.
    pub preview_verse: Option<VerseDisplay>,
    /// Active translation id (e.g. "KJV").
    pub translation: String,
    /// Service timer in whole seconds; `None` when no timer is running.
    /// (Whole seconds — not `Duration` — for a clean JSON number on the wire.)
    pub timer_seconds: Option<u64>,
    /// Current routing relationship to the audience channel (Bullet 4.2).
    pub mode: RoutingMode,
}

impl PastorChannel {
    pub fn new() -> Self {
        Self::default()
    }
}

/// **Operator channel** — the supervision surface, control console only.
///
/// Everything the operator needs to make decisions: the queue, raw detections
/// with confidence, and proactive suggestions. Device health and routing state
/// land here additively in Bullets 4.4 and 4.2 respectively.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct OperatorChannel {
    pub queue: Vec<QueueItem>,
    pub detections: Vec<DetectionResult>,
    pub suggestions: Vec<SuggestedVerse>,
    /// Routing mode the operator currently has selected (Bullet 4.2).
    pub routing_state: RoutingMode,
    /// Health of every registered output endpoint (Bullet 4.4).
    pub device_health: Vec<DeviceStatus>,
}

impl OperatorChannel {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_verse() -> VerseDisplay {
        VerseDisplay {
            book: "Romans".to_string(),
            chapter: 8,
            verse_start: 1,
            verse_end: Some(4),
            reference: "Romans 8:1-4".to_string(),
            text: "There is therefore now no condemnation...".to_string(),
            translation: "KJV".to_string(),
        }
    }

    #[test]
    fn audience_channel_round_trips() {
        let channel = AudienceChannel {
            active_verse: Some(sample_verse()),
            theme_id: "classic".to_string(),
        };
        let json = serde_json::to_string(&channel).expect("serialize");
        let back: AudienceChannel = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.theme_id, "classic");
        assert_eq!(back.active_verse, Some(sample_verse()));
    }

    #[test]
    fn pastor_channel_round_trips() {
        let channel = PastorChannel {
            current_verse: Some(sample_verse()),
            preview_verse: None,
            translation: "KJV".to_string(),
            timer_seconds: Some(125),
            mode: RoutingMode::Preview,
        };
        let json = serde_json::to_string(&channel).expect("serialize");
        let back: PastorChannel = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.current_verse, Some(sample_verse()));
        assert_eq!(back.timer_seconds, Some(125));
        assert_eq!(back.preview_verse, None);
        assert_eq!(back.mode, RoutingMode::Preview);
    }

    #[test]
    fn operator_channel_round_trips() {
        let channel = OperatorChannel {
            queue: vec![QueueItem {
                id: "q1".to_string(),
                verse: sample_verse(),
            }],
            detections: vec![DetectionResult {
                verse: sample_verse(),
                confidence: 0.93,
                source: "direct".to_string(),
            }],
            suggestions: vec![SuggestedVerse {
                verse: sample_verse(),
                score: 0.81,
                reason: "matches sermon topic".to_string(),
            }],
            routing_state: RoutingMode::Independent,
            device_health: vec![DeviceStatus {
                label: "ndi-main".to_string(),
                kind: "ndi".to_string(),
                connection: DeviceConnection::Disconnected,
                last_change_ms: 1_700_000_000_000,
                last_known_verse: Some(sample_verse()),
            }],
        };
        let json = serde_json::to_string(&channel).expect("serialize");
        let back: OperatorChannel = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.queue.len(), 1);
        assert_eq!(back.detections.len(), 1);
        assert_eq!(back.suggestions.len(), 1);
        assert_eq!(back.queue[0].id, "q1");
        assert_eq!(back.routing_state, RoutingMode::Independent);
        assert_eq!(back.device_health.len(), 1);
        assert_eq!(back.device_health[0].connection, DeviceConnection::Disconnected);
        assert_eq!(back.device_health[0].last_known_verse, Some(sample_verse()));
    }

    #[test]
    fn device_status_round_trips_all_connection_states() {
        for state in [
            DeviceConnection::Connected,
            DeviceConnection::Disconnected,
            DeviceConnection::Reconnecting,
        ] {
            let status = DeviceStatus {
                label: "obs".to_string(),
                kind: "obs".to_string(),
                connection: state,
                last_change_ms: 42,
                last_known_verse: None,
            };
            let json = serde_json::to_string(&status).expect("serialize");
            let back: DeviceStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back.connection, state);
            assert_eq!(back.label, "obs");
        }
    }

    #[test]
    fn routing_mode_serializes_snake_case() {
        assert_eq!(serde_json::to_string(&RoutingMode::Locked).unwrap(), "\"locked\"");
        assert_eq!(serde_json::to_string(&RoutingMode::Preview).unwrap(), "\"preview\"");
        assert_eq!(
            serde_json::to_string(&RoutingMode::Independent).unwrap(),
            "\"independent\""
        );
        let back: RoutingMode = serde_json::from_str("\"preview\"").unwrap();
        assert_eq!(back, RoutingMode::Preview);
        assert_eq!(RoutingMode::default(), RoutingMode::Locked);
    }

    /// The bullet's core contract: the audience surface must never carry
    /// operator-only data. Enforced on the serialized JSON so adding a field
    /// later that leaks any of these keys breaks this test loudly.
    #[test]
    fn audience_channel_excludes_operator_only_fields() {
        let channel = AudienceChannel {
            active_verse: Some(sample_verse()),
            theme_id: "classic".to_string(),
        };
        let json = serde_json::to_string(&channel).expect("serialize");
        for forbidden in [
            "confidence",
            "queue",
            "suggestion",
            "detection",
            "preview",
        ] {
            assert!(
                !json.contains(forbidden),
                "AudienceChannel JSON must not contain `{forbidden}`: {json}"
            );
        }
    }

    /// Confidence belongs on the operator channel — proves the exclusion above
    /// is specific to the audience surface, not accidental everywhere.
    #[test]
    fn operator_channel_includes_confidence() {
        let channel = OperatorChannel {
            queue: Vec::new(),
            detections: vec![DetectionResult {
                verse: sample_verse(),
                confidence: 0.67,
                source: "semantic".to_string(),
            }],
            suggestions: Vec::new(),
            routing_state: RoutingMode::default(),
            device_health: Vec::new(),
        };
        let json = serde_json::to_string(&channel).expect("serialize");
        assert!(json.contains("confidence"), "operator JSON should expose confidence: {json}");
    }
}
