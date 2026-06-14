//! Channel routing layer — the backend pub-sub publisher (Phase 4, Bullet 4.2).
//!
//! ARCHITECTURE §11: logical channels own state; physical devices subscribe and
//! display. This module holds the authoritative [`ChannelState`] and publishes it
//! over Tauri's event system:
//!
//! * **Audience** → [`Emitter::emit`] (ALL windows). Reaches every projector /
//!   NDI / OBS output window. The operator (main) window simply ignores it.
//! * **Pastor** / **Operator** → [`Emitter::emit_to`] a specific window label,
//!   so operator-only data (confidence, detections, suggestions) never reaches an
//!   audience device.
//!
//! The Tauri-facing emit logic lives here in the `app` crate, not in
//! `rhema-broadcast`, because that crate has no `tauri` dependency (and must not
//! — unidirectional data flow). `rhema-broadcast` owns the serializable channel
//! *types*; this module owns the *delivery*.

use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rhema_broadcast::ndi::NdiRuntime;
use rhema_broadcast::{
    AudienceChannel, DetectionResult as ChannelDetection, DeviceConnection, DeviceStatus,
    OperatorChannel, PastorChannel, RoutingMode, VerseDisplay,
};
use tauri::{AppHandle, Emitter, Manager, State};

use crate::commands::obs::ObsOverlayServer;

/// Health-probe interval (ARCHITECTURE §11.9 — "every 2 s").
const PING_INTERVAL_SECS: u64 = 2;

/// Window label for the operator console (the main application window).
const OPERATOR_WINDOW: &str = "main";
/// Reserved window label for a dedicated pastor monitor. The window is not built
/// yet (a later output-rollout stage); `emit_to` to a missing label is a
/// harmless no-op, which is the forward-compatible behaviour we want.
const PASTOR_WINDOW: &str = "pastor";

/// Tauri event names (snake_case per the IPC convention).
const AUDIENCE_EVENT: &str = "audience_channel_update";
const PASTOR_EVENT: &str = "pastor_channel_update";
const OPERATOR_EVENT: &str = "operator_channel_update";

/// Authoritative backend channel state — the single source of truth (§11.1).
/// Zustand mirrors it via event listeners in Bullet 4.3; it is never the reverse.
#[derive(Default)]
pub struct ChannelState {
    pub audience: AudienceChannel,
    pub pastor: PastorChannel,
    pub operator: OperatorChannel,
    pub routing_mode: RoutingMode,
}

/// Map an app-level detection into the broadcast channel payload.
///
/// `verse_end`/`translation` are not carried by the app `DetectionResult`
/// (translation is a pastor-channel indicator set on commit), so they default to
/// `None`/empty here — this is the operator's raw-detection view.
fn to_channel_detection(d: &crate::commands::detection::DetectionResult) -> ChannelDetection {
    ChannelDetection {
        verse: VerseDisplay {
            book: d.book_name.clone(),
            chapter: d.chapter as u16,
            verse_start: d.verse as u16,
            verse_end: None,
            reference: d.verse_ref.clone(),
            text: d.verse_text.clone(),
            translation: String::new(),
        },
        confidence: d.confidence as f32,
        source: d.source.clone(),
    }
}

/// Publish the audience channel to ALL windows (`emit`).
fn publish_audience(app: &AppHandle, channel: &AudienceChannel) {
    let _ = app.emit(AUDIENCE_EVENT, channel);
}

/// Publish the pastor channel to the pastor window only (`emit_to`).
fn publish_pastor(app: &AppHandle, channel: &PastorChannel) {
    let _ = app.emit_to(PASTOR_WINDOW, PASTOR_EVENT, channel);
}

/// Publish the operator channel to the operator console window only (`emit_to`).
fn publish_operator(app: &AppHandle, channel: &OperatorChannel) {
    let _ = app.emit_to(OPERATOR_WINDOW, OPERATOR_EVENT, channel);
}

/// Apply a "go live" commit to the channel state, honouring the routing mode.
/// Pure (no IPC) so it is unit-testable without a running app.
///
/// * `Locked`      — pastor mirrors the audience verse; any preview is cleared.
/// * `Preview`     — pastor's current follows the audience verse, but a held
///                   preview verse is preserved (managed by a separate action).
/// * `Independent` — the pastor channel is left untouched (separate navigation).
fn apply_commit(state: &mut ChannelState, verse: Option<VerseDisplay>, theme_id: String) {
    state.audience.active_verse = verse.clone();
    state.audience.theme_id = theme_id;

    match state.routing_mode {
        RoutingMode::Locked => {
            state.pastor.current_verse = verse;
            state.pastor.preview_verse = None;
        }
        RoutingMode::Preview => {
            state.pastor.current_verse = verse;
            // preview_verse intentionally preserved
        }
        RoutingMode::Independent => {
            // Pastor navigates independently — do not move it on an audience commit.
        }
    }

    if let Some(v) = &state.pastor.current_verse {
        state.pastor.translation = v.translation.clone();
    }
}

/// Route detections to the **Operator channel** (the Gap-C egress integration
/// point). Called from `emit_detections` after suppression. Does not hold the
/// state lock across the emit.
pub fn route_detections(app: &AppHandle, kept: &[crate::commands::detection::DetectionResult]) {
    let managed: State<'_, Mutex<ChannelState>> = app.state();
    let snapshot = match managed.lock() {
        Ok(mut state) => {
            state.operator.detections = kept.iter().map(to_channel_detection).collect();
            state.operator.routing_state = state.routing_mode;
            state.operator.clone()
        }
        Err(_) => return,
    };
    publish_operator(app, &snapshot);
}

/// Operator action: set the routing mode and re-publish the pastor + operator
/// channels so all subscribers reflect the change. The frontend toggle (Bullet
/// 4.4) calls this; nothing auto-projects as a result.
#[tauri::command]
pub fn set_routing_mode(
    app: AppHandle,
    state: State<'_, Mutex<ChannelState>>,
    mode: RoutingMode,
) -> Result<(), String> {
    let (pastor, operator) = {
        let mut s = state.lock().map_err(|e| e.to_string())?;
        s.routing_mode = mode;
        s.pastor.mode = mode;
        s.operator.routing_state = mode;
        (s.pastor.clone(), s.operator.clone())
    };
    log::info!("routing: mode set to {mode:?}");
    publish_pastor(&app, &pastor);
    publish_operator(&app, &operator);
    Ok(())
}

/// Operator action: commit a verse live. Updates the authoritative channel state
/// per the current routing mode and publishes the audience + pastor channels.
/// (Frontend wiring lands in Bullet 4.3.)
#[tauri::command]
pub fn commit_live_verse(
    app: AppHandle,
    state: State<'_, Mutex<ChannelState>>,
    verse: Option<VerseDisplay>,
    theme_id: String,
) -> Result<(), String> {
    let (audience, pastor) = {
        let mut s = state.lock().map_err(|e| e.to_string())?;
        apply_commit(&mut s, verse, theme_id);
        (s.audience.clone(), s.pastor.clone())
    };
    publish_audience(&app, &audience);
    publish_pastor(&app, &pastor);
    Ok(())
}

// --- Device health monitor (Bullet 4.4) ------------------------------------

/// A single device liveness reading taken during a probe sweep.
struct DeviceProbe {
    label: &'static str,
    kind: &'static str,
    connected: bool,
}

/// Result of reconciling a fresh probe sweep against the previous snapshot.
struct Reconciled {
    statuses: Vec<DeviceStatus>,
    /// `(label, now_connected)` for each endpoint that changed state — for logs.
    transitions: Vec<(String, bool)>,
    /// True if any endpoint transitioned Disconnected → Connected (needs resync).
    reconnected: bool,
    /// True if any endpoint's connection state changed (or is newly seen).
    changed: bool,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Pure reconciliation: diff a probe sweep against the previous health snapshot.
/// On disconnect the currently-live verse is captured as `last_known_verse`
/// (channel state is preserved, never reset). No IPC — unit-testable.
fn reconcile(
    prev: &[DeviceStatus],
    probes: &[DeviceProbe],
    now_ms: u64,
    current_verse: Option<&VerseDisplay>,
) -> Reconciled {
    let mut statuses = Vec::with_capacity(probes.len());
    let mut transitions = Vec::new();
    let mut reconnected = false;
    let mut changed = false;

    for probe in probes {
        let prior = prev.iter().find(|d| d.label == probe.label);
        match prior {
            Some(p) => {
                let was_connected = matches!(p.connection, DeviceConnection::Connected);
                if was_connected == probe.connected {
                    // No change — carry the prior status forward verbatim.
                    statuses.push(p.clone());
                } else {
                    changed = true;
                    transitions.push((probe.label.to_string(), probe.connected));
                    if probe.connected {
                        reconnected = true;
                    }
                    statuses.push(DeviceStatus {
                        label: probe.label.to_string(),
                        kind: probe.kind.to_string(),
                        connection: if probe.connected {
                            DeviceConnection::Connected
                        } else {
                            DeviceConnection::Disconnected
                        },
                        last_change_ms: now_ms,
                        last_known_verse: if probe.connected {
                            None
                        } else {
                            current_verse.cloned()
                        },
                    });
                }
            }
            None => {
                // First time we have seen this endpoint.
                changed = true;
                statuses.push(DeviceStatus {
                    label: probe.label.to_string(),
                    kind: probe.kind.to_string(),
                    connection: if probe.connected {
                        DeviceConnection::Connected
                    } else {
                        DeviceConnection::Disconnected
                    },
                    last_change_ms: now_ms,
                    last_known_verse: if probe.connected {
                        None
                    } else {
                        current_verse.cloned()
                    },
                });
            }
        }
    }

    Reconciled {
        statuses,
        transitions,
        reconnected,
        changed,
    }
}

/// Probe every registered output endpoint for liveness.
fn probe_devices(app: &AppHandle) -> Vec<DeviceProbe> {
    let mut probes = Vec::new();

    if let Some(ndi) = app.try_state::<Mutex<NdiRuntime>>() {
        if let Ok(rt) = ndi.lock() {
            probes.push(DeviceProbe {
                label: "ndi-main",
                kind: "ndi",
                connected: rt.is_active("main"),
            });
            probes.push(DeviceProbe {
                label: "ndi-alt",
                kind: "ndi",
                connected: rt.is_active("alt"),
            });
        }
    }

    probes.push(DeviceProbe {
        label: "projector-main",
        kind: "projector",
        connected: app.get_webview_window("broadcast").is_some(),
    });
    probes.push(DeviceProbe {
        label: "projector-alt",
        kind: "projector",
        connected: app.get_webview_window("broadcast-alt").is_some(),
    });

    if let Some(obs) = app.try_state::<Mutex<ObsOverlayServer>>() {
        if let Ok(srv) = obs.lock() {
            probes.push(DeviceProbe {
                label: "obs",
                kind: "obs",
                connected: srv.is_running(),
            });
        }
    }

    probes
}

/// Spawn the 2-second device-health monitor (ARCHITECTURE §11.9). On any change
/// it updates `operator.device_health` and re-publishes the operator channel; a
/// reconnect additionally re-publishes the audience channel so the returning
/// device resyncs the current verse. Channel content state is never reset.
pub fn spawn_device_health_monitor(app: AppHandle) {
    let spawn = std::thread::Builder::new()
        .name("device-health".to_string())
        .spawn(move || {
            let mut prev: Vec<DeviceStatus> = Vec::new();
            loop {
                std::thread::sleep(Duration::from_secs(PING_INTERVAL_SECS));

                let probes = probe_devices(&app);
                let now = now_ms();

                let current_verse = {
                    let managed: State<'_, Mutex<ChannelState>> = app.state();
                    let verse = match managed.lock() {
                        Ok(s) => s.audience.active_verse.clone(),
                        Err(_) => continue,
                    };
                    verse
                };

                let result = reconcile(&prev, &probes, now, current_verse.as_ref());
                if !result.changed {
                    prev = result.statuses;
                    continue;
                }

                for (label, connected) in &result.transitions {
                    if *connected {
                        log::info!(
                            "device_health: {label} reconnected, resyncing channel state"
                        );
                    } else {
                        log::warn!(
                            "device_health: {label} disconnected, preserving channel state"
                        );
                    }
                }

                let snapshots = {
                    let managed: State<'_, Mutex<ChannelState>> = app.state();
                    let snap = match managed.lock() {
                        Ok(mut s) => {
                            s.operator.device_health = result.statuses.clone();
                            s.operator.routing_state = s.routing_mode;
                            Some((s.operator.clone(), s.audience.clone()))
                        }
                        Err(_) => None,
                    };
                    snap
                };

                if let Some((operator_snapshot, audience_snapshot)) = snapshots {
                    publish_operator(&app, &operator_snapshot);
                    if result.reconnected {
                        publish_audience(&app, &audience_snapshot);
                    }
                }

                prev = result.statuses;
            }
        });
    if let Err(e) = spawn {
        log::error!("device_health: failed to spawn monitor thread: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verse(reference: &str) -> VerseDisplay {
        VerseDisplay {
            book: "Romans".to_string(),
            chapter: 8,
            verse_start: 1,
            verse_end: None,
            reference: reference.to_string(),
            text: "text".to_string(),
            translation: "KJV".to_string(),
        }
    }

    #[test]
    fn locked_mirrors_pastor_to_audience_and_clears_preview() {
        let mut state = ChannelState {
            routing_mode: RoutingMode::Locked,
            ..Default::default()
        };
        state.pastor.preview_verse = Some(verse("Romans 8:2"));

        apply_commit(&mut state, Some(verse("Romans 8:1")), "classic".to_string());

        assert_eq!(state.audience.active_verse, Some(verse("Romans 8:1")));
        assert_eq!(state.audience.theme_id, "classic");
        assert_eq!(state.pastor.current_verse, Some(verse("Romans 8:1")));
        assert_eq!(state.pastor.preview_verse, None, "Locked clears preview");
        assert_eq!(state.pastor.translation, "KJV");
    }

    #[test]
    fn preview_advances_current_but_keeps_preview() {
        let mut state = ChannelState {
            routing_mode: RoutingMode::Preview,
            ..Default::default()
        };
        state.pastor.preview_verse = Some(verse("Romans 8:2"));

        apply_commit(&mut state, Some(verse("Romans 8:1")), "classic".to_string());

        assert_eq!(state.audience.active_verse, Some(verse("Romans 8:1")));
        assert_eq!(state.pastor.current_verse, Some(verse("Romans 8:1")));
        assert_eq!(
            state.pastor.preview_verse,
            Some(verse("Romans 8:2")),
            "Preview preserves the held preview verse"
        );
    }

    #[test]
    fn independent_leaves_pastor_untouched() {
        let mut state = ChannelState {
            routing_mode: RoutingMode::Independent,
            ..Default::default()
        };
        state.pastor.current_verse = Some(verse("John 1:1"));

        apply_commit(&mut state, Some(verse("Romans 8:1")), "classic".to_string());

        assert_eq!(state.audience.active_verse, Some(verse("Romans 8:1")));
        assert_eq!(
            state.pastor.current_verse,
            Some(verse("John 1:1")),
            "Independent does not move the pastor channel on an audience commit"
        );
    }

    #[test]
    fn detection_maps_to_operator_payload() {
        let app_detection = crate::commands::detection::DetectionResult {
            verse_ref: "Romans 8:1".to_string(),
            verse_text: "There is therefore...".to_string(),
            book_name: "Romans".to_string(),
            book_number: 45,
            chapter: 8,
            verse: 1,
            confidence: 0.93,
            source: "direct".to_string(),
            auto_queued: false,
            raw_score: 0.93,
            minimum_threshold: 0.5,
            auto_queue_threshold: 0.9,
            decision: "surfaced".to_string(),
            explanation: String::new(),
            transcript_snippet: String::new(),
        };
        let mapped = to_channel_detection(&app_detection);
        assert_eq!(mapped.verse.book, "Romans");
        assert_eq!(mapped.verse.chapter, 8);
        assert_eq!(mapped.verse.verse_start, 1);
        assert_eq!(mapped.verse.reference, "Romans 8:1");
        assert_eq!(mapped.source, "direct");
        assert!((mapped.confidence - 0.93).abs() < 1e-6);
    }

    fn probe(label: &'static str, kind: &'static str, connected: bool) -> DeviceProbe {
        DeviceProbe {
            label,
            kind,
            connected,
        }
    }

    #[test]
    fn first_sweep_marks_all_changed() {
        let probes = vec![
            probe("ndi-main", "ndi", true),
            probe("obs", "obs", true),
        ];
        let r = reconcile(&[], &probes, 100, None);
        assert!(r.changed);
        assert!(!r.reconnected, "initial connections are not reconnects");
        assert_eq!(r.statuses.len(), 2);
        assert!(r
            .statuses
            .iter()
            .all(|s| s.connection == DeviceConnection::Connected));
    }

    #[test]
    fn stable_sweep_reports_no_change() {
        let probes = vec![probe("ndi-main", "ndi", true)];
        let first = reconcile(&[], &probes, 100, None);
        let second = reconcile(&first.statuses, &probes, 200, None);
        assert!(!second.changed);
        assert!(second.transitions.is_empty());
        // last_change_ms is carried forward, not refreshed, when unchanged.
        assert_eq!(second.statuses[0].last_change_ms, 100);
    }

    #[test]
    fn disconnect_preserves_last_known_verse() {
        let probes_up = vec![probe("projector-main", "projector", true)];
        let up = reconcile(&[], &probes_up, 100, None);

        let live = verse("Romans 8:4");
        let probes_down = vec![probe("projector-main", "projector", false)];
        let down = reconcile(&up.statuses, &probes_down, 300, Some(&live));

        assert!(down.changed);
        assert!(!down.reconnected);
        assert_eq!(down.statuses[0].connection, DeviceConnection::Disconnected);
        assert_eq!(down.statuses[0].last_known_verse, Some(live));
        assert_eq!(down.statuses[0].last_change_ms, 300);
    }

    #[test]
    fn reconnect_sets_resync_flag() {
        let down = vec![DeviceStatus {
            label: "ndi-main".to_string(),
            kind: "ndi".to_string(),
            connection: DeviceConnection::Disconnected,
            last_change_ms: 100,
            last_known_verse: Some(verse("Romans 8:1")),
        }];
        let probes_up = vec![probe("ndi-main", "ndi", true)];
        let r = reconcile(&down, &probes_up, 500, None);

        assert!(r.changed);
        assert!(r.reconnected, "Disconnected -> Connected must flag a resync");
        assert_eq!(r.statuses[0].connection, DeviceConnection::Connected);
        assert_eq!(r.statuses[0].last_known_verse, None);
    }
}
