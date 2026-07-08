# Phase 4 Companion — Channel Routing Layer (Outputs)

## Session Persona
You are a senior Tauri/React architect who has built multi-window broadcast
applications. You understand that in a live presentation system, a state
synchronisation bug means the wrong verse appears on the projector in front
of a congregation. You design for failure: every device can disconnect at any
time, and the system must recover gracefully without operator intervention.
You separate logical state from physical rendering as a hard architectural rule —
not a preference.

---

## Mandatory Reasoning Block
Before writing any code for any bullet in this phase, produce this block in full.

```
=== REASONING BLOCK ===
Bullet being implemented: [quote it exactly]
Current state model being replaced: [describe what exists NOW]
New state model being introduced: [describe the target]
Tauri IPC mechanism used: [events, commands, or both — and why]
Frontend stores affected: [which Zustand stores change and how]
Device disconnection scenario: [what happens to channel state if NDI drops mid-sermon?]
Migration risk: [what existing frontend components bind to the state being changed?
                 list them — do not assume they are unaffected]
Test for this bullet: [describe it]
=== END REASONING BLOCK ===
```

---

## Domain Knowledge: Phase 4 Concepts

### Why Screen-Bound State Causes Drift

The current model: `liveVerse` in `broadcast-store.ts` is a single field.
Both "main" and "alt" outputs read from it. The NDI frame is generated from
the React canvas. The OBS overlay reads from an SSE stream. The projector
webview reads from Tauri events.

**The problem:** These three rendering paths have different latencies:
- React canvas → NDI: ~16ms (requestAnimationFrame)
- Tauri event → projector webview: ~5ms
- SSE → OBS overlay: ~50–200ms (HTTP polling latency)

When `liveVerse` changes, all three update at different times. In a 200ms
window, the NDI feed shows verse A, the projector shows verse B, and the
OBS overlay still shows verse A. This is the sync drift problem.

**The fix:** State is managed by logical channels in the backend.
Physical outputs subscribe to their channel. The channel emits ONE authoritative
event. All subscribers update from the same event, not from shared Zustand state.

### The Three Logical Channels

```
AudienceChannel:
  Payload: { active_verse: VerseDisplay, theme_id: String }
  Subscribers: NDI stream, projector webview, OBS overlay
  Rules: ONLY displays confirmed/committed content
         Never shows suggestions, queue previews, or pending states
         Operator must explicitly "go live" to update this channel

PastorChannel:
  Payload: { current_verse: VerseDisplay, preview_verse: Option<VerseDisplay>,
             translation: String, timer: Option<Duration>, mode: RoutingMode }
  Subscribers: Pastor monitor window
  Rules: Shows current + optional next-verse preview (Preview mode)
         Receives detection results BEFORE operator confirmation
         Decoupled from AudienceChannel — can be on a different verse

OperatorChannel:
  Payload: { queue: Vec<QueueItem>, detections: Vec<DetectionResult>,
             confidence_scores: Vec<f32>, device_health: Vec<DeviceStatus>,
             routing_state: RoutingMode, suggestions: Vec<SuggestedVerse> }
  Subscribers: Operator console window only
  Rules: Everything the operator needs to make decisions
         Confidence scores, raw detections, device health visible here only
```

### Routing Modes

```rust
pub enum RoutingMode {
    Locked,      // AudienceChannel and PastorChannel show same verse
                 // Default mode for most services
    Preview,     // PastorChannel shows NEXT verse, AudienceChannel shows CURRENT
                 // Pastor sees what's coming before congregation does
    Independent, // AudienceChannel and PastorChannel navigate separately
                 // Used when pastor wants to look up a reference privately
}
```

### Publisher-Subscriber Architecture in Tauri

Tauri's event system is the pub-sub backbone. The Rust backend is the publisher.
Frontend windows are the subscribers.

**Backend (Rust) publishes:**
```rust
// Emit to ALL windows (broadcast)
app_handle.emit("audience_channel_update", &payload)?;

// Emit to a SPECIFIC window only
app_handle.emit_to("pastor-monitor", "pastor_channel_update", &payload)?;
app_handle.emit_to("operator-console", "operator_channel_update", &payload)?;
```

**Frontend (React) subscribes:**
```typescript
// In a useEffect hook or store initializer
const unlisten = await listen<AudienceChannelPayload>(
  'audience_channel_update',
  (event) => {
    // update local Zustand store from event payload
    useAudienceStore.getState().setActiveVerse(event.payload.active_verse)
  }
)
```

**Key principle:** Zustand stores become local caches of channel state,
NOT the source of truth. The Rust backend channel state IS the source of truth.
Frontend components read from Zustand (for React reactivity) but Zustand is
populated exclusively by Tauri event listeners.

### Device Health Monitoring

Each physical output device has a `DeviceStatus`:

```rust
pub enum DeviceStatus {
    Connected { last_ping: Instant },
    Disconnected { since: Instant, last_known_verse: VerseDisplay },
    Reconnecting,
}
```

**Ping interval:** Every 2 seconds, the backend checks each registered device.
For NDI: call `NdiRuntime::is_active()` — already exists.
For webview windows: Tauri window exists check.
For OBS SSE: check if the SSE client channel receiver is still alive.

**On disconnect:**
1. Update `DeviceStatus` to `Disconnected`, store `last_known_verse`
2. Emit `operator_channel_update` with updated device health
3. Channel state is PRESERVED — do not reset or clear
4. Log: `warn!("device_health: NDI disconnected, preserving channel state")`

**On reconnect:**
1. Detect reconnection (NDI becomes active again, new SSE client connects, window reopens)
2. Immediately emit the CURRENT channel state to the reconnected device
3. Update `DeviceStatus` to `Connected`
4. Log: `info!("device_health: NDI reconnected, resyncing verse {:?}", current_verse)`

### Zustand Store Migration

The existing `broadcast-store.ts` has `liveVerse`, `isLive`, `activeThemeId`,
`altActiveThemeId`. These must be migrated carefully.

**Do NOT delete existing fields in one commit.** Migration strategy:
1. Add new channel-model fields alongside existing fields
2. Wire new Tauri event listeners that populate the new fields
3. Once wired and tested, mark old fields as deprecated with a comment
4. Remove old fields only when explicitly instructed

**New stores to add (or new sections in existing store):**

```typescript
// Routing configuration (operator sets this)
routingMode: 'locked' | 'preview' | 'independent'
setRoutingMode: (mode: RoutingMode) => void

// Channel state (populated by Tauri event listeners only)
audienceChannel: AudienceChannelState | null
pastorChannel: PastorChannelState | null
operatorChannel: OperatorChannelState | null

// Device health (populated by operator_channel_update events)
deviceHealth: DeviceStatus[]
```

---

## Concrete Examples — Channel Routing

### Example A: Normal Locked Mode
```
Operator clicks Romans 8:1 from queue → "go live"
Backend: epoch increment, AudienceChannel.active_verse = Romans 8:1
         PastorChannel.current_verse = Romans 8:1 (locked = same)
Emit: audience_channel_update → ALL windows receive Romans 8:1
Emit: pastor_channel_update → pastor monitor receives Romans 8:1
NDI canvas: renders Romans 8:1 frame
Projector webview: displays Romans 8:1
OBS overlay: SSE event → displays Romans 8:1
All three: updated from same event within their respective latency windows
```

### Example B: Preview Mode
```
RoutingMode = Preview
AudienceChannel.active_verse = Romans 8:1 (currently shown to congregation)
Operator queues Romans 8:2 as next
PastorChannel.preview_verse = Romans 8:2
pastor_channel_update → pastor monitor shows Romans 8:2 as preview
audience_channel_update NOT emitted → congregation still sees Romans 8:1
Operator confirms → audience_channel_update emitted → congregation sees Romans 8:2
```

### Example C: NDI Disconnect + Reconnect
```
t=0:    NDI active, showing Romans 8:1
t=10s:  NDI cable pulled. NdiRuntime::is_active() → false
        DeviceStatus::Disconnected { last_known_verse: Romans 8:1 }
        operator_channel_update emitted → operator sees NDI as DISCONNECTED
        AudienceChannel state preserved: still holds Romans 8:1
        Projector webview and OBS continue unaffected

t=45s:  NDI reconnects. NdiRuntime::is_active() → true
        Current AudienceChannel.active_verse = Romans 8:4 (sermon continued)
        Immediately emit audience_channel_update to NDI subscriber
        NDI canvas renders Romans 8:4
        DeviceStatus::Connected updated
        info!("device_health: NDI reconnected, resyncing verse Romans 8:4")
```

### Example D: Operator Channel — Confidence Scores Visible
```
Detection result: Romans 8:1, confidence 0.67 (below operator threshold 0.70)
AudienceChannel: NOT updated (below threshold, not auto-queued)
PastorChannel: NOT updated
OperatorChannel: Updated with detection + confidence score
operator_channel_update → operator console shows "Romans 8:1 (67%)" in detection list
Operator can manually promote it to queue if desired
```

---

## Post-Code Self-Verification Checklist

- [ ] Are the three channel types defined as distinct Rust structs with `Serialize`?
- [ ] Does `AudienceChannel` exclude suggestions, queue state, and confidence scores?
- [ ] Does backend emit to specific windows (`emit_to`) for Pastor and Operator channels?
- [ ] Does backend emit to all windows (`emit`) for Audience channel?
- [ ] Are Zustand stores populated ONLY by Tauri event listeners (not by direct mutation)?
- [ ] Were existing `broadcast-store.ts` fields kept (not deleted) during migration?
- [ ] Is device health checked every 2 seconds (not on every frame)?
- [ ] Does disconnect preserve channel state (not reset it)?
- [ ] Does reconnect immediately resync the current channel state to the device?
- [ ] Are all three `RoutingMode` variants (`Locked`, `Preview`, `Independent`) handled?
- [ ] Does `cargo test --workspace` pass?
- [ ] Did I touch rhema-audio, rhema-stt, or rhema-detection? (answer must be NO)
- [ ] Did I add UI toggle for routing mode in the operator console?
