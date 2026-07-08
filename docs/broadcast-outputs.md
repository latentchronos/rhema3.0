# Broadcast Outputs

Rhema supports multiple broadcast paths so churches can use the production setup they already have.

## Output Types

### External Display

External display opens a Tauri broadcast window on a selected monitor. This is useful for projector workflows or local screen capture.

### NDI

NDI uses the NDI SDK through the Rust broadcast crate. The broadcast output window renders frames and pushes RGBA pixels into the NDI runtime.

### OBS Browser Source

Phase 5 adds a local browser-source server for OBS.

When started, Rhema serves:

```text
http://127.0.0.1:4763/?output=main
http://127.0.0.1:4763/?output=alt
```

Add either URL as an OBS Browser Source. The overlay page connects to Rhema with Server-Sent Events and updates whenever the live verse changes.

The browser source is intentionally local-only:

- host: `127.0.0.1`
- default port: `4763`
- transport: HTTP + Server-Sent Events
- payload source: current broadcast theme and verse data

## Current OBS Limitations

- The initial OBS overlay renders a clean transparent lower-third, not the full Canvas 2D theme renderer.
- OBS WebSocket scene/source control is not implemented yet.
- The local HTTP server is intended for same-machine OBS use.

## Future Work

- Add optional OBS WebSocket control.
- Reuse the full browser renderer in the OBS page for exact theme parity.
- Add authentication or random local tokens if the server is ever exposed beyond localhost.
