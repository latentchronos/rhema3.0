# Security

Rhema is a Tauri desktop app, so the browser surface and native command surface must both stay intentionally small.

## Tauri Content Security Policy

`src-tauri/tauri.conf.json` now uses an explicit CSP instead of `null`.

The policy keeps scripts self-hosted, allows inline styles for the current UI stack, allows local asset/data/blob images, and permits local development/runtime connections used by Tauri IPC, Vite, OBS browser output, and localhost services.

Review the CSP whenever adding:

- external web content;
- remote image or font origins;
- WebSocket integrations;
- embedded browser views;
- third-party analytics or telemetry.

## Capabilities

Capability files live in `src-tauri/capabilities/`.

Current review notes:

- `default.json` grants core and store permissions to the app windows.
- `desktop.json` grants global shortcut permissions.
- Broadcast windows should remain display-only unless a feature explicitly requires commands from those windows.

Before adding a Tauri plugin or command, document why it is needed and keep the permission scoped to the smallest window set that can use it.

## Contributor Rules

- Do not set `security.csp` back to `null`.
- Do not load arbitrary remote pages inside app windows.
- Keep secrets in local settings or OS-backed storage, not committed files.
- Prefer local-only development state under `.agent-progress/` for agent handoff notes.
- Treat STT and presentation integrations as network boundaries: validate inputs, keep tokens out of logs, and document default ports.
