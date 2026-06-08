# Contributor Architecture Notes

Start with `docs/architecture.md` for the system map. This file is the shorter change guide for contributors and AI agents.

## Runtime Boundaries

- Frontend UI lives in `src/`.
- Tauri commands live in `src-tauri/src/commands/`.
- Long-running app state lives in `src-tauri/src/state.rs`.
- Domain crates live in `src-tauri/crates/`.
- Local agent continuation state lives in `.agent-progress/` and must not be committed.

## Change Rules

- Keep audio capture, STT, detection, queueing, and broadcast output as separate concerns.
- Add tests in the crate where behavior lives.
- Keep network integrations behind narrow adapter functions or commands.
- Update public docs when a workflow or architecture boundary changes.
- Update `.agent-progress/progress.md` only after completed code or repo-file changes.

## Production Areas

Audio:

- Preserve default behavior for simple mono/stereo users.
- Treat multi-channel routing and VAD as opt-in controls.

Detection:

- Keep source-specific confidence policy in the backend.
- Expose explanations to the UI instead of duplicating scoring rules in React.

Broadcast:

- Keep display, OBS, NDI, and future ProPresenter outputs payload-compatible.
- Avoid adding output-specific logic to detection.

Security:

- Keep Tauri CSP explicit.
- Scope capabilities to the windows that need them.
- Do not log API keys, presentation tokens, or local secrets.
