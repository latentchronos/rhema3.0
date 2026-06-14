import { invoke } from "@tauri-apps/api/core"

/**
 * Acquire the operator epoch lock (Phase 3.2 invariant: "operator epoch lock
 * beats voice").
 *
 * Call this at the start of any manual operator console action that projects or
 * navigates a verse (present from queue, present a detection, go live). It
 * increments the backend epoch and opens a 500ms lock window, so voice-triggered
 * detections that arrive in that window are discarded by `emit_detections`
 * rather than overriding the operator. The operator's own commit goes through
 * the channel path (`commit_live_verse`), which is not gated by the lock, so it
 * is never self-blocked. Failures are swallowed — the lock is best-effort and
 * must never break an operator action.
 */
export function acquireOperatorLock(): void {
  void invoke("acquire_operator_lock").catch(() => {})
}
