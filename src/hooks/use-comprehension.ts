import { useEffect } from "react"
import { invoke } from "@tauri-apps/api/core"
import {
  useComprehensionStore,
  type StateTransition,
} from "@/stores/comprehension-store"
import { useSettingsStore } from "@/stores"
import { useTauriEvent } from "./use-tauri-event"

/**
 * Subscribes to backend comprehension transitions (§I1's `comprehension_state`
 * event) and caches the latest state. Read-only — comprehension never projects.
 * Also pushes the persisted enable/interval to the backend once at startup so a
 * non-default choice from a previous session takes effect (the model choice is
 * persisted backend-side and applied at load; see `set_comprehension_model`).
 */
export function useComprehension() {
  const setFromTransition = useComprehensionStore((s) => s.setFromTransition)
  useTauriEvent<StateTransition>("comprehension_state", (t) =>
    setFromTransition(t)
  )

  useEffect(() => {
    const { comprehensionEnabled, comprehensionIntervalMs } =
      useSettingsStore.getState()
    void invoke("set_comprehension_config", {
      enabled: comprehensionEnabled,
      intervalMs: comprehensionIntervalMs,
    }).catch(() => {})
  }, [])
}
