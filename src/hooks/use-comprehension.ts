import {
  useComprehensionStore,
  type StateTransition,
} from "@/stores/comprehension-store"
import { useTauriEvent } from "./use-tauri-event"

/**
 * Subscribes to backend comprehension transitions (§I1's `comprehension_state`
 * event) and caches the latest state. Read-only — comprehension never projects.
 */
export function useComprehension() {
  const setFromTransition = useComprehensionStore((s) => s.setFromTransition)
  useTauriEvent<StateTransition>("comprehension_state", (t) =>
    setFromTransition(t)
  )
}
