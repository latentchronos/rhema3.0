import { useBroadcastStore } from "@/stores"
import { useTauriEvent } from "./use-tauri-event"
import { stepLiveVerse } from "./use-broadcast"

/**
 * Listens for backend-emitted voice control commands (Gap 2 live wiring) and
 * drives the same actions as the operator's buttons, so spoken "next verse" /
 * "previous verse" / "clear screen" work. Mount once in a long-lived component.
 */
export function useVoiceCommands() {
  useTauriEvent<string>("voice_command", (action) => {
    if (action === "next") {
      void stepLiveVerse(true)
    } else if (action === "previous") {
      void stepLiveVerse(false)
    } else if (action === "clear") {
      const s = useBroadcastStore.getState()
      s.setLiveVerse(null)
      s.setLive(false)
    }
  })
}
