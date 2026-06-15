import { useTauriEvent } from "./use-tauri-event"
import { applyVoiceNavCommand, type NavCommand } from "./use-broadcast"

/**
 * Listens for backend-emitted structured voice commands (Bullet V5/V6) and
 * drives the navigation cursor: spoken "next verse", "back three verses",
 * "go to verse 7", "chapter 3 verse 16", "clear screen", etc. Absolute jumps and
 * relative steps are validated against the active translation backend-side; an
 * out-of-range target surfaces a toast instead of moving. Mount once.
 */
export function useVoiceCommands() {
  useTauriEvent<NavCommand>("voice_command", (cmd) => {
    void applyVoiceNavCommand(cmd)
  })
}
