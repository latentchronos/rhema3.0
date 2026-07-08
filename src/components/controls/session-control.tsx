import { useEffect, useState } from "react"
import { invoke } from "@tauri-apps/api/core"
import { PlayIcon, SquareIcon } from "lucide-react"
import { cn } from "@/lib/utils"

/**
 * Service-session gate control (§2.4 / Gap 1 / F3). Until the operator starts
 * the service, the backend ignores all detections, voice commands, and
 * suggestions — so sound-check and pre-service chatter can't fire false
 * detections. Transcript still displays regardless.
 */
export function SessionControl() {
  const [active, setActive] = useState(false)

  useEffect(() => {
    void invoke<boolean>("session_status")
      .then(setActive)
      .catch(() => {})
  }, [])

  const toggle = async () => {
    const next = !active
    setActive(next) // optimistic
    try {
      await invoke(next ? "start_session" : "end_session")
    } catch {
      setActive(!next) // revert on failure
    }
  }

  return (
    <button
      type="button"
      onClick={() => void toggle()}
      title={active ? "End service session" : "Start service session"}
      className={cn(
        "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[0.625rem] font-medium uppercase tracking-wider transition-colors",
        active
          ? "bg-emerald-500/15 text-emerald-400"
          : "bg-muted text-muted-foreground hover:bg-muted/80"
      )}
    >
      {active ? (
        <SquareIcon className="size-2.5" />
      ) : (
        <PlayIcon className="size-2.5" />
      )}
      {active ? "Service On" : "Start Service"}
    </button>
  )
}
