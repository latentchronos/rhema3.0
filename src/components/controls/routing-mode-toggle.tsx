import { useBroadcastStore } from "@/stores"
import { cn } from "@/lib/utils"
import type { RoutingMode } from "@/types"

const MODES: { value: RoutingMode; label: string; title: string }[] = [
  { value: "locked", label: "Locked", title: "Audience and Pastor show the same verse" },
  { value: "preview", label: "Preview", title: "Pastor sees the next verse before the audience" },
  {
    value: "independent",
    label: "Indep.",
    title: "Audience and Pastor navigate separately",
  },
]

/**
 * Operator-console routing-mode selector (Bullet 4.4). Sets the channel
 * `RoutingMode`; the store optimistically updates and invokes the backend
 * `set_routing_mode` command, which re-publishes the pastor/operator channels.
 */
export function RoutingModeToggle() {
  const routingMode = useBroadcastStore((s) => s.routingMode)
  const setRoutingMode = useBroadcastStore((s) => s.setRoutingMode)

  return (
    <div
      role="group"
      aria-label="Routing mode"
      className="inline-flex overflow-hidden rounded-md border border-border"
    >
      {MODES.map((m) => {
        const active = routingMode === m.value
        return (
          <button
            key={m.value}
            type="button"
            title={m.title}
            aria-pressed={active}
            onClick={() => setRoutingMode(m.value)}
            className={cn(
              "px-2 py-1 text-[0.625rem] font-medium uppercase tracking-wide transition-colors",
              active
                ? "bg-primary text-primary-foreground"
                : "bg-transparent text-muted-foreground hover:bg-muted"
            )}
          >
            {m.label}
          </button>
        )
      })}
    </div>
  )
}
