import { ServerIcon } from "lucide-react"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { useBroadcastStore } from "@/stores"
import { cn } from "@/lib/utils"

/**
 * Operator-facing device-health surface (Bullet 4.4 / Gap F2). The 2s monitor
 * publishes endpoint health into `deviceHealth`; this shows a connected/total
 * count with a per-endpoint breakdown. Renders nothing until health arrives.
 */
export function DeviceHealthIndicator() {
  const health = useBroadcastStore((s) => s.deviceHealth)
  if (!health || health.length === 0) return null

  const connected = health.filter((d) => d.connection === "connected").length
  const anyDown = connected < health.length

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          title="Output device health"
          className="flex items-center gap-1 rounded px-1.5 py-1 text-[0.625rem] text-muted-foreground transition-colors hover:bg-muted"
        >
          <ServerIcon className={cn("size-3.5", anyDown && "text-red-500")} />
          <span className="tabular-nums">
            {connected}/{health.length}
          </span>
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-56 p-2">
        <div className="mb-1.5 text-[0.625rem] font-medium uppercase tracking-wider text-muted-foreground">
          Output devices
        </div>
        <ul className="space-y-1">
          {health.map((d) => {
            const ok = d.connection === "connected"
            return (
              <li
                key={d.label}
                className="flex items-center justify-between text-xs"
              >
                <span className="text-foreground">{d.label}</span>
                <span
                  className={cn(
                    "flex items-center gap-1",
                    ok ? "text-emerald-500" : "text-red-500"
                  )}
                >
                  <span
                    className={cn(
                      "size-1.5 rounded-full",
                      ok ? "bg-emerald-500" : "bg-red-500"
                    )}
                  />
                  {d.connection}
                </span>
              </li>
            )
          })}
        </ul>
      </PopoverContent>
    </Popover>
  )
}
