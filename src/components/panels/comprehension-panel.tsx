import { PanelHeader } from "@/components/ui/panel-header"
import { ConfidenceDot } from "@/components/ui/confidence-dot"
import { useComprehensionStore } from "@/stores/comprehension-store"

/** "STORY_TELLING" → "Story Telling" for display. */
function humanizeIntent(state: string): string {
  return state
    .toLowerCase()
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ")
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`
}

/**
 * Read-only view of the comprehension observer's current conclusion (dev
 * timeline; the full Phase F UI comes later). Comprehension never projects —
 * this panel only reflects what the observer is understanding right now.
 */
export function ComprehensionPanel() {
  const current = useComprehensionStore((s) => s.current)

  return (
    <div
      data-slot="comprehension-panel"
      className="flex h-full flex-col overflow-hidden rounded-lg border border-border bg-card"
    >
      <PanelHeader title="Comprehension" />

      <div className="min-h-0 flex-1 overflow-y-auto">
        {current === null ? (
          <p className="p-4 text-center text-xs text-muted-foreground">
            The observer's understanding of the sermon will appear here during a
            session
          </p>
        ) : (
          <div className="p-3">
            <div className="flex items-center gap-2">
              <ConfidenceDot confidence={current.confidence} />
              <span className="text-sm font-semibold text-foreground">
                {humanizeIntent(current.state)}
              </span>
              <span className="ml-auto text-xs font-medium text-muted-foreground tabular-nums">
                {formatPercent(current.confidence)}
              </span>
            </div>

            {current.topic && (
              <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                {current.topic}
              </p>
            )}

            {current.passages.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1">
                {current.passages.map((p, i) => (
                  <span
                    key={`${p}-${i}`}
                    className="rounded bg-indigo-500/15 px-1.5 py-0.5 text-[0.6875rem] font-medium text-indigo-300"
                  >
                    {p}
                  </span>
                ))}
              </div>
            )}

            {current.supporting_activities.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1">
                {current.supporting_activities.map((a, i) => (
                  <span
                    key={`${a}-${i}`}
                    className="rounded bg-muted px-1.5 py-0.5 text-[0.5625rem] font-medium tracking-wider text-muted-foreground uppercase"
                  >
                    {humanizeIntent(a)}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
