import { useToastStore } from "@/stores/toast-store"
import { cn } from "@/lib/utils"

/**
 * Renders transient toasts from the toast store (Bullet V6). Mount once near the
 * app root. Bottom-center, click-to-dismiss; nothing rendered when empty.
 */
export function Toaster() {
  const toasts = useToastStore((s) => s.toasts)
  const dismiss = useToastStore((s) => s.dismiss)
  if (toasts.length === 0) return null

  return (
    <div className="pointer-events-none fixed bottom-4 left-1/2 z-50 flex -translate-x-1/2 flex-col items-center gap-2">
      {toasts.map((t) => (
        <button
          key={t.id}
          type="button"
          onClick={() => dismiss(t.id)}
          className={cn(
            "pointer-events-auto max-w-md rounded-md px-3 py-2 text-xs shadow-lg ring-1 backdrop-blur transition-opacity",
            t.tone === "warn"
              ? "bg-amber-500/15 text-amber-200 ring-amber-500/30"
              : "bg-card/90 text-foreground ring-border"
          )}
        >
          {t.message}
        </button>
      ))}
    </div>
  )
}
