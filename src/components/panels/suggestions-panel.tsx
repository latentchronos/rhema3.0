import { useState } from "react"
import { invoke } from "@tauri-apps/api/core"
import { LightbulbIcon, PlayIcon, XIcon } from "lucide-react"
import { PanelHeader } from "@/components/ui/panel-header"
import { Button } from "@/components/ui/button"
import { useBroadcastStore, useBibleStore } from "@/stores"
import { commitLiveVerse } from "@/hooks/use-broadcast"
import { acquireOperatorLock } from "@/lib/operator-lock"
import type { ChannelVerse, Verse } from "@/types"

function channelVerseToVerse(cv: ChannelVerse): Verse {
  return {
    id: 0,
    translation_id: 1,
    book_number: 0, // unknown from the channel payload; name-based render only
    book_name: cv.book,
    book_abbreviation: "",
    chapter: cv.chapter,
    verse: cv.verse_start,
    text: cv.text,
  }
}

/**
 * Operator-only proactive suggestion surface (Phase 5, Bullet 5.3). Reads
 * suggestions from the Operator channel cache and lets the operator present or
 * dismiss them — never auto-projects. Renders nothing when there are no
 * suggestions, so it stays out of the way during normal operation.
 */
export function SuggestionsPanel() {
  const suggestions = useBroadcastStore((s) => s.operatorChannel?.suggestions)
  const [dismissed, setDismissed] = useState<Set<string>>(new Set())

  const visible = (suggestions ?? []).filter((s) => !dismissed.has(s.verse.reference))
  if (visible.length === 0) return null

  const activeTranslation = () => {
    const bible = useBibleStore.getState()
    return (
      bible.translations.find((t) => t.id === bible.activeTranslationId)
        ?.abbreviation ?? "KJV"
    )
  }

  const present = (verse: ChannelVerse) => {
    acquireOperatorLock()
    commitLiveVerse(
      channelVerseToVerse(verse),
      verse.translation || activeTranslation()
    )
  }

  const dismiss = (verse: ChannelVerse) => {
    setDismissed((prev) => new Set(prev).add(verse.reference))
    void invoke("dismiss_suggestion", {
      book: verse.book,
      chapter: verse.chapter,
      verse: verse.verse_start,
    }).catch(() => {})
  }

  return (
    <div
      data-slot="suggestions-panel"
      className="flex shrink-0 flex-col overflow-hidden rounded-lg border border-border bg-card"
    >
      <PanelHeader title="Suggestions" icon={<LightbulbIcon className="size-3.5" />} />
      <div className="max-h-40 overflow-y-auto">
        {visible.map((s) => (
          <div
            key={s.verse.reference}
            className="border-b border-border p-3 last:border-0"
          >
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-foreground">
                {s.verse.reference}
              </span>
              <span className="ml-auto text-xs font-medium text-muted-foreground tabular-nums">
                {Math.round(s.score * 100)}%
              </span>
            </div>
            <p className="mt-1 text-[0.6875rem] leading-snug text-muted-foreground">
              {s.reason}
            </p>
            {s.verse.text && (
              <p className="mt-1 line-clamp-2 text-sm leading-relaxed text-muted-foreground">
                {s.verse.text}
              </p>
            )}
            <div className="mt-2 flex gap-2">
              <Button size="sm" className="gap-1" onClick={() => present(s.verse)}>
                <PlayIcon className="size-3" />
                Present
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="gap-1"
                onClick={() => dismiss(s.verse)}
              >
                <XIcon className="size-3" />
                Dismiss
              </Button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
