import { PanelHeader } from "@/components/ui/panel-header"
import { ConfidenceDot } from "@/components/ui/confidence-dot"
import { Button } from "@/components/ui/button"
import { PlayIcon, PlusIcon } from "lucide-react"
import { useDetection, detectionActions } from "@/hooks/use-detection"
import { bibleActions } from "@/hooks/use-bible"
import { useQueueStore, useBibleStore } from "@/stores"
import { commitLiveVerse } from "@/hooks/use-broadcast"
import { acquireOperatorLock } from "@/lib/operator-lock"
import type { DetectionResult } from "@/types"
import type { QueueItem } from "@/types"

const SOURCE_COLORS: Record<
  string,
  { bg: string; text: string; label: string }
> = {
  direct: { bg: "bg-green-500/15", text: "text-green-600", label: "Direct" },
  contextual: { bg: "bg-blue-500/15", text: "text-blue-600", label: "Context" },
  quotation: { bg: "bg-pink-500/15", text: "text-pink-600", label: "Quote" },
  semantic_local: {
    bg: "bg-indigo-500/15",
    text: "text-indigo-300",
    label: "Semantic",
  },
  semantic_cloud: {
    bg: "bg-purple-500/15",
    text: "text-purple-300",
    label: "Cloud",
  },
}

function SourceBadge({ source }: { source: string }) {
  const style = SOURCE_COLORS[source] ?? {
    bg: "bg-muted",
    text: "text-muted-foreground",
    label: source,
  }
  return (
    <span
      className={`rounded px-1.5 py-0.5 text-[0.5625rem] font-medium tracking-wider uppercase ${style.bg} ${style.text}`}
    >
      {style.label}
    </span>
  )
}

function queueSource(source: DetectionResult["source"]): QueueItem["source"] {
  if (source === "direct" || source === "contextual") return "ai-direct"
  if (source === "semantic_cloud") return "ai-cloud"
  return "ai-semantic"
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`
}

function DetectionCard({ detection }: { detection: DetectionResult }) {
  const handlePresent = () => {
    acquireOperatorLock()
    // Select this verse for preview
    bibleActions.selectVerse({
      id: 0,
      translation_id: 1,
      book_number: detection.book_number,
      book_name: detection.book_name,
      book_abbreviation: "",
      chapter: detection.chapter,
      verse: detection.verse,
      text: detection.verse_text,
    })
    // Navigate book search panel to this verse
    if (detection.book_number > 0) {
      bibleActions.navigateToVerse(
        detection.book_number,
        detection.chapter,
        detection.verse
      )
    }
    // Set broadcast live verse
    const translation =
      useBibleStore
        .getState()
        .translations.find(
          (t) => t.id === useBibleStore.getState().activeTranslationId
        )?.abbreviation ?? "KJV"
    commitLiveVerse(
      {
        id: 0,
        translation_id: 1,
        book_number: detection.book_number,
        book_name: detection.book_name,
        book_abbreviation: "",
        chapter: detection.chapter,
        verse: detection.verse,
        text: detection.verse_text,
      },
      translation
    )
  }

  return (
    <div className="border-b border-border p-3 last:border-0">
      <div className="flex items-center gap-2">
        <ConfidenceDot confidence={detection.confidence} />
        <SourceBadge source={detection.source} />
        <span
          className={
            detection.decision === "auto_queued"
              ? "rounded bg-green-500/10 px-1.5 py-0.5 text-[0.5625rem] font-medium tracking-wider text-green-600 uppercase"
              : "rounded bg-amber-500/10 px-1.5 py-0.5 text-[0.5625rem] font-medium tracking-wider text-amber-600 uppercase"
          }
        >
          {detection.decision === "auto_queued" ? "Auto" : "Review"}
        </span>
        <span className="text-sm font-semibold text-foreground">
          {detection.verse_ref}
        </span>
        <span className="ml-auto text-xs font-medium text-muted-foreground tabular-nums">
          {formatPercent(detection.confidence)}
        </span>
      </div>

      {detection.verse_text && (
        <p className="mt-1 line-clamp-2 text-sm leading-relaxed text-muted-foreground">
          {detection.verse_text}
        </p>
      )}

      <div className="mt-2 grid gap-1 text-[0.6875rem] leading-snug text-muted-foreground">
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          <span>Raw {formatPercent(detection.raw_score)}</span>
          <span>Min {formatPercent(detection.minimum_threshold)}</span>
          <span>Auto {formatPercent(detection.auto_queue_threshold)}</span>
        </div>
        <p>{detection.explanation}</p>
      </div>

      <div className="mt-2 flex gap-2">
        <Button size="sm" className="gap-1" onClick={handlePresent}>
          <PlayIcon className="size-3" />
          Present
        </Button>
        <Button
          variant="outline"
          size="sm"
          className="gap-1"
          onClick={() => {
            useQueueStore.getState().addItem({
              id: crypto.randomUUID(),
              verse: {
                id: 0,
                translation_id: 1,
                book_number: detection.book_number,
                book_name: detection.book_name,
                book_abbreviation: "",
                chapter: detection.chapter,
                verse: detection.verse,
                text: detection.verse_text,
              },
              reference: detection.verse_ref,
              confidence: detection.confidence,
              source: queueSource(detection.source),
              added_at: Date.now(),
            })
          }}
        >
          <PlusIcon className="size-3" />
          Queue
        </Button>
      </div>
    </div>
  )
}

export function DetectionsPanel() {
  const { detections } = useDetection()

  return (
    <div
      data-slot="detections-panel"
      className="flex h-full flex-col overflow-hidden rounded-lg border border-border bg-card"
    >
      <PanelHeader title="Recent detections">
        <button
          onClick={() => detectionActions.clearDetections()}
          className="text-[0.625rem] text-muted-foreground transition-colors hover:text-foreground"
        >
          Clear all
        </button>
      </PanelHeader>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="flex flex-col gap-0">
          {detections.length === 0 && (
            <p className="p-4 text-center text-xs text-muted-foreground">
              Verse detections will appear here during transcription
            </p>
          )}
          {detections.map((detection, i) => (
            <DetectionCard
              key={`${detection.verse_ref}-${i}`}
              detection={detection}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
