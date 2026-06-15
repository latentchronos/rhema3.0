import { useRef, useEffect } from "react"
import { PanelHeader } from "@/components/ui/panel-header"
import { LevelMeter } from "@/components/ui/level-meter"
import { Button } from "@/components/ui/button"
import { MicIcon, MicOffIcon, DownloadIcon } from "lucide-react"
import { invoke } from "@tauri-apps/api/core"
import {
  useTranscriptStore,
  useAudioStore,
  useSettingsStore,
  useDetectionStore,
  useQueueStore,
  useBibleStore,
} from "@/stores"
import { useTauriEvent } from "@/hooks/use-tauri-event"
import { bibleActions } from "@/hooks/use-bible"
import type { TranscriptSegment } from "@/types"
import type { DetectionResult } from "@/types"

export function TranscriptPanel() {
  const segments = useTranscriptStore((s) => s.segments)
  const currentPartial = useTranscriptStore((s) => s.currentPartial)
  const isTranscribing = useTranscriptStore((s) => s.isTranscribing)
  const connectionStatus = useTranscriptStore((s) => s.connectionStatus)
  const audioLevel = useAudioStore((s) => s.level)
  const deepgramApiKey = useSettingsStore((s) => s.deepgramApiKey)
  const scrollRef = useRef<HTMLDivElement>(null)

  // Listen for Tauri events
  useTauriEvent<{ rms: number; peak: number }>("audio_level", (payload) => {
    useAudioStore.getState().setLevel(payload)
  })

  // Connection status events
  useTauriEvent("stt_connected", () => {
    const t = useTranscriptStore.getState()
    t.setConnectionStatus("connected")
    t.setDegradedMode(false) // recovered from any REST fallback
  })
  useTauriEvent("stt_disconnected", () => {
    useTranscriptStore.getState().setConnectionStatus("disconnected")
  })
  useTauriEvent<string>("stt_error", (msg) => {
    const t = useTranscriptStore.getState()
    t.setConnectionStatus("error")
    // Latch a persistent "degraded" indicator when we drop to REST/Hybrid mode.
    if (/hybrid|rest/i.test(msg ?? "")) {
      t.setDegradedMode(true)
    }
  })

  useTauriEvent<{ text: string; is_final: boolean; confidence: number }>(
    "transcript_partial",
    (payload) => {
      useTranscriptStore.getState().setPartial(payload.text)
    }
  )

  useTauriEvent<{ text: string; is_final: boolean; confidence: number }>(
    "transcript_final",
    (payload) => {
      const segment: TranscriptSegment = {
        id: crypto.randomUUID(),
        text: payload.text,
        is_final: true,
        confidence: payload.confidence,
        words: [],
        timestamp: Date.now(),
      }
      useTranscriptStore.getState().addSegment(segment)
    }
  )

  // Listen for voice translation commands: "read in NIV", "switch to ESV"
  useTauriEvent<{ abbreviation: string; translation_id: number }>(
    "translation_command",
    (data) => {
      useBibleStore.getState().setActiveTranslation(data.translation_id)
      console.log(`[VOICE] Translation switched to ${data.abbreviation}`)
    }
  )

  // Listen for detection results from the backend (batch replaces previous detections)
  useTauriEvent<DetectionResult[]>("verse_detections", (detections) => {
    useDetectionStore.getState().addDetections(detections)

    // Auto-navigate book search + select verse for preview/live
    // Handle direct, contextual (reading mode), and high-confidence quotation matches
    const directHit = detections.find(
      (d) =>
        d.source === "direct" ||
        d.source === "contextual" ||
        (d.source === "quotation" && d.auto_queued)
    )
    if (directHit && directHit.book_number > 0) {
      // Select verse immediately so preview/live panels update
      bibleActions.selectVerse({
        id: 0,
        translation_id: useBibleStore.getState().activeTranslationId,
        book_number: directHit.book_number,
        book_name: directHit.book_name,
        book_abbreviation: "",
        chapter: directHit.chapter,
        verse: directHit.verse,
        text: directHit.verse_text,
      })
      // Navigate book search panel to this verse
      useBibleStore.getState().setPendingNavigation({
        bookNumber: directHit.book_number,
        chapter: directHit.chapter,
        verse: directHit.verse,
      })
    }

    // Auto-queue high-confidence detections
    for (const d of detections) {
      if (d.auto_queued) {
        useQueueStore.getState().addItem({
          id: crypto.randomUUID(),
          verse: {
            id: 0,
            translation_id: 1,
            book_number: d.book_number,
            book_name: d.book_name,
            book_abbreviation: "",
            chapter: d.chapter,
            verse: d.verse,
            text: d.verse_text,
          },
          reference: d.verse_ref,
          confidence: d.confidence,
          source:
            d.source === "direct" || d.source === "contextual"
              ? "ai-direct"
              : d.source === "semantic_cloud"
                ? "ai-cloud"
                : "ai-semantic",
          added_at: Date.now(),
        })
      }
    }
  })

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [segments, currentPartial])

  const handleStart = async () => {
    try {
      useTranscriptStore.getState().setConnectionStatus("connecting")
      const { useSettingsStore } = await import("@/stores")
      const settings = useSettingsStore.getState()
      console.log(
        "[AUDIO] Starting with deviceId:",
        settings.audioDeviceId,
        "gain:",
        settings.gain,
        "channel:",
        settings.audioChannelIndex,
        "vad:",
        settings.vadEnabled
      )
      await invoke("start_transcription", {
        apiKey: deepgramApiKey ?? "",
        deviceId: settings.audioDeviceId,
        gain: settings.gain,
        channelIndex: settings.audioChannelIndex,
        vadEnabled: settings.vadEnabled,
      })
      useTranscriptStore.getState().setTranscribing(true)
    } catch (e) {
      console.error("Failed to start transcription:", e)
      useTranscriptStore.getState().setConnectionStatus("error")
      alert(String(e))
    }
  }

  const handleStop = async () => {
    try {
      await invoke("stop_transcription")
      useTranscriptStore.getState().setTranscribing(false)
      useTranscriptStore.getState().setPartial("")
      useTranscriptStore.getState().setConnectionStatus("disconnected")
    } catch (e) {
      console.error("Failed to stop transcription:", e)
    }
  }

  const exportTranscript = () => {
    const segs = useTranscriptStore.getState().segments
    if (segs.length === 0) return
    const stamp = new Date()
    const header = `Rhema transcript — exported ${stamp.toLocaleString()}\n${"=".repeat(48)}\n\n`
    const body = segs.map((s) => s.text).join("\n")
    const blob = new Blob([header + body + "\n"], {
      type: "text/plain;charset=utf-8",
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `rhema-transcript-${stamp.toISOString().slice(0, 19).replace(/[:T]/g, "-")}.txt`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  return (
    <div
      data-slot="transcript-panel"
      className="flex flex-col overflow-hidden rounded-lg border border-border bg-card"
    >
      <PanelHeader
        title="Live transcript"
        icon={<MicIcon className="size-3" />}
      >
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon-sm"
            title="Export transcript"
            aria-label="Export transcript"
            disabled={segments.length === 0}
            onClick={exportTranscript}
          >
            <DownloadIcon className="size-3.5" />
          </Button>
          {isTranscribing && (
            <span
              className={`size-2 rounded-full ${
                connectionStatus === "connected"
                  ? "bg-emerald-500"
                  : connectionStatus === "connecting"
                    ? "animate-pulse bg-amber-500"
                    : connectionStatus === "error"
                      ? "bg-red-500"
                      : "bg-muted-foreground/40"
              }`}
              title={connectionStatus}
            />
          )}
          <LevelMeter level={audioLevel.rms} bars={5} />
        </div>
      </PanelHeader>

      <div ref={scrollRef} className="relative min-h-0 flex-1 overflow-y-auto">
        <div className="flex flex-col gap-2 p-3">
          {/* Faded top gradient */}
          <div className="pointer-events-none absolute inset-x-0 top-0 z-10 h-6 bg-linear-to-b from-card to-transparent" />

          {segments.length === 0 && !currentPartial && !isTranscribing && (
            <p className="text-sm text-muted-foreground">
              Click "Start transcribing" to begin
            </p>
          )}

          {/* Final segments — recent ones brighter, older ones fade */}
          {segments.map((seg, idx) => {
            const distFromEnd = segments.length - 1 - idx
            const opacity =
              distFromEnd === 0
                ? "text-foreground/80"
                : distFromEnd === 1
                  ? "text-foreground/60"
                  : distFromEnd <= 3
                    ? "text-foreground/40"
                    : "text-foreground/25"
            return (
              <p
                key={seg.id}
                className={`text-sm leading-relaxed transition-colors duration-300 ${opacity}`}
              >
                {seg.text}
              </p>
            )
          })}

          {/* Partial (in-progress) text — larger and brighter than final segments */}
          {currentPartial && (
            <p className="border-l-2 border-primary pl-2 text-base leading-relaxed text-foreground">
              {currentPartial}
              <span className="ml-1 inline-block size-1.5 animate-pulse rounded-full bg-primary align-middle" />
            </p>
          )}
        </div>
      </div>

      {/* Bottom control */}
      <div className="flex gap-2 border-t border-border px-3 py-2">
        {isTranscribing ? (
          <Button
            variant="ghost"
            size="sm"
            className="text-destructive hover:text-destructive"
            onClick={handleStop}
          >
            <MicOffIcon className="size-3" />
            Stop transcribing
          </Button>
        ) : (
          <Button variant="ghost" size="sm" onClick={handleStart}>
            <MicIcon className="size-3" />
            Start transcribing
          </Button>
        )}
      </div>
    </div>
  )
}
