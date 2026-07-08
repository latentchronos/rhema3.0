import { create } from "zustand"
import type { TranscriptSegment } from "@/types"

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error"

/**
 * How many finalized segments the store keeps for rendering. The transcript panel maps
 * over this array (and its auto-scroll reads scrollHeight) on every update, so an unbounded
 * list makes render + reflow O(n²) over a long service — the "smooth at first, lags like
 * mad after a while" bug. Older segments scroll off-screen anyway; their text is preserved
 * in `fullLog` for export, so nothing is lost. (Mirrors detection-store's slice(0, 50) and
 * WhisperLive's send_last_n_segments windowing.)
 */
export const MAX_RENDERED_SEGMENTS = 300

interface TranscriptState {
  /** The rendered window — capped at {@link MAX_RENDERED_SEGMENTS}. */
  segments: TranscriptSegment[]
  /**
   * Every finalized segment's text, append-only, in order. NEVER rendered — it exists only
   * so export retains the full service even though the visible list is windowed. Kept as a
   * plain string[] (mutated in place, no React subscribers) so appending stays O(1) and
   * never triggers a re-render.
   */
  fullLog: string[]
  currentPartial: string
  isTranscribing: boolean
  connectionStatus: ConnectionStatus
  /** Latched when the WS drops to slower REST mode; cleared on reconnect (Gap 5). */
  degradedMode: boolean

  addSegment: (segment: TranscriptSegment) => void
  setPartial: (text: string) => void
  setTranscribing: (transcribing: boolean) => void
  setConnectionStatus: (status: ConnectionStatus) => void
  setDegradedMode: (degraded: boolean) => void
  clearTranscript: () => void
}

export const useTranscriptStore = create<TranscriptState>((set) => ({
  segments: [],
  fullLog: [],
  currentPartial: "",
  isTranscribing: false,
  connectionStatus: "disconnected",
  degradedMode: false,

  addSegment: (segment) =>
    set((state) => {
      // Append to the never-rendered export log in place: no subscribers read `fullLog`,
      // so mutating it (rather than spreading a new array each final) keeps this O(1) and
      // avoids a second unbounded O(n) copy per segment.
      state.fullLog.push(segment.text)
      const grown = [...state.segments, segment]
      return {
        segments:
          grown.length > MAX_RENDERED_SEGMENTS
            ? grown.slice(-MAX_RENDERED_SEGMENTS)
            : grown,
        fullLog: state.fullLog,
        currentPartial: "",
      }
    }),
  setPartial: (currentPartial) => set({ currentPartial }),
  setTranscribing: (isTranscribing) => set({ isTranscribing }),
  setConnectionStatus: (connectionStatus) => set({ connectionStatus }),
  setDegradedMode: (degradedMode) => set({ degradedMode }),
  clearTranscript: () => set({ segments: [], fullLog: [], currentPartial: "" }),
}))
