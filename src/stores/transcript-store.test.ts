import { beforeEach, describe, expect, it } from "vitest"
import type { TranscriptSegment } from "@/types"
import { MAX_RENDERED_SEGMENTS, useTranscriptStore } from "./transcript-store"

function seg(text: string): TranscriptSegment {
  return {
    id: `${text}-${Math.random()}`,
    text,
    is_final: true,
    confidence: 1,
    words: [],
    timestamp: 0,
  }
}

describe("transcript store windowing", () => {
  beforeEach(() => {
    useTranscriptStore.getState().clearTranscript()
  })

  it("caps the rendered segments at MAX_RENDERED_SEGMENTS", () => {
    const { addSegment } = useTranscriptStore.getState()
    for (let i = 0; i < MAX_RENDERED_SEGMENTS + 50; i++) addSegment(seg(`s${i}`))

    const { segments } = useTranscriptStore.getState()
    expect(segments).toHaveLength(MAX_RENDERED_SEGMENTS)
    // It keeps the MOST RECENT ones (window slid off the front).
    expect(segments[segments.length - 1].text).toBe(`s${MAX_RENDERED_SEGMENTS + 49}`)
    expect(segments[0].text).toBe("s50")
  })

  it("preserves the full transcript in fullLog for export even past the window", () => {
    const { addSegment } = useTranscriptStore.getState()
    const n = MAX_RENDERED_SEGMENTS + 50
    for (let i = 0; i < n; i++) addSegment(seg(`s${i}`))

    const { fullLog } = useTranscriptStore.getState()
    expect(fullLog).toHaveLength(n) // nothing dropped from the export log
    expect(fullLog[0]).toBe("s0") // the oldest text is still there
    expect(fullLog[n - 1]).toBe(`s${n - 1}`)
  })

  it("clearTranscript empties both the window and the export log", () => {
    const { addSegment } = useTranscriptStore.getState()
    addSegment(seg("a"))
    addSegment(seg("b"))
    useTranscriptStore.getState().clearTranscript()

    const s = useTranscriptStore.getState()
    expect(s.segments).toHaveLength(0)
    expect(s.fullLog).toHaveLength(0)
    expect(s.currentPartial).toBe("")
  })
})
