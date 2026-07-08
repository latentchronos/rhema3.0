import { beforeEach, describe, expect, it, vi } from "vitest"

const emitToMock = vi.fn()
const invokeMock = vi.fn()

vi.mock("@tauri-apps/api/event", () => ({
  emitTo: emitToMock,
}))

vi.mock("@tauri-apps/api/core", () => ({
  invoke: invokeMock,
}))

describe("broadcast store sync", () => {
  beforeEach(async () => {
    emitToMock.mockReset()
    emitToMock.mockResolvedValue(undefined)
    invokeMock.mockReset()
    invokeMock.mockResolvedValue(undefined)
    vi.resetModules()
  })

  it("syncBroadcastOutput emits current theme and verse to both broadcast windows", async () => {
    const { useBroadcastStore } = await import("./broadcast-store")
    const theme = useBroadcastStore.getState().themes[0]
    useBroadcastStore.setState({
      activeThemeId: theme.id,
      altActiveThemeId: theme.id,
      liveVerse: {
        reference: "John 3:16",
        segments: [{ text: "For God so loved the world", verseNumber: 16 }],
      },
    })

    emitToMock.mockClear()
    useBroadcastStore.getState().syncBroadcastOutput()

    expect(emitToMock).toHaveBeenCalledTimes(2)
    expect(emitToMock).toHaveBeenCalledWith(
      "broadcast",
      "broadcast:verse-update",
      expect.objectContaining({
        theme: expect.objectContaining({ id: theme.id }),
        verse: expect.objectContaining({ reference: "John 3:16" }),
      })
    )
    expect(emitToMock).toHaveBeenCalledWith(
      "broadcast-alt",
      "broadcast:verse-update",
      expect.objectContaining({
        theme: expect.objectContaining({ id: theme.id }),
        verse: expect.objectContaining({ reference: "John 3:16" }),
      })
    )
  })
})

describe("broadcast store — Phase 4 channel model (additive)", () => {
  beforeEach(() => {
    emitToMock.mockReset()
    emitToMock.mockResolvedValue(undefined)
    invokeMock.mockReset()
    invokeMock.mockResolvedValue(undefined)
    vi.resetModules()
  })

  const sampleVerse = {
    book: "Romans",
    chapter: 8,
    verse_start: 1,
    verse_end: null,
    reference: "Romans 8:1",
    text: "There is therefore now no condemnation",
    translation: "KJV",
  }

  it("retains all existing (pre-Phase-4) fields", async () => {
    const { useBroadcastStore } = await import("./broadcast-store")
    const s = useBroadcastStore.getState()
    // Existing fields must still exist and keep their defaults (additive migration).
    expect(s.isLive).toBe(false)
    expect(s.liveVerse).toBeNull()
    expect(Array.isArray(s.themes)).toBe(true)
    expect(typeof s.activeThemeId).toBe("string")
    expect(typeof s.altActiveThemeId).toBe("string")
    // New channel fields default to empty/null.
    expect(s.routingMode).toBe("locked")
    expect(s.audienceChannel).toBeNull()
    expect(s.pastorChannel).toBeNull()
    expect(s.operatorChannel).toBeNull()
    expect(s.deviceHealth).toEqual([])
  })

  it("setAudienceChannel/Pastor populate caches without touching liveVerse", async () => {
    const { useBroadcastStore } = await import("./broadcast-store")
    useBroadcastStore.getState().setAudienceChannel({
      active_verse: sampleVerse,
      theme_id: "classic",
    })
    useBroadcastStore.getState().setPastorChannel({
      current_verse: sampleVerse,
      preview_verse: null,
      translation: "KJV",
      timer_seconds: 60,
      mode: "locked",
    })
    const s = useBroadcastStore.getState()
    expect(s.audienceChannel?.active_verse?.reference).toBe("Romans 8:1")
    expect(s.pastorChannel?.timer_seconds).toBe(60)
    // Existing live-output state is untouched by channel caching.
    expect(s.liveVerse).toBeNull()
    expect(s.isLive).toBe(false)
  })

  it("setOperatorChannel mirrors the backend routing_state", async () => {
    const { useBroadcastStore } = await import("./broadcast-store")
    useBroadcastStore.getState().setOperatorChannel({
      queue: [],
      detections: [{ verse: sampleVerse, confidence: 0.93, source: "direct" }],
      suggestions: [],
      routing_state: "preview",
      device_health: [
        {
          label: "ndi-main",
          kind: "ndi",
          connection: "disconnected",
          last_change_ms: 1700000000000,
          last_known_verse: sampleVerse,
        },
      ],
    })
    const s = useBroadcastStore.getState()
    expect(s.operatorChannel?.detections).toHaveLength(1)
    expect(s.routingMode).toBe("preview")
    // device_health flows out into the dedicated deviceHealth field (Bullet 4.4).
    expect(s.deviceHealth).toHaveLength(1)
    expect(s.deviceHealth[0].connection).toBe("disconnected")
    expect(s.deviceHealth[0].last_known_verse?.reference).toBe("Romans 8:1")
  })

  it("setRoutingMode sets locally and invokes the backend command", async () => {
    const { useBroadcastStore } = await import("./broadcast-store")
    useBroadcastStore.getState().setRoutingMode("independent")
    expect(useBroadcastStore.getState().routingMode).toBe("independent")
    expect(invokeMock).toHaveBeenCalledWith("set_routing_mode", {
      mode: "independent",
    })
  })
})

describe("go-live channel commit (Bullet 4.5)", () => {
  beforeEach(() => {
    emitToMock.mockReset()
    emitToMock.mockResolvedValue(undefined)
    invokeMock.mockReset()
    invokeMock.mockResolvedValue(undefined)
    vi.resetModules()
  })

  const verse = {
    id: 1,
    translation_id: 1,
    book_number: 45,
    book_name: "Romans",
    book_abbreviation: "Rom",
    chapter: 8,
    verse: 1,
    text: "There is therefore now no condemnation",
  }

  it("commitLiveVerse maps Verse -> ChannelVerse and invokes commit_live_verse", async () => {
    const { useBroadcastStore } = await import("./broadcast-store")
    const { commitLiveVerse } = await import("../hooks/use-broadcast")

    commitLiveVerse(verse, "KJV")

    // Legacy render path retained.
    expect(useBroadcastStore.getState().liveVerse?.reference).toContain(
      "Romans 8:1"
    )
    // Channel command fired with the lossless mapping + active theme.
    expect(invokeMock).toHaveBeenCalledWith("commit_live_verse", {
      verse: {
        book: "Romans",
        chapter: 8,
        verse_start: 1,
        verse_end: null,
        reference: "Romans 8:1",
        text: "There is therefore now no condemnation",
        translation: "KJV",
      },
      themeId: expect.any(String),
    })
  })

  it("commitLiveVerse(null) blanks the audience channel", async () => {
    const { useBroadcastStore } = await import("./broadcast-store")
    const { commitLiveVerse } = await import("../hooks/use-broadcast")

    commitLiveVerse(null, "KJV")

    expect(useBroadcastStore.getState().liveVerse).toBeNull()
    expect(invokeMock).toHaveBeenCalledWith("commit_live_verse", {
      verse: null,
      themeId: expect.any(String),
    })
  })
})
