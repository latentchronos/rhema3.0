import { invoke } from "@tauri-apps/api/core"
import { useBroadcastStore } from "@/stores/broadcast-store"
import { useBibleStore } from "@/stores/bible-store"
import { acquireOperatorLock } from "@/lib/operator-lock"
import type { ChannelVerse, VerseRenderData } from "@/types"
import type { Verse } from "@/types"

interface NavVerse {
  book_number: number
  book_name: string
  chapter: number
  verse: number
  text: string
  reference: string
}

export function toVerseRenderData(verse: Verse, translation: string): VerseRenderData {
  return {
    reference: `${verse.book_name} ${verse.chapter}:${verse.verse} (${translation})`,
    segments: [{ verseNumber: verse.verse, text: verse.text }],
  }
}

/** Map a Bible `Verse` into the backend channel payload (Bullet 4.5). */
export function toChannelVerse(verse: Verse, translation: string): ChannelVerse {
  return {
    book: verse.book_name,
    chapter: verse.chapter,
    verse_start: verse.verse,
    verse_end: null,
    reference: `${verse.book_name} ${verse.chapter}:${verse.verse}`,
    text: verse.text,
    translation,
  }
}

/**
 * Commit a verse live (Bullet 4.5, Option A — additive).
 *
 * Retains the existing projector path (`setLiveVerse` → emitTo/OBS) AND tells the
 * backend channel layer so `ChannelState.audience`/`pastor` become authoritative
 * (operator console state + device-health reconnect resync depend on this).
 * Passing `null` blanks the audience channel. Failures are swallowed so a channel
 * hiccup never disturbs the proven render path.
 */
export function commitLiveVerse(verse: Verse | null, translation: string) {
  const store = useBroadcastStore.getState()
  // Legacy render path — unchanged.
  store.setLiveVerse(verse ? toVerseRenderData(verse, translation) : null)
  // Channel layer — authoritative backend state.
  void invoke("commit_live_verse", {
    verse: verse ? toChannelVerse(verse, translation) : null,
    themeId: store.activeThemeId,
  }).catch(() => {})
  // Seed the formal navigation cursor so next/previous-verse work from here
  // (Phase 3 follow-up). No-op backend-side when already on this verse.
  if (verse) {
    void invoke("set_cursor_position", {
      bookNumber: verse.book_number,
      chapter: verse.chapter,
      verse: verse.verse,
      translation,
    }).catch(() => {})
  }
}

export function deriveLiveVerse({
  isLive,
  selectedVerse,
  translation,
}: {
  isLive: boolean
  selectedVerse: Verse | null
  translation: string
}): VerseRenderData | null {
  if (!isLive || !selectedVerse) return null
  return toVerseRenderData(selectedVerse, translation)
}

/**
 * Step the live verse forward/backward through the formal navigation cursor
 * (Phase 3 follow-up). Acquires the operator lock (operator beats voice), asks
 * the backend to advance the cursor, and commits the resolved verse live. No-op
 * at a Bible boundary or when nothing is live (cold cursor → backend returns
 * null).
 */
export async function stepLiveVerse(forward: boolean): Promise<void> {
  acquireOperatorLock()
  const cmd = forward ? "next_verse" : "previous_verse"
  const nav = await invoke<NavVerse | null>(cmd).catch(() => null)
  if (!nav) return
  const bible = useBibleStore.getState()
  const translation =
    bible.translations.find((t) => t.id === bible.activeTranslationId)
      ?.abbreviation ?? "KJV"
  commitLiveVerse(
    {
      id: 0,
      translation_id: 1,
      book_number: nav.book_number,
      book_name: nav.book_name,
      book_abbreviation: "",
      chapter: nav.chapter,
      verse: nav.verse,
      text: nav.text,
    },
    translation
  )
}

export const broadcastActions = {
  setLiveVerse: (verse: VerseRenderData | null) =>
    useBroadcastStore.getState().setLiveVerse(verse),
  setLive: (live: boolean) =>
    useBroadcastStore.getState().setLive(live),
  getActiveTheme: () => {
    const s = useBroadcastStore.getState()
    return s.themes.find((t) => t.id === s.activeThemeId) ?? s.themes[0]
  },
}
