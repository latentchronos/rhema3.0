import { invoke } from "@tauri-apps/api/core"
import { useBroadcastStore } from "@/stores/broadcast-store"
import { useBibleStore } from "@/stores/bible-store"
import { pushToast } from "@/stores/toast-store"
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

/**
 * Structured voice navigation command emitted by the backend on `voice_command`
 * (Bullet V5). Mirrors `rhema_detection::NavCommand`'s serde-tagged JSON.
 */
export type NavCommand =
  | { kind: "step"; unit: "verse" | "chapter"; direction: "forward" | "backward"; count: number }
  | { kind: "jump_verse"; verse: number }
  | { kind: "jump_chapter"; chapter: number }
  | { kind: "jump_chapter_verse"; chapter: number; verse: number }
  | { kind: "clear" }

/** Result of a `go_to_reference` / `step_verses` command (Bullet V4). */
export type NavCommandResult =
  | { status: "moved"; verse: NavVerse }
  | { status: "no_change" }
  | { status: "chapter_out_of_range"; book_name: string; requested: number; last_chapter: number }
  | {
      status: "verse_out_of_range"
      book_name: string
      chapter: number
      requested: number
      last_verse: number
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
  commitNavVerse(nav)
}

/** The active translation abbreviation, defaulting to KJV. */
function currentTranslation(): string {
  const bible = useBibleStore.getState()
  return (
    bible.translations.find((t) => t.id === bible.activeTranslationId)
      ?.abbreviation ?? "KJV"
  )
}

/**
 * Commit a backend-resolved `NavVerse` live. Keeps selection in lockstep with
 * the cursor/display (Bug A fix): otherwise the Live panel's effect (keyed on
 * selectedVerse) can re-seed the cursor back to the original verse, so
 * navigation appears to stop after one step.
 */
function commitNavVerse(nav: NavVerse): void {
  const verse: Verse = {
    id: 0,
    translation_id: 1,
    book_number: nav.book_number,
    book_name: nav.book_name,
    book_abbreviation: "",
    chapter: nav.chapter,
    verse: nav.verse,
    text: nav.text,
  }
  useBibleStore.getState().selectVerse(verse)
  commitLiveVerse(verse, currentTranslation())
}

/** Surface an out-of-range / moved outcome from a navigation command. */
function handleNavResult(result: NavCommandResult): void {
  switch (result.status) {
    case "moved":
      commitNavVerse(result.verse)
      break
    case "verse_out_of_range":
      pushToast(
        `${result.book_name} ${result.chapter} has ${result.last_verse} verses — verse ${result.requested} doesn't exist.`,
        "warn"
      )
      break
    case "chapter_out_of_range":
      pushToast(
        `${result.book_name} has ${result.last_chapter} chapters — chapter ${result.requested} doesn't exist.`,
        "warn"
      )
      break
    case "no_change":
      break
  }
}

/**
 * Apply a structured voice navigation command (Bullet V6): absolute jumps and
 * relative steps go through the backend cursor (validated against the active
 * translation) and commit live; "clear" blanks the output. The operator lock is
 * acquired first so a voice command beats a stale detection.
 */
export async function applyVoiceNavCommand(cmd: NavCommand): Promise<void> {
  if (cmd.kind === "clear") {
    acquireOperatorLock()
    const s = useBroadcastStore.getState()
    s.setLive(false)
    commitLiveVerse(null, currentTranslation())
    return
  }

  acquireOperatorLock()
  let result: NavCommandResult | null = null
  switch (cmd.kind) {
    case "step":
      result = await invoke<NavCommandResult>("step_verses", {
        unit: cmd.unit,
        direction: cmd.direction,
        count: cmd.count,
      }).catch(() => null)
      break
    case "jump_verse":
      result = await invoke<NavCommandResult>("go_to_reference", {
        chapter: null,
        verse: cmd.verse,
      }).catch(() => null)
      break
    case "jump_chapter":
      result = await invoke<NavCommandResult>("go_to_reference", {
        chapter: cmd.chapter,
        verse: null,
      }).catch(() => null)
      break
    case "jump_chapter_verse":
      result = await invoke<NavCommandResult>("go_to_reference", {
        chapter: cmd.chapter,
        verse: cmd.verse,
      }).catch(() => null)
      break
  }
  if (result) handleNavResult(result)
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
