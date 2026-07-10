import { create } from "zustand"

/**
 * The observer's current discourse conclusion, as emitted by the Rust backend.
 * Field names match `rhema_comprehension::ComprehensionState`'s serde shape (§8):
 * dominant intent arrives under the key `state` (SCREAMING_SNAKE_CASE), and
 * `topic` is OMITTED (not null) when absent — hence optional here.
 */
export interface ComprehensionState {
  /** Dominant intent, SCREAMING_SNAKE_CASE (e.g. "STORY_TELLING"). */
  state: string
  supporting_activities: string[]
  /** Open-ended passage list; may be empty (§9). Bare "Book Chapter:Verse" strings. */
  passages: string[]
  topic?: string | null
  confidence: number
}

/** A recorded transition between states (§7) — emitted only on a genuine change. */
export interface StateTransition {
  from: ComprehensionState | null
  to: ComprehensionState
  at_ms: number
}

interface Store {
  /** The latest comprehension state, or null before the first transition / after clear. */
  current: ComprehensionState | null
  setFromTransition: (t: StateTransition) => void
  clear: () => void
}

export const useComprehensionStore = create<Store>((set) => ({
  current: null,
  setFromTransition: (t) => set({ current: t.to }),
  clear: () => set({ current: null }),
}))
