import { create } from "zustand"

export type ToastTone = "info" | "warn"

export interface Toast {
  id: number
  message: string
  tone: ToastTone
}

interface ToastState {
  toasts: Toast[]
  push: (message: string, tone?: ToastTone) => void
  dismiss: (id: number) => void
}

let nextId = 1

/**
 * Minimal transient-notification store (Bullet V6). Used to surface
 * non-blocking feedback such as "this chapter has only N verses" when a voice
 * navigation command names a verse/chapter that doesn't exist. Each toast
 * auto-dismisses after a few seconds; clicking one dismisses it early.
 */
export const useToastStore = create<ToastState>((set) => ({
  toasts: [],
  push: (message, tone = "info") => {
    const id = nextId++
    set((s) => ({ toasts: [...s.toasts, { id, message, tone }] }))
    setTimeout(() => {
      set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }))
    }, 4000)
  },
  dismiss: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}))

/** Convenience for non-React callers (hooks, async handlers). */
export const pushToast = (message: string, tone?: ToastTone) =>
  useToastStore.getState().push(message, tone)
