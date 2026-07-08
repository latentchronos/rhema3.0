import { create } from "zustand"

// Tiny localStorage helpers (guarded so the store also works under node/vitest). The STT
// model choice persists across restarts like the theme + audio device do.
const readLS = (k: string): string | null =>
  typeof localStorage !== "undefined" ? localStorage.getItem(k) : null
const writeLS = (k: string, v: string | null) => {
  if (typeof localStorage === "undefined") return
  if (v === null) localStorage.removeItem(k)
  else localStorage.setItem(k, v)
}

interface SettingsState {
  deepgramApiKey: string | null
  openaiApiKey: string | null
  claudeApiKey: string | null
  // Stage-2 multi-provider LLM config (Track L). `llmProvider` is the UI
  // selection: "auto" | "anthropic" | "openai" | "gemini" | "custom".
  llmProvider: string | null
  llmApiKey: string | null
  llmBaseUrl: string | null
  llmModel: string | null
  activeTranslationId: number
  audioDeviceId: string | null
  audioChannelIndex: number | null
  gain: number
  /** Selected on-device STT model (absolute .gguf path); null = use backend default. */
  sttModel: string | null
  /** Whether the selected model is a streaming model; null = defer to the backend. */
  sttStreaming: boolean | null
  vadEnabled: boolean
  commandWakeWord: string | null
  autoMode: boolean
  confidenceThreshold: number
  cooldownMs: number
  onboardingComplete: boolean

  setDeepgramApiKey: (key: string | null) => void
  setOpenaiApiKey: (key: string | null) => void
  setClaudeApiKey: (key: string | null) => void
  setLlmProvider: (provider: string | null) => void
  setLlmApiKey: (key: string | null) => void
  setLlmBaseUrl: (url: string | null) => void
  setLlmModel: (model: string | null) => void
  setActiveTranslationId: (id: number) => void
  setAudioDeviceId: (id: string | null) => void
  setAudioChannelIndex: (id: number | null) => void
  setGain: (gain: number) => void
  /** Persisted; pass both so the picker sets model + streaming mode together. */
  setSttModel: (model: string | null, streaming: boolean | null) => void
  setVadEnabled: (enabled: boolean) => void
  setCommandWakeWord: (word: string | null) => void
  setAutoMode: (auto: boolean) => void
  setConfidenceThreshold: (threshold: number) => void
  setCooldownMs: (ms: number) => void
  setOnboardingComplete: (complete: boolean) => void
}

export const useSettingsStore = create<SettingsState>((set) => ({
  deepgramApiKey: null,
  openaiApiKey: null,
  claudeApiKey: null,
  llmProvider: null,
  llmApiKey: null,
  llmBaseUrl: null,
  llmModel: null,
  activeTranslationId: 1,
  audioDeviceId: null,
  audioChannelIndex: null,
  gain: 1.0,
  sttModel: readLS("rhema.sttModel"),
  sttStreaming: (() => {
    const v = readLS("rhema.sttStreaming")
    return v === null ? null : v === "true"
  })(),
  vadEnabled: false,
  commandWakeWord: null,
  autoMode: false,
  confidenceThreshold: 0.8,
  cooldownMs: 2500,
  onboardingComplete: false,

  setDeepgramApiKey: (deepgramApiKey) => set({ deepgramApiKey }),
  setOpenaiApiKey: (openaiApiKey) => set({ openaiApiKey }),
  setClaudeApiKey: (claudeApiKey) => set({ claudeApiKey }),
  setLlmProvider: (llmProvider) => set({ llmProvider }),
  setLlmApiKey: (llmApiKey) => set({ llmApiKey }),
  setLlmBaseUrl: (llmBaseUrl) => set({ llmBaseUrl }),
  setLlmModel: (llmModel) => set({ llmModel }),
  setActiveTranslationId: (activeTranslationId) => set({ activeTranslationId }),
  setAudioDeviceId: (audioDeviceId) => set({ audioDeviceId }),
  setAudioChannelIndex: (audioChannelIndex) => set({ audioChannelIndex }),
  setGain: (gain) => set({ gain }),
  setSttModel: (sttModel, sttStreaming) => {
    writeLS("rhema.sttModel", sttModel)
    writeLS("rhema.sttStreaming", sttStreaming === null ? null : String(sttStreaming))
    set({ sttModel, sttStreaming })
  },
  setVadEnabled: (vadEnabled) => set({ vadEnabled }),
  setCommandWakeWord: (commandWakeWord) => set({ commandWakeWord }),
  setAutoMode: (autoMode) => set({ autoMode }),
  setConfidenceThreshold: (confidenceThreshold) => set({ confidenceThreshold }),
  setCooldownMs: (cooldownMs) => set({ cooldownMs }),
  setOnboardingComplete: (onboardingComplete) => set({ onboardingComplete }),
}))
