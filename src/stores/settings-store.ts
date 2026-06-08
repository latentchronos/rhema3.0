import { create } from "zustand"

interface SettingsState {
  deepgramApiKey: string | null
  openaiApiKey: string | null
  claudeApiKey: string | null
  activeTranslationId: number
  audioDeviceId: string | null
  audioChannelIndex: number | null
  gain: number
  vadEnabled: boolean
  autoMode: boolean
  confidenceThreshold: number
  cooldownMs: number
  onboardingComplete: boolean

  setDeepgramApiKey: (key: string | null) => void
  setOpenaiApiKey: (key: string | null) => void
  setClaudeApiKey: (key: string | null) => void
  setActiveTranslationId: (id: number) => void
  setAudioDeviceId: (id: string | null) => void
  setAudioChannelIndex: (id: number | null) => void
  setGain: (gain: number) => void
  setVadEnabled: (enabled: boolean) => void
  setAutoMode: (auto: boolean) => void
  setConfidenceThreshold: (threshold: number) => void
  setCooldownMs: (ms: number) => void
  setOnboardingComplete: (complete: boolean) => void
}

export const useSettingsStore = create<SettingsState>((set) => ({
  deepgramApiKey: null,
  openaiApiKey: null,
  claudeApiKey: null,
  activeTranslationId: 1,
  audioDeviceId: null,
  audioChannelIndex: null,
  gain: 1.0,
  vadEnabled: false,
  autoMode: false,
  confidenceThreshold: 0.8,
  cooldownMs: 2500,
  onboardingComplete: false,

  setDeepgramApiKey: (deepgramApiKey) => set({ deepgramApiKey }),
  setOpenaiApiKey: (openaiApiKey) => set({ openaiApiKey }),
  setClaudeApiKey: (claudeApiKey) => set({ claudeApiKey }),
  setActiveTranslationId: (activeTranslationId) => set({ activeTranslationId }),
  setAudioDeviceId: (audioDeviceId) => set({ audioDeviceId }),
  setAudioChannelIndex: (audioChannelIndex) => set({ audioChannelIndex }),
  setGain: (gain) => set({ gain }),
  setVadEnabled: (vadEnabled) => set({ vadEnabled }),
  setAutoMode: (autoMode) => set({ autoMode }),
  setConfidenceThreshold: (confidenceThreshold) => set({ confidenceThreshold }),
  setCooldownMs: (cooldownMs) => set({ cooldownMs }),
  setOnboardingComplete: (onboardingComplete) => set({ onboardingComplete }),
}))
