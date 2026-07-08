import { useState, useEffect, useCallback } from "react"
import { invoke } from "@tauri-apps/api/core"

import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Slider } from "@/components/ui/slider"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
} from "@/components/ui/sidebar"
import {
  MicIcon,
  TvIcon,
  KeyIcon,
  SettingsIcon,
  CheckIcon,
  BookOpenIcon,
  SparklesIcon,
} from "lucide-react"
import { useSettingsStore } from "@/stores"
import type { DeviceInfo } from "@/types/audio"

/* -------------------------------------------------------------------------- */
/*  Nav definition                                                            */
/* -------------------------------------------------------------------------- */

type NavSection = "audio" | "bible" | "display" | "api-keys" | "ai-model"

const navItems: { name: string; id: NavSection; icon: React.ReactNode }[] = [
  {
    name: "Audio",
    id: "audio",
    icon: <MicIcon strokeWidth={2} />,
  },
  {
    name: "Bible",
    id: "bible",
    icon: <BookOpenIcon strokeWidth={2} />,
  },
  {
    name: "Display Mode",
    id: "display",
    icon: <TvIcon strokeWidth={2} />,
  },
  {
    name: "API Keys",
    id: "api-keys",
    icon: <KeyIcon strokeWidth={2} />,
  },
  {
    name: "AI Model",
    id: "ai-model",
    icon: <SparklesIcon strokeWidth={2} />,
  },
]

/* -------------------------------------------------------------------------- */
/*  Section: Audio                                                            */
/* -------------------------------------------------------------------------- */

function AudioSection() {
  const {
    audioDeviceId,
    setAudioDeviceId,
    audioChannelIndex,
    setAudioChannelIndex,
    gain,
    setGain,
    vadEnabled,
    setVadEnabled,
    commandWakeWord,
    setCommandWakeWord,
  } = useSettingsStore()

  const [devices, setDevices] = useState<DeviceInfo[]>([])
  const [loading, setLoading] = useState(true)

  const loadDevices = useCallback(async () => {
    try {
      setLoading(true)
      const result = await invoke<DeviceInfo[]>("get_audio_devices")
      setDevices(result)
    } catch {
      // Tauri command may not be available during dev
      setDevices([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadDevices()
  }, [loadDevices])

  // gain is 0.0-2.0 in store, display as 0-100%
  const gainPercent = Math.round((gain / 2.0) * 100)
  const selectedDevice =
    devices.find((device) => device.id === audioDeviceId) ??
    devices.find((device) => device.is_default)
  const channelCount = selectedDevice?.channels ?? 1

  return (
    <div className="flex flex-col gap-6">
      {/* Device selector */}
      <div className="flex flex-col gap-2">
        <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Input Device
        </label>
        <Select
          value={audioDeviceId ?? "__default__"}
          onValueChange={(v) => {
            setAudioDeviceId(v === "__default__" ? null : v)
            setAudioChannelIndex(null)
          }}
          disabled={loading}
        >
          <SelectTrigger className="h-8 text-xs">
            <SelectValue
              placeholder={loading ? "Loading devices..." : "System default"}
            />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__default__">System default</SelectItem>
            {devices.map((device) => (
              <SelectItem key={device.id} value={device.id}>
                {device.name}
                {device.is_default ? " (default)" : ""}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-[0.625rem] text-muted-foreground">
          Selected device persists across sessions. Leave as system default to
          follow OS audio routing.
        </p>
      </div>

      {channelCount > 1 && (
        <div className="flex flex-col gap-2">
          <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Input Channel
          </label>
          <Select
            value={
              audioChannelIndex === null ? "__mix__" : String(audioChannelIndex)
            }
            onValueChange={(v) =>
              setAudioChannelIndex(v === "__mix__" ? null : Number(v))
            }
          >
            <SelectTrigger className="h-8 text-xs">
              <SelectValue placeholder="Mix all channels" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__mix__">Mix all channels</SelectItem>
              {Array.from({ length: channelCount }, (_, index) => (
                <SelectItem key={index} value={String(index)}>
                  Channel {index + 1}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-[0.625rem] text-muted-foreground">
            Choose one mixer channel when speech is isolated, or mix all channels
            for simple microphone setups.
          </p>
        </div>
      )}

      {/* Input gain */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Input Gain
          </label>
          <span className="text-xs tabular-nums text-muted-foreground">
            {gainPercent}%
          </span>
        </div>
        <Slider
          min={0}
          max={100}
          step={1}
          value={[gainPercent]}
          onValueChange={([v]) => setGain((v / 100) * 2.0)}
        />
        <p className="text-[0.625rem] text-muted-foreground">
          Amplifies the incoming audio signal before transcription. 50% is unity
          gain.
        </p>
      </div>

      <div className="flex items-start justify-between gap-4 rounded-lg border p-3">
        <div className="flex flex-col gap-1">
          <span className="text-xs font-medium text-foreground">
            Voice Activity Detection
          </span>
          <p className="text-[0.625rem] leading-relaxed text-muted-foreground">
            Gate silence locally before audio is sent to transcription. Leave off
            if your mixer or transcription provider already handles silence well.
          </p>
        </div>
        <Button
          type="button"
          size="sm"
          variant={vadEnabled ? "default" : "outline"}
          onClick={() => setVadEnabled(!vadEnabled)}
          className="h-8 min-w-14 text-xs"
        >
          {vadEnabled ? "On" : "Off"}
        </Button>
      </div>

      {/* Wake word */}
      <div className="flex flex-col gap-2">
        <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Voice command wake word (optional)
        </label>
        <Input
          type="text"
          placeholder="e.g. rhema — leave empty to disable"
          value={commandWakeWord ?? ""}
          onChange={(e) => {
            const v = e.target.value.trim()
            setCommandWakeWord(v || null)
          }}
          className="text-xs"
        />
        <p className="text-[0.625rem] leading-relaxed text-muted-foreground">
          When set, voice navigation commands must be prefixed with this word
          (e.g. &quot;rhema next verse&quot;). Leave empty to accept commands without a
          prefix. Disabled by default.
        </p>
      </div>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/*  Section: Display Mode                                                     */
/* -------------------------------------------------------------------------- */

function DisplayModeSection() {
  const {
    autoMode,
    setAutoMode,
    confidenceThreshold,
    setConfidenceThreshold,
  } = useSettingsStore()

  const thresholdPercent = Math.round(confidenceThreshold * 100)

  return (
    <div className="flex flex-col gap-6">
      {/* Mode selector */}
      <div className="flex flex-col gap-3">
        <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Broadcast Mode
        </label>

        <RadioGroup
          value={autoMode ? "auto" : "manual"}
          onValueChange={(v) => setAutoMode(v === "auto")}
          className="gap-3"
        >
          {/* Auto mode */}
          <label
            className={`flex cursor-pointer items-start gap-3 rounded-lg border p-3 transition-colors has-data-[state=checked]:border-primary/50 has-data-[state=checked]:bg-primary/5 has-data-[state=checked]:ring-1 has-data-[state=checked]:ring-primary/20 ${
              !autoMode ? "hover:border-muted-foreground/25" : ""
            }`}
          >
            <RadioGroupItem value="auto" className="mt-0.5" />
            <div className="flex flex-col gap-1">
              <span className="text-xs font-medium text-foreground">Auto</span>
              <p className="text-[0.625rem] leading-relaxed text-muted-foreground">
                Automatically displays the highest-confidence detected verse on
                broadcast output. A 2.5-second cooldown prevents rapid flickering.
                Best for hands-off operation.
              </p>
            </div>
          </label>

          {/* Manual mode */}
          <label
            className={`flex cursor-pointer items-start gap-3 rounded-lg border p-3 transition-colors has-data-[state=checked]:border-primary/50 has-data-[state=checked]:bg-primary/5 has-data-[state=checked]:ring-1 has-data-[state=checked]:ring-primary/20 ${
              autoMode ? "hover:border-muted-foreground/25" : ""
            }`}
          >
            <RadioGroupItem value="manual" className="mt-0.5" />
            <div className="flex flex-col gap-1">
              <span className="text-xs font-medium text-foreground">Manual</span>
              <p className="text-[0.625rem] leading-relaxed text-muted-foreground">
                Nothing goes to broadcast until you explicitly send it. Detected
                verses still appear in the AI Detections panel and queue, but you
                decide which ones to display and when. Best for important services.
              </p>
            </div>
          </label>
        </RadioGroup>
      </div>

      {/* Confidence threshold — only when auto */}
      {autoMode && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Confidence Threshold
            </label>
            <span className="text-xs tabular-nums text-muted-foreground">
              {thresholdPercent}%
            </span>
          </div>
          <Slider
            min={35}
            max={100}
            step={1}
            value={[thresholdPercent]}
            onValueChange={([v]) => setConfidenceThreshold(v / 100)}
          />
          <p className="text-[0.625rem] text-muted-foreground">
            Only verses with confidence above this threshold will be sent to
            broadcast automatically. Higher values reduce false positives.
          </p>
        </div>
      )}
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/*  Section: API Keys                                                         */
/* -------------------------------------------------------------------------- */

function ApiKeysSection() {
  const { deepgramApiKey, setDeepgramApiKey } = useSettingsStore()
  const [keyValue, setKeyValue] = useState(deepgramApiKey ?? "")
  const [saved, setSaved] = useState(false)

  const handleSave = () => {
    setDeepgramApiKey(keyValue || null)
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Deepgram API Key
          </label>
          {deepgramApiKey && (
            <Badge variant="outline" className="text-[0.5rem]">
              Key configured
            </Badge>
          )}
        </div>
        <div className="flex gap-2">
          <Input
            type="password"
            placeholder="Enter your Deepgram API key..."
            value={keyValue}
            onChange={(e) => setKeyValue(e.target.value)}
            className="flex-1 text-xs"
          />
          <Button size="sm" onClick={handleSave}>
            {saved ? (
              <>
                <CheckIcon className="size-3" />
                Saved
              </>
            ) : (
              "Save"
            )}
          </Button>
        </div>
        <p className="text-[0.625rem] text-muted-foreground">
          Required for live transcription. Get a key at{" "}
          <span className="text-primary">deepgram.com</span>
        </p>
      </div>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/*  Section: AI Model (Stage-2 LLM)                                            */
/* -------------------------------------------------------------------------- */

type LlmProviderSel = "auto" | "anthropic" | "openai" | "gemini" | "custom"

interface LlmStatus {
  configured: boolean
  provider: string | null
  model: string | null
  base_url: string | null
}

const providerOptions: { value: LlmProviderSel; label: string }[] = [
  { value: "auto", label: "Auto-detect from key" },
  { value: "anthropic", label: "Claude (Anthropic)" },
  { value: "openai", label: "OpenAI / ChatGPT" },
  { value: "gemini", label: "Google Gemini" },
  { value: "custom", label: "Custom (OpenAI-compatible)" },
]

/** Map the UI selection to the backend `ProviderKind` arg (null = auto-detect). */
function backendProvider(sel: LlmProviderSel): string | null {
  switch (sel) {
    case "auto":
      return null
    case "anthropic":
      return "anthropic"
    case "gemini":
      return "gemini"
    case "openai":
    case "custom":
      return "open_ai_compatible"
  }
}

function AiModelSection() {
  const {
    llmProvider,
    llmApiKey,
    llmBaseUrl,
    llmModel,
    setLlmProvider,
    setLlmApiKey,
    setLlmBaseUrl,
    setLlmModel,
  } = useSettingsStore()

  const [sel, setSel] = useState<LlmProviderSel>(
    (llmProvider as LlmProviderSel) ?? "auto"
  )
  const [keyValue, setKeyValue] = useState(llmApiKey ?? "")
  const [base, setBase] = useState(llmBaseUrl ?? "")
  const [model, setModel] = useState(llmModel ?? "")
  const [status, setStatus] = useState<LlmStatus | null>(null)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    void invoke<LlmStatus>("llm_status").then(setStatus).catch(() => {})
  }, [])

  const handleSave = async () => {
    setError(null)
    if (!keyValue.trim()) {
      await invoke("clear_llm_config").catch(() => {})
      setLlmApiKey(null)
      setStatus({ configured: false, provider: null, model: null, base_url: null })
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
      return
    }
    try {
      await invoke<string>("set_llm_config", {
        provider: backendProvider(sel),
        apiKey: keyValue.trim(),
        baseUrl: sel === "custom" ? base.trim() || null : null,
        model: model.trim() || null,
      })
      setLlmProvider(sel)
      setLlmApiKey(keyValue.trim())
      setLlmBaseUrl(sel === "custom" ? base.trim() || null : null)
      setLlmModel(model.trim() || null)
      const s = await invoke<LlmStatus>("llm_status")
      setStatus(s)
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch (e) {
      setError(String(e))
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <p className="text-[0.625rem] leading-relaxed text-muted-foreground">
        When the speaker mentions a passage without naming it outright, Rhema can
        ask an AI model to help find it. Paste any provider&apos;s key — Claude,
        OpenAI/ChatGPT, Gemini, or any OpenAI-compatible model (DeepSeek, Qwen,
        Groq, local Ollama, …) via the Custom option.
      </p>

      {/* Provider */}
      <div className="flex flex-col gap-2">
        <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Provider
        </label>
        <Select value={sel} onValueChange={(v) => setSel(v as LlmProviderSel)}>
          <SelectTrigger className="h-8 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {providerOptions.map((o) => (
              <SelectItem key={o.value} value={o.value}>
                {o.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* API key */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            API Key
          </label>
          {status?.configured && (
            <Badge variant="outline" className="text-[0.5rem]">
              {status.provider ?? "configured"}
              {status.model ? ` · ${status.model}` : ""}
            </Badge>
          )}
        </div>
        <div className="flex gap-2">
          <Input
            type="password"
            placeholder="Paste your model API key..."
            value={keyValue}
            onChange={(e) => setKeyValue(e.target.value)}
            className="flex-1 text-xs"
          />
          <Button size="sm" onClick={() => void handleSave()}>
            {saved ? (
              <>
                <CheckIcon className="size-3" />
                Saved
              </>
            ) : (
              "Save"
            )}
          </Button>
        </div>
        {error && <p className="text-[0.625rem] text-destructive">{error}</p>}
      </div>

      {/* Custom OpenAI-compatible: base URL */}
      {sel === "custom" && (
        <div className="flex flex-col gap-2">
          <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Base URL
          </label>
          <Input
            placeholder="https://api.deepseek.com"
            value={base}
            onChange={(e) => setBase(e.target.value)}
            className="text-xs"
          />
          <p className="text-[0.625rem] text-muted-foreground">
            The provider&apos;s OpenAI-compatible endpoint root (no trailing
            /v1/chat/completions).
          </p>
        </div>
      )}

      {/* Optional model override */}
      <div className="flex flex-col gap-2">
        <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Model (optional)
        </label>
        <Input
          placeholder="Leave blank for the provider default"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="text-xs"
        />
      </div>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/*  Section titles                                                            */
/* -------------------------------------------------------------------------- */

const sectionTitles: Record<NavSection, string> = {
  audio: "Audio",
  bible: "Bible",
  display: "Display Mode",
  "api-keys": "API Keys",
  "ai-model": "AI Model",
}

/* -------------------------------------------------------------------------- */
/*  Section: Bible Translation                                                */
/* -------------------------------------------------------------------------- */

interface TranslationInfo {
  id: number
  abbreviation: string
  title: string
  language: string
}

function BibleSection() {
  const [translations, setTranslations] = useState<TranslationInfo[]>([])
  const [activeId, setActiveId] = useState<number>(1)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function load() {
      try {
        const [trans, active] = await Promise.all([
          invoke<TranslationInfo[]>("list_translations"),
          invoke<number>("get_active_translation"),
        ])
        setTranslations(trans)
        setActiveId(active)
      } catch (e) {
        console.error("Failed to load translations:", e)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const handleChange = async (value: string) => {
    const id = parseInt(value)
    try {
      await invoke("set_active_translation", { translationId: id })
      setActiveId(id)
      // Update frontend stores so all panels use the new translation
      const { useBibleStore } = await import("@/stores")
      useBibleStore.getState().setActiveTranslation(id)
    } catch (e) {
      console.error("Failed to set translation:", e)
    }
  }

  const englishTranslations = translations.filter((t) => t.language === "en")
  const otherTranslations = translations.filter((t) => t.language !== "en")

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Primary Translation
        </label>
        <Select
          value={String(activeId)}
          onValueChange={handleChange}
          disabled={loading}
        >
          <SelectTrigger className="h-8 text-xs">
            <SelectValue placeholder={loading ? "Loading..." : "Select translation"} />
          </SelectTrigger>
          <SelectContent>
            {englishTranslations.length > 0 && (
              <>
                <div className="px-2 py-1 text-[0.5625rem] font-medium uppercase tracking-wider text-muted-foreground">
                  English
                </div>
                {englishTranslations.map((t) => (
                  <SelectItem key={t.id} value={String(t.id)}>
                    {t.abbreviation} — {t.title}
                  </SelectItem>
                ))}
              </>
            )}
            {otherTranslations.length > 0 && (
              <>
                <div className="mt-1 px-2 py-1 text-[0.5625rem] font-medium uppercase tracking-wider text-muted-foreground">
                  Other Languages
                </div>
                {otherTranslations.map((t) => (
                  <SelectItem key={t.id} value={String(t.id)}>
                    {t.abbreviation} — {t.title}
                  </SelectItem>
                ))}
              </>
            )}
          </SelectContent>
        </Select>
        <p className="text-[0.625rem] text-muted-foreground">
          Detected verses will display in this translation.
          {translations.length > 0 && ` ${translations.length} translations available.`}
        </p>
      </div>
    </div>
  )
}

const sectionComponents: Record<NavSection, React.FC> = {
  audio: AudioSection,
  bible: BibleSection,
  display: DisplayModeSection,
  "api-keys": ApiKeysSection,
  "ai-model": AiModelSection,
}

/* -------------------------------------------------------------------------- */
/*  Main dialog                                                               */
/* -------------------------------------------------------------------------- */

export function SettingsDialog() {
  const [open, setOpen] = useState(false)
  const [activeSection, setActiveSection] = useState<NavSection>("audio")

  const ActiveContent = sectionComponents[activeSection]

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon-sm">
          <SettingsIcon className="size-3.5" />
        </Button>
      </DialogTrigger>
      <DialogContent className="overflow-hidden p-0 md:max-h-[600px] md:max-w-[800px] lg:max-w-[900px]">
        <DialogTitle className="sr-only">Settings</DialogTitle>
        <DialogDescription className="sr-only">
          Configure audio, display mode, and API keys.
        </DialogDescription>
        <SidebarProvider className="items-start">
          <Sidebar collapsible="none" className="hidden md:flex">
            <div className="h-14 border-b border-border border-r px-4 flex items-center" >
              Settings
            </div>
            <SidebarContent className="border-r border-border">
              <SidebarGroup>
                <SidebarGroupContent>
                  <SidebarMenu>
                    {navItems.map((item) => (
                      <SidebarMenuItem key={item.id}>
                        <SidebarMenuButton
                          isActive={item.id === activeSection}
                          onClick={() => setActiveSection(item.id)}
                        >
                          {item.icon}
                          <span>{item.name}</span>
                        </SidebarMenuButton>
                      </SidebarMenuItem>
                    ))}
                  </SidebarMenu>
                </SidebarGroupContent>
              </SidebarGroup>
            </SidebarContent>
          </Sidebar>
          <main className="flex h-[580px] flex-1 flex-col overflow-hidden">
            <header className="flex h-14 shrink-0 items-center gap-2 border-b border-border">
              <div className="flex items-center gap-2 px-4">
                {sectionTitles[activeSection]}
              </div>
            </header>
            <div className="flex flex-1 flex-col overflow-y-auto p-4">
              <ActiveContent />
            </div>
          </main>
        </SidebarProvider>
      </DialogContent>
    </Dialog>
  )
}
