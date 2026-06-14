export interface VerseSegment {
  verseNumber?: number
  text: string
}

export interface VerseRenderData {
  reference: string
  segments: VerseSegment[]
}

// --- Phase 4: Channel routing model (mirrors rhema-broadcast channel types) ---

export type RoutingMode = "locked" | "preview" | "independent"

/** A verse rendered for any channel (mirrors rhema_broadcast::VerseDisplay). */
export interface ChannelVerse {
  book: string
  chapter: number
  verse_start: number
  verse_end: number | null
  reference: string
  text: string
  translation: string
}

/** Audience channel cache — committed content only. */
export interface AudienceChannelState {
  active_verse: ChannelVerse | null
  theme_id: string
}

/** Pastor channel cache — current + optional preview + indicators. */
export interface PastorChannelState {
  current_verse: ChannelVerse | null
  preview_verse: ChannelVerse | null
  translation: string
  timer_seconds: number | null
  mode: RoutingMode
}

export interface ChannelQueueItem {
  id: string
  verse: ChannelVerse
}

export interface ChannelDetection {
  verse: ChannelVerse
  confidence: number
  source: string
}

export interface ChannelSuggestion {
  verse: ChannelVerse
  score: number
  reason: string
}

export type DeviceConnection = "connected" | "disconnected" | "reconnecting"

/** Health of one output endpoint (mirrors rhema_broadcast::DeviceStatus). */
export interface DeviceStatus {
  label: string
  kind: string
  connection: DeviceConnection
  /** UNIX-epoch ms of the last connection-state change. */
  last_change_ms: number
  /** Verse that was live when this endpoint last disconnected. */
  last_known_verse: ChannelVerse | null
}

/** Operator channel cache — supervision surface (operator window only). */
export interface OperatorChannelState {
  queue: ChannelQueueItem[]
  detections: ChannelDetection[]
  suggestions: ChannelSuggestion[]
  routing_state: RoutingMode
  device_health: DeviceStatus[]
}

export interface RenderOptions {
  opacity?: number
  offsetX?: number
  offsetY?: number
  scale?: number // Scale factor for rendering at display size (e.g., 0.42 for 400px panel)
  imageCache?: Map<string, HTMLImageElement>
}

export interface ObsOverlayStatus {
  active: boolean
  port: number | null
  mainUrl: string | null
  altUrl: string | null
  clientCount: number
}

export type TextHorizontalAlign = "left" | "center" | "right" | "justify"
export type TextVerticalAlign = "top" | "middle" | "bottom"
export type TextTransform = "none" | "uppercase" | "lowercase" | "capitalize"
export type TextDecoration = "none" | "underline" | "line-through"

export interface BroadcastTheme {
  id: string
  name: string
  builtin: boolean
  pinned: boolean
  createdAt: number
  updatedAt: number
  resolution: { width: number; height: number }
  background: {
    type: "solid" | "gradient" | "image" | "transparent"
    color: string
    gradient: {
      type: "linear" | "radial"
      angle: number
      stops: { color: string; position: number }[]
    } | null
    image: {
      url: string
      fit: "cover" | "contain" | "stretch"
      blur: number
      brightness: number
      tint: string | null
    } | null
  }
  textBox: {
    enabled: boolean
    color: string
    opacity: number
    borderRadius: number
    padding: number
  }
  verseText: {
    fontFamily: string
    fontSize: number
    fontWeight: number
    color: string
    horizontalAlign?: TextHorizontalAlign
    verticalAlign?: TextVerticalAlign
    textTransform?: TextTransform
    textDecoration?: TextDecoration
    lineHeight: number
    letterSpacing: number
    shadow: { color: string; blur: number; x: number; y: number } | null
    outline: { color: string; width: number } | null
  }
  verseNumbers: {
    visible: boolean
    fontSize: number
    color: string
    superscript: boolean
  }
  reference: {
    fontFamily: string
    fontSize: number
    fontWeight: number
    color: string
    horizontalAlign?: TextHorizontalAlign
    verticalAlign?: TextVerticalAlign
    textTransform?: TextTransform
    textDecoration?: TextDecoration
    uppercase: boolean
    letterSpacing: number
    position: "above" | "below" | "inline"
  }
  layout: {
    anchor:
      | "center"
      | "top-left"
      | "top-center"
      | "top-right"
      | "bottom-left"
      | "bottom-center"
      | "bottom-right"
    offsetX: number
    offsetY: number
    padding: { top: number; right: number; bottom: number; left: number }
    textAlign: "left" | "center" | "right"
    backgroundWidth: number
    backgroundHeight: number
    textAreaWidth: number
    textAreaHeight: number
    referenceGap?: number
  }
  transition: {
    type: "fade" | "slide" | "scale" | "none"
    duration: number
    easing: "linear" | "ease-in" | "ease-out" | "ease-in-out"
    direction: "up" | "down" | "left" | "right"
  }
}
