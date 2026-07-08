export interface DeviceInfo {
  id: string
  name: string
  sample_rate: number
  channels: number
  is_default: boolean
}

export interface AudioLevel {
  rms: number
  peak: number
}

export interface AudioConfig {
  device_id: string | null
  sample_rate: number
  gain: number
  channel_index: number | null
  vad_enabled: boolean
}

/** An on-device STT model offered in Settings (from the `list_stt_models` command). */
export interface SttModelInfo {
  path: string
  label: string
  streaming: boolean
}
