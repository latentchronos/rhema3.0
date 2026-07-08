import { useBroadcastStore } from "@/stores/broadcast-store"
import { useTauriEvent } from "./use-tauri-event"
import type {
  AudienceChannelState,
  OperatorChannelState,
  PastorChannelState,
} from "@/types"

/**
 * Phase 4 channel subscriber. Registers the three Tauri channel listeners and
 * pushes each payload into the broadcast store's channel caches. The backend is
 * the single source of truth; these listeners are the ONLY writers of the
 * channel-state fields (see broadcast-store.ts).
 *
 * Mount once, in a long-lived operator-window component (Dashboard). Note the
 * pastor channel is emitted to a dedicated pastor window that does not exist yet
 * (a later output-rollout stage), so `pastor_channel_update` will not fire in
 * the operator window today — the listener is in place for when it does.
 */
export function useChannels() {
  useTauriEvent<AudienceChannelState>("audience_channel_update", (payload) => {
    useBroadcastStore.getState().setAudienceChannel(payload)
  })
  useTauriEvent<PastorChannelState>("pastor_channel_update", (payload) => {
    useBroadcastStore.getState().setPastorChannel(payload)
  })
  useTauriEvent<OperatorChannelState>("operator_channel_update", (payload) => {
    useBroadcastStore.getState().setOperatorChannel(payload)
  })
}
