# Audio Pipeline

Rhema captures microphone or mixer audio through the `rhema-audio` crate, converts it to mono 16 kHz PCM, emits local level-meter events, and forwards frames to the STT pipeline.

## Capture Configuration

`AudioConfig` supports:

- `device_id`: selected input device, or system default.
- `sample_rate`: target rate requested by the caller. Live STT currently uses 16 kHz for Deepgram linear16.
- `gain`: input gain from `0.0` to `2.0`.
- `channel_index`: optional zero-based source channel. `null` mixes all source channels to mono.
- `vad_enabled`: enables local energy-based voice activity detection before STT.

The default operator behavior is unchanged: mix all channels, unity gain, and VAD off.

## Multi-Channel Inputs

For church mixers and USB interfaces, each physical or bus output may arrive as a separate channel. Rhema now lets the operator choose one channel when speech is isolated, or mix all channels for simpler setups.

Use a single channel when:

- the pastor microphone is on a dedicated mixer bus;
- music and room ambience pollute the full mix;
- STT quality improves when non-speech channels are removed.

Use all-channel mixdown when:

- the device is mono or simple stereo;
- all useful speech is already mixed together;
- the operator does not know the channel layout.

## Voice Activity Detection

The local VAD is energy-based and lives in `src-tauri/crates/audio/src/vad.rs`. It keeps a small pre-buffer so speech onsets are not clipped, emits start/end transitions, and gates silence before audio reaches STT.

VAD is intentionally optional because providers such as Deepgram also perform endpointing. Enable local VAD when long silence creates noisy transcripts or unnecessary network traffic. Leave it off when the provider is already segmenting well.

## Verification

The audio crate includes tests for VAD state transitions and channel mixing. Future regression fixtures should include recorded mixer samples for:

- mono microphone;
- stereo room mix;
- multi-channel mixer bus with isolated speech;
- speech with long pauses.
