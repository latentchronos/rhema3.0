use std::time::{SystemTime, UNIX_EPOCH};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use crossbeam_channel::Sender;

use crate::error::AudioError;
use crate::resample::Resampler;
use crate::types::{AudioConfig, AudioFrame};

/// Holds a live audio capture stream.
/// Dropping this struct (or calling `stop`) will end the capture.
pub struct AudioCapture {
    stream: Stream,
}

impl AudioCapture {
    /// Stop the audio capture, consuming the struct.
    pub fn stop(self) {
        drop(self.stream);
    }
}

/// Start capturing audio from the given device (or default) and send frames
/// through the provided crossbeam sender.
///
/// Audio is converted to mono 16-bit PCM at 16 kHz, with the specified gain
/// applied. If a channel index is provided, only that source channel is used;
/// otherwise all input channels are mixed down to mono.
pub fn start(config: AudioConfig, sender: Sender<AudioFrame>) -> Result<AudioCapture, AudioError> {
    let host = cpal::default_host();

    // Select the device
    log::info!("[AUDIO] Requested device_id: {:?}", config.device_id);

    let device = match &config.device_id {
        Some(id) if !id.is_empty() => {
            let mut found = None;
            let input_devices = host.input_devices().map_err(|e| {
                AudioError::StreamError(format!("Failed to enumerate devices: {e}"))
            })?;
            for d in input_devices {
                if let Ok(name) = d.name() {
                    log::info!("[AUDIO]   Available device: '{}'", name);
                    if name == *id {
                        log::info!("[AUDIO]   ✓ MATCH: '{}'", name);
                        found = Some(d);
                        break;
                    }
                }
            }
            match found {
                Some(d) => {
                    log::info!("[AUDIO] Using requested device: '{}'", id);
                    d
                }
                None => {
                    log::warn!(
                        "[AUDIO] Device '{}' not found! Falling back to default.",
                        id
                    );
                    host.default_input_device()
                        .ok_or(AudioError::NoInputDevices)?
                }
            }
        }
        _ => {
            let d = host
                .default_input_device()
                .ok_or(AudioError::NoInputDevices)?;
            log::info!(
                "[AUDIO] Using default device: '{}'",
                d.name().unwrap_or_default()
            );
            d
        }
    };

    let supported_config = device
        .default_input_config()
        .map_err(|e| AudioError::StreamError(format!("Failed to get default input config: {e}")))?;

    let source_sample_rate = supported_config.sample_rate().0;
    let source_channels = supported_config.channels() as usize;
    let sample_format = supported_config.sample_format();

    let target_sample_rate: u32 = 16_000;
    let gain = config.gain;
    let channel_index = config
        .channel_index
        .and_then(|idx| (idx as usize).lt(&source_channels).then_some(idx as usize));

    let stream_config: StreamConfig = supported_config.into();

    let err_fn = |err: cpal::StreamError| {
        log::error!("Audio stream error: {err}");
    };

    // One stateful resampler per stream (device rate -> 16 kHz). Built per match arm and
    // moved into that arm's callback; only the arm matching `sample_format` is constructed.
    let stream = match sample_format {
        SampleFormat::I16 => {
            let sender = sender.clone();
            let mut resampler = Resampler::new(source_sample_rate, target_sample_rate);
            device.build_input_stream(
                &stream_config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    process_and_send(
                        data,
                        source_channels,
                        gain,
                        channel_index,
                        &mut resampler,
                        &sender,
                    );
                },
                err_fn,
                None,
            )
        }
        SampleFormat::F32 => {
            let sender = sender.clone();
            let mut resampler = Resampler::new(source_sample_rate, target_sample_rate);
            device.build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    // Convert f32 -> i16
                    let i16_data: Vec<i16> = data
                        .iter()
                        .map(|&s| {
                            let clamped = s.clamp(-1.0, 1.0);
                            (clamped * i16::MAX as f32) as i16
                        })
                        .collect();
                    process_and_send(
                        &i16_data,
                        source_channels,
                        gain,
                        channel_index,
                        &mut resampler,
                        &sender,
                    );
                },
                err_fn,
                None,
            )
        }
        SampleFormat::U16 => {
            let sender = sender.clone();
            let mut resampler = Resampler::new(source_sample_rate, target_sample_rate);
            device.build_input_stream(
                &stream_config,
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    // Convert u16 -> i16 (u16 midpoint is 32768)
                    let i16_data: Vec<i16> =
                        data.iter().map(|&s| (s as i32 - 32768) as i16).collect();
                    process_and_send(
                        &i16_data,
                        source_channels,
                        gain,
                        channel_index,
                        &mut resampler,
                        &sender,
                    );
                },
                err_fn,
                None,
            )
        }
        _ => {
            return Err(AudioError::StreamError(format!(
                "Unsupported sample format: {sample_format:?}"
            )));
        }
    }
    .map_err(|e| AudioError::StreamError(format!("Failed to build input stream: {e}")))?;

    stream
        .play()
        .map_err(|e| AudioError::StreamError(format!("Failed to start stream: {e}")))?;

    Ok(AudioCapture { stream })
}

/// Downmix to mono, apply gain, anti-aliased-resample to 16 kHz, and send as AudioFrame.
///
/// The `resampler` is stateful and persists across callbacks (see [`Resampler`]). Its
/// output may be empty while it buffers toward its next chunk — in that case nothing is
/// sent this callback, which is normal.
fn process_and_send(
    samples: &[i16],
    source_channels: usize,
    gain: f32,
    channel_index: Option<usize>,
    resampler: &mut Resampler,
    sender: &Sender<AudioFrame>,
) {
    if samples.is_empty() || source_channels == 0 {
        return;
    }

    // Step 1: select a single channel or downmix to mono by averaging channels.
    let mono = mix_to_mono(samples, source_channels, channel_index);

    // Step 2: move to f32 [-1, 1] and apply gain (gain in the float domain; clamped only
    // at the final i16 conversion so intermediate headroom isn't lost).
    let gained: Vec<f32> = mono.iter().map(|&s| (s as f32 / 32768.0) * gain).collect();

    // Step 3: anti-aliased resample to the target rate. May buffer (empty output).
    let resampled = resampler.process(&gained);
    if resampled.is_empty() {
        return;
    }

    // Step 4: back to i16 for the downstream (i16) pipeline.
    let out: Vec<i16> = resampled
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect();

    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    // Best-effort send; if the receiver is gone, just drop the frame.
    let _ = sender.try_send(AudioFrame {
        samples: out,
        timestamp_ms,
    });
}

fn mix_to_mono(samples: &[i16], source_channels: usize, channel_index: Option<usize>) -> Vec<i16> {
    samples
        .chunks(source_channels)
        .map(|frame| {
            if let Some(idx) = channel_index {
                return frame.get(idx).copied().unwrap_or_default();
            }

            let sum: i32 = frame.iter().map(|&s| s as i32).sum();
            (sum / frame.len().max(1) as i32) as i16
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::mix_to_mono;

    #[test]
    fn mixes_stereo_to_mono() {
        let samples = vec![100, 300, -100, -300];
        assert_eq!(mix_to_mono(&samples, 2, None), vec![200, -200]);
    }

    #[test]
    fn selects_requested_channel() {
        let samples = vec![100, 300, 500, -100, -300, -500];
        assert_eq!(mix_to_mono(&samples, 3, Some(1)), vec![300, -300]);
    }
}
