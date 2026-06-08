use std::time::{SystemTime, UNIX_EPOCH};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use crossbeam_channel::Sender;

use crate::error::AudioError;
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

    let stream = match sample_format {
        SampleFormat::I16 => {
            let sender = sender.clone();
            device.build_input_stream(
                &stream_config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    process_and_send(
                        data,
                        source_channels,
                        source_sample_rate,
                        target_sample_rate,
                        gain,
                        channel_index,
                        &sender,
                    );
                },
                err_fn,
                None,
            )
        }
        SampleFormat::F32 => {
            let sender = sender.clone();
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
                        source_sample_rate,
                        target_sample_rate,
                        gain,
                        channel_index,
                        &sender,
                    );
                },
                err_fn,
                None,
            )
        }
        SampleFormat::U16 => {
            let sender = sender.clone();
            device.build_input_stream(
                &stream_config,
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    // Convert u16 -> i16 (u16 midpoint is 32768)
                    let i16_data: Vec<i16> =
                        data.iter().map(|&s| (s as i32 - 32768) as i16).collect();
                    process_and_send(
                        &i16_data,
                        source_channels,
                        source_sample_rate,
                        target_sample_rate,
                        gain,
                        channel_index,
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

/// Downmix to mono, apply gain, resample to target rate, and send as AudioFrame.
fn process_and_send(
    samples: &[i16],
    source_channels: usize,
    source_rate: u32,
    target_rate: u32,
    gain: f32,
    channel_index: Option<usize>,
    sender: &Sender<AudioFrame>,
) {
    if samples.is_empty() || source_channels == 0 {
        return;
    }

    // Step 1: Select a single channel or downmix to mono by averaging channels.
    let mono = mix_to_mono(samples, source_channels, channel_index);

    // Step 2: Apply gain
    let gained: Vec<i16> = mono
        .iter()
        .map(|&s| {
            let amplified = (s as f32 * gain) as i32;
            amplified.clamp(i16::MIN as i32, i16::MAX as i32) as i16
        })
        .collect();

    // Step 3: Resample to target rate (simple linear interpolation)
    let resampled = if source_rate == target_rate {
        gained
    } else {
        resample(&gained, source_rate, target_rate)
    };

    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let frame = AudioFrame {
        samples: resampled,
        timestamp_ms,
    };

    // Best-effort send; if the receiver is gone, just drop the frame.
    let _ = sender.try_send(frame);
}

/// Simple linear-interpolation resampler.
fn resample(input: &[i16], from_rate: u32, to_rate: u32) -> Vec<i16> {
    if input.is_empty() {
        return Vec::new();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = ((input.len() as f64) / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f64;

        let sample = if idx + 1 < input.len() {
            let a = input[idx] as f64;
            let b = input[idx + 1] as f64;
            (a + (b - a) * frac) as i16
        } else {
            input[input.len() - 1]
        };

        output.push(sample);
    }

    output
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
