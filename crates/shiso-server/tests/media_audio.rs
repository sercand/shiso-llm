//! Audio parts: decoded to clips beside the pictures, in prompt order, and
//! refused loudly when the checkpoint or the build cannot hear them.
//!
//! The tower here is a stand-in with the Gemma frontend's constants: what is
//! under test is the preprocessing path, and a real tower would add a device
//! and a checkpoint without changing a line of it.

use std::num::NonZeroU32;
use std::sync::Arc;

use base64::Engine as _;
use shiso_audio::{
    AudioClip, AudioError, AudioTower, DecodeLimits, Filterbank, Framing, LogScale, MelScale,
    MelSpec, Preemphasis, Spectrum, Window,
};
use shiso_core::{Backend, HfConfig};
use shiso_ir::{Block, MediaSource, Message, Role};
use shiso_server::media::MediaItem;
use shiso_server::tokenizer::message_preprocess::{MediaPolicy, PreprocessError, preprocess};
use shiso_vision::{EncodedRows, ImageLimits, PreprocessSpec};

const RECORDED_CONFIG: &str = include_str!("../../../test-vectors/loader/qwen35-config.json");

/// Two stride-2 stages over the mel rows, capped at `cap` tokens.
#[derive(Debug)]
struct Stand {
    spec: MelSpec,
}

impl AudioTower for Stand {
    fn mel_spec(&self) -> &MelSpec {
        &self.spec
    }
    fn tokens_for(&self, valid_mel: usize) -> Option<usize> {
        let tokens = valid_mel.div_ceil(2).div_ceil(2);
        (tokens <= 750).then_some(tokens)
    }
    fn max_seconds(&self) -> f64 {
        30.0
    }
    fn out_hidden(&self) -> usize {
        8
    }
    fn device_bytes(&self) -> u64 {
        0
    }
    fn encode(&self, _: &Arc<dyn Backend>, _: &[AudioClip]) -> Result<EncodedRows, AudioError> {
        Err(AudioError::Empty)
    }
}

fn stand() -> Stand {
    Stand {
        spec: MelSpec {
            sample_rate: 16_000,
            n_fft: 512,
            win_length: 320,
            hop_length: 160,
            n_mels: 128,
            window: Window::HannPeriodic,
            framing: Framing::LeftPadded {
                pad_to_multiple: 128,
            },
            preemphasis: Preemphasis::None,
            spectrum: Spectrum::Magnitude,
            filterbank: Filterbank::Computed {
                scale: MelScale::Htk,
                slaney_norm: false,
                f_min: 0.0,
                f_max: 8_000.0,
            },
            log_scale: LogScale::Natural { guard: 1e-3 },
        },
    }
}

fn vision_spec() -> PreprocessSpec {
    let vision = HfConfig::from_json_str(RECORDED_CONFIG)
        .expect("the recorded config parses")
        .vision_config()
        .expect("the vision block parses")
        .expect("the recorded config carries a vision block");
    PreprocessSpec::from_vision_config(&vision)
        .expect("the recorded geometry drives a preprocessor")
}

/// A 16 kHz mono 16-bit WAV of `samples` samples of a quiet tone.
fn wav(samples: u32) -> Vec<u8> {
    let data_len = samples * 2;
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&16_000u32.to_le_bytes());
    out.extend_from_slice(&32_000u32.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for n in 0..samples {
        let phase = f64::from(n) * 440.0 * std::f64::consts::TAU / 16_000.0;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the amplitude is 3000, well inside i16"
        )]
        let value = (phase.sin() * 3000.0) as i16;
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn audio_block(bytes: &[u8]) -> Block {
    Block::Audio(MediaSource::Inline(
        base64::engine::general_purpose::STANDARD.encode(bytes),
    ))
}

fn png_block() -> Block {
    let image = image::RgbImage::from_pixel(32, 32, image::Rgb([40, 90, 160]));
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(image)
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .expect("a solid-colour PNG encodes");
    Block::Image(MediaSource::classify(format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&bytes)
    )))
}

fn limits() -> DecodeLimits {
    DecodeLimits::new(NonZeroU32::new(30).expect("non-zero"))
}

#[test]
fn a_wav_clip_decodes_without_ffmpeg_and_sizes_its_pad_run() {
    let tower = stand();
    let policy = MediaPolicy {
        audio: Some(&tower),
        audio_limits: limits(),
        ..MediaPolicy::default()
    };
    // One second: 100 valid mel rows after left padding, so 25 tokens.
    let message = Message::new(Role::User, vec![audio_block(&wav(16_000))]);
    let prepared = preprocess(&[], std::slice::from_ref(&message), &policy).expect("it prepares");

    assert_eq!(prepared.media.len(), 1);
    let MediaItem::Audio(clip) = &prepared.media[0] else {
        panic!("expected an audio item, got {:?}", prepared.media[0]);
    };
    assert_eq!(prepared.pad_counts, vec![clip.tokens()]);
    assert_eq!(prepared.pad_counts[0], prepared.media[0].pad_count(1));
    assert_eq!(clip.tokens(), 25);
    let parts = prepared.messages[0]["content"]
        .as_array()
        .expect("a part list");
    assert_eq!(parts[0]["type"], "audio");
}

#[test]
fn mixed_media_keeps_the_order_the_turn_put_it_in() {
    let tower = stand();
    let spec = vision_spec();
    let policy = MediaPolicy {
        vision: Some(&spec),
        limits: ImageLimits::default(),
        audio: Some(&tower),
        audio_limits: limits(),
        ..MediaPolicy::default()
    };
    let message = Message::new(
        Role::User,
        vec![
            png_block(),
            Block::Text("then".to_owned()),
            audio_block(&wav(8_000)),
            png_block(),
        ],
    );
    let prepared = preprocess(&[], std::slice::from_ref(&message), &policy).expect("it prepares");

    let kinds: Vec<&str> = prepared
        .media
        .iter()
        .map(|item| match item {
            MediaItem::Vision(_) => "vision",
            MediaItem::Audio(_) => "audio",
            _ => "other",
        })
        .collect();
    assert_eq!(kinds, vec!["vision", "audio", "vision"]);
    let merge = spec.geometry.spatial_merge_size();
    let counts: Vec<usize> = prepared
        .media
        .iter()
        .map(|item| item.pad_count(merge))
        .collect();
    assert_eq!(prepared.pad_counts, counts);
    let parts: Vec<&str> = prepared.messages[0]["content"]
        .as_array()
        .expect("a part list")
        .iter()
        .filter_map(|part| part["type"].as_str())
        .collect();
    assert_eq!(parts, vec!["image", "text", "audio", "image"]);
}

#[test]
fn audio_sent_to_a_checkpoint_with_no_audio_tower_is_refused() {
    let spec = vision_spec();
    let policy = MediaPolicy {
        vision: Some(&spec),
        ..MediaPolicy::default()
    };
    let message = Message::new(Role::User, vec![audio_block(&wav(16_000))]);
    let error = preprocess(&[], std::slice::from_ref(&message), &policy)
        .expect_err("no audio tower is attached");
    assert!(
        matches!(error, PreprocessError::NoAudioTower),
        "got {error:?}"
    );
}

#[test]
fn an_mp3_without_ffmpeg_is_refused_naming_the_flag() {
    let tower = stand();
    let policy = MediaPolicy {
        audio: Some(&tower),
        audio_limits: limits(),
        ..MediaPolicy::default()
    };
    let mut mp3 = b"ID3\x04\x00\x00\x00\x00\x00\x00".to_vec();
    mp3.extend_from_slice(&[0u8; 64]);
    let message = Message::new(Role::User, vec![audio_block(&mp3)]);
    let error = preprocess(&[], std::slice::from_ref(&message), &policy)
        .expect_err("an mp3 needs the subprocess decoder");
    assert!(
        matches!(
            error,
            PreprocessError::Audio(AudioError::NeedsDecoder { .. })
        ),
        "got {error:?}"
    );
    assert!(
        error.to_string().contains("--audio-allow-ffmpeg"),
        "the refusal names the flag that would change it: {error}"
    );
}

#[test]
fn a_clip_past_the_decode_limit_is_refused() {
    let tower = stand();
    let policy = MediaPolicy {
        audio: Some(&tower),
        audio_limits: DecodeLimits::new(NonZeroU32::new(1).expect("non-zero")),
        ..MediaPolicy::default()
    };
    let message = Message::new(Role::User, vec![audio_block(&wav(32_000))]);
    let error = preprocess(&[], std::slice::from_ref(&message), &policy)
        .expect_err("two seconds is past a one-second limit");
    assert!(
        matches!(error, PreprocessError::Audio(AudioError::TooLong { .. })),
        "got {error:?}"
    );
}
