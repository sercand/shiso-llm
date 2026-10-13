//! The decode entry point: native WAV, the decoder steps, and the caps.
//!
//! The clips are committed under `tests/fixtures/` (a 440 Hz tone, about 1 s
//! each unless the name says otherwise), generated once by ffmpeg. No test
//! here generates audio with ffmpeg; the subprocess tests only run it as the
//! decoder under test.

use std::num::NonZeroU32;

#[cfg(feature = "decode-ffmpeg")]
use shiso_vision::{Availability, FfmpegPolicy, probe};

use super::{AudioDecodePolicy, DecodeLimits, decode_audio, sniff_container};
use crate::error::AudioError;
use crate::pcm::MODEL_RATE;

const MP3_44K_STEREO: &[u8] = include_bytes!("../tests/fixtures/tone_44k_stereo.mp3");
#[cfg(any(feature = "decode-native", feature = "decode-ffmpeg"))]
const MP3_16K_5S: &[u8] = include_bytes!("../tests/fixtures/tone_16k_5s.mp3");
const M4A_44K: &[u8] = include_bytes!("../tests/fixtures/tone_44k.m4a");
const FLAC_16K: &[u8] = include_bytes!("../tests/fixtures/tone_16k.flac");
const VORBIS_44K_STEREO: &[u8] = include_bytes!("../tests/fixtures/tone_44k_stereo.ogg");
const OPUS_48K: &[u8] = include_bytes!("../tests/fixtures/tone_48k_opus.ogg");

fn no_ffmpeg() -> AudioDecodePolicy {
    AudioDecodePolicy::default()
}

#[cfg(feature = "decode-ffmpeg")]
fn with_ffmpeg() -> AudioDecodePolicy {
    AudioDecodePolicy::default().with_ffmpeg(FfmpegPolicy::default())
}

fn limits(secs: u32) -> DecodeLimits {
    DecodeLimits::new(NonZeroU32::new(secs).expect("non-zero"))
}

#[cfg(feature = "decode-ffmpeg")]
fn have_ffmpeg() -> bool {
    matches!(
        probe(Some(&FfmpegPolicy::default())),
        Availability::Ready { .. }
    )
}

/// A 16-bit PCM WAV of a 440 Hz tone at half scale, `secs` long, at `rate`
/// with `channels`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a half-scale sample is within i16 by construction"
)]
fn wav(secs: u32, rate: u32, channels: u16) -> Vec<u8> {
    let frames = secs * rate;
    let data_len = frames * 2 * u32::from(channels);
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2 * u32::from(channels)).to_le_bytes());
    out.extend_from_slice(&(2 * channels).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for i in 0..frames {
        let t = f64::from(i) / f64::from(rate);
        let s = (0.5 * (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 32_767.0).round() as i16;
        for _ in 0..channels {
            out.extend_from_slice(&s.to_le_bytes());
        }
    }
    out
}

#[test]
fn sniffs_the_common_containers() {
    assert_eq!(sniff_container(b"RIFF\0\0\0\0WAVEfmt "), "wav");
    assert_eq!(sniff_container(b"fLaC\0\0"), "flac");
    assert_eq!(sniff_container(b"OggS\0\0"), "ogg");
    assert_eq!(sniff_container(b"ID3\x04\0"), "mp3");
    assert_eq!(sniff_container(&[0xFF, 0xFB, 0x90, 0x00]), "mp3");
    assert_eq!(sniff_container(b"\0\0\0\x20ftypM4A "), "mp4");
    assert_eq!(sniff_container(&[0x1A, 0x45, 0xDF, 0xA3, 0]), "webm");
    assert_eq!(sniff_container(b"hello"), "unknown");
    assert_eq!(sniff_container(MP3_44K_STEREO), "mp3");
    assert_eq!(sniff_container(M4A_44K), "mp4");
    assert_eq!(sniff_container(FLAC_16K), "flac");
    assert_eq!(sniff_container(VORBIS_44K_STEREO), "ogg");
    assert_eq!(sniff_container(OPUS_48K), "ogg");
}

#[test]
fn wav_needs_no_decoder_and_lands_at_the_model_rate() {
    let pcm = decode_audio(&wav(2, 44_100, 2), &limits(60), &no_ffmpeg()).expect("decodes");
    assert_eq!(pcm.rate(), MODEL_RATE);
    assert_eq!(pcm.samples().len(), 32_000);
}

#[test]
fn the_duration_cap_applies_to_wav() {
    assert!(matches!(
        decode_audio(&wav(3, 16_000, 1), &limits(2), &no_ffmpeg()),
        Err(AudioError::TooLong { max: 2, .. })
    ));
}

#[test]
fn empty_input_is_refused() {
    assert!(matches!(
        decode_audio(&[], &limits(60), &no_ffmpeg()),
        Err(AudioError::Empty)
    ));
}

#[test]
fn a_wav_declaring_an_implausible_rate_is_refused() {
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF\0\0\0\0WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1_000_000_007u32.to_le_bytes());
    wav.extend_from_slice(&2_000_000_014u32.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&200u32.to_le_bytes());
    wav.extend_from_slice(&[0u8; 200]);
    assert!(matches!(
        decode_audio(&wav, &limits(60), &no_ffmpeg()),
        Err(AudioError::BadRate {
            rate: 1_000_000_007
        })
    ));
}

// --- no decoder in the build ------------------------------------------------

#[cfg(not(feature = "decode-native"))]
#[test]
fn other_containers_need_a_decoder_and_say_so() {
    for (bytes, container) in [
        (MP3_44K_STEREO, "mp3"),
        (FLAC_16K, "flac"),
        (M4A_44K, "mp4"),
    ] {
        let err = decode_audio(bytes, &limits(60), &no_ffmpeg()).expect_err("refused");
        assert!(
            matches!(
                &err,
                AudioError::UnsupportedContainer { container: c, codec: None } if *c == container
            ),
            "{err}"
        );
    }
}

// --- the symphonia step -----------------------------------------------------

/// Every fixture symphonia reads, with the sample count it must land in at
/// [`MODEL_RATE`]. The tones are 1 s long. mp3 and FLAC come out exact
/// (gapless decoding trims the mp3 encoder delay); vorbis adds 22 samples;
/// AAC in mp4 keeps its ~1024-frame priming and padding (symphonia's mp4
/// reader is not gapless), 719 samples on this file.
#[cfg(feature = "decode-native")]
const NATIVE_FIXTURES: [(&str, &[u8], std::ops::RangeInclusive<usize>); 4] = [
    ("mp3", MP3_44K_STEREO, 16_000..=16_000),
    ("m4a", M4A_44K, 16_000..=16_800),
    ("flac", FLAC_16K, 16_000..=16_000),
    ("vorbis", VORBIS_44K_STEREO, 16_000..=16_050),
];

#[cfg(feature = "decode-native")]
#[test]
fn every_fixture_decodes_natively_at_the_model_rate() {
    for (name, bytes, expected) in NATIVE_FIXTURES {
        let pcm = decode_audio(bytes, &limits(60), &no_ffmpeg())
            .unwrap_or_else(|err| panic!("{name}: {err}"));
        assert_eq!(pcm.rate(), MODEL_RATE, "{name}");
        let n = pcm.samples().len();
        assert!(expected.contains(&n), "{name}: {n} samples");
        // ffmpeg's sine source is at 1/8 scale (about -3 dB lower where it
        // was upmixed to stereo; AAC's onset overshoots a little). Silence,
        // or a channel sum that was not averaged, falls outside.
        let peak = pcm.samples().iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!((0.05..=0.2).contains(&peak), "{name}: peak {peak}");
    }
}

#[cfg(feature = "decode-native")]
#[test]
fn the_duration_cap_stops_a_native_decode() {
    let err = decode_audio(MP3_16K_5S, &limits(2), &no_ffmpeg()).expect_err("too long");
    // Decoding stopped at the cap plus one second, not at the end of the file.
    assert!(
        matches!(err, AudioError::TooLong { seconds, max: 2 } if (seconds - 3.0).abs() < 0.01),
        "{err}"
    );
    let pcm = decode_audio(MP3_16K_5S, &limits(5), &no_ffmpeg()).expect("within the cap");
    assert_eq!(pcm.samples().len(), 80_000);
}

#[cfg(feature = "decode-native")]
#[test]
fn garbage_behind_an_mp3_signature_is_malformed() {
    let mut junk = b"ID3\x04\0\0\0\0\0\x0a".to_vec();
    junk.extend(
        (0u32..4096).map(|i| u8::try_from(i.wrapping_mul(2_654_435_761) >> 24).expect("one byte")),
    );
    let mut bad_frame = vec![0xFF, 0xFB, 0x90, 0x64];
    bad_frame.extend(std::iter::repeat_n(0xA5u8, 2048));
    for (name, bytes) in [("id3 then junk", junk), ("bad frame", bad_frame)] {
        let err = decode_audio(&bytes, &limits(60), &no_ffmpeg()).expect_err(name);
        assert!(
            matches!(
                err,
                AudioError::Malformed {
                    container: "mp3",
                    ..
                }
            ),
            "{name}: {err}"
        );
    }
}

#[cfg(feature = "decode-native")]
#[test]
fn a_truncated_mp3_decodes_the_frames_it_holds() {
    // An mp3 is a run of self-contained frames with no index to contradict, so
    // a cut-off upload is a shorter clip, not a broken one (ffmpeg reads it the
    // same way). It must not be handed to the next step either.
    let cut = &MP3_44K_STEREO[..MP3_44K_STEREO.len() / 2];
    let pcm = decode_audio(cut, &limits(60), &no_ffmpeg()).expect("the frames before the cut");
    let n = pcm.samples().len();
    assert!((6_000..10_000).contains(&n), "{n} samples");
}

#[cfg(feature = "decode-native")]
#[test]
fn an_mp3_behind_a_tag_larger_than_the_probe_window_decodes() {
    // Embedded cover art makes multi-megabyte ID3v2 tags common. The fixture's
    // own tag is 10 + 0x22 bytes; swap it for one of 1.2 MB of padding.
    const TAG_BODY: u32 = 1_200_000;
    let frames = &MP3_44K_STEREO[10 + 0x22..];
    assert_eq!(&frames[..2], &[0xFF, 0xFB], "the fixture's first frame");
    let mut tagged = b"ID3\x04\x00\x00".to_vec();
    // Syncsafe: seven bits per byte.
    tagged.extend(
        (0..4)
            .rev()
            .map(|i| u8::try_from((TAG_BODY >> (7 * i)) & 0x7F).expect("seven bits")),
    );
    tagged.resize(tagged.len() + usize::try_from(TAG_BODY).expect("fits"), 0);
    tagged.extend_from_slice(frames);
    let pcm = decode_audio(&tagged, &limits(60), &no_ffmpeg()).expect("decodes");
    assert_eq!(pcm.samples().len(), 16_000);
}

#[cfg(feature = "decode-native")]
#[test]
fn a_truncated_flac_is_malformed() {
    let cut = &FLAC_16K[..FLAC_16K.len() / 2];
    let err = decode_audio(cut, &limits(60), &no_ffmpeg()).expect_err("cut mid-frame");
    assert!(
        matches!(
            err,
            AudioError::Malformed {
                container: "flac",
                ..
            }
        ),
        "{err}"
    );
}

#[cfg(feature = "decode-native")]
#[test]
fn opus_without_an_ffmpeg_binary_is_refused_naming_the_codec() {
    // With or without the ffmpeg feature: the policy names no binary, so the
    // chain ends after symphonia.
    let err = decode_audio(OPUS_48K, &limits(60), &no_ffmpeg()).expect_err("no opus decoder");
    assert!(
        matches!(
            &err,
            AudioError::UnsupportedContainer { container: "ogg", codec: Some(c) } if c == "opus"
        ),
        "{err}"
    );
}

// --- the ffmpeg step ----------------------------------------------------------

#[cfg(feature = "decode-ffmpeg")]
#[test]
fn every_fixture_decodes_with_ffmpeg_configured() {
    if !have_ffmpeg() {
        return;
    }
    let policy = with_ffmpeg();
    // Opus always reaches the subprocess (symphonia has no Opus decoder); the
    // others reach it only in a build without `decode-native`.
    for (name, bytes) in [
        ("opus", OPUS_48K),
        ("flac", FLAC_16K),
        ("mp3", MP3_44K_STEREO),
    ] {
        let pcm =
            decode_audio(bytes, &limits(60), &policy).unwrap_or_else(|err| panic!("{name}: {err}"));
        assert_eq!(pcm.rate(), MODEL_RATE, "{name}");
        // mp3 adds encoder delay padding through ffmpeg; allow 50 ms.
        let n = pcm.samples().len();
        assert!((16_000..=16_800).contains(&n), "{name}: {n} samples");
    }
}

#[cfg(all(feature = "decode-ffmpeg", not(feature = "decode-native")))]
#[test]
fn the_duration_cap_applies_to_the_subprocess() {
    if !have_ffmpeg() {
        return;
    }
    assert!(matches!(
        decode_audio(MP3_16K_5S, &limits(2), &with_ffmpeg()),
        Err(AudioError::TooLong { max: 2, .. })
    ));
}
