//! The symphonia step's own seams: downmix, the layer sniff, codec names and
//! the deadline. The fixtures end to end are in `decode_tests.rs`.

use std::num::NonZeroU32;
use std::time::Instant;

use shiso_vision::VisionError;
use symphonia::core::codecs::audio::AudioCodecId;
use symphonia::core::codecs::audio::well_known::CODEC_ID_OPUS;

use super::{Deadline, Native, codec_name, decode, downmix_into, is_mpeg_layer_1_or_2, skip_id3v2};
use crate::decode::DecodeLimits;
use crate::error::AudioError;

const MP3: &[u8] = include_bytes!("../tests/fixtures/tone_44k_stereo.mp3");
const OPUS: &[u8] = include_bytes!("../tests/fixtures/tone_48k_opus.ogg");

fn limits(secs: u32) -> DecodeLimits {
    DecodeLimits::new(NonZeroU32::new(secs).expect("non-zero"))
}

#[test]
fn downmix_averages_the_channels() {
    let mut mono = vec![9.0];
    downmix_into(&[vec![1.0, 0.0], vec![0.0, -1.0]], &mut mono);
    assert_eq!(mono, [9.0, 0.5, -0.5]);
    downmix_into(&[vec![0.25]], &mut mono);
    assert_eq!(mono, [9.0, 0.5, -0.5, 0.25]);
}

#[test]
fn layer_one_and_two_headers_are_told_from_layer_three_and_adts() {
    assert!(!is_mpeg_layer_1_or_2(&[0xFF, 0xFB, 0x90])); // MPEG-1 Layer III
    assert!(is_mpeg_layer_1_or_2(&[0xFF, 0xFD, 0x90])); // MPEG-1 Layer II
    assert!(is_mpeg_layer_1_or_2(&[0xFF, 0xFF, 0x90])); // MPEG-1 Layer I
    assert!(!is_mpeg_layer_1_or_2(&[0xFF, 0xF1, 0x50])); // ADTS AAC
    assert!(!is_mpeg_layer_1_or_2(b"ID3\x04"));
}

#[test]
fn id3v2_tags_are_skipped_by_their_declared_size() {
    // Two tags back to back, the second with a footer.
    let mut bytes = b"ID3\x04\x00\x00\x00\x00\x01\x00".to_vec(); // body 128
    bytes.resize(10 + 128, 0);
    bytes.extend_from_slice(b"ID3\x04\x00\x10\x00\x00\x00\x02xx0123456789");
    bytes.extend_from_slice(&[0xFF, 0xFB]);
    assert_eq!(skip_id3v2(&bytes), &[0xFF, 0xFB]);
    // A size byte with its top bit set is not syncsafe: left for the probe.
    let bad = b"ID3\x04\x00\x00\x80\x00\x00\x00rest";
    assert_eq!(skip_id3v2(bad), bad);
    // A size past the end of the buffer: left for the probe.
    let short = b"ID3\x04\x00\x00\x00\x00\x00\x7Frest";
    assert_eq!(skip_id3v2(short), short);
    assert_eq!(skip_id3v2(b"fLaC"), b"fLaC");
}

#[test]
fn an_unknown_codec_is_named_by_its_id() {
    assert_eq!(codec_name(CODEC_ID_OPUS), "opus");
    assert!(codec_name(AudioCodecId::default()).starts_with("codec id "));
}

#[test]
fn a_passed_deadline_stops_the_decode() {
    let err = decode(MP3, "mp3", limits(60), Deadline::at(Instant::now(), 7))
        .expect_err("the deadline has passed");
    assert!(
        matches!(
            err,
            AudioError::Decoder(VisionError::DecoderTimeout { secs: 7 })
        ),
        "{err}"
    );
}

#[test]
fn a_budget_past_what_an_instant_holds_never_fires() {
    let outcome = decode(MP3, "mp3", limits(60), Deadline::after(u64::MAX)).expect("decodes");
    assert!(matches!(outcome, Native::Decoded(_)), "{outcome:?}");
}

#[test]
fn opus_is_handed_on_with_its_codec_named() {
    let outcome = decode(OPUS, "ogg", limits(60), Deadline::after(20)).expect("not an error");
    assert!(
        matches!(&outcome, Native::Unsupported { codec: Some(c) } if c == "opus"),
        "{outcome:?}"
    );
}
