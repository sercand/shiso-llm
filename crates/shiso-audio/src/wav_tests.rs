#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "test inputs are built from known in-range levels"
)]
//! WAV parsing: every encoding the reader accepts, and the shapes of broken or
//! streamed files it must survive.

use super::{is_wav, parse_wav};
use crate::error::AudioError;

/// Builds a RIFF/WAVE file: `fmt` with the given tag, then `data`.
/// `data_len` overrides the declared data length (streamed files write
/// `0xFFFF_FFFF`).
fn wav(
    tag: u16,
    channels: u16,
    rate: u32,
    bits: u16,
    data: &[u8],
    data_len: Option<u32>,
) -> Vec<u8> {
    let block = channels * bits.div_ceil(8);
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&0u32.to_le_bytes()); // the riff size is not trusted
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&tag.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
    out.extend_from_slice(&block.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    // An unrelated chunk with an odd length, to exercise the pad byte.
    out.extend_from_slice(b"LIST");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&[1, 2, 3, 0]);
    out.extend_from_slice(b"data");
    let len = data_len.unwrap_or(u32::try_from(data.len()).expect("small"));
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(data);
    out
}

#[test]
fn pcm16_stereo_downmixes_by_mean() {
    // Left 0.5, right -0.25 -> mean 0.125.
    let l = (0.5f32 * 32768.0) as i16;
    let r = (-0.25f32 * 32768.0) as i16;
    let mut data = Vec::new();
    for _ in 0..4 {
        data.extend_from_slice(&l.to_le_bytes());
        data.extend_from_slice(&r.to_le_bytes());
    }
    let pcm = parse_wav(&wav(1, 2, 22_050, 16, &data, None)).expect("parses");
    assert_eq!(pcm.rate(), 22_050);
    assert_eq!(pcm.samples().len(), 4);
    for s in pcm.samples() {
        assert!((s - 0.125).abs() < 1e-4, "{s}");
    }
}

#[test]
fn pcm24_pcm32_u8_and_float_decode_to_the_same_level() {
    let level = 0.25f32;
    let i24 = (f64::from(level) * 8_388_608.0) as i32;
    let d24: Vec<u8> = i24.to_le_bytes()[..3].to_vec();
    let d32 = ((f64::from(level) * 2_147_483_648.0) as i32)
        .to_le_bytes()
        .to_vec();
    let d8 = vec![(128.0 + f64::from(level) * 128.0) as u8];
    let float32 = level.to_le_bytes().to_vec();
    let float64 = f64::from(level).to_le_bytes().to_vec();
    for (tag, bits, data) in [
        (1, 24, d24),
        (1, 32, d32),
        (1, 8, d8),
        (3, 32, float32),
        (3, 64, float64),
    ] {
        let pcm = parse_wav(&wav(tag, 1, 16_000, bits, &data, None)).expect("parses");
        assert_eq!(pcm.samples().len(), 1, "tag {tag} bits {bits}");
        assert!(
            (pcm.samples()[0] - level).abs() < 1e-2,
            "tag {tag} bits {bits}: {}",
            pcm.samples()[0]
        );
    }
}

#[test]
fn extensible_resolves_through_the_subformat() {
    let mut file = wav(0xFFFE, 1, 16_000, 16, &[0, 64], None);
    // Rewrite fmt to 40 bytes carrying cbSize=22 and a PCM sub-format GUID.
    let fmt_at = 12;
    let mut fmt = Vec::new();
    fmt.extend_from_slice(b"fmt ");
    fmt.extend_from_slice(&40u32.to_le_bytes());
    fmt.extend_from_slice(&0xFFFEu16.to_le_bytes());
    fmt.extend_from_slice(&1u16.to_le_bytes());
    fmt.extend_from_slice(&16_000u32.to_le_bytes());
    fmt.extend_from_slice(&32_000u32.to_le_bytes());
    fmt.extend_from_slice(&2u16.to_le_bytes());
    fmt.extend_from_slice(&16u16.to_le_bytes());
    fmt.extend_from_slice(&22u16.to_le_bytes());
    fmt.extend_from_slice(&16u16.to_le_bytes());
    fmt.extend_from_slice(&4u32.to_le_bytes());
    fmt.extend_from_slice(&[
        1, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71,
    ]);
    file.splice(fmt_at..fmt_at + 24, fmt);
    let pcm = parse_wav(&file).expect("parses");
    assert!((pcm.samples()[0] - 0.5).abs() < 1e-3);
}

#[test]
fn streamed_length_and_overlong_length_take_what_is_present() {
    // Three whole 16-bit frames plus one stray byte.
    let data = [0u8, 64, 0, 64, 0, 64, 7];
    for declared in [0xFFFF_FFFF, 1000] {
        let pcm = parse_wav(&wav(1, 1, 16_000, 16, &data, Some(declared))).expect("parses");
        assert_eq!(pcm.samples().len(), 3, "declared {declared:#x}");
    }
}

#[test]
fn refusals_are_typed() {
    assert!(matches!(
        parse_wav(&wav(0x55, 1, 16_000, 16, &[0, 0], None)),
        Err(AudioError::UnsupportedWavFormat { tag: 0x55, .. })
    ));
    assert!(matches!(
        parse_wav(&wav(1, 1, 16_000, 12, &[0, 0], None)),
        Err(AudioError::UnsupportedWavFormat { bits: 12, .. })
    ));
    assert!(matches!(
        parse_wav(b"RIFF\0\0\0\0WAVE"),
        Err(AudioError::MalformedWav { .. })
    ));
    assert!(matches!(
        parse_wav(&wav(1, 0, 16_000, 16, &[0, 0], None)),
        Err(AudioError::MalformedWav { .. })
    ));
    assert!(matches!(
        parse_wav(&wav(1, 1, 16_000, 16, &[], None)),
        Err(AudioError::Empty)
    ));
}

#[test]
fn sniffing() {
    assert!(is_wav(&wav(1, 1, 16_000, 16, &[0, 0], None)));
    assert!(!is_wav(b"ID3\x04"));
    assert!(!is_wav(b"RIF"));
}

#[test]
fn a_block_align_that_disagrees_with_the_sample_width_is_refused() {
    // 24-bit samples in 32-bit containers without the extensible header: read
    // at a 3-byte stride this would decode to noise, so it is refused instead.
    let mut file = wav(1, 1, 16_000, 24, &[0, 0, 64, 0, 0, 0, 64, 0], None);
    // block_align sits at byte 32 of the file (12 + 8 + 12).
    file[32..34].copy_from_slice(&4u16.to_le_bytes());
    assert!(matches!(
        parse_wav(&file),
        Err(AudioError::MalformedWav { .. })
    ));
}
