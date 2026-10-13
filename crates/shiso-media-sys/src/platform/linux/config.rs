//! What a codec configuration record says about the stream before any sample is
//! parsed: its chroma format and bit depth, and, for AV1, the OBUs that belong in
//! front of the first sample.
//!
//! The decode chain only falls through to the next decoder on an `Unsupported`
//! from `open`, and the parser reports the stream's real format only once the
//! first sample is fed. So `open` reads the format from the record and checks it
//! against the decoder's capabilities up front; the sequence callback checks the
//! real format again. A record this module cannot read is not an error here - it
//! yields no hint, and the sequence callback has the last word.

use super::ffi::{CHROMA_420, CHROMA_422, CHROMA_444, CHROMA_MONOCHROME};
use crate::HwCodec;

/// A stream's sample format as its configuration record states it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct StreamFormat {
    /// A `cudaVideoChromaFormat` value.
    pub(super) chroma: i32,
    /// Bits per luma sample.
    pub(super) bit_depth: u32,
}

impl StreamFormat {
    /// What a record with no format fields implies: 8-bit 4:2:0, which every
    /// profile without a high-depth or high-chroma extension is.
    pub(super) const BASELINE: Self = Self {
        chroma: CHROMA_420,
        bit_depth: 8,
    };
}

/// Reads the chroma format and bit depth from `config`, or `None` when the record
/// does not say (a Matroska VP9 `CodecPrivate`, an `avcC` without the
/// high-profile trailer) or cannot be read.
pub(super) fn stream_format(codec: HwCodec, config: &[u8]) -> Option<StreamFormat> {
    match codec {
        HwCodec::H264 => avcc_format(config),
        HwCodec::Hevc => hvcc_format(config),
        HwCodec::Vp9 => vpcc_format(config),
        HwCodec::Av1 => av1c_format(config),
    }
}

/// `avcC`: Baseline, Main and Extended (profiles 66, 77, 88) are always 8-bit
/// 4:2:0. Every other profile may carry a trailer after the PPS list with
/// `chroma_format_idc` and `bit_depth_luma_minus8`; muxers write it for all of
/// them, though the record's own syntax only names 100, 110, 122 and 144.
fn avcc_format(c: &[u8]) -> Option<StreamFormat> {
    let profile = *c.get(1)?;
    let mut at = 5usize;
    let sps = *c.get(at)? & 0b1_1111;
    at += 1;
    for _ in 0..sps {
        at = skip_u16_prefixed(c, at)?;
    }
    let pps = *c.get(at)?;
    at += 1;
    for _ in 0..pps {
        at = skip_u16_prefixed(c, at)?;
    }
    if matches!(profile, 66 | 77 | 88) {
        return Some(StreamFormat::BASELINE);
    }
    let Some(&chroma) = c.get(at) else {
        // The trailer is optional in records written by older muxers.
        return None;
    };
    let depth = *c.get(at + 1)?;
    Some(StreamFormat {
        chroma: i32::from(chroma & 0b11),
        bit_depth: u32::from(depth & 0b111) + 8,
    })
}

/// `hvcC`: byte 16 is `chroma_format_idc` in its low two bits, byte 17
/// `bit_depth_luma_minus8` in its low three.
fn hvcc_format(c: &[u8]) -> Option<StreamFormat> {
    let chroma = *c.get(16)? & 0b11;
    let depth = *c.get(17)? & 0b111;
    Some(StreamFormat {
        chroma: i32::from(chroma),
        bit_depth: u32::from(depth) + 8,
    })
}

/// `vpcC` (a version-1 full box payload: version, three flag bytes, profile,
/// level, then `bitDepth:4 chromaSubsampling:3 videoFullRangeFlag:1`). A
/// Matroska VP9 `CodecPrivate` is a different, ID-tagged list whose second byte
/// is a length, never zero, so the zero flag bytes tell them apart.
fn vpcc_format(c: &[u8]) -> Option<StreamFormat> {
    if c.len() < 8 || c[0] != 1 || c[1..4] != [0, 0, 0] {
        return None;
    }
    let packed = c[6];
    let chroma = match (packed >> 1) & 0b111 {
        0 | 1 => CHROMA_420,
        2 => CHROMA_422,
        _ => CHROMA_444,
    };
    Some(StreamFormat {
        chroma,
        bit_depth: u32::from(packed >> 4),
    })
}

/// `av1C`: byte 2 holds `high_bitdepth`, `twelve_bit`, `monochrome`,
/// `chroma_subsampling_x` and `_y` from bit 6 down.
fn av1c_format(c: &[u8]) -> Option<StreamFormat> {
    if *c.first()? & 0x80 == 0 {
        return None;
    }
    let b = *c.get(2)?;
    let bit = |n: u8| (b >> n) & 1 == 1;
    let bit_depth = match (bit(6), bit(5)) {
        (false, _) => 8,
        (true, false) => 10,
        (true, true) => 12,
    };
    let chroma = match (bit(4), bit(3), bit(2)) {
        (true, _, _) => CHROMA_MONOCHROME,
        (false, true, true) => CHROMA_420,
        (false, true, false) => CHROMA_422,
        (false, false, _) => CHROMA_444,
    };
    Some(StreamFormat { chroma, bit_depth })
}

/// The index just past a `u16`-length-prefixed field starting at `at`.
fn skip_u16_prefixed(c: &[u8], at: usize) -> Option<usize> {
    let len = c.get(at..at.checked_add(2)?)?;
    let len = usize::from(u16::from_be_bytes([len[0], len[1]]));
    let end = at.checked_add(2)?.checked_add(len)?;
    (end <= c.len()).then_some(end)
}

/// The OBUs an AV1 configuration record carries for the bitstream (normally the
/// sequence header), to go in front of the first sample.
///
/// An `av1C` record is a four-byte header (its first byte has the marker bit
/// set) followed by `configOBUs`; a configuration without the marker is taken to
/// be bare OBUs already.
pub(super) fn av1_config_obus(config: &[u8]) -> &[u8] {
    match config.first() {
        Some(b) if b & 0x80 != 0 => config.get(4..).unwrap_or(&[]),
        _ => config,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avcc_main_profile_is_baseline_format() {
        let c = [1, 77, 0, 30, 0xff, 0xe1, 0, 1, 0x67, 1, 0, 1, 0x68];
        assert_eq!(
            stream_format(HwCodec::H264, &c),
            Some(StreamFormat::BASELINE)
        );
    }

    #[test]
    fn avcc_high10_trailer_is_read() {
        let mut c = vec![1, 110, 0, 30, 0xff, 0xe1, 0, 1, 0x67, 1, 0, 1, 0x68];
        c.extend_from_slice(&[0xfc | 1, 0xf8 | 2, 0xf8 | 2, 0]);
        assert_eq!(
            stream_format(HwCodec::H264, &c),
            Some(StreamFormat {
                chroma: CHROMA_420,
                bit_depth: 10
            })
        );
        // High 4:4:4 Predictive with a 4:4:4 trailer.
        c[1] = 244;
        c[13] = 0xfc | 3;
        assert_eq!(
            stream_format(HwCodec::H264, &c).map(|f| f.chroma),
            Some(CHROMA_444)
        );
    }

    #[test]
    fn avcc_high_profile_without_trailer_gives_no_hint() {
        let c = [1, 100, 0, 30, 0xff, 0xe1, 0, 1, 0x67, 1, 0, 1, 0x68];
        assert_eq!(stream_format(HwCodec::H264, &c), None);
    }

    #[test]
    fn avcc_with_an_overrunning_sps_gives_no_hint() {
        let c = [1, 100, 0, 30, 0xff, 0xe1, 0, 9, 0x67];
        assert_eq!(stream_format(HwCodec::H264, &c), None);
    }

    #[test]
    fn hvcc_main10_is_read() {
        let mut c = vec![0u8; 23];
        c[0] = 1;
        c[16] = 0xfc | 1;
        c[17] = 0xf8 | 2;
        assert_eq!(
            stream_format(HwCodec::Hevc, &c),
            Some(StreamFormat {
                chroma: CHROMA_420,
                bit_depth: 10
            })
        );
        c[16] = 0xfc | 3;
        assert_eq!(
            stream_format(HwCodec::Hevc, &c).map(|f| f.chroma),
            Some(CHROMA_444)
        );
        assert_eq!(stream_format(HwCodec::Hevc, &c[..10]), None);
    }

    #[test]
    fn vpcc_is_read_and_matroska_private_is_not() {
        let vpcc = [1, 0, 0, 0, 2, 31, (10 << 4) | (1 << 1), 1, 1, 1, 0, 0];
        assert_eq!(
            stream_format(HwCodec::Vp9, &vpcc),
            Some(StreamFormat {
                chroma: CHROMA_420,
                bit_depth: 10
            })
        );
        let matroska = [1, 1, 0, 2, 1, 31, 3, 1, 8, 4, 1, 1];
        assert_eq!(stream_format(HwCodec::Vp9, &matroska), None);
        assert_eq!(stream_format(HwCodec::Vp9, &[]), None);
    }

    #[test]
    fn av1c_bits_are_read() {
        // 8-bit 4:2:0: chroma_subsampling_x and _y set.
        let c = [0x81, 0x08, 0b0000_1100, 0, 0x0a, 0x0b];
        assert_eq!(
            stream_format(HwCodec::Av1, &c),
            Some(StreamFormat::BASELINE)
        );
        let ten = [0x81, 0x08, 0b0100_1100, 0];
        assert_eq!(
            stream_format(HwCodec::Av1, &ten).map(|f| f.bit_depth),
            Some(10)
        );
        let twelve_444 = [0x81, 0x48, 0b0110_0000, 0];
        assert_eq!(
            stream_format(HwCodec::Av1, &twelve_444),
            Some(StreamFormat {
                chroma: CHROMA_444,
                bit_depth: 12
            })
        );
        let mono = [0x81, 0x08, 0b0001_1100, 0];
        assert_eq!(
            stream_format(HwCodec::Av1, &mono).map(|f| f.chroma),
            Some(CHROMA_MONOCHROME)
        );
        assert_eq!(stream_format(HwCodec::Av1, &[0x0a, 0x0b]), None);
    }

    #[test]
    fn av1_config_obus_skip_the_record_header() {
        assert_eq!(
            av1_config_obus(&[0x81, 0, 0x0c, 0, 0x0a, 0x0b]),
            &[0x0a, 0x0b]
        );
        assert!(av1_config_obus(&[0x81, 0, 0x0c]).is_empty());
        assert_eq!(av1_config_obus(&[0x0a, 0x0b]), &[0x0a, 0x0b]);
        assert!(av1_config_obus(&[]).is_empty());
    }
}
