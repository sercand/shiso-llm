//! The checks and copies around a `VideoToolbox` session that need no framework:
//! the NAL-unit length size a configuration record declares, the walk over a
//! length-prefixed sample, and the BGRA-to-RGB row copy.
//!
//! They are plain safe functions so they can be tested without a device, and so
//! the one thing a hostile upload controls before any OS code sees it - the
//! length prefixes - is bounds-checked here rather than trusted to the decoder.

use crate::{HwCodec, HwError};

/// How the samples of a stream are framed, as its configuration record says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Framing {
    /// Length-prefixed NAL units (H.264, HEVC), with a prefix of this many
    /// bytes.
    LengthPrefixed(usize),
    /// Whatever the codec's own syntax is (VP9 frames, AV1 OBUs); the decoder
    /// parses it and nothing here can check more than that it is not empty.
    Opaque,
}

/// Reads the sample framing from a configuration record, refusing a record
/// too short to hold the field or one that declares a length size the
/// container format does not allow (1, 2 and 4 are the only legal sizes).
pub(super) fn framing(codec: HwCodec, config: &[u8]) -> Result<Framing, HwError> {
    // `lengthSizeMinusOne` sits in the low two bits of byte 4 of an `avcC`
    // (ISO/IEC 14496-15 5.3.3.1) and of byte 21 of an `hvcC` (8.3.3.1).
    let (name, at, version) = match codec {
        HwCodec::H264 => ("avcC", 4, 0),
        HwCodec::Hevc => ("hvcC", 21, 0),
        HwCodec::Vp9 | HwCodec::Av1 => return Ok(Framing::Opaque),
    };
    let malformed = |why: String| HwError::Failed {
        detail: format!("malformed {name} configuration record: {why}"),
    };
    // Both records start with `configurationVersion = 1`.
    match config.get(version) {
        Some(1) => {}
        Some(other) => return Err(malformed(format!("configuration version {other}, not 1"))),
        None => return Err(malformed("empty".into())),
    }
    let byte = config
        .get(at)
        .ok_or_else(|| malformed(format!("{} bytes is too short", config.len())))?;
    match (byte & 0b11) + 1 {
        size @ (1 | 2 | 4) => Ok(Framing::LengthPrefixed(usize::from(size))),
        size => Err(malformed(format!("NAL length size {size}"))),
    }
}

/// Checks that a sample is one or more complete units under `framing`: every
/// length prefix is followed by at least that many bytes and nothing is left
/// over. Returns how many NAL units the sample holds.
pub(super) fn check_sample(framing: Framing, sample: &[u8]) -> Result<usize, HwError> {
    if sample.is_empty() {
        return Err(HwError::Failed {
            detail: "an empty sample".into(),
        });
    }
    let Framing::LengthPrefixed(size) = framing else {
        return Ok(1);
    };
    let mut rest = sample;
    let mut units = 0usize;
    while !rest.is_empty() {
        let (prefix, tail) = rest.split_at_checked(size).ok_or_else(|| HwError::Failed {
            detail: format!(
                "a sample ends inside a {size}-byte NAL length prefix after {units} units"
            ),
        })?;
        let len = prefix
            .iter()
            .fold(0usize, |acc, &b| (acc << 8) | usize::from(b));
        if len == 0 {
            return Err(HwError::Failed {
                detail: format!("NAL unit {units} has length 0"),
            });
        }
        rest = tail.get(len..).ok_or_else(|| HwError::Failed {
            detail: format!(
                "NAL unit {units} declares {len} bytes but only {} remain",
                tail.len()
            ),
        })?;
        units += 1;
    }
    Ok(units)
}

/// Copies a locked BGRA pixel buffer into packed RGB8.
///
/// `stride` is the buffer's bytes per row, which is at least `width * 4` and
/// usually more (rows are padded for alignment); the padding is skipped. The
/// last row need only be `width * 4` long, so `src` may stop there.
pub(super) fn bgra_to_rgb(
    src: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<Vec<u8>, HwError> {
    let row_bytes = width
        .checked_mul(4)
        .ok_or_else(|| too_large(width, height))?;
    if stride < row_bytes {
        return Err(HwError::Failed {
            detail: format!("a {width}-pixel BGRA row needs {row_bytes} bytes, stride is {stride}"),
        });
    }
    let out_len = width
        .checked_mul(height)
        .and_then(|px| px.checked_mul(3))
        .ok_or_else(|| too_large(width, height))?;
    let mut rgb = Vec::with_capacity(out_len);
    for row in 0..height {
        let start = row
            .checked_mul(stride)
            .ok_or_else(|| too_large(width, height))?;
        let line = start
            .checked_add(row_bytes)
            .and_then(|end| src.get(start..end))
            .ok_or_else(|| HwError::Failed {
                detail: format!(
                    "row {row} of a {width}x{height} BGRA buffer (stride {stride}) runs past its {} bytes",
                    src.len()
                ),
            })?;
        for px in line.chunks_exact(4) {
            if let [b, g, r, _a] = *px {
                rgb.extend_from_slice(&[r, g, b]);
            }
        }
    }
    Ok(rgb)
}

fn too_large(width: usize, height: usize) -> HwError {
    HwError::Failed {
        detail: format!("a {width}x{height} frame is too large to address"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn avcc(length_size_minus_one: u8) -> Vec<u8> {
        vec![1, 0x64, 0, 0x0a, 0xfc | length_size_minus_one, 0xe0, 0]
    }

    #[test]
    fn framing_reads_the_length_size() {
        assert_eq!(
            framing(HwCodec::H264, &avcc(3)),
            Ok(Framing::LengthPrefixed(4))
        );
        assert_eq!(
            framing(HwCodec::H264, &avcc(1)),
            Ok(Framing::LengthPrefixed(2))
        );
        assert_eq!(
            framing(HwCodec::H264, &avcc(0)),
            Ok(Framing::LengthPrefixed(1))
        );
        let mut hvcc = vec![0u8; 23];
        hvcc[0] = 1;
        hvcc[21] = 0x0f;
        assert_eq!(
            framing(HwCodec::Hevc, &hvcc),
            Ok(Framing::LengthPrefixed(4))
        );
        assert_eq!(framing(HwCodec::Vp9, &[]), Ok(Framing::Opaque));
        assert_eq!(framing(HwCodec::Av1, &[]), Ok(Framing::Opaque));
    }

    #[test]
    fn framing_refuses_bad_records() {
        for bad in [
            &[][..],
            &[1, 2, 3][..],
            &avcc(2)[..],
            &[0, 0x64, 0, 0x0a, 0xff][..],
        ] {
            assert!(
                matches!(framing(HwCodec::H264, bad), Err(HwError::Failed { .. })),
                "{bad:?}"
            );
        }
        assert!(matches!(
            framing(HwCodec::Hevc, &avcc(3)),
            Err(HwError::Failed { .. })
        ));
    }

    #[test]
    fn check_sample_walks_length_prefixes() {
        let four = Framing::LengthPrefixed(4);
        assert_eq!(
            check_sample(four, &[0, 0, 0, 2, 9, 9, 0, 0, 0, 1, 7]),
            Ok(2)
        );
        assert_eq!(check_sample(Framing::LengthPrefixed(2), &[0, 1, 5]), Ok(1));
        assert_eq!(check_sample(Framing::Opaque, &[1, 2, 3]), Ok(1));
    }

    #[test]
    fn check_sample_refuses_overruns_and_empties() {
        let four = Framing::LengthPrefixed(4);
        for bad in [
            &[][..],
            &[0, 0, 0][..],                // ends inside a prefix
            &[0, 0, 0, 5, 1, 2][..],       // declares more than is there
            &[0, 0, 0, 0][..],             // zero-length unit
            &[0, 0, 0, 1, 1, 0, 0][..],    // trailing partial prefix
            &[0xff, 0xff, 0xff, 0xff][..], // hostile length
        ] {
            assert!(
                matches!(check_sample(four, bad), Err(HwError::Failed { .. })),
                "{bad:?}"
            );
        }
        assert!(matches!(
            check_sample(Framing::Opaque, &[]),
            Err(HwError::Failed { .. })
        ));
    }

    #[test]
    fn bgra_to_rgb_skips_row_padding() {
        // 2x2 with a 12-byte stride: 8 bytes of pixels and 4 of padding per row;
        // the last row stops at its pixels.
        let src = [
            1, 2, 3, 255, 4, 5, 6, 255, 0xee, 0xee, 0xee, 0xee, //
            7, 8, 9, 255, 10, 11, 12, 255,
        ];
        let rgb = bgra_to_rgb(&src, 2, 2, 12).expect("fits");
        assert_eq!(rgb, [3, 2, 1, 6, 5, 4, 9, 8, 7, 12, 11, 10]);
    }

    #[test]
    fn bgra_to_rgb_refuses_short_buffers_and_strides() {
        let src = [0u8; 16];
        assert!(matches!(
            bgra_to_rgb(&src, 2, 2, 7),
            Err(HwError::Failed { .. })
        ));
        assert!(matches!(
            bgra_to_rgb(&src, 2, 3, 8),
            Err(HwError::Failed { .. })
        ));
        assert!(matches!(
            bgra_to_rgb(&src, usize::MAX, 1, 8),
            Err(HwError::Failed { .. })
        ));
        assert_eq!(bgra_to_rgb(&src, 2, 2, 8).map(|v| v.len()), Ok(12));
    }
}
