//! What the session needs out of a container's codec configuration record,
//! read in safe code: the parameter-set NAL units of an `avcC` or `hvcC`, and
//! the colour description of a `vpcC` or of the sequence header inside an
//! `av1C`.
//!
//! The colour is the reason this exists. `VideoToolbox` converts to BGRA with
//! the matrix the *format description* names and does not read it from the
//! bitstream; with none named it assumed BT.601 at the fixtures' 160x96,
//! which turns BT.709 content into a plausible picture with the wrong colours
//! (measured: about 29 dB against a BT.709 reference where 50 is right). The
//! H.264 and HEVC constructors in `CoreMedia` read the VUI themselves; for VP9
//! and AV1 the record is read here and the result attached as extensions.

use crate::{HwCodec, HwError};

/// The colour description of a stream, as ITU-T H.273 code points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ColourCodes {
    /// `colour_primaries`.
    pub(super) primaries: u8,
    /// `transfer_characteristics`.
    pub(super) transfer: u8,
    /// `matrix_coefficients`.
    pub(super) matrix: u8,
    /// Whether luma spans the whole sample range.
    pub(super) full_range: bool,
}

/// H.273 "unspecified", for every code.
const UNSPECIFIED: u8 = 2;

impl ColourCodes {
    /// Every code "unspecified", limited range.
    pub(super) const UNSPECIFIED: Self = Self {
        primaries: UNSPECIFIED,
        transfer: UNSPECIFIED,
        matrix: UNSPECIFIED,
        full_range: false,
    };
}

fn malformed(codec: HwCodec, why: impl std::fmt::Display) -> HwError {
    let name = match codec {
        HwCodec::H264 => "avcC",
        HwCodec::Hevc => "hvcC",
        HwCodec::Vp9 => "vpcC",
        HwCodec::Av1 => "av1C",
    };
    HwError::Failed {
        detail: format!("malformed {name} configuration record: {why}"),
    }
}

/// A forward-only reader over a byte slice that fails instead of panicking.
struct Bytes<'a> {
    codec: HwCodec,
    rest: &'a [u8],
}

impl<'a> Bytes<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], HwError> {
        let (head, tail) = self
            .rest
            .split_at_checked(n)
            .ok_or_else(|| malformed(self.codec, "truncated"))?;
        self.rest = tail;
        Ok(head)
    }

    fn u8(&mut self) -> Result<u8, HwError> {
        self.take(1).map(|b| b.first().copied().unwrap_or_default())
    }

    fn u16(&mut self) -> Result<usize, HwError> {
        let b = self.take(2)?;
        Ok(b.iter().fold(0usize, |acc, &x| (acc << 8) | usize::from(x)))
    }

    /// A 16-bit length followed by that many bytes, refused when empty.
    fn nal(&mut self) -> Result<&'a [u8], HwError> {
        let len = self.u16()?;
        if len == 0 {
            return Err(malformed(self.codec, "an empty parameter set"));
        }
        self.take(len)
    }
}

/// The parameter-set NAL units (SPS, PPS, and for HEVC VPS and SEI) of an
/// `avcC` (ISO/IEC 14496-15 5.3.3.1) or `hvcC` (8.3.3.1), in record order.
pub(super) fn parameter_sets(codec: HwCodec, config: &[u8]) -> Result<Vec<&[u8]>, HwError> {
    let mut r = Bytes {
        codec,
        rest: config,
    };
    let mut sets = Vec::new();
    match codec {
        HwCodec::H264 => {
            r.take(5)?;
            let sps = r.u8()? & 0x1f;
            for _ in 0..sps {
                sets.push(r.nal()?);
            }
            let pps = r.u8()?;
            for _ in 0..pps {
                sets.push(r.nal()?);
            }
            // Anything after (the High-profile chroma and bit-depth fields,
            // SPS extensions) repeats what the SPS says; the SPS is what the
            // decoder reads.
        }
        HwCodec::Hevc => {
            r.take(22)?;
            let arrays = r.u8()?;
            for _ in 0..arrays {
                r.u8()?;
                let count = r.u16()?;
                for _ in 0..count {
                    sets.push(r.nal()?);
                }
            }
        }
        HwCodec::Vp9 | HwCodec::Av1 => {
            return Err(malformed(codec, "has no parameter sets"));
        }
    }
    if sets.is_empty() {
        // Legal: an `avc3`/`hev1` track sends its parameter sets in-band and
        // may leave the record's lists empty. The format description needs
        // them up front, so this decoder cannot take the stream - but the
        // stream is not malformed, and the next decoder may.
        return Err(HwError::Unsupported {
            codec,
            reason: "the configuration record carries no parameter sets (they travel in-band), \
                     and the format description needs them up front"
                .into(),
        });
    }
    Ok(sets)
}

/// The colour description a `vpcC` (VP Codec ISO Media File Format Binding,
/// version 1) carries.
pub(super) fn vpcc_colour(config: &[u8]) -> Result<ColourCodes, HwError> {
    let mut r = Bytes {
        codec: HwCodec::Vp9,
        rest: config,
    };
    let version = r.u8()?;
    if version != 1 {
        return Err(malformed(HwCodec::Vp9, format!("version {version}, not 1")));
    }
    // flags(3) profile(1) level(1)
    r.take(5)?;
    let packed = r.u8()?;
    Ok(ColourCodes {
        primaries: r.u8()?,
        transfer: r.u8()?,
        matrix: r.u8()?,
        full_range: packed & 1 == 1,
    })
}

/// What an AV1 sequence header says that the format description needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Av1Sequence {
    /// The colour description.
    pub(super) colour: ColourCodes,
    /// `max_frame_width_minus_1 + 1` and `max_frame_height_minus_1 + 1`.
    pub(super) max_size: (u32, u32),
}

/// The sequence header OBU an `av1C` carries in its `configOBUs`, or `None`
/// when the record carries none (it may not; the first temporal unit then
/// repeats it, and the decoder reads it from there).
pub(super) fn av1c_sequence(config: &[u8]) -> Result<Option<Av1Sequence>, HwError> {
    let mut r = Bytes {
        codec: HwCodec::Av1,
        rest: config,
    };
    let marker = r.u8()?;
    if marker != 0x81 {
        return Err(malformed(
            HwCodec::Av1,
            format!("marker and version byte {marker:#04x}, not 0x81"),
        ));
    }
    r.take(3)?;
    while !r.rest.is_empty() {
        let header = r.u8()?;
        let kind = (header >> 3) & 0x0f;
        if header & 0x04 != 0 {
            r.u8()?;
        }
        if header & 0x02 == 0 {
            return Err(malformed(HwCodec::Av1, "a configOBU without a size field"));
        }
        let size = leb128(&mut r)?;
        let payload = r.take(size)?;
        if kind == 1 {
            return sequence_header(payload).map(Some);
        }
    }
    Ok(None)
}

fn leb128(r: &mut Bytes<'_>) -> Result<usize, HwError> {
    let mut value = 0u64;
    for i in 0..8 {
        let byte = r.u8()?;
        value |= u64::from(byte & 0x7f) << (7 * i);
        if byte & 0x80 == 0 {
            return usize::try_from(value).map_err(|_| malformed(HwCodec::Av1, "an OBU size"));
        }
    }
    Err(malformed(HwCodec::Av1, "an OBU size longer than 8 bytes"))
}

/// A most-significant-bit-first reader for the sequence header.
struct Bits<'a> {
    data: &'a [u8],
    at: usize,
}

impl Bits<'_> {
    fn bit(&mut self) -> Result<u32, HwError> {
        let byte = self
            .data
            .get(self.at / 8)
            .ok_or_else(|| malformed(HwCodec::Av1, "the sequence header is truncated"))?;
        let bit = (byte >> (7 - self.at % 8)) & 1;
        self.at += 1;
        Ok(u32::from(bit))
    }

    fn f(&mut self, n: u32) -> Result<u32, HwError> {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.bit()?;
        }
        Ok(v)
    }

    fn flag(&mut self) -> Result<bool, HwError> {
        self.bit().map(|b| b == 1)
    }

    /// `uvlc()` (AV1 4.10.3), only skipped here.
    fn uvlc(&mut self) -> Result<(), HwError> {
        let mut zeros = 0u32;
        while !self.flag()? {
            zeros += 1;
            if zeros >= 32 {
                return Err(malformed(HwCodec::Av1, "a uvlc longer than 32 bits"));
            }
        }
        self.f(zeros).map(|_| ())
    }
}

/// Walks `sequence_header_obu()` (AV1 5.5.1) as far as `color_config()`.
fn sequence_header(payload: &[u8]) -> Result<Av1Sequence, HwError> {
    let mut b = Bits {
        data: payload,
        at: 0,
    };
    let profile = b.f(3)?;
    b.flag()?; // still_picture
    let reduced = b.flag()?;
    if reduced {
        b.f(5)?; // seq_level_idx[0]
    } else {
        let timing_info = b.flag()?;
        let mut decoder_model_info = false;
        let mut buffer_delay_length = 0;
        if timing_info {
            b.f(32)?; // num_units_in_display_tick
            b.f(32)?; // time_scale
            if b.flag()? {
                b.uvlc()?; // num_ticks_per_picture_minus_1
            }
            decoder_model_info = b.flag()?;
            if decoder_model_info {
                buffer_delay_length = b.f(5)? + 1;
                b.f(32)?; // num_units_in_decoding_tick
                b.f(5)?; // buffer_removal_time_length_minus_1
                b.f(5)?; // frame_presentation_time_length_minus_1
            }
        }
        let initial_display_delay = b.flag()?;
        let operating_points = b.f(5)? + 1;
        for _ in 0..operating_points {
            b.f(12)?; // operating_point_idc
            if b.f(5)? > 7 {
                b.flag()?; // seq_tier
            }
            if decoder_model_info && b.flag()? {
                b.f(buffer_delay_length)?; // decoder_buffer_delay
                b.f(buffer_delay_length)?; // encoder_buffer_delay
                b.flag()?; // low_delay_mode_flag
            }
            if initial_display_delay && b.flag()? {
                b.f(4)?; // initial_display_delay_minus_1
            }
        }
    }
    let width_bits = b.f(4)? + 1;
    let height_bits = b.f(4)? + 1;
    let max_size = (b.f(width_bits)? + 1, b.f(height_bits)? + 1);
    if !reduced && b.flag()? {
        b.f(4)?; // delta_frame_id_length_minus_2
        b.f(3)?; // additional_frame_id_length_minus_1
    }
    b.f(3)?; // use_128x128_superblock, enable_filter_intra, enable_intra_edge_filter
    if !reduced {
        b.f(4)?; // interintra, masked compound, warped motion, dual filter
        let order_hint = b.flag()?;
        if order_hint {
            b.f(2)?; // enable_jnt_comp, enable_ref_frame_mvs
        }
        let force_screen_content_tools = if b.flag()? { 2 } else { b.f(1)? };
        if force_screen_content_tools > 0 && !b.flag()? {
            b.f(1)?; // seq_force_integer_mv
        }
        if order_hint {
            b.f(3)?; // order_hint_bits_minus_1
        }
    }
    b.f(3)?; // enable_superres, enable_cdef, enable_restoration

    // color_config()
    let high_bitdepth = b.flag()?;
    if profile == 2 && high_bitdepth {
        b.flag()?; // twelve_bit
    }
    let mono_chrome = if profile == 1 { false } else { b.flag()? };
    let mut codes = ColourCodes::UNSPECIFIED;
    if b.flag()? {
        codes.primaries = narrow(b.f(8)?);
        codes.transfer = narrow(b.f(8)?);
        codes.matrix = narrow(b.f(8)?);
    }
    // sRGB in 4:4:4 (primaries 1, transfer 13, matrix 0) implies full range
    // and has no color_range bit; neither does it matter for BGRA output.
    let srgb = codes.primaries == 1 && codes.transfer == 13 && codes.matrix == 0;
    codes.full_range = if !mono_chrome && srgb {
        true
    } else {
        b.flag()?
    };
    Ok(Av1Sequence {
        colour: codes,
        max_size,
    })
}

fn narrow(v: u32) -> u8 {
    u8::try_from(v).unwrap_or(UNSPECIFIED)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixture's `avcC` (libx264 High profile, 160x96).
    const AVCC: &[u8] = include_bytes!("../../../tests/fixtures/h264-high-bframes.avcC");
    const HVCC: &[u8] = include_bytes!("../../../tests/fixtures/hevc-main.hvcC");
    const VPCC: &[u8] = include_bytes!("../../../tests/fixtures/vp9-profile0.vpcC");
    const AV1C: &[u8] = include_bytes!("../../../tests/fixtures/av1-main.av1C");

    #[test]
    fn avcc_parameter_sets() {
        let sets = parameter_sets(HwCodec::H264, AVCC).expect("parses");
        assert_eq!(sets.len(), 2);
        assert_eq!(sets[0][0] & 0x1f, 7, "SPS first");
        assert_eq!(sets[1][0] & 0x1f, 8, "then PPS");
    }

    #[test]
    fn hvcc_parameter_sets() {
        let sets = parameter_sets(HwCodec::Hevc, HVCC).expect("parses");
        let kinds: Vec<u8> = sets.iter().map(|s| (s[0] >> 1) & 0x3f).collect();
        assert!(kinds.starts_with(&[32, 33, 34]), "{kinds:?}");
    }

    #[test]
    fn records_without_parameter_sets_are_unsupported() {
        let avcc = [&AVCC[..5], &[0xe0, 0]].concat();
        let hvcc = [&HVCC[..22], &[0]].concat();
        for (codec, record) in [(HwCodec::H264, avcc), (HwCodec::Hevc, hvcc)] {
            let r = parameter_sets(codec, &record);
            assert!(
                matches!(r, Err(HwError::Unsupported { .. })),
                "{codec}: {r:?}"
            );
        }
    }

    #[test]
    fn truncated_records_fail() {
        // The last four bytes (High-profile chroma and bit-depth fields) are
        // optional; everything before them is not.
        for n in 0..AVCC.len() - 4 {
            let r = parameter_sets(HwCodec::H264, &AVCC[..n]);
            assert!(matches!(r, Err(HwError::Failed { .. })), "{n}: {r:?}");
        }
        for n in [0, 10, 22, 23, 30, 100] {
            let r = parameter_sets(HwCodec::Hevc, &HVCC[..n]);
            assert!(matches!(r, Err(HwError::Failed { .. })), "{n}: {r:?}");
        }
        assert!(vpcc_colour(&VPCC[..8]).is_err());
        assert!(av1c_sequence(&AV1C[..10]).is_err());
    }

    #[test]
    fn vpcc_colour_reads_bt709() {
        let c = vpcc_colour(VPCC).expect("parses");
        assert_eq!((c.matrix, c.full_range), (1, false));
    }

    #[test]
    fn av1c_sequence_reads_the_sequence_header() {
        // The encoder signalled the matrix and range only; ffprobe reads the
        // same stream as matrix BT.709, limited range, primaries and transfer
        // unspecified.
        let seq = av1c_sequence(AV1C).expect("parses").expect("has one");
        assert_eq!(
            seq,
            Av1Sequence {
                colour: ColourCodes {
                    primaries: 2,
                    transfer: 2,
                    matrix: 1,
                    full_range: false
                },
                max_size: (160, 96),
            }
        );
        // A record with no configOBUs is allowed and says nothing.
        assert_eq!(av1c_sequence(&AV1C[..4]), Ok(None));
    }
}
