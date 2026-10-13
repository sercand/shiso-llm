//! Length-prefixed H.264/HEVC to Annex-B, for decoders that parse a byte
//! stream.
//!
//! MP4 and Matroska store each NAL unit behind a big-endian length and keep
//! the parameter sets in the configuration record. A byte-stream parser (the
//! NVDEC one) wants the other layout: every NAL unit behind a start code, and
//! the parameter sets in-band ahead of the first picture. `VideoToolbox` takes
//! the stored layout as is, and VP9 and AV1 samples have no NAL units, so this
//! is used by the Linux path only.

use super::{CodecConfig, ConfigFormat, DemuxError};

/// The four-byte start code. Four rather than three because a parameter set
/// or the first NAL unit of an access unit needs the long form, and one form
/// throughout keeps the output easy to check.
const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// Appends `sample`'s NAL units to `out`, each behind a start code.
///
/// `nal_len_size` is the prefix width from the configuration record
/// ([`CodecConfig::nal_length_size`]). An empty NAL unit is dropped: it
/// carries nothing, and a start code followed directly by another start code
/// is something a byte-stream parser may misread.
///
/// # Errors
///
/// [`DemuxError::Malformed`] when `nal_len_size` is not 1, 2 or 4, or when a
/// length runs past the end of the sample (including a sample that ends inside
/// a length prefix). On error `out` may hold the NAL units before the fault.
pub fn avcc_to_annexb(
    sample: &[u8],
    nal_len_size: usize,
    out: &mut Vec<u8>,
) -> Result<(), DemuxError> {
    if !matches!(nal_len_size, 1 | 2 | 4) {
        return Err(DemuxError::malformed(format!(
            "NAL length size {nal_len_size} (must be 1, 2 or 4)"
        )));
    }
    let mut rest = sample;
    while !rest.is_empty() {
        let (prefix, after) = rest
            .split_at_checked(nal_len_size)
            .ok_or_else(|| DemuxError::malformed("sample ends inside a NAL length prefix"))?;
        let len = prefix
            .iter()
            .fold(0usize, |acc, &b| (acc << 8) | usize::from(b));
        let (nal, after) = after
            .split_at_checked(len)
            .ok_or_else(|| DemuxError::malformed("a NAL unit runs past the end of its sample"))?;
        if !nal.is_empty() {
            out.extend_from_slice(&START_CODE);
            out.extend_from_slice(nal);
        }
        rest = after;
    }
    Ok(())
}

/// The parameter sets of an `avcC` or `hvcC` record as an Annex-B prefix:
/// SPS then PPS for H.264; every array the record carries (VPS, SPS, PPS, and
/// any SEI) in stored order for HEVC.
///
/// # Errors
///
/// [`DemuxError::Unsupported`] for a VP9 or AV1 record, which has no NAL
/// parameter sets; [`DemuxError::Malformed`] when the record is truncated or a
/// parameter set's length runs past it.
pub fn parameter_sets_annexb(config: &CodecConfig) -> Result<Vec<u8>, DemuxError> {
    let record = config.record();
    let mut out = Vec::new();
    match config.format() {
        ConfigFormat::Avcc => {
            // version, profile, compatibility, level, length size; then the
            // SPS count in the low five bits.
            let mut rest = record
                .get(5..)
                .ok_or_else(|| DemuxError::malformed("avcC truncated"))?;
            let sps = take_u8(&mut rest, "avcC")? & 0x1f;
            copy_sets(&mut rest, usize::from(sps), &mut out, "avcC")?;
            let pps = take_u8(&mut rest, "avcC")?;
            copy_sets(&mut rest, usize::from(pps), &mut out, "avcC")?;
            // Any trailing bytes are the high-profile chroma and bit-depth
            // extension, which repeats what the SPS already says.
        }
        ConfigFormat::Hvcc => {
            // Twenty-two bytes of profile, tier, level and format fields, then
            // the number of NAL unit arrays.
            let mut rest = record
                .get(22..)
                .ok_or_else(|| DemuxError::malformed("hvcC truncated"))?;
            let arrays = take_u8(&mut rest, "hvcC")?;
            for _ in 0..arrays {
                // completeness flag and NAL unit type: not needed to copy.
                take_u8(&mut rest, "hvcC")?;
                let count = take_u16(&mut rest, "hvcC")?;
                copy_sets(&mut rest, usize::from(count), &mut out, "hvcC")?;
            }
        }
        other => {
            return Err(DemuxError::unsupported(format!(
                "{other:?} carries no NAL parameter sets"
            )));
        }
    }
    Ok(out)
}

/// Copies `count` sets, each behind a 16-bit length, from `rest` to `out`.
fn copy_sets(
    rest: &mut &[u8],
    count: usize,
    out: &mut Vec<u8>,
    what: &str,
) -> Result<(), DemuxError> {
    for _ in 0..count {
        let len = usize::from(take_u16(rest, what)?);
        let (set, after) = rest.split_at_checked(len).ok_or_else(|| {
            DemuxError::malformed(format!("{what} parameter set runs past the record"))
        })?;
        out.extend_from_slice(&START_CODE);
        out.extend_from_slice(set);
        *rest = after;
    }
    Ok(())
}

fn take_u8(rest: &mut &[u8], what: &str) -> Result<u8, DemuxError> {
    let (&byte, after) = rest
        .split_first()
        .ok_or_else(|| DemuxError::malformed(format!("{what} truncated")))?;
    *rest = after;
    Ok(byte)
}

fn take_u16(rest: &mut &[u8], what: &str) -> Result<u16, DemuxError> {
    let (head, after) = rest
        .split_first_chunk::<2>()
        .ok_or_else(|| DemuxError::malformed(format!("{what} truncated")))?;
    *rest = after;
    Ok(u16::from_be_bytes(*head))
}

#[cfg(test)]
#[path = "annexb_tests.rs"]
mod tests;
