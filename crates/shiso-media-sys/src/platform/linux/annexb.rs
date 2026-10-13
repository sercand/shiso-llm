//! Length-prefixed H.264/HEVC samples to Annex-B, in safe code.
//!
//! MP4 and Matroska store H.264 and HEVC access units as NAL units each preceded
//! by a big-endian length of 1 to 4 bytes (the size is in the `avcC`/`hvcC`
//! record), with the parameter sets in that record rather than in the stream.
//! The NVDEC parser reads an elementary stream: NAL units separated by start
//! codes, parameter sets inline. So each sample's length prefixes become start
//! codes, and the record's parameter sets go in front of the first sample.
//!
//! The input is an upload. A length that runs past the end of the sample or the
//! record is an error, never a short read and never an out-of-bounds index.

use crate::HwError;

/// The four-byte start code put before every NAL unit.
const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// What a codec configuration record contributes to the elementary stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ParamSets {
    /// Bytes in each sample's NAL length prefix, 1 to 4.
    pub(super) nal_length_size: usize,
    /// The record's parameter sets (VPS, SPS, PPS as present), in Annex-B form.
    pub(super) prefix: Vec<u8>,
}

/// A cursor over untrusted bytes that refuses to read past the end.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    what: &'static str,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8], what: &'static str) -> Self {
        Self { bytes, at: 0, what }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], HwError> {
        let end = self.at.checked_add(n).ok_or_else(|| self.short())?;
        let out = self.bytes.get(self.at..end).ok_or_else(|| self.short())?;
        self.at = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, HwError> {
        self.take(1).map(|b| b[0])
    }

    fn u16(&mut self) -> Result<usize, HwError> {
        self.take(2)
            .map(|b| usize::from(u16::from_be_bytes([b[0], b[1]])))
    }

    fn skip(&mut self, n: usize) -> Result<(), HwError> {
        self.take(n).map(|_| ())
    }

    fn short(&self) -> HwError {
        HwError::Failed {
            detail: format!("the {} is truncated at byte {}", self.what, self.at),
        }
    }
}

/// Appends one parameter-set NAL unit of `len` bytes read from `reader`.
fn copy_nal(reader: &mut Reader<'_>, out: &mut Vec<u8>) -> Result<(), HwError> {
    let len = reader.u16()?;
    let nal = reader.take(len)?;
    if !nal.is_empty() {
        out.extend_from_slice(&START_CODE);
        out.extend_from_slice(nal);
    }
    Ok(())
}

/// Reads an `AVCDecoderConfigurationRecord` (`avcC`).
///
/// # Errors
///
/// [`HwError::Failed`] for a record that is truncated or not version 1.
pub(super) fn avcc_param_sets(config: &[u8]) -> Result<ParamSets, HwError> {
    let mut r = Reader::new(config, "avcC record");
    let version = r.u8()?;
    if version != 1 {
        return Err(HwError::Failed {
            detail: format!("avcC record version {version}, expected 1"),
        });
    }
    // profile, profile compatibility, level.
    r.skip(3)?;
    let nal_length_size = usize::from(r.u8()? & 0b11) + 1;
    let mut prefix = Vec::new();
    let sps = r.u8()? & 0b1_1111;
    for _ in 0..sps {
        copy_nal(&mut r, &mut prefix)?;
    }
    let pps = r.u8()?;
    for _ in 0..pps {
        copy_nal(&mut r, &mut prefix)?;
    }
    // The high-profile trailer (chroma format, bit depths, SPS extensions) is
    // not needed in the stream: the SPS itself carries those values.
    Ok(ParamSets {
        nal_length_size,
        prefix,
    })
}

/// Reads an `HEVCDecoderConfigurationRecord` (`hvcC`).
///
/// # Errors
///
/// [`HwError::Failed`] for a record that is truncated or not version 1.
pub(super) fn hvcc_param_sets(config: &[u8]) -> Result<ParamSets, HwError> {
    let mut r = Reader::new(config, "hvcC record");
    let version = r.u8()?;
    if version != 1 {
        return Err(HwError::Failed {
            detail: format!("hvcC record version {version}, expected 1"),
        });
    }
    // Profile, tier, level, segmentation, parallelism, chroma, bit depths and
    // frame rate: bytes 1..21.
    r.skip(20)?;
    let nal_length_size = usize::from(r.u8()? & 0b11) + 1;
    let arrays = r.u8()?;
    let mut prefix = Vec::new();
    for _ in 0..arrays {
        // array_completeness, reserved, NAL unit type.
        r.skip(1)?;
        let count = r.u16()?;
        for _ in 0..count {
            copy_nal(&mut r, &mut prefix)?;
        }
    }
    Ok(ParamSets {
        nal_length_size,
        prefix,
    })
}

/// Appends `sample`'s NAL units to `out` with start codes in place of their
/// `nal_length_size`-byte big-endian lengths.
///
/// # Errors
///
/// [`HwError::Failed`] when a length prefix is cut short or a NAL unit's length
/// runs past the end of the sample.
pub(super) fn length_prefixed_to_annexb(
    sample: &[u8],
    nal_length_size: usize,
    out: &mut Vec<u8>,
) -> Result<(), HwError> {
    if !(1..=4).contains(&nal_length_size) {
        return Err(HwError::Failed {
            detail: format!("a NAL length size of {nal_length_size} bytes"),
        });
    }
    let mut r = Reader::new(sample, "sample");
    while r.at < sample.len() {
        let len = r
            .take(nal_length_size)?
            .iter()
            .fold(0usize, |acc, &b| (acc << 8) | usize::from(b));
        let nal = r.take(len).map_err(|_| HwError::Failed {
            detail: format!(
                "a NAL unit of {len} bytes at byte {} runs past the end of a {}-byte sample",
                r.at,
                sample.len()
            ),
        })?;
        if !nal.is_empty() {
            out.extend_from_slice(&START_CODE);
            out.extend_from_slice(nal);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal avcC with one SPS and one PPS and 4-byte lengths.
    fn avcc() -> Vec<u8> {
        let mut c = vec![1, 0x64, 0, 0x1f, 0xff, 0xe1];
        c.extend_from_slice(&[0, 3, 0x67, 0xaa, 0xbb]);
        c.push(1);
        c.extend_from_slice(&[0, 2, 0x68, 0xcc]);
        c
    }

    #[test]
    fn avcc_yields_its_parameter_sets_with_start_codes() {
        let sets = avcc_param_sets(&avcc()).expect("valid avcC");
        assert_eq!(sets.nal_length_size, 4);
        assert_eq!(
            sets.prefix,
            [0, 0, 0, 1, 0x67, 0xaa, 0xbb, 0, 0, 0, 1, 0x68, 0xcc]
        );
    }

    #[test]
    fn avcc_with_a_short_parameter_set_is_refused() {
        let mut c = avcc();
        c.truncate(c.len() - 1);
        assert!(matches!(avcc_param_sets(&c), Err(HwError::Failed { .. })));
        assert!(matches!(
            avcc_param_sets(&[1, 2]),
            Err(HwError::Failed { .. })
        ));
        assert!(matches!(avcc_param_sets(&[]), Err(HwError::Failed { .. })));
    }

    #[test]
    fn avcc_of_the_wrong_version_is_refused() {
        let mut c = avcc();
        c[0] = 0;
        assert!(matches!(avcc_param_sets(&c), Err(HwError::Failed { .. })));
    }

    #[test]
    fn hvcc_yields_vps_sps_pps_in_order() {
        let mut c = vec![1];
        c.extend_from_slice(&[0; 20]);
        c.push(0xf0 | 0b01); // 2-byte lengths
        c.push(3);
        for (ty, body) in [(32u8, &[0x40u8, 1][..]), (33, &[0x42, 1, 2]), (34, &[0x44])] {
            c.push(ty);
            c.extend_from_slice(&[0, 1]);
            let len = u16::try_from(body.len()).expect("small");
            c.extend_from_slice(&len.to_be_bytes());
            c.extend_from_slice(body);
        }
        let sets = hvcc_param_sets(&c).expect("valid hvcC");
        assert_eq!(sets.nal_length_size, 2);
        assert_eq!(
            sets.prefix,
            [
                0, 0, 0, 1, 0x40, 1, 0, 0, 0, 1, 0x42, 1, 2, 0, 0, 0, 1, 0x44
            ]
        );
    }

    #[test]
    fn hvcc_whose_array_count_overruns_is_refused() {
        let mut c = vec![1];
        c.extend_from_slice(&[0; 20]);
        c.extend_from_slice(&[0xf3, 2, 32, 0, 1, 0, 1, 0x40]);
        assert!(matches!(hvcc_param_sets(&c), Err(HwError::Failed { .. })));
    }

    #[test]
    fn sample_nal_units_get_start_codes() {
        let sample = [0, 0, 0, 2, 0x65, 0x88, 0, 0, 0, 1, 0x06];
        let mut out = Vec::new();
        length_prefixed_to_annexb(&sample, 4, &mut out).expect("valid sample");
        assert_eq!(out, [0, 0, 0, 1, 0x65, 0x88, 0, 0, 0, 1, 0x06]);
    }

    #[test]
    fn every_length_size_is_read_big_endian() {
        for size in 1..=4usize {
            let mut sample = vec![0; size - 1];
            sample.push(3);
            sample.extend_from_slice(&[7, 8, 9]);
            let mut out = Vec::new();
            length_prefixed_to_annexb(&sample, size, &mut out).expect("valid");
            assert_eq!(out, [0, 0, 0, 1, 7, 8, 9], "length size {size}");
        }
        let mut out = Vec::new();
        let long = [0x01, 0x00]
            .into_iter()
            .chain(std::iter::repeat_n(0xab, 256))
            .collect::<Vec<u8>>();
        length_prefixed_to_annexb(&long, 2, &mut out).expect("valid");
        assert_eq!(out.len(), 4 + 256);
    }

    #[test]
    fn a_length_past_the_sample_end_is_refused_not_read() {
        let mut out = Vec::new();
        let err = length_prefixed_to_annexb(&[0, 0, 0, 9, 1, 2], 4, &mut out).expect_err("overrun");
        assert!(matches!(err, HwError::Failed { .. }));
        let huge = [0xff, 0xff, 0xff, 0xff, 1];
        assert!(length_prefixed_to_annexb(&huge, 4, &mut out).is_err());
    }

    #[test]
    fn a_cut_length_prefix_is_refused() {
        let mut out = Vec::new();
        assert!(length_prefixed_to_annexb(&[0, 0, 0, 1, 5, 0, 0], 4, &mut out).is_err());
    }

    #[test]
    fn a_nonsense_length_size_is_refused() {
        let mut out = Vec::new();
        assert!(length_prefixed_to_annexb(&[1, 2], 0, &mut out).is_err());
        assert!(length_prefixed_to_annexb(&[1, 2], 5, &mut out).is_err());
    }

    #[test]
    fn empty_nal_units_are_dropped() {
        let mut out = Vec::new();
        length_prefixed_to_annexb(&[0, 0, 0, 1, 0x09], 2, &mut out).expect("valid");
        assert_eq!(out, [0, 0, 0, 1, 0x09]);
    }
}
