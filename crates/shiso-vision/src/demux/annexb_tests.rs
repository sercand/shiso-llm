//! Length-prefixed to Annex-B, on hand-built samples and on the fixtures.

use super::{avcc_to_annexb, parameter_sets_annexb};
use crate::demux::{CodecConfig, ConfigFormat, DemuxError, open};

/// `nals` behind big-endian lengths of `size` bytes.
fn length_prefixed(nals: &[&[u8]], size: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for nal in nals {
        let len = u32::try_from(nal.len()).expect("small").to_be_bytes();
        out.extend_from_slice(&len[4 - size..]);
        out.extend_from_slice(nal);
    }
    out
}

/// The NAL units of an Annex-B stream this module wrote (four-byte start
/// codes only, and no emulation bytes in these test payloads).
fn split_annexb(stream: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut at = 0;
    while at + 4 <= stream.len() {
        if stream[at..at + 4] == [0, 0, 0, 1] {
            starts.push(at + 4);
            at += 4;
        } else {
            at += 1;
        }
    }
    starts
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let end = starts.get(i + 1).map_or(stream.len(), |&next| next - 4);
            &stream[s..end]
        })
        .collect()
}

#[test]
fn round_trip_each_prefix_width() {
    let nals: [&[u8]; 3] = [&[0x09, 0xf0], &[0x65, 0x88, 0x84, 0x21, 0xa0], &[0x06, 5]];
    for size in [1, 2, 4] {
        let mut out = vec![0xaa]; // appends, never clears
        avcc_to_annexb(&length_prefixed(&nals, size), size, &mut out).expect("converts");
        assert_eq!(out[0], 0xaa);
        assert_eq!(split_annexb(&out[1..]), nals, "prefix width {size}");
    }
}

#[test]
fn empty_nal_units_are_dropped() {
    let mut out = Vec::new();
    avcc_to_annexb(&[0, 0, 0, 0, 0, 0, 0, 1, 0x65], 4, &mut out).expect("converts");
    assert_eq!(out, [0, 0, 0, 1, 0x65]);
}

#[test]
fn bounds() {
    let mut out = Vec::new();
    // A length past the sample.
    assert!(matches!(
        avcc_to_annexb(&[0, 0, 0, 9, 0x65, 1], 4, &mut out),
        Err(DemuxError::Malformed { .. })
    ));
    // A sample that ends inside a prefix.
    assert!(matches!(
        avcc_to_annexb(&[0, 0, 0, 1, 0x65, 0, 0], 4, &mut out),
        Err(DemuxError::Malformed { .. })
    ));
    for size in [0, 3, 5, 8] {
        assert!(matches!(
            avcc_to_annexb(&[0, 1, 0x65], size, &mut out),
            Err(DemuxError::Malformed { .. })
        ));
    }
}

#[test]
fn h264_fixture_sample_and_parameter_sets() {
    let bytes = include_bytes!("../../tests/fixtures/h264.mp4");
    let track = open(bytes).expect("demuxes");
    let size = track.config().nal_length_size().expect("avcC");
    let sample = track.sample_bytes(0).expect("first sample");
    let mut out = Vec::new();
    avcc_to_annexb(sample, size, &mut out).expect("converts");
    let nals = split_annexb(&out);
    // Each NAL unit costs its prefix and gains a start code of the same size.
    assert_eq!(out.len(), sample.len());
    // The first sample is an IDR picture: NAL type 5 somewhere in it.
    assert!(
        nals.iter()
            .any(|nal| nal.first().map(|b| b & 0x1f) == Some(5))
    );

    let sets = parameter_sets_annexb(track.config()).expect("avcC");
    let types: Vec<u8> = split_annexb(&sets)
        .iter()
        .map(|nal| nal[0] & 0x1f)
        .collect();
    assert_eq!(types, [7, 8], "one SPS then one PPS");
}

#[test]
fn hevc_fixture_parameter_sets() {
    let bytes = include_bytes!("../../tests/fixtures/hevc.mp4");
    let track = open(bytes).expect("demuxes");
    let sets = parameter_sets_annexb(track.config()).expect("hvcC");
    let types: Vec<u8> = split_annexb(&sets)
        .iter()
        .map(|nal| (nal[0] >> 1) & 0x3f)
        .collect();
    // VPS, SPS, PPS lead; the encoder may add an SEI array after them.
    assert_eq!(types.get(..3), Some(&[32, 33, 34][..]), "{types:?}");
}

#[test]
fn parameter_set_faults() {
    // A truncated SPS length.
    let short = CodecConfig::new(
        ConfigFormat::Avcc,
        vec![1, 0x64, 0, 0x1f, 0xff, 0xe1, 0, 9, 0x67],
    );
    assert!(matches!(
        parameter_sets_annexb(&short),
        Err(DemuxError::Malformed { .. })
    ));
    let hvcc = CodecConfig::new(ConfigFormat::Hvcc, vec![1; 10]);
    assert!(matches!(
        parameter_sets_annexb(&hvcc),
        Err(DemuxError::Malformed { .. })
    ));
    let av1 = CodecConfig::new(ConfigFormat::Av1c, vec![0x81, 0, 0, 0]);
    assert!(matches!(
        parameter_sets_annexb(&av1),
        Err(DemuxError::Unsupported { .. })
    ));
}
