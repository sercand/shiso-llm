//! Matroska codec mapping, timescales, and frames located in the buffer,
//! including the `BlockGroup` frames the ffmpeg-written fixtures do not have.

use std::num::NonZeroU64;

use super::{ReadLog, codec_config, open, timescale};
use crate::demux::{ConfigFormat, DemuxError, SampleData, VideoCodec};

/// One EBML element: its ID bytes, an eight-byte size, its body.
fn el(id: &[u8], body: &[u8]) -> Vec<u8> {
    let mut out = id.to_vec();
    out.push(0x01);
    out.extend_from_slice(&u64::try_from(body.len()).expect("small").to_be_bytes()[1..]);
    out.extend_from_slice(body);
    out
}

/// An unsigned integer element, in as few bytes as it takes.
fn uint(id: &[u8], value: u64) -> Vec<u8> {
    let bytes = value.to_be_bytes();
    let skip = bytes.iter().take(7).take_while(|&&b| b == 0).count();
    el(id, &bytes[skip..])
}

/// A block body: track 1, a relative timestamp, flags, then the frame.
fn block(relative_ms: i16, flags: u8, frame: &[u8]) -> Vec<u8> {
    [&[0x81][..], &relative_ms.to_be_bytes(), &[flags], frame].concat()
}

/// A VP9 `WebM` file with one `SimpleBlock` keyframe at 0 ms and one
/// `BlockGroup` frame each at 33 ms (duration after the block) and 66 ms
/// (duration before it).
fn webm_with_block_group() -> Vec<u8> {
    let header = el(
        &[0x1a, 0x45, 0xdf, 0xa3],
        &[
            uint(&[0x42, 0x86], 1),
            uint(&[0x42, 0xf7], 1),
            uint(&[0x42, 0xf2], 4),
            uint(&[0x42, 0xf3], 8),
            el(&[0x42, 0x82], b"webm"),
            uint(&[0x42, 0x87], 4),
            uint(&[0x42, 0x85], 2),
        ]
        .concat(),
    );
    let info = el(
        &[0x15, 0x49, 0xa9, 0x66],
        &[
            uint(&[0x2a, 0xd7, 0xb1], 1_000_000),
            el(&[0x4d, 0x80], b"t"),
            el(&[0x57, 0x41], b"t"),
        ]
        .concat(),
    );
    let video = el(&[0xe0], &[uint(&[0xb0], 64), uint(&[0xba], 48)].concat());
    let tracks = el(
        &[0x16, 0x54, 0xae, 0x6b],
        &el(
            &[0xae],
            &[
                uint(&[0xd7], 1),
                uint(&[0x73, 0xc5], 1),
                uint(&[0x83], 1),
                el(&[0x86], b"V_VP9"),
                video,
            ]
            .concat(),
        ),
    );
    let cluster = el(
        &[0x1f, 0x43, 0xb6, 0x75],
        &[
            uint(&[0xe7], 0),
            el(&[0xa3], &block(0, 0x80, b"KEYFRAME")),
            el(
                &[0xa0],
                &[el(&[0xa1], &block(33, 0, b"inter")), uint(&[0x9b], 33)].concat(),
            ),
            // A master element's children have no fixed order: the duration
            // may come first, right after a frame that also had one.
            el(
                &[0xa0],
                &[uint(&[0x9b], 33), el(&[0xa1], &block(66, 0, b"swapped"))].concat(),
            ),
        ]
        .concat(),
    );
    [
        header,
        el(&[0x18, 0x53, 0x80, 0x67], &[info, tracks, cluster].concat()),
    ]
    .concat()
}

#[test]
fn block_group_frames_stay_in_the_buffer() {
    let bytes = webm_with_block_group();
    let track = open(&bytes).expect("demuxes");
    assert_eq!(track.codec, VideoCodec::Vp9);
    assert_eq!(track.config.format(), ConfigFormat::MatroskaVp9);
    assert!(track.config.record().is_empty());
    assert_eq!((track.width, track.height), (64, 48));
    assert_eq!(track.timescale.get(), 1000);
    let frames: Vec<(&[u8], i64, bool)> = track
        .samples
        .iter()
        .map(|s| {
            let SampleData::Borrowed(range) = &s.data else {
                panic!("frame was copied");
            };
            (&bytes[range.clone()], s.pts.ticks(), s.keyframe)
        })
        .collect();
    assert_eq!(
        frames,
        [
            (&b"KEYFRAME"[..], 0, true),
            // No keyframe flag on a Block; reported as not a keyframe.
            (&b"inter"[..], 33, false),
            (&b"swapped"[..], 66, false),
        ]
    );
}

#[test]
fn locate_needs_matching_bytes_not_only_length() {
    let bytes = b"abcdefabcxyz";
    let log = ReadLog::default();
    log.push(0, 3);
    log.push(6, 3);
    log.push(9, 3);
    // Latest match wins; a same-length read of other bytes is not a match.
    assert_eq!(log.locate(bytes, b"abc"), Some(6..9));
    assert_eq!(log.locate(bytes, b"qqq"), None);
    log.clear();
    assert_eq!(log.locate(bytes, b"abc"), None);
}

#[test]
fn codec_ids() {
    let cases = [
        ("V_MPEG4/ISO/AVC", VideoCodec::H264, ConfigFormat::Avcc),
        ("V_MPEGH/ISO/HEVC", VideoCodec::Hevc, ConfigFormat::Hvcc),
        ("V_VP9", VideoCodec::Vp9, ConfigFormat::MatroskaVp9),
        ("V_AV1", VideoCodec::Av1, ConfigFormat::Av1c),
    ];
    for (id, codec, format) in cases {
        let (got, config) = codec_config(id, Some(&[1, 2, 3])).expect(id);
        assert_eq!(
            (got, config.format(), config.record()),
            (codec, format, &[1, 2, 3][..])
        );
    }
    // Only VP9 may come without codec private data.
    assert!(codec_config("V_VP9", None).is_ok());
    for id in ["V_MPEG4/ISO/AVC", "V_MPEGH/ISO/HEVC", "V_AV1"] {
        assert!(matches!(
            codec_config(id, None),
            Err(DemuxError::Malformed { .. })
        ));
    }
    for id in ["V_VP8", "V_THEORA", "V_MPEG4/ISO/ASP", ""] {
        assert!(matches!(
            codec_config(id, Some(&[])),
            Err(DemuxError::Unsupported { .. })
        ));
    }
}

#[test]
fn timescales() {
    let scale = |ns| timescale(NonZeroU64::new(ns).expect("non-zero"));
    let pair = |(ts, tick): (std::num::NonZeroU32, i64)| (ts.get(), tick);
    // The default millisecond tick, and a microsecond one, divide a second.
    assert_eq!(pair(scale(1_000_000)), (1000, 1));
    assert_eq!(pair(scale(1_000)), (1_000_000, 1));
    assert_eq!(pair(scale(1)), (1_000_000_000, 1));
    // A tick that does not divide a second is converted to nanoseconds.
    assert_eq!(pair(scale(3)), (1_000_000_000, 3));
    assert_eq!(pair(scale(2_000_000_000)), (1_000_000_000, 2_000_000_000));
}

#[test]
fn no_video_track() {
    let mut bytes = webm_with_block_group();
    // Turn the track type from video (1) into audio (2).
    let at = bytes
        .windows(10)
        .position(|w| w == [0x83, 0x01, 0, 0, 0, 0, 0, 0, 0x01, 0x01])
        .expect("track type element");
    bytes[at + 9] = 2;
    assert!(matches!(open(&bytes), Err(DemuxError::Unsupported { .. })));
}
