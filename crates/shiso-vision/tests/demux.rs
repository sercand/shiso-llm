//! The demuxer against committed fixtures, checked sample by sample against
//! an independent reader's view of the same files.
//!
//! The fixtures were generated once (ffmpeg 9.0.2, `testsrc2` 64x64 at 10 fps
//! for one second) and are committed; nothing here runs ffmpeg. Each
//! `<fixture>.packets.csv` beside them is ffprobe's list of the video packets
//! in file order, `pts,size,pos,flags`, made with
//!
//! ```text
//! ffprobe -v error [-ignore_editlist 1] -select_streams v:0 \
//!     -show_entries packet=pts,pos,size,flags -of csv=p=0 <fixture>
//! ```
//!
//! with `-ignore_editlist 1` for the MP4 files, because this demuxer reports
//! raw sample-table times and does not apply edit lists.
//!
//! The encodes, with `-fflags +bitexact -map_metadata -1` on every output:
//!
//! ```text
//! SRC=-f lavfi -i testsrc2=size=64x64:rate=10:duration=1
//! AUD=-f lavfi -i sine=frequency=440:sample_rate=8000:duration=1
//! X264=-c:v libx264 -preset veryfast -sc_threshold 0 -pix_fmt yuv420p
//! h264.mp4:               AUD SRC -map 0:a -map 1:v -c:a aac -b:a 16k X264 -bf 0 -g 5 -keyint_min 5 -movflags +faststart
//! h264_bframes.mp4:       SRC X264 -bf 2 -b_strategy 0 -g 10 -movflags +faststart
//! h264_trailing_moov.mp4: SRC X264 -bf 0 -g 5            (the muxer's default: moov after mdat)
//! h264_fragmented.mp4:    SRC X264 -bf 0 -g 5 -movflags frag_keyframe+empty_moov
//! hevc.mp4:               SRC -c:v libx265 -x265-params keyint=5:min-keyint=5:bframes=0:scenecut=0 -tag:v hvc1 -movflags +faststart
//! av1.mp4:                SRC -c:v libsvtav1 -g 5 -movflags +faststart
//! vp9.mp4:                SRC -c:v libvpx-vp9 -g 5 -b:v 50k -movflags +faststart
//! vp9.webm:               SRC -c:v libvpx-vp9 -g 5 -b:v 50k
//! h264.mkv:               AUD SRC -map 0:a -map 1:v -c:a aac -b:a 16k X264 -bf 2 -b_strategy 0 -g 5
//! ```
//!
//! `h264.mp4` and `h264.mkv` put an audio track FIRST, so "first video track"
//! is exercised, and their samples interleave with audio chunks, so the chunk
//! map is too.

use shiso_vision::Container;
use shiso_vision::demux::{
    self, ConfigFormat, DemuxError, DemuxedTrack, MAX_SAMPLES, SampleData, SamplePlan, VideoCodec,
};

/// One video packet as ffprobe listed it.
struct Packet {
    pts: i64,
    size: usize,
    pos: usize,
    key: bool,
}

fn packets(csv: &str) -> Vec<Packet> {
    csv.lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            assert_eq!(fields.len(), 4, "csv line {line:?}");
            Packet {
                pts: fields[0].parse().expect("pts"),
                size: fields[1].parse().expect("size"),
                pos: fields[2].parse().expect("pos"),
                key: fields[3].starts_with('K'),
            }
        })
        .collect()
}

/// What a fixture is expected to hold beyond its packet list.
struct Expect {
    container: Container,
    codec: VideoCodec,
    format: ConfigFormat,
    timescale: u32,
    /// MP4 sample ranges are the file positions ffprobe reports; a Matroska
    /// position is the block's, not the frame data's, so it is not compared.
    positions: bool,
}

fn check<'a>(bytes: &'a [u8], csv: &str, expect: &Expect) -> DemuxedTrack<'a> {
    let track = demux::open(bytes).expect("fixture demuxes");
    let truth = packets(csv);
    assert_eq!(track.container(), expect.container);
    assert_eq!(track.codec(), expect.codec);
    assert_eq!(track.config().format(), expect.format);
    assert_eq!((track.width(), track.height()), (64, 64));
    assert_eq!(track.timescale().get(), expect.timescale);
    assert_eq!(track.samples().len(), truth.len(), "sample count");
    assert_eq!(track.samples().len(), 10, "one second at 10 fps");
    for (index, (sample, packet)) in track.samples().iter().zip(&truth).enumerate() {
        let data = track.sample_bytes(index).expect("sample in range");
        assert_eq!(data.len(), packet.size, "sample {index} size");
        assert_eq!(sample.pts().ticks(), packet.pts, "sample {index} pts");
        assert_eq!(sample.keyframe(), packet.key, "sample {index} keyframe");
        // Every fixture's frames are located in the request buffer: nothing
        // is copied.
        let SampleData::Borrowed(range) = sample.data() else {
            panic!("sample {index} was copied");
        };
        if expect.positions {
            assert_eq!(range.start, packet.pos, "sample {index} offset");
        }
        assert_eq!(&bytes[range.clone()], data);
    }
    track
}

const MP4_H264: Expect = Expect {
    container: Container::Mp4,
    codec: VideoCodec::H264,
    format: ConfigFormat::Avcc,
    timescale: 10240,
    positions: true,
};

#[test]
fn h264_mp4_with_audio_first() {
    let bytes = include_bytes!("fixtures/h264.mp4");
    let track = check(
        bytes,
        include_str!("fixtures/h264.mp4.packets.csv"),
        &MP4_H264,
    );
    assert_eq!(track.config().nal_length_size(), Ok(4));
    let keys: Vec<usize> = (0..10).filter(|&i| track.samples()[i].keyframe()).collect();
    assert_eq!(keys, [0, 5], "-g 5 puts a sync sample every five frames");
}

#[test]
fn h264_mp4_with_b_frames_reorders_presentation() {
    let bytes = include_bytes!("fixtures/h264_bframes.mp4");
    let track = check(
        bytes,
        include_str!("fixtures/h264_bframes.mp4.packets.csv"),
        &MP4_H264,
    );
    let pts: Vec<i64> = track.samples().iter().map(|s| s.pts().ticks()).collect();
    assert!(
        pts.windows(2).any(|w| w[1] < w[0]),
        "B-frames show before the reference decoded ahead of them: {pts:?}"
    );
}

#[test]
fn h264_mp4_with_trailing_moov() {
    let bytes = include_bytes!("fixtures/h264_trailing_moov.mp4");
    check(
        bytes,
        include_str!("fixtures/h264_trailing_moov.mp4.packets.csv"),
        &MP4_H264,
    );
}

#[test]
fn hevc_mp4() {
    let bytes = include_bytes!("fixtures/hevc.mp4");
    let track = check(
        bytes,
        include_str!("fixtures/hevc.mp4.packets.csv"),
        &Expect {
            codec: VideoCodec::Hevc,
            format: ConfigFormat::Hvcc,
            ..MP4_H264
        },
    );
    assert_eq!(track.config().nal_length_size(), Ok(4));
}

#[test]
fn av1_mp4() {
    let bytes = include_bytes!("fixtures/av1.mp4");
    let track = check(
        bytes,
        include_str!("fixtures/av1.mp4.packets.csv"),
        &Expect {
            codec: VideoCodec::Av1,
            format: ConfigFormat::Av1c,
            ..MP4_H264
        },
    );
    // av1C: marker bit and version 1 in the first byte.
    assert_eq!(track.config().record().first(), Some(&0x81));
}

#[test]
fn vp9_mp4() {
    let bytes = include_bytes!("fixtures/vp9.mp4");
    let track = check(
        bytes,
        include_str!("fixtures/vp9.mp4.packets.csv"),
        &Expect {
            codec: VideoCodec::Vp9,
            format: ConfigFormat::Vpcc,
            ..MP4_H264
        },
    );
    // vpcC is a full box: version 1, then profile 0.
    assert_eq!(track.config().record().get(..5), Some(&[1, 0, 0, 0, 0][..]));
    assert!(matches!(
        track.config().nal_length_size(),
        Err(DemuxError::Unsupported { .. })
    ));
}

#[test]
fn vp9_webm() {
    let bytes = include_bytes!("fixtures/vp9.webm");
    check(
        bytes,
        include_str!("fixtures/vp9.webm.packets.csv"),
        &Expect {
            container: Container::Matroska,
            codec: VideoCodec::Vp9,
            format: ConfigFormat::MatroskaVp9,
            timescale: 1000,
            positions: false,
        },
    );
}

#[test]
fn h264_mkv_with_audio_first_and_b_frames() {
    let bytes = include_bytes!("fixtures/h264.mkv");
    let track = check(
        bytes,
        include_str!("fixtures/h264.mkv.packets.csv"),
        &Expect {
            container: Container::Matroska,
            codec: VideoCodec::H264,
            format: ConfigFormat::Avcc,
            timescale: 1000,
            positions: false,
        },
    );
    assert_eq!(track.config().nal_length_size(), Ok(4));
    let pts: Vec<i64> = track.samples().iter().map(|s| s.pts().ticks()).collect();
    assert!(pts.windows(2).any(|w| w[1] < w[0]), "{pts:?}");
}

#[test]
fn fragmented_mp4_is_unsupported() {
    let bytes = include_bytes!("fixtures/h264_fragmented.mp4");
    assert!(
        matches!(demux::open(bytes), Err(DemuxError::Unsupported { .. })),
        "{:?}",
        demux::open(bytes)
    );
}

#[test]
fn other_containers_are_unsupported() {
    let avi = b"RIFF\x24\x00\x00\x00AVI LIST\x04\x00\x00\x00hdrl";
    let gif = b"GIF89a\x01\x00\x01\x00\x00\x00\x00;";
    for bytes in [&avi[..], &gif[..], b"", b"not a video at all"] {
        assert!(
            matches!(demux::open(bytes), Err(DemuxError::Unsupported { .. })),
            "{bytes:?}"
        );
    }
}

#[test]
fn plan_over_b_frames_keeps_display_positions() {
    let bytes = include_bytes!("fixtures/h264_bframes.mp4");
    let track = demux::open(bytes).expect("demuxes");
    let plan = SamplePlan::new(&track, 2.0, 768);
    assert!(
        (plan.native_fps() - 10.0).abs() < 1e-9,
        "{}",
        plan.native_fps()
    );
    // One second at 2 fps is two frames, raised to the four-frame floor; the
    // sampler spreads four over ten display positions as 0, 3, 6 and 9.
    assert_eq!(plan.kept(), 4);
    let mut shown: Vec<i64> = track.samples().iter().map(|s| s.pts().ticks()).collect();
    shown.sort_unstable();
    let mut kept: Vec<i64> = (0..track.samples().len())
        .filter(|&i| plan.keep(i))
        .map(|i| track.samples()[i].pts().ticks())
        .collect();
    kept.sort_unstable();
    assert_eq!(kept, [shown[0], shown[3], shown[6], shown[9]]);
}

/// Every fixture, as bytes.
const FIXTURES: [&[u8]; 9] = [
    include_bytes!("fixtures/h264.mp4"),
    include_bytes!("fixtures/h264_bframes.mp4"),
    include_bytes!("fixtures/h264_trailing_moov.mp4"),
    include_bytes!("fixtures/h264_fragmented.mp4"),
    include_bytes!("fixtures/hevc.mp4"),
    include_bytes!("fixtures/av1.mp4"),
    include_bytes!("fixtures/vp9.mp4"),
    include_bytes!("fixtures/vp9.webm"),
    include_bytes!("fixtures/h264.mkv"),
];

/// What any input may demux to: an error, or a track whose every sample lies
/// in the buffer. A panic fails the test on its own.
fn sound(bytes: &[u8]) -> Result<(), DemuxError> {
    let track = demux::open(bytes)?;
    assert!(track.samples().len() <= MAX_SAMPLES);
    assert!(!track.samples().is_empty());
    for index in 0..track.samples().len() {
        assert!(track.sample_bytes(index).is_some(), "sample {index}");
    }
    Ok(())
}

#[test]
fn truncation_never_panics() {
    for fixture in FIXTURES {
        let mp4 = fixture.get(4..8) == Some(b"ftyp");
        for cut in 0..fixture.len() {
            let result = sound(&fixture[..cut]);
            // An MP4 cut anywhere leaves a box overrunning the file. A
            // Matroska cut is read as an early end of file by the crate, so a
            // shorter clip is an acceptable answer there.
            if mp4 {
                assert!(result.is_err(), "MP4 cut at {cut} demuxed");
            }
        }
    }
}

#[test]
fn corruption_never_panics() {
    for fixture in FIXTURES {
        // The index is where the structure is: the first few kilobytes of a
        // faststart MP4 and of a Matroska file, the last few of a trailing
        // moov. Every byte of both ends, three ways.
        let ends =
            (0..fixture.len().min(3072)).chain(fixture.len().saturating_sub(1536)..fixture.len());
        for at in ends {
            for value in [0x00, 0xff, fixture[at] ^ 0x01] {
                let mut bytes = fixture.to_vec();
                bytes[at] = value;
                let _ = sound(&bytes);
            }
        }
    }
}

#[test]
fn errors_are_typed() {
    let fixture = include_bytes!("fixtures/h264.mp4");
    let err = demux::open(&fixture[..fixture.len() - 1]).expect_err("truncated");
    assert!(matches!(err, DemuxError::Malformed { .. }), "{err}");
}
