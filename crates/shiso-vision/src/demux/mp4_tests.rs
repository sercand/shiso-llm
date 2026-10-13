//! The MP4 sample tables, over synthetic box trees small enough to read.

use super::open;
use crate::demux::{DemuxError, MAX_SAMPLES, SampleData, VideoCodec};

fn boxed(kind: &[u8], body: &[u8]) -> Vec<u8> {
    let mut out = u32::try_from(8 + body.len())
        .expect("small")
        .to_be_bytes()
        .to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    out
}

/// A full box: version, three flag bytes, then `body`.
fn full(kind: &[u8], version: u8, body: &[u8]) -> Vec<u8> {
    boxed(kind, &[&[version, 0, 0, 0][..], body].concat())
}

fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_be_bytes()).collect()
}

/// A table box: the entry count, then the entries' words.
fn table(kind: &[u8], version: u8, entries: &[&[u32]]) -> Vec<u8> {
    let mut body = words(&[u32::try_from(entries.len()).expect("small")]);
    for entry in entries {
        body.extend(words(entry));
    }
    full(kind, version, &body)
}

/// The bits of a signed offset, as a `ctts` version 1 stores it.
fn signed(value: i32) -> u32 {
    u32::from_be_bytes(value.to_be_bytes())
}

/// A visual sample entry of `kind`, 64x48, holding `children`.
fn visual_entry(kind: &[u8], width: u16, children: &[u8]) -> Vec<u8> {
    let mut body = vec![0u8; 6];
    body.extend(1u16.to_be_bytes()); // data reference index
    body.extend([0u8; 16]);
    body.extend(width.to_be_bytes());
    body.extend(48u16.to_be_bytes());
    body.extend(words(&[0x0048_0000, 0x0048_0000, 0]));
    body.extend(1u16.to_be_bytes()); // frame count
    body.extend([0u8; 32]); // compressor name
    body.extend(0x0018u16.to_be_bytes());
    body.extend(0xffffu16.to_be_bytes());
    assert_eq!(body.len(), super::VISUAL_ENTRY_FIELDS);
    body.extend_from_slice(children);
    boxed(kind, &body)
}

/// A minimal `avcC`: version, profile, compatibility, level, 4-byte lengths,
/// one SPS, one PPS.
const AVCC: [u8; 15] = [
    1, 0x64, 0, 0x1f, 0xff, 0xe1, 0, 2, 0x67, 0x64, 1, 0, 2, 0x68, 0xee,
];

/// Where the synthetic file's media starts: after a 20-byte `ftyp` and the
/// `mdat` header.
const MEDIA: u32 = 28;

/// The parts of a synthetic file a test changes; [`Spec::default`] is four
/// 10-byte samples in two chunks, 100 ticks apart, no `ctts`, no `stss`.
struct Spec {
    handler: [u8; 4],
    entry: Vec<u8>,
    stsc: Vec<u8>,
    stsz: Vec<u8>,
    stco: Vec<u8>,
    stts: Vec<u8>,
    extra_stbl: Vec<Vec<u8>>,
    extra_moov: Vec<Vec<u8>>,
    extra_top: Vec<Vec<u8>>,
    media: usize,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            handler: *b"vide",
            entry: visual_entry(b"avc1", 64, &boxed(b"avcC", &AVCC)),
            stsc: table(b"stsc", 0, &[&[1, 2, 1]]),
            stsz: full(b"stsz", 0, &words(&[0, 4, 10, 10, 10, 10])),
            stco: table(b"stco", 0, &[&[MEDIA], &[MEDIA + 20]]),
            stts: table(b"stts", 0, &[&[4, 100]]),
            extra_stbl: Vec::new(),
            extra_moov: Vec::new(),
            extra_top: Vec::new(),
            media: 40,
        }
    }
}

impl Spec {
    fn build(&self) -> Vec<u8> {
        let ftyp = boxed(b"ftyp", b"isom\0\0\0\0isom");
        let payload: Vec<u8> = (0..self.media)
            .map(|i| u8::try_from(i % 251).expect("small"))
            .collect();
        let mdat = boxed(b"mdat", &payload);
        let mdhd = full(b"mdhd", 0, &words(&[0, 0, 1000, 400, 0]));
        let hdlr = full(
            b"hdlr",
            0,
            &[&[0u8; 4][..], &self.handler, &[0u8; 13]].concat(),
        );
        let stsd = full(b"stsd", 0, &[&words(&[1])[..], &self.entry].concat());
        let mut stbl = [
            stsd,
            self.stts.clone(),
            self.stsc.clone(),
            self.stsz.clone(),
        ]
        .concat();
        stbl.extend(self.stco.clone());
        for extra in &self.extra_stbl {
            stbl.extend_from_slice(extra);
        }
        let minf = boxed(b"minf", &boxed(b"stbl", &stbl));
        let mdia = boxed(b"mdia", &[mdhd, hdlr, minf].concat());
        let mut moov = boxed(b"trak", &mdia);
        for extra in &self.extra_moov {
            moov.extend_from_slice(extra);
        }
        let mut file = [ftyp, mdat, boxed(b"moov", &moov)].concat();
        for extra in &self.extra_top {
            file.extend_from_slice(extra);
        }
        assert_eq!(
            file.get(20..24),
            Some(&words(&[8 + u32::try_from(self.media).expect("small")])[..])
        );
        file
    }
}

fn ranges(spec: &Spec) -> Vec<std::ops::Range<usize>> {
    open(&spec.build())
        .expect("demuxes")
        .samples
        .into_iter()
        .map(|s| match s.data {
            SampleData::Borrowed(range) => range,
            SampleData::Owned(_) => panic!("MP4 samples are never copied"),
        })
        .collect()
}

fn malformed(spec: &Spec) {
    let result = open(&spec.build());
    assert!(
        matches!(result, Err(DemuxError::Malformed { .. })),
        "{result:?}"
    );
}

fn unsupported(spec: &Spec) {
    let result = open(&spec.build());
    assert!(
        matches!(result, Err(DemuxError::Unsupported { .. })),
        "{result:?}"
    );
}

#[test]
fn reads_the_default_track() {
    let track = open(&Spec::default().build()).expect("demuxes");
    assert_eq!(track.codec, VideoCodec::H264);
    assert_eq!(track.config.record(), &AVCC);
    assert_eq!((track.width, track.height), (64, 48));
    assert_eq!(track.timescale.get(), 1000);
    let pts: Vec<i64> = track.samples.iter().map(|s| s.pts.ticks()).collect();
    assert_eq!(pts, [0, 100, 200, 300]);
    assert!(
        track.samples.iter().all(|s| s.keyframe),
        "no stss: every sample is a sync sample"
    );
    let m = 28;
    assert_eq!(
        ranges(&Spec::default()),
        [m..m + 10, m + 10..m + 20, m + 20..m + 30, m + 30..m + 40]
    );
}

#[test]
fn chunk_map_runs_and_a_short_last_chunk() {
    // Chunk 1 holds one sample, chunks 2 and 3 two each by the second run;
    // the third chunk is declared to hold more than the size table has left.
    let spec = Spec {
        stsz: full(b"stsz", 0, &words(&[0, 4, 10, 10, 10, 10])),
        stco: table(b"stco", 0, &[&[MEDIA], &[MEDIA + 15], &[MEDIA + 35]]),
        stsc: table(b"stsc", 0, &[&[1, 1, 1], &[2, 2, 1]]),
        media: 60,
        ..Spec::default()
    };
    let m = 28;
    assert_eq!(
        ranges(&spec),
        [m..m + 10, m + 15..m + 25, m + 25..m + 35, m + 35..m + 45]
    );
}

#[test]
fn co64_offsets() {
    let spec = Spec {
        stco: {
            let mut body = words(&[2]);
            body.extend(u64::from(MEDIA).to_be_bytes());
            body.extend(u64::from(MEDIA + 20).to_be_bytes());
            full(b"co64", 0, &body)
        },
        ..Spec::default()
    };
    assert_eq!(ranges(&spec).len(), 4);
}

#[test]
fn signed_composition_offsets_reorder_presentation() {
    // I P B P: the B is decoded after the P it is shown before.
    let spec = Spec {
        extra_stbl: vec![table(
            b"ctts",
            1,
            &[&[1, 0], &[1, 100], &[1, signed(-100)], &[1, 0]],
        )],
        ..Spec::default()
    };
    let track = open(&spec.build()).expect("demuxes");
    let pts: Vec<i64> = track.samples.iter().map(|s| s.pts.ticks()).collect();
    assert_eq!(pts, [0, 200, 100, 300]);
}

#[test]
fn sync_sample_table() {
    let spec = Spec {
        extra_stbl: vec![table(b"stss", 0, &[&[1], &[3]])],
        ..Spec::default()
    };
    let track = open(&spec.build()).expect("demuxes");
    let keys: Vec<bool> = track.samples.iter().map(|s| s.keyframe).collect();
    assert_eq!(keys, [true, false, true, false]);
}

#[test]
fn run_lengths_are_bounded_by_the_sample_count() {
    // A run that claims four billion samples expands to the four that exist.
    let spec = Spec {
        stts: table(b"stts", 0, &[&[u32::MAX, 100]]),
        extra_stbl: vec![table(b"ctts", 0, &[&[u32::MAX, 0]])],
        ..Spec::default()
    };
    assert_eq!(ranges(&spec).len(), 4);
}

#[test]
fn size_table_shorter_than_its_count() {
    malformed(&Spec {
        stsz: full(b"stsz", 0, &words(&[0, 1000, 10, 10, 10, 10])),
        ..Spec::default()
    });
}

#[test]
fn uniform_size_bounded_by_the_file() {
    // A uniform size has no table to check against, so its total is.
    malformed(&Spec {
        stsz: full(b"stsz", 0, &words(&[1_000_000, 1000])),
        ..Spec::default()
    });
}

#[test]
fn sample_cap_applies_only_to_a_backed_count() {
    let over = u32::try_from(MAX_SAMPLES + 1).expect("fits");
    // Over the cap with nothing behind it: a broken index, refused, never
    // passed on to another decoder.
    malformed(&Spec {
        stsz: full(b"stsz", 0, &words(&[1, over])),
        ..Spec::default()
    });
    malformed(&Spec {
        stsz: full(b"stsz", 0, &words(&[0, u32::MAX, 10, 10])),
        ..Spec::default()
    });
    // Over the cap with the bytes to back it: a long file, unsupported here.
    unsupported(&Spec {
        stsz: full(b"stsz", 0, &words(&[1, over])),
        media: MAX_SAMPLES + 1,
        ..Spec::default()
    });
}

#[test]
fn uniform_size() {
    let spec = Spec {
        stsz: full(b"stsz", 0, &words(&[10, 4])),
        ..Spec::default()
    };
    assert_eq!(ranges(&spec), ranges(&Spec::default()));
}

#[test]
fn sample_past_the_file() {
    malformed(&Spec {
        stco: table(b"stco", 0, &[&[MEDIA], &[1_000_000]]),
        ..Spec::default()
    });
    malformed(&Spec {
        stco: table(b"stco", 0, &[&[MEDIA], &[u32::MAX - 5]]),
        ..Spec::default()
    });
}

#[test]
fn time_tables_shorter_than_the_track() {
    malformed(&Spec {
        stts: table(b"stts", 0, &[&[3, 100]]),
        ..Spec::default()
    });
    malformed(&Spec {
        extra_stbl: vec![table(b"ctts", 0, &[&[2, 0]])],
        ..Spec::default()
    });
}

#[test]
fn chunk_map_faults() {
    // First chunks start at one and only increase.
    malformed(&Spec {
        stsc: table(b"stsc", 0, &[&[0, 2, 1]]),
        ..Spec::default()
    });
    malformed(&Spec {
        stsc: table(b"stsc", 0, &[&[1, 1, 1], &[1, 1, 1]]),
        ..Spec::default()
    });
    // A run past the chunk table.
    malformed(&Spec {
        stsc: table(b"stsc", 0, &[&[1, 1, 1]]),
        ..Spec::default()
    });
    malformed(&Spec {
        stsc: table(b"stsc", 0, &[&[1, 0, 1], &[u32::MAX, 4, 1]]),
        ..Spec::default()
    });
    // A second sample description.
    unsupported(&Spec {
        stsc: table(b"stsc", 0, &[&[1, 2, 2]]),
        ..Spec::default()
    });
}

#[test]
fn sync_sample_faults() {
    malformed(&Spec {
        extra_stbl: vec![table(b"stss", 0, &[&[0]])],
        ..Spec::default()
    });
    malformed(&Spec {
        extra_stbl: vec![table(b"stss", 0, &[&[5]])],
        ..Spec::default()
    });
}

#[test]
fn sample_entry_faults() {
    unsupported(&Spec {
        entry: visual_entry(b"mp4v", 64, &[]),
        ..Spec::default()
    });
    malformed(&Spec {
        entry: visual_entry(b"avc1", 64, &boxed(b"colr", b"nclx\0\x01\0\x01\0\x01\0")),
        ..Spec::default()
    });
    malformed(&Spec {
        entry: visual_entry(b"avc1", 0, &boxed(b"avcC", &AVCC)),
        ..Spec::default()
    });
}

#[test]
fn sample_entry_padding_is_tolerated() {
    // A zero terminator after the children, as some writers leave.
    let spec = Spec {
        entry: visual_entry(
            b"avc1",
            64,
            &[&boxed(b"avcC", &AVCC)[..], &[0, 0, 0, 0]].concat(),
        ),
        ..Spec::default()
    };
    assert_eq!(ranges(&spec).len(), 4);
}

#[test]
fn not_video() {
    unsupported(&Spec {
        handler: *b"soun",
        ..Spec::default()
    });
}

#[test]
fn fragmented() {
    unsupported(&Spec {
        extra_moov: vec![boxed(b"mvex", &full(b"trex", 0, &[0u8; 20]))],
        ..Spec::default()
    });
    unsupported(&Spec {
        extra_top: vec![boxed(b"moof", &full(b"mfhd", 0, &words(&[1])))],
        ..Spec::default()
    });
}

#[test]
fn boxes_that_do_not_tile() {
    let mut file = Spec::default().build();
    file.push(0);
    assert!(matches!(open(&file), Err(DemuxError::Malformed { .. })));
}
