//! The software AV1 decoder against committed IVF clips.
//!
//! The clips under `tests/fixtures/` are ten frames of a 160x96 test pattern,
//! encoded once (8-bit and 10-bit 4:2:0) and committed:
//!
//! ```sh
//! ffmpeg -f lavfi -i testsrc2=size=160x96:rate=10 -frames:v 10 -pix_fmt yuv420p \
//!   -c:v libsvtav1 -preset 8 -crf 35 -f ivf av1_8bit.ivf   # yuv420p10le: av1_10bit.ivf
//! ```
//!
//! The reference frames they are compared with live under the gitignored
//! `test-vectors/references/media-sys/`: the same clips decoded and converted
//! to RGB by ffmpeg, with nearest-neighbour chroma like the `colour` module
//! (its default filtered upsample of the 10-bit clip sits 31 dB from any
//! nearest-chroma conversion of this pattern's sharp colour edges):
//!
//! ```sh
//! ffmpeg -i av1_8bit.ivf -sws_flags neighbor+accurate_rnd+bitexact \
//!   -f rawvideo -pix_fmt rgb24 av1_8bit.rgb24                # and av1_10bit
//! ```
//!
//! The AV1 decode itself is normative, so what the reference checks
//! independently is the plane reading and the colour conversion. A missing
//! reference skips only the comparison, with a line on stderr, so read the
//! output - a skipped comparison and a passing one look alike. Everything else
//! here (frame counts, sizes, timestamps, refusals) needs no reference and
//! always runs.
//!
//! No encoder available here writes 4:4:4, 4:2:2, 12-bit or monochrome AV1, so
//! those refusals are tested with hand-built sequence headers: a header alone is
//! what the decoder has to refuse, and it is the part those streams differ in.
#![cfg(feature = "av1-rav1d")]

use std::path::{Path, PathBuf};

use shiso_media_sys::{Av1Decoder, HwCodec, HwDecoderDesc, HwError, HwFrame, HwTimestamp};

const WIDTH: u32 = 160;
const HEIGHT: u32 = 96;
const FRAMES: usize = 10;
/// The floor every decoded frame has to clear against its reference: high
/// enough to catch a wrong plane, stride or matrix, low enough for two YUV to
/// RGB conversions that round differently.
const PSNR_FLOOR_DB: f64 = 40.0;

fn fixture(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn reference_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/media-sys")
        .join(name)
}

#[expect(
    clippy::print_stderr,
    reason = "a skipped comparison has to say so in the test output, and a run its numbers"
)]
fn say(line: &str) {
    eprintln!("{line}");
}

// --- IVF ---------------------------------------------------------------------

/// One frame of an IVF file: its timestamp and a temporal unit.
struct IvfFrame {
    pts: i64,
    data: Vec<u8>,
}

/// Reads an IVF file: a 32-byte header (`DKIF`, version, header size, fourcc,
/// width, height, rate, scale, frame count), then frames of a 12-byte header
/// (size, 64-bit timestamp) and their data.
fn read_ivf(bytes: &[u8]) -> Vec<IvfFrame> {
    assert_eq!(&bytes[..4], b"DKIF", "not an IVF file");
    assert_eq!(&bytes[8..12], b"AV01", "not an AV1 IVF file");
    let header_len = usize::from(u16::from_le_bytes([bytes[6], bytes[7]]));
    let mut at = header_len;
    let mut frames = Vec::new();
    while at < bytes.len() {
        let size = u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes"));
        let pts = u64::from_le_bytes(bytes[at + 4..at + 12].try_into().expect("8 bytes"));
        let start = at + 12;
        let end = start + usize::try_from(size).expect("frame size fits usize");
        frames.push(IvfFrame {
            pts: i64::try_from(pts).expect("timestamp fits i64"),
            data: bytes[start..end].to_vec(),
        });
        at = end;
    }
    frames
}

// --- OBUs and av1C -------------------------------------------------------------

const OBU_SEQUENCE_HEADER: u8 = 1;

/// Reads a LEB128 value, returning it and its length in bytes.
fn leb128(bytes: &[u8]) -> (usize, usize) {
    let mut value = 0usize;
    for (i, byte) in bytes.iter().enumerate().take(8) {
        value |= usize::from(byte & 0x7f) << (7 * i);
        if byte & 0x80 == 0 {
            return (value, i + 1);
        }
    }
    panic!("a LEB128 value longer than 8 bytes");
}

fn write_leb128(mut value: usize, out: &mut Vec<u8>) {
    loop {
        let byte = u8::try_from(value & 0x7f).expect("seven bits");
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// The whole sequence header OBU (header and payload) in a temporal unit.
fn sequence_header_obu(temporal_unit: &[u8]) -> Vec<u8> {
    let mut at = 0;
    while at < temporal_unit.len() {
        let header = temporal_unit[at];
        let kind = (header >> 3) & 0x0f;
        let extension = usize::from(header & 0x04 != 0);
        assert!(header & 0x02 != 0, "OBUs in a sample carry their size");
        let (size, size_len) = leb128(&temporal_unit[at + 1 + extension..]);
        let end = at + 1 + extension + size_len + size;
        if kind == OBU_SEQUENCE_HEADER {
            return temporal_unit[at..end].to_vec();
        }
        at = end;
    }
    panic!("no sequence header in the first temporal unit");
}

/// An `av1C` record: marker and version, profile and level, the layout flags,
/// no presentation delay, then `obus`.
fn av1c(flags: u8, obus: &[u8]) -> Vec<u8> {
    let mut record = vec![0x81, 0x00, flags, 0x00];
    record.extend_from_slice(obus);
    record
}

/// `av1C` flags for 4:2:0: both chroma subsampling bits, plus `high_bitdepth`
/// for a 10-bit stream.
fn flags_420(ten_bit: bool) -> u8 {
    if ten_bit { 0x4c } else { 0x0c }
}

/// Builds an OBU bit by bit.
struct Bits {
    bytes: Vec<u8>,
    used: u32,
}

impl Bits {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            used: 0,
        }
    }

    fn put(&mut self, value: u32, width: u32) {
        for shift in (0..width).rev() {
            if self.used.is_multiple_of(8) {
                self.bytes.push(0);
            }
            let bit = u8::try_from((value >> shift) & 1).expect("one bit");
            let last = self.bytes.last_mut().expect("a byte was pushed");
            *last |= bit << (7 - self.used % 8);
            self.used += 1;
        }
    }

    /// The trailing one bit and zero padding every OBU payload ends with.
    fn finish(mut self) -> Vec<u8> {
        self.put(1, 1);
        self.bytes
    }
}

/// The layouts a hand-built sequence header can declare.
#[derive(Clone, Copy, Debug)]
enum Crafted {
    /// Profile 0, 8-bit 4:2:0: the one the decoder accepts, so the builder is
    /// shown to produce a header it reads.
    Yuv420,
    /// Profile 0, monochrome.
    Mono,
    /// Profile 1, 8-bit 4:4:4.
    Yuv444,
    /// Profile 2, 10-bit 4:2:2.
    Yuv422,
    /// Profile 2, 12-bit 4:2:0.
    TwelveBit,
}

/// A sequence header OBU for a 160x96 still picture with the reduced header,
/// declaring `layout`.
fn crafted_sequence_header(layout: Crafted) -> Vec<u8> {
    let profile = match layout {
        Crafted::Yuv420 | Crafted::Mono => 0,
        Crafted::Yuv444 => 1,
        Crafted::Yuv422 | Crafted::TwelveBit => 2,
    };
    let mut bits = Bits::new();
    bits.put(profile, 3); // seq_profile
    bits.put(1, 1); // still_picture
    bits.put(1, 1); // reduced_still_picture_header
    bits.put(0, 5); // seq_level_idx[0]
    bits.put(7, 4); // frame_width_bits_minus_1: 8 bits
    bits.put(6, 4); // frame_height_bits_minus_1: 7 bits
    bits.put(WIDTH - 1, 8); // max_frame_width_minus_1
    bits.put(HEIGHT - 1, 7); // max_frame_height_minus_1
    bits.put(0, 1); // use_128x128_superblock
    bits.put(0, 1); // enable_filter_intra
    bits.put(0, 1); // enable_intra_edge_filter
    bits.put(0, 1); // enable_superres
    bits.put(0, 1); // enable_cdef
    bits.put(0, 1); // enable_restoration
    // color_config
    let high_bitdepth = matches!(layout, Crafted::Yuv422 | Crafted::TwelveBit);
    bits.put(u32::from(high_bitdepth), 1);
    if profile == 2 && high_bitdepth {
        bits.put(u32::from(matches!(layout, Crafted::TwelveBit)), 1); // twelve_bit
    }
    if profile != 1 {
        bits.put(u32::from(matches!(layout, Crafted::Mono)), 1); // mono_chrome
    }
    bits.put(0, 1); // color_description_present_flag
    bits.put(0, 1); // color_range
    match layout {
        Crafted::Mono => {}
        Crafted::Yuv420 => {
            bits.put(0, 2); // chroma_sample_position
            bits.put(0, 1); // separate_uv_delta_q
        }
        Crafted::Yuv444 | Crafted::Yuv422 => bits.put(0, 1), // separate_uv_delta_q
        Crafted::TwelveBit => {
            bits.put(1, 1); // subsampling_x
            bits.put(1, 1); // subsampling_y
            bits.put(0, 2); // chroma_sample_position
            bits.put(0, 1); // separate_uv_delta_q
        }
    }
    bits.put(0, 1); // film_grain_params_present
    let payload = bits.finish();
    let mut obu = vec![OBU_SEQUENCE_HEADER << 3 | 0x02];
    write_leb128(payload.len(), &mut obu);
    obu.extend_from_slice(&payload);
    obu
}

// --- decoding ----------------------------------------------------------------

fn desc(config: &[u8]) -> HwDecoderDesc<'_> {
    HwDecoderDesc::new(HwCodec::Av1, config, WIDTH, HEIGHT)
}

/// Decodes every frame of `frames`, keeping those `keep` picks, then flushes.
/// Returns what `decode` produced and what `flush` produced.
fn decode_all(
    config: &[u8],
    frames: &[IvfFrame],
    keep: impl Fn(usize) -> bool,
) -> (Vec<HwFrame>, Vec<HwFrame>) {
    let mut decoder = Av1Decoder::open(&desc(config)).expect("open");
    let mut out = Vec::new();
    for (i, frame) in frames.iter().enumerate() {
        out.extend(
            decoder
                .decode(&frame.data, HwTimestamp::new(frame.pts), keep(i))
                .unwrap_or_else(|e| panic!("decode frame {i}: {e}")),
        );
    }
    let flushed = decoder.flush().expect("flush");
    (out, flushed)
}

fn psnr(got: &[u8], want: &[u8]) -> f64 {
    assert_eq!(got.len(), want.len());
    let sum: f64 = got
        .iter()
        .zip(want)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    #[expect(
        clippy::cast_precision_loss,
        reason = "a frame's byte count is far below 2^52"
    )]
    let mse = sum / got.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

/// Decodes `clip` with and without its sequence header in an `av1C` record and
/// checks the frames: count, size, timestamps in display order, and against
/// `reference` when it is present.
fn check_clip(clip: &str, reference: &str, ten_bit: bool) {
    let frames = read_ivf(&fixture(clip));
    assert_eq!(frames.len(), FRAMES, "{clip}: frame count in the file");
    let record = av1c(flags_420(ten_bit), &sequence_header_obu(&frames[0].data));
    let reference_file = reference_path(reference);
    let reference = std::fs::read(&reference_file).ok();
    if reference.is_none() {
        say(&format!(
            "SKIP {clip}: no reference at {}; the PSNR comparison did not run, \
             frames are checked for count, size and order only",
            reference_file.display()
        ));
    }
    for (label, config) in [("av1C", record.as_slice()), ("in-band", &[][..])] {
        let (decoded, flushed) = decode_all(config, &frames, |_| true);
        let all: Vec<HwFrame> = decoded.into_iter().chain(flushed).collect();
        assert_eq!(all.len(), FRAMES, "{clip} ({label}): frames out");
        let mut want_pts: Vec<i64> = frames.iter().map(|f| f.pts).collect();
        want_pts.sort_unstable();
        let got_pts: Vec<i64> = all.iter().map(|f| f.pts().ticks()).collect();
        assert_eq!(got_pts, want_pts, "{clip} ({label}): display order");
        for frame in &all {
            assert_eq!((frame.width(), frame.height()), (WIDTH, HEIGHT));
        }
        if let Some(reference) = &reference {
            let size = usize::try_from(WIDTH * HEIGHT * 3).expect("fits");
            assert_eq!(reference.len(), size * FRAMES, "{clip}: reference size");
            let mut worst = f64::INFINITY;
            for (i, (frame, want)) in all.iter().zip(reference.chunks_exact(size)).enumerate() {
                let db = psnr(frame.rgb(), want);
                assert!(
                    db >= PSNR_FLOOR_DB,
                    "{clip} ({label}) frame {i}: {db:.2} dB < {PSNR_FLOOR_DB} dB"
                );
                worst = worst.min(db);
            }
            say(&format!(
                "{clip} ({label}): worst frame {worst:.2} dB against the reference"
            ));
        }
    }
}

#[test]
fn eight_bit_clip_decodes_in_display_order() {
    check_clip("av1_8bit.ivf", "av1_8bit.rgb24", false);
}

#[test]
fn ten_bit_clip_decodes_in_display_order() {
    check_clip("av1_10bit.ivf", "av1_10bit.rgb24", true);
}

#[test]
fn dropped_frames_are_decoded_but_not_returned() {
    let frames = read_ivf(&fixture("av1_8bit.ivf"));
    let (decoded, flushed) = decode_all(&[], &frames, |i| i % 3 == 0);
    let all: Vec<i64> = decoded
        .iter()
        .chain(&flushed)
        .map(|f| f.pts().ticks())
        .collect();
    // Every third sample in decode order; the clip's frames are one per
    // temporal unit, so the kept timestamps are those samples' own.
    let mut want: Vec<i64> = frames
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 3 == 0)
        .map(|(_, f)| f.pts)
        .collect();
    want.sort_unstable();
    assert_eq!(all, want);
}

#[test]
fn flush_drains_and_a_fresh_decoder_holds_nothing() {
    let mut fresh = Av1Decoder::open(&desc(&[])).expect("open");
    assert!(fresh.flush().expect("flush").is_empty());
    let frames = read_ivf(&fixture("av1_10bit.ivf"));
    let (out, flushed) = decode_all(&[], &frames, |_| true);
    assert_eq!(out.len() + flushed.len(), FRAMES);
}

#[test]
fn other_codecs_are_unsupported() {
    for codec in [HwCodec::H264, HwCodec::Hevc, HwCodec::Vp9] {
        let err = Av1Decoder::open(&HwDecoderDesc::new(codec, &[], WIDTH, HEIGHT))
            .expect_err("only AV1 opens");
        assert!(
            matches!(err, HwError::Unsupported { codec: c, .. } if c == codec),
            "{codec}: {err:?}"
        );
    }
}

#[test]
fn a_malformed_av1c_record_fails() {
    for record in [&[0x81][..], &[0x81, 0, 0x0c], &[0x01, 0, 0x0c, 0]] {
        let err = Av1Decoder::open(&desc(record)).expect_err("malformed record");
        assert!(matches!(err, HwError::Failed { .. }), "{record:?}: {err:?}");
    }
    // A sequence header OBU that claims more payload than it has.
    let mut truncated = crafted_sequence_header(Crafted::Yuv420);
    truncated.truncate(truncated.len() - 2);
    let err = Av1Decoder::open(&desc(&av1c(0x0c, &truncated))).expect_err("truncated header");
    assert!(matches!(err, HwError::Failed { .. }), "{err:?}");
}

#[test]
fn av1c_flags_refuse_layouts_without_a_sequence_header() {
    // 4:4:4 (no subsampling), 4:2:2 (x only), 12-bit, monochrome.
    for flags in [0x00, 0x08, 0x6c, 0x1c] {
        let err = Av1Decoder::open(&desc(&av1c(flags, &[]))).expect_err("unsupported");
        assert!(
            matches!(
                err,
                HwError::Unsupported {
                    codec: HwCodec::Av1,
                    ..
                }
            ),
            "{flags:#04x}: {err:?}"
        );
    }
    assert!(Av1Decoder::open(&desc(&av1c(0x0c, &[]))).is_ok());
    assert!(Av1Decoder::open(&desc(&av1c(0x4c, &[]))).is_ok());
}

#[test]
fn sequence_headers_refuse_layouts_in_av1c_and_in_band() {
    // The builder makes a header the decoder reads, both ways.
    let good = crafted_sequence_header(Crafted::Yuv420);
    Av1Decoder::open(&desc(&av1c(0x0c, &good))).expect("a 4:2:0 header opens");
    let mut decoder = Av1Decoder::open(&desc(&[])).expect("open");
    assert!(
        decoder
            .decode(&good, HwTimestamp::new(0), true)
            .expect("a header alone")
            .is_empty()
    );

    for layout in [
        Crafted::Mono,
        Crafted::Yuv444,
        Crafted::Yuv422,
        Crafted::TwelveBit,
    ] {
        let header = crafted_sequence_header(layout);
        // The record's flags claim 4:2:0; the header inside is what refuses.
        let err = Av1Decoder::open(&desc(&av1c(0x0c, &header))).expect_err("in av1C");
        assert!(
            matches!(err, HwError::Unsupported { .. }),
            "{layout:?} in av1C: {err:?}"
        );
        let mut decoder = Av1Decoder::open(&desc(&[])).expect("open");
        let err = decoder
            .decode(&header, HwTimestamp::new(0), true)
            .expect_err("in band");
        assert!(
            matches!(err, HwError::Unsupported { .. }),
            "{layout:?} in band: {err:?}"
        );
    }
}

#[test]
fn bad_samples_fail() {
    let mut decoder = Av1Decoder::open(&desc(&[])).expect("open");
    let err = decoder
        .decode(&[], HwTimestamp::new(0), true)
        .expect_err("empty sample");
    assert!(matches!(err, HwError::Failed { .. }), "{err:?}");

    // An OBU header claiming a payload far longer than the sample.
    let mut decoder = Av1Decoder::open(&desc(&[])).expect("open");
    let err = decoder
        .decode(&[0x32, 0xff, 0xff, 0x03, 0x00], HwTimestamp::new(0), true)
        .expect_err("overlong OBU");
    assert!(matches!(err, HwError::Failed { .. }), "{err:?}");
}

/// A small deterministic generator, so a failing mutation can be replayed.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(n).expect("fits")).expect("fits")
    }
}

// Why the two damaged-input tests below are ignored. The library this decoder
// wraps (1.1.0, the latest release) aborts the process on a sample whose tile
// data fails to decode: with one frame context, its error path unwraps a frame
// header already released, inside an `extern "C"` function, so the panic cannot
// unwind. Upstream fixed it in PR 1503 on `main`; run these with `--ignored`
// against a release that carries that fix, then drop the `ignore`s.

/// One flipped bit aborts: the 8-bit clip opened from its av1C record, with
/// bit 0 of byte 226 of its fourth temporal unit flipped. The damaged sample
/// is the first one that fails, so a caller that stops at the first error is
/// not protected. It must come back as an error.
#[test]
#[ignore = "aborts on rav1d 1.1.0: fixed upstream in PR 1503, not yet released"]
fn a_damaged_tile_fails() {
    let frames = read_ivf(&fixture("av1_8bit.ivf"));
    let record = av1c(flags_420(false), &sequence_header_obu(&frames[0].data));
    let mut decoder = Av1Decoder::open(&desc(&record)).expect("open");
    for frame in &frames[..3] {
        decoder
            .decode(&frame.data, HwTimestamp::new(frame.pts), true)
            .expect("the undamaged samples decode");
    }
    let mut damaged = frames[3].data.clone();
    damaged[226] ^= 1;
    let err = decoder
        .decode(&damaged, HwTimestamp::new(frames[3].pts), true)
        .expect_err("a damaged sample");
    assert!(matches!(err, HwError::Failed { .. }), "{err:?}");
}

/// Damaged input must come back as `Ok` or `Err`, never a panic or an abort:
/// each case truncates or corrupts one sample of a clip and decodes the whole
/// clip on a fresh decoder. Every frame that does come out has the clip's
/// size.
#[test]
#[ignore = "aborts on rav1d 1.1.0: fixed upstream in PR 1503, not yet released"]
fn damaged_samples_never_crash() {
    let mut rng = XorShift(0x5eed_a1c1_dead_beef);
    let mut cases = 0;
    for clip in ["av1_8bit.ivf", "av1_10bit.ivf"] {
        let frames = read_ivf(&fixture(clip));
        let record = av1c(0x0c, &sequence_header_obu(&frames[0].data));
        for case in 0..150 {
            let target = rng.below(frames.len());
            let mut sample = frames[target].data.clone();
            let at = rng.below(sample.len());
            match case % 3 {
                0 => sample.truncate(at.max(1)),
                1 => sample[at] ^= 1 << rng.below(8),
                _ => {
                    for _ in 0..4 {
                        let at = rng.below(sample.len());
                        sample[at] = u8::try_from(rng.next() & 0xff).expect("one byte");
                    }
                }
            }
            // Half the cases open with an av1C record, and half of those with
            // its sequence header damaged too.
            let mut config = if case % 2 == 0 {
                record.clone()
            } else {
                Vec::new()
            };
            if case % 4 == 0 && config.len() > 4 {
                let at = 4 + rng.below(config.len() - 4);
                config[at] ^= 0x10;
            }
            let Ok(mut decoder) = Av1Decoder::open(&desc(&config)) else {
                cases += 1;
                continue;
            };
            for (i, frame) in frames.iter().enumerate() {
                let bytes = if i == target { &sample } else { &frame.data };
                if let Ok(out) = decoder.decode(bytes, HwTimestamp::new(frame.pts), true) {
                    for frame in out {
                        assert_eq!(
                            frame.rgb().len(),
                            frame.width() as usize * frame.height() as usize * 3
                        );
                    }
                }
            }
            let _ = decoder.flush();
            cases += 1;
        }
    }
    assert_eq!(cases, 300);
}
