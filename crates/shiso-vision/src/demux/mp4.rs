//! MP4/MOV: the first video track's sample tables, read in place.
//!
//! The path is `moov/trak/mdia` (whose `hdlr` says `vide`), then
//! `minf/stbl`, whose tables between them say where every sample is and when
//! it plays:
//!
//! - `stsd` — the sample entry, which names the codec and carries its
//!   configuration record (`avcC`, `hvcC`, `vpcC`, `av1C`) and the size;
//! - `stsz` — each sample's size;
//! - `stco`/`co64` and `stsc` — where each chunk starts and how many samples
//!   it holds, which together with the sizes give each sample's byte range;
//! - `stts` and `ctts` — decode-time deltas and the composition offset added to
//!   them for the presentation time;
//! - `stss` — the sync samples; without it every sample is one.
//!
//! # What is ignored, and why that is safe
//!
//! The edit list (`edts/elst`) is not applied. It shifts or trims the timeline
//! for playback, most often by the composition delay of the first frame; the
//! sampler only reads the ORDER and SPACING of presentation times, both of
//! which a shift leaves alone. A trim could keep a few lead-in frames a player
//! would hide, which is a frame or two of extra context, not a wrong frame.
//!
//! # Hostile input
//!
//! Every count is checked against the bytes that would back it before
//! anything proportional to it is allocated, and a count no table backs (a
//! uniform sample size, a run length) is bounded by the sample count, which is
//! itself capped at [`MAX_SAMPLES`]. All arithmetic is checked, and every
//! sample's range is checked to lie inside the buffer, so a decoder is never
//! handed a range that would make it read past the upload.

use std::num::NonZeroU32;

use super::{
    CodecConfig, ConfigFormat, DemuxError, MAX_SAMPLES, ParsedTrack, Pts, Sample, SampleData,
    VideoCodec,
};
use crate::mp4::{Span, boxes, header};

/// Bytes of a visual sample entry before its child boxes: the sample entry
/// header (8), then the visual fields up to and including `pre_defined` (70).
const VISUAL_ENTRY_FIELDS: usize = 78;

/// Reads the first video track of an MP4 or `QuickTime` file.
pub(super) fn open(bytes: &[u8]) -> Result<ParsedTrack, DemuxError> {
    let top = boxes(bytes, 0, bytes.len())
        .ok_or_else(|| DemuxError::malformed("top-level boxes do not tile the file"))?;
    // A fragmented file keeps its samples in `moof` boxes the tables below
    // never mention; checked before the tables, because its `moov` holds empty
    // ones that would otherwise read as a track with no samples.
    if find(&top, *b"moof").is_some() {
        return Err(DemuxError::unsupported("fragmented MP4 (moof boxes)"));
    }
    let moov = find(&top, *b"moov").ok_or_else(|| DemuxError::malformed("no moov box"))?;
    let moov_children = children(bytes, moov, "moov")?;
    if find(&moov_children, *b"mvex").is_some() {
        return Err(DemuxError::unsupported("fragmented MP4 (mvex box)"));
    }
    for trak in moov_children.iter().filter(|span| &span.kind == b"trak") {
        let trak_children = children(bytes, *trak, "trak")?;
        let mdia = require(&trak_children, *b"mdia")?;
        let mdia_children = children(bytes, mdia, "mdia")?;
        let hdlr = body(bytes, require(&mdia_children, *b"hdlr")?)?;
        // Full-box header (4), `pre_defined` (4), then the handler type.
        if hdlr.get(8..12) == Some(b"vide") {
            return read_track(bytes, &mdia_children);
        }
    }
    Err(DemuxError::unsupported("no video track"))
}

/// Reads one video track from the children of its `mdia` box.
fn read_track(bytes: &[u8], mdia: &[Span]) -> Result<ParsedTrack, DemuxError> {
    let timescale = media_timescale(body(bytes, require(mdia, *b"mdhd")?)?)?;
    let minf = children(bytes, require(mdia, *b"minf")?, "minf")?;
    let stbl = children(bytes, require(&minf, *b"stbl")?, "stbl")?;
    let entry = SampleEntry::read(bytes, require(&stbl, *b"stsd")?)?;

    let sizes = SampleSizes::read(bytes, &stbl)?;
    let ranges = sample_ranges(bytes, &stbl, &sizes)?;
    let pts = presentation_times(bytes, &stbl, sizes.count)?;
    let keyframes = keyframes(bytes, &stbl, sizes.count)?;
    let samples = ranges
        .into_iter()
        .zip(pts)
        .zip(keyframes)
        .map(|((range, pts), keyframe)| Sample {
            data: SampleData::Borrowed(range),
            pts,
            keyframe,
        })
        .collect();
    Ok(ParsedTrack {
        codec: entry.codec,
        config: entry.config,
        width: entry.width,
        height: entry.height,
        timescale,
        samples,
    })
}

/// The first box of `kind` among `spans`.
fn find(spans: &[Span], kind: [u8; 4]) -> Option<Span> {
    spans.iter().copied().find(|span| span.kind == kind)
}

/// The first box of `kind` among `spans`, which the format requires.
fn require(spans: &[Span], kind: [u8; 4]) -> Result<Span, DemuxError> {
    find(spans, kind).ok_or_else(|| DemuxError::malformed(format!("missing {} box", fourcc(kind))))
}

/// The boxes inside container `span`, which must tile its body exactly.
fn children(bytes: &[u8], span: Span, what: &str) -> Result<Vec<Span>, DemuxError> {
    let from = body_start(bytes, span)?;
    boxes(bytes, from, span.end())
        .ok_or_else(|| DemuxError::malformed(format!("{what} children do not tile the box")))
}

/// Where `span`'s body starts, checked against its end (a 64-bit size field
/// leaves a 16-byte header that a box shorter than that cannot hold).
fn body_start(bytes: &[u8], span: Span) -> Result<usize, DemuxError> {
    let from = span.start.saturating_add(header(bytes, span));
    if from > span.end() {
        return Err(DemuxError::malformed(format!(
            "{} box shorter than its header",
            fourcc(span.kind)
        )));
    }
    Ok(from)
}

/// The body of `span`, after its header.
fn body(bytes: &[u8], span: Span) -> Result<&[u8], DemuxError> {
    let from = body_start(bytes, span)?;
    bytes
        .get(from..span.end())
        .ok_or_else(|| DemuxError::malformed("box runs past the file"))
}

/// A box kind, printable.
fn fourcc(kind: [u8; 4]) -> String {
    kind.iter()
        .map(|&b| {
            if b.is_ascii_graphic() || b == b' ' {
                char::from(b)
            } else {
                '?'
            }
        })
        .collect()
}

/// `N` bytes of `bytes` at `at`, or `Malformed` naming `what`.
fn array<const N: usize>(bytes: &[u8], at: usize, what: &str) -> Result<[u8; N], DemuxError> {
    at.checked_add(N)
        .and_then(|end| bytes.get(at..end))
        .and_then(|slice| <[u8; N]>::try_from(slice).ok())
        .ok_or_else(|| DemuxError::malformed(format!("{what} truncated")))
}

fn u16_at(bytes: &[u8], at: usize, what: &str) -> Result<u16, DemuxError> {
    array(bytes, at, what).map(u16::from_be_bytes)
}

fn u32_at(bytes: &[u8], at: usize, what: &str) -> Result<u32, DemuxError> {
    array(bytes, at, what).map(u32::from_be_bytes)
}

fn i32_at(bytes: &[u8], at: usize, what: &str) -> Result<i32, DemuxError> {
    array(bytes, at, what).map(i32::from_be_bytes)
}

fn u64_at(bytes: &[u8], at: usize, what: &str) -> Result<u64, DemuxError> {
    array(bytes, at, what).map(u64::from_be_bytes)
}

/// A 32-bit count as a `usize`. Saturates on a target narrower than 32 bits,
/// where the bounds check that follows every count then refuses it.
fn count(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

/// The table of `count` entries of `width` bytes starting at `at` in a box
/// body, checked to fit before anything is read from it or allocated for it.
fn table<'b>(
    body: &'b [u8],
    at: usize,
    count: usize,
    width: usize,
    what: &str,
) -> Result<&'b [u8], DemuxError> {
    count
        .checked_mul(width)
        .and_then(|len| at.checked_add(len))
        .and_then(|end| body.get(at..end))
        .ok_or_else(|| {
            DemuxError::malformed(format!(
                "{what} declares {count} entries its box cannot hold"
            ))
        })
}

/// The media timescale from `mdhd`, whose layout depends on its version.
fn media_timescale(mdhd: &[u8]) -> Result<NonZeroU32, DemuxError> {
    // Version 1 widens the creation and modification times to 64 bits.
    let at = match mdhd.first() {
        Some(0) => 12,
        Some(1) => 20,
        Some(v) => return Err(DemuxError::malformed(format!("mdhd version {v}"))),
        None => return Err(DemuxError::malformed("mdhd truncated")),
    };
    NonZeroU32::new(u32_at(mdhd, at, "mdhd")?)
        .ok_or_else(|| DemuxError::malformed("mdhd timescale is zero"))
}

/// What the first `stsd` entry says about the codec.
struct SampleEntry {
    codec: VideoCodec,
    config: CodecConfig,
    width: u32,
    height: u32,
}

impl SampleEntry {
    fn read(bytes: &[u8], stsd: Span) -> Result<Self, DemuxError> {
        let from = body_start(bytes, stsd)?;
        let declared = u32_at(bytes, from.saturating_add(4), "stsd")?;
        if declared == 0 {
            return Err(DemuxError::malformed("stsd has no sample entries"));
        }
        let entries = boxes(bytes, from.saturating_add(8), stsd.end())
            .ok_or_else(|| DemuxError::malformed("stsd entries do not tile the box"))?;
        let entry = entries
            .first()
            .copied()
            .ok_or_else(|| DemuxError::malformed("stsd has no sample entries"))?;
        let (codec, config_kind, format) = match &entry.kind {
            b"avc1" | b"avc3" => (VideoCodec::H264, b"avcC", ConfigFormat::Avcc),
            b"hvc1" | b"hev1" => (VideoCodec::Hevc, b"hvcC", ConfigFormat::Hvcc),
            b"vp09" => (VideoCodec::Vp9, b"vpcC", ConfigFormat::Vpcc),
            b"av01" => (VideoCodec::Av1, b"av1C", ConfigFormat::Av1c),
            other => {
                return Err(DemuxError::unsupported(format!(
                    "video sample entry {}",
                    fourcc(*other)
                )));
            }
        };
        let fields = body(bytes, entry)?;
        // Six reserved bytes, the data reference index, then sixteen bytes of
        // pre-defined and reserved fields before the size.
        let width = u32::from(u16_at(fields, 24, "visual sample entry")?);
        let height = u32::from(u16_at(fields, 26, "visual sample entry")?);
        if width == 0 || height == 0 {
            return Err(DemuxError::malformed("visual sample entry has a zero size"));
        }
        let children_from = body_start(bytes, entry)?.saturating_add(VISUAL_ENTRY_FIELDS);
        let config = entry_children(bytes, children_from, entry.end())?
            .into_iter()
            .find(|span| &span.kind == config_kind)
            .ok_or_else(|| {
                DemuxError::malformed(format!(
                    "{} sample entry without its {} box",
                    fourcc(entry.kind),
                    fourcc(*config_kind)
                ))
            })?;
        Ok(Self {
            codec,
            config: CodecConfig::new(format, body(bytes, config)?.to_vec()),
            width,
            height,
        })
    }
}

/// The child boxes of a sample entry, from `from` to `to`.
///
/// More lenient than [`boxes`]: `QuickTime` writers may close the list with a
/// zero terminator or leave a few padding bytes, so the walk stops at a zero
/// size or when fewer bytes remain than a header needs. A box that overruns
/// the entry is still refused.
fn entry_children(bytes: &[u8], from: usize, to: usize) -> Result<Vec<Span>, DemuxError> {
    let mut out = Vec::new();
    let mut at = from;
    while to.saturating_sub(at) >= 8 {
        let size = u32_at(bytes, at, "sample entry child")?;
        if size == 0 {
            break;
        }
        let kind = array::<4>(bytes, at.saturating_add(4), "sample entry child")?;
        let len = count(size);
        let end = at
            .checked_add(len)
            .filter(|&end| len >= 8 && end <= to)
            .ok_or_else(|| DemuxError::malformed("sample entry child overruns its entry"))?;
        out.push(Span {
            kind,
            start: at,
            len,
        });
        at = end;
    }
    Ok(out)
}

/// The `stsz` table: either one size for every sample or one per sample.
struct SampleSizes<'b> {
    count: usize,
    uniform: u32,
    table: &'b [u8],
}

impl<'b> SampleSizes<'b> {
    fn read(bytes: &'b [u8], stbl: &[Span]) -> Result<Self, DemuxError> {
        let Some(stsz) = find(stbl, *b"stsz") else {
            return Err(if find(stbl, *b"stz2").is_some() {
                DemuxError::unsupported("compact sample sizes (stz2)")
            } else {
                DemuxError::malformed("missing stsz box")
            });
        };
        let body = body(bytes, stsz)?;
        let uniform = u32_at(body, 4, "stsz")?;
        let count = count(u32_at(body, 8, "stsz")?);
        if count == 0 {
            return Err(DemuxError::malformed("video track has no samples"));
        }
        // The count is checked against the bytes behind it BEFORE the cap: a
        // count nothing backs is a broken index (refused), and only a backed
        // one over the cap is a long file another step may take.
        let table = if uniform == 0 {
            table(body, 12, count, 4, "stsz")?
        } else {
            // No table backs a uniform size, so bound the count by the bytes
            // it claims: every sample is at least one byte of the file.
            let total = u64::from(uniform).checked_mul(u64::try_from(count).unwrap_or(u64::MAX));
            if total.is_none_or(|total| total > u64::try_from(bytes.len()).unwrap_or(u64::MAX)) {
                return Err(DemuxError::malformed(
                    "stsz declares more sample bytes than the file holds",
                ));
            }
            &[]
        };
        if count > MAX_SAMPLES {
            return Err(DemuxError::unsupported(format!(
                "{count} samples, over the cap of {MAX_SAMPLES}"
            )));
        }
        Ok(Self {
            count,
            uniform,
            table,
        })
    }

    /// The size of sample `index` (decode order).
    fn get(&self, index: usize) -> Result<u32, DemuxError> {
        if self.uniform != 0 {
            return Ok(self.uniform);
        }
        u32_at(self.table, index.saturating_mul(4), "stsz")
    }
}

/// Each sample's byte range, from the chunk offsets, the sample-to-chunk map
/// and the sizes.
fn sample_ranges(
    bytes: &[u8],
    stbl: &[Span],
    sizes: &SampleSizes<'_>,
) -> Result<Vec<std::ops::Range<usize>>, DemuxError> {
    let (offsets, wide) = match (find(stbl, *b"stco"), find(stbl, *b"co64")) {
        (Some(stco), _) => (body(bytes, stco)?, false),
        (None, Some(co64)) => (body(bytes, co64)?, true),
        (None, None) => return Err(DemuxError::malformed("missing stco/co64 box")),
    };
    let width = if wide { 8 } else { 4 };
    let chunks = count(u32_at(offsets, 4, "chunk offsets")?);
    let offsets = table(offsets, 8, chunks, width, "chunk offsets")?;
    let chunk_offset = |chunk: usize| -> Result<u64, DemuxError> {
        let at = chunk.saturating_mul(width);
        if wide {
            u64_at(offsets, at, "co64")
        } else {
            u32_at(offsets, at, "stco").map(u64::from)
        }
    };

    let stsc = body(bytes, require(stbl, *b"stsc")?)?;
    let runs = count(u32_at(stsc, 4, "stsc")?);
    let runs = table(stsc, 8, runs, 12, "stsc")?;
    let file_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    let mut out = Vec::with_capacity(sizes.count);
    let mut previous_first = 0u64;
    let mut runs_iter = runs.chunks_exact(12).peekable();
    while let Some(run) = runs_iter.next() {
        let first = u64::from(u32_at(run, 0, "stsc")?);
        let per_chunk = u32_at(run, 4, "stsc")?;
        let description = u32_at(run, 8, "stsc")?;
        if first <= previous_first {
            return Err(DemuxError::malformed(
                "stsc first chunks are not increasing",
            ));
        }
        previous_first = first;
        if description != 1 {
            return Err(DemuxError::unsupported(
                "samples that switch sample descriptions",
            ));
        }
        // A run lasts until the next run's first chunk, or to the last chunk.
        let next_first = match runs_iter.peek() {
            Some(next) => u64::from(u32_at(next, 0, "stsc")?),
            None => u64::try_from(chunks).unwrap_or(u64::MAX).saturating_add(1),
        };
        let mut chunk = first;
        // Bounded by the chunk table: a chunk past it is refused below, and
        // first chunks only increase, so no run revisits a chunk.
        while chunk < next_first && out.len() < sizes.count {
            let index = usize::try_from(chunk - 1).unwrap_or(usize::MAX);
            if index >= chunks {
                return Err(DemuxError::malformed(
                    "stsc names a chunk stco does not have",
                ));
            }
            let mut at = chunk_offset(index)?;
            for _ in 0..per_chunk {
                if out.len() == sizes.count {
                    break;
                }
                let end = at
                    .checked_add(u64::from(sizes.get(out.len())?))
                    .filter(|&end| end <= file_len)
                    .ok_or_else(|| DemuxError::malformed("a sample runs past the file"))?;
                let start = usize::try_from(at)
                    .map_err(|_| DemuxError::malformed("sample offset out of range"))?;
                let stop = usize::try_from(end)
                    .map_err(|_| DemuxError::malformed("sample offset out of range"))?;
                out.push(start..stop);
                at = end;
            }
            chunk += 1;
        }
    }
    if out.len() < sizes.count {
        return Err(DemuxError::malformed(format!(
            "the chunk map places {} of {} samples",
            out.len(),
            sizes.count
        )));
    }
    Ok(out)
}

/// Expands a run-length table of `(count, value)` pairs into `samples` values.
///
/// Run lengths are not backed by any bytes, so expansion stops at `samples`
/// however large a run claims to be; fewer than `samples` values is refused.
fn expand_runs(
    runs: &[u8],
    samples: usize,
    what: &str,
    mut each: impl FnMut(&[u8]) -> Result<i64, DemuxError>,
) -> Result<Vec<i64>, DemuxError> {
    let mut out = Vec::with_capacity(samples);
    for run in runs.chunks_exact(8) {
        if out.len() == samples {
            break;
        }
        let length = count(u32_at(run, 0, what)?);
        let value = each(run)?;
        let take = length.min(samples - out.len());
        out.extend(std::iter::repeat_n(value, take));
    }
    if out.len() < samples {
        return Err(DemuxError::malformed(format!(
            "{what} covers {} of {samples} samples",
            out.len()
        )));
    }
    Ok(out)
}

/// Presentation time of each sample: its decode time from `stts` plus its
/// composition offset from `ctts`, when there is one.
fn presentation_times(bytes: &[u8], stbl: &[Span], samples: usize) -> Result<Vec<Pts>, DemuxError> {
    let stts = body(bytes, require(stbl, *b"stts")?)?;
    let runs = count(u32_at(stts, 4, "stts")?);
    let deltas = expand_runs(table(stts, 8, runs, 8, "stts")?, samples, "stts", |run| {
        u32_at(run, 4, "stts").map(i64::from)
    })?;
    let offsets = match find(stbl, *b"ctts") {
        // Read as signed in both versions: version 1 says so, and version 0
        // files with negative offsets stored the same bits are common enough
        // that players read them this way too.
        Some(ctts) => {
            let ctts = body(bytes, ctts)?;
            let runs = count(u32_at(ctts, 4, "ctts")?);
            Some(expand_runs(
                table(ctts, 8, runs, 8, "ctts")?,
                samples,
                "ctts",
                |run| i32_at(run, 4, "ctts").map(i64::from),
            )?)
        }
        None => None,
    };
    let mut out = Vec::with_capacity(samples);
    let mut decode_time = 0i64;
    for (index, delta) in deltas.into_iter().enumerate() {
        let offset = offsets
            .as_ref()
            .and_then(|o| o.get(index).copied())
            .unwrap_or(0);
        let pts = decode_time
            .checked_add(offset)
            .ok_or_else(|| DemuxError::malformed("presentation time overflows"))?;
        out.push(Pts::new(pts));
        decode_time = decode_time
            .checked_add(delta)
            .ok_or_else(|| DemuxError::malformed("decode time overflows"))?;
    }
    Ok(out)
}

/// Which samples a decoder may start from: those `stss` lists, or every
/// sample when there is no `stss`.
fn keyframes(bytes: &[u8], stbl: &[Span], samples: usize) -> Result<Vec<bool>, DemuxError> {
    let Some(stss) = find(stbl, *b"stss") else {
        return Ok(vec![true; samples]);
    };
    let stss = body(bytes, stss)?;
    let entries = count(u32_at(stss, 4, "stss")?);
    let numbers = table(stss, 8, entries, 4, "stss")?;
    let mut out = vec![false; samples];
    for number in numbers.chunks_exact(4) {
        // Sample numbers are one-based.
        let index = count(u32_at(number, 0, "stss")?)
            .checked_sub(1)
            .ok_or_else(|| DemuxError::malformed("stss names sample 0"))?;
        let slot = out
            .get_mut(index)
            .ok_or_else(|| DemuxError::malformed("stss names a sample past the track"))?;
        *slot = true;
    }
    Ok(out)
}

#[cfg(test)]
#[path = "mp4_tests.rs"]
mod tests;
