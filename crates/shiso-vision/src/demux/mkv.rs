//! Matroska/WebM: the first video track, through the `matroska-demuxer` crate.
//!
//! # Where a frame's bytes come from
//!
//! The crate copies each frame into a buffer it owns and reports no position.
//! Copying every frame would double the media's footprint, so the reader it is
//! given records the offset and length of every read it makes, and a frame
//! whose bytes match one of the reads made while fetching it is kept as that
//! range of the request buffer instead. Matching on the bytes, not only the
//! length, is what makes this safe against a change in how the crate reads: a
//! frame that cannot be located is kept as its own copy
//! ([`SampleData::Owned`]), never as a guessed range. The current position is
//! NOT a usable answer on its own: for a frame stored in a `BlockGroup` the
//! crate goes on to read the group's other elements before it returns.
//!
//! # Hostile input
//!
//! The crate bounds every element it allocates by the bytes remaining, which
//! is the property that matters for memory. Two things it does not do are
//! handled here:
//!
//! - Its block-timestamp arithmetic can overflow on crafted input, which panics
//!   in a build with overflow checks. The whole parse runs under
//!   [`std::panic::catch_unwind`], and a panic is a [`DemuxError::Malformed`]:
//!   a parse of request bytes must not be able to take the request thread
//!   down.
//! - It treats a read error as the end of the file, so a truncated upload
//!   demuxes to the frames before the cut rather than to an error. That is a
//!   short clip, not a wrong one; a cut that leaves no video frame at all is
//!   refused.
//!
//! The frame loop is capped at [`MAX_SAMPLES`] video frames and a multiple of
//! that across all tracks, so a file of empty laced frames cannot spin it.

use std::cell::RefCell;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::num::{NonZeroU32, NonZeroU64};
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use matroska_demuxer::{Frame, MatroskaFile, TrackType};

use super::{
    CodecConfig, ConfigFormat, DemuxError, MAX_SAMPLES, ParsedTrack, Pts, Sample, SampleData,
    VideoCodec,
};

/// Frames of every track the loop reads before giving up. Audio frames
/// interleave with video at a few per video frame; this leaves room for that
/// while still bounding a file built to be nothing but tiny laced frames.
const MAX_FRAMES_ALL_TRACKS: usize = MAX_SAMPLES * 8;

/// Nanoseconds per second: Matroska timestamps are a tick count whose length
/// in nanoseconds the segment declares.
const NANOS_PER_SECOND: u64 = 1_000_000_000;

/// Reads the first video track of a Matroska or `WebM` file.
pub(super) fn open(bytes: &[u8]) -> Result<ParsedTrack, DemuxError> {
    catch_unwind(AssertUnwindSafe(|| read(bytes)))
        .unwrap_or_else(|_| Err(DemuxError::malformed("the Matroska parser panicked")))
}

fn read(bytes: &[u8]) -> Result<ParsedTrack, DemuxError> {
    // The crate owns its reader and gives no way back to it, so the read log
    // is shared between the reader and this loop.
    let log = ReadLog::default();
    let mut file =
        MatroskaFile::open(Recorded::new(bytes, log.clone())).map_err(|e| malformed(&e))?;
    let track = file
        .tracks()
        .iter()
        .find(|t| t.track_type() == TrackType::Video)
        .ok_or_else(|| DemuxError::unsupported("no video track"))?;
    if track.content_encodings().is_some() {
        // Compressed or encrypted frames are not codec bytes as stored.
        return Err(DemuxError::unsupported("Matroska content encoding"));
    }
    let (codec, config) = codec_config(track.codec_id(), track.codec_private())?;
    let video = track
        .video()
        .ok_or_else(|| DemuxError::malformed("video track without video settings"))?;
    let width = side(video.pixel_width())?;
    let height = side(video.pixel_height())?;
    let number = track.track_number().get();
    let (timescale, tick) = timescale(file.info().timestamp_scale());

    let mut samples = Vec::new();
    let mut frame = Frame::default();
    let mut seen = 0usize;
    loop {
        log.clear();
        // The crate refuses a `BlockDuration` when the frame it fills already
        // has one, and only resets it once a block is read; a group whose
        // duration precedes its block (legal: a master's children have no
        // order) would otherwise fail after any frame that had a duration.
        frame.duration = None;
        if !file.next_frame(&mut frame).map_err(|e| malformed(&e))? {
            break;
        }
        seen += 1;
        if seen > MAX_FRAMES_ALL_TRACKS {
            return Err(DemuxError::unsupported(format!(
                "more than {MAX_FRAMES_ALL_TRACKS} frames across all tracks"
            )));
        }
        if frame.track != number {
            continue;
        }
        if samples.len() == MAX_SAMPLES {
            return Err(DemuxError::unsupported(format!(
                "more than {MAX_SAMPLES} video frames"
            )));
        }
        let ticks = i64::try_from(frame.timestamp)
            .ok()
            .and_then(|t| t.checked_mul(tick))
            .ok_or_else(|| DemuxError::malformed("frame timestamp out of range"))?;
        let data = match log.locate(bytes, &frame.data) {
            Some(range) => SampleData::Borrowed(range),
            None => SampleData::Owned(std::mem::take(&mut frame.data)),
        };
        samples.push(Sample {
            data,
            pts: Pts::new(ticks),
            keyframe: frame.is_keyframe.unwrap_or(false),
        });
    }
    if samples.is_empty() {
        return Err(DemuxError::malformed("video track has no frames"));
    }
    Ok(ParsedTrack {
        codec,
        config,
        width,
        height,
        timescale,
        samples,
    })
}

/// The crate's error, as a refusal: by the time it parses, the magic said
/// Matroska.
fn malformed(error: &matroska_demuxer::DemuxError) -> DemuxError {
    DemuxError::malformed(format!("Matroska: {error}"))
}

/// Maps a Matroska codec ID and its codec private data onto a codec and its
/// configuration record.
pub(super) fn codec_config(
    codec_id: &str,
    private: Option<&[u8]>,
) -> Result<(VideoCodec, CodecConfig), DemuxError> {
    let (codec, format) = match codec_id {
        "V_MPEG4/ISO/AVC" => (VideoCodec::H264, ConfigFormat::Avcc),
        "V_MPEGH/ISO/HEVC" => (VideoCodec::Hevc, ConfigFormat::Hvcc),
        "V_VP9" => (VideoCodec::Vp9, ConfigFormat::MatroskaVp9),
        "V_AV1" => (VideoCodec::Av1, ConfigFormat::Av1c),
        other => {
            return Err(DemuxError::unsupported(format!(
                "Matroska video codec {other}"
            )));
        }
    };
    let record = match (format, private) {
        (_, Some(record)) => record.to_vec(),
        // VP9 needs nothing before the first frame; its codec private data is
        // an optional feature list.
        (ConfigFormat::MatroskaVp9, None) => Vec::new(),
        (_, None) => {
            return Err(DemuxError::malformed(format!(
                "{codec_id} track without codec private data"
            )));
        }
    };
    Ok((codec, CodecConfig::new(format, record)))
}

/// A pixel dimension, as the `u32` every decoder takes.
fn side(pixels: NonZeroU64) -> Result<u32, DemuxError> {
    u32::try_from(pixels.get()).map_err(|_| DemuxError::malformed("video size out of range"))
}

/// The track timescale and the factor each raw timestamp is multiplied by.
///
/// The segment says how many nanoseconds one tick lasts (a millisecond by
/// default). When that divides a second the ticks are kept as they are, at
/// `1e9 / scale` per second; otherwise they are converted to nanoseconds, so
/// the timescale stays an exact integer either way.
pub(super) fn timescale(scale: NonZeroU64) -> (NonZeroU32, i64) {
    let nanos = NonZeroU32::new(1_000_000_000).unwrap_or(NonZeroU32::MIN);
    if NANOS_PER_SECOND.is_multiple_of(scale.get())
        && let Some(per_second) = u32::try_from(NANOS_PER_SECOND / scale.get())
            .ok()
            .and_then(NonZeroU32::new)
    {
        return (per_second, 1);
    }
    (nanos, i64::try_from(scale.get()).unwrap_or(i64::MAX))
}

/// The `(offset, length)` of every read made since the last [`Self::clear`].
#[derive(Debug, Clone, Default)]
pub(super) struct ReadLog(Rc<RefCell<Vec<(usize, usize)>>>);

impl ReadLog {
    /// Forgets the reads made so far, before the next frame is fetched.
    fn clear(&self) {
        self.0.borrow_mut().clear();
    }

    fn push(&self, start: usize, len: usize) {
        self.0.borrow_mut().push((start, len));
    }

    /// The range of `bytes` holding exactly `data`, among the logged reads;
    /// the latest match wins.
    pub(super) fn locate(&self, bytes: &[u8], data: &[u8]) -> Option<Range<usize>> {
        self.0.borrow().iter().rev().find_map(|&(start, len)| {
            let range = start..start.checked_add(len)?;
            (len == data.len() && bytes.get(range.clone()) == Some(data)).then_some(range)
        })
    }
}

/// The request buffer as the crate's reader, logging each read it makes.
#[derive(Debug)]
pub(super) struct Recorded<'a> {
    cursor: Cursor<&'a [u8]>,
    log: ReadLog,
}

impl<'a> Recorded<'a> {
    pub(super) fn new(bytes: &'a [u8], log: ReadLog) -> Self {
        Self {
            cursor: Cursor::new(bytes),
            log,
        }
    }
}

impl Read for Recorded<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let start = usize::try_from(self.cursor.position()).unwrap_or(usize::MAX);
        let n = self.cursor.read(buf)?;
        if n > 0 {
            self.log.push(start, n);
        }
        Ok(n)
    }
}

impl Seek for Recorded<'_> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.cursor.seek(pos)
    }
}

#[cfg(test)]
#[path = "mkv_tests.rs"]
mod tests;
