//! Container demux: from request bytes to one video track's coded samples.
//!
//! The in-process decoders (the OS decoder, the pure-Rust AV1 decoder) take
//! coded samples, not files, so something has to read the container first.
//! This module is that something, and it is always compiled: it links no codec
//! and makes no decode decision, it only says where each sample is, when it is
//! shown, and whether a decoder can start from it.
//!
//! # What is read
//!
//! - **MP4/MOV**, in house ([`open`] dispatches to it on an `ftyp` magic). The
//!   whole file is in memory, so the index may come before or after the media;
//!   moving it is a concern of the ffmpeg path only.
//! - **Matroska/WebM**, through the `matroska-demuxer` crate.
//!
//! Anything else, a fragmented MP4 and a file with no video track this module
//! reads included, is [`DemuxError::Unsupported`]: the chain hands those bytes
//! to the ffmpeg step untouched. A file that claims to be one of the two
//! containers and then does not parse is [`DemuxError::Malformed`], which the
//! chain refuses rather than retries, because a decoder that tolerates a
//! corrupt index is not evidence the index was right.
//!
//! # Samples are ranges, not copies
//!
//! A [`Sample`] names a byte range of the request buffer. An upload can be tens
//! of megabytes and the decoders read each sample exactly once, so a second
//! copy of the media would double the peak for nothing. The one exception is a
//! Matroska frame the demuxer crate hands back without a position this module
//! can verify (see [`SampleData`]); that frame keeps its own bytes, and
//! [`DemuxedTrack::sample_bytes`] hides the difference from every caller.
//!
//! # Order
//!
//! Samples are listed in DECODE order, the order a decoder must be fed. With
//! B-frames that is not the order frames are shown in; the presentation
//! timestamp ([`Sample::pts`]) is what says where a frame lands on screen, and
//! [`SamplePlan`] is where the two orders are reconciled.

mod annexb;
mod mkv;
mod mp4;
mod plan;

use std::num::NonZeroU32;
use std::ops::Range;

pub use annexb::{avcc_to_annexb, parameter_sets_annexb};
pub use plan::SamplePlan;

use crate::error::Container;
use crate::video::sniff_container;

/// The most samples one track may declare.
///
/// About nine and a half hours at 30 frames per second, far past anything the
/// sampler will keep (it stops at a few hundred frames), and small enough that
/// the per-sample tables (a range, a timestamp and a flag each, plus the
/// sampler's reorder table) stay in the tens of megabytes. A track over the cap
/// is [`DemuxError::Unsupported`] rather than malformed: it is a legitimate,
/// just very long, file, and the ffmpeg step samples it with its own caps.
pub const MAX_SAMPLES: usize = 1 << 20;

/// Why a container could not be demuxed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DemuxError {
    /// A well-formed input this demuxer does not read: AVI, a fragmented MP4,
    /// no readable video track, an unknown codec, a track over
    /// [`MAX_SAMPLES`]. The decode chain passes these to the next step.
    #[error("demux unsupported: {reason}")]
    Unsupported {
        /// What was not supported, for the log and the error message.
        reason: String,
    },
    /// The container claims a format and then breaks its rules: a box that
    /// overruns its parent, a sample table shorter than its count, a sample
    /// outside the buffer. The decode chain refuses these.
    #[error("malformed container: {reason}")]
    Malformed {
        /// What was broken, for the log and the error message.
        reason: String,
    },
}

impl DemuxError {
    /// An [`DemuxError::Unsupported`] with `reason`.
    pub(crate) fn unsupported(reason: impl Into<String>) -> Self {
        Self::Unsupported {
            reason: reason.into(),
        }
    }

    /// A [`DemuxError::Malformed`] with `reason`.
    pub(crate) fn malformed(reason: impl Into<String>) -> Self {
        Self::Malformed {
            reason: reason.into(),
        }
    }
}

/// The video codecs the in-process decoders may be offered.
///
/// H.264 and HEVC are here because the OS decoders take them; nothing in this
/// crate decodes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VideoCodec {
    /// H.264 / AVC.
    H264,
    /// H.265 / HEVC.
    Hevc,
    /// VP9.
    Vp9,
    /// AV1.
    Av1,
}

impl std::fmt::Display for VideoCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::H264 => "h264",
            Self::Hevc => "hevc",
            Self::Vp9 => "vp9",
            Self::Av1 => "av1",
        })
    }
}

/// Which record a [`CodecConfig`] holds.
///
/// The bytes alone do not say: a Matroska VP9 track's codec private data is a
/// different structure from an MP4 `vpcC` box, and a decoder that parsed one as
/// the other would read garbage profile bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigFormat {
    /// An `AVCDecoderConfigurationRecord` (`avcC`): H.264 parameter sets and
    /// the NAL length size.
    Avcc,
    /// An `HEVCDecoderConfigurationRecord` (`hvcC`): HEVC parameter sets and
    /// the NAL length size.
    Hvcc,
    /// The body of an MP4 `vpcC` box, starting with its full-box version and
    /// flags.
    Vpcc,
    /// An `AV1CodecConfigurationRecord` (`av1C`), the same record in both
    /// containers.
    Av1c,
    /// A Matroska `V_VP9` track's codec private data: an optional list of
    /// codec features, often absent, so the record may be empty.
    MatroskaVp9,
}

/// The codec configuration record a decoder needs before the first sample.
///
/// Owned: it is a few hundred bytes at most, and a decoder session outlives
/// the parse that found it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecConfig {
    format: ConfigFormat,
    record: Vec<u8>,
}

impl CodecConfig {
    /// Wraps `record`, which must be in `format`.
    #[must_use]
    pub fn new(format: ConfigFormat, record: Vec<u8>) -> Self {
        Self { format, record }
    }

    /// Which record this is.
    #[must_use]
    pub fn format(&self) -> ConfigFormat {
        self.format
    }

    /// The raw record bytes, as stored in the container.
    #[must_use]
    pub fn record(&self) -> &[u8] {
        &self.record
    }

    /// Bytes in each sample's NAL length prefix (1, 2 or 4), for the
    /// length-prefixed H.264 and HEVC formats.
    ///
    /// # Errors
    ///
    /// [`DemuxError::Unsupported`] for a format whose samples are not
    /// length-prefixed NAL units (VP9, AV1); [`DemuxError::Malformed`] when the
    /// record is too short to hold the field or declares a size of 3, which
    /// neither record format allows.
    pub fn nal_length_size(&self) -> Result<usize, DemuxError> {
        // `lengthSizeMinusOne` is the low two bits of byte 4 in `avcC` and of
        // byte 21 in `hvcC`.
        let at = match self.format {
            ConfigFormat::Avcc => 4,
            ConfigFormat::Hvcc => 21,
            other => {
                return Err(DemuxError::unsupported(format!(
                    "{other:?} samples carry no NAL length prefix"
                )));
            }
        };
        let byte = self.record.get(at).ok_or_else(|| {
            DemuxError::malformed("codec config too short for its NAL length size")
        })?;
        match (byte & 0b11) + 1 {
            3 => Err(DemuxError::malformed(
                "codec config declares a 3-byte NAL length",
            )),
            size => Ok(usize::from(size)),
        }
    }
}

/// A presentation timestamp, in the track's timescale ticks.
///
/// Signed because a composition offset may move a frame before the decode
/// time of the first sample. A newtype so a tick count is never mistaken for a
/// sample index or a frame count, which are also integers on the same paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pts(i64);

impl Pts {
    /// A timestamp of `ticks` timescale units.
    #[must_use]
    pub fn new(ticks: i64) -> Self {
        Self(ticks)
    }

    /// The tick count.
    #[must_use]
    pub fn ticks(self) -> i64 {
        self.0
    }
}

/// Where a sample's bytes live.
///
/// MP4 samples are always [`SampleData::Borrowed`]. A Matroska frame is too
/// whenever this module can locate it in the request buffer and the bytes
/// there match what the demuxer crate returned, which is every frame the crate
/// reads with a single read (all of them, at the version pinned here).
/// [`SampleData::Owned`] is the fallback for a frame it cannot locate, so a
/// change in the crate's read pattern costs a copy, not a wrong sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SampleData {
    /// A range of the request buffer, checked to lie inside it.
    Borrowed(Range<usize>),
    /// The sample's own bytes.
    Owned(Vec<u8>),
}

/// One coded sample (an access unit, a temporal unit, a frame).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sample {
    data: SampleData,
    pts: Pts,
    keyframe: bool,
}

impl Sample {
    /// Where the bytes live. Read them through [`DemuxedTrack::sample_bytes`].
    #[must_use]
    pub fn data(&self) -> &SampleData {
        &self.data
    }

    /// When the sample is shown, in the track's timescale.
    #[must_use]
    pub fn pts(&self) -> Pts {
        self.pts
    }

    /// Whether a decoder may start from this sample.
    ///
    /// For MP4 this is the sync-sample table, or every sample when the file has
    /// none. For Matroska it is the `SimpleBlock` keyframe flag; a frame stored
    /// in a `BlockGroup` carries no such flag the demuxer crate exposes and is
    /// reported as `false`, which costs a caller that seeks a later start point
    /// and nothing else (the decoders here always start from the first sample).
    #[must_use]
    pub fn keyframe(&self) -> bool {
        self.keyframe
    }
}

/// One video track, demuxed: what codec, how big, and its samples in decode
/// order.
#[derive(Debug, Clone)]
pub struct DemuxedTrack<'a> {
    bytes: &'a [u8],
    container: Container,
    codec: VideoCodec,
    config: CodecConfig,
    width: u32,
    height: u32,
    timescale: NonZeroU32,
    samples: Vec<Sample>,
}

impl DemuxedTrack<'_> {
    /// The container the track was read from.
    #[must_use]
    pub fn container(&self) -> Container {
        self.container
    }

    /// The track's codec.
    #[must_use]
    pub fn codec(&self) -> VideoCodec {
        self.codec
    }

    /// The codec configuration record.
    #[must_use]
    pub fn config(&self) -> &CodecConfig {
        &self.config
    }

    /// Coded width in pixels, as the container declares it.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Coded height in pixels, as the container declares it.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Timestamp ticks per second.
    #[must_use]
    pub fn timescale(&self) -> NonZeroU32 {
        self.timescale
    }

    /// The samples, in decode order.
    #[must_use]
    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    /// The coded bytes of sample `index`, or [`None`] when there is no such
    /// sample.
    #[must_use]
    pub fn sample_bytes(&self, index: usize) -> Option<&[u8]> {
        match &self.samples.get(index)?.data {
            SampleData::Borrowed(range) => self.bytes.get(range.clone()),
            SampleData::Owned(bytes) => Some(bytes),
        }
    }
}

/// Reads the first video track of `bytes`.
///
/// Dispatches on the magic bytes, as [`crate::decode_clip`] does: an `ftyp`
/// box is MP4/MOV, an EBML header is Matroska/WebM.
///
/// # Errors
///
/// [`DemuxError::Unsupported`] for any other container (AVI, GIF, unknown), a
/// fragmented MP4, a file with no video track or an unknown codec, or a track
/// over [`MAX_SAMPLES`]; [`DemuxError::Malformed`] when the container's own
/// structure is broken, including a sample outside the buffer.
pub fn open(bytes: &[u8]) -> Result<DemuxedTrack<'_>, DemuxError> {
    let container = sniff_container(bytes);
    let track = match container {
        Container::Mp4 => mp4::open(bytes)?,
        Container::Matroska => mkv::open(bytes)?,
        other => {
            return Err(DemuxError::unsupported(format!(
                "{other} is not a container this demuxer reads"
            )));
        }
    };
    Ok(DemuxedTrack {
        bytes,
        container,
        codec: track.codec,
        config: track.config,
        width: track.width,
        height: track.height,
        timescale: track.timescale,
        samples: track.samples,
    })
}

/// What a container reader produces; [`open`] adds the buffer and container.
#[derive(Debug)]
struct ParsedTrack {
    codec: VideoCodec,
    config: CodecConfig,
    width: u32,
    height: u32,
    timescale: NonZeroU32,
    samples: Vec<Sample>,
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
