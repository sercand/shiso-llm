//! The types every decoder shares.

/// A video codec an OS decoder may handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HwCodec {
    /// H.264 / AVC.
    H264,
    /// H.265 / HEVC.
    Hevc,
    /// VP9.
    Vp9,
    /// AV1.
    Av1,
}

impl std::fmt::Display for HwCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::H264 => "h264",
            Self::Hevc => "hevc",
            Self::Vp9 => "vp9",
            Self::Av1 => "av1",
        })
    }
}

/// What a decoder is opened with.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct HwDecoderDesc<'a> {
    /// The stream's codec.
    pub codec: HwCodec,
    /// The container's codec configuration record: `avcC`, `hvcC`, `vpcC` or
    /// `av1C`, or a Matroska `CodecPrivate`, exactly as stored.
    pub config: &'a [u8],
    /// Coded width in pixels.
    pub width: u32,
    /// Coded height in pixels.
    pub height: u32,
}

impl<'a> HwDecoderDesc<'a> {
    /// Describes a stream.
    #[must_use]
    pub const fn new(codec: HwCodec, config: &'a [u8], width: u32, height: u32) -> Self {
        Self {
            codec,
            config,
            width,
            height,
        }
    }
}

/// A presentation timestamp in the track's timescale ticks.
///
/// A newtype so a timestamp cannot be passed where a size or an index is
/// expected, and so the unit (ticks, not seconds) is part of the type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HwTimestamp(i64);

impl HwTimestamp {
    /// A timestamp of `ticks` timescale ticks.
    #[must_use]
    pub const fn new(ticks: i64) -> Self {
        Self(ticks)
    }

    /// The tick count.
    #[must_use]
    pub const fn ticks(self) -> i64 {
        self.0
    }
}

/// One decoded frame: packed RGB8, three bytes per pixel, row-major.
#[derive(Clone, PartialEq, Eq)]
pub struct HwFrame {
    width: u32,
    height: u32,
    rgb: Vec<u8>,
    pts: HwTimestamp,
}

impl HwFrame {
    /// Wraps a packed RGB8 buffer.
    ///
    /// # Errors
    ///
    /// [`HwError::Failed`] when the buffer does not hold exactly
    /// `width * height * 3` bytes: a stride mistake in a decoder must not
    /// become a garbled frame that decodes, encodes and answers.
    pub fn new(width: u32, height: u32, rgb: Vec<u8>, pts: HwTimestamp) -> Result<Self, HwError> {
        let expected = usize::try_from(u64::from(width) * u64::from(height) * 3).ok();
        if expected != Some(rgb.len()) {
            return Err(HwError::Failed {
                detail: format!(
                    "a {width}x{height} frame needs {} bytes of RGB, got {}",
                    u64::from(width) * u64::from(height) * 3,
                    rgb.len()
                ),
            });
        }
        Ok(Self {
            width,
            height,
            rgb,
            pts,
        })
    }

    /// Width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The packed RGB8 pixels.
    #[must_use]
    pub fn rgb(&self) -> &[u8] {
        &self.rgb
    }

    /// Takes the pixels.
    #[must_use]
    pub fn into_rgb(self) -> Vec<u8> {
        self.rgb
    }

    /// The frame's presentation timestamp.
    #[must_use]
    pub fn pts(&self) -> HwTimestamp {
        self.pts
    }
}

/// Summarised rather than derived; a frame is megabytes of pixel data.
impl std::fmt::Debug for HwFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HwFrame")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("pts", &self.pts)
            .finish_non_exhaustive()
    }
}

/// Why a decoder could not be opened or fed.
///
/// Only [`HwError::Unavailable`] and [`HwError::Unsupported`] let a caller try
/// the next decoder. [`HwError::Failed`] means the decoder was there and broke on
/// this input, and retrying the same bytes elsewhere would hide a corrupt upload
/// behind whichever decoder tolerates it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum HwError {
    /// This machine has no OS decoder: no framework, no driver library, no
    /// device.
    #[error("no OS video decoder is available: {reason}")]
    Unavailable {
        /// What was missing.
        reason: String,
    },
    /// There is a decoder but it does not handle this codec, size, bit depth or
    /// profile.
    #[error("the OS decoder does not support {codec} here: {reason}")]
    Unsupported {
        /// The codec that was asked for.
        codec: HwCodec,
        /// What the decoder objected to.
        reason: String,
    },
    /// The decoder was there and failed on this input.
    #[error("the video decoder failed: {detail}")]
    Failed {
        /// The decoder's own complaint.
        detail: String,
    },
}

/// What this machine's OS decoder reported at boot.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct HwProbe {
    /// The decoder's name, such as `VideoToolbox` or `NVDEC`.
    pub decoder: &'static str,
    /// The codecs it reports it can decode.
    pub codecs: Vec<HwCodec>,
}
