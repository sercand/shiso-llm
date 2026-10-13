//! OS video decoders behind a safe API.
//!
//! Every decoder a server reaches for that lives outside the process's own safe
//! code is here, so that `unsafe` is confined to one crate and the crates above
//! it (`shiso-vision`, the server) stay `forbid(unsafe_code)`: `VideoToolbox` on
//! macOS, the NVDEC driver library on Linux, and, with `decode-native`, the C-ABI
//! of the AV1 decoder. A decoder that is not there is a typed outcome
//! ([`HwError::Unavailable`]), never a panic or a link error: the OS libraries
//! are reached at run time on Linux and through system frameworks on macOS, so a
//! build is the same binary on a machine with the hardware and one without.
//!
//! The API is the same for every decoder. A [`HwDecoder`] is opened from the
//! container's codec configuration record, fed one container sample at a time
//! in decode order, and hands back finished frames in display order as packed
//! RGB8. Dropping it releases every OS handle it holds.
//!
//! What is deliberately NOT here: demuxing (that is `shiso-vision`), sampling
//! policy (a caller passes `keep = false` for frames it will drop, so they are
//! decoded - later frames may reference them - but not colour-converted), and any
//! dependency on a compute backend. This crate names neither `shiso-cuda` nor
//! `shiso-metal`.
#![allow(
    unsafe_code,
    reason = "OS video decoder FFI: VideoToolbox through objc2 on macOS, the NVDEC driver \
              library through dlopen on Linux, and the C-ABI call into the AV1 decoder"
)]

#[cfg(feature = "decode-native")]
mod av1;
pub mod colour;
mod platform;
mod types;

#[cfg(feature = "decode-native")]
pub use av1::Av1Decoder;
pub use types::{HwCodec, HwDecoderDesc, HwError, HwFrame, HwProbe, HwTimestamp};

/// An OS hardware or system video decoder session.
///
/// One session decodes one elementary stream. It is not `Sync`: the platform
/// decoders are driven from one thread at a time, and a server that decodes on
/// several request threads opens a session per request.
#[derive(Debug)]
pub struct HwDecoder {
    inner: platform::Session,
}

impl HwDecoder {
    /// Opens a decoder for the stream `desc` describes.
    ///
    /// # Errors
    ///
    /// [`HwError::Unavailable`] when this machine has no OS decoder at all;
    /// [`HwError::Unsupported`] when it has one but not for this codec, size,
    /// bit depth or profile - the caller falls through to the next decoder on
    /// either; [`HwError::Failed`] when the decoder was there and broke.
    pub fn open(desc: &HwDecoderDesc<'_>) -> Result<Self, HwError> {
        platform::Session::open(desc).map(|inner| Self { inner })
    }

    /// Decodes one container sample, in decode order.
    ///
    /// Samples are as the container stores them: length-prefixed NAL units for
    /// H.264 and HEVC, a low-overhead OBU sequence for AV1. Returns the frames
    /// that became ready, in display order; a decoder that reorders returns
    /// fewer than it was given until [`HwDecoder::flush`]. With `keep = false` the sample is decoded
    /// but its frame is not converted to RGB, so it is not returned.
    ///
    /// # Errors
    ///
    /// [`HwError::Failed`] when the decoder rejects the sample.
    pub fn decode(
        &mut self,
        sample: &[u8],
        pts: HwTimestamp,
        keep: bool,
    ) -> Result<Vec<HwFrame>, HwError> {
        self.inner.decode(sample, pts, keep)
    }

    /// Emits every frame the decoder still holds, in display order.
    ///
    /// # Errors
    ///
    /// [`HwError::Failed`] when the decoder fails while draining.
    pub fn flush(&mut self) -> Result<Vec<HwFrame>, HwError> {
        self.inner.flush()
    }
}

/// Reports what this machine's OS decoder can do, for the boot log.
///
/// # Errors
///
/// [`HwError::Unavailable`] when there is none.
pub fn probe() -> Result<HwProbe, HwError> {
    platform::probe()
}
