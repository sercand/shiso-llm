//! The AV1 decoder, for machines whose OS has none.

use crate::{HwDecoderDesc, HwError, HwFrame, HwTimestamp};

/// An AV1 decode session.
#[derive(Debug)]
pub struct Av1Decoder;

impl Av1Decoder {
    /// Opens a decoder for an AV1 stream.
    ///
    /// # Errors
    ///
    /// [`HwError::Unsupported`] for anything but 8- and 10-bit 4:2:0.
    pub fn open(_desc: &HwDecoderDesc<'_>) -> Result<Self, HwError> {
        Err(HwError::Unavailable {
            reason: "the AV1 decoder is not built yet".into(),
        })
    }

    /// Decodes one container sample, in decode order.
    ///
    /// # Errors
    ///
    /// [`HwError::Failed`] when the decoder rejects the sample.
    pub fn decode(
        &mut self,
        _sample: &[u8],
        _pts: HwTimestamp,
        _keep: bool,
    ) -> Result<Vec<HwFrame>, HwError> {
        Err(HwError::Unavailable {
            reason: "the AV1 decoder is not built yet".into(),
        })
    }

    /// Emits every frame the decoder still holds, in display order.
    ///
    /// # Errors
    ///
    /// [`HwError::Failed`] when the decoder fails while draining.
    pub fn flush(&mut self) -> Result<Vec<HwFrame>, HwError> {
        Err(HwError::Unavailable {
            reason: "the AV1 decoder is not built yet".into(),
        })
    }
}
