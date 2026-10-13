//! `VideoToolbox`, not implemented yet: every call answers [`HwError::Unavailable`].

use crate::{HwDecoderDesc, HwError, HwFrame, HwProbe, HwTimestamp};

/// A `VideoToolbox` session.
#[derive(Debug)]
pub(crate) struct Session;

#[expect(
    clippy::unused_self,
    reason = "a stub until the real session lands; the signatures are the contract"
)]
impl Session {
    pub(crate) fn open(_desc: &HwDecoderDesc<'_>) -> Result<Self, HwError> {
        Err(unavailable())
    }

    pub(crate) fn decode(
        &mut self,
        _sample: &[u8],
        _pts: HwTimestamp,
        _keep: bool,
    ) -> Result<Vec<HwFrame>, HwError> {
        Err(unavailable())
    }

    pub(crate) fn flush(&mut self) -> Result<Vec<HwFrame>, HwError> {
        Err(unavailable())
    }
}

pub(crate) fn probe() -> Result<HwProbe, HwError> {
    Err(unavailable())
}

fn unavailable() -> HwError {
    HwError::Unavailable {
        reason: "VideoToolbox support is not built yet".into(),
    }
}
