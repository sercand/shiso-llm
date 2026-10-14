//! A platform with no OS video decoder: every call answers
//! [`HwError::Unavailable`], so the crate builds everywhere and the decode chain
//! falls through to the next step.

use crate::{HwDecoderDesc, HwError, HwFrame, HwProbe, HwTimestamp};

/// A session that cannot exist.
#[derive(Debug)]
pub(crate) struct Session;

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
        reason: "this platform has no OS video decoder".into(),
    }
}
