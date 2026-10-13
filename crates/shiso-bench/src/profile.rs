//! Per-kernel GPU time for a bench phase.
//!
//! A profiled phase is never the timed one. A profile region submits every
//! launch on its own, so its numbers say where the time goes, not how long the
//! phase takes; a runner measures the phase batched first and prints both.
//! The same rule as the rest of this crate in a different form: no perf path
//! records a profile and no profile is a perf number.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use shiso_core::backend::Backend;
use shiso_core::error::CoreError;
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::model::DecodeStep;
use shiso_core::{ProfileRegion, ProfileReport};

use crate::decode::{Rig, RigError};

/// Why a profile could not be taken or written.
#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    /// The device cannot time launches one by one.
    #[error("the {backend} backend cannot time launches")]
    NoProfiler {
        /// The backend that was asked.
        backend: &'static str,
    },
    /// The device refused while the region was open.
    #[error(transparent)]
    Device(#[from] CoreError),
    /// The model refused.
    #[error(transparent)]
    Rig(#[from] RigError),
    /// The report could not be written.
    #[error("writing the profile to {path}")]
    Write {
        /// Where it was going.
        path: PathBuf,
        /// What the filesystem said.
        #[source]
        source: std::io::Error,
    },
}

/// Runs `f` inside a profile region on `backend` and returns its result with
/// the report.
///
/// # Errors
///
/// [`ProfileError::NoProfiler`] when `backend` cannot time launches, and
/// whatever `f` or the region refuses.
pub fn scoped<T, E: From<ProfileError>>(
    backend: &dyn Backend,
    f: impl FnOnce() -> Result<T, E>,
) -> Result<(T, ProfileReport), E> {
    let profiler = backend.as_profiler().ok_or(ProfileError::NoProfiler {
        backend: backend.backend_name(),
    })?;
    let region = ProfileRegion::open(profiler).map_err(ProfileError::from)?;
    let value = f()?;
    let report = region.finish().map_err(ProfileError::from)?;
    Ok((value, report))
}

/// Writes `{"context": context, "report": report}` to `path`, pretty-printed.
///
/// # Errors
///
/// [`ProfileError::Write`] when the file cannot be written.
pub fn write(path: &Path, context: &Value, report: &ProfileReport) -> Result<(), ProfileError> {
    let body = json!({ "context": context, "report": report.to_json() });
    let text = serde_json::to_string_pretty(&body).map_err(|error| ProfileError::Write {
        path: path.to_path_buf(),
        source: error.into(),
    })?;
    std::fs::write(path, text).map_err(|source| ProfileError::Write {
        path: path.to_path_buf(),
        source,
    })
}

/// Decode steps from the frontier: `warm` unprofiled, then `steps` inside one
/// region.
///
/// The prefill that reaches the frontier and the warm steps run outside the
/// region, so the report is the steady decode step only.
///
/// # Errors
///
/// Whatever the device or the model reports.
pub fn decode_profiled(
    rig: &mut Rig,
    warm: u32,
    steps: u32,
) -> Result<ProfileReport, ProfileError> {
    let (session, first) = rig.stand_at_frontier()?;
    let mut token = first;
    for _ in 0..warm {
        token = decode_one(rig, session, token)?;
    }
    let backend = std::sync::Arc::clone(&rig.backend);
    let ((), report) = scoped(backend.as_ref(), || -> Result<(), ProfileError> {
        for _ in 0..steps {
            token = decode_one(rig, session, token)?;
        }
        Ok(())
    })?;
    rig.model.close_session(session)?;
    Ok(report)
}

/// One step: the launch, the readback the next step's token comes from.
fn decode_one(rig: &mut Rig, session: SessionId, token: TokenId) -> Result<TokenId, ProfileError> {
    {
        let out = rig.model.decode_batch(&[DecodeStep { session, token }])?;
        rig.backend.copy_d2h(out.row(0)?, &mut rig.row)?;
    }
    Ok(rig.greedy())
}

/// One prefill of the rig's prompt inside a region, session open to close.
///
/// # Errors
///
/// Whatever the device or the model reports.
pub fn prefill_profiled(rig: &mut Rig) -> Result<ProfileReport, ProfileError> {
    let backend = std::sync::Arc::clone(&rig.backend);
    let ((), report) = scoped(backend.as_ref(), || -> Result<(), ProfileError> {
        let session = rig.model.open_session()?;
        {
            let last = rig.model.prefill_last(session, &rig.prompt)?;
            rig.backend.copy_d2h(last.row(0)?, &mut rig.row)?;
        }
        rig.model.close_session(session)?;
        Ok(())
    })?;
    Ok(report)
}
