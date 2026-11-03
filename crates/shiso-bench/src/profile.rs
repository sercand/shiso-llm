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
/// region, with `streams` sessions advancing together.
///
/// The prefill that reaches the frontier and the warm steps run outside the
/// region, so the report is the steady decode step only. Every stream past the
/// first stands at the end of a salted prompt of its own, as the batched
/// throughput phase does, so the rows of a step are different sessions at
/// different positions rather than copies of one.
///
/// # Errors
///
/// Whatever the device or the model reports.
pub fn decode_profiled(
    rig: &mut Rig,
    warm: u32,
    steps: u32,
    streams: u32,
) -> Result<ProfileReport, ProfileError> {
    let mut sessions = Vec::new();
    let mut tokens = Vec::new();
    for row in 0..streams.max(1) {
        let (session, first) = if row == 0 {
            rig.stand_at_frontier()?
        } else {
            rig.stand_at_frontier_salted(row)?
        };
        sessions.push(session);
        tokens.push(first);
    }
    for _ in 0..warm {
        decode_rows(rig, &sessions, &mut tokens)?;
    }
    let backend = std::sync::Arc::clone(&rig.backend);
    let ((), report) = scoped(backend.as_ref(), || -> Result<(), ProfileError> {
        for _ in 0..steps {
            decode_rows(rig, &sessions, &mut tokens)?;
        }
        Ok(())
    })?;
    for session in sessions {
        rig.model.close_session(session)?;
    }
    Ok(report)
}

/// One step over every session: the launch, then each row's readback, which is
/// where each row's next token comes from.
fn decode_rows(
    rig: &mut Rig,
    sessions: &[SessionId],
    tokens: &mut [TokenId],
) -> Result<(), ProfileError> {
    let batch: Vec<DecodeStep> = sessions
        .iter()
        .zip(tokens.iter())
        .map(|(session, token)| DecodeStep {
            session: *session,
            token: *token,
        })
        .collect();
    let mut host = vec![vec![0u8; rig.row.len()]; batch.len()];
    {
        let out = rig.model.decode_batch(&batch)?;
        for (row, dst) in host.iter_mut().enumerate() {
            let row = u32::try_from(row).unwrap_or(u32::MAX);
            rig.backend.copy_d2h(out.row(row)?, dst)?;
        }
    }
    for (token, bytes) in tokens.iter_mut().zip(&host) {
        rig.row.copy_from_slice(bytes);
        *token = rig.greedy();
    }
    Ok(())
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
