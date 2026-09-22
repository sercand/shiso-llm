//! The resume-latency gate.
//!
//! # This module measures nothing
//!
//! The checkpoint bench owns the resume protocol: it prefills a prompt cold,
//! stores the prefix, resumes onto it, and checks that both runs continue with
//! byte-identical tokens. That bench is the only thing that produces these
//! numbers, and re-deriving them here would mean two definitions of what a
//! resume costs. So this module reads that file and grades it.
//!
//! # There is no external number to compare against
//!
//! The implementation this port follows keeps its snapshot save and restore
//! deliberately outside both of its timing windows, and its snapshot is an
//! in-memory blob rather than the on-disk checkpoint this port resumes from.
//! There is therefore no upstream resume latency, and inventing one would be
//! worse than having none. The gate is entirely self-referential in the timing
//! and absolute in the invariants:
//!
//! - the ratio, against this port's own recorded resume time;
//! - `resume < cold prefill`, because a resume that costs more than the prefill
//!   it replaces has no reason to exist;
//! - the continuation was byte-identical, which the recording bench checks and
//!   the gate refuses to grade a timing without. A resume that is fast and
//!   different is not a fast resume.

use std::path::{Path, PathBuf};

use crate::baseline::{Baseline, Comparison, Invariant, Report, ResumeEntry};

/// What can stop the gate from reading a recorded file.
#[derive(Debug, thiserror::Error)]
pub enum KvError {
    /// The file is not where the gate expects it.
    #[error("no resume baseline at {path}")]
    Missing {
        /// The path that was tried.
        path: PathBuf,
    },
    /// The file is there but is not the shape the gate reads.
    #[error("the resume baseline at {path} does not parse")]
    Malformed {
        /// The file that would not parse.
        path: PathBuf,
        /// What the parser said.
        #[source]
        source: serde_json::Error,
    },
}

/// Reads one machine-keyed resume file.
///
/// # Errors
///
/// [`KvError::Missing`] when the file is absent and [`KvError::Malformed`] when
/// it will not parse.
pub fn read(path: &Path) -> Result<Baseline<ResumeEntry>, KvError> {
    let text = std::fs::read_to_string(path).map_err(|_| KvError::Missing {
        path: path.to_path_buf(),
    })?;
    serde_json::from_str(&text).map_err(|source| KvError::Malformed {
        path: path.to_path_buf(),
        source,
    })
}

/// Grades one resume record against its baseline entry.
///
/// The invariants are pushed whether or not the ratio passes: a report that
/// stopped at the first failure would hide the one fact worth knowing, which is
/// whether the resumed run said the same thing.
pub fn grade(record: &ResumeEntry, baseline: &ResumeEntry, report: &mut Report) {
    report.comparisons.push(Comparison {
        name: "resume ms".to_owned(),
        baseline: baseline.resume_ms,
        measured: record.resume_ms,
    });
    report.invariants.push(Invariant {
        name: "the resume costs less than the cold prefill it replaces".to_owned(),
        held: record.resume_ms < record.cold_prefill_ms,
    });
    report.invariants.push(Invariant {
        name: "both runs continued with identical tokens".to_owned(),
        held: record.continuation_tokens_equal,
    });
}
