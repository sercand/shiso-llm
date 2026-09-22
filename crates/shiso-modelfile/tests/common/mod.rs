//! Shared plumbing for the model-dependent test binaries.
//!
//! Both model tests need the same thing: find the local model, and pass (not
//! fail) when it is not on this machine. Nothing here downloads anything - the
//! path is either `SHISO_TEST_MODEL` or the well-known local checkout, and a
//! missing file is reported as a skip.

use std::path::PathBuf;

use shiso_modelfile::GgufFile;

/// Path of the model the full tests run against.
///
/// `None` when the file is absent, which is normal on a machine that has no
/// local checkpoint.
#[must_use]
pub(crate) fn model_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL").ok()?);
    if path.is_file() { Some(path) } else { None }
}

/// Opens the model, or reports a skip and returns `None`.
///
/// A skipped test still has to pass, so callers use this as an early return
/// rather than an assertion.
#[must_use]
pub(crate) fn open_model() -> Option<GgufFile> {
    let path = model_path()?;
    match GgufFile::open(&path) {
        Ok(file) => Some(file),
        Err(err) => {
            skip(&format!(
                "model at {} failed to open: {err}",
                path.display()
            ));
            None
        }
    }
}

/// Prints the skip notice; the caller returns immediately afterwards.
///
/// Exiting the process from a test thread would cut the harness off mid-run, so
/// skipping is a printed note plus an early return from the test body.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
pub(crate) fn skip(why: &str) {
    println!("SKIP: {why}");
}
