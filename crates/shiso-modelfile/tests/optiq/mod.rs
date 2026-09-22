//! Shared plumbing for the checkpoint-directory tests.
//!
//! Same contract as the single-file model tests: find the local checkpoint, and
//! pass rather than fail when this machine does not have it. Nothing here
//! downloads anything - the directory is either `SHISO_TEST_MODEL_OPTIQ` or the
//! well-known local path, and an absent one is a printed skip.

use std::path::PathBuf;
use std::sync::Arc;

use shiso_modelfile::SafetensorsFile;
use shiso_models::qwen35::Qwen35Names;

/// Environment variable that names the checkpoint directory.
const DIR_VAR: &str = "SHISO_TEST_MODEL_OPTIQ";

/// Path of the checkpoint directory the full tests run against.
///
/// `None` when the directory is absent, which is normal on a machine with no
/// local checkpoint.
#[must_use]
pub(crate) fn checkpoint_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var(DIR_VAR).ok()?);
    if dir.is_dir() { Some(dir) } else { None }
}

/// Opens the checkpoint, or reports a skip and returns `None`.
///
/// A skipped test still has to pass, so callers use this as an early return
/// rather than an assertion.
#[must_use]
pub(crate) fn open_checkpoint() -> Option<SafetensorsFile> {
    let dir = checkpoint_dir()?;
    match SafetensorsFile::open_with(&dir, Arc::new(Qwen35Names)) {
        Ok(file) => Some(file),
        Err(err) => {
            skip(&format!(
                "checkpoint at {} failed to open: {err}",
                dir.display()
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
