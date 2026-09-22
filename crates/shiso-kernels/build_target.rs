//! The device-compiler abstraction the nvcc lane drives.
//!
//! One trait so the orchestration in `build.rs` never names a compiler, and the
//! Linux lane can grow a second backend without the job planner learning about
//! it. The argument list itself is composed in `build_plan.rs`, where it can
//! be tested without a compiler on the machine; this file is only the process.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::build_plan::compose_nvcc_argv;

/// A compiler that turns one CUDA translation unit into one loadable artefact.
///
/// `Send + Sync` because the lane compiles jobs on several threads at once: a
/// clean build is dominated by template instantiation in a handful of units, so
/// the difference between sequential and parallel here is minutes.
pub(super) trait CompileTarget: Send + Sync {
    /// Extension of the artefact this target produces, without the dot.
    fn output_extension(&self) -> &'static str;

    /// Compile `source` for `arch` into `out`.
    ///
    /// Returns the diagnostic as an error string rather than printing it, so
    /// the caller can collect every failure in a parallel batch and report them
    /// together instead of racing them onto the build log.
    fn compile(
        &self,
        source: &Path,
        arch: &str,
        flags: &[String],
        out: &Path,
    ) -> Result<(), String>;
}

/// The NVIDIA compiler, emitting PTX.
#[derive(Debug)]
pub(super) struct NvccTarget {
    nvcc: PathBuf,
}

impl NvccTarget {
    /// Locate `nvcc`, or report that there is none.
    ///
    /// Returning `None` rather than failing is what lets a machine with no CUDA
    /// toolkit — which is every development machine in this project — build the
    /// crate: the caller turns a missing compiler into an empty registry and a
    /// log line. Only a build that explicitly asked for a real compile on a
    /// platform that ships one treats it as fatal.
    pub(super) fn find() -> Option<Self> {
        if let Some(home) = std::env::var_os("CUDA_HOME") {
            let candidate = PathBuf::from(home).join("bin/nvcc");
            if candidate.is_file() {
                return Some(Self { nvcc: candidate });
            }
        }
        for root in ["/usr/local/cuda", "/opt/cuda", "/usr"] {
            let candidate = PathBuf::from(root).join("bin/nvcc");
            if candidate.is_file() {
                return Some(Self { nvcc: candidate });
            }
        }
        // Last resort: whatever `nvcc` the PATH resolves. Deliberately last, so
        // a CUDA_HOME that was set on purpose is never shadowed by a stray one.
        which("nvcc").map(|nvcc| Self { nvcc })
    }
}

impl CompileTarget for NvccTarget {
    fn output_extension(&self) -> &'static str {
        "ptx"
    }

    fn compile(
        &self,
        source: &Path,
        arch: &str,
        flags: &[String],
        out: &Path,
    ) -> Result<(), String> {
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
        }
        let argv = compose_nvcc_argv(source, arch, flags, out);
        let output = Command::new(&self.nvcc)
            .args(&argv)
            .output()
            .map_err(|e| format!("cannot run {}: {e}", self.nvcc.display()))?;
        if output.status.success() {
            return Ok(());
        }
        Err(format!(
            "{} ({arch}) failed:\n{}",
            source.display(),
            String::from_utf8_lossy(&output.stderr).trim_end()
        ))
    }
}

/// The first executable named `name` on `PATH`.
fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}
