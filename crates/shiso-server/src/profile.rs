//! `--profile-dir`: one GPU-time-per-kernel report per request.
//!
//! Each service that runs device work for a request opens a region on its
//! own backend before the work and writes the report after it. The services
//! open separate backend instances, so a report holds only its own service's
//! kernels; profiling one kind of request at a time is what keeps the GPU
//! times themselves undisturbed.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context as _;
use shiso_core::profile::elapsed_ns;
use shiso_core::{Backend, ProfileReport, Profiler};

/// Where reports go, and the device whose launches they time.
#[derive(Debug, Clone)]
pub struct ProfileSink {
    dir: Arc<Path>,
    backend: Arc<dyn Backend>,
}

/// A region the sink opened and has not written yet.
#[derive(Debug)]
pub struct OpenRegion {
    started: Instant,
    label: String,
}

impl OpenRegion {
    /// Names the report after an id minted once the region was already open.
    pub fn relabel(&mut self, label: String) {
        self.label = label;
    }
}

impl ProfileSink {
    /// A sink writing into `dir`, which is created if absent.
    ///
    /// # Errors
    ///
    /// When `backend` cannot time launches, or `dir` cannot be created.
    pub fn new(dir: &Path, backend: Arc<dyn Backend>) -> anyhow::Result<Self> {
        if backend.as_profiler().is_none() {
            anyhow::bail!(
                "--profile-dir: the {} backend cannot time launches",
                backend.backend_name()
            );
        }
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating --profile-dir {}", dir.display()))?;
        Ok(Self {
            dir: Arc::from(dir),
            backend,
        })
    }

    fn profiler(&self) -> &dyn Profiler {
        self.backend
            .as_profiler()
            .expect("`new` refused a backend without a profiler")
    }

    /// Opens a region; `None`, logged, when the backend refuses.
    pub fn begin(&self, label: String) -> Option<OpenRegion> {
        match self.profiler().start() {
            Ok(()) => Some(OpenRegion {
                started: Instant::now(),
                label,
            }),
            Err(error) => {
                tracing::warn!(error = %error.chain(), "a profile region did not open");
                None
            }
        }
    }

    /// Closes `region` and writes its report; returns the path written, or
    /// `None` after logging why nothing was.
    ///
    /// A failure here never fails the request: the report is a side channel.
    pub fn end(&self, region: OpenRegion, kind: &str) -> Option<PathBuf> {
        let OpenRegion { started, label } = region;
        let samples = match self.profiler().stop() {
            Ok(samples) => samples,
            Err(error) => {
                tracing::warn!(error = %error.chain(), "a profile region did not close cleanly");
                return None;
            }
        };
        let report = ProfileReport::from_samples(&samples, elapsed_ns(started));
        let path = self
            .dir
            .join(format!("{kind}-{}-{label}.json", std::process::id()));
        let body = serde_json::json!({
            "kind": kind,
            "id": label,
            "report": report.to_json(),
        });
        let written = serde_json::to_string_pretty(&body)
            .map_err(std::io::Error::from)
            .and_then(|text| std::fs::write(&path, text));
        if let Err(error) = written {
            tracing::warn!(%error, path = %path.display(), "a profile report was not written");
            return None;
        }
        tracing::info!(
            path = %path.display(),
            launches = report.launches,
            serialized_ns = report.serialized_ns,
            "wrote a profile report"
        );
        Some(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_relabelled_region_is_written_under_its_new_name() {
        let mut region = OpenRegion {
            started: Instant::now(),
            label: "pending-1".to_owned(),
        };
        region.relabel("00000000000000aa".to_owned());
        assert_eq!(region.label, "00000000000000aa");
    }
}
