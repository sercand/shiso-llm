//! Host hardware probing and capability profiling.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sysinfo::{RefreshKind, System};

use crate::error::HubError;

/// Supported hardware execution backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// Apple Silicon Metal backend.
    Metal,
    /// NVIDIA CUDA backend.
    Cuda,
}

impl Backend {
    /// Returns the backend as a string slice (`"metal"` or `"cuda"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Information about available CUDA hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaInfo {
    /// Compute capability (e.g. 89 for Ada Lovelace, 100 for Blackwell).
    pub sm: u32,
    /// Total VRAM across usable devices in bytes.
    pub vram_bytes: u64,
}

/// Information about the Metal device.
///
/// Keyed on the Apple GPU family rather than on the chip's name: kernels are
/// written against a family's feature set, and an M1 Pro, Max and Ultra share
/// the M1's family. What tells those apart for a model is memory, which
/// [`MetalInfo::working_set_bytes`] answers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetalInfo {
    /// The device's name (e.g. `Apple M1 Max`), for display only.
    pub name: String,
    /// Highest Apple GPU family supported: 7 = M1, 8 = M2, 9 = M3/M4, 10 = M5.
    pub family: u32,
    /// Working set the device recommends for one process, in bytes.
    pub working_set_bytes: u64,
}

/// Host profile describing available backends and hardware memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostProfile {
    /// Hardware backends compiled into this binary.
    pub backends: Vec<Backend>,
    /// Total physical RAM in bytes.
    pub ram_bytes: u64,
    /// CUDA hardware facts, when available.
    pub cuda: Option<CudaInfo>,
    /// Metal device facts, when available.
    pub metal: Option<MetalInfo>,
}

impl HostProfile {
    /// Creates a host profile with probed system RAM and caller-supplied backends.
    #[must_use]
    pub fn probe(backends: Vec<Backend>, cuda: Option<CudaInfo>, metal: Option<MetalInfo>) -> Self {
        Self {
            backends,
            ram_bytes: probe_ram(),
            cuda,
            metal,
        }
    }
}

/// Probes the system for total physical RAM in bytes.
#[must_use]
pub fn probe_ram() -> u64 {
    let mut sys = System::new_with_specifics(
        RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::nothing().with_ram()),
    );
    sys.refresh_memory();
    sys.total_memory()
}

/// Probes available free disk space on the filesystem containing `path`.
///
/// Asks the filesystem of the nearest existing ancestor directly. Enumerating
/// every mount instead took 25s on a developer machine with a few disk images
/// and network volumes attached, once per download.
///
/// # Errors
/// Returns [`HubError::Io`] if free disk space cannot be determined.
pub fn probe_free_disk(path: &Path) -> Result<u64, HubError> {
    let existing = path
        .ancestors()
        .find(|candidate| candidate.exists())
        .unwrap_or_else(|| Path::new("."));
    let stat = rustix::fs::statvfs(existing).map_err(|e| HubError::Io {
        path: existing.to_path_buf(),
        source: e.into(),
    })?;
    Ok(stat.f_bavail.saturating_mul(stat.f_frsize))
}
