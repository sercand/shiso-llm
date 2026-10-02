//! Host hardware probing and capability profiling.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sysinfo::{Disks, RefreshKind, System};

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

/// Host profile describing available backends and hardware memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostProfile {
    /// Hardware backends compiled into this binary.
    pub backends: Vec<Backend>,
    /// Total physical RAM in bytes.
    pub ram_bytes: u64,
    /// CUDA hardware facts, when available.
    pub cuda: Option<CudaInfo>,
}

impl HostProfile {
    /// Creates a host profile with probed system RAM and caller-supplied backends.
    #[must_use]
    pub fn probe(backends: Vec<Backend>, cuda: Option<CudaInfo>) -> Self {
        Self {
            backends,
            ram_bytes: probe_ram(),
            cuda,
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
/// # Errors
/// Returns [`HubError::Io`] if free disk space cannot be determined.
pub fn probe_free_disk(path: &Path) -> Result<u64, HubError> {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let disks = Disks::new_with_refreshed_list();

    // Find the disk mount with the longest matching prefix
    let mut best_match: Option<(usize, u64)> = None;
    for disk in &disks {
        let mount_point = disk.mount_point();
        if canonical.starts_with(mount_point) {
            let match_len = mount_point.as_os_str().len();
            if best_match.as_ref().is_none_or(|(len, _)| match_len > *len) {
                best_match = Some((match_len, disk.available_space()));
            }
        }
    }

    if let Some((_, space)) = best_match {
        Ok(space)
    } else if let Some(first) = disks.first() {
        Ok(first.available_space())
    } else {
        Err(HubError::Other(
            "could not determine free disk space".to_string(),
        ))
    }
}
