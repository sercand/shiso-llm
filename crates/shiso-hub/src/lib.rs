//! shiso-hub - model repository client, digest-verified downloads and blob store.
//!
//! Provides recipe manifests, HTTPS static registry client, variant selection,
//! resumable SHA-256-verified downloads and content-addressed blob storage.
#![forbid(unsafe_code)]

pub mod config;
pub mod download;
pub mod error;
pub mod host;
pub mod progress;
pub mod recipe;
pub mod registry;
pub mod select;
pub mod store;
pub mod validate;

pub use config::{
    DEFAULT_HF_ENDPOINT, DEFAULT_MAX_PARALLEL_FILES, DEFAULT_REGISTRY_URL, HfToken, HubConfig,
    RegistryEndpoint,
};
pub use download::{DownloadFileDesc, Downloader};
pub use error::HubError;
pub use host::{
    Backend, CudaInfo, HostProfile, MetalInfo, ServingDevice, probe_free_disk, probe_ram,
};
pub use progress::ProgressEvent;
pub use recipe::{
    CURRENT_SCHEMA_VERSION, CardSource, CommitSha, Component, ExpertResidency, FileEntry,
    HostConstraints, Index, IndexModelEntry, IndexVariantEntry, InstalledFileEntry,
    InstalledMetadata, ModalCapabilities, Modality, ModelInfo, ModelName, ModelRef, QualityScope,
    QualityScore, Recipe, RecipeFeatures, Requirements, Role, RolePath, ServerDefaults, Sha256,
    Task, Variant, VariantKey,
};
pub use registry::RegistryClient;
pub use select::{SelectedVariant, backends_fit_host, fits_host, host_backend, select_variant};
pub use store::{BlobStore, GcStats, InstalledVariantSummary};
pub use validate::{FamilyValidationFacts, ValidationMode, validate_recipe};

#[cfg(test)]
mod tests;
