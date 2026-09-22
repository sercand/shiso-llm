//! The Splash package: what it says it holds, the digests that say it still
//! does, and the reader that puts a file's sections on a device.
//!
//! A Splash package is not a checkpoint in this family's other sense. There is
//! no safetensors index and no `config.json` subtree to recognise: there is a
//! `manifest.json` listing every artifact with its SHA-256, a `layout.json`
//! saying what each `.bin` holds and at which offset, and the `.bin` files
//! themselves, already quantized and already padded to the widths the kernels
//! want. The two index readers need no device and read no weight; they exist so
//! that [`loader`], which does both, can be told where a tensor is instead of
//! deriving it.
//!
//! The two are read in that order for a reason worth stating once: the layout
//! is what makes a byte range a tensor, and the manifest is what makes the
//! bytes in that range the ones the packer wrote. A package that passes the
//! layout and fails the manifest still loads, still runs, and answers from
//! whatever half-downloaded file it found - which is the failure mode this
//! module exists to make impossible rather than merely unlikely.

pub mod layout;
pub mod loader;
pub mod manifest;

pub use layout::{FileLayout, Layout, Section, SectionDtype};
pub use loader::{LayerBuffers, SectionBuffer, SectionError, load_layer, slice_section};
pub use manifest::{
    Artifact, Converter, DraftGeometry, ExecutionGeometry, Format, LayerKind, Manifest,
    SCHEMA_VERSION, TargetGeometry, Upstream, UpstreamRef, VerifyError, verify,
};

#[cfg(test)]
#[path = "package_tests.rs"]
mod tests;
