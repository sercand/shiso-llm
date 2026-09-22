//! Compiled device kernels, embedded in the binary and looked up by name.
//!
//! The crate has no dependencies and, apart from the registry the build script
//! generates, no code. A caller asks for a module by name and gets the bytes
//! back:
//!
//! ```no_run
//! let bytes = shiso_kernels::ptx("mmq");
//! ```
//!
//! # What a lookup means
//!
//! One registry, PTX from the NVIDIA lane, and one name space over it. A module
//! name is a single registration, so a caller that knows a kernel's name needs
//! to know nothing else to reach it.
//!
//! The registry is empty on any build that did not explicitly ask for a
//! compile, which is the default everywhere. That is the normal state of this
//! crate on a development machine, and every accessor answers an empty slice
//! rather than failing to compile or panicking. A loader handed zero bytes
//! fails the module load with the device's own error, which is a far better
//! place to find out than a panic on a thread deep inside a build.
//!
//! # Staleness
//!
//! The registry arrives through `include!` of a file the build script wrote,
//! and cargo does not track build-script output as a recompile trigger. It
//! would happily keep a module list from a previous kernel set indefinitely.
//! [`KERNEL_SET_HASH`] is what closes that: it is a fingerprint of the
//! generated registry, referenced here with `env!`, so a changed kernel set
//! changes this crate's own fingerprint and forces the rebuild.

include!(concat!(env!("SHISO_KERNELS_OUT_DIR"), "/target_ptx.rs"));

/// Content fingerprint of the kernel set embedded in this binary.
///
/// Twelve hex characters over the generated registry sources. Two binaries with
/// the same value embed the same modules built from the same artefact paths; a
/// different value means the set changed. Referencing it is what makes cargo
/// rebuild this crate when the build script regenerates the registry — see the
/// crate documentation.
pub const KERNEL_SET_HASH: &str = env!("SHISO_KERNEL_SET_HASH");
