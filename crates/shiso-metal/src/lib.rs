//! shiso-metal - the Metal compute backend.
//!
//! The device is opened once, the MSL kernel library is compiled at startup, and
//! every launch goes through [`shiso_core::Backend`], so an engine holds
//! `Arc<dyn Backend>` and never names this crate.
//!
//! Launch geometry is CUDA semantics: `LaunchArgs::grid` is the threadgroup
//! count and `LaunchArgs::block` is the threads per threadgroup. Nothing here
//! divides, rounds up, or otherwise reinterprets a grid.
//!
//! The backend is built only with the `metal` feature *and* on a target that has
//! the Objective-C runtime. Everywhere else the crate presents the same two
//! names against an uninhabited type whose constructor refuses, which is what
//! keeps `cargo check --workspace` working on Linux: a Cargo feature cannot be
//! target-conditional, so the predicate is the feature and the target together,
//! and the Objective-C dependencies live in a target table.
#![cfg_attr(not(all(feature = "metal", target_os = "macos")), forbid(unsafe_code))]
#![cfg_attr(
    all(feature = "metal", target_os = "macos"),
    expect(unsafe_code, reason = "objc2 message sends into Metal")
)]

#[cfg(all(feature = "metal", target_os = "macos"))]
mod compile;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod device;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod pipeline;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod run;

#[cfg(not(all(feature = "metal", target_os = "macos")))]
mod stub;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use device::{MetalBackend, metal_available};
#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub use stub::{MetalBackend, metal_available};
