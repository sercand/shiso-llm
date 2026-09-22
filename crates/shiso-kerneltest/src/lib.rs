//! The dense, generic, qwen35, qwen35moe and vision kernel sets' device cases,
//! written against no backend in particular.
//!
//! Every case here is a fixture, a launch through [`shiso_core::Backend`] and a
//! comparison against a plain Rust `f32` loop over the same numbers. None of it
//! names a device. That is the whole point of the crate: the engine names its
//! kernels itself (`KernelRef::new("dense", "kernel_dense_...")`) and each
//! backend resolves the name its own way, so a twin written for a second device
//! is graded by the same case that grades the first, from its first run rather
//! than after the fact.
//!
//! **The dependency runs from the backends to here, never back.** This crate
//! must not depend on `shiso-metal` or `shiso-cuda`; if it did, it would be a
//! relocated Metal test suite rather than a neutral one, and a case could
//! quietly grow a device assumption that only one backend satisfies. The
//! concrete handle a backend has to keep alive travels through [`Gpu::new`] as
//! `Box<dyn Any>` for exactly that reason.
//!
//! Each backend carries a thin driver per file - `crates/shiso-metal/tests/
//! dense_*.rs`, `q35_*.rs` and `crates/shiso-cuda/tests/` the same - holding the
//! `#[test]` wrappers and nothing else: open a device, or say on stderr why the
//! cases did not run.

pub mod dense;
pub mod generic;
pub mod harness;
pub mod oracle;
pub mod qwen35;
pub mod qwen35moe;
pub mod vision;

pub use crate::harness::{Gpu, drain, f32_bytes, f32_values, run};
