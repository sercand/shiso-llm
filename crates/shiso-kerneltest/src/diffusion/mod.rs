//! Device cases for the `diffusion` module entries.
//!
//! Most of the diffusion pack is still graded from
//! `crates/shiso-metal/tests/dit_kernels.rs`, which predates this crate's
//! rule; the per-row INT8 GEMM and its `ConvRot` rotation start here, so a
//! second backend's twin of them is graded by the same cases from its first
//! run, as are the residual VAE decoder's fused entries ([`vae`]).

pub mod int8;
pub mod vae;
