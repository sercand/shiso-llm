//! Device cases for the `diffusion` module entries.
//!
//! Most of the diffusion pack is still graded from
//! `crates/shiso-metal/tests/dit_kernels.rs`, which predates this crate's
//! rule; the per-row INT8 GEMM and its `ConvRot` rotation start here, so a
//! second backend's twin of them is graded by the same cases from its first
//! run, as are the residual VAE decoder's fused entries ([`vae`]) and the
//! prefix-cached transformer's causal attention, table `RoPE` and
//! activations ([`prefix`]), and the GGUF k-quant linears ([`kquant`]).

pub mod int8;
pub mod kquant;
pub mod prefix;
pub mod vae;
