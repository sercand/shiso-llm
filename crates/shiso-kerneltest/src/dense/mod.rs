//! The dense kernel set's cases, one module per entry-point family.
//!
//! The split mirrors the five test binaries these came out of, and the five
//! headings of `crates/shiso-metal/metal/dense/KERNELS.md`: the flat
//! activations and the scaled residual norm, the attention family and its
//! rotary norms, the per-layer embedding pair, the mlx-affine set, and the
//! `q4_0` linear path that every Gemma 4 layer is read through.
//!
//! Every function a backend driver calls here asserts by panicking, because
//! that is the whole of what a case does. `missing_panics_doc` reads each of
//! them as a function with an undocumented edge case, which is the opposite of
//! what they are, so it is turned off for the whole subtree rather than
//! answered thirty times over with the same sentence.
#![expect(
    clippy::missing_panics_doc,
    reason = "a case asserts by panicking; that is its contract, not an edge case"
)]

pub mod affine;
pub mod attn;
pub mod block_draft;
pub mod exact_affine;
pub mod iq;
pub mod misc;
pub mod ple;
pub mod q4_0;
pub mod routed;
pub mod ssm;
pub mod vision;
