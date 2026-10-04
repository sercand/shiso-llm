//! The ternary kernel set's cases: the blockwise rotation a rotated
//! checkpoint's activations pass through, and the matmuls over its ternary
//! and 2-bit affine weights.
//!
//! Every function a backend driver calls here asserts by panicking, which is
//! the whole of what a case does.
#![expect(
    clippy::missing_panics_doc,
    reason = "a case asserts by panicking; that is its contract, not an edge case"
)]

pub mod fixtures;
pub mod matmul;
pub mod rotate;

/// Every entry point this set's cases launch, for a backend's pipeline check.
pub const ENTRIES: &[&str] = &[rotate::ENTRY];
