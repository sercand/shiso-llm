//! Device cases for the `generic` and `qwen35` module entries a dense pass
//! reaches.
//!
//! The `dense` module beside this one carries the entries Gemma 4 needs and
//! qwen35 does not. These are the other half: the entries a Gemma 4 pass
//! dispatches under `launch::GENERIC` and `launch::Q35`, which are older than
//! this phase on Metal and new on CUDA, and which had no case of their own
//! anywhere in the tree - so a twin of one of them was graded by nothing until
//! a whole model answered wrongly.
//!
//! Every case compares the device against the simplest reading of the formula,
//! written as a plain Rust `f32` loop, so a disagreement points at the kernel
//! or its launch geometry rather than at a second subtle implementation.
//!
//! The `#[test]` wrappers live in `crates/shiso-cuda/tests/generic.rs`. Metal has
//! no driver for most of this module: two of these entries take the simdgroup
//! count as a Metal function constant, which most cases here do not pass, and the
//! CUDA twins read it off the launch instead. The cases are written
//! backend-neutrally anyway - nothing here names a device. The exception is
//! `rows_q8`: its two cases pass their function constants explicitly through
//! `KernelRef::specialized`, so they can run on Metal
//! (`crates/shiso-metal/tests/generic_q8.rs`) as well as on any backend that
//! ignores the table.
//!
//! Every function a backend driver calls here asserts by panicking, because
//! that is the whole of what a case does, so `missing_panics_doc` is turned off
//! for the subtree rather than answered with the same sentence in every case.
#![expect(
    clippy::missing_panics_doc,
    reason = "a case asserts by panicking; that is its contract, not an edge case"
)]

pub mod glu;
pub mod norm;
pub mod plane;
pub mod product;
pub mod rows_q8;
