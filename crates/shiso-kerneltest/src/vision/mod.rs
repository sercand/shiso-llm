//! The vision kernel set's cases.
//!
//! These lived in `crates/shiso-metal/tests/vision_kernels.rs` and named no
//! backend even there — only the harness beside them did. Moving them here is
//! what lets one case grade both the Metal kernel and its CUDA twin, which is
//! the whole of what makes a twin a twin rather than a second implementation
//! graded against itself.
//!
//! **Every launch here names the module `"vision"`**, which is
//! `shiso_engine::launch::VISION` and what both encoders name. On Metal that is
//! a label on one flat library; on CUDA it is the key of a separately compiled
//! image, and until `kernels/cuda/vision/vit.cu` existed every one of these
//! launches failed with "this build embeds no kernel module vision" — the same
//! shape of defect `HANDOFF.md` records eleven times over, an argument rather
//! than arithmetic.
//!
//! Every function a backend driver calls here asserts by panicking, because
//! that is the whole of what a case does. `missing_panics_doc` reads each of
//! them as a function with an undocumented edge case, which is the opposite of
//! what they are, so it is turned off for the whole subtree rather than
//! answered a dozen times over with the same sentence.
#![expect(
    clippy::missing_panics_doc,
    reason = "a case asserts by panicking; that is its contract, not an edge case"
)]

pub mod quant;
pub mod tower;
