//! The qwen35 kernel set's cases, one module per entry-point family.
//!
//! The split mirrors the five test binaries these came out of, and the
//! headings of `kernels/metal/qwen35/KERNELS.md`: the gated delta
//! network and its four small helpers, the `q5_K`/`q6_K`/`q4_K` linear path,
//! the fused norm-and-rotate pair with its two multimodal siblings, the gated
//! attention family, and the chunk kernel's lane handout replayed as index
//! arithmetic.
//!
//! These lived in `crates/shiso-metal/tests/q35_*.rs` and named no backend even
//! there - only the harness beside them did. Moving them here is what lets one
//! case grade both the Metal kernel and its CUDA twin, which is the whole of
//! what makes a twin a twin rather than a second implementation graded against
//! itself.
//!
//! **Every launch here names the module `"qwen35"`, which is
//! `shiso_engine::launch::Q35`.** Eight of the launches said `"q35"` before the
//! move and two said `"qwen35"`, and nothing noticed: Metal compiles one flat
//! library and resolves an entry by name alone, so a module name is a label
//! there. On CUDA it is the key of a separately compiled image, and the eight
//! failed at the first launch with "this build embeds no kernel module q35".
//! That is one wrong *argument* rather than any wrong arithmetic, which is the
//! failure mode `HANDOFF.md` records eleven times over.
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

pub mod attn;
pub mod chunk_fold;
pub mod gdn;
pub mod gemv;
pub mod norm_rope;
