//! The multimodal half of a prefill: where a picture's vectors enter the
//! residual stream, and where its tokens sit in the rotation.
//!
//! Nothing in this module encodes anything. By the time a stack sees a request
//! with media the tower has already run, somewhere above, and what arrives here
//! is a block of rows on the device plus the grids that say how many of them
//! each item produced. Two things then happen inside the pass, and they are the
//! whole of the multimodal path:
//!
//! - the embedding gather runs over every token as usual, and the rows at the
//!   placeholder positions are immediately overwritten with the tower's;
//! - the rotary embedding reads three position streams instead of one, because
//!   a placeholder token's position is a grid coordinate rather than a count.
//!
//! # Text-only requests do not come through here
//!
//! A request with no media takes the same launches it took before this module
//! existed: the same embedding gather, the same one-dimensional rotary entry
//! points, the same argument blocks. That is not an optimization, it is the
//! regression guarantee — three position streams that agree reproduce the
//! one-dimensional rotation arithmetically, but "arithmetically" is not
//! "bit-for-bit", and the text path is pinned to a recorded vector.
//!
//! # What the row count buys
//!
//! [`VisionPrefill::new`] refuses a block whose row count is not the sum of the
//! items' placeholder counts. That check is the only thing standing between a
//! tower and a prompt that disagree and a request that succeeds anyway: every
//! buffer would be the right size, every kernel would launch, and the answer
//! would be about a rearranged picture. There is no second expansion of the
//! placeholder run anywhere below this — the count the items carry is the one
//! the rendered prompt used, and the mismatch is reported rather than repaired.

mod mrope;

use shiso_core::backend::{Backend, KernelArg, KernelRef};
use shiso_core::error::CoreResult;
use shiso_core::{DeviceView, Extent3};

use crate::launch::{self, BLOCK_LINEAR, SpliceArgs, VISION};

pub(crate) use mrope::{Streams, Walk, rotary_shift_at};
pub use shiso_core::media::{PadTokens, VisionGrid, VisionPrefill};

/// What one chunk of a multimodal prefill adds to the pass.
///
/// The three position streams and the splice map are already on the device, in
/// scratch, so what travels is the encoder block and its height. The rest of the
/// pass reads the streams straight out of scratch, exactly as it reads the
/// linear positions.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MediaChunk<'a> {
    /// The whole encoder block for the request.
    pub(crate) rows: DeviceView<'a>,
    /// Rows in that block, so an index past it is refused on the device.
    pub(crate) row_count: u32,
}

/// Overwrites the embedding rows at the placeholder positions with the tower's.
///
/// `hidden_rows` is the chunk's freshly gathered embeddings and `map` is one
/// `int32` per token: the encoder row that token takes, or a negative number for
/// a token that keeps the embedding it was given.
///
/// # Errors
///
/// [`CoreError::Backend`] naming the kernel when the launch fails.
pub(crate) fn splice(
    backend: &dyn Backend,
    hidden_rows: DeviceView<'_>,
    map: DeviceView<'_>,
    media: MediaChunk<'_>,
    hidden: u32,
    n_tok: u32,
) -> CoreResult<()> {
    let args = SpliceArgs {
        rows: n_tok,
        hidden,
        src_rows: media.row_count,
    };
    launch::run(
        backend,
        KernelRef::new(VISION, "kernel_vit_splice_rows"),
        Extent3::new(n_tok, 1, 1),
        Extent3::new(BLOCK_LINEAR, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(hidden_rows),
            KernelArg::Buffer(media.rows),
            KernelArg::Buffer(map),
        ],
    )
    .map_err(|err| launch::launch_failure("kernel_vit_splice_rows", err))
}
