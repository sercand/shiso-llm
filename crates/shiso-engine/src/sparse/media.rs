//! The one launch this family's pass adds for a picture: the splice that
//! replaces a placeholder token's embedding with the row the tower produced.
//!
//! Everything else the media path needs is a row rather than a launch. The
//! rotary table already carries three axes and a media chunk fills them with
//! three different numbers; the position row is unchanged, because a
//! placeholder token occupies a cache row like any other token; the splice map
//! is one `int32` a token and comes out of the same walk as the axes. So the
//! whole of the device-side seam is here: one entry, between the row gather and
//! the residual broadcast.
//!
//! # The placeholder ids stay placeholder ids, unlike the dense family's
//!
//! The dense stack rewrites every placeholder token to id zero before it
//! uploads the id row (`crates/shiso-engine/src/stack.rs`, `ple_tokens`), so
//! the gather there fetches vocabulary row zero for a row the splice is about
//! to overwrite anyway. This family does not, and the difference is not
//! cosmetic: the id row is also what the n-gram block hashes its trigrams
//! from, and a trigram is a fact about the token stream the model was trained
//! on. The placeholder id is a real row of a 248320-entry vocabulary, so
//! nothing reads out of bounds, and keeping it is what makes the hashed
//! context the one the checkpoint saw.
//!
//! There is no reference for this on this checkpoint - the engine it was ported
//! from has no media path for this family at all - so it is a reasoned choice
//! rather than a measured one, and it is written here because that is the sort
//! of choice a later session needs to be able to find.
//!
//! # The argument block is one operand, and three scalars would have been wrong
//!
//! `kernel_vit_splice_rows` declares its first parameter as a
//! `vit_args_splice` **by value**, and every other entry this family dispatches
//! takes its widths as separate scalars. Standing three `Operand::U32`s in for
//! the struct's three fields is the obvious move and it does not work, for a
//! reason that is invisible in the C and plain in the PTX:
//!
//! ```text
//! .visible .entry kernel_vit_splice_rows(
//!     .param .align 4 .b8 kernel_vit_splice_rows_param_0[12],
//!     .param .u64 kernel_vit_splice_rows_param_1, ...
//! ```
//!
//! The block is **one** declared parameter, and
//! `crates/shiso-cuda/src/launch.rs`'s packer hands the driver one pointer per
//! declared parameter - not one contiguous buffer the driver carves. Three
//! `u32` slots would give the entry a four-byte first parameter where it
//! declares twelve, and it would read the two slots after it as the rest of
//! its own struct and then take a pointer out of a slot holding half of one.
//! Every launch would succeed. The three `KERNEL_NAMES.md` rows are right and
//! the reasoning from the C signature was not, which is
//! [`crate::sparse::bind::Operand::SpliceArgs`]'s whole reason to exist.
//!
//! [`SPLICE_ARGS_LAYOUT`] stays as the assertion that the struct is still the
//! twelve bytes the entry declares, because that is the half a reader cannot
//! see from here.

use shiso_core::media::VisionPrefill;

use crate::launch::SpliceArgs;
use crate::sparse::bind::{Args, Operand};
use crate::sparse::kernels::{self, Entry, MOD_VISION};
use crate::sparse::plan::Plan;
use crate::sparse::scratch::Buf;

/// The entry the splice launches, named once.
pub const SPLICE_ENTRY: &str = "kernel_vit_splice_rows";

/// The argument block's size and field offsets, as the three scalars below
/// assume them.
///
/// A `const` rather than a comment because the struct is declared in another
/// module and could gain a field there without anything in this one noticing:
/// a fourth field would push the three buffers along by a slot and the splice
/// would read its map as its source.
pub const SPLICE_ARGS_LAYOUT: (usize, usize, usize, usize) = (
    size_of::<SpliceArgs>(),
    core::mem::offset_of!(SpliceArgs, rows),
    core::mem::offset_of!(SpliceArgs, hidden),
    core::mem::offset_of!(SpliceArgs, src_rows),
);

const _: () = assert!(
    SPLICE_ARGS_LAYOUT.0 == 12
        && SPLICE_ARGS_LAYOUT.1 == 0
        && SPLICE_ARGS_LAYOUT.2 == 4
        && SPLICE_ARGS_LAYOUT.3 == 8,
    "the splice argument block is three u32 fields with no padding, which is \
     what the three scalar operands below stand in for"
);

/// Overwrites the gathered embedding of every placeholder token with the
/// encoder row the splice map names.
///
/// Only the grid is decided here. The three widths are the argument block's,
/// and it is built at issue time from the pass, so the height the kernel
/// bounds its reads by is the height the pass was planned against rather than
/// a second copy of it.
///
/// Infallible, unlike the other plan builders in this family: there is one
/// entry, it is instantiated for no element type, and every bound it could
/// check is checked on the device or by the walk that built the map.
#[must_use]
pub fn splice(n_tok: u32) -> Plan {
    let mut plan = Plan::new();
    plan.launch(
        Entry::plain(MOD_VISION, SPLICE_ENTRY),
        kernels::splice(n_tok),
        // `args, hidden_rows, src, map` - four parameters, which is what the
        // entry declares. The widths travel inside `args`, which the issuer
        // builds from the pass rather than from these arguments.
        Args::of(&[
            Operand::SpliceArgs,
            Operand::Scratch(Buf::Embed),
            Operand::MediaRows,
            Operand::Scratch(Buf::SpliceMap),
        ]),
    );
    plan
}

/// The one position rule this family walks every request by: a placeholder
/// token's three axes are a grid coordinate. The dense families read theirs off
/// the config because two of them disagree; there is one of this one.
pub(crate) const MEDIA_POSITIONS: crate::shape::MediaPositions =
    crate::shape::MediaPositions::Mrope3D;

/// A multimodal prefill part way through its chunks.
///
/// The walk is the part that cannot be rebuilt per chunk: it carries the
/// running multimodal position, which after a single image is already a
/// different number from the linear one, and how far into the encoder block
/// the splice has got. The rotary prefix is the second part, and it is this
/// family's own - the dense family's three planar streams have no lookback to
/// carry. A prompt split across sub-chunks has to see the same rows it would
/// have seen unsplit, and these two are what make the split invisible.
#[derive(Debug)]
pub(crate) struct MediaRun<'a> {
    /// The request's encoder block and the grids that make it addressable.
    pub(crate) prefill: VisionPrefill<'a>,
    /// The walk over this request's token stream.
    pub(crate) walk: crate::vision::Walk<'a>,
    /// The previous chunk's last [`shiso_core::page::MROPE_BACK`] rotary rows,
    /// empty before the first chunk has run.
    pub(crate) prefix: Vec<i32>,
}

impl<'a> MediaRun<'a> {
    /// Starts a run over `prefill`, with the walk's sequence position at
    /// `start` and its rotary position `rope_shift` behind it.
    pub(crate) fn new(
        prefill: &VisionPrefill<'a>,
        start: u32,
        rope_shift: u32,
        positions: crate::shape::MediaPositions,
    ) -> Self {
        Self {
            prefill: *prefill,
            walk: crate::vision::Walk::new(prefill.grids(), prefill.pads(), start, positions)
                .at_rotary(start.saturating_sub(rope_shift)),
            prefix: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::plan::Step;

    #[test]
    fn the_argument_block_is_still_the_twelve_bytes_the_entry_declares() {
        // The entry declares `.param .align 4 .b8 [12]`, and the block bound
        // into that slot is this struct. A field added to `SpliceArgs` fails
        // the `const` assertion above; a field REORDERED there fails here.
        assert_eq!(SPLICE_ARGS_LAYOUT, (12, 0, 4, 8));
    }

    #[test]
    fn the_operands_are_the_kernels_four_parameters_in_declaration_order() {
        // FOUR, not six. The entry declares its argument block as one
        // twelve-byte parameter and the packer gives the driver one pointer
        // per declared parameter, so the three widths travel inside the block.
        let plan = splice(7);
        let step = plan.steps().first().expect("one launch");
        let ops = step.operands().expect("the list fits a launch");
        assert_eq!(
            ops,
            [
                Operand::SpliceArgs,
                Operand::Scratch(Buf::Embed),
                Operand::MediaRows,
                Operand::Scratch(Buf::SpliceMap),
            ]
        );
    }

    #[test]
    fn the_grid_is_one_block_per_token_row_and_not_one_per_element() {
        // The kernel reads its row from `blockIdx.x` and loops the width, so a
        // grid sized by elements would run `hidden` blocks per token and every
        // block but the first `n_tok` would return having read a map entry
        // nobody wrote.
        let plan = splice(7);
        match plan.steps().first().expect("one launch") {
            Step::Kernel { launch, entry, .. } => {
                assert_eq!(launch.grid.x, 7);
                assert_eq!(launch.block.x, 256);
                assert_eq!(entry.module(), MOD_VISION);
            }
            other => panic!("a splice is an ordinary launch: {other:?}"),
        }
    }
}
