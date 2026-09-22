//! The row gather that opens a pass: one vocabulary row per token id.
//!
//! This is the only stage of the stack whose kernel is chosen by the *stored*
//! element type of a tensor rather than by a shape, and the three entries it
//! chooses between are not a tier ladder: they are three different storage
//! formats of the same table, and exactly one of them can read a given
//! checkpoint. A table stored in a fourth format is therefore a refusal naming
//! the format, not a fallback - a gather that dequantized a table the wrong way
//! produces a residual nobody can attribute to a token.
//!
//! # Why the argument lists differ between them
//!
//! They came from different families and were never unified, and this module
//! does not unify them either: each launch is the parameter list its own kernel
//! declares, in its own order. Two of the three take the vocabulary size and
//! clamp an id outside it to row zero; the block-quantized one does not, so the
//! caller owns the range there. Reordering either list to look like the other
//! would compile.

use shiso_core::DType;
use shiso_core::error::{CoreError, CoreResult};

use crate::sparse::bind::{Args, Operand};
use crate::sparse::kernels::{self, Entry, MOD_EMBED};
use crate::sparse::plan::Plan;
use crate::sparse::shape::SparseShape;

/// The three operands a gather moves between.
///
/// A struct rather than three arguments because all three are the same type and
/// a swapped pair would compile into a launch that reads its own destination.
///
/// Deliberately NOT `#[non_exhaustive]`: the stack builds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bound {
    /// The token ids to look up.
    pub ids: Operand,
    /// The vocabulary table to look them up in.
    pub table: Operand,
    /// Where the gathered rows go.
    pub dst: Operand,
}

impl Bound {
    /// One gather's three operands.
    #[must_use]
    pub const fn new(ids: Operand, table: Operand, dst: Operand) -> Self {
        Self { ids, table, dst }
    }
}

/// Copies of one row this gather writes per token.
///
/// The half-precision entry can fan a row out across a residual's streams as it
/// gathers. This stack does not ask it to: the residual broadcast is its own
/// launch, which also has to run for the other two storage formats, so the
/// gather writes one row per token and the broadcast is left where every path
/// shares it.
const COPIES_PER_TOKEN: u32 = 1;

/// The gather for a table stored as `dtype`.
///
/// # Errors
///
/// [`CoreError::Unsupported`] for a storage format no gather entry reads.
pub fn rows(shape: &SparseShape, dtype: DType, bound: Bound, n_tok: u32) -> CoreResult<Plan> {
    let elements = u64::from(n_tok) * u64::from(COPIES_PER_TOKEN) * u64::from(shape.hidden);
    let launch = kernels::elementwise(elements);
    let mut plan = Plan::new();
    match dtype {
        // `out, tokens, w, n_vocab, n_tokens, n_embd, n_hc`
        DType::F16 => plan.launch(
            Entry::plain(MOD_EMBED, "embed_tokens_hc_kernel"),
            launch,
            Args::of(&[
                bound.dst,
                bound.ids,
                bound.table,
                Operand::U32(shape.vocab),
                Operand::U32(n_tok),
                Operand::U32(shape.hidden),
                Operand::U32(COPIES_PER_TOKEN),
            ]),
        ),
        // `out, weights, tokens, n_tokens, n_embd, n_vocab` - the table comes
        // before the ids here, which is the whole reason these lists are
        // written out one kernel at a time.
        DType::BF16 => plan.launch(
            Entry::plain(MOD_EMBED, "glm53_embedding_bf16_kernel"),
            launch,
            Args::of(&[
                bound.dst,
                bound.table,
                bound.ids,
                Operand::U32(n_tok),
                Operand::U32(shape.hidden),
                Operand::U32(shape.vocab),
            ]),
        ),
        // `out, tokens, w, n_tokens, n_embd`
        DType::Q8_0 => plan.launch(
            Entry::plain(MOD_EMBED, "glm_embed_tokens_q8_0_kernel"),
            launch,
            Args::of(&[
                bound.dst,
                bound.ids,
                bound.table,
                Operand::U32(n_tok),
                Operand::U32(shape.hidden),
            ]),
        ),
        _ => {
            return Err(CoreError::Unsupported {
                what: "a vocabulary table stored in a format no row gather in this kernel set \
                       reads",
                backend: "engine",
            });
        }
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::scratch::Buf;

    fn bound() -> Bound {
        Bound::new(
            Operand::Scratch(Buf::Ids),
            Operand::Scratch(Buf::Embed),
            Operand::Scratch(Buf::Embed),
        )
    }

    fn shape() -> SparseShape {
        use crate::shape::{GdnShape, LayerKind};
        use crate::sparse::shape::{GqaShape, IndexerShape, MoeShape, MropeSections};
        SparseShape {
            layers: vec![LayerKind::Attn],
            hidden: 2_560,
            vocab: 248_320,
            streams: 4,
            stream_rank: 320,
            gqa: GqaShape::new(24, 2, 256, 64),
            gdn: GdnShape::new(48, 16, 128, 4),
            moe: MoeShape::new(512, 10, 1, 640),
            indexer: IndexerShape::new(4, 128, 2_048, 4),
            ple: None,
            mrope: MropeSections::new(11, 11, 10),
            rms_eps: 1.0e-6,
            rope_theta: 1.0e7,
        }
    }

    #[test]
    fn the_three_storage_formats_reach_three_different_entries() {
        let shape = shape();
        for (dtype, entry) in [
            (DType::F16, "embed_tokens_hc_kernel"),
            (DType::BF16, "glm53_embedding_bf16_kernel"),
            (DType::Q8_0, "glm_embed_tokens_q8_0_kernel"),
        ] {
            let plan = rows(&shape, dtype, bound(), 8).expect("a stored format with a gather");
            assert_eq!(plan.names(), vec![entry], "{dtype:?}");
        }
    }

    #[test]
    fn a_format_no_entry_reads_is_refused_rather_than_approximated() {
        let err = rows(&shape(), DType::Q4_K, bound(), 8)
            .expect_err("no gather entry reads a k-quant table");
        assert!(matches!(err, CoreError::Unsupported { .. }), "{err:?}");
    }

    #[test]
    fn the_grid_covers_one_row_per_token_and_not_one_per_stream() {
        // The residual broadcast is a launch of its own, so the gather writes
        // `n_tok * n_embd` elements. Sizing it for the streams as well would
        // write four times the destination.
        let shape = shape();
        let plan = rows(&shape, DType::F16, bound(), 512).expect("half precision has a gather");
        let step = plan.steps().first().expect("one launch");
        let want = (512_u64 * u64::from(shape.hidden)).div_ceil(256);
        match step {
            crate::sparse::plan::Step::Kernel { launch, args, .. } => {
                assert_eq!(u64::from(launch.grid.x), want);
                assert_eq!(launch.block.x, 256);
                let ops = args.operands().expect("the list fits a launch");
                assert_eq!(ops[6], Operand::U32(1), "one copy of a row per token");
            }
            other => panic!("a gather is an ordinary launch: {other:?}"),
        }
    }
}
