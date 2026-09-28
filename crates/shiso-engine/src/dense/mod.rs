//! The parametric block launches, and the kernel set behind them.
//!
//! "dense" is the family whose widths arrive as arguments rather than as
//! constants: one attention entry point serves every checkpoint of this block
//! structure, so a second such model is a different set of numbers and not a
//! second set of kernels. That is why these launches live in the engine while
//! qwen35's live in `shiso-models` — there is nothing checkpoint-specific left in
//! them to move out.
//!
//! [`DenseKernels`] is the whole of the public surface. Behind it is [`attn`],
//! the block's five launches in the order the cache requires them.

pub mod attn;

use shiso_core::backend::Backend;
use shiso_core::error::{CoreError, CoreResult};

use crate::attn::{Attn, AttnCache, AttnGeometry, AttnScratch};
use crate::blocks::{AttnIo, BlockKernels, GdnIo};
use crate::gdn::{Gdn, GdnGeometry, GdnScratch};
use crate::launch::{DENSE_ATTN_HEAD_DIM, WARP};
use crate::shape::{HybridShape, LayerKind};

/// The dense family's block launcher.
///
/// A unit struct rather than a value carrying a measurement: everything this
/// family's kernels need is either in the layer's [`AttnGeometry`] or in the
/// argument block derived from it, and nothing here is a fact about the machine
/// the way the recurrent family's lane split is.
#[derive(Debug, Default, Clone, Copy)]
pub struct DenseKernels;

impl BlockKernels for DenseKernels {
    /// Refuses a shape outside what this set was compiled for.
    ///
    /// Five facts, and each of them is a wrong answer rather than a crash if it
    /// goes unchecked. A head wider than the staging every threadgroup
    /// allocation is sized by overruns that memory. A head that is not a whole
    /// number of simdgroups is a prefill store whose final loop leaves part of
    /// each row unwritten, because the loop is unguarded and the 32 lanes cover
    /// a row between them. A frequency ladder narrower than the rotated prefix
    /// makes the rotation's partner index land back inside the loop's own range,
    /// where two lanes race over one dimension. A query head count that is not a
    /// whole multiple of the key/value head count - zero on either side
    /// included - is a grouping the kernels cannot form: they address a key head
    /// as `h / (n_head / n_head_kv)`, an integer division that silently sends
    /// the last queries to a head they do not belong to, or divides by zero. And
    /// an output gate is a second projection this family neither reads nor
    /// applies, so a checkpoint that has one would be run with the gate silently
    /// dropped.
    ///
    /// A recurrent slot is the sixth, and it is a different kind of statement:
    /// this set has no recurrent entry point at all. It is refused here rather
    /// than at the launch so that the disagreement is a load failure instead of
    /// a residual stream that is short one block's contribution.
    fn validate(&self, shape: &HybridShape) -> CoreResult<()> {
        // The state-space block's launches are the engine's own and serve any
        // family; the gated delta-rule's are a family's.
        if shape.count(LayerKind::Gdn) > 0 {
            return Err(CoreError::Unsupported {
                what: "a gated delta-rule slot on the dense kernel set",
                backend: "engine",
            });
        }
        if shape.attn_output_gate {
            return Err(CoreError::Unsupported {
                what: "an attention output gate on the dense kernel set",
                backend: "engine",
            });
        }
        for (rank, layer) in shape.attn.iter().enumerate() {
            if layer.head_dim > DENSE_ATTN_HEAD_DIM {
                return Err(CoreError::InvalidValue {
                    key: "dense head width",
                    value: format!(
                        "attention slot {rank} has a {}-wide head, and the set stages {DENSE_ATTN_HEAD_DIM}",
                        layer.head_dim
                    ),
                });
            }
            if !layer.head_dim.is_multiple_of(WARP) {
                return Err(CoreError::InvalidValue {
                    key: "dense head width",
                    value: format!(
                        "attention slot {rank} has a {}-wide head, which {WARP} lanes do not cover \
                         a whole number of times",
                        layer.head_dim
                    ),
                });
            }
            if layer.q_heads == 0 || layer.kv_heads == 0 {
                return Err(CoreError::InvalidValue {
                    key: "dense head counts",
                    value: format!(
                        "attention slot {rank} has {} query heads over {} key/value heads, and a \
                         block with no heads of either kind has no grouping to compute",
                        layer.q_heads, layer.kv_heads
                    ),
                });
            }
            if !layer.q_heads.is_multiple_of(layer.kv_heads) {
                return Err(CoreError::InvalidValue {
                    key: "dense head counts",
                    value: format!(
                        "attention slot {rank} has {} query heads over {} key/value heads, which \
                         the kernels' `h / (n_head / n_head_kv)` does not group evenly",
                        layer.q_heads, layer.kv_heads
                    ),
                });
            }
            if !layer.qk_norm && layer.n_rot != 0 {
                return Err(CoreError::InvalidValue {
                    key: "dense rotary width",
                    value: format!(
                        "attention slot {rank} rotates {} dimensions but normalizes nothing, and \
                         the set's rotation is fused into its norm",
                        layer.n_rot
                    ),
                });
            }
            if layer.freq_dims < layer.n_rot {
                return Err(CoreError::InvalidValue {
                    key: "dense rotary width",
                    value: format!(
                        "attention slot {rank} rotates {} dimensions over a {}-wide ladder",
                        layer.n_rot, layer.freq_dims
                    ),
                });
            }
        }
        Ok(())
    }

    fn attention(
        &self,
        backend: &dyn Backend,
        attn: &Attn,
        geometry: &AttnGeometry,
        scratch: &AttnScratch,
        io: AttnIo<'_>,
        cache: &AttnCache<'_>,
        n_tok: u32,
    ) -> CoreResult<()> {
        attn::run(backend, attn, geometry, scratch, io, cache, n_tok)
    }

    fn recurrent(
        &self,
        _backend: &dyn Backend,
        _gdn: &Gdn,
        _geometry: &GdnGeometry,
        _scratch: &GdnScratch,
        _io: GdnIo<'_>,
        _n_tok: u32,
    ) -> CoreResult<()> {
        // A stack built from this set has no recurrent slot at all, so reaching
        // this arm means the shape and the kernel set disagree about what the
        // model is. Refusing names that disagreement; a silent no-op would leave
        // the residual stream short one block's contribution and every later
        // layer plausible.
        Err(CoreError::Unsupported {
            what: "a recurrent block on the dense kernel set",
            backend: "engine",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::DenseKernels;
    use crate::blocks::BlockKernels;
    use crate::shape::{
        Activation, AttnLayer, GdnShape, HybridShape, LayerKind, MediaPositions, NormScheme,
    };
    use shiso_core::error::CoreError;

    /// A Gemma-shaped stack of `attn.len()` attention slots and nothing else.
    fn shape(attn: Vec<AttnLayer>) -> HybridShape {
        HybridShape {
            layers: vec![LayerKind::Attn; attn.len()],
            attn,
            hidden: 2048,
            vocab: 128,
            intermediate: 4096,
            gdn: GdnShape::new(0, 0, 0, 1),
            attn_output_gate: false,
            rms_eps: 1e-6,
            activation: Activation::GeGluTanh,
            norms: NormScheme::PrePost,
            embed_scale: None,
            final_softcap: None,
            layer_scalar: true,
            ple: None,
            media_positions: MediaPositions::Linear,
            v_norm: true,
            moe: None,
            ssm: None,
            no_ffn: Vec::new(),
        }
    }

    #[test]
    fn a_shape_the_set_was_compiled_for_is_accepted() {
        let attn = vec![
            AttnLayer::new(8, 2, 256, 128, 256, 1e4),
            AttnLayer::new(8, 2, 512, 512, 512, 1e6),
        ];
        DenseKernels
            .validate(&shape(attn))
            .expect("two heads inside the staging the set allocates");
    }

    /// The staging every threadgroup allocation is sized by is a constant, so a
    /// wider head is a read past the end of memory that was allocated correctly.
    #[test]
    fn a_head_wider_than_the_staging_is_refused() {
        let attn = vec![
            AttnLayer::new(8, 2, 256, 256, 256, 1e4),
            AttnLayer::new(8, 2, 640, 256, 640, 1e4),
        ];
        match DenseKernels.validate(&shape(attn)) {
            Err(CoreError::InvalidValue { key, value }) => {
                assert_eq!(key, "dense head width");
                assert!(
                    value.contains("slot 1"),
                    "the refusal names the slot: {value}"
                );
            }
            other => panic!("expected a head-width refusal, got {other:?}"),
        }
    }

    /// The prefill store's final loop is unguarded: 32 lanes cover a row between
    /// them, so a head that is not a whole number of them leaves a tail nobody
    /// writes.
    #[test]
    fn a_head_the_simdgroup_does_not_cover_is_refused() {
        let attn = vec![AttnLayer::new(8, 2, 48, 32, 48, 1e4)];
        assert!(matches!(
            DenseKernels.validate(&shape(attn)),
            Err(CoreError::InvalidValue { key, .. }) if key == "dense head width"
        ));
    }

    /// Below `freq_dims >= n_rot` the rotation's partner index lands back inside
    /// the loop's own range and two lanes race over one dimension.
    ///
    /// The shape's own check refuses `n_rot > freq_dims` too, and this is not
    /// redundant with it: this is the statement of what the KERNEL requires,
    /// stated where the kernel is, and it is the one that has to move if the
    /// rotation ever stops requiring it.
    #[test]
    fn a_ladder_narrower_than_the_rotated_prefix_is_refused() {
        let attn = vec![AttnLayer::new(8, 2, 256, 256, 128, 1e4)];
        assert!(matches!(
            DenseKernels.validate(&shape(attn)),
            Err(CoreError::InvalidValue { key, .. }) if key == "dense rotary width"
        ));
    }

    /// `n_head / n_head_kv` is the group size the kernels divide a query head by
    /// to reach its key head, so no query head at all is a grouping that does
    /// not exist.
    #[test]
    fn a_slot_with_no_query_head_is_refused() {
        let attn = vec![AttnLayer::new(0, 2, 256, 256, 256, 1e4)];
        match DenseKernels.validate(&shape(attn)) {
            Err(CoreError::InvalidValue { key, value }) => {
                assert_eq!(key, "dense head counts");
                assert!(
                    value.contains("0 query heads") && value.contains("2 key/value heads"),
                    "the refusal names both counts: {value}"
                );
            }
            other => panic!("expected a head-count refusal, got {other:?}"),
        }
    }

    /// And no key/value head is the division itself: `n_head / n_head_kv` by
    /// zero.
    #[test]
    fn a_slot_with_no_key_value_head_is_refused() {
        let attn = vec![AttnLayer::new(8, 0, 256, 256, 256, 1e4)];
        match DenseKernels.validate(&shape(attn)) {
            Err(CoreError::InvalidValue { key, value }) => {
                assert_eq!(key, "dense head counts");
                assert!(
                    value.contains("8 query heads") && value.contains("0 key/value heads"),
                    "the refusal names both counts: {value}"
                );
            }
            other => panic!("expected a head-count refusal, got {other:?}"),
        }
    }

    /// A count that divides unevenly is the quiet one: the division truncates,
    /// and the tail of the query heads reads a key head that is not theirs.
    #[test]
    fn query_heads_that_do_not_group_are_refused() {
        let attn = vec![
            AttnLayer::new(8, 2, 256, 256, 256, 1e4),
            AttnLayer::new(6, 4, 256, 256, 256, 1e4),
        ];
        match DenseKernels.validate(&shape(attn)) {
            Err(CoreError::InvalidValue { key, value }) => {
                assert_eq!(key, "dense head counts");
                assert!(
                    value.contains("slot 1")
                        && value.contains("6 query heads")
                        && value.contains("4 key/value heads"),
                    "the refusal names the slot and both counts: {value}"
                );
            }
            other => panic!("expected a head-count refusal, got {other:?}"),
        }
    }

    /// An output gate is a second projection this family never reads. A
    /// checkpoint that carries one would run with its gate silently dropped.
    #[test]
    fn an_output_gate_is_refused() {
        let mut shape = shape(vec![AttnLayer::new(8, 2, 256, 256, 256, 1e4)]);
        shape.attn_output_gate = true;
        assert!(matches!(
            DenseKernels.validate(&shape),
            Err(CoreError::Unsupported { .. })
        ));
    }

    /// And a recurrent slot, which this set has no entry point for at all: a
    /// load failure rather than a residual stream short one block.
    #[test]
    fn a_recurrent_slot_is_refused() {
        let mut shape = shape(vec![AttnLayer::new(8, 2, 256, 256, 256, 1e4)]);
        shape.layers.insert(0, LayerKind::Gdn);
        assert!(matches!(
            DenseKernels.validate(&shape),
            Err(CoreError::Unsupported { .. })
        ));
    }
}
