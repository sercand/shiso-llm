//! Which build may read a stored checkpoint back.
//!
//! [`crate::ckpt`] decides *when* an entry is written. This module answers the
//! other question - which bytes a build may resume from - and it is separate
//! because the two change for different reasons: a policy changes when a
//! scheduler learns something, and a layout changes only when a shape does.
//!
//! # The layout words are part of every key
//!
//! Every count a reader would otherwise have to assume is stated: the layer
//! counts, the vocabulary, the width of a key/value row, and the element counts
//! the recurrent state is sized from. The store folds these words, the page
//! geometry and the device's kernel-set fingerprint into one identity digest, and
//! that digest into the name of every page and every checkpoint. A build whose
//! words differ therefore never even finds another build's files - which is the
//! failure that matters here, because loading another architecture's rows
//! produces fluent, confident nonsense rather than an error.
//!
//! The recurrent state itself is the session's own stack, slot by slot in
//! ordinal order, each slot's convolution window followed by its state matrix.
//! That is the order [`crate::stack`] snapshots in, and a resume copies it
//! straight back: no element is read, reordered or converted anywhere along the
//! way, so a restored session is a byte image of the one that was saved.
//!
//! # What identity the engine cannot supply
//!
//! Two of the words are not derivable from a geometry: which checkpoint family
//! wrote the file, and how its weights were quantized. Both are facts about a
//! model, so they arrive as [`CkptIdentity`] from the crate that knows which
//! model this is.

use shiso_kvstore::PathLayout;

use crate::layers::Geometry;
use crate::shape::{HybridShape, LayerKind};

/// Bytes per stored key/value element: the cache is f16 end to end.
///
/// Stated here as well as in [`crate::kv`] on purpose. This one is a *guard value*
/// written into the file, and the other is what the pool is sized with; if the two
/// ever disagree that is a bug worth a compile-time edit in two places rather than a
/// silent cross-load.
const CACHE_ELEMENT_BYTES: u32 = 2;

/// Who wrote a checkpoint, in the two terms a geometry cannot express.
///
/// Neither field is configurable. A model tag is a fact about which family's blocks
/// the bytes came out of, and the quantization is read off the weights that produced
/// them, so both are derived by the crate that loaded the checkpoint and passed
/// through here unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CkptIdentity {
    /// Which family's geometry the stored bytes assume.
    ///
    /// A shared cache directory stays unambiguous only if two families never claim
    /// the same tag, so the values are assigned once, by the crate that defines the
    /// family, and never by a caller.
    pub model_id: u8,
    /// Quantization width of the weights that produced the stored rows.
    ///
    /// Two widths are not interchangeable: the same prompt through a 4-bit and a
    /// 2-bit build produces different key/value rows, and a cross-load would answer
    /// from the wrong ones without any sign of trouble. The disk tier only accepts 2
    /// and 4, which is what makes a build whose weights are neither refuse to
    /// checkpoint at all rather than store rows nothing can check.
    pub quant_bits: u8,
}

impl CkptIdentity {
    /// The identity of a family tagged `model_id` whose weights are `quant_bits`
    /// wide.
    #[must_use]
    pub const fn new(model_id: u8, quant_bits: u8) -> Self {
        Self {
            model_id,
            quant_bits,
        }
    }
}

/// The geometry guard a stored payload carries, derived from the running build.
///
/// Every number here is either a count from the shape or a width the cache was built
/// with, so a file written by this build and a file this build will accept are
/// described by the same function. `ctx` is the running context capacity: a file
/// prefilled under a larger one is refused, because its path can be longer than this
/// build has room for.
#[must_use]
pub(crate) fn path_layout(
    shape: &HybridShape,
    geometry: &Geometry,
    ctx: u32,
    identity: CkptIdentity,
) -> PathLayout {
    PathLayout {
        model_id: identity.model_id,
        quant_bits: identity.quant_bits,
        ctx_size: ctx,
        kv_width: kv_row_bytes(shape),
        n_layers: shape.n_layers(),
        n_attn_layers: shape.count(LayerKind::Attn),
        n_gdn_layers: shape.count(LayerKind::Gdn),
        vocab: geometry.vocab,
        gdn_conv_elements: conv_elements(shape),
        gdn_state_elements: state_elements(shape),
        // Neither plane exists in a dense hybrid stack. They are written as zero
        // rather than omitted, because a reader compares all nineteen words: a family
        // that grows one of these planes must not be able to load a file written
        // before it had one.
        ple_window_elements: 0,
        hc_width: 0,
        indexer: false,
        draft_head: false,
    }
}

/// Bytes of one position's key row in the widest attention layer.
///
/// The widest rather than a per-layer list, because this is one word of a guard a
/// reader compares whole: a family whose slots differ in width is separated from
/// one whose slots are all this wide by the layer counts beside it, and a family
/// whose every slot is one width - which is every family that has written a file
/// so far - gets exactly the number it always got.
///
/// Saturating rather than checked: the value is a guard, and a shape wide enough to
/// overflow it would have failed to allocate its pool long before anything asked for
/// a checkpoint.
#[must_use]
pub(crate) fn kv_row_bytes(shape: &HybridShape) -> u32 {
    shape
        .attn
        .iter()
        .map(|layer| {
            layer
                .kv_heads
                .saturating_mul(layer.head_dim)
                .saturating_mul(CACHE_ELEMENT_BYTES)
        })
        .max()
        .unwrap_or(0)
}

/// Elements of one recurrent slot's convolution window: `conv_kernel - 1` rows of
/// the projection's channels.
#[must_use]
pub(crate) fn conv_elements(shape: &HybridShape) -> u32 {
    let qk = shape.gdn.qk_heads.saturating_mul(shape.gdn.head_dim);
    let v = shape.gdn.v_heads.saturating_mul(shape.gdn.head_dim);
    let channels = qk.saturating_mul(2).saturating_add(v);
    shape
        .gdn
        .conv_kernel
        .saturating_sub(1)
        .saturating_mul(channels)
}

/// Elements of one recurrent slot's state matrix: one square matrix per value head.
#[must_use]
pub(crate) fn state_elements(shape: &HybridShape) -> u32 {
    shape
        .gdn
        .v_heads
        .saturating_mul(shape.gdn.head_dim)
        .saturating_mul(shape.gdn.head_dim)
}

#[cfg(test)]
mod tests {

    use super::{conv_elements, kv_row_bytes, state_elements};
    use crate::kv;
    use crate::shape::{
        Activation, AttnLayer, GdnShape, HybridShape, LayerKind, MediaPositions, NormScheme,
    };

    /// A four-slot stack in the real one's pattern, with every width cut down.
    fn shape() -> HybridShape {
        HybridShape {
            layers: vec![
                LayerKind::Gdn,
                LayerKind::Gdn,
                LayerKind::Gdn,
                LayerKind::Attn,
            ],
            attn: vec![AttnLayer::new(2, 1, 32, 16, 16, 10_000.0)],
            hidden: 64,
            vocab: 128,
            intermediate: 128,
            gdn: GdnShape::new(2, 1, 32, 4),
            attn_output_gate: true,
            rms_eps: 1e-6,
            activation: Activation::SwiGlu,
            norms: NormScheme::PreOnly,
            embed_scale: None,
            final_softcap: None,
            layer_scalar: false,
            ple: None,
            media_positions: MediaPositions::Mrope3D,
            v_norm: false,
            moe: None,
        }
    }

    #[test]
    fn the_guard_element_counts_are_the_ones_the_cache_sizes_its_buffers_from() {
        // The payload's element counts and the cache's byte counts are derived
        // independently -- one guards a file, the other allocates a buffer -- and this
        // is the assertion that keeps the two statements of the same geometry in
        // step. A disagreement here would write a file whose state section is sized
        // for a stack the session does not have.
        let shape = shape();
        assert_eq!(
            u64::from(conv_elements(&shape)) * 4,
            kv::conv_state_bytes(&shape),
            "the conv window is the same number of f32s in both derivations"
        );
        assert_eq!(
            u64::from(state_elements(&shape)) * 4,
            kv::state_bytes(&shape),
            "the state matrix is the same number of f32s in both derivations"
        );
    }

    #[test]
    fn a_key_row_is_one_f16_per_head_channel() {
        // One key/value head of 32 channels, two bytes each.
        assert_eq!(kv_row_bytes(&shape()), 64);
    }

    /// A stack of attention slots and nothing else, whose recurrent widths are
    /// the zeros a family without a recurrent block carries.
    fn dense_shape() -> HybridShape {
        HybridShape {
            layers: vec![LayerKind::Attn, LayerKind::Attn],
            attn: vec![AttnLayer::new(2, 1, 32, 16, 16, 10_000.0); 2],
            gdn: GdnShape::new(0, 0, 0, 1),
            ..shape()
        }
    }

    #[test]
    fn a_stack_with_no_recurrent_slots_describes_a_payload_with_no_state_section() {
        let shape = dense_shape();
        assert_eq!(conv_elements(&shape), 0);
        assert_eq!(state_elements(&shape), 0);
        assert_eq!(
            kv::geometry(&shape).ckpt_bytes,
            0,
            "the blob is the recurrent stack alone, and there is none"
        );

        assert!(
            kv_row_bytes(&shape) > 0,
            "the attention rows are still described"
        );
    }
}
