//! The fixed shape table for this checkpoint family, and the metadata that has
//! to agree with it.
//!
//! Every dimension the kernels were written for is a constant below, and a
//! checkpoint that claims a different one is refused by name. That is the whole
//! point of the table: a file whose head width disagrees would otherwise produce
//! wrong logits with no diagnostic at all, because every row read from the wrong
//! offset still decodes to finite numbers.
//!
//! # Why this table is here and not below
//!
//! It was in the shared crate once, which meant the bottom of the stack carried
//! one checkpoint family's dimensions and a match arm on its name. What the
//! forward pass actually needs is a [`SparseShape`]; which numbers fill it in is
//! a fact about *this* family and belongs in the crate that can read this
//! family's files.
//!
//! # Why the geometry comes out of the checkpoint's own metadata
//!
//! There is no separate configuration document for this family: the container
//! carries every dimension as metadata under its own architecture prefix, and
//! the vocabulary size is only knowable from the tokenizer array. So the reader
//! here is a metadata reader rather than a document parser, and the check it
//! performs is between the file's own claims and this table.

use std::sync::Arc;

use shiso_core::config::HfConfig;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{LayerIdx, TokenId};
use shiso_engine::shape::{GdnShape, LayerKind};
use shiso_engine::sparse::shape::{
    GqaShape, IndexerShape, MoeShape, MropeSections, PleShape, SparseShape,
};
use shiso_modelfile::{GgufFile, GgufMetadata, MetaValue};
use shiso_ple::{PleParams, PleParamsDesc, PleRowType};

/// The container architecture whose metadata this crate can read.
pub const ARCH: &str = "qwen4exp";

/// The `model_type` spellings a config document may use for this family.
///
/// Two, because the two places the family is named disagree and both are
/// authoritative where they are written. The container's metadata carries
/// `qwen4exp`, which is [`ARCH`]; the published checkpoint's `config.json`
/// carries `qwen4_exp`, which is what the transformers class is registered
/// under. Neither is a mistake to be corrected in the other's favour, and a
/// reader that knew only the first refused the real checkpoint's own document.
const ARCHES: [&str; 2] = [ARCH, "qwen4_exp"];

/// Layer slots.
pub const N_LAYERS: u32 = 48;
/// Width of one residual stream.
pub const HIDDEN: u32 = 2_560;
/// Vocabulary size.
pub const VOCAB: u32 = 248_320;
/// Attention query heads.
pub const Q_HEADS: u32 = 24;
/// Attention key and value heads.
pub const KV_HEADS: u32 = 2;
/// Width of one attention head, which is not `hidden / q_heads`.
pub const HEAD_DIM: u32 = 256;
/// Leading dimensions of a head the rotary embedding touches.
pub const N_ROT: u32 = 64;
/// Residual streams the hyper-connections carry.
pub const STREAMS: u32 = 4;
/// Inner width of the low-rank pair that mixes the streams.
pub const STREAM_RANK: u32 = 320;
/// Routed experts per layer.
pub const EXPERTS: u32 = 512;
/// Experts one token is routed to.
pub const EXPERTS_USED: u32 = 10;
/// Always-on experts every token additionally pays for.
pub const EXPERTS_SHARED: u32 = 1;
/// Inner width of one expert.
pub const EXPERT_INTERMEDIATE: u32 = 640;
/// Recurrent value heads.
pub const GDN_V_HEADS: u32 = 48;
/// Recurrent query and key heads.
pub const GDN_QK_HEADS: u32 = 16;
/// Width of one recurrent head.
pub const GDN_HEAD_DIM: u32 = 128;
/// Width of the recurrent causal convolution.
pub const GDN_CONV: u32 = 4;
/// Every slot whose index is one below a multiple of this holds attention.
pub const FULL_ATTENTION_INTERVAL: u32 = 4;
/// Indexer query heads.
pub const INDEXER_HEADS: u32 = 4;
/// Width of one indexer head.
pub const INDEXER_HEAD_DIM: u32 = 128;
/// Positions the indexer's selection may reach.
pub const INDEXER_TOP_K: u32 = 2_048;
/// Positions one pooled indexer key block averages over.
pub const INDEXER_COMPRESS: u32 = 4;
/// The one slot that folds in the n-gram block.
pub const PLE_LAYER: u32 = 1;
/// Gram widths the n-gram hash table is partitioned into.
pub const PLE_NGRAM: u32 = 3;
/// Heads each gram width contributes.
pub const PLE_HEADS_PER_NGRAM: u32 = 8;
/// Width of one n-gram head's embedding row.
pub const PLE_HEAD_DIM: u32 = 160;
/// Width of the n-gram block's dilated convolution.
pub const PLE_CONV: u32 = 4;
/// The n-gram table's own sequence boundary token.
///
/// Not the chat end-of-turn token and not in the chat vocabulary at all: it is
/// the id the hash reads for a position before the start of a sequence, and it
/// truncates a gram window transitively wherever it appears.
pub const PLE_EOS_TOKEN: u32 = 248_044;
/// Rotary frequency base.
pub const ROPE_THETA: f32 = 1.0e7;
/// Epsilon every normalization adds.
pub const RMS_EPS: f32 = 1.0e-6;
/// Positions the rotary table was trained over.
pub const ROPE_ORIGINAL_CONTEXT: u64 = 262_144;
/// Rotary frequency pairs driven by the time axis.
pub const MROPE_TIME: u32 = 11;
/// Rotary frequency pairs driven by the height axis.
pub const MROPE_HEIGHT: u32 = 11;
/// Rotary frequency pairs driven by the width axis.
pub const MROPE_WIDTH: u32 = 10;

/// The streamed n-gram table's own numbers.
///
/// Everything about that table the geometry does not already say: the
/// multipliers the hash mixes each gram slot with, where each hash head's rows
/// begin and how many it has, and the table's own sequence boundary token.
/// They are here rather than in the engine because they are metadata of one
/// checkpoint family, and here rather than in [`SparseShape`] because they are
/// facts about a file that is never made resident rather than about the
/// arithmetic of a layer.
///
/// Deliberately NOT `#[non_exhaustive]`: a reader for a second container format
/// has to build one, and a field added here is a number that reader is not
/// supplying yet, which should stop its compile rather than silently default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramHash {
    /// One multiplier per gram slot, all odd.
    pub multipliers: Vec<u64>,
    /// Where each hash head's rows start in the table.
    pub head_offsets: Vec<u64>,
    /// How many rows each hash head has.
    pub head_vocab: Vec<u64>,
    /// The table's own sequence boundary token.
    pub eos_token: TokenId,
}

impl NgramHash {
    /// Rows the whole table holds, summed over the heads.
    #[must_use]
    pub fn total_rows(&self) -> u64 {
        self.head_vocab.iter().copied().sum()
    }

    /// The checked reader parameters for a table stored as `row_type`.
    ///
    /// `head_bias` is the table's per-head bias, head-major, which the trellis
    /// codec finishes every weight with and the block-quantized codec has none
    /// of. Every consistency check - odd multipliers, heads that tile the table
    /// without a gap, a bias of the right length - happens inside, once, and is
    /// why everything downstream can read these numbers without asking again.
    ///
    /// # Errors
    ///
    /// Whatever [`PleParams::new`] reports, carried into the shared error
    /// vocabulary.
    pub fn params(
        &self,
        ple: PleShape,
        row_type: PleRowType,
        head_bias: Option<Arc<Vec<u16>>>,
    ) -> CoreResult<PleParams> {
        Ok(PleParams::new(PleParamsDesc {
            row_type,
            ngram_size: ple.ngram,
            heads_per_ngram: ple.heads_per_ngram,
            head_dim: ple.head_dim,
            eos_token: self.eos_token,
            multipliers: self.multipliers.clone(),
            head_offsets: self.head_offsets.clone(),
            head_vocab: self.head_vocab.clone(),
            head_bias,
        })?)
    }
}

/// What this family's checkpoint says, once it has been checked.
///
/// Three things, and they are together because a checkpoint that produced one of
/// them without the others would be half-loaded: the geometry runs the stack,
/// the end-of-turn token is what a scheduler compares against, and the attention
/// interval is what the tensor binding cross-checks the file's own per-layer
/// table with.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Qwen4ExpConfig {
    shape: SparseShape,
    eos_token: TokenId,
    full_attention_interval: u32,
    ngram: NgramHash,
}

/// Whether slot `il` holds attention.
///
/// A slot is an attention slot when its index is one below a multiple of the
/// interval, which makes the *last* slot always one. The container states this
/// twice - as the interval and as a per-layer table - and the two are checked
/// against each other, because the tensor set follows the schedule and a
/// disagreement would bind a recurrent slot's weights into an attention slot.
#[must_use]
pub const fn is_attention(il: u32, interval: u32) -> bool {
    interval != 0 && (il + 1).is_multiple_of(interval)
}

/// The layer schedule an interval implies.
#[must_use]
pub fn layer_kinds(n_layers: u32, interval: u32) -> Vec<LayerKind> {
    (0..n_layers)
        .map(|il| {
            if is_attention(il, interval) {
                LayerKind::Attn
            } else {
                LayerKind::Gdn
            }
        })
        .collect()
}

/// The geometry this family's kernels were written for.
#[must_use]
pub fn shape() -> SparseShape {
    SparseShape {
        layers: layer_kinds(N_LAYERS, FULL_ATTENTION_INTERVAL),
        hidden: HIDDEN,
        vocab: VOCAB,
        streams: STREAMS,
        stream_rank: STREAM_RANK,
        gqa: GqaShape::new(Q_HEADS, KV_HEADS, HEAD_DIM, N_ROT),
        gdn: GdnShape::new(GDN_V_HEADS, GDN_QK_HEADS, GDN_HEAD_DIM, GDN_CONV),
        moe: MoeShape::new(EXPERTS, EXPERTS_USED, EXPERTS_SHARED, EXPERT_INTERMEDIATE),
        indexer: IndexerShape::new(
            INDEXER_HEADS,
            INDEXER_HEAD_DIM,
            INDEXER_TOP_K,
            INDEXER_COMPRESS,
        ),
        ple: Some(PleShape::new(
            LayerIdx::new(PLE_LAYER),
            PLE_NGRAM,
            PLE_HEADS_PER_NGRAM,
            PLE_HEAD_DIM,
            PLE_CONV,
        )),
        mrope: MropeSections::new(MROPE_TIME, MROPE_HEIGHT, MROPE_WIDTH),
        rms_eps: RMS_EPS,
        rope_theta: ROPE_THETA,
    }
}

impl Qwen4ExpConfig {
    /// The four facts, given directly.
    ///
    /// The reading path is [`Qwen4ExpConfig::from_metadata`]; this exists for a
    /// caller that already has the geometry - a test that needs a stack small
    /// enough to fit in a fixture, or a container that carries its dimensions
    /// some other way.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a geometry with no n-gram slot. Every
    /// checkpoint of this family folds one in, and a config carrying a hash
    /// table for a stack that would never gather a row is a value nothing can
    /// use.
    pub fn new(
        shape: SparseShape,
        eos_token: TokenId,
        full_attention_interval: u32,
        ngram: NgramHash,
    ) -> CoreResult<Self> {
        if shape.ple.is_none() {
            return Err(CoreError::ShapeMismatch {
                field: "n-gram slot",
                expected: 1,
                got: 0,
            });
        }
        Ok(Self {
            shape,
            eos_token,
            full_attention_interval,
            ngram,
        })
    }

    /// The streamed n-gram table's own numbers.
    #[must_use]
    pub const fn ngram(&self) -> &NgramHash {
        &self.ngram
    }

    /// The geometry.
    #[must_use]
    pub const fn shape(&self) -> &SparseShape {
        &self.shape
    }

    /// The token that ends a turn.
    #[must_use]
    pub const fn eos_token(&self) -> TokenId {
        self.eos_token
    }

    /// The attention interval the layer schedule follows.
    #[must_use]
    pub const fn full_attention_interval(&self) -> u32 {
        self.full_attention_interval
    }

    /// Whether a parsed config document names this family.
    ///
    /// Separate from [`Qwen4ExpConfig::from_hf`] so that a caller can refuse an
    /// unservable checkpoint before it pays for a device or opens a weights
    /// file.
    #[must_use]
    pub fn describes(config: &HfConfig) -> bool {
        ARCHES.contains(&config.model_type())
    }

    /// Reads the geometry for the checkpoint at `checkpoint`, whose family the
    /// config document at `config` names.
    ///
    /// The document is used for recognition and for nothing else, and that is
    /// not a shortcut. This family carries no geometry in a config document:
    /// every dimension is in the container's own metadata, and the vocabulary
    /// size is only knowable from the tokenizer array there. A reader that took
    /// dimensions from the document would be parsing key names no checkpoint of
    /// this family has ever written, and would agree with the container by
    /// coincidence or disagree with it silently.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownArch`] when the document describes another family -
    /// which is the one failure a caller reads as "not mine" - and whatever
    /// [`Qwen4ExpConfig::from_metadata`] reports once the container is open.
    pub fn from_hf(config: &HfConfig, checkpoint: &std::path::Path) -> CoreResult<Self> {
        if !Self::describes(config) {
            return Err(CoreError::UnknownArch {
                model_type: config.model_type().to_owned(),
            });
        }
        let file = GgufFile::open_shared(checkpoint).map_err(|source| CoreError::Backend {
            backend: "qwen4exp",
            op: "read the checkpoint's metadata",
            source: Box::new(source),
        })?;
        Self::from_metadata(file.metadata())
    }

    /// Reads and checks the geometry out of a container's metadata.
    ///
    /// Every dimension below is compared with the constant the kernels were
    /// written for. Nothing is adopted from the file: a value that disagrees is
    /// a refusal naming the key, because a stack built from a shape the kernels
    /// were not compiled for produces confident nonsense rather than an error.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownArch`] when the metadata describes another
    /// architecture, [`CoreError::ConfigMissing`] when a required key is absent,
    /// and [`CoreError::ShapeMismatch`] naming the first dimension that
    /// disagrees.
    pub fn from_metadata(meta: &GgufMetadata) -> CoreResult<Self> {
        if meta.architecture() != Some(ARCH) {
            return Err(CoreError::UnknownArch {
                model_type: meta.architecture().unwrap_or_default().to_owned(),
            });
        }
        expect(meta, "block_count", N_LAYERS)?;
        expect(meta, "embedding_length", HIDDEN)?;
        expect(meta, "attention.head_count", Q_HEADS)?;
        expect(meta, "attention.head_count_kv", KV_HEADS)?;
        expect(meta, "attention.key_length", HEAD_DIM)?;
        expect(meta, "attention.value_length", HEAD_DIM)?;
        expect(meta, "rope.dimension_count", N_ROT)?;
        expect(meta, "expert_count", EXPERTS)?;
        expect(meta, "expert_used_count", EXPERTS_USED)?;
        expect(meta, "expert_feed_forward_length", EXPERT_INTERMEDIATE)?;
        expect(
            meta,
            "expert_shared_feed_forward_length",
            EXPERT_INTERMEDIATE,
        )?;
        // The recurrent block is described in the container's state-space
        // vocabulary: the group count is the query and key head count, the time
        // step rank is the value head count, the state size is the head width
        // they share, and the inner size is the width they multiply out to.
        expect(meta, "ssm.group_count", GDN_QK_HEADS)?;
        expect(meta, "ssm.time_step_rank", GDN_V_HEADS)?;
        expect(meta, "ssm.state_size", GDN_HEAD_DIM)?;
        expect(meta, "ssm.conv_kernel", GDN_CONV)?;
        expect(meta, "ssm.inner_size", GDN_V_HEADS * GDN_HEAD_DIM)?;
        expect(meta, "full_attention_interval", FULL_ATTENTION_INTERVAL)?;
        expect(meta, "hyper_connection.count", STREAMS)?;
        expect(meta, "hyper_connection.low_rank", STREAM_RANK)?;
        expect(meta, "attention.indexer.head_count", INDEXER_HEADS)?;
        expect(meta, "attention.indexer.key_length", INDEXER_HEAD_DIM)?;
        expect(meta, "attention.indexer.top_k", INDEXER_TOP_K)?;
        expect(meta, "ple.ngram_size", PLE_NGRAM)?;
        expect(meta, "ple.heads_per_ngram", PLE_HEADS_PER_NGRAM)?;
        expect(meta, "ple.conv_kernel", PLE_CONV)?;
        expect(meta, "ple.eos_token_id", PLE_EOS_TOKEN)?;
        // The hash heads times one row's width is the residual: the n-gram
        // block's decoded rows ARE a residual-width activation, laid out head
        // after head, so a file whose row width disagrees would decode into the
        // wrong columns of it.
        expect(meta, "embedding_length_per_layer_input", PLE_HEAD_DIM)?;

        // There is no vocabulary-size key in this architecture; the tokenizer
        // array is the only source, and its length is the number.
        let tokens = meta
            .get_array("tokenizer.ggml.tokens")
            .ok_or_else(|| missing("tokenizer.ggml.tokens"))?;
        let vocab = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        if vocab != VOCAB {
            return Err(CoreError::ShapeMismatch {
                field: "tokenizer.ggml.tokens",
                expected: u64::from(VOCAB),
                got: u64::from(vocab),
            });
        }

        let eos = meta
            .get_u32("tokenizer.ggml.eos_token_id")
            .ok_or_else(|| missing("tokenizer.ggml.eos_token_id"))?;
        let schedule = layer_schedule(meta)?;
        Ok(Self {
            shape: shape(),
            eos_token: TokenId::new(eos),
            full_attention_interval: schedule,
            ngram: ngram_hash(meta)?,
        })
    }
}

/// Reads one architecture-prefixed key and checks it against the table.
fn expect(meta: &GgufMetadata, suffix: &'static str, want: u32) -> CoreResult<()> {
    let got = meta
        .arch_u32(suffix)
        .ok_or_else(|| missing(&format!("{ARCH}.{suffix}")))?;
    if got == want {
        return Ok(());
    }
    Err(CoreError::ShapeMismatch {
        field: suffix,
        expected: u64::from(want),
        got: u64::from(got),
    })
}

/// One metadata value as an unsigned count, whatever width it was written at.
///
/// Writers disagree about the width a small count is stored in, and a reader
/// that insisted on one of them would refuse a file for a reason that has
/// nothing to do with the model.
fn unsigned(value: &MetaValue) -> Option<u64> {
    match value {
        MetaValue::U8(v) => Some(u64::from(*v)),
        MetaValue::U16(v) => Some(u64::from(*v)),
        MetaValue::U32(v) => Some(u64::from(*v)),
        MetaValue::U64(v) => Some(*v),
        MetaValue::I8(v) => u64::try_from(*v).ok(),
        MetaValue::I16(v) => u64::try_from(*v).ok(),
        MetaValue::I32(v) => u64::try_from(*v).ok(),
        MetaValue::I64(v) => u64::try_from(*v).ok(),
        _ => None,
    }
}

/// The n-gram table's own numbers, out of the container's metadata.
///
/// Three arrays, and nothing about them is checkable from a single value: the
/// multipliers have to be odd or the hash stops being a bijection on the low
/// bits and half the table folds onto the other half, and the head ranges have
/// to tile the table without a gap or the identifier computed for a head is out
/// of range. Both checks are the reader's, in [`NgramHash::params`], and both
/// are refusals rather than clamps - a clamped row decodes to finite numbers.
fn ngram_hash(meta: &GgufMetadata) -> CoreResult<NgramHash> {
    let heads = (PLE_NGRAM - 1) * PLE_HEADS_PER_NGRAM;
    Ok(NgramHash {
        multipliers: u64_array(meta, "qwen4exp.ple.layer_multipliers", PLE_NGRAM)?,
        head_offsets: u64_array(meta, "qwen4exp.ple.head_offsets", heads)?,
        head_vocab: u64_array(meta, "qwen4exp.ple.head_vocab_sizes", heads)?,
        eos_token: TokenId::new(PLE_EOS_TOKEN),
    })
}

/// One metadata array of counts, at the length this family's geometry implies.
///
/// The length is checked here rather than left to the reader below so that a
/// file carrying a table for a different head count is refused by the key that
/// is wrong instead of by a downstream arithmetic error.
fn u64_array(meta: &GgufMetadata, key: &str, want: u32) -> CoreResult<Vec<u64>> {
    let array = meta.get_array(key).ok_or_else(|| missing(key))?;
    let len = u32::try_from(array.len()).unwrap_or(u32::MAX);
    if len != want {
        return Err(CoreError::ShapeMismatch {
            field: "n-gram table array",
            expected: u64::from(want),
            got: u64::from(len),
        });
    }
    array
        .iter()
        .map(|entry| {
            entry
                .ok()
                .as_ref()
                .and_then(unsigned)
                .ok_or_else(|| CoreError::InvalidValue {
                    key: "n-gram table array",
                    value: format!("{key} holds an entry that is not a count"),
                })
        })
        .collect()
}

/// Names an absent key.
fn missing(key: &str) -> CoreError {
    CoreError::ConfigMissing {
        key: key.to_owned(),
    }
}

/// Checks the per-layer compression table against the interval, and returns the
/// interval.
///
/// The container states the schedule twice, and this is where the two are made
/// to agree. A file whose table disagreed with its own interval would bind a
/// recurrent slot's weights into an attention slot - which fails at load only
/// because the two slots have different tensor names, and would be silent if
/// they did not.
fn layer_schedule(meta: &GgufMetadata) -> CoreResult<u32> {
    let interval = meta
        .arch_u32("full_attention_interval")
        .ok_or_else(|| missing(&format!("{ARCH}.full_attention_interval")))?;
    let Some(ratios) = meta.get_array("qwen4exp.attention.compress_ratios") else {
        // Not every writer emits the table; the interval alone is enough, and a
        // file without it is not thereby suspect.
        return Ok(interval);
    };
    let len = u32::try_from(ratios.len()).unwrap_or(u32::MAX);
    if len < N_LAYERS {
        return Err(CoreError::ShapeMismatch {
            field: "attention.compress_ratios",
            expected: u64::from(N_LAYERS),
            got: u64::from(len),
        });
    }
    for il in 0..N_LAYERS {
        let value = ratios
            .get(il as usize)
            .ok()
            .as_ref()
            .and_then(unsigned)
            .unwrap_or(0);
        // A nonzero compression ratio marks an attention slot: it is how many
        // positions that slot's indexer pools into one key block, and a
        // recurrent slot has no indexer to pool for.
        let attends = value != 0;
        if attends != is_attention(il, interval) {
            return Err(CoreError::InvalidValue {
                key: "attention.compress_ratios",
                value: format!(
                    "slot {il} says {} but an interval of {interval} says {}",
                    if attends { "attention" } else { "recurrent" },
                    if is_attention(il, interval) {
                        "attention"
                    } else {
                        "recurrent"
                    }
                ),
            });
        }
        if attends && value != u64::from(INDEXER_COMPRESS) {
            return Err(CoreError::ShapeMismatch {
                field: "attention.compress_ratios",
                expected: u64::from(INDEXER_COMPRESS),
                got: value,
            });
        }
    }
    Ok(interval)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_last_slot_is_always_an_attention_slot() {
        // The schedule is stated as "one below a multiple of the interval",
        // which puts an attention slot at the top of the stack whatever the
        // layer count is - and the draft head above it reads that slot's cache.
        let kinds = layer_kinds(N_LAYERS, FULL_ATTENTION_INTERVAL);
        assert_eq!(kinds.last(), Some(&LayerKind::Attn));
        assert_eq!(kinds.first(), Some(&LayerKind::Gdn));
        assert_eq!(kinds.len(), N_LAYERS as usize);
    }

    #[test]
    fn three_slots_in_four_are_recurrent() {
        let kinds = layer_kinds(N_LAYERS, FULL_ATTENTION_INTERVAL);
        let attn = kinds.iter().filter(|k| **k == LayerKind::Attn).count();
        assert_eq!(attn, 12, "forty-eight slots at one in four");
        assert_eq!(kinds.len() - attn, 36);
    }

    #[test]
    fn the_shape_this_crate_carries_passes_the_engine_s_own_checks() {
        shape()
            .validate()
            .expect("the table the kernels were written for has to be self-consistent");
    }

    #[test]
    fn the_rotary_sections_tile_the_frequency_pairs_exactly() {
        // Thirty-two pairs over three axes. A triple that did not tile would
        // leave the tail pairs unrotated, which no assertion downstream catches.
        assert_eq!(MROPE_TIME + MROPE_HEIGHT + MROPE_WIDTH, N_ROT / 2);
        assert!(shape().mrope.covers(N_ROT));
    }

    #[test]
    fn the_n_gram_heads_multiply_out_to_the_residual_width() {
        let ple = shape().ple.expect("this family has an n-gram block");
        assert_eq!(ple.heads(), 16);
        assert_eq!(ple.heads() * ple.head_dim, HIDDEN);
        assert_eq!(
            ple.history(),
            9,
            "three dilated taps at a gram count of three"
        );
    }

    #[test]
    fn the_derived_widths_are_what_the_kernels_index_by() {
        let shape = shape();
        assert_eq!(shape.stream_dim(), 10_240);
        assert_eq!(shape.q_dim(), 6_144);
        assert_eq!(shape.kv_dim(), 512);
        assert_eq!(shape.gdn_qk_dim(), 2_048);
        assert_eq!(shape.gdn_v_dim(), 6_144);
        assert_eq!(shape.gdn_in_dim(), 10_240);
        assert_eq!(shape.indexer_q_dim(), 512);
    }

    #[test]
    fn the_selection_engages_only_once_the_context_outgrows_its_budget() {
        let shape = shape();
        // Five hundred and twelve complete blocks is the budget; the pass after
        // the one that completes the five hundred and thirteenth is the first
        // that selects.
        assert!(!shape.indexer.sparse_at(2_048));
        assert!(!shape.indexer.sparse_at(2_051));
        assert!(shape.indexer.sparse_at(2_052));
        assert_eq!(shape.indexer.budget_blocks(), 512);
    }
}
