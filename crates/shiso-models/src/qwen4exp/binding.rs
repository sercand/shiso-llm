//! What this family's tensors are called, and which role each one fills.
//!
//! This is the whole of the naming knowledge for this checkpoint family, and it
//! is here rather than in the reader or the engine for the same reason the shape
//! table is: a reader that knew these names could not open another family's
//! file, and an engine that knew them could not run one.
//!
//! # A binding is views, not bytes
//!
//! Every role resolves to a [`TensorView`] into the open container. Nothing is
//! copied and nothing is decoded; the model type above moves each view to the
//! device afterwards, and the container's mapping is what it moves *from*. So a
//! binding that succeeded means the file has every tensor the pass will ask for,
//! at the shape it will ask for it - which is the failure worth catching before
//! a gigabyte of transfers starts rather than in the middle of them.
//!
//! # Why a missing tensor names itself
//!
//! Every lookup below goes through one function that formats the slot number
//! into the name and reports the name it could not find. A checkpoint missing
//! one of five hundred tensors is then a message a reader can act on rather than
//! a count that does not add up.

use shiso_core::error::{CoreError, CoreResult};
use shiso_engine::shape::LayerKind;
use shiso_modelfile::{GgufFile, TensorView};

use crate::qwen4exp::config::{Qwen4ExpConfig, is_attention};

/// The four tensors one stream mix reads.
#[derive(Debug)]
pub struct MixRoles<'a> {
    /// Per-stream norm weight.
    pub norm: TensorView<'a>,
    /// The contraction down to the low-rank vector.
    pub down: TensorView<'a>,
    /// The expansion back to the full stream width.
    pub up: TensorView<'a>,
    /// The per-stream logits the matching combine injects.
    ///
    /// Absent on the mix that collapses the streams for the vocabulary
    /// projection: that one has no residual to inject back into, and an option
    /// says so where a fourth required tensor would have to be invented.
    pub inject: Option<TensorView<'a>>,
}

/// The eight tensors one routed feed-forward reads.
#[derive(Debug)]
pub struct ExpertRoles<'a> {
    /// The router's own projection.
    pub router: TensorView<'a>,
    /// Every routed expert's gate projection, stacked.
    pub gate: TensorView<'a>,
    /// Every routed expert's up projection, stacked.
    pub up: TensorView<'a>,
    /// Every routed expert's down projection, stacked.
    pub down: TensorView<'a>,
    /// The shared expert's scalar gate.
    pub shared_router: TensorView<'a>,
    /// The shared expert's gate projection.
    pub shared_gate: TensorView<'a>,
    /// The shared expert's up projection.
    pub shared_up: TensorView<'a>,
    /// The shared expert's down projection.
    pub shared_down: TensorView<'a>,
}

/// The ten tensors one attention slot reads.
#[derive(Debug)]
pub struct AttnRoles<'a> {
    /// The query projection, which carries the per-head output gate in its own
    /// rows.
    pub q: TensorView<'a>,
    /// The key projection.
    pub k: TensorView<'a>,
    /// The value projection.
    pub v: TensorView<'a>,
    /// Norm over one query head.
    pub q_norm: TensorView<'a>,
    /// Norm over one key head.
    pub k_norm: TensorView<'a>,
    /// The projection back to the residual width.
    pub output: TensorView<'a>,
    /// The indexer's query projection.
    pub idx_q: TensorView<'a>,
    /// The indexer's key projection.
    pub idx_k: TensorView<'a>,
    /// Norm over one indexer query head.
    pub idx_q_norm: TensorView<'a>,
    /// Norm over one indexer key head.
    pub idx_k_norm: TensorView<'a>,
}

/// The nine tensors one recurrent slot reads.
#[derive(Debug)]
pub struct RecurrentRoles<'a> {
    /// The fused query, key and value projection.
    pub qkv: TensorView<'a>,
    /// The output gate's projection.
    pub gate: TensorView<'a>,
    /// The causal convolution's kernel.
    pub conv1d: TensorView<'a>,
    /// Per-head decay base.
    pub a: TensorView<'a>,
    /// The decay projection.
    pub alpha: TensorView<'a>,
    /// The write-strength projection.
    pub beta: TensorView<'a>,
    /// Per-head decay bias.
    pub dt_bias: TensorView<'a>,
    /// Norm over one value head.
    pub norm: TensorView<'a>,
    /// The projection back to the residual width.
    pub out: TensorView<'a>,
}

/// The six tensors the n-gram block reads.
#[derive(Debug)]
pub struct NgramRoles<'a> {
    /// The projection that turns decoded rows into per-stream keys.
    pub key: TensorView<'a>,
    /// The projection that turns decoded rows into values.
    pub value: TensorView<'a>,
    /// Norm over the keys.
    pub norm_key: TensorView<'a>,
    /// Norm over the residual the keys are gated against.
    pub norm_query: TensorView<'a>,
    /// Norm over the gated value the convolution reads.
    pub norm_conv: TensorView<'a>,
    /// The dilated convolution's kernel.
    pub conv1d: TensorView<'a>,
}

/// Which token mixer a slot holds, together with that mixer's tensors.
#[derive(Debug)]
pub enum BlockRoles<'a> {
    /// A recurrent slot.
    Recurrent(Box<RecurrentRoles<'a>>),
    /// An attention slot.
    Attn(Box<AttnRoles<'a>>),
}

impl BlockRoles<'_> {
    /// Which kind of slot this is.
    #[must_use]
    pub const fn kind(&self) -> LayerKind {
        match self {
            Self::Recurrent(_) => LayerKind::Gdn,
            Self::Attn(_) => LayerKind::Attn,
        }
    }
}

/// One bound layer slot.
#[derive(Debug)]
pub struct BoundLayer<'a> {
    /// The mix that feeds the token mixer.
    pub attn_mix: MixRoles<'a>,
    /// The mix that feeds the routed feed-forward.
    pub ffn_mix: MixRoles<'a>,
    /// The token mixer.
    pub block: BlockRoles<'a>,
    /// The routed feed-forward.
    pub experts: ExpertRoles<'a>,
    /// The n-gram block, on the one slot that folds one in.
    pub ngram: Option<NgramRoles<'a>>,
}

/// Every tensor a draft head has to carry.
///
/// The head's block is bound by the same three helpers a target layer's is,
/// against the sidecar rather than the checkpoint: its tensors are named
/// `blk.0.*` and are the same shape and format as a target attention slot's, so
/// there is nothing here for them to do differently. What is the head's own is
/// the four tensors in front of the block and the mix behind it.
#[derive(Debug)]
pub(crate) struct DraftBinding<'a> {
    /// Norm over the token embedding the head starts from.
    pub(crate) norm_embed: TensorView<'a>,
    /// Norm over the target hidden state it starts from.
    pub(crate) norm_hidden: TensorView<'a>,
    /// The projection of the normed embedding.
    pub(crate) fc_embed: TensorView<'a>,
    /// The projection of the normed hidden state.
    pub(crate) fc_hidden: TensorView<'a>,
    /// The head's single layer.
    pub(crate) layer: BoundLayer<'a>,
    /// The mix that collapses the head's streams.
    pub(crate) output_mix: MixRoles<'a>,
}

/// Every tensor a checkpoint of this family has to carry.
#[derive(Debug)]
pub struct ModelBinding<'a> {
    /// The vocabulary table.
    pub token_embd: TensorView<'a>,
    /// The output projection, when the file stores one of its own.
    pub output: Option<TensorView<'a>>,
    /// The mix that collapses the streams for the vocabulary projection.
    pub output_mix: MixRoles<'a>,
    /// One entry per layer slot, index zero first.
    pub layers: Vec<BoundLayer<'a>>,
    /// The n-gram hash table, when the file carries one.
    ///
    /// It is separate from the block that reads it because it is not a weight in
    /// the usual sense: it is hundreds of millions of rows that are streamed
    /// from disk a handful at a time rather than made resident.
    pub ngram_table: Option<TensorView<'a>>,
    /// The per-head bias the trellis row codec finishes every weight with.
    ///
    /// A tensor of the table rather than of the layer, and absent for a
    /// checkpoint whose rows are block-quantized, because that codec has no
    /// bias at all. Small enough to be resident, unlike the table it belongs
    /// to.
    pub ngram_bias: Option<TensorView<'a>>,
}

/// One tensor by name, or a failure that names it.
fn require<'a>(file: &'a GgufFile, name: &str) -> CoreResult<TensorView<'a>> {
    file.get(name).ok_or_else(|| CoreError::ConfigMissing {
        key: name.to_owned(),
    })
}

/// The stream mix at one slot, on one side of it.
///
/// `side` is the half of the layer this mix feeds, which is the only thing that
/// differs between the two: everything else about them is identical, which is
/// why one function covers both.
fn mix<'f>(file: &'f GgufFile, il: u32, side: &str) -> CoreResult<MixRoles<'f>> {
    Ok(MixRoles {
        norm: require(file, &format!("blk.{il}.hc_{side}_norm.weight"))?,
        down: require(file, &format!("blk.{il}.hc_{side}_down.weight"))?,
        up: require(file, &format!("blk.{il}.hc_{side}_up.weight"))?,
        inject: Some(require(file, &format!("blk.{il}.hc_{side}_inject.weight"))?),
    })
}

/// The attention slot at `il`.
fn attn(file: &GgufFile, il: u32) -> CoreResult<AttnRoles<'_>> {
    Ok(AttnRoles {
        q: require(file, &format!("blk.{il}.attn_q.weight"))?,
        k: require(file, &format!("blk.{il}.attn_k.weight"))?,
        v: require(file, &format!("blk.{il}.attn_v.weight"))?,
        q_norm: require(file, &format!("blk.{il}.attn_q_norm.weight"))?,
        k_norm: require(file, &format!("blk.{il}.attn_k_norm.weight"))?,
        output: require(file, &format!("blk.{il}.attn_output.weight"))?,
        idx_q: require(file, &format!("blk.{il}.indexer.q_proj.weight"))?,
        idx_k: require(file, &format!("blk.{il}.indexer.k_proj.weight"))?,
        idx_q_norm: require(file, &format!("blk.{il}.indexer.q_norm.weight"))?,
        idx_k_norm: require(file, &format!("blk.{il}.indexer.k_norm.weight"))?,
    })
}

/// The recurrent slot at `il`.
///
/// Its tensors are named in the container's state-space vocabulary, which is
/// what the writer emits: the convolution, the decay base and the decay bias all
/// carry that prefix even though the block is a gated delta rule rather than a
/// state-space model.
fn recurrent(file: &GgufFile, il: u32) -> CoreResult<RecurrentRoles<'_>> {
    Ok(RecurrentRoles {
        qkv: require(file, &format!("blk.{il}.attn_qkv.weight"))?,
        gate: require(file, &format!("blk.{il}.attn_gate.weight"))?,
        conv1d: require(file, &format!("blk.{il}.ssm_conv1d.weight"))?,
        a: require(file, &format!("blk.{il}.ssm_a"))?,
        alpha: require(file, &format!("blk.{il}.ssm_alpha.weight"))?,
        beta: require(file, &format!("blk.{il}.ssm_beta.weight"))?,
        dt_bias: require(file, &format!("blk.{il}.ssm_dt.bias"))?,
        norm: require(file, &format!("blk.{il}.ssm_norm.weight"))?,
        out: require(file, &format!("blk.{il}.ssm_out.weight"))?,
    })
}

/// The routed feed-forward at `il`.
fn experts(file: &GgufFile, il: u32) -> CoreResult<ExpertRoles<'_>> {
    Ok(ExpertRoles {
        router: require(file, &format!("blk.{il}.ffn_gate_inp.weight"))?,
        gate: require(file, &format!("blk.{il}.ffn_gate_exps.weight"))?,
        up: require(file, &format!("blk.{il}.ffn_up_exps.weight"))?,
        down: require(file, &format!("blk.{il}.ffn_down_exps.weight"))?,
        shared_router: require(file, &format!("blk.{il}.ffn_gate_inp_shexp.weight"))?,
        shared_gate: require(file, &format!("blk.{il}.ffn_gate_shexp.weight"))?,
        shared_up: require(file, &format!("blk.{il}.ffn_up_shexp.weight"))?,
        shared_down: require(file, &format!("blk.{il}.ffn_down_shexp.weight"))?,
    })
}

/// The n-gram block at `il`.
fn ngram(file: &GgufFile, il: u32) -> CoreResult<NgramRoles<'_>> {
    Ok(NgramRoles {
        key: require(file, &format!("blk.{il}.ple_key.weight"))?,
        value: require(file, &format!("blk.{il}.ple_value.weight"))?,
        norm_key: require(file, &format!("blk.{il}.ple_norm_key.weight"))?,
        norm_query: require(file, &format!("blk.{il}.ple_norm_query.weight"))?,
        norm_conv: require(file, &format!("blk.{il}.ple_norm_conv.weight"))?,
        conv1d: require(file, &format!("blk.{il}.ple_conv1d.weight"))?,
    })
}

/// The name the streamed n-gram hash table is stored under.
pub const NGRAM_TABLE: &str = "per_layer_token_embd.weight";

/// The name that table's per-head bias is stored under.
pub const NGRAM_BIAS: &str = "per_layer_token_embd.bias";

/// Binds every tensor a checkpoint of this family carries.
///
/// The layer schedule comes from the config rather than from the file's tensor
/// set: a slot is bound as attention or as recurrent because the schedule says
/// so, and a file whose tensors disagree fails on the first missing name. That
/// is the right order, because the schedule is also what the ordinals and the
/// cache's per-layer planes are derived from.
///
/// # Errors
///
/// [`CoreError::ConfigMissing`] naming the first tensor the file does not
/// carry.
pub fn bind_qwen4exp<'f>(
    file: &'f GgufFile,
    config: &Qwen4ExpConfig,
) -> CoreResult<ModelBinding<'f>> {
    let interval = config.full_attention_interval();
    let n_layers = config.shape().n_layers();
    let mut layers = Vec::with_capacity(n_layers as usize);
    for il in 0..n_layers {
        let block = if is_attention(il, interval) {
            BlockRoles::Attn(Box::new(attn(file, il)?))
        } else {
            BlockRoles::Recurrent(Box::new(recurrent(file, il)?))
        };
        let ngram_here = config
            .shape()
            .ple
            .is_some_and(|ple| ple.layer.get() == il)
            .then(|| ngram(file, il))
            .transpose()?;
        layers.push(BoundLayer {
            attn_mix: mix(file, il, "attn")?,
            ffn_mix: mix(file, il, "ffn")?,
            block,
            experts: experts(file, il)?,
            ngram: ngram_here,
        });
    }
    Ok(ModelBinding {
        token_embd: require(file, "token_embd.weight")?,
        // A file without its own projection reads the embedding table the other
        // way round, which is why this is an option rather than a requirement.
        output: file.get("output.weight"),
        output_mix: MixRoles {
            norm: require(file, "output_hc_norm.weight")?,
            down: require(file, "output_hc_down.weight")?,
            up: require(file, "output_hc_up.weight")?,
            inject: None,
        },
        layers,
        ngram_table: file.get(NGRAM_TABLE),
        ngram_bias: file.get(NGRAM_BIAS),
    })
}

/// The architecture a draft head sidecar declares.
const DRAFT_ARCH: &str = "qwen4exp-mtp";

/// Binds a draft head sidecar against the target it drafts for.
///
/// The head is refused rather than adapted when its widths are not the
/// target's: it is a *second model* over the same residual stream, and a head
/// whose block was built at a different width would bind tensors of the wrong
/// shape into launches sized by the target's geometry. Every one of them would
/// succeed.
///
/// # Errors
///
/// [`CoreError::UnknownArch`] when the file is not a draft head at all,
/// [`CoreError::ShapeMismatch`] when it is one for a different residual width,
/// and [`CoreError::ConfigMissing`] naming the first tensor it does not hold.
pub(crate) fn bind_draft<'f>(
    file: &'f GgufFile,
    config: &Qwen4ExpConfig,
) -> CoreResult<DraftBinding<'f>> {
    let meta = file.metadata();
    let arch = meta.architecture().unwrap_or_default().to_owned();
    if arch != DRAFT_ARCH {
        return Err(CoreError::UnknownArch { model_type: arch });
    }
    let hidden = u64::from(config.shape().hidden);
    let declared = meta.arch_u32("embedding_length").map_or(0, u64::from);
    if declared != hidden {
        return Err(CoreError::ShapeMismatch {
            field: "draft head residual width",
            expected: hidden,
            got: declared,
        });
    }
    let blocks = meta.arch_u32("block_count").unwrap_or(0);
    if blocks != 1 {
        return Err(CoreError::ShapeMismatch {
            field: "draft head block count",
            expected: 1,
            got: u64::from(blocks),
        });
    }
    Ok(DraftBinding {
        norm_embed: require(file, "mtp.pre_fc_norm_embedding.weight")?,
        norm_hidden: require(file, "mtp.pre_fc_norm_hidden.weight")?,
        fc_embed: require(file, "mtp.fc_embedding.weight")?,
        fc_hidden: require(file, "mtp.fc_hidden.weight")?,
        layer: BoundLayer {
            attn_mix: mix(file, 0, "attn")?,
            ffn_mix: mix(file, 0, "ffn")?,
            block: BlockRoles::Attn(Box::new(attn(file, 0)?)),
            experts: experts(file, 0)?,
            ngram: None,
        },
        output_mix: MixRoles {
            norm: require(file, "mtp.hc_norm.weight")?,
            down: require(file, "mtp.hc_down.weight")?,
            up: require(file, "mtp.hc_up.weight")?,
            inject: None,
        },
    })
}
