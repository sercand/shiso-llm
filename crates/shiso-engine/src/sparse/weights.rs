//! The roles a resident checkpoint fills, and the descriptors a plan is built
//! from.
//!
//! Two parallel vocabularies live here and they are not redundant.
//!
//! The *role* structs - [`StreamMix`], [`ExpertBank`], [`SparseAttn`] and the
//! rest - hold device memory. A model crate fills them in once at load, and the
//! forward pass reads them. They name what a tensor does in the architecture,
//! never what it is called in a file: which file the weights came from and what
//! its tensors were named are facts about one checkpoint family, and a crate
//! that knew them could not run a second.
//!
//! The *descriptor* structs - [`WeightDesc`] and its aggregates - hold three
//! numbers per projection and no memory at all. Every planning decision in this
//! module reads only those three, so making them a separate type is what lets
//! the whole ladder be exercised without a device: a test states the element
//! type and the two widths and asks what the pass would do.
//!
//! # Why a mix is one type used three times
//!
//! Each layer mixes the residual streams twice - once for the token mixer and
//! once for the routed feed-forward - and the pass ends with a third mix that
//! collapses the streams for the vocabulary projection. All three are the same
//! sequence of four tensors, and the last differs only in having nothing to
//! inject back into a residual. One type with an `Option` says that; three
//! types would say that they might diverge, which they do not.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use shiso_core::DType;
use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, DeviceView, HostBytes, import_host};
use shiso_core::error::{CoreError, CoreResult};
use shiso_modelfile::quant::{self, block_bytes_for};
use shiso_modelfile::reader::TensorShape;

use crate::shape::LayerKind;

/// Which span of a weight's payload an operand names.
///
/// Only the trellis formats have parts; every other format is one span, and
/// asking for a part of one is a refusal rather than an offset into the middle
/// of its rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightPart {
    /// The whole payload, which is what every non-trellis format is.
    Whole,
    /// The packed tiles.
    Tiles,
    /// The input scale vector, one half-precision word per contraction element.
    InputScales,
    /// The output scale vector, one half-precision word per output row.
    OutputScales,
}

/// One projection weight, resident on the device.
///
/// The buffer is private on purpose: a kernel is never handed a buffer, only a
/// [`DeviceView`] that cannot outlive this, and that is the whole reason the
/// type exists. A weight that has been dropped cannot be read by a launch that
/// was queued before it went.
///
/// # Why this is not the dense family's weight type
///
/// That one refuses, at import, any format for which no *dense* matvec exists,
/// which is right for the family whose dispatch table it is checking against and
/// wrong here: this family's ladder covers formats that one has no kernel for,
/// and the two ladders should not have to agree. And a trellis tensor's payload
/// is its tiles plus two scale vectors, which that type's size check does not
/// know about, so a trellis weight could not be imported through it at all.
pub struct Weight {
    /// Element type the weight was stored as.
    pub dtype: DType,
    /// Elements in one weight row: the projection's contraction width.
    pub in_dim: u32,
    /// Weight rows: the projection's output width.
    pub out_dim: u32,
    /// Weight matrices stacked behind one tensor, which is one for every
    /// projection except the routed expert bank.
    ///
    /// The routed feed-forward stores its experts as a single `[k, n, E]`
    /// tensor - all `E` tile blocks, then all `E` input scale vectors, then all
    /// `E` output ones - and the kernel is handed the base of each of those
    /// three regions and strides through them itself. So the count belongs to
    /// the payload's extent, not to the projection's geometry: every dimension
    /// a launch computes with is still one expert's.
    pub slots: u32,
    buffer: DeviceBuffer,
}

impl Weight {
    /// Bytes ONE matrix of this shape and format occupies in a container.
    ///
    /// Per matrix rather than per tensor, because that is what both callers
    /// want: a stacked bank multiplies it by its slot count, and the routed
    /// down projection passes it to a kernel that strides by it.
    ///
    /// Read from the container reader's own accounting rather than recomputed,
    /// which is what makes the tail the trellis formats append - an input scale
    /// vector and an output one, inside the tensor's own span - the same number
    /// on both sides of the seam. Two independent formulas here would differ
    /// exactly on the format where the difference is a silent over-read.
    #[must_use]
    pub fn payload_bytes(dtype: DType, in_dim: u32, out_dim: u32) -> Option<u64> {
        let shape = TensorShape::new(&[u64::from(in_dim), u64::from(out_dim)]).ok()?;
        block_bytes_for(dtype, &shape)
    }

    /// Wraps `host[range]` as a device weight, without copying where the
    /// backend can.
    ///
    /// The range is checked against the weight's own size rather than trusted,
    /// because the two come from different places - the shape from the binding,
    /// the extent from the container's directory - and a payload shorter than
    /// the declared shape should fail here rather than read off the end of a
    /// mapping.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] when `range` is not exactly the weight's
    /// size, and whatever [`import_host`] reports.
    pub fn import(
        backend: &Arc<dyn Backend>,
        dtype: DType,
        in_dim: u32,
        out_dim: u32,
        slots: u32,
        host: Arc<dyn HostBytes>,
        range: Range<usize>,
    ) -> CoreResult<Self> {
        if slots == 0 {
            return Err(CoreError::ShapeMismatch {
                field: "weight slots",
                expected: 1,
                got: 0,
            });
        }
        let total = Self::payload_bytes(dtype, in_dim, out_dim)
            .and_then(|one| one.checked_mul(u64::from(slots)))
            .ok_or(CoreError::ShapeMismatch {
                field: "weight bytes",
                expected: u64::MAX,
                got: u64::from(in_dim),
            })?;
        let given = u64::try_from(range.end.saturating_sub(range.start)).unwrap_or(u64::MAX);
        if given != total {
            return Err(CoreError::ShapeMismatch {
                field: "weight import bytes",
                expected: total,
                got: given,
            });
        }
        let buffer = import_host(backend, host, range)?;
        Ok(Self {
            dtype,
            in_dim,
            out_dim,
            slots,
            buffer,
        })
    }

    /// The whole weight, as a kernel argument.
    #[must_use]
    pub fn view(&self) -> DeviceView<'_> {
        self.buffer.view()
    }

    /// One span of the payload, for the formats whose payload has parts.
    ///
    /// A trellis tensor is three things in one contiguous range - the tiles, an
    /// input scale vector of the contraction width, an output one of the output
    /// width - and the cooperative kernel takes all three as separate pointers.
    /// Slicing them here rather than at the launch site is what keeps the one
    /// piece of layout knowledge in the file that already owns the size of the
    /// whole.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] when the format has no such part, and
    /// [`CoreError::OutOfBounds`] when the computed span runs past the resident
    /// copy, which means the layout and the import disagree.
    pub fn part(&self, part: WeightPart) -> CoreResult<DeviceView<'_>> {
        let Some(bits) = quant::exl3::k_bits(self.dtype) else {
            return match part {
                WeightPart::Whole => Ok(self.buffer.view()),
                WeightPart::Tiles | WeightPart::InputScales | WeightPart::OutputScales => {
                    Err(CoreError::Unsupported {
                        what: "a trellis payload part of a weight stored in another format",
                        backend: "engine",
                    })
                }
            };
        };
        // Each of the three regions holds every slot's copy back to back, so a
        // stacked bank scales all three and the kernel strides inside them.
        let slots = u64::from(self.slots);
        let elements = u64::from(self.in_dim).saturating_mul(u64::from(self.out_dim));
        let tiles = elements.saturating_mul(u64::from(bits)) / 8 * slots;
        let input = u64::from(self.in_dim)
            .saturating_mul(2)
            .saturating_mul(slots);
        let output = u64::from(self.out_dim)
            .saturating_mul(2)
            .saturating_mul(slots);
        match part {
            WeightPart::Whole => Ok(self.buffer.view()),
            WeightPart::Tiles => self.buffer.slice(0, tiles),
            WeightPart::InputScales => self.buffer.slice(tiles, input),
            WeightPart::OutputScales => self.buffer.slice(tiles.saturating_add(input), output),
        }
    }

    /// Imports a half-precision matrix repacked to [`DType::F16Pack12`], or
    /// `None` when its contraction is not one the packing covers - the caller
    /// then imports the halves as they are.
    ///
    /// `halves` is the tensor's stored bytes, row-major over `out_dim` rows of
    /// `in_dim`; see [`crate::sparse::pack12`] for what the packing is and why
    /// it is lossless.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when the upload fails.
    pub fn packed_half(
        backend: &Arc<dyn Backend>,
        in_dim: u32,
        out_dim: u32,
        halves: &[u8],
    ) -> CoreResult<Option<Self>> {
        let bits = halves
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect::<Vec<_>>();
        let Some(bytes) = crate::sparse::pack12::pack(&bits, in_dim, out_dim) else {
            return Ok(None);
        };
        // Copied rather than imported: an import keeps its host bytes alive
        // beside the device's copy, and on a device that shares its memory
        // with the host these are a gibibyte nobody reads again. The ledger
        // tag is the caller's scope, as an import's is.
        let buffer = shiso_core::buffer::alloc_tagged(
            backend,
            shiso_core::ledger::current(),
            u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        )?;
        backend.copy_h2d(buffer.view(), &bytes)?;
        Ok(Some(Self {
            dtype: DType::F16Pack12,
            in_dim,
            out_dim,
            slots: 1,
            buffer,
        }))
    }

    /// Wraps a buffer this crate produced as a weight of the given shape.
    ///
    /// The import above is for bytes a container holds; this is for bytes a
    /// launch wrote - the one case being a vocabulary head gathered down to the
    /// blocks a draft is likely to want. No length check, because there is no
    /// second party to disagree with: the caller sized the allocation from
    /// [`Weight::payload_bytes`] of the very shape it passes here.
    #[must_use]
    pub const fn over(dtype: DType, in_dim: u32, out_dim: u32, buffer: DeviceBuffer) -> Self {
        Self {
            dtype,
            in_dim,
            out_dim,
            slots: 1,
            buffer,
        }
    }

    /// Bytes the resident copy occupies.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.buffer.len()
    }

    /// Whether the weight is empty, which no bound tensor ever is.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

impl fmt::Debug for Weight {
    /// Names the weight by its shape and format, and its allocation by size.
    ///
    /// Not derived, because a device allocation has no host-readable content:
    /// the derived form would print an opaque identifier that tells a reader
    /// nothing and hides the three numbers that actually distinguish one weight
    /// from another.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Weight")
            .field("dtype", &self.dtype)
            .field("in_dim", &self.in_dim)
            .field("out_dim", &self.out_dim)
            .field("slots", &self.slots)
            .field("bytes", &self.buffer.len())
            .finish()
    }
}

/// The three numbers a projection's tier is chosen from.
///
/// Deliberately NOT `#[non_exhaustive]`: a test and a model crate both build
/// these, and a new field would be a number neither of them is supplying yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightDesc {
    /// Element type the weight is stored as.
    pub dtype: DType,
    /// Elements in one weight row: the projection's contraction width.
    pub in_dim: u32,
    /// Weight rows: the projection's output width.
    pub out_dim: u32,
}

impl WeightDesc {
    /// The three numbers, given directly.
    #[must_use]
    pub const fn new(dtype: DType, in_dim: u32, out_dim: u32) -> Self {
        Self {
            dtype,
            in_dim,
            out_dim,
        }
    }

    /// Whether two weights are the same shape and format, which is what a
    /// grouped launch over a pair requires.
    #[must_use]
    pub const fn matches(&self, other: &Self) -> bool {
        self.in_dim == other.in_dim
            && self.out_dim == other.out_dim
            && matches!(
                (self.dtype, other.dtype),
                (DType::Exl3K4, DType::Exl3K4)
                    | (DType::Exl3K5, DType::Exl3K5)
                    | (DType::Exl3K6, DType::Exl3K6)
                    | (DType::F16, DType::F16)
                    | (DType::BF16, DType::BF16)
                    | (DType::F32, DType::F32)
            )
    }
}

impl From<&Weight> for WeightDesc {
    fn from(weight: &Weight) -> Self {
        Self {
            dtype: weight.dtype,
            in_dim: weight.in_dim,
            out_dim: weight.out_dim,
        }
    }
}

/// The low-rank pair that mixes the residual streams, and the norm in front of
/// it.
#[derive(Debug)]
pub struct StreamMix {
    /// Per-stream norm weight the mix reads, when the previous block's combine
    /// did not already leave the normed residual behind.
    pub norm: DeviceBuffer,
    /// The contraction down to the low-rank vector.
    pub down: Weight,
    /// The expansion back to the full stream width.
    pub up: Weight,
    /// The per-stream logits the matching combine injects, absent on the mix
    /// that collapses the streams at the end of a pass.
    pub inject: Option<Weight>,
}

/// The three widths a mix's plan is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MixDesc {
    /// The contraction down to the low-rank vector.
    pub down: WeightDesc,
    /// The expansion back to the full stream width.
    pub up: WeightDesc,
    /// The injection logits, absent on the collapsing mix.
    pub inject: Option<WeightDesc>,
}

impl StreamMix {
    /// The widths this mix plans from.
    #[must_use]
    pub fn desc(&self) -> MixDesc {
        MixDesc {
            down: WeightDesc::from(&self.down),
            up: WeightDesc::from(&self.up),
            inject: self.inject.as_ref().map(WeightDesc::from),
        }
    }
}

/// The routed feed-forward's weights: the router, the expert stack, and the
/// shared expert every token pays for.
#[derive(Debug)]
pub struct ExpertBank {
    /// The router's own projection, one logit per expert.
    pub router: Weight,
    /// Every routed expert's gate projection, stacked.
    pub gate: Weight,
    /// Every routed expert's up projection, stacked.
    pub up: Weight,
    /// Every routed expert's down projection, stacked.
    pub down: Weight,
    /// The shared expert's scalar gate.
    pub shared_router: Weight,
    /// The shared expert's gate projection.
    pub shared_gate: Weight,
    /// The shared expert's up projection.
    pub shared_up: Weight,
    /// The shared expert's down projection.
    pub shared_down: Weight,
}

/// The widths a routed feed-forward's plan is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertDesc {
    /// The router's projection.
    pub router: WeightDesc,
    /// One routed expert's gate.
    pub gate: WeightDesc,
    /// One routed expert's up.
    pub up: WeightDesc,
    /// One routed expert's down.
    pub down: WeightDesc,
    /// The shared expert's scalar gate.
    pub shared_router: WeightDesc,
    /// The shared expert's gate.
    pub shared_gate: WeightDesc,
    /// The shared expert's up.
    pub shared_up: WeightDesc,
    /// The shared expert's down.
    pub shared_down: WeightDesc,
}

impl ExpertBank {
    /// The widths this bank plans from.
    #[must_use]
    pub fn desc(&self) -> ExpertDesc {
        ExpertDesc {
            router: WeightDesc::from(&self.router),
            gate: WeightDesc::from(&self.gate),
            up: WeightDesc::from(&self.up),
            down: WeightDesc::from(&self.down),
            shared_router: WeightDesc::from(&self.shared_router),
            shared_gate: WeightDesc::from(&self.shared_gate),
            shared_up: WeightDesc::from(&self.shared_up),
            shared_down: WeightDesc::from(&self.shared_down),
        }
    }
}

/// The narrow tower that chooses which key blocks an attention block reads.
#[derive(Debug)]
pub struct Indexer {
    /// The indexer's query projection.
    pub q: Weight,
    /// The indexer's key projection.
    pub k: Weight,
    /// Norm over one indexer query head.
    pub q_norm: DeviceBuffer,
    /// Norm over one indexer key head.
    pub k_norm: DeviceBuffer,
}

/// The widths an indexer's plan is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexerDesc {
    /// The indexer's query projection.
    pub q: WeightDesc,
    /// The indexer's key projection.
    pub k: WeightDesc,
}

impl Indexer {
    /// The widths this indexer plans from.
    #[must_use]
    pub fn desc(&self) -> IndexerDesc {
        IndexerDesc {
            q: WeightDesc::from(&self.q),
            k: WeightDesc::from(&self.k),
        }
    }
}

/// An attention slot's weights.
#[derive(Debug)]
pub struct SparseAttn {
    /// The query projection, which carries the per-head output gate in its own
    /// rows: the first launch of the block splits the two apart.
    pub q: Weight,
    /// The key projection.
    pub k: Weight,
    /// The value projection.
    pub v: Weight,
    /// The projection back to the residual width.
    pub output: Weight,
    /// Norm over one query head.
    pub q_norm: DeviceBuffer,
    /// Norm over one key head.
    pub k_norm: DeviceBuffer,
    /// The block's indexer.
    pub indexer: Indexer,
}

/// The widths an attention slot's plan is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnDesc {
    /// The query projection, gate rows included.
    pub q: WeightDesc,
    /// The key projection.
    pub k: WeightDesc,
    /// The value projection.
    pub v: WeightDesc,
    /// The projection back to the residual width.
    pub output: WeightDesc,
    /// The block's indexer.
    pub indexer: IndexerDesc,
}

impl SparseAttn {
    /// The widths this slot plans from.
    #[must_use]
    pub fn desc(&self) -> AttnDesc {
        AttnDesc {
            q: WeightDesc::from(&self.q),
            k: WeightDesc::from(&self.k),
            v: WeightDesc::from(&self.v),
            output: WeightDesc::from(&self.output),
            indexer: self.indexer.desc(),
        }
    }
}

/// A recurrent slot's weights.
#[derive(Debug)]
pub struct Recurrent {
    /// The fused query, key and value projection.
    pub qkv: Weight,
    /// The output gate's projection.
    pub gate: Weight,
    /// The decay projection.
    pub alpha: Weight,
    /// The write-strength projection.
    pub beta: Weight,
    /// The projection back to the residual width.
    pub out: Weight,
    /// The causal convolution's kernel.
    pub conv1d: DeviceBuffer,
    /// Per-head decay base.
    pub a: DeviceBuffer,
    /// Per-head decay bias.
    pub dt_bias: DeviceBuffer,
    /// Norm over one value head, applied by the output gate.
    pub norm: DeviceBuffer,
}

/// The widths a recurrent slot's plan is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecurrentDesc {
    /// The fused query, key and value projection.
    pub qkv: WeightDesc,
    /// The output gate's projection.
    pub gate: WeightDesc,
    /// The decay projection.
    pub alpha: WeightDesc,
    /// The write-strength projection.
    pub beta: WeightDesc,
    /// The projection back to the residual width.
    pub out: WeightDesc,
}

impl Recurrent {
    /// The widths this slot plans from.
    #[must_use]
    pub fn desc(&self) -> RecurrentDesc {
        RecurrentDesc {
            qkv: WeightDesc::from(&self.qkv),
            gate: WeightDesc::from(&self.gate),
            alpha: WeightDesc::from(&self.alpha),
            beta: WeightDesc::from(&self.beta),
            out: WeightDesc::from(&self.out),
        }
    }
}

/// The n-gram block's weights.
#[derive(Debug)]
pub struct NgramBlock {
    /// The projection that turns decoded rows into per-stream keys.
    pub key: Weight,
    /// The projection that turns decoded rows into values.
    pub value: Weight,
    /// Norm over the keys.
    pub norm_key: DeviceBuffer,
    /// Norm over the residual the keys are gated against.
    pub norm_query: DeviceBuffer,
    /// Norm over the gated value the convolution reads.
    pub norm_conv: DeviceBuffer,
    /// The dilated convolution's kernel.
    pub conv1d: DeviceBuffer,
}

/// The widths an n-gram block's plan is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramDesc {
    /// The key projection.
    pub key: WeightDesc,
    /// The value projection.
    pub value: WeightDesc,
    /// Element type of one gathered row, which chooses the decode kernel.
    pub row: DType,
}

impl NgramBlock {
    /// The widths this block plans from, given the format its gathered rows
    /// arrive in.
    ///
    /// The row format is an argument because it is a property of the *table*,
    /// which is streamed from disk and is not one of the tensors held here.
    #[must_use]
    pub fn desc(&self, row: DType) -> NgramDesc {
        NgramDesc {
            key: WeightDesc::from(&self.key),
            value: WeightDesc::from(&self.value),
            row,
        }
    }
}

/// Which block a slot holds, together with that block's weights.
///
/// An enum rather than two options: a slot holds exactly one block, and a pair
/// of options would let it hold both or neither.
#[derive(Debug)]
pub enum SparseBlock {
    /// A recurrent slot, whose ordinal indexes the session's recurrent stack.
    Recurrent(Recurrent),
    /// An attention slot, whose ordinal indexes the cache's per-layer planes.
    Attn(SparseAttn),
}

impl SparseBlock {
    /// Which kind of slot this is.
    #[must_use]
    pub const fn kind(&self) -> LayerKind {
        match self {
            Self::Recurrent(_) => LayerKind::Gdn,
            Self::Attn(_) => LayerKind::Attn,
        }
    }
}

/// Which block a slot holds, as widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockDesc {
    /// A recurrent slot.
    Recurrent(RecurrentDesc),
    /// An attention slot.
    Attn(AttnDesc),
}

impl BlockDesc {
    /// Which kind of slot this describes.
    #[must_use]
    pub const fn kind(&self) -> LayerKind {
        match self {
            Self::Recurrent(_) => LayerKind::Gdn,
            Self::Attn(_) => LayerKind::Attn,
        }
    }
}

/// One layer slot: its two stream mixes, its token mixer, its routed
/// feed-forward, and the n-gram block on the one slot that has one.
#[derive(Debug)]
pub struct SparseLayer {
    /// The mix that feeds the token mixer.
    pub attn_mix: StreamMix,
    /// The mix that feeds the routed feed-forward.
    pub ffn_mix: StreamMix,
    /// The token mixer.
    pub block: SparseBlock,
    /// The routed feed-forward.
    pub experts: ExpertBank,
    /// The n-gram block, on the one slot that folds one in.
    pub ngram: Option<NgramBlock>,
    /// Rank of this slot among the slots of its own kind, counted from zero.
    ///
    /// A recurrent slot's rank indexes the session's recurrent stack; an
    /// attention slot's rank indexes the cache pool's per-layer planes. Both
    /// were allocated in slot order, so the rank is the index in either case.
    pub ordinal: u32,
}

/// One layer slot, as widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerDesc {
    /// The mix that feeds the token mixer.
    pub attn_mix: MixDesc,
    /// The mix that feeds the routed feed-forward.
    pub ffn_mix: MixDesc,
    /// The token mixer.
    pub block: BlockDesc,
    /// The routed feed-forward.
    pub experts: ExpertDesc,
    /// The n-gram block, on the one slot that folds one in.
    pub ngram: Option<NgramDesc>,
    /// Rank of this slot among the slots of its own kind, counted from zero.
    ///
    /// Carried on the descriptor as well as on the slot because it is what a
    /// launch binds: a recurrent slot's rank names the session state it writes,
    /// and an attention slot's names the cache planes it reads.
    pub ordinal: u32,
}

impl SparseLayer {
    /// The widths this slot plans from.
    ///
    /// `ngram_row` is `None` on a stack with no streamed table, and a slot that
    /// holds n-gram weights without one plans no n-gram block: the decode
    /// kernel is chosen by the row format, so there is nothing to plan from.
    /// The stack refuses that pairing before it ever gets here.
    #[must_use]
    pub fn desc(&self, ngram_row: Option<DType>) -> LayerDesc {
        LayerDesc {
            attn_mix: self.attn_mix.desc(),
            ffn_mix: self.ffn_mix.desc(),
            block: match &self.block {
                SparseBlock::Recurrent(r) => BlockDesc::Recurrent(r.desc()),
                SparseBlock::Attn(a) => BlockDesc::Attn(a.desc()),
            },
            experts: self.experts.desc(),
            ngram: self
                .ngram
                .as_ref()
                .zip(ngram_row)
                .map(|(block, row)| block.desc(row)),
            ordinal: self.ordinal,
        }
    }
}

/// The draft head's own weights.
///
/// Not a [`SparseWeights`], because it is not a model: it holds no vocabulary
/// table and no output projection, and reads the target's for both. What it
/// does hold is one layer of the same architecture at the same widths, the two
/// normalizations and two projections that turn a token and a target hidden
/// state into that layer's input, and the mix that collapses its streams for
/// the shared vocabulary head.
#[derive(Debug)]
pub struct DraftHead {
    /// Norm over the token embedding, at the residual width.
    pub norm_embed: DeviceBuffer,
    /// Norm over the target hidden state, at the multi-stream width.
    pub norm_hidden: DeviceBuffer,
    /// The projection of the normed embedding.
    pub fc_embed: Weight,
    /// The projection of the normed hidden state, applied per stream.
    pub fc_hidden: Weight,
    /// The head's single layer.
    pub layer: SparseLayer,
    /// The mix that collapses the streams for the vocabulary projection.
    pub output_mix: StreamMix,
    /// The vocabulary head narrowed to the rows a draft is likely to want.
    ///
    /// Absent is the whole head, which is correct and costs a draft most of
    /// what a target step costs. See [`crate::sparse::headslice`].
    pub head: Option<crate::sparse::headslice::HeadSlice>,
}

/// Every resident weight, in the order a forward pass reads them.
#[derive(Debug)]
pub struct SparseWeights {
    /// The vocabulary table the pass gathers rows out of.
    pub token_embd: Weight,
    /// The output projection when the file stores one of its own.
    pub lm_head: Option<Weight>,
    /// The mix that collapses the streams for the vocabulary projection.
    ///
    /// It stands in for an output norm: there is no separate one, because the
    /// mix already normalizes what it reads.
    pub output_mix: StreamMix,
    /// One entry per layer slot, index zero first.
    pub layers: Vec<SparseLayer>,
    /// The draft head, on a build that speculates.
    pub draft: Option<DraftHead>,
}

impl SparseWeights {
    /// The matrix the logits come out of: the file's own projection when it has
    /// one, and the embedding table otherwise.
    #[must_use]
    pub fn output_matrix(&self) -> &Weight {
        self.lm_head.as_ref().unwrap_or(&self.token_embd)
    }
}

/// Every width a pass plans from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDesc {
    /// The vocabulary table.
    pub token_embd: WeightDesc,
    /// The output projection.
    pub output: WeightDesc,
    /// The collapsing mix.
    pub output_mix: MixDesc,
    /// One entry per layer slot, index zero first.
    pub layers: Vec<LayerDesc>,
    /// The draft head's widths, on a build that speculates.
    pub draft: Option<crate::sparse::draft::DraftDesc>,
}

impl SparseWeights {
    /// The widths this checkpoint plans from.
    ///
    /// `ngram_row` is the element type one gathered n-gram row arrives in,
    /// which is a property of the streamed table rather than of any tensor held
    /// here, and so has to be supplied - and is absent exactly when there is no
    /// such table.
    #[must_use]
    pub fn desc(&self, ngram_row: Option<DType>) -> ModelDesc {
        ModelDesc {
            token_embd: WeightDesc::from(&self.token_embd),
            output: WeightDesc::from(self.output_matrix()),
            output_mix: self.output_mix.desc(),
            layers: self
                .layers
                .iter()
                .map(|layer| layer.desc(ngram_row))
                .collect(),
            // The head folds in no n-gram block, so its layer's widths do not
            // depend on the streamed table at all.
            draft: self
                .draft
                .as_ref()
                .map(|head| crate::sparse::draft::DraftDesc {
                    fc_embed: WeightDesc::from(&head.fc_embed),
                    fc_hidden: WeightDesc::from(&head.fc_hidden),
                    layer: head.layer.desc(None),
                    output_mix: head.output_mix.desc(),
                    vocabulary: head
                        .head
                        .as_ref()
                        .map(|slice| WeightDesc::from(slice.weight())),
                }),
        }
    }
}
