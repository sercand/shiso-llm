//! What each launch is handed: its arguments, named rather than addressed.
//!
//! A planned step says which kernel runs and on what grid. That is half of a
//! launch. The other half is the argument list, and it is the half a port gets
//! wrong in the way that is hardest to see: an activation and its destination
//! swapped, a stride passed where an offset belongs, a norm weight bound one
//! role over. None of those fail. They produce a number.
//!
//! # Why an operand is a name and not a pointer
//!
//! A plan is built once and read many times, by a test on a machine with no
//! device as much as by the executor on one that has. A pointer could only be
//! produced on the second. So an argument is an [`Operand`] - "the residual",
//! "this layer's key projection", "the session's page table", "the row count" -
//! and turning one into a device range is a separate step that happens where
//! the buffers are.
//!
//! That separation is what makes argument order checkable without hardware. A
//! test reads the operand list a step carries and compares it against the
//! kernel's own parameter list; the comparison is exact, it names the position
//! that is wrong, and it needs nothing to run.
//!
//! # Ordinals are the kernels' own
//!
//! Arguments are in kernel declaration order, which is what
//! [`shiso_core::backend::KernelArg`] means by ordinal. A pointer parameter a
//! launcher passes as null - the checkpoint slab of the convolution and the
//! recurrence, the next block's norm weight where the chain breaks - is
//! [`Operand::Null`], which is eight zero bytes in a pointer-sized slot rather
//! than a zero-length buffer, because a zero-length buffer is still an address.

use shiso_core::backend::{CoopLaunch, KernelArg};
use shiso_core::buffer::DeviceView;
use shiso_core::error::{CoreError, CoreResult};

use crate::sparse::paged::{PagedKv, Plane};
use crate::sparse::rows::{Layout, PtrArray};
use crate::sparse::scratch::{Buf, Scratch};
use crate::sparse::stack::SparseSession;
use crate::sparse::weights::{SparseWeights, WeightPart};

/// Arguments one launch may carry.
///
/// The widest kernel in this set declares eighteen parameters; the bound is
/// above that with room, and an argument list that reaches it is refused rather
/// than truncated.
pub const MAX_ARGS: usize = 24;

/// Rows one binder may hold a session for.
///
/// One row per run of the widest segmented pass ([`MAX_PROMPT_RUNS`]), which
/// is wider than a decode batch: a prompt-role pass binds each run's own state
/// by run. This bounds the binder only. The per-row pointer arrays are packed
/// for a decode batch's or a verify's rows alone, at most
/// [`MAX_DECODE_BATCH`] of them, into a slab the scratch sizes at the logits
/// block's height ([`LOGIT_BLOCK_ROWS`]).
///
/// [`MAX_PROMPT_RUNS`]: crate::sparse::stack::MAX_PROMPT_RUNS
/// [`MAX_DECODE_BATCH`]: crate::sparse::stack::MAX_DECODE_BATCH
/// [`LOGIT_BLOCK_ROWS`]: crate::sparse::stack::LOGIT_BLOCK_ROWS
pub const MAX_ROWS: usize = crate::sparse::stack::MAX_PROMPT_RUNS as usize;

/// Which bound tensor an operand names.
///
/// Roles rather than indices: a plan holds the role and the layer, and the one
/// place that turns the pair into a device range is [`Binder::weight`]. A
/// tensor bound one role over is then a single wrong name in a plan a test can
/// read, instead of an offset nothing can check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// The vocabulary table.
    TokenEmbd,
    /// The projection onto the vocabulary.
    Output,
    /// The collapsing mix's per-stream norm.
    OutputMixNorm,
    /// The collapsing mix's contraction.
    OutputMixDown,
    /// The collapsing mix's expansion.
    OutputMixUp,
    /// The token mixer's stream-mix norm.
    AttnMixNorm,
    /// The token mixer's stream-mix contraction.
    AttnMixDown,
    /// The token mixer's stream-mix expansion.
    AttnMixUp,
    /// The token mixer's injection logits.
    AttnMixInject,
    /// The routed feed-forward's stream-mix norm.
    FfnMixNorm,
    /// The routed feed-forward's stream-mix contraction.
    FfnMixDown,
    /// The routed feed-forward's stream-mix expansion.
    FfnMixUp,
    /// The routed feed-forward's injection logits.
    FfnMixInject,
    /// The query projection, output-gate rows included.
    AttnQuery,
    /// The key projection.
    AttnKey,
    /// The value projection.
    AttnValue,
    /// The projection back to the residual width.
    AttnOutput,
    /// Norm over one query head.
    AttnQueryNorm,
    /// Norm over one key head.
    AttnKeyNorm,
    /// The indexer's query projection.
    IndexerQuery,
    /// The indexer's key projection.
    IndexerKey,
    /// Norm over one indexer query head.
    IndexerQueryNorm,
    /// Norm over one indexer key head.
    IndexerKeyNorm,
    /// The recurrent block's fused query, key and value projection.
    RecurrentQkv,
    /// The recurrent block's output-gate projection.
    RecurrentGate,
    /// The recurrent block's decay projection.
    RecurrentAlpha,
    /// The recurrent block's write-strength projection.
    RecurrentBeta,
    /// The recurrent block's projection back to the residual width.
    RecurrentOut,
    /// The recurrent block's convolution kernel.
    RecurrentConv1d,
    /// The recurrent block's per-head decay base.
    RecurrentDecayBase,
    /// The recurrent block's per-head decay bias.
    RecurrentDecayBias,
    /// Norm over one recurrent value head.
    RecurrentNorm,
    /// The expert router's projection.
    Router,
    /// The routed experts' stacked gate projection.
    ExpertGate,
    /// The routed experts' stacked up projection.
    ExpertUp,
    /// The routed experts' stacked down projection.
    ExpertDown,
    /// The shared expert's scalar gate.
    SharedRouter,
    /// The shared expert's gate projection.
    SharedGate,
    /// The shared expert's up projection.
    SharedUp,
    /// The shared expert's down projection.
    SharedDown,
    /// The n-gram block's key projection.
    NgramKey,
    /// The n-gram block's value projection.
    NgramValue,
    /// Norm over the n-gram keys.
    NgramNormKey,
    /// Norm over the residual the n-gram keys are gated against.
    NgramNormQuery,
    /// Norm over the gated value the n-gram convolution reads.
    NgramNormConv,
    /// The n-gram block's dilated convolution kernel.
    NgramConv1d,
    /// The draft head's norm over the token embedding it starts from.
    DraftNormEmbed,
    /// The draft head's norm over the target hidden state it starts from.
    ///
    /// One RMS over the whole multi-stream vector, not one per stream: the
    /// weight is that wide and the normalization is that wide with it.
    DraftNormHidden,
    /// The draft head's projection of the normed embedding.
    DraftFcEmbed,
    /// The draft head's projection of the normed hidden state, applied to every
    /// stream with the same matrix.
    DraftFcHidden,
    /// The vocabulary head narrowed to the rows a draft is likely to want.
    DraftVocabulary,
}

impl Role {
    /// Whether this role names a tensor the model holds once rather than once
    /// per layer.
    #[must_use]
    pub const fn is_model_wide(self) -> bool {
        matches!(
            self,
            Self::TokenEmbd
                | Self::Output
                | Self::OutputMixNorm
                | Self::OutputMixDown
                | Self::OutputMixUp
                | Self::DraftNormEmbed
                | Self::DraftNormHidden
                | Self::DraftFcEmbed
                | Self::DraftFcHidden
                | Self::DraftVocabulary
        )
    }
}

/// Which of the two weight sets a reference names.
///
/// A model that speculates carries two: the target's layers, and a draft head
/// whose single layer is the same architecture at the same widths. Naming the
/// set is therefore not a convenience - layer zero exists in both, holds
/// tensors of identical shape and dtype in both, and a reference that resolved
/// against the wrong one would bind a weight of exactly the right size. Every
/// launch would succeed and the draft would predict from the target's first
/// layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WeightSet {
    /// The model that is being decoded.
    #[default]
    Target,
    /// The draft head that proposes for it.
    Draft,
}

/// One layer of one weight set.
///
/// A newtype over the index rather than the index itself, for the reason
/// [`WeightSet`] gives: the two sets' layer zeros are indistinguishable by
/// shape, so the set has to travel with the index everywhere the index does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayerRef {
    /// Which weight set.
    pub set: WeightSet,
    /// The layer's index within that set.
    pub index: u32,
}

impl LayerRef {
    /// A layer of the model being decoded.
    #[must_use]
    pub const fn target(index: u32) -> Self {
        Self {
            set: WeightSet::Target,
            index,
        }
    }

    /// The draft head's only layer.
    #[must_use]
    pub const fn draft() -> Self {
        Self {
            set: WeightSet::Draft,
            index: 0,
        }
    }
}

/// One bound tensor: which layer holds it, what it is to that layer, and which
/// span of its payload.
///
/// Deliberately NOT `#[non_exhaustive]`: every block family builds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WeightRef {
    /// The layer, whose index is ignored for a role its set holds once.
    pub layer: LayerRef,
    /// What the tensor is to that layer.
    pub role: Role,
    /// Which span of the payload, for the formats whose payload has parts.
    pub part: WeightPart,
}

impl WeightRef {
    /// One of a layer's own tensors, whole.
    #[must_use]
    pub const fn layer(layer: LayerRef, role: Role) -> Self {
        Self {
            layer,
            role,
            part: WeightPart::Whole,
        }
    }

    /// One of the tensors the model being decoded holds once, whole.
    #[must_use]
    pub const fn model(role: Role) -> Self {
        Self {
            layer: LayerRef::target(0),
            role,
            part: WeightPart::Whole,
        }
    }

    /// One of the tensors the draft head holds once, whole.
    #[must_use]
    pub const fn draft_model(role: Role) -> Self {
        Self {
            layer: LayerRef::draft(),
            role,
            part: WeightPart::Whole,
        }
    }

    /// The same tensor, narrowed to one span of its payload.
    #[must_use]
    pub const fn spanning(mut self, part: WeightPart) -> Self {
        self.part = part;
        self
    }
}

/// One kernel parameter, named by where it comes from.
///
/// `F32` carries a bit pattern rather than a float so that two plans compare
/// equal exactly when they would launch the same bytes; [`Operand::f32`] is the
/// constructor a call site uses and hides it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Operand {
    /// An activation buffer, whole.
    Scratch(Buf),
    /// An activation buffer from a byte offset on.
    ///
    /// The offset is the *launcher's*, not a slice of convenience: several
    /// kernels are launched twice over the two halves of one projection, and
    /// the only thing that tells the two launches apart is where their operand
    /// starts.
    ScratchFrom(Buf, u64),
    /// A bound tensor.
    Weight(WeightRef),
    /// The session's page table.
    ///
    /// Names *the* session, so a binder holding a batch refuses it rather than
    /// answering with the first row's - which is the whole of the bug the
    /// batched kernels exist to close, and the one shape of it that a planner
    /// could reintroduce by forgetting to widen a launch.
    Pages,
    /// One row's page table, by row.
    ///
    /// What a scalar cache-addressing launch binds inside a batched pass. Row
    /// zero of a one-row pass is that pass's only session, so this is not a
    /// batch-only operand: it is the addressed form of the one above.
    PagesOf(u32),
    /// The device array of one page table per row.
    ///
    /// The `const int32_t *const *pages_arr` the batched key/value store, the
    /// batched attention and the batched indexer store take. Filled once per
    /// step by [`crate::sparse::rows::pack`].
    PageTables,
    /// The device array of one recurrent slot's convolution window per row.
    RecurrentConvs(u32),
    /// The device array of one recurrent slot's state matrix per row.
    RecurrentStates(u32),
    /// The device array of one recurrent slot's convolution rollback records
    /// per row, which a segmented verify's single convolution launch writes.
    RecurrentConvCkpts(u32),
    /// The device array of one recurrent slot's state rollback records per
    /// row, which a segmented verify's single recurrence launch writes.
    RecurrentStateCkpts(u32),
    /// The device array of one n-gram convolution window per row.
    NgramWindows,
    /// One attention slot's whole allocation for one cache plane.
    ///
    /// `rank` is the slot's rank among attention slots, which is the order the
    /// pool allocated in - not the layer index, because three slots in four
    /// hold no attention block and own no cache at all.
    CachePlane {
        /// The slot's rank among attention slots.
        rank: u32,
        /// Which plane.
        plane: Plane,
    },
    /// One recurrent slot's convolution window.
    RecurrentConv(u32),
    /// One recurrent slot's state matrix.
    RecurrentState(u32),
    /// One recurrent slot's per-position convolution windows, which a
    /// speculative rollback copies back from.
    ///
    /// A slab of one record per drafted row, in row order. Present only on a
    /// session opened for speculation; binding it on any other names a buffer
    /// that does not exist, which is a refusal rather than a launch writing
    /// past something else.
    RecurrentConvCkpt(u32),
    /// One recurrent slot's per-position state matrices, laid out the same way.
    RecurrentStateCkpt(u32),
    /// One run's convolution window for one recurrent slot: what a prompt-role
    /// segmented pass's per-run convolution binds, in the way
    /// [`Operand::PagesOf`] names one row's page table.
    RecurrentConvOf {
        /// The slot's rank among recurrent slots.
        slot: u32,
        /// The run, which is the binder's row.
        run: u32,
    },
    /// One run's state matrix for one recurrent slot.
    RecurrentStateOf {
        /// The slot's rank among recurrent slots.
        slot: u32,
        /// The run, which is the binder's row.
        run: u32,
    },
    /// One segment's n-gram convolution window, by segment.
    ///
    /// What a scalar launch binds inside a segmented pass - several sessions'
    /// runs of consecutive rows, each launched for its own session - in the
    /// way [`Operand::PagesOf`] names one row's page table.
    NgramWindowOf(u32),
    /// One segment's per-position n-gram windows.
    NgramWindowCkptOf(u32),
    /// The n-gram block's dilated convolution window.
    NgramWindow,
    /// The stack's snapshot buffer from this byte offset: a checkpoint blob's
    /// layout on the device, which a prompt pass that ends between two grid
    /// points writes the session's state into as it stood at the last one.
    Snapshot(u64),
    /// The n-gram block's per-position windows, which a speculative rollback
    /// copies back from.
    NgramWindowCkpt,
    /// The residual row a previous pass staged for the draft head.
    ///
    /// A session's buffer rather than scratch, because it has to survive the
    /// pass that reads it: the head's first guess starts from the row the
    /// *target* last produced, and by the time the head runs, the scratch that
    /// held it is the head's own working space.
    DraftRow,
    /// The quantized n-gram rows this pass gathered, staged on the device.
    NgramRows,
    /// The splice's argument block, by value.
    ///
    /// The one by-value struct parameter in this family's whole kernel set, and
    /// it is a variant of its own because the packer gives the driver **one
    /// pointer per declared parameter**: `kernel_vit_splice_rows` declares its
    /// block as a single `.param .align 4 .b8 [12]`, so three `u32` operands
    /// standing in for its three fields would hand the driver four pointers
    /// where it wants four but with the first three of them four bytes wide
    /// and the fourth a pointer read from the wrong slot. Every launch would
    /// still succeed.
    ///
    /// The value itself is built by the issuer from the pass it is issuing, so
    /// the block and the plan cannot disagree about a width.
    SpliceArgs,
    /// The whole encoder block of a multimodal request, which the splice reads
    /// the placeholder rows out of.
    ///
    /// A per-pass binding rather than a scratch buffer, for the reason
    /// [`Operand::NgramRows`] is one: the block is a request's, not a chunk's -
    /// a prompt split into several passes splices out of the same block - and
    /// nothing in this stack allocated it. A pass with no media cannot name it,
    /// and says so rather than binding whatever was last at that address.
    MediaRows,
    /// The per-head bias the trellis row codec finishes each weight with.
    ///
    /// A tensor of the streamed table rather than of the layer, which is why it
    /// is an operand of its own: the six tensors the n-gram block holds are
    /// weights the checkpoint made resident, and this one travels with the
    /// table the rows were read from. The block-quantized codec has no bias and
    /// never names it.
    NgramBias,
    /// The lock chain at the head of the per-stream cooperative scratch.
    ///
    /// Every cooperative launch on a stream shares it and leaves it zeroed, so
    /// it sits at a fixed offset of zero and the staging behind it is what
    /// grows.
    CoopLocks,
    /// A byte offset into the staging region of the per-stream cooperative
    /// scratch, which begins where the lock chain ends.
    ///
    /// The offset is the launch's own carve-up of that region - a transformed
    /// activation slab, an expanded weight, one of the routed block's three
    /// staging buffers - so two operands of one launch naming the same offset
    /// is a plan a reader can see is wrong.
    CoopStaging(u64),
    /// The expert-to-slot map the affine expert stages of one layer read, by
    /// layer index: the identity while the banks are resident, and the stream's
    /// own map for that layer while they are streamed.
    ExpertMap(u32),
    /// The identity expert-to-slot map of a resident bank, whatever the
    /// target's banks are: a draft head's experts are never streamed, and
    /// reading a streamed target layer's map against them would send every
    /// column to the slot some other layer's expert sits at.
    ResidentExpertMap,
    /// A pointer parameter the launcher passes as null.
    Null,
    /// An unsigned 32-bit scalar.
    U32(u32),
    /// A signed 32-bit scalar.
    I32(i32),
    /// An unsigned 64-bit scalar.
    U64(u64),
    /// A single-precision scalar, as its bit pattern.
    F32(u32),
}

impl Operand {
    /// A single-precision scalar.
    #[must_use]
    pub const fn f32(value: f32) -> Self {
        Self::F32(value.to_bits())
    }

    /// The same operand, advanced by `bytes`.
    ///
    /// What a pass of a multi-pass projection reads and writes: the launch is
    /// the same, the rows it covers are the next ones along. A scalar has no
    /// address to advance, so advancing one is the scalar unchanged.
    #[must_use]
    pub const fn advanced(self, bytes: u64) -> Self {
        match self {
            Self::Scratch(buf) => Self::ScratchFrom(buf, bytes),
            Self::ScratchFrom(buf, at) => Self::ScratchFrom(buf, at.saturating_add(bytes)),
            other => other,
        }
    }

    /// Whether this operand names device memory rather than a scalar.
    #[must_use]
    pub const fn is_pointer(self) -> bool {
        matches!(
            self,
            Self::Scratch(_)
                | Self::ScratchFrom(..)
                | Self::Weight(_)
                | Self::Pages
                | Self::PagesOf(_)
                | Self::PageTables
                | Self::RecurrentConvs(_)
                | Self::RecurrentStates(_)
                | Self::RecurrentConvCkpts(_)
                | Self::RecurrentStateCkpts(_)
                | Self::NgramWindows
                | Self::CachePlane { .. }
                | Self::RecurrentConv(_)
                | Self::RecurrentState(_)
                | Self::RecurrentConvCkpt(_)
                | Self::RecurrentStateCkpt(_)
                | Self::RecurrentConvOf { .. }
                | Self::RecurrentStateOf { .. }
                | Self::NgramWindowOf(_)
                | Self::NgramWindowCkptOf(_)
                | Self::NgramWindow
                | Self::Snapshot(_)
                | Self::NgramWindowCkpt
                | Self::NgramRows
                | Self::NgramBias
                | Self::MediaRows
                | Self::SpliceArgs
                | Self::CoopLocks
                | Self::CoopStaging(_)
                | Self::ExpertMap(_)
                | Self::ResidentExpertMap
                | Self::Null
        )
    }
}

/// One launch's argument list, in kernel declaration order.
///
/// A fixed array rather than a vector, because a plan is `Copy` and a heap
/// allocation per launch in a pass that issues thousands of them is a cost
/// nothing here needs to pay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Args {
    ops: [Operand; MAX_ARGS],
    len: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self::new()
    }
}

impl Args {
    /// An empty argument list.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ops: [Operand::Null; MAX_ARGS],
            len: 0,
        }
    }

    /// The list `ops` spells, in kernel declaration order.
    ///
    /// A list longer than a launch can carry is recorded as its true length and
    /// refused by [`Args::operands`], rather than silently truncated: a
    /// truncated argument list is a launch that reads whatever the driver had
    /// in the slot.
    #[must_use]
    pub fn of(ops: &[Operand]) -> Self {
        let mut args = Self::new();
        args.len = ops.len();
        for (slot, op) in args.ops.iter_mut().zip(ops.iter()) {
            *slot = *op;
        }
        args
    }

    /// How many parameters the list carries.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the list is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The operands, in order.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] when the list was built from more operands
    /// than a launch can carry.
    pub fn operands(&self) -> CoreResult<&[Operand]> {
        self.ops.get(..self.len).ok_or(CoreError::ShapeMismatch {
            field: "kernel argument count",
            expected: MAX_ARGS as u64,
            got: self.len as u64,
        })
    }
}

/// What the streamed n-gram tier puts in front of a pass.
///
/// One value rather than two optional fields on the binder, because the two are
/// not independent: a stack that has staged rows at all has exactly one answer
/// about whether its row codec carries a bias, and a binder holding the bias of
/// a table whose rows it cannot name is a state nothing should be able to
/// build.
#[derive(Debug, Clone, Copy)]
pub struct NgramViews<'a> {
    /// Where this pass's gathered rows were staged.
    pub rows: DeviceView<'a>,
    /// The per-head bias the trellis codec finishes with, absent for the
    /// block-quantized codec, which has none.
    pub bias: Option<DeviceView<'a>>,
}

/// Everything an operand can name, in one place.
///
/// Borrowed rather than owned, and borrowed all at once, because the views it
/// hands out have to outlive the launch that reads them and cannot outlive the
/// buffers they name.
#[derive(Debug)]
pub struct Binder<'a> {
    scratch: &'a Scratch,
    weights: &'a SparseWeights,
    rows: [Option<&'a SparseSession>; MAX_ROWS],
    len: usize,
    cache: &'a PagedKv,
    ngram: Option<NgramViews<'a>>,
    media: Option<DeviceView<'a>>,
    splice: Option<&'a crate::launch::SpliceArgs>,
    coop: Option<&'a dyn CoopLaunch>,
    stream: Option<&'a crate::sparse::stream::ExpertStream>,
    snapshot: Option<DeviceView<'a>>,
}

impl<'a> Binder<'a> {
    /// Binds against one session's state and the stack's shared activations.
    #[must_use]
    pub const fn new(
        scratch: &'a Scratch,
        weights: &'a SparseWeights,
        session: &'a SparseSession,
        cache: &'a PagedKv,
    ) -> Self {
        let mut rows = [None; MAX_ROWS];
        rows[0] = Some(session);
        Self {
            scratch,
            weights,
            rows,
            len: 1,
            cache,
            ngram: None,
            media: None,
            splice: None,
            coop: None,
            stream: None,
            snapshot: None,
        }
    }

    /// Binds against one session per row of a decode batch, in row order.
    ///
    /// A fixed array rather than a borrowed slice, and that is not an
    /// optimization: the views this hands out carry the binder's own lifetime,
    /// so a slice would have to outlive every launch it is bound for, and the
    /// caller that assembles the rows is a stack method whose locals do not.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] for an empty batch or one wider than
    /// [`MAX_ROWS`].
    pub fn batched(
        scratch: &'a Scratch,
        weights: &'a SparseWeights,
        sessions: &[&'a SparseSession],
        cache: &'a PagedKv,
    ) -> CoreResult<Self> {
        if sessions.is_empty() || sessions.len() > MAX_ROWS {
            return Err(CoreError::Unsupported {
                what: "a batch wider than one binder holds",
                backend: "engine",
            });
        }
        let mut rows = [None; MAX_ROWS];
        for (slot, session) in rows.iter_mut().zip(sessions.iter()) {
            *slot = Some(*session);
        }
        Ok(Self {
            scratch,
            weights,
            rows,
            len: sessions.len(),
            cache,
            ngram: None,
            media: None,
            splice: None,
            coop: None,
            stream: None,
            snapshot: None,
        })
    }

    /// Rows this binder holds one session for.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.len
    }

    /// The session behind one row.
    ///
    /// # Errors
    ///
    /// [`CoreError::OutOfBounds`] for a row past the end of the batch, which is
    /// a plan built for a wider batch than the one being issued.
    pub fn row(&self, row: u32) -> CoreResult<&'a SparseSession> {
        let index = usize::try_from(row).unwrap_or(usize::MAX);
        self.rows
            .get(index)
            .copied()
            .flatten()
            .filter(|_| index < self.len)
            .ok_or(CoreError::OutOfBounds {
                offset: u64::from(row),
                len: 1,
                buffer_bytes: self.len as u64,
            })
    }

    /// The one session a pass over one sequence belongs to.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] when this binder holds a batch. An operand
    /// that names *the* session, resolved against a batch, would answer with
    /// the first row's page table or the first row's recurrent state for every
    /// row - which is not a slower answer for the other rows, it is another
    /// sequence's context read as theirs.
    fn only(&self) -> CoreResult<&'a SparseSession> {
        if self.len != 1 {
            return Err(CoreError::Unsupported {
                what: "an operand naming the pass's one session, bound against a batch of \
                       several - the launch that reads it has a batched sibling \
                       taking a per-row array, and this build's planner did not \
                       use it",
                backend: "engine",
            });
        }
        self.row(0)
    }

    /// The layout the per-row pointer arrays are packed at for this batch.
    ///
    /// The same derivation the pack itself uses, and deliberately the same
    /// function: a reader and a writer that computed the offsets separately
    /// would be two chances to disagree about where an array starts.
    fn layout(&self) -> CoreResult<Layout> {
        let rows = self
            .rows
            .get(..self.len)
            .ok_or(CoreError::OutOfBounds {
                offset: 0,
                len: self.len as u64,
                buffer_bytes: MAX_ROWS as u64,
            })?
            .iter()
            .copied()
            .flatten()
            .collect::<Vec<_>>();
        Ok(Layout::of_rows(&rows))
    }

    /// One per-row pointer array, as the range the pack wrote it into.
    fn array(&self, array: PtrArray) -> CoreResult<DeviceView<'a>> {
        let (offset, len) = self.layout()?.span(array).ok_or(CoreError::Unsupported {
            what: "a per-row pointer array this stack packs none of - the n-gram windows \
                   of a stack with no n-gram block, or a recurrent slot past \
                   the last one",
            backend: "engine",
        })?;
        self.scratch.view(Buf::RowPointers)?.slice(offset, len)
    }

    /// Names the staged n-gram rows this pass gathered, and the bias its codec
    /// finishes with.
    #[must_use]
    pub const fn with_ngram(mut self, views: NgramViews<'a>) -> Self {
        self.ngram = Some(views);
        self
    }

    /// Names the encoder block this request's placeholder rows are spliced out
    /// of.
    ///
    /// Per pass rather than per stack: a text request binds none, and a
    /// multimodal one binds the same block for every chunk its prompt was split
    /// into, because the splice map each chunk carries indexes that one block.
    #[must_use]
    pub const fn with_media(mut self, rows: DeviceView<'a>) -> Self {
        self.media = Some(rows);
        self
    }

    /// Names the splice's by-value argument block.
    ///
    /// Borrowed rather than carried by value because it is bound as bytes, and
    /// the bytes have to outlive the launch that reads them.
    #[must_use]
    pub(crate) const fn with_splice(mut self, args: &'a crate::launch::SpliceArgs) -> Self {
        self.splice = Some(args);
        self
    }

    /// Names the stack's snapshot buffer, which exists once a checkpoint tier
    /// is open.
    #[must_use]
    pub const fn with_snapshot(mut self, snapshot: DeviceView<'a>) -> Self {
        self.snapshot = Some(snapshot);
        self
    }

    /// Names the device the cooperative scratch is reserved on.
    ///
    /// The device rather than the range, and that is the whole of why: the
    /// reservation is a step of the pass, so the range does not exist when a
    /// binder is built and does exist by the time the launch that reads it is
    /// bound. A binder holding a range taken before the reservation would name
    /// the previous pass's slab.
    #[must_use]
    pub const fn with_coop(mut self, coop: &'a dyn CoopLaunch) -> Self {
        self.coop = Some(coop);
        self
    }

    /// The same binder, resolving the routed banks and the expert maps
    /// through a stream rather than through resident weights.
    #[must_use]
    pub const fn with_stream(mut self, stream: &'a crate::sparse::stream::ExpertStream) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Issues a [`crate::sparse::plan::Step::Fetch`]: brings one layer's routed
    /// experts into the stream's slots, or does nothing on a stack whose banks
    /// are resident.
    ///
    /// # Errors
    ///
    /// Whatever [`crate::sparse::stream::ExpertStream::fetch`] reports.
    pub fn fetch_experts(
        &self,
        backend: &dyn shiso_core::backend::Backend,
        layer: u32,
        ids: u32,
    ) -> CoreResult<()> {
        let Some(stream) = self.stream else {
            return Ok(());
        };
        stream.fetch(backend, layer, self.scratch.view(Buf::MoeIds)?, ids)
    }

    /// Resolves one argument list into what a launch takes.
    ///
    /// # Errors
    ///
    /// Whatever [`Args::operands`] reports for an over-long list, and whatever
    /// [`Binder::arg`] reports for an operand this context cannot name.
    pub fn bind(&self, args: Args, out: &mut Vec<KernelArg<'a>>) -> CoreResult<()> {
        out.clear();
        for op in args.operands()? {
            out.push(self.arg(*op)?);
        }
        Ok(())
    }

    /// Resolves one operand.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] for an operand naming state this context does
    /// not hold - the staged n-gram rows of a stack with no n-gram block, the
    /// cooperative scratch of a device that reserved none - and
    /// [`CoreError::OutOfBounds`] for a recurrent slot or layer past the end.
    #[expect(
        clippy::too_many_lines,
        reason = "one arm per operand: this match is the vocabulary's only \
                  resolver, and a split would give an operand two places it \
                  might be answered"
    )]
    pub fn arg(&self, op: Operand) -> CoreResult<KernelArg<'a>> {
        Ok(match op {
            Operand::Scratch(buf) => KernelArg::Buffer(self.scratch.view(buf)?),
            Operand::ScratchFrom(buf, offset) => {
                KernelArg::Buffer(self.scratch.view_from(buf, offset)?)
            }
            Operand::Weight(reference) => KernelArg::Buffer(self.weight(reference)?),
            Operand::Pages => KernelArg::Buffer(self.only()?.pages()),
            Operand::PagesOf(row) => KernelArg::Buffer(self.row(row)?.pages()),
            Operand::PageTables => KernelArg::Buffer(self.array(PtrArray::Pages)?),
            Operand::RecurrentConvs(slot) => {
                KernelArg::Buffer(self.array(PtrArray::RecurrentConv(slot))?)
            }
            Operand::RecurrentStates(slot) => {
                KernelArg::Buffer(self.array(PtrArray::RecurrentState(slot))?)
            }
            Operand::RecurrentConvCkpts(slot) => {
                KernelArg::Buffer(self.array(PtrArray::RecurrentConvCkpt(slot))?)
            }
            Operand::RecurrentStateCkpts(slot) => {
                KernelArg::Buffer(self.array(PtrArray::RecurrentStateCkpt(slot))?)
            }
            Operand::NgramWindows => KernelArg::Buffer(self.array(PtrArray::NgramWindows)?),
            Operand::CachePlane { rank, plane } => {
                KernelArg::Buffer(self.cache.plane(rank, plane)?)
            }
            Operand::RecurrentConv(slot) => KernelArg::Buffer(self.only()?.recurrent(slot)?.0),
            Operand::RecurrentState(slot) => KernelArg::Buffer(self.only()?.recurrent(slot)?.1),
            Operand::RecurrentConvOf { slot, run } => {
                KernelArg::Buffer(self.row(run)?.recurrent(slot)?.0)
            }
            Operand::RecurrentStateOf { slot, run } => {
                KernelArg::Buffer(self.row(run)?.recurrent(slot)?.1)
            }
            Operand::RecurrentConvCkpt(slot) => {
                KernelArg::Buffer(self.only()?.recurrent_rollback(slot)?.0)
            }
            Operand::RecurrentStateCkpt(slot) => {
                KernelArg::Buffer(self.only()?.recurrent_rollback(slot)?.1)
            }
            Operand::NgramWindowOf(row) => KernelArg::Buffer(self.row(row)?.ngram_window().ok_or(
                CoreError::Unsupported {
                    what: "the n-gram convolution window of a session whose stack has no n-gram \
                           block",
                    backend: "engine",
                },
            )?),
            Operand::NgramWindowCkptOf(row) => {
                KernelArg::Buffer(self.row(row)?.ngram_window_rollback().ok_or(
                    CoreError::Unsupported {
                        what: "the per-position n-gram windows of a session that was not opened \
                               for speculation",
                        backend: "engine",
                    },
                )?)
            }
            Operand::NgramWindowCkpt => {
                KernelArg::Buffer(self.only()?.ngram_window_rollback().ok_or(
                    CoreError::Unsupported {
                        what: "the per-position n-gram windows of a session that was not opened \
                               for speculation",
                        backend: "engine",
                    },
                )?)
            }
            Operand::NgramWindow => {
                KernelArg::Buffer(self.only()?.ngram_window().ok_or(CoreError::Unsupported {
                    what: "the n-gram convolution window of a session whose stack has no n-gram \
                           block",
                    backend: "engine",
                })?)
            }
            Operand::Snapshot(offset) => {
                let snapshot = self.snapshot.ok_or(CoreError::Unsupported {
                    what: "the snapshot buffer of a stack with no checkpoint tier open",
                    backend: "engine",
                })?;
                KernelArg::Buffer(snapshot.slice(offset, snapshot.len().saturating_sub(offset))?)
            }
            Operand::DraftRow => {
                KernelArg::Buffer(self.only()?.pending_draft().ok_or(CoreError::Unsupported {
                    what: "the staged draft row of a session that has not staged one, which \
                           would draft from whatever the previous request left there",
                    backend: "engine",
                })?)
            }
            Operand::NgramRows => KernelArg::Buffer(
                self.ngram
                    .ok_or(CoreError::Unsupported {
                        what: "the streamed n-gram rows of a pass that gathered none",
                        backend: "engine",
                    })?
                    .rows,
            ),
            Operand::SpliceArgs => KernelArg::Bytes(bytemuck::bytes_of(self.splice.ok_or(
                CoreError::Unsupported {
                    what: "the splice argument block of a pass that carries no media, whose \
                           three widths nothing built",
                    backend: "engine",
                },
            )?)),
            Operand::MediaRows => KernelArg::Buffer(self.media.ok_or(CoreError::Unsupported {
                what: "the encoder block of a pass that carries no media, which would splice \
                       the placeholder rows out of whatever was last at that address",
                backend: "engine",
            })?),
            Operand::NgramBias => {
                KernelArg::Buffer(self.ngram.and_then(|views| views.bias).ok_or(
                    CoreError::Unsupported {
                        what: "the per-head bias of an n-gram table whose row codec carries none",
                        backend: "engine",
                    },
                )?)
            }
            Operand::ExpertMap(layer) if self.stream.is_some() => KernelArg::Buffer(
                self.stream
                    .ok_or(CoreError::Unsupported {
                        what: "an expert map on a stack with no stream",
                        backend: "engine",
                    })?
                    .map(layer)?,
            ),
            Operand::ExpertMap(_) | Operand::ResidentExpertMap => KernelArg::Buffer(
                self.weights
                    .expert_map
                    .as_ref()
                    .ok_or(CoreError::Unsupported {
                        what: "the expert-to-slot map of a model whose expert banks are not affine",
                        backend: "engine",
                    })?
                    .view(),
            ),
            Operand::CoopLocks => KernelArg::Buffer(self.coop_at(0)?),
            Operand::CoopStaging(offset) => {
                KernelArg::Buffer(self.coop_at(crate::sparse::exl3::STAGING_BASE + offset)?)
            }
            // Eight zero bytes in a pointer-sized slot. A zero-length buffer
            // would not do: that is still an address, and the kernels that take
            // one of these test the pointer itself.
            Operand::Null => KernelArg::U64(0),
            Operand::U32(value) => KernelArg::U32(value),
            Operand::I32(value) => KernelArg::I32(value),
            Operand::U64(value) => KernelArg::U64(value),
            Operand::F32(bits) => KernelArg::F32(f32::from_bits(bits)),
        })
    }

    /// Resolves one operand that has to name device memory.
    ///
    /// The multiply tier takes its operands as a descriptor rather than an
    /// argument array, so it needs the range itself rather than a launch
    /// argument. An operand that is a scalar is a refusal here rather than a
    /// number reinterpreted as an address.
    ///
    /// # Errors
    ///
    /// Whatever [`Binder::arg`] reports, and [`CoreError::Unsupported`] when
    /// the operand names a scalar.
    pub fn buffer(&self, op: Operand) -> CoreResult<DeviceView<'a>> {
        match self.arg(op)? {
            KernelArg::Buffer(view) => Ok(view),
            _ => Err(CoreError::Unsupported {
                what: "a scalar where a matrix multiply's operand belongs",
                backend: "engine",
            }),
        }
    }

    /// The cooperative scratch from a byte offset on.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] when the device reserved none, and
    /// [`CoreError::OutOfBounds`] when the reservation is shorter than the
    /// carve-up a launch was planned with - which is the one failure a plan
    /// built against a larger chunk and issued against a smaller reservation
    /// has, and it is better named here than as a launch that reads past the
    /// end.
    fn coop_at(&self, offset: u64) -> CoreResult<DeviceView<'a>> {
        let slab = self
            .coop
            .and_then(CoopLaunch::scratch)
            .ok_or(CoreError::Unsupported {
                what: "cooperative launch scratch on a device that reserved none",
                backend: "engine",
            })?;
        let len = slab
            .len()
            .checked_sub(offset)
            .ok_or(CoreError::OutOfBounds {
                offset,
                len: 0,
                buffer_bytes: slab.len(),
            })?;
        slab.slice(offset, len)
    }

    /// The device range one bound tensor occupies.
    ///
    /// # Errors
    ///
    /// [`CoreError::OutOfBounds`] for a layer past the end,
    /// [`CoreError::Unsupported`] when the role names a tensor the layer does
    /// not hold - a recurrent role on an attention slot, an n-gram role on a
    /// slot that folds in no n-gram block, an injection on the collapsing mix.
    pub fn weight(&self, reference: WeightRef) -> CoreResult<DeviceView<'a>> {
        let part = reference.part;
        // A norm or bias vector is one span by construction, so asking for a
        // trellis part of one is a naming error rather than an offset.
        let vector = |buffer: &'a shiso_core::buffer::DeviceBuffer| -> CoreResult<DeviceView<'a>> {
            if part == WeightPart::Whole {
                Ok(buffer.view())
            } else {
                Err(CoreError::Unsupported {
                    what: "a trellis payload part of a norm vector",
                    backend: "engine",
                })
            }
        };
        let head = || {
            self.weights.draft.as_ref().ok_or(CoreError::Unsupported {
                what: "a draft head's tensor on a model that carries no draft head",
                backend: "engine",
            })
        };
        if reference.role.is_model_wide() {
            return match reference.role {
                Role::TokenEmbd => self.weights.token_embd.part(part),
                Role::Output => self.weights.output_matrix().part(part),
                Role::OutputMixNorm => vector(&self.weights.output_mix.norm),
                Role::OutputMixDown => self.weights.output_mix.down.part(part),
                Role::OutputMixUp => self.weights.output_mix.up.part(part),
                Role::DraftNormEmbed => vector(&head()?.norm_embed),
                Role::DraftNormHidden => vector(&head()?.norm_hidden),
                Role::DraftFcEmbed => head()?.fc_embed.part(part),
                Role::DraftFcHidden => head()?.fc_hidden.part(part),
                _ => head()?
                    .head
                    .as_ref()
                    .ok_or(CoreError::Unsupported {
                        what: "a narrowed vocabulary head on a build that has none",
                        backend: "engine",
                    })?
                    .weight()
                    .part(part),
            };
        }
        // The draft head's own mix is model-wide *to the head*, so it answers
        // to the same three roles the target's collapsing mix does - which is
        // why the set is read before the role and not after it.
        if reference.layer.set == WeightSet::Draft {
            return match reference.role {
                Role::OutputMixNorm => vector(&head()?.output_mix.norm),
                Role::OutputMixDown => head()?.output_mix.down.part(part),
                Role::OutputMixUp => head()?.output_mix.up.part(part),
                _ => Self::layer_weight(&head()?.layer, reference, part),
            };
        }
        if let Some(stream) = self.stream
            && matches!(
                reference.role,
                Role::ExpertGate | Role::ExpertUp | Role::ExpertDown
            )
        {
            return stream.bank(reference.role, part);
        }
        let index = usize::try_from(reference.layer.index).unwrap_or(usize::MAX);
        let layer = self
            .weights
            .layers
            .get(index)
            .ok_or(CoreError::OutOfBounds {
                offset: u64::from(reference.layer.index),
                len: 1,
                buffer_bytes: self.weights.layers.len() as u64,
            })?;
        Self::layer_weight(layer, reference, part)
    }

    /// One tensor of one layer, whichever set the layer came from.
    ///
    /// Split out of [`Binder::weight`] because the two weight sets reach it by
    /// different routes - the target's through a layer index, the draft head's
    /// through the head - and a second copy of the role table would be a second
    /// place for the two to disagree.
    fn layer_weight(
        layer: &'a crate::sparse::weights::SparseLayer,
        reference: WeightRef,
        part: WeightPart,
    ) -> CoreResult<DeviceView<'a>> {
        use crate::sparse::weights::SparseBlock;

        let vector = |buffer: &'a shiso_core::buffer::DeviceBuffer| -> CoreResult<DeviceView<'a>> {
            if part == WeightPart::Whole {
                Ok(buffer.view())
            } else {
                Err(CoreError::Unsupported {
                    what: "a trellis payload part of a norm vector",
                    backend: "engine",
                })
            }
        };
        let missing = |what: &'static str| CoreError::Unsupported {
            what,
            backend: "engine",
        };
        let attn = || match &layer.block {
            SparseBlock::Attn(attn) => Ok(attn),
            SparseBlock::Recurrent(_) => Err(missing(
                "an attention tensor of a slot that holds the recurrent block",
            )),
        };
        let recurrent = || match &layer.block {
            SparseBlock::Recurrent(gdn) => Ok(gdn),
            SparseBlock::Attn(_) => Err(missing(
                "a recurrent tensor of a slot that holds the attention block",
            )),
        };
        let ngram = || {
            layer
                .ngram
                .as_ref()
                .ok_or_else(|| missing("an n-gram tensor of a slot that folds in no n-gram block"))
        };
        match reference.role {
            Role::AttnMixNorm => vector(&layer.attn_mix.norm),
            Role::AttnMixDown => layer.attn_mix.down.part(part),
            Role::AttnMixUp => layer.attn_mix.up.part(part),
            Role::AttnMixInject => layer
                .attn_mix
                .inject
                .as_ref()
                .ok_or_else(|| missing("an injection the token mixer's stream mix does not hold"))?
                .part(part),
            Role::FfnMixNorm => vector(&layer.ffn_mix.norm),
            Role::FfnMixDown => layer.ffn_mix.down.part(part),
            Role::FfnMixUp => layer.ffn_mix.up.part(part),
            Role::FfnMixInject => layer
                .ffn_mix
                .inject
                .as_ref()
                .ok_or_else(|| missing("an injection the feed-forward's stream mix does not hold"))?
                .part(part),
            Role::AttnQuery => attn()?.q.part(part),
            Role::AttnKey => attn()?.k.part(part),
            Role::AttnValue => attn()?.v.part(part),
            Role::AttnOutput => attn()?.output.part(part),
            Role::AttnQueryNorm => vector(&attn()?.q_norm),
            Role::AttnKeyNorm => vector(&attn()?.k_norm),
            Role::IndexerQuery => attn()?.indexer.q.part(part),
            Role::IndexerKey => attn()?.indexer.k.part(part),
            Role::IndexerQueryNorm => vector(&attn()?.indexer.q_norm),
            Role::IndexerKeyNorm => vector(&attn()?.indexer.k_norm),
            Role::RecurrentQkv => recurrent()?.qkv.part(part),
            Role::RecurrentGate => recurrent()?.gate.part(part),
            Role::RecurrentAlpha => recurrent()?.alpha.part(part),
            Role::RecurrentBeta => recurrent()?.beta.part(part),
            Role::RecurrentOut => recurrent()?.out.part(part),
            Role::RecurrentConv1d => vector(&recurrent()?.conv1d),
            Role::RecurrentDecayBase => vector(&recurrent()?.a),
            Role::RecurrentDecayBias => vector(&recurrent()?.dt_bias),
            Role::RecurrentNorm => vector(&recurrent()?.norm),
            Role::Router => layer.experts.router.part(part),
            Role::ExpertGate => layer.experts.gate.part(part),
            Role::ExpertUp => layer.experts.up.part(part),
            Role::ExpertDown => layer.experts.down.part(part),
            Role::SharedRouter => layer.experts.shared_router.part(part),
            Role::SharedGate => layer.experts.shared_gate.part(part),
            Role::SharedUp => layer.experts.shared_up.part(part),
            Role::SharedDown => layer.experts.shared_down.part(part),
            Role::NgramKey => ngram()?.key.part(part),
            Role::NgramValue => ngram()?.value.part(part),
            Role::NgramNormKey => vector(&ngram()?.norm_key),
            Role::NgramNormQuery => vector(&ngram()?.norm_query),
            Role::NgramNormConv => vector(&ngram()?.norm_conv),
            Role::NgramConv1d => vector(&ngram()?.conv1d),
            // Every role either set holds once, answered above.
            Role::TokenEmbd
            | Role::Output
            | Role::OutputMixNorm
            | Role::OutputMixDown
            | Role::OutputMixUp
            | Role::DraftNormEmbed
            | Role::DraftNormHidden
            | Role::DraftFcEmbed
            | Role::DraftFcHidden
            | Role::DraftVocabulary => unreachable_model_wide(),
        }
    }
}

/// The arm no caller reaches, as a refusal rather than a panic.
///
/// The model-wide roles are answered before the per-layer match runs, so this
/// is the arm the compiler needs and nothing takes. It refuses rather than
/// panicking, because a future role added to one list and not the other should
/// be an error a caller sees, not a process that stops.
fn unreachable_model_wide<'a>() -> CoreResult<DeviceView<'a>> {
    Err(CoreError::Unsupported {
        what: "a model-wide tensor resolved through the per-layer table",
        backend: "engine",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_argument_list_records_its_true_length_rather_than_truncating() {
        let ops = [Operand::U32(1); MAX_ARGS + 2];
        let args = Args::of(&ops);
        assert_eq!(args.len(), MAX_ARGS + 2);
        let err = args
            .operands()
            .expect_err("a list longer than a launch can carry is refused");
        assert!(matches!(err, CoreError::ShapeMismatch { .. }), "{err:?}");
    }

    #[test]
    fn two_lists_that_would_launch_the_same_bytes_compare_equal() {
        // The single-precision scalars travel as bit patterns for exactly this:
        // a plan is compared, and two floats that are the same value are the
        // same launch.
        let left = Args::of(&[Operand::f32(1.0e-6), Operand::U32(4)]);
        let right = Args::of(&[Operand::f32(1.0e-6), Operand::U32(4)]);
        assert_eq!(left, right);
        assert_ne!(left, Args::of(&[Operand::f32(1.0e-5), Operand::U32(4)]));
    }

    #[test]
    fn a_null_pointer_is_a_pointer_sized_slot_and_not_an_empty_buffer() {
        assert!(Operand::Null.is_pointer());
        assert!(!Operand::U32(0).is_pointer());
    }

    #[test]
    fn the_model_wide_roles_are_the_five_a_layer_does_not_hold() {
        for role in [
            Role::TokenEmbd,
            Role::Output,
            Role::OutputMixNorm,
            Role::OutputMixDown,
            Role::OutputMixUp,
        ] {
            assert!(role.is_model_wide(), "{role:?}");
        }
        for role in [
            Role::AttnMixNorm,
            Role::ExpertDown,
            Role::RecurrentQkv,
            Role::NgramConv1d,
        ] {
            assert!(!role.is_model_wide(), "{role:?}");
        }
    }
}
