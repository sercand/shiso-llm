//! The geometry of the stack this crate runs.
//!
//! These are architecture facts, not product facts. A slot holds a recurrent
//! block or an attention block; an attention block has query heads, fewer
//! key/value heads and a rotated prefix; a recurrent block has value heads,
//! fewer query/key heads and a short causal convolution. Every checkpoint this
//! crate can run is some filling-in of those numbers, and the crate that knows
//! WHICH checkpoint - what it is called, which config file describes it, which
//! token ends a turn - is a different crate.
//!
//! # Why an attention slot states its own widths
//!
//! One [`AttnLayer`] per attention slot rather than one set of widths for the
//! whole stack, because a family exists whose slots are not all the same: two
//! head widths in one model, a sliding window on some slots and not others, and
//! slots that read an earlier slot's key/value runs instead of writing their
//! own. A single set of widths beside a layer list could not say any of that,
//! and every number the pass reads has to be a number some slot actually has.
//!
//! # Why this is not in the shared vocabulary crate
//!
//! It was, along with a named shape per model and an enum to pick between them.
//! That made the bottom of the stack carry the names of the two checkpoints
//! above it, so a third meant editing the crate that everything depends on. What
//! the bottom of the stack actually needs to know about a model is in
//! [`shiso_core::ModelInfo`]: a handful of numbers a scheduler reads. The
//! geometry a *forward pass* needs belongs with the forward pass, which is here.

use std::num::NonZeroU32;

use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::LayerIdx;

/// Which block a layer slot holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerKind {
    /// A gated delta-rule recurrent block. It has no key/value rows; its memory
    /// is a convolution window and a state matrix that belong to one session.
    Gdn,
    /// An attention block, which owns key/value rows in the shared cache.
    Attn,
    /// A selective state-space block: a causal convolution and a diagonal
    /// state recurrence, whose memory is a convolution window and one state
    /// matrix per head that belong to one session - a recurrent slot, as
    /// [`LayerKind::Gdn`] is, with its geometry in [`HybridShape::ssm`].
    Ssm,
    /// A gated short convolution: two elementwise gates around a depthwise
    /// causal convolution of a few taps over the residual width. Its memory is
    /// the last few gated inputs and nothing else, which belongs to one
    /// session - a recurrent slot, with its geometry in
    /// [`HybridShape::short_conv`].
    ShortConv,
}

/// Where an attention slot's keys and values live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvSource {
    /// The slot writes and reads its own runs.
    Own,
    /// The slot reads the runs of an earlier attention slot and writes nothing.
    Shared {
        /// Slot whose runs this one reads. It must own them itself.
        owner: LayerIdx,
    },
}

/// The softmax scale of an attention slot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttnScale {
    /// The usual `1 / sqrt(head_dim)`.
    InvSqrtHeadDim,
    /// No scaling at all, which is a family's decision rather than an omission.
    One,
}

impl AttnScale {
    /// The number the scores are multiplied by, for a head of `head_dim`.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "a head width is a small integer, and the scale it implies is a float"
    )]
    pub fn value(self, head_dim: u32) -> f32 {
        match self {
            Self::InvSqrtHeadDim => 1.0 / (head_dim as f32).sqrt(),
            Self::One => 1.0,
        }
    }
}

/// One attention slot's geometry.
///
/// `#[non_exhaustive]` because a slot is built through [`AttnLayer::new`] and
/// then narrowed by the `with_*` methods: a family that has none of the
/// narrowings writes one constructor call, and a field added here is a field
/// only the families that need it have to name.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct AttnLayer {
    /// Query heads.
    pub q_heads: u32,
    /// Key/value heads. `q_heads / kv_heads` queries share each one.
    pub kv_heads: u32,
    /// Width of one head, which in these models is not `hidden / q_heads`.
    pub head_dim: u32,
    /// Dimensions rotated: pairs `(i, i + freq_dims/2)` for `i < n_rot/2`.
    pub n_rot: u32,
    /// Width the frequency ladder is computed over: `angle_i = pos * theta^(-2i/freq_dims)`.
    pub freq_dims: u32,
    /// Rotary frequency base.
    pub rope_theta: f32,
    /// Sliding window in positions; `None` is full causal attention.
    pub window: Option<NonZeroU32>,
    /// Where this slot's keys and values live.
    pub kv: KvSource,
    /// The softmax scale this slot applies.
    pub scale: AttnScale,
    /// Whether the query and key heads are RMS-normalized before the rotation.
    ///
    /// Independent of the rotation, which [`Self::n_rot`] states: `false` with
    /// a non-zero `n_rot` is a slot that rotates its heads and normalizes
    /// neither, and `false` with an `n_rot` of zero takes the projections as
    /// they are.
    pub qk_norm: bool,
}

impl AttnLayer {
    /// A full-causal slot that owns its cache and scales by `1 / sqrt(head_dim)`.
    ///
    /// The three narrowings a family may need - a sliding window, a shared
    /// cache, a different scale - are the `with_*` methods below, so the widths
    /// every attention slot has are the whole of this signature.
    #[must_use]
    pub const fn new(
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        n_rot: u32,
        freq_dims: u32,
        rope_theta: f32,
    ) -> Self {
        Self {
            q_heads,
            kv_heads,
            head_dim,
            n_rot,
            freq_dims,
            rope_theta,
            window: None,
            kv: KvSource::Own,
            scale: AttnScale::InvSqrtHeadDim,
            qk_norm: true,
        }
    }

    /// The same slot, with neither a query nor a key norm.
    #[must_use]
    pub const fn without_qk_norm(self) -> Self {
        Self {
            qk_norm: false,
            ..self
        }
    }

    /// The same slot, seeing only the last `w` positions. `0` is no window.
    #[must_use]
    pub const fn with_window(self, w: u32) -> Self {
        Self {
            window: NonZeroU32::new(w),
            ..self
        }
    }

    /// The same slot, reading `owner`'s key/value runs and writing none.
    #[must_use]
    pub const fn shared_from(self, owner: LayerIdx) -> Self {
        Self {
            kv: KvSource::Shared { owner },
            ..self
        }
    }

    /// The same slot, with `s` in place of the usual inverse square root.
    #[must_use]
    pub const fn with_scale(self, s: AttnScale) -> Self {
        Self { scale: s, ..self }
    }

    /// Whether this slot writes the runs it reads.
    #[must_use]
    pub const fn owns_kv(&self) -> bool {
        matches!(self.kv, KvSource::Own)
    }
}

/// Which feed-forward activation the stack applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// `silu(gate) * up`.
    SwiGlu,
    /// `gelu_tanh(gate) * up`.
    GeGluTanh,
    /// `max(up, 0)^2`, with no gate projection at all.
    Relu2,
}

/// Which norms a layer slot carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormScheme {
    /// input norm and the feed-forward's pre-norm only (qwen35).
    PreOnly,
    /// plus a post-attention norm and a post-feed-forward norm (Gemma).
    PrePost,
}

/// How a media prompt is positioned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaPositions {
    /// Three grid streams the rotation reads in sections.
    Mrope3D,
    /// One position per token, media included.
    Linear,
}

/// The per-layer embedding table's widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PerLayerShape {
    /// Width of one slot's per-layer embedding row.
    pub dim: u32,
    /// Rows the table holds, which is a vocabulary of its own.
    pub vocab: u32,
}

impl PerLayerShape {
    /// The two widths of a per-layer embedding table.
    #[must_use]
    pub const fn new(dim: u32, vocab: u32) -> Self {
        Self { dim, vocab }
    }
}

/// Recurrent geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct GdnShape {
    /// State heads, one square state matrix each.
    pub v_heads: u32,
    /// Query/key heads, of which there are fewer than state heads. A state head
    /// reads the query/key head at its own index modulo this count.
    pub qk_heads: u32,
    /// Width of one head.
    pub head_dim: u32,
    /// Width of the short causal convolution over the projections.
    pub conv_kernel: u32,
}

impl GdnShape {
    /// The four widths of a recurrent block.
    #[must_use]
    pub const fn new(v_heads: u32, qk_heads: u32, head_dim: u32, conv_kernel: u32) -> Self {
        Self {
            v_heads,
            qk_heads,
            head_dim,
            conv_kernel,
        }
    }
}

/// A selective state-space block's widths.
///
/// `heads * head_dim` channels are mixed; `groups` input and output projections
/// of `state` values each are shared by `heads / groups` consecutive heads; the
/// convolution runs over the mixed channels and both projections together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsmShape {
    /// State heads, one `head_dim x state` state matrix each.
    pub heads: u32,
    /// Channels one head mixes.
    pub head_dim: u32,
    /// Values one head's state carries per channel.
    pub state: u32,
    /// Groups the heads share input and output projections in.
    pub groups: u32,
    /// Width of the short causal convolution.
    pub conv_kernel: u32,
}

impl SsmShape {
    /// Channels the heads mix: `heads * head_dim`.
    #[must_use]
    pub const fn inner(&self) -> u32 {
        self.heads * self.head_dim
    }

    /// Channels the convolution runs over: the mixed ones and both
    /// projections, `inner + 2 * groups * state`.
    #[must_use]
    pub const fn conv_channels(&self) -> u32 {
        self.inner() + 2 * self.groups * self.state
    }

    /// Width of the block's input projection: gate, convolved channels and
    /// one step size per head.
    #[must_use]
    pub const fn in_width(&self) -> u32 {
        self.inner() + self.conv_channels() + self.heads
    }
}

/// A gated short-convolution block's widths.
///
/// The convolution runs over the residual width itself, so the one number a
/// checkpoint states is how many taps it has; everything else is
/// [`HybridShape::hidden`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShortConvShape {
    /// Width of the causal convolution, the current input included.
    pub taps: u32,
}

/// Experts one token may reach, above which both routers write past their
/// chosen-weight and chosen-id arrays.
///
/// Sixteen, because that is the width both routers declare and neither bounds:
/// `__shared__ float chosen_w[16]` / `chosen_i[16]` at
/// `kernels/cuda/q4e/moe.cu:44-47`, and the `threadgroup` pair the Metal port
/// carries faithfully at
/// `kernels/metal/qwen35moe/moe_route.metal:161-162`. Both index
/// them by `n_used` with no test. The CUDA comment claims its entry point
/// rejects anything wider; it does not, and the constant that comment cites
/// does not exist in this tree. That file is fidelity-gated and its bytes are
/// left alone, so this host constant is the only rejection there is.
const ROUTER_CHOSEN_SLOTS: u32 = 16;

/// Router geometry: how many experts exist, how many a token reaches, and how
/// wide each one is.
///
/// Here rather than in `crate::sparse::shape`, where it was written, because a
/// routed feed-forward turned out not to be a fact about the sparse stack: a
/// dense-hybrid checkpoint carries one too ([`HybridShape::moe`]), with one
/// residual stream and no indexer anywhere in it. The sparse module re-exports
/// this type, so nothing there changed but where the definition lives.
///
/// Deliberately NOT `#[non_exhaustive]`. A model crate has to build one of
/// these, and the attribute exists so a library can add a field without
/// breaking callers it cannot recompile - which here would only mean the
/// callers cannot build it at all. A new field is a number some checkpoint
/// family is not supplying yet, and a compile error at that crate is how that
/// should surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeShape {
    /// Routed experts the layer holds.
    pub experts: u32,
    /// Experts one token is routed to.
    pub used: u32,
    /// Always-on experts every token additionally pays for.
    pub shared: u32,
    /// Inner width of one routed expert's gate, up and down projections.
    ///
    /// **Not** the shared expert's, which is [`HybridShape::intermediate`] -
    /// the shared expert IS the slot's dense feed-forward, and a family whose
    /// two widths differ states each of them once.
    pub intermediate: u32,
}

impl MoeShape {
    /// The four widths of one routed feed-forward.
    #[must_use]
    pub const fn new(experts: u32, used: u32, shared: u32, intermediate: u32) -> Self {
        Self {
            experts,
            used,
            shared,
            intermediate,
        }
    }

    /// Expert columns one chunk of `n_tok` rows produces.
    ///
    /// This is the row count every stage after the router counts in: the gate,
    /// the up, the activation and the down all run one row per (token, slot)
    /// pair, and the sort that groups them by expert sorts exactly this many
    /// columns.
    #[must_use]
    pub const fn slot_rows(&self, n_tok: u32) -> u64 {
        n_tok as u64 * self.used as u64
    }
}

/// The geometry of one hybrid recurrent/attention stack.
///
/// The layer list is the only place the slot count lives: a separate
/// `n_layers` field beside it would be a second copy of the same fact, and the
/// two could disagree. Everything that wants a count reads
/// [`HybridShape::n_layers`], which counts the list.
///
/// Deliberately NOT `#[non_exhaustive]`, unlike most public types here. That
/// attribute exists so a library can add a field without breaking its callers,
/// and this is a type whose callers SHOULD break: every field is a geometry fact
/// the forward pass reads, so a new one means there is a number no existing model
/// crate is supplying. A compile error naming each of them is the correct outcome;
/// a silent default would be a wrong answer.
///
/// # Why the router is an `Option` here and not a third shape
///
/// A `Qwen3_5MoeForConditionalGeneration` checkpoint is this skeleton exactly:
/// one residual stream, a recurrent block three slots in four, a gated
/// attention block in the fourth, one set of widths for every attention slot.
/// What it does NOT have is the four things
/// [`crate::sparse::shape::SparseShape`] requires - several residual streams,
/// the low-rank pair that mixes them, an indexer tower, an n-gram block - so
/// that shape could only hold it by making three of its own fields mean
/// "absent", and the sparse pass would then be running hyper-connections over
/// one stream and an indexer over nothing.
///
/// The only thing it has that this shape did not is a routed feed-forward, and
/// a feed-forward is one step of the layer loop. So the choice was one
/// `Option` here against a second copy of `crate::layers` and `crate::stack` -
/// about 2,700 lines - whose only difference from the first would be that one
/// step. The cost of the `Option` is borne by every dense family exactly once,
/// as a `moe: None` at its construction site; the cost of the copy would have
/// been borne at every later edit to the layer loop, twice, forever.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridShape {
    /// Block kind of every slot, index zero first.
    pub layers: Vec<LayerKind>,
    /// One per [`LayerKind::Attn`] slot, in slot order. Length must equal the
    /// number of attention slots, which [`HybridShape::validate`] checks.
    pub attn: Vec<AttnLayer>,
    /// Residual-stream width.
    pub hidden: u32,
    /// Vocabulary size, which is also the logit row width.
    pub vocab: u32,
    /// Feed-forward inner width.
    ///
    /// For a routed stack this is the SHARED expert's width, because the shared
    /// expert is exactly the slot's dense feed-forward: the same three matrices
    /// over the same rows, running whatever the router decided. One routed
    /// expert's width is [`MoeShape::intermediate`], which a family whose two
    /// widths differ states separately.
    pub intermediate: u32,
    /// Recurrent block geometry.
    pub gdn: GdnShape,
    /// Whether the attention output is gated by a second projection carried in
    /// the query projection's own rows.
    pub attn_output_gate: bool,
    /// Epsilon every normalization in the stack adds.
    pub rms_eps: f32,
    /// The feed-forward's activation.
    pub activation: Activation,
    /// Which norms a slot carries.
    pub norms: NormScheme,
    /// Multiplier on the gathered token rows (Gemma: `sqrt(hidden)`); `None` is 1.
    pub embed_scale: Option<f32>,
    /// `cap * tanh(logits / cap)` on the logits row; `None` is off.
    pub final_softcap: Option<f32>,
    /// Whether each slot carries a scalar its output is multiplied by.
    pub layer_scalar: bool,
    /// The per-layer embedding table, for a family that has one.
    pub ple: Option<PerLayerShape>,
    /// How a media prompt is positioned.
    pub media_positions: MediaPositions,
    /// Whether the value projection is RMS-normalized (unweighted) before caching.
    pub v_norm: bool,
    /// The router, for a stack whose feed-forward is a mixture of experts.
    ///
    /// `None` is a dense feed-forward, which is what every slot of this shape
    /// carried before a routed hybrid arrived. See the type's own note for why
    /// an `Option` here beat a third shape.
    pub moe: Option<MoeShape>,
    /// State-space block geometry, for a stack with [`LayerKind::Ssm`] slots.
    ///
    /// A stack's recurrent slots are one kind: gated delta-rule ([`Self::gdn`])
    /// or state-space (this), never both, which [`Self::validate`] checks.
    pub ssm: Option<SsmShape>,
    /// Gated short-convolution geometry, for a stack with
    /// [`LayerKind::ShortConv`] slots.
    ///
    /// Like [`Self::ssm`], a third recurrent kind that never shares a stack
    /// with either of the other two, which [`Self::validate`] checks.
    pub short_conv: Option<ShortConvShape>,
    /// Slots that carry no feed-forward half, in ascending order.
    ///
    /// Empty for every family whose slots all have one. A checkpoint whose
    /// residual blocks alternate a mixing block and a feed-forward, with some
    /// mixing blocks followed directly by another, maps the pair to one slot
    /// and the unpaired mixing block to a slot listed here: two residual
    /// blocks in a row are exactly what one slot computes, so the pairing
    /// changes no arithmetic and saves a close per pair.
    pub no_ffn: Vec<LayerIdx>,
}

impl HybridShape {
    /// Total layer slots.
    #[must_use]
    pub fn n_layers(&self) -> u32 {
        u32::try_from(self.layers.len()).unwrap_or(u32::MAX)
    }

    /// Block kind of one slot, or `None` past the last one.
    ///
    /// The bound is in the return type rather than in an assertion, so a caller
    /// that walks off the end gets `None` instead of a panic.
    #[must_use]
    pub fn layer_kind(&self, il: LayerIdx) -> Option<LayerKind> {
        self.layers.get(il.get() as usize).copied()
    }

    /// How many slots hold `kind`.
    #[must_use]
    pub fn count(&self, kind: LayerKind) -> u32 {
        let n = self.layers.iter().filter(|slot| **slot == kind).count();
        u32::try_from(n).unwrap_or(u32::MAX)
    }

    /// How many slots keep a session's recurrent state: the gated delta-rule,
    /// the state-space and the short-convolution ones.
    #[must_use]
    pub fn recurrent_count(&self) -> u32 {
        self.count(LayerKind::Gdn)
            .saturating_add(self.count(LayerKind::Ssm))
            .saturating_add(self.count(LayerKind::ShortConv))
    }

    /// Refuses a stack that mixes recurrent kinds or names one without its
    /// geometry: the session's recurrent records are one size, and each
    /// kind's state is its own size.
    fn validate_recurrent(&self) -> CoreResult<()> {
        let kinds = [LayerKind::Gdn, LayerKind::Ssm, LayerKind::ShortConv]
            .into_iter()
            .filter(|kind| self.count(*kind) > 0)
            .count();
        if kinds > 1 {
            return Err(CoreError::Unsupported {
                what: "more than one recurrent block kind in one stack",
                backend: "engine",
            });
        }
        if self.count(LayerKind::Ssm) > 0 && self.ssm.is_none() {
            return Err(CoreError::ShapeMismatch {
                field: "state-space geometry",
                expected: 1,
                got: 0,
            });
        }
        if self.count(LayerKind::ShortConv) > 0 && self.short_conv.is_none() {
            return Err(CoreError::ShapeMismatch {
                field: "short-convolution geometry",
                expected: 1,
                got: 0,
            });
        }
        self.validate_no_ffn()
    }

    /// Refuses a [`Self::no_ffn`] list that is not ascending slots of this
    /// stack.
    fn validate_no_ffn(&self) -> CoreResult<()> {
        let mut last = None;
        for il in &self.no_ffn {
            if il.get() >= self.n_layers() || last.is_some_and(|prev: u32| prev >= il.get()) {
                return Err(CoreError::InvalidValue {
                    key: "slots without a feed-forward",
                    value: format!(
                        "{:?} is not an ascending list of slots below {}",
                        self.no_ffn,
                        self.n_layers()
                    ),
                });
            }
            last = Some(il.get());
        }
        Ok(())
    }

    /// Whether slot `il` has a feed-forward half.
    #[must_use]
    pub fn has_ffn(&self, il: LayerIdx) -> bool {
        !self.no_ffn.contains(&il)
    }

    /// Rank of slot `il` among the attention slots, or `None` when it holds a
    /// recurrent block or does not exist.
    fn attn_rank(&self, il: LayerIdx) -> Option<u32> {
        let slot = il.get() as usize;
        if self.layers.get(slot).copied()? != LayerKind::Attn {
            return None;
        }
        let rank = self
            .layers
            .iter()
            .take(slot)
            .filter(|kind| **kind == LayerKind::Attn)
            .count();
        u32::try_from(rank).ok()
    }

    /// The attention descriptor of slot `il`, or `None` for a recurrent slot.
    #[must_use]
    pub fn attn_layer(&self, il: LayerIdx) -> Option<&AttnLayer> {
        let rank = self.attn_rank(il)?;
        self.attn.get(rank as usize)
    }

    /// The attention slots that own runs, as `(slot index, &AttnLayer)`, in slot
    /// order.
    ///
    /// This is the cache's allocation order: a slot's position in this list is
    /// the rank its key and value runs were allocated at, which is what
    /// [`HybridShape::kv_owner_rank`] answers with.
    #[must_use]
    pub fn kv_owning_attn(&self) -> Vec<(LayerIdx, &AttnLayer)> {
        let mut owners = Vec::new();
        let mut rank = 0usize;
        for (slot, kind) in self.layers.iter().enumerate() {
            if *kind != LayerKind::Attn {
                continue;
            }
            if let Some(layer) = self.attn.get(rank)
                && layer.owns_kv()
            {
                owners.push((
                    LayerIdx::new(u32::try_from(slot).unwrap_or(u32::MAX)),
                    layer,
                ));
            }
            rank += 1;
        }
        owners
    }

    /// Rank among owning slots of the runs slot `il` reads: its own rank for
    /// [`KvSource::Own`], the owner's for [`KvSource::Shared`].
    ///
    /// `None` for a non-attention slot, an owner that is not an attention slot,
    /// or an owner that is itself `Shared` - none of which is an answer, and all
    /// of which [`HybridShape::validate`] refuses before a pass can meet them.
    #[must_use]
    pub fn kv_owner_rank(&self, il: LayerIdx) -> Option<u32> {
        let layer = self.attn_layer(il)?;
        let owner = match layer.kv {
            KvSource::Own => il,
            KvSource::Shared { owner } => owner,
        };
        // One scan rather than [`HybridShape::kv_owning_attn`] and a search
        // through it: this is asked once per attention slot at load, and the
        // list it would have built is thrown away after one comparison.
        let mut rank = 0u32;
        let mut attn_rank = 0usize;
        for (slot, kind) in self.layers.iter().enumerate() {
            if *kind != LayerKind::Attn {
                continue;
            }
            let candidate = self.attn.get(attn_rank);
            attn_rank += 1;
            if !candidate.is_some_and(AttnLayer::owns_kv) {
                continue;
            }
            if u32::try_from(slot).ok()? == owner.get() {
                return Some(rank);
            }
            rank = rank.saturating_add(1);
        }
        None
    }

    /// Largest `head_dim` over attention slots (0 with none).
    #[must_use]
    pub fn max_head_dim(&self) -> u32 {
        self.attn
            .iter()
            .map(|layer| layer.head_dim)
            .max()
            .unwrap_or(0)
    }

    /// Checks the facts the rest of the crate then reads without checking.
    ///
    /// Six of them, and each is a wrong answer rather than a crash if it goes
    /// unchecked: an attention list that does not cover the attention slots
    /// silently gives a slot another slot's widths; a shared owner that is not
    /// an owning attention slot at a lower index names runs that do not exist or
    /// are not filled yet; a sharer whose key/value widths are not its owner's
    /// reads those runs at the wrong stride; a rotary width wider than the
    /// ladder it is computed over rotates dimensions no frequency was derived
    /// for; a router that reaches more experts than the layer holds sends every
    /// token past the end of the expert list; and a router that reaches more
    /// than [`ROUTER_CHOSEN_SLOTS`] overruns the two arrays both backends'
    /// routers pick into.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] when `attn.len()` is not the attention slot
    /// count or the router's two counts cannot both be true, and
    /// [`CoreError::InvalidValue`] for a bad owner, a rotary width the head
    /// cannot carry, or a router wider than the chosen-expert arrays.
    pub fn validate(&self) -> CoreResult<()> {
        // One recurrent kind a stack: the session's recurrent records are one
        // size, and each kind's state is its own size.
        self.validate_recurrent()?;
        // Before the attention walk, because it is the cheapest of the six and
        // the one whose failure is widest: a token routed past the expert list
        // reads a matrix that is not an expert at all.
        if let Some(moe) = self.moe
            && (moe.used == 0 || moe.used > moe.experts)
        {
            return Err(CoreError::ShapeMismatch {
                field: "experts per token",
                expected: u64::from(moe.experts),
                got: u64::from(moe.used),
            });
        }
        // Second, and separate, because a router can be perfectly coherent
        // against its own expert list and still be wider than the arrays the
        // routers pick into. The overrun is not a fault: it writes into
        // adjacent threadgroup or shared storage, and what comes back is a
        // fluent answer routed through the wrong experts, with every count
        // downstream still consistent. One check here covers both backends and
        // both families, because `HybridStack::new` is the only door.
        if let Some(moe) = self.moe
            && moe.used > ROUTER_CHOSEN_SLOTS
        {
            return Err(CoreError::InvalidValue {
                key: "experts per token",
                value: format!(
                    "{used} exceeds the {ROUTER_CHOSEN_SLOTS} slots both routers' chosen-weight \
                     and chosen-id arrays hold, at kernels/cuda/q4e/moe.cu:44-47 and \
                     kernels/metal/qwen35moe/moe_route.metal:161-162; neither bounds \
                     its writes, so a wider router silently answers from the wrong experts",
                    used = moe.used
                ),
            });
        }
        let slots = u64::from(self.count(LayerKind::Attn));
        let listed = u64::try_from(self.attn.len()).unwrap_or(u64::MAX);
        if listed != slots {
            return Err(CoreError::ShapeMismatch {
                field: "attention layers",
                expected: slots,
                got: listed,
            });
        }
        for (slot, kind) in self.layers.iter().enumerate() {
            if *kind != LayerKind::Attn {
                continue;
            }
            let il = LayerIdx::new(u32::try_from(slot).unwrap_or(u32::MAX));
            let Some(layer) = self.attn_layer(il) else {
                continue;
            };
            if let KvSource::Shared { owner } = layer.kv {
                let source = self
                    .attn_layer(owner)
                    .filter(|source| source.owns_kv() && owner.get() < il.get())
                    .ok_or_else(|| CoreError::InvalidValue {
                        key: "kv source",
                        value: format!(
                            "slot {slot} reads slot {}'s runs, which is not an attention slot \
                             that owns runs at a lower index",
                            owner.get()
                        ),
                    })?;
                // A run is sized from the slot that owns it, and the sharer
                // indexes it with its own widths: two slots that disagree about
                // either would read a row at the wrong stride, inside a buffer
                // whose length is right. Nothing downstream can notice that.
                if source.kv_heads != layer.kv_heads || source.head_dim != layer.head_dim {
                    return Err(CoreError::InvalidValue {
                        key: "kv source",
                        value: format!(
                            "slot {slot} reads {}x{} rows out of slot {}'s {}x{} runs",
                            layer.kv_heads,
                            layer.head_dim,
                            owner.get(),
                            source.kv_heads,
                            source.head_dim
                        ),
                    });
                }
            }
            if layer.n_rot > layer.freq_dims || layer.freq_dims > layer.head_dim {
                return Err(CoreError::InvalidValue {
                    key: "rotary width",
                    value: format!(
                        "slot {slot} rotates {} of a {}-wide ladder over a {}-wide head",
                        layer.n_rot, layer.freq_dims, layer.head_dim
                    ),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Activation, AttnLayer, AttnScale, GdnShape, HybridShape, LayerKind, MediaPositions,
        MoeShape, NormScheme,
    };
    use shiso_core::CoreError;
    use shiso_core::ids::LayerIdx;

    /// A stack of nothing but attention slots, one per entry of `attn`, with the
    /// family switches a Gemma-like checkpoint carries.
    fn gemma_like(attn: Vec<AttnLayer>) -> HybridShape {
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
            embed_scale: Some(45.254_834),
            final_softcap: Some(30.0),
            layer_scalar: true,
            ple: None,
            media_positions: MediaPositions::Linear,
            v_norm: true,
            moe: None,
            ssm: None,
            short_conv: None,
            no_ffn: Vec::new(),
        }
    }

    #[test]
    fn owner_rank_follows_one_hop_of_sharing() {
        // 6 attention slots: 0..=3 own, 4 shares 2, 5 shares 3.
        let mut attn = vec![AttnLayer::new(8, 2, 256, 256, 256, 1e4); 6];
        attn[3] = AttnLayer::new(8, 2, 512, 128, 512, 1e6);
        attn[4] = attn[2].shared_from(LayerIdx::new(2));
        attn[5] = attn[3].shared_from(LayerIdx::new(3));
        let shape = gemma_like(attn);
        shape.validate().expect("valid");
        assert_eq!(shape.kv_owning_attn().len(), 4);
        assert_eq!(shape.kv_owner_rank(LayerIdx::new(4)), Some(2));
        assert_eq!(shape.kv_owner_rank(LayerIdx::new(5)), Some(3));
        assert_eq!(shape.kv_owner_rank(LayerIdx::new(1)), Some(1));
        assert_eq!(shape.max_head_dim(), 512);
    }

    #[test]
    fn a_shared_owner_that_is_itself_shared_is_refused() {
        let mut attn = vec![AttnLayer::new(8, 2, 256, 256, 256, 1e4); 6];
        attn[4] = attn[2].shared_from(LayerIdx::new(2));
        attn[5] = attn[4].shared_from(LayerIdx::new(4));
        let shape = gemma_like(attn);
        shape
            .validate()
            .expect_err("a chain of shares has no run to read");
        // The rank is the answer the pass would have taken, and there is none.
        assert_eq!(shape.kv_owner_rank(LayerIdx::new(5)), None);
    }

    #[test]
    fn a_sharer_whose_key_value_widths_are_not_its_owner_s_is_refused() {
        // The runs are sized from the owner, and the sharer indexes them with
        // its own widths. Two slots that disagree read rows at the wrong stride
        // out of a buffer whose length is right, which nothing downstream can
        // notice.
        let mut attn = vec![AttnLayer::new(8, 2, 256, 256, 256, 1e4); 3];
        attn[2] = AttnLayer::new(8, 4, 256, 256, 256, 1e4).shared_from(LayerIdx::new(0));
        gemma_like(attn)
            .validate()
            .expect_err("four key/value heads cannot read two heads' runs");

        let mut attn = vec![AttnLayer::new(8, 2, 256, 256, 256, 1e4); 3];
        attn[2] = AttnLayer::new(8, 2, 512, 256, 512, 1e4).shared_from(LayerIdx::new(0));
        gemma_like(attn)
            .validate()
            .expect_err("a 512-wide head cannot read 256-wide runs");

        // The same slot with its owner's widths is what a sharer looks like.
        let mut attn = vec![AttnLayer::new(8, 2, 256, 256, 256, 1e4); 3];
        attn[2] = attn[0].shared_from(LayerIdx::new(0)).with_window(512);
        gemma_like(attn)
            .validate()
            .expect("a window is not a cache width");
    }

    #[test]
    fn the_attention_list_must_match_the_slot_count() {
        let attn = vec![AttnLayer::new(8, 2, 256, 256, 256, 1e4); 5];
        let mut shape = gemma_like(attn);
        shape.layers = vec![LayerKind::Attn; 6];
        shape
            .validate()
            .expect_err("five descriptors do not cover six attention slots");
    }

    #[test]
    fn a_rotary_width_wider_than_its_ladder_is_refused() {
        let attn = vec![AttnLayer::new(8, 2, 256, 300, 256, 1e4); 1];
        gemma_like(attn)
            .validate()
            .expect_err("a slot cannot rotate more dimensions than it has frequencies");
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "1.0 is the scale exactly, and a tolerance would hide a scale that is not"
    )]
    fn scale_values() {
        assert_eq!(AttnScale::One.value(512), 1.0);
        assert!((AttnScale::InvSqrtHeadDim.value(256) - 0.0625).abs() < 1e-7);
    }

    #[test]
    fn a_router_that_reaches_more_experts_than_the_layer_holds_is_refused() {
        let attn = vec![AttnLayer::new(8, 2, 256, 256, 256, 1e4); 1];
        let dense = gemma_like(attn);
        // A dense stack states no router, and that is not a router of zero
        // experts: nothing in the pass reads the field at all.
        dense.validate().expect("a dense feed-forward");

        let routed = HybridShape {
            moe: Some(MoeShape::new(256, 8, 1, 512)),
            ..dense.clone()
        };
        routed
            .validate()
            .expect("eight of two hundred and fifty-six");

        let past_the_end = HybridShape {
            moe: Some(MoeShape::new(8, 10, 1, 512)),
            ..dense.clone()
        };
        past_the_end
            .validate()
            .expect_err("ten experts cannot be picked out of eight");

        let none_at_all = HybridShape {
            moe: Some(MoeShape::new(256, 0, 1, 512)),
            ..dense
        };
        none_at_all
            .validate()
            .expect_err("a token routed to no expert has no feed-forward");
    }

    #[test]
    fn a_router_wider_than_the_chosen_expert_arrays_is_refused_by_both_file_names() {
        let attn = vec![AttnLayer::new(8, 2, 256, 256, 256, 1e4); 1];
        let dense = gemma_like(attn);

        // Sixteen is the width both arrays hold, so sixteen still fits.
        let exactly_full = HybridShape {
            moe: Some(MoeShape::new(256, 16, 1, 512)),
            ..dense.clone()
        };
        exactly_full
            .validate()
            .expect("sixteen is what the arrays hold");

        // Seventeen is coherent against a 256-expert list -- the check above it
        // passes -- and overruns the arrays anyway. That is the whole reason
        // this is a second check rather than a wider first one.
        let past_the_arrays = HybridShape {
            moe: Some(MoeShape::new(256, 17, 1, 512)),
            ..dense
        };
        match past_the_arrays
            .validate()
            .expect_err("seventeen picks do not fit sixteen slots")
        {
            CoreError::InvalidValue { key, value } => {
                assert_eq!(key, "experts per token");
                // Both arrays, by file and line, because a reader who hits this
                // has to find the two declarations and there is nothing in
                // either file that will lead them to the other.
                assert!(value.contains("kernels/cuda/q4e/moe.cu:44-47"), "{value}");
                assert!(
                    value.contains("kernels/metal/qwen35moe/moe_route.metal:161-162"),
                    "{value}"
                );
                assert!(value.contains("17"), "{value}");
            }
            other => panic!("expected InvalidValue, got {other:?}"),
        }
    }

    #[test]
    fn a_routed_column_is_one_token_and_one_expert_slot() {
        let moe = MoeShape::new(256, 8, 1, 512);
        assert_eq!(moe.slot_rows(1), 8);
        assert_eq!(moe.slot_rows(512), 4_096);
    }

    #[test]
    fn a_window_is_a_count_or_nothing_at_all() {
        let full = AttnLayer::new(8, 2, 256, 256, 256, 1e4);
        assert_eq!(full.window, None);
        assert_eq!(
            full.with_window(512).window.map(std::num::NonZeroU32::get),
            Some(512)
        );
        assert_eq!(full.with_window(0).window, None);
    }
}
