//! The hyper-connection streams: how a block reads the residual and how its
//! output gets back in.
//!
//! The residual here is not one vector per token but [`SparseShape::streams`]
//! of them, laid out stream-outer so one token is a contiguous run of
//! `streams * hidden` floats. Every block reads a single mixed vector and
//! writes a single contribution, so something has to fold the streams down on
//! the way in and fan the contribution back out on the way out. That is what a
//! *mix* and a *combine* are, and between them they are the only launches every
//! single layer performs whatever block it holds.
//!
//! # The mix and the combine are one chain, not two operations
//!
//! The combine that closes a block does two things in one pass over the
//! residual: it adds the block's contribution back into the streams, and it
//! normalizes the result with the *next* block's norm weight. So the next mix
//! usually has no norm to do - its input is already normed - and the chain
//! breaks in exactly three places: the first island of a pass, because nothing
//! preceded it; an island whose n-gram block reads and rewrites the residual
//! between the update and the mix; and the last layer, whose successor is
//! either the collapsing mix or nothing at all.
//!
//! [`MixInput`] is that chain expressed as a type, so a call site cannot leave
//! it ambiguous.

use shiso_core::error::{CoreError, CoreResult};

use crate::sparse::bind::{Args, LayerRef, Operand, Role, WeightRef};
use crate::sparse::kernels::{self, Elem, Entry, MOD_HC, TemplateArgs};
use crate::sparse::matmul::{self, Bound, Device};
use crate::sparse::plan::Plan;
use crate::sparse::scratch::Buf;
use crate::sparse::shape::SparseShape;
use crate::sparse::weights::MixDesc;

/// Whether a mix has to normalize its input, or was handed it already normed.
///
/// An enum rather than a `bool`, because the two readings of `false` - "do not
/// normalize" and "there is nothing to normalize yet" - are the difference
/// between a correct pass and one that runs its first block on an unnormalized
/// residual.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixInput {
    /// The previous combine left the normed residual behind, so this mix reads
    /// it and launches no norm of its own.
    AlreadyNormed,
    /// Nothing normed this residual, so the mix normalizes it first.
    Raw,
}

/// Whether the normalization chain is whole across one seam.
///
/// The decision, without the weight: [`chain`] answers it from the layer
/// structure alone, and the pass turns a whole link into the weight the
/// combine binds. Two types rather than one because the two are decided in
/// different places and by different things.
///
/// [`chain`]: crate::sparse::island::chain
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainLink {
    /// The combine leaves the next block's normed input behind.
    Chained,
    /// The chain breaks here.
    Broken,
}

/// What the combine that closes a block leaves behind for whatever runs next.
///
/// A role rather than a flag: the combine binds the next mix's norm weight, so
/// "chained" has to say *which* weight, and "broken" is the null pointer the
/// launcher passes in its place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextNorm {
    /// The weight of the next block's mix, which this combine applies in the
    /// same pass that updates the residual.
    Chained(WeightRef),
    /// Nothing: the chain breaks here, and whatever runs next normalizes for
    /// itself.
    Broken,
}

impl NextNorm {
    /// The operand the launcher passes for the next norm.
    #[must_use]
    pub const fn operand(self) -> Operand {
        match self {
            Self::Chained(weight) => Operand::Weight(weight),
            Self::Broken => Operand::Null,
        }
    }
}

/// Which four tensors one stream mix reads.
///
/// Deliberately NOT `#[non_exhaustive]`: a caller builds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MixRoles {
    /// The layer that holds them.
    pub layer: LayerRef,
    /// The per-stream norm.
    pub norm: Role,
    /// The contraction.
    pub down: Role,
    /// The expansion.
    pub up: Role,
    /// The injection logits, absent on the collapsing mix.
    pub inject: Option<Role>,
}

impl MixRoles {
    /// The token mixer's stream mix, for one layer.
    #[must_use]
    pub const fn token_mixer(layer: LayerRef) -> Self {
        Self {
            layer,
            norm: Role::AttnMixNorm,
            down: Role::AttnMixDown,
            up: Role::AttnMixUp,
            inject: Some(Role::AttnMixInject),
        }
    }

    /// The routed feed-forward's stream mix, for one layer.
    #[must_use]
    pub const fn experts(layer: LayerRef) -> Self {
        Self {
            layer,
            norm: Role::FfnMixNorm,
            down: Role::FfnMixDown,
            up: Role::FfnMixUp,
            inject: Some(Role::FfnMixInject),
        }
    }

    /// The mix that collapses the streams at the end of the draft head's pass.
    #[must_use]
    pub const fn draft_output() -> Self {
        Self {
            layer: LayerRef::draft(),
            norm: Role::OutputMixNorm,
            down: Role::OutputMixDown,
            up: Role::OutputMixUp,
            inject: None,
        }
    }

    /// The mix that collapses the streams at the end of a pass.
    #[must_use]
    pub const fn output() -> Self {
        Self {
            layer: LayerRef::target(0),
            norm: Role::OutputMixNorm,
            down: Role::OutputMixDown,
            up: Role::OutputMixUp,
            inject: None,
        }
    }

    /// One of these roles as a reference into the layer that holds it.
    #[must_use]
    pub const fn at(&self, role: Role) -> WeightRef {
        WeightRef {
            layer: self.layer,
            role,
            part: crate::sparse::weights::WeightPart::Whole,
        }
    }
}

/// The broadcast that opens a pass: the embedding tiled across every stream.
#[must_use]
pub fn init(shape: &SparseShape, n_tok: u32) -> Plan {
    let mut plan = Plan::new();
    plan.launch(
        Entry::plain(MOD_HC, "q4e_hc_init_kernel"),
        kernels::hc_init(shape.hidden, shape.streams, n_tok),
        // `res, embed, n_embd, n_hc, n_tok`
        Args::of(&[
            Operand::Scratch(Buf::Residual),
            Operand::Scratch(Buf::Embed),
            Operand::U32(shape.hidden),
            Operand::U32(shape.streams),
            Operand::U32(n_tok),
        ]),
    );
    plan
}

/// The same broadcast with a per-stream term added, which is how a draft head
/// starts from the target's hidden state rather than from an embedding alone.
///
/// `embed` is the per-token term and `hidden` the per-stream one: the draft
/// head's two projections, of the token it was given and of the target's last
/// hidden state, each already at the width the residual wants. Both are
/// operands rather than fixed buffers because neither is where a target pass
/// leaves its embedding - this launch opens a pass that starts from two
/// projections rather than from a gather.
#[must_use]
pub fn init_add(shape: &SparseShape, embed: Operand, hidden: Operand, n_tok: u32) -> Plan {
    let mut plan = Plan::new();
    plan.launch(
        Entry::plain(MOD_HC, "q4e_hc_init_add_kernel"),
        kernels::hc_init(shape.hidden, shape.streams, n_tok),
        // `res, embed, h, n_embd, n_hc, n_tok`
        Args::of(&[
            Operand::Scratch(Buf::Residual),
            embed,
            hidden,
            Operand::U32(shape.hidden),
            Operand::U32(shape.streams),
            Operand::U32(n_tok),
        ]),
    );
    plan
}

/// The stream norm, which three call sites share: a mix whose chain broke, and
/// the n-gram block's three normalizations.
///
/// `dst` and `src` are separate because one of those call sites norms a buffer
/// into itself and another norms the residual into a buffer of its own.
#[must_use]
pub fn norm(
    shape: &SparseShape,
    weight: WeightRef,
    src: Operand,
    dst: Operand,
    n_tok: u32,
) -> Plan {
    norm_of(
        shape.hidden,
        shape.streams,
        shape.rms_eps,
        weight,
        src,
        dst,
        n_tok,
    )
}

/// The same normalization at a width and a group count of the caller's.
///
/// The stream norm above is this with the two read off the shape. The draft
/// head needs neither: it normalizes a token embedding as **one** group of the
/// residual width, and a whole multi-stream vector as **one** group of the
/// multi-stream width - a single RMS over all four streams together, which is
/// what the weight it carries is wide enough for. Reading the group count off
/// the shape there would normalize each stream separately against a quarter of
/// the weight, which is four normalizations of the right width and the wrong
/// numbers.
#[must_use]
pub fn norm_of(
    width: u32,
    groups: u32,
    eps: f32,
    weight: WeightRef,
    src: Operand,
    dst: Operand,
    n_tok: u32,
) -> Plan {
    let mut plan = Plan::new();
    plan.launch(
        Entry::template(MOD_HC, "q4e_hc_norm_kernel", TemplateArgs::Elem(Elem::F32)),
        kernels::hc_norm(groups, n_tok),
        // `out, res, w, n_embd, n_hc, eps`
        Args::of(&[
            dst,
            src,
            Operand::Weight(weight),
            Operand::U32(width),
            Operand::U32(groups),
            Operand::f32(eps),
        ]),
    );
    plan
}

/// One stream mix.
///
/// The order is the whole of it: normalize (when the chain broke), contract to
/// the low-rank vector, scale by the stream count and activate, expand back,
/// collapse the gated streams into the block's input, and - for the two mixes
/// that have one - project the per-stream logits the matching combine injects.
///
/// The scale sits *inside* the activation, before the expansion. Moving it
/// outside changes the gate materially rather than by a rounding step, which is
/// why it is a launch of its own rather than a factor folded into a weight.
///
/// # Errors
///
/// Whatever [`matmul::project`] reports for a weight this build cannot
/// project.
pub fn mix(
    shape: &SparseShape,
    desc: &MixDesc,
    roles: MixRoles,
    input: MixInput,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let mut plan = Plan::new();
    if input == MixInput::Raw {
        plan.append(norm(
            shape,
            roles.at(roles.norm),
            Operand::Scratch(Buf::Residual),
            Operand::Scratch(Buf::Normed),
            n_tok,
        ));
    }
    matmul::project(
        &mut plan,
        desc.down,
        Bound::new(
            roles.at(roles.down),
            Operand::Scratch(Buf::Normed),
            Operand::Scratch(Buf::LowRank),
        ),
        n_tok,
        device,
    )?;
    plan.launch(
        Entry::plain(MOD_HC, "q4e_scale_silu_kernel"),
        kernels::elementwise(u64::from(n_tok) * u64::from(shape.stream_rank)),
        // `x, inv_hc, n` - in place, and the reciprocal is the stream count's.
        Args::of(&[
            Operand::Scratch(Buf::LowRank),
            Operand::f32(stream_reciprocal(shape.streams)),
            Operand::U64(u64::from(n_tok) * u64::from(shape.stream_rank)),
        ]),
    );
    matmul::project(
        &mut plan,
        desc.up,
        Bound::new(
            roles.at(roles.up),
            Operand::Scratch(Buf::LowRank),
            Operand::Scratch(Buf::Expanded),
        ),
        n_tok,
        device,
    )?;
    plan.launch(
        Entry::template(
            MOD_HC,
            "q4e_hc_collapse_kernel",
            TemplateArgs::Elem(Elem::F32),
        ),
        kernels::hc_collapse(n_tok),
        // `out, xn, up, n_embd, n_hc`
        Args::of(&[
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::Normed),
            Operand::Scratch(Buf::Expanded),
            Operand::U32(shape.hidden),
            Operand::U32(shape.streams),
        ]),
    );
    if let (Some(inject), Some(role)) = (desc.inject, roles.inject) {
        matmul::project(
            &mut plan,
            inject,
            Bound::new(
                roles.at(role),
                Operand::Scratch(Buf::Normed),
                Operand::Scratch(Buf::Inject),
            ),
            n_tok,
            device,
        )?;
    }
    Ok(plan)
}

/// The fused residual update and next-block norm that closes a block.
///
/// One launch, not two. Running an add and then a norm instead would round the
/// residual twice - once when the sum is stored and again when it is read back -
/// so the residual the next layer sees would not be the one this layer
/// intended.
///
/// `next` is the *caller's* to decide: it changes which weight the launch binds
/// and nothing about the grid, and where the chain breaks it is the null
/// pointer that turns this launch into the residual update alone.
///
/// # Errors
///
/// [`CoreError::Unsupported`] when the stream width is wider than the fused
/// kernel's fixed block can cover, which its launcher refuses outright rather
/// than leaving a tail of the residual unwritten.
pub fn combine(shape: &SparseShape, next: NextNorm, n_tok: u32) -> CoreResult<Plan> {
    if !kernels::hc_combine_fits(shape.hidden) {
        return Err(CoreError::Unsupported {
            what: "a residual stream wider than the fused combine's block can cover",
            backend: "engine",
        });
    }
    let mut plan = Plan::new();
    plan.launch(
        Entry::template(
            MOD_HC,
            "q4e_hc_combine_norm_kernel",
            TemplateArgs::Elem(Elem::F32),
        ),
        kernels::hc_combine(shape.streams, n_tok),
        // `res, xn, block_out, inject, w, n_embd, n_hc, eps`
        Args::of(&[
            Operand::Scratch(Buf::Residual),
            Operand::Scratch(Buf::Normed),
            Operand::Scratch(Buf::BlockOut),
            Operand::Scratch(Buf::Inject),
            next.operand(),
            Operand::U32(shape.hidden),
            Operand::U32(shape.streams),
            Operand::f32(shape.rms_eps),
        ]),
    );
    Ok(plan)
}

/// The mix that ends a pass.
///
/// It collapses the streams for the vocabulary projection and stands in for an
/// output norm; there is no separate output-norm tensor, and nothing to inject
/// back into a residual - which is why its descriptor has no injection weight
/// and this is the one mix whose input is never already normed.
///
/// # Errors
///
/// Whatever [`mix`] reports.
pub fn collapse(
    shape: &SparseShape,
    desc: &MixDesc,
    roles: MixRoles,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    if desc.inject.is_some() {
        return Err(CoreError::InvalidValue {
            key: "output mix",
            value: "the collapsing mix has no residual to inject into".to_owned(),
        });
    }
    mix(shape, desc, roles, MixInput::Raw, n_tok, device)
}

/// The reciprocal of the stream count, which the activation scales by.
///
/// Narrowed through a sixteen-bit conversion rather than cast: the stream count
/// is a handful, the conversion is exact there, and a count that did not fit
/// would be a geometry error rather than a silently rounded scale.
fn stream_reciprocal(streams: u32) -> f32 {
    let streams = f32::from(u16::try_from(streams).unwrap_or(u16::MAX));
    if streams == 0.0 { 0.0 } else { 1.0 / streams }
}
