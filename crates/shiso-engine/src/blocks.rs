//! The seam between the stack and the kernels one checkpoint family launches.
//!
//! Everything a block reads and writes is the engine's: the weights, the derived
//! geometry, the scratch rows, the session's cache and recurrent state. Which
//! kernels those bytes are fed to is not. Two families with the same block
//! structure normalize, rotate and reduce differently enough that a shared launch
//! site would be a switch on the checkpoint in the middle of the forward pass, so
//! the launches leave the engine through [`BlockKernels`] and the family that
//! owns the kernels owns the code that dispatches them.
//!
//! # Why the views arrive in a struct
//!
//! A block takes six or seven views and one of them is new plumbing. Passing
//! them positionally put nine arguments on a function whose reader has to keep
//! the order in their head; [`AttnIo`] and [`GdnIo`] name them once, and adding
//! a view is then a field a family can ignore rather than a signature every
//! family has to be edited for.

use shiso_core::backend::Backend;
use shiso_core::buffer::DeviceView;
use shiso_core::error::CoreResult;

use crate::attn::{Attn, AttnCache, AttnGeometry, AttnScratch, Positions};
use crate::gdn::{Gdn, GdnGeometry, GdnScratch};

/// The views one attention block reads and writes for one chunk.
#[derive(Debug, Clone, Copy)]
pub struct AttnIo<'a> {
    /// Normalized residual rows, `[n_tok, hidden]` f32.
    pub input: DeviceView<'a>,
    /// The chunk's positions, linear or three-stream.
    pub pos: Positions<'a>,
    /// Last key position each query may read, one `int32` per token.
    ///
    /// A text token's own position. A placeholder token's bound is the last
    /// position of its media item, so attention within the item is
    /// bidirectional. qwen35's kernels ignore it.
    pub bound: DeviceView<'a>,
    /// The block's contribution to the stream, `[n_tok, hidden]` f32.
    pub block_out: DeviceView<'a>,
}

/// Where a recurrent block leaves the state it stood on after each row.
///
/// A speculative pass runs `1 + k` rows and may keep only a prefix of them. The
/// rows it discards have already been folded into the recurrent state, which is
/// the one piece of a session a later pass cannot recompute and cannot ignore:
/// the cache rows past the frontier are overwritten before anything reads them,
/// but a state matrix that walked three tokens too far is what every token
/// after it is then conditioned on. Nothing fails; the answers just stop being
/// the model's.
///
/// So a pass that may be rolled back records the state as it goes, and a
/// rejection copies one record back. Both halves are recorded, because the
/// convolution window and the state matrix are one position between them and a
/// rewind that restored one of them would stand the block on half a token.
///
/// # Why this is a trail and not a kernel argument
///
/// The reference implementation of this family's sibling has the recurrence
/// kernel write a checkpoint per token. That needs a parameter on four CUDA
/// entry points and their four Metal twins, and this tree cannot run the Metal
/// half. What it does instead is ask the family to run the two *sequential*
/// stages - the convolution and the recurrence - a row at a time and copy the
/// state out between them, which is bit-for-bit the same arithmetic: both
/// kernels carry their state in registers across the row loop and write it back
/// at the end, so splitting the loop only round-trips numbers through the f32
/// buffer they were already going to be written to.
///
/// It costs the extra launches and one extra read-write of the state per row.
/// Measured against the alternative it costs nothing that was not already going
/// to be paid: the batched stages either side of it - every projection, which
/// is where a decode step's bytes actually go - stay batched.
#[derive(Debug, Clone, Copy)]
pub struct StateTrail<'a> {
    /// `rows` records of the convolution window, in row order.
    pub conv: DeviceView<'a>,
    /// `rows` records of the state matrix, in row order.
    pub state: DeviceView<'a>,
    /// Rows the trail has room for, which is the depth the session was opened
    /// at.
    ///
    /// A pass wider than this is **refused**, with
    /// [`shiso_core::CoreError::OutOfBounds`] from the slice that would have
    /// written past the slab. That is deliberate and it is the safe direction:
    /// silently leaving the overflowing rows unrecorded would let a rollback
    /// name a row whose record was never written, which is the failure this
    /// whole mechanism exists to prevent. What keeps a pass inside the trail is
    /// [`crate::mtp::draft_budget`], which clamps a step's drafts to the depth
    /// [`crate::mtp::SpecDepth`] sized both the trail and the chain by.
    pub rows: u32,
}

/// Whether a pass records the per-row state a rejection would wind back to.
///
/// An enum and not a `bool`, and that is the whole point of the type. It is
/// passed to [`crate::layers::forward`], which already carries eleven
/// arguments and a `too_many_arguments` waiver, and the failure a `bool` there
/// allows is silent: `forward(.., false)` compiles, speculation still runs,
/// the chain drafts, the target verifies, and [`crate::mtp::Rollback`] then
/// copies back a slab **the allocator last wrote**. The session stands on a
/// recurrent state belonging to no position and every token after it is
/// conditioned on that. No launch fails and no test fails, because the
/// byte-identity test passes trivially when speculation is off.
///
/// Naming the two cases does not make that impossible, but it makes it
/// unwriteable by accident: `Trail::None` at a verify site is a sentence a
/// reviewer can read, where `false` is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trail {
    /// Leave the state after each row in the session's trail. Only a verify
    /// pass asks for this, because only a verify pass can be wound back.
    Record,
    /// Record nothing, which is every prefill and every plain decode step.
    None,
}

impl Trail {
    /// Whether this pass records.
    #[must_use]
    pub const fn records(self) -> bool {
        matches!(self, Self::Record)
    }
}

/// One recurrent slot's whole memory of one session's prefix.
///
/// The two halves travel together because they are one position between them:
/// a block handed one session's window and another's state matrix would be
/// standing on half a token, and every launch would still succeed.
#[derive(Debug, Clone, Copy)]
pub struct GdnState<'a> {
    /// The convolution window, rewritten in place.
    pub conv: DeviceView<'a>,
    /// The recurrent state matrix, rewritten in place.
    pub state: DeviceView<'a>,
}

/// Which session's recurrent state each row of a pass folds into.
///
/// An enum and not a slice-of-one, because the two cases are two different
/// launch shapes rather than two sizes of the same one. A chunk walks its rows
/// **inside** one launch - the convolution and the recurrence each carry their
/// state in registers across a row loop - which is what makes a 512-row prefill
/// two launches instead of a thousand. A batch cannot: its rows are different
/// conversations, and no launch can fold row one into row zero's state matrix.
///
/// So a family reading this cannot serve a batch by pretending it is a chunk.
/// The variant says which it is, and [`GdnStates::PerRow`] is the one that
/// costs a launch per row for the two sequential stages. That cost is bounded
/// by [`crate::MAX_DECODE_BATCH`] and is the part of a decode step batching
/// does not amortize: a recurrent state is per session and reading four of them
/// costs four reads however they are issued.
#[derive(Debug, Clone, Copy)]
pub enum GdnStates<'a> {
    /// Every row of the pass is one session's, at consecutive positions: a
    /// prefill chunk, a single-session decode step, or a verify pass.
    Chunk(GdnState<'a>),
    /// Row `i` is `rows[i]`'s own single token - a batched decode step.
    ///
    /// The slice is exactly as long as the pass is wide; a family that finds
    /// otherwise refuses rather than running the rows it happens to have.
    PerRow(&'a [GdnState<'a>]),
}

/// The views one recurrent block reads and writes for one chunk.
#[derive(Debug, Clone, Copy)]
pub struct GdnIo<'a> {
    /// Normalized residual rows, `[n_tok, hidden]` f32.
    pub input: DeviceView<'a>,
    /// The state each row folds into, rewritten in place.
    pub states: GdnStates<'a>,
    /// The block's contribution to the stream, `[n_tok, hidden]` f32.
    pub block_out: DeviceView<'a>,
    /// Where to leave the state after each row, for a pass a rejection may have
    /// to undo.
    ///
    /// `None` on every pass that cannot be rolled back, which is every prefill
    /// and every plain decode step. A family that does not implement
    /// speculation may ignore it; what it must not do is accept a trail and
    /// leave it unwritten, because a rollback would then restore whatever the
    /// allocator last left there.
    pub trail: Option<StateTrail<'a>>,
}

/// The launches a checkpoint family supplies for its two block kinds.
///
/// One value per stack, built at load and read at every block. It carries
/// whatever the family measured or chose once - the recurrent kernel's lane
/// split, for instance - which is why it is a value rather than a pair of free
/// functions.
pub trait BlockKernels: Send + Sync + std::fmt::Debug {
    /// Refuses a shape these kernels cannot run, before a stack is built over
    /// it.
    ///
    /// Not the same question [`crate::shape::HybridShape::validate`] asks. That
    /// one is about a shape that contradicts itself and is wrong for every
    /// family; this one is about a shape that is perfectly coherent and happens
    /// to be outside what THIS set of kernels was compiled for - a head wider
    /// than the staging its threadgroups allocate, say. A kernel cannot refuse:
    /// it would read past its own threadgroup memory and produce numbers. So the
    /// host refuses instead, at load, rather than at the first forward pass of a
    /// model somebody has already waited for.
    ///
    /// The default accepts everything, which is the right answer for a family
    /// whose kernels were written for one checkpoint and are dispatched over its
    /// own widths.
    ///
    /// # Errors
    ///
    /// Whatever the family reports for a shape it cannot run, naming the slot.
    fn validate(&self, _shape: &crate::shape::HybridShape) -> CoreResult<()> {
        Ok(())
    }

    /// Runs one attention block over `n_tok` rows of `io.input`.
    ///
    /// # Errors
    ///
    /// Whatever the family reports for a width or format it cannot carry, and
    /// [`shiso_core::CoreError::Backend`] naming the kernel when a launch fails.
    #[expect(
        clippy::too_many_arguments,
        reason = "the weights, the geometry, the scratch and the session's cache are four \
                  different owners' contributions to one launch, and the views that are \
                  the caller's are already bundled in `AttnIo`"
    )]
    fn attention(
        &self,
        backend: &dyn Backend,
        attn: &Attn,
        geometry: &AttnGeometry,
        scratch: &AttnScratch,
        io: AttnIo<'_>,
        cache: &AttnCache<'_>,
        n_tok: u32,
    ) -> CoreResult<()>;

    /// Runs one recurrent block over `n_tok` rows of `io.input`.
    ///
    /// # Errors
    ///
    /// Whatever the family reports for a width or format it cannot carry, and
    /// [`shiso_core::CoreError::Backend`] naming the kernel when a launch fails.
    fn recurrent(
        &self,
        backend: &dyn Backend,
        gdn: &Gdn,
        geometry: &GdnGeometry,
        scratch: &GdnScratch,
        io: GdnIo<'_>,
        n_tok: u32,
    ) -> CoreResult<()>;
}
