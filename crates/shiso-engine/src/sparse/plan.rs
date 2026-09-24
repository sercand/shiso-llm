//! The ordered work one pass through the stack is made of.
//!
//! Every block family in this module answers with a [`Plan`] rather than
//! issuing launches itself, and the reason is what this port can and cannot
//! check. The numbers a kernel produces need the kernel to run; the *order* it
//! runs in, the grid it runs on, which of six projection tiers a weight takes
//! and where the residual chain breaks do not - and those are the decisions
//! that a port gets wrong. Making them a value makes every one of them an
//! assertion about a returned plan instead of an argument about a trace nobody
//! without the hardware can produce.
//!
//! # Why the steps are an enum rather than a list of kernels
//!
//! Not all of the work is a kernel. A wide projection goes to the device's BLAS
//! tier, which has no entry point at all; a trellis projection launches
//! cooperatively, which is a different launch call with a grid that must fit the
//! device rather than merely describe the work; and the checkpoint and cache
//! paths move bytes. Four variants say which of those a step is, so a reader of
//! a plan cannot mistake a cooperative grid for an ordinary one.

use shiso_core::backend::{GemmDType, GemmOp, GemmReduction};
use shiso_core::error::{CoreError, CoreResult};

use crate::sparse::bind::{Args, Operand};
use crate::sparse::exl3::{GemmShape, Staging};
use crate::sparse::kernels::{Entry, Launch};

/// Which cooperative trellis launch a step is.
///
/// Named rather than inferred from the grid, because the three have different
/// argument shapes on the device side and only their block counts look alike.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TrellisOp {
    /// One weight, one output.
    Dense,
    /// A gate and its up projection behind one launch.
    Pair,
    /// One stacked weight behind one launch, a slot per selected matrix.
    Fanout,
    /// Every routed expert's three projections behind one launch.
    Routed,
}

impl TrellisOp {
    /// How this launch is reported.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dense => "trellis gemm",
            Self::Pair => "trellis pair",
            Self::Fanout => "trellis fan-out",
            Self::Routed => "trellis experts",
        }
    }
}

/// One unit of work in a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Step {
    /// An ordinary kernel launch.
    Kernel {
        /// The entry point.
        entry: Entry,
        /// Its grid, block and dynamic shared memory.
        launch: Launch,
        /// What it is handed, in kernel declaration order.
        args: Args,
    },
    /// A cooperative trellis launch, whose whole grid has to be resident at
    /// once because its blocks hand partial sums down a lock chain.
    ///
    /// The trellis kernels are template instantiations of a different vendored
    /// tree than the block kernels, so the entry point is named the way every
    /// other template in this module is - by the template and its arguments -
    /// and resolved through the table a build captures off its compiled
    /// modules.
    Trellis {
        /// Which of the three cooperative launches this is.
        op: TrellisOp,
        /// The entry point.
        entry: Entry,
        /// Tiling shape the weight is read under.
        shape: GemmShape,
        /// Trellis width.
        bits: u32,
        /// Blocks in one group, which is the grid's `x` extent.
        ///
        /// A dense projection has one group and this is its whole grid. The
        /// grouped and routed forms do not: their kernels read `blockIdx.z` as
        /// which group a block belongs to and `gridDim.x` as how wide that
        /// group is, so a flattened grid of the same block count puts every
        /// block in group zero and tells each of them the group is the whole
        /// grid.
        blocks: u32,
        /// Groups the launch starts, which is the grid's `z` extent.
        groups: u32,
        /// Threads per block, which the tiling shape fixes.
        block_dim: u32,
        /// Dynamic shared memory the launch reserves.
        smem: u32,
        /// What it carves out of the per-stream scratch.
        staging: Staging,
        /// What it is handed, in kernel declaration order.
        args: Args,
    },
    /// A matrix multiply on the device's BLAS tier.
    ///
    /// The dimensions are the tier's own, which is column-major: `m` is the
    /// width of the projection's output, `n` the rows of the chunk, and the
    /// left operand is the weight. A weight stored one output row at a time is
    /// read transposed and one expanded from a trellis is not, which is the
    /// difference `op_a` carries and the reason it is not a stride.
    Blas {
        /// How the left operand is read.
        op_a: GemmOp,
        /// How the right operand is read.
        op_b: GemmOp,
        /// Rows of the product.
        m: u32,
        /// Columns of the product.
        n: u32,
        /// Shared inner dimension.
        k: u32,
        /// Leading dimension of the left operand.
        lda: u32,
        /// Leading dimension of the right operand.
        ldb: u32,
        /// Leading dimension of the accumulator.
        ldc: u32,
        /// Left operand.
        a: Operand,
        /// Right operand.
        b: Operand,
        /// Accumulator.
        c: Operand,
        /// Element type of the operands.
        dtype: GemmDType,
        /// Whether a row may be summed differently at another row count.
        reduction: GemmReduction,
    },
    /// A reservation of per-stream scratch, which some launches need and which
    /// cannot be made while a stream is capturing.
    ///
    /// It is a step rather than a side effect of the launch that needs it,
    /// because *when* it happens is the whole of its correctness: a reservation
    /// issued inside a capture fails, and one issued after the launch that
    /// reads it is a buffer read before it exists.
    Reserve {
        /// Bytes the following launch needs staged.
        bytes: u64,
    },
    /// Zeros a scratch buffer before the launch that accumulates into it.
    ///
    /// A step rather than something the allocator does once, because these
    /// buffers are reused every layer and every pass: the routed sort's
    /// histogram is accumulated with `atomicAdd`, and the same row is read back
    /// as the scatter's cursor, so both only mean anything if they start at
    /// zero. Without it the counts of forty-eight layers pile up, expert spans
    /// grow past the staging any group has, and the answer is fluent for four
    /// layers and then NaN.
    Clear {
        /// The buffer to zero.
        buffer: Operand,
    },
    /// A point in a branch of concurrent work; see [`Branch`].
    Branch(Branch),
    /// An ordinary launch that carves the per-stream staging the cooperative
    /// launches use - the decode-shaped trellis projection, whose transformed
    /// activation and partial sums live there.
    Staged {
        /// The entry point.
        entry: Entry,
        /// Its grid and block.
        launch: Launch,
        /// What it carves out of the per-stream scratch.
        staging: Staging,
        /// What it is handed, in kernel declaration order.
        args: Args,
    },
    /// A trellis weight expanded once into half precision, before the BLAS
    /// multiply that follows it.
    ///
    /// An ordinary grid rather than a cooperative one - one block per tile of
    /// the expanded weight - but the weight it writes lives in the same
    /// per-stream staging the cooperative launches carve, which is why it is a
    /// tier of its own rather than a kernel step.
    Expand {
        /// The entry point.
        entry: Entry,
        /// Contraction width of the weight.
        k: u32,
        /// Output width of the weight.
        n: u32,
        /// Trellis width.
        bits: u32,
        /// Its grid and block.
        launch: Launch,
        /// What it carves out of the per-stream scratch.
        staging: Staging,
        /// What it is handed, in kernel declaration order.
        args: Args,
    },
}

/// The three points that bracket work run beside the main stream.
///
/// `Fork`, the branch's launches, `Resume`, the main stream's launches, `Join`:
/// the branch waits for everything before the fork, the main stream runs on
/// beside it from the resume, and nothing after the join starts before the
/// branch is done. The branch must neither read what the main stream's
/// launches write nor write what they read, and it must not use the
/// per-stream cooperative scratch while they do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Branch {
    /// Opens the branch.
    Fork,
    /// Ends the branch's launches.
    Resume,
    /// Waits for the branch.
    Join,
}

impl Step {
    /// An ordinary launch.
    #[must_use]
    pub const fn kernel(entry: Entry, launch: Launch, args: Args) -> Self {
        Self::Kernel {
            entry,
            launch,
            args,
        }
    }

    /// The operands this step is handed, borrowed from the step.
    ///
    /// The list a reader compares against a kernel's own parameter list. It
    /// borrows rather than copying so a caller can hold it across an
    /// assertion; [`Step::args`] is the whole list by value, for a caller that
    /// wants to pass it on.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] when the step was built from more operands
    /// than a launch can carry.
    pub fn operands(&self) -> CoreResult<&[Operand]> {
        match self {
            Self::Kernel { args, .. }
            | Self::Trellis { args, .. }
            | Self::Expand { args, .. }
            | Self::Staged { args, .. } => args.operands(),
            Self::Blas { .. } | Self::Reserve { .. } | Self::Clear { .. } | Self::Branch(_) => {
                Ok(&[])
            }
        }
    }

    /// What this step is handed.
    ///
    /// Empty for the two tiers whose arguments are a descriptor rather than a
    /// parameter array: the matrix multiply, whose operands travel in a
    /// descriptor the device's library reads, and the reservation, which
    /// launches nothing.
    #[must_use]
    pub const fn args(&self) -> Args {
        match self {
            Self::Kernel { args, .. }
            | Self::Trellis { args, .. }
            | Self::Expand { args, .. }
            | Self::Staged { args, .. } => *args,
            Self::Blas { .. } | Self::Reserve { .. } | Self::Clear { .. } | Self::Branch(_) => {
                Args::new()
            }
        }
    }

    /// What this step carves out of the per-stream cooperative scratch.
    #[must_use]
    pub const fn staging(&self) -> Staging {
        match self {
            Self::Trellis { staging, .. }
            | Self::Expand { staging, .. }
            | Self::Staged { staging, .. } => *staging,
            Self::Kernel { .. }
            | Self::Blas { .. }
            | Self::Reserve { .. }
            | Self::Clear { .. }
            | Self::Branch(_) => Staging::none(),
        }
    }

    /// The entry point this step launches, or `None` for the tiers that have
    /// none.
    #[must_use]
    pub const fn entry(&self) -> Option<Entry> {
        match self {
            Self::Kernel { entry, .. }
            | Self::Trellis { entry, .. }
            | Self::Expand { entry, .. }
            | Self::Staged { entry, .. } => Some(*entry),
            Self::Blas { .. } | Self::Reserve { .. } | Self::Clear { .. } | Self::Branch(_) => None,
        }
    }

    /// The name this step is reported by, which for a tier with no entry point
    /// is the tier.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Kernel { entry, .. } | Self::Staged { entry, .. } => entry.name(),
            Self::Trellis { op, .. } => op.name(),
            Self::Blas { .. } => "blas gemm",
            Self::Expand { .. } => "trellis expand",
            Self::Reserve { .. } => "stream scratch",
            Self::Clear { .. } => "zero a buffer",
            Self::Branch(Branch::Fork) => "fork",
            Self::Branch(Branch::Resume) => "resume",
            Self::Branch(Branch::Join) => "join",
        }
    }
}

/// An ordered list of steps.
///
/// A newtype over the vector rather than the vector itself, because the two
/// things a caller does with one - append another plan's steps, and read them
/// in order - are the whole interface, and a bare `Vec` would also offer
/// removal from the middle, which no caller should have.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Plan {
    steps: Vec<Step>,
    staging: Staging,
}

impl Plan {
    /// An empty plan.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            steps: Vec::new(),
            staging: Staging::none(),
        }
    }

    /// Appends one step.
    ///
    /// The plan also takes the widest per-stream staging any step it holds
    /// carves. A reservation has to be made once, before the first launch that
    /// reads it, because growing it later would move an address a captured
    /// launch has already baked in - so the number a pass reserves is a
    /// property of the whole plan and is accumulated here rather than asked for
    /// per launch.
    pub fn push(&mut self, step: Step) {
        self.staging = self.staging.union(step.staging());
        self.steps.push(step);
    }

    /// Appends one ordinary launch.
    pub fn launch(&mut self, entry: Entry, launch: Launch, args: Args) {
        self.steps.push(Step::kernel(entry, launch, args));
    }

    /// Appends everything in `other`, in order.
    pub fn append(&mut self, other: Self) {
        self.staging = self.staging.union(other.staging);
        self.steps.extend(other.steps);
    }

    /// The widest per-stream staging any step in this plan carves.
    #[must_use]
    pub const fn staging(&self) -> Staging {
        self.staging
    }

    /// Bytes a stream must have reserved before this plan's first cooperative
    /// launch, or zero when it holds none.
    #[must_use]
    pub const fn coop_reservation(&self) -> u64 {
        self.staging.reservation()
    }

    /// The steps, in order.
    #[must_use]
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// How many steps the plan holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the plan is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// The entry and tier names in order, which is what a test reads a plan by.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        self.steps.iter().map(Step::name).collect()
    }

    /// Whether any step launches `name`.
    #[must_use]
    pub fn launches(&self, name: &str) -> bool {
        self.steps.iter().any(|step| step.name() == name)
    }

    /// How many steps launch `name`.
    #[must_use]
    pub fn count(&self, name: &str) -> usize {
        self.steps.iter().filter(|step| step.name() == name).count()
    }

    /// Checks that every launch has work to do.
    ///
    /// A zero extent is not a launch of nothing: the driver refuses it, so a
    /// plan that contains one is a plan that cannot be executed, and finding
    /// that out here rather than at the first token is the whole point of the
    /// plan being a value.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] naming the first step whose grid or block
    /// has a zero dimension.
    pub fn validate(&self) -> CoreResult<()> {
        for step in &self.steps {
            let (grid, block) = match step {
                Step::Kernel { launch, .. }
                | Step::Expand { launch, .. }
                | Step::Staged { launch, .. } => (launch.grid.volume(), launch.block.volume()),
                Step::Trellis {
                    blocks,
                    groups,
                    block_dim,
                    ..
                } => (
                    u64::from(*blocks).saturating_mul(u64::from(*groups)),
                    u64::from(*block_dim),
                ),
                Step::Blas { .. } | Step::Reserve { .. } | Step::Clear { .. } | Step::Branch(_) => {
                    continue;
                }
            };
            if grid == 0 || block == 0 {
                return Err(CoreError::ShapeMismatch {
                    field: "launch extent",
                    expected: 1,
                    got: 0,
                });
            }
        }
        Ok(())
    }
}
