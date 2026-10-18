//! Issuing a planned pass: resolving each step's operands and handing the
//! result to the device.
//!
//! Everything above this decides; this is the only file in the module that
//! calls [`shiso_core::backend::Backend::run`]. Keeping the two apart is what
//! lets the deciding half be exercised on a machine with no device and the
//! issuing half be exercised against a device that records what it was handed
//! rather than doing it.
//!
//! # What this build can issue, and what a device can still refuse
//!
//! Every one of the five step kinds is issued here. Three of them need a
//! capability the device may not have, and the refusal in that case names the
//! capability rather than the step: a cooperative launch and the reservation it
//! reads need [`shiso_core::backend::CoopLaunch`], and a multiply needs
//! [`shiso_core::backend::BlasBackend`]. A device with neither runs the block
//! kernels and refuses the rest by name, which is what a fake backend in a unit
//! test does.
//!
//! The one thing that is still a build artifact rather than a device fact is
//! the entry name of a template instantiation. Those are mangled, they are read
//! off a compiled module, and an empty table makes every template dispatch -
//! including the trellis projections, which are all templates - a typed refusal
//! naming the missing capture rather than a guessed name.

use shiso_core::Extent3;
use shiso_core::backend::{Backend, CoopLaunch, GemmDesc, GraphBackend, KernelArg, LaunchArgs};
use shiso_core::error::{CoreError, CoreResult};

use crate::sparse::bind::{Binder, Operand};
use crate::sparse::island::{self, Island, IslandRun};
use crate::sparse::kernels::EntryTable;
use crate::sparse::plan::{Branch, Step, TrellisOp};
use crate::sparse::scratch::Buf;
use crate::sparse::stack::{IslandSpan, Pass};
use crate::sparse::tap::{ResidualTap, TapPoint};

/// What happened when a pass ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct PassReport {
    /// Kernel launches issued.
    pub launches: usize,
    /// Scratch reservations made.
    pub reservations: usize,
    /// Islands served from a captured graph.
    pub replayed: usize,
    /// Islands encoded and captured.
    pub captured: usize,
    /// Islands encoded directly.
    pub eager: usize,
    /// Multiplies issued on the device's matrix-multiply tier.
    pub multiplies: usize,
    /// Scratch buffers zeroed before the launches that accumulate into them.
    pub cleared: usize,
}

/// Whether each step is followed by a device barrier.
///
/// A diagnostic switch, off unless `SHISO_STEP_SYNC=1`, and it validates the one
/// release path rather than changing it: the steps issued are identical and only
/// the reporting point moves. It exists because a faulting launch is reported by
/// whichever *later* call happens to synchronise - on this stack usually a
/// multiply several layers along - so without it the name in the error belongs
/// to an innocent kernel.
fn step_sync() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SHISO_STEP_SYNC").as_deref() == Ok("1") || step_dump())
}

/// Whether each step's destination is read back after it runs.
///
/// A second diagnostic switch, under `SHISO_STEP_DUMP=1`, and it implies the
/// barrier: reading a buffer back before the launch that wrote it has finished
/// would report the previous pass's numbers.
fn step_dump() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("SHISO_STEP_DUMP").as_deref(), Ok("1" | "2")))
}

/// Whether every operand of every step is read back, not just the destination.
///
/// `SHISO_STEP_DUMP=2`. The routed launch reports its inputs either way; this
/// widens that to the whole pass, which is what a comparison of two passes that
/// should agree needs when the step that first differs is reading rather than
/// writing.
fn step_dump_all() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SHISO_STEP_DUMP").as_deref() == Ok("2"))
}

/// The grid, block and staging a step asks for, for a diagnostic.
fn step_geometry(step: &Step) -> String {
    match step {
        Step::Kernel { launch, .. } | Step::Expand { launch, .. } | Step::Staged { launch, .. } => {
            format!(
                "grid {}x{}x{} block {}x{}x{} smem {}",
                launch.grid.x,
                launch.grid.y,
                launch.grid.z,
                launch.block.x,
                launch.block.y,
                launch.block.z,
                launch.smem,
            )
        }
        Step::Trellis {
            op,
            blocks,
            groups,
            block_dim,
            smem,
            staging,
            bits,
            shape,
            ..
        } => format!(
            "{} k{} shape{} grid {}x1x{} block {} smem {} staging {}",
            op.name(),
            bits,
            shape.get(),
            blocks,
            groups,
            block_dim,
            smem,
            staging.bytes,
        ),
        Step::Reserve { bytes } => format!("{bytes} bytes"),
        Step::Fetch { layer, ids } => format!("layer {layer} ids {ids}"),
        Step::Clear { .. } => "zeros".to_owned(),
        Step::Branch(branch) => format!("{branch:?}"),
        Step::Blas { m, n, k, .. } => format!("m {m} n {n} k {k}"),
    }
}

/// Which operand a step writes its result into.
///
/// Zero for every launcher in this module except the three cooperative
/// trellis forms, whose first operand is the activation they read: the dense
/// projection takes `A, B, C`, the pair takes `A, B_base, B_stride, C`, and the
/// routed block takes four staging pointers before the slot output it fills.
fn destination(step: &Step) -> usize {
    match step {
        Step::Trellis { op, .. } => match op {
            TrellisOp::Dense => 2,
            TrellisOp::Pair | TrellisOp::Fanout => 3,
            TrellisOp::Routed => 4,
        },
        Step::Kernel { .. }
        | Step::Expand { .. }
        | Step::Staged { .. }
        | Step::Blas { .. }
        | Step::Reserve { .. }
        | Step::Clear { .. }
        | Step::Branch(_)
        | Step::Fetch { .. } => 0,
    }
}

/// What a step is called in a diagnostic.
fn step_name(step: &Step) -> &'static str {
    match step {
        Step::Kernel { entry, .. }
        | Step::Expand { entry, .. }
        | Step::Trellis { entry, .. }
        | Step::Staged { entry, .. } => entry.name(),
        Step::Reserve { .. } => "reserve scratch",
        Step::Clear { .. } => "zero a buffer",
        Step::Branch(_) => "branch",
        Step::Blas { .. } => "matrix multiply",
        Step::Fetch { .. } => "fetch routed experts",
    }
}

/// Issues the steps of a plan against one device and one session's state.
#[derive(Debug)]
pub struct Exec<'a> {
    backend: &'a dyn Backend,
    entries: EntryTable,
    binder: Binder<'a>,
}

impl<'a> Exec<'a> {
    /// Binds a plan against a device.
    ///
    /// `entries` is the table a build captured off its compiled modules; an
    /// empty one makes every template dispatch a typed refusal rather than a
    /// guessed name.
    #[must_use]
    pub const fn new(backend: &'a dyn Backend, entries: EntryTable, binder: Binder<'a>) -> Self {
        Self {
            backend,
            entries,
            binder,
        }
    }

    /// The operand resolver this executor binds through.
    #[must_use]
    pub const fn binder(&self) -> &Binder<'a> {
        &self.binder
    }

    /// Issues one step.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] for a tier this build cannot issue, naming
    /// which one and why; whatever the binder reports for an operand this
    /// context cannot name; and whatever the device reports for the launch.
    pub fn run_step(&self, step: &Step) -> CoreResult<()> {
        match step {
            // An ordinary launch, and the expansion of a trellis weight, which
            // has an ordinary grid too: the only thing that sets the second
            // apart is that it writes into the per-stream staging the
            // cooperative launches carve, which is a fact about the plan rather
            // than about the launch call.
            Step::Kernel {
                entry,
                launch,
                args,
            }
            | Step::Expand {
                entry,
                launch,
                args,
                ..
            }
            | Step::Staged {
                entry,
                launch,
                args,
                ..
            } => {
                let kernel = self.entries.resolve(*entry)?;
                let mut bound = Vec::with_capacity(args.len());
                self.binder.bind(*args, &mut bound)?;
                self.backend.run(
                    kernel,
                    &LaunchArgs {
                        grid: launch.grid,
                        block: launch.block,
                        smem_bytes: launch.smem,
                        args: &bound,
                    },
                )
            }
            // Reserving is the one non-launch step this build can carry out,
            // and *when* it happens is the whole of its correctness: the launch
            // that reads the slab comes next, and a reservation attempted
            // inside a capture is refused by the device rather than silently
            // moving an address the capture has baked in.
            Step::Reserve { bytes } => self.coop()?.reserve_scratch(*bytes),
            // Zeroing is a step, not a side effect of allocation: see
            // `Step::Clear`.
            Step::Clear { buffer } => self.backend.fill_zero(self.binder.buffer(*buffer)?),
            Step::Branch(Branch::Fork) => self.backend.fork(),
            Step::Branch(Branch::Resume) => self.backend.resume(),
            Step::Branch(Branch::Join) => self.backend.join(),
            Step::Fetch { layer, ids } => self.binder.fetch_experts(self.backend, *layer, *ids),
            // The whole grid of a trellis launch has to be resident at once,
            // because its blocks hand partial sums down the lock chain at the
            // head of the per-stream scratch. That is a different launch call,
            // not a larger grid.
            Step::Trellis {
                entry,
                blocks,
                groups,
                block_dim,
                smem,
                args,
                ..
            } => {
                let kernel = self.entries.resolve(*entry)?;
                let mut bound = Vec::with_capacity(args.len());
                self.binder.bind(*args, &mut bound)?;
                self.coop()?.run_cooperative(
                    kernel,
                    &LaunchArgs {
                        // Three-dimensional on purpose: `blockIdx.z` is which
                        // group a block belongs to and `gridDim.x` is how wide
                        // that group is. A dense launch has one group.
                        grid: Extent3::new(*blocks, 1, *groups),
                        block: Extent3::linear(*block_dim),
                        smem_bytes: *smem,
                        args: &bound,
                    },
                )
            }
            Step::Blas {
                op_a,
                op_b,
                m,
                n,
                k,
                lda,
                ldb,
                ldc,
                a,
                b,
                c,
                dtype,
                reduction,
            } => self
                .backend
                .as_blas()
                .ok_or(CoreError::Unsupported {
                    what: "a projection on a device with no matrix-multiply tier",
                    backend: "engine",
                })?
                .gemm(&GemmDesc {
                    op_a: *op_a,
                    op_b: *op_b,
                    m: *m,
                    n: *n,
                    k: *k,
                    lda: *lda,
                    ldb: *ldb,
                    ldc: *ldc,
                    a: self.binder.buffer(*a)?,
                    b: self.binder.buffer(*b)?,
                    c: self.binder.buffer(*c)?,
                    // The projections here replace their destination rather
                    // than accumulating into it: a nonzero second scale would
                    // add this chunk's result onto the previous chunk's.
                    alpha: 1.0,
                    beta: 0.0,
                    dtype: *dtype,
                    reduction: *reduction,
                }),
        }
    }

    /// Reads back the buffer a step wrote and reports what is in it.
    ///
    /// Under the same knob as the barrier, and for the same reason: a wrong
    /// answer forty layers deep is found by bisecting the pass, and the only
    /// way to bisect a pass is to look at what each step left behind. A
    /// summary rather than a window, because a window onto the first eight
    /// words of a `[tokens][hidden]` buffer only ever shows the first token -
    /// which is how a pass whose second row was already NaN read as healthy
    /// for four layers.
    ///
    /// Everything here reports rather than fails. A step whose destination is
    /// a scalar, a device that will not read back, a buffer larger than the
    /// window - none of them is a reason to stop a pass that is only being
    /// watched.
    fn dump_destination(&self, at: usize, step: &Step) {
        if !step_dump() {
            return;
        }
        let Ok(operands) = step.operands() else {
            return;
        };
        // The routed expert launch is the one step whose inputs are worth
        // reporting whole: it takes four staging regions, nine weight spans and
        // two routing tables, and which of them a bad row came from is not
        // recoverable from the destination alone.
        if step_dump_all()
            || matches!(
                step,
                Step::Trellis {
                    op: TrellisOp::Routed,
                    ..
                }
            )
        {
            for (which, &operand) in operands.iter().enumerate() {
                if which != destination(step) {
                    self.summarize(at, step, which, operand, "read");
                }
            }
        }
        let Some(&dst) = operands.get(destination(step)) else {
            return;
        };
        self.summarize(at, step, destination(step), dst, "wrote");
    }

    /// Reads one operand back and reports what is in it.
    fn summarize(&self, at: usize, step: &Step, which: usize, operand: Operand, what: &str) {
        /// Most a single read-back copies, so a weight-sized operand does not
        /// turn a diagnostic into a transfer benchmark.
        ///
        /// Wide enough for every activation this family carries at the widest
        /// chunk it runs - the residual is `chunk * hidden * streams` floats -
        /// because the sums below are only comparable to another engine's when
        /// both cover the whole tensor.
        const CAP: u64 = 64 << 20;
        let Ok(view) = self.binder.buffer(operand) else {
            return;
        };
        let want = view.len().min(CAP) / size_of::<f32>() as u64 * size_of::<f32>() as u64;
        let (Ok(head), Ok(want)) = (view.slice(0, want), usize::try_from(want)) else {
            return;
        };
        let mut bytes = vec![0u8; want];
        if self.backend.copy_d2h(head, &mut bytes).is_err() {
            return;
        }
        let words = bytes
            .as_chunks::<{ size_of::<f32>() }>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word));
        let mut nans = 0usize;
        let mut zeros = 0usize;
        let mut count = 0usize;
        let mut low = f32::INFINITY;
        let mut high = f32::NEG_INFINITY;
        let mut head = Vec::new();
        let mut where_nan: Vec<usize> = Vec::new();
        let mut last_nan = 0usize;
        let mut first_live = None;
        let mut last_live = 0usize;
        // In double precision, and reported alongside the extremes, because
        // these three are what another implementation of this model prints for
        // the same tensor: a sum cancels and is a poor comparator on its own,
        // the absolute sum does not, and together they say whether two passes
        // that should agree do.
        let mut sum = 0.0f64;
        let mut l1 = 0.0f64;
        let mut l2 = 0.0f64;
        for value in words {
            if head.len() < 4 {
                head.push(value);
            }
            if value.is_nan() {
                nans += 1;
                if where_nan.len() < 8 {
                    where_nan.push(count);
                }
                last_nan = count;
            } else {
                if value == 0.0 {
                    zeros += 1;
                } else {
                    first_live.get_or_insert(count);
                    last_live = count;
                }
                low = low.min(value);
                high = high.max(value);
                sum += f64::from(value);
                l1 += f64::from(value.abs());
                l2 += f64::from(value) * f64::from(value);
            }
            count += 1;
        }
        tracing::info!(
            at,
            step = step_name(step),
            which,
            buffer = ?operand,
            count,
            nans,
            zeros,
            ?where_nan,
            last_nan,
            first_live,
            last_live,
            low,
            high,
            sum,
            l1,
            l2 = l2.sqrt(),
            ?head,
            "{what}",
        );
    }

    /// The device's cooperative launch capability.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] when the device has none.
    fn coop(&self) -> CoreResult<&dyn CoopLaunch> {
        self.backend.as_coop().ok_or(CoreError::Unsupported {
            what: "a cooperative launch on a device that cannot keep a whole grid resident",
            backend: "engine",
        })
    }

    /// Issues a run of steps in order, and says how many were launches.
    ///
    /// # Errors
    ///
    /// Whatever [`Exec::run_step`] reports for the first step that fails. The
    /// steps before it have already been issued: a pass is a stream, and there
    /// is no undo.
    pub fn run_steps(&self, steps: &[Step]) -> CoreResult<PassReport> {
        let mut report = PassReport::default();
        for (at, step) in steps.iter().enumerate() {
            let started = step_sync().then(std::time::Instant::now);
            self.run_step(step).inspect_err(|error| {
                tracing::error!(at, step = step_name(step), error = ?error, "a step would not issue");
            })?;
            // A launch failure is asynchronous: the call that reports it is
            // whichever later call synchronises, which on this stack is usually
            // a multiply several layers on. Under the knob every step is
            // followed by a barrier, so the first step to fault is the one
            // named.
            if step_sync() {
                self.backend.sync().inspect_err(|error| {
                    tracing::error!(
                        at,
                        step = step_name(step),
                        geometry = %step_geometry(step),
                        error = ?error,
                        "the device faulted on this step",
                    );
                })?;
                // With a barrier after it, a step's wall clock is its own device
                // time rather than the pipeline's. The total over a pass is
                // longer than the pass really takes - nothing overlaps any more -
                // but which launches the time is *in* is what a profile is for,
                // and that is what the serialization makes readable.
                tracing::info!(
                    at,
                    step = step_name(step),
                    geometry = %step_geometry(step),
                    micros = started.map_or(0, |at| at.elapsed().as_micros()),
                    "step",
                );
                self.dump_destination(at, step);
            }
            match step {
                Step::Kernel { .. }
                | Step::Trellis { .. }
                | Step::Expand { .. }
                | Step::Staged { .. } => {
                    report.launches = report.launches.saturating_add(1);
                }
                Step::Reserve { .. } => {
                    report.reservations = report.reservations.saturating_add(1);
                }
                Step::Blas { .. } => report.multiplies = report.multiplies.saturating_add(1),
                Step::Clear { .. } => report.cleared = report.cleared.saturating_add(1),
                Step::Branch(_) | Step::Fetch { .. } => {}
            }
        }
        Ok(report)
    }

    /// Issues a whole pass, replaying each island's captured graph where the
    /// device has one.
    ///
    /// The prologue and the epilogue - the row gather, the residual broadcast,
    /// the collapsing mix and the vocabulary projection - are outside every
    /// island and are always encoded directly: their row count varies with the
    /// chunk, which is the thing a capture cannot carry.
    ///
    /// # Errors
    ///
    /// Whatever [`Exec::run_step`] reports, and [`CoreError::InvalidValue`]
    /// when an island's key does not fit the slot it is filed under.
    pub fn run_pass(
        &self,
        pass: &Pass,
        graphs: Option<&dyn GraphBackend>,
    ) -> CoreResult<PassReport> {
        self.run_pass_tapped(pass, graphs, None)
    }

    /// Issues a whole pass, copying one residual row into `tap` after the
    /// prologue and after every island when a tap is given.
    ///
    /// A tapped pass is run with `graphs` withheld, so nothing is replayed and
    /// nothing is captured: see [`crate::sparse::tap`]. Without a tap this is
    /// [`Exec::run_pass`] exactly.
    ///
    /// # Errors
    ///
    /// As [`Exec::run_pass`], plus whatever the barrier and the copy report:
    /// unlike the step dump a tap was asked for, so a row it cannot read is a
    /// failure rather than a line missing from a log.
    pub fn run_pass_tapped(
        &self,
        pass: &Pass,
        graphs: Option<&dyn GraphBackend>,
        tap: Option<&mut ResidualTap>,
    ) -> CoreResult<PassReport> {
        self.run_pass_landed(pass, graphs, tap, None)
    }

    /// Issues a whole pass as [`Exec::run_pass_tapped`] does, calling `land`
    /// exactly once on the way: in front of the island
    /// [`Pass::ngram_island`] names, or before the first step when it names
    /// none.
    ///
    /// `land` is the host work the island needs done first - awaiting the
    /// n-gram reads and uploading them - and everything issued before it reads
    /// none of what it writes, so the device runs those launches while the
    /// host does it. A pass is one stream either way: the upload `land` makes
    /// is ordered behind the launches already issued and in front of the ones
    /// after it, exactly where the rows were always read.
    ///
    /// # Errors
    ///
    /// As [`Exec::run_pass_tapped`], plus whatever `land` reports. The steps
    /// issued before a failing `land` have already been issued: a pass is a
    /// stream, and there is no undo.
    pub fn run_pass_landed(
        &self,
        pass: &Pass,
        graphs: Option<&dyn GraphBackend>,
        mut tap: Option<&mut ResidualTap>,
        mut land: Option<&mut dyn FnMut() -> CoreResult<()>>,
    ) -> CoreResult<PassReport> {
        let steps = pass.plan.steps();
        let graphs = graphs.filter(|_| pass.capturable && tap.is_none());
        if let Some(tap) = tap.as_deref_mut() {
            tap.restart();
        }
        let landing_island = pass
            .ngram_island
            .filter(|island| *island < pass.islands.len());
        if landing_island.is_none()
            && let Some(land) = land.take()
        {
            land()?;
        }
        let mut report = PassReport::default();
        let mut at = 0usize;
        for (index, span) in pass.islands.iter().enumerate() {
            report = merge(report, self.run_steps(prefix(steps, at, span.start)?)?);
            if landing_island == Some(index)
                && let Some(land) = land.take()
            {
                land()?;
            }
            // The prologue is everything in front of the first island. It ends
            // with the broadcast; the reservation placed there writes no
            // activation.
            if index == 0
                && let Some(tap) = tap.as_deref_mut()
            {
                self.copy_residual(tap, TapPoint::Embed, pass.rows)?;
            }
            report = merge(report, self.run_island(steps, *span, graphs)?);
            if let Some(tap) = tap.as_deref_mut() {
                let point = match span.key.island {
                    Island::Mixer => TapPoint::Mixer(span.key.layer),
                    Island::Experts => TapPoint::Layer(span.key.layer),
                };
                self.copy_residual(tap, point, pass.rows)?;
            }
            at = span.start.saturating_add(span.len);
        }
        report = merge(report, self.run_steps(tail(steps, at))?);
        Ok(report)
    }

    /// Copies the last of `rows` residual rows into `tap` under `point`.
    ///
    /// The barrier comes first: without it the copy races the launches still
    /// writing the row and reports the previous island's numbers.
    fn copy_residual(&self, tap: &mut ResidualTap, point: TapPoint, rows: u32) -> CoreResult<()> {
        let width = tap.row_len() as u64 * size_of::<f32>() as u64;
        let last = rows.checked_sub(1).ok_or(CoreError::ShapeMismatch {
            field: "tapped pass rows",
            expected: 1,
            got: 0,
        })?;
        self.backend.sync()?;
        let view = self.binder.buffer(Operand::Scratch(Buf::Residual))?;
        let row = view.slice(u64::from(last).saturating_mul(width), width)?;
        let mut bytes = vec![0u8; usize::try_from(width).unwrap_or(usize::MAX)];
        self.backend.copy_d2h(row, &mut bytes)?;
        let values = bytes
            .as_chunks::<{ size_of::<f32>() }>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
        tap.push(point, values)
    }

    /// Runs one island, replaying its captured graph when there is one.
    ///
    /// # Errors
    ///
    /// Whatever the island's steps report, and [`CoreError::InvalidValue`] for
    /// a key that does not fit its slot. An island with no slot at all - a
    /// batched one - is encoded rather than refused.
    pub fn run_island(
        &self,
        steps: &[Step],
        span: IslandSpan,
        graphs: Option<&dyn GraphBackend>,
    ) -> CoreResult<PassReport> {
        let end = span.start.saturating_add(span.len);
        let body = steps.get(span.start..end).ok_or(CoreError::OutOfBounds {
            offset: span.start as u64,
            len: span.len as u64,
            buffer_bytes: steps.len() as u64,
        })?;
        let key = span.key.pack()?;
        let mut issued = PassReport::default();
        let run = island::run(graphs, key, || {
            issued = self.run_steps(body)?;
            Ok(())
        })?;
        let mut report = issued;
        match run {
            // A replay issued the island's launches on the device without this
            // process encoding them, so the launch count is what the body
            // holds rather than what was bound here.
            IslandRun::Replayed => {
                report.replayed = 1;
                report.launches = body
                    .iter()
                    .filter(|step| {
                        matches!(
                            step,
                            Step::Kernel { .. }
                                | Step::Trellis { .. }
                                | Step::Expand { .. }
                                | Step::Staged { .. }
                        )
                    })
                    .count();
                report.multiplies = body
                    .iter()
                    .filter(|step| matches!(step, Step::Blas { .. }))
                    .count();
            }
            IslandRun::Captured => report.captured = 1,
            IslandRun::Eager => report.eager = 1,
        }
        Ok(report)
    }
}

/// The steps between where the last island ended and where this one starts.
fn prefix(steps: &[Step], at: usize, start: usize) -> CoreResult<&[Step]> {
    steps.get(at..start).ok_or(CoreError::OutOfBounds {
        offset: at as u64,
        len: start.saturating_sub(at) as u64,
        buffer_bytes: steps.len() as u64,
    })
}

/// The steps after the last island.
fn tail(steps: &[Step], at: usize) -> &[Step] {
    steps.get(at..).unwrap_or(&[])
}

/// Adds two reports.
const fn merge(left: PassReport, right: PassReport) -> PassReport {
    PassReport {
        launches: left.launches.saturating_add(right.launches),
        reservations: left.reservations.saturating_add(right.reservations),
        replayed: left.replayed.saturating_add(right.replayed),
        captured: left.captured.saturating_add(right.captured),
        eager: left.eager.saturating_add(right.eager),
        multiplies: left.multiplies.saturating_add(right.multiplies),
        cleared: left.cleared.saturating_add(right.cleared),
    }
}

/// The bound arguments one step would be launched with, without launching it.
///
/// What a test reads when it is checking that a kernel is handed the buffers
/// its own parameter list names, in the order it names them.
///
/// # Errors
///
/// Whatever the binder reports for an operand this context cannot name.
pub fn bound_args<'a>(binder: &Binder<'a>, step: &Step) -> CoreResult<Vec<KernelArg<'a>>> {
    let mut out = Vec::new();
    binder.bind(step.args(), &mut out)?;
    Ok(out)
}
