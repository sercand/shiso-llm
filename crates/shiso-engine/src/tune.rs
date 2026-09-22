//! Picking, by measurement, which shape of a kernel this device runs fastest.
//!
//! The launch geometry of a hand-written kernel is not a constant: how many
//! lanes a threadgroup can carry is a register-budget question, and the budget
//! is a property of the device and of the compiled pipeline. A number that fits
//! one part is a refused launch on a smaller one and money left on the table on
//! a larger one. So a family compiles the same kernel at several widths and
//! this module asks the machine which of them to run.
//!
//! What it does, in order: probe each candidate shape for the thread ceiling
//! the device reports for it, trial-dispatch the ones wide enough, time them,
//! quantise away differences smaller than the measurement's own noise, and pick
//! the fastest - falling back down a ladder of weaker signals when there is no
//! timing to go on.
//!
//! # Why nothing here names a model or a device
//!
//! Two things used to live in one file: this machinery, and qwen35's table of
//! nine GDN entry points. Only the second is specific to anything. The
//! machinery reaches the device through [`Backend::max_block_threads`] and
//! through a closure the caller hands in; it learns which shapes exist from a
//! [`TunableShape`] the caller defines. It therefore knows nothing about GDN,
//! about qwen35, or about Metal.
//!
//! In particular it is not Metal-only. `max_block_threads` has a `None`
//! default, and a backend that cannot answer produces a table of `None`s, which
//! [`choose_shape`] reads as "nobody had an opinion" and answers with the
//! reference shape the build has always used. A CUDA build therefore behaves
//! exactly as it did before this module existed until somebody implements the
//! probe there (`cuFuncGetAttribute`), at which point the same ladder starts
//! choosing for it with no edit here.
//!
//! # What a family has to supply
//!
//! A gemma or a deepseek table is a sibling of qwen35's `gdn_tune.rs`, not an
//! edit to this one. It owes three things:
//!
//! - a type implementing [`TunableShape`], usually an enum of compiled widths;
//! - a closure that issues **one** launch of a given shape at a given probe
//!   extent, over whatever scratch that family's kernels read. That closure is
//!   the seam: it is the only place a state buffer, an argument block or a grid
//!   is named, and this module never sees any of them;
//! - the floor contract that closure has to honour, which is
//!   [`TunableShape`]'s own doc: a launch at [`FLOOR_EXTENT`] must be declined
//!   by the kernel before it touches memory.
//!
//! # What "viable" means
//!
//! A row in the table can fail two different ways, and they are not the same
//! fact. `limit` is what `max_block_threads` said *before* a launch was ever
//! attempted; it is a lane-count question and nothing else. But threadgroup
//! *memory* is only checked at dispatch, so a shape can report a limit wide
//! enough for its lane count and still be refused when a launch is actually
//! encoded - that is exactly how one over-budget GDN allocation slipped through
//! an earlier round of this work. So a candidate carries both facts, and
//! viability requires both: wide enough, and not refused.
//!
//! Its callers are outside this crate. The qwen35 table is
//! `shiso-models/src/qwen35/kernels/gdn_tune.rs`, which is where the nine
//! entry points, their lane counts and the scratch one timing run reads all
//! live; nothing about any of that is visible from here.

use std::fmt::Debug;
use std::time::Instant;

use shiso_core::backend::{Backend, KernelRef};
use shiso_core::{CoreError, CoreResult};

/// One compiled shape of a kernel, as much of it as the tuner needs to see.
///
/// Three facts, and deliberately no more. The tuner has to name the pipeline to
/// probe it ([`entry`](Self::entry) and [`module`](Self::module)) and has to
/// know the width the shape is launched at, because that is the number the
/// device's reported ceiling is compared against. Everything else a launch
/// needs - threadgroup memory, buffers, the grid, the argument block - belongs
/// to the family and is bound inside the closure the family hands to
/// [`measure`].
///
/// `Copy` because a shape is a label the table copies per row, and `Debug`
/// because a refusal is logged with the shape that was refused.
///
/// # The one contract this trait cannot express in its methods
///
/// [`measure`] times every implementor twice: once at the real work size the
/// caller asks for, and once at [`FLOOR_EXTENT`], to learn how much of the
/// first number is this device's own launch overhead rather than this
/// kernel's work. That second measurement is only honest if a launch at
/// `FLOOR_EXTENT` truly does nothing - so **every kernel a `TunableShape`
/// names must decline at `FLOOR_EXTENT` before its first memory access**,
/// the same pipeline, lanes, threadgroup memory and grid as the real launch,
/// doing nothing once it starts.
///
/// Nothing in this trait's signature checks that, because it is a property of
/// the kernel body the shape points at, not of the Rust type. GDN's kernels
/// satisfy it for free - each opens with `if (head_dim != Q35_GDN_D) return;`,
/// a guard load-bearing kernel authors wrote for a *numerical* reason (see
/// `gdn_tune.rs`), sitting ahead of the first memory access in every variant -
/// but a family without a guard shaped like that has to add one before it can
/// use [`measure`] at all. A family that skips this turns the floor probe
/// into a second full timing of real work, silently: nothing here fails, the
/// floor is just wrong, and the `debug` line beside every row becomes
/// nonsense in a way no test catches.
pub trait TunableShape: Copy + Debug {
    /// The entry point this shape was compiled as.
    fn entry(self) -> &'static str;

    /// Threads per threadgroup this shape must be launched at.
    ///
    /// Not a maximum and not a suggestion: a kernel's index arithmetic assumes
    /// its own lane count, so this is the width the device has to accept for
    /// the shape to be usable at all.
    fn lanes(self) -> u32;

    /// The kernel module the entry point was compiled into.
    fn module(self) -> &'static str;

    /// The kernel this shape names, as the backend wants it.
    ///
    /// Provided rather than required: a shape that can say where it lives and
    /// what it is called can always say which kernel it is, and having one
    /// spelling of the pair keeps the probe and the dispatch naming the same
    /// pipeline.
    fn kernel(self) -> KernelRef {
        KernelRef::new(self.module(), self.entry())
    }
}

/// One shape as the probe and the benchmark found it.
#[derive(Debug, Clone, Copy)]
pub struct Candidate<S> {
    /// The shape this row describes.
    pub shape: S,
    /// Threads the device will accept for it; `None` when it could not say.
    pub limit: Option<u32>,
    /// Best observed launch, in nanoseconds; `None` when it was not timed.
    pub nanos: Option<u64>,
    /// Whether a trial dispatch of this shape was refused by the device.
    ///
    /// Distinct from `limit` being `None`: a shape can report a limit that
    /// covers its lane count and still be refused, because threadgroup
    /// memory is a dispatch-time check that `max_block_threads` does not
    /// make. `limit` alone cannot tell "never attempted" apart from
    /// "attempted and rejected", since both would otherwise read as
    /// `nanos: None`. That is why the distinction gets its own field rather
    /// than being inferred from the other two.
    pub refused: bool,
}

impl<S: TunableShape> Candidate<S> {
    /// Whether the device will accept this shape at the width it needs.
    fn viable(&self) -> bool {
        match self.limit {
            Some(limit) => limit >= self.shape.lanes() && !self.refused,
            None => false,
        }
    }
}

/// The shape to dispatch, given what the probe and the benchmark found.
///
/// `what` is the capability an empty table failed to provide, in the words the
/// family would use - "a GDN threadgroup shape this device will launch". It is
/// the caller's because this module does not know what it is choosing among,
/// and an error naming only "a shape" would send a reader here rather than to
/// the kernels that could not be launched.
///
/// # Errors
///
/// [`CoreError::Unsupported`] when the device accepts no shape at all, which
/// ends the load rather than the first request.
pub fn choose_shape<S: TunableShape>(
    what: &'static str,
    reference: S,
    candidates: &[Candidate<S>],
) -> CoreResult<S> {
    // Nobody had an opinion, so there is nothing to act on and the shape
    // the build has always used stands.
    if candidates.iter().all(|row| row.limit.is_none()) {
        return Ok(reference);
    }

    let mut best: Option<Candidate<S>> = None;
    for row in candidates.iter().filter(|row| row.viable()) {
        let better = match (&best, row.nanos) {
            // A timed shape beats an untimed one; between two timed ones
            // the faster wins; between two untimed ones the wider does,
            // which is the only signal left. Every viable row is judged this
            // way on its own reported facts - the limit is not monotone in
            // lane count (a measured device has reported a *narrower*
            // shape with a *smaller* limit than a wider sibling), so there
            // is no shortcut that stops at the first one that fits.
            //
            // [`measure`]'s callers have a documented path - scratch
            // allocation failing - where every row comes back untimed with
            // real limits, so this untimed-vs-untimed comparison is not a
            // hypothetical corner; it is the whole table on that path.
            (None, _) => true,
            (Some(cur), Some(ns)) => match cur.nanos {
                Some(cur_ns) => ns < cur_ns,
                None => true,
            },
            // Strict `>`, not `>=`: two untimed rows at equal lane counts -
            // a family's narrowest ordinary shape and its always-fits fallback
            // are typically both at the same width - leave `better` false
            // and `best` keeps the row already held, which is the *earlier*
            // entry in `candidates`. That is deliberate, not incidental:
            // every caller this module has lists a family best-first and its
            // fallback last, so a tie resolving to the earlier entry means an
            // ordinary shape always beats its own fallback rather than losing
            // to it on a coin flip. This makes a caller's array order
            // load-bearing: reordering it for readability is a behavioural
            // change to which kernel ships on an untimed table, not a
            // cosmetic edit.
            (Some(cur), None) => cur.nanos.is_none() && row.shape.lanes() > cur.shape.lanes(),
        };
        if better {
            best = Some(*row);
        }
    }

    best.map(|row| row.shape).ok_or(CoreError::Unsupported {
        what,
        backend: reference.module(),
    })
}

// ---------------------------------------------------------------------------------------
// The device half
// ---------------------------------------------------------------------------------------

/// Warm-up launches, discarded.
///
/// Enough to build the pipeline, fault the scratch in and let the clocks
/// settle; not enough to matter against the timed run that follows.
const WARMUP_REPS: u32 = 5;

/// Timed samples per shape. The minimum of these is the estimate.
const TIMED_REPS: u32 = 20;

/// The sample window cannot be empty.
///
/// `best_of` takes the minimum of [`TIMED_REPS`] samples, and a minimum over
/// nothing is not a slow shape - it is no measurement at all. Caught here, at
/// the constant, so that lowering it to zero is a build failure rather than a
/// table of shapes that all appear to be exactly as fast as each other.
const _: () = assert!(
    TIMED_REPS > 0,
    "TIMED_REPS is the window best_of minimises over; zero of them has no minimum",
);

/// Launches encoded per timed sample, before the queue is drained once.
///
/// The sync is not optional - without it the clock measures encoding rather
/// than running - but one sync per launch charges a whole command-buffer round
/// trip to every sample, and on this hardware that round trip is a few hundred
/// microseconds. Measured on an M1 Max it was larger than the *entire* cost of
/// a decode-length recurrent launch, which means a per-launch sync was
/// minimising a number that was mostly a constant: it compresses the
/// differences between shapes, and two shapes closer together than the round
/// trip's own variance are exactly the close call this tuner exists to settle.
///
/// So the round trip is amortised. Ten launches go into the buffer, the queue
/// is drained once, and the elapsed time is divided by ten; the sync still
/// stands between the clock and the encode, so what is measured is still
/// execution.
///
/// Chaining them is honest rather than convenient. These kernels rewrite their
/// state in place, so ten back-to-back launches are serially dependent - the
/// second cannot start before the first has written the state the second reads.
/// That is the shape a real decode has. Ten launches over *independent* buffers
/// would overlap on the device and time as something faster than the block will
/// ever run.
const LAUNCHES_PER_SAMPLE: u32 = 10;

/// The probe extent used to time a launch that does no work.
///
/// The contract this rests on - every [`TunableShape`] a family names must
/// decline at this extent before touching memory - is documented on the
/// trait, where the person who has to satisfy it will read it; this constant
/// is just the number. For GDN, `FLOOR_EXTENT` is `0` and the guard is
/// `if (head_dim != Q35_GDN_D) return;`, ahead of the first memory access in
/// every variant, wrappers included, because both the register layout and the
/// rotated sweep are built from the 128-wide head - so a launch at `0` runs
/// the same pipeline, at the same lane count, with the same threadgroup
/// memory and the same grid, and returns immediately.
///
/// That makes it the floor: what a launch of *this shape* costs before any of
/// its work is counted. It is measured rather than assumed because a timing
/// that silently contains an unknown constant cannot be compared with one
/// taken on another machine, and this is the number that says how much of a
/// row is the device and how much is the kernel.
///
/// # This depends on a guard nobody wrote for the tuner's benefit
///
/// GDN's `if (head_dim != Q35_GDN_D) return;` exists for a *numerical*
/// reason - the register layout and the rotated sweep are only correct at
/// the compiled-in head width, so a mismatched width has to refuse rather
/// than compute a wrong answer (see `gdn_tune.rs` and the Tasks that added
/// and then re-anchored it). The floor probe above rides on that guard
/// existing and on it sitting ahead of every memory access; it does not add
/// a reason of its own for the guard to stay there. A future variant that
/// moves the guard below its first read or write - for whatever numerical
/// reason seems right at the time - quietly turns the floor into a second
/// full timing of real work. Nothing fails: the row still gets a number, the
/// `debug` line still logs one, and both are wrong in a way no test here
/// catches, because there is no signal that distinguishes "returned
/// immediately" from "ran fast".
pub const FLOOR_EXTENT: u32 = 0;

/// What one shape's trial run produced.
#[derive(Debug, Clone, Copy)]
struct Trial {
    /// Best observed launch in nanoseconds, or `None` if it never ran clean.
    nanos: Option<u64>,
    /// How far apart this shape's own samples landed, as an interquartile
    /// range over [`TIMED_REPS`] of them.
    ///
    /// The scale on which `nanos` can be believed. Two shapes whose timings
    /// differ by less than this are not two different timings; they are one
    /// timing taken twice. [`tie_band`] turns the column of these into the
    /// band the whole table is judged on.
    spread: Option<u64>,
    /// What a launch of this shape costs before any of its work is counted.
    ///
    /// Diagnostic, not a decision: it is logged beside `nanos` so a reader can
    /// tell a kernel that is genuinely fast from a row that is mostly the
    /// device's own launch overhead. It is deliberately *not* subtracted from
    /// `nanos` - two numbers with independent noise subtract to a difference
    /// that can go negative, and a tuner reporting a negative time would be
    /// worse than one reporting an inflated one.
    floor: Option<u64>,
    /// Whether the device turned the dispatch away.
    refused: bool,
}

impl Trial {
    /// A shape that was never attempted: too wide for its own limit.
    const SKIPPED: Self = Self {
        nanos: None,
        spread: None,
        floor: None,
        refused: false,
    };

    /// A shape the device tried and would not take.
    const REFUSED: Self = Self {
        nanos: None,
        spread: None,
        floor: None,
        refused: true,
    };
}

/// One timed sample: [`LAUNCHES_PER_SAMPLE`] launches, one drain, divided.
///
/// The division is what makes this a per-launch figure, and the single drain
/// is what stops a command-buffer round trip being charged to each one. The
/// sync still separates the clock from the encode, so the result is execution
/// time and not encoding time.
///
/// # Errors
///
/// Whatever the dispatch or the drain reported.
fn sample<S, L>(backend: &dyn Backend, shape: S, probe_extent: u32, launch: &L) -> CoreResult<u64>
where
    S: TunableShape,
    L: Fn(S, u32) -> CoreResult<()>,
{
    let started = Instant::now();
    for _ in 0..LAUNCHES_PER_SAMPLE {
        launch(shape, probe_extent)?;
    }
    // Without this the measurement is of encoding, not of running.
    backend.sync()?;
    let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    Ok(elapsed / u64::from(LAUNCHES_PER_SAMPLE))
}

/// The best of [`TIMED_REPS`] samples, or a refusal.
///
/// The minimum, not the mean: the question is how fast this shape can go, and
/// every other sample is that plus noise from something else on the device.
///
/// Two numbers come back, taken the same way over the same pipeline: the
/// shape at the caller's real probe extent, and the shape at [`FLOOR_EXTENT`],
/// which every kernel in a tunable family declines before it touches memory.
/// The second is the launch overhead the first contains.
///
/// Every failure in this window is attributed to `shape`, including one
/// reported by `sync` rather than by the launch itself. That is not a guess.
/// A launch that is over its lane count or its threadgroup memory budget is
/// refused when it is encoded, while a fault in the kernel body surfaces when
/// the command buffer is committed - and the only work in this buffer is this
/// shape's, because the window opens and closes on a drained queue.
fn time_shape<S, L>(backend: &dyn Backend, shape: S, probe_extent: u32, launch: &L) -> Trial
where
    S: TunableShape,
    L: Fn(S, u32) -> CoreResult<()>,
{
    /// Logs a refusal without ending the tuner: one bad shape is a dropped
    /// candidate, not a failed load.
    fn refuse<S: TunableShape>(shape: S, error: &CoreError) -> Trial {
        tracing::warn!(
            entry = shape.entry(),
            lanes = shape.lanes(),
            shape = ?shape,
            error = %error.chain(),
            "the device refused a launch shape; it will not be considered",
        );
        Trial::REFUSED
    }

    /// [`TIMED_REPS`] samples at one probe extent, as the best of them and
    /// how far apart they landed.
    ///
    /// Both, because they answer different questions and only one of them used
    /// to be kept. The minimum is how fast this shape can go; the spread is how
    /// much the answer moves when nothing about the shape has changed, which is
    /// the only honest scale on which to call two shapes different.
    fn best_of<S, L>(
        backend: &dyn Backend,
        shape: S,
        probe_extent: u32,
        launch: &L,
    ) -> CoreResult<(u64, u64)>
    where
        S: TunableShape,
        L: Fn(S, u32) -> CoreResult<()>,
    {
        let mut samples = Vec::with_capacity(TIMED_REPS as usize);
        for _ in 0..TIMED_REPS {
            samples.push(sample(backend, shape, probe_extent, launch)?);
        }
        // Unreachable while `TIMED_REPS` is positive, which the assertion
        // beside it makes a build failure to change. Loud rather than
        // `unwrap_or(u64::MAX)`: a fabricated minimum would take its place in
        // the table as an ordinary timing, and the fastest row is what every
        // other row is then judged against.
        let best = samples
            .iter()
            .copied()
            .min()
            .expect("TIMED_REPS samples were taken, so there is a fastest one");
        Ok((best, sample_spread(&mut samples)))
    }

    for _ in 0..WARMUP_REPS {
        if let Err(error) = launch(shape, probe_extent) {
            return refuse(shape, &error);
        }
    }
    // Launches accumulate into an open command buffer, so the queue has to be
    // drained before the clock starts - otherwise the first sample carries the
    // warm-up's execution and every later one carries nothing at all.
    if let Err(error) = backend.sync() {
        return refuse(shape, &error);
    }

    let (nanos, spread) = match best_of(backend, shape, probe_extent, launch) {
        Ok(measured) => measured,
        Err(error) => return refuse(shape, &error),
    };
    // The floor is measured after the work, on a pipeline the work has already
    // warmed, so nothing it reports is first-launch cost. It is cheap by
    // construction - these launches return immediately - so probing every
    // shape rather than one costs a few milliseconds across the table.
    let floor = best_of(backend, shape, FLOOR_EXTENT, launch);
    if let Err(error) = &floor {
        // A floor that cannot be taken is not a reason to drop a shape that
        // ran: the timing above is still a timing, only now without the
        // number that says how much of it was overhead.
        tracing::debug!(
            entry = shape.entry(),
            error = %error.chain(),
            "no launch floor for this shape; its timing stands unannotated",
        );
    }
    Trial {
        nanos: Some(nanos),
        spread: Some(spread),
        floor: floor.ok().map(|(best, _)| best),
        refused: false,
    }
}

/// The interquartile range of one shape's timing samples.
///
/// The spread the ordering is exposed to: what the same measurement of the
/// same shape does when nothing has changed. Sorts in place, because the
/// caller has no further use for the order it handed in.
///
/// Interquartile rather than largest-minus-smallest, and that is the whole
/// reason this is a function and not an expression. Over twenty samples the
/// extremes are one scheduling hiccup each - some other process taking the
/// GPU for a frame - so a max-minus-min band would be set by the worst thing
/// that happened during the probe and would grow every time the probe was run
/// for longer. The middle half is the part that describes the measurement
/// rather than its interruptions.
///
/// Zero for fewer than four samples, where there are no quartiles to take.
/// That disables quantising, which is the right answer: a spread estimated
/// from three numbers would be a worse input than none.
fn sample_spread(samples: &mut [u64]) -> u64 {
    samples.sort_unstable();
    let count = samples.len();
    if count < 4 {
        return 0;
    }
    samples[count * 3 / 4].saturating_sub(samples[count / 4])
}

/// The table a device that could not be timed at all still produces.
///
/// The limits without the timings. A caller whose scratch allocation failed has
/// nothing to launch and so nothing to measure, but what the device says it
/// would accept is still worth reporting: [`choose_shape`] falls back on width
/// alone, which is a documented path through it rather than a corner.
pub fn unmeasured<S: TunableShape>(backend: &dyn Backend, shapes: &[S]) -> Vec<Candidate<S>> {
    shapes
        .iter()
        .map(|&shape| Candidate {
            shape,
            limit: backend.max_block_threads(shape.kernel()),
            nanos: None,
            refused: false,
        })
        .collect()
}

/// Times every shape this device will run, at the caller's own geometry.
///
/// One row per shape, in the order they were given, whatever happened to each:
/// a shape too wide for its own limit is skipped, one the device turned away at
/// dispatch is marked refused, and the rest carry a timing. Those three
/// outcomes are what [`choose_shape`] reads, and they are kept distinct because
/// a refusal and a skip look identical from `nanos` alone - and a selector that
/// confused them would keep proposing a kernel the device has already declined
/// to run.
///
/// `launch` issues exactly one dispatch of the shape it is handed at the
/// probe extent it is handed, over scratch the caller has already allocated
/// and holds for the duration. It is called with `probe_extent` for the timed
/// runs and with [`FLOOR_EXTENT`] for the floor probe, and nothing else about
/// a family crosses into this module.
///
/// Every shape is probed. Viability is not monotone in lane count: a measured
/// device has reported a *larger* limit for a *wider* shape than for its
/// narrower sibling, so stopping at the first shape that does not fit would
/// drop shapes that do.
///
/// Nothing here fails. A device that will run none of them produces a table
/// of skips and refusals, and turning that into a load failure is
/// [`choose_shape`]'s job, where the message can name what was wanted.
///
/// A timing is a per-launch figure amortised over `LAUNCHES_PER_SAMPLE`
/// chained launches, and the `debug` line beside it carries the launch floor
/// the figure still contains. Read the two together: a row whose timing is
/// close to its floor is a row where this device's launch overhead, not this
/// kernel, is most of what was measured. The floor is reported and never
/// calibrated against - it describes a different fact from the one the tie
/// band needs.
///
/// The timing that reaches a [`Candidate`] is snapped to whole tie bands above
/// the fastest row in the table (see `tie_band` and `quantise_from`); the
/// raw figure is logged beside it. The rule that produces is exactly the one
/// wanted: *the winner is the fastest candidate, and any candidate
/// indistinguishable from it defers to caller ordering.*
///
/// That is deliberately done here and not in [`choose_shape`]: the measurement
/// layer is the only one that knows how noisy its own numbers are, and the
/// decision layer should go on comparing plain numbers.
pub fn measure<S, L>(
    backend: &dyn Backend,
    shapes: &[S],
    probe_extent: u32,
    launch: L,
) -> Vec<Candidate<S>>
where
    S: TunableShape,
    L: Fn(S, u32) -> CoreResult<()>,
{
    // Measured first, judged second. The band is a property of the table as a
    // whole, so no row can be quantised until every row has been taken.
    let rows: Vec<(S, Option<u32>, Trial)> = shapes
        .iter()
        .map(|&shape| {
            let limit = backend.max_block_threads(shape.kernel());
            let wide_enough = matches!(limit, Some(limit) if limit >= shape.lanes());
            let trial = if wide_enough {
                time_shape(backend, shape, probe_extent, &launch)
            } else {
                Trial::SKIPPED
            };
            (shape, limit, trial)
        })
        .collect();

    let band = tie_band(rows.iter().map(|&(_, _, trial)| trial.spread));
    // The fastest row is what every other row is judged against. Only a timed
    // row can be the reference, and only a row that passed the lane check was
    // ever timed, so this is the minimum over exactly the viable candidates
    // without having to ask which those are a second time.
    let best = rows
        .iter()
        .filter_map(|&(_, _, trial)| trial.nanos)
        .min()
        .unwrap_or(0);
    rows.into_iter()
        .map(|(shape, limit, trial)| {
            let nanos = trial.nanos.map(|nanos| quantise_from(nanos, best, band));
            // Both figures, always, plus how many bands from the fastest this
            // row landed. A winner that changed because of the band is the
            // first thing somebody reading a slow-decode report needs to see,
            // and a quantised number on its own hides that it moved.
            tracing::debug!(
                entry = shape.entry(),
                lanes = shape.lanes(),
                limit,
                raw_nanos = trial.nanos,
                nanos,
                bands_above_best = trial.nanos.map(|nanos| bands_above(nanos, best, band)),
                spread_nanos = trial.spread,
                floor_nanos = trial.floor,
                band_nanos = band,
                refused = trial.refused,
                "launch shape measured",
            );
            Candidate {
                shape,
                limit,
                nanos,
                refused: trial.refused,
            }
        })
        .collect()
}

/// The width of the band inside which two timings are called equal.
///
/// The median per-shape sample spread on the table. Each row's spread is the
/// interquartile range of the [`TIMED_REPS`] samples that row's own minimum
/// was taken from (see [`sample_spread`]), so it says how much that timing
/// moves when nothing about the shape has changed. A difference between two
/// rows smaller than that is not a result; it is one measurement taken twice.
///
/// # Why the samples and not the launch floors
///
/// An earlier form of this took the spread of the launch floors, and it was
/// calibrated to the wrong quantity by about two orders of magnitude. Floor
/// probes are short, uniform, and return before touching memory, so they
/// cluster tightly - a measured table had all of them inside 213 ns - while
/// the real timings they were meant to describe moved 21 us on one shape
/// across three runs of the same probe. A band that narrow suppresses almost
/// nothing. The samples are the scatter the ordering is actually exposed to,
/// and the tuner was already collecting twenty of them per shape and
/// throwing all but the minimum away.
///
/// # Why the median of the rows and not the widest of them
///
/// The widest is the tempting choice, because a comparison is only as
/// trustworthy as the noisier of the two numbers in it and the failure
/// directions are not symmetric: a band too wide declares a tie, which falls
/// through to the caller's ordering and picks the conventional shape, while a
/// band too narrow invents an ordering out of noise and states it as a
/// finding. A fallback against a wrong answer - so lean wide.
///
/// It was written that way first and the measurements argued it down. Noise
/// here is not uniform across rows: on one measured table the spreads were
/// 98067, 46142 and 4412 ns, a factor of twenty-two between the noisiest row
/// and the quietest. The noisiest was the *winner*, whose 98 us interquartile
/// range sat on a 95 us minimum, and taking the widest let that one row erase
/// the whole table - a shape genuinely 66% slower came out tied with the
/// one that beat it. Leaning wide is not free; it is paid for in results the
/// tuner had and threw away.
///
/// The median is the noise of a typical row, which is the scale most pairs on
/// the table are actually compared at. On the same four measured tables it
/// kept every winner the widest produced, tied the pair that genuinely could
/// not be told apart, and kept the 63 us difference that could. An even count
/// takes the upper of the two middles, which is the one concession back to
/// leaning wide.
///
/// Zero when no row could report a spread, which leaves timings untouched -
/// with nothing to estimate the scatter from, inventing a band would be worse
/// than believing the numbers.
fn tie_band(spreads: impl IntoIterator<Item = Option<u64>>) -> u64 {
    let mut spreads: Vec<u64> = spreads.into_iter().flatten().collect();
    if spreads.is_empty() {
        return 0;
    }
    spreads.sort_unstable();
    spreads[spreads.len() / 2]
}

/// Whole bands from `best` to `nanos`, which is how far out of the running
/// this row is.
///
/// Zero means "indistinguishable from the fastest thing measured", which is
/// the only distinction that decides an outcome.
fn bands_above(nanos: u64, best: u64, band: u64) -> u64 {
    if band == 0 {
        return 0;
    }
    nanos.saturating_sub(best) / band
}

/// `nanos` snapped down to whole bands *above the table's fastest row*.
///
/// The grid is anchored at `best` rather than at zero, and that is the whole
/// of this function. Quantising to absolute multiples ties two values that
/// land in the same band, which is not the same as tying two values within a
/// band *of each other*: a pair straddling a multiple stays ordered however
/// close it is. That failure was reachable on real data - a measured pair 2742
/// ns apart under a band of 2587 fell either side of a boundary - so the
/// boundary is put where it cannot do that. Anchored at the minimum, every row
/// within one band of the fastest maps onto the fastest exactly, and the
/// comparison that decides the winner is always against a value sitting at the
/// bottom of its own band.
///
/// The result is still nanoseconds, and deliberately so: it is the fastest
/// time observed plus however many whole bands this shape is above it. Storing
/// the band count itself would express the same ordering, but it would put "2
/// bands above the fastest" in a field called `nanos`, and the next reader
/// would be entitled to believe that number was a time.
///
/// This also sidesteps the trap a pairwise epsilon comparison sets. "Within
/// one band" is not transitive - `a` near `b` and `b` near `c` does not make
/// `a` near `c` - so comparing pairs would make the answer depend on the order
/// they were visited. Every row here is compared against one fixed reference
/// instead, which cannot be intransitive.
///
/// Two rows that come out equal are then settled by [`choose_shape`]'s
/// strict `<`, which leaves the incumbent standing - the earlier entry in the
/// caller's array. That is the caller-ordering tie-break Task 6 already
/// documented and tested for untimed rows, reached by falling through it
/// rather than by adding a second ordering rule beside it.
fn quantise_from(nanos: u64, best: u64, band: u64) -> u64 {
    if band == 0 {
        return nanos;
    }
    best.saturating_add(bands_above(nanos, best, band).saturating_mul(band))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in for a family's shape enum.
    ///
    /// The selection cases are about the ladder and not about GDN, so they are
    /// written against a shape that is nothing but the three facts the trait
    /// asks for. That the ladder can be exercised without a kernel anywhere in
    /// sight is the property this file exists to have.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Shape {
        /// The entry point, which here is only a label to tell rows apart.
        entry: &'static str,
        /// The width the row is judged against its limit at.
        lanes: u32,
    }

    impl TunableShape for Shape {
        fn entry(self) -> &'static str {
            self.entry
        }

        fn lanes(self) -> u32 {
            self.lanes
        }

        fn module(self) -> &'static str {
            "test"
        }
    }

    /// The widths two independent families might offer: `A` a three-step
    /// ladder plus a 128-lane fallback that always fits, `B` a four-step
    /// ladder plus its own fallback. Neither is GDN's actual table - the
    /// selection cases are about the ladder, not about any one family's
    /// kernels - but the shape is the one a real family has: a reference
    /// width, narrower and wider siblings, and a fallback that is never the
    /// fastest choice but is always a viable one. Two pairs share a lane
    /// count on purpose - that is what makes the tie cases below reachable.
    const A_NARROW: Shape = Shape {
        entry: "a_narrow",
        lanes: 128,
    };
    const A_MID: Shape = Shape {
        entry: "a_mid",
        lanes: 256,
    };
    const A_WIDE: Shape = Shape {
        entry: "a_wide",
        lanes: 512,
    };
    const A_FALLBACK: Shape = Shape {
        entry: "a_fallback",
        lanes: 128,
    };
    const B_NARROW: Shape = Shape {
        entry: "b_narrow",
        lanes: 128,
    };
    const B_WIDE: Shape = Shape {
        entry: "b_wide",
        lanes: 512,
    };
    const B_WIDEST: Shape = Shape {
        entry: "b_widest",
        lanes: 1024,
    };
    const B_FALLBACK: Shape = Shape {
        entry: "b_fallback",
        lanes: 128,
    };

    /// What a family would name in the error a dry table produces. Its words
    /// are the caller's, so the cases supply one rather than asserting on it.
    const WHAT: &str = "a threadgroup shape this device will launch";

    /// [`choose_shape`] with the capability name every case here shares.
    fn choose(reference: Shape, candidates: &[Candidate<Shape>]) -> CoreResult<Shape> {
        choose_shape(WHAT, reference, candidates)
    }

    /// Builds a row with a trial dispatch that was not refused, which is
    /// every test's baseline: the refusal path gets its own test below.
    fn candidate(shape: Shape, limit: Option<u32>, nanos: Option<u64>) -> Candidate<Shape> {
        Candidate {
            shape,
            limit,
            nanos,
            refused: false,
        }
    }

    /// A backend with no opinion leaves the reference standing, which is
    /// exactly the behaviour a build had before any of this existed.
    #[test]
    fn no_opinion_anywhere_keeps_the_reference() {
        let table = [
            candidate(A_NARROW, None, None),
            candidate(A_MID, None, None),
            candidate(A_WIDE, None, None),
        ];
        assert_eq!(choose(A_WIDE, &table).expect("a choice"), A_WIDE);
    }

    /// A shape that reported nothing while its siblings reported numbers
    /// is one this device will not build, so it is not a candidate.
    #[test]
    fn a_silent_variant_among_speaking_ones_is_dropped() {
        let table = [
            candidate(A_NARROW, Some(1024), Some(900)),
            candidate(A_MID, None, None),
            candidate(A_WIDE, Some(448), None),
        ];
        assert_eq!(choose(A_WIDE, &table).expect("a choice"), A_NARROW);
    }

    /// Fastest wins, not widest: the whole reason the tuner times anything.
    #[test]
    fn the_fastest_timed_variant_wins() {
        let table = [
            candidate(A_NARROW, Some(1024), Some(900)),
            candidate(A_MID, Some(1024), Some(400)),
            candidate(A_WIDE, Some(1024), Some(650)),
        ];
        assert_eq!(choose(A_WIDE, &table).expect("a choice"), A_MID);
    }

    /// A shape too wide for its own limit is not viable however fast a
    /// number happens to be sitting next to it.
    #[test]
    fn a_variant_wider_than_its_limit_is_not_viable() {
        let table = [
            candidate(A_NARROW, Some(128), Some(900)),
            candidate(A_WIDE, Some(448), Some(10)),
        ];
        assert_eq!(choose(A_WIDE, &table).expect("a choice"), A_NARROW);
    }

    /// The last resort is still a result.
    #[test]
    fn only_the_fallback_surviving_is_a_choice_not_a_failure() {
        let table = [
            candidate(A_NARROW, Some(96), None),
            candidate(A_FALLBACK, Some(256), Some(5_000)),
        ];
        assert_eq!(choose(A_WIDE, &table).expect("a choice"), A_FALLBACK);
    }

    /// Nothing viable is a load failure, named, rather than a launch that
    /// the device refuses one request later.
    #[test]
    fn nothing_viable_is_an_error() {
        let table = [
            candidate(A_NARROW, Some(96), None),
            candidate(A_FALLBACK, Some(64), None),
        ];
        assert!(matches!(
            choose(A_WIDE, &table),
            Err(CoreError::Unsupported { .. }),
        ));
    }

    /// A limit wide enough is not the whole story: threadgroup memory is
    /// only checked at dispatch, so a shape the caller actually tried to
    /// launch and had refused must be dropped even though its limit and its
    /// timing both look fine sitting in the table.
    #[test]
    fn a_refused_variant_is_not_viable_however_fast_it_timed() {
        let table = [
            Candidate {
                shape: B_FALLBACK,
                limit: Some(1024),
                nanos: Some(1),
                refused: true,
            },
            candidate(B_NARROW, Some(128), Some(500)),
        ];
        assert_eq!(choose(B_WIDEST, &table).expect("a choice"), B_NARROW);
    }

    /// Two viable rows that neither one got timed - [`measure`]'s documented
    /// fallback when the caller's scratch allocation itself fails - have only
    /// their lane count to be judged by, and the wider one is the better bet.
    #[test]
    fn the_wider_of_two_untimed_variants_wins() {
        let table = [
            candidate(B_NARROW, Some(128), None),
            candidate(B_WIDE, Some(512), None),
        ];
        assert_eq!(choose(B_WIDEST, &table).expect("a choice"), B_WIDE);
    }

    /// `A_NARROW` and `A_FALLBACK` both ask for 128 lanes, so an untimed
    /// table cannot break the tie on width. It resolves to whichever the
    /// caller listed first, and every caller lists a family best-first with
    /// its fallback last - so the ordinary `A_NARROW` must win here, not the
    /// fallback it is tied with.
    #[test]
    fn a_tie_between_untimed_variants_goes_to_the_earlier_entry() {
        let table = [
            candidate(A_NARROW, Some(128), None),
            candidate(A_FALLBACK, Some(128), None),
        ];
        assert_eq!(choose(A_WIDE, &table).expect("a choice"), A_NARROW);
    }

    // -- the tie band, which is `measure`'s and not `choose_shape`'s ---------

    /// A band width, as a test input and nothing more.
    ///
    /// It is not a spread any current probe reports: it came from a superseded
    /// round that derived the band from the launch floors, which cluster two
    /// orders of magnitude tighter than the timings they were meant to
    /// describe. Measured sample spreads run 3846 to 116354 ns and the bands
    /// [`tie_band`] takes from them 11596 to 40667 ns, so nothing below should
    /// be read as a report of this machine. The cases are about the
    /// arithmetic - what quantising does to a pair inside one band and to a
    /// pair outside it - and that arithmetic is scale-free, which is why a
    /// small band and small gaps keep them legible.
    const MEASURED_BAND: u64 = 2_587;

    /// The band is the typical row's noise, not the worst row's.
    ///
    /// The literals are a measured table: one row's samples scattered over
    /// 98 us while another's stayed inside 4.4 us, and letting the first set
    /// the band erased a difference between two other rows that was real.
    #[test]
    fn the_band_is_the_median_sample_spread() {
        assert_eq!(tie_band([Some(98_067), Some(46_142), Some(4_412)]), 46_142);
        // A row that could not report one is not counted as quiet.
        assert_eq!(tie_band([None, Some(2_587)]), 2_587);
        // An even count takes the upper middle, the concession to leaning wide.
        assert_eq!(tie_band([Some(900), Some(2_587)]), 2_587);
        // A table where nothing could be measured cannot say how noisy it is,
        // so it says nothing rather than guessing.
        assert_eq!(tie_band([None, None]), 0);
        // A band of zero has to leave the timings alone rather than divide.
        assert_eq!(quantise_from(110_145, 110_145, 0), 110_145);
    }

    /// The spread is the middle half, so one hiccup does not set the band.
    ///
    /// The point of an interquartile range here: the twentieth sample being
    /// ten times the others is some other process taking the GPU for a frame,
    /// and a band set by that would grow every time the probe ran longer. The
    /// two sets below differ only in their extremes and must agree.
    #[test]
    fn the_spread_ignores_the_extremes() {
        let mut calm: Vec<u64> = (0..20).map(|i| 100_000 + i * 100).collect();
        let mut hiccup = calm.clone();
        hiccup[0] = 1;
        hiccup[19] = 10_000_000;

        let quiet = sample_spread(&mut calm);
        assert_eq!(quiet, sample_spread(&mut hiccup));
        // Quartiles of twenty evenly spaced samples: the 16th less the 6th.
        assert_eq!(quiet, 1_000);
        // Too few samples to have quartiles at all is answered with silence.
        assert_eq!(sample_spread(&mut [5, 900_000]), 0);
    }

    /// Two timings inside one band are equal by the time they are compared,
    /// so the choice falls through to the caller's ordering and the earlier
    /// entry stands.
    ///
    /// This is the behaviour the band exists to produce. The first figure is
    /// what an M1 Max reported for `_s1` at one token; the second is a
    /// neighbour 855 ns above it. That is the case worth suppressing: an
    /// earlier run of this probe, before the launch floor was amortised away,
    /// confidently ordered the two 128-lane recurrent shapes the *opposite*
    /// way by 138 us. A gap that changes sign between runs is not an ordering,
    /// and quantising is what stops the tuner presenting one as though it
    /// were.
    #[test]
    fn two_timings_inside_one_band_resolve_to_the_earlier_entry() {
        let raw = [110_145u64, 111_000];
        let best = raw.iter().copied().min().expect("a fastest row");
        let quantised: Vec<u64> = raw
            .iter()
            .map(|&ns| quantise_from(ns, best, MEASURED_BAND))
            .collect();
        assert_eq!(
            quantised[0], quantised[1],
            "two timings inside one band must come out equal",
        );
        let table = [
            candidate(A_NARROW, Some(832), Some(quantised[0])),
            candidate(A_FALLBACK, Some(384), Some(quantised[1])),
        ];
        assert_eq!(
            choose(A_WIDE, &table).expect("a choice"),
            A_NARROW,
            "a tie resolves to the earlier entry, which callers list best-first",
        );
    }

    /// A pair closer than the band that straddles a multiple of it still ties.
    ///
    /// This is the case the previous form got wrong, and the reason the grid
    /// is anchored at the table's fastest row rather than at zero. Quantising
    /// to absolute multiples asks "do these land in the same band", which is
    /// not the question; the question is "is this within a band of the best".
    /// 110145 and 111500 are 1355 ns apart under a band of 2587 - barely half
    /// a band - but 111241 is 43 x 2587, so the absolute grid put a boundary
    /// between them and ordered them anyway. Anchored at the minimum there is
    /// no boundary to straddle, because the minimum is by construction at the
    /// bottom of its own band.
    #[test]
    fn a_pair_straddling_a_band_boundary_still_ties() {
        let raw = [110_145u64, 111_500];
        assert!(
            raw[1] - raw[0] < MEASURED_BAND,
            "the pair has to be closer than one band for this to be the case",
        );
        // The absolute grid this replaced would have separated them.
        assert_ne!(
            (raw[0] / MEASURED_BAND) * MEASURED_BAND,
            (raw[1] / MEASURED_BAND) * MEASURED_BAND,
            "the pair has to straddle a multiple for this to be the case",
        );

        let best = raw.iter().copied().min().expect("a fastest row");
        let table = [
            candidate(
                A_NARROW,
                Some(832),
                Some(quantise_from(raw[0], best, MEASURED_BAND)),
            ),
            candidate(
                A_FALLBACK,
                Some(384),
                Some(quantise_from(raw[1], best, MEASURED_BAND)),
            ),
        ];
        assert_eq!(table[0].nanos, table[1].nanos, "they must now tie");
        assert_eq!(choose(A_WIDE, &table).expect("a choice"), A_NARROW);
    }

    /// The tie is with the *fastest* row, not with whichever row came first.
    ///
    /// A slower row that happens to be listed earlier must not drag the
    /// reference up to itself: the band is measured from the best thing on the
    /// table, so a shape a comfortable distance behind it still loses even
    /// though the row above it in the array is nearer.
    #[test]
    fn the_band_is_measured_from_the_fastest_row_not_the_first() {
        // Listed worst-first on purpose. 200000 is over thirty bands above
        // 110145, so nothing about the order it is visited in can save it.
        let raw = [200_000u64, 110_145, 111_000];
        let best = raw.iter().copied().min().expect("a fastest row");
        let table = [
            candidate(
                A_MID,
                Some(832),
                Some(quantise_from(raw[0], best, MEASURED_BAND)),
            ),
            candidate(
                A_NARROW,
                Some(832),
                Some(quantise_from(raw[1], best, MEASURED_BAND)),
            ),
            candidate(
                A_FALLBACK,
                Some(384),
                Some(quantise_from(raw[2], best, MEASURED_BAND)),
            ),
        ];
        assert_ne!(table[0].nanos, table[1].nanos, "the slow row must not tie");
        assert_eq!(table[1].nanos, table[2].nanos, "the close pair must tie");
        assert_eq!(choose(A_WIDE, &table).expect("a choice"), A_NARROW);
    }

    /// A difference comfortably wider than the band survives quantising, so
    /// the band suppresses noise without suppressing the result.
    ///
    /// The figures are a real measurement - GDN's recurrent shapes at
    /// thirty-two tokens, where the always-fits fallback leads by 162 us,
    /// sixty bands - carried over onto these stand-in shapes. It wins from the
    /// *later* position in the array, which is what makes this the converse of
    /// the case above rather than a restatement of it.
    #[test]
    fn a_difference_wider_than_the_band_still_picks_the_faster() {
        let raw = [944_958u64, 782_462];
        let best = raw.iter().copied().min().expect("a fastest row");
        let table = [
            candidate(
                A_NARROW,
                Some(832),
                Some(quantise_from(raw[0], best, MEASURED_BAND)),
            ),
            candidate(
                A_FALLBACK,
                Some(384),
                Some(quantise_from(raw[1], best, MEASURED_BAND)),
            ),
        ];
        assert_eq!(choose(A_WIDE, &table).expect("a choice"), A_FALLBACK);
    }
}
