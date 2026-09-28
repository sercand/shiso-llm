//! Which shape of a GDN kernel this device runs fastest.
//!
//! The lane split of the two recurrent kernels is not a constant: it is a
//! register-budget question, and the budget is a property of the device and
//! of the compiled pipeline. A number that fits one part is a refused launch
//! on a smaller one and money left on the table on a larger one.
//!
//! Every variant produces the same bits (see `q35_canonical4`), so which one
//! a machine picks varies performance and never output. That is what makes
//! it safe to measure at every startup and to let the answer differ between
//! two runs on one machine.
//!
//! # What is here and what is next door
//!
//! This file is qwen35's table and nothing else: nine entry points, the lane
//! count and threadgroup memory each was compiled for, the scratch one timing
//! run reads, and the chunk hybrid's seventh buffer. Deciding *from* that
//! table - probing the device, timing, quantising away differences smaller
//! than the measurement's own noise, and walking the fallback ladder - is
//! [`shiso_engine::tune`], which knows none of it.
//!
//! The split is not tidiness. The machinery is neither Metal-specific nor
//! qwen35-specific, and a gemma or deepseek table is a sibling of this file
//! rather than an edit to the tuner: it owes an enum implementing
//! [`TunableShape`], a closure that launches one of its shapes, and the floor
//! contract that closure honours. This file is the worked example of all three.
//!
//! # Where it is read
//!
//! [`GdnShape::choose`] is called once per stack, by [`super::Qwen35Kernels::choose`],
//! which [`crate::qwen35::Qwen35Model::load`] reaches before the stack is built;
//! the pair it returns is carried on the launcher and read by `gdn::recurrence`
//! at every launch. There is no fixed block constant behind it to fall back on:
//! the two that used to name 512 and 1024 lanes were deleted rather than
//! corrected, because a constant was the original bug and a corrected constant
//! is the same bug on a different part.

use shiso_core::backend::{Backend, KernelArg};
use shiso_core::buffer::{DeviceView, RawBuffer};
use shiso_core::{CoreResult, Extent3};

use shiso_engine::gdn::GdnGeometry;
use shiso_engine::launch::{self, Q35, RecurrentArgs};
use shiso_engine::tune::{self, Candidate, TunableShape};

/// Threadgroup bytes the recurrent hybrid stages on top of its scratch.
///
/// One quarter of the state row: 128 lanes each holding 32 floats. The
/// arithmetic is spelled out rather than written as a literal because the
/// *next* quarter is what does not fit, and a reader who can see the
/// factorisation can see why.
const RECURRENT_STAGE_BYTES: u32 = 128 * 32 * 4;

/// What the tuner is told when no GDN shape will run.
///
/// The words are this file's because the tuner does not know what it was
/// choosing among; an error naming only "a shape" would send a reader to the
/// selection code rather than to the kernels that could not be launched.
const NO_VIABLE_SHAPE: &str = "a GDN threadgroup shape this device will launch";

/// One compiled shape of a GDN kernel.
///
/// Every entry point below is Task 5's kernel compiled for a fixed number of
/// lanes; none of the nine differ in what they compute, only in how much of
/// the device's threadgroup budget they ask for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Variant {
    /// Recurrent, 128 lanes, four quarters per lane.
    S1,
    /// Recurrent, 256 lanes, two quarters per lane.
    S2,
    /// Recurrent, 512 lanes, one quarter per lane. The reference.
    S4,
    /// Recurrent, 128 lanes, one quarter of the state staged. Always fits.
    RecurrentHybrid,
    /// Chunk, 128 lanes.
    C1,
    /// Chunk, 256 lanes.
    C2,
    /// Chunk, 512 lanes.
    C4,
    /// Chunk, 1024 lanes. The reference.
    C8,
    /// Chunk, 128 lanes, deltas spilled to device memory. Always fits.
    ChunkHybrid,
}

impl Variant {
    /// The entry point this variant was compiled as.
    pub(crate) const fn entry(self) -> &'static str {
        match self {
            Self::S1 => "kernel_q35_gdn_recurrent_s1",
            Self::S2 => "kernel_q35_gdn_recurrent_s2",
            Self::S4 => "kernel_q35_gdn_recurrent_s4",
            Self::RecurrentHybrid => "kernel_q35_gdn_recurrent_hybrid",
            Self::C1 => "kernel_q35_gdn_chunk_c1",
            Self::C2 => "kernel_q35_gdn_chunk_c2",
            Self::C4 => "kernel_q35_gdn_chunk_c4",
            Self::C8 => "kernel_q35_gdn_chunk_c8",
            Self::ChunkHybrid => "kernel_q35_gdn_chunk_hybrid",
        }
    }

    /// The entry point that also records the state after every row, for a
    /// verify pass, or `None` for a variant with no recording twin - the
    /// chunked and hybrid shapes, which a verify-sized pass never picks.
    pub(crate) const fn trail_entry(self) -> Option<&'static str> {
        match self {
            Self::S1 => Some("kernel_q35_gdn_recurrent_trail_s1"),
            Self::S2 => Some("kernel_q35_gdn_recurrent_trail_s2"),
            Self::S4 => Some("kernel_q35_gdn_recurrent_trail_s4"),
            _ => None,
        }
    }

    /// Threads per threadgroup this variant must be launched at. Not a
    /// maximum and not a suggestion: the index arithmetic assumes it.
    pub(crate) const fn lanes(self) -> u32 {
        match self {
            Self::S1 | Self::RecurrentHybrid | Self::C1 | Self::ChunkHybrid => 128,
            Self::S2 | Self::C2 => 256,
            Self::S4 | Self::C4 => 512,
            Self::C8 => 1024,
        }
    }

    /// Threadgroup memory this variant's launch has to reserve.
    ///
    /// Not a property the device will report and not one the pipeline
    /// carries: dynamic threadgroup memory is bound by the *launch*, so
    /// getting it wrong is either a refused dispatch or a kernel reading past
    /// the end of its own scratch. Each number therefore comes from the
    /// kernel source rather than from a guess at what a shape would want.
    ///
    /// This is also why it is not one of the three facts [`TunableShape`]
    /// asks for: the tuner never encodes a launch itself, so the only code
    /// that needs the number is the dispatch below.
    ///
    /// The two hybrids are where the numbers stop being uniform. The
    /// recurrent hybrid stages *one quarter* of the state row - 128 lanes of
    /// 32 floats - beside the scratch it already had. Staging two quarters
    /// would be 33924 B, over the 32768 B every Apple Silicon GPU family is
    /// built to, so the quarter is a budget the family fixes and not a
    /// tuning knob. The chunk hybrid stages nothing extra at all: its deltas
    /// go to device memory, which is the whole reason it takes a seventh
    /// buffer.
    pub(crate) const fn smem_bytes(self) -> u32 {
        match self {
            Self::S1 | Self::S2 | Self::S4 => launch::RECURRENT_SMEM_BYTES,
            Self::RecurrentHybrid => launch::RECURRENT_SMEM_BYTES + RECURRENT_STAGE_BYTES,
            Self::C1 | Self::C2 | Self::C4 | Self::C8 | Self::ChunkHybrid => {
                launch::CHUNK_SMEM_BYTES
            }
        }
    }

    /// Whether this variant takes the seventh buffer, the spilled deltas.
    ///
    /// Exactly one does. Binding it to any other variant would put a buffer
    /// at an index that kernel does not declare, and leaving it off the one
    /// that needs it is a launch reading an unbound pointer, so the question
    /// is asked of the variant rather than answered at each call site.
    pub(crate) const fn spills_deltas(self) -> bool {
        matches!(self, Self::ChunkHybrid)
    }
}

impl TunableShape for Variant {
    /// Delegates to the inherent `const fn`, which is what the dispatch and
    /// the log lines already call: one spelling of the nine names, not two.
    fn entry(self) -> &'static str {
        Self::entry(self)
    }

    fn lanes(self) -> u32 {
        Self::lanes(self)
    }

    /// Every one of the nine is compiled into the qwen35 kernel set.
    fn module(self) -> &'static str {
        Q35
    }
}

/// The variant to dispatch, given what the probe and the benchmark found.
///
/// The ladder itself is [`tune::choose_shape`]; all this adds is the sentence
/// a device that will run nothing at all is refused with.
///
/// # Errors
///
/// [`shiso_core::CoreError::Unsupported`] when the device accepts no variant,
/// which ends the load rather than the first request.
pub(crate) fn choose_variant(
    reference: Variant,
    candidates: &[Candidate<Variant>],
) -> CoreResult<Variant> {
    tune::choose_shape(NO_VIABLE_SHAPE, reference, candidates)
}

/// The recurrent variants, in the order a tie resolves through.
///
/// Best first and the hybrid last, which is load-bearing rather than tidy:
/// [`tune::choose_shape`] settles an untimed tie at equal lane counts on the
/// earlier entry, and `s1` and the hybrid are both 128 lanes. Listed this way
/// the register-resident shape wins that tie; reordering the array is a change
/// to which kernel a device with no timings ships.
const RECURRENT: [Variant; 4] = [
    Variant::S1,
    Variant::S2,
    Variant::S4,
    Variant::RecurrentHybrid,
];

/// The chunk variants, on the same rule: best first, hybrid last.
const CHUNK: [Variant; 5] = [
    Variant::C1,
    Variant::C2,
    Variant::C4,
    Variant::C8,
    Variant::ChunkHybrid,
];

/// The shape the build has always dispatched the per-token walk at.
///
/// It is the reference, not a default: [`tune::choose_shape`] returns it only
/// when no candidate reported a limit at all, which is a backend that cannot
/// answer the probe rather than a device that answered narrowly.
const RECURRENT_REFERENCE: Variant = Variant::S4;

/// The same, for the 64-token run.
const CHUNK_REFERENCE: Variant = Variant::C8;

/// Environment override, diagnostic only: pins a shape and skips the
/// benchmark.
///
/// Follows `SHISO_METAL_SOURCE_DIR` (`shiso-metal/src/compile.rs`) in being the
/// knob you reach for to make an A/B measurable or a user's report
/// reproducible, and in not being something a normal run reads. It is spelled
/// as one or more entry-point suffixes separated by commas - `s1`, `c4`,
/// `s1,chybrid` - and a side nobody named keeps its reference shape, which is
/// the pre-tuner behaviour and not necessarily a shape this device will launch.
/// That is the point: pinning is how you reproduce a refusal as well as a
/// timing.
const VARIANT_ENV: &str = "SHISO_GDN_VARIANT";

/// The shapes this stack dispatches.
///
/// Two, because the block runs two different kernels and the fastest split of
/// one says nothing about the fastest split of the other: the per-token walk
/// carries a state row per lane and the 64-token run carries a delta array
/// beside it, so they hit the register budget at different widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GdnShape {
    /// The per-token walk, for decode and short prefill.
    pub recurrent: Variant,
    /// The 64-token run, for long prefill.
    pub chunk: Variant,
}

impl GdnShape {
    /// Measures the device and picks a shape for each kernel.
    ///
    /// Two benchmarks, at the two token counts the block is actually launched
    /// at: one token for the walk and a whole run for the chunk. Timing both at
    /// one width would rank the chunk variants on a length the chunk path never
    /// sees, since `gdn::recurrence` only reaches it at
    /// [`launch::GDN_CHUNK_MIN`] tokens and up.
    ///
    /// It costs launches, and the number is worth knowing before this is moved
    /// anywhere hotter than a load: [`tune::measure`] warms, times and probes
    /// the floor of every shape the device will run, which is 405 dispatches
    /// each - five warm-up, two hundred timed, two hundred at the floor width -
    /// and so up to 3645 across the nine. They are all launches of the block
    /// this stack is about to run at the length it runs them, so the bill is
    /// the same order as a few tokens of decode, and it is paid once per
    /// process rather than per request.
    ///
    /// # Errors
    ///
    /// [`shiso_core::CoreError::Unsupported`] when the device accepts no shape
    /// of either kernel, which ends the load rather than the first request.
    pub(crate) fn choose(backend: &dyn Backend, geometry: &GdnGeometry) -> CoreResult<Self> {
        if let Ok(pinned) = std::env::var(VARIANT_ENV) {
            if let Some(shape) = Self::pinned(&pinned) {
                tracing::info!(
                    %pinned,
                    recurrent = shape.recurrent.entry(),
                    chunk = shape.chunk.entry(),
                    "GDN shape pinned by the environment; the device was not measured",
                );
                return Ok(shape);
            }
            // Measured rather than refused: the pin is a diagnostic, and a
            // typo in one should not be the reason a server will not start.
            // Loud, though - the timings below are about to answer a question
            // nobody asked.
            tracing::warn!(%pinned, "unrecognised {VARIANT_ENV}; measuring instead");
        }

        let recurrent = choose_variant(
            RECURRENT_REFERENCE,
            &measure(backend, geometry, &RECURRENT, 1),
        )?;
        let chunk = choose_variant(
            CHUNK_REFERENCE,
            &measure(backend, geometry, &CHUNK, launch::GDN_CHUNK_RUN),
        )?;
        Ok(Self { recurrent, chunk })
    }

    /// The shape `pinned` names, or `None` if any part of it names nothing.
    ///
    /// Refused rather than ignored: a misspelled pin that quietly measured
    /// instead would report the timings of a shape nobody asked about, and the
    /// whole use of this knob is to know exactly which kernel ran.
    fn pinned(pinned: &str) -> Option<Self> {
        let mut shape = Self {
            recurrent: RECURRENT_REFERENCE,
            chunk: CHUNK_REFERENCE,
        };
        for token in pinned.split(',') {
            match token.trim() {
                "s1" => shape.recurrent = Variant::S1,
                "s2" => shape.recurrent = Variant::S2,
                "s4" => shape.recurrent = Variant::S4,
                "rhybrid" => shape.recurrent = Variant::RecurrentHybrid,
                "c1" => shape.chunk = Variant::C1,
                "c2" => shape.chunk = Variant::C2,
                "c4" => shape.chunk = Variant::C4,
                "c8" => shape.chunk = Variant::C8,
                "chybrid" => shape.chunk = Variant::ChunkHybrid,
                _ => return None,
            }
        }
        Some(shape)
    }
}

// ---------------------------------------------------------------------------------------
// The device half
// ---------------------------------------------------------------------------------------

/// Scale of the carried state the probe starts from.
///
/// Small, not zero. A zero state is a perfectly ordinary input to this
/// recurrence and a misleading one to time: the first token's update would
/// then be the only thing in the row, and a shape whose cost depends on what
/// it is carrying would be measured at its emptiest.
const STATE_SCALE: f32 = 0.01;

/// Seeds for the four filled rows.
///
/// Distinct so that two rows the recurrence multiplies together are not the
/// same sequence, which would make `<k, q>` a sum of squares and hide a sign
/// error behind a number that is right by symmetry.
const QKV_SEED: u32 = 0x5117_0001;
const STATE_SEED: u32 = 0x5117_0002;
const DECAY_SEED: u32 = 0x5117_0003;
const BETA_SEED: u32 = 0x5117_0004;

/// Slots of [`TuneScratch`], in the order the kernels bind them.
///
/// The kernel's own argument order, so that reading the dispatch below beside
/// `gdn.rs`'s `recurrence` shows the two binding the same things in the same
/// places. The seventh is the odd one out and is bound only by the variant
/// that declares it.
const SLOT_ATTN_OUT: usize = 0;
const SLOT_STATE: usize = 1;
const SLOT_QKV: usize = 2;
const SLOT_DECAY: usize = 3;
const SLOT_BETA: usize = 4;
const SLOT_SPILL: usize = 5;

/// A `u32` width as a host length.
///
/// Every width this module handles is a head count or a token count, so the
/// saturation can only be reached on a target whose `usize` is narrower than
/// the geometry - where the allocation that follows would fail anyway.
fn count(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

/// Deterministic `f32` fixture data: a small LCG mapped to `[-1, 1)`.
///
/// The same generator the kernel tests fill their fixtures from, written out
/// again here because a library crate cannot import from a test crate. Fixed
/// rather than random for the usual reason and for one more: a tuner that
/// measured a different input on every boot would be choosing between shapes
/// on a number that moved underneath it.
fn signal(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 0x9E37_79B9;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let bits = u16::try_from((state >> 8) & 0xFFFF).unwrap_or(0);
            f32::from(bits) / f32::from(u16::MAX) * 2.0 - 1.0
        })
        .collect()
}

/// The mix rows the delta rule reads, with the query and key halves
/// normalized the way `kernel_q35_gdn_l2norm` normalizes them upstream.
///
/// Normalizing is what keeps the recurrence bounded rather than merely
/// realistic: the update is only stable while `beta * <k, k>` stays near one,
/// and an unnormalized key row would drive the state to infinity within a
/// chunk. A timing run over infinities and NaNs is not a timing run of this
/// kernel - denormal and NaN handling is its own cost - so the fixture is the
/// kernel's real input range and not an arbitrary one.
fn qkv_fill(geometry: &GdnGeometry, tokens: usize) -> Vec<f32> {
    let stride = count(geometry.channels);
    let head = count(geometry.head_dim);
    let normalized = 2 * count(geometry.qk_heads);
    let mut rows = signal(tokens.saturating_mul(stride), QKV_SEED);
    for token in 0..tokens {
        for slot in 0..normalized {
            let at = token * stride + slot * head;
            let chunk = &mut rows[at..at + head];
            let norm = chunk.iter().map(|value| value * value).sum::<f32>().sqrt();
            // A zero row cannot come out of this generator, but dividing by a
            // norm without looking is how a fixture change turns into NaNs
            // nobody traces back to here.
            if norm > 0.0 {
                for value in chunk.iter_mut() {
                    *value /= norm;
                }
            }
        }
    }
    rows
}

/// The decay gate, in the range `kernel_q35_gdn_gates` leaves it in.
///
/// That kernel produces `softplus(alpha + bias) * a` with `a` negative, so
/// the value is negative and both consumers exponentiate it directly. A run
/// of 64 tokens multiplies 64 of those together, so the fixture stays in
/// `[-0.03, -0.01]`: a decay a chunk carry survives.
fn decay_fill(len: usize) -> Vec<f32> {
    signal(len, DECAY_SEED)
        .into_iter()
        .map(|value| -0.01 - 0.02 * (0.5 + 0.5 * value))
        .collect()
}

/// The write strength, inside the `(0, 1)` range a sigmoid leaves it in.
fn beta_fill(len: usize) -> Vec<f32> {
    signal(len, BETA_SEED)
        .into_iter()
        .map(|value| 0.2 + 0.6 * (0.5 + 0.5 * value))
        .collect()
}

/// The buffers one round of timing launches reads and writes.
///
/// Allocated through `alloc_raw` rather than through [`shiso_core::alloc`]
/// because the tuner is handed a `&dyn Backend` and that helper wants an
/// `Arc`. The ownership the helper provides is reproduced here instead: every
/// handle goes into the list the moment it exists, so a failure part way
/// through construction frees what came before rather than leaking it, and
/// `Drop` returns the rest.
///
/// Sized at the caller's own geometry. A synthetic shape would answer a
/// question nobody asked - the split that wins at 32 value heads is not
/// necessarily the one that wins at 8 - and the whole of it is about two
/// megabytes of state plus a few rows, which is not a memory event beside
/// the weights already resident.
///
/// # One allocation, read and rewritten by every shape in the table
///
/// [`measure`] allocates exactly one of these per call and every shape it
/// times - warm-up launches, timed samples and the floor probe alike -
/// dispatches against the same `SLOT_STATE` row rather than a fresh one per
/// shape. These kernels rewrite their state in place, so the first shape in
/// `RECURRENT`/`CHUNK` (`gdn_tune.rs`) is timed on [`STATE_SEED`]'s fill and
/// every shape after it is timed on whatever its predecessors left the row
/// holding. The state is not re-seeded between shapes, on purpose: the decay
/// this recurrence applies converges any starting row to the same O(1) fixed
/// point within a handful of tokens (see [`STATE_SCALE`] and [`decay_fill`]),
/// so nothing here has been observed to drift toward the denormals or
/// infinities that would actually bias a timing.
///
/// It is still worth saying plainly rather than leaving implicit, because
/// [`shiso_engine::tune::choose_shape`]'s tie-break already makes the *order* of
/// `RECURRENT`/`CHUNK` load-bearing for which shape wins an untimed or a
/// tied table - and this makes that same order quietly load-bearing for the
/// timings that feed it, too: whichever shape runs first sees a state no
/// prior launch has touched, and every shape after it does not. A future
/// change to a kernel's convergence behaviour - not just to the tuner - is
/// what would turn this from a documented non-issue into a real one, and the
/// fix at that point is a fresh `TuneScratch` (or at least a re-seeded state
/// row) per shape rather than per call.
#[derive(Debug)]
struct TuneScratch<'b> {
    /// The device the handles below belong to and are returned to.
    backend: &'b dyn Backend,
    /// Every allocation, indexed by the `SLOT_*` constants.
    raws: Vec<RawBuffer>,
}

impl<'b> TuneScratch<'b> {
    /// Allocates and fills every row a launch over `n_tok` tokens needs.
    ///
    /// The spill row is allocated for every measurement rather than only when
    /// the chunk hybrid is among the variants: it is one run of deltas per
    /// value head, a few megabytes at most, and a scratch whose contents
    /// depend on which variants the caller happened to list is a scratch that
    /// times differently depending on the list.
    ///
    /// # Errors
    ///
    /// [`shiso_core::CoreError::Backend`] when the device refuses an
    /// allocation or a host-to-device copy.
    fn new(backend: &'b dyn Backend, geometry: &GdnGeometry, n_tok: u32) -> CoreResult<Self> {
        let tokens = count(n_tok);
        let heads = count(geometry.v_heads);
        let head = count(geometry.head_dim);
        let mut scratch = Self {
            backend,
            raws: Vec::with_capacity(6),
        };
        // Order is the `SLOT_*` order, which is the kernels' binding order.
        scratch.push(&vec![0.0; tokens.saturating_mul(count(geometry.width))])?;
        scratch.push(
            &signal(heads.saturating_mul(head).saturating_mul(head), STATE_SEED)
                .into_iter()
                .map(|value| value * STATE_SCALE)
                .collect::<Vec<f32>>(),
        )?;
        scratch.push(&qkv_fill(geometry, tokens))?;
        scratch.push(&decay_fill(tokens.saturating_mul(heads)))?;
        scratch.push(&beta_fill(tokens.saturating_mul(heads)))?;
        scratch.push(&vec![
            0.0;
            heads
                .saturating_mul(count(launch::GDN_CHUNK_RUN))
                .saturating_mul(head)
        ])?;
        Ok(scratch)
    }

    /// Allocates one row and uploads it.
    fn push(&mut self, data: &[f32]) -> CoreResult<()> {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        let raw = self.backend.alloc_raw(len)?;
        // Recorded before the upload, so a copy that fails still frees.
        self.raws.push(raw);
        self.backend
            .copy_h2d(DeviceView::from_raw(raw, 0, len), bytes)
    }

    /// The whole of one slot, as a kernel argument.
    fn view(&self, slot: usize) -> DeviceView<'_> {
        let raw = self.raws[slot];
        DeviceView::from_raw(raw, 0, raw.bytes())
    }
}

impl Drop for TuneScratch<'_> {
    /// Returns every allocation. A `Drop` cannot report, so a refused free is
    /// logged rather than swallowed: leaked device memory should be visible in
    /// the trace even where it cannot be handled.
    fn drop(&mut self) {
        for raw in self.raws.drain(..) {
            if let Err(error) = self.backend.free_raw(raw) {
                tracing::error!(
                    id = %raw.id(),
                    bytes = raw.bytes(),
                    error = %error.chain(),
                    "the GDN tuner could not free its scratch",
                );
            }
        }
    }
}

/// The argument block for one launch, at a stated head width.
///
/// `head_dim` is a parameter rather than simply `geometry.head_dim` so that
/// the floor probe can pass [`shiso_engine::tune`]'s floor width and get a launch the
/// kernel declines. Every other field is the caller's real geometry either
/// way: the two launches differ in one number, so what separates their timings
/// is the work and not the setup.
fn recurrent_args(geometry: &GdnGeometry, n_tok: u32, head_dim: u32) -> RecurrentArgs {
    RecurrentArgs {
        head_dim,
        n_head_k: geometry.qk_heads,
        n_head_v: geometry.v_heads,
        k_offset: geometry.k_offset(),
        v_offset: geometry.v_offset(),
        stride: geometry.channels,
        n_tok,
    }
}

/// Issues one launch of `variant`, the way `gdn.rs` issues the reference one.
///
/// Field for field the same argument block, the same grid - one threadgroup
/// per value head - and the same buffer order; the variant supplies the entry
/// name, the lane count and the threadgroup memory, which is the entire
/// difference between the nine. The seventh buffer is bound only for the one
/// variant that declares it.
///
/// This is the whole of what [`shiso_engine::tune`] is handed: everything it does not
/// know about GDN - the buffers, the argument block, the grid, the seventh
/// buffer - is bound here and nowhere else.
///
/// # Errors
///
/// [`shiso_core::CoreError::Backend`] naming the entry point when the device
/// refuses the dispatch.
fn dispatch(
    backend: &dyn Backend,
    scratch: &TuneScratch<'_>,
    geometry: &GdnGeometry,
    variant: Variant,
    args: &RecurrentArgs,
) -> CoreResult<()> {
    let mut params = vec![
        KernelArg::Bytes(bytemuck::bytes_of(args)),
        KernelArg::Buffer(scratch.view(SLOT_ATTN_OUT)),
        KernelArg::Buffer(scratch.view(SLOT_STATE)),
        KernelArg::Buffer(scratch.view(SLOT_QKV)),
        KernelArg::Buffer(scratch.view(SLOT_DECAY)),
        KernelArg::Buffer(scratch.view(SLOT_BETA)),
    ];
    if variant.spills_deltas() {
        params.push(KernelArg::Buffer(scratch.view(SLOT_SPILL)));
    }
    launch::run(
        backend,
        TunableShape::kernel(variant),
        Extent3::linear(geometry.v_heads),
        Extent3::new(variant.lanes(), 1, 1),
        variant.smem_bytes(),
        &params,
    )
    .map_err(|err| launch::launch_failure(variant.entry(), err))
}

/// Times every variant this device will run, at this model's own geometry.
///
/// The measurement is [`tune::measure`]'s; what this adds is the scratch those
/// launches read and the closure that binds it. One row per variant, in the
/// order they were given, carrying whatever happened to each - skipped,
/// refused, or timed.
///
/// Nothing here fails. A device that will run none of them produces a table
/// of skips and refusals, and turning that into a load failure is
/// [`choose_variant`]'s job, where the message can name what was wanted.
///
/// Scratch that cannot be allocated is the one path that reports no timings at
/// all. The limits are still worth having: [`choose_variant`] falls back on
/// width alone, which is a documented path through it rather than a corner.
///
/// Nothing is allocated until the probe says something can be timed. A backend
/// that does not answer [`Backend::max_block_threads`] - every non-Metal one
/// today - would otherwise allocate a state row per head and free it again
/// without launching anything, which is a visible event on a device the load is
/// still filling.
pub(crate) fn measure(
    backend: &dyn Backend,
    geometry: &GdnGeometry,
    variants: &[Variant],
    n_tok: u32,
) -> Vec<Candidate<Variant>> {
    let any_viable = variants.iter().any(|variant| {
        matches!(
            backend.max_block_threads(TunableShape::kernel(*variant)),
            Some(limit) if limit >= variant.lanes(),
        )
    });
    if !any_viable {
        return tune::unmeasured(backend, variants);
    }

    let scratch = match TuneScratch::new(backend, geometry, n_tok) {
        Ok(scratch) => scratch,
        Err(error) => {
            tracing::warn!(
                error = %error.chain(),
                "the GDN tuner could not allocate; shapes will be chosen unmeasured",
            );
            return tune::unmeasured(backend, variants);
        }
    };

    tune::measure(backend, variants, geometry.head_dim, |variant, head_dim| {
        dispatch(
            backend,
            &scratch,
            geometry,
            variant,
            &recurrent_args(geometry, n_tok, head_dim),
        )
    })
}

#[cfg(test)]
mod device_tests {
    use super::*;

    use std::num::NonZeroU64;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use shiso_core::CoreError;
    use shiso_core::backend::{Backend, KernelRef, LaunchArgs};
    use shiso_core::buffer::{DeviceView, RawBuffer};

    use shiso_core::DType;

    use shiso_engine::gdn::GdnGeometry;
    // The architecture's widths, not the tuner's pair of variants: both are
    // called `GdnShape`, and this module has the tuner's one in scope through
    // `super::*`.
    use shiso_engine::shape::GdnShape as GdnWidths;

    /// The device the timing cases run on, or `None` with a note on stderr.
    ///
    /// Which variants a device will run is a property of the machine, so a
    /// machine with no Metal device at all skips rather than fails: the case
    /// has nothing to say there, and saying it loudly would turn every
    /// non-macOS check into a red build.
    #[expect(
        clippy::print_stderr,
        reason = "the device cases are opt-in, so the reason they did not run belongs on stderr"
    )]
    fn test_backend() -> Option<Arc<dyn Backend>> {
        if !shiso_metal::metal_available() {
            eprintln!("note: no Metal device on this machine, skipping the GDN tuner cases");
            return None;
        }
        let metal = shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");
        Some(metal)
    }

    /// The model's own GDN widths: 32 value heads over 16 key heads, at the
    /// 128-wide head every one of these kernels is built around.
    fn test_geometry() -> GdnGeometry {
        GdnGeometry::new(&GdnWidths::new(32, 16, 128, 4)).expect("a GDN geometry")
    }

    /// Every q35 kernel the stack dispatches at a block it does not measure,
    /// with the block it is dispatched at.
    ///
    /// Read off the launch sites rather than off the kernel sources: what this
    /// case is checking is the number the *host* asks for, so a table built
    /// from what the kernels were written for would agree with itself and miss
    /// exactly the disagreement that produced the original bug. The GDN
    /// recurrence is absent because it no longer has a fixed block; it is
    /// covered below, through the shape the tuner picks for it.
    const FIXED_BLOCKS: [(&str, u32); 12] = [
        ("kernel_q35_gdn_conv", launch::BLOCK_LINEAR),
        ("kernel_q35_gdn_l2norm", launch::BLOCK_128),
        ("kernel_q35_gdn_gates", launch::BLOCK_LINEAR),
        ("kernel_q35_gdn_out_gate", launch::BLOCK_128),
        ("kernel_q35_q_norm_rope", launch::BLOCK_128),
        ("kernel_q35_q_norm_rope_mrope", launch::BLOCK_128),
        ("kernel_q35_k_norm_rope_kv", launch::BLOCK_128),
        ("kernel_q35_k_norm_rope_kv_mrope", launch::BLOCK_128),
        ("kernel_q35_attn_decode", launch::BLOCK_LINEAR),
        ("kernel_q35_attn_prefill", launch::BLOCK_LINEAR),
        ("kernel_q35_attn_gate_mul", launch::BLOCK_LINEAR),
        ("kernel_get_rows_q6_K_f32", launch::GET_ROWS_BLOCK),
    ];

    /// Asserts one kernel's launch width against what the device allows.
    ///
    /// A kernel that does not resolve is a failure rather than a skip: every
    /// name here is one this build dispatches, so a missing limit means the
    /// pipeline the stack is about to launch does not exist.
    fn fits(backend: &dyn Backend, kernel: KernelRef, block: u32) {
        let limit = backend
            .max_block_threads(kernel)
            .unwrap_or_else(|| panic!("{} resolves on this device", kernel.entry));
        assert!(
            limit >= block,
            "{} is dispatched with {block} threads and this device allows {limit}",
            kernel.entry,
        );
    }

    /// Every shape the stack can dispatch is one this device will accept.
    ///
    /// This is the case the original bug needed: `kernel_q35_gdn_recurrent`
    /// asked for 512 threads on a device that allows 448, and nothing said so
    /// until a request arrived and the launch was refused. It is deliberately
    /// not `#[ignore]`d, unlike the timing cases around it - a gate that only
    /// runs when somebody remembers to ask for it is not a gate - and it skips
    /// rather than fails on a machine with no Metal device.
    ///
    /// Both halves of the dispatch are covered, because they fail differently.
    /// The kernels with a fixed block are checked against that constant, which
    /// is the shape of the original bug. The recurrence has no constant left to
    /// check, so what is checked there is the answer the tuner gives this
    /// machine - including, on a device where nothing fits, that `choose`
    /// refuses at load rather than handing back something unlaunchable.
    #[test]
    fn every_dispatched_shape_fits_this_device() {
        let Some(backend) = test_backend() else {
            return;
        };
        let backend = backend.as_ref();
        for (entry, block) in FIXED_BLOCKS {
            fits(backend, KernelRef::new(Q35, entry), block);
        }
        // The two matvec families reach the device through a specialization
        // table, and a pipeline's thread ceiling is a property of the
        // specialized function: asking about the unspecialized name would be
        // asking about a pipeline this crate never launches.
        let ext_block = launch::ext_block();
        for dtype in [DType::Q5_K, DType::Q6_K] {
            for r1ptg in 1..=5 {
                for nxpsg in [4, 8, 16] {
                    let (Some(entry), Some(consts)) = (
                        launch::ext_name(dtype, r1ptg),
                        launch::ext_consts_static(nxpsg),
                    ) else {
                        continue;
                    };
                    let kernel = KernelRef::specialized(Q35, entry, consts);
                    fits(backend, kernel, ext_block.x * ext_block.y * ext_block.z);
                }
            }
        }
        for dtype in [DType::Q4_K, DType::Q8_0, DType::F16] {
            let Some(entry) = launch::mm_name(dtype) else {
                continue;
            };
            let kernel = KernelRef::specialized(Q35, entry, launch::mm_consts());
            fits(backend, kernel, launch::GEMM_BLOCK);
        }

        let shape =
            GdnShape::choose(backend, &test_geometry()).expect("this device runs some GDN shape");
        for variant in [shape.recurrent, shape.chunk] {
            fits(backend, TunableShape::kernel(variant), variant.lanes());
        }
    }

    /// The probe fills a row per variant and times the ones the device
    /// will run. Which ones those are is a property of the device, so the
    /// assertion is about the shape of the table, not its contents.
    #[test]
    #[ignore = "requires a Metal device"]
    fn measuring_fills_a_row_per_variant() {
        let Some(backend) = test_backend() else {
            return;
        };
        let geometry = test_geometry();
        let variants = [Variant::S1, Variant::S2, Variant::S4];
        let table = measure(backend.as_ref(), &geometry, &variants, 1);

        assert_eq!(table.len(), variants.len());
        for (row, want) in table.iter().zip(variants) {
            assert_eq!(row.shape, want);
            // A viable row was timed; a non-viable one was not launched.
            let viable = matches!(row.limit, Some(limit) if limit >= want.lanes());
            assert_eq!(row.nanos.is_some(), viable, "{want:?}");
        }
        assert!(
            table.iter().any(|row| row.nanos.is_some()),
            "no recurrent variant was viable on this device",
        );
    }

    /// A backend wide enough for every lane count that refuses one entry
    /// point's dispatch.
    #[derive(Debug)]
    struct RefusingBackend {
        /// The entry whose launches are refused.
        refuse: &'static str,
        /// Hands out allocation ids; nothing behind them is ever read.
        next: AtomicU64,
    }

    impl Backend for RefusingBackend {
        fn backend_name(&self) -> &'static str {
            "refusing"
        }

        fn alloc_raw(&self, bytes: u64) -> CoreResult<RawBuffer> {
            let id = self.next.fetch_add(1, Ordering::Relaxed);
            let id = NonZeroU64::new(id).expect("a non-zero allocation id");
            Ok(RawBuffer::new(id, bytes))
        }

        fn free_raw(&self, _raw: RawBuffer) -> CoreResult<()> {
            Ok(())
        }

        fn copy_h2d(&self, _dst: DeviceView<'_>, _src: &[u8]) -> CoreResult<()> {
            Ok(())
        }

        fn copy_d2h(&self, _src: DeviceView<'_>, _dst: &mut [u8]) -> CoreResult<()> {
            Ok(())
        }

        fn copy_d2d(&self, _dst: DeviceView<'_>, _src: DeviceView<'_>) -> CoreResult<()> {
            Ok(())
        }

        fn max_block_threads(&self, _kernel: KernelRef) -> Option<u32> {
            Some(1024)
        }

        fn run(&self, kernel: KernelRef, _args: &LaunchArgs<'_>) -> CoreResult<()> {
            if kernel.entry == self.refuse {
                return Err(CoreError::Backend {
                    backend: "refusing",
                    op: "run",
                    source: Box::new(std::io::Error::other(
                        "threadgroup memory is over this device's budget",
                    )),
                });
            }
            Ok(())
        }

        fn sync(&self) -> CoreResult<()> {
            Ok(())
        }
    }

    /// A variant the device refuses at dispatch is recorded as refused, not
    /// as merely untimed.
    ///
    /// The distinction is the whole reason `refused` is a field: threadgroup
    /// *memory* is checked when a launch is encoded and not by
    /// `max_block_threads`, so a variant can report a limit wide enough for
    /// its lane count, be tried, and be turned away. Read off `nanos` alone
    /// that is indistinguishable from "never attempted", and `choose_variant`
    /// would keep offering a kernel the device will not run.
    ///
    /// The refusal is staged rather than provoked on a real device on
    /// purpose: every shipped variant fits the 32768 B family budget since
    /// Task 4, so there is no launch this hardware actually turns away. A
    /// case that waited for one would pass by never running.
    #[test]
    fn a_dispatch_refusal_is_recorded_as_refused() {
        let backend = RefusingBackend {
            refuse: Variant::ChunkHybrid.entry(),
            next: AtomicU64::new(1),
        };
        let geometry = test_geometry();
        let variants = [Variant::C1, Variant::ChunkHybrid];
        let table = measure(&backend, &geometry, &variants, 64);

        assert_eq!(table.len(), 2);
        assert!(!table[0].refused, "the accepted variant was not refused");
        assert!(table[0].nanos.is_some(), "the accepted variant was timed");
        assert!(table[1].refused, "the refused variant is marked refused");
        assert_eq!(table[1].nanos, None, "a refused variant carries no timing");
        assert_eq!(
            table[1].limit,
            Some(1024),
            "the limit is still reported: the refusal was not a lane-count one",
        );
    }
}
