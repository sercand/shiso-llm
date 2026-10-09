//! The qwen35 recurrent block: one projection, a short convolution, a gated
//! delta rule.
//!
//! Every kernel here is the qwen35 one, and every launch geometry comes from
//! `metal/qwen35/KERNELS.md` rather than from a guess at what the block would
//! want. The block runs over a whole chunk of tokens per launch, so a prefill
//! step and a decode step differ by `n_tok` and by nothing else in this file.
//!
//! # The two states are not scratch
//!
//! The convolution window and the delta-rule state are both per-session state:
//! the kernels rewrite them in place, and a checkpoint is a copy of exactly these
//! bytes. They therefore arrive as views the caller owns, and nothing in this
//! module allocates them or gets to decide how long they live.
//!
//! # Why the two recurrence paths are not interchangeable
//!
//! [`launch::GDN_CHUNK_MIN`] splits the two recurrence paths on chunk length. The
//! chunked kernel is faster over a long span but produces only the state at the
//! end of a run, so it cannot fill the per-token states a speculative rollback
//! checkpoint is made of; that is the whole reason the threshold exists and not a
//! statement about which one is faster.

use shiso_core::Extent3;
use shiso_core::backend::{Backend, KernelArg, KernelRef};
use shiso_core::buffer::DeviceView;
use shiso_core::error::{CoreError, CoreResult};

use shiso_engine::blocks::{GdnIo, GdnState, GdnStates};
use shiso_engine::gdn::{Gdn, GdnGeometry, GdnScratch};
use shiso_engine::launch::{
    self, BLOCK_128, BLOCK_LINEAR, ConvArgs, GDN_NORM_EPS, GateArgs, HEAD_SMEM_BYTES, L2NormArgs,
    OutGateArgs, Q35, RecurrentArgs,
};
use shiso_engine::projection;

use super::gdn_tune;

/// A run of rows inside a chunk's scratch, as the two sequential stages take
/// it.
///
/// The convolution and the recurrence are the only two stages of this block
/// that carry state from one row to the next, so they are the only two a
/// speculative pass has to be able to stop between. Both index every row they
/// touch off a base pointer, so "rows `first..first + count`" is expressible as
/// three advanced bases and a smaller `n_tok`, and this is the arithmetic that
/// does it once instead of at each of the five call sites.
#[derive(Debug, Clone, Copy)]
struct RowSpan {
    /// First row of the run, counted from the chunk's own row zero.
    first: u32,
    /// Rows in the run.
    count: u32,
}

impl RowSpan {
    /// The whole chunk.
    const fn all(count: u32) -> Self {
        Self { first: 0, count }
    }

    /// Row `at` alone.
    const fn one(at: u32) -> Self {
        Self {
            first: at,
            count: 1,
        }
    }

    /// This run's bytes of a buffer laid out as `stride` bytes a row.
    ///
    /// # Errors
    ///
    /// [`shiso_core::CoreError::OutOfBounds`] when the run reaches past the
    /// buffer, which is a chunk wider than the scratch was allocated for.
    fn slice(self, view: DeviceView<'_>, stride: u64) -> CoreResult<DeviceView<'_>> {
        view.slice(
            stride.saturating_mul(u64::from(self.first)),
            stride.saturating_mul(u64::from(self.count)),
        )
    }
}

/// The short causal convolution, over the whole chunk at once.
///
/// One threadgroup owns a channel and walks the chunk inside the launch, so the
/// window it carries between chunks stays in `conv_state` for the length of the
/// generation and the launch itself is one threadgroup group per channel block.
fn conv(
    backend: &dyn Backend,
    gdn: &Gdn,
    geometry: &GdnGeometry,
    scratch: &GdnScratch,
    conv_state: DeviceView<'_>,
    rows: RowSpan,
) -> CoreResult<()> {
    let args = ConvArgs {
        channels: geometry.channels,
        conv_kernel: geometry.conv_kernel,
        n_tok: rows.count,
    };
    let stride = geometry.mix_row_bytes();
    let span = stride.saturating_mul(u64::from(rows.count));
    let at = stride.saturating_mul(u64::from(rows.first));
    launch::run(
        backend,
        KernelRef::new(Q35, "kernel_q35_gdn_conv"),
        Extent3::linear(geometry.channels.div_ceil(BLOCK_LINEAR)),
        Extent3::new(BLOCK_LINEAR, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(scratch.mix.view().slice(at, span)?),
            KernelArg::Buffer(conv_state),
            KernelArg::Buffer(scratch.x.view().slice(at, span)?),
            KernelArg::Buffer(gdn.conv1d.view()),
        ],
    )
    .map_err(|err| launch::launch_failure("kernel_q35_gdn_conv", err))
}

/// The convolution over rows `0 .. n_tok`, writing the window after every row
/// but the last into `trail`, in one launch.
fn conv_recorded(
    backend: &dyn Backend,
    gdn: &Gdn,
    geometry: &GdnGeometry,
    scratch: &GdnScratch,
    conv_state: DeviceView<'_>,
    trail: DeviceView<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    const NAME: &str = "kernel_q35_gdn_conv_trail";
    let need = conv_state
        .len()
        .saturating_mul(u64::from(n_tok.saturating_sub(1)));
    if trail.len() < need {
        return Err(CoreError::OutOfBounds {
            offset: 0,
            len: need,
            buffer_bytes: trail.len(),
        });
    }
    let args = ConvArgs {
        channels: geometry.channels,
        conv_kernel: geometry.conv_kernel,
        n_tok,
    };
    launch::run(
        backend,
        KernelRef::new(Q35, NAME),
        Extent3::linear(geometry.channels.div_ceil(BLOCK_LINEAR)),
        Extent3::new(BLOCK_LINEAR, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(scratch.mix.view()),
            KernelArg::Buffer(conv_state),
            KernelArg::Buffer(scratch.x.view()),
            KernelArg::Buffer(gdn.conv1d.view()),
            KernelArg::Buffer(trail),
        ],
    )
    .map_err(|err| launch::launch_failure(NAME, err))
}

/// Normalizes both the query and the key half of the mix, in place.
///
/// The kernel takes a head family by offset, so the two halves are two launches
/// of one entry point rather than a wider argument block the kernel would only
/// have to split again.
fn normalize(
    backend: &dyn Backend,
    geometry: &GdnGeometry,
    scratch: &GdnScratch,
    n_tok: u32,
) -> CoreResult<()> {
    for offset in [0, geometry.k_offset()] {
        let args = L2NormArgs {
            head_dim: geometry.head_dim,
            n_heads: geometry.qk_heads,
            stride: geometry.channels,
            offset,
        };
        launch::run(
            backend,
            KernelRef::new(Q35, "kernel_q35_gdn_l2norm"),
            Extent3::new(geometry.qk_heads, n_tok, 1),
            Extent3::new(BLOCK_128, 1, 1),
            HEAD_SMEM_BYTES,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(scratch.mix.view()),
            ],
        )
        .map_err(|err| launch::launch_failure("kernel_q35_gdn_l2norm", err))?;
    }
    Ok(())
}

/// Runs the two per-head gates over the chunk.
///
/// Both drivers are projected from the same residual rows the mix was projected
/// from, so the block reads the block's own input three times instead of keeping
/// a second copy of the residual stream around for them.
fn gates(
    backend: &dyn Backend,
    gdn: &Gdn,
    geometry: &GdnGeometry,
    scratch: &GdnScratch,
    input: DeviceView<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    projection::projection(backend, &gdn.alpha, n_tok, input, scratch.alpha_proj.view())?;
    projection::projection(backend, &gdn.beta, n_tok, input, scratch.beta_proj.view())?;
    let args = GateArgs {
        n_head: geometry.v_heads,
        n_tok,
    };
    launch::run(
        backend,
        KernelRef::new(Q35, "kernel_q35_gdn_gates"),
        Extent3::linear(
            geometry
                .v_heads
                .saturating_mul(n_tok)
                .div_ceil(BLOCK_LINEAR),
        ),
        Extent3::new(BLOCK_LINEAR, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(scratch.decay.view()),
            KernelArg::Buffer(scratch.beta_out.view()),
            KernelArg::Buffer(scratch.alpha_proj.view()),
            KernelArg::Buffer(scratch.beta_proj.view()),
            KernelArg::Buffer(gdn.dt_bias.view()),
            KernelArg::Buffer(gdn.a.view()),
        ],
    )
    .map_err(|err| launch::launch_failure("kernel_q35_gdn_gates", err))
}

/// Walks the recurrence over the chunk, refreshing `state` as it goes.
///
/// The chunked kernel covers 64-token runs and writes only the run's final state,
/// which is why it cannot serve a path that has to record a state per token; the
/// threshold below is where the walk becomes the cheaper of the two, and the
/// caller that has to fill rollback checkpoints picks the path instead.
fn recurrence(
    backend: &dyn Backend,
    geometry: &GdnGeometry,
    scratch: &GdnScratch,
    shape: gdn_tune::GdnShape,
    state: DeviceView<'_>,
    rows: RowSpan,
) -> CoreResult<()> {
    let n_tok = rows.count;
    let args = RecurrentArgs {
        head_dim: geometry.head_dim,
        n_head_k: geometry.qk_heads,
        n_head_v: geometry.v_heads,
        k_offset: geometry.k_offset(),
        v_offset: geometry.v_offset(),
        stride: geometry.channels,
        n_tok,
    };
    let variant = if n_tok >= launch::GDN_CHUNK_MIN {
        shape.chunk
    } else {
        shape.recurrent
    };
    // Every one of the three rows this entry walks is indexed off its own base
    // pointer by the row number, so a run over rows `[first, first + count)` is
    // the same launch with all three bases advanced. That is what makes a
    // per-row walk bit-identical to a single-shot one: the kernel does the same
    // arithmetic in the same order on the same numbers, and the only thing that
    // changes is how often the state row round-trips through the f32 buffer it
    // was going to be written to anyway.
    let mix = rows.slice(scratch.mix.view(), geometry.mix_row_bytes())?;
    let out = rows.slice(scratch.attn_out.view(), geometry.out_row_bytes())?;
    let head = geometry.head_row_bytes();
    let mut binds = vec![
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(out),
        KernelArg::Buffer(state),
        KernelArg::Buffer(mix),
        KernelArg::Buffer(rows.slice(scratch.decay.view(), head)?),
        KernelArg::Buffer(rows.slice(scratch.beta_out.view(), head)?),
    ];
    // One variant declares a seventh buffer, so the argument list is a
    // property of the variant and not of this call site: the six above are the
    // contract every entry point keeps, and the spill is bound only by the one
    // that asks for it.
    if variant.spills_deltas() {
        binds.push(KernelArg::Buffer(scratch.spill.view()));
    }
    launch::run(
        backend,
        KernelRef::new(Q35, variant.entry()),
        Extent3::linear(geometry.v_heads),
        Extent3::new(variant.lanes(), 1, 1),
        variant.smem_bytes(),
        &binds,
    )
    .map_err(|err| launch::launch_failure(variant.entry(), err))
}

/// The recurrence over rows `0 .. n_tok`, writing the state after every row
/// but the last into `trail`, in one launch of the variant's recording twin.
fn recurrence_recorded(
    backend: &dyn Backend,
    geometry: &GdnGeometry,
    scratch: &GdnScratch,
    variant: (gdn_tune::Variant, &'static str),
    state: DeviceView<'_>,
    trail: DeviceView<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    let (variant, entry) = variant;
    let args = RecurrentArgs {
        head_dim: geometry.head_dim,
        n_head_k: geometry.qk_heads,
        n_head_v: geometry.v_heads,
        k_offset: geometry.k_offset(),
        v_offset: geometry.v_offset(),
        stride: geometry.channels,
        n_tok,
    };
    // A trail too short for the pass's records is refused here, before the
    // kernel writes past it.
    let need = state
        .len()
        .saturating_mul(u64::from(n_tok.saturating_sub(1)));
    if trail.len() < need {
        return Err(CoreError::OutOfBounds {
            offset: 0,
            len: need,
            buffer_bytes: trail.len(),
        });
    }
    launch::run(
        backend,
        KernelRef::new(Q35, entry),
        Extent3::linear(geometry.v_heads),
        Extent3::new(variant.lanes(), 1, 1),
        variant.smem_bytes(),
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(scratch.attn_out.view()),
            KernelArg::Buffer(state),
            KernelArg::Buffer(scratch.mix.view()),
            KernelArg::Buffer(scratch.decay.view()),
            KernelArg::Buffer(scratch.beta_out.view()),
            KernelArg::Buffer(trail),
        ],
    )
    .map_err(|err| launch::launch_failure(entry, err))
}

/// Applies the per-head normalization and the output gate.
///
/// The gate is a projection row, not a scalar: the kernel reads one head-width of
/// `z` per value head, which is why the gate projection is as wide as the block's
/// output rather than one float per head.
fn out_gate(
    backend: &dyn Backend,
    gdn: &Gdn,
    geometry: &GdnGeometry,
    scratch: &GdnScratch,
    n_tok: u32,
) -> CoreResult<()> {
    let args = OutGateArgs {
        head_dim: geometry.head_dim,
        n_head: geometry.v_heads,
        eps: GDN_NORM_EPS,
    };
    launch::run(
        backend,
        KernelRef::new(Q35, "kernel_q35_gdn_out_gate"),
        Extent3::linear(n_tok.saturating_mul(geometry.v_heads)),
        Extent3::new(BLOCK_128, 1, 1),
        HEAD_SMEM_BYTES,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(scratch.gated.view()),
            KernelArg::Buffer(scratch.attn_out.view()),
            KernelArg::Buffer(scratch.z.view()),
            KernelArg::Buffer(gdn.norm.view()),
        ],
    )
    .map_err(|err| launch::launch_failure("kernel_q35_gdn_out_gate", err))
}

/// Runs one recurrent block over `n_tok` rows of `input`.
///
/// `shape` is the pair of kernel shapes this device was measured to run
/// fastest, chosen once when the stack was built. It arrives as an argument
/// rather than sitting in [`GdnGeometry`] because it is a fact about the
/// machine and not about the architecture: two processes on two boxes run the
/// same geometry through different entry points, and bit for bit the same
/// answer comes out.
///
/// `io.states` is the session's own state, or one per row, and is rewritten in
/// place: it is the block's whole memory of the prefix, and a caller that handed
/// in a fresh pair would get a block that had never seen the prompt.
/// `block_out` receives the block's contribution to the residual stream and
/// belongs to the caller, which is what adds it in.
///
/// # The three shapes, and what separates them
///
/// The projections, the normalization and the gates are launched over every row
/// of the pass whichever shape it is: none of them carries anything between
/// rows, so a row's answer does not depend on which rows it was issued with.
/// Only the convolution and the recurrence do, and they are what the three
/// shapes differ in:
///
/// - a **chunk** walks its rows inside one launch of each, which is what makes
///   a 512-row prefill two launches rather than a thousand;
/// - a **recorded** chunk splits that walk so a rejection has a state per row
///   to wind back to, at the same arithmetic (`RowSpan`'s header says why);
/// - a **batch** has no walk to make: its rows are different sessions, so each
///   one folds into its own state matrix and the two sequential stages cost one
///   launch per row. That is the part of a decode step batching does not
///   amortize, and it is bounded by `MAX_DECODE_BATCH`.
///
/// # Errors
///
/// [`CoreError::Unsupported`] for a recorded batch, which is a speculative step
/// over several sessions at once and is not a shape this stack builds;
/// [`CoreError::ShapeMismatch`] when the per-row states and the pass disagree
/// about how wide it is; whatever [`projection::projection`] reports for a
/// width or format it cannot carry, and [`CoreError::Backend`] naming the
/// kernel when a launch fails.
pub(super) fn run(
    backend: &dyn Backend,
    gdn: &Gdn,
    geometry: &GdnGeometry,
    scratch: &GdnScratch,
    shape: gdn_tune::GdnShape,
    io: GdnIo<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    let input = io.input;
    projection::projection(backend, &gdn.qkv, n_tok, input, scratch.x.view())?;
    let whole = RowSpan::all(n_tok);
    match (io.states, io.trail) {
        (GdnStates::Chunk(one), None) => {
            conv(backend, gdn, geometry, scratch, one.conv, whole)?;
            normalize(backend, geometry, scratch, n_tok)?;
            gates(backend, gdn, geometry, scratch, input, n_tok)?;
            recurrence(backend, geometry, scratch, shape, one.state, whole)?;
        }
        (GdnStates::Chunk(one), Some(trail)) => {
            // The convolution first, row by row, so that every row's window is
            // recorded; then the two per-row stages over the whole chunk,
            // because neither carries anything between rows; then the
            // recurrence, row by row, for the same reason the convolution was.
            //
            // Not one loop over all four. The normalization and the gates read
            // the mix the convolution wrote, and the recurrence reads what they
            // left - so a single loop would either interleave four launches per
            // row where two will do, or read a row the batched stages had not
            // reached yet.
            conv_recorded(backend, gdn, geometry, scratch, one.conv, trail.conv, n_tok)?;
            normalize(backend, geometry, scratch, n_tok)?;
            gates(backend, gdn, geometry, scratch, input, n_tok)?;
            // One launch that writes every row's record as it goes, where the
            // variant has a recording twin; the per-row walk otherwise. The two
            // leave the same bits in the trail and in the state.
            //
            // The recording twin runs at the widest split the device accepts
            // (`GdnShape::recorder`), whatever the tuner chose for the plain
            // recurrence. The three splits fold to the same bits by
            // construction (`q35_gdn_recurrent_impl`), so this changes no
            // number; what it changes is the trail's write pattern. At one
            // lane a column a head is one 128-thread block and each store
            // instruction scatters over 32 lines; measured on the dense 27B
            // with a block drafter, that shape spent 417 us a layer writing an
            // eight-row trail - 19 ms of a 145 ms verify step.
            let recorder = shape.recorder.and_then(|recorder| {
                recorder
                    .trail_entry()
                    .filter(|_| n_tok < launch::GDN_CHUNK_MIN)
                    .map(|entry| (recorder, entry))
            });
            match recorder {
                Some(recorder) => recurrence_recorded(
                    backend,
                    geometry,
                    scratch,
                    recorder,
                    one.state,
                    trail.state,
                    n_tok,
                )?,
                None => walk(n_tok, |row| {
                    recurrence(backend, geometry, scratch, shape, one.state, row.span)?;
                    record(backend, trail.state, one.state, row)
                })?,
            }
        }
        (GdnStates::PerRow(rows), None) => {
            per_row(rows, n_tok)?;
            for (at, row) in rows.iter().enumerate() {
                let at = u32::try_from(at).unwrap_or(u32::MAX);
                conv(backend, gdn, geometry, scratch, row.conv, RowSpan::one(at))?;
            }
            normalize(backend, geometry, scratch, n_tok)?;
            gates(backend, gdn, geometry, scratch, input, n_tok)?;
            for (at, row) in rows.iter().enumerate() {
                let at = u32::try_from(at).unwrap_or(u32::MAX);
                recurrence(
                    backend,
                    geometry,
                    scratch,
                    shape,
                    row.state,
                    RowSpan::one(at),
                )?;
            }
        }
        (GdnStates::PerRow(_), Some(_)) => {
            // A trail records "the state after row i" so a rejection can wind
            // one session back to it. Over several sessions there is no such
            // sequence: row i's state belongs to a different conversation from
            // row i-1's, and a rollback would restore one session's matrix over
            // another's. Refused rather than recorded into the wrong slab -
            // nothing builds this shape, and this is what says so if something
            // later tries.
            return Err(CoreError::Unsupported {
                what: "a recurrent trail over a batch of several sessions, whose rows \
                       are not one sequence for a rollback to wind back along",
                backend: "qwen35",
            });
        }
    }
    projection::projection(backend, &gdn.gate, n_tok, input, scratch.z.view())?;
    out_gate(backend, gdn, geometry, scratch, n_tok)?;
    projection::projection(backend, &gdn.out, n_tok, scratch.gated.view(), io.block_out)
}

/// Refuses per-row states that do not name every row of the pass.
///
/// The two sequential stages index the scratch by the row number and the state
/// by the slice position, so a slice short of the pass would leave the rows past
/// it folded into nothing - every launch succeeding, the residual stream short
/// one block's contribution for those rows - and a slice longer than it would
/// fold rows the pass never wrote.
fn per_row(rows: &[GdnState<'_>], n_tok: u32) -> CoreResult<()> {
    let got = u64::try_from(rows.len()).unwrap_or(u64::MAX);
    if got == u64::from(n_tok) {
        return Ok(());
    }
    Err(CoreError::ShapeMismatch {
        field: "per-row recurrent states",
        expected: u64::from(n_tok),
        got,
    })
}

/// One row of a recorded walk: which row it is, and how many follow it.
#[derive(Debug, Clone, Copy)]
struct Recorded {
    /// The row itself.
    span: RowSpan,
    /// Rows in the whole walk, which is what says whether this one is the last.
    total: u32,
}

/// Runs `step` for each row of a chunk of `n_tok` rows, in order.
fn walk(n_tok: u32, mut step: impl FnMut(Recorded) -> CoreResult<()>) -> CoreResult<()> {
    for at in 0..n_tok {
        step(Recorded {
            span: RowSpan::one(at),
            total: n_tok,
        })?;
    }
    Ok(())
}

/// Copies the state a row left into that row's record of the trail.
///
/// The last row of the walk writes nothing. Its record would be the state the
/// session is already standing on, and a rollback to it is the rollback that
/// [`shiso_engine::mtp::Rollback::new`] answers `None` for.
fn record(
    backend: &dyn Backend,
    trail: DeviceView<'_>,
    live: DeviceView<'_>,
    row: Recorded,
) -> CoreResult<()> {
    if row.span.first.saturating_add(1) >= row.total {
        return Ok(());
    }
    let stride = live.len();
    backend.copy_d2d(
        trail.slice(stride.saturating_mul(u64::from(row.span.first)), stride)?,
        live,
    )
}
