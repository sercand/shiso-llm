//! The routed feed-forward: a router over hundreds of experts, plus a shared
//! expert every token pays for.
//!
//! Every layer holds one. The router scores every expert for every token and
//! keeps the best handful; each surviving (token, expert) pair becomes a
//! *column*; the columns run through the chosen experts' gate, up, activation
//! and down; and the results are recombined into one contribution per token,
//! with the shared expert's own output added on top.
//!
//! # Where the arithmetic is, and why it is worth isolating
//!
//! Almost nothing in this file is a shape. The column count, the sorted layout,
//! the grid the down projection runs on, the threshold that decides whether
//! columns are grouped by expert - all of them are arithmetic over the routing,
//! and every one of them is checkable without a device. That matters more here
//! than anywhere else in the pass, because the failure mode of getting the
//! routing wrong is not a crash: it is a token answered by the wrong ten
//! experts, which reads as a model that is merely a bit worse.
//!
//! # Two expert stages, and one of them this build cannot reach
//!
//! A trellis expert stack runs gate, up, activation and down as one cooperative
//! launch per expert group. Anything else runs them staged - and the staged
//! gate-and-up belongs to a different kernel set than the one vendored here, so
//! it is a typed refusal naming the missing companion rather than a slower path
//! that quietly produces different numbers.
//!
//! That is also why [`block`] takes no [`RowOrigin`]. The fused launch groups
//! the expert slots *one token* was routed to - its fan-out counts
//! [`MoeShape::used`] and its grid is one group per expert, neither of which
//! depends on the row count - so what the rows beside a token routed to changes
//! neither which blocks run nor what any of them sums. The grouping that does
//! depend on the rows being related is the down projection's, which sweeps one
//! expert's row across every column routed to it, and that is [`RowOrigin`]'s
//! own question in [`down`]. A gate on the fused launch could only ever turn a
//! decode batch into the refusal above, because there is no second path for it
//! to fall to.
//!
//! [`MoeShape::used`]: crate::sparse::shape::MoeShape::used

use shiso_core::DType;
use shiso_core::error::{CoreError, CoreResult};

use crate::sparse::bind::{Args, LayerRef, Operand, Role, WeightRef};
use crate::sparse::exl3::{self, Fanout};
use crate::sparse::kernels::{self, Entry, MOD_EXL3, MOD_MOE, TemplateArgs};
use crate::sparse::matmul::{self, Bound, Device};
use crate::sparse::plan::{Plan, Step, TrellisOp};
use crate::sparse::scratch::Buf;
use crate::sparse::shape::SparseShape;
use crate::sparse::weights::{ExpertDesc, WeightPart};

/// Rows from which columns are grouped by expert before the down projection.
///
/// The grouped kernel reads one expert's row into registers once and sweeps
/// every column routed to it; the per-column kernel reads the whole row again
/// for each column. At one token those are the same traffic, and at a
/// five-hundred-row chunk the second is five hundred times too much.
pub const GROUP_MIN_ROWS: u32 = 4;

/// Rows from which columns are grouped when a chunk's rows belong to unrelated
/// sequences.
///
/// Consecutive tokens of one sequence route to overlapping experts, so grouping
/// them pays from a handful of rows. The rows of a batched decode step are
/// unrelated sequences: there the grouping gains very little and changes the
/// rounding of a path that is otherwise identical to a single-sequence decode,
/// so it waits for a row count that is a real chunk.
pub const GROUP_MIN_UNRELATED_ROWS: u32 = 32;

/// Whether a chunk's rows come from one sequence or from several.
///
/// An enum rather than a `bool` because the threshold it selects differs by a
/// factor of eight, and at a call site `true` says nothing about which way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RowOrigin {
    /// One sequence's consecutive positions: a prefill chunk, or a speculative
    /// verify.
    ///
    /// The default because it is the only origin a pass of one row can have,
    /// and one row is what most passes are.
    #[default]
    Sequential,
    /// One row per sequence, from sequences that have nothing to do with each
    /// other.
    Unrelated,
}

impl RowOrigin {
    /// Whether a chunk of `n_tok` rows groups its columns by expert.
    #[must_use]
    pub const fn groups(self, n_tok: u32) -> bool {
        match self {
            Self::Sequential => n_tok >= GROUP_MIN_ROWS,
            Self::Unrelated => n_tok >= GROUP_MIN_UNRELATED_ROWS,
        }
    }
}

/// The router's own stage: one projection, then the selection.
///
/// The selection kernel stages the whole logit row in shared memory, one float
/// per expert, which is why the expert count reaches its launch.
///
/// # Errors
///
/// Whatever [`matmul::project`] reports for the router's projection.
pub fn route(
    shape: &SparseShape,
    desc: &ExpertDesc,
    layer: LayerRef,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let mut plan = Plan::new();
    matmul::project(
        &mut plan,
        desc.router,
        Bound::new(
            WeightRef::layer(layer, Role::Router),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::MoeRouter),
        ),
        n_tok,
        device,
    )?;
    plan.launch(
        Entry::plain(MOD_MOE, "q4e_moe_route_kernel"),
        kernels::moe_route(shape.moe.experts, n_tok),
        // `ids, weights, logits, n_expert, n_used`
        Args::of(&[
            Operand::Scratch(Buf::MoeIds),
            Operand::Scratch(Buf::MoeWeights),
            Operand::Scratch(Buf::MoeRouter),
            Operand::U32(shape.moe.experts),
            Operand::U32(shape.moe.used),
        ]),
    );
    Ok(plan)
}

/// The sort that groups the routed columns by expert.
///
/// Three passes - count, scan, scatter - over the columns rather than over the
/// tokens, because a column is what an expert receives. The scan is one warp in
/// one block: it runs over the expert count and is serial by construction.
///
/// The per-expert count row is the scatter's cursor as well as the count pass's
/// destination: the scan has already read it into the bounds by then, and the
/// scatter needs one counter per expert and nothing else.
#[must_use]
pub fn sort(shape: &SparseShape, n_tok: u32) -> Plan {
    let rows = shape.moe.slot_rows(n_tok);
    let columns = u32::try_from(rows).unwrap_or(u32::MAX);
    let mut plan = Plan::new();
    // The histogram is accumulated with `atomicAdd` and the buffer outlives the
    // layer, so without this the counts of every earlier layer are still in it:
    // expert spans grow past the staging a group has, and the fused launch
    // starts producing NaN rows around the fifth routed block.
    plan.push(Step::Clear {
        buffer: Operand::Scratch(Buf::MoeCounts),
    });
    plan.launch(
        Entry::plain(MOD_MOE, "q4e_moe_sort_count_kernel"),
        kernels::moe_sort_count(rows),
        // `counts, ids, n_rows, n_expert`
        Args::of(&[
            Operand::Scratch(Buf::MoeCounts),
            Operand::Scratch(Buf::MoeIds),
            Operand::U32(columns),
            Operand::U32(shape.moe.experts),
        ]),
    );
    plan.launch(
        Entry::plain(MOD_MOE, "q4e_moe_sort_scan_kernel"),
        kernels::moe_sort_scan(),
        // `bounds, active, counts, n_expert`
        Args::of(&[
            Operand::Scratch(Buf::MoeBounds),
            Operand::Scratch(Buf::MoeActive),
            Operand::Scratch(Buf::MoeCounts),
            Operand::U32(shape.moe.experts),
        ]),
    );
    // The same row is the scatter's per-expert cursor. The scan has read the
    // counts into the bounds by now, so zeroing here is what makes the cursor a
    // cursor rather than the histogram it was a moment ago.
    plan.push(Step::Clear {
        buffer: Operand::Scratch(Buf::MoeCounts),
    });
    plan.launch(
        Entry::plain(MOD_MOE, "q4e_moe_sort_scatter_kernel"),
        kernels::moe_sort_scatter(rows),
        // `sorted, cursor, bounds, ids, n_rows, n_expert`
        Args::of(&[
            Operand::Scratch(Buf::MoeSorted),
            Operand::Scratch(Buf::MoeCounts),
            Operand::Scratch(Buf::MoeBounds),
            Operand::Scratch(Buf::MoeIds),
            Operand::U32(columns),
            Operand::U32(shape.moe.experts),
        ]),
    );
    plan
}

/// The stored element type ids the routed down projection was instantiated for.
///
/// Two, and they are the two an expert stack is shipped in when it is not a
/// trellis. A third would be a template instantiation this build does not
/// carry, which is why this is a lookup that can fail rather than a cast.
#[must_use]
pub const fn down_template_type(dtype: DType) -> Option<u32> {
    match dtype {
        DType::Q5_1 => Some(7),
        DType::Q8_0 => Some(8),
        _ => None,
    }
}

/// The experts' own stage, for a trellis stack: one cooperative launch.
///
/// # Errors
///
/// [`CoreError::Unsupported`] when no built tiling shape divides the expert
/// weights, and whatever [`exl3::check_shape`] reports for a weight the codec
/// cannot tile at all.
pub fn fused_experts(
    shape: &SparseShape,
    layer: LayerRef,
    bits: u32,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    exl3::check_shape(shape.hidden, shape.moe.intermediate, bits)?;
    let fanout = Fanout::Grouped {
        slots: shape.moe.used,
        input_per_slot: true,
    };
    let tiling = exl3::select_shape(shape.hidden, shape.moe.intermediate, bits, fanout).ok_or(
        CoreError::Unsupported {
            what: "an expert stack no built tiling shape divides",
            backend: "engine",
        },
    )?;
    let grid = exl3::routed_grid(
        device.max_co_resident,
        device.max_co_resident,
        shape.moe.experts,
    );
    // The staging is per group and per token, and an undersized one is silent
    // out-of-bounds corruption rather than a failed launch - so its three
    // buffers are carved at offsets derived from the grid rather than guessed
    // at. The reservation itself belongs to the pass, not to this launch: it
    // cannot be made while a stream is capturing, and this launch sits inside
    // an island that may be.
    let staging = exl3::routed_staging(grid, n_tok, shape.hidden, shape.moe.intermediate);
    let at = |role: Role, span| Operand::Weight(WeightRef::layer(layer, role).spanning(span));
    let width = |value: u32| Operand::I32(i32::try_from(value).unwrap_or(i32::MAX));
    let mut plan = Plan::new();
    plan.push(Step::Trellis {
        op: TrellisOp::Routed,
        entry: Entry::template(MOD_EXL3, kernels::EXL3_MOE, TemplateArgs::TrellisBits(bits)),
        shape: tiling,
        bits,
        blocks: grid.group_size,
        groups: grid.groups,
        // The kernel's own width, not the tiling shape's: this entry point is
        // `__launch_bounds__` at `EXL3_MOE_BLOCK_DIM` and divides its work by
        // that number, so a narrower block from a shape that happens to carry
        // one would leave most of each tile unwritten.
        block_dim: kernels::EXL3_MOE_BLOCK_DIM,
        smem: kernels::EXL3_SMEM_MAX,
        staging: exl3::Staging::of(staging.bytes),
        // `hidden_state, temp_state, temp_intermediate, temp_act, slot_out,
        //  gate_trellis, gate_suh, gate_svh, up_trellis, up_suh, up_svh,
        //  down_trellis, down_suh, down_svh, expert_bounds, slot_sorted,
        //  num_tokens, hidden_dim, intermediate_dim, num_experts,
        //  num_experts_per_tok, locks`
        //
        // The routing is the expert-sorted column map the sort stage built:
        // expert `e` owns the sorted slots between `bounds[e]` and
        // `bounds[e + 1]`, and the launch writes each assignment's unweighted
        // contribution at its own slot, which is what lets the combine apply
        // the routing weights in a fixed order.
        args: Args::of(&[
            Operand::Scratch(Buf::Mixed),
            Operand::CoopStaging(staging.state),
            Operand::CoopStaging(staging.intermediate),
            Operand::CoopStaging(staging.act),
            Operand::Scratch(Buf::MoeDown),
            at(Role::ExpertGate, WeightPart::Tiles),
            at(Role::ExpertGate, WeightPart::InputScales),
            at(Role::ExpertGate, WeightPart::OutputScales),
            at(Role::ExpertUp, WeightPart::Tiles),
            at(Role::ExpertUp, WeightPart::InputScales),
            at(Role::ExpertUp, WeightPart::OutputScales),
            at(Role::ExpertDown, WeightPart::Tiles),
            at(Role::ExpertDown, WeightPart::InputScales),
            at(Role::ExpertDown, WeightPart::OutputScales),
            Operand::Scratch(Buf::MoeBounds),
            Operand::Scratch(Buf::MoeSorted),
            width(n_tok),
            width(shape.hidden),
            width(shape.moe.intermediate),
            width(shape.moe.experts),
            width(shape.moe.used),
            Operand::CoopLocks,
        ]),
    });
    Ok(plan)
}

/// Rows from which the routed block takes the fused launch.
///
/// Below it - which is a decode step and nothing else - the block runs as two
/// grouped launches instead. See [`grouped_experts`] for why.
pub const FUSED_MIN_ROWS: u32 = 2;

/// The experts' own stage for a single row, as two grouped launches.
///
/// # Why one token does not want the fused launch
///
/// The fused kernel gives each expert a *group* of blocks and walks the groups
/// over the experts that received columns. Its group count comes from the
/// device - on a forty-eight-block device that is six groups of eight - so ten
/// active experts go round twice, and the second round runs four experts wide
/// on a device sized for six. Measured on this box against a standalone A/B of
/// the two paths on this checkpoint's widths: 165 us a layer fused, 97 us
/// grouped, where 97 us is what the 23 MB a layer streams costs at the memory
/// bandwidth. The reference draws the same line, at the same place, for the
/// same reason.
///
/// The grouped form gives each *slot* its own group, so every expert the token
/// routed to streams at once: the gate and the up projections are twenty slots
/// behind one launch and the down projection is ten behind another. It also
/// needs no expert-sorted column map, because a slot names its matrix directly
/// - so a decode step skips the sort's four launches as well.
///
/// # Why it is a single row only
///
/// A slot's input is one row. Several rows per slot would mean replicating the
/// activation once per (row, expert) before the launch and reading each
/// expert's weights once per column rather than once - which is the trade the
/// fused kernel exists to avoid, and at a prefill chunk it is hundreds of times
/// the traffic. So this is the decode arm, and [`block`] takes it on the row
/// count rather than offering it as a choice.
///
/// # Errors
///
/// [`CoreError::Unsupported`] when no built tiling shape divides the expert
/// weights, and whatever [`exl3::check_shape`] reports for a weight the codec
/// cannot tile at all.
pub fn grouped_experts(
    shape: &SparseShape,
    layer: LayerRef,
    bits: u32,
    device: Device,
) -> CoreResult<Plan> {
    let hidden = shape.hidden;
    let inter = shape.moe.intermediate;
    let used = shape.moe.used;
    exl3::check_shape(hidden, inter, bits)?;
    exl3::check_shape(inter, hidden, bits)?;
    let at = |role: Role, span| Operand::Weight(WeightRef::layer(layer, role).spanning(span));
    let width = |value: u32| Operand::I32(i32::try_from(value).unwrap_or(i32::MAX));
    let unsupported = || CoreError::Unsupported {
        what: "an expert stack no built tiling shape divides",
        backend: "engine",
    };
    let mut plan = Plan::new();

    // Gate and up share their input and their slot list, so they are one launch
    // over twice the slots: slot `j` past the split is the second tensor's slot
    // `j - split`, reading the same expert id and writing the other output.
    let slots = used.saturating_mul(2);
    let pair = Fanout::Grouped {
        slots,
        input_per_slot: false,
    };
    let tiling = exl3::select_shape(hidden, inter, bits, pair).ok_or_else(unsupported)?;
    let grid = exl3::mgemm_grid(hidden, inter, tiling, slots, device.max_co_resident);
    plan.push(Step::Trellis {
        op: TrellisOp::Pair,
        entry: matmul::trellis_entry(kernels::EXL3_MGEMM, bits, tiling),
        shape: tiling,
        bits,
        blocks: grid.per_group,
        groups: grid.concurrency,
        block_dim: tiling.block_dim(),
        smem: kernels::EXL3_SMEM_MAX,
        // One transformed slab per slot, both tensors' slots counted: the
        // kernel indexes this by the *global* slot, so sizing it for one
        // tensor's worth is a write past the end rather than a failed launch.
        staging: exl3::Staging::hadamard(slots, 1, hidden),
        // `A, B_base, B_stride, C, size_m, size_k, size_n, locks, suh_base,
        //  A_had, svh_base, B_indices, bszm_in, bszm_out, B_base2, suh_base2,
        //  svh_base2, C2, split`
        args: Args::of(&[
            Operand::Scratch(Buf::Mixed),
            at(Role::ExpertGate, WeightPart::Tiles),
            Operand::U64(exl3::expert_tile_stride(hidden, inter, bits)),
            Operand::Scratch(Buf::MoeGate),
            Operand::U32(1),
            Operand::U32(hidden),
            Operand::U32(inter),
            Operand::CoopLocks,
            at(Role::ExpertGate, WeightPart::InputScales),
            Operand::CoopStaging(0),
            at(Role::ExpertGate, WeightPart::OutputScales),
            Operand::Scratch(Buf::MoeIds),
            Operand::I32(1),
            width(slots),
            at(Role::ExpertUp, WeightPart::Tiles),
            at(Role::ExpertUp, WeightPart::InputScales),
            at(Role::ExpertUp, WeightPart::OutputScales),
            Operand::Scratch(Buf::MoeUp),
            width(used),
        ]),
    });

    plan.append(activate(
        shape,
        Operand::Scratch(Buf::MoeGate),
        Operand::Scratch(Buf::MoeUp),
        Operand::Scratch(Buf::MoeMid),
        1,
    ));

    // The down projection is one tensor, and its slots do *not* share an input:
    // each reads the row its own expert activated.
    let down = Fanout::Grouped {
        slots: used,
        input_per_slot: true,
    };
    let tiling = exl3::select_shape(inter, hidden, bits, down).ok_or_else(unsupported)?;
    let grid = exl3::mgemm_grid(inter, hidden, tiling, used, device.max_co_resident);
    plan.push(Step::Trellis {
        op: TrellisOp::Fanout,
        entry: matmul::trellis_entry(kernels::EXL3_MGEMM, bits, tiling),
        shape: tiling,
        bits,
        blocks: grid.per_group,
        groups: grid.concurrency,
        block_dim: tiling.block_dim(),
        smem: kernels::EXL3_SMEM_MAX,
        staging: exl3::Staging::hadamard(used, 1, inter),
        // The same argument block with no second tensor: the null base is what
        // makes the split unreachable, and the split is still the slot count so
        // that a read of it cannot address past the list.
        args: Args::of(&[
            Operand::Scratch(Buf::MoeMid),
            at(Role::ExpertDown, WeightPart::Tiles),
            Operand::U64(exl3::expert_tile_stride(inter, hidden, bits)),
            Operand::Scratch(Buf::MoeDown),
            Operand::U32(1),
            Operand::U32(inter),
            Operand::U32(hidden),
            Operand::CoopLocks,
            at(Role::ExpertDown, WeightPart::InputScales),
            Operand::CoopStaging(0),
            at(Role::ExpertDown, WeightPart::OutputScales),
            Operand::Scratch(Buf::MoeIds),
            width(used),
            width(used),
            Operand::Null,
            Operand::Null,
            Operand::Null,
            Operand::Null,
            width(used),
        ]),
    });
    Ok(plan)
}

/// The activation between the up projection and the down one.
///
/// `out, gate, up, n`. The three buffers are the caller's because this kernel
/// serves two stages with different ones: the routed stage runs it over every
/// routed column, and the shared expert over every token.
#[must_use]
pub fn activate(shape: &SparseShape, gate: Operand, up: Operand, out: Operand, n_tok: u32) -> Plan {
    let n = shape.moe.slot_rows(n_tok) * u64::from(shape.moe.intermediate);
    let mut plan = Plan::new();
    plan.launch(
        Entry::plain(MOD_MOE, "q4e_swiglu_kernel"),
        kernels::elementwise(n),
        Args::of(&[out, gate, up, Operand::U64(n)]),
    );
    plan
}

/// The routed down projection, which is where the column grouping shows up.
///
/// # Errors
///
/// [`CoreError::Unsupported`] for a stored element type the down template was
/// not instantiated for.
pub fn down(
    shape: &SparseShape,
    desc: &ExpertDesc,
    layer: LayerRef,
    origin: RowOrigin,
    n_tok: u32,
) -> CoreResult<Plan> {
    let stored = down_template_type(desc.down.dtype).ok_or(CoreError::Unsupported {
        what: "a routed down projection at a stored element type this build never instantiated",
        backend: "engine",
    })?;
    let rows = u32::try_from(shape.moe.slot_rows(n_tok)).unwrap_or(u32::MAX);
    let blocks = matmul::quant_blocks_per_row(shape.moe.intermediate)
        .and_then(|blocks| u32::try_from(blocks).ok())
        .ok_or(CoreError::ShapeMismatch {
            field: "expert intermediate width",
            expected: 32,
            got: u64::from(shape.moe.intermediate),
        })?;
    let expert_bytes = crate::sparse::weights::Weight::payload_bytes(
        desc.down.dtype,
        desc.down.in_dim,
        desc.down.out_dim,
    )
    .ok_or(CoreError::ShapeMismatch {
        field: "expert slab bytes",
        expected: 1,
        got: 0,
    })?;
    let expert_bytes = Operand::U32(u32::try_from(expert_bytes).unwrap_or(u32::MAX));
    let weight = Operand::Weight(WeightRef::layer(layer, Role::ExpertDown));
    let mut plan = Plan::new();
    if origin.groups(n_tok) {
        plan.launch(
            Entry::template(
                MOD_MOE,
                "q4e_moe_down_grouped_kernel",
                TemplateArgs::StoredTypeBlocks { stored, blocks },
            ),
            kernels::moe_down_grouped(shape.hidden, rows, shape.moe.experts),
            // `dst, W, x, sorted_cols, bounds, active, n_expert, out_dim,
            //  in_dim, expert_bytes`
            Args::of(&[
                Operand::Scratch(Buf::MoeDown),
                weight,
                Operand::Scratch(Buf::MoeMid),
                Operand::Scratch(Buf::MoeSorted),
                Operand::Scratch(Buf::MoeBounds),
                Operand::Scratch(Buf::MoeActive),
                Operand::U32(shape.moe.experts),
                Operand::U32(shape.hidden),
                Operand::U32(shape.moe.intermediate),
                expert_bytes,
            ]),
        );
    } else {
        plan.launch(
            Entry::template(
                MOD_MOE,
                "q4e_moe_down_kernel",
                TemplateArgs::StoredType(stored),
            ),
            kernels::moe_down(shape.hidden, rows),
            // `dst, W, x, ids, out_dim, in_dim, n_used, expert_bytes`
            Args::of(&[
                Operand::Scratch(Buf::MoeDown),
                weight,
                Operand::Scratch(Buf::MoeMid),
                Operand::Scratch(Buf::MoeIds),
                Operand::U32(shape.hidden),
                Operand::U32(shape.moe.intermediate),
                Operand::U32(shape.moe.used),
                expert_bytes,
            ]),
        );
    }
    Ok(plan)
}

/// The recombination of every column into one contribution per token.
#[must_use]
pub fn combine(shape: &SparseShape, n_tok: u32) -> Plan {
    let mut plan = Plan::new();
    plan.launch(
        Entry::plain(MOD_MOE, "q4e_moe_combine_kernel"),
        kernels::moe_combine(shape.hidden, n_tok),
        // `out, down, weights, n_embd, n_used`
        Args::of(&[
            Operand::Scratch(Buf::BlockOut),
            Operand::Scratch(Buf::MoeDown),
            Operand::Scratch(Buf::MoeWeights),
            Operand::U32(shape.hidden),
            Operand::U32(shape.moe.used),
        ]),
    );
    plan
}

/// The shared expert: dense over every token, with a scalar gate of its own.
///
/// It runs after the routed contribution is already in place and adds onto it,
/// which is why its last launch is an add rather than a write.
///
/// # Errors
///
/// Whatever [`matmul::project_pair`] and [`matmul::project`] report.
pub fn shared(
    shape: &SparseShape,
    desc: &ExpertDesc,
    layer: LayerRef,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let at = |role: Role| WeightRef::layer(layer, role);
    let n = u64::from(n_tok) * u64::from(shape.moe.intermediate);
    let mut plan = Plan::new();
    matmul::project_pair(
        &mut plan,
        desc.shared_gate,
        desc.shared_up,
        Bound::new(
            at(Role::SharedGate),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::SharedGate),
        ),
        Bound::new(
            at(Role::SharedUp),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::SharedUp),
        ),
        n_tok,
        device,
    )?;
    plan.launch(
        Entry::plain(MOD_MOE, "q4e_swiglu_kernel"),
        kernels::elementwise(n),
        // `out, gate, up, n`
        Args::of(&[
            Operand::Scratch(Buf::SharedMid),
            Operand::Scratch(Buf::SharedGate),
            Operand::Scratch(Buf::SharedUp),
            Operand::U64(n),
        ]),
    );
    matmul::project(
        &mut plan,
        desc.shared_down,
        Bound::new(
            at(Role::SharedDown),
            Operand::Scratch(Buf::SharedMid),
            Operand::Scratch(Buf::SharedOut),
        ),
        n_tok,
        device,
    )?;
    matmul::project(
        &mut plan,
        desc.shared_router,
        Bound::new(
            at(Role::SharedRouter),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::SharedLogit),
        ),
        n_tok,
        device,
    )?;
    plan.launch(
        Entry::plain(MOD_MOE, "q4e_shared_add_kernel"),
        kernels::moe_combine(shape.hidden, n_tok),
        // `out, shared, logit, n_embd`
        Args::of(&[
            Operand::Scratch(Buf::BlockOut),
            Operand::Scratch(Buf::SharedOut),
            Operand::Scratch(Buf::SharedLogit),
            Operand::U32(shape.hidden),
        ]),
    );
    Ok(plan)
}

/// One routed feed-forward, in launch order.
///
/// # Errors
///
/// [`CoreError::Unsupported`] when the expert stack is stored in a format whose
/// staged gate-and-up belongs to a kernel set this build does not carry, and
/// whatever the stages report.
pub fn block(
    shape: &SparseShape,
    desc: &ExpertDesc,
    layer: LayerRef,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let Some(bits) = matmul::trellis_bits(desc.gate.dtype) else {
        return Err(CoreError::Unsupported {
            what: "a staged expert gate and up projection, whose kernel is not in this kernel set",
            backend: "engine",
        });
    };
    let mut plan = route(shape, desc, layer, n_tok, device)?;
    if n_tok < FUSED_MIN_ROWS {
        plan.append(grouped_experts(shape, layer, bits, device)?);
    } else {
        plan.append(sort(shape, n_tok));
        plan.append(fused_experts(shape, layer, bits, n_tok, device)?);
    }
    plan.append(combine(shape, n_tok));
    plan.append(shared(shape, desc, layer, n_tok, device)?);
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_routed_column_is_one_token_and_one_expert_slot() {
        let moe = crate::sparse::shape::MoeShape::new(512, 10, 1, 640);
        assert_eq!(moe.slot_rows(1), 10);
        assert_eq!(moe.slot_rows(512), 5_120);
    }

    #[test]
    fn unrelated_rows_wait_for_a_real_chunk_before_they_group() {
        assert!(!RowOrigin::Sequential.groups(3));
        assert!(RowOrigin::Sequential.groups(4));
        assert!(!RowOrigin::Unrelated.groups(8));
        assert!(RowOrigin::Unrelated.groups(32));
    }

    #[test]
    fn a_down_projection_outside_the_instantiated_types_is_refused_by_name() {
        assert_eq!(down_template_type(DType::Q8_0), Some(8));
        assert_eq!(down_template_type(DType::Q5_1), Some(7));
        assert_eq!(down_template_type(DType::Q4_K), None);
    }
}
