//! The latent routed feed-forward: the experts run inside a narrower space the
//! block input is projected into, and their summed outputs are projected back.
//!
//! ```text
//! xl     = x . D^T                      [T, G * L]   one dense projection
//! z[g]   = sum over the token's columns j with group(e_j) == g of
//!              w_j * Down_e( silu(Gate_e . xl[g]) * (Up_e . xl[g]) )
//! routed = z . U^T                      [T, H]       one dense projection
//! ```
//!
//! An expert reads the latent row of its own group, `labels[expert]`, so the
//! gate and up launches take the activation row `token * G + group` where the
//! plain stage takes `token`. The experts' down projections then write latent
//! rows, and one combine adds them up per (token, group) instead of per token.
//! Combining before `U` is the same linear map as applying `U` to every
//! column, with two ordinary dense projections in place of a per-column one.
//!
//! Everything except the two projections, the labels and the combine is the
//! plain NVFP4 stage: the same router, the same sort, the same expert
//! kernels at the latent width in place of the residual width, and the same
//! shared expert after it. That is why this module plans only the seams.
//!
//! # A step pass skips the groups nobody routes to
//!
//! A token's experts cover a few of the groups, and nothing reads the rest:
//! `D`'s rows for an unused group are no expert's input, and `U`'s terms
//! over an unused group multiply the zeros the combine left. On a step or
//! verify pass, whose projections are bound by the bytes they stream, one
//! small launch writes which groups each block of rows uses and the two
//! projections take entries that leave the others out. Every value that is
//! read is the plain entries' value bit for bit, so which pass a row shares
//! still moves nothing. What it measured is on [`projection`].
//!
//! # Refused rather than approximated
//!
//! A latent bank in any format but NVFP4, over a prompt that quantizes its
//! activations, on a draft head, or on a device whose kernel set lacks the
//! latent entries has no kernel that would give its answer. Each is an error
//! naming itself, because the stages that would run instead read the residual
//! width and would produce finite numbers.

use shiso_core::DType;
use shiso_core::error::{CoreError, CoreResult};

use crate::sparse::bind::{Args, LayerRef, Operand, Role, WeightRef, WeightSet};
use crate::sparse::ggml;
use crate::sparse::kernels::{self, Entry, MOD_GGML, MOD_MODELOPT, MOD_MOE};
use crate::sparse::matmul::{self, Bound, Device, LatentKernels};
use crate::sparse::moe::{self, NVFP4_GEMM_K_ALIGN};
use crate::sparse::plan::{Plan, Step};
use crate::sparse::scratch::Buf;
use crate::sparse::shape::SparseShape;
use crate::sparse::weights::{ExpertDesc, LatentDesc, WeightDesc, WeightPart};

/// Rows a pass may have: the combine puts the token on the grid's second axis,
/// which the driver caps at 65535.
pub const LATENT_MAX_ROWS: u32 = 65_535;

/// Groups a step pass's mask word has a bit for.
pub const LATENT_MASK_GROUPS: u32 = 32;

/// Which side of a step projection the unused groups are on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    /// `D`: the output rows of an unused group.
    Down,
    /// `U`: the contraction terms of an unused group.
    Up,
}

/// Whether this pass's two projections take the entries that skip unused
/// groups: a step or verify pass over `Q8_0` projections on the lanes-a-row
/// step kernels, with few enough groups for a mask word.
fn masks(latent: &LatentDesc, n_tok: u32, device: Device) -> bool {
    latent.groups <= LATENT_MASK_GROUPS
        && latent.down.dtype == DType::Q8_0
        && latent.up.dtype == DType::Q8_0
        && matches!(matmul::step_tier(device), ggml::StepTier::Runs)
        && matmul::ggml_step(DType::Q8_0, ggml::step_stage(n_tok).0, device).is_some()
}

/// One of the stage's two projections: the plain projection, or on a pass
/// [`masks`] accepts the step entry that leaves the unused groups out.
///
/// The masked entries are the plain step entries with two more arguments
/// (`out, w, x, in_dim, out_dim, rows, lanes, masks, width`), on the plain
/// launch.
///
/// Measured on a GB10 serving a 48-layer pack with eight groups of 1792,
/// whose two projections are 78 MB of the 190 MB a layer streams a token:
/// plain decode 18.9 -> 21.7 tok/s, greedy text unchanged. A five-row verify
/// pass does not move (27-43 tok/s either way): its rows' groups together
/// cover nearly every group, and a group any row uses is read for all.
fn projection(
    plan: &mut Plan,
    weight: WeightDesc,
    bound: Bound,
    side: Side,
    latent: &LatentDesc,
    pass: (u32, Device),
) -> CoreResult<()> {
    let (n_tok, device) = pass;
    if !masks(latent, n_tok, device) {
        return matmul::project(plan, weight, bound, n_tok, device);
    }
    let four = ggml::step_stage(n_tok).1 == 4;
    let name = match (side, four) {
        (Side::Down, false) => "q4e_g_step_latent_down_q8_0_kernel",
        (Side::Down, true) => "q4e_g_step_latent_down4_q8_0_kernel",
        (Side::Up, false) => "q4e_g_step_latent_up_q8_0_kernel",
        (Side::Up, true) => "q4e_g_step_latent_up4_q8_0_kernel",
    };
    let lanes = ggml::step_lanes(weight.in_dim, matmul::step_tier(device));
    plan.launch(
        Entry::plain(MOD_GGML, name),
        ggml::step_launch(weight.out_dim, n_tok, lanes),
        Args::of(&[
            bound.dst,
            Operand::Weight(bound.weight),
            bound.src,
            Operand::U32(weight.in_dim),
            Operand::U32(weight.out_dim),
            Operand::U32(n_tok),
            Operand::U32(lanes),
            Operand::Scratch(Buf::LatentMask),
            Operand::U32(latent.width),
        ]),
    );
    Ok(())
}

/// A refusal by name.
fn refuse(what: &'static str) -> CoreError {
    CoreError::Unsupported {
        what,
        backend: "engine",
    }
}

/// Checks everything the latent stage depends on and returns the stage's
/// widths.
///
/// # Errors
///
/// [`CoreError::Unsupported`] for each case the module documentation lists,
/// and for a bank whose widths disagree with the latent widths.
fn check(
    shape: &SparseShape,
    desc: &ExpertDesc,
    layer: LayerRef,
    device: Device,
) -> CoreResult<LatentDesc> {
    let latent = desc.latent.ok_or(refuse(
        "a latent routed stage over a bank with no latent stage",
    ))?;
    if layer.set != WeightSet::Target {
        return Err(refuse("a latent routed stage on a draft head"));
    }
    if [desc.gate, desc.up, desc.down]
        .iter()
        .any(|w| w.dtype != DType::Nvfp4Gs16)
    {
        return Err(refuse(
            "a latent routed stage over experts that are not NVFP4",
        ));
    }
    if device.lowp_prompt() {
        return Err(refuse(
            "a latent routed stage in a prompt pass that quantizes its activations",
        ));
    }
    if device.latent_kernels != LatentKernels::Present {
        return Err(refuse(
            "a latent routed stage on a device whose kernel set lacks the latent expert kernels",
        ));
    }
    let hidden = shape.hidden;
    let inter = shape.moe.intermediate;
    let rows = latent.groups.checked_mul(latent.width);
    if latent.groups == 0 || latent.width == 0 || rows.is_none() {
        return Err(refuse("a latent routed stage with no groups or no width"));
    }
    // The experts' down output is staged in a buffer sized for the residual
    // width.
    if latent.width > hidden {
        return Err(refuse(
            "a latent routed stage wider than the residual width its scratch is sized for",
        ));
    }
    let widths = [
        (desc.gate, latent.width, inter),
        (desc.up, latent.width, inter),
        (desc.down, inter, latent.width),
    ];
    if widths
        .iter()
        .any(|(w, i, o)| w.in_dim != *i || w.out_dim != *o)
    {
        return Err(refuse(
            "latent routed experts whose widths are not the latent width and the inner width",
        ));
    }
    let rows = rows.unwrap_or(0);
    if latent.down.in_dim != hidden || latent.down.out_dim != rows {
        return Err(refuse(
            "a latent down projection that is not residual width to groups times latent width",
        ));
    }
    if latent.up.in_dim != rows || latent.up.out_dim != hidden {
        return Err(refuse(
            "a latent up projection that is not groups times latent width to residual width",
        ));
    }
    Ok(latent)
}

/// One latent routed feed-forward, in launch order: the router and the sort,
/// the projection into the latent rows, the experts, the per-group combine,
/// the projection back, and the shared expert.
///
/// # Errors
///
/// [`CoreError::Unsupported`] for each case the module documentation lists,
/// for widths that disagree with the latent stage's or that the NVFP4 prompt
/// kernels cannot stride over, and whatever
/// [`matmul::project`] and the shared expert's planner report.
pub fn block(
    shape: &SparseShape,
    desc: &ExpertDesc,
    layer: LayerRef,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let latent = check(shape, desc, layer, device)?;
    if n_tok > LATENT_MAX_ROWS {
        return Err(refuse(
            "a latent routed stage over more rows than a launch grid has rows for",
        ));
    }
    // Never the fused route launch: it ends in a combine that adds the shared
    // expert to the routed sum, and this stage's routed sum is the up
    // projection's, which has to land in `BlockOut` first.
    let mut plan = if moe::route_sorts(shape, n_tok) {
        let mut plan = moe::router_logits(desc, layer, n_tok, device)?;
        plan.append(moe::route_sort(shape, n_tok, device));
        plan
    } else {
        let mut plan = moe::route(shape, desc, layer, n_tok, device)?;
        plan.append(moe::sort(shape, n_tok));
        plan
    };
    if masks(&latent, n_tok, device) {
        plan.launch(
            Entry::plain(MOD_MOE, "q4e_latent_groups_kernel"),
            kernels::latent_groups(n_tok),
            // `masks, ids, labels, n_tok, n_used`
            Args::of(&[
                Operand::Scratch(Buf::LatentMask),
                Operand::Scratch(Buf::MoeIds),
                Operand::ExpertLabels(layer.index),
                Operand::U32(n_tok),
                Operand::U32(shape.moe.used),
            ]),
        );
    }
    projection(
        &mut plan,
        latent.down,
        Bound::new(
            WeightRef::layer(layer, Role::LatentDown),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::Latent),
        ),
        Side::Down,
        &latent,
        (n_tok, device),
    )?;
    plan.push(Step::Fetch {
        layer: layer.index,
        ids: u32::try_from(shape.moe.slot_rows(n_tok)).unwrap_or(u32::MAX),
    });
    plan.append(experts(shape, &latent, layer, n_tok, device)?);
    plan.launch(
        Entry::plain(MOD_MOE, "q4e_latent_combine_kernel"),
        kernels::latent_combine(latent.width, latent.groups, n_tok),
        // `out, down, weights, ids, labels, width, n_groups, n_used`
        Args::of(&[
            Operand::Scratch(Buf::LatentSum),
            Operand::Scratch(Buf::MoeDown),
            Operand::Scratch(Buf::MoeWeights),
            Operand::Scratch(Buf::MoeIds),
            Operand::ExpertLabels(layer.index),
            Operand::U32(latent.width),
            Operand::U32(latent.groups),
            Operand::U32(shape.moe.used),
        ]),
    );
    projection(
        &mut plan,
        latent.up,
        Bound::new(
            WeightRef::layer(layer, Role::LatentUp),
            Operand::Scratch(Buf::LatentSum),
            Operand::Scratch(Buf::BlockOut),
        ),
        Side::Up,
        &latent,
        (n_tok, device),
    )?;
    moe::with_shared(shape, desc, layer, n_tok, device, plan)
}

/// The gate-and-up, activation and down launches over the latent rows.
///
/// The plain NVFP4 stage's entries and grids, at the latent width: a prompt
/// pass takes the tensor-core tiles, a step the warp-per-row kernels (one
/// token's, or the held-row entry for a wider pass), and gate and up take the
/// labelled twins that pick each column's activation row by its expert's
/// group.
fn experts(
    shape: &SparseShape,
    latent: &LatentDesc,
    layer: LayerRef,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let inter = shape.moe.intermediate;
    let width = latent.width;
    let gemm = device.row_invariant();
    if gemm
        && !(width.is_multiple_of(NVFP4_GEMM_K_ALIGN) && inter.is_multiple_of(NVFP4_GEMM_K_ALIGN))
    {
        return Err(refuse(
            "a latent prompt expert stage whose widths are not multiples of 128",
        ));
    }
    let columns = u32::try_from(shape.moe.slot_rows(n_tok)).unwrap_or(u32::MAX);
    let groups = columns.min(shape.moe.experts);
    let at = |role: Role, part| Operand::Weight(WeightRef::layer(layer, role).spanning(part));
    let map = Operand::ExpertMap(layer.index);
    let tail = |in_dim: u32, out_dim: u32| {
        [
            Operand::Scratch(Buf::MoeSorted),
            Operand::Scratch(Buf::MoeBounds),
            Operand::Scratch(Buf::MoeActive),
            map,
            Operand::U32(in_dim),
            Operand::U32(out_dim),
            Operand::U32(shape.moe.experts),
            Operand::U32(shape.moe.used),
        ]
    };
    let (gate_up, down) = if gemm {
        (
            "q4e_nvfp4_latent_gemm_gate_up_kernel",
            "q4e_nvfp4_experts_gemm_down_kernel",
        )
    } else {
        (
            if n_tok == 1 {
                "q4e_nvfp4_latent_gate_up_act_kernel"
            } else {
                "q4e_nvfp4_latent_gate_up_act_held_kernel"
            },
            "q4e_nvfp4_experts_down_kernel",
        )
    };
    let (gate_up_launch, down_launch) = moe::nvfp4_grids(gemm, inter, width, groups);
    // `out_gate, out_up, gate_w, gate_s, gate_g, up_w, up_s, up_g, x, sorted,
    //  bounds, active, map, in_dim, out_dim, n_expert, n_used, labels,
    //  n_groups`, and on a step the one output `out_mid` in place of the
    // first two.
    let outputs: &[Operand] = if gemm {
        &[Operand::Scratch(Buf::MoeGate), Operand::Scratch(Buf::MoeUp)]
    } else {
        &[Operand::Scratch(Buf::MoeMid)]
    };
    let mut ops = outputs.to_vec();
    ops.extend([
        at(Role::ExpertGate, WeightPart::Whole),
        at(Role::ExpertGate, WeightPart::GroupScales),
        at(Role::ExpertGate, WeightPart::GlobalScale),
        at(Role::ExpertUp, WeightPart::Whole),
        at(Role::ExpertUp, WeightPart::GroupScales),
        at(Role::ExpertUp, WeightPart::GlobalScale),
        Operand::Scratch(Buf::Latent),
    ]);
    ops.extend(tail(width, inter));
    ops.extend([
        Operand::ExpertLabels(layer.index),
        Operand::U32(latent.groups),
    ]);
    let mut plan = Plan::new();
    plan.launch(
        Entry::plain(MOD_MODELOPT, gate_up),
        gate_up_launch,
        Args::of(&ops),
    );
    if gemm {
        plan.append(moe::activate(
            shape,
            Operand::Scratch(Buf::MoeGate),
            Operand::Scratch(Buf::MoeUp),
            Operand::Scratch(Buf::MoeMid),
            n_tok,
        ));
    }
    // `out, down_w, down_s, down_g, mid, sorted, bounds, active, map, in_dim,
    //  out_dim, n_expert, n_used`
    let mut ops = vec![
        Operand::Scratch(Buf::MoeDown),
        at(Role::ExpertDown, WeightPart::Whole),
        at(Role::ExpertDown, WeightPart::GroupScales),
        at(Role::ExpertDown, WeightPart::GlobalScale),
        Operand::Scratch(Buf::MoeMid),
    ];
    ops.extend(tail(inter, width));
    plan.launch(
        Entry::plain(MOD_MODELOPT, down),
        down_launch,
        Args::of(&ops),
    );
    Ok(plan)
}

#[cfg(test)]
#[path = "latent_tests.rs"]
mod tests;
