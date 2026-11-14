//! The latent stage's plan: the order of its stages, what it refuses, that a
//! bank without it plans as before, and what its scratch costs.

use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use shiso_core::DType;
use shiso_core::backend::{Backend, KernelRef, LaunchArgs};
use shiso_core::buffer::{DeviceView, RawBuffer};
use shiso_core::error::{CoreError, CoreResult};

use super::*;
use crate::shape::{GdnShape, LayerKind};
use crate::sparse::scratch::Scratch;
use crate::sparse::shape::{GqaShape, IndexerShape, MoeShape, MropeSections};
use crate::sparse::weights::WeightDesc;

const HIDDEN: u32 = 2_560;
const INTER: u32 = 640;
const GROUPS: u32 = 8;
const WIDTH: u32 = 1_792;

/// The published geometry, four layers of it.
fn sparse_shape() -> SparseShape {
    SparseShape {
        layers: vec![
            LayerKind::Gdn,
            LayerKind::Gdn,
            LayerKind::Gdn,
            LayerKind::Attn,
        ],
        hidden: HIDDEN,
        vocab: 248_320,
        streams: 4,
        stream_rank: 320,
        gqa: GqaShape::new(24, 2, 256, 64),
        gdn: GdnShape::new(48, 16, 128, 4),
        moe: MoeShape::new(512, 10, 1, INTER),
        indexer: IndexerShape::new(4, 128, 2_048, 4),
        ple: None,
        mrope: MropeSections::new(11, 11, 10),
        rms_eps: 1.0e-6,
        rope_theta: 1.0e7,
    }
}

fn w(dtype: DType, in_dim: u32, out_dim: u32) -> WeightDesc {
    WeightDesc::new(dtype, in_dim, out_dim)
}

/// An NVFP4 bank at the published widths, without a latent stage.
fn plain_desc() -> ExpertDesc {
    ExpertDesc {
        router: w(DType::BF16, HIDDEN, 512),
        gate: w(DType::Nvfp4Gs16, HIDDEN, INTER),
        up: w(DType::Nvfp4Gs16, HIDDEN, INTER),
        down: w(DType::Nvfp4Gs16, INTER, HIDDEN),
        shared_router: w(DType::BF16, HIDDEN, 1),
        shared_gate: w(DType::Mxfp8E4M3G32, HIDDEN, INTER),
        shared_up: w(DType::Mxfp8E4M3G32, HIDDEN, INTER),
        shared_down: w(DType::Mxfp8E4M3G32, INTER, HIDDEN),
        latent: None,
    }
}

/// The same bank with the latent stage of the published checkpoint: experts at
/// 1792, eight groups, `Q8_0` projections either side.
fn latent_desc() -> ExpertDesc {
    ExpertDesc {
        gate: w(DType::Nvfp4Gs16, WIDTH, INTER),
        up: w(DType::Nvfp4Gs16, WIDTH, INTER),
        down: w(DType::Nvfp4Gs16, INTER, WIDTH),
        latent: Some(LatentDesc {
            down: w(DType::Q8_0, HIDDEN, GROUPS * WIDTH),
            up: w(DType::Q8_0, GROUPS * WIDTH, HIDDEN),
            groups: GROUPS,
            width: WIDTH,
        }),
        ..plain_desc()
    }
}

/// A step device on the CUDA kernel set, latent entries included.
fn step() -> Device {
    Device {
        blas: true,
        prompt_matmul: matmul::PromptMatmul::Tensor,
        route_sort: matmul::RouteSort::Fused,
        latent_kernels: LatentKernels::Present,
        ..Device::default()
    }
}

/// The same device in a prompt pass.
fn prompt() -> Device {
    Device {
        role: matmul::PassRole::Prompt { snapshot: None },
        ..step()
    }
}

fn device_for(n_tok: u32) -> Device {
    if n_tok > 32 { prompt() } else { step() }
}

fn names(plan: &Plan) -> Vec<&'static str> {
    plan.steps()
        .iter()
        .filter_map(Step::entry)
        .map(|e| e.name())
        .collect()
}

/// Index of the first step whose entry is `name`.
fn at(plan: &Plan, name: &str) -> usize {
    plan.steps()
        .iter()
        .position(|s| s.entry().is_some_and(|e| e.name() == name))
        .unwrap_or_else(|| panic!("{name} is not in {:?}", names(plan)))
}

/// Index of the first kernel step for which `pred(entry name, operands)` holds.
fn find(plan: &Plan, what: &str, pred: impl Fn(&str, &[Operand]) -> bool) -> usize {
    plan.steps()
        .iter()
        .position(|s| {
            s.entry()
                .is_some_and(|e| pred(e.name(), s.operands().expect("a launch's operands")))
        })
        .unwrap_or_else(|| panic!("no {what} in {:?}", names(plan)))
}

fn refused(result: CoreResult<Plan>) -> &'static str {
    match result.expect_err("a refusal") {
        CoreError::Unsupported { what, .. } => what,
        other => panic!("an unsupported refusal, not {other:?}"),
    }
}

#[test]
fn latent_plan_orders_stages() {
    let shape = sparse_shape();
    let desc = latent_desc();
    let layer = LayerRef::target(0);
    for (n_tok, gate_up) in [
        (1, "q4e_nvfp4_latent_gate_up_act_kernel"),
        (4, "q4e_nvfp4_latent_gate_up_act_held_kernel"),
        (512, "q4e_nvfp4_latent_gemm_gate_up_kernel"),
    ] {
        let plan = block(&shape, &desc, layer, n_tok, device_for(n_tok)).expect("plans");
        plan.validate().expect("every launch has work");
        let names = names(&plan);
        let msg = format!("{n_tok} tokens: {names:?}");
        assert!(!plan.launches("q4e_ffn_route_kernel"), "{msg}");
        assert!(!plan.launches("q4e_ffn_route_rows_kernel"), "{msg}");
        assert!(!plan.launches("q4e_ffn_combine_kernel"), "{msg}");
        let route = at(
            &plan,
            names
                .iter()
                .find(|n| n.starts_with("q4e_moe_route"))
                .expect(&msg),
        );
        // D: the block input into the latent rows, whichever kernel the Q8_0
        // weight takes at this width.
        let project_down = find(&plan, "D projection", |_, ops| {
            ops.first() == Some(&Operand::Scratch(Buf::Latent))
                && ops.get(2) == Some(&Operand::Scratch(Buf::Mixed))
        });
        let fetch = plan
            .steps()
            .iter()
            .position(|s| matches!(s, Step::Fetch { layer: 0, .. }))
            .expect(&msg);
        let gate_up_at = at(&plan, gate_up);
        let down = find(&plan, "expert down", |name, ops| {
            name.starts_with("q4e_nvfp4_experts") && name.ends_with("down_kernel") && {
                // `out, w, s, g, mid, sorted, bounds, active, map, in_dim, out_dim, ..`
                ops.get(9) == Some(&Operand::U32(INTER))
                    && ops.get(10) == Some(&Operand::U32(WIDTH))
            }
        });
        let combine = at(&plan, "q4e_latent_combine_kernel");
        let project_up = find(&plan, "U projection", |_, ops| {
            ops.first() == Some(&Operand::Scratch(Buf::BlockOut))
                && ops.get(2) == Some(&Operand::Scratch(Buf::LatentSum))
        });
        let shared_add = at(&plan, "q4e_shared_add_kernel");
        assert!(
            route < project_down
                && project_down < fetch
                && fetch < gate_up_at
                && gate_up_at < down
                && down < combine
                && combine < project_up
                && project_up < shared_add,
            "{msg}: {route} {project_down} {fetch} {gate_up_at} {down} {combine} {project_up} {shared_add}"
        );
        assert_eq!(
            shared_add,
            plan.len() - 1,
            "the shared expert closes: {msg}"
        );
        // The plain stage's entries never run: they would read the block
        // input and write residual-width rows.
        assert!(
            names
                .iter()
                .all(|n| !n.starts_with("q4e_nvfp4_experts_gate_up")
                    && !n.starts_with("q4e_nvfp4_experts_gemm_gate_up")),
            "{msg}"
        );
        // Gate and up read the latent rows, with the labels and the group
        // count appended.
        let ops = plan.steps()[gate_up_at].operands().expect("operands");
        assert!(ops.contains(&Operand::Scratch(Buf::Latent)), "{msg}");
        assert!(!ops.contains(&Operand::Scratch(Buf::Mixed)), "{msg}");
        assert_eq!(ops[ops.len() - 2], Operand::ExpertLabels(0), "{msg}");
        assert_eq!(ops[ops.len() - 1], Operand::U32(GROUPS), "{msg}");
        // The combine: `out, down, weights, ids, labels, width, n_groups, n_used`.
        assert_eq!(
            plan.steps()[combine].operands().expect("operands"),
            [
                Operand::Scratch(Buf::LatentSum),
                Operand::Scratch(Buf::MoeDown),
                Operand::Scratch(Buf::MoeWeights),
                Operand::Scratch(Buf::MoeIds),
                Operand::ExpertLabels(0),
                Operand::U32(WIDTH),
                Operand::U32(GROUPS),
                Operand::U32(10),
            ],
            "{msg}"
        );
    }
}

#[test]
fn latent_launch_grids_use_the_latent_width() {
    let shape = sparse_shape();
    let desc = latent_desc();
    let layer = LayerRef::target(0);
    // The step grid: 4 gate-and-up rows and 8 down rows a block, per expert.
    let plan = block(&shape, &desc, layer, 1, step()).expect("plans");
    let launch = |name: &str| match plan.steps()[at(&plan, name)] {
        Step::Kernel { launch, .. } => launch,
        other => panic!("a kernel, not {other:?}"),
    };
    let (gate_up_grid, down_grid) = moe::nvfp4_grids(false, INTER, WIDTH, 10);
    assert_eq!(launch("q4e_nvfp4_latent_gate_up_act_kernel"), gate_up_grid);
    assert_eq!(launch("q4e_nvfp4_experts_down_kernel"), down_grid);
    assert_eq!(
        launch("q4e_latent_combine_kernel"),
        kernels::latent_combine(WIDTH, GROUPS, 1)
    );
    // 8 * 1792 elements a token, 256 threads a block, one grid row a token.
    assert_eq!(
        kernels::latent_combine(WIDTH, GROUPS, 3).grid,
        shiso_core::Extent3::new(56, 3, 1)
    );
}

#[test]
fn latent_plans_full_prompt_chunk() {
    let shape = sparse_shape();
    let plan = block(&shape, &latent_desc(), LayerRef::target(0), 8_192, prompt())
        .expect("a default prompt chunk plans");
    match plan.steps()[at(&plan, "q4e_latent_combine_kernel")] {
        Step::Kernel { launch, .. } => {
            assert_eq!(launch.grid, shiso_core::Extent3::new(56, 8_192, 1));
        }
        other => panic!("a kernel, not {other:?}"),
    }
    block(
        &shape,
        &latent_desc(),
        LayerRef::target(0),
        65_535,
        prompt(),
    )
    .expect("the widest");
    let what = refused(block(
        &shape,
        &latent_desc(),
        LayerRef::target(0),
        65_536,
        prompt(),
    ));
    assert!(what.contains("grid has rows"), "{what}");
}

#[test]
fn latent_refuses_non_nvfp4_bank() {
    let shape = sparse_shape();
    let layer = LayerRef::target(0);
    // Through `moe::block`, so that a format with a planner of its own cannot
    // take a latent bank around the refusal.
    for dtype in [DType::Fp8E4M3Block128, DType::MlxAffine4Gs64, DType::Q8_0] {
        let desc = ExpertDesc {
            gate: w(dtype, WIDTH, INTER),
            ..latent_desc()
        };
        for device in [step(), prompt()] {
            let what = refused(moe::block(&shape, &desc, layer, 4, device));
            assert!(what.contains("not NVFP4"), "{dtype:?}: {what}");
        }
    }
}

#[test]
fn latent_refuses_lowp_activations() {
    let shape = sparse_shape();
    let device = Device {
        prefill_activations: matmul::PrefillActivations::Lowp,
        ..prompt()
    };
    let what = refused(block(
        &shape,
        &latent_desc(),
        LayerRef::target(0),
        512,
        device,
    ));
    assert!(what.contains("quantizes its activations"), "{what}");
    // A step pass keeps its weight-only kernels under the option, as the
    // plain stage does.
    let step_device = Device {
        prefill_activations: matmul::PrefillActivations::Lowp,
        ..step()
    };
    block(&shape, &latent_desc(), LayerRef::target(0), 1, step_device).expect("a step plans");
}

#[test]
fn latent_refuses_backend_without_kernels() {
    let shape = sparse_shape();
    let device = Device {
        latent_kernels: LatentKernels::Absent,
        ..step()
    };
    let what = refused(block(
        &shape,
        &latent_desc(),
        LayerRef::target(0),
        1,
        device,
    ));
    assert!(what.contains("lacks the latent expert kernels"), "{what}");
}

#[test]
fn latent_refuses_a_draft_head_and_widths_that_disagree() {
    let shape = sparse_shape();
    let layer = LayerRef::target(0);
    let what = refused(block(&shape, &latent_desc(), LayerRef::draft(), 1, step()));
    assert!(what.contains("draft head"), "{what}");

    let mut desc = latent_desc();
    desc.gate = w(DType::Nvfp4Gs16, HIDDEN, INTER);
    assert!(refused(block(&shape, &desc, layer, 1, step())).contains("latent width"));
    let mut desc = latent_desc();
    desc.latent = desc.latent.map(|l| LatentDesc {
        down: w(DType::Q8_0, HIDDEN, WIDTH),
        ..l
    });
    assert!(refused(block(&shape, &desc, layer, 1, step())).contains("down projection"));
    let mut desc = latent_desc();
    desc.latent = desc.latent.map(|l| LatentDesc {
        up: w(DType::Q8_0, GROUPS * WIDTH, HIDDEN + 32),
        ..l
    });
    assert!(refused(block(&shape, &desc, layer, 1, step())).contains("up projection"));
    // The down output is staged at the residual width.
    let mut wide = latent_desc();
    wide.latent = wide.latent.map(|l| LatentDesc {
        width: HIDDEN + 128,
        ..l
    });
    assert!(refused(block(&shape, &wide, layer, 1, step())).contains("wider than"));
    // A prompt stage strides over its contraction in 128s.
    let mut odd = latent_desc();
    odd.gate = w(DType::Nvfp4Gs16, 1_776, INTER);
    odd.up = odd.gate;
    odd.down = w(DType::Nvfp4Gs16, INTER, 1_776);
    odd.latent = odd.latent.map(|l| LatentDesc {
        down: w(DType::Q8_0, HIDDEN, GROUPS * 1_776),
        up: w(DType::Q8_0, GROUPS * 1_776, HIDDEN),
        width: 1_776,
        ..l
    });
    assert!(refused(block(&shape, &odd, layer, 512, prompt())).contains("multiples of 128"));
    block(&shape, &odd, layer, 1, step()).expect("a step has no such stride");
}

/// FNV-1a over a plan's debug text: every entry, grid and operand of it.
fn digest(plan: &Plan) -> u64 {
    format!("{:?}", plan.steps())
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
}

#[test]
fn plain_bank_plan_unchanged() {
    let shape = sparse_shape();
    let desc = plain_desc();
    let layer = LayerRef::target(0);
    // Digests of the plans this bank had before the latent stage existed,
    // taken from the commit before it: the extraction of the NVFP4 grids and
    // the dispatch in front of `staged_block` move no launch of a plain bank.
    for (n_tok, device, want) in [
        (1, step(), PLAIN_STEP_1),
        (4, step(), PLAIN_STEP_4),
        (512, prompt(), PLAIN_PROMPT_512),
    ] {
        let plan = moe::block(&shape, &desc, layer, n_tok, device).expect("plans");
        assert!(
            names(&plan).iter().all(|n| !n.contains("latent")),
            "{:?}",
            names(&plan)
        );
        assert_eq!(digest(&plan), want, "{n_tok} tokens: {:?}", names(&plan));
    }
}

// How these were produced: `digest` above (FNV-1a over the `Debug` text
// of `plan.steps()`) of `moe::block` for `plain_desc()` at the devices named in
// `plain_bank_plan_unchanged`, run at commit f4a9e2ed (before the latent
// stage) in a clean worktree with an equivalent throwaway test that printed
// the three values. To regenerate after an intended change to a plain bank's
// plan, print `digest(&plan)` in that test, check the plan diff is intended,
// and paste the new values.
const PLAIN_STEP_1: u64 = 16_962_141_729_033_441_160;
const PLAIN_STEP_4: u64 = 4_646_864_847_006_760_388;
const PLAIN_PROMPT_512: u64 = 5_869_114_533_540_638_767;

/// Allocations with ids and lengths and no bytes.
#[derive(Debug, Default)]
struct Sizes {
    next: AtomicU64,
}

impl Backend for Sizes {
    fn backend_name(&self) -> &'static str {
        "sizes"
    }

    fn alloc_raw(&self, bytes: u64) -> CoreResult<RawBuffer> {
        let id = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        Ok(RawBuffer::new(
            NonZeroU64::new(id).expect("ids start at one"),
            bytes,
        ))
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

    fn run(&self, _kernel: KernelRef, _args: &LaunchArgs<'_>) -> CoreResult<()> {
        Ok(())
    }

    fn sync(&self) -> CoreResult<()> {
        Ok(())
    }
}

fn scratch(rows: u32, latent_columns: u32) -> Scratch {
    let backend: Arc<dyn Backend> = Arc::new(Sizes::default());
    Scratch::new(
        &backend,
        &sparse_shape(),
        rows,
        8,
        16_384,
        0,
        latent_columns,
    )
    .expect("size-only allocations")
}

#[test]
fn scratch_latent_zero_without_latent_layers() {
    let scratch = scratch(512, 0);
    for buf in [Buf::Latent, Buf::LatentSum, Buf::LatentMask] {
        assert_eq!(scratch.view(buf).expect("allocated").len(), 0, "{buf:?}");
    }
    // The other buffers are as before.
    assert_eq!(
        scratch.view(Buf::MoeDown).expect("allocated").len(),
        512 * 10 * 2_560 * 4
    );
}

#[test]
fn scratch_latent_sized() {
    for rows in [1_u32, 4, 512] {
        let scratch = scratch(rows, GROUPS * WIDTH);
        let want = u64::from(rows) * 8 * 1_792 * 4;
        for buf in [Buf::Latent, Buf::LatentSum] {
            assert_eq!(scratch.view(buf).expect("allocated").len(), want, "{buf:?}");
        }
        // One mask word per block of eight rows.
        assert_eq!(
            scratch.view(Buf::LatentMask).expect("allocated").len(),
            u64::from(rows.div_ceil(8)) * 4,
            "mask words for {rows} rows"
        );
    }
    // The ledger's tag and the zero the shape-only sizing answers.
    assert_eq!(Buf::Latent.bytes(&sparse_shape(), 512, 8, 0), 0);
    assert_eq!(Buf::Latent.tag(), shiso_core::ledger::MemTag::MoeScratch);
}

/// A step or verify pass writes the groups its rows use once, before the
/// down projection, and both projections take the entries that leave the
/// rest out; a prompt pass, whose projections are the tensor-core tiles,
/// launches none of the three.
#[test]
fn a_step_pass_skips_unused_groups_and_a_prompt_pass_does_not() {
    let shape = sparse_shape();
    let desc = latent_desc();
    let layer = LayerRef::target(0);
    for (n_tok, down, up) in [
        (
            1,
            "q4e_g_step_latent_down_q8_0_kernel",
            "q4e_g_step_latent_up_q8_0_kernel",
        ),
        (
            5,
            "q4e_g_step_latent_down4_q8_0_kernel",
            "q4e_g_step_latent_up4_q8_0_kernel",
        ),
    ] {
        let plan = block(&shape, &desc, layer, n_tok, step()).expect("plans");
        plan.validate().expect("every launch has work");
        let groups = at(&plan, "q4e_latent_groups_kernel");
        let (down, up) = (at(&plan, down), at(&plan, up));
        let route = at(
            &plan,
            names(&plan)
                .iter()
                .find(|n| n.starts_with("q4e_moe_route"))
                .expect("a route launch"),
        );
        assert!(
            route < groups && groups < down && down < up,
            "{:?}",
            names(&plan)
        );
        for at in [down, up] {
            let Step::Kernel { args, launch, .. } = &plan.steps()[at] else {
                panic!("a projection is a kernel step");
            };
            let ops = args.operands().expect("plain operands");
            // `out, w, x, in_dim, out_dim, rows, lanes, masks, width`
            assert_eq!(ops.len(), 9);
            assert_eq!(ops[7], Operand::Scratch(Buf::LatentMask));
            assert_eq!(ops[8], Operand::U32(WIDTH));
            // The plain step launch at this pass width.
            assert_eq!(launch.grid.y, n_tok.div_ceil(ggml::STEP_COLS));
        }
    }
    let plan = block(&shape, &desc, layer, 512, prompt()).expect("plans");
    for name in names(&plan) {
        assert!(
            name != "q4e_latent_groups_kernel" && !name.starts_with("q4e_g_step_latent"),
            "a prompt pass launched {name}"
        );
    }
    // More groups than a mask word has bits keeps the plain projections.
    let mut wide = latent_desc();
    if let Some(latent) = wide.latent.as_mut() {
        latent.groups = LATENT_MASK_GROUPS + 1;
    }
    assert!(!masks(&wide.latent.expect("latent"), 1, step()));
}
