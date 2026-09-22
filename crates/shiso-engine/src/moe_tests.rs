//! The eight launches a routed block issues, and what each is handed.
//!
//! Not a numerical check - the kernels are graded in
//! `crates/shiso-kerneltest/src/qwen35moe/experts.rs`, stage by stage, on a
//! device. This is the other half: **the sequence itself is new**, and every
//! seam in it is a convention neither kernel can check. The routed column is
//! `token * used + slot`; the gate and up read activation row `col / used`
//! while the down reads its own row `col`; the down writes column-major over
//! (token, slot) because that is what the combine reads; the per-expert scale
//! is an array indexed by expert id.
//!
//! A kernel handed the wrong buffer for any of those runs, finishes and writes
//! a plausible number.

use std::sync::Arc;

use shiso_core::backend::Backend;
use std::num::NonZeroU64;

use shiso_core::{CoreError, DType, DeviceBuffer, Extent3};

use super::{ActDtype, ExpertCodec, ExpertSlab, Moe, MoeIo, MoeScratch, Q4E_MOE};
use crate::ffn::Ffn;
use crate::projection::Weight;
use crate::recorder::{Bytes, Launch, Recorder};
use crate::shape::MoeShape;

/// A routed block small enough to reason about and awkward enough to catch a
/// stride: six experts, two per token, a hidden of 32 and an inner of 16.
const HIDDEN: u32 = 32;
const INTER: u32 = 16;
const EXPERTS: u32 = 6;
const USED: u32 = 2;

fn shape() -> MoeShape {
    MoeShape::new(EXPERTS, USED, 1, INTER)
}

/// An NVFP4 weight of `[out_dim, in_dim]`, through the real constructor.
fn nvfp4(backend: &Arc<dyn Backend>, in_dim: u32, out_dim: u32) -> Weight {
    let packed = (in_dim / 2) as usize * out_dim as usize;
    let groups = (in_dim / 16) as usize * out_dim as usize;
    let mut host = vec![0x11u8; packed];
    host.extend(std::iter::repeat_n(0x38u8, groups));
    Weight::import_nvfp4(
        backend,
        in_dim,
        out_dim,
        Arc::new(Bytes(host)),
        0..packed,
        packed..packed + groups,
        3.1e-5,
    )
    .expect("an nvfp4 weight")
}

/// A `[out_dim, in_dim]` f32 weight, for the router and the shared gate.
fn dense(backend: &Arc<dyn Backend>, in_dim: u32, out_dim: u32) -> Weight {
    let bytes = in_dim as usize * out_dim as usize * 4;
    Weight::import(
        backend,
        DType::F32,
        in_dim,
        out_dim,
        Arc::new(Bytes(vec![0u8; bytes])),
        0..bytes,
    )
    .expect("a dense weight")
}

/// An NVFP4 expert slab of `experts` matrices of `[out_dim, in_dim]`.
fn slab(backend: &Arc<dyn Backend>, in_dim: u32, out_dim: u32) -> ExpertSlab {
    let packed = (u64::from(in_dim) / 2) * u64::from(out_dim) * u64::from(EXPERTS);
    let groups = (u64::from(in_dim) / 16) * u64::from(out_dim) * u64::from(EXPERTS);
    ExpertSlab::Nvfp4 {
        weights: shiso_core::alloc(backend, packed).expect("allocation"),
        scales: shiso_core::alloc(backend, groups).expect("allocation"),
        globals: shiso_core::alloc(backend, u64::from(EXPERTS) * 4).expect("allocation"),
    }
}

/// An affine group-64 expert slab of the same shape.
///
/// Every buffer is a DIFFERENT size from every other, so the three ids the
/// launch carries cannot coincide and an assertion on their order is an
/// assertion about order rather than about three interchangeable handles.
fn affine_slab(backend: &Arc<dyn Backend>, in_dim: u32, out_dim: u32) -> ExpertSlab {
    let packed = (u64::from(in_dim) / 2) * u64::from(out_dim) * u64::from(EXPERTS);
    let groups = (u64::from(in_dim) / 64).max(1) * u64::from(out_dim) * u64::from(EXPERTS);
    ExpertSlab::Q4Affine {
        weights: shiso_core::alloc(backend, packed).expect("allocation"),
        scales: shiso_core::alloc(backend, groups * 2).expect("allocation"),
        biases: shiso_core::alloc(backend, groups * 2).expect("allocation"),
    }
}

/// A `ggml` super-block stack: one buffer, because the format carries its
/// scales inside the block.
///
/// The byte count is what the codec's block size implies for the shape, so a
/// fixture cannot state a slab the entries would read past the end of.
fn block_slab(
    backend: &Arc<dyn Backend>,
    in_dim: u32,
    out_dim: u32,
    codec: ExpertCodec,
) -> ExpertSlab {
    let (elements, bytes) = match codec {
        ExpertCodec::Q4K => (256u64, 144u64),
        ExpertCodec::Q5K => (256, 176),
        ExpertCodec::Q6K => (256, 210),
        ExpertCodec::Q8_0 => (32, 34),
        other => panic!("{} is not a super-block codec", other.name()),
    };
    let blocks = u64::from(EXPERTS) * u64::from(out_dim) * (u64::from(in_dim) / elements);
    ExpertSlab::Block {
        weights: shiso_core::alloc(backend, blocks * bytes).expect("allocation"),
        codec,
    }
}

/// The buffer ids a slab contributes, in the order it holds them.
fn slab_ids(slab: &ExpertSlab) -> Vec<NonZeroU64> {
    match slab {
        ExpertSlab::Block { weights, .. } => vec![weights.view().id()],
        ExpertSlab::Nvfp4 {
            weights,
            scales,
            globals,
        } => vec![weights.view().id(), scales.view().id(), globals.view().id()],
        ExpertSlab::Q4Affine {
            weights,
            scales,
            biases,
        } => vec![weights.view().id(), scales.view().id(), biases.view().id()],
    }
}

/// A routed block and the scratch it runs over.
struct Fixture {
    backend: Arc<dyn Backend>,
    recorder: Arc<Recorder>,
    moe: Moe,
    shared: Ffn,
    scratch: MoeScratch,
    input: DeviceBuffer,
    out: DeviceBuffer,
}

fn fixture(n_tok: u32) -> Fixture {
    fixture_in(n_tok, ExpertCodec::Nvfp4, ActDtype::F32)
}

/// The same block with its three stacks in `codec` and its activations in
/// `dtype`, built through the real constructor.
fn fixture_in(n_tok: u32, codec: ExpertCodec, dtype: ActDtype) -> Fixture {
    fixture_stages(n_tok, codec, codec, dtype)
}

/// The same, with the down stack in a codec of its own.
///
/// Two codecs rather than one because the container this family is served from
/// has two: gate and up in `Q4_K`, down in `Q5_K`.
fn fixture_stages(n_tok: u32, gate_up: ExpertCodec, down: ExpertCodec, dtype: ActDtype) -> Fixture {
    let recorder = Recorder::new();
    let backend: Arc<dyn Backend> = recorder.clone();
    let rows = u64::from(HIDDEN) * u64::from(n_tok) * 4;
    let build = |codec: ExpertCodec, in_dim, out_dim| match codec {
        ExpertCodec::Nvfp4 => slab(&backend, in_dim, out_dim),
        ExpertCodec::Q4Affine => affine_slab(&backend, in_dim, out_dim),
        block => block_slab(&backend, in_dim, out_dim, block),
    };
    let stack = |in_dim, out_dim| build(gate_up, in_dim, out_dim);
    let down_stack = |in_dim, out_dim| build(down, in_dim, out_dim);
    Fixture {
        moe: Moe::new(
            dense(&backend, HIDDEN, EXPERTS),
            stack(HIDDEN, INTER),
            stack(HIDDEN, INTER),
            down_stack(INTER, HIDDEN),
            dense(&backend, HIDDEN, 1),
            dtype,
        )
        .expect("a block whose gate and up agree"),
        shared: Ffn {
            gate: nvfp4(&backend, HIDDEN, INTER),
            up: nvfp4(&backend, HIDDEN, INTER),
            down: nvfp4(&backend, INTER, HIDDEN),
        },
        scratch: MoeScratch::new(&backend, shape(), HIDDEN, n_tok).expect("scratch"),
        input: shiso_core::alloc(&backend, rows).expect("allocation"),
        out: shiso_core::alloc(&backend, rows).expect("allocation"),
        recorder,
        backend,
    }
}

/// Runs the block and returns what the recorder saw.
fn launches(n_tok: u32) -> (Fixture, Vec<Launch>) {
    launches_in(n_tok, ExpertCodec::Nvfp4, ActDtype::F32)
}

/// The same, for a block in `codec` and `dtype`.
fn launches_in(n_tok: u32, codec: ExpertCodec, dtype: ActDtype) -> (Fixture, Vec<Launch>) {
    run_fixture(fixture_in(n_tok, codec, dtype), n_tok)
}

/// The same, for a block whose down stack is a codec of its own.
fn launches_in_stages(
    n_tok: u32,
    gate_up: ExpertCodec,
    down: ExpertCodec,
) -> (Fixture, Vec<Launch>) {
    run_fixture(fixture_stages(n_tok, gate_up, down, ActDtype::F32), n_tok)
}

/// Runs a built fixture and returns it with what the recorder saw.
fn run_fixture(f: Fixture, n_tok: u32) -> (Fixture, Vec<Launch>) {
    super::run(
        f.backend.as_ref(),
        &f.moe,
        &f.shared,
        shape(),
        &f.scratch,
        MoeIo {
            input: f.input.view(),
            block_out: f.out.view(),
        },
        n_tok,
    )
    .expect("a routed block");
    let seen = f.recorder.launches();
    (f, seen)
}

#[test]
fn the_block_is_the_fifteen_launches_it_should_be_in_the_order_it_should_be() {
    let (_f, seen) = launches(1);
    let names: Vec<(&str, &str)> = seen
        .iter()
        .map(|launch| (launch.module, launch.entry))
        .collect();
    assert_eq!(
        names,
        vec![
            // The router's projection, then the selection. The router is
            // UNQUANTIZED in this checkpoint - `hf_quant_config.json` names no
            // target for `mlp.gate` - so it takes the generic dense matvec and
            // not either of this family's codecs. A reader that assumed
            // "inside `mlp`, therefore NVFP4" would decode the routing logits
            // from nibbles and send every token to the wrong experts.
            ("generic", "kernel_mul_mv_f32_f32_4"),
            (Q4E_MOE, "q4e_moe_route_kernel"),
            // The counting sort: histogram, scan, scatter.
            (Q4E_MOE, "q4e_moe_sort_count_kernel"),
            (Q4E_MOE, "q4e_moe_sort_scan_kernel"),
            (Q4E_MOE, "q4e_moe_sort_scatter_kernel"),
            // The experts: gate and up in one launch, the activation, down.
            ("qwen35moe", "kernel_q35moe_experts_gate_up_nvfp4_f32"),
            (Q4E_MOE, "q4e_swiglu_kernel"),
            ("qwen35moe", "kernel_q35moe_experts_down_nvfp4_f32"),
            // One contribution per token.
            (Q4E_MOE, "q4e_moe_combine_kernel"),
            // The shared expert: three projections, its activation, its
            // scalar gate's projection, and the gated add.
            ("qwen35moe", "kernel_q35moe_mv_nvfp4_f32"),
            ("qwen35moe", "kernel_q35moe_mv_nvfp4_f32"),
            ("generic", "kernel_swiglu_flat_f32"),
            ("qwen35moe", "kernel_q35moe_mv_nvfp4_f32"),
            // The shared expert's scalar gate, which is unquantized too.
            ("generic", "kernel_mul_mv_f32_f32_4"),
            (Q4E_MOE, "q4e_shared_add_kernel"),
        ],
        "the launch sequence changed"
    );
}

#[test]
fn the_router_s_shared_memory_is_one_float_per_expert() {
    // `q4e_moe_route_kernel` stages the whole logit row in dynamic shared
    // memory. Too little is a read past the allocation inside a kernel that
    // completes; too much is only waste, and the two are indistinguishable
    // from the launch's exit code.
    let (_f, seen) = launches(1);
    let route = seen
        .iter()
        .find(|launch| launch.entry == "q4e_moe_route_kernel")
        .expect("the router ran");
    assert_eq!(route.smem, EXPERTS * 4);
    assert_eq!(route.grid, Extent3::linear(1), "one block per token");
    // Five buffers and scalars, no by-value struct: this entry takes its two
    // widths as `uint32_t` parameters, unlike the ones this family wrote.
    assert_eq!(route.buffers(), 3);
    assert!(route.args.is_empty());
}

#[test]
fn the_expert_stages_take_their_slabs_in_threes_and_the_sorted_map_after() {
    let (_f, seen) = launches(1);
    let gate_up = seen
        .iter()
        .find(|l| l.entry == "kernel_q35moe_experts_gate_up_nvfp4_f32")
        .expect("the gate and up ran");
    // Twelve buffers: three per slab for gate and up, the activations, the
    // three-part sorted map, and two outputs. A launch one buffer short slides
    // every later operand up by one and reads the sorted map as a weight.
    assert_eq!(gate_up.buffers(), 12);
    assert_eq!(gate_up.args.len(), 16, "one by-value struct of four widths");

    let down = seen
        .iter()
        .find(|l| l.entry == "kernel_q35moe_experts_down_nvfp4_f32")
        .expect("the down ran");
    // Eight: one slab, the activated intermediate, the three-part map, one
    // output.
    assert_eq!(down.buffers(), 8);
    assert_eq!(down.args.len(), 16);
}

#[test]
fn the_expert_extent_is_the_columns_a_chunk_has_and_never_more_experts() {
    // The entries index the ACTIVE experts, and a chunk cannot activate more
    // experts than it has columns. At one token that is two blocks rather than
    // six; past three tokens it is every expert.
    for (n_tok, want_y) in [(1u32, 2u32), (2, 4), (3, 6), (8, 6), (64, 6)] {
        let (_f, seen) = launches(n_tok);
        let gate_up = seen
            .iter()
            .find(|l| l.entry == "kernel_q35moe_experts_gate_up_nvfp4_f32")
            .expect("the gate and up ran");
        assert_eq!(
            gate_up.grid,
            Extent3::new(INTER.div_ceil(4), want_y, 1),
            "at {n_tok} tokens"
        );
        let down = seen
            .iter()
            .find(|l| l.entry == "kernel_q35moe_experts_down_nvfp4_f32")
            .expect("the down ran");
        assert_eq!(
            down.grid,
            Extent3::new(HIDDEN.div_ceil(4), want_y, 1),
            "at {n_tok} tokens"
        );
    }
}

#[test]
fn the_sort_runs_over_the_columns_and_the_scan_over_the_experts() {
    // Three passes with two different notions of "how many": the histogram and
    // the scatter walk the routed COLUMNS, the scan walks the EXPERTS and is
    // one block because it is serial by construction. A histogram launched
    // over the experts would count four of ten columns at five tokens.
    let (_f, seen) = launches(5);
    let columns = 5 * USED;
    for entry in ["q4e_moe_sort_count_kernel", "q4e_moe_sort_scatter_kernel"] {
        let pass = seen.iter().find(|l| l.entry == entry).expect(entry);
        assert_eq!(pass.grid, Extent3::linear(columns.div_ceil(256).max(1)));
    }
    let scan = seen
        .iter()
        .find(|l| l.entry == "q4e_moe_sort_scan_kernel")
        .expect("the scan ran");
    assert_eq!(scan.grid, Extent3::linear(1));
    assert_eq!(scan.block, Extent3::linear(32), "one warp");
}

#[test]
fn the_activation_covers_every_routed_column_and_not_every_token() {
    // `columns * intermediate`, not `tokens * intermediate`. The second is
    // right at one expert per token and silently leaves the other slots'
    // intermediate holding whatever the previous layer left, which decodes to
    // a fluent answer from half the experts.
    let (_f, seen) = launches(5);
    let swiglu = seen
        .iter()
        .find(|l| l.module == Q4E_MOE && l.entry == "q4e_swiglu_kernel")
        .expect("the activation ran");
    let want = u64::from(5 * USED) * u64::from(INTER);
    assert_eq!(want, 160);
    assert_eq!(
        swiglu.grid,
        Extent3::linear(
            u32::try_from(want.div_ceil(256))
                .expect("a small count")
                .max(1)
        )
    );
}

#[test]
fn a_shape_whose_router_is_not_the_weight_s_width_is_refused() {
    let f = fixture(1);
    let wrong = MoeShape::new(EXPERTS + 1, USED, 1, INTER);
    let err = super::run(
        f.backend.as_ref(),
        &f.moe,
        &f.shared,
        wrong,
        &f.scratch,
        MoeIo {
            input: f.input.view(),
            block_out: f.out.view(),
        },
        1,
    )
    .expect_err("seven experts against a six-wide router");
    match err {
        CoreError::ShapeMismatch {
            field,
            expected,
            got,
        } => {
            assert_eq!(field, "router output width");
            assert_eq!((expected, got), (7, 6));
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn the_three_built_expert_entries_are_named_and_the_fourth_is_not() {
    use super::{ExpertStageKind, expert_entry};
    assert_eq!(
        expert_entry(ExpertStageKind::GateUp, ExpertCodec::Nvfp4, ActDtype::F32),
        Some("kernel_q35moe_experts_gate_up_nvfp4_f32")
    );
    assert_eq!(
        expert_entry(ExpertStageKind::Down, ExpertCodec::Q4Affine, ActDtype::Bf16),
        Some("kernel_q35moe_experts_down_q4affine_bf16")
    );
    assert_eq!(
        expert_entry(
            ExpertStageKind::GateUp,
            ExpertCodec::Q4Affine,
            ActDtype::F32
        ),
        Some("kernel_q35moe_experts_gate_up_q4affine_f32")
    );
    // Nothing asks for an NVFP4 stack in bf16, so nothing is built for it and
    // the refusal names the pair rather than launching something else.
    assert_eq!(
        expert_entry(ExpertStageKind::Down, ExpertCodec::Nvfp4, ActDtype::Bf16),
        None
    );
}

#[test]
fn an_affine_slab_carries_biases_where_an_nvfp4_slab_carries_globals() {
    // The two codecs' third buffer is not the same quantity: NVFP4 carries one
    // f32 per expert, affine carries one bias per group. Asserting the COUNT
    // proves nothing - it was three before the enum existed. What has to hold
    // is that the affine arm hands its three over in the order the entry reads
    // them: written `[weights, biases, scales]` the compiler is equally happy,
    // exhaustiveness is still satisfied, and the kernel silently multiplies by
    // a bias and adds a scale.
    //
    // In `f32` rather than `bf16`: the two affine arms bind identically, and
    // `bf16` is refused at the constructor until the routed scratch carries a
    // dtype - see `a_bf16_block_is_refused_while_the_routed_scratch_is_f32_only`.
    let (f, seen) = launches_in(1, ExpertCodec::Q4Affine, ActDtype::F32);
    assert_eq!(f.moe.gate.buffer_count(), 3);

    let gate_up = seen
        .iter()
        .find(|l| l.entry.contains("experts_gate_up"))
        .expect("the gate and up ran");
    assert_eq!(
        gate_up.entry, "kernel_q35moe_experts_gate_up_q4affine_f32",
        "the affine gate-and-up entry"
    );
    // Twelve as in the NVFP4 case, and the first six are the two slabs in
    // order: weights, scales, biases, then the same for `up`.
    assert_eq!(gate_up.buffers(), 12);
    let mut want = slab_ids(&f.moe.gate);
    want.extend(slab_ids(&f.moe.up));
    assert_eq!(
        gate_up.ids[..6],
        want[..],
        "the affine arm's three buffers, per slab, in kernel order"
    );

    let down = seen
        .iter()
        .find(|l| l.entry.contains("experts_down"))
        .expect("the down ran");
    assert_eq!(
        down.entry, "kernel_q35moe_experts_down_q4affine_f32",
        "the affine down entry"
    );
    assert_eq!(down.buffers(), 8);
    assert_eq!(
        down.ids[..3],
        slab_ids(&f.moe.down)[..],
        "the down slab's three buffers in kernel order"
    );
}

#[test]
fn a_gate_and_up_that_disagree_about_the_codec_are_refused_by_name() {
    // The failure this forecloses: gate and up are ONE launch and the codec is
    // that entry's template argument, so a pair that disagreed would launch one
    // codec's entry over the other codec's bytes and finish.
    //
    // Down is deliberately NOT part of this rule any more - see
    // `a_down_stack_may_be_a_different_codec_from_the_gate_and_up`.
    let recorder = Recorder::new();
    let backend: Arc<dyn Backend> = recorder;
    let err = Moe::new(
        dense(&backend, HIDDEN, EXPERTS),
        slab(&backend, HIDDEN, INTER),
        affine_slab(&backend, HIDDEN, INTER),
        slab(&backend, INTER, HIDDEN),
        dense(&backend, HIDDEN, 1),
        ActDtype::F32,
    )
    .expect_err("an affine up projection beside an nvfp4 gate");
    match err {
        CoreError::InvalidValue { key, value } => {
            assert_eq!(key, "routed expert stack codec");
            // Named, not categorised: both codecs and the reason are in the
            // message an operator reads.
            assert_eq!(
                value,
                "up is q4affine where gate is nvfp4, and the two share one launch"
            );
        }
        other => panic!("expected InvalidValue, got {other:?}"),
    }
}

#[test]
fn a_down_stack_may_be_a_different_codec_from_the_gate_and_up() {
    // `unsloth`'s `Qwen3.6-35B-A3B-UD-Q4_K_XL` stores every expert's gate and
    // up in `Q4_K` and every down in `Q5_K`. The constructor refused that
    // combination before this - "down is q5k where gate is q4k" - and the block
    // could not be loaded at all.
    //
    // What makes it safe is not the absence of the check but where the two
    // halves of a launch now come from: `entry_for` reads the stage's OWN
    // stack's codec and `slab` hands over that same stack's buffers, so the
    // entry and the pointer count cannot disagree. This case asserts both
    // halves, because asserting that the constructor accepted would not.
    let (f, seen) = launches_in_stages(1, ExpertCodec::Q4K, ExpertCodec::Q5K);
    let gate_up = seen
        .iter()
        .find(|l| l.entry.contains("experts_gate_up"))
        .expect("the gate and up ran");
    assert_eq!(gate_up.entry, "kernel_q35moe_experts_gate_up_q4k_f32");
    // One buffer per super-block stack, not three: the scales are inside the
    // blocks. Two stacks, then the activation, the three routing rows and the
    // two outputs.
    assert_eq!(gate_up.buffers(), 2 + 1 + 3 + 2);
    assert_eq!(gate_up.ids[..1], slab_ids(&f.moe.gate)[..]);
    assert_eq!(gate_up.ids[1..2], slab_ids(&f.moe.up)[..]);

    let down = seen
        .iter()
        .find(|l| l.entry.contains("experts_down"))
        .expect("the down ran");
    assert_eq!(down.entry, "kernel_q35moe_experts_down_q5k_f32");
    assert_eq!(down.buffers(), 1 + 1 + 3 + 1);
    assert_eq!(down.ids[..1], slab_ids(&f.moe.down)[..]);
}

#[test]
fn a_stage_nothing_was_built_for_is_refused_at_the_load_and_again_at_the_launch() {
    let recorder = Recorder::new();
    let backend: Arc<dyn Backend> = recorder.clone();

    // At the load: `Moe::new` refuses the pair rather than handing back a
    // block whose first decode would die five launches in.
    let err = Moe::new(
        dense(&backend, HIDDEN, EXPERTS),
        slab(&backend, HIDDEN, INTER),
        slab(&backend, HIDDEN, INTER),
        slab(&backend, INTER, HIDDEN),
        dense(&backend, HIDDEN, 1),
        ActDtype::Bf16,
    )
    .expect_err("nvfp4 in bf16 is not built");
    match &err {
        CoreError::InvalidValue { key, value } => {
            assert_eq!(*key, "routed expert stage");
            assert_eq!(value, "nvfp4/bf16 was never built; nothing asks for it");
        }
        other => panic!("expected InvalidValue, got {other:?}"),
    }

    // And again at the launch. `run`'s own refusal is unreachable through the
    // constructor, so it is driven here through the struct literal the module
    // can still write - otherwise the wiring of `expert_entry`'s `None` into
    // `run` is compile-checked and never executed.
    let f = fixture(1);
    let smuggled = Moe {
        router: f.moe.router,
        gate: f.moe.gate,
        up: f.moe.up,
        down: f.moe.down,
        shared_gate: f.moe.shared_gate,
        gate_up_codec: ExpertCodec::Nvfp4,
        down_codec: ExpertCodec::Nvfp4,
        dtype: ActDtype::Bf16,
    };
    let err = super::run(
        f.backend.as_ref(),
        &smuggled,
        &f.shared,
        shape(),
        &f.scratch,
        MoeIo {
            input: f.input.view(),
            block_out: f.out.view(),
        },
        1,
    )
    .expect_err("a block that got past the constructor");
    match err {
        CoreError::InvalidValue { key, value } => {
            assert_eq!(key, "routed expert stage");
            assert_eq!(value, "nvfp4/bf16 was never built; nothing asks for it");
        }
        other => panic!("expected InvalidValue, got {other:?}"),
    }
    // It refused BEFORE issuing an expert launch: the router and the sort ran,
    // no expert stage did.
    let entries: Vec<&str> = recorder.launches().iter().map(|l| l.entry).collect();
    assert!(
        !entries.iter().any(|e| e.contains("experts_")),
        "no expert stage was issued, saw {entries:?}"
    );
}

#[test]
fn a_bf16_block_is_refused_while_the_routed_scratch_is_f32_only() {
    // The two `bf16` expert entries are compiled and graded on a device -
    // `crates/shiso-kerneltest/src/qwen35moe/q4affine.rs` measures both - but
    // nothing can drive them yet, and the way they would fail is the bad way.
    // `MoeScratch::new` sizes gate/up/mid/down at four bytes an element while
    // `q4e_swiglu_kernel` and `q4e_moe_combine_kernel` take `float *`, so an
    // f32 buffer is TWICE the bytes a bf16 stage needs: every access stays in
    // bounds, the expert entry reads f32 bit patterns as `bfloat`, the
    // activation reads `bfloat` output as f32, and the block finishes without
    // a fault and writes plausible nonsense. No case would catch it.
    //
    // So the constructor refuses it rather than half-wiring it, and the
    // message says WHY: an operator who sees "was never built" would go
    // looking for a kernel that is in fact there and graded.
    let recorder = Recorder::new();
    let backend: Arc<dyn Backend> = recorder;
    let err = Moe::new(
        dense(&backend, HIDDEN, EXPERTS),
        affine_slab(&backend, HIDDEN, INTER),
        affine_slab(&backend, HIDDEN, INTER),
        affine_slab(&backend, INTER, HIDDEN),
        dense(&backend, HIDDEN, 1),
        ActDtype::Bf16,
    )
    .expect_err("the routed scratch cannot feed a bf16 stage");
    match err {
        CoreError::InvalidValue { key, value } => {
            assert_eq!(key, "routed expert stage");
            assert!(
                value.contains("entries exist and are graded"),
                "the refusal must not read as `never built`: {value}"
            );
            assert!(
                value.contains("f32-only"),
                "the refusal must name the reason: {value}"
            );
        }
        other => panic!("expected InvalidValue, got {other:?}"),
    }
    // The same pair still resolves to a name, which is the point of refusing
    // here: the entries are real, the plumbing is not.
    assert_eq!(
        super::expert_entry(
            super::ExpertStageKind::GateUp,
            ExpertCodec::Q4Affine,
            ActDtype::Bf16
        ),
        Some("kernel_q35moe_experts_gate_up_q4affine_bf16")
    );
}
