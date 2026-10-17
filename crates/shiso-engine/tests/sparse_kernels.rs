//! Every kernel the sparse stack can dispatch, checked against the kernel set's
//! own name and launch-geometry table.
//!
//! The table is a document in this repository, and it is the single source of
//! truth for what a kernel is called and how it is launched. This binary reads
//! it and compares. That is the one check on this machine that can catch the
//! failure mode a port actually has here - a name that does not exist, or a name
//! that exists and is launched on a grid its launcher would not have used -
//! because the kernels cannot run and their numbers cannot be looked at.
//!
//! # Why the plans are built from the block planners rather than from a stack
//!
//! A stack needs a device and imported weights; the block planners need neither,
//! and between them they emit every entry the pass can reach. Building the
//! coverage this way also means a path nothing exercises - a kernel the ladder
//! can select but the fixture never reaches - is still checked.

use std::collections::BTreeSet;
use std::path::PathBuf;

use shiso_core::DType;
use shiso_core::ids::LayerIdx;
use shiso_engine::shape::{GdnShape, LayerKind};
use shiso_engine::sparse::bind::{Args, LayerRef, Operand, Role, WeightRef};
use shiso_engine::sparse::gather;
use shiso_engine::sparse::hyper::{self, MixInput, MixRoles, NextNorm};
use shiso_engine::sparse::indexer::{self, Scorer};
use shiso_engine::sparse::kernels::{self, Elem};
use shiso_engine::sparse::matmul::{self, Bound, Device, PassRole};
use shiso_engine::sparse::moe::{self, RowOrigin};
use shiso_engine::sparse::mtp::SpecDepth;
use shiso_engine::sparse::plan::Plan;
use shiso_engine::sparse::ple::{self, NgramContext};
use shiso_engine::sparse::qsa::{self, AttnContext, AttnPath, AttnSlot};
use shiso_engine::sparse::recurrent::{self, ChunkPolicy, Recurrence};
use shiso_engine::sparse::scratch::Buf;
use shiso_engine::sparse::shape::{
    GqaShape, IndexerShape, MoeShape, MropeSections, PleShape, SparseShape,
};
use shiso_engine::sparse::weights::{
    AttnDesc, ExpertDesc, IndexerDesc, MixDesc, NgramDesc, RecurrentDesc, WeightDesc,
};

/// Positions the plans here are built for. Past the selection budget, which is
/// where the score buffer's stride and that budget stop being the same number.
const CTX: u32 = 16_384;

/// The attention slot every attention plan here is built for: the fourth
/// layer, the first attention slot, over a pool of a thousand pages.
const SLOT: AttnSlot = AttnSlot {
    layer: LayerRef::target(3),
    rank: 0,
    pool_slots: 1_024,
    ctx: CTX,
    kv: shiso_core::page::KvElem::F16,
};

/// The shape this binary plans against: the shipping one.
///
/// Not a cut-down fixture. Every precondition in the ladder is stated on a
/// width - a head that is 256 wide, an expert stack the trellis codec can tile,
/// four indexer heads - so a narrowed shape would quietly take different paths
/// and check different rows of the table.
fn shape() -> SparseShape {
    SparseShape {
        layers: (0..48)
            .map(|il| {
                if (il + 1_u32).is_multiple_of(4) {
                    LayerKind::Attn
                } else {
                    LayerKind::Gdn
                }
            })
            .collect(),
        hidden: 2_560,
        vocab: 248_320,
        streams: 4,
        stream_rank: 320,
        gqa: GqaShape::new(24, 2, 256, 64),
        gdn: GdnShape::new(48, 16, 128, 4),
        moe: MoeShape::new(512, 10, 1, 640),
        indexer: IndexerShape::new(4, 128, 2_048, 4),
        ple: Some(PleShape::new(LayerIdx::new(1), 3, 8, 160, 4)),
        mrope: MropeSections::new(11, 11, 10),
        rms_eps: 1.0e-6,
        rope_theta: 1.0e7,
    }
}

const GPU: Device = Device {
    blas: true,
    capturing: false,
    max_co_resident: 48,
    half_block_input: false,
    role: PassRole::Step,
    prompt_matmul: matmul::PromptMatmul::Rows,
    prefill_activations: matmul::PrefillActivations::A16,
    route_sort: matmul::RouteSort::Tiny,
    normed: matmul::NormedStreams::Stored,
};

fn half(in_dim: u32, out_dim: u32) -> WeightDesc {
    WeightDesc::new(DType::F16, in_dim, out_dim)
}

fn trellis(in_dim: u32, out_dim: u32) -> WeightDesc {
    WeightDesc::new(DType::Exl3K4, in_dim, out_dim)
}

fn mix_desc(shape: &SparseShape, inject: bool) -> MixDesc {
    MixDesc {
        down: half(shape.stream_dim(), shape.stream_rank),
        up: half(shape.stream_rank, shape.stream_dim()),
        inject: inject.then(|| half(shape.stream_dim(), shape.streams)),
    }
}

fn attn_desc(shape: &SparseShape) -> AttnDesc {
    AttnDesc {
        q: half(shape.hidden, 2 * shape.q_dim()),
        k: half(shape.hidden, shape.kv_dim()),
        v: half(shape.hidden, shape.kv_dim()),
        output: half(shape.q_dim(), shape.hidden),
        indexer: IndexerDesc {
            q: half(shape.hidden, shape.indexer_q_dim()),
            k: half(shape.hidden, shape.indexer.head_dim),
        },
    }
}

fn recurrent_desc(shape: &SparseShape) -> RecurrentDesc {
    RecurrentDesc {
        qkv: half(shape.hidden, shape.gdn_in_dim()),
        gate: half(shape.hidden, shape.gdn_v_dim()),
        alpha: half(shape.hidden, shape.gdn.v_heads),
        beta: half(shape.hidden, shape.gdn.v_heads),
        out: half(shape.gdn_v_dim(), shape.hidden),
    }
}

fn expert_desc(shape: &SparseShape) -> ExpertDesc {
    let inter = shape.moe.intermediate;
    ExpertDesc {
        router: half(shape.hidden, shape.moe.experts),
        gate: trellis(shape.hidden, inter),
        up: trellis(shape.hidden, inter),
        down: trellis(inter, shape.hidden),
        shared_router: half(shape.hidden, 1),
        shared_gate: trellis(shape.hidden, inter),
        shared_up: trellis(shape.hidden, inter),
        shared_down: trellis(inter, shape.hidden),
    }
}

fn ngram_desc(shape: &SparseShape, row: DType) -> NgramDesc {
    NgramDesc {
        key: half(shape.hidden, shape.stream_dim()),
        value: half(shape.hidden, shape.hidden),
        row,
    }
}

/// Every plan the sparse stack can produce, at the row counts that reach every
/// branch of every ladder.
#[expect(
    clippy::too_many_lines,
    reason = "every branch of every ladder in one place is the point: split \
              across helpers, a path nothing reaches would stop being visible \
              as a gap in one list"
)]
fn every_path() -> Plan {
    let shape = shape();
    let mut plan = Plan::new();
    // The row gather, at every stored format of a vocabulary table: which entry
    // runs is decided by the format alone, so all three are reachable from one
    // stack and none of them is reached by the fixture's own weights.
    for dtype in [DType::F16, DType::BF16, DType::Q8_0] {
        plan.append(
            gather::rows(
                &shape,
                dtype,
                gather::Bound::new(
                    Operand::Scratch(Buf::Ids),
                    Operand::Weight(WeightRef::model(Role::TokenEmbd)),
                    Operand::Scratch(Buf::Embed),
                ),
                8,
            )
            .expect("every stored format here has a gather"),
        );
    }
    // The packed half-precision tier at every rung of its ladder: the
    // per-row kernel, its passes, and both expansions a prefill chunk takes -
    // the narrow router's to floats for the strict multiply, the stream mix's
    // to halves for the half-precision one.
    for (in_dim, out_dim, rows) in [
        (shape.hidden, shape.moe.experts, 1_u32),
        (shape.hidden, shape.moe.experts, 32),
        (shape.hidden, shape.moe.experts, 512),
        (shape.stream_dim(), shape.stream_rank, 512),
    ] {
        matmul::project(
            &mut plan,
            WeightDesc::new(DType::F16Pack12, in_dim, out_dim),
            Bound::new(
                WeightRef::layer(LayerRef::target(0), Role::Router),
                Operand::Scratch(Buf::Mixed),
                Operand::Scratch(Buf::MoeRouter),
            ),
            rows,
            GPU,
        )
        .expect("a packed half-precision weight has a tier at every row count");
    }
    // The brain-float module of a tensor-core device at every rung: the three
    // step kernels with and without a split, both prompt GEMMs.
    let tensor = Device {
        prompt_matmul: matmul::PromptMatmul::Tensor,
        ..GPU
    };
    let prompt = Device {
        role: PassRole::Prompt { snapshot: None },
        ..tensor
    };
    for (in_dim, out_dim, rows, device) in [
        (shape.stream_dim(), shape.stream_rank, 1_u32, tensor),
        (shape.hidden, shape.moe.experts, 5, tensor),
        (shape.hidden, shape.stream_dim(), 20, tensor),
        (shape.stream_dim(), shape.stream_rank, 512, prompt),
        (shape.hidden, shape.moe.experts, 512, prompt),
    ] {
        matmul::project(
            &mut plan,
            WeightDesc::new(DType::BF16, in_dim, out_dim),
            Bound::new(
                WeightRef::layer(LayerRef::target(0), Role::Router),
                Operand::Scratch(Buf::Mixed),
                Operand::Scratch(Buf::MoeRouter),
            ),
            rows,
            device,
        )
        .expect("a brain-float weight has a tier at every row count");
    }
    plan.append(brain_mixes(&shape, prompt));
    plan.append(hyper::init(&shape, 1));
    plan.append(hyper::init_add(
        &shape,
        Operand::Scratch(Buf::Mixed),
        Operand::Scratch(Buf::Expanded),
        1,
    ));
    for rows in [1_u32, 4, 16, 512] {
        // The half-precision streams are a prefill chunk's, so they are only
        // planned where a chunk is wider than the per-row kernels.
        let elems: &[Elem] = if rows > 16 {
            &[Elem::F32, Elem::F16]
        } else {
            &[Elem::F32]
        };
        for &elem in elems {
            for input in [MixInput::Raw, MixInput::AlreadyNormed] {
                plan.append(
                    hyper::mix(
                        &shape,
                        &mix_desc(&shape, true),
                        MixRoles::token_mixer(LayerRef::target(0)),
                        input,
                        rows,
                        GPU,
                        elem,
                    )
                    .expect("a half-precision mix has a tier"),
                );
            }
            for next in [
                NextNorm::Broken,
                NextNorm::Chained(WeightRef::layer(LayerRef::target(1), Role::AttnMixNorm)),
            ] {
                plan.append(
                    hyper::combine(&shape, next, rows, elem, matmul::NormedStreams::Stored)
                        .expect("the residual fits the fused combine"),
                );
            }
            plan.append(
                hyper::collapse(
                    &shape,
                    &mix_desc(&shape, false),
                    MixRoles::output(),
                    rows,
                    GPU,
                    elem,
                )
                .expect("the collapsing mix has a tier"),
            );
        }
        // Both recurrences: the sequential one at every row count, the chunked
        // one only where the policy allows it.
        // Both recurrences, and both draft depths: the per-position checkpoint
        // slab is a pointer at one depth and a null at the other, and the
        // declaration says a pointer belongs in that slot either way.
        for policy in [
            ChunkPolicy {
                min_rows: 0,
                ..ChunkPolicy::new()
            },
            ChunkPolicy::new(),
        ] {
            for spec in [SpecDepth::NONE, SpecDepth::new(4)] {
                let policy = Recurrence {
                    chunk: policy,
                    spec,
                    origin: RowOrigin::Sequential,
                };
                plan.append(
                    recurrent::block(
                        &shape,
                        &recurrent_desc(&shape),
                        LayerRef::target(0),
                        0,
                        policy,
                        rows,
                        GPU,
                    )
                    .expect("a half-precision recurrent block has a tier"),
                );
            }
        }
        // Every attention path, by driving the three device grants and the
        // selection independently.
        for ctx in [
            AttnContext::default(),
            AttnContext {
                tiled: true,
                ..AttnContext::default()
            },
            AttnContext {
                tensor_core: true,
                tiled: true,
                ..AttnContext::default()
            },
            AttnContext {
                selected: true,
                ..AttnContext::default()
            },
            AttnContext {
                selected: true,
                tensor_core: true,
                tiled: true,
                origin: RowOrigin::Sequential,
                batch_selected: 0,
                ..AttnContext::default()
            },
        ] {
            plan.append(
                qsa::block(&shape, &attn_desc(&shape), SLOT, ctx, rows, GPU)
                    .expect("a half-precision attention block has a tier"),
            );
        }
        // Both scorers.
        for scorer in [Scorer::Scalar, Scorer::TensorCore] {
            plan.append(
                indexer::select(
                    &shape,
                    &attn_desc(&shape).indexer,
                    indexer::IndexerSlot::new(LayerRef::target(3), 0, CTX),
                    scorer,
                    rows,
                    GPU,
                )
                .expect("a half-precision indexer has a tier"),
            );
        }
        plan.append(
            moe::block(&shape, &expert_desc(&shape), LayerRef::target(0), rows, GPU)
                .expect("a trellis expert stack has a routed stage"),
        );
        if rows == 1 {
            // The draft head's single row takes the grouped launches, which is
            // the one routed stage the cooperative fan-out still serves.
            plan.append(
                moe::block(&shape, &expert_desc(&shape), LayerRef::draft(), rows, GPU)
                    .expect("a trellis expert stack has a draft routed stage"),
            );
        }
        // Three rows, a verify pass's width: the one row count that still
        // takes the fused cooperative stage.
        // The fused cooperative stage is no longer on the routed block's own
        // ladder - every pass of a few rows takes the grouped launches, every
        // wider one the weight-stationary ones - but it is still a planner, and
        // its bindings are still checked.
        if rows == 4 {
            plan.append(
                moe::fused_experts(&shape, LayerRef::target(0), 4, 3, GPU)
                    .expect("a trellis expert stack has a fused stage"),
            );
        }
        // The staged down projection, which the fused stage does not reach.
        let mut staged = expert_desc(&shape);
        staged.down = WeightDesc::new(DType::Q8_0, shape.moe.intermediate, shape.hidden);
        plan.append(
            moe::down(
                &shape,
                &staged,
                LayerRef::target(0),
                RowOrigin::Sequential,
                rows,
            )
            .expect("a block-quantized down projection was instantiated"),
        );
        plan.append(
            moe::down(
                &shape,
                &staged,
                LayerRef::target(0),
                RowOrigin::Unrelated,
                1,
            )
            .expect("a block-quantized down projection was instantiated"),
        );
        plan.append(moe::activate(
            &shape,
            Operand::Scratch(Buf::SharedGate),
            Operand::Scratch(Buf::SharedUp),
            Operand::Scratch(Buf::SharedMid),
            rows,
        ));
        let ple = shape.ple.expect("this shape has an n-gram block");
        // Both row codecs a checkpoint of this family may ship its table in.
        // The trellis one takes a pointer the other does not - the hash head's
        // bias - so the two are different argument lists for the same position
        // in the block, and a build that reached only one of them would leave
        // the other's binding unchecked against its own declaration.
        for row in [DType::F16, DType::Exl3Ngram6] {
            for spec in [SpecDepth::NONE, SpecDepth::new(4)] {
                plan.append(
                    ple::block(
                        &shape,
                        ple,
                        &ngram_desc(&shape, row),
                        LayerRef::target(1),
                        NgramContext {
                            spec,
                            origin: RowOrigin::Sequential,
                        },
                        rows,
                        GPU,
                    )
                    .expect("both n-gram row codecs have a decode kernel at this width"),
                );
            }
        }
    }
    // The batched decode: one row per sequence, at every width the logits
    // block admits. Six kernels in this set carry per-sequence state through a
    // launch that serves a whole batch, and until this loop existed none of the
    // six was dispatched anywhere - so none of them had ever been compared
    // against its own declaration.
    for rows in [2_u32, 8] {
        plan.append(
            recurrent::block(
                &shape,
                &recurrent_desc(&shape),
                LayerRef::target(0),
                0,
                Recurrence {
                    chunk: ChunkPolicy::new(),
                    spec: SpecDepth::NONE,
                    origin: RowOrigin::Unrelated,
                },
                rows,
                GPU,
            )
            .expect("a half-precision recurrent block has a tier"),
        );
        // Both grants, because the batched attention entry sits above the whole
        // ladder and a grant must not move it.
        for ctx in [
            AttnContext {
                origin: RowOrigin::Unrelated,
                ..AttnContext::default()
            },
            AttnContext {
                selected: true,
                tensor_core: true,
                tiled: true,
                origin: RowOrigin::Unrelated,
                batch_selected: 0b01,
                ..AttnContext::default()
            },
        ] {
            plan.append(
                qsa::block(&shape, &attn_desc(&shape), SLOT, ctx, rows, GPU)
                    .expect("a half-precision attention block has a tier"),
            );
        }
        let ple = shape.ple.expect("this shape has an n-gram block");
        for row in [DType::F16, DType::Exl3Ngram6] {
            plan.append(
                ple::block(
                    &shape,
                    ple,
                    &ngram_desc(&shape, row),
                    LayerRef::target(1),
                    NgramContext {
                        spec: SpecDepth::NONE,
                        origin: RowOrigin::Unrelated,
                    },
                    rows,
                    GPU,
                )
                .expect("both n-gram row codecs have a decode kernel at this width"),
            );
        }
    }
    // The single-precision tier, which no weight in the fixture's own formats
    // reaches: a very wide contraction into a handful of outputs, which is the
    // shape its contraction split exists for, and the fold that follows it.
    matmul::project(
        &mut plan,
        WeightDesc::new(DType::F32, shape.stream_dim(), shape.streams),
        Bound::new(
            WeightRef::layer(LayerRef::target(0), Role::AttnMixInject),
            Operand::Scratch(Buf::Normed),
            Operand::Scratch(Buf::Inject),
        ),
        1,
        GPU,
    )
    .expect("a single-precision projection has a tier");
    // The trellis tiers: the cooperative kernel at one row and the expand-then-
    // multiply path at a chunk wide enough to be worth expanding once.
    for rows in [1_u32, 512] {
        matmul::project(
            &mut plan,
            trellis(shape.hidden, shape.hidden),
            Bound::new(
                WeightRef::layer(LayerRef::target(0), Role::AttnOutput),
                Operand::Scratch(Buf::Mixed),
                Operand::Scratch(Buf::BlockOut),
            ),
            rows,
            GPU,
        )
        .expect("a trellis projection has a tier");
    }
    // The MXFP8 tier: the one-row and multi-row step kernels, a batched
    // verify's passes of the latter, and the tensor-core prompt kernel.
    let prompt = Device {
        role: PassRole::Prompt { snapshot: None },
        ..GPU
    };
    for (out_dim, rows, device) in [
        (shape.hidden, 1_u32, GPU),
        (shape.moe.experts, 1, GPU),
        (shape.hidden, 5, GPU),
        (shape.moe.experts, 5, GPU),
        (shape.hidden, 20, GPU),
        (shape.hidden, 333, prompt),
    ] {
        matmul::project(
            &mut plan,
            WeightDesc::new(DType::Mxfp8E4M3G32, shape.hidden, out_dim),
            Bound::new(
                WeightRef::layer(LayerRef::target(0), Role::AttnOutput),
                Operand::Scratch(Buf::Mixed),
                Operand::Scratch(Buf::BlockOut),
            ),
            rows,
            device,
        )
        .expect("an MXFP8 projection has a tier at every row count");
    }
    // The low-precision prompt path: the W8A8 projection, and the W4A4 routed
    // stage over an NVFP4 bank with its brain-float combine.
    let lowp = Device {
        prefill_activations: matmul::PrefillActivations::Lowp,
        ..prompt
    };
    matmul::project(
        &mut plan,
        WeightDesc::new(DType::Mxfp8E4M3G32, shape.hidden, shape.hidden),
        Bound::new(
            WeightRef::layer(LayerRef::target(0), Role::AttnOutput),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::BlockOut),
        ),
        333,
        lowp,
    )
    .expect("a W8A8 prompt projection");
    let nvfp4 = |in_dim, out_dim| WeightDesc::new(DType::Nvfp4Gs16, in_dim, out_dim);
    let fp4_experts = ExpertDesc {
        gate: nvfp4(shape.hidden, shape.moe.intermediate),
        up: nvfp4(shape.hidden, shape.moe.intermediate),
        down: nvfp4(shape.moe.intermediate, shape.hidden),
        ..expert_desc(&shape)
    };
    plan.append(
        moe::affine_experts(&shape, &fp4_experts, LayerRef::target(0), 333, lowp)
            .expect("a W4A4 routed stage"),
    );
    plan.append(moe::combine_lowp(&shape, 333));
    // The MXFP8 group launches: a recurrent layer's four input projections,
    // attention's query, key, value and indexer key, and the shared expert's
    // pair, at a decode row, a verify, a batched verify's passes and (the
    // separate tensor-core launches) a prompt chunk.
    plan.append(mxfp8_groups(&shape, prompt));
    plan.launch(
        kernels::Entry::plain(kernels::MOD_MISC, "q4e_argmax_rows_kernel"),
        kernels::argmax_rows(9),
        Args::of(&[
            Operand::Scratch(Buf::Argmax),
            Operand::Scratch(Buf::Logits),
            Operand::U32(shape.vocab),
        ]),
    );
    plan
}

/// The block planners over MXFP8 projection weights, where the projections
/// of one input share group launches.
fn mxfp8_groups(shape: &SparseShape, prompt: Device) -> Plan {
    let mx = |in_dim: u32, out_dim: u32| WeightDesc::new(DType::Mxfp8E4M3G32, in_dim, out_dim);
    let recurrent = mxfp8_recurrent(shape);
    let attn = AttnDesc {
        q: mx(shape.hidden, 2 * shape.q_dim()),
        k: mx(shape.hidden, shape.kv_dim()),
        v: mx(shape.hidden, shape.kv_dim()),
        output: mx(shape.q_dim(), shape.hidden),
        indexer: IndexerDesc {
            q: mx(shape.hidden, shape.indexer_q_dim()),
            k: mx(shape.hidden, shape.indexer.head_dim),
        },
    };
    let inter = shape.moe.intermediate;
    let experts = ExpertDesc {
        shared_gate: mx(shape.hidden, inter),
        shared_up: mx(shape.hidden, inter),
        shared_down: mx(inter, shape.hidden),
        ..expert_desc(shape)
    };
    let mut plan = Plan::new();
    for (rows, device) in [(1_u32, GPU), (5, GPU), (20, GPU), (333, prompt)] {
        plan.append(
            recurrent::block(
                shape,
                &recurrent,
                LayerRef::target(0),
                0,
                Recurrence {
                    chunk: ChunkPolicy::new(),
                    spec: SpecDepth::new(4),
                    origin: RowOrigin::Sequential,
                },
                rows,
                device,
            )
            .expect("an MXFP8 recurrent block has a tier"),
        );
        plan.append(
            qsa::block(shape, &attn, SLOT, AttnContext::default(), rows, device)
                .expect("an MXFP8 attention block has a tier"),
        );
        plan.append(
            moe::shared_expert(shape, &experts, LayerRef::target(0), rows, device)
                .expect("an MXFP8 shared expert has a tier"),
        );
        // The NVFP4 routed stages beside it: the activated gate and up on a
        // step (the one activation launch left is the shared expert's), the
        // tensor-core tiles and the activation launch on a prompt.
        let nv = |in_dim: u32, out_dim: u32| WeightDesc::new(DType::Nvfp4Gs16, in_dim, out_dim);
        let routed = ExpertDesc {
            gate: nv(shape.hidden, inter),
            up: nv(shape.hidden, inter),
            down: nv(inter, shape.hidden),
            ..experts
        };
        let block = moe::block(shape, &routed, LayerRef::target(0), rows, device)
            .expect("an NVFP4 routed block has a tier");
        if device.row_invariant() {
            assert!(block.launches("q4e_nvfp4_experts_gemm_gate_up_kernel"));
        } else {
            assert!(
                block.launches(if rows == 1 {
                    "q4e_nvfp4_experts_gate_up_act_kernel"
                } else {
                    "q4e_nvfp4_experts_gate_up_act_held_kernel"
                }) && block.count("q4e_swiglu_kernel") == 1,
                "a step's routed activation rides on gate and up: {:?}",
                block.names()
            );
            // The shared expert forks off beside the router and the routed
            // stages, and joins before its gate logit.
            let names = block.names();
            let at = |what: &str| names.iter().position(|n| *n == what);
            assert!(
                at("fork") < at("q4e_swiglu_kernel")
                    && at("q4e_swiglu_kernel") < at("resume")
                    && at("resume") < at("fetch routed experts")
                    && at("q4e_moe_combine_kernel") < at("join")
                    && at("join") < at("q4e_shared_add_kernel")
                    && at("fork").is_some(),
                "{names:?}"
            );
        }
        plan.append(block);
    }
    plan
}

/// The stream mixes over brain-float weights on a tensor-core device: a
/// prompt pass takes the fused pair and collapse launches, a step the
/// separate ones.
fn brain_mixes(shape: &SparseShape, prompt: Device) -> Plan {
    let bf = |in_dim: u32, out_dim: u32| WeightDesc::new(DType::BF16, in_dim, out_dim);
    let step = Device {
        role: PassRole::Step,
        ..prompt
    };
    let scaled = Device {
        normed: matmul::NormedStreams::Scaled,
        ..prompt
    };
    let mut plan = Plan::new();
    for inject in [true, false] {
        let desc = MixDesc {
            down: bf(shape.stream_dim(), shape.stream_rank),
            up: bf(shape.stream_rank, shape.stream_dim()),
            inject: inject.then(|| bf(shape.stream_dim(), shape.streams)),
        };
        let roles = if inject {
            MixRoles::token_mixer(LayerRef::target(0))
        } else {
            MixRoles::output()
        };
        // A prompt pass: the pair and the fused collapse, over the stored
        // normed streams or, chained on a scaled pass, norming as they read.
        for (input, device, norming) in [
            (MixInput::Raw, prompt, false),
            (MixInput::AlreadyNormed, prompt, false),
            (MixInput::Raw, scaled, false),
            (MixInput::AlreadyNormed, scaled, true),
        ] {
            let mix = hyper::mix(shape, &desc, roles, input, 512, device, Elem::F32)
                .expect("a brain-float mix has a tier");
            let names = mix.names();
            let collapse = if norming {
                "q4e_bf16_gemm_collapse_norm_kernel"
            } else {
                "q4e_bf16_gemm_collapse_kernel"
            };
            let pair = if norming {
                "q4e_bf16_gemm_pair_norm_kernel"
            } else {
                "q4e_bf16_gemm_pair_kernel"
            };
            assert!(mix.launches(collapse), "{names:?}");
            assert_eq!(mix.launches(pair), inject, "{names:?}");
            assert!(!mix.launches("q4e_hc_collapse_kernel"), "{names:?}");
            plan.append(mix);
        }
        // A step or verify pass, a batched one's passes included: the step
        // pair and the fused step, at every width.
        for rows in [1_u32, 4, 12] {
            let mix = hyper::mix(
                shape,
                &desc,
                roles,
                MixInput::AlreadyNormed,
                rows,
                step,
                Elem::F32,
            )
            .expect("a brain-float mix has a tier");
            let names = mix.names();
            assert_eq!(
                mix.count("q4e_bf16_mix_step_kernel"),
                rows.div_ceil(8) as usize,
                "{names:?}"
            );
            let pair = if rows == 1 {
                "q4e_bf16_gemv_pair_kernel"
            } else {
                "q4e_bf16_rows_pair_kernel"
            };
            assert_eq!(mix.launches(pair), inject, "{names:?}");
            assert!(!mix.launches("q4e_hc_collapse_kernel"), "{names:?}");
            assert!(!mix.launches("q4e_scale_silu_kernel"), "{names:?}");
            plan.append(mix);
        }
    }
    // The chained combine of a scaled pass leaves the scale; a broken one is
    // the residual update alone either way.
    let next = NextNorm::Chained(WeightRef::layer(LayerRef::target(1), Role::AttnMixNorm));
    let combine = hyper::combine(shape, next, 512, Elem::F32, matmul::NormedStreams::Scaled)
        .expect("the residual fits");
    assert!(combine.launches("q4e_hc_combine_scale_kernel"));
    plan.append(combine);
    plan
}

/// A recurrent layer whose five projections are MXFP8.
fn mxfp8_recurrent(shape: &SparseShape) -> RecurrentDesc {
    let mx = |in_dim: u32, out_dim: u32| WeightDesc::new(DType::Mxfp8E4M3G32, in_dim, out_dim);
    RecurrentDesc {
        qkv: mx(shape.hidden, shape.gdn_in_dim()),
        gate: mx(shape.hidden, shape.gdn_v_dim()),
        alpha: mx(shape.hidden, shape.gdn.v_heads),
        beta: mx(shape.hidden, shape.gdn.v_heads),
        out: mx(shape.gdn_v_dim(), shape.hidden),
    }
}

#[test]
fn a_recurrent_decode_row_projects_its_mxfp8_input_in_one_launch() {
    let shape = shape();
    let decode = recurrent::block(
        &shape,
        &mxfp8_recurrent(&shape),
        LayerRef::target(0),
        0,
        Recurrence::default(),
        1,
        GPU,
    )
    .expect("an MXFP8 recurrent decode row has a tier");
    assert_eq!(
        (
            decode.count("q4e_mxfp8_gemv_group_kernel"),
            decode.count("q4e_mxfp8_gemv_kernel"),
            decode.count("q4e_mxfp8_gemv_narrow_kernel"),
        ),
        (1, 1, 0),
        "a recurrent decode row projects its input in one group launch and its \
         output in one more: {:?}",
        decode.names()
    );
}

/// The kernel set's own name table.
fn kernel_names() -> String {
    read_kernel_file("q4e/KERNEL_NAMES.md") + &read_kernel_file("q4e_lowp/KERNEL_NAMES.md")
}

/// One file of the vendored kernel tree.
fn read_kernel_file(relative: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/cuda")
        .join(relative);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("{} is readable: {err}", path.display()))
}

/// The trellis family's own sources.
///
/// That family has no name table of its own and does not need one: every entry
/// it exposes is a template, so the thing a dispatch has to agree with is the
/// template's definition in the source rather than a list somebody maintains
/// beside it.
fn exl3_sources() -> String {
    [
        "exl3/exl3_kernel_map.cuh",
        "exl3/exl3_gemm_kernel.cuh",
        "exl3/exl3_moe_common.cuh",
        "exl3/exl3_moe_kernel.cuh",
        "exl3/exl3_moe_ws.cuh",
        "exl3/exl3_dgemv.cuh",
        "exl3/exl3_reconstruct.cuh",
        "exl3/KERNEL.toml",
    ]
    .iter()
    .map(|name| read_kernel_file(name))
    .collect::<Vec<_>>()
    .join("\n")
}

#[test]
fn every_entry_the_pass_can_dispatch_is_in_the_kernel_set_s_own_table() {
    let table = kernel_names();
    let exl3 = exl3_sources();
    let plan = every_path();
    let mut checked = BTreeSet::new();
    for step in plan.steps() {
        let Some(entry) = step.entry() else {
            continue;
        };
        let name = entry.name();
        assert!(
            table.contains(&format!("`{name}`"))
                || exl3.contains(&format!("void {name}"))
                || exl3.contains(&format!("\n{name}(")),
            "{name} is dispatched by the pass but is declared by no vendored unit"
        );
        checked.insert(name);
    }
    // A coverage floor rather than an exact count: the point is that the walk
    // above reached a broad set rather than one arm of one ladder.
    assert!(
        checked.len() >= 30,
        "only {} distinct entries were reached: {checked:?}",
        checked.len()
    );
}

#[test]
fn every_dispatched_module_is_one_the_kernel_set_registers() {
    let table = kernel_names();
    // The trellis family registers its one module in its own manifest, which is
    // where a reader who wants to know what `exl3` is has to look anyway. That
    // manifest has an empty `[modules]`, so the registration is the source stem
    // itself — which is what this matches, rather than any bare occurrence of
    // the name elsewhere in the file.
    let registered = read_kernel_file("exl3/KERNEL.toml");
    for step in every_path().steps() {
        let Some(entry) = step.entry() else {
            continue;
        };
        let module = entry.module();
        assert!(
            table.contains(&format!("`{module}`"))
                || registered.contains(&format!("\"{module}.cu\"")),
            "{module} is dispatched by the pass but is not a module any kernel set registers"
        );
    }
}

#[test]
fn no_launch_the_pass_emits_has_an_empty_grid_or_block() {
    every_path()
        .validate()
        .expect("a launch with a zero extent is a launch that cannot run");
}

#[test]
fn the_attention_ladder_reaches_every_one_of_its_six_paths() {
    let shape = shape();
    let none = AttnContext::default();
    let tiled = AttnContext {
        tiled: true,
        ..none
    };
    let tc = AttnContext {
        tensor_core: true,
        tiled: true,
        ..none
    };
    // One row, no grants: the split kernel, because the head width is the one
    // its slabs are sized for and one row is below the tiled path's floor.
    assert_eq!(qsa::path(&shape, none, 1), AttnPath::Split);
    // A wide chunk with no grants falls all the way to the always-correct one.
    assert_eq!(qsa::path(&shape, none, 512), AttnPath::Dense);
    // With the tiled grant, a chunk takes the tiled kernel and a row still
    // cannot.
    assert_eq!(qsa::path(&shape, tiled, 512), AttnPath::Tiled);
    assert_eq!(qsa::path(&shape, tiled, 1), AttnPath::Split);
    // With the tensor-core grant a chunk takes that instead.
    assert_eq!(qsa::path(&shape, tc, 512), AttnPath::TensorCore);
    // And once the selection is live, both selected paths.
    let selected = AttnContext {
        selected: true,
        ..none
    };
    assert_eq!(qsa::path(&shape, selected, 1), AttnPath::Gathered);
    assert_eq!(
        qsa::path(
            &shape,
            AttnContext {
                selected: true,
                ..tc
            },
            1
        ),
        AttnPath::GatheredTensorCore
    );
}

#[test]
fn a_single_slice_tensor_core_attention_writes_its_own_destination() {
    let shape = shape();
    let selected_tc = AttnContext {
        selected: true,
        tensor_core: true,
        tiled: true,
        origin: RowOrigin::Sequential,
        batch_selected: 0,
        ..AttnContext::default()
    };
    // Two key heads over sixteen rows is past the slice threshold, so one slice
    // covers it and there is nothing to fold.
    let wide = qsa::attend(&shape, SLOT, selected_tc, 32);
    assert_eq!(wide.count("q4e_qsa_attention_combine_kernel"), 0);
    // One row is below it, so the work is sliced and folded.
    let narrow = qsa::attend(&shape, SLOT, selected_tc, 1);
    assert_eq!(narrow.count("q4e_qsa_attention_combine_kernel"), 1);
}

#[test]
fn the_geometry_of_every_kernel_is_the_expression_its_launcher_computes() {
    // A spot check across the modules, at the shipped shape. Each of these is a
    // row of the table read back as arithmetic rather than as a number: the
    // formula is what is being checked, and a number would be right once.
    let shape = shape();
    let n_tok = 512;

    let init = kernels::hc_init(shape.hidden, shape.streams, n_tok);
    assert_eq!(init.grid.x, (512 * 4 * 2_560_u32).div_ceil(256));
    assert_eq!(init.block.x, 256);

    let norm = kernels::hc_norm(shape.streams, n_tok);
    assert_eq!(norm.grid.x, 512 * 4);

    let route = kernels::moe_route(shape.moe.experts, n_tok);
    assert_eq!(route.grid.x, 512);
    assert_eq!(route.smem, 512 * 4, "one float per expert");

    let conv = kernels::gdn_conv(shape.gdn_in_dim());
    assert_eq!(conv.grid.x, 10_240_u32.div_ceil(256));

    let l2 = kernels::gdn_l2norm(shape.gdn.qk_heads, n_tok);
    assert_eq!((l2.grid.x, l2.grid.y), (16, 512));
    assert_eq!(l2.block.x, 128);

    let recurrent = kernels::gdn_recurrent(shape.gdn.head_dim, shape.gdn.v_heads);
    assert_eq!(recurrent.grid.x, 48);
    assert_eq!(recurrent.block.x, 128 * kernels::GDN_SPLIT);
    assert_eq!(recurrent.smem, 66_048);

    // The fused decode step keeps the recurrence's geometry and tile.
    assert_eq!(
        kernels::gdn_step(shape.gdn.head_dim, shape.gdn.v_heads),
        recurrent
    );

    let chunk = kernels::gdn_chunk(shape.gdn.head_dim, shape.gdn.v_heads);
    assert_eq!(chunk.block.x, 128 * kernels::GDN_CSPLIT);
    assert_eq!(chunk.smem, 99_072);

    let rope = kernels::qsa_q_norm_rope(shape.gqa.q_heads, n_tok, shape.gqa.head_dim);
    assert_eq!((rope.grid.x, rope.grid.y), (24, 512));
    assert_eq!(rope.smem, 256 * 4, "one float per head dimension");

    let dense = kernels::qsa_attention(shape.gqa.q_heads, n_tok, shape.gqa.head_dim);
    assert_eq!(dense.block.x, 32 * kernels::ATTN_WARPS);
    assert_eq!(dense.smem, kernels::ATTN_WARPS * 256 * 4);

    let tiled = kernels::qsa_attention_tiled(shape.gqa.q_heads, n_tok, shape.gqa.head_dim);
    assert_eq!(tiled.grid.y, 512_u32.div_ceil(kernels::ATTN_QT));
    assert_eq!(tiled.smem, 50_176);

    let split = kernels::qsa_attention_split(shape.gqa.kv_heads, 4);
    // One block per row, each holding all twelve query heads of its key head,
    // so a row's key range is its own.
    assert_eq!(split.grid.x, 4);
    assert_eq!(split.grid.y, kernels::ATTN_SPLITS);
    assert_eq!(split.grid.z, 2);

    let score = kernels::idx_score(shape.indexer.heads, n_tok, shape.indexer.budget_blocks());
    assert_eq!(score.grid.x, 512_u32.div_ceil(kernels::IDX_BT));
    assert_eq!(score.grid.y, 512_u32.div_ceil(kernels::IDX_QT));
    assert_eq!(score.smem, 32_768);

    let topk = kernels::idx_topk(n_tok);
    assert_eq!(topk.block.x, 1_024);

    let argmax = kernels::argmax_rows(9);
    assert_eq!((argmax.grid.x, argmax.block.x), (9, 1_024));

    // The six batched entries, whose rows of the table are the scalar ones with
    // a `y` dimension of rows added - and for three of them nothing else at all.
    // A grid that lost the row dimension would run every row's work on row zero
    // and leave the rest of the batch holding whatever was last at its address.
    let rows = 8;
    let conv_batch = kernels::gdn_conv_batch(shape.gdn_in_dim(), rows);
    assert_eq!(conv_batch.grid.x, 10_240_u32.div_ceil(256));
    assert_eq!(conv_batch.grid.y, rows);
    assert_eq!(conv_batch.block.x, 256);

    let rec_batch = kernels::gdn_recurrent_batch(shape.gdn.head_dim, shape.gdn.v_heads, rows);
    assert_eq!((rec_batch.grid.x, rec_batch.grid.y), (48, rows));
    assert_eq!(rec_batch.block.x, 128 * kernels::GDN_SPLIT);
    assert_eq!(rec_batch.smem, 66_048);

    let ple_batch = kernels::ple_conv_batch(shape.stream_dim(), rows);
    assert_eq!(ple_batch.grid.y, rows);

    let store_batch = kernels::qsa_store_kv_batch(shape.gqa.kv_heads, rows, shape.gqa.head_dim);
    assert_eq!(
        store_batch,
        kernels::qsa_store_kv(shape.gqa.kv_heads, rows, shape.gqa.head_dim),
        "the batched store differs from its sibling in an argument, not a grid"
    );
    let attn_batch = kernels::qsa_attention_batch(shape.gqa.q_heads, rows, shape.gqa.head_dim);
    assert_eq!(
        attn_batch,
        kernels::qsa_attention(shape.gqa.q_heads, rows, shape.gqa.head_dim)
    );

    // And the pooled key pass a batch launches per row covers one position, so
    // its grid is the one block the truncating divide plus one gives - the
    // block that also refills the row's raw key ring.
    assert_eq!(kernels::idx_ring_pool(1, kernels::IDX_R).grid.x, 1);

    // Both n-gram row decodes: one block per (token, hash head) either way, and
    // the block width is the whole of the difference - a warp for the
    // block-quantized codec, and the row width itself for the trellis, whose
    // ring wraps at that width so one thread decodes one weight.
    let ple = shape.ple.expect("this shape has an n-gram block");
    let dequant = kernels::ple_dequant(ple.heads(), n_tok);
    assert_eq!((dequant.grid.x, dequant.block.x), (512 * 16, 32));
    let trellis = kernels::ple_dequant_trellis(ple.heads(), n_tok, ple.head_dim);
    assert_eq!((trellis.grid.x, trellis.block.x), (512 * 16, 160));
}

#[test]
fn the_pooled_key_grid_is_not_a_ceiling_and_the_table_says_so() {
    // The one trap the kernel set's table records for a dispatcher, restated
    // here as arithmetic: a truncating divide plus one, which agrees with a
    // ceiling everywhere except at a multiple of the ratio.
    let table = kernel_names();
    assert!(
        table.contains("is **not** a ceiling"),
        "the table still records the trap this test exists for"
    );
    for n_tok in [1_u32, 3, 4, 5, 255, 256, 257, 512] {
        let want = n_tok / kernels::IDX_R + 1;
        assert_eq!(kernels::idx_pool(n_tok, kernels::IDX_R).grid.x, want);
    }
    assert_ne!(
        kernels::idx_pool(256, kernels::IDX_R).grid.x,
        kernels::blocks_for(256, u64::from(kernels::IDX_R))
    );
}

// ---------------------------------------------------------------------------
// argument binding
// ---------------------------------------------------------------------------

/// One kernel's declared parameter list, read out of the vendored source.
///
/// The kernels are in this repository, so what a launch has to hand them is not
/// a matter of opinion: it is the parameter list in the file. This binary reads
/// those lists and compares them against what the pass binds, position by
/// position. That is the one check available on a machine that cannot run a
/// kernel, and it catches the failure argument binding actually has - a
/// missing argument, an extra one, or a scalar where a pointer belongs - none
/// of which fail at the device either, because a pointer-shaped hole in an
/// argument list is filled with whatever the driver had there.
#[derive(Debug, Clone)]
struct Signature {
    /// Whether each parameter is a pointer, in declaration order.
    pointers: Vec<bool>,
}

/// Every kernel signature in the vendored translation units.
fn signatures() -> std::collections::BTreeMap<String, Signature> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../kernels/cuda/q4e");
    let mut out = std::collections::BTreeMap::new();
    for unit in [
        "hc.cu",
        "moe.cu",
        "gdn.cu",
        "gdn_tc.cu",
        "ple.cu",
        "matvec.cu",
        "qsa.cu",
        "qsa_sparse.cu",
        "idx.cu",
        "misc.cu",
        "embed.cu",
        "exact.cu",
        "kv8.cu",
        "mxfp8.cu",
        "idxring.cu",
        "../q4e_lowp/nvfp4_a4.cu",
        "../q4e_lowp/mxfp8_a8.cu",
        "bf16.cu",
        "modelopt.cu",
    ] {
        let path = dir.join(unit);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("{} is readable: {err}", path.display()));
        for (name, params) in parse_kernels(&text) {
            out.insert(name, Signature { pointers: params });
        }
    }
    // The trellis family declares its three cooperative entries through macros
    // rather than in the parameter list, so the list a launch has to match is
    // the macro's, and the expansion kernel spells its own out with the opening
    // parenthesis on its own line. Neither shape is what the scan above reads,
    // and both are read here rather than transcribed, for the same reason: a
    // transcribed list is a second copy of the thing being checked.
    for (name, macro_name) in [
        (kernels::EXL3_GEMM, "EXL3_GEMM_ARGS"),
        (kernels::EXL3_MGEMM, "EXL3_MGEMM_ARGS"),
        (kernels::EXL3_MOE, "EXL3_MOE_KERNEL_ARGS"),
    ] {
        let source = exl3_sources();
        let pointers = parse_macro_params(&source, macro_name)
            .unwrap_or_else(|| panic!("{macro_name} is declared in the trellis family's sources"));
        out.insert(name.to_owned(), Signature { pointers });
    }
    // The weight-stationary routed block is plain kernels in one header of the
    // trellis family's own, named without the `_kernel` suffix the scan above
    // keys on, so each is looked up by the name the pass dispatches.
    let ws = read_kernel_file("exl3/exl3_moe_ws.cuh");
    for name in [
        kernels::EXL3_MOE_WS_PREP,
        kernels::EXL3_MOE_WS_INVERT,
        kernels::EXL3_MOE_WS_PREP_TOK,
        kernels::EXL3_MOE_WS_GATEUP,
        kernels::EXL3_MOE_WS_DOWN,
        kernels::EXL3_MOE_WS_DOWN_H,
        kernels::EXL3_MOE_WS_COMBINE,
        kernels::EXL3_MOE_WSD_PREP,
        kernels::EXL3_MOE_WSD_GATEUP,
        kernels::EXL3_MOE_WSD_ACT,
        kernels::EXL3_MOE_WSD_DOWN,
        kernels::EXL3_MOE_WSD_COMBINE,
    ] {
        let pointers = parse_line_definition(&ws, name)
            .unwrap_or_else(|| panic!("{name} is defined in the weight-stationary header"));
        out.insert(name.to_owned(), Signature { pointers });
    }
    // The decode-shaped dense projection, in a header of its own.
    let dgemv = read_kernel_file("exl3/exl3_dgemv.cuh");
    for name in [kernels::EXL3_DGEMV_HAD, kernels::EXL3_DGEMV] {
        let pointers = parse_line_definition(&dgemv, name)
            .unwrap_or_else(|| panic!("{name} is defined in the decode-shaped header"));
        out.insert(name.to_owned(), Signature { pointers });
    }
    let reconstruct = read_kernel_file("exl3/exl3_reconstruct.cuh");
    let pointers = parse_split_declaration(&reconstruct, kernels::EXL3_RECONSTRUCT)
        .unwrap_or_else(|| panic!("{} is declared", kernels::EXL3_RECONSTRUCT));
    out.insert(kernels::EXL3_RECONSTRUCT.to_owned(), Signature { pointers });
    out
}

/// Pulls the per-parameter "is a pointer" list out of an argument-list macro.
///
/// The macro's body is one parameter per continued line; the continuations and
/// the blank grouping lines are dropped, and what is left is split the same way
/// an ordinary parameter list is.
fn parse_macro_params(text: &str, macro_name: &str) -> Option<Vec<bool>> {
    let start = text.find(&format!("#define {macro_name}"))?;
    let mut body = String::new();
    for line in text[start..].lines().skip(1) {
        let trimmed = line.trim_end();
        let continues = trimmed.ends_with('\\');
        body.push_str(trimmed.trim_end_matches('\\'));
        body.push(' ');
        if !continues {
            break;
        }
    }
    Some(
        body.split(',')
            .filter(|param| !param.trim().is_empty())
            .map(|param| param.contains('*'))
            .collect(),
    )
}

/// The parameter list of a kernel whose name opens its own line and whose
/// definition follows the list.
fn parse_line_definition(text: &str, name: &str) -> Option<Vec<bool>> {
    let mut at = 0usize;
    while let Some(found) = text[at..].find(&format!("\n{name}(")) {
        let open = at + found + name.len() + 2;
        at = open;
        let rest = &text[open..];
        let close = rest.find(')')?;
        if !rest[close + 1..].trim_start().starts_with('{') {
            continue;
        }
        return Some(
            rest[..close]
                .split(',')
                .filter(|param| !param.trim().is_empty())
                .map(|param| param.contains('*'))
                .collect(),
        );
    }
    None
}

/// The same, for a declaration whose opening parenthesis is on its own line.
fn parse_split_declaration(text: &str, name: &str) -> Option<Vec<bool>> {
    let at = text.find(&format!("void {name}"))?;
    let rest = &text[at..];
    let open = rest.find('(')?;
    let close = rest.find(')')?;
    Some(
        rest[open + 1..close]
            .split(',')
            .filter(|param| !param.trim().is_empty())
            .map(|param| param.contains('*'))
            .collect(),
    )
}

/// Pulls `name -> per-parameter "is a pointer"` out of one translation unit.
///
/// A hand-rolled scan rather than a parser: the declarations here are all of
/// one shape - `__global__ ... name(...) {` - and anything this misses shows up
/// as a kernel the binding test cannot find, which is a loud failure rather
/// than a silent skip.
fn parse_kernels(text: &str) -> Vec<(String, Vec<bool>)> {
    let mut out = Vec::new();
    let mut at = 0usize;
    // The parameter list opens at the `(` that follows the entry name, which is
    // not always the first `(` after `__global__`: several of these carry a
    // launch-bounds attribute in between. So the scan looks for the name
    // itself, which every one of them ends the same way.
    while let Some(found) = text[at..].find("_kernel(") {
        let close_name = at + found + "_kernel".len();
        at = close_name + 1;
        let head = &text[..close_name];
        let name_start = head
            .rfind(|c: char| !(c.is_alphanumeric() || c == '_'))
            .map_or(0, |index| index + 1);
        let name = head[name_start..].to_owned();
        // A definition opens a body; a declaration and an explicit
        // instantiation end in a semicolon and carry no argument names.
        let rest = &text[close_name + 1..];
        let Some(close) = rest.find(')') else {
            continue;
        };
        if !rest[close + 1..].trim_start().starts_with('{') {
            continue;
        }
        // Only the entries that are kernels: a device helper whose name happens
        // to end this way would have no `__global__` in front of it.
        if !head[..name_start].trim_end().ends_with("void")
            && !head[..name_start].contains("__global__")
        {
            continue;
        }
        let pointers = rest[..close]
            .split(',')
            .map(|param| param.contains('*'))
            .collect::<Vec<_>>();
        out.push((name, pointers));
    }
    out
}

#[test]
fn every_launch_binds_exactly_the_parameters_its_kernel_declares() {
    let signatures = signatures();
    let plan = every_path();
    let mut seen = BTreeSet::new();
    for step in plan.steps() {
        let Some(entry) = step.entry() else {
            continue;
        };
        let name = entry.name();
        let signature = signatures
            .get(name)
            .unwrap_or_else(|| panic!("{name} is launched but no vendored unit declares it"));
        let operands = step
            .operands()
            .unwrap_or_else(|err| panic!("{name} binds a list a launch can carry: {err:?}"));
        assert_eq!(
            operands.len(),
            signature.pointers.len(),
            "{name} declares {} parameters and the pass binds {}",
            signature.pointers.len(),
            operands.len()
        );
        for (index, (operand, is_pointer)) in
            operands.iter().zip(signature.pointers.iter()).enumerate()
        {
            assert_eq!(
                operand.is_pointer(),
                *is_pointer,
                "{name} parameter {index} is {} in the kernel and {operand:?} in the pass",
                if *is_pointer { "a pointer" } else { "a scalar" }
            );
        }
        seen.insert(name);
    }
    // The floor rises with the set the pass can reach: the row gather at its
    // three storage formats and the trellis family's four entries are all
    // bound now, and each of them is checked against a declaration in the
    // vendored source rather than against a list in this file.
    assert!(
        seen.len() >= 53,
        "only {} distinct entries were bound: {seen:?}",
        seen.len()
    );
    for entry in [
        "embed_tokens_hc_kernel",
        "embed_tokens_bf16_kernel",
        "embed_tokens_q8_0_kernel",
        kernels::EXL3_GEMM,
        kernels::EXL3_MGEMM,
        kernels::EXL3_MOE,
        kernels::EXL3_RECONSTRUCT,
        "q4e_f32_to_f16_kernel",
        "q4e_mxfp8_gemv_kernel",
        "q4e_mxfp8_gemv_narrow_kernel",
        "q4e_mxfp8_rows_kernel",
        "q4e_mxfp8_rows_narrow_kernel",
        "q4e_mxfp8_gemm_kernel",
        "q4e_mxfp8_a8_quant_kernel",
        "q4e_mxfp8_a8_gemm_kernel",
        "q4e_nvfp4_a4_quant_kernel",
        "q4e_nvfp4_a4_quant_bf16_kernel",
        "q4e_nvfp4_a4_gate_up_kernel",
        "q4e_nvfp4_a4_down_kernel",
        "q4e_nvfp4_a4_combine_kernel",
        "q4e_mxfp8_gemv_group_kernel",
        "q4e_mxfp8_gemv_group_narrow_kernel",
        "q4e_mxfp8_rows_group_kernel",
    ] {
        assert!(seen.contains(entry), "{entry} was never bound: {seen:?}");
    }
    // The six that carry per-sequence state through a batched launch. They are
    // named here rather than left to the floor above because they were the last
    // entries in the set nothing dispatched: a floor that happened to be met by
    // other kernels would have gone on saying nothing about them.
    for entry in BATCHED {
        assert!(seen.contains(entry), "{entry} was never bound: {seen:?}");
    }
}

/// Every attention block shape the ladder can take at `slot`: each grant and
/// selection state, one sequence's rows and a batch's, at a decode, a verify, a
/// short chunk and a prefill width.
fn attention_blocks(slot: AttnSlot) -> Plan {
    let shape = shape();
    let mut plan = Plan::new();
    for rows in [1_u32, 4, 16, 512] {
        for ctx in [
            AttnContext::default(),
            AttnContext {
                tiled: true,
                ..AttnContext::default()
            },
            AttnContext {
                tensor_core: true,
                tiled: true,
                ..AttnContext::default()
            },
            AttnContext {
                selected: true,
                ..AttnContext::default()
            },
            AttnContext {
                selected: true,
                tensor_core: true,
                tiled: true,
                ..AttnContext::default()
            },
            AttnContext {
                selected: true,
                tensor_core: true,
                tiled: true,
                dense_rows: rows / 2,
                ..AttnContext::default()
            },
            AttnContext {
                origin: RowOrigin::Unrelated,
                ..AttnContext::default()
            },
            AttnContext {
                selected: true,
                tensor_core: true,
                tiled: true,
                origin: RowOrigin::Unrelated,
                batch_selected: 0b01,
                ..AttnContext::default()
            },
        ] {
            plan.append(
                qsa::block(&shape, &attn_desc(&shape), slot, ctx, rows, GPU)
                    .expect("a half-precision attention block has a tier"),
            );
        }
        // The batched reader, which a block no longer reaches - a batch
        // attends one row at a time - but the ladder still names.
        plan.append(qsa::attend(
            &shape,
            slot,
            AttnContext {
                origin: RowOrigin::Unrelated,
                ..AttnContext::default()
            },
            rows,
        ));
    }
    plan
}

#[test]
fn an_fp8_cache_changes_the_entry_of_every_cache_kernel_and_nothing_else() {
    // The fp8 kernels are the f16 ones with the cache element changed, so the
    // plan over an fp8 slot has to be the f16 plan launch for launch - the same
    // grid, block, shared memory and operands - with only the store's and the
    // readers' entries swapped for their twins. Anything else moving would be
    // a path chosen by the element rather than by the shape.
    let fp8_slot = AttnSlot {
        kv: shiso_core::page::KvElem::Fp8E4m3,
        ..SLOT
    };
    let f16 = attention_blocks(SLOT);
    let fp8 = attention_blocks(fp8_slot);
    assert_eq!(f16.steps().len(), fp8.steps().len(), "launch counts");
    let twin = |name: &str| {
        kernels::CacheKernel::ALL
            .into_iter()
            .find(|kernel| kernel.name(shiso_core::page::KvElem::F16).1 == name)
            .map(|kernel| kernel.name(shiso_core::page::KvElem::Fp8E4m3))
    };
    let mut swapped = BTreeSet::new();
    for (a, b) in f16.steps().iter().zip(fp8.steps()) {
        let (
            shiso_engine::sparse::plan::Step::Kernel {
                entry: ea,
                launch: la,
                ..
            },
            shiso_engine::sparse::plan::Step::Kernel {
                entry: eb,
                launch: lb,
                ..
            },
        ) = (a, b)
        else {
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
            continue;
        };
        assert_eq!(
            format!("{la:?}"),
            format!("{lb:?}"),
            "{}: geometry",
            ea.name()
        );
        assert_eq!(
            a.operands().expect("binds"),
            b.operands().expect("binds"),
            "{}: operands",
            ea.name()
        );
        match twin(ea.name()) {
            Some((module, name)) => {
                assert_eq!((eb.module(), eb.name()), (module, name), "{}", ea.name());
                swapped.insert(name);
            }
            None => assert_eq!(
                (ea.module(), ea.name()),
                (eb.module(), eb.name()),
                "a kernel that does not touch the cache planes kept its entry"
            ),
        }
    }
    let every: BTreeSet<_> = kernels::CacheKernel::ALL
        .into_iter()
        .map(|kernel| kernel.name(shiso_core::page::KvElem::Fp8E4m3).1)
        .collect();
    assert_eq!(
        swapped, every,
        "every fp8 entry is reachable from the ladder"
    );
}

#[test]
fn every_fp8_cache_launch_binds_the_parameters_its_kernel_declares() {
    let signatures = signatures();
    let table = kernel_names();
    let plan = attention_blocks(AttnSlot {
        kv: shiso_core::page::KvElem::Fp8E4m3,
        ..SLOT
    });
    for step in plan.steps() {
        let Some(entry) = step.entry() else {
            continue;
        };
        let name = entry.name();
        if entry.module() != kernels::MOD_KV8 {
            continue;
        }
        assert!(
            table.contains(&format!("`{name}`")),
            "{name} is not in the name table"
        );
        let signature = signatures
            .get(name)
            .unwrap_or_else(|| panic!("{name} is launched but kv8.cu does not declare it"));
        let operands = step.operands().expect("binds");
        assert_eq!(
            operands.len(),
            signature.pointers.len(),
            "{name}: parameter count"
        );
        for (index, (operand, is_pointer)) in
            operands.iter().zip(signature.pointers.iter()).enumerate()
        {
            assert_eq!(
                operand.is_pointer(),
                *is_pointer,
                "{name} parameter {index}"
            );
        }
    }
}

/// The five kernels that carry per-sequence state through one batched launch,
/// and which parameter of each is the per-row array.
///
/// The index is the position in the *kernel's own* declaration, and the test
/// below reads that declaration rather than trusting this list for anything but
/// the position it points at. The batched attention entry is not among them any
/// more: a batch attends one row at a time through the single-row kernels, over
/// each row's own page table, so a row computes what it would alone.
/// The raw indexer key store's batched entry is not here: no raw key is
/// stored per position any more, and the pooled pass that took its place reads
/// each row's ring by row instead.
const BATCHED: [&str; 4] = [
    "q4e_gdn_conv_batch_kernel",
    "q4e_gdn_recurrent_batch_kernel",
    "q4e_qsa_store_kv_batch_kernel",
    "q4e_ple_conv_batch_kernel",
];

#[test]
fn every_batched_entry_binds_a_per_row_array_where_its_declaration_takes_one() {
    // The parameter each of the six takes in place of the one buffer its scalar
    // sibling takes: `float * const *states` or `const int32_t *const *pages`.
    // Both are pointers in the declaration, so the position check in the test
    // above cannot tell an array from a single buffer - which is exactly the
    // mistake that would reintroduce the bug, because binding one session's
    // page table there compiles, launches and reads another sequence's cache.
    let plan = every_path();
    let signatures = signatures();
    let arrays = [
        ("q4e_gdn_conv_batch_kernel", 1, Operand::RecurrentConvs(0)),
        (
            "q4e_gdn_recurrent_batch_kernel",
            1,
            Operand::RecurrentStates(0),
        ),
        ("q4e_qsa_store_kv_batch_kernel", 14, Operand::PageTables),
        ("q4e_ple_conv_batch_kernel", 1, Operand::NgramWindows),
    ];
    for (name, index, want) in arrays {
        let signature = signatures
            .get(name)
            .unwrap_or_else(|| panic!("{name} is declared by a vendored unit"));
        assert_eq!(
            signature.pointers.get(index),
            Some(&true),
            "{name} parameter {index} is not a pointer in its own declaration"
        );
        let step = plan
            .steps()
            .iter()
            .find(|step| step.name() == name)
            .unwrap_or_else(|| panic!("{name} is dispatched by the pass"));
        let operands = step.operands().expect("the list fits a launch");
        assert_eq!(operands[index], want, "{name} parameter {index}");
    }
}

#[test]
fn a_batched_launch_carries_no_row_count_and_no_checkpoint_slab() {
    // Each row of a batch is exactly one token, so the batched recurrence and
    // the two batched convolutions have neither a chain to scan nor a position
    // to record - and their declarations say so by being shorter than their
    // scalar siblings'. A dispatcher that padded the list to the scalar length
    // would be handing the kernel arguments it does not declare, which is a
    // launch that reads whatever the driver had in the slot.
    let signatures = signatures();
    for (batched, scalar) in [
        ("q4e_gdn_conv_batch_kernel", "q4e_gdn_conv_kernel"),
        ("q4e_gdn_recurrent_batch_kernel", "q4e_gdn_recurrent_kernel"),
        ("q4e_ple_conv_batch_kernel", "q4e_ple_conv_kernel"),
    ] {
        let short = signatures.get(batched).expect("declared").pointers.len();
        let long = signatures.get(scalar).expect("declared").pointers.len();
        assert!(
            short < long,
            "{batched} declares {short} parameters and {scalar} declares {long}"
        );
    }
    // The three cache-addressing entries are the same length as their siblings:
    // the page table is replaced, not added to.
    for (batched, scalar) in [
        ("q4e_qsa_store_kv_batch_kernel", "q4e_qsa_store_kv_kernel"),
        ("q4e_qsa_attention_batch_kernel", "q4e_qsa_attention_kernel"),
        ("q4e_idx_store_k_batch_kernel", "q4e_idx_store_k_kernel"),
    ] {
        assert_eq!(
            signatures.get(batched).expect("declared").pointers,
            signatures.get(scalar).expect("declared").pointers,
            "{batched} and {scalar} differ in more than which page table they read"
        );
    }
}

#[test]
fn the_batched_attention_entry_sits_above_the_whole_ladder() {
    // Every other entry in the set stages several queries under one page table,
    // so no grant and no row count may move a batch off the batched one.
    let shape = shape();
    let batched = AttnContext {
        origin: RowOrigin::Unrelated,
        ..AttnContext::default()
    };
    for rows in [1_u32, 2, 8] {
        assert_eq!(qsa::path(&shape, batched, rows), AttnPath::Batched);
        assert_eq!(
            qsa::path(
                &shape,
                AttnContext {
                    selected: true,
                    tensor_core: true,
                    tiled: true,
                    ..batched
                },
                rows
            ),
            AttnPath::Batched,
            "a grant moved a batch off the only batched attention entry"
        );
    }
}

#[test]
fn the_pooled_key_pass_reads_the_rotary_table_from_its_prefix_and_nothing_else_does() {
    // The prefix exists because this one launch derives a pooled row's position
    // by rounding down to the compression boundary, which can name a position
    // a few slots before the chunk starts. Every other launch that reads the
    // table reads the chunk's own first row, and a launch that read the prefix
    // instead would rotate every row by the lookback.
    let shape = shape();
    let plan = indexer::store(
        &shape,
        &attn_desc(&shape).indexer,
        indexer::IndexerSlot::new(LayerRef::target(3), 0, CTX),
        RowOrigin::Sequential,
        8,
        GPU,
    )
    .expect("a half-precision indexer has a tier");
    let pool = plan
        .steps()
        .iter()
        .find(|step| step.name() == "q4e_idx_ring_pool_kernel")
        .expect("the storing half pools");
    let operands = pool.operands().expect("the list fits a launch");
    assert_eq!(
        operands[9],
        Operand::Scratch(Buf::Mrope),
        "the pooled pass reads the table whole"
    );

    let attn = qsa::block(
        &shape,
        &attn_desc(&shape),
        SLOT,
        AttnContext::default(),
        8,
        GPU,
    )
    .expect("a half-precision attention block has a tier");
    let rope = attn
        .steps()
        .iter()
        .find(|step| step.name() == "q4e_qsa_q_norm_rope_kernel")
        .expect("the block rotates its query");
    let operands = rope.operands().expect("the list fits a launch");
    assert!(
        matches!(operands[8], Operand::ScratchFrom(Buf::Mrope, offset) if offset > 0),
        "the query rotation reads the chunk's own first row, not the prefix: {:?}",
        operands[8]
    );
}

#[test]
fn the_storing_half_is_one_pooled_pass_per_session_over_its_own_ring_and_its_own_keys() {
    // No raw key plane is written any more: the only launch after the key
    // projection is the pooled pass, and it reads the session's ring. A chunk
    // binds the pass's one session; a batch binds row `r`'s ring and row `r`'s
    // key row in row `r`'s launch - a ring bound by the wrong row would pool
    // one sequence's block out of another's raw keys.
    let shape = shape();
    let desc = attn_desc(&shape).indexer;
    let slot = indexer::IndexerSlot::new(LayerRef::target(3), 2, CTX);
    let pools = |origin, rows| {
        indexer::store(&shape, &desc, slot, origin, rows, GPU)
            .expect("a half-precision indexer has a tier")
            .steps()
            .iter()
            .filter(|step| step.entry().is_some() && step.name().starts_with("q4e_idx"))
            .map(|step| {
                (
                    step.name().to_owned(),
                    step.operands().expect("the list fits a launch").to_vec(),
                )
            })
            .collect::<Vec<_>>()
    };
    let chunk = pools(RowOrigin::Sequential, 8);
    assert_eq!(chunk.len(), 1, "one launch stores a chunk: {chunk:?}");
    assert_eq!(chunk[0].0, "q4e_idx_ring_pool_kernel");
    assert_eq!(chunk[0].1[1], Operand::IndexerRing(2));
    assert_eq!(chunk[0].1[2], Operand::Scratch(Buf::IdxKey));
    assert_eq!(
        (chunk[0].1[14], chunk[0].1[15]),
        (Operand::IndexerRing(2), Operand::U32(kernels::IDX_NO_TAP)),
        "a pass with no tap hands the kernel its own ring and no row"
    );
    let batch = pools(RowOrigin::Unrelated, 3);
    assert_eq!(batch.len(), 3, "a batch pools once per row: {batch:?}");
    for (row, (name, operands)) in (0_u32..).zip(&batch) {
        assert_eq!(name, "q4e_idx_ring_pool_kernel");
        assert_eq!(operands[1], Operand::IndexerRingOf { rank: 2, row });
        assert_eq!(
            operands[2],
            Operand::ScratchFrom(Buf::IdxKey, u64::from(row) * 128 * 4)
        );
        assert_eq!(operands[5], Operand::PagesOf(row));
    }
    // And a tap reaches the kernel only on the one-session pass that asks.
    let tap = indexer::RingTap {
        row: 5,
        offset: 4_096,
    };
    let tapped = indexer::store(
        &shape,
        &desc,
        slot.with_tap(Some(tap)),
        RowOrigin::Sequential,
        8,
        GPU,
    )
    .expect("a half-precision indexer has a tier");
    let step = tapped
        .steps()
        .iter()
        .find(|step| step.name() == "q4e_idx_ring_pool_kernel")
        .expect("the storing half pools");
    let operands = step.operands().expect("the list fits a launch");
    assert_eq!(
        (operands[14], operands[15]),
        (Operand::Snapshot(4_096), Operand::U32(5))
    );
}

#[test]
fn the_two_recurrent_normalizations_differ_only_in_the_offset_they_are_handed() {
    let shape = shape();
    let plan = recurrent::block(
        &shape,
        &recurrent_desc(&shape),
        LayerRef::target(0),
        0,
        Recurrence::default(),
        8,
        GPU,
    )
    .expect("a half-precision recurrent block has a tier");
    let offsets = plan
        .steps()
        .iter()
        .filter(|step| step.name() == "q4e_gdn_l2norm_kernel")
        .map(|step| {
            let operands = step.operands().expect("the list fits a launch");
            (operands[0], operands[4])
        })
        .collect::<Vec<_>>();
    assert_eq!(offsets.len(), 2);
    // Same buffer, two offsets: the query half of the fused projection and the
    // key half. Two different buffers, or one offset twice, would leave one of
    // the two unnormalized.
    assert_eq!(offsets[0].0, offsets[1].0);
    assert_eq!(offsets[0].1, Operand::U32(0));
    assert_eq!(offsets[1].1, Operand::U32(shape.gdn_qk_dim()));
}

/// The recurrent block's launches at `rows` rows under `policy`, as entry
/// names in order.
fn recurrent_entries(policy: &Recurrence, rows: u32, device: Device) -> Vec<&'static str> {
    let shape = shape();
    recurrent::block(
        &shape,
        &recurrent_desc(&shape),
        LayerRef::target(0),
        0,
        *policy,
        rows,
        device,
    )
    .expect("a half-precision recurrent block has a tier")
    .steps()
    .iter()
    .filter_map(|step| step.entry().map(|entry| entry.name()))
    .collect()
}

#[test]
fn a_decode_row_plans_the_fused_step_and_a_verify_window_plans_the_chain() {
    const CHAIN: [&str; 5] = [
        "q4e_gdn_l2norm_kernel",
        "q4e_gdn_gates_kernel",
        "q4e_gdn_recurrent_kernel",
        "q4e_gdn_out_gate_kernel",
        "q4e_gdn_conv_kernel",
    ];
    let shape = shape();
    assert!(recurrent::step_eligible(&shape), "the shipped geometry");
    for spec in [SpecDepth::NONE, SpecDepth::new(4)] {
        let policy = Recurrence {
            spec,
            ..Recurrence::default()
        };
        // One row of one session's decode: the convolution, then the step, and
        // none of the four launches it replaces.
        let decode = recurrent_entries(&policy, 1, GPU);
        assert_eq!(
            decode
                .iter()
                .filter(|name| **name == "q4e_gdn_step_kernel")
                .count(),
            1
        );
        assert!(decode.contains(&"q4e_gdn_conv_kernel"));
        for name in &CHAIN[..4] {
            assert!(!decode.contains(name), "{name} ran beside the step");
        }
        // Between the projections, two launches where the chain has six: the
        // convolution, then the step - and every input projection ahead of
        // them, so only the output projection follows.
        let gdn: Vec<&str> = decode
            .iter()
            .copied()
            .filter(|name| name.starts_with("q4e_gdn"))
            .collect();
        assert_eq!(gdn, ["q4e_gdn_conv_kernel", "q4e_gdn_step_kernel"]);
        let after = decode
            .iter()
            .position(|name| *name == "q4e_gdn_step_kernel")
            .expect("the step ran");
        let mut out = Plan::new();
        matmul::project(
            &mut out,
            recurrent_desc(&shape).out,
            Bound::new(
                WeightRef::layer(LayerRef::target(0), Role::RecurrentOut),
                Operand::Scratch(Buf::GdnGated),
                Operand::Scratch(Buf::BlockOut),
            ),
            1,
            GPU,
        )
        .expect("the output projection has a tier");
        let out: Vec<&str> = out
            .steps()
            .iter()
            .filter_map(|step| step.entry().map(|entry| entry.name()))
            .collect();
        assert_eq!(
            decode[after + 1..],
            out[..],
            "only the output projection follows the step: {decode:?}"
        );
        // A verify window is more than one row and a rollback may reach it, so
        // it keeps the chain that records; so does a decode batch, whose rows
        // carry a state each, and a one-row prompt pass.
        for rows in [2_u32, 5] {
            let verify = recurrent_entries(&policy, rows, GPU);
            assert!(!verify.contains(&"q4e_gdn_step_kernel"), "{rows} rows");
            for name in CHAIN {
                assert!(verify.contains(&name), "{name} missing at {rows} rows");
            }
        }
    }
    let unrelated = Recurrence {
        origin: RowOrigin::Unrelated,
        ..Recurrence::default()
    };
    assert!(!unrelated.takes_step(1, PassRole::Step));
    assert!(!recurrent_entries(&unrelated, 1, GPU).contains(&"q4e_gdn_step_kernel"));
    let prompt = PassRole::Prompt { snapshot: None };
    assert!(!Recurrence::default().takes_step(1, prompt));
    let no_chunk = Recurrence {
        chunk: ChunkPolicy {
            min_rows: 0,
            ..ChunkPolicy::new()
        },
        ..Recurrence::default()
    };
    assert!(
        !recurrent_entries(
            &no_chunk,
            1,
            Device {
                role: prompt,
                ..GPU
            }
        )
        .contains(&"q4e_gdn_step_kernel")
    );
}

#[test]
fn a_head_width_the_step_kernel_is_not_sized_for_keeps_the_chain() {
    let mut shape = shape();
    shape.gdn.head_dim = 64;
    assert!(!recurrent::step_eligible(&shape));
    assert!(Recurrence::default().takes_step(1, PassRole::Step));
    let plan = recurrent::block(
        &shape,
        &recurrent_desc(&shape),
        LayerRef::target(0),
        0,
        Recurrence::default(),
        1,
        GPU,
    )
    .expect("a half-precision recurrent block has a tier");
    assert_eq!(plan.count("q4e_gdn_step_kernel"), 0);
    assert_eq!(plan.count("q4e_gdn_recurrent_kernel"), 1);
}

#[test]
fn the_combine_binds_the_next_block_s_norm_and_a_null_where_the_chain_breaks() {
    let shape = shape();
    let chained = hyper::combine(
        &shape,
        NextNorm::Chained(WeightRef::layer(LayerRef::target(1), Role::AttnMixNorm)),
        1,
        Elem::F32,
        matmul::NormedStreams::Stored,
    )
    .expect("the residual fits the fused combine");
    let operands = chained.steps()[0]
        .operands()
        .expect("the list fits a launch");
    assert_eq!(
        operands[4],
        Operand::Weight(WeightRef::layer(LayerRef::target(1), Role::AttnMixNorm))
    );
    let broken = hyper::combine(
        &shape,
        NextNorm::Broken,
        1,
        Elem::F32,
        matmul::NormedStreams::Scaled,
    )
    .expect("the residual fits");
    let operands = broken.steps()[0]
        .operands()
        .expect("the list fits a launch");
    assert_eq!(
        operands[4],
        Operand::Null,
        "a broken chain is a null pointer, which is what turns this launch into \
         the residual update alone"
    );
}

#[test]
fn the_tensor_core_attention_is_handed_a_destination_or_a_slab_and_never_both() {
    use shiso_engine::sparse::bind::Operand;
    use shiso_engine::sparse::scratch::Buf;

    // The kernel decides what to write by asking whether its partial pointer is
    // null: with a slab it stores an unnormalized partial for the combine, and
    // without one it normalizes and writes the destination. Handing it both is
    // the slab branch at a split count whose caller does not fold, so the
    // destination keeps the previous chunk's attention and the block
    // contributes numbers nothing in this pass computed. That is invisible at
    // one token, because the path is only reached from sixteen.
    let shape = shape();
    let ctx = AttnContext {
        tensor_core: true,
        tiled: true,
        ..AttnContext::default()
    };
    for rows in [16_u32, 512] {
        assert_eq!(qsa::path(&shape, ctx, rows), AttnPath::TensorCore);
        let plan = qsa::block(&shape, &attn_desc(&shape), SLOT, ctx, rows, GPU)
            .expect("the tensor-core path has a tier");
        let step = plan
            .steps()
            .iter()
            .find(|step| step.name() == "q4e_qsa_attention_tc_exact_kernel")
            .expect("the tensor-core entry is the one that ran");
        let args = step.operands().expect("the list fits a launch");
        let folded = plan.count("q4e_qsa_attention_combine_kernel") > 0;
        match (args[0], args[1]) {
            (Operand::Scratch(Buf::AttnOut), Operand::Null) => assert!(
                !folded,
                "a launch that wrote the destination itself must not be folded again",
            ),
            (Operand::Null, Operand::Scratch(Buf::AttnPartial)) => assert!(
                folded,
                "a launch that wrote only slabs has to be folded into the destination",
            ),
            other => panic!("exactly one of the two is a buffer, got {other:?}"),
        }
    }
}

#[test]
fn the_routed_experts_carve_three_staging_buffers_that_do_not_overlap() {
    use shiso_engine::sparse::exl3;
    use shiso_engine::sparse::plan::Step;

    let shape = shape();
    let plan = moe::fused_experts(&shape, LayerRef::target(0), 4, 3, GPU)
        .expect("a trellis expert stack has a fused stage");
    let fused = plan
        .steps()
        .iter()
        .find(|step| matches!(step, Step::Trellis { .. }))
        .expect("the fused stage is a cooperative launch");
    // The launch does not reserve: the reservation belongs to the pass, in
    // front of every island, because it cannot be made mid-capture and growing
    // it would move an address a captured island has baked in. What the launch
    // does carry is the size it needs, which the plan accumulates.
    assert_eq!(plan.count("stream scratch"), 0);
    assert!(plan.staging().bytes > 0, "the fused stage carves staging");
    assert_eq!(
        plan.coop_reservation(),
        exl3::STAGING_BASE + plan.staging().bytes,
        "a pass reserves the lock chain, the padding that realigns the staging \
         behind it, and the widest staging any of its launches carves"
    );

    // The three staging buffers are one region carved in the kernel's own
    // order, and two of them at the same offset would be a launch whose gate
    // output is its own input.
    let ops = fused.operands().expect("the list fits a launch");
    let offsets = [ops[1], ops[2], ops[3]].map(|op| match op {
        Operand::CoopStaging(at) => at,
        other => panic!("the three staging buffers are offsets: {other:?}"),
    });
    assert_eq!(offsets[0], 0, "the gathered input opens the region");
    assert!(
        offsets[0] < offsets[1] && offsets[1] < offsets[2],
        "{offsets:?}"
    );
    assert!(
        offsets[2] < plan.staging().bytes,
        "every buffer starts inside the region: {offsets:?}"
    );
    // And the lock chain is a fourth operand, at a fixed offset of zero rather
    // than inside the staging: every launch on a stream shares it and leaves it
    // zeroed, so it may not move when the staging grows.
    assert_eq!(ops[21], Operand::CoopLocks);
}

#[test]
fn a_trellis_pair_stays_one_launch_up_to_the_expand_threshold_and_splits_past_it() {
    use shiso_engine::sparse::plan::Step;

    let shape = shape();
    let inter = shape.moe.intermediate;
    let pair_of = |out: u32, rows| {
        let mut plan = Plan::new();
        matmul::project_pair(
            &mut plan,
            trellis(shape.hidden, out),
            trellis(shape.hidden, out),
            Bound::new(
                WeightRef::layer(LayerRef::target(0), Role::SharedGate),
                Operand::Scratch(Buf::Mixed),
                Operand::Scratch(Buf::SharedGate),
            ),
            Bound::new(
                WeightRef::layer(LayerRef::target(0), Role::SharedUp),
                Operand::Scratch(Buf::Mixed),
                Operand::Scratch(Buf::SharedUp),
            ),
            rows,
            GPU,
        )
        .expect("a matched trellis pair has a tier");
        plan
    };
    let pair = |rows| pair_of(inter, rows);
    // Narrow enough for the decode-shaped projection, the pair is two of
    // those at every row count that projection covers.
    let decoded = pair(kernels::EXL3_RECONSTRUCT_ROWS);
    assert_eq!(decoded.count("trellis pair"), 0);
    assert_eq!(decoded.count(kernels::EXL3_DGEMV), 2);
    assert_eq!(decoded.count("trellis expand"), 0);
    // One row past it each weight is expanded once and multiplied: there is no
    // grouped form of the expanded path, so the pair stops being one launch.
    let split = pair(kernels::EXL3_RECONSTRUCT_ROWS + 1);
    assert_eq!(split.count("trellis pair"), 0);
    assert_eq!(split.count("trellis expand"), 2);
    assert_eq!(split.count("blas gemm"), 2);
    // A pair wide on both sides is one grouped launch up to the threshold,
    // which runs both weights over half a device each rather than one after
    // the other.
    let grouped = pair_of(2_048, kernels::EXL3_RECONSTRUCT_ROWS);
    assert_eq!(grouped.count("trellis pair"), 1);
    assert_eq!(grouped.count("trellis expand"), 0);
    // And the grouped launch's two output slots are different buffers: one
    // launch writing one destination twice is a gate that is its own up.
    let step = grouped
        .steps()
        .iter()
        .find(|step| matches!(step, Step::Trellis { .. }))
        .expect("the pair is one cooperative launch");
    let ops = step.operands().expect("the list fits a launch");
    assert_eq!(ops.len(), 19);
    assert_ne!(ops[3], ops[17], "the two slots write different buffers");
    assert_eq!(ops[12], Operand::I32(1), "both slots read the same rows");
    assert_eq!(ops[13], Operand::I32(2), "two output slots");
    assert_eq!(
        ops[18],
        Operand::I32(1),
        "the second tensor starts at slot one"
    );
}

#[test]
fn a_dense_trellis_projection_is_handed_the_three_spans_of_its_payload_in_order() {
    use shiso_engine::sparse::plan::Step;
    use shiso_engine::sparse::weights::WeightPart;

    let shape = shape();
    let mut plan = Plan::new();
    // One row: far below the threshold past which the weight is expanded once,
    // so this takes the cooperative kernel.
    matmul::project(
        &mut plan,
        trellis(shape.hidden, shape.hidden),
        Bound::new(
            WeightRef::layer(LayerRef::target(0), Role::AttnOutput),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::BlockOut),
        ),
        1,
        GPU,
    )
    .expect("a trellis projection has a tier");
    let step = plan
        .steps()
        .iter()
        .find(|step| matches!(step, Step::Trellis { .. }))
        .expect("one row takes the cooperative kernel");
    let ops = step.operands().expect("the list fits a launch");
    // `A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh`. The payload is
    // one allocation with three spans, and the codec applies the input scales
    // before the matmul and the output scales after it: the two swapped is a
    // projection that runs and is wrong.
    assert_eq!(ops.len(), 10);
    assert_eq!(
        ops[1],
        Operand::Weight(
            WeightRef::layer(LayerRef::target(0), Role::AttnOutput).spanning(WeightPart::Tiles)
        )
    );
    assert_eq!(ops[3], Operand::U32(1));
    assert_eq!(ops[4], Operand::U32(shape.hidden));
    assert_eq!(ops[5], Operand::U32(shape.hidden));
    assert_eq!(ops[6], Operand::CoopLocks);
    assert_eq!(
        ops[7],
        Operand::Weight(
            WeightRef::layer(LayerRef::target(0), Role::AttnOutput)
                .spanning(WeightPart::InputScales)
        )
    );
    assert_eq!(ops[8], Operand::CoopStaging(0));
    assert_eq!(
        ops[9],
        Operand::Weight(
            WeightRef::layer(LayerRef::target(0), Role::AttnOutput)
                .spanning(WeightPart::OutputScales)
        )
    );
}

#[test]
fn the_two_multiply_tiers_differ_in_which_operand_they_read_transposed() {
    use shiso_core::backend::GemmOp;
    use shiso_engine::sparse::plan::Step;

    let shape = shape();
    // A half-precision weight is stored one output row at a time, so the
    // column-major tier reads it transposed and its leading dimension counts
    // the contraction.
    let mut wide = Plan::new();
    matmul::project(
        &mut wide,
        half(shape.hidden, shape.hidden),
        Bound::new(
            WeightRef::layer(LayerRef::target(0), Role::AttnOutput),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::BlockOut),
        ),
        512,
        GPU,
    )
    .expect("a half-precision projection has a tier");
    let Some(Step::Blas {
        op_a, m, n, k, lda, ..
    }) = wide.steps().iter().find(|s| matches!(s, Step::Blas { .. }))
    else {
        panic!("a wide half-precision chunk reaches the multiply tier");
    };
    assert_eq!(*op_a, GemmOp::Transposed);
    assert_eq!((*m, *n, *k), (shape.hidden, 512, shape.hidden));
    assert_eq!(*lda, shape.hidden);
    // And the rounding pass runs first: the tier has no path that reads a
    // single-precision right operand against a half-precision left one.
    let names = wide.names();
    let convert = names
        .iter()
        .position(|name| *name == "q4e_f32_to_f16_kernel")
        .expect("the activation is rounded before the multiply");
    let gemm = names
        .iter()
        .position(|name| *name == "blas gemm")
        .expect("the multiply is planned");
    assert!(convert < gemm, "{names:?}");

    // An expanded trellis weight comes out output-major, which the tier wants
    // without a transposition, and its leading dimension counts the output.
    let mut expanded = Plan::new();
    matmul::project(
        &mut expanded,
        trellis(shape.hidden, shape.hidden),
        Bound::new(
            WeightRef::layer(LayerRef::target(0), Role::AttnOutput),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::BlockOut),
        ),
        512,
        GPU,
    )
    .expect("a trellis projection has a tier");
    let Some(Step::Blas { op_a, lda, .. }) = expanded
        .steps()
        .iter()
        .find(|s| matches!(s, Step::Blas { .. }))
    else {
        panic!("a wide trellis chunk is expanded and multiplied");
    };
    assert_eq!(*op_a, GemmOp::AsStored);
    assert_eq!(*lda, shape.hidden);
    // The expansion is one block per tile of the expanded weight, not a flat
    // elementwise launch over its elements.
    let Some(Step::Expand { launch, .. }) = expanded
        .steps()
        .iter()
        .find(|s| matches!(s, Step::Expand { .. }))
    else {
        panic!("the weight is expanded first");
    };
    assert_eq!(launch.grid.x, shape.hidden / 128);
    assert_eq!(launch.grid.y, shape.hidden / 128);
    assert_eq!(launch.block.x, 256);
}
