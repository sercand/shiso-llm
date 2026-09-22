//! The routed feed-forward, end to end on the device, against a host reference
//! that does the whole block in plain Rust.
//!
//! Six of the eight launches here are `q4e_moe`'s own entries, extracted
//! verbatim from the reference engine and fidelity-gated - the router, the
//! three sort passes, the activation and the combine. Two are this family's,
//! because the sparse family's expert stages are a trellis cooperative launch
//! and a template over `{Q5_1, Q8_0}` and neither reads NVFP4.
//!
//! # Why the whole block and not the two new entries alone
//!
//! The two new entries are the only new arithmetic, but they are not the only
//! new thing: the *sequence* is new, and every seam in it is a place where a
//! correct kernel can be handed the wrong argument. The routed column `col` is
//! `token * n_used + slot`; the gate and up stages read activation row
//! `col / n_used` while the down stage reads its own row `col`; the down stage
//! writes column-major over (token, slot) because that is what the combine
//! reads. Four conventions, none of them checkable by looking at one kernel,
//! and all four are what this file grades.
//!
//! # Two codecs, one reference
//!
//! Every case below runs twice: once over NVFP4 and once over q4-affine. Only
//! the two expert entries and the widths their packers allow differ -
//! `ExpertCodec` holds both, and the affine fixture is two storage tiles wide
//! where the NVFP4 one is a single undivided slab. The six launches around them
//! are the same launches with the same arguments, because the router, the sort
//! passes, the activation and the combine read `f32` and know nothing about the
//! container.
//!
//! **The host reference is one implementation for both.** It works in `f32`
//! after each codec's own host decoder has run, which is the whole point: a
//! reference that had to be told which container it was grading would be two
//! references, and two references cannot say the two codecs agree. Their
//! measured divergences differ - `BUDGET` and `AFFINE_BUDGET` are each
//! recorded from a run rather than chosen - and every case prints what it saw.
//!
//! **The router's own law is graded here too, and it is not this tree's.**
//! `q4e_moe_route_kernel` softmaxes over every expert, takes the top `n_used`
//! and renormalizes over those - which is what `vllm`'s `Qwen3NextSparseMoeBlock`
//! does, `renormalize=getattr(config, "norm_topk_prob", True)` with this
//! config stating no `norm_topk_prob`. The host reference below implements
//! that law independently, so the case says the two agree rather than assuming
//! the entry was the right one to reach for.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "the kernel argument structs take i32 dimensions while the fixtures are \
              written in usize, and every fixture dimension and index here is a small \
              literal - the same allowance `crate::qwen35::gemv` takes for the same reason"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::qwen35moe::q4affine::{
    affine_fixture_weights, bf16_bytes, dequant_q4_affine, pack_q4_affine_tiled,
};
use bytemuck::{Pod, Zeroable};
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};
use shiso_modelfile::nvfp4::{GROUP, Nvfp4Layout, PER_BYTE, dequant_nvfp4};

/// This family's own module.
const MODULE: &str = "qwen35moe";

/// The sparse family's module, whose six general entries this family reaches
/// unchanged.
///
/// A module name is the key of a compiled image and says nothing about which
/// family may name it - `crates/shiso-engine/src/launch.rs` has a table of
/// exactly these decisions. What makes these six safe to share is not the name
/// but the fidelity gate: they are extracted verbatim and
/// `scripts/q4e_ptx_oracle.sh --mode=fidelity` refuses a tree in which any of
/// them has changed.
const Q4E_MOE: &str = "q4e_moe";

/// Threads per block of the linear `q4e_moe` entries.
const BLOCK_LINEAR: u32 = 256;

/// Threads per block of the two expert entries, which is
/// `SHISO_Q35MOE_EXPERT_BLOCK`.
const EXPERT_BLOCK: u32 = 128;

/// Output rows one expert block covers, which is
/// `SHISO_Q35MOE_EXPERT_WARPS`.
const EXPERT_ROWS: u32 = 4;

/// What a product here is allowed to diverge by, as a fraction of what the sum
/// accumulated.
///
/// **Not the codec cases' four ulps, and the difference is measured rather
/// than assumed.** Those grade ONE dot product; a routed block is a dot, a
/// `silu` through the device's fast exponential, a second dot over the result
/// and a weighted sum of two of those. The number below is what this fixture
/// actually produces; every case prints its own.
const BUDGET: f32 = 8.0 * f32::EPSILON;

/// What an intermediate stage is allowed to diverge by, as a fraction of that
/// stage's own largest magnitude.
const STAGE_BUDGET: f32 = 8.0 * f32::EPSILON;

/// What a product over the q4-affine codec is allowed to diverge by, as a
/// fraction of what the sum accumulated.
///
/// **Measured, and deliberately not the NVFP4 number above.** The two codecs
/// are graded at different widths - see [`ExpertCodec::shape`] - so their
/// summation orders differ by more than the container does, and reusing one
/// budget for both would be a claim neither run made.
///
/// An M-series device reports 7.345e-7 of what the sum carried, at five tokens
/// and at one alike - inside the NVFP4 fixture's `8 * EPSILON`, over reduction
/// depths of 256 and 512 against that fixture's 64 and 48. The budget is about
/// five times the measurement, which is headroom for another device's
/// fused-multiply-add placement and nothing like enough to hide a seam error:
/// those miss by whole factors.
const AFFINE_BUDGET: f32 = 4.0e-6;

/// What the routing weights over the q4-affine codec may diverge by.
///
/// The router reads `f32` logits and knows nothing about the codec, so this is
/// the same kernel the NVFP4 run grades - but it is measured here rather than
/// inherited. Both fixtures report 5.039e-8 of the stage's largest weight, at
/// five tokens and at one; the budget is about five times that.
///
/// **It is deliberately not shared with the projections.** A stage budget wide
/// enough for `mid` would leave this one 1,190 times loose, which is not a
/// measurement of anything.
const AFFINE_WEIGHT_BUDGET: f32 = 2.5e-7;

/// What the gate or up projection over the q4-affine codec may diverge by.
///
/// One number for the two because they are one launch over one pair of widths:
/// 5.704e-7 of the stage's largest value over a five-token chunk and 1.329e-6
/// over one token, identical between gate and up for the reason
/// [`Slab::affine`] records. The budget is about four and a half times the
/// larger measurement.
const AFFINE_PROJECTION_BUDGET: f32 = 6.0e-6;

/// What `silu(gate) * up` over the q4-affine codec may diverge by.
///
/// **The largest number this file measures, and it is this stage's alone.**
/// `mid` carries the gate stage's *absolute* error scaled by an element of the
/// up stage - `d(mid) = silu'(g) * dg * u + silu(g) * du` - while being
/// normalized by the largest value `mid` itself holds, and at one token that
/// denominator is 1.65 against an up stage reaching 7.18. That reports
/// 1.4205e-5, which is the ratio of two stages' magnitudes and not an error in
/// the activation: the whole block's own divergence, which is what
/// [`AFFINE_BUDGET`] holds, is nineteen times smaller. The budget is about four
/// times the measurement, and it is this stage's own so that the four stages
/// measuring 40 to 1,200 times tighter are each held to what they measure.
const AFFINE_MID_BUDGET: f32 = 6.0e-5;

/// What the down projection over the q4-affine codec may diverge by.
///
/// 6.690e-7 of the stage's largest value over a five-token chunk and 1.463e-6
/// over one token, over twice the gate projection's reduction depth. The budget
/// is about four times the larger measurement.
const AFFINE_DOWN_BUDGET: f32 = 6.0e-6;

/// Outputs per storage tile of a resident affine slab, `Q35MOE_STORAGE_N`.
///
/// A constant of the kernel set rather than an argument, so a fixture packed
/// at any other width is packed for a kernel that does not exist. It is why
/// the affine geometry below is not the NVFP4 one.
const STORAGE_N: usize = 256;

/// One graded stage of the routed block.
///
/// A budget is per stage and not per block, because the five stages do not
/// measure alike: see [`AFFINE_MID_BUDGET`] for the one that is two orders
/// wider than its neighbours and why that is the comparison rather than the
/// kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// The routing weights, which no codec touches.
    Weights,
    /// The gate or the up projection - one budget, one launch, one pair of
    /// widths.
    Projection,
    /// `silu(gate) * up`.
    Mid,
    /// The down projection, at the gate's transposed widths.
    Down,
}

/// The container the three expert projections are packed in.
///
/// The block around them is the same either way - the router, the three sort
/// passes, the activation and the combine read `f32` and know nothing about
/// the codec - so this picks two entry names, the geometry their packers
/// allow, and the budget each was measured at. **The host reference is one
/// implementation for both**, working in `f32` after the codec's own host
/// decoder has run, which is what makes the two runs comparable at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpertCodec {
    /// NVFP4: e2m1 codes, an e4m3 scale per group of 16 and one `f32`
    /// `weight_scale_2` per expert.
    Nvfp4,
    /// q4-affine: 4-bit codes, a `bfloat` scale and bias per group of 64, laid
    /// out group-major within a storage tile.
    Q4Affine,
}

impl ExpertCodec {
    /// The fused gate-and-up entry for this codec, in `f32` activations.
    ///
    /// `ActDtype::Bf16` has no entry here on purpose: `MoeScratch` allocates
    /// `f32` and the activation and combine entries take `float*`, so a bf16
    /// block would read `f32` bit patterns as `bfloat` without faulting.
    /// `Moe::new` refuses that pair, and this file grades the pair the engine
    /// can actually drive.
    const fn gate_up_entry(self) -> &'static str {
        match self {
            Self::Nvfp4 => "kernel_q35moe_experts_gate_up_nvfp4_f32",
            Self::Q4Affine => "kernel_q35moe_experts_gate_up_q4affine_f32",
        }
    }

    /// The down entry for this codec, in `f32` activations.
    const fn down_entry(self) -> &'static str {
        match self {
            Self::Nvfp4 => "kernel_q35moe_experts_down_nvfp4_f32",
            Self::Q4Affine => "kernel_q35moe_experts_down_q4affine_f32",
        }
    }

    /// The fixture geometry this codec's packer allows, at `tokens` tokens.
    ///
    /// **The two differ, and the difference is the codec's and not the case's.**
    /// NVFP4 runs at a hidden of 64 and an inner of 48: small enough to
    /// reference by hand and awkward enough that a stride shows, with 48 not a
    /// multiple of the four output rows a block covers. The affine packer
    /// cannot be handed either number - a group is 64 codes deep, so `in_dim`
    /// is a whole number of 64s, and a slab is a whole number of
    /// [`STORAGE_N`]-wide storage tiles - so its fixture is 256 wide by 512
    /// deep. That is not a relaxation: 512 is **two** storage tiles, where 48
    /// and 64 would both have been one, so the affine run grades a tile index
    /// the NVFP4 run has no way to reach.
    ///
    /// Everything else is held equal on purpose. Six experts at two per token
    /// is the same routing, over the same logits, so the two runs choose the
    /// same experts in the same order and a divergence between them is
    /// arithmetic rather than routing.
    const fn shape(self, tokens: usize) -> Shape {
        match self {
            Self::Nvfp4 => Shape {
                hidden: 64,
                inter: 48,
                experts: 6,
                used: 2,
                tokens,
            },
            Self::Q4Affine => Shape {
                hidden: STORAGE_N,
                inter: 2 * STORAGE_N,
                experts: 6,
                used: 2,
                tokens,
            },
        }
    }

    /// What a routed output of this codec may diverge by.
    const fn budget(self) -> f32 {
        match self {
            Self::Nvfp4 => BUDGET,
            Self::Q4Affine => AFFINE_BUDGET,
        }
    }

    /// What `stage` of this codec may diverge by.
    ///
    /// NVFP4 holds all five to one number, which is what its CUDA run was
    /// graded at and is not this branch's to move without a device to measure
    /// on. The affine run measured each stage and each is held to its own.
    const fn stage_budget(self, stage: Stage) -> f32 {
        match self {
            Self::Nvfp4 => STAGE_BUDGET,
            Self::Q4Affine => match stage {
                Stage::Weights => AFFINE_WEIGHT_BUDGET,
                Stage::Projection => AFFINE_PROJECTION_BUDGET,
                Stage::Mid => AFFINE_MID_BUDGET,
                Stage::Down => AFFINE_DOWN_BUDGET,
            },
        }
    }
}

/// The four widths a routed expert stage is launched over.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ExpertArgs {
    in_dim: i32,
    out_dim: i32,
    n_expert: i32,
    n_used: i32,
}
const _: () = assert!(size_of::<ExpertArgs>() == 16);

/// The geometry of the fixture, small enough to reference by hand and awkward
/// enough to catch a stride.
struct Shape {
    hidden: usize,
    inter: usize,
    experts: usize,
    used: usize,
    tokens: usize,
}

impl Shape {
    /// Routed columns one chunk produces.
    const fn columns(&self) -> usize {
        self.tokens * self.used
    }
}

/// One projection's experts: the three buffers an expert entry reads, and the
/// dense matrices the host reference multiplies against.
///
/// **Both codecs hand the entry the same three buffers in the same order** -
/// codes, a per-group scale, and a third parameter - so only the contents and
/// the widths differ and one struct serves both. Holding the two parameter
/// buffers as bytes rather than as `f32` is what makes that true: NVFP4's third
/// buffer is one `f32` per expert and the affine codec's is one `bfloat` per
/// group, and neither is the other's shape.
///
/// The dense matrices are decoded once, here, rather than per routed column.
/// The host decode is exact either way - it is a table lookup and a multiply -
/// and hoisting it keeps a reference that already walks two million products
/// from walking two million divisions as well.
struct Slab {
    /// Every expert's codes, concatenated at the entry's expert stride.
    codes: Vec<u8>,
    /// Every expert's group scales, in the container's own float.
    scales: Vec<u8>,
    /// NVFP4's one `f32` per expert, or the affine codec's per-group biases.
    params: Vec<u8>,
    /// Each expert's matrix, row-major `[out_dim, in_dim]`, decoded by that
    /// codec's own host decoder.
    dense: Vec<Vec<f32>>,
}

impl Slab {
    /// One projection's experts, packed in `codec`.
    fn new(codec: ExpertCodec, experts: usize, out_dim: usize, in_dim: usize, seed: usize) -> Self {
        match codec {
            ExpertCodec::Nvfp4 => Self::nvfp4(experts, out_dim, in_dim, seed),
            ExpertCodec::Q4Affine => Self::affine(experts, out_dim, in_dim, seed),
        }
    }

    /// An NVFP4 slab whose bytes walk the whole code range, with a different
    /// global per expert.
    ///
    /// **The globals differ**, which is the point: every expert of a layer has
    /// its own `weight_scale_2` in this checkpoint, and a kernel that read
    /// expert zero's for all of them would be wrong by whatever the spread is.
    /// A fixture whose globals were all equal could not tell that apart from
    /// correct code.
    fn nvfp4(experts: usize, out_dim: usize, in_dim: usize, seed: usize) -> Self {
        let packed = out_dim * in_dim / PER_BYTE;
        let groups = out_dim * in_dim / GROUP;
        let mut codes = Vec::with_capacity(experts * packed);
        let mut scales = Vec::with_capacity(experts * groups);
        let mut globals = Vec::with_capacity(experts);
        for e in 0..experts {
            for i in 0..packed {
                codes.push(((i * 7 + e * 13 + seed) % 256) as u8);
            }
            for i in 0..groups {
                // `0x30..0x48` spans e4m3 exponents either side of one, which
                // keeps the decoded magnitudes in a range a dot product can
                // still distinguish.
                scales.push((0x30 + ((i + e * 3 + seed) % 24)) as u8);
            }
            globals.push(((e + 1) as f32).mul_add(1.7e-5, 3.1e-5));
        }
        let dense = (0..experts)
            .map(|e| {
                let mut out = Vec::new();
                dequant_nvfp4(
                    Nvfp4Layout::new(out_dim, in_dim, globals[e]),
                    &codes[e * packed..(e + 1) * packed],
                    &scales[e * groups..(e + 1) * groups],
                    &mut out,
                )
                .expect("the fixture's own geometry");
                out
            })
            .collect();
        Self {
            codes,
            scales,
            params: f32_bytes(&globals),
            dense,
        }
    }

    /// A q4-affine slab, one storage-tiled pack per expert.
    ///
    /// The values are [`affine_fixture_weights`]', which every code in `0..=15`
    /// and every group parameter of; `seed` and the expert id rotate the output
    /// rows, so gate, up and down are three different matrices and every expert
    /// is a fourth. An entry that read the gate slab for the up projection, or
    /// expert zero's rows for all six, would otherwise produce the right
    /// numbers.
    ///
    /// **"Different" means "differently permuted" for the gate and up pair, and
    /// a reader who does not know that will chase a copy-paste bug that is not
    /// there.** The rotation is `(e * 37 + seed * 91) % out_dim`, so at
    /// `out_dim = 512` the gate slab is the base matrix rotated by `e * 37` and
    /// the up slab the same base rotated by `e * 37 + 89`: both are cyclic row
    /// permutations of one matrix, which is why the two print byte-identical
    /// worst divergences and identical stage scales. A swapped gate and up
    /// buffer is still caught, because the two shifts differ by 89 rows. `down`
    /// is a genuinely different matrix - 256 by 512 against the pair's 512 by
    /// 256, so its base is a different call - and it prints its own numbers.
    ///
    /// The pack is [`pack_q4_affine_tiled`] rather than a packer of this file's
    /// own, and the parameters are uploaded through the same [`bf16_bytes`] the
    /// codec's own cases use. That is not tidiness: the container holds
    /// `bfloat` scales and biases, `pack_q4_affine` rounds both to `bfloat`
    /// *before* it chooses codes, and a slab packed against `f32` parameters is
    /// a reference for a container nobody writes.
    fn affine(experts: usize, out_dim: usize, in_dim: usize, seed: usize) -> Self {
        let base = affine_fixture_weights(out_dim, in_dim);
        let slabs: Vec<_> = (0..experts)
            .map(|e| {
                let shift = (e * 37 + seed * 91) % out_dim;
                let mut values = vec![0f32; out_dim * in_dim];
                for n in 0..out_dim {
                    let src = (n + shift) % out_dim;
                    values[n * in_dim..(n + 1) * in_dim]
                        .copy_from_slice(&base[src * in_dim..(src + 1) * in_dim]);
                }
                pack_q4_affine_tiled(&values, out_dim, in_dim, STORAGE_N)
            })
            .collect();
        let scales: Vec<f32> = slabs.iter().flat_map(|s| s.scales.clone()).collect();
        let biases: Vec<f32> = slabs.iter().flat_map(|s| s.biases.clone()).collect();
        let dense = slabs
            .iter()
            .map(|slab| {
                (0..out_dim * in_dim)
                    .map(|i| dequant_q4_affine(slab, i))
                    .collect()
            })
            .collect();
        Self {
            codes: slabs.iter().flat_map(|s| s.weights.clone()).collect(),
            scales: bf16_bytes(&scales),
            params: bf16_bytes(&biases),
            dense,
        }
    }
}

/// The three projections one routed block multiplies by.
///
/// Held together because they are always built, uploaded and read as a set,
/// and because a routed block that took them one at a time would hand nine
/// buffers to every function that touches a weight.
struct Projections {
    gate: Slab,
    up: Slab,
    down: Slab,
}

impl Projections {
    /// The three, packed in `codec` at this shape's widths.
    ///
    /// The three seeds differ, so gate, up and down are three different
    /// matrices: an entry that read the gate slab for the up projection would
    /// otherwise produce numbers the reference agrees with.
    fn new(codec: ExpertCodec, shape: &Shape) -> Self {
        Self {
            gate: Slab::new(codec, shape.experts, shape.inter, shape.hidden, 0),
            up: Slab::new(codec, shape.experts, shape.inter, shape.hidden, 91),
            down: Slab::new(codec, shape.experts, shape.hidden, shape.inter, 173),
        }
    }
}

/// The routing a host reference derives from one token's logits.
///
/// Softmax over every expert, the top `used` by probability, then renormalized
/// over the chosen ones - independently of the kernel, from the reference
/// implementation's own definition.
fn route(logits: &[f32], used: usize) -> (Vec<usize>, Vec<f32>) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|v| v / sum).collect();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    // By logit, keeping the lower id on a tie, which is what the entry's
    // repeated block argmax does: `e` only grows, so the first of equal values
    // wins.
    order.sort_by(|a, b| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });
    let chosen: Vec<usize> = order.into_iter().take(used).collect();
    let total: f32 = chosen.iter().map(|e| probs[*e]).sum();
    let total = total.max(1e-20);
    let weights = chosen.iter().map(|e| probs[*e] / total).collect();
    (chosen, weights)
}

/// One reference answer, and what each element of it accumulated.
struct Reference {
    values: Vec<f32>,
    magnitude: Vec<f32>,
}

/// The same six stages, computed on the host.
struct RefStages {
    ids: Vec<i32>,
    weights: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    mid: Vec<f32>,
    down: Vec<f32>,
    out: Reference,
}

/// The whole routed block, in plain Rust over the same bytes, stage by stage.
fn reference(shape: &Shape, slabs: &Projections, logits: &[f32], x: &[f32]) -> RefStages {
    let columns = shape.columns();
    let mut out_ids = vec![-1i32; columns];
    let mut out_weights = vec![0.0f32; columns];
    let mut out_gate = vec![0.0f32; columns * shape.inter];
    let mut out_up = vec![0.0f32; columns * shape.inter];
    let mut out_mid = vec![0.0f32; columns * shape.inter];
    let mut out_down = vec![0.0f32; columns * shape.hidden];
    let mut values = vec![0.0f32; shape.tokens * shape.hidden];
    let mut magnitude = vec![0.0f32; shape.tokens * shape.hidden];
    for tok in 0..shape.tokens {
        let (experts, weights) = route(
            &logits[tok * shape.experts..(tok + 1) * shape.experts],
            shape.used,
        );
        for (slot, (&expert, &weight)) in experts.iter().zip(&weights).enumerate() {
            let col = tok * shape.used + slot;
            out_ids[col] = expert as i32;
            out_weights[col] = weight;
            let g = &slabs.gate.dense[expert];
            let u = &slabs.up.dense[expert];
            let d = &slabs.down.dense[expert];
            for i in 0..shape.inter {
                let mut gsum = 0.0f32;
                let mut usum = 0.0f32;
                for k in 0..shape.hidden {
                    gsum += g[i * shape.hidden + k] * x[tok * shape.hidden + k];
                    usum += u[i * shape.hidden + k] * x[tok * shape.hidden + k];
                }
                out_gate[col * shape.inter + i] = gsum;
                out_up[col * shape.inter + i] = usum;
                // `silu(gate) * up`, which is `q4e_swiglu_kernel`.
                out_mid[col * shape.inter + i] = (gsum / (1.0 + (-gsum).exp())) * usum;
            }
            for e in 0..shape.hidden {
                let mut sum = 0.0f32;
                let mut carried = 0.0f32;
                for i in 0..shape.inter {
                    let term = d[e * shape.inter + i] * out_mid[col * shape.inter + i];
                    sum += term;
                    carried += term.abs();
                }
                out_down[col * shape.hidden + e] = sum;
                values[tok * shape.hidden + e] += sum * weight;
                // What the WHOLE two-level sum accumulated, not what it came
                // out as. Each `down` element carries an absolute error set by
                // its own dot product over `inter` terms, and the combine adds
                // `used` of those; an output element where the two nearly
                // cancel is a small number carrying a large sum's rounding,
                // and normalizing by the element itself reports that as a
                // kernel defect. This is the same mistake the FP8 oracle beside
                // this made on its first run, one level deeper.
                magnitude[tok * shape.hidden + e] += carried * weight.abs();
            }
        }
    }
    RefStages {
        ids: out_ids,
        weights: out_weights,
        gate: out_gate,
        up: out_up,
        mid: out_mid,
        down: out_down,
        out: Reference { values, magnitude },
    }
}

/// Compares every stage in order and reports the first that disagrees.
///
/// Absolute, scaled by the largest magnitude in the reference stage, because
/// the intermediate stages are not sums whose conditioning travels with them.
#[expect(clippy::print_stdout, reason = "the measurement is this case's output")]
fn compare_stages(got: &Stages, want: &RefStages, codec: ExpertCodec, label: &str) {
    assert_eq!(
        got.ids, want.ids,
        "{label}: the routing chose other experts"
    );
    let named: [(&str, Stage, &[f32], &[f32]); 5] = [
        ("weights", Stage::Weights, &got.weights, &want.weights),
        ("gate", Stage::Projection, &got.gate, &want.gate),
        ("up", Stage::Projection, &got.up, &want.up),
        ("mid", Stage::Mid, &got.mid, &want.mid),
        ("down", Stage::Down, &got.down, &want.down),
    ];
    for (name, stage, mine, theirs) in named {
        let budget = codec.stage_budget(stage);
        assert_eq!(mine.len(), theirs.len(), "{label}/{name}: element count");
        let scale = theirs
            .iter()
            .fold(0.0f32, |acc, v| acc.max(v.abs()))
            .max(f32::MIN_POSITIVE);
        let mut worst = 0.0f32;
        let mut at = 0usize;
        for (i, (a, b)) in mine.iter().zip(theirs).enumerate() {
            let d = (a - b).abs() / scale;
            if d > worst {
                worst = d;
                at = i;
            }
        }
        println!(
            "{label}/{name}: worst {worst:e} at element {at} (stage scale {scale:e}, \
             budget {budget:e})"
        );
        assert!(
            worst <= budget,
            "{label}/{name}: element {at} was {} against {} ({worst:e} of the stage's own \
             largest magnitude {scale:e}, budget {budget:e})",
            mine[at],
            theirs[at],
        );
    }
}

/// Compares and reports, naming the worst divergence and the budget.
#[expect(clippy::print_stdout, reason = "the measurement is this case's output")]
fn assert_close(got: &[f32], want: &Reference, budget: f32, label: &str) {
    assert_eq!(got.len(), want.values.len(), "{label}: element count");
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (a, b)) in got.iter().zip(&want.values).enumerate() {
        let scale = want.magnitude[i].max(f32::MIN_POSITIVE);
        let relative = (a - b).abs() / scale;
        if relative > worst {
            worst = relative;
            worst_at = i;
        }
    }
    println!(
        "{label}: worst divergence {worst:e} of what the sum carried, at element {worst_at}, \
         budget {budget:e}"
    );
    assert!(
        worst <= budget,
        "{label}: element {worst_at} was {} against {} over a sum carrying {} \
         ({worst:e}, budget {budget:e})",
        got[worst_at],
        want.values[worst_at],
        want.magnitude[worst_at],
    );
}

/// The device buffers one routed block needs.
struct Scratch {
    ids: DeviceBuffer,
    weights: DeviceBuffer,
    counts: DeviceBuffer,
    bounds: DeviceBuffer,
    active: DeviceBuffer,
    sorted: DeviceBuffer,
    gate: DeviceBuffer,
    up: DeviceBuffer,
    mid: DeviceBuffer,
    down: DeviceBuffer,
    out: DeviceBuffer,
}

/// `n` zeroed `int32` slots.
fn zeros_i32(gpu: &Gpu, n: usize) -> DeviceBuffer {
    gpu.upload(&vec![0u8; n * 4])
}

/// `n` poisoned `f32` slots, so a stage that skips one is a failure rather
/// than a value that happened to already be right.
fn poison(gpu: &Gpu, n: usize) -> DeviceBuffer {
    gpu.upload(&f32_bytes(&vec![f32::from_bits(0xA5A5_A5A5); n]))
}

/// What every stage of the block wrote, read back.
///
/// Returned as a whole rather than just the answer, because a routed block is
/// six launches and a divergence in the last of them says nothing about which.
/// When a case fails it names the first stage that disagrees, which is the
/// difference between "the numbers are wrong" and "the down projection is
/// reading its input row from the wrong column".
#[derive(Debug)]
pub struct Stages {
    /// One expert id per routed column.
    pub ids: Vec<i32>,
    /// One routing weight per routed column.
    pub weights: Vec<f32>,
    /// The expert-sorted column map.
    pub sorted: Vec<i32>,
    /// The gate projection, `[columns, inter]`.
    pub gate: Vec<f32>,
    /// The up projection, `[columns, inter]`.
    pub up: Vec<f32>,
    /// `silu(gate) * up`, `[columns, inter]`.
    pub mid: Vec<f32>,
    /// The down projection, `[columns, hidden]`.
    pub down: Vec<f32>,
    /// One contribution per token, `[tokens, hidden]`.
    pub out: Vec<f32>,
}

/// Reads a whole device buffer as `int32` values.
fn read_i32(gpu: &Gpu, buffer: &DeviceBuffer) -> Vec<i32> {
    gpu.read(buffer)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| i32::from_le_bytes(*w))
        .collect()
}

/// The weight slabs, uploaded.
struct Resident {
    gate_w: DeviceBuffer,
    gate_s: DeviceBuffer,
    gate_p: DeviceBuffer,
    up_w: DeviceBuffer,
    up_s: DeviceBuffer,
    up_p: DeviceBuffer,
    down_w: DeviceBuffer,
    down_s: DeviceBuffer,
    down_p: DeviceBuffer,
}

impl Resident {
    /// Uploads the three slabs, nine buffers in all: each is packed codes,
    /// group scales and the codec's third parameter, already in the bytes the
    /// entry reads.
    fn new(gpu: &Gpu, w: &Projections) -> Self {
        Self {
            gate_w: gpu.upload(&w.gate.codes),
            gate_s: gpu.upload(&w.gate.scales),
            gate_p: gpu.upload(&w.gate.params),
            up_w: gpu.upload(&w.up.codes),
            up_s: gpu.upload(&w.up.scales),
            up_p: gpu.upload(&w.up.params),
            down_w: gpu.upload(&w.down.codes),
            down_s: gpu.upload(&w.down.scales),
            down_p: gpu.upload(&w.down.params),
        }
    }
}

/// The router and the counting sort: the four launches that decide which
/// expert sees which column, and in what order.
fn route_and_sort(gpu: &Gpu, shape: &Shape, scratch: &Scratch, logits: &DeviceBuffer) {
    let columns = shape.columns();
    let linear = |n: usize| Extent3::linear((n as u32).div_ceil(BLOCK_LINEAR));

    // Softmax over every expert, the top `used`, renormalized over those.
    run(
        gpu,
        KernelRef::new(Q4E_MOE, "q4e_moe_route_kernel"),
        &LaunchArgs {
            grid: Extent3::linear(shape.tokens as u32),
            block: Extent3::linear(BLOCK_LINEAR),
            smem_bytes: (shape.experts * 4) as u32,
            args: &[
                KernelArg::Buffer(scratch.ids.view()),
                KernelArg::Buffer(scratch.weights.view()),
                KernelArg::Buffer(logits.view()),
                KernelArg::U32(shape.experts as u32),
                KernelArg::U32(shape.used as u32),
            ],
        },
    );

    // The counting sort that groups the columns by expert: histogram, scan,
    // scatter. The scatter's cursor is a row of its own here; the sparse
    // family reuses the histogram row and zeroes it in between, which is the
    // same thing with one fewer allocation and one more thing to get wrong.
    run(
        gpu,
        KernelRef::new(Q4E_MOE, "q4e_moe_sort_count_kernel"),
        &LaunchArgs {
            grid: linear(columns),
            block: Extent3::linear(BLOCK_LINEAR),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(scratch.counts.view()),
                KernelArg::Buffer(scratch.ids.view()),
                KernelArg::U32(columns as u32),
                KernelArg::U32(shape.experts as u32),
            ],
        },
    );
    run(
        gpu,
        KernelRef::new(Q4E_MOE, "q4e_moe_sort_scan_kernel"),
        &LaunchArgs {
            grid: Extent3::linear(1),
            block: Extent3::linear(32),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(scratch.bounds.view()),
                KernelArg::Buffer(scratch.active.view()),
                KernelArg::Buffer(scratch.counts.view()),
                KernelArg::U32(shape.experts as u32),
            ],
        },
    );
    let cursor = zeros_i32(gpu, shape.experts);
    run(
        gpu,
        KernelRef::new(Q4E_MOE, "q4e_moe_sort_scatter_kernel"),
        &LaunchArgs {
            grid: linear(columns),
            block: Extent3::linear(BLOCK_LINEAR),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(scratch.sorted.view()),
                KernelArg::Buffer(cursor.view()),
                KernelArg::Buffer(scratch.bounds.view()),
                KernelArg::Buffer(scratch.ids.view()),
                KernelArg::U32(columns as u32),
                KernelArg::U32(shape.experts as u32),
            ],
        },
    );
}

/// The three launches that turn a routed column into one expert's answer:
/// the fused gate-and-up, the activation, and the down projection.
fn expert_stages(
    gpu: &Gpu,
    shape: &Shape,
    scratch: &Scratch,
    w: &Resident,
    x_buf: &DeviceBuffer,
    codec: ExpertCodec,
) {
    let columns = shape.columns();
    // The gate and up projections, grouped by expert.
    let expert_grid = |out_dim: usize| {
        Extent3::new(
            (out_dim as u32).div_ceil(EXPERT_ROWS),
            (columns.min(shape.experts)) as u32,
            1,
        )
    };
    let gate_args = ExpertArgs {
        in_dim: shape.hidden as i32,
        out_dim: shape.inter as i32,
        n_expert: shape.experts as i32,
        n_used: shape.used as i32,
    };
    run(
        gpu,
        KernelRef::new(MODULE, codec.gate_up_entry()),
        &LaunchArgs {
            grid: expert_grid(shape.inter),
            block: Extent3::linear(EXPERT_BLOCK),
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&gate_args)),
                KernelArg::Buffer(w.gate_w.view()),
                KernelArg::Buffer(w.gate_s.view()),
                KernelArg::Buffer(w.gate_p.view()),
                KernelArg::Buffer(w.up_w.view()),
                KernelArg::Buffer(w.up_s.view()),
                KernelArg::Buffer(w.up_p.view()),
                KernelArg::Buffer(x_buf.view()),
                KernelArg::Buffer(scratch.sorted.view()),
                KernelArg::Buffer(scratch.bounds.view()),
                KernelArg::Buffer(scratch.active.view()),
                KernelArg::Buffer(scratch.gate.view()),
                KernelArg::Buffer(scratch.up.view()),
            ],
        },
    );

    // `silu(gate) * up` over every routed column.
    let elements = (columns * shape.inter) as u64;
    run(
        gpu,
        KernelRef::new(Q4E_MOE, "q4e_swiglu_kernel"),
        &LaunchArgs {
            grid: Extent3::linear((elements as u32).div_ceil(BLOCK_LINEAR)),
            block: Extent3::linear(BLOCK_LINEAR),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(scratch.mid.view()),
                KernelArg::Buffer(scratch.gate.view()),
                KernelArg::Buffer(scratch.up.view()),
                KernelArg::U64(elements),
            ],
        },
    );

    // The down projection, grouped by the same map.
    let down_args = ExpertArgs {
        in_dim: shape.inter as i32,
        out_dim: shape.hidden as i32,
        n_expert: shape.experts as i32,
        n_used: shape.used as i32,
    };
    run(
        gpu,
        KernelRef::new(MODULE, codec.down_entry()),
        &LaunchArgs {
            grid: expert_grid(shape.hidden),
            block: Extent3::linear(EXPERT_BLOCK),
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&down_args)),
                KernelArg::Buffer(w.down_w.view()),
                KernelArg::Buffer(w.down_s.view()),
                KernelArg::Buffer(w.down_p.view()),
                KernelArg::Buffer(scratch.mid.view()),
                KernelArg::Buffer(scratch.sorted.view()),
                KernelArg::Buffer(scratch.bounds.view()),
                KernelArg::Buffer(scratch.active.view()),
                KernelArg::Buffer(scratch.down.view()),
            ],
        },
    );
}

/// Runs the whole routed block on the device and returns every stage.
fn run_block(
    gpu: &Gpu,
    codec: ExpertCodec,
    shape: &Shape,
    weights: &Projections,
    logits: &[f32],
    x: &[f32],
) -> Stages {
    let columns = shape.columns();
    let scratch = Scratch {
        ids: zeros_i32(gpu, columns),
        weights: poison(gpu, columns),
        counts: zeros_i32(gpu, shape.experts),
        bounds: zeros_i32(gpu, shape.experts + 1),
        active: zeros_i32(gpu, shape.experts + 1),
        sorted: zeros_i32(gpu, columns),
        gate: poison(gpu, columns * shape.inter),
        up: poison(gpu, columns * shape.inter),
        mid: poison(gpu, columns * shape.inter),
        down: poison(gpu, columns * shape.hidden),
        out: poison(gpu, shape.tokens * shape.hidden),
    };
    let logits_buf = gpu.upload(&f32_bytes(logits));
    let x_buf = gpu.upload(&f32_bytes(x));
    let w = Resident::new(gpu, weights);

    route_and_sort(gpu, shape, &scratch, &logits_buf);

    expert_stages(gpu, shape, &scratch, &w, &x_buf, codec);

    // One contribution per token, weighted by the routing.
    run(
        gpu,
        KernelRef::new(Q4E_MOE, "q4e_moe_combine_kernel"),
        &LaunchArgs {
            grid: Extent3::new(
                (shape.hidden as u32).div_ceil(BLOCK_LINEAR),
                shape.tokens as u32,
                1,
            ),
            block: Extent3::linear(BLOCK_LINEAR),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(scratch.out.view()),
                KernelArg::Buffer(scratch.down.view()),
                KernelArg::Buffer(scratch.weights.view()),
                KernelArg::U32(shape.hidden as u32),
                KernelArg::U32(shape.used as u32),
            ],
        },
    );
    drain(gpu);
    Stages {
        ids: read_i32(gpu, &scratch.ids),
        weights: gpu.read_f32(&scratch.weights),
        sorted: read_i32(gpu, &scratch.sorted),
        gate: gpu.read_f32(&scratch.gate),
        up: gpu.read_f32(&scratch.up),
        mid: gpu.read_f32(&scratch.mid),
        down: gpu.read_f32(&scratch.down),
        out: gpu.read_f32(&scratch.out),
    }
}

/// Router logits that put different experts first for different tokens.
fn logits(shape: &Shape) -> Vec<f32> {
    let mut out = Vec::with_capacity(shape.tokens * shape.experts);
    for tok in 0..shape.tokens {
        for e in 0..shape.experts {
            // A different permutation per token, so no expert is chosen by
            // every token and no token routes to a contiguous run - which is
            // what makes the sorted column map do any work at all.
            out.push((((e * 5 + tok * 3) % shape.experts) as f32).mul_add(0.37, -1.1));
        }
    }
    out
}

/// Activations that are not all one magnitude.
fn activations(shape: &Shape) -> Vec<f32> {
    let mut out = Vec::with_capacity(shape.tokens * shape.hidden);
    for tok in 0..shape.tokens {
        for k in 0..shape.hidden {
            out.push((((tok * shape.hidden + k) % 61) as f32).mul_add(0.041, -1.2));
        }
    }
    out
}

/// The routed block at `tokens` tokens, in one codec, against the host
/// reference.
///
/// The body of the two chunk cases below and their two affine twins. What the
/// codec changes is the two entry names and the widths their packers allow;
/// what it does not change is [`reference`], which works in `f32` after the
/// host decoder has run and is the one thing both codecs are held to.
fn routed_block(gpu: &Gpu, codec: ExpertCodec, tokens: usize, label: &str) {
    let shape = codec.shape(tokens);
    let weights = Projections::new(codec, &shape);
    let logits = logits(&shape);
    let x = activations(&shape);

    let want = reference(&shape, &weights, &logits, &x);
    let got = run_block(gpu, codec, &shape, &weights, &logits, &x);
    compare_stages(&got, &want, codec, label);
    assert_close(&got.out, &want.out, codec.budget(), label);
}

/// The first row of a chunk against the same token run alone, in one codec.
fn first_row_against_the_same_token(gpu: &Gpu, codec: ExpertCodec, label: &str) {
    let chunk = codec.shape(5);
    let single = codec.shape(1);
    let weights = Projections::new(codec, &chunk);
    let all = logits(&chunk);
    let x = activations(&chunk);

    let many = run_block(gpu, codec, &chunk, &weights, &all, &x);
    let one = run_block(
        gpu,
        codec,
        &single,
        &weights,
        &all[..chunk.experts],
        &x[..chunk.hidden],
    );
    let want = reference(&single, &weights, &all[..chunk.experts], &x[..chunk.hidden]);
    assert_close(
        &many.out[..chunk.hidden],
        &Reference {
            magnitude: want.out.magnitude,
            values: one.out,
        },
        codec.budget(),
        label,
    );
}

/// The routed block at a chunk of tokens, against the host reference.
///
/// # Panics
///
/// When a launch is refused or the block diverges past the budget.
pub fn the_routed_block_matches_the_host_reference(gpu: &Gpu) {
    // Every width is deliberately not the checkpoint's: 6 experts at 2 per
    // token over a hidden of 64 and an inner of 48 is small enough that the
    // reference runs in a test and awkward enough that a stride is visible.
    // 48 is not a multiple of the four output rows a block covers, and 6
    // experts with 2 used means some experts get several columns and some get
    // none. `ExpertCodec::shape` holds that geometry and the affine codec's.
    routed_block(gpu, ExpertCodec::Nvfp4, 5, "routed block");
}

/// The same chunk over the q4-affine codec.
///
/// Not a second grading of the same thing: the affine fixture is 256 wide by
/// 512 deep, which is **two** storage tiles where the NVFP4 fixture is one
/// undivided slab, so the tile index in the affine entry's parameter
/// arithmetic is reached here and nowhere else in this file. The six launches
/// around the two expert entries are the same launches, with the same
/// arguments, and the reference is the same function.
///
/// # Panics
///
/// When a launch is refused or the block diverges past the budget.
pub fn the_routed_block_in_q4_affine_matches_the_host_reference(gpu: &Gpu) {
    routed_block(gpu, ExpertCodec::Q4Affine, 5, "routed block in q4-affine");
}

/// The same block at one token, which is what a decode step runs.
///
/// Not a duplicate of the case above. At one token every active expert owns
/// exactly one column, so the grouped sweep degenerates to the per-column
/// form - and this is what says the single path really does serve both,
/// rather than there being a row count at which the model starts answering
/// differently.
///
/// # Panics
///
/// When a launch is refused or the block diverges past the budget.
pub fn the_routed_block_at_one_token_matches_the_same_reference(gpu: &Gpu) {
    routed_block(gpu, ExpertCodec::Nvfp4, 1, "routed block at one token");
}

/// The q4-affine block at one token, against the same reference.
///
/// # Panics
///
/// When a launch is refused or the block diverges past the budget.
pub fn the_routed_block_in_q4_affine_at_one_token_matches_the_same_reference(gpu: &Gpu) {
    routed_block(
        gpu,
        ExpertCodec::Q4Affine,
        1,
        "routed block in q4-affine at one token",
    );
}

/// The first token of a chunk answers what it answers alone.
///
/// The routing is per token and the expert stages are grouped across tokens,
/// so a chunk and a single token take genuinely different paths through the
/// sorted map. If the first row of a five-token chunk differed from the same
/// prompt's single-token answer, the model would answer differently depending
/// on how the prompt was chunked - which is an open item in this repository's
/// own handoff for a different family.
///
/// # Panics
///
/// When a launch is refused or the two disagree.
pub fn the_first_row_of_a_chunk_is_what_that_token_answers_alone(gpu: &Gpu) {
    first_row_against_the_same_token(
        gpu,
        ExpertCodec::Nvfp4,
        "the first row of a chunk against the same token alone",
    );
}

/// The same chunk-against-single agreement over the q4-affine codec.
///
/// The affine entries sweep a whole expert's span of the sorted map with the
/// weight row held across the columns routed to it, so a chunk of five tokens
/// and a chunk of one really do run different loop trip counts over the same
/// arithmetic. That the two agree is the claim, and it is a claim per codec.
///
/// # Panics
///
/// When a launch is refused or the two disagree.
pub fn the_first_row_of_a_q4_affine_chunk_is_what_that_token_answers_alone(gpu: &Gpu) {
    first_row_against_the_same_token(
        gpu,
        ExpertCodec::Q4Affine,
        "the first row of a q4-affine chunk against the same token alone",
    );
}

/// The shared expert's gated add, which is the only part of it that is not an
/// ordinary projection.
///
/// **Step 3 of this plan's Task 5 needed no new kernel, and this case is where
/// that is written down.** The shared expert is a dense feed-forward at the
/// routed experts' own width whose three matrices are NVFP4, so it runs
/// through `projection::projection` and the entries Task 4 landed, and its
/// activation is the same `q4e_swiglu_kernel` the routed columns take. What is
/// left over is one scalar per token - `shared_expert_gate` is a `[1, hidden]`
/// projection - pushed through a sigmoid and multiplied into the shared
/// output as it is added onto the routed one. `q4e_shared_add_kernel` is
/// exactly that, and this grades it.
///
/// The gate is a SCALAR per token and not a vector per channel. A kernel that
/// read `logit[e]` instead of `logit[t]` would still produce a finite,
/// plausibly-scaled residual, so the fixture gives different tokens very
/// different gates and asserts every channel of a token takes the same one.
///
/// # Panics
///
/// When the launch is refused or the gated add diverges past the budget.
#[expect(clippy::print_stdout, reason = "the measurement is this case's output")]
pub fn the_shared_expert_s_gate_is_one_scalar_per_token(gpu: &Gpu) {
    const TOKENS: usize = 4;
    const HIDDEN: usize = 67;

    // Gates spanning the sigmoid's whole useful range, including both signs,
    // so a token that read another token's gate is off by a visible factor.
    let logits: Vec<f32> = vec![-4.0, -0.25, 0.75, 3.5];
    let routed: Vec<f32> = (0..TOKENS * HIDDEN)
        .map(|i| ((i % 29) as f32).mul_add(0.07, -1.0))
        .collect();
    let shared: Vec<f32> = (0..TOKENS * HIDDEN)
        .map(|i| ((i % 17) as f32).mul_add(-0.13, 0.9))
        .collect();

    let mut want = routed.clone();
    for (tok, logit) in logits.iter().enumerate() {
        let gate = 1.0 / (1.0 + (-logit).exp());
        for e in 0..HIDDEN {
            want[tok * HIDDEN + e] += shared[tok * HIDDEN + e] * gate;
        }
    }

    let out = gpu.upload(&f32_bytes(&routed));
    let shared_buf = gpu.upload(&f32_bytes(&shared));
    let logit_buf = gpu.upload(&f32_bytes(&logits));
    run(
        gpu,
        KernelRef::new(Q4E_MOE, "q4e_shared_add_kernel"),
        &LaunchArgs {
            grid: Extent3::new((HIDDEN as u32).div_ceil(BLOCK_LINEAR), TOKENS as u32, 1),
            block: Extent3::linear(BLOCK_LINEAR),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(shared_buf.view()),
                KernelArg::Buffer(logit_buf.view()),
                KernelArg::U32(HIDDEN as u32),
            ],
        },
    );
    drain(gpu);
    let got = gpu.read_f32(&out);

    let scale = want
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()))
        .max(f32::MIN_POSITIVE);
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        let d = (a - b).abs() / scale;
        if d > worst {
            worst = d;
            at = i;
        }
    }
    println!("shared expert gated add: worst {worst:e} at element {at}, budget {STAGE_BUDGET:e}");
    assert!(
        worst <= STAGE_BUDGET,
        "element {at} (token {}, channel {}) was {} against {}",
        at / HIDDEN,
        at % HIDDEN,
        got[at],
        want[at],
    );

    // And the gate really is per token: the ratio of what was added to what
    // was offered is one number for a whole row, and four different numbers
    // across the four rows.
    for (tok, logit) in logits.iter().enumerate() {
        let gate = 1.0 / (1.0 + (-logit).exp());
        let added = got[tok * HIDDEN] - routed[tok * HIDDEN];
        let offered = shared[tok * HIDDEN];
        assert!(
            (added / offered - gate).abs() < 1e-5,
            "token {tok} was gated by {} and not by {gate}",
            added / offered,
        );
    }
}

#[cfg(test)]
mod geometry {
    //! This checkpoint's real widths against each entry's own bounds.
    //!
    //! No device, and that is the point: **256 experts at 8 per token over a
    //! width of 512 is a narrower, deeper fan-out than the sparse family's 512
    //! at 10 over 640**, and the six entries this family borrows were written
    //! for the second. Each of their bounds is checked here against the first,
    //! rather than assumed to scale, because a grid or a shared-memory request
    //! that is wrong at these widths is a launch that succeeds and writes part
    //! of its output.

    use super::{BLOCK_LINEAR, EXPERT_ROWS};

    /// The checkpoint, measured: `docs/superpowers/plans/2026-09-20-qwen35moe-cuda.md`.
    const HIDDEN: u32 = 2048;
    const INTER: u32 = 512;
    const EXPERTS: u32 = 256;
    const USED: u32 = 8;
    /// The engine's widest prefill chunk, `stack::MAX_PREFILL_CHUNK`.
    const MAX_CHUNK: u32 = 512;

    #[test]
    fn the_router_s_two_fixed_bounds_hold_at_these_widths() {
        // `q4e_moe_route_kernel` stages the whole logit row in dynamic shared
        // memory, one float per expert, and the host passes `n_expert * 4`.
        assert_eq!(EXPERTS * 4, 1024, "the shared request, in bytes");
        // GB10 allows 48 KiB of dynamic shared memory per block without an
        // opt-in, and this is a fortieth of that.
        const { assert!(EXPERTS * 4 <= 48 * 1024) };

        // Its `chosen_w` and `chosen_i` are `[16]`, fixed. The sparse family
        // reaches 10; this one reaches 8. A checkpoint past 16 would overrun
        // two shared arrays with no diagnostic at all.
        const { assert!(USED <= 16, "the router's chosen_* arrays are [16]") };

        // One thread per expert with 256 threads: the kernel's loops stride by
        // `blockDim.x`, so any expert count works, but at exactly 256 each
        // thread holds one and the block argmax is one round.
        assert_eq!(EXPERTS, BLOCK_LINEAR);
    }

    #[test]
    fn the_expert_stages_cover_every_output_row_at_these_widths() {
        // Each block covers `EXPERT_ROWS` output rows, one per warp, and the
        // host rounds up. 512 gate rows is a whole number of blocks; 2048 down
        // rows likewise - but the kernel guards `row >= out_dim` anyway,
        // because a family whose inner width is not a multiple of four is a
        // config change away.
        assert_eq!(INTER.div_ceil(EXPERT_ROWS), 128);
        assert_eq!(HIDDEN.div_ceil(EXPERT_ROWS), 512);
        assert_eq!(INTER % EXPERT_ROWS, 0);
    }

    #[test]
    fn the_active_expert_extent_is_bounded_by_the_columns_a_chunk_has() {
        // `blockIdx.y` indexes the experts that received a column, and the
        // host sizes that extent as `min(columns, n_expert)`: a chunk cannot
        // activate more experts than it has columns. At one token that is 8
        // blocks rather than 256, which is the difference the sparse family
        // measured at half its kernel's time.
        for tokens in [1u32, 2, 8, 32, MAX_CHUNK] {
            let columns = tokens * USED;
            let extent = columns.min(EXPERTS);
            assert!(extent <= EXPERTS);
            assert!(extent <= columns);
        }
        assert_eq!(
            USED.min(EXPERTS),
            8,
            "a decode step, whose one token routes to eight"
        );
        assert_eq!((MAX_CHUNK * USED).min(EXPERTS), 256, "the widest chunk");
    }

    #[test]
    fn the_routed_column_count_fits_every_width_it_is_passed_as() {
        // The sort passes take the column count as a `u32` and the activation
        // takes an element count as a `u64`. The widest chunk is 4,096 columns
        // and 2,097,152 activation elements.
        let columns = u64::from(MAX_CHUNK) * u64::from(USED);
        assert_eq!(columns, 4_096);
        assert!(u32::try_from(columns).is_ok());
        assert_eq!(columns * u64::from(INTER), 2_097_152);
    }

    #[test]
    fn an_expert_slab_at_these_widths_fits_the_offsets_the_entries_compute() {
        // The entries index a slab as `expert * out_dim * (in_dim / 2)`, in
        // `int64_t`. The largest of the three is the down projection's:
        // 256 experts of 2048 rows of 256 packed bytes.
        let down = u64::from(EXPERTS) * u64::from(HIDDEN) * u64::from(INTER / 2);
        assert_eq!(down, 134_217_728, "128 MiB of packed bytes a layer");
        // Comfortably inside `i64`, and past `i32` - which is why the offsets
        // in those entries are computed in `int64_t` rather than promoted from
        // an `int` product that would have wrapped at the ninth expert.
        assert!(i64::try_from(down).is_ok());
        assert!(u32::try_from(down).is_ok());
        let all_three = down * 2 + u64::from(EXPERTS) * u64::from(INTER) * u64::from(HIDDEN / 2);
        assert!(all_three < u64::from(u32::MAX) * 2, "{all_three}");
    }
}
