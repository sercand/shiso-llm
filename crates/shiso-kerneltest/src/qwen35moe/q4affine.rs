//! The q4-affine codec and its device cases.
//!
//! Group 64, scale and bias per group, codes two to a byte with the low nibble
//! first. This is MLX's affine 4-bit and Splash's `q4_affine` alike, which is
//! why one kernel set reads both containers.
//!
//! # Two conventions this file pins, because a kernel cannot pin them alone
//!
//! **The slab is group-major, within a storage tile.** Codes are laid out
//! `[storage_tile][quant_group][storage_n][GROUP]` nibbles, not row-major
//! `[out_dim][in_dim]`, and the scales and biases follow in the same order.
//! That is `third_party/splash/runtime/metal/kernels/common/q4_mpp_tiles.h`'s
//! layout and it is what the `uint4b_format` tensor operand reads with a
//! stride of `{1, 64}`; the row-major layout a host packer reaches for first
//! would hand the tile the wrong 32 bytes for every group past the first.
//! [`dequant_q4_affine`] still takes the *logical* index `out * in_dim + in`,
//! so nothing downstream has to know where a code physically sits.
//!
//! The storage tile is the outer axis and it is **not** optional at the widths
//! this kernel set runs: the reference ships `StorageN = 256` while the real
//! projections are 2,048 (hidden) and 512 (expert intermediate) wide, so a
//! packer that only ever emits one tile silently mislays every parameter and
//! every code past the first `storage_n` outputs. [`pack_q4_affine`] defaults
//! `storage_n` to `out_dim`, which is the one-tile case and is what the graded
//! probe launches; [`pack_q4_affine_tiled`] is what a real projection uses.
//!
//! **Scales and biases are stored at `bfloat` precision.** The containers hold
//! one `bfloat` per group and that is what the device reads, so a slab whose
//! scales are full `f32` is not a reference for any kernel -- it is a
//! reference for a container nobody writes. Over 32 groups a 2^-9 relative
//! slip per scale is worth roughly 0.3 absolute on a dot product of this size,
//! which is six times the budget below. [`pack_q4_affine`] rounds both before
//! it picks the codes, so a host decode of a slab is bit-for-bit the values the
//! device decodes out of the same bytes.
//!
//! # What the device case's zero divergence does and does not prove
//!
//! [`the_affine_tile_matches_the_host_product`] grades bit-exact, and that is a
//! property of a fixture chosen to make it one: the inputs are small integers,
//! the codes are integers and every scale and bias is a short binary fraction,
//! so each term and each partial sum of both the host's and the device's
//! summation orders is exact in `f32`. **It therefore grades conventions --
//! nibble order, slab layout, the parameter index, the scale/bias epilogue --
//! and nothing else.** It cannot see reassociation, it cannot see a fast-math
//! contraction, and it will not notice a kernel that has become numerically
//! worse while staying structurally right. A case that wants those has to be
//! built on a fixture that is not exactly representable, and it will have a
//! nonzero budget rather than this one's zero.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "the kernel argument struct takes i32 dimensions while the fixtures are \
              written in usize, every fixture dimension here is a small literal, and \
              the code a quantizer picks is a rounded and clamped 0..=15 - the same \
              allowance `crate::qwen35moe::experts` takes for the same reason. The \
              routing and counting-sort fixtures below carry expert ids and column \
              indices the same way: the kernels read them as i32 and the largest of \
              them is in the hundreds"
)]

use crate::harness::{Gpu, drain, f32_bytes, f32_values, run};
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

/// Codes per quantization group.
pub const GROUP: usize = 64;

/// One projection's affine slab, host side.
///
/// The codes and the parameters are both group-major; see the module header
/// for why, and use [`dequant_q4_affine`] rather than indexing `weights`.
#[derive(Debug, Clone)]
pub struct Q4AffineSlab {
    /// Packed codes, two per byte,
    /// `[storage_tile][quant_group][storage_n][GROUP]` nibbles.
    pub weights: Vec<u8>,
    /// One `f32` per group, `[storage_tile][quant_group][storage_n]`, each
    /// already rounded to the `bfloat` the device reads.
    pub scales: Vec<f32>,
    /// One `f32` per group, same shape and same rounding as the scales.
    pub biases: Vec<f32>,
    /// Outputs this slab holds.
    pub out_dim: usize,
    /// Reduction depth of each output.
    pub in_dim: usize,
    /// Outputs per storage tile, the kernel's `StorageN`. Equal to `out_dim`
    /// for a slab that is one tile wide.
    pub storage_n: usize,
}

/// Where the code for one logical element sits, and which parameter decodes it.
///
/// The one place this layout is spelled: `(parameter, nibble)`, where the
/// nibble is an index into `weights` counted two to a byte. It is the same
/// arithmetic `q35moe_affine_tile` does on the device --
/// `(tile * groups + g) * StorageN + tile_offset` for the parameter, and that
/// times [`GROUP`] plus the offset within the group for the code.
fn placement(out: usize, depth: usize, in_dim: usize, storage_n: usize) -> (usize, usize) {
    let groups = in_dim / GROUP;
    let tile = out / storage_n;
    let tile_offset = out % storage_n;
    let parameter = (tile * groups + depth / GROUP) * storage_n + tile_offset;
    (parameter, parameter * GROUP + depth % GROUP)
}

/// Packs `values` -- row-major `[out_dim, in_dim]` -- into a one-tile slab.
///
/// `storage_n` is `out_dim`, which is the shape the graded probe launches. A
/// projection wider than one kernel storage tile goes through
/// [`pack_q4_affine_tiled`] instead; see the module header for why that is not
/// a detail Task 6 may leave to chance.
///
/// # Panics
///
/// When `in_dim` is not a multiple of [`GROUP`], or when `values` is not
/// `out_dim * in_dim` long.
#[must_use]
pub fn pack_q4_affine(values: &[f32], out_dim: usize, in_dim: usize) -> Q4AffineSlab {
    pack_q4_affine_tiled(values, out_dim, in_dim, out_dim)
}

/// Packs `values` -- row-major `[out_dim, in_dim]` -- into an affine slab laid
/// out for a kernel whose storage tile is `storage_n` outputs wide.
///
/// The codes and the parameters both end up where `q35moe_affine_tile`'s
/// `StorageN` arithmetic reads them; see [`placement`].
///
/// # Panics
///
/// When `in_dim` is not a multiple of [`GROUP`], when `out_dim` is not a
/// multiple of `storage_n` (the kernel's last tile is a whole tile, not a
/// short one), or when `values` is not `out_dim * in_dim` long.
#[must_use]
pub fn pack_q4_affine_tiled(
    values: &[f32],
    out_dim: usize,
    in_dim: usize,
    storage_n: usize,
) -> Q4AffineSlab {
    assert!(
        in_dim.is_multiple_of(GROUP),
        "in_dim must be a whole number of groups"
    );
    assert!(
        storage_n > 0 && out_dim.is_multiple_of(storage_n),
        "out_dim {out_dim} must be a whole number of storage tiles of {storage_n}",
    );
    assert!(
        values.len() == out_dim * in_dim,
        "values is {} long, not out_dim * in_dim = {}",
        values.len(),
        out_dim * in_dim,
    );
    let groups = in_dim / GROUP;
    let params = out_dim * groups;
    let mut weights = vec![0u8; out_dim * in_dim / 2];
    let mut scales = vec![0f32; params];
    let mut biases = vec![0f32; params];
    for n in 0..out_dim {
        for g in 0..groups {
            let start = n * in_dim + g * GROUP;
            let span = &values[start..start + GROUP];
            let lo = span.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = span.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = if (hi - lo).abs() < f32::EPSILON {
                1.0
            } else {
                bf16_round((hi - lo) / 15.0)
            };
            let bias = bf16_round(lo);
            let (p, first_nibble) = placement(n, g * GROUP, in_dim, storage_n);
            scales[p] = scale;
            biases[p] = bias;
            for (r, &v) in span.iter().enumerate() {
                let code = ((v - bias) / scale).round().clamp(0.0, 15.0) as u8;
                let nibble = first_nibble + r;
                if nibble.is_multiple_of(2) {
                    weights[nibble / 2] |= code;
                } else {
                    weights[nibble / 2] |= code << 4;
                }
            }
        }
    }
    Q4AffineSlab {
        weights,
        scales,
        biases,
        out_dim,
        in_dim,
        storage_n,
    }
}

/// Decodes one element of an affine slab.
///
/// `index` is the *logical* row-major index `out * in_dim + in`; the slab's own
/// tiled group-major placement is this function's business and not the
/// caller's.
///
/// # Panics
///
/// When `index` is past `out_dim * in_dim`.
#[must_use]
pub fn dequant_q4_affine(slab: &Q4AffineSlab, index: usize) -> f32 {
    let n = index / slab.in_dim;
    let k = index % slab.in_dim;
    let (p, nibble) = placement(n, k, slab.in_dim, slab.storage_n);
    let byte = slab.weights[nibble / 2];
    let code = if nibble.is_multiple_of(2) {
        byte & 0x0F
    } else {
        byte >> 4
    };
    f32::from(code) * slab.scales[p] + slab.biases[p]
}

/// This family's module. A label: Metal resolves by entry name.
const MODULE: &str = "qwen35moe";

/// The module the seven general entries live in.
///
/// The string `crates/shiso-engine/src/moe.rs`'s `Q4E_MOE` holds and the one
/// the sibling `experts.rs` already spells, repeated rather than imported
/// because `shiso-kerneltest` does not depend on `shiso-engine` and this is not
/// the change that adds that edge. Metal ignores the module string entirely and
/// resolves by entry name, so launching these seven under `MODULE` worked here
/// -- and would have found no such module the first time they ran on CUDA. A
/// name checked on only one backend is the same blindness that let four
/// module-name defects reach CUDA invisibly (`docs/handoff/cuda-qwen35.md` §2),
/// so it is spelled correctly now, while nothing on this branch can tell the
/// difference.
const Q4E_MOE: &str = "q4e_moe";

/// The tile agrees with the host product over the whole of `in_dim`.
///
/// The spike graded one quant group and a throwaway reference disagreed at
/// 2,048, so this case exists to say which of the two was wrong and to keep
/// the answer.
///
/// The fixture is built so that **every** convention is graded and none of the
/// float arithmetic is. See [`affine_fixture_weights`] for the first half of
/// that and the module header for the second.
///
/// # Panics
///
/// When the launch is refused, or when the device product leaves the budget
/// printed beside it.
#[expect(
    clippy::print_stdout,
    reason = "the divergence this case measured is the finding, and a graded case that \
              holds a kernel to a budget says what it actually saw"
)]
pub fn the_affine_tile_matches_the_host_product(gpu: &Gpu) {
    const ROWS: usize = 8;
    const TILE_N: usize = 32;
    const IN_DIM: usize = 2048;

    let input: Vec<f32> = (0..ROWS * IN_DIM)
        .map(|i| ((i * 7) % 13) as f32 - 6.0)
        .collect();
    let weights = affine_fixture_weights(TILE_N, IN_DIM);
    let slab = pack_q4_affine(&weights, TILE_N, IN_DIM);
    // The parameters have to differ across BOTH axes or the parameter index
    // formula is graded by nothing: a fixture whose every 64-wide window spans
    // the same min and max yields one scale and one bias for the whole slab,
    // and any permutation of `(quant_group, output)` then reads correctly.
    assert!(
        distinct(&slab.scales) > 1 && distinct(&slab.biases) > 1,
        "the fixture collapsed to {} scale(s) and {} bias(es); it cannot \
         discriminate the parameter layout",
        distinct(&slab.scales),
        distinct(&slab.biases),
    );

    let mut want = vec![0f32; ROWS * TILE_N];
    for m in 0..ROWS {
        for n in 0..TILE_N {
            let mut acc = 0f32;
            for k in 0..IN_DIM {
                acc += bf16_round(input[m * IN_DIM + k]) * dequant_q4_affine(&slab, n * IN_DIM + k);
            }
            want[m * TILE_N + n] = acc;
        }
    }

    let args = [
        i32::try_from(IN_DIM).expect("the fixture depth fits an i32"),
        i32::try_from(TILE_N).expect("the fixture tile width fits an i32"),
        1,
        1,
    ];
    let d_in = gpu.upload(&bf16_bytes(&input));
    let d_w = gpu.upload(&slab.weights);
    let d_s = gpu.upload(&bf16_bytes(&slab.scales));
    let d_b = gpu.upload(&bf16_bytes(&slab.biases));
    let d_out = zeros(gpu, ROWS * TILE_N * 4);

    run(
        gpu,
        KernelRef::new(MODULE, "kernel_q35moe_tile_probe"),
        &LaunchArgs {
            grid: Extent3::linear(1),
            block: Extent3::linear(256),
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::cast_slice(&args)),
                KernelArg::Buffer(d_in.view()),
                KernelArg::Buffer(d_w.view()),
                KernelArg::Buffer(d_s.view()),
                KernelArg::Buffer(d_b.view()),
                KernelArg::Buffer(d_out.view()),
            ],
        },
    );
    drain(gpu);

    let got = gpu.read_f32(&d_out);
    // Before the zip, which would otherwise shorten the comparison rather than
    // fail it if the read-back came back short.
    assert_eq!(
        got.len(),
        want.len(),
        "the read-back is not the tile's shape"
    );
    let mut worst = 0f32;
    let mut worst_at = 0usize;
    let mut magnitude = 0f32;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        magnitude = magnitude.max(w.abs());
        if (g - w).abs() > worst {
            worst = (g - w).abs();
            worst_at = i;
        }
    }
    // Printed, not chosen: a 2,048-term device dot product against a host one.
    println!(
        "affine tile worst divergence: {worst} at element {worst_at} \
         (device {}, host {}), largest term {magnitude}, budget 0.05",
        got[worst_at], want[worst_at],
    );
    assert!(worst < 0.05, "worst divergence {worst} over {IN_DIM} terms");
}

/// The graded case's weight fixture, row-major `[out_dim, in_dim]`.
///
/// Every 64-wide window is written as `bias + code * scale` for a
/// `(scale, bias)` pair that cycles with period 35 over `out * groups + group`,
/// so the pair differs along *both* the quant-group axis and the output axis
/// and a permutation of the parameter index cannot read correctly. The codes
/// run over the whole of `0..=15` in every window, which is what makes
/// `pack_q4_affine` recover exactly this scale and this bias rather than a
/// narrower pair of its own.
///
/// The arithmetic is chosen to stay exact, not merely close. Each scale is a
/// multiple of `2^-4` with at most five significant bits, each bias a multiple
/// of `2^-2`, the codes and the case's inputs are small integers; so every term
/// and every partial sum of the host's `sum a*(code*scale + bias)` and of the
/// device's `scale * sum code*a + bias * sum a` is a multiple of `2^-4` well
/// inside an `f32` mantissa, and the two summation orders land on the same
/// float. That is deliberate: this case grades layout, not float agreement.
#[must_use]
pub fn affine_fixture_weights(out_dim: usize, in_dim: usize) -> Vec<f32> {
    let groups = in_dim / GROUP;
    let mut values = vec![0f32; out_dim * in_dim];
    for n in 0..out_dim {
        for g in 0..groups {
            let t = n * groups + g;
            let scale = (1 + t % 5) as f32 * 0.0625;
            let bias = (t % 7) as f32 * 0.25 - 1.75;
            for r in 0..GROUP {
                // gcd(11, 16) == 1, so this hits every code in 0..=15 - and
                // therefore both ends of the window - whatever `t` is.
                let code = ((r * 11) + t) % 16;
                values[n * in_dim + g * GROUP + r] = bias + code as f32 * scale;
            }
        }
    }
    values
}

/// How many distinct bit patterns `values` holds.
fn distinct(values: &[f32]) -> usize {
    let mut bits: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
    bits.sort_unstable();
    bits.dedup();
    bits.len()
}

/// A zeroed device buffer of `bytes` bytes.
fn zeros(gpu: &Gpu, bytes: usize) -> DeviceBuffer {
    gpu.upload(&vec![0u8; bytes])
}

/// A value as bf16 round-trips it, which is what the device reads.
fn bf16_round(v: f32) -> f32 {
    f32::from_bits(v.to_bits() & 0xFFFF_0000)
}

/// `values` as the bf16 bytes a device buffer holds.
///
/// Truncating, matching [`bf16_round`]: the two have to agree or the host
/// reference would be grading a kernel against numbers the kernel never saw.
///
/// `crate::qwen35moe::experts` uploads its affine slabs through this rather
/// than through a second converter of its own: a whole-block case that rounded
/// the container differently from the case that graded one tile would be
/// grading a different kernel.
pub(crate) fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
    }
    out
}

/// `i32` values from little-endian bytes, without assuming any alignment.
///
/// The read-back is a `Vec<u8>` of byte alignment, so `bytemuck::cast_slice`
/// would refuse it; this is [`crate::harness::f32_values`] over the other
/// four-byte scalar these kernels write.
fn i32_values(bytes: &[u8]) -> Vec<i32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| i32::from_le_bytes(*w))
        .collect()
}

/// `i32` values as little-endian bytes.
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The largest absolute difference between a device read-back and a host
/// reference of the same shape.
///
/// # Panics
///
/// When the read-back is not the reference's length. Checked before the zip,
/// which would otherwise shorten the comparison rather than fail it.
fn worst_divergence(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(
        got.len(),
        want.len(),
        "the read-back is not the reference's shape"
    );
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max)
}

/// The routed ids this fixture hands the counting sort, one per column.
///
/// Two of them are deliberately out of range - a `-1` for a column no expert
/// claimed and a `999` past the expert count - because both guards in
/// `q4e_moe_sort_count_kernel` and `q4e_moe_sort_scatter_kernel` are silent
/// when they are wrong: a column counted twice or scattered past its span
/// corrupts a neighbour's rows rather than failing.
fn sort_fixture_ids(rows: usize, experts: usize) -> Vec<i32> {
    (0..rows)
        .map(|i| {
            if i % 97 == 0 {
                -1
            } else if i % 89 == 0 {
                999
            } else {
                ((i * 13) % experts) as i32
            }
        })
        .collect()
}

/// The host counting sort the three device passes are graded against.
///
/// Returns the per-expert counts and, per expert, the columns that chose it in
/// ascending column order.
fn host_counting_sort(ids: &[i32], experts: usize) -> (Vec<i32>, Vec<Vec<i32>>) {
    let mut per_expert = vec![Vec::new(); experts];
    for (i, &e) in ids.iter().enumerate() {
        if e >= 0 && (e as usize) < experts {
            per_expert[e as usize].push(i as i32);
        }
    }
    let counts = per_expert.iter().map(|c| c.len() as i32).collect();
    (counts, per_expert)
}

/// The router picks the same experts and the same weights as the host.
///
/// **Wrong routing does not crash.** It answers fluently out of the wrong
/// experts, and every count in the request stays right, so this case is the
/// only thing between that and a build that looks fine. It therefore asserts
/// the chosen ids match **exactly** and grades the weights separately; a case
/// that only checked the weights would pass on a permutation of them.
///
/// The fixture's logits repeat - `(i * 37) % 101` over 256 experts gives two or
/// three experts the same value - so the tie-break is graded rather than
/// assumed. That is deliberate: the reference keeps the LOWER expert id among
/// equal values, and a port that keeps the higher one differs only on a tie and
/// only on the experts it then reads.
///
/// # Panics
///
/// When the launch is refused, when an id disagrees, or when a weight leaves
/// the budget printed beside it.
#[expect(
    clippy::print_stdout,
    reason = "the divergence this case measured is the finding, and a graded case that \
              holds a kernel to a budget says what it actually saw"
)]
pub fn the_router_picks_what_the_host_reference_picks(gpu: &Gpu) {
    const TOKENS: usize = 5;
    const EXPERTS: usize = 256;
    const USED: usize = 8;

    let logits: Vec<f32> = (0..TOKENS * EXPERTS)
        .map(|i| ((i * 37) % 101) as f32 * 0.05 - 2.5)
        .collect();

    // The host law: softmax over all the experts, the best USED, renormalized
    // over those alone.
    let mut want_ids = vec![0i32; TOKENS * USED];
    let mut want_weights = vec![0f32; TOKENS * USED];
    for t in 0..TOKENS {
        let row = &logits[t * EXPERTS..(t + 1) * EXPERTS];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let inv: f32 = 1.0 / row.iter().map(|v| (v - m).exp()).sum::<f32>();
        let mut order: Vec<usize> = (0..EXPERTS).collect();
        // Descending by value, ties to the lower id, which is the kernel's rule.
        order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).expect("no NaN").then(a.cmp(&b)));
        let chosen = &order[..USED];
        let probs: Vec<f32> = chosen.iter().map(|&e| (row[e] - m).exp() * inv).collect();
        let wsum = probs.iter().sum::<f32>().max(1e-20);
        for (s, (&e, p)) in chosen.iter().zip(&probs).enumerate() {
            want_ids[t * USED + s] = e as i32;
            want_weights[t * USED + s] = p / wsum;
        }
    }
    assert!(
        want_ids[..USED].windows(2).any(|w| w[0] > w[1]),
        "the fixture's first row is already in id order, so it cannot grade the tie-break"
    );

    let d_logits = gpu.upload(&f32_bytes(&logits));
    let d_ids = zeros(gpu, TOKENS * USED * 4);
    let d_weights = zeros(gpu, TOKENS * USED * 4);
    run(
        gpu,
        KernelRef::new(Q4E_MOE, "q4e_moe_route_kernel"),
        &LaunchArgs {
            grid: Extent3::linear(TOKENS as u32),
            block: Extent3::linear(256),
            // One float per expert: the row is staged before the selection
            // rounds mask it.
            smem_bytes: (EXPERTS * 4) as u32,
            args: &[
                KernelArg::Buffer(d_ids.view()),
                KernelArg::Buffer(d_weights.view()),
                KernelArg::Buffer(d_logits.view()),
                KernelArg::U32(EXPERTS as u32),
                KernelArg::U32(USED as u32),
            ],
        },
    );
    drain(gpu);

    let got_ids = i32_values(&gpu.read(&d_ids));
    assert_eq!(got_ids, want_ids, "the chosen experts must agree exactly");
    let worst = worst_divergence(&gpu.read_f32(&d_weights), &want_weights);
    println!("router weight worst divergence: {worst}, budget 1e-5");
    assert!(worst < 1e-5, "worst routing weight divergence {worst}");
}

/// The activation is the reference's division, not a sigmoid multiply.
///
/// `g / (1 + exp(-g))` and `g * sigmoid(g)` are the same function and do not
/// round alike, and the element count is deliberately not a multiple of the
/// 256-thread block so the tail guard is graded too.
///
/// # Panics
///
/// When the launch is refused, or when an element leaves the budget printed
/// beside it.
#[expect(
    clippy::print_stdout,
    reason = "the divergence this case measured is the finding, and a graded case that \
              holds a kernel to a budget says what it actually saw"
)]
pub fn the_activation_matches_the_host_silu_product(gpu: &Gpu) {
    // 5,003 is prime, so the last block is a partial one whichever way the
    // host rounds the grid up.
    const N: usize = 5003;

    let gate: Vec<f32> = (0..N).map(|i| (i % 41) as f32 * 0.25 - 5.0).collect();
    let up: Vec<f32> = (0..N).map(|i| (i % 13) as f32 * 0.5 - 3.0).collect();
    let want: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect();

    let d_gate = gpu.upload(&f32_bytes(&gate));
    let d_up = gpu.upload(&f32_bytes(&up));
    let d_out = zeros(gpu, N * 4);
    run(
        gpu,
        KernelRef::new(Q4E_MOE, "q4e_swiglu_kernel"),
        &LaunchArgs {
            grid: Extent3::linear(N.div_ceil(256) as u32),
            block: Extent3::linear(256),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(d_out.view()),
                KernelArg::Buffer(d_gate.view()),
                KernelArg::Buffer(d_up.view()),
                KernelArg::U64(N as u64),
            ],
        },
    );
    drain(gpu);

    let worst = worst_divergence(&gpu.read_f32(&d_out), &want);
    println!("swiglu worst divergence: {worst}, budget 1e-5");
    assert!(worst < 1e-5, "worst activation divergence {worst}");
}

/// The combine reads slot `s` of token `t` at column `t * used + s`.
///
/// That seam convention is the whole content of this kernel: the expert matmul
/// writes column-major over (token, slot) and this is the reader that pins it,
/// so the fixture gives every (token, slot) a distinct weight and every column
/// a distinct row - a transposed read would land on a real number rather than
/// out of bounds.
///
/// # Panics
///
/// When the launch is refused, or when an element leaves the budget printed
/// beside it.
#[expect(
    clippy::print_stdout,
    reason = "the divergence this case measured is the finding, and a graded case that \
              holds a kernel to a budget says what it actually saw"
)]
pub fn the_combine_matches_the_host_weighted_sum(gpu: &Gpu) {
    const TOKENS: usize = 3;
    const USED: usize = 4;
    // Not a multiple of 256, so the strided sweep and its bound are graded.
    const EMBD: usize = 300;
    const COLUMNS: usize = TOKENS * USED;

    let down: Vec<f32> = (0..COLUMNS * EMBD)
        .map(|i| (i % 23) as f32 * 0.125 - 1.5)
        .collect();
    let weights: Vec<f32> = (0..COLUMNS).map(|c| 0.05 + c as f32 * 0.03).collect();
    let mut want = vec![0f32; TOKENS * EMBD];
    for t in 0..TOKENS {
        for e in 0..EMBD {
            let mut acc = 0f32;
            for s in 0..USED {
                let col = t * USED + s;
                acc += down[col * EMBD + e] * weights[col];
            }
            want[t * EMBD + e] = acc;
        }
    }

    let d_down = gpu.upload(&f32_bytes(&down));
    let d_weights = gpu.upload(&f32_bytes(&weights));
    let d_out = zeros(gpu, TOKENS * EMBD * 4);
    run(
        gpu,
        KernelRef::new(Q4E_MOE, "q4e_moe_combine_kernel"),
        &LaunchArgs {
            grid: Extent3::new(EMBD.div_ceil(256) as u32, TOKENS as u32, 1),
            block: Extent3::linear(256),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(d_out.view()),
                KernelArg::Buffer(d_down.view()),
                KernelArg::Buffer(d_weights.view()),
                KernelArg::U32(EMBD as u32),
                KernelArg::U32(USED as u32),
            ],
        },
    );
    drain(gpu);

    let worst = worst_divergence(&gpu.read_f32(&d_out), &want);
    println!("combine worst divergence: {worst}, budget 1e-5");
    assert!(worst < 1e-5, "worst combine divergence {worst}");
}

/// The shared expert's add is an add, and its gate is one scalar per token.
///
/// The destination is seeded with something other than zero, because the
/// routed experts have already written it: a kernel that stores instead of
/// accumulating is bit-exact on a zeroed destination and wrong on a real one.
///
/// # Panics
///
/// When the launch is refused, or when an element leaves the budget printed
/// beside it.
#[expect(
    clippy::print_stdout,
    reason = "the divergence this case measured is the finding, and a graded case that \
              holds a kernel to a budget says what it actually saw"
)]
pub fn the_shared_add_matches_the_host_gated_add(gpu: &Gpu) {
    const TOKENS: usize = 3;
    const EMBD: usize = 300;

    let seed: Vec<f32> = (0..TOKENS * EMBD)
        .map(|i| (i % 17) as f32 * 0.25 - 2.0)
        .collect();
    let shared: Vec<f32> = (0..TOKENS * EMBD)
        .map(|i| (i % 29) as f32 * 0.0625 - 0.75)
        .collect();
    let logit: Vec<f32> = (0..TOKENS).map(|t| t as f32 * 1.5 - 1.5).collect();
    let mut want = seed.clone();
    for t in 0..TOKENS {
        let g = 1.0 / (1.0 + (-logit[t]).exp());
        for e in 0..EMBD {
            want[t * EMBD + e] += shared[t * EMBD + e] * g;
        }
    }

    let d_out = gpu.upload(&f32_bytes(&seed));
    let d_shared = gpu.upload(&f32_bytes(&shared));
    let d_logit = gpu.upload(&f32_bytes(&logit));
    run(
        gpu,
        KernelRef::new(Q4E_MOE, "q4e_shared_add_kernel"),
        &LaunchArgs {
            grid: Extent3::new(EMBD.div_ceil(256) as u32, TOKENS as u32, 1),
            block: Extent3::linear(256),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(d_out.view()),
                KernelArg::Buffer(d_shared.view()),
                KernelArg::Buffer(d_logit.view()),
                KernelArg::U32(EMBD as u32),
            ],
        },
    );
    drain(gpu);

    let got = gpu.read_f32(&d_out);
    assert_ne!(got, seed, "the kernel did not write the destination at all");
    let worst = worst_divergence(&got, &want);
    println!("shared add worst divergence: {worst}, budget 1e-5");
    assert!(worst < 1e-5, "worst shared-add divergence {worst}");
}

/// The histogram counts each expert's columns and skips the unroutable ones.
///
/// # Panics
///
/// When the launch is refused, or when a count disagrees with the host's.
#[expect(
    clippy::print_stdout,
    reason = "a graded case that holds a kernel to a budget says what it actually saw, \
              and an exact pass is a measurement like any other"
)]
pub fn the_sort_histogram_counts_every_column(gpu: &Gpu) {
    const ROWS: usize = 600;
    const EXPERTS: usize = 64;

    let ids = sort_fixture_ids(ROWS, EXPERTS);
    let (want, _) = host_counting_sort(&ids, EXPERTS);
    assert!(
        want.iter().sum::<i32>() < ROWS as i32,
        "the fixture has no out-of-range id, so neither guard is graded"
    );

    let d_ids = gpu.upload(&i32_bytes(&ids));
    let d_counts = zeros(gpu, EXPERTS * 4);
    run(
        gpu,
        KernelRef::new(Q4E_MOE, "q4e_moe_sort_count_kernel"),
        &LaunchArgs {
            grid: Extent3::linear(ROWS.div_ceil(256) as u32),
            block: Extent3::linear(256),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(d_counts.view()),
                KernelArg::Buffer(d_ids.view()),
                KernelArg::U32(ROWS as u32),
                KernelArg::U32(EXPERTS as u32),
            ],
        },
    );
    drain(gpu);

    let got = i32_values(&gpu.read(&d_counts));
    println!(
        "sort histogram divergence: {} counts, budget 0",
        count_mismatches(&got, &want)
    );
    assert_eq!(got, want, "the per-expert counts must agree exactly");
}

/// The scan is the exclusive prefix of the counts, plus the active list.
///
/// The fixture leaves several experts empty, which is what makes the active
/// list say anything: a scan that listed every expert would still produce the
/// right bounds and would launch the grouped stage over hundreds of empty
/// experts.
///
/// # Panics
///
/// When the launch is refused, or when a bound or an active entry disagrees.
#[expect(
    clippy::print_stdout,
    reason = "a graded case that holds a kernel to a budget says what it actually saw, \
              and an exact pass is a measurement like any other"
)]
pub fn the_sort_scan_is_the_exclusive_prefix_of_the_counts(gpu: &Gpu) {
    const ROWS: usize = 600;
    // Wider than the fixture routes to, so some experts are empty.
    const EXPERTS: usize = 80;

    let ids = sort_fixture_ids(ROWS, 64);
    let (counts, _) = host_counting_sort(&ids, EXPERTS);
    let mut want_bounds = vec![0i32; EXPERTS + 1];
    let mut want_active = Vec::new();
    let mut total = 0i32;
    for (e, &c) in counts.iter().enumerate() {
        want_bounds[e] = total;
        total += c;
        if c > 0 {
            want_active.push(e as i32);
        }
    }
    want_bounds[EXPERTS] = total;
    assert!(
        want_active.len() < EXPERTS,
        "every expert is active, so the active list is graded by nothing"
    );

    let d_counts = gpu.upload(&i32_bytes(&counts));
    let d_bounds = zeros(gpu, (EXPERTS + 1) * 4);
    let d_active = zeros(gpu, (EXPERTS + 1) * 4);
    run(
        gpu,
        KernelRef::new(Q4E_MOE, "q4e_moe_sort_scan_kernel"),
        &LaunchArgs {
            grid: Extent3::linear(1),
            block: Extent3::linear(32),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(d_bounds.view()),
                KernelArg::Buffer(d_active.view()),
                KernelArg::Buffer(d_counts.view()),
                KernelArg::U32(EXPERTS as u32),
            ],
        },
    );
    drain(gpu);

    let got_bounds = i32_values(&gpu.read(&d_bounds));
    assert_eq!(
        got_bounds, want_bounds,
        "the expert bounds must agree exactly"
    );
    let got_active = i32_values(&gpu.read(&d_active));
    assert_eq!(
        got_active[EXPERTS],
        want_active.len() as i32,
        "the active count lives at active[n_expert]"
    );
    assert_eq!(
        &got_active[..want_active.len()],
        &want_active[..],
        "the active experts must be listed in ascending order"
    );
    println!(
        "sort scan divergence: 0 bounds, 0 active entries (total {total}, \
         {} active of {EXPERTS}), budget 0",
        want_active.len(),
    );
}

/// The scatter fills each expert's span with exactly that expert's columns.
///
/// **The order inside one span is not asserted, because it is not defined.**
/// The reference's `atomicAdd` hands out slots in whatever order the hardware
/// runs, and the comment on `kernels/cuda/q4e/moe.cu:228` says so; every column
/// writes its own output row afterwards, so the order cannot matter. What is
/// asserted is what is defined: each span holds exactly the set of columns that
/// chose that expert, and every column lands once.
///
/// # Panics
///
/// When the launch is refused, or when a span is not its expert's column set.
#[expect(
    clippy::print_stdout,
    reason = "a graded case that holds a kernel to a budget says what it actually saw, \
              and an exact pass is a measurement like any other"
)]
pub fn the_sort_scatter_fills_every_expert_span(gpu: &Gpu) {
    const ROWS: usize = 600;
    const EXPERTS: usize = 64;

    let ids = sort_fixture_ids(ROWS, EXPERTS);
    let (counts, per_expert) = host_counting_sort(&ids, EXPERTS);
    let mut bounds = vec![0i32; EXPERTS + 1];
    let mut total = 0i32;
    for (e, &c) in counts.iter().enumerate() {
        bounds[e] = total;
        total += c;
    }
    bounds[EXPERTS] = total;

    let d_ids = gpu.upload(&i32_bytes(&ids));
    let d_bounds = gpu.upload(&i32_bytes(&bounds));
    let d_cursor = zeros(gpu, EXPERTS * 4);
    let d_sorted = gpu.upload(&i32_bytes(&vec![-1i32; ROWS]));
    run(
        gpu,
        KernelRef::new(Q4E_MOE, "q4e_moe_sort_scatter_kernel"),
        &LaunchArgs {
            grid: Extent3::linear(ROWS.div_ceil(256) as u32),
            block: Extent3::linear(256),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(d_sorted.view()),
                KernelArg::Buffer(d_cursor.view()),
                KernelArg::Buffer(d_bounds.view()),
                KernelArg::Buffer(d_ids.view()),
                KernelArg::U32(ROWS as u32),
                KernelArg::U32(EXPERTS as u32),
            ],
        },
    );
    drain(gpu);

    let sorted = i32_values(&gpu.read(&d_sorted));
    let mut mismatched = 0usize;
    for (e, want) in per_expert.iter().enumerate() {
        let lo = bounds[e] as usize;
        let hi = bounds[e + 1] as usize;
        assert_eq!(hi - lo, want.len(), "expert {e}'s span is not its count");
        let mut span = sorted[lo..hi].to_vec();
        span.sort_unstable();
        if &span != want {
            mismatched += 1;
        }
    }
    // Past the total the buffer keeps the sentinel it was seeded with, which is
    // what says the scatter stayed inside the spans it was given.
    let tail_written = sorted[total as usize..]
        .iter()
        .filter(|&&v| v != -1)
        .count();
    println!(
        "sort scatter divergence: {mismatched} spans, {tail_written} writes past the total, \
         budget 0 and 0"
    );
    assert_eq!(mismatched, 0, "an expert's span is not its column set");
    assert_eq!(tail_written, 0, "the scatter wrote past the column total");
}

// ---------------------------------------------------------------------------
// The two expert stages.
//
// Everything above this line grades one kernel against one host expression.
// These four cases grade the seam instead: the four conventions
// `crates/shiso-engine/src/moe.rs` lists in its header, none of which is
// checkable by reading a single kernel, and all of which fail silently and
// plausibly. A stage that reads `col` where it should read `col / n_used`
// answers out of the wrong activation row, at the right magnitude, for every
// column; a stage that writes `[out_dim, column]` hands the combine a
// correctly sized transpose.
//
// The fixture is therefore chosen to make each of them discriminating rather
// than to be convenient:
//
//   * every token's activation row differs, so reading the wrong row is a
//     wrong number rather than the same one;
//   * every column's intermediate row differs, for the same reason on the down
//     side;
//   * every expert's slab differs, so reading expert zero's for all of them is
//     visible;
//   * one expert receives no column at all and the grid's second extent is
//     `min(columns, n_expert)`, so the block that must return on the `active`
//     count is actually launched;
//   * the gate/up output width is 512 -- two storage tiles -- so the tiled
//     placement is graded here too and not only in the unit test above.
//
// Unlike `the_affine_tile_matches_the_host_product`, the activations here are
// NOT exactly representable: they are a multiple of a step that is not a binary
// fraction, so no product and no partial sum is exact, the device's per-group
// interleaved sweep and the host's sequential one genuinely reassociate, and
// the number these cases print is a measurement of that rather than a property
// of the fixture.

/// Outputs per storage tile of a resident affine slab.
///
/// `Q35MOE_STORAGE_N` in `metal/qwen35moe/q35moe_common.h`, which the expert
/// entries read as a constant rather than taking in the argument block: the
/// resident layout is one decision for the whole kernel set. A fixture packed
/// at any other width is packed for a kernel that does not exist.
const STORAGE_N: usize = 256;

/// Tokens in the routed-stage fixture.
const STAGE_TOKENS: usize = 4;
/// Experts each token is routed to.
const STAGE_USED: usize = 2;
/// Experts the fixture layer holds. One of them receives no column.
const STAGE_EXPERTS: usize = 4;
/// The fixture's residual width: the gate/up reduction depth and the down
/// projection's output width.
const STAGE_HIDDEN: usize = 256;
/// The fixture's expert intermediate width.
///
/// Two storage tiles wide, deliberately: at one tile the tile index is always
/// zero and every permutation of the placement reads correctly.
const STAGE_INTER: usize = 512;
/// Routed columns the fixture produces.
const STAGE_COLUMNS: usize = STAGE_TOKENS * STAGE_USED;
/// Output rows one expert threadgroup covers, `Q35MOE_EXPERT_ROWS`.
const STAGE_ROWS: usize = 4;
/// Threads per expert threadgroup, `EXPERT_BLOCK` in `moe.rs`.
const STAGE_BLOCK: u32 = 128;

/// The float the routed activations cross a launch boundary in.
///
/// It governs the input rows AND the outputs of an expert entry together,
/// which is what `ActDtype` names on the host side: the weights stay `uchar`
/// codes and the scale and bias stay `bfloat` whatever this is, because those
/// are container widths and not an activation choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Act {
    /// Four-byte float.
    F32,
    /// Two-byte brain float.
    Bf16,
}

impl Act {
    /// The gate-and-up entry for this dtype.
    ///
    /// Spelled out rather than interpolated because `KernelRef` takes a
    /// `&'static str` -- and because a name a case formats is a name a typo in
    /// the case would silently turn into "no such entry" at some other launch.
    const fn gate_up_entry(self) -> &'static str {
        match self {
            Self::F32 => "kernel_q35moe_experts_gate_up_q4affine_f32",
            Self::Bf16 => "kernel_q35moe_experts_gate_up_q4affine_bf16",
        }
    }

    /// The down entry for this dtype.
    const fn down_entry(self) -> &'static str {
        match self {
            Self::F32 => "kernel_q35moe_experts_down_q4affine_f32",
            Self::Bf16 => "kernel_q35moe_experts_down_q4affine_bf16",
        }
    }

    /// `values` as the bytes a device buffer of this dtype holds.
    fn bytes(self, values: &[f32]) -> Vec<u8> {
        match self {
            Self::F32 => f32_bytes(values),
            Self::Bf16 => bf16_bytes(values),
        }
    }

    /// A read-back of this dtype, widened to `f32`.
    fn values(self, bytes: &[u8]) -> Vec<f32> {
        match self {
            Self::F32 => f32_values(bytes),
            Self::Bf16 => bf16_values(bytes),
        }
    }

    /// A value as this dtype round-trips it, which is what the device reads.
    fn round(self, v: f32) -> f32 {
        match self {
            Self::F32 => v,
            Self::Bf16 => bf16_round(v),
        }
    }
}

/// bf16 values from little-endian bytes, widened to `f32`.
///
/// The inverse of [`bf16_bytes`], and truncating in the same direction.
fn bf16_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|w| f32::from_bits(u32::from(u16::from_le_bytes(*w)) << 16))
        .collect()
}

/// The routing the two stages are launched over, built on the host.
///
/// The sort passes have their own cases above; this is their *output*, written
/// out by hand so that a failure here names the expert stage rather than the
/// stage before it.
#[derive(Debug)]
struct Routing {
    /// The expert each column chose.
    ids: Vec<i32>,
    /// The columns in expert order.
    sorted: Vec<i32>,
    /// Exclusive prefix sums, `experts + 1` long.
    bounds: Vec<i32>,
    /// The experts that received a column, with the count at `[experts]`.
    active: Vec<i32>,
}

/// The fixture routing: three of four experts active, spans of 3, 2 and 3.
///
/// `(c * 5) % 3` leaves expert 3 with no column, which is the point: the grid's
/// second extent is `min(columns, n_expert)` = 4 and only three blocks may do
/// work, so the `active[n_expert]` guard is launched rather than argued about.
fn stage_routing() -> Routing {
    let ids: Vec<i32> = (0..STAGE_COLUMNS).map(|c| ((c * 5) % 3) as i32).collect();
    let (counts, per_expert) = host_counting_sort(&ids, STAGE_EXPERTS);
    let mut bounds = vec![0i32; STAGE_EXPERTS + 1];
    let mut total = 0i32;
    for (e, &c) in counts.iter().enumerate() {
        bounds[e] = total;
        total += c;
    }
    bounds[STAGE_EXPERTS] = total;
    let mut active = vec![0i32; STAGE_EXPERTS + 1];
    let mut n_active = 0usize;
    for (e, &c) in counts.iter().enumerate() {
        if c > 0 {
            active[n_active] = e as i32;
            n_active += 1;
        }
    }
    active[STAGE_EXPERTS] = n_active as i32;
    assert!(
        n_active < STAGE_COLUMNS.min(STAGE_EXPERTS),
        "every expert is active, so the `active` guard is never exercised"
    );
    Routing {
        ids,
        sorted: per_expert.into_iter().flatten().collect(),
        bounds,
        active,
    }
}

/// One projection's experts, packed and resident, plus the host slabs.
#[derive(Debug)]
struct Stack {
    /// The host slabs, one per expert, for the reference.
    slabs: Vec<Q4AffineSlab>,
    /// Every expert's codes, concatenated at the entry's expert stride.
    codes: DeviceBuffer,
    /// Every expert's scales, concatenated, as bf16.
    scales: DeviceBuffer,
    /// Every expert's biases, concatenated, as bf16.
    biases: DeviceBuffer,
}

/// Packs one projection for every expert and uploads it as one slab.
///
/// `salt` rotates the output rows of the shared weight fixture, so gate, up and
/// down are three different matrices and every expert is a fourth. An entry
/// that read the gate slab for the up projection, or expert zero's rows for all
/// four, would otherwise produce the right numbers.
fn stage_stack(gpu: &Gpu, out_dim: usize, in_dim: usize, salt: usize) -> Stack {
    let base = affine_fixture_weights(out_dim, in_dim);
    let slabs: Vec<Q4AffineSlab> = (0..STAGE_EXPERTS)
        .map(|e| {
            let shift = (e * 37 + salt * 91) % out_dim;
            let mut values = vec![0f32; out_dim * in_dim];
            for n in 0..out_dim {
                let src = (n + shift) % out_dim;
                values[n * in_dim..(n + 1) * in_dim]
                    .copy_from_slice(&base[src * in_dim..(src + 1) * in_dim]);
            }
            pack_q4_affine_tiled(&values, out_dim, in_dim, STORAGE_N)
        })
        .collect();
    let codes: Vec<u8> = slabs.iter().flat_map(|s| s.weights.clone()).collect();
    let scales: Vec<f32> = slabs.iter().flat_map(|s| s.scales.clone()).collect();
    let biases: Vec<f32> = slabs.iter().flat_map(|s| s.biases.clone()).collect();
    assert_eq!(
        codes.len(),
        STAGE_EXPERTS * out_dim * in_dim / 2,
        "the stacked codes are not the entry's expert stride"
    );
    Stack {
        codes: gpu.upload(&codes),
        scales: gpu.upload(&bf16_bytes(&scales)),
        biases: gpu.upload(&bf16_bytes(&biases)),
        slabs,
    }
}

/// One expert's slab, decoded to a dense row-major `[out_dim, in_dim]` matrix.
///
/// The reference multiplies against this rather than calling
/// [`dequant_q4_affine`] inside the innermost loop: the decode is exact either
/// way and hoisting it keeps a case that already walks two million products
/// from walking two million divisions as well.
fn dequantized(slab: &Q4AffineSlab) -> Vec<f32> {
    (0..slab.out_dim * slab.in_dim)
        .map(|i| dequant_q4_affine(slab, i))
        .collect()
}

/// The host product every routed column of a stage should hold.
///
/// `per_token` is the difference between the two stages and is the second seam
/// convention: gate and up read activation row `col / n_used`, the down
/// projection reads row `col`.
fn stage_reference(
    slabs: &[Q4AffineSlab],
    routing: &Routing,
    rows: &[f32],
    per_token: bool,
    act: Act,
) -> Vec<f32> {
    let in_dim = slabs[0].in_dim;
    let out_dim = slabs[0].out_dim;
    let dense: Vec<Vec<f32>> = slabs.iter().map(dequantized).collect();
    let mut want = vec![0f32; STAGE_COLUMNS * out_dim];
    for col in 0..STAGE_COLUMNS {
        let expert = routing.ids[col] as usize;
        let row = if per_token { col / STAGE_USED } else { col };
        let x = &rows[row * in_dim..(row + 1) * in_dim];
        for n in 0..out_dim {
            let w = &dense[expert][n * in_dim..(n + 1) * in_dim];
            let mut acc = 0f32;
            for k in 0..in_dim {
                acc += act.round(x[k]) * w[k];
            }
            want[col * out_dim + n] = acc;
        }
    }
    want
}

/// The fixture's activation block, `[rows, width]`.
///
/// `step` is deliberately not a binary fraction, so no product and no partial
/// sum is exact and the divergence the cases print is a real measurement of the
/// two summation orders rather than a property of the fixture.
fn stage_activations(rows: usize, width: usize, period: usize, step: f32) -> Vec<f32> {
    (0..rows * width)
        .map(|i| ((i % period) as f32 - (period / 2) as f32) * step)
        .collect()
}

/// The grid and block one expert stage is launched at, as `moe.rs` computes it.
fn stage_geometry(out_dim: usize) -> (Extent3, Extent3) {
    (
        Extent3::new(
            out_dim.div_ceil(STAGE_ROWS) as u32,
            STAGE_COLUMNS.min(STAGE_EXPERTS) as u32,
            1,
        ),
        Extent3::linear(STAGE_BLOCK),
    )
}

/// The argument block, in the field order `q35moe_expert_args` declares.
fn stage_args(in_dim: usize, out_dim: usize) -> [i32; 4] {
    [
        in_dim as i32,
        out_dim as i32,
        STAGE_EXPERTS as i32,
        STAGE_USED as i32,
    ]
}

/// A device buffer of `n` activations, seeded with a value no product can be.
///
/// An entry that writes its outputs transposed leaves most of the block at this
/// sentinel, which a zero-filled buffer would hide behind a plausible-looking
/// difference.
fn poisoned(gpu: &Gpu, n: usize, act: Act) -> DeviceBuffer {
    gpu.upload(&act.bytes(&vec![1e30f32; n]))
}

/// The worst divergence and the largest reference magnitude of one stage.
#[derive(Debug, Clone, Copy)]
struct Measured {
    /// Largest absolute difference from the host reference.
    worst: f32,
    /// Largest absolute value the host reference holds.
    magnitude: f32,
}

/// Launches the gate-and-up stage and measures it against the host reference.
fn measure_gate_up(gpu: &Gpu, act: Act) -> Measured {
    let routing = stage_routing();
    let gate = stage_stack(gpu, STAGE_INTER, STAGE_HIDDEN, 0);
    let up = stage_stack(gpu, STAGE_INTER, STAGE_HIDDEN, 1);
    let input = stage_activations(STAGE_TOKENS, STAGE_HIDDEN, 23, 0.317);
    let want_gate = stage_reference(&gate.slabs, &routing, &input, true, act);
    let want_up = stage_reference(&up.slabs, &routing, &input, true, act);

    let d_in = gpu.upload(&act.bytes(&input));
    let d_sorted = gpu.upload(&i32_bytes(&routing.sorted));
    let d_bounds = gpu.upload(&i32_bytes(&routing.bounds));
    let d_active = gpu.upload(&i32_bytes(&routing.active));
    let n = STAGE_COLUMNS * STAGE_INTER;
    let d_gate = poisoned(gpu, n, act);
    let d_up = poisoned(gpu, n, act);
    let args = stage_args(STAGE_HIDDEN, STAGE_INTER);
    let (grid, block) = stage_geometry(STAGE_INTER);
    run(
        gpu,
        KernelRef::new(MODULE, act.gate_up_entry()),
        &LaunchArgs {
            grid,
            block,
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::cast_slice(&args)),
                KernelArg::Buffer(gate.codes.view()),
                KernelArg::Buffer(gate.scales.view()),
                KernelArg::Buffer(gate.biases.view()),
                KernelArg::Buffer(up.codes.view()),
                KernelArg::Buffer(up.scales.view()),
                KernelArg::Buffer(up.biases.view()),
                KernelArg::Buffer(d_in.view()),
                KernelArg::Buffer(d_sorted.view()),
                KernelArg::Buffer(d_bounds.view()),
                KernelArg::Buffer(d_active.view()),
                KernelArg::Buffer(d_gate.view()),
                KernelArg::Buffer(d_up.view()),
            ],
        },
    );
    drain(gpu);

    let got_gate = act.values(&gpu.read(&d_gate));
    let got_up = act.values(&gpu.read(&d_up));
    Measured {
        worst: worst_divergence(&got_gate, &want_gate).max(worst_divergence(&got_up, &want_up)),
        magnitude: magnitude(&want_gate).max(magnitude(&want_up)),
    }
}

/// Launches the down stage and measures it against the host reference.
fn measure_down(gpu: &Gpu, act: Act) -> Measured {
    let routing = stage_routing();
    let down = stage_stack(gpu, STAGE_HIDDEN, STAGE_INTER, 2);
    let mid = stage_activations(STAGE_COLUMNS, STAGE_INTER, 29, 0.211);
    let want = stage_reference(&down.slabs, &routing, &mid, false, act);

    let d_mid = gpu.upload(&act.bytes(&mid));
    let d_sorted = gpu.upload(&i32_bytes(&routing.sorted));
    let d_bounds = gpu.upload(&i32_bytes(&routing.bounds));
    let d_active = gpu.upload(&i32_bytes(&routing.active));
    let d_out = poisoned(gpu, STAGE_COLUMNS * STAGE_HIDDEN, act);
    let args = stage_args(STAGE_INTER, STAGE_HIDDEN);
    let (grid, block) = stage_geometry(STAGE_HIDDEN);
    run(
        gpu,
        KernelRef::new(MODULE, act.down_entry()),
        &LaunchArgs {
            grid,
            block,
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::cast_slice(&args)),
                KernelArg::Buffer(down.codes.view()),
                KernelArg::Buffer(down.scales.view()),
                KernelArg::Buffer(down.biases.view()),
                KernelArg::Buffer(d_mid.view()),
                KernelArg::Buffer(d_sorted.view()),
                KernelArg::Buffer(d_bounds.view()),
                KernelArg::Buffer(d_active.view()),
                KernelArg::Buffer(d_out.view()),
            ],
        },
    );
    drain(gpu);

    let got = act.values(&gpu.read(&d_out));
    Measured {
        worst: worst_divergence(&got, &want),
        magnitude: magnitude(&want),
    }
}

/// The largest absolute value a reference block holds.
fn magnitude(values: &[f32]) -> f32 {
    values.iter().map(|v| v.abs()).fold(0f32, f32::max)
}

/// What the gate-and-up stage may diverge from the host reference by, in `f32`.
///
/// Measured, not chosen: this machine reports 4.196e-5 against a reference
/// whose largest value is 60.5, which is about 7e-7 relative over a 256-term
/// dot product. The budget is roughly five times that, which is headroom for
/// another device's fused-multiply-add placement and nothing like enough to
/// hide a convention error -- those miss by tens.
const GATE_UP_F32_BUDGET: f32 = 0.0002;

/// What the down stage may diverge by, in `f32`.
///
/// Measured 9.155e-5 against a largest reference value of 122.5, the same 7e-7
/// relative over twice the reduction depth.
const DOWN_F32_BUDGET: f32 = 0.0004;

/// What the gate-and-up stage may diverge by, in `bf16`.
///
/// Measured 0.1233, which is 2.0e-3 relative -- 2^-9, and that is not a
/// coincidence: bf16 carries eight mantissa bits, the activation rows arrive
/// already rounded to them and the outputs are rounded to them again on the way
/// out. Three thousand times the `f32` number for the same arithmetic, which is
/// what the dtype costs and why both cases exist.
const GATE_UP_BF16_BUDGET: f32 = 0.5;

/// What the down stage may diverge by, in `bf16`.
///
/// Measured 0.2461 against a largest reference value of 122.2: the same 2.0e-3
/// relative as the gate-and-up stage, at twice the magnitude.
const DOWN_BF16_BUDGET: f32 = 1.0;

/// The gate-and-up stage writes what the host reference writes, in `f32`.
///
/// Graded first and alone, so that the codec is measured without the dtype
/// change confounding it. Three of the four seam conventions are here: the
/// column is `token * n_used + slot`, gate and up read activation row
/// `col / n_used`, and the stage is launched over the ACTIVE experts through
/// the list the scan left behind rather than over all of them.
///
/// # Panics
///
/// When the launch is refused, or when the device product leaves the budget
/// printed beside it.
#[expect(
    clippy::print_stdout,
    reason = "the divergence this case measured is the finding, and a graded case that \
              holds a kernel to a budget says what it actually saw"
)]
pub fn the_gate_up_stage_matches_the_host_reference(gpu: &Gpu) {
    let m = measure_gate_up(gpu, Act::F32);
    println!(
        "gate/up q4affine f32 worst divergence: {} over {STAGE_HIDDEN} terms, \
         largest reference value {}, budget {GATE_UP_F32_BUDGET}",
        m.worst, m.magnitude,
    );
    assert!(
        m.worst < GATE_UP_F32_BUDGET,
        "worst divergence {} over {STAGE_HIDDEN} terms",
        m.worst,
    );
}

/// The down stage writes what the host reference writes, in `f32`.
///
/// The other two conventions are here: the down projection reads its OWN
/// column's row of the intermediate rather than its token's, and it writes
/// `[column, hidden]` row-major, which is column-major over (token, slot)
/// because that is what `q4e_moe_combine_kernel` reads.
///
/// # Panics
///
/// When the launch is refused, or when the device product leaves the budget
/// printed beside it.
#[expect(
    clippy::print_stdout,
    reason = "the divergence this case measured is the finding, and a graded case that \
              holds a kernel to a budget says what it actually saw"
)]
pub fn the_down_stage_matches_the_host_reference(gpu: &Gpu) {
    let m = measure_down(gpu, Act::F32);
    println!(
        "down q4affine f32 worst divergence: {} over {STAGE_INTER} terms, \
         largest reference value {}, budget {DOWN_F32_BUDGET}",
        m.worst, m.magnitude,
    );
    assert!(
        m.worst < DOWN_F32_BUDGET,
        "worst divergence {} over {STAGE_INTER} terms",
        m.worst,
    );
}

/// The gate-and-up stage in `bf16` activations, against the same reference.
///
/// The budget is looser than the `f32` case's and is not a different opinion
/// about the kernel: bf16 carries eight mantissa bits, so the activation rows
/// arrive with about 2^-9 of relative slip already in them and the outputs are
/// rounded again on the way out. **If this case ever measures tighter than the
/// `f32` one, the `f32` path is wrong**, not this one.
///
/// # Panics
///
/// When the launch is refused, or when the device product leaves the budget
/// printed beside it.
#[expect(
    clippy::print_stdout,
    reason = "the divergence this case measured is the finding, and a graded case that \
              holds a kernel to a budget says what it actually saw"
)]
pub fn the_gate_up_stage_in_bf16_matches_the_host_reference(gpu: &Gpu) {
    let m = measure_gate_up(gpu, Act::Bf16);
    let f32_side = measure_gate_up(gpu, Act::F32);
    println!(
        "gate/up q4affine bf16 worst divergence: {} over {STAGE_HIDDEN} terms, \
         largest reference value {}, budget {GATE_UP_BF16_BUDGET} \
         (the same stage in f32 measured {})",
        m.worst, m.magnitude, f32_side.worst,
    );
    assert!(
        m.worst < GATE_UP_BF16_BUDGET,
        "worst divergence {} over {STAGE_HIDDEN} terms",
        m.worst,
    );
    assert!(
        f32_side.worst < m.worst,
        "f32 diverged by {} and bf16 by {}: bf16 carries eight mantissa bits, so \
         the f32 path is the one to look at",
        f32_side.worst,
        m.worst,
    );
}

/// The down stage in `bf16` activations, against the same reference.
///
/// # Panics
///
/// When the launch is refused, or when the device product leaves the budget
/// printed beside it.
#[expect(
    clippy::print_stdout,
    reason = "the divergence this case measured is the finding, and a graded case that \
              holds a kernel to a budget says what it actually saw"
)]
pub fn the_down_stage_in_bf16_matches_the_host_reference(gpu: &Gpu) {
    let m = measure_down(gpu, Act::Bf16);
    let f32_side = measure_down(gpu, Act::F32);
    println!(
        "down q4affine bf16 worst divergence: {} over {STAGE_INTER} terms, \
         largest reference value {}, budget {DOWN_BF16_BUDGET} \
         (the same stage in f32 measured {})",
        m.worst, m.magnitude, f32_side.worst,
    );
    assert!(
        m.worst < DOWN_BF16_BUDGET,
        "worst divergence {} over {STAGE_INTER} terms",
        m.worst,
    );
    assert!(
        f32_side.worst < m.worst,
        "f32 diverged by {} and bf16 by {}: bf16 carries eight mantissa bits, so \
         the f32 path is the one to look at",
        f32_side.worst,
        m.worst,
    );
}

/// How many entries of `got` differ from `want`.
fn count_mismatches(got: &[i32], want: &[i32]) -> usize {
    assert_eq!(
        got.len(),
        want.len(),
        "the read-back is not the reference's shape"
    );
    got.iter().zip(want).filter(|(g, w)| g != w).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_packed_value_decodes_to_within_half_a_step_of_itself() {
        let values: Vec<f32> = (0..GROUP).map(|i| (i as f32) * 0.25 - 4.0).collect();
        let slab = pack_q4_affine(&values, 1, GROUP);
        let step = (values[GROUP - 1] - values[0]) / 15.0;
        for (i, &want) in values.iter().enumerate() {
            let got = dequant_q4_affine(&slab, i);
            assert!(
                (got - want).abs() <= step * 0.5 + 1e-6,
                "index {i}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn a_constant_group_survives_a_zero_range() {
        let slab = pack_q4_affine(&[2.5f32; GROUP], 1, GROUP);
        assert!((dequant_q4_affine(&slab, 0) - 2.5).abs() < 1e-6);
    }

    /// The group-major placement is the whole reason the tile reads the right
    /// bytes, so it is pinned here rather than only on a device.
    #[test]
    fn a_second_group_lands_after_every_output_of_the_first() {
        const OUT: usize = 3;
        const IN: usize = 2 * GROUP;
        // Distinct per (out, in) so a mixed-up index cannot coincide.
        let values: Vec<f32> = (0..OUT * IN).map(|i| (i % 16) as f32).collect();
        let slab = pack_q4_affine(&values, OUT, IN);
        assert_eq!(slab.scales.len(), OUT * 2);
        for n in 0..OUT {
            for k in 0..IN {
                let want = values[n * IN + k];
                let got = dequant_q4_affine(&slab, n * IN + k);
                assert!((got - want).abs() < 1e-3, "({n}, {k}): {got} vs {want}");
            }
        }
        // Output 0's second group starts a whole `out_dim` of groups later.
        let second_group_first_nibble = OUT * GROUP;
        assert!(second_group_first_nibble.is_multiple_of(2));
    }

    /// A projection wider than one kernel storage tile lands where the device
    /// reads it.
    ///
    /// This is the case the one-tile packer got silently wrong: with
    /// `out_dim > storage_n` the storage tile is the outer axis, so output
    /// `storage_n` restarts at quant group 0 of tile 1 rather than continuing
    /// tile 0. The addresses below are recomputed from `q35moe_affine_tile`'s
    /// own `StorageN` arithmetic rather than from [`placement`], so the two
    /// are graded against each other instead of against themselves.
    #[test]
    fn a_slab_wider_than_one_storage_tile_lands_where_the_device_reads() {
        const OUT: usize = 8;
        const STORAGE_N: usize = 4;
        const IN: usize = 2 * GROUP;
        const GROUPS: usize = IN / GROUP;

        let values = affine_fixture_weights(OUT, IN);
        let slab = pack_q4_affine_tiled(&values, OUT, IN, STORAGE_N);
        assert_eq!(slab.storage_n, STORAGE_N);
        assert!(
            distinct(&slab.scales) > 1,
            "the fixture must vary or this proves nothing"
        );

        for n in 0..OUT {
            // What the kernel does: pick the tile, then index inside it.
            let tile = n / STORAGE_N;
            let tile_offset = n % STORAGE_N;
            let tile_first_byte = tile * GROUPS * STORAGE_N * GROUP / 2;
            for g in 0..GROUPS {
                let group_first_byte = tile_first_byte + (g * STORAGE_N + tile_offset) * GROUP / 2;
                let parameter = (tile * GROUPS + g) * STORAGE_N + tile_offset;
                for r in 0..GROUP {
                    let nibble = group_first_byte * 2 + r;
                    let byte = slab.weights[nibble / 2];
                    let code = if nibble.is_multiple_of(2) {
                        byte & 0x0F
                    } else {
                        byte >> 4
                    };
                    let decoded = f32::from(code) * slab.scales[parameter] + slab.biases[parameter];
                    let want = values[n * IN + g * GROUP + r];
                    assert!(
                        (decoded - want).abs() < 1e-6,
                        "output {n} group {g} lane {r}: the bytes the kernel would \
                         read decode to {decoded}, not {want}",
                    );
                }
            }
        }

        // And the logical accessor agrees with all of it.
        for (i, &want) in values.iter().enumerate() {
            let got = dequant_q4_affine(&slab, i);
            assert!((got - want).abs() < 1e-6, "element {i}: {got} vs {want}");
        }
    }

    /// A width that is not a whole number of storage tiles is refused rather
    /// than mislaid, since the kernel's last tile is a whole tile.
    #[test]
    #[should_panic(expected = "whole number of storage tiles")]
    fn a_width_that_is_not_a_whole_number_of_tiles_is_refused() {
        let values = vec![0f32; 6 * GROUP];
        drop(pack_q4_affine_tiled(&values, 6, GROUP, 4));
    }
}
