//! Device cases for the per-layer embedding pair.
//!
//! Gemma 4 E4B carries one `q6_K` block per layer slot per token id, off a table
//! far too large to hold resident. Two kernels turn a row of that table into the
//! per-slot vector the layer loop multiplies its gate by: `prepare` normalizes
//! the model's own projection, dequantizes the row's slot and mixes the two, and
//! `gate_mul` applies the mix at the one slot whose turn it is.
//!
//! Both are checked against a plain Rust reading of the same formula, with the
//! `q6_K` packer and dequantizer written out here so the fixture never depends
//! on the kernel it is testing. The packer is the inverse of
//! `dequantize_dense_q6_K` (`metal/qwen35/gemv_ext_q5k_q6k.metal`), which is the
//! decoder `kernel_dense_ple_prepare_q6_K` reuses.
//!
//! The `affine4` twin of `prepare` reads the same formula off the MLX sidecar's
//! row layout instead of a `q6_K` one, and is checked the same way: an unpack
//! written out here, against a fixture whose slot count is deliberately not the
//! shipped checkpoint's, so that a kernel carrying that checkpoint's offsets as
//! literals fails rather than passes.
//!
//! The `#[test]` wrappers live in each backend's own driver -
//! `crates/shiso-metal/tests/dense_ple.rs`,
//! `crates/shiso-cuda/tests/dense_ple.rs`. [`ENTRIES`] is here rather than
//! there because the twin list is a property of the kernel set, not of a
//! device.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "kernel argument structs take u32 dimensions while the fixtures are written in usize, and every fixture dimension here is a small literal"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use bytemuck::{Pod, Zeroable};
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};
use shiso_modelfile::quant::BlockQ6K;

use crate::oracle::TOL_QUANT;

/// Every entry point this module's cases launch.
pub const ENTRIES: [&str; 3] = [
    "kernel_dense_ple_prepare_q6_K",
    "kernel_dense_ple_gate_mul",
    "kernel_dense_ple_prepare_affine4",
];

/// Width of one slot's per-layer row, which is also one `q6_K` block.
const DIM: usize = 256;
/// Layer slots the fixture table carries per token.
const LAYERS: usize = 2;
/// Tokens the fixture chunk carries.
const N_TOK: usize = 3;
/// Rows the resident fixture table holds, which the ids below address into.
const VOCAB: usize = 8;
/// Threads per `prepare` threadgroup, mirroring `shiso_engine::launch::BLOCK_128`.
const PREPARE_BLOCK: u32 = 128;
/// Threadgroup bytes `prepare` asks for: the block sum's slots plus the staged
/// row, mirroring `shiso_engine::launch::DENSE_PLE_SMEM_BYTES`.
const PREPARE_SMEM: u32 = 4 * (33 + DIM as u32);
/// Threads per threadgroup of the flat gate kernel, mirroring
/// `shiso_engine::launch::BLOCK_LINEAR`.
const LINEAR_BLOCK: u32 = 256;

/// `f16` bits for 1.0, the block scale every fixture block carries.
const ONE_F16: u16 = 0x3C00;

/// Signed per-16-element `q6_K` scales. Both signs appear, which is what makes
/// `q6_K` symmetric and is the half of its range a fixture of one sign misses.
const Q6_SCALES: [i8; 16] = [
    17, -33, 5, -1, 64, -64, 22, -7, 1, -1, 31, -31, 12, -12, 40, -40,
];

/// The `qh` bit position carrying the top two bits of each 32-element quarter,
/// in the order `dequantize_dense_q6_K` walks them.
const QH_SHIFTS: [u32; 4] = [0, 2, 4, 6];

// -- argument blocks --------------------------------------------------------

/// `dense_args_ple`, mirroring `shiso_engine::launch::DensePleArgs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PleArgs {
    n_tok: u32,
    n_layers: u32,
    dim: u32,
    row_bytes: u32,
    resident: u32,
    eps: f32,
    proj_scale: f32,
    embed_scale: f32,
    mix_scale: f32,
}

const _: () = assert!(size_of::<PleArgs>() == 36);

/// `dense_args_ple_gate`, mirroring `shiso_engine::launch::DensePleGateArgs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PleGateArgs {
    n_tok: u32,
    n_layers: u32,
    dim: u32,
    layer: u32,
}

const _: () = assert!(size_of::<PleGateArgs>() == 16);

// -- the q6_K fixture -------------------------------------------------------

/// Raw six-bit weights for one table block: the whole `0..64` range, so all four
/// quarters of a block and both nibble halves of `ql` are covered.
fn q6_values(block: usize) -> [u8; DIM] {
    let mut values = [0u8; DIM];
    for (e, slot) in values.iter_mut().enumerate() {
        *slot = ((e * 3 + block * 5) % 64) as u8;
    }
    values
}

/// Packs 256 raw six-bit values into a `q6_K` block: the inverse of
/// [`dequant_q6_k`]. Element `e` sits at `l = e % 32` inside its quarter, so the
/// quarter picks the `ql` byte (odd quarters live in the pair's second 32
/// bytes), the nibble (low for quarters 0 and 1, high for the next two) and the
/// two-bit field of `qh`.
fn pack_q6_k(values: &[u8; DIM]) -> BlockQ6K {
    let mut block = BlockQ6K {
        ql: [0u8; 128],
        qh: [0u8; 64],
        scales: Q6_SCALES,
        d: ONE_F16,
    };
    for (e, &v) in values.iter().enumerate() {
        let half = e / 128;
        let quarter = (e % 128) / 32;
        let l = e % 32;
        block.ql[half * 64 + (quarter % 2) * 32 + l] |= (v & 0x0F) << ((quarter / 2) * 4);
        block.qh[half * 32 + l] |= (v >> 4) << QH_SHIFTS[quarter];
    }
    block
}

/// Element `e` of a `q6_K` block as `dequantize_dense_q6_K` reads it: low and
/// high nibbles of `ql` alternate with the two-bit fields of `qh`, and the `-32`
/// bias is folded before the multiply.
fn dequant_q6_k(block: &BlockQ6K, e: usize) -> f32 {
    let half = e / 128;
    let quarter = (e % 128) / 32;
    let l = e % 32;
    let scale = f32::from(block.scales[half * 8 + quarter * 2 + l / 16]);
    let d = crate::oracle::f16_to_f32(block.d) * scale;
    let ql = block.ql[half * 64 + (quarter % 2) * 32 + l] >> ((quarter / 2) * 4);
    let qh = block.qh[half * 32 + l] >> QH_SHIFTS[quarter];
    let v = u32::from(ql & 0x0F) | (u32::from(qh & 3) << 4);
    d * (v as i32 - 32) as f32
}

/// The table's blocks, laid out row-major: `rows` rows of `LAYERS` blocks each.
///
/// Block `row * LAYERS + l` is seeded by its own index, so a kernel that read
/// the wrong row or the wrong slot of the right row reads different numbers.
fn table(rows: usize) -> Vec<BlockQ6K> {
    (0..rows * LAYERS)
        .map(|block| pack_q6_k(&q6_values(block)))
        .collect()
}

// -- helpers ----------------------------------------------------------------

/// Launches one dense entry point, naming the geometry if the driver refuses it.
fn launch(
    gpu: &Gpu,
    entry: &'static str,
    grid: Extent3,
    block: Extent3,
    smem_bytes: u32,
    args: &[KernelArg<'_>],
) {
    run(
        gpu,
        KernelRef::new("dense", entry),
        &LaunchArgs {
            grid,
            block,
            smem_bytes,
            args,
        },
    );
}

/// An f32 device buffer of `len` zeros.
fn zeros(gpu: &Gpu, len: usize) -> DeviceBuffer {
    gpu.upload(&f32_bytes(&vec![0.0f32; len]))
}

/// Asserts every value agrees within [`TOL_QUANT`], scaled by the expected
/// magnitude.
fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (index, (&g, &w)) in got.iter().zip(want).enumerate() {
        let bound = TOL_QUANT * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= bound,
            "{what}: element {index} is {g}, expected {w} (tolerance {bound})",
        );
    }
}

/// The tanh approximation of the Gaussian error linear unit, as
/// `dense_gelu_tanh` writes it.
fn gelu_tanh(v: f32) -> f32 {
    let c = 0.797_884_6_f32;
    0.5 * v * (1.0 + (c * (v + 0.044_715 * v * v * v)).tanh())
}

/// The scales the fixture launches carry, which are Gemma 4's own: the
/// projection is divided by `sqrt(hidden)`, the embedding multiplied by
/// `sqrt(dim)`, and the sum halved in quadrature.
struct Scales {
    eps: f32,
    proj: f32,
    embed: f32,
    mix: f32,
}

const SCALES: Scales = Scales {
    eps: 1.0e-6,
    // 1 / sqrt(hidden) at a plausible hidden of 2048.
    proj: 0.022_097_087,
    // sqrt(dim) at dim 256.
    embed: 16.0,
    // 1 / sqrt(2).
    mix: 0.707_106_77,
};

/// The whole `prepare` formula, per (token, slot), as a plain `f32` loop.
///
/// `row` says which table row this token reads, which is the token id in the
/// resident mode and the token's own index in the streamed one.
fn prepare_oracle(
    proj: &[f32],
    norm_w: &[f32],
    blocks: &[BlockQ6K],
    row: impl Fn(usize) -> usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; N_TOK * LAYERS * DIM];
    for t in 0..N_TOK {
        for l in 0..LAYERS {
            let at = (t * LAYERS + l) * DIM;
            // The mean square is over the SCALED projection, so the scale
            // reaches the magnitudes the normalization divides by as well as
            // the value it then multiplies.
            let mut sum = 0.0f32;
            for i in 0..DIM {
                let v = proj[at + i] * SCALES.proj;
                sum += v * v;
            }
            let inv = 1.0 / (sum / DIM as f32 + SCALES.eps).sqrt();
            let block = &blocks[row(t) * LAYERS + l];
            for i in 0..DIM {
                let p = proj[at + i] * SCALES.proj * inv * norm_w[i];
                let e = dequant_q6_k(block, i) * SCALES.embed;
                out[at + i] = (p + e) * SCALES.mix;
            }
        }
    }
    out
}

/// Runs `prepare` on the device in one of its two addressing modes.
fn prepare_on_device(
    gpu: &Gpu,
    proj: &[f32],
    norm_w: &[f32],
    blocks: &[BlockQ6K],
    ids: &[i32],
    resident: bool,
) -> Vec<f32> {
    let d_proj = gpu.upload(&f32_bytes(proj));
    let d_norm = gpu.upload(&f32_bytes(norm_w));
    let d_rows = gpu.upload(bytemuck::cast_slice(blocks));
    let d_ids = gpu.upload(bytemuck::cast_slice(ids));
    let out = zeros(gpu, N_TOK * LAYERS * DIM);
    let args = PleArgs {
        n_tok: N_TOK as u32,
        n_layers: LAYERS as u32,
        dim: DIM as u32,
        row_bytes: (LAYERS * size_of::<BlockQ6K>()) as u32,
        resident: u32::from(resident),
        eps: SCALES.eps,
        proj_scale: SCALES.proj,
        embed_scale: SCALES.embed,
        mix_scale: SCALES.mix,
    };
    launch(
        gpu,
        "kernel_dense_ple_prepare_q6_K",
        Extent3::new(LAYERS as u32, N_TOK as u32, 1),
        Extent3::new(PREPARE_BLOCK, 1, 1),
        PREPARE_SMEM,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(d_proj.view()),
            KernelArg::Buffer(d_norm.view()),
            KernelArg::Buffer(d_rows.view()),
            KernelArg::Buffer(d_ids.view()),
        ],
    );
    drain(gpu);
    gpu.read_f32(&out)
}

// -- prepare ----------------------------------------------------------------

/// A resident table is indexed by the token id, and the whole formula holds.
pub fn a_resident_table_mixes_the_normalized_projection_with_the_row_its_id_names(gpu: &Gpu) {
    let proj = crate::oracle::signal(N_TOK * LAYERS * DIM, 31);
    let norm_w = crate::oracle::signal(DIM, 32);
    let blocks = table(VOCAB);
    // Deliberately not 0, 1, 2: a kernel that read `t` instead of `ids[t]` would
    // agree with the oracle on an identity id row.
    let ids: [i32; N_TOK] = [5, 0, 3];

    let got = prepare_on_device(gpu, &proj, &norm_w, &blocks, &ids, true);
    let want = prepare_oracle(&proj, &norm_w, &blocks, |t| ids[t] as usize);
    close(&got, &want, "prepare (resident)");

    // And the ids really are what picked the rows: the identity oracle is a
    // different answer, which is what makes the case above mean something.
    let identity = prepare_oracle(&proj, &norm_w, &blocks, |t| t);
    assert!(
        got.iter()
            .zip(&identity)
            .any(|(g, w)| (g - w).abs() > TOL_QUANT * w.abs().max(1.0)),
        "the ids chose the rows"
    );
}

/// A streamed table has already been narrowed to the chunk, so row `t` is
/// token `t`'s.
pub fn a_streamed_table_reads_the_row_that_was_staged_for_each_token(gpu: &Gpu) {
    let proj = crate::oracle::signal(N_TOK * LAYERS * DIM, 33);
    let norm_w = crate::oracle::signal(DIM, 34);
    // The staging holds exactly the chunk's rows, in token order.
    let blocks = table(N_TOK);
    let ids: [i32; N_TOK] = [5, 0, 3];

    let got = prepare_on_device(gpu, &proj, &norm_w, &blocks, &ids, false);
    let want = prepare_oracle(&proj, &norm_w, &blocks, |t| t);
    close(&got, &want, "prepare (streamed)");
}

/// Each slot of a token reads its own block, not the first one.
///
/// A kernel that dropped the layer term from either the row offset or the
/// destination would still produce finite numbers of the right magnitude for
/// every token, which is exactly the failure this catches.
pub fn the_two_slots_of_one_token_are_two_different_rows_of_the_table(gpu: &Gpu) {
    // One norm weight and one projection everywhere, so the ONLY thing that can
    // differ between the two slots is which block they decoded.
    let proj = vec![0.5f32; N_TOK * LAYERS * DIM];
    let norm_w = vec![1.0f32; DIM];
    let blocks = table(N_TOK);
    let ids = [0i32; N_TOK];

    let got = prepare_on_device(gpu, &proj, &norm_w, &blocks, &ids, false);
    let slot = |t: usize, l: usize| &got[(t * LAYERS + l) * DIM..(t * LAYERS + l + 1) * DIM];
    assert_ne!(slot(0, 0), slot(0, 1), "the two slots decode two blocks");
    assert_ne!(slot(0, 0), slot(1, 0), "the two tokens read two rows");
}

// -- prepare, from the MLX affine sidecar -----------------------------------

/// Quantization groups one slot of an affine row carries.
const AFFINE_GROUPS: usize = DIM / 64;
/// Bytes of packed words one slot of an affine row carries.
const AFFINE_WORDS_BYTES: usize = DIM / 2;
/// Bytes of scales - and, separately, of biases - one slot carries.
const AFFINE_SCALES_BYTES: usize = AFFINE_GROUPS * 2;
/// Row stride: every slot's words, then every slot's scales, then the biases.
const AFFINE_ROW_BYTES: usize = LAYERS * AFFINE_WORDS_BYTES + 2 * LAYERS * AFFINE_SCALES_BYTES;

/// One affine table row per entry, laid out the way `ple_repack` writes them.
///
/// The nibbles walk the row so that no two slots, and no two rows, decode to
/// the same values; the scales and biases are distinct per group for the same
/// reason.
fn affine_table(rows: usize) -> Vec<u8> {
    let mut out = vec![0u8; rows * AFFINE_ROW_BYTES];
    for r in 0..rows {
        let row = &mut out[r * AFFINE_ROW_BYTES..(r + 1) * AFFINE_ROW_BYTES];
        for (index, byte) in row[..LAYERS * AFFINE_WORDS_BYTES].iter_mut().enumerate() {
            let low = (index + 3 * r) % 16;
            let high = (index / 2 + r + 7) % 16;
            *byte = (low | (high << 4)) as u8;
        }
        let scales = &mut row[LAYERS * AFFINE_WORDS_BYTES..];
        for g in 0..2 * LAYERS * AFFINE_GROUPS {
            let value = if g < LAYERS * AFFINE_GROUPS {
                // Scales, kept positive and of a plausible magnitude.
                half::bf16::from_f32(0.05 + 0.01 * ((g + r) % 5) as f32)
            } else {
                // Biases, which the kernel adds after the scale.
                half::bf16::from_f32(-0.2 + 0.05 * ((g + 2 * r) % 7) as f32)
            };
            scales[2 * g..2 * g + 2].copy_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    out
}

/// One weight of one slot of one affine row, unpacked.
fn dequant_affine4(table: &[u8], row: usize, slot: usize, element: usize) -> f32 {
    let base = row * AFFINE_ROW_BYTES;
    let group = element / 64;
    let byte = table[base + slot * AFFINE_WORDS_BYTES + element / 2];
    let nibble = if element.is_multiple_of(2) {
        byte & 0x0f
    } else {
        byte >> 4
    };
    let scales_base = base + LAYERS * AFFINE_WORDS_BYTES;
    let biases_base = scales_base + LAYERS * AFFINE_SCALES_BYTES;
    let at = slot * AFFINE_SCALES_BYTES + group * 2;
    let read = |from: usize| {
        half::bf16::from_bits(u16::from_le_bytes([table[from], table[from + 1]])).to_f32()
    };
    f32::from(nibble) * read(scales_base + at) + read(biases_base + at)
}

/// The `prepare` formula again, over an affine row instead of a `q6_K` one.
fn prepare_affine_oracle(
    proj: &[f32],
    norm_w: &[f32],
    table: &[u8],
    row: impl Fn(usize) -> usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; N_TOK * LAYERS * DIM];
    for t in 0..N_TOK {
        for l in 0..LAYERS {
            let at = (t * LAYERS + l) * DIM;
            let mut sum = 0.0f32;
            for i in 0..DIM {
                let v = proj[at + i] * SCALES.proj;
                sum += v * v;
            }
            let inv = 1.0 / (sum / DIM as f32 + SCALES.eps).sqrt();
            for i in 0..DIM {
                let p = proj[at + i] * SCALES.proj * inv * norm_w[i];
                let e = dequant_affine4(table, row(t), l, i) * SCALES.embed;
                out[at + i] = (p + e) * SCALES.mix;
            }
        }
    }
    out
}

/// Runs the affine `prepare` on the device in one of its two addressing modes.
fn prepare_affine_on_device(
    gpu: &Gpu,
    proj: &[f32],
    norm_w: &[f32],
    table: &[u8],
    ids: &[i32],
    resident: bool,
) -> Vec<f32> {
    let d_proj = gpu.upload(&f32_bytes(proj));
    let d_norm = gpu.upload(&f32_bytes(norm_w));
    let d_rows = gpu.upload(table);
    let d_ids = gpu.upload(bytemuck::cast_slice(ids));
    let out = zeros(gpu, N_TOK * LAYERS * DIM);
    let args = PleArgs {
        n_tok: N_TOK as u32,
        n_layers: LAYERS as u32,
        dim: DIM as u32,
        row_bytes: AFFINE_ROW_BYTES as u32,
        resident: u32::from(resident),
        eps: SCALES.eps,
        proj_scale: SCALES.proj,
        embed_scale: SCALES.embed,
        mix_scale: SCALES.mix,
    };
    launch(
        gpu,
        "kernel_dense_ple_prepare_affine4",
        Extent3::new(LAYERS as u32, N_TOK as u32, 1),
        Extent3::new(PREPARE_BLOCK, 1, 1),
        PREPARE_SMEM,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(d_proj.view()),
            KernelArg::Buffer(d_norm.view()),
            KernelArg::Buffer(d_rows.view()),
            KernelArg::Buffer(d_ids.view()),
        ],
    );
    drain(gpu);
    gpu.read_f32(&out)
}

/// The `affine4` twin of `prepare`, over a resident table indexed by token id.
///
/// The same formula as the `q6_K` case above, read off the mlx sidecar's row
/// layout instead. The fixture's slot count is deliberately not the shipped
/// checkpoint's, so a kernel carrying that checkpoint's row offsets as literals
/// fails here rather than passing at the one shape it was written against.
pub fn the_affine_prepare_mixes_the_projection_with_the_row_its_id_names(gpu: &Gpu) {
    let proj = crate::oracle::signal(N_TOK * LAYERS * DIM, 35);
    let norm_w = crate::oracle::signal(DIM, 36);
    let table = affine_table(VOCAB);
    let ids: [i32; N_TOK] = [5, 0, 3];

    let got = prepare_affine_on_device(gpu, &proj, &norm_w, &table, &ids, true);
    let want = prepare_affine_oracle(&proj, &norm_w, &table, |t| ids[t] as usize);
    close(&got, &want, "affine prepare (resident)");

    let identity = prepare_affine_oracle(&proj, &norm_w, &table, |t| t);
    assert!(
        got.iter()
            .zip(&identity)
            .any(|(g, w)| (g - w).abs() > TOL_QUANT * w.abs().max(1.0)),
        "the ids chose the rows"
    );
}

/// The streamed staging holds exactly the chunk's rows, in token order.
pub fn the_affine_prepare_reads_the_row_that_was_staged_for_each_token(gpu: &Gpu) {
    let proj = crate::oracle::signal(N_TOK * LAYERS * DIM, 37);
    let norm_w = crate::oracle::signal(DIM, 38);
    let table = affine_table(N_TOK);
    let ids: [i32; N_TOK] = [5, 0, 3];

    let got = prepare_affine_on_device(gpu, &proj, &norm_w, &table, &ids, false);
    let want = prepare_affine_oracle(&proj, &norm_w, &table, |t| t);
    close(&got, &want, "affine prepare (streamed)");
}

/// Each slot reads its own span of the row's words, scales and biases.
///
/// The three bases are derived inside the kernel from the slot count it is
/// handed; a kernel that carried the shipped checkpoint's forty-two-slot
/// offsets instead would read past this fixture's rows entirely.
pub fn the_two_slots_of_one_affine_row_are_two_different_spans(gpu: &Gpu) {
    let proj = vec![0.5f32; N_TOK * LAYERS * DIM];
    let norm_w = vec![1.0f32; DIM];
    let table = affine_table(N_TOK);
    let ids = [0i32; N_TOK];

    let got = prepare_affine_on_device(gpu, &proj, &norm_w, &table, &ids, false);
    let slot = |t: usize, l: usize| &got[(t * LAYERS + l) * DIM..(t * LAYERS + l + 1) * DIM];
    assert_ne!(slot(0, 0), slot(0, 1), "the two slots decode two spans");
    assert_ne!(slot(0, 0), slot(1, 0), "the two tokens read two rows");
}

// -- gate_mul ---------------------------------------------------------------

/// The gate is activated and multiplied by ONE slot of the per-layer block.
pub fn the_gate_is_the_activation_times_the_named_slots_row(gpu: &Gpu) {
    /// The slot this case reads, which is deliberately not the first one.
    const LAYER: usize = 1;

    let gate = crate::oracle::signal(N_TOK * DIM, 41);
    let ple = crate::oracle::signal(N_TOK * LAYERS * DIM, 42);
    let d_gate = gpu.upload(&f32_bytes(&gate));
    let d_ple = gpu.upload(&f32_bytes(&ple));

    let args = PleGateArgs {
        n_tok: N_TOK as u32,
        n_layers: LAYERS as u32,
        dim: DIM as u32,
        layer: LAYER as u32,
    };
    let n = (N_TOK * DIM) as u32;
    launch(
        gpu,
        "kernel_dense_ple_gate_mul",
        Extent3::linear(n.div_ceil(LINEAR_BLOCK)),
        Extent3::new(LINEAR_BLOCK, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_gate.view()),
            KernelArg::Buffer(d_ple.view()),
        ],
    );
    drain(gpu);

    let mut want = vec![0.0f32; N_TOK * DIM];
    for t in 0..N_TOK {
        for i in 0..DIM {
            want[t * DIM + i] = gelu_tanh(gate[t * DIM + i]) * ple[(t * LAYERS + LAYER) * DIM + i];
        }
    }
    close(&gpu.read_f32(&d_gate), &want, "gate_mul");

    // Slot 0 is a different answer, so the layer field really chose slot 1.
    let other: Vec<f32> = (0..N_TOK * DIM)
        .map(|k| gelu_tanh(gate[k]) * ple[(k / DIM * LAYERS) * DIM + k % DIM])
        .collect();
    assert!(
        gpu.read_f32(&d_gate)
            .iter()
            .zip(&other)
            .any(|(g, w)| (g - w).abs() > TOL_QUANT * w.abs().max(1.0)),
        "the layer field chose the slot"
    );
}
