//! Cases for the MLX packed affine matrix-vector (qmv) and matrix-matrix (qmm)
//! kernels.
//!
//! Evaluates `kernel_dense_affine_qmv_4_f32`, `kernel_dense_affine_qmv_8_f32`,
//! `kernel_dense_affine_qmm_4_f32`, and `kernel_dense_affine_qmm_8_f32` against
//! the host reference in `shiso_modelfile::mlx_quant::dequant_affine` followed by an
//! f32 host matmul, and `kernel_dense_embed_affine4_f32` against a per-column
//! unpack of the same layout.
//!
//! The `#[test]` wrappers live in each backend's own driver -
//! `crates/shiso-metal/tests/dense_affine.rs`,
//! `crates/shiso-cuda/tests/dense_affine.rs`. [`ENTRIES`] is here rather than
//! there because the twin list is a property of the kernel set, not of a
//! device.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "fixture dimensions are small positive integers"
)]
#![expect(clippy::similar_names, reason = "sum_x and sum_qx mirror math")]
#![expect(
    clippy::many_single_char_names,
    reason = "math loops use single-character variable names"
)]
#![expect(
    clippy::too_many_arguments,
    reason = "test helper passes all tensor dimensions and slices"
)]
#![expect(
    clippy::print_stdout,
    reason = "check_rows prints the margin it measured, which is what makes a near-miss readable when a case is run by hand under --ignored"
)]
#![expect(
    clippy::cast_sign_loss,
    reason = "fixture token ids are small non-negative literals"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use bytemuck::{Pod, Zeroable};
use half::bf16;
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

/// Argument block mirroring `dense_args_affine`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DenseAffineArgs {
    pub in_dim: u32,
    pub out_dim: u32,
    pub n_tok: u32,
    pub bits: u32,
}

const _: () = assert!(size_of::<DenseAffineArgs>() == 16);

/// Launches one dense entry point.
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

/// Simple deterministic LCG pseudo-random generator for fixture generation.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.0 >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }

    fn next_bf16(&mut self) -> bf16 {
        bf16::from_f32(self.next_f32() * 0.1)
    }
}

/// Every entry point this module's cases launch.
pub const ENTRIES: [&str; 5] = [
    "kernel_dense_affine_qmv_4_f32",
    "kernel_dense_affine_qmv_8_f32",
    "kernel_dense_affine_qmm_4_f32",
    "kernel_dense_affine_qmm_8_f32",
    "kernel_dense_embed_affine4_f32",
];
/// Compares `got` and `want` rows with relative and absolute tolerance.
fn check_rows(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut max_diff = 0.0f32;
    let mut max_rel = 0.0f32;
    for (index, (&g, &w)) in got.iter().zip(want).enumerate() {
        let diff = (g - w).abs();
        let rel = if w.abs() > 1e-4 { diff / w.abs() } else { diff };
        if diff > max_diff {
            max_diff = diff;
        }
        if rel > max_rel {
            max_rel = rel;
        }
        let tol = (w.abs() * 2e-3).max(2e-4);
        assert!(
            diff <= tol,
            "{what}: element {index} is {g}, expected {w} (diff {diff}, tol {tol}, rel {rel})",
        );
    }
    println!("{what}: max_diff={max_diff:.6}, max_rel={max_rel:.6}");
}

/// Host reference: dequantizes affine weights using `dequant_affine` to f16, then matmuls with x.
fn host_affine_matmul(
    words: &[u32],
    scales: &[bf16],
    biases: &[bf16],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
    n_tok: usize,
    bits: u32,
) -> Vec<f32> {
    let words_per_row = in_dim * bits as usize / 32;
    let groups_per_row = in_dim / 64;
    let mut want = vec![0.0f32; n_tok * out_dim];
    let mask = (1u32 << bits) - 1;

    for r in 0..out_dim {
        let row_w = &words[r * words_per_row..(r + 1) * words_per_row];
        let row_s = &scales[r * groups_per_row..(r + 1) * groups_per_row];
        let row_b = &biases[r * groups_per_row..(r + 1) * groups_per_row];

        for t in 0..n_tok {
            let x_tok = &x[t * in_dim..(t + 1) * in_dim];
            let mut row_acc = 0.0f32;
            for g in 0..groups_per_row {
                let s = f32::from(row_s[g]);
                let b = f32::from(row_b[g]);
                let mut sum_qx = 0.0f32;
                let mut sum_x = 0.0f32;
                for i in 0..64 {
                    let c = i;
                    let elem_idx = g * 64 + i;
                    let word_idx = g * (64 * bits as usize / 32) + (c * bits as usize / 32);
                    let shift = (c * bits as usize) % 32;
                    let q = (row_w[word_idx] >> shift) & mask;
                    let xc = x_tok[elem_idx];
                    sum_qx += (q as f32) * xc;
                    sum_x += xc;
                }
                row_acc += s * sum_qx + b * sum_x;
            }
            want[t * out_dim + r] = row_acc;
        }
    }
    want
}

/// Runs a qmv or qmm test on GPU and checks against host reference.
fn run_affine_test(gpu: &Gpu, in_dim: usize, out_dim: usize, n_tok: usize, bits: u32, seed: u64) {
    let mut lcg = Lcg::new(seed);
    let words_per_row = in_dim * bits as usize / 32;
    let groups_per_row = in_dim / 64;

    let mut words = vec![0u32; out_dim * words_per_row];
    for w in &mut words {
        *w = lcg.next_u32();
    }

    let mut scales = vec![bf16::ZERO; out_dim * groups_per_row];
    for s in &mut scales {
        *s = lcg.next_bf16();
    }

    let mut biases = vec![bf16::ZERO; out_dim * groups_per_row];
    for b in &mut biases {
        *b = lcg.next_bf16();
    }

    let mut x = vec![0.0f32; n_tok * in_dim];
    for v in &mut x {
        *v = lcg.next_f32();
    }

    let want = host_affine_matmul(&words, &scales, &biases, &x, out_dim, in_dim, n_tok, bits);

    let d_w = gpu.upload(bytemuck::cast_slice::<u32, u8>(&words));
    let d_scales = gpu.upload(bytemuck::cast_slice::<bf16, u8>(&scales));
    let d_biases = gpu.upload(bytemuck::cast_slice::<bf16, u8>(&biases));
    let d_x = gpu.upload(&f32_bytes(&x));
    let d_y = zeros(gpu, n_tok * out_dim);

    let args = DenseAffineArgs {
        in_dim: in_dim as u32,
        out_dim: out_dim as u32,
        n_tok: n_tok as u32,
        bits,
    };

    let (entry, grid) = if n_tok == 1 {
        let name = match bits {
            4 => "kernel_dense_affine_qmv_4_f32",
            8 => "kernel_dense_affine_qmv_8_f32",
            _ => panic!("unsupported bits"),
        };
        (name, Extent3::linear(out_dim as u32))
    } else {
        let name = match bits {
            4 => "kernel_dense_affine_qmm_4_f32",
            8 => "kernel_dense_affine_qmm_8_f32",
            _ => panic!("unsupported bits"),
        };
        // The band width the matrix entries carry, mirrored from
        // `shiso_engine::projection::AFFINE_GEMM_TOKENS`.
        (
            name,
            Extent3::new(out_dim as u32, (n_tok as u32).div_ceil(8), 1),
        )
    };

    let block = Extent3::linear(32);
    launch(
        gpu,
        entry,
        grid,
        block,
        0,
        &[
            KernelArg::Buffer(d_w.view()),
            KernelArg::Buffer(d_scales.view()),
            KernelArg::Buffer(d_biases.view()),
            KernelArg::Buffer(d_x.view()),
            KernelArg::Buffer(d_y.view()),
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
        ],
    );
    drain(gpu);

    let got = gpu.read_f32(&d_y);
    check_rows(&got, &want, entry);
}

/// The 4-bit matvec against `dequant_affine` followed by an f32 host matmul.
pub fn qmv_4_matches_the_host_dequant_matmul(gpu: &Gpu) {
    run_affine_test(gpu, 2560, 2048, 1, 4, 12345);
}

/// The 8-bit matvec against `dequant_affine` followed by an f32 host matmul.
pub fn qmv_8_matches_the_host_dequant_matmul(gpu: &Gpu) {
    run_affine_test(gpu, 2560, 10240, 1, 8, 54321);
}

/// The 4-bit GEMM at a prefill width, which is the entry a chunk of more than
/// one token takes.
pub fn qmm_4_matches_at_n_tok_64(gpu: &Gpu) {
    run_affine_test(gpu, 2560, 2048, 64, 4, 67890);
}

/// The 8-bit GEMM at a prefill width, which is the entry a chunk of more than
/// one token takes.
pub fn qmm_8_matches_at_n_tok_64(gpu: &Gpu) {
    run_affine_test(gpu, 2560, 10240, 64, 8, 98765);
}

/// An output width that is not a whole number of the row tile, so the tail
/// tile's guard is exercised rather than assumed.
pub fn a_row_count_that_is_not_a_tile_multiple_is_handled(gpu: &Gpu) {
    run_affine_test(gpu, 256, 2049, 1, 4, 112_233);
}

/// A chunk that is not a whole number of bands, so the last one is partial.
///
/// Thirteen tokens is one full band of eight and one of five. The matrix
/// entries carry a band per threadgroup and clamp the last one against the
/// chunk length; a kernel that carried the band width instead would write five
/// rows of somebody else's chunk past the end of this one, and at a whole band
/// - which sixty-four tokens is - nothing would ever notice.
pub fn qmm_4_matches_at_a_chunk_that_is_not_a_whole_band(gpu: &Gpu) {
    run_affine_test(gpu, 2560, 512, 13, 4, 246_802);
}

/// The 8-bit twin of the partial-band case above, at the same thirteen tokens.
pub fn qmm_8_matches_at_a_chunk_that_is_not_a_whole_band(gpu: &Gpu) {
    run_affine_test(gpu, 2560, 512, 13, 8, 135_791);
}

/// Two tokens: shorter than one band, which is the smallest chunk the matrix
/// entries ever see, since a single token takes the matvec instead. The tail
/// lanes read the last column over again and are dropped at the write, so a
/// kernel that wrote them would show up here as a row of the wrong answer.
pub fn qmm_4_matches_at_a_chunk_shorter_than_one_band(gpu: &Gpu) {
    run_affine_test(gpu, 256, 128, 2, 4, 314_159);
}

/// `w * scale + bias` uses the bias, checked with every weight zero.
///
/// With zero weights the answer is the bias alone, so a kernel that dropped the
/// bias term answers zero everywhere and a kernel that read it from the wrong
/// group answers somebody else's number. Neither is visible in the cases above,
/// where the bias is a small correction on a large product.
pub fn the_group_bias_is_applied(gpu: &Gpu) {
    let in_dim = 256usize;
    let out_dim = 64usize;
    let groups = in_dim / 64;

    // All zero weights
    let words = vec![0u32; out_dim * (in_dim * 4 / 32)];
    let scales = vec![bf16::from_f32(1.0); out_dim * groups];

    // Non-zero biases
    let mut biases = vec![bf16::ZERO; out_dim * groups];
    for (i, b) in biases.iter_mut().enumerate() {
        *b = bf16::from_f32(f32::midpoint(i as f32, 1.0));
    }

    // One-hot probe x: 1.0 at index 0, 0 everywhere else
    let mut x = vec![0.0f32; in_dim];
    x[0] = 1.0;

    let d_w = gpu.upload(bytemuck::cast_slice::<u32, u8>(&words));
    let d_scales = gpu.upload(bytemuck::cast_slice::<bf16, u8>(&scales));
    let d_biases = gpu.upload(bytemuck::cast_slice::<bf16, u8>(&biases));
    let d_x = gpu.upload(&f32_bytes(&x));
    let d_y = zeros(gpu, out_dim);

    let args = DenseAffineArgs {
        in_dim: in_dim as u32,
        out_dim: out_dim as u32,
        n_tok: 1,
        bits: 4,
    };

    launch(
        gpu,
        "kernel_dense_affine_qmv_4_f32",
        Extent3::linear(out_dim as u32),
        Extent3::linear(32),
        0,
        &[
            KernelArg::Buffer(d_w.view()),
            KernelArg::Buffer(d_scales.view()),
            KernelArg::Buffer(d_biases.view()),
            KernelArg::Buffer(d_x.view()),
            KernelArg::Buffer(d_y.view()),
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
        ],
    );
    drain(gpu);

    let got = gpu.read_f32(&d_y);

    // With zero weights and one-hot x at index 0 (which is in group 0):
    // out[r] = scale[r, 0] * 0 + bias[r, 0] * 1.0 = bias[r, 0]
    for r in 0..out_dim {
        let want = f32::from(biases[r * groups]);
        let diff = (got[r] - want).abs();
        assert!(
            diff < 1e-4,
            "bias mismatch at row {r}: got {}, want {want}",
            got[r]
        );
    }
}

/// Argument block mirroring `dense_args_embed_affine`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DenseEmbedAffineArgs {
    pub hidden: u32,
    pub n_tok: u32,
}

const _: () = assert!(size_of::<DenseEmbedAffineArgs>() == 8);

/// Columns one threadgroup of the gather covers, as `layers::embed` launches it.
const EMBED_BLOCK: u32 = 64;

/// The embedding gather dequantizes the packed row each id names.
///
/// The hidden width is deliberately not a whole number of threadgroups, so the
/// tail block's bounds check is exercised rather than assumed.
pub fn the_affine_gather_dequantizes_the_rows_the_ids_name(gpu: &Gpu) {
    // A hidden width that is not a whole number of threadgroups, so the tail
    // block's bounds check is exercised rather than assumed: 320 columns is
    // five groups and 5.5 threadgroups after the one-thread-per-column launch.
    let (hidden, vocab) = (320usize, 11usize);
    let ids: Vec<i32> = vec![7, 0, 10, 3];
    let n_tok = ids.len();

    let mut lcg = Lcg::new(0x51ae_9f31);
    let words_per_row = hidden / 8;
    let groups_per_row = hidden / 64;
    let mut words = vec![0u32; vocab * words_per_row];
    for w in &mut words {
        *w = lcg.next_u32();
    }
    let mut scales = vec![bf16::ZERO; vocab * groups_per_row];
    for s in &mut scales {
        *s = lcg.next_bf16();
    }
    let mut biases = vec![bf16::ZERO; vocab * groups_per_row];
    for b in &mut biases {
        *b = lcg.next_bf16();
    }

    // The reference: every column is its own nibble, scaled and shifted by the
    // group it falls in. Written per column rather than per group so that a
    // kernel that has the nibble order right and the group order wrong fails.
    let mut want = vec![0.0f32; n_tok * hidden];
    for (t, &id) in ids.iter().enumerate() {
        let row = id as usize;
        for col in 0..hidden {
            let group = col / 64;
            let word = words[row * words_per_row + col / 8];
            let q = (word >> ((col % 8) * 4)) & 0x0f;
            let s = f32::from(scales[row * groups_per_row + group]);
            let b = f32::from(biases[row * groups_per_row + group]);
            want[t * hidden + col] = (q as f32) * s + b;
        }
    }

    let d_words = gpu.upload(bytemuck::cast_slice::<u32, u8>(&words));
    let d_scales = gpu.upload(bytemuck::cast_slice::<bf16, u8>(&scales));
    let d_biases = gpu.upload(bytemuck::cast_slice::<bf16, u8>(&biases));
    let d_ids = gpu.upload(bytemuck::cast_slice::<i32, u8>(&ids));
    let d_out = zeros(gpu, n_tok * hidden);

    let args = DenseEmbedAffineArgs {
        hidden: hidden as u32,
        n_tok: n_tok as u32,
    };
    launch(
        gpu,
        "kernel_dense_embed_affine4_f32",
        Extent3::new((hidden as u32).div_ceil(EMBED_BLOCK), n_tok as u32, 1),
        Extent3::linear(EMBED_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_out.view()),
            KernelArg::Buffer(d_words.view()),
            KernelArg::Buffer(d_scales.view()),
            KernelArg::Buffer(d_biases.view()),
            KernelArg::Buffer(d_ids.view()),
        ],
    );
    drain(gpu);

    check_rows(
        &gpu.read_f32(&d_out),
        &want,
        "kernel_dense_embed_affine4_f32",
    );
}

/// The gather is indexed by id and not by token: the same id twice, two
/// positions apart, produces the same row twice.
pub fn the_affine_gather_reads_one_row_per_token_and_repeats_a_repeated_id(gpu: &Gpu) {
    // The same id twice, two positions apart: a kernel that indexed the table
    // by token rather than by id would answer four different rows here.
    let (hidden, vocab) = (128usize, 5usize);
    let ids: Vec<i32> = vec![4, 1, 4, 0];
    let n_tok = ids.len();

    let mut lcg = Lcg::new(0x00c0_ffee);
    let words_per_row = hidden / 8;
    let groups_per_row = hidden / 64;
    let mut words = vec![0u32; vocab * words_per_row];
    for w in &mut words {
        *w = lcg.next_u32();
    }
    let scales = vec![bf16::from_f32(0.25); vocab * groups_per_row];
    let biases = vec![bf16::from_f32(-1.5); vocab * groups_per_row];

    let d_words = gpu.upload(bytemuck::cast_slice::<u32, u8>(&words));
    let d_scales = gpu.upload(bytemuck::cast_slice::<bf16, u8>(&scales));
    let d_biases = gpu.upload(bytemuck::cast_slice::<bf16, u8>(&biases));
    let d_ids = gpu.upload(bytemuck::cast_slice::<i32, u8>(&ids));
    let d_out = zeros(gpu, n_tok * hidden);

    let args = DenseEmbedAffineArgs {
        hidden: hidden as u32,
        n_tok: n_tok as u32,
    };
    launch(
        gpu,
        "kernel_dense_embed_affine4_f32",
        Extent3::new((hidden as u32).div_ceil(EMBED_BLOCK), n_tok as u32, 1),
        Extent3::linear(EMBED_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_out.view()),
            KernelArg::Buffer(d_words.view()),
            KernelArg::Buffer(d_scales.view()),
            KernelArg::Buffer(d_biases.view()),
            KernelArg::Buffer(d_ids.view()),
        ],
    );
    drain(gpu);

    let got = gpu.read_f32(&d_out);
    assert_eq!(
        got[..hidden],
        got[2 * hidden..3 * hidden],
        "the same id gathered two different rows"
    );
    assert_ne!(
        got[..hidden],
        got[hidden..2 * hidden],
        "two different ids gathered the same row"
    );
}
