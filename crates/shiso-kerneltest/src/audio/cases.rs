//! Device cases for the audio encoder kernel set.
//!
//! Every case compares the device against the plainest reading of its formula,
//! a Rust loop in f32 or f64, and where a structural property pins the
//! behaviour (a masked key is never seen, a query with no key yields zeros, a
//! tile edge writes nothing past it) the case checks that as well as the
//! numbers.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "argument blocks take u32 dimensions while fixtures are usize literals; the oracles average over those small counts and index padded positions signed"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "the oracles use the formulas' own names - q, k, v, i, f, g, o - and renaming them would make them worse transcriptions"
)]

use bytemuck::{Pod, Zeroable};
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::oracle;
use crate::{Gpu, drain, f32_bytes, run};

mod fused;
pub use fused::{
    add_rms_norm_is_the_axpy_and_rms_norm_it_replaces, argmax_wide_matches_the_row_argmax,
    matvec_silu_is_the_products_and_silu_mul, matvec3_is_three_matvecs,
    qk_norm_rope_append_is_the_chain_it_replaces,
};

/// Every entry the set exports, in contract-table order. Each kernel group
/// appends its entries here; the pipeline check resolves exactly this list.
pub const ENTRIES: &[&str] = &[
    "kernel_audio_unary_f32",
    "kernel_audio_axpy_f32",
    "kernel_audio_glu_f32",
    "kernel_audio_layer_norm_f32",
    "kernel_audio_rms_norm_f32",
    "kernel_audio_gemm_f32w",
    "kernel_audio_gemm_bf16w",
    "kernel_audio_gemm_f16w",
    "kernel_audio_gemm_q8_0w",
    "kernel_audio_gemm64_f32w",
    "kernel_audio_gemm64_bf16w",
    "kernel_audio_gemm64_f16w",
    "kernel_audio_gemm64_q8_0w",
    "kernel_audio_gemv_f32w",
    "kernel_audio_gemv_bf16w",
    "kernel_audio_gemv_f16w",
    "kernel_audio_gemv_q8_0w",
    "kernel_audio_split_sum_f32",
    "kernel_audio_conv2d_s2_f32",
    "kernel_audio_dwconv1d_f32",
    "kernel_audio_unfold_f32",
    "kernel_audio_transpose_f32",
    "kernel_audio_window_attn_f32",
    "kernel_audio_rel_scores_f32",
    "kernel_audio_rope_half_f32",
    "kernel_audio_lstm_cell_f32",
    "kernel_audio_greedy_window_f32",
    "kernel_audio_greedy_step_u32",
    "kernel_audio_embed_row_f32",
    "kernel_audio_lstm_cell_gated_f32",
    "kernel_audio_keep_f32",
    "kernel_audio_bias_rows_f32",
    "kernel_audio_argmax_rows_f32",
    "kernel_audio_im2col_s2_f32",
    "kernel_audio_causal_attn_f32",
    "kernel_audio_decode_attn_f32",
    "kernel_audio_decode_attn_merge_f32",
    "kernel_audio_kv_append_f32",
    "kernel_audio_rope_half_state_f32",
    "kernel_audio_decode_step_u32",
    "kernel_audio_embed_token_f32w",
    "kernel_audio_embed_token_bf16w",
    "kernel_audio_embed_token_f16w",
    "kernel_audio_embed_token_q8_0w",
    "kernel_audio_embed_token_affine8w",
    "kernel_audio_silu_mul_f32",
    "kernel_audio_matvec_f32w",
    "kernel_audio_matvec_bf16w",
    "kernel_audio_matvec_f16w",
    "kernel_audio_matvec_q8_0w",
    "kernel_audio_matvec3_f32w",
    "kernel_audio_matvec3_bf16w",
    "kernel_audio_matvec3_f16w",
    "kernel_audio_matvec3_q8_0w",
    "kernel_audio_matvec_silu_f32w",
    "kernel_audio_matvec_silu_bf16w",
    "kernel_audio_matvec_silu_f16w",
    "kernel_audio_matvec_silu_q8_0w",
    "kernel_audio_add_rms_norm_f32",
    "kernel_audio_qk_norm_rope_append_f32",
    "kernel_audio_argmax_split_f32",
    "kernel_audio_argmax_merge_f32",
];

const LINEAR_BLOCK: u32 = 256;
const GEMM_TILE: u32 = 32;
/// Output columns one GEMM threadgroup produces.
const GEMM_TILE_N: u32 = 64;
/// Output columns and rows one matrix-vector threadgroup produces.
const GEMV_TG_COLS: u32 = 16;
const GEMV_TG_ROWS: u32 = 8;
/// Output columns and threads of one matrix-vector-by-runs threadgroup.
const MATVEC_COLS: u32 = 4;
const MATVEC_BLOCK: u32 = 64;
/// Queries one attention simdgroup takes at once.
const ATTN_QUERY_BLOCK: u32 = 8;
/// Distance rows one relative-score simdgroup takes at once.
const REL_ROWS: u32 = 4;
const GEMM_BLOCK: u32 = 128;
const GEMM_SMEM: u32 = 12_288;
/// Rows, threads and threadgroup bytes of the 64-row GEMM.
const GEMM64_TILE_M: u32 = 64;
const GEMM64_BLOCK: u32 = 256;
const GEMM64_SMEM: u32 = 20_480;
const NORM_SMEM: u32 = 132;

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
        KernelRef::new("audio", entry),
        &LaunchArgs {
            grid,
            block,
            smem_bytes,
            args,
        },
    );
}

fn blocks(count: usize, per_block: u32) -> u32 {
    count.div_ceil(per_block as usize) as u32
}

fn zeros(gpu: &Gpu, len: usize) -> DeviceBuffer {
    gpu.upload(&f32_bytes(&vec![0.0f32; len]))
}

/// Asserts agreement within `tol`, relative to max(|want|, 1).
fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what}[{i}]: {g} is not finite");
        assert!(
            (g - w).abs() <= tol * w.abs().max(1.0),
            "{what}[{i}]: got {g}, want {w}"
        );
    }
}

/// `audio_args_unary`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct UnaryArgs {
    count: u32,
    op: u32,
    a: f32,
    b: f32,
}
const _: () = assert!(size_of::<UnaryArgs>() == 16);

/// `audio_args_axpy`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AxpyArgs {
    count: u32,
    alpha: f32,
}
const _: () = assert!(size_of::<AxpyArgs>() == 8);

/// `audio_args_glu`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GluArgs {
    rows: u32,
    channels: u32,
}
const _: () = assert!(size_of::<GluArgs>() == 8);

/// `audio_args_silu_mul`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SiluMulArgs {
    count: u32,
}
const _: () = assert!(size_of::<SiluMulArgs>() == 4);

fn unary_oracle(op: u32, x: f32, a: f32, b: f32) -> f32 {
    match op {
        0 => x.max(0.0),
        1 => x / (1.0 + (-x).exp()),
        2 => 1.0 / (1.0 + (-x).exp()),
        3 => 0.5 * x * (1.0 + libm_erf(x * std::f32::consts::FRAC_1_SQRT_2)),
        4 => x.tanh(),
        5 => x.clamp(a, b),
        6 => x * a,
        _ => unreachable!("op codes are the contract's"),
    }
}

/// erf in f64, which is exact to f32 precision; the kernel's is within 1.5e-7.
fn libm_erf(x: f32) -> f32 {
    // Abramowitz-Stegun 7.1.26 in f64 is within 1.5e-7, below the case tolerance.
    let (s, a) = (x.signum(), f64::from(x.abs()));
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let p = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    (f64::from(s) * (1.0 - p * (-a * a).exp())) as f32
}

/// Each op code of the unary kernel against its own formula, over a range
/// wide enough to reach every activation's saturation.
pub fn every_unary_op_matches_its_formula(gpu: &Gpu) {
    let x: Vec<f32> = oracle::signal(1000, 7).iter().map(|v| v * 6.0).collect();
    let input = gpu.upload(&f32_bytes(&x));
    for op in 0..=6u32 {
        let (a, b) = if op == 5 { (-1.5, 2.0) } else { (0.75, 0.0) };
        let out = zeros(gpu, x.len());
        let args = UnaryArgs {
            count: x.len() as u32,
            op,
            a,
            b,
        };
        launch(
            gpu,
            "kernel_audio_unary_f32",
            Extent3::linear(blocks(x.len(), LINEAR_BLOCK)),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(input.view()),
            ],
        );
        drain(gpu);
        let want: Vec<f32> = x.iter().map(|&v| unary_oracle(op, v, a, b)).collect();
        close(&gpu.read_f32(&out), &want, 2e-6, &format!("unary op {op}"));
    }
}

/// The unary kernel is safe with the output bound to its own input.
pub fn unary_runs_in_place(gpu: &Gpu) {
    let x = oracle::signal(300, 3);
    let buf = gpu.upload(&f32_bytes(&x));
    let args = UnaryArgs {
        count: 300,
        op: 1,
        a: 0.0,
        b: 0.0,
    };
    launch(
        gpu,
        "kernel_audio_unary_f32",
        Extent3::linear(2),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(buf.view()),
            KernelArg::Buffer(buf.view()),
        ],
    );
    drain(gpu);
    let want: Vec<f32> = x.iter().map(|&v| unary_oracle(1, v, 0.0, 0.0)).collect();
    close(&gpu.read_f32(&buf), &want, 2e-6, "in-place silu");
}

/// `y += alpha * x`.
pub fn axpy_adds_a_scaled_copy(gpu: &Gpu) {
    let (y0, x) = (oracle::signal(777, 1), oracle::signal(777, 2));
    let y = gpu.upload(&f32_bytes(&y0));
    let xb = gpu.upload(&f32_bytes(&x));
    let args = AxpyArgs {
        count: 777,
        alpha: 0.5,
    };
    launch(
        gpu,
        "kernel_audio_axpy_f32",
        Extent3::linear(blocks(777, LINEAR_BLOCK)),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(y.view()),
            KernelArg::Buffer(xb.view()),
        ],
    );
    drain(gpu);
    let want: Vec<f32> = y0.iter().zip(&x).map(|(a, b)| a + 0.5 * b).collect();
    close(&gpu.read_f32(&y), &want, 1e-6, "axpy");
}

/// The gated linear unit takes the value from the first half of each row and
/// the gate from the second, not the other way round.
pub fn glu_gates_the_first_half_with_the_second(gpu: &Gpu) {
    let (rows, c) = (5usize, 70usize);
    let x = oracle::signal(rows * 2 * c, 9);
    let input = gpu.upload(&f32_bytes(&x));
    let out = zeros(gpu, rows * c);
    let args = GluArgs {
        rows: rows as u32,
        channels: c as u32,
    };
    launch(
        gpu,
        "kernel_audio_glu_f32",
        Extent3::linear(blocks(rows * c, LINEAR_BLOCK)),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(input.view()),
        ],
    );
    drain(gpu);
    let mut want = Vec::with_capacity(rows * c);
    for r in 0..rows {
        for ch in 0..c {
            let (v, g) = (x[r * 2 * c + ch], x[r * 2 * c + c + ch]);
            want.push(v / (1.0 + (-g).exp()));
        }
    }
    close(&gpu.read_f32(&out), &want, 2e-6, "glu");
}

/// The gated MLP's product takes `silu` of the gate, not of the value,
/// over a count that is not a whole number of blocks, and runs in place on
/// the gate.
pub fn silu_mul_gates_the_up_rows(gpu: &Gpu) {
    let count = 1000usize;
    let gate: Vec<f32> = oracle::signal(count, 11).iter().map(|v| v * 8.0).collect();
    let up = oracle::signal(count, 12);
    let g = gpu.upload(&f32_bytes(&gate));
    let u = gpu.upload(&f32_bytes(&up));
    let args = SiluMulArgs {
        count: count as u32,
    };
    launch(
        gpu,
        "kernel_audio_silu_mul_f32",
        Extent3::linear(blocks(count, LINEAR_BLOCK)),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(g.view()),
            KernelArg::Buffer(g.view()),
            KernelArg::Buffer(u.view()),
        ],
    );
    drain(gpu);
    let want: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&x, &y)| x / (1.0 + (-x).exp()) * y)
        .collect();
    close(&gpu.read_f32(&g), &want, 2e-6, "silu_mul");
}

/// `audio_args_norm`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct NormArgs {
    rows: u32,
    dim: u32,
    eps: f32,
    flags: u32,
}
const _: () = assert!(size_of::<NormArgs>() == 16);

fn layer_norm_oracle(x: &[f32], dim: usize, w: &[f32], b: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(dim) {
        let mean = row.iter().map(|v| f64::from(*v)).sum::<f64>() / dim as f64;
        let var = row
            .iter()
            .map(|v| (f64::from(*v) - mean).powi(2))
            .sum::<f64>()
            / dim as f64;
        let rstd = 1.0 / (var + f64::from(eps)).sqrt();
        for (i, v) in row.iter().enumerate() {
            let y = (f64::from(*v) - mean) * rstd * f64::from(w[i])
                + b.map_or(0.0, |b| f64::from(b[i]));
            out.push(y as f32);
        }
    }
    out
}

fn norm_launch(
    gpu: &Gpu,
    entry: &'static str,
    args: NormArgs,
    out: &DeviceBuffer,
    input: &DeviceBuffer,
    extra: &[&DeviceBuffer],
) {
    let mut kargs = vec![
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(input.view()),
    ];
    kargs.extend(extra.iter().map(|b| KernelArg::Buffer(b.view())));
    launch(
        gpu,
        entry,
        Extent3::linear(args.rows),
        Extent3::linear(LINEAR_BLOCK),
        NORM_SMEM,
        &kargs,
    );
    drain(gpu);
}

/// The layer norm subtracts the mean, scales by the deviation and applies the
/// weight, with and without its bias.
pub fn layer_norm_matches_with_and_without_bias(gpu: &Gpu) {
    let (rows, dim) = (7usize, 1024usize);
    let x = oracle::signal(rows * dim, 11);
    let (w, b) = (oracle::signal(dim, 12), oracle::signal(dim, 13));
    let (xin, wb, bb) = (
        gpu.upload(&f32_bytes(&x)),
        gpu.upload(&f32_bytes(&w)),
        gpu.upload(&f32_bytes(&b)),
    );
    for (flags, bias) in [(2u32, Some(&b[..])), (0u32, None)] {
        let out = zeros(gpu, rows * dim);
        let args = NormArgs {
            rows: rows as u32,
            dim: dim as u32,
            eps: 1e-5,
            flags,
        };
        norm_launch(
            gpu,
            "kernel_audio_layer_norm_f32",
            args,
            &out,
            &xin,
            &[&wb, &bb],
        );
        close(
            &gpu.read_f32(&out),
            &layer_norm_oracle(&x, dim, &w, bias, 1e-5),
            2e-5,
            "layer norm",
        );
    }
}

/// A row with a large mean and a tiny variance keeps its precision: the
/// variance is taken about the mean, not as a difference of squares.
pub fn layer_norm_keeps_precision_on_a_large_mean_small_variance_row(gpu: &Gpu) {
    let dim = 512usize;
    let x: Vec<f32> = oracle::signal(dim, 5)
        .iter()
        .map(|v| 1.0e4 + 1.0e-2 * v)
        .collect();
    let w = vec![1.0f32; dim];
    let (xin, wb) = (gpu.upload(&f32_bytes(&x)), gpu.upload(&f32_bytes(&w)));
    let out = zeros(gpu, dim);
    norm_launch(
        gpu,
        "kernel_audio_layer_norm_f32",
        NormArgs {
            rows: 1,
            dim: dim as u32,
            eps: 1e-5,
            flags: 0,
        },
        &out,
        &xin,
        &[&wb, &wb],
    );
    close(
        &gpu.read_f32(&out),
        &layer_norm_oracle(&x, dim, &w, None, 1e-5),
        5e-3,
        "large-mean layer norm",
    );
}

/// The RMS norm, with its weight and in the scale-free form.
pub fn rms_norm_matches_with_and_without_weight(gpu: &Gpu) {
    let (rows, dim) = (4usize, 1536usize);
    let x = oracle::signal(rows * dim, 21);
    let w = oracle::signal(dim, 22);
    let (xin, wb) = (gpu.upload(&f32_bytes(&x)), gpu.upload(&f32_bytes(&w)));
    for (flags, weight) in [(1u32, Some(&w[..])), (0u32, None)] {
        let out = zeros(gpu, rows * dim);
        norm_launch(
            gpu,
            "kernel_audio_rms_norm_f32",
            NormArgs {
                rows: rows as u32,
                dim: dim as u32,
                eps: 1e-6,
                flags,
            },
            &out,
            &xin,
            &[&wb],
        );
        let want: Vec<f32> = x
            .chunks(dim)
            .flat_map(|r| oracle::rms_norm(r, weight, 1e-6))
            .collect();
        close(&gpu.read_f32(&out), &want, 2e-5, "rms norm");
    }
}

/// `audio_args_gemm`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GemmArgs {
    m: u32,
    n: u32,
    k: u32,
    flags: u32,
    k_chunk: u32,
    splits: u32,
}
const _: () = assert!(size_of::<GemmArgs>() == 24);

fn gemm_oracle(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0f64;
            for i in 0..k {
                acc += f64::from(x[r * k + i]) * f64::from(w[c * k + i]);
            }
            y[r * n + c] = (acc + bias.map_or(0.0, |b| f64::from(b[c]))) as f32;
        }
    }
    y
}

/// Weight bytes and the values the oracle must use, per weight type.
fn weight_fixture(kind: &str, len: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let raw = oracle::signal(len, seed);
    match kind {
        "f32w" => (f32_bytes(&raw), raw),
        "bf16w" => {
            let bits: Vec<u16> = raw.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            (
                bits.iter().flat_map(|b| b.to_le_bytes()).collect(),
                bits.iter()
                    .map(|b| f32::from_bits(u32::from(*b) << 16))
                    .collect(),
            )
        }
        "f16w" => {
            let halves: Vec<half::f16> = raw.iter().map(|v| half::f16::from_f32(*v)).collect();
            (
                halves.iter().flat_map(|h| h.to_le_bytes()).collect(),
                halves.iter().map(|h| h.to_f32()).collect(),
            )
        }
        "q8_0w" => {
            // GGUF Q8_0: blocks of an f16 scale and 32 signed bytes along each row.
            let (mut bytes, mut values) = (Vec::new(), Vec::new());
            for block in raw.chunks(32) {
                let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let d = half::f16::from_f32(amax / 127.0);
                bytes.extend_from_slice(&d.to_le_bytes());
                for v in block {
                    let q = if d.to_f32() == 0.0 {
                        0i8
                    } else {
                        (v / d.to_f32()).round() as i8
                    };
                    bytes.push(q.to_le_bytes()[0]);
                    values.push(f32::from(q) * d.to_f32());
                }
            }
            (bytes, values)
        }
        _ => unreachable!("four weight types"),
    }
}

/// Which multiply a case launches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GemmMode {
    /// The tiled GEMM over all of k.
    Tiled,
    /// The tiled GEMM over `splits` slices of k into partial planes, closed by
    /// the split-sum pass.
    Split(usize),
    /// The matrix-vector kernel.
    Gemv,
    /// The 64-row tiled GEMM over `splits` slices of k (1: all of it).
    Tall(usize),
    /// The matrix-vector-by-runs kernel, one threadgroup row per input row.
    Matvec,
}

fn gemm_entry(kind: &str, mode: GemmMode) -> &'static str {
    match (mode, kind) {
        (GemmMode::Gemv, "q8_0w") => "kernel_audio_gemv_q8_0w",
        (GemmMode::Gemv, "f32w") => "kernel_audio_gemv_f32w",
        (GemmMode::Gemv, "bf16w") => "kernel_audio_gemv_bf16w",
        (GemmMode::Gemv, _) => "kernel_audio_gemv_f16w",
        (GemmMode::Matvec, "q8_0w") => "kernel_audio_matvec_q8_0w",
        (GemmMode::Matvec, "f32w") => "kernel_audio_matvec_f32w",
        (GemmMode::Matvec, "bf16w") => "kernel_audio_matvec_bf16w",
        (GemmMode::Matvec, _) => "kernel_audio_matvec_f16w",
        (GemmMode::Tall(_), "q8_0w") => "kernel_audio_gemm64_q8_0w",
        (GemmMode::Tall(_), "f32w") => "kernel_audio_gemm64_f32w",
        (GemmMode::Tall(_), "bf16w") => "kernel_audio_gemm64_bf16w",
        (GemmMode::Tall(_), _) => "kernel_audio_gemm64_f16w",
        (_, "q8_0w") => "kernel_audio_gemm_q8_0w",
        (_, "f32w") => "kernel_audio_gemm_f32w",
        (_, "bf16w") => "kernel_audio_gemm_bf16w",
        _ => "kernel_audio_gemm_f16w",
    }
}

fn gemm_case(
    gpu: &Gpu,
    kind: &'static str,
    (m, n, k): (usize, usize, usize),
    with_bias: bool,
    mode: GemmMode,
) {
    let x = oracle::signal(m * k, 31);
    let (wbytes, wvals) = weight_fixture(kind, n * k, 32);
    let bias = oracle::signal(n, 33);
    let (xb, wb, bb) = (
        gpu.upload(&f32_bytes(&x)),
        gpu.upload(&wbytes),
        gpu.upload(&f32_bytes(&bias)),
    );
    // One guard row past the end: the kernel must not write it.
    let y = gpu.upload(&f32_bytes(&vec![f32::NAN; m * n + n]));
    let splits = match mode {
        GemmMode::Split(s) | GemmMode::Tall(s) => s,
        GemmMode::Tiled | GemmMode::Gemv | GemmMode::Matvec => 1,
    };
    let k_tiles = k.div_ceil(GEMM_TILE as usize);
    let k_chunk = k_tiles.div_ceil(splits) * GEMM_TILE as usize;
    let splits = k.div_ceil(k_chunk);
    let args = GemmArgs {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        flags: u32::from(with_bias),
        k_chunk: k_chunk as u32,
        splits: splits as u32,
    };
    let entry = gemm_entry(kind, mode);
    let parts = gpu.upload(&f32_bytes(&vec![f32::NAN; splits * m * n]));
    let target = if splits > 1 { &parts } else { &y };
    let (grid, block, smem) = match mode {
        GemmMode::Gemv => (
            Extent3::new(blocks(n, GEMV_TG_COLS), blocks(m, GEMV_TG_ROWS), 1),
            GEMM_BLOCK,
            0,
        ),
        GemmMode::Matvec => (
            Extent3::new(blocks(n, MATVEC_COLS), m as u32, 1),
            MATVEC_BLOCK,
            0,
        ),
        GemmMode::Tall(_) => (
            Extent3::new(
                blocks(n, GEMM_TILE_N),
                blocks(m, GEMM64_TILE_M),
                splits as u32,
            ),
            GEMM64_BLOCK,
            GEMM64_SMEM,
        ),
        GemmMode::Tiled | GemmMode::Split(_) => (
            Extent3::new(blocks(n, GEMM_TILE_N), blocks(m, GEMM_TILE), splits as u32),
            GEMM_BLOCK,
            GEMM_SMEM,
        ),
    };
    launch(
        gpu,
        entry,
        grid,
        Extent3::linear(block),
        smem,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(target.view()),
            KernelArg::Buffer(xb.view()),
            KernelArg::Buffer(wb.view()),
            KernelArg::Buffer(bb.view()),
        ],
    );
    if splits > 1 {
        launch(
            gpu,
            "kernel_audio_split_sum_f32",
            Extent3::linear(blocks(m * n, LINEAR_BLOCK)),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(y.view()),
                KernelArg::Buffer(parts.view()),
                KernelArg::Buffer(bb.view()),
            ],
        );
    }
    drain(gpu);
    let got = gpu.read_f32(&y);
    let want = gemm_oracle(&x, &wvals, with_bias.then_some(&bias[..]), m, n, k);
    close(
        &got[..m * n],
        &want,
        1e-4,
        &format!("gemm {kind} {mode:?} {m}x{n}x{k}"),
    );
    assert!(
        got[m * n..].iter().all(|v| v.is_nan()),
        "gemm {kind} {mode:?}: wrote past the output"
    );
}

/// The tiled GEMM for every weight type over square, ragged, single-row and
/// multi-tile shapes, with and without its bias, writing nothing past `m x n`.
pub fn gemm_matches_for_every_weight_type_and_ragged_shape(gpu: &Gpu) {
    for kind in ["f32w", "bf16w", "f16w"] {
        for &(m, n, k, bias) in &[
            (64usize, 64usize, 64usize, true),
            (37, 5, 33, false),
            (1, 1300, 1152, true),
            (130, 96, 1024, true),
        ] {
            gemm_case(gpu, kind, (m, n, k), bias, GemmMode::Tiled);
        }
    }
    // Q8_0 rows are whole blocks, so its shapes keep k a multiple of 32.
    for &(m, n, k, bias) in &[
        (37usize, 5usize, 64usize, false),
        (1, 1300, 1152, true),
        (130, 96, 1024, true),
    ] {
        gemm_case(gpu, "q8_0w", (m, n, k), bias, GemmMode::Tiled);
    }
}

/// Split k: slices into partial planes, summed with the bias once, match the
/// whole-k product - including a last slice shorter than the others and a
/// k that is not a whole number of tiles.
pub fn split_k_gemm_matches_the_whole_product(gpu: &Gpu) {
    for kind in ["f32w", "bf16w", "f16w"] {
        for &(m, n, k, splits) in &[(19usize, 70usize, 1000usize, 4usize), (3, 64, 97, 3)] {
            gemm_case(gpu, kind, (m, n, k), true, GemmMode::Split(splits));
        }
    }
    gemm_case(gpu, "q8_0w", (19, 70, 1024), true, GemmMode::Split(5));
}

/// The 64-row GEMM for every weight type: a whole tile, rows that end
/// mid-tile, several row tiles, ragged columns and k, and split k.
pub fn tall_gemm_matches_for_every_weight_type(gpu: &Gpu) {
    for kind in ["f32w", "bf16w", "f16w"] {
        for &(m, n, k, bias, splits) in &[
            (64usize, 64usize, 64usize, true, 1usize),
            (33, 5, 33, false, 1),
            (200, 130, 1024, true, 1),
            (130, 70, 1000, true, 4),
        ] {
            gemm_case(gpu, kind, (m, n, k), bias, GemmMode::Tall(splits));
        }
    }
    for &(m, n, k, splits) in &[(200usize, 130usize, 1024usize, 1usize), (37, 96, 1024, 5)] {
        gemm_case(gpu, "q8_0w", (m, n, k), true, GemmMode::Tall(splits));
    }
}

/// The matrix-vector kernel for every weight type over one to a few rows,
/// ragged column counts and a k that is not a multiple of four.
pub fn gemv_matches_for_every_weight_type(gpu: &Gpu) {
    for kind in ["f32w", "bf16w", "f16w"] {
        for &(m, n, k, bias) in &[
            (1usize, 1300usize, 1152usize, true),
            (4, 37, 64, false),
            (3, 18, 33, true),
            (9, 20, 40, true),
        ] {
            gemm_case(gpu, kind, (m, n, k), bias, GemmMode::Gemv);
        }
    }
    for &(m, n, k) in &[(1usize, 1300usize, 1152usize), (4, 37, 64)] {
        gemm_case(gpu, "q8_0w", (m, n, k), true, GemmMode::Gemv);
    }
}

/// The matrix-vector-by-runs kernel for every weight type: one row and a
/// few, column counts that end mid-threadgroup, a k shorter than one
/// simdgroup's sweep and one of several sweeps, with and without the bias,
/// writing nothing past `m x n`.
pub fn matvec_matches_for_every_weight_type(gpu: &Gpu) {
    for kind in ["f32w", "bf16w", "f16w"] {
        for &(m, n, k, bias) in &[
            (1usize, 1301usize, 1152usize, true),
            (3, 37, 64, false),
            (1, 6, 8, true),
            (2, 9, 2056, true),
        ] {
            gemm_case(gpu, kind, (m, n, k), bias, GemmMode::Matvec);
        }
    }
    for &(m, n, k, bias) in &[
        (1usize, 1301usize, 1152usize, true),
        (3, 37, 64, false),
        (1, 5, 2080, true),
    ] {
        gemm_case(gpu, "q8_0w", (m, n, k), bias, GemmMode::Matvec);
    }
}

/// `audio_args_conv2d`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Conv2dArgs {
    t_in: u32,
    f_in: u32,
    c_in: u32,
    t_out: u32,
    f_out: u32,
    c_out: u32,
    pad_t: u32,
    pad_f: u32,
    depthwise: u32,
    has_bias: u32,
    relu: u32,
    in_valid: u32,
    out_valid: u32,
    batch: u32,
}
const _: () = assert!(size_of::<Conv2dArgs>() == 56);

struct ConvCase {
    t_in: usize,
    f_in: usize,
    c_in: usize,
    c_out: usize,
    t_out: usize,
    f_out: usize,
    pad_t: usize,
    pad_f: usize,
    depthwise: bool,
    bias: bool,
    relu: bool,
    in_valid: usize,
    out_valid: usize,
    /// Independent planes, stacked.
    batch: usize,
}

fn conv_oracle(c: &ConvCase, x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    // Each plane is its own image: its first rows are padding, never the
    // previous plane's last.
    x.chunks(c.t_in * c.f_in * c.c_in)
        .flat_map(|x| conv_plane(c, x, w, b))
        .collect()
}

fn conv_plane(c: &ConvCase, x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let per = if c.depthwise { 1 } else { c.c_in };
    let mut out = vec![0.0f32; c.t_out * c.f_out * c.c_out];
    for t in 0..c.t_out {
        if t >= c.out_valid {
            continue;
        }
        for f in 0..c.f_out {
            for co in 0..c.c_out {
                let mut acc = if c.bias { f64::from(b[co]) } else { 0.0 };
                for kt in 0..3 {
                    for kf in 0..3 {
                        let (ti, fi) = (
                            (2 * t + kt) as isize - c.pad_t as isize,
                            (2 * f + kf) as isize - c.pad_f as isize,
                        );
                        if ti < 0
                            || fi < 0
                            || ti as usize >= c.t_in.min(c.in_valid)
                            || fi as usize >= c.f_in
                        {
                            continue;
                        }
                        for g in 0..per {
                            let ci = if c.depthwise { co } else { g };
                            let xv = x[(ti as usize * c.f_in + fi as usize) * c.c_in + ci];
                            acc += f64::from(w[((co * per + g) * 3 + kt) * 3 + kf]) * f64::from(xv);
                        }
                    }
                }
                let v = acc as f32;
                out[(t * c.f_out + f) * c.c_out + co] = if c.relu { v.max(0.0) } else { v };
            }
        }
    }
    out
}

fn conv_run(gpu: &Gpu, c: &ConvCase) {
    let per = if c.depthwise { 1 } else { c.c_in };
    let x = oracle::signal(c.batch * c.t_in * c.f_in * c.c_in, 41);
    let w = oracle::signal(c.c_out * per * 9, 42);
    let b = oracle::signal(c.c_out, 43);
    let (xb, wb, bb) = (
        gpu.upload(&f32_bytes(&x)),
        gpu.upload(&f32_bytes(&w)),
        gpu.upload(&f32_bytes(&b)),
    );
    let total = c.batch * c.t_out * c.f_out * c.c_out;
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; total]));
    let args = Conv2dArgs {
        t_in: c.t_in as u32,
        f_in: c.f_in as u32,
        c_in: c.c_in as u32,
        t_out: c.t_out as u32,
        f_out: c.f_out as u32,
        c_out: c.c_out as u32,
        pad_t: c.pad_t as u32,
        pad_f: c.pad_f as u32,
        depthwise: u32::from(c.depthwise),
        has_bias: u32::from(c.bias),
        relu: u32::from(c.relu),
        in_valid: c.in_valid as u32,
        out_valid: c.out_valid as u32,
        batch: c.batch as u32,
    };
    launch(
        gpu,
        "kernel_audio_conv2d_s2_f32",
        Extent3::linear(blocks(total, LINEAR_BLOCK)),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(xb.view()),
            KernelArg::Buffer(wb.view()),
            KernelArg::Buffer(bb.view()),
        ],
    );
    drain(gpu);
    close(
        &gpu.read_f32(&out),
        &conv_oracle(c, &x, &w, &b),
        2e-5,
        "conv2d",
    );
}

/// The stride-2 conv2d against a zero-padded direct oracle: causal and
/// symmetric padding, full and depthwise, odd widths and valid-row cuts on
/// both sides.
pub fn conv2d_matches_causal_symmetric_full_and_depthwise(gpu: &Gpu) {
    // Causal time (2 left, 1 implicit right), freq (2,1), 1 -> 16 channels, ReLU, a masked tail.
    conv_run(
        gpu,
        &ConvCase {
            t_in: 21,
            f_in: 128,
            c_in: 1,
            c_out: 16,
            t_out: 11,
            f_out: 65,
            pad_t: 2,
            pad_f: 2,
            depthwise: false,
            bias: true,
            relu: true,
            in_valid: 21,
            out_valid: 9,
            batch: 1,
        },
    );
    // Depthwise over an odd freq width with an input valid-row cut.
    conv_run(
        gpu,
        &ConvCase {
            t_in: 11,
            f_in: 65,
            c_in: 16,
            c_out: 16,
            t_out: 6,
            f_out: 33,
            pad_t: 2,
            pad_f: 2,
            depthwise: true,
            bias: true,
            relu: false,
            in_valid: 8,
            out_valid: 6,
            batch: 1,
        },
    );
    // Symmetric pad 1, no bias, full 8 -> 4 channels (the second shape of a two-stage stack).
    conv_run(
        gpu,
        &ConvCase {
            t_in: 13,
            f_in: 64,
            c_in: 8,
            c_out: 4,
            t_out: 7,
            f_out: 32,
            pad_t: 1,
            pad_f: 1,
            depthwise: false,
            bias: false,
            relu: false,
            in_valid: 13,
            out_valid: 7,
            batch: 1,
        },
    );
}

/// A stack of planes convolves each on its own: a plane's first output row
/// reads zero padding, not the previous plane's last rows, and the valid-row
/// cuts apply within every plane.
pub fn conv2d_keeps_batched_planes_apart(gpu: &Gpu) {
    // Symmetric pad 1, full 1 -> 6 channels over five planes of an odd height.
    conv_run(
        gpu,
        &ConvCase {
            t_in: 9,
            f_in: 20,
            c_in: 1,
            c_out: 6,
            t_out: 5,
            f_out: 10,
            pad_t: 1,
            pad_f: 1,
            depthwise: false,
            bias: true,
            relu: false,
            in_valid: 9,
            out_valid: 5,
            batch: 5,
        },
    );
    // Full 6 -> 5 channels over three planes with both cuts inside each.
    conv_run(
        gpu,
        &ConvCase {
            t_in: 10,
            f_in: 11,
            c_in: 6,
            c_out: 5,
            t_out: 5,
            f_out: 6,
            pad_t: 1,
            pad_f: 1,
            depthwise: false,
            bias: true,
            relu: true,
            in_valid: 7,
            out_valid: 4,
            batch: 3,
        },
    );
}

/// The stride-2 conv2d's patches: every plane, tap and channel lands where
/// the GEMM reads it, zero at the pads, past the input cut and on output rows
/// past the output cut; and patches times the kernel is the direct conv.
///
/// # Panics
///
/// When a patch value differs, or the product differs from the direct conv.
pub fn im2col_lays_out_the_patches_of_every_plane(gpu: &Gpu) {
    let c = ConvCase {
        t_in: 11,
        f_in: 13,
        c_in: 5,
        c_out: 7,
        t_out: 6,
        f_out: 7,
        pad_t: 1,
        pad_f: 1,
        depthwise: false,
        bias: true,
        relu: false,
        in_valid: 9,
        out_valid: 5,
        batch: 3,
    };
    let x = oracle::signal(c.batch * c.t_in * c.f_in * c.c_in, 61);
    let taps = 9 * c.c_in;
    let total = c.batch * c.t_out * c.f_out * taps;
    let mut want = vec![0.0f32; total];
    for b in 0..c.batch {
        for t in 0..c.out_valid {
            for f in 0..c.f_out {
                for kt in 0..3 {
                    for kf in 0..3 {
                        let (ti, fi) = (
                            (2 * t + kt) as isize - c.pad_t as isize,
                            (2 * f + kf) as isize - c.pad_f as isize,
                        );
                        if ti < 0 || fi < 0 || ti as usize >= c.in_valid || fi as usize >= c.f_in {
                            continue;
                        }
                        for ci in 0..c.c_in {
                            want[((b * c.t_out + t) * c.f_out + f) * taps
                                + (kt * 3 + kf) * c.c_in
                                + ci] = x
                                [((b * c.t_in + ti as usize) * c.f_in + fi as usize) * c.c_in + ci];
                        }
                    }
                }
            }
        }
    }
    let xb = gpu.upload(&f32_bytes(&x));
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; total]));
    let args = Conv2dArgs {
        t_in: c.t_in as u32,
        f_in: c.f_in as u32,
        c_in: c.c_in as u32,
        t_out: c.t_out as u32,
        f_out: c.f_out as u32,
        c_out: c.c_out as u32,
        pad_t: c.pad_t as u32,
        pad_f: c.pad_f as u32,
        depthwise: 0,
        has_bias: 0,
        relu: 0,
        in_valid: c.in_valid as u32,
        out_valid: c.out_valid as u32,
        batch: c.batch as u32,
    };
    launch(
        gpu,
        "kernel_audio_im2col_s2_f32",
        Extent3::linear(blocks(total, LINEAR_BLOCK)),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(xb.view()),
        ],
    );
    drain(gpu);
    let got = gpu.read_f32(&out);
    // A copy: exact.
    assert_eq!(got, want, "im2col");
    // The patches against the kernel, reordered to `[c_out][tap][c_in]`, are
    // the direct conv on every valid row (without the bias, which the conv
    // writes on rows the patches leave zero).
    let w = oracle::signal(c.c_out * c.c_in * 9, 62);
    let direct = conv_oracle(&ConvCase { bias: false, ..c }, &x, &w, &[]);
    for (row, patch) in got.chunks(taps).enumerate() {
        for co in 0..c.c_out {
            let mut acc = 0.0f64;
            for tap in 0..9 {
                for ci in 0..c.c_in {
                    acc += f64::from(w[(co * c.c_in + ci) * 9 + tap])
                        * f64::from(patch[tap * c.c_in + ci]);
                }
            }
            let d = direct[row * c.c_out + co];
            assert!(
                (acc as f32 - d).abs() <= 1e-5 * d.abs().max(1.0),
                "row {row} channel {co}: {acc} against {d}"
            );
        }
    }
}

/// `audio_args_dwconv1d`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DwConv1dArgs {
    rows: u32,
    channels: u32,
    taps: u32,
}
const _: () = assert!(size_of::<DwConv1dArgs>() == 12);

/// `audio_args_unfold`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct UnfoldArgs {
    rows: u32,
    channels: u32,
    taps: u32,
    pad: u32,
}
const _: () = assert!(size_of::<UnfoldArgs>() == 16);

/// `audio_args_transpose`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct TransposeArgs {
    batch: u32,
    rows: u32,
    cols: u32,
}
const _: () = assert!(size_of::<TransposeArgs>() == 12);

/// The depthwise conv1d reads `taps` rows starting at its own, over an input
/// the caller already left-padded.
pub fn dwconv1d_sums_taps_over_pre_padded_rows(gpu: &Gpu) {
    for &(rows, c, taps) in &[(20usize, 96usize, 9usize), (7, 1024, 5), (1, 64, 9)] {
        let x = oracle::signal((rows + taps - 1) * c, 51);
        let w = oracle::signal(c * taps, 52);
        let (xb, wb) = (gpu.upload(&f32_bytes(&x)), gpu.upload(&f32_bytes(&w)));
        let out = zeros(gpu, rows * c);
        let args = DwConv1dArgs {
            rows: rows as u32,
            channels: c as u32,
            taps: taps as u32,
        };
        launch(
            gpu,
            "kernel_audio_dwconv1d_f32",
            Extent3::linear(blocks(rows * c, LINEAR_BLOCK)),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(xb.view()),
                KernelArg::Buffer(wb.view()),
            ],
        );
        drain(gpu);
        let mut want = vec![0.0f32; rows * c];
        for t in 0..rows {
            for ch in 0..c {
                want[t * c + ch] = (0..taps)
                    .map(|j| w[ch * taps + j] * x[(t + j) * c + ch])
                    .sum();
            }
        }
        close(&gpu.read_f32(&out), &want, 2e-5, "dwconv1d");
    }
}

/// Unfold stacks each row with its neighbours, zero past either edge.
pub fn unfold_stacks_neighbouring_rows_with_zero_edges(gpu: &Gpu) {
    let (rows, c, taps, pad) = (9usize, 5usize, 3usize, 1usize);
    let x = oracle::signal(rows * c, 61);
    let xb = gpu.upload(&f32_bytes(&x));
    let out = zeros(gpu, rows * taps * c);
    let args = UnfoldArgs {
        rows: rows as u32,
        channels: c as u32,
        taps: taps as u32,
        pad: pad as u32,
    };
    launch(
        gpu,
        "kernel_audio_unfold_f32",
        Extent3::linear(blocks(rows * taps * c, LINEAR_BLOCK)),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(xb.view()),
        ],
    );
    drain(gpu);
    let mut want = vec![0.0f32; rows * taps * c];
    for t in 0..rows {
        for j in 0..taps {
            for ch in 0..c {
                let src = t as isize + j as isize - pad as isize;
                want[t * taps * c + j * c + ch] = if src < 0 || src as usize >= rows {
                    0.0
                } else {
                    x[src as usize * c + ch]
                };
            }
        }
    }
    close(&gpu.read_f32(&out), &want, 0.0, "unfold");
}

/// The batched transpose swaps the inner two axes and leaves the batch alone.
pub fn transpose_swaps_the_inner_two_axes(gpu: &Gpu) {
    let (b, r, c) = (3usize, 17usize, 256usize);
    let x = oracle::signal(b * r * c, 71);
    let xb = gpu.upload(&f32_bytes(&x));
    let out = zeros(gpu, b * r * c);
    let args = TransposeArgs {
        batch: b as u32,
        rows: r as u32,
        cols: c as u32,
    };
    launch(
        gpu,
        "kernel_audio_transpose_f32",
        Extent3::linear(blocks(b * r * c, LINEAR_BLOCK)),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(xb.view()),
        ],
    );
    drain(gpu);
    let mut want = vec![0.0f32; b * r * c];
    for bb in 0..b {
        for rr in 0..r {
            for cc in 0..c {
                want[(bb * c + cc) * r + rr] = x[(bb * r + rr) * c + cc];
            }
        }
    }
    close(&gpu.read_f32(&out), &want, 0.0, "transpose");
}

/// `audio_args_window_attn`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct WindowAttnArgs {
    n_q: u32,
    n_k: u32,
    heads: u32,
    head_dim: u32,
    q_abs0: u32,
    k_abs0: u32,
    k_end: u32,
    chunk: u32,
    left_chunks: u32,
    d_min: i32,
    n_d: u32,
    scale: f32,
    ac_scale: f32,
    softcap: f32,
    has_bias_uv: u32,
    has_q_dim_scale: u32,
}
const _: () = assert!(size_of::<WindowAttnArgs>() == 64);

/// `audio_args_rope`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RopeArgs {
    rows: u32,
    heads: u32,
    head_dim: u32,
    pos0: u32,
    theta: f32,
}
const _: () = assert!(size_of::<RopeArgs>() == 20);

struct AttnCase {
    n_q: usize,
    n_k: usize,
    heads: usize,
    hd: usize,
    q_abs0: usize,
    k_abs0: usize,
    k_end: usize,
    chunk: usize,
    left_chunks: usize,
    d_min: i32,
    n_d: usize,
    scale: f32,
    ac_scale: f32,
    softcap: f32,
    bias_uv: bool,
    qds: bool,
}

/// The tensors one attention case runs over.
struct AttnFixture {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    bu: Vec<f32>,
    bv: Vec<f32>,
    qds: Vec<f32>,
    rel: Vec<f32>,
}

fn attn_oracle(c: &AttnCase, fx: &AttnFixture) -> Vec<f32> {
    let AttnFixture {
        q,
        k,
        v,
        bu,
        bv,
        qds,
        rel,
    } = fx;
    let w = c.heads * c.hd;
    let mut out = vec![0.0f32; c.n_q * w];
    for i in 0..c.n_q {
        let qa = c.q_abs0 + i;
        for h in 0..c.heads {
            let mut scores = Vec::new();
            for j in 0..c.n_k {
                let ka = c.k_abs0 + j;
                let (qc, kc) = (qa / c.chunk, ka / c.chunk);
                if ka >= c.k_end || kc > qc || qc - kc > c.left_chunks {
                    continue;
                }
                let d = qa as i64 - ka as i64;
                let (mut ac, mut bd) = (0.0f64, 0.0f64);
                for e in 0..c.hd {
                    let qv = f64::from(q[i * w + h * c.hd + e])
                        * if c.qds { f64::from(qds[e]) } else { 1.0 };
                    let (u, vb) = if c.bias_uv {
                        (f64::from(bu[h * c.hd + e]), f64::from(bv[h * c.hd + e]))
                    } else {
                        (0.0, 0.0)
                    };
                    ac += (qv + u) * f64::from(k[j * w + h * c.hd + e]);
                    let r = d - i64::from(c.d_min);
                    if c.n_d > 0 && r >= 0 && (r as usize) < c.n_d {
                        bd += (qv + vb) * f64::from(rel[r as usize * w + h * c.hd + e]);
                    }
                }
                let mut s = f64::from(c.scale) * (f64::from(c.ac_scale) * ac + bd);
                if c.softcap > 0.0 {
                    s = f64::from(c.softcap) * (s / f64::from(c.softcap)).tanh();
                }
                scores.push((j, s));
            }
            if scores.is_empty() {
                continue;
            }
            let mx = scores.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = scores.iter().map(|x| (x.1 - mx).exp()).sum();
            for e in 0..c.hd {
                let acc: f64 = scores
                    .iter()
                    .map(|&(j, s)| (s - mx).exp() / z * f64::from(v[j * w + h * c.hd + e]))
                    .sum();
                out[i * w + h * c.hd + e] = acc as f32;
            }
        }
    }
    out
}

fn attn_run(gpu: &Gpu, c: &AttnCase) {
    let w = c.heads * c.hd;
    let (q, k, v) = (
        oracle::signal(c.n_q * w, 81),
        oracle::signal(c.n_k * w, 82),
        oracle::signal(c.n_k * w, 83),
    );
    let (bu, bv) = (oracle::signal(w, 84), oracle::signal(w, 85));
    let qds: Vec<f32> = oracle::signal(c.hd, 86)
        .iter()
        .map(|x| 0.5 + x.abs())
        .collect();
    let rel = oracle::signal(c.n_d.max(1) * w, 87);
    let up = |x: &[f32]| gpu.upload(&f32_bytes(x));
    let (qb, kb, vb, bub, bvb, qdb, rb) =
        (up(&q), up(&k), up(&v), up(&bu), up(&bv), up(&qds), up(&rel));
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; c.n_q * w]));
    let args = WindowAttnArgs {
        n_q: c.n_q as u32,
        n_k: c.n_k as u32,
        heads: c.heads as u32,
        head_dim: c.hd as u32,
        q_abs0: c.q_abs0 as u32,
        k_abs0: c.k_abs0 as u32,
        k_end: c.k_end as u32,
        chunk: c.chunk as u32,
        left_chunks: c.left_chunks as u32,
        d_min: c.d_min,
        n_d: c.n_d as u32,
        scale: c.scale,
        ac_scale: c.ac_scale,
        softcap: c.softcap,
        has_bias_uv: u32::from(c.bias_uv),
        has_q_dim_scale: u32::from(c.qds),
    };
    // The relative term is scored first, once per (query, head, distance).
    let scores = gpu.upload(&f32_bytes(&vec![
        0.0f32;
        c.n_q.max(1) * c.heads * c.n_d.max(1)
    ]));
    if c.n_d > 0 {
        launch(
            gpu,
            "kernel_audio_rel_scores_f32",
            Extent3::new(
                blocks(c.n_d, REL_ROWS),
                c.heads as u32,
                blocks(c.n_q, ATTN_QUERY_BLOCK),
            ),
            Extent3::linear(32),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(scores.view()),
                KernelArg::Buffer(qb.view()),
                KernelArg::Buffer(bvb.view()),
                KernelArg::Buffer(qdb.view()),
                KernelArg::Buffer(rb.view()),
            ],
        );
    }
    launch(
        gpu,
        "kernel_audio_window_attn_f32",
        Extent3::new(c.heads as u32, blocks(c.n_q, ATTN_QUERY_BLOCK), 1),
        Extent3::linear(32),
        ATTN_QUERY_BLOCK * c.hd as u32 * 4,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(qb.view()),
            KernelArg::Buffer(kb.view()),
            KernelArg::Buffer(vb.view()),
            KernelArg::Buffer(bub.view()),
            KernelArg::Buffer(qdb.view()),
            KernelArg::Buffer(scores.view()),
        ],
    );
    drain(gpu);
    let fx = AttnFixture {
        q,
        k,
        v,
        bu,
        bv,
        qds,
        rel,
    };
    close(
        &gpu.read_f32(&out),
        &attn_oracle(c, &fx),
        5e-5,
        "window attention",
    );
}

/// Chunked-limited attention with a relative-position term and per-head
/// content and position biases, offline and as a streaming step over a cache.
pub fn window_attention_chunked_relpos_matches(gpu: &Gpu) {
    // Offline chunked-limited, R = 3: chunk 4, 14 left chunks, rel rows for d in [-3, 59].
    attn_run(
        gpu,
        &AttnCase {
            n_q: 90,
            n_k: 90,
            heads: 2,
            hd: 128,
            q_abs0: 0,
            k_abs0: 0,
            k_end: 90,
            chunk: 4,
            left_chunks: 14,
            d_min: -3,
            n_d: 63,
            scale: 128f32.powf(-0.5),
            ac_scale: 1.0,
            softcap: 0.0,
            bias_uv: true,
            qds: false,
        },
    );
    // Streaming step: 4 queries at absolute 60.., keys = 56 cached + 4 new starting at absolute 8.
    attn_run(
        gpu,
        &AttnCase {
            n_q: 4,
            n_k: 60,
            heads: 2,
            hd: 128,
            q_abs0: 60,
            k_abs0: 8,
            k_end: 64,
            chunk: 4,
            left_chunks: 14,
            d_min: -3,
            n_d: 63,
            scale: 128f32.powf(-0.5),
            ac_scale: 1.0,
            softcap: 0.0,
            bias_uv: true,
            qds: false,
        },
    );
}

/// A 12-key sliding window with a per-dimension query scale, a key-only
/// content scale and a tanh softcap.
pub fn window_attention_sliding_softcapped_matches(gpu: &Gpu) {
    // Sliding 12-key window (chunk 1, 11 left), per-dim query scale, k-only scale on ac, softcap 50.
    attn_run(
        gpu,
        &AttnCase {
            n_q: 30,
            n_k: 30,
            heads: 2,
            hd: 128,
            q_abs0: 0,
            k_abs0: 0,
            k_end: 30,
            chunk: 1,
            left_chunks: 11,
            d_min: 0,
            n_d: 13,
            scale: 1.0,
            ac_scale: 1.894_635_3,
            softcap: 50.0,
            bias_uv: false,
            qds: true,
        },
    );
}

/// Plain bidirectional attention over the whole sequence.
pub fn window_attention_full_bidirectional_matches(gpu: &Gpu) {
    attn_run(
        gpu,
        &AttnCase {
            n_q: 70,
            n_k: 70,
            heads: 3,
            hd: 64,
            q_abs0: 0,
            k_abs0: 0,
            k_end: 70,
            chunk: 70,
            left_chunks: 0,
            d_min: 0,
            n_d: 0,
            scale: 0.125,
            ac_scale: 1.0,
            softcap: 0.0,
            bias_uv: false,
            qds: false,
        },
    );
}

/// A query that sees no key writes zeros, not NaN.
pub fn a_query_with_no_visible_key_yields_zeros(gpu: &Gpu) {
    // k_end at 0: nothing is visible to anyone.
    attn_run(
        gpu,
        &AttnCase {
            n_q: 3,
            n_k: 8,
            heads: 1,
            hd: 64,
            q_abs0: 0,
            k_abs0: 0,
            k_end: 0,
            chunk: 4,
            left_chunks: 1,
            d_min: 0,
            n_d: 0,
            scale: 0.125,
            ac_scale: 1.0,
            softcap: 0.0,
            bias_uv: false,
            qds: false,
        },
    );
}

/// The half-split rotary embedding pairs dimension `m` with `m + hd/2`.
pub fn rope_half_rotates_pairs_across_the_halves(gpu: &Gpu) {
    let (rows, heads, hd, pos0, theta) = (9usize, 2usize, 64usize, 5u32, 10_000.0f32);
    let x = oracle::signal(rows * heads * hd, 91);
    let buf = gpu.upload(&f32_bytes(&x));
    let args = RopeArgs {
        rows: rows as u32,
        heads: heads as u32,
        head_dim: hd as u32,
        pos0,
        theta,
    };
    let pairs = rows * heads * hd / 2;
    launch(
        gpu,
        "kernel_audio_rope_half_f32",
        Extent3::linear(blocks(pairs, LINEAR_BLOCK)),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(buf.view()),
        ],
    );
    drain(gpu);
    let mut want = x.clone();
    let half = hd / 2;
    for r in 0..rows {
        for h in 0..heads {
            for m in 0..half {
                let ang =
                    f64::from(pos0 + r as u32) * f64::from(theta).powf(-2.0 * m as f64 / hd as f64);
                let base = (r * heads + h) * hd;
                let (a, b) = (f64::from(x[base + m]), f64::from(x[base + m + half]));
                want[base + m] = (a * ang.cos() - b * ang.sin()) as f32;
                want[base + m + half] = (b * ang.cos() + a * ang.sin()) as f32;
            }
        }
    }
    close(&gpu.read_f32(&buf), &want, 2e-5, "rope half");
}

/// `audio_args_lstm`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LstmArgs {
    hidden: u32,
}
const _: () = assert!(size_of::<LstmArgs>() == 4);

/// The LSTM cell reads its gates in (i, f, g, o) order and updates the cell
/// state in place.
pub fn lstm_cell_matches_the_gate_equations(gpu: &Gpu) {
    let hdim = 640usize;
    let gates: Vec<f32> = oracle::signal(4 * hdim, 101)
        .iter()
        .map(|v| v * 3.0)
        .collect();
    let c0 = oracle::signal(hdim, 102);
    let (gb, cb, hb) = (
        gpu.upload(&f32_bytes(&gates)),
        gpu.upload(&f32_bytes(&c0)),
        zeros(gpu, hdim),
    );
    let args = LstmArgs {
        hidden: hdim as u32,
    };
    launch(
        gpu,
        "kernel_audio_lstm_cell_f32",
        Extent3::linear(blocks(hdim, LINEAR_BLOCK)),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(hb.view()),
            KernelArg::Buffer(cb.view()),
            KernelArg::Buffer(gb.view()),
        ],
    );
    drain(gpu);
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let (mut want_c, mut want_h) = (vec![0.0f32; hdim], vec![0.0f32; hdim]);
    for u in 0..hdim {
        let (i, f, g, o) = (
            gates[u],
            gates[hdim + u],
            gates[2 * hdim + u],
            gates[3 * hdim + u],
        );
        want_c[u] = sig(f) * c0[u] + sig(i) * g.tanh();
        want_h[u] = sig(o) * want_c[u].tanh();
    }
    close(&gpu.read_f32(&cb), &want_c, 2e-6, "lstm c");
    close(&gpu.read_f32(&hb), &want_h, 2e-6, "lstm h");
}

/// `audio_args_bias_rows`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BiasRowsArgs {
    rows: u32,
    cols: u32,
}
const _: () = assert!(size_of::<BiasRowsArgs>() == 8);

/// The bias add broadcasts one row over every row and adds, not overwrites.
pub fn bias_rows_adds_one_row_to_every_row(gpu: &Gpu) {
    let (rows, cols) = (37usize, 640usize);
    let (y0, b) = (oracle::signal(rows * cols, 111), oracle::signal(cols, 112));
    let (y, bb) = (gpu.upload(&f32_bytes(&y0)), gpu.upload(&f32_bytes(&b)));
    let args = BiasRowsArgs {
        rows: rows as u32,
        cols: cols as u32,
    };
    launch(
        gpu,
        "kernel_audio_bias_rows_f32",
        Extent3::linear(blocks(rows * cols, LINEAR_BLOCK)),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(y.view()),
            KernelArg::Buffer(bb.view()),
        ],
    );
    drain(gpu);
    let want: Vec<f32> = y0
        .iter()
        .enumerate()
        .map(|(i, v)| v + b[i % cols])
        .collect();
    close(&gpu.read_f32(&y), &want, 1e-6, "bias rows");
}

/// `audio_args_argmax`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ArgmaxArgs {
    rows: u32,
    cols: u32,
}
const _: () = assert!(size_of::<ArgmaxArgs>() == 8);

/// Threads one argmax threadgroup spreads a row over.
const ARGMAX_BLOCK: u32 = 256;

/// The row argmax is the first index of the largest value: a tie keeps the
/// lowest index, a NaN never wins, and a row with nothing above negative
/// infinity answers 0.
///
/// # Panics
///
/// When an id differs from the strictly-greater scan's.
pub fn argmax_rows_keeps_the_lowest_index_of_the_largest(gpu: &Gpu) {
    let (rows, cols) = (6usize, 16_385usize);
    let mut x = oracle::signal(rows * cols, 121);
    // Row 1: a tie between two columns far apart, in different threads.
    x[cols + 9000] = 50.0;
    x[cols + 700] = 50.0;
    // Row 2: a NaN at the front and the maximum at the very end.
    x[2 * cols] = f32::NAN;
    x[3 * cols - 1] = 60.0;
    // Row 3: nothing above negative infinity.
    x[3 * cols..4 * cols].fill(f32::NEG_INFINITY);
    // Row 4: a tie inside one thread's stride.
    x[4 * cols + 3] = 70.0;
    x[4 * cols + 3 + ARGMAX_BLOCK as usize] = 70.0;
    let input = gpu.upload(&f32_bytes(&x));
    let out = zeros(gpu, rows);
    let args = ArgmaxArgs {
        rows: rows as u32,
        cols: cols as u32,
    };
    launch(
        gpu,
        "kernel_audio_argmax_rows_f32",
        Extent3::linear(rows as u32),
        Extent3::linear(ARGMAX_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(input.view()),
        ],
    );
    drain(gpu);
    let want: Vec<u32> = x
        .chunks_exact(cols)
        .map(|r| {
            let mut best = (0u32, f32::NEG_INFINITY);
            for (i, &v) in r.iter().enumerate() {
                if v > best.1 {
                    best = (i as u32, v);
                }
            }
            best.0
        })
        .collect();
    assert_eq!(want[1], 700);
    assert_eq!(want[3], 0);
    let got: Vec<u32> = gpu.read_f32(&out).iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, want);
}

/// `audio_args_greedy_step`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GreedyStepArgs {
    rows: u32,
    avail: u32,
    blank: u32,
    max_symbols: u32,
    slot: u32,
}
const _: () = assert!(size_of::<GreedyStepArgs>() == 20);

/// `audio_args_greedy_window`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GreedyWindowArgs {
    rows: u32,
    cols: u32,
    avail: u32,
}
const _: () = assert!(size_of::<GreedyWindowArgs>() == 12);

/// `audio_args_embed_row`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct EmbedRowArgs {
    width: u32,
    vocab: u32,
}
const _: () = assert!(size_of::<EmbedRowArgs>() == 8);

/// `audio_args_keep`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct KeepArgs {
    count: u32,
}
const _: () = assert!(size_of::<KeepArgs>() == 4);

const GREEDY_BLANK: u32 = 99;
const GREEDY_NONE: u32 = u32::MAX;

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn read_u32(gpu: &Gpu, buffer: &DeviceBuffer) -> Vec<u32> {
    gpu.read_f32(buffer).iter().map(|v| v.to_bits()).collect()
}

/// Greedy search's state as the kernels keep it: frame, symbols on it, done,
/// last token, emitted.
fn greedy_state(gpu: &Gpu, t: u32, sym: u32, token: u32, emit: u32) -> DeviceBuffer {
    gpu.upload(&u32_bytes(&[t, sym, 0, token, emit, 0, 0, 0]))
}

/// Runs the step kernel against a scripted joint until it is done: at frame
/// `f` the joint's argmax is the next token `frames[f]` still owes, blank once
/// it owes none, and emitting a token pays it. `avail` grows through `stops`,
/// as frames arrive in pieces. Returns `(token, frame)` in emission order.
fn greedy_run(gpu: &Gpu, frames: &[Vec<u32>], rows: usize, stops: &[usize]) -> Vec<(u32, u32)> {
    let mut progress = vec![0usize; frames.len()];
    let state = greedy_state(gpu, 0, 0, GREEDY_BLANK, 0);
    let out = gpu.upload(&u32_bytes(&[0, 0]));
    let mut emitted = Vec::new();
    for &avail in stops {
        for _ in 0..4 * frames.len() * 12 {
            let s = read_u32(gpu, &state);
            let ids: Vec<u32> = (0..rows)
                .map(|r| {
                    let f = s[0] as usize + r;
                    frames
                        .get(f)
                        .and_then(|owed| owed.get(progress[f]))
                        .copied()
                        .unwrap_or(GREEDY_BLANK)
                })
                .collect();
            let ids = gpu.upload(&u32_bytes(&ids));
            let args = GreedyStepArgs {
                rows: rows as u32,
                avail: avail as u32,
                blank: GREEDY_BLANK,
                max_symbols: 10,
                slot: 0,
            };
            launch(
                gpu,
                "kernel_audio_greedy_step_u32",
                Extent3::linear(1),
                Extent3::linear(1),
                0,
                &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(state.view()),
                    KernelArg::Buffer(out.view()),
                    KernelArg::Buffer(ids.view()),
                ],
            );
            drain(gpu);
            let (o, s) = (read_u32(gpu, &out), read_u32(gpu, &state));
            if o[0] == GREEDY_NONE {
                assert_eq!(s[4], 0, "a blank step lowers the flag");
            } else {
                assert_eq!(s[4], 1, "an emitting step raises the flag");
                assert_eq!(s[3], o[0], "and records its token");
                progress[o[1] as usize] += 1;
                emitted.push((o[0], o[1]));
            }
            if s[2] != 0 {
                break;
            }
        }
    }
    emitted
}

/// The step kernel is greedy search: blank frames are skipped, a token keeps
/// its frame, a frame gives at most ten symbols, the window's width changes
/// nothing, and a search stopped at the last frame resumes where it stopped.
///
/// # Panics
///
/// When a step decides differently from the rules.
pub fn greedy_step_follows_the_search_rules(gpu: &Gpu) {
    let frames = vec![vec![], vec![5], vec![], vec![], vec![6, 7], vec![]];
    assert_eq!(
        greedy_run(gpu, &frames, 4, &[6]),
        vec![(5, 1), (6, 4), (7, 4)]
    );
    let many = vec![(0..11).collect::<Vec<u32>>(), vec![], vec![]];
    let got = greedy_run(gpu, &many, 2, &[3]);
    assert_eq!(got.len(), 10, "{got:?}");
    assert!(got.iter().all(|(_, f)| *f == 0), "{got:?}");
    let frames = vec![
        vec![],
        vec![],
        vec![1],
        vec![],
        vec![2, 3],
        vec![],
        vec![],
        vec![4],
    ];
    let reference = greedy_run(gpu, &frames, 1, &[8]);
    assert_eq!(reference, vec![(1, 2), (2, 4), (3, 4), (4, 7)]);
    for rows in [2usize, 3, 8, 64] {
        assert_eq!(
            greedy_run(gpu, &frames, rows, &[8]),
            reference,
            "rows {rows}"
        );
    }
    let frames = vec![vec![], vec![1], vec![], vec![2], vec![]];
    assert_eq!(greedy_run(gpu, &frames, 4, &[2, 5]), vec![(1, 1), (2, 3)]);
}

/// The window reads its first row from the search state, adds the prediction
/// and rectifies, and is zero past the last pending row.
///
/// # Panics
///
/// When a value differs from the formula.
pub fn greedy_window_adds_the_prediction_past_the_frame(gpu: &Gpu) {
    let (pending_rows, cols, rows, t) = (7usize, 40usize, 4usize, 5u32);
    let pending = oracle::signal(pending_rows * cols, 131);
    let dec = oracle::signal(cols, 132);
    let (pb, db) = (
        gpu.upload(&f32_bytes(&pending)),
        gpu.upload(&f32_bytes(&dec)),
    );
    let state = greedy_state(gpu, t, 0, 0, 0);
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; rows * cols]));
    let args = GreedyWindowArgs {
        rows: rows as u32,
        cols: cols as u32,
        avail: pending_rows as u32,
    };
    launch(
        gpu,
        "kernel_audio_greedy_window_f32",
        Extent3::linear(blocks(rows * cols, LINEAR_BLOCK)),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(pb.view()),
            KernelArg::Buffer(db.view()),
            KernelArg::Buffer(state.view()),
        ],
    );
    drain(gpu);
    let want: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let (r, c) = (i / cols, i % cols);
            let src = t as usize + r;
            if src < pending_rows {
                (pending[src * cols + c] + dec[c]).max(0.0)
            } else {
                0.0
            }
        })
        .collect();
    close(&gpu.read_f32(&out), &want, 0.0, "greedy window");
}

/// The embedding row is the state's last token's, and zeros for a token past
/// the table.
///
/// # Panics
///
/// When a value differs from the table.
pub fn embed_row_takes_the_last_tokens_row(gpu: &Gpu) {
    let (vocab, width) = (9usize, 300usize);
    let table = oracle::signal(vocab * width, 141);
    let tb = gpu.upload(&f32_bytes(&table));
    for token in [0u32, 7, 9] {
        let state = greedy_state(gpu, 0, 0, token, 1);
        let out = gpu.upload(&f32_bytes(&vec![f32::NAN; width]));
        let args = EmbedRowArgs {
            width: width as u32,
            vocab: vocab as u32,
        };
        launch(
            gpu,
            "kernel_audio_embed_row_f32",
            Extent3::linear(blocks(width, LINEAR_BLOCK)),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(tb.view()),
                KernelArg::Buffer(state.view()),
            ],
        );
        drain(gpu);
        let k = token as usize;
        let want = if k < vocab {
            table[k * width..(k + 1) * width].to_vec()
        } else {
            vec![0.0; width]
        };
        close(
            &gpu.read_f32(&out),
            &want,
            0.0,
            &format!("embed row {token}"),
        );
    }
}

/// The gated cell always hands h' on, and keeps h' and c' as the state only
/// when the last step emitted; kept, they are the plain cell's exactly.
///
/// # Panics
///
/// When a kept or unkept value is wrong.
pub fn gated_lstm_cell_keeps_the_state_only_on_emit(gpu: &Gpu) {
    let hdim = 640usize;
    let gates: Vec<f32> = oracle::signal(4 * hdim, 151)
        .iter()
        .map(|v| v * 3.0)
        .collect();
    let (h0, c0) = (oracle::signal(hdim, 152), oracle::signal(hdim, 153));
    let gb = gpu.upload(&f32_bytes(&gates));
    // The plain cell's answer is the reference.
    let (plain_h, plain_c) = (zeros(gpu, hdim), gpu.upload(&f32_bytes(&c0)));
    let args = LstmArgs {
        hidden: hdim as u32,
    };
    let grid = Extent3::linear(blocks(hdim, LINEAR_BLOCK));
    launch(
        gpu,
        "kernel_audio_lstm_cell_f32",
        grid,
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(plain_h.view()),
            KernelArg::Buffer(plain_c.view()),
            KernelArg::Buffer(gb.view()),
        ],
    );
    drain(gpu);
    let (want_h, want_c) = (gpu.read_f32(&plain_h), gpu.read_f32(&plain_c));
    for emit in [1u32, 0] {
        let (x, h, c) = (
            zeros(gpu, hdim),
            gpu.upload(&f32_bytes(&h0)),
            gpu.upload(&f32_bytes(&c0)),
        );
        let state = greedy_state(gpu, 0, 0, 0, emit);
        launch(
            gpu,
            "kernel_audio_lstm_cell_gated_f32",
            grid,
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(x.view()),
                KernelArg::Buffer(h.view()),
                KernelArg::Buffer(c.view()),
                KernelArg::Buffer(gb.view()),
                KernelArg::Buffer(state.view()),
            ],
        );
        drain(gpu);
        close(&gpu.read_f32(&x), &want_h, 0.0, "gated cell x");
        let (kept_h, kept_c) = if emit == 1 {
            (&want_h, &want_c)
        } else {
            (&h0, &c0)
        };
        close(
            &gpu.read_f32(&h),
            kept_h,
            0.0,
            &format!("gated cell h, emit {emit}"),
        );
        close(
            &gpu.read_f32(&c),
            kept_c,
            0.0,
            &format!("gated cell c, emit {emit}"),
        );
    }
}

/// Keep copies only when the last step emitted.
///
/// # Panics
///
/// When the destination is wrong.
pub fn keep_copies_only_on_emit(gpu: &Gpu) {
    let count = 1000usize;
    let (src, dst0) = (oracle::signal(count, 161), oracle::signal(count, 162));
    let sb = gpu.upload(&f32_bytes(&src));
    for emit in [0u32, 1] {
        let db = gpu.upload(&f32_bytes(&dst0));
        let state = greedy_state(gpu, 0, 0, 0, emit);
        let args = KeepArgs {
            count: count as u32,
        };
        launch(
            gpu,
            "kernel_audio_keep_f32",
            Extent3::linear(blocks(count, LINEAR_BLOCK)),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(db.view()),
                KernelArg::Buffer(sb.view()),
                KernelArg::Buffer(state.view()),
            ],
        );
        drain(gpu);
        let want = if emit == 1 { &src } else { &dst0 };
        close(&gpu.read_f32(&db), want, 0.0, &format!("keep, emit {emit}"));
    }
}

// ---- a causal decoder over a device KV cache -------------------------------

/// `audio_args_cache_attn`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CacheAttnArgs {
    n_q: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    pos0: u32,
    cap: u32,
    scale: f32,
    splits: u32,
}
const _: () = assert!(size_of::<CacheAttnArgs>() == 32);

/// `audio_args_kv_append`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct KvAppendArgs {
    rows: u32,
    kv_heads: u32,
    head_dim: u32,
    cap: u32,
}
const _: () = assert!(size_of::<KvAppendArgs>() == 16);

/// `audio_args_decode_step`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DecodeStepArgs {
    eos0: u32,
    eos1: u32,
    max_tokens: u32,
    advance: u32,
    slot: u32,
}
const _: () = assert!(size_of::<DecodeStepArgs>() == 20);

/// `audio_args_embed_token`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct EmbedTokenArgs {
    width: u32,
    vocab: u32,
    group: u32,
}
const _: () = assert!(size_of::<EmbedTokenArgs>() == 12);

/// Splits one decode-attention threadgroup runs, query heads one simdgroup
/// takes per kv head, and the merge's threads per head.
const DECODE_SPLITS_PER_TG: u32 = 4;
const DECODE_GROUP_TILE: u32 = 2;
const DECODE_MERGE_THREADS: u32 = 256;
/// The decoder state's words: cache rows filled, tokens emitted, done, last
/// token, emitted, stop reason.
const DEC_POS: usize = 0;
const DEC_COUNT: usize = 1;
const DEC_DONE: usize = 2;
const DEC_TOKEN: usize = 3;
const DEC_EMIT: usize = 4;
const DEC_STOP: usize = 5;
const DEC_STOP_EOS: u32 = 1;
const DEC_STOP_CAP: u32 = 2;

fn decode_state(gpu: &Gpu, pos: u32, token: u32) -> DeviceBuffer {
    gpu.upload(&u32_bytes(&[pos, 0, 0, token, 0, 0, 0, 0]))
}

/// One layer's cache plane, `[kv_heads][cap][hd]`: rows below `valid` hold
/// the signal, rows at or past it NaN, so a kernel that reads a row it may
/// not see answers NaN.
fn cache_plane(kv_heads: usize, cap: usize, hd: usize, valid: usize, seed: u32) -> Vec<f32> {
    let mut plane = oracle::signal(kv_heads * cap * hd, seed);
    for h in 0..kv_heads {
        for r in valid..cap {
            plane[(h * cap + r) * hd..(h * cap + r + 1) * hd].fill(f32::NAN);
        }
    }
    plane
}

/// Causal GQA attention over a cache plane, in f64: query `i` of head `h`
/// sits at `pos0 + i` and sees cache rows `0..=pos0 + i` of kv head
/// `h / (heads / kv_heads)`.
struct CacheAttnOracle<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    n_q: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    pos0: usize,
    cap: usize,
    scale: f32,
}

fn cache_attn_oracle(o: &CacheAttnOracle<'_>) -> Vec<f32> {
    let w = o.heads * o.hd;
    let group = o.heads / o.kv_heads;
    let mut out = vec![0.0f32; o.n_q * w];
    for i in 0..o.n_q {
        for h in 0..o.heads {
            let kvh = h / group;
            let seen = o.pos0 + i + 1;
            let scores: Vec<f64> = (0..seen)
                .map(|j| {
                    let dot: f64 = (0..o.hd)
                        .map(|e| {
                            f64::from(o.q[i * w + h * o.hd + e])
                                * f64::from(o.k[(kvh * o.cap + j) * o.hd + e])
                        })
                        .sum();
                    f64::from(o.scale) * dot
                })
                .collect();
            let mx = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = scores.iter().map(|s| (s - mx).exp()).sum();
            for e in 0..o.hd {
                let acc: f64 = scores
                    .iter()
                    .enumerate()
                    .map(|(j, s)| (s - mx).exp() / z * f64::from(o.v[(kvh * o.cap + j) * o.hd + e]))
                    .sum();
                out[i * w + h * o.hd + e] = acc as f32;
            }
        }
    }
    out
}

/// Prefill attention is causal and grouped: each query sees the cache up to
/// its own absolute position, and a query head reads its group's kv head,
/// from a cache offset of zero and of 37 rows. Rows past the last query's
/// position are NaN, so a row read that should not be is caught.
///
/// # Panics
///
/// When an output differs from the oracle by more than 2e-5.
pub fn causal_attention_matches_for_gqa_and_a_cache_offset(gpu: &Gpu) {
    let (heads, kv_heads, hd, n_q, cap) = (4usize, 2usize, 128usize, 13usize, 64usize);
    let scale = 1.0 / (hd as f32).sqrt();
    for pos0 in [0usize, 37] {
        let valid = pos0 + n_q;
        let q = oracle::signal(n_q * heads * hd, 171 + pos0 as u32);
        let k = cache_plane(kv_heads, cap, hd, valid, 172);
        let v = cache_plane(kv_heads, cap, hd, valid, 173);
        let (qb, kb, vb) = (
            gpu.upload(&f32_bytes(&q)),
            gpu.upload(&f32_bytes(&k)),
            gpu.upload(&f32_bytes(&v)),
        );
        let out = gpu.upload(&f32_bytes(&vec![f32::NAN; n_q * heads * hd]));
        let args = CacheAttnArgs {
            n_q: n_q as u32,
            heads: heads as u32,
            kv_heads: kv_heads as u32,
            head_dim: hd as u32,
            pos0: pos0 as u32,
            cap: cap as u32,
            scale,
            splits: 0,
        };
        launch(
            gpu,
            "kernel_audio_causal_attn_f32",
            Extent3::new(heads as u32, blocks(n_q, ATTN_QUERY_BLOCK), 1),
            Extent3::linear(32),
            ATTN_QUERY_BLOCK * hd as u32 * 4,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(qb.view()),
                KernelArg::Buffer(kb.view()),
                KernelArg::Buffer(vb.view()),
            ],
        );
        drain(gpu);
        let want = cache_attn_oracle(&CacheAttnOracle {
            q: &q,
            k: &k,
            v: &v,
            n_q,
            heads,
            kv_heads,
            hd,
            pos0,
            cap,
            scale,
        });
        close(
            &gpu.read_f32(&out),
            &want,
            2e-5,
            &format!("causal attention, pos0 {pos0}"),
        );
    }
}

/// Decode attention takes its length from the state: the one query of each
/// head at position `pos` sees rows `0..=pos`, for a length of one row, a
/// length shorter than the splits, and several blocks per split with a
/// ragged tail, split one way, five ways and 32 ways, for query groups of
/// one, two and four heads; the merge closes the splits into the same
/// answer. The rows past the length are NaN.
///
/// # Panics
///
/// When an output differs from the oracle by more than 2e-5.
pub fn decode_attention_reads_its_length_from_the_state(gpu: &Gpu) {
    // Groups of two (one tile), four (two tiles) and one (a tile's edge).
    for (heads, kv_heads, hd) in [(16usize, 8usize, 128usize), (8, 2, 128), (3, 3, 64)] {
        decode_attention_case(gpu, heads, kv_heads, hd);
    }
}

fn decode_attention_case(gpu: &Gpu, heads: usize, kv_heads: usize, hd: usize) {
    let cap = 1100usize;
    let scale = 1.0 / (hd as f32).sqrt();
    let q = oracle::signal(heads * hd, 181);
    let qb = gpu.upload(&f32_bytes(&q));
    for pos in [0usize, 3, 1037] {
        let k = cache_plane(kv_heads, cap, hd, pos + 1, 182);
        let v = cache_plane(kv_heads, cap, hd, pos + 1, 183);
        let (kb, vb) = (gpu.upload(&f32_bytes(&k)), gpu.upload(&f32_bytes(&v)));
        let state = decode_state(gpu, pos as u32, 0);
        let want = cache_attn_oracle(&CacheAttnOracle {
            q: &q,
            k: &k,
            v: &v,
            n_q: 1,
            heads,
            kv_heads,
            hd,
            pos0: pos,
            cap,
            scale,
        });
        for splits in [1usize, 5, 32] {
            let partials = gpu.upload(&f32_bytes(&vec![f32::NAN; heads * splits * (hd + 2)]));
            let out = gpu.upload(&f32_bytes(&vec![f32::NAN; heads * hd]));
            let args = CacheAttnArgs {
                n_q: 1,
                heads: heads as u32,
                kv_heads: kv_heads as u32,
                head_dim: hd as u32,
                pos0: 0,
                cap: cap as u32,
                scale,
                splits: splits as u32,
            };
            launch(
                gpu,
                "kernel_audio_decode_attn_f32",
                Extent3::new(
                    kv_heads as u32,
                    blocks(splits, DECODE_SPLITS_PER_TG),
                    blocks(heads / kv_heads, DECODE_GROUP_TILE),
                ),
                Extent3::linear(32 * DECODE_SPLITS_PER_TG),
                0,
                &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(partials.view()),
                    KernelArg::Buffer(qb.view()),
                    KernelArg::Buffer(kb.view()),
                    KernelArg::Buffer(vb.view()),
                    KernelArg::Buffer(state.view()),
                ],
            );
            launch(
                gpu,
                "kernel_audio_decode_attn_merge_f32",
                Extent3::linear(heads as u32),
                Extent3::linear(DECODE_MERGE_THREADS),
                0,
                &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(out.view()),
                    KernelArg::Buffer(partials.view()),
                ],
            );
            drain(gpu);
            close(
                &gpu.read_f32(&out),
                &want,
                2e-5,
                &format!(
                    "decode attention, {heads}/{kv_heads} heads of {hd}, pos {pos}, {splits} splits"
                ),
            );
        }
    }
}

/// The append writes its rows at the state's position, head by head into
/// the `[kv_heads][cap][hd]` planes, leaves every other row alone, and
/// drops a row that would land at or past the capacity.
///
/// # Panics
///
/// When a row is written where it should not be, or not where it should.
pub fn kv_append_writes_at_the_state_position(gpu: &Gpu) {
    let (kv_heads, hd, cap, rows) = (2usize, 128usize, 20usize, 3usize);
    for pos in [5usize, 18] {
        let (k0, v0) = (
            oracle::signal(kv_heads * cap * hd, 191),
            oracle::signal(kv_heads * cap * hd, 192),
        );
        let (kn, vn) = (
            oracle::signal(rows * kv_heads * hd, 193),
            oracle::signal(rows * kv_heads * hd, 194),
        );
        let (kc, vc) = (gpu.upload(&f32_bytes(&k0)), gpu.upload(&f32_bytes(&v0)));
        let (kb, vb) = (gpu.upload(&f32_bytes(&kn)), gpu.upload(&f32_bytes(&vn)));
        let state = decode_state(gpu, pos as u32, 0);
        let args = KvAppendArgs {
            rows: rows as u32,
            kv_heads: kv_heads as u32,
            head_dim: hd as u32,
            cap: cap as u32,
        };
        launch(
            gpu,
            "kernel_audio_kv_append_f32",
            Extent3::linear(blocks(rows * kv_heads * hd, LINEAR_BLOCK)),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(kc.view()),
                KernelArg::Buffer(vc.view()),
                KernelArg::Buffer(kb.view()),
                KernelArg::Buffer(vb.view()),
                KernelArg::Buffer(state.view()),
            ],
        );
        drain(gpu);
        let place = |plane: &[f32], new: &[f32]| {
            let mut want = plane.to_vec();
            for r in 0..rows {
                if pos + r >= cap {
                    continue;
                }
                for h in 0..kv_heads {
                    let dst = (h * cap + pos + r) * hd;
                    let src = (r * kv_heads + h) * hd;
                    want[dst..dst + hd].copy_from_slice(&new[src..src + hd]);
                }
            }
            want
        };
        close(
            &gpu.read_f32(&kc),
            &place(&k0, &kn),
            0.0,
            &format!("kv append k, pos {pos}"),
        );
        close(
            &gpu.read_f32(&vc),
            &place(&v0, &vn),
            0.0,
            &format!("kv append v, pos {pos}"),
        );
        assert_eq!(
            read_u32(gpu, &state)[DEC_POS],
            pos as u32,
            "the append leaves the position to the step"
        );
    }
}

/// Queues a batch of decode steps over scripted argmax ids with no read
/// between them, as the decoder does, and returns the slots and the state.
fn decode_batch(
    gpu: &Gpu,
    state: &DeviceBuffer,
    ids: &[u32],
    max_tokens: u32,
    first_advance: u32,
) -> (Vec<u32>, Vec<u32>) {
    let out = gpu.upload(&u32_bytes(&vec![7777; ids.len()]));
    let id_bufs: Vec<DeviceBuffer> = ids
        .iter()
        .map(|&id| gpu.upload(&u32_bytes(&[id])))
        .collect();
    for (slot, idb) in id_bufs.iter().enumerate() {
        let args = DecodeStepArgs {
            eos0: 151_643,
            eos1: 151_645,
            max_tokens,
            advance: if slot == 0 { first_advance } else { 1 },
            slot: slot as u32,
        };
        launch(
            gpu,
            "kernel_audio_decode_step_u32",
            Extent3::linear(1),
            Extent3::linear(1),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(state.view()),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(idb.view()),
            ],
        );
    }
    drain(gpu);
    (read_u32(gpu, &out), read_u32(gpu, state))
}

/// The step appends each argmax to its slot and advances the position, stops
/// on either end id without emitting it, stops after the token cap with the
/// last token kept, and once stopped every later step in the batch writes
/// the no-token mark and changes nothing.
///
/// # Panics
///
/// When a slot or a state word differs from the rules.
pub fn decode_step_stops_on_eos_and_goes_quiet(gpu: &Gpu) {
    let none = u32::MAX;
    // The first step takes the prefill's argmax: nothing was appended for it,
    // so the position stays where the host left it.
    let state = decode_state(gpu, 10, 0);
    let (out, s) = decode_batch(gpu, &state, &[5, 6, 151_645, 7, 8], 100, 0);
    assert_eq!(out, vec![5, 6, none, none, none]);
    assert_eq!(s[DEC_POS], 12, "steps 1 and 2 each fed one row");
    assert_eq!(s[DEC_COUNT], 2);
    assert_eq!(s[DEC_DONE], 1);
    assert_eq!(s[DEC_TOKEN], 6, "the end id is not the last token");
    assert_eq!(s[DEC_EMIT], 0);
    assert_eq!(s[DEC_STOP], DEC_STOP_EOS);
    // The other end id, with every step fed.
    let state = decode_state(gpu, 3, 42);
    let (out, s) = decode_batch(gpu, &state, &[9, 151_643, 1], 100, 1);
    assert_eq!(out, vec![9, none, none]);
    assert_eq!(s[DEC_POS], 5);
    assert_eq!((s[DEC_COUNT], s[DEC_TOKEN]), (1, 9));
    assert_eq!(s[DEC_STOP], DEC_STOP_EOS);
    // The cap keeps its last token and then goes quiet.
    let state = decode_state(gpu, 0, 42);
    let (out, s) = decode_batch(gpu, &state, &[1, 2, 3, 4, 5], 3, 1);
    assert_eq!(out, vec![1, 2, 3, none, none]);
    assert_eq!((s[DEC_POS], s[DEC_COUNT], s[DEC_TOKEN]), (3, 3, 3));
    assert_eq!(
        (s[DEC_DONE], s[DEC_STOP], s[DEC_EMIT]),
        (1, DEC_STOP_CAP, 0)
    );
    // A batch still running leaves done low and the emit flag up.
    let state = decode_state(gpu, 0, 42);
    let (out, s) = decode_batch(gpu, &state, &[11, 12], 100, 1);
    assert_eq!(out, vec![11, 12]);
    assert_eq!((s[DEC_POS], s[DEC_COUNT], s[DEC_TOKEN]), (2, 2, 12));
    assert_eq!((s[DEC_DONE], s[DEC_STOP], s[DEC_EMIT]), (0, 0, 1));
    // A later batch on the same state carries on from it.
    let (out, s) = decode_batch(gpu, &state, &[13, 151_645], 100, 1);
    assert_eq!(out, vec![13, none]);
    assert_eq!((s[DEC_POS], s[DEC_COUNT], s[DEC_TOKEN]), (4, 3, 13));
}

/// MLX 8-bit affine rows: per group of `group` elements a bf16 scale and bias,
/// the bytes packed four to a little-endian u32. Returns the words', scales'
/// and biases' bytes.
fn affine8_fixture(rows: usize, width: usize, group: usize, seed: u32) -> [Vec<u8>; 3] {
    let raw = oracle::signal(rows * width, seed);
    let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
    let (mut words, mut scales, mut biases) = (Vec::new(), Vec::new(), Vec::new());
    for g in raw.chunks(group) {
        let lo = g.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = g.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let s = f32::from_bits(u32::from(bf16((hi - lo) / 255.0)) << 16);
        let b = f32::from_bits(u32::from(bf16(lo)) << 16);
        scales.extend_from_slice(&bf16(s).to_le_bytes());
        biases.extend_from_slice(&bf16(b).to_le_bytes());
        for v in g {
            words.push(((v - b) / s).round().clamp(0.0, 255.0) as u8);
        }
    }
    [words, scales, biases]
}

/// Element `i` of an affine fixture as the kernel dequantises it: one fused
/// multiply-add of the byte, the group's scale and its bias.
fn affine8_value(fx: &[Vec<u8>; 3], group: usize, i: usize) -> f32 {
    let g = i / group;
    let bf16 =
        |b: &[u8]| f32::from_bits(u32::from(u16::from_le_bytes([b[2 * g], b[2 * g + 1]])) << 16);
    f32::from(fx[0][i]).mul_add(bf16(&fx[1]), bf16(&fx[2]))
}

/// The embedding row of the state's last token, for every storage the text
/// embedding comes in, dequantised exactly; zeros for a token past the table.
///
/// # Panics
///
/// When a value differs from the table's.
pub fn embed_token_takes_the_state_token_row(gpu: &Gpu) {
    let (vocab, width, group) = (9usize, 320usize, 64usize);
    let dense: Vec<(&str, &'static str)> = vec![
        ("f32w", "kernel_audio_embed_token_f32w"),
        ("bf16w", "kernel_audio_embed_token_bf16w"),
        ("f16w", "kernel_audio_embed_token_f16w"),
        ("q8_0w", "kernel_audio_embed_token_q8_0w"),
    ];
    let args = EmbedTokenArgs {
        width: width as u32,
        vocab: vocab as u32,
        group: group as u32,
    };
    let grid = Extent3::linear(blocks(width, LINEAR_BLOCK));
    let row_of = |values: &[f32], token: u32| -> Vec<f32> {
        let k = token as usize;
        if k < vocab {
            values[k * width..(k + 1) * width].to_vec()
        } else {
            vec![0.0; width]
        }
    };
    for (kind, entry) in dense {
        let (bytes, values) = weight_fixture(kind, vocab * width, 201);
        let tb = gpu.upload(&bytes);
        for token in [0u32, 7, 9] {
            let state = decode_state(gpu, 0, token);
            let out = gpu.upload(&f32_bytes(&vec![f32::NAN; width]));
            launch(
                gpu,
                entry,
                grid,
                Extent3::linear(LINEAR_BLOCK),
                0,
                &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(out.view()),
                    KernelArg::Buffer(tb.view()),
                    KernelArg::Buffer(state.view()),
                ],
            );
            drain(gpu);
            close(
                &gpu.read_f32(&out),
                &row_of(&values, token),
                0.0,
                &format!("embed token {kind} {token}"),
            );
        }
    }
    let fx = affine8_fixture(vocab, width, group, 202);
    let values: Vec<f32> = (0..vocab * width)
        .map(|i| affine8_value(&fx, group, i))
        .collect();
    let (wb, sb, bb) = (gpu.upload(&fx[0]), gpu.upload(&fx[1]), gpu.upload(&fx[2]));
    for token in [0u32, 7, 9] {
        let state = decode_state(gpu, 0, token);
        let out = gpu.upload(&f32_bytes(&vec![f32::NAN; width]));
        launch(
            gpu,
            "kernel_audio_embed_token_affine8w",
            grid,
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(wb.view()),
                KernelArg::Buffer(sb.view()),
                KernelArg::Buffer(bb.view()),
                KernelArg::Buffer(state.view()),
            ],
        );
        drain(gpu);
        close(
            &gpu.read_f32(&out),
            &row_of(&values, token),
            0.0,
            &format!("embed token affine8w {token}"),
        );
    }
}

/// The state-driven rotary embedding is the plain one at the state's position
/// plus the offset, bit for bit.
///
/// # Panics
///
/// When a value differs from the plain rotation's.
pub fn rope_half_state_takes_its_position_from_the_state(gpu: &Gpu) {
    let (rows, heads, hd, theta) = (2usize, 16usize, 128usize, 1.0e6f32);
    let (pos, offset) = (4093u32, 1u32);
    let x = oracle::signal(rows * heads * hd, 211);
    let pairs = rows * heads * hd / 2;
    let rotate = |entry: &'static str, pos0: u32, state: Option<&DeviceBuffer>| {
        let buf = gpu.upload(&f32_bytes(&x));
        let args = RopeArgs {
            rows: rows as u32,
            heads: heads as u32,
            head_dim: hd as u32,
            pos0,
            theta,
        };
        let mut bind = vec![
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(buf.view()),
        ];
        if let Some(s) = state {
            bind.push(KernelArg::Buffer(s.view()));
        }
        launch(
            gpu,
            entry,
            Extent3::linear(blocks(pairs, LINEAR_BLOCK)),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &bind,
        );
        drain(gpu);
        gpu.read_f32(&buf)
    };
    let state = decode_state(gpu, pos, 0);
    let got = rotate("kernel_audio_rope_half_state_f32", offset, Some(&state));
    let want = rotate("kernel_audio_rope_half_f32", pos + offset, None);
    close(&got, &want, 0.0, "rope at the state's position");
}
