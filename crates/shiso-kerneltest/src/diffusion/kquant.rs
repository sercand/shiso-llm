//! The GGUF k-quant linears: `Q4_K` and `Q6_K` weight rows decoded while the
//! simdgroup tile stages them.
//!
//! Each case builds random blocks with the loader's own block structs, so
//! the byte layout comes from a struct definition rather than this file,
//! and grades the device product against a plain `f64` product over the
//! host-decoded weight. `Q4_K` decodes through the loader's
//! [`dequant_q4_k`]; `Q6_K` has no host decoder there, so this file writes
//! its formula out directly - the run order and the bit planes are what
//! the kernel's index math must reproduce. The fused entry is graded
//! against the separate swiglu and linear it replaces.
#![expect(
    clippy::cast_possible_truncation,
    reason = "fixture dimensions are small literals marshalled as the kernels' u32 argument fields"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "fixture indices and codes converted to floats are small, exact in f32"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "the oracles are written in the formula's own letters (m, n, k, x, w), which is what makes them checkable against it"
)]

use bytemuck::{Pod, Zeroable};
use half::f16;
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};
use shiso_modelfile::quant::{BlockQ4K, BlockQ6K, dequant_q4_k};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// Every entry point this file's cases launch.
pub const ENTRIES: [&str; 4] = [
    "kernel_dit_gemm64_q4k",
    "kernel_dit_gemm64_q6k",
    "kernel_dit_gemm64_fused_q4k",
    "kernel_dit_gemm64_fused_q6k",
];

const MODULE: &str = "diffusion";
/// Weights per k-quant block.
const QK: usize = 256;
/// Output tile edge of the simdgroup GEMM.
const G64_TILE: u32 = 64;
/// Threads per simdgroup-tile threadgroup.
const G64_THREADS: u32 = 128;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GemmArgs {
    m: u32,
    n: u32,
    k: u32,
    has_bias: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GemmFusedArgs {
    m: u32,
    n: u32,
    cols_a: u32,
    inner: u32,
    gu_stride: u32,
    gu_offset: u32,
    has_bias: u32,
}

const _: () = assert!(size_of::<GemmArgs>() == 16);
const _: () = assert!(size_of::<GemmFusedArgs>() == 28);

/// A pseudo-random byte stream from `seed`.
fn bytes(count: usize, seed: u32) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9) | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect()
}

/// `count` random `Q4_K` blocks with small, finite super-block scales.
fn q4k_blocks(count: usize, seed: u32) -> Vec<BlockQ4K> {
    let raw = bytes(count * size_of::<BlockQ4K>(), seed);
    let mut blocks: Vec<BlockQ4K> = raw
        .chunks_exact(size_of::<BlockQ4K>())
        .map(bytemuck::pod_read_unaligned)
        .collect();
    for (i, b) in blocks.iter_mut().enumerate() {
        b.d = f16::from_f32(0.002 + 0.0005 * (i % 7) as f32).to_bits();
        b.dmin = f16::from_f32(0.001 + 0.0003 * (i % 5) as f32).to_bits();
    }
    blocks
}

/// `count` random `Q6_K` blocks with a small, finite block scale.
fn q6k_blocks(count: usize, seed: u32) -> Vec<BlockQ6K> {
    let raw = bytes(count * size_of::<BlockQ6K>(), seed);
    let mut blocks: Vec<BlockQ6K> = raw
        .chunks_exact(size_of::<BlockQ6K>())
        .map(bytemuck::pod_read_unaligned)
        .collect();
    for (i, b) in blocks.iter_mut().enumerate() {
        b.d = f16::from_f32(0.0004 + 0.0001 * (i % 3) as f32).to_bits();
    }
    blocks
}

/// One `Q6_K` block's 256 weights: per half of 128, run `r` of 32 takes the
/// low (r = 0, 1) or high (r = 2, 3) nibbles of `ql[32 * (r & 1)..]`, bits
/// `2r, 2r + 1` of `qh`, minus 32, times the int8 scale of its 16.
fn dequant_q6_k(b: &BlockQ6K) -> [f32; QK] {
    let d = f16::from_bits(b.d).to_f32();
    let mut out = [0.0f32; QK];
    for half in 0..2 {
        for r in 0..4 {
            for l in 0..32 {
                let ql = b.ql[64 * half + 32 * (r & 1) + l];
                let lo = if r >= 2 { ql >> 4 } else { ql & 0xF };
                let hi = (b.qh[32 * half + l] >> (2 * r)) & 3;
                let q = i32::from(lo | (hi << 4)) - 32;
                let scale = f32::from(b.scales[8 * half + l / 16 + 2 * r]);
                out[128 * half + 32 * r + l] = d * scale * q as f32;
            }
        }
    }
    out
}

/// A k-quant weight of `n` rows by `k` columns: its bytes and its decode.
struct Weight {
    bytes: Vec<u8>,
    dense: Vec<f32>,
}

fn weight(format: &str, n: usize, k: usize, seed: u32) -> Weight {
    let count = n * k / QK;
    if format == "q4k" {
        let blocks = q4k_blocks(count, seed);
        Weight {
            bytes: bytemuck::cast_slice(&blocks).to_vec(),
            dense: blocks.iter().flat_map(dequant_q4_k).collect(),
        }
    } else {
        let blocks = q6k_blocks(count, seed);
        Weight {
            bytes: bytemuck::cast_slice(&blocks).to_vec(),
            dense: blocks.iter().flat_map(dequant_q6_k).collect(),
        }
    }
}

/// `x . w^T` in `f64`.
fn product(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let dot: f64 = (0..k)
                .map(|i| f64::from(x[r * k + i]) * f64::from(w[c * k + i]))
                .sum();
            out[r * n + c] = dot as f32;
        }
    }
    out
}

fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let rms =
        (want.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / want.len() as f64).sqrt() as f32;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let bound = 2e-4 * w.abs().max(rms);
        assert!(
            (g - w).abs() <= bound,
            "{what}[{i}]: device {g} vs oracle {w} (bound {bound})"
        );
    }
}

fn launch(gpu: &Gpu, entry: &'static str, m: usize, n: usize, args: &[KernelArg<'_>]) {
    run(
        gpu,
        KernelRef::new(MODULE, entry),
        &LaunchArgs {
            grid: Extent3::new(
                (n as u32).div_ceil(G64_TILE),
                (m as u32).div_ceil(G64_TILE),
                1,
            ),
            block: Extent3::linear(G64_THREADS),
            smem_bytes: 0,
            args,
        },
    );
}

/// `kernel_dit_gemm64_q4k` and `kernel_dit_gemm64_q6k` are the product over
/// the host-decoded weight, with a bias, on a shape that leaves ragged row
/// and column tiles and crosses three blocks along each row.
///
/// # Panics
///
/// When the device disagrees with the oracle.
pub fn gemm_kquant_matches_the_decoded_product(gpu: &Gpu) {
    let (m, n, k) = (70usize, 67usize, 3 * QK);
    for (format, entry) in [
        ("q4k", "kernel_dit_gemm64_q4k"),
        ("q6k", "kernel_dit_gemm64_q6k"),
    ] {
        let w = weight(format, n, k, 11);
        let x = signal(m * k, 12);
        let bias = signal(n, 13);
        let mut want = product(&x, &w.dense, m, n, k);
        for r in 0..m {
            for c in 0..n {
                want[r * n + c] += bias[c];
            }
        }
        let d_w = gpu.upload(&w.bytes);
        let d_x = gpu.upload(&f32_bytes(&x));
        let d_b = gpu.upload(&f32_bytes(&bias));
        let out: DeviceBuffer = gpu.upload(&f32_bytes(&vec![0.0; m * n]));
        let args = GemmArgs {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            has_bias: 1,
        };
        launch(
            gpu,
            entry,
            m,
            n,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(d_x.view()),
                KernelArg::Buffer(d_w.view()),
                KernelArg::F32(1.0),
                KernelArg::Buffer(d_b.view()),
            ],
        );
        drain(gpu);
        close(&gpu.read_f32(&out), &want, entry);
    }
}

/// `kernel_dit_gemm64_fused_q4k` and `_q6k` add `gate * linear([a |
/// silu(g) * u])` onto the residual, as the separate passes would.
///
/// # Panics
///
/// When the device disagrees with the oracle.
pub fn fused_kquant_is_the_unfused_composition(gpu: &Gpu) {
    let (m, n, cols_a, inner) = (37usize, 70usize, QK, QK);
    let k = cols_a + inner;
    for (format, entry) in [
        ("q4k", "kernel_dit_gemm64_fused_q4k"),
        ("q6k", "kernel_dit_gemm64_fused_q6k"),
    ] {
        let w = weight(format, n, k, 21);
        let a = signal(m * cols_a, 22);
        let gu = signal(m * 2 * inner, 23);
        let residual = signal(m * n, 24);
        let gate = signal(n, 25);
        let x: Vec<f32> = (0..m)
            .flat_map(|r| {
                let row = &gu[r * 2 * inner..][..2 * inner];
                a[r * cols_a..][..cols_a].iter().copied().chain(
                    (0..inner).map(move |i| row[i] / (1.0 + (-row[i]).exp()) * row[inner + i]),
                )
            })
            .collect();
        let lin = product(&x, &w.dense, m, n, k);
        let want: Vec<f32> = (0..m * n)
            .map(|i| residual[i] + gate[i % n] * lin[i])
            .collect();
        let d_w = gpu.upload(&w.bytes);
        let d_a = gpu.upload(&f32_bytes(&a));
        let d_gu = gpu.upload(&f32_bytes(&gu));
        let d_gate = gpu.upload(&f32_bytes(&gate));
        let stand_in = gpu.upload(&f32_bytes(&[0.0]));
        let out = gpu.upload(&f32_bytes(&residual));
        let args = GemmFusedArgs {
            m: m as u32,
            n: n as u32,
            cols_a: cols_a as u32,
            inner: inner as u32,
            gu_stride: (2 * inner) as u32,
            gu_offset: 0,
            has_bias: 0,
        };
        launch(
            gpu,
            entry,
            m,
            n,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(d_a.view()),
                KernelArg::Buffer(d_gu.view()),
                KernelArg::Buffer(d_w.view()),
                KernelArg::F32(1.0),
                KernelArg::Buffer(stand_in.view()),
                KernelArg::Buffer(d_gate.view()),
            ],
        );
        drain(gpu);
        close(&gpu.read_f32(&out), &want, entry);
    }
}
