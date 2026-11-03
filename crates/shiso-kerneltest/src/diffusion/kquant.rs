//! The GGUF block-quant linears: `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` and
//! `Q8_0` weight rows decoded while the simdgroup tile stages them.
//!
//! Each case builds random blocks with the loader's own block structs, so
//! the byte layout comes from a struct definition rather than this file,
//! and grades the device product against a plain `f64` product over the
//! host-decoded weight. `Q4_K` decodes through the loader's
//! [`dequant_q4_k`] and `Q2_K` through its `decode_q2_k_block`; the other
//! formats have no host decoder there, so this file writes their formulas
//! out directly - the run order and the bit planes are what the kernel's
//! index math must reproduce. The fused entry is graded against the
//! separate swiglu and linear it replaces.
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
use shiso_modelfile::iq_quant::decode_q2_k_block;
use shiso_modelfile::quant::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8_0, dequant_q4_k,
};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// Every entry point this file's cases launch.
pub const ENTRIES: [&str; 12] = [
    "kernel_dit_gemm64_q2k",
    "kernel_dit_gemm64_q3k",
    "kernel_dit_gemm64_q4k",
    "kernel_dit_gemm64_q5k",
    "kernel_dit_gemm64_q6k",
    "kernel_dit_gemm64_q8_0",
    "kernel_dit_gemm64_fused_q2k",
    "kernel_dit_gemm64_fused_q3k",
    "kernel_dit_gemm64_fused_q4k",
    "kernel_dit_gemm64_fused_q5k",
    "kernel_dit_gemm64_fused_q6k",
    "kernel_dit_gemm64_fused_q8_0",
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

/// The block formats these cases cover.
#[derive(Debug, Clone, Copy)]
enum Format {
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
}

impl Format {
    /// Weights per block: the reduction length must be a multiple of it.
    fn elems(self) -> usize {
        match self {
            Self::Q8_0 => 32,
            _ => QK,
        }
    }

    fn entry(self, fused: bool) -> &'static str {
        match (self, fused) {
            (Self::Q2K, false) => "kernel_dit_gemm64_q2k",
            (Self::Q3K, false) => "kernel_dit_gemm64_q3k",
            (Self::Q4K, false) => "kernel_dit_gemm64_q4k",
            (Self::Q5K, false) => "kernel_dit_gemm64_q5k",
            (Self::Q6K, false) => "kernel_dit_gemm64_q6k",
            (Self::Q8_0, false) => "kernel_dit_gemm64_q8_0",
            (Self::Q2K, true) => "kernel_dit_gemm64_fused_q2k",
            (Self::Q3K, true) => "kernel_dit_gemm64_fused_q3k",
            (Self::Q4K, true) => "kernel_dit_gemm64_fused_q4k",
            (Self::Q5K, true) => "kernel_dit_gemm64_fused_q5k",
            (Self::Q6K, true) => "kernel_dit_gemm64_fused_q6k",
            (Self::Q8_0, true) => "kernel_dit_gemm64_fused_q8_0",
        }
    }
}

/// `count` random blocks of `B`, each given small, finite fp16 scales by
/// `fix` (which gets the block and its index).
fn blocks<B: Pod>(count: usize, seed: u32, fix: impl Fn(&mut B, usize)) -> Vec<B> {
    let raw = bytes(count * size_of::<B>(), seed);
    let mut blocks: Vec<B> = raw
        .chunks_exact(size_of::<B>())
        .map(bytemuck::pod_read_unaligned)
        .collect();
    for (i, b) in blocks.iter_mut().enumerate() {
        fix(b, i);
    }
    blocks
}

fn half_bits(v: f32) -> u16 {
    f16::from_f32(v).to_bits()
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

/// The 6-bit `(scale, min)` pair `j` (0..8) of a `Q4_K`/`Q5_K` block's 12
/// scale bytes.
fn scale_min(j: usize, sc: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (sc[j] & 63, sc[j + 4] & 63)
    } else {
        (
            (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4),
            (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4),
        )
    }
}

/// One `Q5_K` block: as `Q4_K`, with bit `2c` (low nibbles) or `2c + 1`
/// (high nibbles) of `qh[l]` adding 16 to run `c`'s code.
fn dequant_q5_k(b: &BlockQ5K) -> [f32; QK] {
    let d = f16::from_bits(b.d).to_f32();
    let dmin = f16::from_bits(b.dmin).to_f32();
    let mut out = [0.0f32; QK];
    for c in 0..4 {
        let (s1, m1) = scale_min(2 * c, &b.scales);
        let (s2, m2) = scale_min(2 * c + 1, &b.scales);
        for l in 0..32 {
            let q = b.qs[32 * c + l];
            let lo = i32::from(q & 0xF) + if b.qh[l] >> (2 * c) & 1 == 1 { 16 } else { 0 };
            let hi = i32::from(q >> 4)
                + if b.qh[l] >> (2 * c + 1) & 1 == 1 {
                    16
                } else {
                    0
                };
            out[64 * c + l] = d * f32::from(s1) * lo as f32 - dmin * f32::from(m1);
            out[64 * c + 32 + l] = d * f32::from(s2) * hi as f32 - dmin * f32::from(m2);
        }
    }
    out
}

/// One `Q3_K` block: sixteen 6-bit scales (minus 32) in 12 bytes, the low
/// 2 bits of each weight in `qs` (two planes of 128; four runs of 32 per
/// plane at shifts 0, 2, 4, 6) and the third bit inverted in `hmask`:
/// a clear bit subtracts 4.
fn dequant_q3_k(b: &BlockQ3K) -> [f32; QK] {
    let d = f16::from_bits(b.d).to_f32();
    let scale = |is: usize| -> i32 {
        let low = if is < 8 {
            b.scales[is] & 0xF
        } else {
            b.scales[is - 8] >> 4
        };
        let high = (b.scales[8 + (is & 3)] >> (2 * (is >> 2))) & 3;
        i32::from(low | (high << 4)) - 32
    };
    let mut out = [0.0f32; QK];
    for plane in 0..2 {
        for j in 0..4 {
            for sub in 0..2 {
                let dl = d * scale(8 * plane + 2 * j + sub) as f32;
                for l in 0..16 {
                    let at = 32 * plane + 16 * sub + l;
                    let low = i32::from((b.qs[at] >> (2 * j)) & 3);
                    let high = if b.hmask[16 * sub + l] >> (4 * plane + j) & 1 == 1 {
                        0
                    } else {
                        4
                    };
                    out[128 * plane + 32 * j + 16 * sub + l] = dl * (low - high) as f32;
                }
            }
        }
    }
    out
}

/// A block-quant weight of `n` rows by `k` columns: its bytes and its decode.
struct Weight {
    bytes: Vec<u8>,
    dense: Vec<f32>,
}

fn weight(format: Format, n: usize, k: usize, seed: u32) -> Weight {
    let count = n * k / format.elems();
    let pack = |bytes: &[u8], dense: Vec<f32>| Weight {
        bytes: bytes.to_vec(),
        dense,
    };
    match format {
        Format::Q2K => {
            let b = blocks::<BlockQ2K>(count, seed, |b, i| {
                b.d = half_bits(0.003 + 0.0005 * (i % 7) as f32);
                b.dmin = half_bits(0.002 + 0.0003 * (i % 5) as f32);
            });
            let dense = b
                .iter()
                .flat_map(|b| {
                    let mut out = [0.0f32; QK];
                    decode_q2_k_block(bytemuck::cast_ref(b), &mut out);
                    out
                })
                .collect();
            pack(bytemuck::cast_slice(&b), dense)
        }
        Format::Q3K => {
            let b = blocks::<BlockQ3K>(count, seed, |b, i| {
                b.d = half_bits(0.0008 + 0.0002 * (i % 7) as f32);
            });
            pack(
                bytemuck::cast_slice(&b),
                b.iter().flat_map(dequant_q3_k).collect(),
            )
        }
        Format::Q4K => {
            let b = blocks::<BlockQ4K>(count, seed, |b, i| {
                b.d = half_bits(0.002 + 0.0005 * (i % 7) as f32);
                b.dmin = half_bits(0.001 + 0.0003 * (i % 5) as f32);
            });
            pack(
                bytemuck::cast_slice(&b),
                b.iter().flat_map(dequant_q4_k).collect(),
            )
        }
        Format::Q5K => {
            let b = blocks::<BlockQ5K>(count, seed, |b, i| {
                b.d = half_bits(0.001 + 0.0003 * (i % 7) as f32);
                b.dmin = half_bits(0.0005 + 0.0002 * (i % 5) as f32);
            });
            pack(
                bytemuck::cast_slice(&b),
                b.iter().flat_map(dequant_q5_k).collect(),
            )
        }
        Format::Q6K => {
            let b = blocks::<BlockQ6K>(count, seed, |b, i| {
                b.d = half_bits(0.0004 + 0.0001 * (i % 3) as f32);
            });
            pack(
                bytemuck::cast_slice(&b),
                b.iter().flat_map(dequant_q6_k).collect(),
            )
        }
        Format::Q8_0 => {
            let b = blocks::<BlockQ8_0>(count, seed, |b, i| {
                b.d = half_bits(0.01 + 0.002 * (i % 5) as f32);
            });
            let dense = b
                .iter()
                .flat_map(|b| {
                    let d = f16::from_bits(b.d).to_f32();
                    b.qs.iter().map(move |&q| d * f32::from(q))
                })
                .collect();
            pack(bytemuck::cast_slice(&b), dense)
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

/// The reduction lengths a format is graded at: one block, and three.
fn ks(format: Format) -> [usize; 2] {
    let e = format.elems();
    [e, 3 * e]
}

/// The product over the host-decoded weight, with a bias, on a shape that
/// leaves ragged row and column tiles.
fn gemm_case(gpu: &Gpu, format: Format) {
    let entry = format.entry(false);
    for k in ks(format) {
        let (m, n) = (70usize, 67usize);
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
        close(&gpu.read_f32(&out), &want, &format!("{entry} k={k}"));
    }
}

/// `gate * linear([a | silu(g) * u])` added onto the residual, as the
/// separate passes would.
fn fused_case(gpu: &Gpu, format: Format) {
    let entry = format.entry(true);
    for k in ks(format) {
        // Split the reduction into a dense part and a swiglu part, both
        // multiples of four.
        let (cols_a, inner) = (k / 2, k - k / 2);
        let (m, n) = (37usize, 70usize);
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
        close(&gpu.read_f32(&out), &want, &format!("{entry} k={k}"));
    }
}

/// Defines the public per-format cases: one plain GEMM and one fused.
macro_rules! cases {
    ($($format:ident => $gemm:ident, $fused:ident;)*) => {
        $(
            /// The plain tiled GEMM over this format's host-decoded weight,
            /// at one block and at three blocks along each row.
            ///
            /// # Panics
            ///
            /// When the device disagrees with the oracle.
            pub fn $gemm(gpu: &Gpu) {
                gemm_case(gpu, Format::$format);
            }

            /// The fused entry over this format, against the separate swiglu
            /// and linear it replaces.
            ///
            /// # Panics
            ///
            /// When the device disagrees with the oracle.
            pub fn $fused(gpu: &Gpu) {
                fused_case(gpu, Format::$format);
            }
        )*
    };
}

cases! {
    Q2K => gemm64_q2k_matches_dequant_oracle, gemm64_fused_q2k_is_the_unfused_composition;
    Q3K => gemm64_q3k_matches_dequant_oracle, gemm64_fused_q3k_is_the_unfused_composition;
    Q4K => gemm64_q4k_matches_dequant_oracle, gemm64_fused_q4k_is_the_unfused_composition;
    Q5K => gemm64_q5k_matches_dequant_oracle, gemm64_fused_q5k_is_the_unfused_composition;
    Q6K => gemm64_q6k_matches_dequant_oracle, gemm64_fused_q6k_is_the_unfused_composition;
    Q8_0 => gemm64_q8_0_matches_dequant_oracle, gemm64_fused_q8_0_is_the_unfused_composition;
}
