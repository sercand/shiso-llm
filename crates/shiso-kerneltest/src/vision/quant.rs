//! The tower's projections over quantized weights: `kernel_vit_qmm_mxfp8` and
//! `kernel_vit_qmm_nvfp4` (`kernels/cuda/vision/vit_quant.cuh`), graded
//! against a host decode of the same codes and a sum in `f64`.
//!
//! The activation side is the split operand the encoder stages, two
//! brain-float planes; the fixture builds them on the host the way
//! `kernel_vit_to_bf16` does (the nearest brain float, then the nearest to
//! what it left), so kernel and oracle multiply one operand.
//!
//! The decode in the oracle is the container crate's
//! ([`shiso_modelfile::fp8::e4m3`], [`shiso_modelfile::nvfp4::e2m1`],
//! [`shiso_modelfile::mxfp8::ue8m0`]) and not the kernel's bit placement, so a
//! slip in either is a disagreement rather than a shared mistake.
//!
//! CUDA-only entries: a backend without them has no case to run.
#![expect(
    clippy::cast_possible_truncation,
    reason = "kernel argument structs take u32 dimensions while the fixtures are written in usize, and every fixture dimension here is a tower width or a patch count"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "m, n and k are the projection's own names, the kernel's and the BLAS call's"
)]

use bytemuck::{Pod, Zeroable};
use shiso_core::{
    DeviceBuffer, Extent3, GemmDType, GemmDesc, GemmOp, GemmReduction, KernelArg, KernelRef,
    LaunchArgs,
};
use shiso_modelfile::fp8::e4m3;
use shiso_modelfile::mxfp8::ue8m0;
use shiso_modelfile::nvfp4::e2m1;

use crate::bench::Fixture;
use crate::oracle;
use crate::{Gpu, drain, run};

/// The two tilings, by the outputs and patch rows a (square) tile covers:
/// the wide entries at 128, the `_narrow` ones at 64.
const TILES: [u32; 2] = [128, 64];

/// `VIT_Q_SMEM_OF(tile, tile)`: three steps of raw codes and both planes, two
/// decoded tiles.
const fn smem(tile: u32) -> u32 {
    3 * (tile * 32 + 2 * tile * 32 * 2) + 2 * tile * 32 * 2
}

/// `vit_args_qmm`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct QmmArgs {
    m: u32,
    n: u32,
    k: u32,
    global: f32,
}

/// The two stored formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    /// E4M3 codes, a `ue8m0` scale per 32.
    Mxfp8,
    /// E2M1 codes two a byte, an E4M3 scale per 16, an f32 scale per tensor.
    Nvfp4,
}

impl Format {
    const fn entry(self, tile: u32) -> &'static str {
        match (self, tile) {
            (Self::Mxfp8, 128) => "kernel_vit_qmm_mxfp8",
            (Self::Nvfp4, 128) => "kernel_vit_qmm_nvfp4",
            (Self::Mxfp8, _) => "kernel_vit_qmm_mxfp8_narrow",
            (Self::Nvfp4, _) => "kernel_vit_qmm_nvfp4_narrow",
        }
    }
}

/// A quantized weight `[n, k]` as stored, with its decoded values.
struct QWeight {
    codes: Vec<u8>,
    scales: Vec<u8>,
    global: f32,
    /// `[n * k]`, the f32 product the host decode makes, the tensor scale
    /// included.
    values: Vec<f32>,
}

/// A deterministic byte stream.
fn bytes(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed | 0x9E37_79B9;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect()
}

/// An MXFP8 weight: codes clear of the two NaN codes, scales within a few
/// octaves of one.
fn mxfp8_weight(n: usize, k: usize, seed: u32) -> QWeight {
    let codes: Vec<u8> = bytes(n * k, seed)
        .into_iter()
        .map(|c| if c & 0x7F == 0x7F { c ^ 0x01 } else { c })
        .collect();
    let scales: Vec<u8> = bytes(n * k / 32, seed ^ 0x51)
        .into_iter()
        .map(|b| 118 + b % 12)
        .collect();
    let values = codes
        .iter()
        .enumerate()
        .map(|(i, &c)| e4m3(c) * ue8m0(scales[i / 32]).unwrap_or(0.0))
        .collect();
    QWeight {
        codes,
        scales,
        global: 1.0,
        values,
    }
}

/// An NVFP4 weight: every nibble, scales in the E4M3 range a calibrated
/// tensor uses, and a tensor scale that is not a power of two.
fn nvfp4_weight(n: usize, k: usize, seed: u32) -> QWeight {
    let codes = bytes(n * k / 2, seed);
    let scales: Vec<u8> = bytes(n * k / 16, seed ^ 0x77)
        .into_iter()
        .map(|b| 0x28 + b % 0x30)
        .collect();
    let global = 0.000_731_2_f32;
    let values = (0..n * k)
        .map(|i| {
            let byte = codes[i / 2];
            let code = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            e2m1(code) * (e4m3(scales[i / 16]) * global)
        })
        .collect();
    QWeight {
        codes,
        scales,
        global,
        values,
    }
}

/// `x` as the split operand: the nearest brain floats, then the nearest to
/// the remainder, `m * k` of each; and the value the two sum to.
fn split(x: &[f32]) -> (Vec<u8>, Vec<f64>) {
    let hi: Vec<half::bf16> = x.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let lo: Vec<half::bf16> = x
        .iter()
        .zip(&hi)
        .map(|(&v, h)| half::bf16::from_f32(v - h.to_f32()))
        .collect();
    let bytes = hi
        .iter()
        .chain(&lo)
        .flat_map(|h| h.to_bits().to_le_bytes())
        .collect();
    let sum = hi
        .iter()
        .zip(&lo)
        .map(|(h, l)| f64::from(h.to_f32()) + f64::from(l.to_f32()))
        .collect();
    (bytes, sum)
}

/// Launches one projection.
fn launch(
    gpu: &Gpu,
    (format, tile): (Format, u32),
    out: &DeviceBuffer,
    planes: &DeviceBuffer,
    weight: (&DeviceBuffer, &DeviceBuffer, f32),
    (m, n, k): (usize, usize, usize),
) {
    let args = QmmArgs {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        global: weight.2,
    };
    run(
        gpu,
        KernelRef::new("vision", format.entry(tile)),
        &LaunchArgs {
            grid: Extent3::new((n as u32).div_ceil(tile), (m as u32).div_ceil(tile), 1),
            block: Extent3::linear(tile * tile / 32),
            smem_bytes: smem(tile),
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(planes.view()),
                KernelArg::Buffer(weight.0.view()),
                KernelArg::Buffer(weight.1.view()),
            ],
        },
    );
}

/// Runs one shape and grades every output against the `f64` sum over the
/// host decode. The bound is the f32 accumulation's: a few units of 2^-23 of
/// the sum of magnitudes, which is far below what a misplaced code, scale or
/// tile would move.
fn check(gpu: &Gpu, format: Format, (m, n, k): (usize, usize, usize), seed: u32) {
    let weight = match format {
        Format::Mxfp8 => mxfp8_weight(n, k, seed),
        Format::Nvfp4 => nvfp4_weight(n, k, seed),
    };
    let x = oracle::signal(m * k, seed ^ 0x99);
    let (plane_bytes, xs) = split(&x);
    let d_planes = gpu.upload(&plane_bytes);
    let d_codes = gpu.upload(&weight.codes);
    let d_scales = gpu.upload(&weight.scales);
    // Each output's `f64` sum and its sum of magnitudes, once for both tilings.
    let reference: Vec<(f64, f64)> = (0..m * n)
        .map(|i| {
            let (r, o) = (i / n, i % n);
            (0..k).fold((0f64, 0f64), |(sum, mag), c| {
                let p = f64::from(weight.values[o * k + c]) * xs[r * k + c];
                (sum + p, mag + p.abs())
            })
        })
        .collect();
    for tile in TILES {
        let out = gpu.upload(&vec![0xFFu8; m * n * 4]);
        launch(
            gpu,
            (format, tile),
            &out,
            &d_planes,
            (&d_codes, &d_scales, weight.global),
            (m, n, k),
        );
        drain(gpu);
        grade(
            &gpu.read_f32(&out),
            &reference,
            (m, n, k),
            format.entry(tile),
        );
    }
}

/// Grades one launch's output against the `f64` sum.
fn grade(got: &[f32], reference: &[(f64, f64)], (m, n, k): (usize, usize, usize), entry: &str) {
    assert_eq!(got.len(), reference.len(), "{entry}: output length");
    for (i, (&g, &(want, mag))) in got.iter().zip(reference).enumerate() {
        let g = f64::from(g);
        assert!(
            g.is_finite() && (g - want).abs() <= 1e-5 * mag + 1e-30,
            "{entry}: [{m} x {n} x {k}] row {} output {} is {g}, expected {want} (sum of magnitudes {mag})",
            i / n,
            i % n,
        );
    }
}

/// The MXFP8 projection matches the host decode at the tower's attention
/// widths, with patch rows and outputs that are not whole tiles.
pub fn the_mxfp8_projection_matches_the_host_decode(gpu: &Gpu) {
    check(gpu, Format::Mxfp8, (257, 300, 1152), 3);
    check(gpu, Format::Mxfp8, (5, 40, 64), 4);
}

/// The NVFP4 projection matches the host decode, over the second
/// feed-forward layer's ragged contraction (4304 is not a whole number of
/// 32-wide steps) and the merger's.
pub fn the_nvfp4_projection_matches_the_host_decode(gpu: &Gpu) {
    check(gpu, Format::Nvfp4, (131, 150, 4304), 5);
    check(gpu, Format::Nvfp4, (200, 260, 1152), 6);
    check(gpu, Format::Nvfp4, (3, 17, 48), 7);
}

// -- benches ----------------------------------------------------------------

/// The projection at a tower shape: `m` patch rows into `n` outputs over `k`.
fn bench_q(gpu: &Gpu, format: Format, (m, n, k): (usize, usize, usize)) -> Fixture {
    bench_tiled(gpu, (format, 128), (m, n, k))
}

/// [`bench_q`] at a chosen tiling.
fn bench_tiled(
    gpu: &Gpu,
    (format, tile): (Format, u32),
    (m, n, k): (usize, usize, usize),
) -> Fixture {
    let (codes, scales) = match format {
        Format::Mxfp8 => (bytes(n * k, 1), vec![127u8; n * k / 32]),
        Format::Nvfp4 => (bytes(n * k / 2, 1), vec![0x38u8; n * k / 16]),
    };
    let planes = gpu.upload(&vec![0x3Fu8; m * k * 4]);
    let d_codes = gpu.upload(&codes);
    let d_scales = gpu.upload(&scales);
    let out = gpu.upload(&vec![0u8; m * n * 4]);
    let weight_bytes = (codes.len() + scales.len()) as u64;
    let bytes = weight_bytes + (m * k * 4 + m * n * 4) as u64;
    // The useful product; the kernel runs it twice, once a plane.
    let flops = 2 * (m * n * k) as u64;
    Fixture::new(bytes, flops, move |gpu| {
        launch(
            gpu,
            (format, tile),
            &out,
            &planes,
            (&d_codes, &d_scales, 1.0),
            (m, n, k),
        );
    })
}

/// The path this replaces: two brain-float BLAS calls, one a plane, over the
/// weight decoded to brain floats at load.
fn bench_blas(gpu: &Gpu, (m, n, k): (usize, usize, usize)) -> Fixture {
    let weight = gpu.upload(&vec![0x3Fu8; n * k * 2]);
    let planes = gpu.upload(&vec![0x3Fu8; m * k * 4]);
    let out = gpu.upload(&vec![0u8; m * n * 4]);
    let bytes = (n * k * 2 + m * k * 4 + m * n * 4) as u64;
    let flops = 2 * (m * n * k) as u64;
    Fixture::new(bytes, flops, move |gpu| {
        let Some(blas) = gpu.device().as_blas() else {
            panic!("the BLAS baseline needs a device with a BLAS");
        };
        let plane = (m * k * 2) as u64;
        for (half, beta) in [(0u64, 0.0f32), (1, 1.0)] {
            blas.gemm(&GemmDesc {
                op_a: GemmOp::Transposed,
                op_b: GemmOp::AsStored,
                m: n as u32,
                n: m as u32,
                k: k as u32,
                lda: k as u32,
                ldb: k as u32,
                ldc: n as u32,
                a: weight.view(),
                b: planes
                    .slice(half * plane, plane)
                    .expect("the plane lies inside the operand"),
                c: out.view(),
                alpha: 1.0,
                beta,
                dtype: GemmDType::BF16,
                reduction: GemmReduction::Fastest,
            })
            .expect("the device runs the BLAS baseline");
        }
    })
}

/// The fused projection's shape at 4096 patch rows.
pub(crate) fn bench_mxfp8_qkv_4096(gpu: &Gpu) -> Fixture {
    bench_q(gpu, Format::Mxfp8, (4096, 3456, 1152))
}
/// The attention output's shape at 4096 patch rows.
pub(crate) fn bench_mxfp8_proj_4096(gpu: &Gpu) -> Fixture {
    bench_q(gpu, Format::Mxfp8, (4096, 1152, 1152))
}
/// The first feed-forward layer at 4096 patch rows.
pub(crate) fn bench_nvfp4_fc1_4096(gpu: &Gpu) -> Fixture {
    bench_q(gpu, Format::Nvfp4, (4096, 4304, 1152))
}
/// The second feed-forward layer at 4096 patch rows.
pub(crate) fn bench_nvfp4_fc2_4096(gpu: &Gpu) -> Fixture {
    bench_q(gpu, Format::Nvfp4, (4096, 1152, 4304))
}
/// The merger's first layer at 1024 merged rows (4096 patches).
pub(crate) fn bench_nvfp4_merger_1024(gpu: &Gpu) -> Fixture {
    bench_q(gpu, Format::Nvfp4, (1024, 4608, 4608))
}
/// The fused projection's shape at 16384 patch rows.
pub(crate) fn bench_mxfp8_qkv_16384(gpu: &Gpu) -> Fixture {
    bench_q(gpu, Format::Mxfp8, (16_384, 3456, 1152))
}
/// The first feed-forward layer at 16384 patch rows.
pub(crate) fn bench_nvfp4_fc1_16384(gpu: &Gpu) -> Fixture {
    bench_q(gpu, Format::Nvfp4, (16_384, 4304, 1152))
}
/// The second feed-forward layer at 16384 patch rows.
pub(crate) fn bench_nvfp4_fc2_16384(gpu: &Gpu) -> Fixture {
    bench_q(gpu, Format::Nvfp4, (16_384, 1152, 4304))
}
/// The narrow tile at a 224-pixel picture's attention output (196 rows).
pub(crate) fn bench_mxfp8_proj_196_narrow(gpu: &Gpu) -> Fixture {
    bench_tiled(gpu, (Format::Mxfp8, 64), (196, 1152, 1152))
}
/// The wide tile at the same shape.
pub(crate) fn bench_mxfp8_proj_196(gpu: &Gpu) -> Fixture {
    bench_q(gpu, Format::Mxfp8, (196, 1152, 1152))
}
/// The narrow tile at a 224-pixel picture's second feed-forward layer.
pub(crate) fn bench_nvfp4_fc2_196_narrow(gpu: &Gpu) -> Fixture {
    bench_tiled(gpu, (Format::Nvfp4, 64), (196, 1152, 4304))
}
/// The BLAS pair at a 224-pixel picture's attention output.
pub(crate) fn bench_blas_proj_196(gpu: &Gpu) -> Fixture {
    bench_blas(gpu, (196, 1152, 1152))
}
/// The BLAS pair at a 224-pixel picture's second feed-forward layer.
pub(crate) fn bench_blas_fc2_196(gpu: &Gpu) -> Fixture {
    bench_blas(gpu, (196, 1152, 4304))
}
/// The BLAS pair at the fused projection's shape, 4096 patch rows.
pub(crate) fn bench_blas_qkv_4096(gpu: &Gpu) -> Fixture {
    bench_blas(gpu, (4096, 3456, 1152))
}
/// The BLAS pair at the attention output's shape, 4096 patch rows.
pub(crate) fn bench_blas_proj_4096(gpu: &Gpu) -> Fixture {
    bench_blas(gpu, (4096, 1152, 1152))
}
/// The BLAS pair at the first feed-forward layer, 4096 patch rows.
pub(crate) fn bench_blas_fc1_4096(gpu: &Gpu) -> Fixture {
    bench_blas(gpu, (4096, 4304, 1152))
}
/// The BLAS pair at the second feed-forward layer, 4096 patch rows.
pub(crate) fn bench_blas_fc2_4096(gpu: &Gpu) -> Fixture {
    bench_blas(gpu, (4096, 1152, 4304))
}
/// The BLAS pair at the merger's first layer, 1024 merged rows.
pub(crate) fn bench_blas_merger_1024(gpu: &Gpu) -> Fixture {
    bench_blas(gpu, (1024, 4608, 4608))
}
/// The BLAS pair at the fused projection's shape, 16384 patch rows.
pub(crate) fn bench_blas_qkv_16384(gpu: &Gpu) -> Fixture {
    bench_blas(gpu, (16_384, 3456, 1152))
}
/// The BLAS pair at the first feed-forward layer, 16384 patch rows.
pub(crate) fn bench_blas_fc1_16384(gpu: &Gpu) -> Fixture {
    bench_blas(gpu, (16_384, 4304, 1152))
}
/// The BLAS pair at the second feed-forward layer, 16384 patch rows.
pub(crate) fn bench_blas_fc2_16384(gpu: &Gpu) -> Fixture {
    bench_blas(gpu, (16_384, 1152, 4304))
}
