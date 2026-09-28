//! Cases for the row-exact affine matmul (`dense_exact_affine.metal`).
//!
//! Two properties, checked separately because they fail for different reasons:
//!
//! - **Value:** every entry agrees with the host dequantize-then-matmul within
//!   a float tolerance. This is what a wrong unpack, a wrong group or a wrong
//!   scale shows up in.
//! - **Row exactness:** a row's bits do not depend on how many rows share the
//!   call. Row `r` of an `R`-row matrix call equals the same row alone, for
//!   every `R` from 1 to 33, and the one-row matvec equals the matrix call's
//!   one-row answer. This is what a speculative verify window rests on, and no
//!   tolerance is involved.
//!
//! The launch geometry mirrors `shiso_engine::projection`'s exact plan. Only the
//! split count - a constant inside the kernel - decides the bits; the tile
//! counts chosen here only spread outputs over threadgroups, so the row
//! exactness asserted below holds for the engine's choice too.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "fixture dimensions are small positive integers"
)]
#![expect(
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    reason = "the host reference is written in the matrix algebra's own index names"
)]
#![expect(
    clippy::print_stdout,
    reason = "the value cases print the margin they measured, which is what makes a near-miss readable under --ignored"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use bytemuck::{Pod, Zeroable};
use half::bf16;
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

/// Argument block mirroring `dense_args_exact_affine`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ExactAffineArgs {
    in_dim: u32,
    out_dim: u32,
    n_rows: u32,
    pad: u32,
}

const _: () = assert!(size_of::<ExactAffineArgs>() == 16);

/// Chunks the kernels deal a row's groups to (`DENSE_EXACT_SPLITS`).
const SPLITS: u32 = 8;

/// Every entry point this module's cases launch.
pub const ENTRIES: [&str; 16] = [
    "kernel_dense_exact_qmv_4_nr1_f32",
    "kernel_dense_exact_qmv_4_nr2_f32",
    "kernel_dense_exact_qmv_8_nr1_f32",
    "kernel_dense_exact_qmv_8_nr2_f32",
    "kernel_dense_exact_qmm_4_nt1_rt1_f32",
    "kernel_dense_exact_qmm_4_nt2_rt1_f32",
    "kernel_dense_exact_qmm_4_nt4_rt1_f32",
    "kernel_dense_exact_qmm_4_nt1_rt2_f32",
    "kernel_dense_exact_qmm_4_nt2_rt2_f32",
    "kernel_dense_exact_qmm_4_nt4_rt2_f32",
    "kernel_dense_exact_qmm_8_nt1_rt1_f32",
    "kernel_dense_exact_qmm_8_nt2_rt1_f32",
    "kernel_dense_exact_qmm_8_nt4_rt1_f32",
    "kernel_dense_exact_qmm_8_nt1_rt2_f32",
    "kernel_dense_exact_qmm_8_nt2_rt2_f32",
    "kernel_dense_exact_qmm_8_nt4_rt2_f32",
];

/// A packed weight and its inputs on the device.
struct Fixture {
    w: DeviceBuffer,
    scales: DeviceBuffer,
    biases: DeviceBuffer,
    x: DeviceBuffer,
    host_w: Vec<u32>,
    host_s: Vec<bf16>,
    host_b: Vec<bf16>,
    host_x: Vec<f32>,
    k: usize,
    n: usize,
    bits: u32,
}

/// A deterministic generator for fixtures.
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}

impl Fixture {
    fn new(gpu: &Gpu, k: usize, n: usize, rows: usize, bits: u32, seed: u64) -> Self {
        let mut lcg = Lcg(seed);
        let words = k * bits as usize / 32;
        let groups = k / 64;
        let host_w: Vec<u32> = (0..n * words).map(|_| lcg.next_u32()).collect();
        let host_s: Vec<bf16> = (0..n * groups)
            .map(|_| bf16::from_f32(lcg.next_f32() * 0.02))
            .collect();
        let host_b: Vec<bf16> = (0..n * groups)
            .map(|_| bf16::from_f32(lcg.next_f32() * 0.1))
            .collect();
        let host_x: Vec<f32> = (0..rows * k).map(|_| lcg.next_f32()).collect();
        Self {
            w: gpu.upload(bytemuck::cast_slice(&host_w)),
            scales: gpu.upload(bytemuck::cast_slice(&host_s)),
            biases: gpu.upload(bytemuck::cast_slice(&host_b)),
            x: gpu.upload(&f32_bytes(&host_x)),
            host_w,
            host_s,
            host_b,
            host_x,
            k,
            n,
            bits,
        }
    }

    /// The host dequantize-then-matmul, row `r` of the inputs.
    fn host_row(&self, r: usize) -> Vec<f32> {
        let bits = self.bits as usize;
        let per = 32 / bits;
        let words = self.k / per;
        let groups = self.k / 64;
        let mask = (1u32 << bits) - 1;
        let x = &self.host_x[r * self.k..(r + 1) * self.k];
        (0..self.n)
            .map(|o| {
                let mut acc = 0.0f64;
                for kk in 0..self.k {
                    let word = self.host_w[o * words + kk / per];
                    let q = (word >> ((kk % per) * bits)) & mask;
                    let g = kk / 64;
                    let s = f64::from(f32::from(self.host_s[o * groups + g]));
                    let b = f64::from(f32::from(self.host_b[o * groups + g]));
                    acc += (f64::from(q) * s + b) * f64::from(x[kk]);
                }
                acc as f32
            })
            .collect()
    }
}

/// The one-row launch: entry, grid, block.
fn qmv_plan(bits: u32, n: u32, k: u32) -> (&'static str, Extent3, Extent3) {
    let (nr, sgs) = match (n > 2048, k >= 4096) {
        (false, _) => (1, 8),
        (true, false) => (2, 2),
        (true, true) => (2, 4),
    };
    let per_tg = sgs * (32 / SPLITS) * nr;
    let entry = match (bits, nr) {
        (4, 1) => "kernel_dense_exact_qmv_4_nr1_f32",
        (4, _) => "kernel_dense_exact_qmv_4_nr2_f32",
        (8, 1) => "kernel_dense_exact_qmv_8_nr1_f32",
        _ => "kernel_dense_exact_qmv_8_nr2_f32",
    };
    (
        entry,
        Extent3::linear(n.div_ceil(per_tg)),
        Extent3::linear(sgs * 32),
    )
}

/// The any-row launch: entry, grid, block.
fn qmm_plan(bits: u32, rows: u32, n: u32) -> (&'static str, Extent3, Extent3) {
    let rt = rows.div_ceil(8).min(2);
    let nt = if n.is_multiple_of(32) {
        4
    } else if n.is_multiple_of(16) {
        2
    } else {
        1
    };
    let entry = match (bits, nt, rt) {
        (4, 1, 1) => "kernel_dense_exact_qmm_4_nt1_rt1_f32",
        (4, 2, 1) => "kernel_dense_exact_qmm_4_nt2_rt1_f32",
        (4, 4, 1) => "kernel_dense_exact_qmm_4_nt4_rt1_f32",
        (4, 1, _) => "kernel_dense_exact_qmm_4_nt1_rt2_f32",
        (4, 2, _) => "kernel_dense_exact_qmm_4_nt2_rt2_f32",
        (4, _, _) => "kernel_dense_exact_qmm_4_nt4_rt2_f32",
        (_, 1, 1) => "kernel_dense_exact_qmm_8_nt1_rt1_f32",
        (_, 2, 1) => "kernel_dense_exact_qmm_8_nt2_rt1_f32",
        (_, 4, 1) => "kernel_dense_exact_qmm_8_nt4_rt1_f32",
        (_, 1, _) => "kernel_dense_exact_qmm_8_nt1_rt2_f32",
        (_, 2, _) => "kernel_dense_exact_qmm_8_nt2_rt2_f32",
        _ => "kernel_dense_exact_qmm_8_nt4_rt2_f32",
    };
    (
        entry,
        Extent3::new(n.div_ceil(8 * nt), rows.div_ceil(8 * rt), 1),
        Extent3::linear(SPLITS * 32),
    )
}

/// Runs one launch over `rows` rows of `x` starting at row `first`, returning
/// the outputs.
fn launch(gpu: &Gpu, f: &Fixture, first: usize, rows: usize, one_row_kernel: bool) -> Vec<f32> {
    let (entry, grid, block) = if one_row_kernel {
        assert_eq!(rows, 1, "the matvec takes one row");
        qmv_plan(f.bits, f.n as u32, f.k as u32)
    } else {
        qmm_plan(f.bits, rows as u32, f.n as u32)
    };
    let out = gpu.upload(&f32_bytes(&vec![0.0f32; rows * f.n]));
    let x =
        f.x.view()
            .slice((first * f.k * 4) as u64, (rows * f.k * 4) as u64)
            .expect("the rows lie inside the fixture's inputs");
    let args = ExactAffineArgs {
        in_dim: f.k as u32,
        out_dim: f.n as u32,
        n_rows: rows as u32,
        pad: 0,
    };
    run(
        gpu,
        KernelRef::new("dense", entry),
        &LaunchArgs {
            grid,
            block,
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(f.w.view()),
                KernelArg::Buffer(f.scales.view()),
                KernelArg::Buffer(f.biases.view()),
                KernelArg::Buffer(x),
                KernelArg::Buffer(out.view()),
            ],
        },
    );
    drain(gpu);
    gpu.read_f32(&out)
}

/// Checks `got` against the host within a float tolerance scaled by the row.
fn check_value(got: &[f32], want: &[f32], what: &str) {
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
    let mut worst = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let diff = (g - w).abs();
        worst = worst.max(diff);
        assert!(
            diff <= 1e-4 * scale,
            "{what}: output {i} is {g}, the host says {w} (diff {diff}, row scale {scale})"
        );
    }
    println!("{what}: worst |diff| {worst:e} at row scale {scale}");
}

/// Both kernels agree with the host at one shape.
fn values_at(gpu: &Gpu, k: usize, n: usize, bits: u32) {
    let rows = 11;
    let f = Fixture::new(
        gpu,
        k,
        n,
        rows,
        bits,
        0x5eed ^ (k * n) as u64 ^ u64::from(bits),
    );
    let want0 = f.host_row(0);
    check_value(&launch(gpu, &f, 0, 1, true), &want0, "qmv row 0");
    let all = launch(gpu, &f, 0, rows, false);
    for r in [0, 5, rows - 1] {
        check_value(&all[r * n..(r + 1) * n], &f.host_row(r), "qmm row");
    }
}

/// Row `r` of every call width equals the same row alone, and the matvec
/// equals the matrix kernel's one-row answer, bit for bit.
fn rows_are_exact_at(gpu: &Gpu, k: usize, n: usize, bits: u32) {
    let max_rows = 33;
    let f = Fixture::new(
        gpu,
        k,
        n,
        max_rows,
        bits,
        0xabc ^ (k + n) as u64 ^ u64::from(bits),
    );
    let alone: Vec<Vec<f32>> = (0..max_rows)
        .map(|r| launch(gpu, &f, r, 1, false))
        .collect();
    for r in 0..max_rows {
        let mv = launch(gpu, &f, r, 1, true);
        let same = mv
            .iter()
            .zip(&alone[r])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(
            same,
            "{k}x{n} {bits}-bit: the matvec's row {r} differs from the matrix kernel's one-row answer"
        );
    }
    for width in 2..=max_rows {
        let got = launch(gpu, &f, 0, width, false);
        for r in 0..width {
            let row = &got[r * n..(r + 1) * n];
            let same = row
                .iter()
                .zip(&alone[r])
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(
                same,
                "{k}x{n} {bits}-bit: row {r} of a {width}-row call differs from the row alone"
            );
        }
    }
}

/// Both 4-bit kernels agree with the host dequantize-then-matmul.
pub fn values_match_the_host_4bit(gpu: &Gpu) {
    values_at(gpu, 512, 96, 4);
    values_at(gpu, 2560, 2056, 4);
    // Long contractions: several staging batches, and the wide launch that
    // keeps four simdgroups on one staging.
    values_at(gpu, 5120, 4104, 4);
    values_at(gpu, 17408, 2056, 4);
}

/// Both 8-bit kernels agree with the host dequantize-then-matmul.
pub fn values_match_the_host_8bit(gpu: &Gpu) {
    values_at(gpu, 512, 96, 8);
    values_at(gpu, 2560, 2056, 8);
    values_at(gpu, 5120, 4104, 8);
}

/// Row exactness at small shapes, including an output count that is a multiple
/// of 8 but not of 16 or 32 (the narrowest tile) and a contraction shorter
/// than one chunk per split.
pub fn rows_are_exact_at_small_shapes(gpu: &Gpu) {
    for bits in [4, 8] {
        rows_are_exact_at(gpu, 256, 40, bits);
        rows_are_exact_at(gpu, 1024, 48, bits);
        rows_are_exact_at(gpu, 2560, 512, bits);
    }
}

/// Row exactness at the dense 27B checkpoint's projection shapes.
pub fn rows_are_exact_at_model_shapes(gpu: &Gpu) {
    for (k, n) in [
        (5120, 17408),
        (17408, 5120),
        (5120, 10240),
        (6144, 5120),
        (5120, 48),
    ] {
        rows_are_exact_at(gpu, k, n, 4);
    }
    for (k, n) in [(2560, 10240), (10240, 2560)] {
        rows_are_exact_at(gpu, k, n, 8);
    }
}
