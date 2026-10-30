//! The `q8_0` projection over a few activation rows (`dense_q8_rows.metal`):
//! against the host product, and a row's answer the same whatever else shares
//! the launch.

#![expect(
    clippy::cast_possible_truncation,
    reason = "fixture dimensions are small"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use shiso_core::{Extent3, KernelArg, KernelRef, LaunchArgs};

/// Every instantiation, by row count.
pub const ENTRIES: [&str; 8] = [
    "kernel_dense_q8_0_rows_f32_r1",
    "kernel_dense_q8_0_rows_f32_r2",
    "kernel_dense_q8_0_rows_f32_r3",
    "kernel_dense_q8_0_rows_f32_r4",
    "kernel_dense_q8_0_rows_f32_r5",
    "kernel_dense_q8_0_rows_f32_r6",
    "kernel_dense_q8_0_rows_f32_r7",
    "kernel_dense_q8_0_rows_f32_r8",
];

const IN: usize = 2688;
const OUT: usize = 37;

/// `[OUT][IN/32]` blocks: an f16 scale then 32 signed bytes.
fn weights() -> (Vec<u8>, Vec<f64>) {
    let scales = crate::oracle::signal(OUT * IN / 32, 81);
    let mut bytes = Vec::with_capacity(OUT * IN / 32 * 34);
    let mut dense = Vec::with_capacity(OUT * IN);
    for (b, s) in scales.iter().enumerate() {
        let d = half::f16::from_f32(s * 0.01);
        bytes.extend_from_slice(&d.to_bits().to_le_bytes());
        for j in 0..32u32 {
            let raw = ((b as u32 * 32 + j).wrapping_mul(2_654_435_761) >> 24) as u8;
            let q = raw.cast_signed();
            bytes.push(raw);
            dense.push(f64::from(d.to_f32()) * f64::from(q));
        }
    }
    (bytes, dense)
}

fn launch(gpu: &Gpu, rows: usize, w: &[u8], x: &[f32]) -> Vec<f32> {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        in_dim: u32,
        out_dim: u32,
    }
    let d_w = gpu.upload(w);
    let d_x = gpu.upload(&f32_bytes(x));
    let d_y = gpu.upload(&f32_bytes(&vec![0.0; rows * OUT]));
    let args = Args {
        in_dim: IN as u32,
        out_dim: OUT as u32,
    };
    run(
        gpu,
        KernelRef::new("dense", ENTRIES[rows - 1]),
        &LaunchArgs {
            grid: Extent3::linear((OUT as u32).div_ceil(2)),
            block: Extent3::linear(128),
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d_w.view()),
                KernelArg::Buffer(d_x.view()),
                KernelArg::Buffer(d_y.view()),
            ],
        },
    );
    drain(gpu);
    gpu.read_f32(&d_y)
}

/// Every row count against the host product.
pub fn q8_rows_matches_the_host_product(gpu: &Gpu) {
    let (w, dense) = weights();
    for rows in 1..=8usize {
        let x = crate::oracle::signal(rows * IN, 83);
        let got = launch(gpu, rows, &w, &x);
        for t in 0..rows {
            for o in 0..OUT {
                let want: f64 = (0..IN)
                    .map(|k| dense[o * IN + k] * f64::from(x[t * IN + k]))
                    .sum();
                let have = f64::from(got[t * OUT + o]);
                assert!(
                    (have - want).abs() <= 1e-3 * (1.0 + want.abs()),
                    "rows {rows}, row {t}, output {o}: {have} against {want}"
                );
            }
        }
    }
}

/// Row 0's answer is bit-identical alone and sharing the launch with seven
/// other rows.
pub fn q8_rows_answer_does_not_depend_on_the_row_count(gpu: &Gpu) {
    let (w, _) = weights();
    let x = crate::oracle::signal(8 * IN, 85);
    let two = launch(gpu, 1, &w, &x[..IN]);
    let eight = launch(gpu, 8, &w, &x);
    assert!(
        two[..OUT]
            .iter()
            .zip(&eight[..OUT])
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "row 0 changed with the row count"
    );
}
