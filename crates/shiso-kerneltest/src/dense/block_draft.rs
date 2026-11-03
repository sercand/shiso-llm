//! Cases for the block drafter's two kernels (`dense_block_draft.metal`).
//!
//! Both are held to a host oracle: the strided row copy exactly, since it
//! moves bits, and the grouped dynamic causal convolution within a float
//! tolerance, since its host twin folds the same terms in the same order but
//! need not round identically.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "fixture dimensions are small positive integers"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use bytemuck::{Pod, Zeroable};
use shiso_core::{Extent3, KernelArg, KernelRef, LaunchArgs};

/// Every entry point this module's cases launch.
pub const ENTRIES: [&str; 2] = ["kernel_dense_copy_rows_f32", "kernel_dense_dyn_conv_f32"];

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CopyRowsArgs {
    n_rows: u32,
    width: u32,
    dst_stride: u32,
    dst_offset: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DynConvArgs {
    n_rows: u32,
    hidden: u32,
    group: u32,
    taps: u32,
    part: u32,
    pad: [u32; 3],
}

fn signal(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i as u32).wrapping_mul(2_654_435_761) ^ seed) % 1000) as f32 / 500.0 - 1.0)
        .collect()
}

fn launch(gpu: &Gpu, entry: &'static str, n: u32, args: &[KernelArg<'_>]) {
    run(
        gpu,
        KernelRef::new("dense", entry),
        &LaunchArgs {
            grid: Extent3::linear(n.div_ceil(256)),
            block: Extent3::linear(256),
            smem_bytes: 0,
            args,
        },
    );
    drain(gpu);
}

/// Each row lands at its stride and offset, and nothing else in the
/// destination moves.
pub fn copy_rows_lands_each_row_in_its_slot(gpu: &Gpu) {
    let (rows, width, slots, slot) = (5usize, 96usize, 3usize, 1usize);
    let src = signal(rows * width, 7);
    let before = signal(rows * width * slots, 8);
    let src_buf = gpu.upload(&f32_bytes(&src));
    let dst_buf = gpu.upload(&f32_bytes(&before));
    let args = CopyRowsArgs {
        n_rows: rows as u32,
        width: width as u32,
        dst_stride: (width * slots) as u32,
        dst_offset: (width * slot) as u32,
    };
    launch(
        gpu,
        "kernel_dense_copy_rows_f32",
        (rows * width) as u32,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(src_buf.view()),
            KernelArg::Buffer(dst_buf.view()),
        ],
    );
    let got = gpu.read_f32(&dst_buf);
    for r in 0..rows {
        for s in 0..slots {
            for c in 0..width {
                let at = r * width * slots + s * width + c;
                let want = if s == slot {
                    src[r * width + c]
                } else {
                    before[at]
                };
                assert_eq!(
                    got[at].to_bits(),
                    want.to_bits(),
                    "row {r} slot {s} column {c}"
                );
            }
        }
    }
}

/// Both halves of the convolution against the host rule: causal along the
/// rows, a static kernel per channel and a dynamic one per channel group.
pub fn dyn_conv_matches_the_host_rule(gpu: &Gpu) {
    let (rows, hidden, group, taps) = (8usize, 64usize, 16usize, 2usize);
    let groups = hidden / group;
    let x = signal(rows * hidden, 11);
    let dynk = signal(rows * 2 * taps * groups, 12);
    let base = signal(2 * taps * hidden, 13);
    for part in 0..2usize {
        let x_buf = gpu.upload(&f32_bytes(&x));
        let d_buf = gpu.upload(&f32_bytes(&dynk));
        let b_buf = gpu.upload(&f32_bytes(&base));
        let out_buf = gpu.upload(&f32_bytes(&vec![0.0; rows * hidden]));
        let args = DynConvArgs {
            n_rows: rows as u32,
            hidden: hidden as u32,
            group: group as u32,
            taps: taps as u32,
            part: part as u32,
            pad: [0; 3],
        };
        launch(
            gpu,
            "kernel_dense_dyn_conv_f32",
            (rows * hidden) as u32,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(x_buf.view()),
                KernelArg::Buffer(d_buf.view()),
                KernelArg::Buffer(b_buf.view()),
                KernelArg::Buffer(out_buf.view()),
            ],
        );
        let got = gpu.read_f32(&out_buf);
        for t in 0..rows {
            for c in 0..hidden {
                let mut want = 0.0f32;
                for o in 0..taps.min(t + 1) {
                    let v = x[(t - o) * hidden + c];
                    want += base[(part * taps + o) * hidden + c] * v;
                    want += dynk[((t * 2 + part) * taps + o) * groups + c / group] * v;
                }
                let have = got[t * hidden + c];
                assert!(
                    (have - want).abs() <= 1e-5 * (1.0 + want.abs()),
                    "part {part} row {t} channel {c}: {have} against {want}"
                );
            }
        }
    }
}

/// The CUDA-only multi-row projections a block drafter's `q4_0` and `q8_0`
/// weights opt into (`dense_rows_mv.cuh`). Not in [`ENTRIES`]: the Metal set
/// has no twin, and reads those weights through its extended matvec.
pub const ROWS_MV_ENTRIES: [&str; 6] = [
    "kernel_dense_rows_mv_q4_0_f32_r2",
    "kernel_dense_rows_mv_q4_0_f32_r4",
    "kernel_dense_rows_mv_q4_0_f32_r8",
    "kernel_dense_rows_mv_q8_0_f32_r2",
    "kernel_dense_rows_mv_q8_0_f32_r4",
    "kernel_dense_rows_mv_q8_0_f32_r8",
];

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RowsMvArgs {
    in_dim: u32,
    out_dim: u32,
    n_rows: u32,
    pad0: u32,
}

/// `count` quant blocks of 32 weights with varied scales, as stored: an `f16`
/// scale and then `q4_0` nibbles (`bytes == 18`) or `q8_0` bytes
/// (`bytes == 34`); and every weight decoded, block after block.
fn quant_blocks(count: usize, bytes: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let mut raw = Vec::with_capacity(count * bytes);
    let mut values = Vec::with_capacity(count * 32);
    for b in 0..count {
        let h = (b as u32).wrapping_mul(2_654_435_761) ^ seed;
        let d = half::f16::from_f32(0.002 + (h % 97) as f32 * 0.0005);
        raw.extend_from_slice(&d.to_le_bytes());
        let d = d.to_f32();
        let q = |j: u32| h.wrapping_add(j.wrapping_mul(40_503)).rotate_left(j % 31);
        if bytes == 18 {
            let mut decoded = [0.0f32; 32];
            for j in 0..16u32 {
                let byte = (q(j) & 0xff) as u8;
                raw.push(byte);
                decoded[j as usize] = (i32::from(byte & 0x0f) - 8) as f32 * d;
                decoded[j as usize + 16] = (i32::from(byte >> 4) - 8) as f32 * d;
            }
            values.extend_from_slice(&decoded);
        } else {
            for j in 0..32u32 {
                let byte = (q(j) & 0xff) as u8;
                raw.push(byte);
                values.push(f32::from(byte.cast_signed()) * d);
            }
        }
    }
    (raw, values)
}

/// Both multi-row projections, at one to eight rows and at widths that leave a
/// partial chunk of the contraction and a partial block of outputs, within a
/// float tolerance of a host product over the decoded weights.
///
/// # Panics
///
/// When an output strays from the host product.
pub fn rows_mv_matches_the_host_product(gpu: &Gpu) {
    for (i, entry) in ROWS_MV_ENTRIES.iter().copied().enumerate() {
        let bytes = if i < 3 { 18usize } else { 34usize };
        let per_warp = [2u32, 4, 8][i % 3];
        for (in_dim, out_dim) in [(2560u32, 640u32), (1088, 37), (4096, 1024), (2560, 4096)] {
            let blocks = (in_dim / 32) as usize;
            let (raw, weights) = quant_blocks(blocks * out_dim as usize, bytes, in_dim ^ out_dim);
            let w_dev = gpu.upload(&raw);
            for n_rows in 1..=8u32 {
                let x = signal((n_rows * in_dim) as usize, 7 + n_rows);
                let x_dev = gpu.upload(&f32_bytes(&x));
                let out_dev = gpu.upload(&vec![0u8; (n_rows * out_dim * 4) as usize]);
                let args = RowsMvArgs {
                    in_dim,
                    out_dim,
                    n_rows,
                    pad0: 0,
                };
                run(
                    gpu,
                    KernelRef::new("dense", entry),
                    &LaunchArgs {
                        grid: Extent3::linear(out_dim.div_ceil(4 * per_warp)),
                        block: Extent3::linear(128),
                        smem_bytes: 0,
                        args: &[
                            KernelArg::Bytes(bytemuck::bytes_of(&args)),
                            KernelArg::Buffer(w_dev.view()),
                            KernelArg::Buffer(x_dev.view()),
                            KernelArg::Buffer(out_dev.view()),
                        ],
                    },
                );
                drain(gpu);
                let got = gpu.read_f32(&out_dev);
                let k = in_dim as usize;
                for n in 0..n_rows as usize {
                    for o in 0..out_dim as usize {
                        let want: f64 = (0..k)
                            .map(|i| f64::from(weights[o * k + i]) * f64::from(x[n * k + i]))
                            .sum();
                        let g = f64::from(got[n * out_dim as usize + o]);
                        assert!(
                            (g - want).abs() <= 1e-4 * (1.0 + want.abs()),
                            "{entry} {in_dim}x{out_dim} rows {n_rows}: out[{n}][{o}] = {g}, \
                             expected {want}"
                        );
                    }
                }
            }
        }
    }
}
