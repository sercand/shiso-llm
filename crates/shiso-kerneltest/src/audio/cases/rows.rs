//! The multi-row matrix-vector kernels: a block of rows through one pass over
//! the weights must give every row, bit for bit, what the one-row kernels
//! give it - they are how a decoder batch keeps each sequence's ids.

use shiso_core::{DeviceBuffer, Extent3, KernelArg};

use super::affine::affine;
use super::fused::same;
use super::{GemmArgs, MATVEC_BLOCK, MATVEC_COLS, blocks, launch, weight_fixture};
use crate::{Gpu, drain, f32_bytes, oracle};

/// Input rows one multi-row threadgroup takes.
const MATVEC_ROWS: u32 = 8;

/// One weight as the kernels bind it: the main buffer and, for an affine
/// weight, its scales and biases.
struct Weight {
    main: DeviceBuffer,
    aux: Option<(DeviceBuffer, DeviceBuffer)>,
}

fn weight(gpu: &Gpu, kind: &str, n: usize, k: usize, seed: u32) -> Weight {
    if kind == "affine8w" {
        let a = affine(gpu, n, k, seed);
        return Weight {
            main: a.q,
            aux: Some((a.scales, a.biases)),
        };
    }
    Weight {
        main: gpu.upload(&weight_fixture(kind, n * k, seed).0),
        aux: None,
    }
}

fn entry(kind: &str, silu: bool, rows: bool) -> &'static str {
    match (kind, silu, rows) {
        ("f32w", false, false) => "kernel_audio_matvec_f32w",
        ("bf16w", false, false) => "kernel_audio_matvec_bf16w",
        ("f16w", false, false) => "kernel_audio_matvec_f16w",
        ("q8_0w", false, false) => "kernel_audio_matvec_q8_0w",
        ("affine8w", false, false) => "kernel_audio_matvec_affine8w",
        ("f32w", false, true) => "kernel_audio_matvec_rows_f32w",
        ("bf16w", false, true) => "kernel_audio_matvec_rows_bf16w",
        ("f16w", false, true) => "kernel_audio_matvec_rows_f16w",
        ("q8_0w", false, true) => "kernel_audio_matvec_rows_q8_0w",
        ("affine8w", false, true) => "kernel_audio_matvec_rows_affine8w",
        ("f32w", true, false) => "kernel_audio_matvec_silu_f32w",
        ("bf16w", true, false) => "kernel_audio_matvec_silu_bf16w",
        ("f16w", true, false) => "kernel_audio_matvec_silu_f16w",
        ("q8_0w", true, false) => "kernel_audio_matvec_silu_q8_0w",
        ("affine8w", true, false) => "kernel_audio_matvec_silu_affine8w",
        ("f32w", true, true) => "kernel_audio_matvec_silu_rows_f32w",
        ("bf16w", true, true) => "kernel_audio_matvec_silu_rows_bf16w",
        ("f16w", true, true) => "kernel_audio_matvec_silu_rows_f16w",
        ("q8_0w", true, true) => "kernel_audio_matvec_silu_rows_q8_0w",
        _ => "kernel_audio_matvec_silu_rows_affine8w",
    }
}

/// `m` rows of `x` through `ws` (one weight, or the gate and up pair with
/// `silu`), on the one-row kernel (a threadgroup row per input row) or the
/// multi-row one; returns the `[m][n]` output and the guard row past it.
fn run(
    gpu: &Gpu,
    (kind, silu, rows): (&str, bool, bool),
    (m, n, k): (usize, usize, usize),
    x: &DeviceBuffer,
    ws: &[Weight],
    bias: Option<&DeviceBuffer>,
) -> Vec<f32> {
    let y = gpu.upload(&f32_bytes(&vec![f32::NAN; m * n + n]));
    let args = GemmArgs {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        flags: u32::from(bias.is_some()),
        k_chunk: k as u32,
        splits: 1,
    };
    let mut bind = vec![
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(y.view()),
        KernelArg::Buffer(x.view()),
    ];
    bind.extend(ws.iter().map(|w| KernelArg::Buffer(w.main.view())));
    if !silu {
        bind.push(KernelArg::Buffer(bias.unwrap_or(x).view()));
    }
    // An affine weight's scales, then its biases, each for every weight.
    if ws[0].aux.is_some() {
        let aux: Vec<&(DeviceBuffer, DeviceBuffer)> =
            ws.iter().filter_map(|w| w.aux.as_ref()).collect();
        bind.extend(aux.iter().map(|a| KernelArg::Buffer(a.0.view())));
        bind.extend(aux.iter().map(|a| KernelArg::Buffer(a.1.view())));
    }
    let grid_rows = if rows {
        blocks(m, MATVEC_ROWS)
    } else {
        m as u32
    };
    launch(
        gpu,
        entry(kind, silu, rows),
        Extent3::new(blocks(n, MATVEC_COLS), grid_rows, 1),
        Extent3::linear(MATVEC_BLOCK),
        0,
        &bind,
    );
    drain(gpu);
    gpu.read_f32(&y)
}

/// The multi-row product and gated pair over every weight type the one-row
/// kernels read: one row, a partial block, a whole block and a block and a
/// half, a column count that ends mid-threadgroup, with and without the
/// bias - each row bit for bit the one-row kernel's, nothing written past
/// `m x n`.
///
/// # Panics
///
/// When a row differs from the one-row kernel's, or a kernel writes past its
/// output.
pub fn matvec_rows_are_each_row_alone(gpu: &Gpu) {
    let (n, k) = (37usize, 1152usize);
    for kind in ["f32w", "bf16w", "f16w", "q8_0w", "affine8w"] {
        let ws = [weight(gpu, kind, n, k, 351), weight(gpu, kind, n, k, 352)];
        let bias = gpu.upload(&f32_bytes(&oracle::signal(n, 353)));
        for m in [1usize, 3, 8, 13] {
            let x = gpu.upload(&f32_bytes(&oracle::signal(m * k, 354 + m as u32)));
            for (silu, with_bias) in [(false, true), (false, false), (true, false)] {
                let w = if silu { &ws[..] } else { &ws[..1] };
                let b = with_bias.then_some(&bias);
                let what = format!("{kind} silu {silu} bias {with_bias} {m} rows");
                let want = run(gpu, (kind, silu, false), (m, n, k), &x, w, b);
                let got = run(gpu, (kind, silu, true), (m, n, k), &x, w, b);
                assert!(
                    want[..m * n].iter().all(|v| v.is_finite()),
                    "{what}: the one-row kernel wrote every value"
                );
                same(&got[..m * n], &want[..m * n], &what);
                assert!(
                    got[m * n..].iter().all(|v| v.is_nan()),
                    "{what}: wrote past the output"
                );
            }
        }
    }
}
