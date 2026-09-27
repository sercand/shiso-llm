//! The audio set's own products over an MLX 8-bit affine weight (groups of
//! 64, a bf16 scale and bias each): the tiled GEMMs against the plainest
//! reading of `y = x * (q * scale + bias)^T (+ b)`, the one-product
//! matrix-vector kernel against the same, and the three-product and gated
//! kernels against the one-product chain they replace, value for value.

use shiso_core::{DeviceBuffer, Extent3, KernelArg};

use super::fused::{Matvec3Args, same};
use super::{
    GEMM_BLOCK, GEMM_SMEM, GEMM_TILE, GEMM_TILE_N, GEMM64_BLOCK, GEMM64_SMEM, GEMM64_TILE_M,
    GemmArgs, LINEAR_BLOCK, MATVEC_BLOCK, MATVEC_COLS, SiluMulArgs, blocks, close, gemm_oracle,
    launch, zeros,
};
use crate::{Gpu, drain, f32_bytes, oracle};

/// Elements of one scale-and-bias group.
const GROUP: usize = 64;

/// An `[n][k]` affine weight on the device, and its values as the kernels
/// form them (`q * scale + bias`, fused, in f32).
struct Affine {
    q: DeviceBuffer,
    scales: DeviceBuffer,
    biases: DeviceBuffer,
    values: Vec<f32>,
}

fn affine(gpu: &Gpu, n: usize, k: usize, seed: u32) -> Affine {
    let bytes: Vec<u8> = oracle::signal(n * k, seed)
        .iter()
        .map(|v| ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8)
        .collect();
    let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
    let widen = |b: u16| f32::from_bits(u32::from(b) << 16);
    let scales: Vec<u16> = oracle::signal(n * k / GROUP, seed + 1)
        .iter()
        .map(|v| bf16(0.002 + 0.001 * v))
        .collect();
    let biases: Vec<u16> = oracle::signal(n * k / GROUP, seed + 2)
        .iter()
        .map(|v| bf16(-0.2 + 0.05 * v))
        .collect();
    let values = bytes
        .iter()
        .enumerate()
        .map(|(i, &q)| f32::from(q).mul_add(widen(scales[i / GROUP]), widen(biases[i / GROUP])))
        .collect();
    let halves = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|b| b.to_le_bytes()).collect() };
    Affine {
        q: gpu.upload(&bytes),
        scales: gpu.upload(&halves(&scales)),
        biases: gpu.upload(&halves(&biases)),
        values,
    }
}

fn gemm_args(m: usize, n: usize, k: usize, bias: bool) -> GemmArgs {
    GemmArgs {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        flags: u32::from(bias),
        k_chunk: k as u32,
        splits: 1,
    }
}

/// The 32-row and 64-row tiled GEMMs over an affine weight, with and without
/// the bias, a ragged column count and rows that end mid-tile, writing
/// nothing past `m x n`; and the one-product matrix-vector kernel over one
/// row and a few.
///
/// # Panics
///
/// When a product is off its oracle or a kernel writes past its output.
pub fn affine8_products_match_the_dequantised_weight(gpu: &Gpu) {
    for &(m, n, k, bias, tall) in &[
        (20usize, 70usize, 128usize, true, false),
        (130, 96, 1024, true, true),
        (40, 37, 192, false, true),
    ] {
        let w = affine(gpu, n, k, 401);
        let x = oracle::signal(m * k, 402);
        let b = oracle::signal(n, 403);
        let (xb, bb) = (gpu.upload(&f32_bytes(&x)), gpu.upload(&f32_bytes(&b)));
        let y = gpu.upload(&f32_bytes(&vec![f32::NAN; m * n + n]));
        let args = gemm_args(m, n, k, bias);
        let (entry, tile_m, block, smem) = if tall {
            (
                "kernel_audio_gemm64_affine8w",
                GEMM64_TILE_M,
                GEMM64_BLOCK,
                GEMM64_SMEM,
            )
        } else {
            (
                "kernel_audio_gemm_affine8w",
                GEMM_TILE,
                GEMM_BLOCK,
                GEMM_SMEM,
            )
        };
        launch(
            gpu,
            entry,
            Extent3::new(blocks(n, GEMM_TILE_N), blocks(m, tile_m), 1),
            Extent3::linear(block),
            smem,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(y.view()),
                KernelArg::Buffer(xb.view()),
                KernelArg::Buffer(w.q.view()),
                KernelArg::Buffer(bb.view()),
                KernelArg::Buffer(w.scales.view()),
                KernelArg::Buffer(w.biases.view()),
            ],
        );
        drain(gpu);
        let got = gpu.read_f32(&y);
        let want = gemm_oracle(&x, &w.values, bias.then_some(&b[..]), m, n, k);
        let what = format!("affine {entry} {m}x{n}x{k}");
        close(&got[..m * n], &want, 1e-4, &what);
        assert!(
            got[m * n..].iter().all(|v| v.is_nan()),
            "{what}: wrote past"
        );
    }
    for &(m, n, k, bias) in &[(1usize, 1301usize, 1024usize, true), (3, 37, 128, false)] {
        let w = affine(gpu, n, k, 411);
        let x = oracle::signal(m * k, 412);
        let b = oracle::signal(n, 413);
        let (xb, bb) = (gpu.upload(&f32_bytes(&x)), gpu.upload(&f32_bytes(&b)));
        let y = gpu.upload(&f32_bytes(&vec![f32::NAN; m * n + n]));
        let args = gemm_args(m, n, k, bias);
        matvec_affine(gpu, &args, &y, &xb, &w, &bb);
        drain(gpu);
        let got = gpu.read_f32(&y);
        let want = gemm_oracle(&x, &w.values, bias.then_some(&b[..]), m, n, k);
        close(
            &got[..m * n],
            &want,
            1e-4,
            &format!("affine matvec {m}x{n}x{k}"),
        );
        assert!(
            got[m * n..].iter().all(|v| v.is_nan()),
            "affine matvec: wrote past"
        );
    }
}

fn matvec_affine(
    gpu: &Gpu,
    args: &GemmArgs,
    y: &DeviceBuffer,
    x: &DeviceBuffer,
    w: &Affine,
    b: &DeviceBuffer,
) {
    launch(
        gpu,
        "kernel_audio_matvec_affine8w",
        Extent3::new(blocks(args.n as usize, MATVEC_COLS), args.m, 1),
        Extent3::linear(MATVEC_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(args)),
            KernelArg::Buffer(y.view()),
            KernelArg::Buffer(x.view()),
            KernelArg::Buffer(w.q.view()),
            KernelArg::Buffer(b.view()),
            KernelArg::Buffer(w.scales.view()),
            KernelArg::Buffer(w.biases.view()),
        ],
    );
}

/// The three-product and gated kernels over affine weights equal the
/// one-product kernel's chain, value for value.
///
/// # Panics
///
/// When a value differs from the chain's.
pub fn affine8_fused_products_are_the_chain(gpu: &Gpu) {
    let (k, ns) = (1024usize, [37usize, 16, 9]);
    let x = gpu.upload(&f32_bytes(&oracle::signal(k, 421)));
    let ws: Vec<Affine> = ns
        .iter()
        .enumerate()
        .map(|(i, &n)| affine(gpu, n, k, 422 + 7 * i as u32))
        .collect();
    let ones: Vec<DeviceBuffer> = ns.iter().map(|&n| zeros(gpu, n)).collect();
    for ((y, w), &n) in ones.iter().zip(&ws).zip(&ns) {
        matvec_affine(gpu, &gemm_args(1, n, k, false), y, &x, w, &x);
    }
    let threes: Vec<DeviceBuffer> = ns.iter().map(|&n| zeros(gpu, n)).collect();
    let args = Matvec3Args {
        n0: ns[0] as u32,
        n1: ns[1] as u32,
        n2: ns[2] as u32,
        k: k as u32,
    };
    let groups: usize = ns.iter().map(|n| n.div_ceil(MATVEC_COLS as usize)).sum();
    let mut bind = vec![
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(threes[0].view()),
        KernelArg::Buffer(threes[1].view()),
        KernelArg::Buffer(threes[2].view()),
        KernelArg::Buffer(x.view()),
    ];
    bind.extend(ws.iter().map(|w| KernelArg::Buffer(w.q.view())));
    bind.extend(ws.iter().map(|w| KernelArg::Buffer(w.scales.view())));
    bind.extend(ws.iter().map(|w| KernelArg::Buffer(w.biases.view())));
    launch(
        gpu,
        "kernel_audio_matvec3_affine8w",
        Extent3::linear(groups as u32),
        Extent3::linear(MATVEC_BLOCK),
        0,
        &bind,
    );
    drain(gpu);
    for (i, (a, b)) in threes.iter().zip(&ones).enumerate() {
        same(
            &gpu.read_f32(a),
            &gpu.read_f32(b),
            &format!("affine matvec3 part {i}"),
        );
    }
    // The gated pair over two rows.
    let (n, rows) = (37usize, 2usize);
    let xs = oracle::signal(rows * k, 431);
    let (wg, wu) = (affine(gpu, n, k, 432), affine(gpu, n, k, 439));
    let mut want = Vec::new();
    for r in 0..rows {
        let xr = gpu.upload(&f32_bytes(&xs[r * k..(r + 1) * k]));
        let (g, u) = (zeros(gpu, n), zeros(gpu, n));
        matvec_affine(gpu, &gemm_args(1, n, k, false), &g, &xr, &wg, &xr);
        matvec_affine(gpu, &gemm_args(1, n, k, false), &u, &xr, &wu, &xr);
        let silu = SiluMulArgs { count: n as u32 };
        launch(
            gpu,
            "kernel_audio_silu_mul_f32",
            Extent3::linear(blocks(n, LINEAR_BLOCK)),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&silu)),
                KernelArg::Buffer(g.view()),
                KernelArg::Buffer(g.view()),
                KernelArg::Buffer(u.view()),
            ],
        );
        drain(gpu);
        want.extend(gpu.read_f32(&g));
    }
    let xb = gpu.upload(&f32_bytes(&xs));
    let y = zeros(gpu, rows * n);
    let args = gemm_args(rows, n, k, false);
    launch(
        gpu,
        "kernel_audio_matvec_silu_affine8w",
        Extent3::new(blocks(n, MATVEC_COLS), rows as u32, 1),
        Extent3::linear(MATVEC_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(y.view()),
            KernelArg::Buffer(xb.view()),
            KernelArg::Buffer(wg.q.view()),
            KernelArg::Buffer(wu.q.view()),
            KernelArg::Buffer(wg.scales.view()),
            KernelArg::Buffer(wu.scales.view()),
            KernelArg::Buffer(wg.biases.view()),
            KernelArg::Buffer(wu.biases.view()),
        ],
    );
    drain(gpu);
    same(&gpu.read_f32(&y), &want, "affine matvec_silu");
}
