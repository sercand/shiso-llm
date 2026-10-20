//! The decode-shaped trellis projection at every built width, graded against
//! the expansion.
//!
//! `exl3_dgemv` is shiso's own kernel over the vendored trellis decoder: it
//! stages tiles through shared memory in pieces whose count depends on the
//! width, and reads them back per warp by a tile index that does too. The
//! expansion (`reconstruct_had_kernel`) is the vendored kernel, which writes
//! the weight out in the original basis with both rotations and both scale
//! vectors applied. Both decode the same tiles through the same decoder, so a
//! staging or indexing slip at one width shows up as a disagreement, while the
//! decoder itself is the vendored one either way.
//!
//! The oracle is the expanded weight times the activation in `f64`. The two
//! paths round in different places - the projection rounds the rotated
//! activation to half precision, the expansion rounds the weight and its
//! rotation's butterflies - so they agree to a few parts in a thousand of the
//! output's RMS, where a misread tile is off by the output's whole size. Rows
//! of the same pass must also be bit-identical whatever else shares the pass.
//!
//! The geometry is transcribed from `shiso_engine::sparse::kernels` and
//! `sparse::matmul`, which this crate may not depend on; each constant names
//! the one it mirrors.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "kernel arguments are u32 and i32 while the fixtures index in usize; every \
              size here is a model dimension far below i32::MAX"
)]

use half::f16;
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::{f16_to_f32, signal};

/// The module every entry lives in, mirroring `kernels::MOD_EXL3`.
const MODULE: &str = "exl3";
/// Warps one decode-shaped block carries, mirroring `kernels::EXL3_DGEMV_WARPS`.
const DG_WARPS: u32 = 4;
/// Pipeline stages of the decode-shaped instances, mirroring
/// `kernels::EXL3_DGEMV_STAGES`.
const DG_STAGES: u32 = 4;
/// Contraction one stage covers, mirroring `kernels::EXL3_DGEMV_KT`.
const DG_KT: u32 = 64;
/// Threads one expansion block carries, mirroring `exl3::RECONSTRUCT_THREADS`.
const RECONSTRUCT_THREADS: u32 = 256;
/// The expansion's tile edge, mirroring `exl3::RECONSTRUCT_TILE_K` and `_N`.
const RECONSTRUCT_TILE: u32 = 128;

/// Shapes `(in_dim, out_dim)` of this family's dense trellis tensors: a square
/// hidden projection, the shared expert's down projection, and the narrow
/// indexer projection.
const SHAPES: [(u32, u32); 3] = [(2_560, 2_560), (640, 2_560), (2_560, 128)];

/// Row counts: one (a decode step), a verify pass, and one past sixteen, which
/// the engine runs on the two-tile instance.
const ROWS: [u32; 3] = [1, 5, 20];

/// RMS of the disagreement, as a fraction of the oracle's RMS, that rounding
/// may account for. Measured agreement is about 1e-3; a misread tile is ~1.
const RMS_TOLERANCE: f64 = 1e-2;

/// One trellis tensor on the device.
struct Trellis {
    /// `[k / 16, n / 16, 16 * bits]` words.
    tiles: DeviceBuffer,
    /// `k` half-precision input scales.
    suh: DeviceBuffer,
    /// `n` half-precision output scales.
    svh: DeviceBuffer,
}

/// Half-precision scales of mixed sign and a spread of magnitudes, as a
/// quantizer's sign-folded scales are.
fn scales(len: usize, seed: u32) -> Vec<u8> {
    signal(len, seed)
        .iter()
        .flat_map(|&v| {
            let magnitude = 0.5 + v.abs();
            let signed = if v < 0.0 { -magnitude } else { magnitude };
            f16::from_f32(signed).to_bits().to_le_bytes()
        })
        .collect()
}

/// A tensor of arbitrary codes: every word decodes to a finite weight, so
/// random words are a valid tensor of the width.
fn trellis(gpu: &Gpu, k: u32, n: u32, bits: u32) -> Trellis {
    let words = (k / 16) as usize * (n / 16) as usize * 16 * bits as usize;
    let mut state = 0x2545_f491_u32 ^ (k << 8) ^ n ^ (bits << 24);
    let tiles: Vec<u8> = (0..words)
        .flat_map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as u16).to_le_bytes()
        })
        .collect();
    Trellis {
        tiles: gpu.upload(&tiles),
        suh: gpu.upload(&scales(k as usize, 0x5a5a ^ k ^ bits)),
        svh: gpu.upload(&scales(n as usize, 0xa5a5 ^ n ^ bits)),
    }
}

/// A template instance of the exl3 module, as this device names it.
fn entry(gpu: &Gpu, logical: &str) -> KernelRef {
    gpu.device()
        .resolve_entry(MODULE, logical)
        .unwrap_or_else(|| panic!("the exl3 module has no entry {logical}"))
}

/// The weight the expansion writes: `[k][n]` half precision in the original
/// basis, widened.
fn expanded(gpu: &Gpu, w: &Trellis, k: u32, n: u32, bits: u32) -> Vec<f32> {
    let out = gpu.upload(&vec![0u8; k as usize * n as usize * 2]);
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(w.tiles.view()),
        KernelArg::Buffer(w.suh.view()),
        KernelArg::Buffer(w.svh.view()),
        KernelArg::I32((n / 16) as i32),
        KernelArg::I32(0),
    ];
    run(
        gpu,
        entry(gpu, &format!("_Z22reconstruct_had_kernelILi{bits}ELi2EEv")),
        &LaunchArgs {
            grid: Extent3::new(n / RECONSTRUCT_TILE, k / RECONSTRUCT_TILE, 1),
            block: Extent3::linear(RECONSTRUCT_THREADS),
            smem_bytes: 0,
            args: &args,
        },
    );
    drain(gpu);
    gpu.read(&out)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f16_to_f32(u16::from_le_bytes(*pair)))
        .collect()
}

/// Dynamic shared memory the engine launches the projection with, mirroring
/// `kernels::exl3_dgemv_smem`.
const fn dgemv_smem(bits: u32, tiles: u32) -> u32 {
    let tile_words = 256 * bits / 16;
    let a_stage = tiles * 16 * DG_KT;
    let b_stage = (DG_KT / 16) * DG_WARPS * tile_words;
    DG_STAGES * (a_stage * 2 + b_stage * 2)
}

/// One decode-shaped projection of `x` (`[rows, k]`) through a `bits`-wide
/// tensor: the input transform, then the slices and their fold. `[rows, n]`.
fn dgemv(gpu: &Gpu, w: &Trellis, x: &[f32], dims: (u32, u32, u32, u32), ksplit: u32) -> Vec<f32> {
    let (rows, k, n, bits) = dims;
    // The engine's choice, and it is made from the row count: two m16 tiles a
    // pass past sixteen rows.
    let tiles = if rows > 16 { 2 } else { 1 };
    let padded = rows.next_multiple_of(16);
    let xs = gpu.upload(&f32_bytes(x));
    let a_frag = gpu.upload(&vec![0u8; padded as usize * k as usize * 2]);
    let had_args = [
        KernelArg::Buffer(xs.view()),
        KernelArg::Buffer(a_frag.view()),
        KernelArg::Buffer(w.suh.view()),
        KernelArg::I32(rows as i32),
        KernelArg::I32(k as i32),
    ];
    let warps = padded * (k / 128);
    run(
        gpu,
        entry(gpu, "_Z14exl3_dgemv_had"),
        &LaunchArgs {
            grid: Extent3::linear(warps.div_ceil(8)),
            block: Extent3::linear(256),
            smem_bytes: 0,
            args: &had_args,
        },
    );
    let out = gpu.upload(&vec![0u8; rows as usize * n as usize * 4]);
    let part = gpu.upload(&vec![0u8; ksplit as usize * rows as usize * n as usize * 4]);
    let tickets = gpu.upload(&vec![0u8; (n / 128) as usize * 4]);
    let args = [
        KernelArg::Buffer(a_frag.view()),
        KernelArg::Buffer(w.tiles.view()),
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(part.view()),
        KernelArg::Buffer(tickets.view()),
        KernelArg::Buffer(w.svh.view()),
        KernelArg::I32(rows as i32),
        KernelArg::I32(k as i32),
        KernelArg::I32(n as i32),
        KernelArg::I32(ksplit as i32),
    ];
    run(
        gpu,
        entry(
            gpu,
            &format!("_Z10exl3_dgemvILi{bits}ELi2ELi{DG_STAGES}ELi{DG_WARPS}ELi{tiles}EEv"),
        ),
        &LaunchArgs {
            grid: Extent3::new(n / (16 * DG_WARPS), ksplit, 1),
            block: Extent3::linear(DG_WARPS * 32),
            smem_bytes: dgemv_smem(bits, tiles),
            args: &args,
        },
    );
    drain(gpu);
    let tickets_after = gpu.read(&tickets);
    assert!(
        tickets_after.iter().all(|&b| b == 0),
        "K = {bits}: the fold left a ticket set, so the next launch would skip its fold"
    );
    gpu.read_f32(&out)
}

/// `x` times the expanded weight in `f64`; `[rows, n]`.
fn oracle(x: &[f32], wide: &[f32], rows: usize, k: usize, n: usize) -> Vec<f64> {
    let mut y = vec![0f64; rows * n];
    for r in 0..rows {
        let xr = &x[r * k..(r + 1) * k];
        let yr = &mut y[r * n..(r + 1) * n];
        for (kk, &xv) in xr.iter().enumerate() {
            let wrow = &wide[kk * n..(kk + 1) * n];
            for (acc, &wv) in yr.iter_mut().zip(wrow) {
                *acc += f64::from(xv) * f64::from(wv);
            }
        }
    }
    y
}

/// RMS of `got - want` over RMS of `want`.
fn relative_rms(got: &[f32], want: &[f64]) -> f64 {
    let (mut err, mut norm) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        err += (f64::from(g) - w).powi(2);
        norm += w * w;
    }
    (err / norm.max(f64::MIN_POSITIVE)).sqrt()
}

/// The decode-shaped projection agrees with the expanded weight at `bits`, on
/// every dense shape, row count and slice count; and a row's bits do not
/// depend on how many rows share its pass.
///
/// # Panics
///
/// On any disagreement past rounding, a non-finite output, or a row that
/// changes with the pass around it.
pub fn the_decode_projection_is_the_expanded_weight(gpu: &Gpu, bits: u32) {
    for (k, n) in SHAPES {
        let w = trellis(gpu, k, n, bits);
        let wide = expanded(gpu, &w, k, n, bits);
        assert!(
            wide.iter().all(|v| v.is_finite()),
            "K = {bits} {k}x{n}: the expansion wrote a non-finite weight"
        );
        let max_rows = *ROWS.iter().max().unwrap_or(&1) as usize;
        let x = signal(max_rows * k as usize, 0xdec0 ^ k ^ n);
        let want = oracle(&x, &wide, max_rows, k as usize, n as usize);
        let mut row0: Option<Vec<f32>> = None;
        for ksplit in [1, 3] {
            for rows in ROWS {
                let used = rows as usize * k as usize;
                let got = dgemv(gpu, &w, &x[..used], (rows, k, n, bits), ksplit);
                let expect = &want[..rows as usize * n as usize];
                assert!(
                    got.iter().all(|v| v.is_finite()),
                    "K = {bits} {k}x{n} rows {rows} ksplit {ksplit}: a non-finite output"
                );
                let rel = relative_rms(&got, expect);
                assert!(
                    rel < RMS_TOLERANCE,
                    "K = {bits} {k}x{n} rows {rows} ksplit {ksplit}: relative RMS error {rel:.3e} \
                     against the expanded weight"
                );
                // Row 0 is the same row in every pass of one slice count.
                if ksplit == 1 {
                    let first = got[..n as usize].to_vec();
                    match &row0 {
                        None => row0 = Some(first),
                        Some(held) => assert!(
                            held.iter()
                                .zip(&first)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "K = {bits} {k}x{n}: row 0 changed when {rows} rows shared the pass"
                        ),
                    }
                }
            }
        }
    }
}
