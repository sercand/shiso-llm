//! The sparse family's NVFP4 n-gram row decode, graded bit for bit against a
//! host restatement built from the loader's own `e2m1` and `e4m3` decoders and
//! the brain-float encoder, in the reference table's order:
//! `bf16((e2m1 * e4m3) * global)`.
//!
//! The streaming tier's host codec (`shiso_ple::codec`) is graded against the
//! same restatement over every code and scale byte in its own tests, so the
//! device, the host codec and this oracle are one function three times.
#![expect(
    clippy::cast_possible_truncation,
    reason = "fixture dimensions are small literals; every case asserts by panicking"
)]

use half::bf16;
use shiso_core::{Extent3, KernelArg, KernelRef, LaunchArgs};
use shiso_modelfile::fp8::e4m3;
use shiso_modelfile::nvfp4::e2m1;

use crate::harness::{Gpu, drain, run};

/// Weights in one row of the checkpoint's table.
const HEAD_DIM: usize = 160;
/// Bytes one gathered row occupies: the codes, then a scale byte per sixteen.
const ROW_BYTES: usize = HEAD_DIM / 2 + HEAD_DIM / 16;
/// Hash heads a position reads.
const N_HEADS: usize = 16;

/// Scale bytes at the edges of `e4m3`: zero both ways, the smallest and
/// largest subnormals, the smallest normal, one, the largest finite value
/// (`0x7E`, 448) and their negatives.
const EDGE_SCALES: [u8; 10] = [0x00, 0x80, 0x01, 0x07, 0x08, 0x38, 0x7E, 0x81, 0x87, 0xFE];

/// One weight as the reference table holds it.
fn oracle(row: &[u8], i: usize, global: f32) -> f32 {
    let byte = row[i / 2];
    let code = if i.is_multiple_of(2) {
        byte & 0x0F
    } else {
        byte >> 4
    };
    let scale = e4m3(row[HEAD_DIM / 2 + i / 16]);
    bf16::from_f32((e2m1(code) * scale) * global).to_f32()
}

/// A deterministic byte stream.
fn bytes(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect()
}

/// Rows that reach every code under every edge scale, then random rows.
fn fixture(n_rows: usize) -> Vec<u8> {
    let mut rows = Vec::with_capacity(n_rows * ROW_BYTES);
    for r in 0..n_rows {
        if r < EDGE_SCALES.len() {
            // Each group holds codes 0..16 in order; the row's ten groups run
            // through the edge scales, rotated so every scale meets every
            // group position across the edge rows.
            rows.extend((0..HEAD_DIM / 2).map(|j| {
                let even = (2 * j % 16) as u8;
                even | ((even + 1) << 4)
            }));
            rows.extend((0..HEAD_DIM / 16).map(|g| EDGE_SCALES[(g + r) % EDGE_SCALES.len()]));
        } else {
            let mut row = bytes(ROW_BYTES, 0x5EED_0000 ^ r as u32);
            // A real exporter never writes the NaN scale; it has a case of
            // its own below.
            for s in &mut row[HEAD_DIM / 2..] {
                if *s & 0x7F == 0x7F {
                    *s &= 0xFE;
                }
            }
            rows.extend(row);
        }
    }
    rows
}

/// Runs the decode over `rows` and checks every weight against the oracle.
fn check(gpu: &Gpu, rows: &[u8], global: f32) {
    let n_rows = rows.len() / ROW_BYTES;
    let rows_b = gpu.upload(rows);
    let out = gpu.upload(&vec![0u8; n_rows * HEAD_DIM * 4]);
    let global_b = gpu.upload(&global.to_le_bytes());
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(rows_b.view()),
        KernelArg::U32(HEAD_DIM as u32),
        KernelArg::U32(N_HEADS as u32),
        KernelArg::U32(ROW_BYTES as u32),
        KernelArg::U32(n_rows as u32),
        KernelArg::Buffer(global_b.view()),
    ];
    run(
        gpu,
        KernelRef::new("q4e_modelopt", "q4e_ple_dequant_nvfp4_kernel"),
        &LaunchArgs::new(
            Extent3::linear(n_rows as u32),
            Extent3::linear(HEAD_DIM as u32),
            &args,
        ),
    );
    drain(gpu);
    let got = gpu.read_f32(&out);
    for r in 0..n_rows {
        let (t, h) = (r / N_HEADS, r % N_HEADS);
        let row = &rows[r * ROW_BYTES..(r + 1) * ROW_BYTES];
        for i in 0..HEAD_DIM {
            let want = oracle(row, i, global);
            let g = got[(t * N_HEADS + h) * HEAD_DIM + i];
            if want.is_nan() {
                assert!(g.is_nan(), "row {r} weight {i}: {g}, want NaN");
            } else {
                assert_eq!(
                    g.to_bits(),
                    want.to_bits(),
                    "row {r} weight {i}: {g} vs {want}"
                );
            }
        }
    }
}

/// NVFP4 n-gram rows decode to exactly the host's weights: every code under
/// the edge scales (subnormal, zero, `0x7E`), random rows, head-slowest
/// placement over three positions, under the checkpoint's table scale and
/// under one whose products land between brain floats.
pub fn nvfp4_ple_row_decode_equals_host_decode(gpu: &Gpu) {
    let rows = fixture(3 * N_HEADS);
    check(gpu, &rows, 3.324_236e-5);
    check(gpu, &rows, 0.123_456_79);
}

/// The not-a-number scale byte decodes its group to NaN rather than to a
/// value an infinity-walking rounding carry would make of it.
pub fn an_nvfp4_ple_row_under_the_nan_scale_decodes_to_nan(gpu: &Gpu) {
    let mut rows = fixture(N_HEADS);
    rows[HEAD_DIM / 2] = 0x7F;
    rows[ROW_BYTES + HEAD_DIM / 2 + 3] = 0xFF;
    check(gpu, &rows, 3.324_236e-5);
}
