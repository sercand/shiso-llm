//! The low-precision prompt kernels (`q4e_nvfp4_a4`, `q4e_mxfp8_a8`): the
//! routed NVFP4 expert stages with NVFP4 activations and the dense MXFP8
//! projections with MXFP8 activations, each on the block-scaled tensor-core
//! instruction of its format.
//!
//! Three kinds of case, for both formats:
//!
//! - **The quantizers, bit for bit** against a host restatement of the
//!   rounding ([`fp4_block`], [`quant_mxfp8`]): every code byte and scale
//!   byte, over rows built to land on the places a quantizer goes wrong -
//!   values exactly on a rounding tie, blocks that saturate, all-zero blocks
//!   and rows, scales in the subnormal range, negative values that round to
//!   zero.
//! - **The GEMMs against the host product over the quantized operands**: the
//!   device's own quantized activations are read back and multiplied on the
//!   host in `f64`, so the only slack left is the instruction's accumulation;
//!   and **against the weight-only kernels**, which is the error activation
//!   quantization introduces - measured and returned for the driver to report,
//!   with only a loose bound asserted.
//! - **Row invariance**: a row computed in a narrow pass and inside a wide one
//!   is the same bits.
//!
//! The entries exist only in a build for a family-specific architecture; a
//! device whose kernel set lacks them is reported by the driver as a skip.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::missing_panics_doc,
    clippy::many_single_char_names,
    reason = "fixture dimensions and ids are small literals; every case asserts by \
              panicking; the host products use the formulas' own one-letter names"
)]

use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::bench::Fixture;
use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;
use crate::qwen4exp::fp8::e4m3;
use crate::qwen4exp::mxfp8::{self as wonly, Matrix, PassDesc};
use crate::qwen4exp::nvfp4::{
    BENCH_SHAPE, Bank, DevRoute, Down, GateUp, PROMPT, Route, bench_bank, bench_routing, e2m1,
    i32_bytes, matrix_bytes, views, zeroed,
};

/// The W4A4 module, mirroring `kernels::MOD_NVFP4_A4`.
pub const A4_MODULE: &str = "q4e_nvfp4_a4";
/// The W8A8 module, mirroring `kernels::MOD_MXFP8_A8`.
pub const A8_MODULE: &str = "q4e_mxfp8_a8";

const A4_QUANT: &str = "q4e_nvfp4_a4_quant_kernel";
const A4_QUANT_BF16: &str = "q4e_nvfp4_a4_quant_bf16_kernel";
const A4_COMBINE: &str = "q4e_nvfp4_a4_combine_kernel";
const A4_GATE_UP: &str = "q4e_nvfp4_a4_gate_up_kernel";
const A4_DOWN: &str = "q4e_nvfp4_a4_down_kernel";
const A8_QUANT: &str = "q4e_mxfp8_a8_quant_kernel";
const A8_GEMM: &str = "q4e_mxfp8_a8_gemm_kernel";

/// Threads of the W4A4 GEMM (`LP_A4_THREADS`).
const A4_THREADS: u32 = 256;
/// Blocks one row tile of the W4A4 GEMM is split over (`LP_A4_SPLIT`).
const A4_SPLIT: u32 = 4;
/// Columns one W4A4 tile covers (`LP_A4_BN`).
pub const A4_BN: u32 = 128;
/// Dynamic shared memory of the W4A4 GEMM (`LP_A4_SMEM`): two stages of
/// 128 weight rows and [`A4_BN`] columns at 256 contraction elements, and
/// three words a column.
pub const A4_SMEM: u32 = 2 * (128 * 128 + 128 * 16 + A4_BN * 128 + A4_BN * 16) + A4_BN * 12;
/// Threads of the W8A8 GEMM (`LP_A8_THREADS`).
const A8_THREADS: u32 = 256;
/// Rows and outputs of one W8A8 tile (`LP_A8_BM`, `LP_A8_BN`).
const A8_TILE: u32 = 128;
/// Dynamic shared memory of the W8A8 GEMM (`LP_A8_SMEM`): four stages of two
/// 128 x 64 code tiles and two chunks of 16 scale bytes for 256 rows.
pub const A8_SMEM: u32 = 4 * 2 * 128 * 64 + 2 * 256 * 16;

/// Whether this device's kernel set carries the low-precision entries, which
/// only a build for a family-specific architecture does.
#[must_use]
pub fn available(gpu: &Gpu) -> bool {
    [
        (A4_MODULE, A4_QUANT),
        (A4_MODULE, A4_QUANT_BF16),
        (A4_MODULE, A4_COMBINE),
        (A4_MODULE, A4_GATE_UP),
        (A4_MODULE, A4_DOWN),
        (A8_MODULE, A8_QUANT),
        (A8_MODULE, A8_GEMM),
    ]
    .iter()
    .all(|&(m, e)| {
        gpu.device()
            .max_block_threads(KernelRef::new(m, e))
            .is_some()
    })
}

// --- host restatements -------------------------------------------------------

/// `f32` to `e4m3`, round to nearest even, saturating at 448; NaN is `0x7F`.
/// The sign of a zero is kept.
#[must_use]
pub fn to_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a >= 448.0 {
        return sign | 0x7E;
    }
    if a < 2f32.powi(-6) {
        // Subnormal: steps of 2^-9; a code of 8 is the smallest normal.
        let m = (f64::from(a) * 512.0).round_ties_even() as u8;
        return sign | m;
    }
    let bits = a.to_bits();
    let mut e = ((bits >> 23) & 0xFF) as i32 - 127;
    let frac = bits & 0x7F_FFFF;
    let mut q = frac >> 20;
    let rem = frac & 0xF_FFFF;
    if rem > 0x8_0000 || (rem == 0x8_0000 && q & 1 == 1) {
        q += 1;
    }
    if q == 8 {
        q = 0;
        e += 1;
    }
    sign | (((e + 7) as u8) << 3) | q as u8
}

/// One block of sixteen activations as NVFP4 under the global scale `gs`:
/// the packed codes (element `i` in nibble `i`) and the `e4m3` scale byte.
///
/// `sf = e4m3_rn(min(amax * gs / 6, 448))`; a value's code is the `e2m1`
/// value nearest `|x| / (e4m3(sf) / gs)`, found against seven thresholds
/// scaled by that step - each a single rounded product, exactly as the device
/// forms them - ties to even, the sign set only on a nonzero code.
#[must_use]
pub fn fp4_block(v: &[f32], gs: f32) -> (u64, u8) {
    let amax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    let sf = (amax * gs / 6.0).min(448.0);
    let sbyte = to_e4m3(sf);
    let qs = e4m3(sbyte);
    let mut packed = 0u64;
    if qs != 0.0 && gs != 0.0 {
        let vs = qs / gs;
        let [t025, t075, t125, t175, t250, t350, t500] =
            [0.25f32, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0].map(|c| vs * c);
        for (i, &q) in v.iter().enumerate() {
            let mag = q.abs();
            let mut n: u64 = if mag > t025 && mag < t075 {
                1
            } else if mag >= t075 && mag <= t125 {
                2
            } else if mag > t125 && mag < t175 {
                3
            } else if mag >= t175 && mag <= t250 {
                4
            } else if mag > t250 && mag < t350 {
                5
            } else if mag >= t350 && mag <= t500 {
                6
            } else if mag > t500 {
                7
            } else {
                0
            };
            if n != 0 && q < 0.0 {
                n |= 8;
            }
            packed |= n << (4 * i);
        }
    }
    (packed, sbyte)
}

/// How a quantizer launch picks a row's global scale.
#[derive(Debug, Clone, Copy)]
pub enum GlobalScale<'a> {
    /// `448 * 6 / amax(row)`, zero for a row of zeros.
    Dynamic,
    /// `table[idx[row]]`, or `table[0]` for every row without `idx`.
    Static {
        /// The scales.
        table: &'a [f32],
        /// Which scale each row takes.
        idx: Option<&'a [i32]>,
    },
}

/// `x [rows, k]` quantized as the device lays it out: codes `[rows][k / 2]`,
/// scales `[rows][k / 16]`, then the `f32` global of every row.
#[must_use]
pub fn quant_nvfp4(x: &[f32], rows: usize, k: usize, gs: GlobalScale<'_>) -> Vec<u8> {
    let mut codes = vec![0u8; rows * k / 2];
    let mut scales = vec![0u8; rows * k / 16];
    let mut globals = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &x[r * k..(r + 1) * k];
        let g = match gs {
            GlobalScale::Dynamic => {
                let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
                if amax > 0.0 { 2688.0 / amax } else { 0.0 }
            }
            GlobalScale::Static { table, idx } => table[idx.map_or(0, |i| i[r] as usize)],
        };
        for (b, block) in row.chunks_exact(16).enumerate() {
            let (packed, sbyte) = fp4_block(block, g);
            codes[r * k / 2 + b * 8..r * k / 2 + b * 8 + 8].copy_from_slice(&packed.to_le_bytes());
            scales[r * k / 16 + b] = sbyte;
        }
        globals.push(g);
    }
    codes.extend_from_slice(&scales);
    codes.extend_from_slice(&f32_bytes(&globals));
    codes
}

/// A quantized NVFP4 activation buffer, read back into its parts.
struct A4Rows<'a> {
    q: &'a [u8],
    rows: usize,
    k: usize,
}

impl A4Rows<'_> {
    /// Element `j` of row `r` before the global: `e2m1 * e4m3(scale)`.
    fn raw(&self, r: usize, j: usize) -> f64 {
        let byte = self.q[r * self.k / 2 + j / 2];
        let code = if j.is_multiple_of(2) {
            byte & 0xF
        } else {
            byte >> 4
        };
        let s = self.q[self.rows * self.k / 2 + r * self.k / 16 + j / 16];
        f64::from(e2m1(code)) * f64::from(e4m3(s))
    }

    /// Row `r`'s global scale.
    fn gs(&self, r: usize) -> f32 {
        let at = self.rows * self.k / 2 + self.rows * self.k / 16 + 4 * r;
        f32::from_le_bytes(self.q[at..at + 4].try_into().expect("four bytes"))
    }
}

/// The exact `ceil(log2(r))` of a positive finite `f32`.
fn ceil_log2(r: f32) -> i32 {
    let bits = r.to_bits();
    let ex = (bits >> 23) & 0xFF;
    let mant = bits & 0x7F_FFFF;
    if ex == 0 {
        let msb = 31 - mant.leading_zeros() as i32;
        return msb - 149 + i32::from(mant & (mant - 1) != 0);
    }
    ex as i32 - 127 + i32::from(mant != 0)
}

/// `x [rows, k]` as MXFP8, as the device lays it out: codes `[rows][k]`, then
/// the scale bytes `[rows][k / 32]`. Per row per 32:
/// `e = clamp(ceil(log2(amax / 448)), -127, 127)` (0 for zeros),
/// `code = e4m3_rn(x / 2^e)`, scale byte `e + 127`.
#[must_use]
pub fn quant_mxfp8(x: &[f32], rows: usize, k: usize) -> Vec<u8> {
    let mut codes = Vec::with_capacity(rows * k + rows * k / 32);
    let mut scales = Vec::with_capacity(rows * k / 32);
    for group in x[..rows * k].chunks_exact(32) {
        let amax = group.iter().fold(0f32, |m, v| m.max(v.abs()));
        let e = if amax > 0.0 {
            ceil_log2(amax / 448.0).clamp(-127, 127)
        } else {
            0
        };
        let inv = f32::from_bits(((127 - e) as u32) << 23);
        codes.extend(group.iter().map(|&v| to_e4m3(v * inv)));
        scales.push((e + 127) as u8);
    }
    codes.extend_from_slice(&scales);
    codes
}

/// Asserts two byte strings are equal, naming the first difference and what
/// it belongs to.
fn same_bytes(what: &str, got: &[u8], want: &[u8], name: impl Fn(usize) -> String) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(g, w)| g != w) {
        panic!(
            "{what}: byte {i} ({}) is {:#04x}, the host says {:#04x}",
            name(i),
            got[i],
            want[i]
        );
    }
}

// --- the quantizers ----------------------------------------------------------

/// Rows of 256 built to reach every branch of the NVFP4 rounding: blocks on
/// the scaled thresholds exactly, saturating blocks, all-zero blocks,
/// subnormal and vanishing scales, negative values that round to zero.
fn fp4_fixture(rows: usize, k: usize) -> Vec<f32> {
    let mut x = signal(rows * k, 0xF4F4);
    for r in 0..rows {
        let row = &mut x[r * k..(r + 1) * k];
        // Magnitudes spread over many decades, row by row.
        let mag = 2f32.powi((r as i32 % 23) - 11);
        for v in row.iter_mut() {
            *v *= mag;
        }
        // Block 1: zeros, and a negative zero.
        row[16..32].fill(0.0);
        row[17] = -0.0;
        // Block 2: an outlier, so the rest of the block rounds to small codes
        // and negative ones to zero.
        row[33] = 900.0 * mag;
        row[34] = -0.01 * mag;
        // Block 3: the thresholds of the step, exactly, when the block's
        // scale is exact: amax 6 under a global of one is a scale of one.
        let ties = [
            6.0f32, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5,
            -5.0, 0.5,
        ];
        row[48..64].copy_from_slice(&ties);
        // Block 4: vanishing values, whose scale is subnormal or zero.
        for (i, v) in row[64..80].iter_mut().enumerate() {
            *v = (i as f32 - 7.5) * 1e-7;
        }
        // Block 5: a huge block, saturating under a large static global.
        for (i, v) in row[80..96].iter_mut().enumerate() {
            *v = (i as f32 - 8.0) * 37.0;
        }
    }
    // One row of zeros.
    x[k..2 * k].fill(0.0);
    x
}

/// The element of a quantizer's input rows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rows {
    F32,
    Bf16,
}

/// Runs the NVFP4 quantizer over `x [rows, k]`; the quantized buffer.
fn device_quant_nvfp4(
    gpu: &Gpu,
    x: &DeviceBuffer,
    elem: Rows,
    rows: usize,
    k: usize,
    gs: GlobalScale<'_>,
) -> DeviceBuffer {
    let q = zeroed(gpu, (rows * k / 2 + rows * k / 16 + rows * 4) as u64);
    let tables = match gs {
        GlobalScale::Dynamic => None,
        GlobalScale::Static { table, idx } => Some((
            gpu.upload(&f32_bytes(table)),
            idx.map(|i| gpu.upload(&i32_bytes(i))),
        )),
    };
    let null = KernelArg::U64(0);
    let args = [
        KernelArg::Buffer(q.view()),
        KernelArg::Buffer(x.view()),
        tables
            .as_ref()
            .map_or(null, |(t, _)| KernelArg::Buffer(t.view())),
        tables
            .as_ref()
            .and_then(|(_, i)| i.as_ref())
            .map_or(null, |i| KernelArg::Buffer(i.view())),
        KernelArg::U32(rows as u32),
        KernelArg::U32(k as u32),
    ];
    run(
        gpu,
        KernelRef::new(
            A4_MODULE,
            if elem == Rows::F32 {
                A4_QUANT
            } else {
                A4_QUANT_BF16
            },
        ),
        &LaunchArgs::new(
            Extent3::new((rows as u32).div_ceil(4), 1, 1),
            Extent3::linear(128),
            &args,
        ),
    );
    q
}

/// A float rounded to the nearest brain float, as its bit pattern.
fn bf16_bits(v: f32) -> u16 {
    half::bf16::from_f32(v).to_bits()
}

/// The NVFP4 quantizer returns the host's bytes - codes, scales and globals -
/// under a dynamic global, a single static one (one large enough to saturate
/// every block's scale) and a static one chosen per row; over `f32` rows and
/// over brain-float rows (the down stage's input); at a width the quantizer
/// holds in registers and at one past it, which it reads twice.
pub fn the_nvfp4_quantizer_matches_the_host_bit_for_bit(gpu: &Gpu) {
    for (rows, k) in [(37usize, 256usize), (9, 2_688)] {
        let x = fp4_fixture(rows, k);
        let xb: Vec<u16> = x.iter().map(|&v| bf16_bits(v)).collect();
        let xw: Vec<f32> = xb
            .iter()
            .map(|&b| half::bf16::from_bits(b).to_f32())
            .collect();
        let inputs = [
            (Rows::F32, gpu.upload(&f32_bytes(&x)), &x),
            (
                Rows::Bf16,
                gpu.upload(&xb.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<_>>()),
                &xw,
            ),
        ];
        let idx: Vec<i32> = (0..rows as i32).map(|r| r % 3).collect();
        let table = [1.0f32, 3.883e-3_f32.recip(), 1.0e4];
        let unit = [1.0f32];
        let big = [1.0e4f32];
        let scales_of = [
            ("dynamic", GlobalScale::Dynamic),
            (
                "static 1",
                GlobalScale::Static {
                    table: &unit,
                    idx: None,
                },
            ),
            (
                "static 1e4",
                GlobalScale::Static {
                    table: &big,
                    idx: None,
                },
            ),
            (
                "static per row",
                GlobalScale::Static {
                    table: &table,
                    idx: Some(&idx),
                },
            ),
        ];
        for ((what, gs), (elem, xs, host)) in scales_of
            .iter()
            .flat_map(|g| inputs.iter().map(move |i| (g, i)))
        {
            let q = device_quant_nvfp4(gpu, xs, *elem, rows, k, *gs);
            drain(gpu);
            let got = gpu.read(&q);
            let want = quant_nvfp4(host, rows, k, *gs);
            let codes = rows * k / 2;
            let scales = rows * k / 16;
            let input = if *elem == Rows::F32 { "f32" } else { "bf16" };
            same_bytes(
                &format!("nvfp4 quantizer, {what}, {input} rows of {k}"),
                &got,
                &want,
                |i| {
                    if i < codes {
                        format!("row {} code byte {}", i / (k / 2), i % (k / 2))
                    } else if i < codes + scales {
                        let j = i - codes;
                        format!("row {} scale {}", j / (k / 16), j % (k / 16))
                    } else {
                        format!("global of row {}", (i - codes - scales) / 4)
                    }
                },
            );
        }
    }
}

/// Groups of 32 built to reach every branch of the MXFP8 rounding: amax an
/// exact power of two times 448 and one ulp past it, values on `e4m3` ties,
/// values in the `e4m3` subnormal range, all-zero groups, a negative zero,
/// and magnitudes from 1e-30 to 1e30.
fn fp8_fixture(rows: usize, k: usize) -> Vec<f32> {
    let mut x = signal(rows * k, 0xF8F8);
    for (g, group) in x.chunks_exact_mut(32).enumerate() {
        let mag = 10f32.powi((g as i32 % 61) - 30);
        for v in group.iter_mut() {
            *v *= mag;
        }
        match g % 7 {
            0 => group.fill(0.0),
            1 => {
                // amax / 448 exactly 8: e = 3, and the ties of [1, 2) at 2^3.
                group[0] = 448.0 * 8.0;
                for (i, v) in group[1..].iter_mut().enumerate() {
                    *v = 8.0 * (1.0 + (2 * (i % 8) + 1) as f32 / 16.0);
                }
                group[31] = -0.0;
            }
            2 => {
                // One ulp past it: e = 4.
                group[0] = f32::from_bits((448.0f32 * 8.0).to_bits() + 1);
            }
            3 => {
                // Codes in the subnormal range of `e4m3`, ties included.
                group[0] = 448.0;
                for (i, v) in group[1..].iter_mut().enumerate() {
                    *v = (i as f32 + 0.5) * 2f32.powi(-9);
                }
            }
            _ => {}
        }
    }
    x
}

/// Runs the MXFP8 quantizer over `x [rows, k]`.
fn device_quant_mxfp8(gpu: &Gpu, x: &DeviceBuffer, rows: usize, k: usize) -> DeviceBuffer {
    let q = zeroed(gpu, (rows * k + rows * k / 32) as u64);
    let quads = (rows * k / 4) as u32;
    run(
        gpu,
        KernelRef::new(A8_MODULE, A8_QUANT),
        &LaunchArgs::new(
            Extent3::new(quads.div_ceil(256), 1, 1),
            Extent3::linear(256),
            &[
                KernelArg::Buffer(q.view()),
                KernelArg::Buffer(x.view()),
                KernelArg::U32(rows as u32),
                KernelArg::U32(k as u32),
            ],
        ),
    );
    q
}

/// The MXFP8 quantizer returns the host's codes and scale bytes.
pub fn the_mxfp8_quantizer_matches_the_host_bit_for_bit(gpu: &Gpu) {
    let (rows, k) = (29usize, 512usize);
    let x = fp8_fixture(rows, k);
    let xs = gpu.upload(&f32_bytes(&x));
    let q = device_quant_mxfp8(gpu, &xs, rows, k);
    drain(gpu);
    let got = gpu.read(&q);
    let want = quant_mxfp8(&x, rows, k);
    same_bytes("mxfp8 quantizer", &got, &want, |i| {
        if i < rows * k {
            format!("row {} code {} (value {:e})", i / k, i % k, x[i])
        } else {
            let j = i - rows * k;
            format!("row {} scale {}", j / (k / 32), j % (k / 32))
        }
    });
}

// --- the W4A4 expert stages --------------------------------------------------

/// One routing of a W4A4 run.
struct A4Case<'a> {
    ids: &'a [i32],
    map: &'a [u32],
    x: &'a [f32],
    hidden: usize,
    inter: usize,
    n_expert: usize,
    n_used: usize,
}

/// What a W4A4 run produced: the quantized token rows, `silu(g) * u` per
/// column, the quantized columns, and the down rows.
struct A4Out {
    xq: Vec<u8>,
    mid: Vec<f32>,
    midq: Vec<u8>,
    down: Vec<f32>,
}

/// One W4A4 gate-and-up launch.
fn a4_gate_up(
    gpu: &Gpu,
    mid: &DeviceBuffer,
    banks: [&[DeviceBuffer; 3]; 2],
    xq: &DeviceBuffer,
    route: &DevRoute,
    dims: (u32, u32, u32, u32, u32),
) {
    let (hidden, inter, n_expert, n_used, rows) = dims;
    let [gate, up] = banks;
    let args = [
        KernelArg::Buffer(mid.view()),
        KernelArg::Buffer(gate[0].view()),
        KernelArg::Buffer(gate[1].view()),
        KernelArg::Buffer(gate[2].view()),
        KernelArg::Buffer(up[0].view()),
        KernelArg::Buffer(up[1].view()),
        KernelArg::Buffer(up[2].view()),
        KernelArg::Buffer(xq.view()),
        KernelArg::Buffer(route.sorted.view()),
        KernelArg::Buffer(route.bounds.view()),
        KernelArg::Buffer(route.active.view()),
        KernelArg::Buffer(route.map.view()),
        KernelArg::U32(hidden),
        KernelArg::U32(inter),
        KernelArg::U32(n_expert),
        KernelArg::U32(n_used),
        KernelArg::U32(rows),
    ];
    let mut launch = LaunchArgs::new(
        Extent3::new(
            A4_SPLIT * inter.div_ceil(64),
            (route.columns as u32).min(n_expert),
            1,
        ),
        Extent3::linear(A4_THREADS),
        &args,
    );
    launch.smem_bytes = A4_SMEM;
    run(gpu, KernelRef::new(A4_MODULE, A4_GATE_UP), &launch);
}

/// One W4A4 down launch.
fn a4_down(
    gpu: &Gpu,
    out: &DeviceBuffer,
    bank: &[DeviceBuffer; 3],
    midq: &DeviceBuffer,
    route: &DevRoute,
    dims: (u32, u32, u32, u32, u32),
) {
    let (hidden, inter, n_expert, n_used, rows) = dims;
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(bank[0].view()),
        KernelArg::Buffer(bank[1].view()),
        KernelArg::Buffer(bank[2].view()),
        KernelArg::Buffer(midq.view()),
        KernelArg::Buffer(route.sorted.view()),
        KernelArg::Buffer(route.bounds.view()),
        KernelArg::Buffer(route.active.view()),
        KernelArg::Buffer(route.map.view()),
        KernelArg::U32(inter),
        KernelArg::U32(hidden),
        KernelArg::U32(n_expert),
        KernelArg::U32(n_used),
        KernelArg::U32(rows),
    ];
    let mut launch = LaunchArgs::new(
        Extent3::new(
            A4_SPLIT * hidden.div_ceil(128),
            (route.columns as u32).min(n_expert),
            1,
        ),
        Extent3::linear(A4_THREADS),
        &args,
    );
    launch.smem_bytes = A4_SMEM;
    run(gpu, KernelRef::new(A4_MODULE, A4_DOWN), &launch);
}

/// Quantize, gate and up, quantize, down: the W4A4 routed stage as the
/// engine plans it, with dynamic globals.
fn a4_run(gpu: &Gpu, banks: &[[DeviceBuffer; 3]; 3], case: &A4Case<'_>) -> A4Out {
    let (hidden, inter) = (case.hidden, case.inter);
    let cols = case.ids.len();
    let n_tok = case.x.len() / hidden;
    let route = Route::new(case.ids, case.n_expert).upload(gpu, case.map, cols);
    let xs = gpu.upload(&f32_bytes(case.x));
    let xq = device_quant_nvfp4(gpu, &xs, Rows::F32, n_tok, hidden, GlobalScale::Dynamic);
    // Brain-float NaNs, so a row the kernel never writes fails.
    let mid = gpu.upload(&vec![0xFFu8; cols * inter * 2]);
    a4_gate_up(
        gpu,
        &mid,
        [&banks[0], &banks[1]],
        &xq,
        &route,
        (
            hidden as u32,
            inter as u32,
            case.n_expert as u32,
            case.n_used as u32,
            n_tok as u32,
        ),
    );
    let midq = device_quant_nvfp4(gpu, &mid, Rows::Bf16, cols, inter, GlobalScale::Dynamic);
    let down = gpu.upload(&vec![0xFFu8; cols * hidden * 2]);
    a4_down(
        gpu,
        &down,
        &banks[2],
        &midq,
        &route,
        (
            hidden as u32,
            inter as u32,
            case.n_expert as u32,
            case.n_used as u32,
            cols as u32,
        ),
    );
    drain(gpu);
    A4Out {
        xq: gpu.read(&xq),
        mid: read_bf16(gpu, &mid),
        midq: gpu.read(&midq),
        down: read_bf16(gpu, &down),
    }
}

/// A buffer of brain floats, widened.
fn read_bf16(gpu: &Gpu, buf: &DeviceBuffer) -> Vec<f32> {
    gpu.read(buf)
        .chunks_exact(2)
        .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect()
}

/// A brain float's rounding: half an ulp, 2^-9 of the value, and a margin.
const BF16_HALF_ULP: f64 = 1.0 / 256.0;

/// Weight `(o, k)` of `bank`'s slot `slot` before its global:
/// `e2m1 * e4m3(scale)`.
fn raw_w(bank: &Bank, slot: usize, o: usize, k: usize) -> f64 {
    let row = slot * bank.out_dim + o;
    let byte = bank.codes[(row * bank.in_dim + k) / 2];
    let code = if k.is_multiple_of(2) {
        byte & 0xF
    } else {
        byte >> 4
    };
    f64::from(e2m1(code)) * f64::from(e4m3(bank.scales[(row * bank.in_dim + k) / 16]))
}

/// `(sum, sum |.|)` of row `o` of `bank`'s slot against quantized row `r`,
/// before either global.
fn raw_dot(bank: &Bank, slot: usize, o: usize, xq: &A4Rows<'_>, r: usize) -> (f64, f64) {
    let (mut sum, mut abs) = (0.0f64, 0.0f64);
    for k in 0..bank.in_dim {
        let p = raw_w(bank, slot, o, k) * xq.raw(r, k);
        sum += p;
        abs += p.abs();
    }
    (sum, abs)
}

/// The instruction's sums against the host product of the quantized operands,
/// as a fraction of the product's absolute sum: only the accumulation is
/// left. The worst measured over the cases below on a GB10 is what this bound
/// is set from.
pub const A4_TOLERANCE: f64 = 1e-5;

/// The scale an epilogue multiplies a sum by: `global / gs`, rounded once,
/// or zero for a row of zeros.
fn alpha(global: f32, gs: f32) -> f64 {
    if gs == 0.0 {
        0.0
    } else {
        f64::from(global / gs)
    }
}

/// What a W4A4 case measured: the worst accumulation error against the host
/// product of the quantized operands, and the error activation quantization
/// adds against the weight-only stages (`||a4 - a16|| / ||a16||` of the
/// `silu(g) * u` rows and of the down rows).
#[derive(Debug, Clone, Copy)]
pub struct A4Report {
    /// Worst accumulation error, as a fraction of the absolute sum.
    pub accumulation: f64,
    /// Relative error of the intermediate rows against the weight-only stage.
    pub mid_vs_a16: f64,
    /// Relative error of the down rows against the weight-only stage fed the
    /// weight-only intermediate.
    pub down_vs_a16: f64,
}

/// `||a - b|| / ||b||`.
fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        num += (f64::from(*x) - f64::from(*y)).powi(2);
        den += f64::from(*y).powi(2);
    }
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
}

/// Both W4A4 stages over a sorted column list through a slot map that is not
/// the identity, at this family's widths, against the host product of the
/// device's own quantized operands; then the same routing through the
/// weight-only prompt stages, for the error activation quantization adds.
///
/// The token rows carry a few outliers, as a residual stream does, so the
/// per-row global and the per-block scales both have something to do.
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "one routing through both stages and through the weight-only ones, \
              graded in one place, is the case"
)]
pub fn the_a4_stages_match_the_host_product_of_their_operands(gpu: &Gpu) -> A4Report {
    let (hidden, inter, n_expert, n_used, slots) = (2560usize, 640usize, 16usize, 3usize, 6usize);
    let gate = Bank::new(slots, inter, hidden, 271);
    let up = Bank::new(slots, inter, hidden, 272);
    let down = Bank::new(slots, hidden, inter, 273);
    // Six tokens: some experts shared, one expert taking more than one tile's
    // worth is the invariance case's job.
    let ids: Vec<i32> = vec![4, 9, 1, 9, 13, 4, 1, 4, 9, 13, 9, 1, 4, 13, 1, 9, 4, 13];
    let n_tok = ids.len() / n_used;
    let mut map = vec![u32::MAX; n_expert];
    for (e, slot) in [(1usize, 5u32), (4, 0), (9, 3), (13, 2)] {
        map[e] = slot;
    }
    let mut x = signal(n_tok * hidden, 281);
    for t in 0..n_tok {
        x[t * hidden + 17 * (t + 1)] = 40.0;
        x[t * hidden + 1001 + t] = -25.0;
    }
    let banks = [gate.upload(gpu), up.upload(gpu), down.upload(gpu)];
    let got = a4_run(
        gpu,
        &banks,
        &A4Case {
            ids: &ids,
            map: &map,
            x: &x,
            hidden,
            inter,
            n_expert,
            n_used,
        },
    );
    let cols = ids.len();
    let xq = A4Rows {
        q: &got.xq,
        rows: n_tok,
        k: hidden,
    };
    let midq = A4Rows {
        q: &got.midq,
        rows: cols,
        k: inter,
    };
    let mut worst = 0.0f64;
    for (col, &e) in ids.iter().enumerate() {
        let slot = map[e as usize] as usize;
        let t = col / n_used;
        let (ag, au) = (
            alpha(gate.globals[slot], xq.gs(t)),
            alpha(up.globals[slot], xq.gs(t)),
        );
        for o in 0..inter {
            let (g, g_abs) = raw_dot(&gate, slot, o, &xq, t);
            let (u, u_abs) = raw_dot(&up, slot, o, &xq, t);
            let (g, u) = (g * ag, u * au);
            let (g_abs, u_abs) = (g_abs * ag, u_abs * au);
            let silu = g / (1.0 + (-g).exp());
            let want = silu * u;
            // The device's g and u are each off by at most the tolerance of
            // their absolute sums; what that does to `silu(g) * u`:
            // |d silu / dg| < 1.1.
            let budget = A4_TOLERANCE * (1.1 * g_abs * u.abs() + silu.abs() * u_abs)
                + BF16_HALF_ULP * want.abs()
                + 1e-30;
            let have = f64::from(got.mid[col * inter + o]);
            let err = (have - want).abs();
            let scale = 1.1 * g_abs * u.abs() + silu.abs() * u_abs;
            worst = worst
                .max((err - BF16_HALF_ULP * want.abs()).max(0.0) / scale.max(f64::MIN_POSITIVE));
            assert!(
                err <= budget,
                "a4 gate/up: column {col} row {o}: {have:e} against {want:e} (budget {budget:e})"
            );
        }
        let ad = alpha(down.globals[slot], midq.gs(col));
        for o in 0..hidden {
            let (d, d_abs) = raw_dot(&down, slot, o, &midq, col);
            let want = d * ad;
            let have = f64::from(got.down[col * hidden + o]);
            // The brain-float rounding of the output first, then what is left.
            let err = ((have - want).abs() - BF16_HALF_ULP * want.abs()).max(0.0)
                / (d_abs * ad).max(f64::MIN_POSITIVE);
            worst = worst.max(err);
            assert!(
                err <= A4_TOLERANCE,
                "a4 down: column {col} row {o}: {have:e} against {want:e}, {err:e} of the \
                 absolute sum"
            );
        }
    }

    // The weight-only stages over the same routing and the same rows.
    let route = Route::new(&ids, n_expert).upload(gpu, &map, cols);
    let xs = gpu.upload(&f32_bytes(&x));
    let og = zeroed(gpu, (cols * inter * 4) as u64);
    let ou = zeroed(gpu, (cols * inter * 4) as u64);
    GateUp {
        out_gate: og.view(),
        out_up: ou.view(),
        gate: views(&banks[0]),
        up: views(&banks[1]),
        x: xs.view(),
        route: &route,
        hidden: hidden as u32,
        inter: inter as u32,
        n_expert: n_expert as u32,
        n_used: n_used as u32,
    }
    .launch(gpu, &PROMPT);
    drain(gpu);
    let (g16, u16) = (gpu.read_f32(&og), gpu.read_f32(&ou));
    let mid16: Vec<f32> = g16
        .iter()
        .zip(&u16)
        .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    let mid_b = gpu.upload(&f32_bytes(&mid16));
    let od = zeroed(gpu, (cols * hidden * 4) as u64);
    Down {
        out: od.view(),
        weights: views(&banks[2]),
        mid: mid_b.view(),
        route: &route,
        hidden: hidden as u32,
        inter: inter as u32,
        n_expert: n_expert as u32,
        n_used: n_used as u32,
    }
    .launch(gpu, &PROMPT);
    drain(gpu);
    let down16 = gpu.read_f32(&od);
    let report = A4Report {
        accumulation: worst,
        mid_vs_a16: rel_l2(&got.mid, &mid16),
        down_vs_a16: rel_l2(&got.down, &down16),
    };
    // Four-bit activations cost about a tenth of a row on data like this:
    // the fixture's activations are uniform, the worst case for a format
    // whose steps widen toward the top (the RMS rounding error of a uniform
    // value under `e2m1` is 11% of its RMS), and the down stage carries the
    // intermediate's error too. GB10: 12.7% and 15.2%. A quarter would be a
    // broken scale, a misread block or a neighbour's global.
    assert!(
        report.mid_vs_a16 < 0.25 && report.down_vs_a16 < 0.25,
        "activation quantization error out of range: {report:?}"
    );
    report
}

/// A W4A4 column equals itself computed in a narrower chunk, bit for bit:
/// the first token's columns alone, and as the first of `n_tok` tokens whose
/// other columns land on the same experts and on others - one tile, several,
/// and several over the split.
pub fn the_a4_columns_do_not_depend_on_the_chunk(gpu: &Gpu) {
    let (hidden, inter, n_expert, n_used, slots) = (2560usize, 640usize, 32usize, 4usize, 8usize);
    let banks = [
        Bank::new(slots, inter, hidden, 371).upload(gpu),
        Bank::new(slots, inter, hidden, 372).upload(gpu),
        Bank::new(slots, hidden, inter, 373).upload(gpu),
    ];
    let map: Vec<u32> = (0..n_expert as u32)
        .map(|e| (e * 5 + 3) % slots as u32)
        .collect();
    for n_tok in [3usize, 40, 300] {
        let mut ids = vec![3i32, 7, 11, 30];
        for t in 1..n_tok as i32 {
            ids.extend([(t * 13) % 32, 7, (t * 7 + 1) % 32, 3]);
        }
        for t in 1..n_tok {
            let row = &mut ids[t * n_used..(t + 1) * n_used];
            for i in 0..n_used {
                while row[..i].contains(&row[i]) {
                    row[i] = (row[i] + 1) % 32;
                }
            }
        }
        let x = signal(n_tok * hidden, 381);
        let case = |ids: &'_ [i32], x: &'_ [f32]| -> A4Out {
            a4_run(
                gpu,
                &banks,
                &A4Case {
                    ids,
                    map: &map,
                    x,
                    hidden,
                    inter,
                    n_expert,
                    n_used,
                },
            )
        };
        let wide = case(&ids, &x);
        let alone = case(&ids[..n_used], &x[..hidden]);
        for (what, w, a) in [
            ("silu(g) * u", &wide.mid[..n_used * inter], &alone.mid[..]),
            ("down", &wide.down[..n_used * hidden], &alone.down[..]),
        ] {
            for (i, (p, q)) in w.iter().zip(a).enumerate() {
                assert_eq!(
                    p.to_bits(),
                    q.to_bits(),
                    "a4 {what} {i}: {p} in a pass of {n_tok} tokens vs {q} alone"
                );
            }
        }
    }
}

// --- the W8A8 projections ----------------------------------------------------

/// Quantizes `x [rows, in]` and runs the W8A8 GEMM; the quantized
/// activations and `[rows, out]`.
fn a8(gpu: &Gpu, w: &wonly::Device, m: &Matrix, x: &[f32], rows: u32) -> (Vec<u8>, Vec<f32>) {
    let xs = gpu.upload(&f32_bytes(x));
    let xq = device_quant_mxfp8(gpu, &xs, rows as usize, m.in_dim as usize);
    let out = wonly::sentinel(gpu, rows as usize * m.out_dim as usize);
    a8_launch(
        gpu,
        &out,
        w,
        &xq,
        PassDesc {
            in_dim: m.in_dim,
            out_dim: m.out_dim,
            rows,
        },
    );
    drain(gpu);
    (gpu.read(&xq), gpu.read_f32(&out))
}

fn a8_launch(gpu: &Gpu, out: &DeviceBuffer, w: &wonly::Device, xq: &DeviceBuffer, pass: PassDesc) {
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(w.codes.view()),
        KernelArg::Buffer(w.scales.view()),
        KernelArg::Buffer(xq.view()),
        KernelArg::U32(pass.in_dim),
        KernelArg::U32(pass.out_dim),
        KernelArg::U32(pass.rows),
    ];
    let mut launch = LaunchArgs::new(
        Extent3::new(
            pass.out_dim.div_ceil(A8_TILE),
            pass.rows.div_ceil(A8_TILE),
            1,
        ),
        Extent3::linear(A8_THREADS),
        &args,
    );
    launch.smem_bytes = A8_SMEM;
    run(gpu, KernelRef::new(A8_MODULE, A8_GEMM), &launch);
}

/// The W8A8 GEMM's budget against the host product of the quantized
/// operands, as a fraction of the row's absolute sum.
pub const A8_TOLERANCE: f64 = 1e-5;

/// What the W8A8 case measured: the worst accumulation error, and the error
/// activation quantization adds against the weight-only prompt kernel.
#[derive(Debug, Clone, Copy)]
pub struct A8Report {
    /// Worst accumulation error, as a fraction of the absolute sum.
    pub accumulation: f64,
    /// `||a8 - a16|| / ||a16||` over every shape's checked outputs.
    pub vs_a16: f64,
}

/// The W8A8 GEMM at the family's projection shapes whose contraction is a
/// multiple of 512, against the host product of the device's own quantized
/// activations; bit for bit against itself across prompt widths; and against
/// the weight-only prompt kernel, for the error activation quantization adds.
#[must_use]
pub fn the_a8_gemm_matches_the_host_and_holds_its_rows(gpu: &Gpu) -> A8Report {
    const WIDEST: u32 = 333;
    let mut worst = 0.0f64;
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (i, &(out_dim, in_dim)) in wonly::SHAPES.iter().enumerate() {
        if in_dim % 512 != 0 {
            continue;
        }
        let m = Matrix::new(out_dim, in_dim, 0x7E44 + i as u64);
        let dev = wonly::Device::upload(gpu, &m);
        let width = in_dim as usize;
        let od = out_dim as usize;
        let mut x = signal(WIDEST as usize * width, 0xA8A8 ^ in_dim ^ out_dim);
        for t in 0..WIDEST as usize {
            x[t * width + (t * 37) % width] = 30.0;
        }
        let (xq, full) = a8(gpu, &dev, &m, &x, WIDEST);
        for rows in [1u32, 7, 64, 128, 129] {
            let (_, got) = a8(gpu, &dev, &m, &x[..rows as usize * width], rows);
            wonly::same_bits(
                &format!("a8 [{out_dim} x {in_dim}] width {rows} against 333"),
                &got,
                &full[..rows as usize * od],
            );
        }
        let a16 = wonly::mxfp8(gpu, wonly::GEMM, &dev, &m, &x, WIDEST);
        let stride = if out_dim > 1_024 { 7 } else { 1 };
        let n = WIDEST as usize * width;
        for t in (0..WIDEST as usize).step_by(5) {
            for o in (0..od).step_by(stride) {
                let wr = &m.values[o * width..(o + 1) * width];
                let (mut sum, mut abs) = (0.0f64, 0.0f64);
                for (k, w) in wr.iter().enumerate() {
                    let code = xq[t * width + k];
                    let e = i32::from(xq[n + t * width / 32 + k / 32]) - 127;
                    let p = f64::from(*w) * f64::from(e4m3(code)) * 2f64.powi(e);
                    sum += p;
                    abs += p.abs();
                }
                let got = f64::from(full[t * od + o]);
                let err = (got - sum).abs() / abs.max(f64::MIN_POSITIVE);
                worst = worst.max(err);
                assert!(
                    err <= A8_TOLERANCE,
                    "a8 [{out_dim} x {in_dim}] row {t} output {o}: {got:e} against {sum:e}, \
                     {err:e} of the absolute sum"
                );
                let b = f64::from(a16[t * od + o]);
                num += (got - b).powi(2);
                den += b.powi(2);
            }
        }
    }
    let report = A8Report {
        accumulation: worst,
        vs_a16: (num / den.max(f64::MIN_POSITIVE)).sqrt(),
    };
    assert!(
        report.vs_a16 < 0.05,
        "activation quantization error out of range: {report:?}"
    );
    report
}

// --- bench -------------------------------------------------------------------

/// Which W4A4 kernel a bench fixture times.
#[derive(Clone, Copy, PartialEq, Eq)]
enum A4Stage {
    QuantX,
    GateUp,
    QuantMid,
    Down,
    Combine,
}

/// One W4A4 kernel over `n_tok` tokens of the bench routing. Its inputs are
/// quantized once at build time, so the fixture launches exactly the one
/// kernel. Bytes and flops are counted as the weight-only fixtures count
/// them, with the quantized activations' and the brain-float rows' own bytes.
#[expect(
    clippy::too_many_lines,
    reason = "one arm per kernel, each a transcription of its buffers"
)]
fn bench_a4(gpu: &Gpu, stage: A4Stage, n_tok: u32) -> Fixture {
    let (hidden, inter, n_expert, n_used) = BENCH_SHAPE;
    let cols = n_tok * n_used;
    let map: Vec<u32> = (0..n_expert).collect();
    let route = Route::new(&bench_routing(n_tok, 1), n_expert as usize);
    let distinct = route.distinct() as u64;
    let route = route.upload(gpu, &map, cols as usize);
    let q_bytes = |rows: u32, k: u32| u64::from(rows) * (u64::from(k) * 9 / 16 + 4);
    let bf16_rows = |rows: u32, k: u32| u64::from(rows) * u64::from(k) * 2;
    match stage {
        A4Stage::QuantX | A4Stage::QuantMid => {
            let (rows, k, entry, elem_bytes) = if stage == A4Stage::QuantX {
                (n_tok, hidden, A4_QUANT, 4)
            } else {
                (cols, inter, A4_QUANT_BF16, 2)
            };
            let x = zeroed(gpu, u64::from(rows) * u64::from(k) * elem_bytes);
            let fill = signal((rows * k) as usize, 0x5EED);
            let bytes_in: Vec<u8> = if elem_bytes == 4 {
                f32_bytes(&fill)
            } else {
                fill.iter()
                    .flat_map(|&v| bf16_bits(v).to_le_bytes())
                    .collect()
            };
            gpu.device()
                .copy_h2d(x.view(), &bytes_in)
                .expect("host-to-device copy");
            let q = zeroed(gpu, q_bytes(rows, k));
            let bytes = u64::from(rows) * u64::from(k) * elem_bytes + q_bytes(rows, k);
            Fixture::new(bytes, 0, move |gpu| {
                run(
                    gpu,
                    KernelRef::new(A4_MODULE, entry),
                    &LaunchArgs::new(
                        Extent3::new(rows.div_ceil(4), 1, 1),
                        Extent3::linear(128),
                        &[
                            KernelArg::Buffer(q.view()),
                            KernelArg::Buffer(x.view()),
                            KernelArg::U64(0),
                            KernelArg::U64(0),
                            KernelArg::U32(rows),
                            KernelArg::U32(k),
                        ],
                    ),
                );
            })
        }
        A4Stage::GateUp => {
            let gate = bench_bank(gpu, n_expert, inter, hidden, 0x6A7E);
            let up = bench_bank(gpu, n_expert, inter, hidden, 0x0B0B);
            let x = gpu.upload(&f32_bytes(&signal((n_tok * hidden) as usize, 0x5EED)));
            let xq = device_quant_nvfp4(
                gpu,
                &x,
                Rows::F32,
                n_tok as usize,
                hidden as usize,
                GlobalScale::Dynamic,
            );
            drain(gpu);
            let mid = zeroed(gpu, bf16_rows(cols, inter));
            let bytes = distinct * 2 * matrix_bytes(inter, hidden)
                + q_bytes(n_tok, hidden)
                + bf16_rows(cols, inter);
            let flops = 2 * 2 * u64::from(cols) * u64::from(inter) * u64::from(hidden);
            Fixture::new(bytes, flops, move |gpu| {
                a4_gate_up(
                    gpu,
                    &mid,
                    [&gate, &up],
                    &xq,
                    &route,
                    (hidden, inter, n_expert, n_used, n_tok),
                );
            })
        }
        A4Stage::Down => {
            let down = bench_bank(gpu, n_expert, hidden, inter, 0xD0D0);
            let mid = gpu.upload(&f32_bytes(&signal((cols * inter) as usize, 0x3EED)));
            let midq = device_quant_nvfp4(
                gpu,
                &mid,
                Rows::F32,
                cols as usize,
                inter as usize,
                GlobalScale::Dynamic,
            );
            drain(gpu);
            let od = zeroed(gpu, bf16_rows(cols, hidden));
            let bytes = distinct * matrix_bytes(hidden, inter)
                + q_bytes(cols, inter)
                + bf16_rows(cols, hidden);
            let flops = 2 * u64::from(cols) * u64::from(inter) * u64::from(hidden);
            Fixture::new(bytes, flops, move |gpu| {
                a4_down(
                    gpu,
                    &od,
                    &down,
                    &midq,
                    &route,
                    (hidden, inter, n_expert, n_used, cols),
                );
            })
        }
        A4Stage::Combine => {
            let fill = signal((cols * hidden) as usize, 0xC0C0);
            let down = gpu.upload(
                &fill
                    .iter()
                    .flat_map(|&v| bf16_bits(v).to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            let w = gpu.upload(&f32_bytes(&signal(cols as usize, 0xC1)));
            let out = zeroed(gpu, u64::from(n_tok) * u64::from(hidden) * 4);
            let bytes = bf16_rows(cols, hidden) + u64::from(n_tok) * u64::from(hidden) * 4;
            Fixture::new(bytes, 0, move |gpu| {
                a4_combine(gpu, &out, &down, &w, hidden, n_used, n_tok);
            })
        }
    }
}

/// One W4A4 combine launch over `n_tok` tokens.
fn a4_combine(
    gpu: &Gpu,
    out: &DeviceBuffer,
    down: &DeviceBuffer,
    weights: &DeviceBuffer,
    hidden: u32,
    n_used: u32,
    n_tok: u32,
) {
    run(
        gpu,
        KernelRef::new(A4_MODULE, A4_COMBINE),
        &LaunchArgs::new(
            Extent3::new(hidden.div_ceil(512), n_tok, 1),
            Extent3::linear(256),
            &[
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(down.view()),
                KernelArg::Buffer(weights.view()),
                KernelArg::U32(hidden),
                KernelArg::U32(n_used),
            ],
        ),
    );
}

/// The W4A4 combine sums every token's brain-float rows in slot order, from
/// zero, one rounded product and one rounded add a slot - the host's `f32`
/// arithmetic, bit for bit.
pub fn the_a4_combine_sums_in_slot_order(gpu: &Gpu) {
    let (hidden, n_used, n_tok) = (2560u32, 10u32, 7u32);
    let cols = (n_tok * n_used) as usize;
    let rows: Vec<u16> = signal(cols * hidden as usize, 0xC0)
        .iter()
        .map(|&v| bf16_bits(v * 30.0))
        .collect();
    let w = signal(cols, 0xC1);
    let down = gpu.upload(
        &rows
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let wb = gpu.upload(&f32_bytes(&w));
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; (n_tok * hidden) as usize]));
    a4_combine(gpu, &out, &down, &wb, hidden, n_used, n_tok);
    drain(gpu);
    let got = gpu.read_f32(&out);
    for t in 0..n_tok as usize {
        for e in 0..hidden as usize {
            let mut acc = 0f32;
            for s in 0..n_used as usize {
                let col = t * n_used as usize + s;
                let d = half::bf16::from_bits(rows[col * hidden as usize + e]).to_f32();
                acc += d * w[col];
            }
            let have = got[t * hidden as usize + e];
            assert_eq!(
                have.to_bits(),
                acc.to_bits(),
                "a4 combine token {t} element {e}: {have} against {acc}"
            );
        }
    }
}

/// The weight-only stage's combine (`q4e_moe_combine_kernel`) over `f32`
/// routed rows, which the W4A4 combine replaces.
fn bench_combine_f32(gpu: &Gpu, n_tok: u32) -> Fixture {
    let (hidden, _, _, n_used) = BENCH_SHAPE;
    let cols = n_tok * n_used;
    let down = gpu.upload(&f32_bytes(&signal((cols * hidden) as usize, 0xC0C0)));
    let w = gpu.upload(&f32_bytes(&signal(cols as usize, 0xC1)));
    let out = zeroed(gpu, u64::from(n_tok) * u64::from(hidden) * 4);
    let bytes = u64::from(cols) * u64::from(hidden) * 4 + u64::from(n_tok) * u64::from(hidden) * 4;
    Fixture::new(bytes, 0, move |gpu| {
        run(
            gpu,
            KernelRef::new("q4e_moe", "q4e_moe_combine_kernel"),
            &LaunchArgs::new(
                Extent3::new(hidden.div_ceil(256), n_tok, 1),
                Extent3::linear(256),
                &[
                    KernelArg::Buffer(out.view()),
                    KernelArg::Buffer(down.view()),
                    KernelArg::Buffer(w.view()),
                    KernelArg::U32(hidden),
                    KernelArg::U32(n_used),
                ],
            ),
        );
    })
}

/// The weight-only stage's activation between gate-and-up and down
/// (`q4e_swiglu_kernel`), which the W4A4 gate-and-up folds into its epilogue:
/// the weight-only stage's total is its gate-and-up plus this.
fn bench_swiglu(gpu: &Gpu, n_tok: u32) -> Fixture {
    let (_, inter, _, n_used) = BENCH_SHAPE;
    let n = u64::from(n_tok * n_used) * u64::from(inter);
    let g = gpu.upload(&f32_bytes(&signal(n as usize, 0x51)));
    let u = gpu.upload(&f32_bytes(&signal(n as usize, 0x52)));
    let out = zeroed(gpu, n * 4);
    Fixture::new(3 * n * 4, 0, move |gpu| {
        run(
            gpu,
            KernelRef::new("q4e_moe", "q4e_swiglu_kernel"),
            &LaunchArgs::new(
                Extent3::new(n.div_ceil(256) as u32, 1, 1),
                Extent3::linear(256),
                &[
                    KernelArg::Buffer(out.view()),
                    KernelArg::Buffer(g.view()),
                    KernelArg::Buffer(u.view()),
                    KernelArg::U64(n),
                ],
            ),
        );
    })
}

/// The W8A8 GEMM (its activations quantized once at build time), or its
/// quantizer, at one shape.
fn bench_a8(gpu: &Gpu, out_dim: u32, in_dim: u32, rows: u32, quant: bool) -> Fixture {
    let x = gpu.upload(&f32_bytes(&signal(rows as usize * in_dim as usize, 0x0B0B)));
    let xq = device_quant_mxfp8(gpu, &x, rows as usize, in_dim as usize);
    drain(gpu);
    let q_bytes = u64::from(rows) * u64::from(in_dim) * 33 / 32;
    if quant {
        let quads = rows * in_dim / 4;
        return Fixture::new(
            u64::from(rows) * u64::from(in_dim) * 4 + q_bytes,
            0,
            move |gpu| {
                run(
                    gpu,
                    KernelRef::new(A8_MODULE, A8_QUANT),
                    &LaunchArgs::new(
                        Extent3::new(quads.div_ceil(256), 1, 1),
                        Extent3::linear(256),
                        &[
                            KernelArg::Buffer(xq.view()),
                            KernelArg::Buffer(x.view()),
                            KernelArg::U32(rows),
                            KernelArg::U32(in_dim),
                        ],
                    ),
                );
            },
        );
    }
    let m = Matrix::new(out_dim, in_dim, 0xBE7C);
    let weights: Vec<wonly::Device> = (0..2).map(|_| wonly::Device::upload(gpu, &m)).collect();
    let out = zeroed(gpu, u64::from(rows) * u64::from(out_dim) * 4);
    let bytes = (m.codes.len() + m.scales.len()) as u64
        + q_bytes
        + 4 * u64::from(rows) * u64::from(out_dim);
    let flops = 2 * u64::from(rows) * u64::from(in_dim) * u64::from(out_dim);
    let pass = PassDesc {
        in_dim,
        out_dim,
        rows,
    };
    let next = std::cell::Cell::new(0usize);
    Fixture::new(bytes, flops, move |gpu| {
        let dev = &weights[next.get() % weights.len()];
        next.set(next.get().wrapping_add(1));
        a8_launch(gpu, &out, dev, &xq, pass);
    })
}

/// `q4e_nvfp4_a4_quant_kernel` over a 2,048-token chunk's token rows.
pub(crate) fn bench_a4_quant_x_2048(gpu: &Gpu) -> Fixture {
    bench_a4(gpu, A4Stage::QuantX, 2048)
}
/// `q4e_nvfp4_a4_quant_kernel` over a 2,048-token chunk's routed columns.
pub(crate) fn bench_a4_quant_mid_2048(gpu: &Gpu) -> Fixture {
    bench_a4(gpu, A4Stage::QuantMid, 2048)
}
/// The W4A4 gate-and-up stage over a 2,048-token chunk.
pub(crate) fn bench_a4_gate_up_2048(gpu: &Gpu) -> Fixture {
    bench_a4(gpu, A4Stage::GateUp, 2048)
}
/// The W4A4 down stage over a 2,048-token chunk.
pub(crate) fn bench_a4_down_2048(gpu: &Gpu) -> Fixture {
    bench_a4(gpu, A4Stage::Down, 2048)
}
/// The W4A4 gate-and-up stage over a 4,096-token chunk.
pub(crate) fn bench_a4_gate_up_4096(gpu: &Gpu) -> Fixture {
    bench_a4(gpu, A4Stage::GateUp, 4096)
}
/// The W4A4 down stage over a 4,096-token chunk.
pub(crate) fn bench_a4_down_4096(gpu: &Gpu) -> Fixture {
    bench_a4(gpu, A4Stage::Down, 4096)
}
/// The W4A4 gate-and-up stage over an 8,192-token chunk.
pub(crate) fn bench_a4_gate_up_8192(gpu: &Gpu) -> Fixture {
    bench_a4(gpu, A4Stage::GateUp, 8192)
}
/// The W4A4 down stage over an 8,192-token chunk.
pub(crate) fn bench_a4_down_8192(gpu: &Gpu) -> Fixture {
    bench_a4(gpu, A4Stage::Down, 8192)
}
/// `q4e_nvfp4_a4_quant_kernel` over a 4,096-token chunk's routed columns.
pub(crate) fn bench_a4_quant_mid_4096(gpu: &Gpu) -> Fixture {
    bench_a4(gpu, A4Stage::QuantMid, 4096)
}
/// The W4A4 combine over a 2,048-token chunk.
pub(crate) fn bench_a4_combine_2048(gpu: &Gpu) -> Fixture {
    bench_a4(gpu, A4Stage::Combine, 2048)
}
/// The W4A4 combine over an 8,192-token chunk.
pub(crate) fn bench_a4_combine_8192(gpu: &Gpu) -> Fixture {
    bench_a4(gpu, A4Stage::Combine, 8192)
}
/// `q4e_moe_combine_kernel` over a 2,048-token chunk.
pub(crate) fn bench_combine_f32_2048(gpu: &Gpu) -> Fixture {
    bench_combine_f32(gpu, 2048)
}
/// `q4e_moe_combine_kernel` over an 8,192-token chunk.
pub(crate) fn bench_combine_f32_8192(gpu: &Gpu) -> Fixture {
    bench_combine_f32(gpu, 8192)
}
/// `q4e_nvfp4_a4_quant_bf16_kernel` over an 8,192-token chunk's routed columns.
pub(crate) fn bench_a4_quant_mid_8192(gpu: &Gpu) -> Fixture {
    bench_a4(gpu, A4Stage::QuantMid, 8192)
}
/// `q4e_nvfp4_a4_quant_kernel` over an 8,192-token chunk's token rows.
pub(crate) fn bench_a4_quant_x_8192(gpu: &Gpu) -> Fixture {
    bench_a4(gpu, A4Stage::QuantX, 8192)
}
/// `q4e_swiglu_kernel` over an 8,192-token chunk's routed columns.
pub(crate) fn bench_swiglu_8192(gpu: &Gpu) -> Fixture {
    bench_swiglu(gpu, 8192)
}
/// `q4e_swiglu_kernel` over a 2,048-token chunk's routed columns.
pub(crate) fn bench_swiglu_2048(gpu: &Gpu) -> Fixture {
    bench_swiglu(gpu, 2048)
}
/// `q4e_swiglu_kernel` over a 4,096-token chunk's routed columns.
pub(crate) fn bench_swiglu_4096(gpu: &Gpu) -> Fixture {
    bench_swiglu(gpu, 4096)
}
/// The W8A8 GEMM at the recurrent input's shape over 2,048 rows.
pub(crate) fn bench_a8_gemm2048_10240x2560(gpu: &Gpu) -> Fixture {
    bench_a8(gpu, 10_240, 2_560, 2048, false)
}
/// The W8A8 GEMM at the attention query's shape over 2,048 rows.
pub(crate) fn bench_a8_gemm2048_12288x2560(gpu: &Gpu) -> Fixture {
    bench_a8(gpu, 12_288, 2_560, 2048, false)
}
/// The W8A8 GEMM at the shared expert's shape over 2,048 rows.
pub(crate) fn bench_a8_gemm2048_6144x2560(gpu: &Gpu) -> Fixture {
    bench_a8(gpu, 6_144, 2_560, 2048, false)
}
/// The W8A8 GEMM at the recurrent output's shape over 2,048 rows.
pub(crate) fn bench_a8_gemm2048_2560x6144(gpu: &Gpu) -> Fixture {
    bench_a8(gpu, 2_560, 6_144, 2048, false)
}
/// The MXFP8 quantizer over 2,048 rows of 2,560.
pub(crate) fn bench_a8_quant_2048x2560(gpu: &Gpu) -> Fixture {
    bench_a8(gpu, 0, 2_560, 2048, true)
}
/// The MXFP8 quantizer over 2,048 rows of 6,144.
pub(crate) fn bench_a8_quant_2048x6144(gpu: &Gpu) -> Fixture {
    bench_a8(gpu, 0, 6_144, 2048, true)
}
