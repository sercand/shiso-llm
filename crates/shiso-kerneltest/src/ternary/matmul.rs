//! Cases for the ternary and 2-bit affine matmuls and row gather, against
//! `shiso_modelfile::ternary`'s host decoders followed by an `f64` matmul.
//!
//! The fixtures are encoded here, by code written apart from both the decoder
//! and the kernels: a random code matrix is packed into each format, so one
//! matrix of numbers reaches the device three ways. That is what lets the
//! cross-format case demand bit equality between `PQ2_0` and its 2-bit affine
//! re-encoding.

#![expect(
    clippy::many_single_char_names,
    reason = "w, s, b, x and y are the matmul's own names for its operands"
)]

use bytemuck::{Pod, Zeroable};
use half::f16;
use shiso_core::{DType, DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};
use shiso_modelfile::ternary::{decode_affine2, decode_ternary};

use super::fixtures::{Lcg, rel_err};
use crate::harness::{Gpu, drain, f32_bytes, run};

/// Mirrors `ternary_args_mv`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MvArgs {
    in_dim: u32,
    out_dim: u32,
    n_tok: u32,
    splits: u32,
}

/// Mirrors `ternary_args_rows`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RowsArgs {
    in_dim: u32,
    n_tok: u32,
}

/// Inputs per group.
const GROUP: usize = 128;

/// Token blockings the matvec is built with.
pub const TOKEN_BLOCKS: [u32; 4] = [1, 2, 4, 8];

/// One of the three storage formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// `PQ2_0` blocks.
    Pq2,
    /// `PTQ1_0` blocks.
    Ptq1,
    /// MLX 2-bit affine words with `f16` scale and bias arrays.
    Mlx2,
}

impl Format {
    /// The suffix the entry names spell.
    const fn tag(self) -> &'static str {
        match self {
            Self::Pq2 => "pq2_0",
            Self::Ptq1 => "ptq1_0",
            Self::Mlx2 => "mlx2",
        }
    }
}

/// The matvec entry for `format` at `t` tokens a launch.
#[must_use]
pub fn mv_entry(format: Format, t: u32) -> String {
    format!("kernel_ternary_mv_{}_f32_t{t}", format.tag())
}

/// The row-gather entry for `format`.
#[must_use]
pub fn rows_entry(format: Format) -> String {
    format!("kernel_ternary_get_rows_{}_f32", format.tag())
}

/// A matrix in all three encodings, and its host decode.
struct Fixture {
    in_dim: usize,
    out_dim: usize,
    pq2: Vec<u8>,
    ptq1: Vec<u8>,
    words: Vec<u8>,
    scales: Vec<u8>,
    biases: Vec<u8>,
}

impl Fixture {
    /// Random codes (0..=2, plus one code 3 per row when `allow_three`) and
    /// scales; `PTQ1_0` gets the same codes with every 3 clamped to 2.
    fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let mut rng = Lcg::new(seed);
        let groups = in_dim / GROUP * out_dim;
        let mut pq2 = Vec::with_capacity(groups * 34);
        let mut ptq1 = Vec::with_capacity(groups * 28);
        let mut words = Vec::with_capacity(groups * 32);
        let mut scales = Vec::with_capacity(groups * 2);
        let mut biases = Vec::with_capacity(groups * 2);
        for g in 0..groups {
            let mut codes = [0_u8; GROUP];
            for c in &mut codes {
                *c = u8::try_from(rng.next_u32() % 3).unwrap_or(0);
            }
            // The 2-bit formats can hold +2 * scale; a decoder that clamps it
            // to +1 must fail here rather than on the first file that uses it.
            let mut wide = codes;
            wide[g % GROUP] = 3;
            let d = f16::from_f32(0.01 + 0.05 * rng.next_f32().abs());
            pq2.extend(d.to_le_bytes());
            let packed = pack2(&wide);
            pq2.extend(&packed);
            words.extend(&packed);
            scales.extend(d.to_le_bytes());
            biases.extend((-d).to_le_bytes());
            ptq1.extend(pack_trits(&codes));
            ptq1.extend(d.to_le_bytes());
        }
        Self {
            in_dim,
            out_dim,
            pq2,
            ptq1,
            words,
            scales,
            biases,
        }
    }

    /// The matrix `format` stores, decoded on the host.
    fn decoded(&self, format: Format) -> Vec<f32> {
        match format {
            Format::Pq2 => decode_ternary(DType::Pq2_0, &self.pq2, self.in_dim, self.out_dim),
            Format::Ptq1 => decode_ternary(DType::Ptq1_0, &self.ptq1, self.in_dim, self.out_dim),
            Format::Mlx2 => decode_affine2(
                &self.words,
                &self.scales,
                &self.biases,
                self.in_dim,
                self.out_dim,
            ),
        }
        .expect("fixture decodes")
    }

    /// The device buffers `format` launches with: weight, scales, biases.
    fn upload(&self, gpu: &Gpu, format: Format) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer) {
        match format {
            Format::Pq2 => (
                gpu.upload(&self.pq2),
                gpu.upload(&[0; 4]),
                gpu.upload(&[0; 4]),
            ),
            Format::Ptq1 => (
                gpu.upload(&self.ptq1),
                gpu.upload(&[0; 4]),
                gpu.upload(&[0; 4]),
            ),
            Format::Mlx2 => (
                gpu.upload(&self.words),
                gpu.upload(&self.scales),
                gpu.upload(&self.biases),
            ),
        }
    }
}

/// Four 2-bit codes a byte, from the low bits up.
fn pack2(codes: &[u8; GROUP]) -> Vec<u8> {
    codes
        .chunks_exact(4)
        .map(|c| c[0] | (c[1] << 2) | (c[2] << 4) | (c[3] << 6))
        .collect()
}

/// Packs base-3 digits most significant first into the byte that decodes to
/// them: `ceil(value * 256 / 243)`.
fn trit_byte(digits: &[u8]) -> u8 {
    let mut value = 0_u32;
    for &d in digits {
        value = value * 3 + u32::from(d);
    }
    for _ in digits.len()..5 {
        value *= 3;
    }
    u8::try_from((value * 256).div_ceil(243)).unwrap_or(u8::MAX)
}

/// The 26 code bytes of one `PTQ1_0` block.
fn pack_trits(codes: &[u8; GROUP]) -> Vec<u8> {
    let mut out = Vec::with_capacity(26);
    for m in 0..16 {
        let d: Vec<u8> = (0..5).map(|n| codes[n * 16 + m]).collect();
        out.push(trit_byte(&d));
    }
    for m in 0..8 {
        let d: Vec<u8> = (0..5).map(|n| codes[80 + n * 8 + m]).collect();
        out.push(trit_byte(&d));
    }
    for h in 0..2 {
        let d: Vec<u8> = (0..4).map(|n| codes[120 + n * 2 + h]).collect();
        out.push(trit_byte(&d));
    }
    out
}

/// `y[t][o] = sum_k w[o][k] x[t][k]` in `f64`.
fn host_matmul(w: &[f32], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let n_tok = x.len() / in_dim;
    let mut y = vec![0.0_f32; n_tok * out_dim];
    for t in 0..n_tok {
        for o in 0..out_dim {
            let mut acc = 0.0_f64;
            for k in 0..in_dim {
                acc += f64::from(w[o * in_dim + k]) * f64::from(x[t * in_dim + k]);
            }
            #[expect(clippy::cast_possible_truncation, reason = "the oracle rounds once")]
            let v = acc as f32;
            y[t * out_dim + o] = v;
        }
    }
    y
}

/// K-splits for a shape, the rule `shiso_engine::launch::ternary_splits` uses.
fn splits(in_dim: usize, out_dim: usize) -> u32 {
    let tgs = out_dim.div_ceil(8);
    let sweeps = in_dim.div_ceil(512);
    let mut s = 1;
    while tgs * s < 256 && s * 2 <= 8 && sweeps >= s * 2 {
        s *= 2;
    }
    u32::try_from(s).unwrap_or(1)
}

/// Runs the matvec at token blocking `t` over `n_tok` rows and returns `y`.
fn launch_mv(gpu: &Gpu, fx: &Fixture, format: Format, t: u32, x: &[f32]) -> Vec<f32> {
    let n_tok = x.len() / fx.in_dim;
    let (w, s, b) = fx.upload(gpu, format);
    let xb = gpu.upload(&f32_bytes(x));
    let y = gpu.upload(&f32_bytes(&vec![0.0; n_tok * fx.out_dim]));
    let as_u32 = |v: usize| u32::try_from(v).expect("fixture dimension");
    let split = splits(fx.in_dim, fx.out_dim);
    let args = MvArgs {
        in_dim: as_u32(fx.in_dim),
        out_dim: as_u32(fx.out_dim),
        n_tok: as_u32(n_tok),
        splits: split,
    };
    let entry = mv_entry(format, t);
    run(
        gpu,
        KernelRef::new("ternary", leak(entry)),
        &LaunchArgs {
            grid: Extent3::new(as_u32(fx.out_dim.div_ceil(8)), as_u32(n_tok).div_ceil(t), 1),
            block: Extent3::linear(64 * split),
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(w.view()),
                KernelArg::Buffer(s.view()),
                KernelArg::Buffer(b.view()),
                KernelArg::Buffer(xb.view()),
                KernelArg::Buffer(y.view()),
            ],
        },
    );
    drain(gpu);
    gpu.read_f32(&y)
}

/// An entry name with the lifetime `KernelRef` asks for. Cases build a few
/// dozen names once per run, so the leak is bounded.
fn leak(name: String) -> &'static str {
    Box::leak(name.into_boxed_str())
}

/// Relative tolerance of an f32 matvec over a few thousand products.
const TOL: f32 = 2e-5;

/// Every format, every token blocking, at widths this family uses.
pub fn mv_matches_the_host_matmul(gpu: &Gpu) {
    for (seed, (in_dim, out_dim)) in [(256, 64), (5120, 48), (6144, 40), (1024, 8), (17408, 16)]
        .into_iter()
        .enumerate()
    {
        let fx = Fixture::new(in_dim, out_dim, 100 + seed as u64);
        for format in [Format::Pq2, Format::Ptq1, Format::Mlx2] {
            let w = fx.decoded(format);
            for t in TOKEN_BLOCKS {
                for n_tok in [1_usize, 3, 8] {
                    let x = Lcg::new(7 + n_tok as u64).floats(n_tok * in_dim);
                    let want = host_matmul(&w, &x, in_dim, out_dim);
                    let got = launch_mv(gpu, &fx, format, t, &x);
                    let err = rel_err(&got, &want);
                    assert!(
                        err <= TOL,
                        "{format:?} t{t} {n_tok} rows ({in_dim}x{out_dim}): {err}"
                    );
                }
            }
        }
    }
}

/// `PQ2_0` and its 2-bit affine re-encoding produce the same bits.
pub fn pq2_and_mlx2_agree_bitwise(gpu: &Gpu) {
    let fx = Fixture::new(5120, 64, 300);
    for t in TOKEN_BLOCKS {
        let x = Lcg::new(301).floats(5 * 5120);
        let a = launch_mv(gpu, &fx, Format::Pq2, t, &x);
        let b = launch_mv(gpu, &fx, Format::Mlx2, t, &x);
        let same = a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits());
        assert!(same, "t{t}: PQ2_0 and MLX 2-bit differ");
    }
}

/// The row gather decodes exactly the rows the ids name.
pub fn get_rows_matches_the_host_decode(gpu: &Gpu) {
    let fx = Fixture::new(1024, 16, 400);
    let ids: [i32; 3] = [0, 7, 3];
    for format in [Format::Pq2, Format::Ptq1, Format::Mlx2] {
        let w = fx.decoded(format);
        let (wb, s, b) = fx.upload(gpu, format);
        let id_buf = gpu.upload(bytemuck::cast_slice(&ids));
        let y = gpu.upload(&f32_bytes(&vec![0.0; ids.len() * fx.in_dim]));
        let args = RowsArgs {
            in_dim: 1024,
            n_tok: 3,
        };
        run(
            gpu,
            KernelRef::new("ternary", leak(rows_entry(format))),
            &LaunchArgs {
                grid: Extent3::new(1024 / 128, 3, 1),
                block: Extent3::linear(128),
                smem_bytes: 0,
                args: &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(wb.view()),
                    KernelArg::Buffer(s.view()),
                    KernelArg::Buffer(b.view()),
                    KernelArg::Buffer(id_buf.view()),
                    KernelArg::Buffer(y.view()),
                ],
            },
        );
        drain(gpu);
        let got = gpu.read_f32(&y);
        let want: Vec<f32> = ids
            .iter()
            .flat_map(|&id| {
                let r = usize::try_from(id).expect("id");
                w[r * 1024..(r + 1) * 1024].to_vec()
            })
            .collect();
        let same = got
            .iter()
            .zip(&want)
            .all(|(g, w)| g.to_bits() == w.to_bits());
        assert!(same, "{format:?} rows differ from the host decode");
    }
}

/// The staged-tile matmul entry for `format` at `tn` tokens a threadgroup.
#[must_use]
pub fn mm_entry(format: Format, tn: u32) -> String {
    format!("kernel_ternary_mm_{}_f32_n{tn}", format.tag())
}

/// The register-fragment matmul entry for `format` (two to eight rows).
#[must_use]
pub fn mmv_entry(format: Format) -> String {
    format!("kernel_ternary_mmv_{}_f32", format.tag())
}

/// Runs one of the two matmul shapes over all of `x`'s rows.
fn launch_mm(gpu: &Gpu, fx: &Fixture, format: Format, tile: Option<u32>, x: &[f32]) -> Vec<f32> {
    let n_tok = x.len() / fx.in_dim;
    let (w, s, b) = fx.upload(gpu, format);
    let xb = gpu.upload(&f32_bytes(x));
    let y = gpu.upload(&f32_bytes(&vec![0.0; n_tok * fx.out_dim]));
    let as_u32 = |v: usize| u32::try_from(v).expect("fixture dimension");
    let args = MvArgs {
        in_dim: as_u32(fx.in_dim),
        out_dim: as_u32(fx.out_dim),
        n_tok: as_u32(n_tok),
        splits: 1,
    };
    let (entry, tokens) = match tile {
        Some(tn) => (mm_entry(format, tn), as_u32(n_tok).div_ceil(tn)),
        None => (mmv_entry(format), 1),
    };
    run(
        gpu,
        KernelRef::new("ternary", leak(entry)),
        &LaunchArgs {
            grid: Extent3::new(as_u32(fx.out_dim.div_ceil(32)), tokens, 1),
            block: Extent3::linear(128),
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(w.view()),
                KernelArg::Buffer(s.view()),
                KernelArg::Buffer(b.view()),
                KernelArg::Buffer(xb.view()),
                KernelArg::Buffer(y.view()),
            ],
        },
    );
    drain(gpu);
    gpu.read_f32(&y)
}

/// The two matmul shapes, every format, against the host product: the
/// register shape at two to eight rows, the staged one at both tile widths
/// over passes that leave a partial tile.
pub fn mm_matches_the_host_matmul(gpu: &Gpu) {
    for (seed, (in_dim, out_dim)) in [(256, 64), (5120, 40), (1024, 96)].into_iter().enumerate() {
        let fx = Fixture::new(in_dim, out_dim, 500 + seed as u64);
        for format in [Format::Pq2, Format::Ptq1, Format::Mlx2] {
            let w = fx.decoded(format);
            for (tile, n_tok) in [
                (None, 2_usize),
                (None, 5),
                (None, 8),
                (Some(8), 3),
                (Some(8), 8),
                (Some(32), 9),
                (Some(32), 40),
            ] {
                let x = Lcg::new(9 + n_tok as u64).floats(n_tok * in_dim);
                let want = host_matmul(&w, &x, in_dim, out_dim);
                let got = launch_mm(gpu, &fx, format, tile, &x);
                let err = rel_err(&got, &want);
                assert!(
                    err <= 1e-4,
                    "{format:?} {tile:?} {n_tok} rows ({in_dim}x{out_dim}): {err}"
                );
            }
        }
    }
}

/// Every matvec, matmul and gather entry name.
#[must_use]
pub fn entries() -> Vec<String> {
    let mut out = Vec::new();
    for format in [Format::Pq2, Format::Ptq1, Format::Mlx2] {
        for t in TOKEN_BLOCKS {
            out.push(mv_entry(format, t));
        }
        out.push(rows_entry(format));
        out.push(mmv_entry(format));
        out.push(mm_entry(format, 8));
        out.push(mm_entry(format, 32));
    }
    out
}
