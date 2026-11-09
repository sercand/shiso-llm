//! The sparse family's GGML block entries: dense projections, the routed
//! expert stages and the vocabulary gather over weights kept in their GGUF
//! blocks, graded against a host decode of the same bytes.
//!
//! The 256-weight i-quants are decoded on the host by the loader's own block
//! decoders; the three formats it has none for (`q8_0`, `iq4_nl`, `q2_0`) and
//! `q6_K` are restated here from the block layout, so a kernel and the host
//! disagree whenever either misreads a bit.
//!
//! Block bytes are random apart from each scale field, which is set to a small
//! finite half so every block decodes to finite weights. The budgets are
//! relative to the magnitude of the sums: the device folds a row lane-major,
//! the host in index order.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "fixture dimensions are small literals, codes are a few bits wide, and the ids \
              and column indices the kernels read as i32 are below a thousand"
)]
#![expect(
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    reason = "every case asserts by panicking, and the routed case runs both expert stages as \
              one sequence"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;
use half::f16;
use shiso_core::{DType, Extent3, KernelArg, KernelRef, LaunchArgs};
use shiso_modelfile::iq_quant::decode_block;

/// The module every entry here is filed under.
const MODULE: &str = "q4e_ggml";

/// Output rows one threadgroup covers (`Q4E_G_ROWS`).
const ROWS: usize = 4;
/// Activation rows one pass carries (`Q4E_G_COLS`).
const COLS: usize = 8;

/// A stored format the entries are instantiated for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Single precision, read 32 floats at a time.
    F32,
    /// 32 signed bytes and a half scale.
    Q8_0,
    /// 6-bit `K`-quant, 256 weights.
    Q6K,
    /// 3-bit grid i-quant, 256 weights.
    Iq3Xxs,
    /// 4-bit non-linear codebook, 32 weights.
    Iq4Nl,
    /// 2-bit grid i-quant, 256 weights.
    Iq2S,
    /// 4-bit non-linear codebook with 6-bit sub-scales, 256 weights.
    Iq4Xs,
    /// 2-bit codes standing for -1..2 steps, 64 weights.
    Q2_0,
}

/// Every format, in instantiation order.
pub const FORMATS: [Format; 8] = [
    Format::F32,
    Format::Q8_0,
    Format::Q6K,
    Format::Iq3Xxs,
    Format::Iq4Nl,
    Format::Iq2S,
    Format::Iq4Xs,
    Format::Q2_0,
];

const IQ4NL: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0,
    89.0, 113.0,
];

impl Format {
    /// The name the entries carry.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q8_0 => "q8_0",
            Self::Q6K => "q6_k",
            Self::Iq3Xxs => "iq3_xxs",
            Self::Iq4Nl => "iq4_nl",
            Self::Iq2S => "iq2_s",
            Self::Iq4Xs => "iq4_xs",
            Self::Q2_0 => "q2_0",
        }
    }

    /// `(weights, bytes)` of one block.
    #[must_use]
    pub const fn block(self) -> (usize, usize) {
        match self {
            Self::F32 => (32, 128),
            Self::Q8_0 => (32, 34),
            Self::Q6K => (256, 210),
            Self::Iq3Xxs => (256, 98),
            Self::Iq4Nl => (32, 18),
            Self::Iq2S => (256, 82),
            Self::Iq4Xs => (256, 136),
            Self::Q2_0 => (64, 18),
        }
    }

    /// Byte offset of the block's half scale.
    const fn scale_at(self) -> usize {
        match self {
            Self::Q6K => 208,
            _ => 0,
        }
    }

    /// One random block with a small finite scale.
    fn random_block(self, rng: &mut u32) -> Vec<u8> {
        if self == Self::F32 {
            return f32_bytes(&signal(32, next(rng)));
        }
        let (_, bytes) = self.block();
        let mut b: Vec<u8> = (0..bytes).map(|_| next(rng) as u8).collect();
        let scale = 0.002 + (next(rng) % 1000) as f32 * 1e-5;
        let at = self.scale_at();
        b[at..at + 2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
        if self == Self::Q6K {
            // Sub-block scales are signed bytes; keep them small.
            for s in &mut b[192..208] {
                *s = ((next(rng) % 33) as i32 - 16) as i8 as u8;
            }
        }
        b
    }

    /// Decodes one block on the host.
    fn decode(self, b: &[u8]) -> Vec<f32> {
        let d = f16::from_le_bytes([b[self.scale_at()], b[self.scale_at() + 1]]).to_f32();
        match self {
            Self::Q8_0 => b[2..34].iter().map(|&q| d * f32::from(q as i8)).collect(),
            Self::Iq4Nl => {
                let mut out = vec![0f32; 32];
                for j in 0..16 {
                    out[j] = d * IQ4NL[usize::from(b[2 + j] & 0xF)];
                    out[j + 16] = d * IQ4NL[usize::from(b[2 + j] >> 4)];
                }
                out
            }
            Self::Q2_0 => (0..64)
                .map(|e| {
                    let q = (b[2 + e / 4] >> ((e % 4) * 2)) & 3;
                    d * (f32::from(q) - 1.0)
                })
                .collect(),
            Self::Q6K => {
                let mut out = vec![0f32; 256];
                for n in 0..2 {
                    for l in 0..32 {
                        let ql = &b[n * 64..];
                        let qh = b[128 + n * 32 + l];
                        let sc = |i: usize| f32::from(b[192 + n * 8 + i] as i8);
                        let is = l / 16;
                        let q =
                            |low: u8, hi: u8| f32::from(((low & 0xF) | ((hi & 3) << 4)) as i8 - 32);
                        out[n * 128 + l] = d * sc(is) * q(ql[l], qh);
                        out[n * 128 + l + 32] = d * sc(is + 2) * q(ql[l + 32], qh >> 2);
                        out[n * 128 + l + 64] = d * sc(is + 4) * q(ql[l] >> 4, qh >> 4);
                        out[n * 128 + l + 96] = d * sc(is + 6) * q(ql[l + 32] >> 4, qh >> 6);
                    }
                }
                out
            }
            Self::F32 => b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect(),
            Self::Iq3Xxs | Self::Iq2S | Self::Iq4Xs => {
                let dtype = match self {
                    Self::Iq3Xxs => DType::IQ3_XXS,
                    Self::Iq2S => DType::IQ2_S,
                    _ => DType::IQ4_XS,
                };
                let mut out = vec![0f32; 256];
                decode_block(dtype, b, &mut out).expect("a whole i-quant block");
                out
            }
        }
    }
}

/// Which entry of a format's four.
#[derive(Debug, Clone, Copy)]
enum Stage {
    Matvec,
    GateUp,
    Down,
    Embed,
    Gemm,
    GemmGateUp,
    GemmDown,
    StepMatvec,
    StepMatvec4,
    StepGateUp,
    StepDown,
}

/// The entry name of `stage` for `format`.
fn entry(format: Format, stage: Stage) -> &'static str {
    macro_rules! names {
        ($n:literal) => {
            match stage {
                Stage::Matvec => concat!("q4e_g_matvec_", $n, "_kernel"),
                Stage::GateUp => concat!("q4e_g_experts_gate_up_", $n, "_kernel"),
                Stage::Down => concat!("q4e_g_experts_down_", $n, "_kernel"),
                Stage::Embed => concat!("q4e_g_embed_", $n, "_kernel"),
                Stage::Gemm => concat!("q4e_g_gemm_", $n, "_kernel"),
                Stage::GemmGateUp => concat!("q4e_g_experts_gemm_gate_up_", $n, "_kernel"),
                Stage::GemmDown => concat!("q4e_g_experts_gemm_down_", $n, "_kernel"),
                Stage::StepMatvec => concat!("q4e_g_step_matvec_", $n, "_kernel"),
                Stage::StepMatvec4 => concat!("q4e_g_step_matvec4_", $n, "_kernel"),
                Stage::StepGateUp => concat!("q4e_g_step_gate_up_", $n, "_kernel"),
                Stage::StepDown => concat!("q4e_g_step_down_", $n, "_kernel"),
            }
        };
    }
    match format {
        Format::F32 => names!("f32"),
        Format::Q8_0 => names!("q8_0"),
        Format::Q6K => names!("q6_k"),
        Format::Iq3Xxs => names!("iq3_xxs"),
        Format::Iq4Nl => names!("iq4_nl"),
        Format::Iq2S => names!("iq2_s"),
        Format::Iq4Xs => names!("iq4_xs"),
        Format::Q2_0 => names!("q2_0"),
    }
}

fn next(rng: &mut u32) -> u32 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 17;
    *rng ^= *rng << 5;
    *rng
}

/// A bank of `slots` matrices of `out_dim` rows, stored and decoded.
struct Bank {
    bytes: Vec<u8>,
    values: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl Bank {
    fn new(format: Format, slots: usize, out_dim: usize, in_dim: usize, seed: u32) -> Self {
        let (per, _) = format.block();
        assert_eq!(in_dim % per, 0, "{}: width {in_dim}", format.name());
        let mut rng = seed | 1;
        let mut bytes = Vec::new();
        let mut values = Vec::new();
        for _ in 0..slots * out_dim * in_dim / per {
            let b = format.random_block(&mut rng);
            values.extend(format.decode(&b));
            bytes.extend(b);
        }
        Self {
            bytes,
            values,
            in_dim,
            out_dim,
        }
    }

    /// The bank laid out with `pad` bytes after every slot, as a streamed
    /// pool sized for a wider layer holds it; returns the bytes and the
    /// stride between slots.
    fn padded(&self, slots: usize, pad: usize) -> (Vec<u8>, u32) {
        let natural = self.bytes.len() / slots;
        let stride = natural + pad;
        let mut out = vec![0xA5u8; slots * stride];
        for slot in 0..slots {
            out[slot * stride..slot * stride + natural]
                .copy_from_slice(&self.bytes[slot * natural..(slot + 1) * natural]);
        }
        (out, stride as u32)
    }

    fn project(&self, slot: usize, x: &[f32]) -> Vec<f32> {
        (0..self.out_dim)
            .map(|o| {
                let row = &self.values[(slot * self.out_dim + o) * self.in_dim..][..self.in_dim];
                row.iter()
                    .zip(x)
                    .map(|(w, v)| f64::from(*w) * f64::from(*v))
                    .sum::<f64>() as f32
            })
            .collect()
    }
}

fn assert_close(what: &str, got: &[f32], want: &[f32], rel: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mean = want.iter().map(|v| v.abs()).sum::<f32>() / want.len().max(1) as f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = rel * w.abs().max(mean).max(1e-6);
        assert!(
            (g - w).abs() <= tol,
            "{what}: element {i} is {g}, the host says {w} (tolerance {tol})"
        );
    }
}

fn matvec(gpu: &Gpu, format: Format, bank: &Bank, x: &[f32], rows: usize) -> Vec<f32> {
    let w = gpu.upload(&bank.bytes);
    let xs = gpu.upload(&f32_bytes(&x[..rows * bank.in_dim]));
    let out = gpu.upload(&vec![0u8; rows * bank.out_dim * 4]);
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(w.view()),
        KernelArg::Buffer(xs.view()),
        KernelArg::U32(bank.in_dim as u32),
        KernelArg::U32(bank.out_dim as u32),
        KernelArg::U32(rows as u32),
    ];
    let grid = Extent3::new(
        bank.out_dim.div_ceil(ROWS) as u32,
        rows.div_ceil(COLS) as u32,
        1,
    );
    run(
        gpu,
        KernelRef::new(MODULE, entry(format, Stage::Matvec)),
        &LaunchArgs::new(grid, Extent3::linear((32 * ROWS) as u32), &args),
    );
    drain(gpu);
    gpu.read_f32(&out)
}

/// Every format's dense projection matches the host at one row and at a row
/// count with a partial last pass, and a row's result does not move when
/// fewer rows share its pass.
pub fn the_dense_projection_matches_the_host_in_every_format(gpu: &Gpu) {
    for (i, format) in FORMATS.into_iter().enumerate() {
        let (out_dim, in_dim, rows) = (37usize, 512usize, 11usize);
        let bank = Bank::new(format, 1, out_dim, in_dim, 101 + i as u32);
        let x = signal(rows * in_dim, 7 + i as u32);
        let got = matvec(gpu, format, &bank, &x, rows);
        let want: Vec<f32> = (0..rows)
            .flat_map(|r| bank.project(0, &x[r * in_dim..(r + 1) * in_dim]))
            .collect();
        assert_close(format.name(), &got, &want, 2e-4);
        let one = matvec(gpu, format, &bank, &x, 1);
        assert!(
            one == got[..out_dim],
            "{}: row 0 moved alone",
            format.name()
        );
    }
}

/// The routed stages: gate and up over a token's columns, then down over each
/// column's own row, every expert reached through the slot map - in slots
/// padded wider than one expert, as a streamed pool sized for another layer's
/// format lays them out.
pub fn the_routed_stages_match_the_host_in_every_format(gpu: &Gpu) {
    for (i, format) in FORMATS.into_iter().enumerate() {
        routed(gpu, format, Tier::Step, 5, 300 + 10 * i as u32);
    }
}

/// Which kernels a routed case runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tier {
    /// The step entries, one warp an output row.
    Step,
    /// The prompt entries on tensor cores, activations rounded to halves.
    Prompt,
    /// The CUDA step entries at these lanes a row of gate/up and of down.
    Lanes(u32, u32),
}

/// One routed case over `tokens` tokens: gate and up, then down, against the
/// host. Returns the gate stage's output, which reads nothing but the
/// activations.
fn routed(gpu: &Gpu, format: Format, tier: Tier, tokens: usize, seed: u32) -> Vec<f32> {
    let (hidden, inter, experts, used) = (512usize, 256usize, 6usize, 2usize);
    // (entry, output rows a block, threads, dynamic shared memory) of each
    // stage.
    let (gu, dn) = match tier {
        Tier::Step => (
            (Stage::GateUp, ROWS, 32 * ROWS, 0),
            (Stage::Down, ROWS, 32 * ROWS, 0),
        ),
        Tier::Lanes(gl, dl) => (
            (Stage::StepGateUp, 4 * 16 / gl as usize, 128, 0),
            (Stage::StepDown, 4 * 32 / dl as usize, 128, 0),
        ),
        Tier::Prompt => (
            (
                Stage::GemmGateUp,
                GEMM_GATE_UP_ROWS / 2,
                2 * GEMM_GATE_UP_ROWS,
                gemm_smem_half(GEMM_GATE_UP_ROWS, GEMM_EXPERT_COLS),
            ),
            (
                Stage::GemmDown,
                GEMM_DOWN_ROWS,
                2 * GEMM_DOWN_ROWS,
                gemm_smem_half(GEMM_DOWN_ROWS, GEMM_EXPERT_COLS),
            ),
        ),
    };
    let budget = 2e-4;
    {
        let gate = Bank::new(format, experts, inter, hidden, seed);
        let up = Bank::new(format, experts, inter, hidden, seed + 1);
        let down = Bank::new(format, experts, hidden, inter, seed + 2);
        let x = signal(tokens * hidden, seed + 3);
        let x = if tier == Tier::Prompt { halves(&x) } else { x };
        // Token t picks experts (t + k) % experts; slot map reverses the bank.
        let ids: Vec<usize> = (0..tokens * used)
            .map(|c| (c / used + c % used) % experts)
            .collect();
        let map: Vec<u32> = (0..experts).map(|e| (experts - 1 - e) as u32).collect();
        let mut sorted = Vec::new();
        let mut bounds = vec![0i32; experts + 1];
        for (e, bound) in bounds.iter_mut().take(experts).enumerate() {
            *bound = sorted.len() as i32;
            sorted.extend((0..ids.len()).filter(|&c| ids[c] == e).map(|c| c as i32));
        }
        bounds[experts] = sorted.len() as i32;
        let mut active: Vec<i32> = (0..experts as i32)
            .filter(|&e| bounds[e as usize + 1] > bounds[e as usize])
            .collect();
        let n_active = active.len() as i32;
        active.resize(experts, 0);
        active.push(n_active);

        let cols = ids.len();
        let i32_le = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let u32_le = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let (gate_bytes, gate_stride) = gate.padded(experts, 96);
        let (up_bytes, up_stride) = up.padded(experts, 96);
        let (down_bytes, down_stride) = down.padded(experts, 160);
        let (gw, uw, dw) = (
            gpu.upload(&gate_bytes),
            gpu.upload(&up_bytes),
            gpu.upload(&down_bytes),
        );
        // The routed prompt entries read their activations as halves.
        let act = |v: &[f32]| {
            if tier == Tier::Prompt {
                half_bytes(v)
            } else {
                f32_bytes(v)
            }
        };
        let xs = gpu.upload(&act(&x));
        let sorted_b = gpu.upload(&i32_le(&sorted));
        let bounds_b = gpu.upload(&i32_le(&bounds));
        let active_b = gpu.upload(&i32_le(&active));
        let map_b = gpu.upload(&u32_le(&map));
        let out_gate = gpu.upload(&vec![0u8; cols * inter * 4]);
        let out_up = gpu.upload(&vec![0u8; cols * inter * 4]);
        run(
            gpu,
            KernelRef::new(MODULE, entry(format, gu.0)),
            &LaunchArgs {
                smem_bytes: gu.3,
                ..LaunchArgs::new(
                    Extent3::new(inter.div_ceil(gu.1) as u32, experts as u32, 1),
                    Extent3::linear(gu.2 as u32),
                    &[
                        KernelArg::Buffer(out_gate.view()),
                        KernelArg::Buffer(out_up.view()),
                        KernelArg::Buffer(gw.view()),
                        KernelArg::Buffer(uw.view()),
                        KernelArg::Buffer(xs.view()),
                        KernelArg::Buffer(sorted_b.view()),
                        KernelArg::Buffer(bounds_b.view()),
                        KernelArg::Buffer(active_b.view()),
                        KernelArg::Buffer(map_b.view()),
                        KernelArg::U32(hidden as u32),
                        KernelArg::U32(inter as u32),
                        KernelArg::U32(experts as u32),
                        KernelArg::U32(used as u32),
                        KernelArg::U32(gate_stride),
                        KernelArg::U32(up_stride),
                        KernelArg::U32(match tier {
                            Tier::Lanes(gl, _) => gl,
                            _ => 0,
                        }),
                    ][..if matches!(tier, Tier::Lanes(..)) {
                        16
                    } else {
                        15
                    }],
                )
            },
        );
        drain(gpu);
        let got_gate = gpu.read_f32(&out_gate);
        let got_up = gpu.read_f32(&out_up);
        let mut want_gate = Vec::new();
        let mut want_up = Vec::new();
        for (c, &e) in ids.iter().enumerate() {
            let xr = &x[(c / used) * hidden..][..hidden];
            let slot = map[e] as usize;
            want_gate.extend(gate.project(slot, xr));
            want_up.extend(up.project(slot, xr));
        }
        assert_close(
            &format!("{} gate", format.name()),
            &got_gate,
            &want_gate,
            budget,
        );
        assert_close(&format!("{} up", format.name()), &got_up, &want_up, budget);

        // Down over the host's own activations, so it is graded on its own.
        let mid: Vec<f32> = want_gate
            .iter()
            .zip(&want_up)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        // The fixture's random banks are not normalized, so a column's
        // activations can pass the half range; scaled into it, they test the
        // same sums.
        let mid = if tier == Tier::Prompt {
            let peak = mid.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
            halves(&mid.iter().map(|v| v / peak).collect::<Vec<_>>())
        } else {
            mid
        };
        let mid_b = gpu.upload(&act(&mid));
        let out_down = gpu.upload(&vec![0u8; cols * hidden * 4]);
        run(
            gpu,
            KernelRef::new(MODULE, entry(format, dn.0)),
            &LaunchArgs {
                smem_bytes: dn.3,
                ..LaunchArgs::new(
                    Extent3::new(hidden.div_ceil(dn.1) as u32, experts as u32, 1),
                    Extent3::linear(dn.2 as u32),
                    &[
                        KernelArg::Buffer(out_down.view()),
                        KernelArg::Buffer(dw.view()),
                        KernelArg::Buffer(mid_b.view()),
                        KernelArg::Buffer(sorted_b.view()),
                        KernelArg::Buffer(bounds_b.view()),
                        KernelArg::Buffer(active_b.view()),
                        KernelArg::Buffer(map_b.view()),
                        KernelArg::U32(inter as u32),
                        KernelArg::U32(hidden as u32),
                        KernelArg::U32(experts as u32),
                        KernelArg::U32(used as u32),
                        KernelArg::U32(down_stride),
                        KernelArg::U32(match tier {
                            Tier::Lanes(_, dl) => dl,
                            _ => 0,
                        }),
                    ][..if matches!(tier, Tier::Lanes(..)) {
                        13
                    } else {
                        12
                    }],
                )
            },
        );
        drain(gpu);
        let got_down = gpu.read_f32(&out_down);
        let want_down: Vec<f32> = ids
            .iter()
            .enumerate()
            .flat_map(|(c, &e)| down.project(map[e] as usize, &mid[c * inter..][..inter]))
            .collect();
        assert_close(
            &format!("{} down", format.name()),
            &got_down,
            &want_down,
            budget,
        );
        got_gate
    }
}

/// Output rows and columns a dense prompt block covers (`Q4E_GG_DENSE_BM`,
/// `Q4E_GG_DENSE_BN`).
const GEMM_ROWS: usize = 128;
const GEMM_COLS: usize = 128;
/// Output rows a gate/up and a down prompt block cover (`Q4E_GG_GATE_UP_BM`,
/// half of each matrix; `Q4E_GG_DOWN_BM`), and the columns a routed block
/// takes at a time (`Q4E_GG_EXPERT_BN`).
const GEMM_GATE_UP_ROWS: usize = 256;
const GEMM_DOWN_ROWS: usize = 128;
const GEMM_EXPERT_COLS: usize = 64;

/// `Q4E_GG_SMEM(rows, cols)`.
/// `Q4E_GG_SMEM_H(rows, cols)`: a routed prompt block's three half
/// activation buffers.
const fn gemm_smem_half(rows: usize, cols: usize) -> u32 {
    ((2 * rows + 3 * cols) * 40 * 2 + 2 * rows * 2 * 4) as u32
}

/// Values as little-endian halves.
fn half_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| f16::from_f32(x).to_le_bytes())
        .collect()
}

const fn gemm_smem(rows: usize, cols: usize) -> u32 {
    (2 * (rows + cols) * 40 * 2 + 2 * rows * 2 * 4) as u32
}

/// Every value rounded to half precision, as the prompt kernels round their
/// activations; the host oracle then sees the operands the device multiplies.
fn halves(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| f16::from_f32(x).to_f32()).collect()
}

fn gemm(gpu: &Gpu, format: Format, bank: &Bank, x: &[f32], rows: usize) -> Vec<f32> {
    let w = gpu.upload(&bank.bytes);
    let xs = gpu.upload(&f32_bytes(&x[..rows * bank.in_dim]));
    let out = gpu.upload(&vec![0u8; rows * bank.out_dim * 4]);
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(w.view()),
        KernelArg::Buffer(xs.view()),
        KernelArg::U32(bank.in_dim as u32),
        KernelArg::U32(bank.out_dim as u32),
        KernelArg::U32(rows as u32),
    ];
    let grid = Extent3::new(
        bank.out_dim.div_ceil(GEMM_ROWS) as u32,
        rows.div_ceil(GEMM_COLS) as u32,
        1,
    );
    run(
        gpu,
        KernelRef::new(MODULE, entry(format, Stage::Gemm)),
        &LaunchArgs {
            smem_bytes: gemm_smem(GEMM_ROWS, GEMM_COLS),
            ..LaunchArgs::new(grid, Extent3::linear((2 * GEMM_ROWS) as u32), &args)
        },
    );
    drain(gpu);
    gpu.read_f32(&out)
}

/// Every block format's prompt projection on tensor cores matches the host
/// over half-rounded activations, across partial row and column tiles, and a
/// row's result does not move with the chunk width.
pub fn the_prompt_projection_matches_the_host_in_every_block_format(gpu: &Gpu) {
    for (i, format) in FORMATS.into_iter().skip(1).enumerate() {
        let (out_dim, in_dim, rows) = (137usize, 512usize, 70usize);
        let bank = Bank::new(format, 1, out_dim, in_dim, 501 + i as u32);
        let x = halves(&signal(rows * in_dim, 17 + i as u32));
        let got = gemm(gpu, format, &bank, &x, rows);
        let want: Vec<f32> = (0..rows)
            .flat_map(|r| bank.project(0, &x[r * in_dim..(r + 1) * in_dim]))
            .collect();
        assert_close(format.name(), &got, &want, 2e-4);
        let few = gemm(gpu, format, &bank, &x, 5);
        assert!(
            few == got[..5 * out_dim],
            "{}: a row moved with the chunk width",
            format.name()
        );
    }
}

/// The prompt pass's routed stages on tensor cores match the host, including
/// an expert with more columns than one tile, and a column's result does not
/// move when the pass is narrower.
pub fn the_prompt_routed_stages_match_the_host_in_every_block_format(gpu: &Gpu) {
    for (i, format) in FORMATS.into_iter().skip(1).enumerate() {
        let seed = 700 + 10 * i as u32;
        // 200 tokens of two picks over six experts put about 67 columns on
        // each: a tile of 64 and a remainder.
        let wide = routed(gpu, format, Tier::Prompt, 200, seed);
        let narrow = routed(gpu, format, Tier::Prompt, 3, seed);
        let inter = 256;
        assert!(
            narrow[..inter] == wide[..inter],
            "{}: column 0 moved with the pass width",
            format.name()
        );
    }
}

/// The vocabulary gather decodes a token's row exactly.
pub fn the_vocabulary_gather_decodes_rows_exactly(gpu: &Gpu) {
    for (i, format) in [Format::Q6K, Format::Q8_0].into_iter().enumerate() {
        let (vocab, embd) = (40usize, 512usize);
        let bank = Bank::new(format, 1, vocab, embd, 900 + i as u32);
        let tokens: Vec<i32> = vec![3, 39, 0, 17];
        let w = gpu.upload(&bank.bytes);
        let ids = gpu.upload(
            &tokens
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        let out = gpu.upload(&vec![0u8; tokens.len() * embd * 4]);
        run(
            gpu,
            KernelRef::new(MODULE, entry(format, Stage::Embed)),
            &LaunchArgs::new(
                Extent3::new(tokens.len() as u32, 1, 1),
                Extent3::linear(32),
                &[
                    KernelArg::Buffer(out.view()),
                    KernelArg::Buffer(ids.view()),
                    KernelArg::Buffer(w.view()),
                    KernelArg::U32(tokens.len() as u32),
                    KernelArg::U32(embd as u32),
                ],
            ),
        );
        drain(gpu);
        let got = gpu.read_f32(&out);
        let want: Vec<f32> = tokens
            .iter()
            .flat_map(|&t| bank.values[t as usize * embd..][..embd].to_vec())
            .collect();
        assert!(got == want, "{}: gathered rows differ", format.name());
    }
}

/// The n-gram table's `IQ4_NL` row decode places each gathered row at its
/// token and head slot, exactly.
pub fn the_ngram_row_decode_places_iq4_nl_rows_exactly(gpu: &Gpu) {
    let (head_dim, n_heads, n_tok) = (160usize, 3usize, 2usize);
    let n_rows = n_heads * n_tok;
    let bank = Bank::new(Format::Iq4Nl, 1, n_rows, head_dim, 1234);
    let row_bytes = head_dim / 32 * 18;
    let rows = gpu.upload(&bank.bytes);
    let out = gpu.upload(&vec![0u8; n_rows * head_dim * 4]);
    run(
        gpu,
        KernelRef::new("q4e_ple", "q4e_ple_dequant_kernel"),
        &LaunchArgs::new(
            Extent3::new(n_rows as u32, 1, 1),
            Extent3::linear(32),
            &[
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(rows.view()),
                KernelArg::U32(head_dim as u32),
                KernelArg::U32(n_heads as u32),
                KernelArg::U32(row_bytes as u32),
                KernelArg::U32(n_rows as u32),
            ],
        ),
    );
    drain(gpu);
    // Row r is token r / n_heads, head r % n_heads: the gather order is
    // already the output order.
    assert!(
        gpu.read_f32(&out) == bank.values,
        "decoded n-gram rows differ"
    );
}

fn step_matvec(
    gpu: &Gpu,
    format: Format,
    bank: &Bank,
    x: &[f32],
    rows: usize,
    lpr: u32,
    four: bool,
) -> Vec<f32> {
    let w = gpu.upload(&bank.bytes);
    let xs = gpu.upload(&f32_bytes(&x[..rows * bank.in_dim]));
    let out = gpu.upload(&vec![0u8; rows * bank.out_dim * 4]);
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(w.view()),
        KernelArg::Buffer(xs.view()),
        KernelArg::U32(bank.in_dim as u32),
        KernelArg::U32(bank.out_dim as u32),
        KernelArg::U32(rows as u32),
        KernelArg::U32(lpr),
    ];
    let per_block = 128 / lpr as usize * if four { 4 } else { 1 };
    let grid = Extent3::new(
        bank.out_dim.div_ceil(per_block) as u32,
        rows.div_ceil(COLS) as u32,
        1,
    );
    run(
        gpu,
        KernelRef::new(
            MODULE,
            entry(
                format,
                if four {
                    Stage::StepMatvec4
                } else {
                    Stage::StepMatvec
                },
            ),
        ),
        &LaunchArgs::new(grid, Extent3::linear(128), &args),
    );
    drain(gpu);
    gpu.read_f32(&out)
}

/// The CUDA step entries: every format's dense projection at a half
/// warp, a warp and a whole block a row, and the routed stages at a half warp
/// and a warp, match the host; a row's result does not move when it runs
/// alone, which is what lets a verify reproduce a step.
pub fn the_lane_step_entries_match_the_host_in_every_block_format(gpu: &Gpu) {
    for (i, format) in FORMATS.into_iter().enumerate() {
        let (out_dim, in_dim, rows) = (37usize, 1024usize, 11usize);
        let bank = Bank::new(format, 1, out_dim, in_dim, 1301 + i as u32);
        let x = signal(rows * in_dim, 31 + i as u32);
        let want: Vec<f32> = (0..rows)
            .flat_map(|r| bank.project(0, &x[r * in_dim..(r + 1) * in_dim]))
            .collect();
        for lpr in [16, 32, 128] {
            let got = step_matvec(gpu, format, &bank, &x, rows, lpr, false);
            assert_close(&format!("{} lpr {lpr}", format.name()), &got, &want, 2e-4);
            let one = step_matvec(gpu, format, &bank, &x, 1, lpr, false);
            assert!(
                one == got[..out_dim],
                "{} lpr {lpr}: row 0 moved alone",
                format.name()
            );
            // Four output rows a unit sum each row as one does.
            let four = step_matvec(gpu, format, &bank, &x, rows, lpr, true);
            assert!(
                four == got,
                "{} lpr {lpr}: four rows a unit moved a row",
                format.name()
            );
        }
        // Single precision has no routed step entries.
        if format == Format::F32 {
            continue;
        }
        for (gl, dl) in [(32, 16), (16, 32)] {
            routed(gpu, format, Tier::Lanes(gl, dl), 5, 1500 + 10 * i as u32);
        }
    }
}
