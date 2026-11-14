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
    /// 4-bit linear codes, each its nibble less eight, 32 weights. It has
    /// only the prompt and step projection entries (a latent stage's
    /// projections), so it is not in [`FORMATS`].
    Q4_0,
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
            Self::Q4_0 => "q4_0",
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
            Self::Iq4Nl | Self::Q4_0 => (32, 18),
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
            Self::Q4_0 => {
                let mut out = vec![0f32; 32];
                for j in 0..16 {
                    out[j] = d * (f32::from(b[2 + j] & 0xF) - 8.0);
                    out[j + 16] = d * (f32::from(b[2 + j] >> 4) - 8.0);
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
    StepGateUpAct,
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
                Stage::StepGateUpAct => concat!("q4e_g_step_gate_up_act_", $n, "_kernel"),
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
        Format::Q4_0 => names!("q4_0"),
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

/// The lanes-a-row step entries: every format's dense projection from two
/// lanes to a whole block a row, and the routed stages from two lanes to a
/// warp, match the host; a row's result does not move when it runs
/// alone, which is what lets a verify reproduce a step.
pub fn the_lane_step_entries_match_the_host_in_every_block_format(gpu: &Gpu) {
    for (i, format) in FORMATS.into_iter().enumerate() {
        let (out_dim, in_dim, rows) = (37usize, 1024usize, 11usize);
        let bank = Bank::new(format, 1, out_dim, in_dim, 1301 + i as u32);
        let x = signal(rows * in_dim, 31 + i as u32);
        let want: Vec<f32> = (0..rows)
            .flat_map(|r| bank.project(0, &x[r * in_dim..(r + 1) * in_dim]))
            .collect();
        for lpr in [1, 2, 8, 16, 32, 128] {
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
        for (gl, dl) in [(32, 16), (16, 32), (8, 2), (2, 1), (4, 4), (1, 2)] {
            routed(gpu, format, Tier::Lanes(gl, dl), 5, 1500 + 10 * i as u32);
        }
    }
}

/// The fused gate/up+SwiGLU step entry writes exactly what the unfused gate/up
/// followed by the device's own `q4e_swiglu_kernel` writes, bit for bit, at
/// every unit-path lane count and for every block format. The activation rides
/// in the gate/up epilogue, so this is the check that it rounds the same way
/// the separate kernel does (`g / (1 + precise::exp(-g)) * u`, not a sigmoid).
pub fn the_fused_gate_up_activation_matches_the_unfused_chain(gpu: &Gpu) {
    for (i, format) in FORMATS.into_iter().enumerate() {
        // Single precision has no routed step entries.
        if format == Format::F32 {
            continue;
        }
        let (hidden, inter, experts, used) = (512usize, 256usize, 6usize, 2usize);
        let tokens = 3usize;
        let gate = Bank::new(format, experts, inter, hidden, 2101 + i as u32);
        let up = Bank::new(format, experts, inter, hidden, 2102 + i as u32);
        let x = signal(tokens * hidden, 2103 + i as u32);
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
        let gw = gpu.upload(&gate_bytes);
        let uw = gpu.upload(&up_bytes);
        let xs = gpu.upload(&f32_bytes(&x));
        let sorted_b = gpu.upload(&i32_le(&sorted));
        let bounds_b = gpu.upload(&i32_le(&bounds));
        let active_b = gpu.upload(&i32_le(&active));
        let map_b = gpu.upload(&u32_le(&map));
        // The unit path the decode step takes: `lpr * 8 <= 32`.
        for gl in [1u32, 2, 4] {
            let out_gate = gpu.upload(&vec![0u8; cols * inter * 4]);
            let out_up = gpu.upload(&vec![0u8; cols * inter * 4]);
            let per_block = 4 * 16 / gl as usize;
            let grid = Extent3::new(inter.div_ceil(per_block) as u32, experts as u32, 1);
            let routing = [
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
                KernelArg::U32(gl),
            ];
            let mut unfused = vec![
                KernelArg::Buffer(out_gate.view()),
                KernelArg::Buffer(out_up.view()),
            ];
            unfused.extend_from_slice(&routing);
            run(
                gpu,
                KernelRef::new(MODULE, entry(format, Stage::StepGateUp)),
                &LaunchArgs::new(grid, Extent3::linear(128), &unfused),
            );
            drain(gpu);
            // The device's own activation, as the unfused chain runs it.
            let mid_ref = gpu.upload(&vec![0u8; cols * inter * 4]);
            let n = (cols * inter) as u64;
            run(
                gpu,
                KernelRef::new("q4e_moe", "q4e_swiglu_kernel"),
                &LaunchArgs::new(
                    Extent3::linear(n.div_ceil(256) as u32),
                    Extent3::linear(256),
                    &[
                        KernelArg::Buffer(mid_ref.view()),
                        KernelArg::Buffer(out_gate.view()),
                        KernelArg::Buffer(out_up.view()),
                        KernelArg::U64(n),
                    ],
                ),
            );
            drain(gpu);
            // The fused entry: one output, the activation in the epilogue.
            let mid_fused = gpu.upload(&vec![0u8; cols * inter * 4]);
            let mut fused = vec![KernelArg::Buffer(mid_fused.view())];
            fused.extend_from_slice(&routing);
            run(
                gpu,
                KernelRef::new(MODULE, entry(format, Stage::StepGateUpAct)),
                &LaunchArgs::new(grid, Extent3::linear(128), &fused),
            );
            drain(gpu);
            let want = gpu.read_f32(&mid_ref);
            let got = gpu.read_f32(&mid_fused);
            assert!(
                got == want,
                "{} gl {gl}: fused activation moved a bit",
                format.name()
            );
        }
    }
}

/// Stored bytes of `slots` matrices of `out_dim` rows, with no decoded copy:
/// a bench shape is too large to decode on the host for nothing.
fn bench_bytes(format: Format, slots: usize, out_dim: usize, in_dim: usize, seed: u32) -> Vec<u8> {
    let (per, _) = format.block();
    let mut rng = seed | 1;
    let mut bytes = Vec::new();
    for _ in 0..slots * out_dim * in_dim / per {
        bytes.extend(format.random_block(&mut rng));
    }
    bytes
}

/// A dense step projection of `rows` activation rows at `lpr` lanes a row,
/// at the stage the engine picks for that row count, for [`crate::bench`].
fn bench_step_matvec(
    gpu: &Gpu,
    format: Format,
    (out_dim, in_dim): (usize, usize),
    rows: usize,
    lpr: u32,
) -> crate::bench::Fixture {
    let bytes = bench_bytes(format, 1, out_dim, in_dim, 0x51ED);
    let weight_bytes = bytes.len() as u64;
    let w = gpu.upload(&bytes);
    let xs = gpu.upload(&f32_bytes(&signal(rows * in_dim, 77)));
    let out = gpu.upload(&vec![0u8; rows * out_dim * 4]);
    let four = rows > 2;
    let per_block = 128 / lpr as usize * if four { 4 } else { 1 };
    let grid = Extent3::new(
        out_dim.div_ceil(per_block) as u32,
        rows.div_ceil(COLS) as u32,
        1,
    );
    let stage = if four {
        Stage::StepMatvec4
    } else {
        Stage::StepMatvec
    };
    crate::bench::Fixture::new(
        weight_bytes,
        2 * (rows * out_dim * in_dim) as u64,
        move |gpu| {
            let args = [
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(w.view()),
                KernelArg::Buffer(xs.view()),
                KernelArg::U32(in_dim as u32),
                KernelArg::U32(out_dim as u32),
                KernelArg::U32(rows as u32),
                KernelArg::U32(lpr),
            ];
            run(
                gpu,
                KernelRef::new(MODULE, entry(format, stage)),
                &LaunchArgs::new(grid, Extent3::linear(128), &args),
            );
        },
    )
}

/// The extended `q6_K` matvec's decode specialization: two simdgroups, sixteen
/// lanes a row (`shiso_engine::launch::ext_consts`, `ext_nxpsg` at one token).
const EXT_Q6_DECODE: [shiso_core::FnConst; 2] = [
    shiso_core::FnConst {
        index: 600,
        value: shiso_core::FnConstValue::I16(2),
    },
    shiso_core::FnConst {
        index: 601,
        value: shiso_core::FnConstValue::I16(16),
    },
];

/// The extended `q6_K` matvec (`qwen35/gemv_ext_q5k_q6k.metal`) at one
/// activation row, for an A/B against the packed step entry at the same shape.
fn bench_ext_q6_k(gpu: &Gpu, (out_dim, in_dim): (usize, usize)) -> crate::bench::Fixture {
    use crate::generic::product::MulMvExtArgs;
    let bytes = bench_bytes(Format::Q6K, 1, out_dim, in_dim, 0x51ED);
    let weight_bytes = bytes.len() as u64;
    let row_bytes = weight_bytes / out_dim as u64;
    let w = gpu.upload(&bytes);
    let xs = gpu.upload(&f32_bytes(&signal(in_dim, 77)));
    let out = gpu.upload(&vec![0u8; out_dim * 4]);
    let args = MulMvExtArgs {
        ne00: in_dim as i32,
        ne01: out_dim as i32,
        ne02: 1,
        pad0: [0; 4],
        nb00: row_bytes,
        nb01: row_bytes,
        nb02: weight_bytes,
        nb03: weight_bytes,
        ne10: in_dim as i32,
        ne11: 1,
        ne12: 1,
        pad1: [0; 4],
        nb10: 4,
        nb11: (in_dim * 4) as u64,
        nb12: (in_dim * 4) as u64,
        nb13: (in_dim * 4) as u64,
        ne0: out_dim as i32,
        ne1: 1,
        r2: 1,
        r3: 1,
        pad2: [0; 4],
    };
    // Four output rows a threadgroup at sixteen lanes a row and two simdgroups.
    let grid = Extent3::new(out_dim.div_ceil(4) as u32, 1, 1);
    crate::bench::Fixture::new(weight_bytes, 2 * (out_dim * in_dim) as u64, move |gpu| {
        run(
            gpu,
            KernelRef::specialized("qwen35", "kernel_mul_mv_ext_q6_K_f32_r1_1", &EXT_Q6_DECODE),
            &LaunchArgs::new(
                grid,
                Extent3::new(32, 2, 1),
                &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(w.view()),
                    KernelArg::Buffer(xs.view()),
                    KernelArg::Buffer(out.view()),
                ],
            ),
        );
    })
}

/// The routed tokens, experts a token and bank slots of a bench stage: a
/// three-row verify of a ten-of-many router, a few experts shared.
const BENCH_ROUTED: (usize, usize, usize) = (3, 10, 30);

/// A routed step stage - gate and up, or down - over `tokens` rows of
/// [`BENCH_ROUTED`]'s router at `lpr` lanes a row, for [`crate::bench`].
fn bench_step_routed(
    gpu: &Gpu,
    format: Format,
    gate_up: bool,
    (hidden, inter): (usize, usize),
    lpr: u32,
    tokens: usize,
) -> crate::bench::Fixture {
    let (_, used, experts) = BENCH_ROUTED;
    // Token t picks experts 8 t .. 8 t + 9, so neighbours share two.
    let ids: Vec<usize> = (0..tokens * used)
        .map(|c| (8 * (c / used) + c % used) % experts)
        .collect();
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
    let n_active = active.len();
    active.resize(experts, 0);
    active.push(n_active as i32);
    let map: Vec<u32> = (0..experts as u32).collect();
    let i32_le = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let u32_le = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let (out_dim, in_dim) = if gate_up {
        (inter, hidden)
    } else {
        (hidden, inter)
    };
    let bank = bench_bytes(format, experts, out_dim, in_dim, 0xBA5E);
    let stride = (bank.len() / experts) as u32;
    let matrices = if gate_up { 2 } else { 1 };
    let read = (bank.len() / experts * n_active * matrices) as u64;
    let flops = 2 * (ids.len() * out_dim * in_dim * matrices) as u64;
    let w = gpu.upload(&bank);
    let w2 = gpu.upload(&bank);
    let x_rows = if gate_up { tokens } else { ids.len() };
    let xs = gpu.upload(&f32_bytes(&signal(x_rows * in_dim, 78)));
    let sorted_b = gpu.upload(&i32_le(&sorted));
    let bounds_b = gpu.upload(&i32_le(&bounds));
    let active_b = gpu.upload(&i32_le(&active));
    let map_b = gpu.upload(&u32_le(&map));
    let out = gpu.upload(&vec![0u8; ids.len() * out_dim * 4]);
    let out2 = gpu.upload(&vec![0u8; ids.len() * out_dim * 4]);
    let per_block = if gate_up { 4 * 16 } else { 4 * 32 } / lpr as usize;
    let grid = Extent3::new(out_dim.div_ceil(per_block) as u32, experts as u32, 1);
    crate::bench::Fixture::new(read, flops, move |gpu| {
        let shape = [
            KernelArg::U32(in_dim as u32),
            KernelArg::U32(out_dim as u32),
            KernelArg::U32(experts as u32),
            KernelArg::U32(used as u32),
        ];
        let routing = [
            KernelArg::Buffer(xs.view()),
            KernelArg::Buffer(sorted_b.view()),
            KernelArg::Buffer(bounds_b.view()),
            KernelArg::Buffer(active_b.view()),
            KernelArg::Buffer(map_b.view()),
        ];
        let mut args = Vec::with_capacity(16);
        if gate_up {
            args.extend([
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(out2.view()),
                KernelArg::Buffer(w.view()),
                KernelArg::Buffer(w2.view()),
            ]);
            args.extend(routing);
            args.extend(shape);
            args.extend([KernelArg::U32(stride), KernelArg::U32(stride)]);
        } else {
            args.extend([KernelArg::Buffer(out.view()), KernelArg::Buffer(w.view())]);
            args.extend(routing);
            args.extend(shape);
            args.push(KernelArg::U32(stride));
        }
        args.push(KernelArg::U32(lpr));
        let stage = if gate_up {
            Stage::StepGateUp
        } else {
            Stage::StepDown
        };
        run(
            gpu,
            KernelRef::new(MODULE, entry(format, stage)),
            &LaunchArgs::new(grid, Extent3::linear(128), &args),
        );
    })
}

/// The routed and dense widths of the bench shapes: the published sparse
/// checkpoint's.
const BENCH_HIDDEN: usize = 2560;
const BENCH_INTER: usize = 640;

macro_rules! bench_cases {
    ($($name:ident => $body:expr;)*) => {
        $(
            #[doc = concat!("`", stringify!($name), "` for [`crate::bench`].")]
            pub(crate) fn $name(gpu: &Gpu) -> crate::bench::Fixture {
                let build: fn(&Gpu) -> crate::bench::Fixture = $body;
                build(gpu)
            }
        )*
    };
}

// The lanes a row are the ones the engine picks for these widths
// (`shiso_engine::sparse::ggml::step_lanes`, `step_expert_lanes`).
bench_cases! {
    bench_gate_up_iq2_s_l2 => |g| bench_step_routed(g, Format::Iq2S, true, (BENCH_HIDDEN, BENCH_INTER), 2, 3);
    bench_gate_up_iq2_s_l1 => |g| bench_step_routed(g, Format::Iq2S, true, (BENCH_HIDDEN, BENCH_INTER), 1, 3);
    bench_gate_up_iq2_s_l4 => |g| bench_step_routed(g, Format::Iq2S, true, (BENCH_HIDDEN, BENCH_INTER), 4, 3);
    bench_gate_up_iq3_xxs_l2 => |g| bench_step_routed(g, Format::Iq3Xxs, true, (BENCH_HIDDEN, BENCH_INTER), 2, 3);
    bench_gate_up_iq3_xxs_l4 => |g| bench_step_routed(g, Format::Iq3Xxs, true, (BENCH_HIDDEN, BENCH_INTER), 4, 3);
    bench_down_iq4_nl_l2 => |g| bench_step_routed(g, Format::Iq4Nl, false, (BENCH_HIDDEN, BENCH_INTER), 2, 3);
    bench_down_iq4_nl_l1 => |g| bench_step_routed(g, Format::Iq4Nl, false, (BENCH_HIDDEN, BENCH_INTER), 1, 3);
    bench_down_iq4_nl_l4 => |g| bench_step_routed(g, Format::Iq4Nl, false, (BENCH_HIDDEN, BENCH_INTER), 4, 3);
    bench_gate_up_iq2_s_t1_l4 => |g| bench_step_routed(g, Format::Iq2S, true, (BENCH_HIDDEN, BENCH_INTER), 4, 1);
    bench_gate_up_iq2_s_t1_l2 => |g| bench_step_routed(g, Format::Iq2S, true, (BENCH_HIDDEN, BENCH_INTER), 2, 1);
    bench_gate_up_iq2_s_t1_l1 => |g| bench_step_routed(g, Format::Iq2S, true, (BENCH_HIDDEN, BENCH_INTER), 1, 1);
    bench_gate_up_iq3_xxs_t1_l4 => |g| bench_step_routed(g, Format::Iq3Xxs, true, (BENCH_HIDDEN, BENCH_INTER), 4, 1);
    bench_down_iq4_nl_t1_l2 => |g| bench_step_routed(g, Format::Iq4Nl, false, (BENCH_HIDDEN, BENCH_INTER), 2, 1);
    bench_down_iq4_nl_t1_l1 => |g| bench_step_routed(g, Format::Iq4Nl, false, (BENCH_HIDDEN, BENCH_INTER), 1, 1);
    bench_q6_k_2560x2560_r3_l8 => |g| bench_step_matvec(g, Format::Q6K, (2560, 2560), 3, 8);
    bench_q6_k_2560x2560_r1_l8 => |g| bench_step_matvec(g, Format::Q6K, (2560, 2560), 1, 8);
    bench_q6_k_2560x2560_r1_l16 => |g| bench_step_matvec(g, Format::Q6K, (2560, 2560), 1, 16);
    bench_q6_k_2560x2560_r1_l32 => |g| bench_step_matvec(g, Format::Q6K, (2560, 2560), 1, 32);
    bench_q6_k_10240x2560_r1_l8 => |g| bench_step_matvec(g, Format::Q6K, (10240, 2560), 1, 8);
    bench_q6_k_10240x2560_r1_l16 => |g| bench_step_matvec(g, Format::Q6K, (10240, 2560), 1, 16);
    bench_q6_k_10240x2560_r1_l32 => |g| bench_step_matvec(g, Format::Q6K, (10240, 2560), 1, 32);
    bench_q6_k_6144x2560_r1_l8 => |g| bench_step_matvec(g, Format::Q6K, (6144, 2560), 1, 8);
    bench_q6_k_2560x6144_r1_l16 => |g| bench_step_matvec(g, Format::Q6K, (2560, 6144), 1, 16);
    bench_ext_q6_k_2560x2560 => |g| bench_ext_q6_k(g, (2560, 2560));
    bench_ext_q6_k_10240x2560 => |g| bench_ext_q6_k(g, (10240, 2560));
    bench_ext_q6_k_6144x2560 => |g| bench_ext_q6_k(g, (6144, 2560));
    bench_ext_q6_k_2560x6144 => |g| bench_ext_q6_k(g, (2560, 6144));
    bench_q8_0_10240x320_r3_l1 => |g| bench_step_matvec(g, Format::Q8_0, (10240, 320), 3, 1);
    bench_q8_0_10240x320_r1_l1 => |g| bench_step_matvec(g, Format::Q8_0, (10240, 320), 1, 1);
    bench_q8_0_10240x320_r1_l4 => |g| bench_step_matvec(g, Format::Q8_0, (10240, 320), 1, 4);
    bench_q6_k_head_r1_l8 => |g| bench_step_matvec(g, Format::Q6K, (62080, 2560), 1, 8);
}

/// The four-bit linear format's two entries - the prompt projection on tensor
/// cores and the lanes-a-row step projection - against the host, as the
/// block formats' own cases hold theirs: across partial tiles, at a warp and
/// a block of lanes a row, with a row unmoved by the pass width, by running
/// alone, or by four output rows a unit.
///
/// # Panics
///
/// When an output is outside the host's tolerance, a row moves, or the device
/// refuses a launch or a read.
pub fn the_four_bit_projection_entries_match_the_host(gpu: &Gpu) {
    let format = Format::Q4_0;
    let (out_dim, in_dim, rows) = (137usize, 512usize, 70usize);
    let bank = Bank::new(format, 1, out_dim, in_dim, 2101);
    let x = halves(&signal(rows * in_dim, 2102));
    let got = gemm(gpu, format, &bank, &x, rows);
    let want: Vec<f32> = (0..rows)
        .flat_map(|r| bank.project(0, &x[r * in_dim..(r + 1) * in_dim]))
        .collect();
    assert_close("q4_0 prompt", &got, &want, 2e-4);
    assert!(
        gemm(gpu, format, &bank, &x, 5) == got[..5 * out_dim],
        "q4_0: a row moved with the chunk width"
    );

    let (out_dim, in_dim, rows) = (37usize, 1024usize, 11usize);
    let bank = Bank::new(format, 1, out_dim, in_dim, 2103);
    let x = signal(rows * in_dim, 2104);
    let want: Vec<f32> = (0..rows)
        .flat_map(|r| bank.project(0, &x[r * in_dim..(r + 1) * in_dim]))
        .collect();
    for lpr in [32, 128] {
        let got = step_matvec(gpu, format, &bank, &x, rows, lpr, false);
        assert_close(&format!("q4_0 lpr {lpr}"), &got, &want, 2e-4);
        assert!(
            step_matvec(gpu, format, &bank, &x, 1, lpr, false) == got[..out_dim],
            "q4_0 lpr {lpr}: row 0 moved alone"
        );
        assert!(
            step_matvec(gpu, format, &bank, &x, rows, lpr, true) == got,
            "q4_0 lpr {lpr}: four rows a unit moved a row"
        );
    }
}
