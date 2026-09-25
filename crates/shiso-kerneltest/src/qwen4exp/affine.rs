//! The sparse family's MLX affine entries: projections, the vocabulary gather,
//! the routed expert stages and the n-gram row decode, graded against a host
//! restatement of MLX's own dequantization.
//!
//! The fixtures are the checkpoint's own layout - codes row-major, eight to a
//! little-endian word with the lowest nibble first, one brain-float scale and
//! bias per group - so a case that passes is a case that read the bytes the
//! loader hands the device, not a repacked copy of them.
//!
//! The budgets are relative to the magnitude of the sums, because the device
//! folds each row lane-major and the host folds it in index order: the two
//! differ by reassociation only, which at these widths is a few units in the
//! sixth significant digit.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "fixture dimensions are small literals, codes are 0..=15, and the expert ids \
              and column indices the kernels read as i32 are in the hundreds"
)]
#![expect(
    clippy::missing_panics_doc,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "every case asserts by panicking; the gather and row decode are exact, so they \
              compare floats exactly; w/s/b are the three arrays of an affine matrix; and a \
              case that runs both expert stages reads best as one sequence"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

/// The module every entry here is filed under.
const MODULE: &str = "q4e_matvec";

/// One affine matrix bank in the checkpoint's layout.
#[derive(Debug, Clone)]
pub struct AffineBank {
    /// Codes, `[slot][out][in / 8]` words.
    pub words: Vec<u32>,
    /// Brain-float scale bit patterns, `[slot][out][in / group]`.
    pub scales: Vec<u16>,
    /// Brain-float bias bit patterns, same shape.
    pub biases: Vec<u16>,
    /// Contraction width.
    pub in_dim: usize,
    /// Output rows per slot.
    pub out_dim: usize,
    /// Codes per group.
    pub group: usize,
}

/// A float rounded to the nearest brain float, as a bit pattern.
fn bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let round = ((bits >> 16) & 1) + 0x7FFF;
    ((bits + round) >> 16) as u16
}

/// A brain-float bit pattern widened exactly.
fn bf16_value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

impl AffineBank {
    /// A deterministic bank of `slots` matrices.
    #[must_use]
    pub fn new(slots: usize, out_dim: usize, in_dim: usize, group: usize, seed: u32) -> Self {
        let groups = in_dim / group;
        let codes = signal(slots * out_dim * in_dim, seed);
        let params = signal(slots * out_dim * groups * 2, seed ^ 0x5bd1);
        let mut words = vec![0u32; slots * out_dim * in_dim / 8];
        for (i, c) in codes.iter().enumerate() {
            let q = ((c.abs() * 97.0) as u32) & 0xF;
            words[i / 8] |= q << (4 * (i % 8));
        }
        let scales = (0..slots * out_dim * groups)
            .map(|i| bf16_bits(0.01 + params[2 * i].abs() * 0.02))
            .collect();
        let biases = (0..slots * out_dim * groups)
            .map(|i| bf16_bits(params[2 * i + 1] * 0.1))
            .collect();
        Self {
            words,
            scales,
            biases,
            in_dim,
            out_dim,
            group,
        }
    }

    /// Element `k` of row `o` of slot `slot`.
    #[must_use]
    pub fn at(&self, slot: usize, o: usize, k: usize) -> f32 {
        let row = slot * self.out_dim + o;
        let index = row * self.in_dim + k;
        let q = (self.words[index / 8] >> (4 * (index % 8))) & 0xF;
        let p = row * (self.in_dim / self.group) + k / self.group;
        q as f32 * bf16_value(self.scales[p]) + bf16_value(self.biases[p])
    }

    /// `W[slot] x` in f64, the host reference.
    #[must_use]
    pub fn project(&self, slot: usize, x: &[f32]) -> Vec<f32> {
        (0..self.out_dim)
            .map(|o| {
                (0..self.in_dim)
                    .map(|k| f64::from(self.at(slot, o, k)) * f64::from(x[k]))
                    .sum::<f64>() as f32
            })
            .collect()
    }

    /// The three device arrays.
    #[must_use]
    pub fn upload(&self, gpu: &Gpu) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer) {
        let words: Vec<u8> = self.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let scales: Vec<u8> = self.scales.iter().flat_map(|w| w.to_le_bytes()).collect();
        let biases: Vec<u8> = self.biases.iter().flat_map(|w| w.to_le_bytes()).collect();
        (gpu.upload(&words), gpu.upload(&scales), gpu.upload(&biases))
    }
}

/// Asserts every value is within `rel` of the reference, relative to the
/// larger of its own magnitude and the row's mean magnitude.
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

/// One projection case.
fn projection(gpu: &Gpu, out_dim: usize, in_dim: usize, rows: usize, seed: u32) {
    let bank = AffineBank::new(1, out_dim, in_dim, 64, seed);
    let x = signal(rows * in_dim, seed ^ 0x77);
    let (w, s, b) = bank.upload(gpu);
    let xs = gpu.upload(&f32_bytes(&x));
    let out = gpu.upload(&vec![0u8; rows * out_dim * 4]);
    let (entry, tokens) = if rows == 1 {
        ("q4e_affine_mv_kernel", 1)
    } else {
        ("q4e_affine_mm_kernel", 8)
    };
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(w.view()),
        KernelArg::Buffer(s.view()),
        KernelArg::Buffer(b.view()),
        KernelArg::Buffer(xs.view()),
        KernelArg::U32(in_dim as u32),
        KernelArg::U32(out_dim as u32),
        KernelArg::U32(rows as u32),
    ];
    let want: Vec<f32> = (0..rows)
        .flat_map(|r| bank.project(0, &x[r * in_dim..(r + 1) * in_dim]))
        .collect();
    let grid = Extent3::new(out_dim.div_ceil(8) as u32, rows.div_ceil(tokens) as u32, 1);
    // Every contraction split the host may choose sums the same product.
    for splits in [1u32, 2, 4] {
        run(
            gpu,
            KernelRef::new(MODULE, entry),
            &LaunchArgs::new(grid, Extent3::linear(64 * splits), &args),
        );
        drain(gpu);
        let got = gpu.read_f32(&out);
        assert_close(entry, &got, &want, 2e-4);
    }
}

/// A decode step's projections at the checkpoint's shapes, including the
/// contractions that are not a multiple of the 256 codes a simdgroup reads in
/// one sweep.
pub fn the_one_row_projection_matches_the_host(gpu: &Gpu) {
    projection(gpu, 512, 2560, 1, 1);
    projection(gpu, 2560, 640, 1, 2);
    projection(gpu, 10240, 320, 1, 3);
    projection(gpu, 13, 6144, 1, 4);
}

/// A prefill chunk's projections, with a row count that leaves a partial
/// last threadgroup of activation rows.
pub fn the_multi_row_projection_matches_the_host(gpu: &Gpu) {
    projection(gpu, 320, 10240, 5, 5);
    projection(gpu, 640, 2560, 19, 6);
    projection(gpu, 10240, 320, 11, 7);
    projection(gpu, 2560, 640, 9, 8);
}

/// One prompt projection on the matrix units, at `rows` activation rows.
fn gemm(
    gpu: &Gpu,
    bank: &AffineBank,
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
    rows: usize,
) -> Vec<f32> {
    let (w, s, b) = bank.upload(gpu);
    let xs = gpu.upload(&f32_bytes(&x[..rows * in_dim]));
    let out = gpu.upload(&vec![0u8; rows * out_dim * 4]);
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(w.view()),
        KernelArg::Buffer(s.view()),
        KernelArg::Buffer(b.view()),
        KernelArg::Buffer(xs.view()),
        KernelArg::U32(in_dim as u32),
        KernelArg::U32(out_dim as u32),
        KernelArg::U32(rows as u32),
    ];
    let grid = Extent3::new(out_dim.div_ceil(32) as u32, rows.div_ceil(32) as u32, 1);
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_affine_gemm_kernel"),
        &LaunchArgs::new(grid, Extent3::linear(128), &args),
    );
    drain(gpu);
    gpu.read_f32(&out)
}

/// The prompt projection matches the host, and a row comes out bit for bit
/// the same whichever rows share its pass - the promise a prompt pass makes.
pub fn the_prompt_projection_matches_the_host_at_every_width(gpu: &Gpu) {
    for (out_dim, in_dim, rows, seed) in [
        (640usize, 2560usize, 45usize, 11u32),
        (70, 320, 33, 12),
        (320, 10240, 7, 13),
    ] {
        let bank = AffineBank::new(1, out_dim, in_dim, 64, seed);
        let x = signal(rows * in_dim, seed ^ 0x55);
        let got = gemm(gpu, &bank, &x, out_dim, in_dim, rows);
        let want: Vec<f32> = (0..rows)
            .flat_map(|r| bank.project(0, &x[r * in_dim..(r + 1) * in_dim]))
            .collect();
        assert_close("q4e_affine_gemm_kernel", &got, &want, 2e-4);
        for narrow in [1usize, 5] {
            let fewer = gemm(gpu, &bank, &x, out_dim, in_dim, narrow);
            assert!(
                fewer == got[..narrow * out_dim],
                "{out_dim}x{in_dim}: the first {narrow} rows moved when the pass narrowed"
            );
        }
    }
}

/// One brain-float prompt projection on the matrix units at `rows` rows.
fn bf16_gemm(
    gpu: &Gpu,
    w: &[u16],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
    rows: usize,
) -> Vec<f32> {
    let wb = gpu.upload(&w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    let xs = gpu.upload(&f32_bytes(&x[..rows * in_dim]));
    let out = gpu.upload(&vec![0u8; rows * out_dim * 4]);
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(wb.view()),
        KernelArg::Buffer(xs.view()),
        KernelArg::U32(in_dim as u32),
        KernelArg::U32(out_dim as u32),
        KernelArg::U32(rows as u32),
    ];
    let grid = Extent3::new(out_dim.div_ceil(32) as u32, rows.div_ceil(32) as u32, 1);
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_bf16_gemm_kernel"),
        &LaunchArgs::new(grid, Extent3::linear(128), &args),
    );
    drain(gpu);
    gpu.read_f32(&out)
}

/// The brain-float prompt projection matches the host at the checkpoint's
/// shapes - the router, the recurrent gates, the stream mixes' injections -
/// and a row comes out the same whichever rows share its pass.
pub fn the_prompt_brain_float_projection_matches_the_host_at_every_width(gpu: &Gpu) {
    for (out_dim, in_dim, rows, seed) in [
        (512usize, 2560usize, 40usize, 21u32),
        (4, 10240, 37, 22),
        (48, 2560, 3, 23),
    ] {
        let w: Vec<u16> = signal(out_dim * in_dim, seed)
            .iter()
            .map(|v| bf16_bits(*v * 0.05))
            .collect();
        let x = signal(rows * in_dim, seed ^ 0x33);
        let got = bf16_gemm(gpu, &w, &x, out_dim, in_dim, rows);
        let want: Vec<f32> = (0..rows)
            .flat_map(|r| {
                let xr = &x[r * in_dim..(r + 1) * in_dim];
                let w = &w;
                (0..out_dim).map(move |o| {
                    let mut acc = 0f64;
                    for k in 0..in_dim {
                        acc += f64::from(bf16_value(w[o * in_dim + k])) * f64::from(xr[k]);
                    }
                    acc as f32
                })
            })
            .collect();
        assert_close("q4e_bf16_gemm_kernel", &got, &want, 2e-4);
        for narrow in [1usize, 2] {
            let fewer = bf16_gemm(gpu, &w, &x, out_dim, in_dim, narrow);
            assert!(
                fewer == got[..narrow * out_dim],
                "{out_dim}x{in_dim}: the first {narrow} rows moved when the pass narrowed"
            );
        }
    }
}

/// The vocabulary gather dequantizes each token's row and fans it out.
pub fn the_affine_gather_matches_the_host(gpu: &Gpu) {
    let (vocab, embd, hc) = (37usize, 2560usize, 2usize);
    let bank = AffineBank::new(1, vocab, embd, 64, 11);
    let tokens: Vec<i32> = vec![3, 36, 0, -1, 99];
    let (w, s, b) = bank.upload(gpu);
    let ids = gpu.upload(
        &tokens
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let n = tokens.len() * hc * embd;
    let out = gpu.upload(&vec![0u8; n * 4]);
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(ids.view()),
        KernelArg::Buffer(w.view()),
        KernelArg::Buffer(s.view()),
        KernelArg::Buffer(b.view()),
        KernelArg::U32(vocab as u32),
        KernelArg::U32(tokens.len() as u32),
        KernelArg::U32(embd as u32),
        KernelArg::U32(hc as u32),
    ];
    run(
        gpu,
        KernelRef::new("q4e_embed", "q4e_affine_embed_hc_kernel"),
        &LaunchArgs::new(
            Extent3::linear(n.div_ceil(256) as u32),
            Extent3::linear(256),
            &args,
        ),
    );
    drain(gpu);
    let got = gpu.read_f32(&out);
    for (t, &tok) in tokens.iter().enumerate() {
        let row = if tok < 0 || tok as usize >= vocab {
            0
        } else {
            tok as usize
        };
        for h in 0..hc {
            for d in 0..embd {
                let g = got[(t * hc + h) * embd + d];
                let w = bank.at(0, row, d);
                assert!(g == w, "token {t} stream {h} element {d}: {g} vs {w}");
            }
        }
    }
}

/// Little-endian `i32` bytes.
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Both routed stages over a sorted column list, with the bank reached
/// through a slot map that is not the identity: the case a streamed bank
/// takes, where expert `e` sits wherever the stream put it.
pub fn the_expert_stages_follow_the_slot_map(gpu: &Gpu) {
    // The checkpoint's intermediate width takes the down projection's row
    // split; a width that is not a multiple of 128 takes the sweep layout.
    expert_stages(gpu, 640);
    expert_stages(gpu, 576);
}

/// One expert-stage case at intermediate width `inter`.
fn expert_stages(gpu: &Gpu, inter: usize) {
    // Two tokens, three slots each: columns 0..6.
    let ids: Vec<i32> = vec![4, 9, 1, 9, 13, 4];
    // Expert -> slot; only the routed experts have one.
    let mut map = vec![u32::MAX; 16];
    for (slot, e) in [1usize, 4, 9, 13].iter().enumerate() {
        map[*e] = [5u32, 0, 3, 2][slot];
    }
    expert_case(gpu, inter, &ids, 16, &map, false);
}

/// The routed stages over `ids` - three slots a token - checked against the
/// host, on the prompt kernels when `gemm`; returns gate, up and down.
fn expert_case(
    gpu: &Gpu,
    inter: usize,
    ids: &[i32],
    n_expert: usize,
    map: &[u32],
    gemm: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (hidden, n_used, slots) = (2560usize, 3usize, 6usize);
    let gate = AffineBank::new(slots, inter, hidden, 64, 21);
    let up = AffineBank::new(slots, inter, hidden, 64, 22);
    let down = AffineBank::new(slots, hidden, inter, 64, 23);
    let n_tok = ids.len() / n_used;
    // The sort's outputs, computed on the host.
    let mut bounds = vec![0i32; n_expert + 1];
    for &e in ids {
        bounds[e as usize + 1] += 1;
    }
    for e in 0..n_expert {
        bounds[e + 1] += bounds[e];
    }
    let mut cursor = bounds.clone();
    let mut sorted = vec![0i32; ids.len()];
    for (col, &e) in ids.iter().enumerate() {
        sorted[cursor[e as usize] as usize] = col as i32;
        cursor[e as usize] += 1;
    }
    let mut active: Vec<i32> = (0..n_expert as i32)
        .filter(|&e| bounds[e as usize + 1] > bounds[e as usize])
        .collect();
    let n_active = active.len() as i32;
    active.resize(n_expert, 0);
    active.push(n_active);

    let x = signal(n_tok * hidden, 31);
    let (gate_up_entry, down_entry, block, per) = if gemm {
        (
            "q4e_affine_experts_gemm_gate_up_kernel",
            "q4e_affine_experts_gemm_down_kernel",
            128,
            (64, 64),
        )
    } else {
        let rows = if inter.is_multiple_of(128) && inter <= 1024 {
            16
        } else {
            8
        };
        (
            "q4e_affine_experts_gate_up_kernel",
            "q4e_affine_experts_down_kernel",
            64,
            (8, rows),
        )
    };
    let (gw, gs, gb) = gate.upload(gpu);
    let (uw, us, ub) = up.upload(gpu);
    let (dw, ds, db) = down.upload(gpu);
    let xs = gpu.upload(&f32_bytes(&x));
    let sorted_b = gpu.upload(&i32_bytes(&sorted));
    let bounds_b = gpu.upload(&i32_bytes(&bounds));
    let active_b = gpu.upload(&i32_bytes(&active));
    let map_b = gpu.upload(&map.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    let cols = ids.len();
    let og = gpu.upload(&vec![0u8; cols * inter * 4]);
    let ou = gpu.upload(&vec![0u8; cols * inter * 4]);
    let args = [
        KernelArg::Buffer(og.view()),
        KernelArg::Buffer(ou.view()),
        KernelArg::Buffer(gw.view()),
        KernelArg::Buffer(gs.view()),
        KernelArg::Buffer(gb.view()),
        KernelArg::Buffer(uw.view()),
        KernelArg::Buffer(us.view()),
        KernelArg::Buffer(ub.view()),
        KernelArg::Buffer(xs.view()),
        KernelArg::Buffer(sorted_b.view()),
        KernelArg::Buffer(bounds_b.view()),
        KernelArg::Buffer(active_b.view()),
        KernelArg::Buffer(map_b.view()),
        KernelArg::U32(hidden as u32),
        KernelArg::U32(inter as u32),
        KernelArg::U32(n_expert as u32),
        KernelArg::U32(n_used as u32),
    ];
    let tiles = if gemm {
        2 * inter.div_ceil(per.0)
    } else {
        inter.div_ceil(per.0)
    };
    let grid = Extent3::new(tiles as u32, cols.min(n_expert) as u32, 1);
    run(
        gpu,
        KernelRef::new("q4e_moe", gate_up_entry),
        &LaunchArgs::new(grid, Extent3::linear(block), &args),
    );
    drain(gpu);
    let got_gate = gpu.read_f32(&og);
    let got_up = gpu.read_f32(&ou);
    let mut mid = vec![0f32; cols * inter];
    for (col, &e) in ids.iter().enumerate() {
        let slot = map[e as usize] as usize;
        let xr = &x[(col / n_used) * hidden..(col / n_used + 1) * hidden];
        let g = gate.project(slot, xr);
        let u = up.project(slot, xr);
        assert_close("gate", &got_gate[col * inter..(col + 1) * inter], &g, 2e-4);
        assert_close("up", &got_up[col * inter..(col + 1) * inter], &u, 2e-4);
        for i in 0..inter {
            mid[col * inter + i] = g[i] / (1.0 + (-g[i]).exp()) * u[i];
        }
    }
    let mid_b = gpu.upload(&f32_bytes(&mid));
    let od = gpu.upload(&vec![0u8; cols * hidden * 4]);
    let args = [
        KernelArg::Buffer(od.view()),
        KernelArg::Buffer(dw.view()),
        KernelArg::Buffer(ds.view()),
        KernelArg::Buffer(db.view()),
        KernelArg::Buffer(mid_b.view()),
        KernelArg::Buffer(sorted_b.view()),
        KernelArg::Buffer(bounds_b.view()),
        KernelArg::Buffer(active_b.view()),
        KernelArg::Buffer(map_b.view()),
        KernelArg::U32(inter as u32),
        KernelArg::U32(hidden as u32),
        KernelArg::U32(n_expert as u32),
        KernelArg::U32(n_used as u32),
    ];
    let grid = Extent3::new(hidden.div_ceil(per.1) as u32, cols.min(n_expert) as u32, 1);
    run(
        gpu,
        KernelRef::new("q4e_moe", down_entry),
        &LaunchArgs::new(grid, Extent3::linear(block), &args),
    );
    drain(gpu);
    let got_down = gpu.read_f32(&od);
    for (col, &e) in ids.iter().enumerate() {
        let slot = map[e as usize] as usize;
        let want = down.project(slot, &mid[col * inter..(col + 1) * inter]);
        assert_close(
            "down",
            &got_down[col * hidden..(col + 1) * hidden],
            &want,
            2e-4,
        );
    }
    (got_gate, got_up, got_down)
}

/// The prompt kernels of both routed stages match the host, loop over an
/// expert whose columns fill more than one tile, and give each column the same
/// bits whichever columns share its pass.
pub fn the_prompt_expert_stages_match_the_host_at_every_width(gpu: &Gpu) {
    let n_expert = 8usize;
    // Forty tokens, three distinct experts each, expert 2 in every one: forty
    // columns for expert 2, two tiles.
    let others = [0i32, 1, 3, 4, 5, 6, 7];
    let ids: Vec<i32> = (0..40usize)
        .flat_map(|t| {
            let a = others[t % 7];
            let b = others[(t * 3 + 1) % 7];
            let b = if b == a { others[(t % 7 + 1) % 7] } else { b };
            [2, a, b]
        })
        .collect();
    let map: Vec<u32> = (0..n_expert as u32).map(|e| (e * 5 + 1) % 6).collect();
    let (g, u, d) = expert_case(gpu, 640, &ids, n_expert, &map, true);
    let narrow = &ids[..5 * 3];
    let (g5, u5, d5) = expert_case(gpu, 640, narrow, n_expert, &map, true);
    assert!(g5 == g[..g5.len()], "gate moved when the pass narrowed");
    assert!(u5 == u[..u5.len()], "up moved when the pass narrowed");
    assert!(d5 == d[..d5.len()], "down moved when the pass narrowed");
}

/// The n-gram rows decode from the concatenated affine row format into the
/// head-slowest placement.
pub fn the_ngram_rows_decode_from_the_affine_row(gpu: &Gpu) {
    let (head_dim, n_heads, n_tok) = (160usize, 4usize, 3usize);
    let n_rows = n_heads * n_tok;
    let bank = AffineBank::new(1, n_rows, head_dim, 32, 41);
    let groups = head_dim / 32;
    let row_bytes = head_dim / 2 + groups * 4;
    let mut rows = Vec::with_capacity(n_rows * row_bytes);
    for r in 0..n_rows {
        let words = &bank.words[r * head_dim / 8..(r + 1) * head_dim / 8];
        rows.extend(words.iter().flat_map(|w| w.to_le_bytes()));
        rows.extend(
            bank.scales[r * groups..(r + 1) * groups]
                .iter()
                .flat_map(|w| w.to_le_bytes()),
        );
        rows.extend(
            bank.biases[r * groups..(r + 1) * groups]
                .iter()
                .flat_map(|w| w.to_le_bytes()),
        );
    }
    let rows_b = gpu.upload(&rows);
    let out = gpu.upload(&vec![0u8; n_rows * head_dim * 4]);
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(rows_b.view()),
        KernelArg::U32(head_dim as u32),
        KernelArg::U32(n_heads as u32),
        KernelArg::U32(row_bytes as u32),
        KernelArg::U32(n_rows as u32),
    ];
    run(
        gpu,
        KernelRef::new("q4e_ple", "q4e_ple_dequant_affine_kernel"),
        &LaunchArgs::new(
            Extent3::linear(n_rows as u32),
            Extent3::linear(head_dim as u32),
            &args,
        ),
    );
    drain(gpu);
    let got = gpu.read_f32(&out);
    for r in 0..n_rows {
        let (t, h) = (r / n_heads, r % n_heads);
        for i in 0..head_dim {
            let g = got[(t * n_heads + h) * head_dim + i];
            let w = bank.at(0, r, i);
            assert!(g == w, "row {r} element {i}: {g} vs {w}");
        }
    }
}
