//! The sparse family's FP8 entries: the block-scaled routed expert stages, the
//! FP8 n-gram row decode and the brain-float vocabulary gather, graded against
//! a host restatement of the same decode.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::missing_panics_doc,
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "fixture dimensions and ids are small literals; every case asserts by panicking; \
              the decode and the gather are exact; a case running both expert stages reads \
              best as one sequence"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;
use shiso_core::{Extent3, KernelArg, KernelRef, LaunchArgs};

/// One E4M3 byte as a float.
#[must_use]
pub fn e4m3(b: u8) -> f32 {
    let e = u32::from((b >> 3) & 0xF);
    let m = u32::from(b & 7);
    let v = if e == 0 {
        m as f32 * 2f32.powi(-9)
    } else if e == 15 && m == 7 {
        f32::NAN
    } else {
        (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7)
    };
    if b & 0x80 != 0 { -v } else { v }
}

/// A float rounded to the nearest brain float, as a bit pattern.
fn bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    ((bits + ((bits >> 16) & 1) + 0x7FFF) >> 16) as u16
}

/// A brain-float bit pattern widened exactly.
fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// A block-scaled bank of `slots` matrices `[out, in]`.
struct Bank {
    codes: Vec<u8>,
    scales: Vec<u16>,
    in_dim: usize,
    out_dim: usize,
}

impl Bank {
    fn new(slots: usize, out_dim: usize, in_dim: usize, seed: u32) -> Self {
        let codes = signal(slots * out_dim * in_dim, seed)
            .iter()
            .map(|v| {
                // Avoid the NaN pattern: a real exporter never writes it.
                let b = ((v.abs() * 997.0) as u32 & 0xFF) as u8;
                if b & 0x7F == 0x7F { b & 0xFE } else { b }
            })
            .collect();
        let blocks = slots * out_dim.div_ceil(128) * in_dim.div_ceil(128);
        let scales = signal(blocks, seed ^ 0x33)
            .iter()
            .map(|v| bf16_bits(0.001 + v.abs() * 0.004))
            .collect();
        Self {
            codes,
            scales,
            in_dim,
            out_dim,
        }
    }

    fn at(&self, slot: usize, o: usize, k: usize) -> f32 {
        let row = slot * self.out_dim + o;
        let blocks = self.in_dim.div_ceil(128);
        let brow = slot * self.out_dim.div_ceil(128) + o / 128;
        e4m3(self.codes[row * self.in_dim + k]) * bf16(self.scales[brow * blocks + k / 128])
    }

    fn project(&self, slot: usize, x: &[f32]) -> Vec<f32> {
        (0..self.out_dim)
            .map(|o| {
                (0..self.in_dim)
                    .map(|k| f64::from(self.at(slot, o, k)) * f64::from(x[k]))
                    .sum::<f64>() as f32
            })
            .collect()
    }
}

fn close(what: &str, got: &[f32], want: &[f32]) {
    let mean = want.iter().map(|v| v.abs()).sum::<f32>() / want.len() as f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = 2e-4 * w.abs().max(mean).max(1e-6);
        assert!((g - w).abs() <= tol, "{what}: {i}: {g} vs {w}");
    }
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Both FP8 expert stages over a sorted column list through a slot map that is
/// not the identity.
pub fn the_fp8_expert_stages_follow_the_slot_map(gpu: &Gpu) {
    let (hidden, inter, n_expert, n_used, slots) = (2560usize, 640usize, 16usize, 3usize, 6usize);
    let gate = Bank::new(slots, inter, hidden, 51);
    let up = Bank::new(slots, inter, hidden, 52);
    let down = Bank::new(slots, hidden, inter, 53);
    let ids: Vec<i32> = vec![4, 9, 1, 9, 13, 4];
    let n_tok = ids.len() / n_used;
    let mut map = vec![u32::MAX; n_expert];
    for (e, slot) in [(1usize, 5u32), (4, 0), (9, 3), (13, 2)] {
        map[e] = slot;
    }
    let mut bounds = vec![0i32; n_expert + 1];
    for &e in &ids {
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
    let x = signal(n_tok * hidden, 61);
    let up_b = |bank: &Bank| {
        (
            gpu.upload(&bank.codes),
            gpu.upload(&u16_bytes(&bank.scales)),
        )
    };
    let (gw, gs) = up_b(&gate);
    let (uw, us) = up_b(&up);
    let (dw, ds) = up_b(&down);
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
        KernelArg::Buffer(uw.view()),
        KernelArg::Buffer(us.view()),
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
    run(
        gpu,
        KernelRef::new("q4e_moe", "q4e_fp8_experts_gate_up_kernel"),
        &LaunchArgs::new(
            Extent3::new(inter.div_ceil(8) as u32, cols.min(n_expert) as u32, 1),
            Extent3::linear(256),
            &args,
        ),
    );
    drain(gpu);
    let got_g = gpu.read_f32(&og);
    let got_u = gpu.read_f32(&ou);
    let mut mid = vec![0f32; cols * inter];
    for (col, &e) in ids.iter().enumerate() {
        let slot = map[e as usize] as usize;
        let xr = &x[(col / n_used) * hidden..(col / n_used + 1) * hidden];
        let g = gate.project(slot, xr);
        let u = up.project(slot, xr);
        close("gate", &got_g[col * inter..(col + 1) * inter], &g);
        close("up", &got_u[col * inter..(col + 1) * inter], &u);
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
    run(
        gpu,
        KernelRef::new("q4e_moe", "q4e_fp8_experts_down_kernel"),
        &LaunchArgs::new(
            Extent3::new(hidden.div_ceil(8) as u32, cols.min(n_expert) as u32, 1),
            Extent3::linear(256),
            &args,
        ),
    );
    drain(gpu);
    let got_d = gpu.read_f32(&od);
    for (col, &e) in ids.iter().enumerate() {
        let want = down.project(
            map[e as usize] as usize,
            &mid[col * inter..(col + 1) * inter],
        );
        close("down", &got_d[col * hidden..(col + 1) * hidden], &want);
    }
}

/// FP8 n-gram rows decode under the table's one scale, head-slowest.
pub fn the_fp8_ngram_rows_decode_under_the_table_scale(gpu: &Gpu) {
    let (head_dim, n_heads, n_rows) = (160usize, 4usize, 12usize);
    let rows: Vec<u8> = (0..n_rows * head_dim)
        .map(|i| ((i * 37 + 11) % 255) as u8)
        .collect();
    let scale = 0.037f32;
    let rows_b = gpu.upload(&rows);
    let out = gpu.upload(&vec![0u8; n_rows * head_dim * 4]);
    let scale_b = gpu.upload(&scale.to_le_bytes());
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(rows_b.view()),
        KernelArg::U32(head_dim as u32),
        KernelArg::U32(n_heads as u32),
        KernelArg::U32(head_dim as u32),
        KernelArg::U32(n_rows as u32),
        KernelArg::Buffer(scale_b.view()),
    ];
    run(
        gpu,
        KernelRef::new("q4e_ple", "q4e_ple_dequant_fp8_kernel"),
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
            let want = e4m3(rows[r * head_dim + i]) * scale;
            let g = got[(t * n_heads + h) * head_dim + i];
            // The device's power of two is the fast one, a unit in the last
            // place off at worst.
            let ok = (g - want).abs() <= want.abs() * 1e-6 || (g.is_nan() && want.is_nan());
            assert!(ok, "row {r} {i}: {g} vs {want}");
        }
    }
}

/// The brain-float gather widens each token's row exactly and zeroes an id
/// outside the vocabulary.
pub fn the_brain_float_gather_widens_exactly(gpu: &Gpu) {
    let (vocab, embd) = (41usize, 2560usize);
    let table: Vec<u16> = signal(vocab * embd, 71)
        .iter()
        .map(|v| bf16_bits(*v))
        .collect();
    let tokens: Vec<i32> = vec![3, 40, -1, 41, 0];
    let tb = gpu.upload(&u16_bytes(&table));
    let ids = gpu.upload(&i32_bytes(&tokens));
    let n = tokens.len() * embd;
    let out = gpu.upload(&vec![0u8; n * 4]);
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(tb.view()),
        KernelArg::Buffer(ids.view()),
        KernelArg::U32(tokens.len() as u32),
        KernelArg::U32(embd as u32),
        KernelArg::U32(vocab as u32),
    ];
    run(
        gpu,
        KernelRef::new("q4e_embed", "embed_tokens_bf16_kernel"),
        &LaunchArgs::new(
            Extent3::linear(n.div_ceil(256) as u32),
            Extent3::linear(256),
            &args,
        ),
    );
    drain(gpu);
    let got = gpu.read_f32(&out);
    for (t, &tok) in tokens.iter().enumerate() {
        for d in 0..embd {
            let want = if tok < 0 || tok as usize >= vocab {
                0.0
            } else {
                bf16(table[tok as usize * embd + d])
            };
            assert!(got[t * embd + d] == want, "token {t} element {d}");
        }
    }
}
