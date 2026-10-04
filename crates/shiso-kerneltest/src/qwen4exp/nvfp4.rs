//! The sparse family's NVFP4 routed expert stages, graded against a host
//! restatement of the decode: four-bit codes, an `e4m3` scale per sixteen
//! weights, and a per-slot `f32` above both.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    reason = "fixture dimensions and ids are small literals; every case asserts by panicking; \
              a case running both expert stages reads best as one sequence"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;
use crate::qwen4exp::fp8::e4m3;
use shiso_core::{Extent3, KernelArg, KernelRef, LaunchArgs};

/// The module the entries live in.
const MODULE: &str = "q4e_modelopt";

/// One four-bit code as a float.
fn e2m1(code: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[usize::from(code & 7)];
    if code & 8 != 0 { -v } else { v }
}

/// An NVFP4 bank of `slots` matrices `[out, in]`.
struct Bank {
    codes: Vec<u8>,
    scales: Vec<u8>,
    globals: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl Bank {
    fn new(slots: usize, out_dim: usize, in_dim: usize, seed: u32) -> Self {
        let codes = signal(slots * out_dim * in_dim / 2, seed)
            .iter()
            .map(|v| ((v.abs() * 9973.0) as u32 & 0xFF) as u8)
            .collect();
        // Every scale byte below the NaN pattern, the subnormal ones included:
        // the device decode has to survive a flushed subnormal, and these are
        // the bytes that would show it.
        let scales = signal(slots * out_dim * in_dim / 16, seed ^ 0x55)
            .iter()
            .map(|v| ((v.abs() * 7919.0) as u32 % 0x7F) as u8)
            .collect();
        let globals = (0..slots)
            .map(|s| 1.0e-4 * (1.0 + s as f32 * 0.37))
            .collect();
        Self {
            codes,
            scales,
            globals,
            in_dim,
            out_dim,
        }
    }

    fn at(&self, slot: usize, o: usize, k: usize) -> f32 {
        let row = slot * self.out_dim + o;
        let byte = self.codes[(row * self.in_dim + k) / 2];
        let code = if k.is_multiple_of(2) {
            byte & 0xF
        } else {
            byte >> 4
        };
        let scale = e4m3(self.scales[(row * self.in_dim + k) / 16]) * self.globals[slot];
        e2m1(code) * scale
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

fn close(what: &str, got: &[f32], want: &[f32], rel: f32) {
    let mean = want.iter().map(|v| v.abs()).sum::<f32>() / want.len() as f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = rel * w.abs().max(mean).max(1e-12);
        assert!((g - w).abs() <= tol, "{what}: {i}: {g} vs {w}");
    }
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// One stage set: entries, rows a block, threads a block, and the tolerance
/// against the host product.
struct Stages {
    gate_up: &'static str,
    down: &'static str,
    rows: usize,
    threads: u32,
    /// Copies of the gate's row tiles the gate-and-up grid holds: the step
    /// kernel covers gate and up in one block, the prompt kernel tiles them
    /// one after the other.
    copies: usize,
    rel: f32,
}

/// Both NVFP4 expert stages over a sorted column list through a slot map that
/// is not the identity, at this family's widths: the warp-per-row step
/// kernels.
pub fn the_nvfp4_expert_stages_follow_the_slot_map(gpu: &Gpu) {
    stages(
        gpu,
        &Stages {
            gate_up: "q4e_nvfp4_experts_gate_up_kernel",
            down: "q4e_nvfp4_experts_down_kernel",
            rows: 8,
            threads: 256,
            copies: 1,
            rel: 2e-4,
        },
    );
}

/// The same, through the tensor-core prompt kernels. Their weights decode
/// exactly but their activations are rounded to halves, which is the whole of
/// the wider budget: eleven bits of mantissa on every input of a 2,560-term
/// sum.
pub fn the_nvfp4_prompt_stages_follow_the_slot_map(gpu: &Gpu) {
    stages(
        gpu,
        &Stages {
            gate_up: "q4e_nvfp4_experts_gemm_gate_up_kernel",
            down: "q4e_nvfp4_experts_gemm_down_kernel",
            rows: 64,
            threads: 128,
            copies: 2,
            rel: 4e-3,
        },
    );
}

fn stages(gpu: &Gpu, st: &Stages) {
    let (hidden, inter, n_expert, n_used, slots) = (2560usize, 640usize, 16usize, 3usize, 6usize);
    let gate = Bank::new(slots, inter, hidden, 71);
    let up = Bank::new(slots, inter, hidden, 72);
    let down = Bank::new(slots, hidden, inter, 73);
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
    let x = signal(n_tok * hidden, 81);
    let up_b = |bank: &Bank| {
        (
            gpu.upload(&bank.codes),
            gpu.upload(&bank.scales),
            gpu.upload(&f32_bytes(&bank.globals)),
        )
    };
    let (gw, gs, gg) = up_b(&gate);
    let (uw, us, ug) = up_b(&up);
    let (dw, ds, dg) = up_b(&down);
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
        KernelArg::Buffer(gg.view()),
        KernelArg::Buffer(uw.view()),
        KernelArg::Buffer(us.view()),
        KernelArg::Buffer(ug.view()),
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
        KernelRef::new(MODULE, st.gate_up),
        &LaunchArgs::new(
            Extent3::new(
                (st.copies * inter.div_ceil(st.rows)) as u32,
                cols.min(n_expert) as u32,
                1,
            ),
            Extent3::linear(st.threads),
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
        close("gate", &got_g[col * inter..(col + 1) * inter], &g, st.rel);
        close("up", &got_u[col * inter..(col + 1) * inter], &u, st.rel);
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
        KernelArg::Buffer(dg.view()),
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
        KernelRef::new(MODULE, st.down),
        &LaunchArgs::new(
            Extent3::new(
                hidden.div_ceil(st.rows) as u32,
                cols.min(n_expert) as u32,
                1,
            ),
            Extent3::linear(st.threads),
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
        close(
            "down",
            &got_d[col * hidden..(col + 1) * hidden],
            &want,
            st.rel,
        );
    }
}
