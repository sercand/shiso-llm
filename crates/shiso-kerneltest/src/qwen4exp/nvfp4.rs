//! The sparse family's NVFP4 routed expert stages, graded against a host
//! restatement of the decode: four-bit codes, an `e4m3` scale per sixteen
//! weights, and a per-slot `f32` above both.
//!
//! Beside the grading, two invariance cases hold the property speculative
//! decoding and chunked prompts lean on - a column's bits depend on its own
//! token and expert, never on what else shares the launch - and the bench
//! fixtures time the four entries under a skewed, prompt-like routing.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "fixture dimensions and ids are small literals"
)]

use crate::bench::Fixture;
use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;
use crate::qwen4exp::fp8::e4m3;
use shiso_core::{DeviceBuffer, DeviceView, Extent3, KernelArg, KernelRef, LaunchArgs, alloc};

/// The module the entries live in.
const MODULE: &str = "q4e_modelopt";

/// One four-bit code as a float.
pub(crate) fn e2m1(code: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[usize::from(code & 7)];
    if code & 8 != 0 { -v } else { v }
}

/// An NVFP4 bank of `slots` matrices `[out, in]`.
pub(crate) struct Bank {
    pub(crate) codes: Vec<u8>,
    pub(crate) scales: Vec<u8>,
    pub(crate) globals: Vec<f32>,
    pub(crate) in_dim: usize,
    pub(crate) out_dim: usize,
}

impl Bank {
    pub(crate) fn new(slots: usize, out_dim: usize, in_dim: usize, seed: u32) -> Self {
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

    pub(crate) fn at(&self, slot: usize, o: usize, k: usize) -> f32 {
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

    pub(crate) fn project(&self, slot: usize, x: &[f32]) -> Vec<f32> {
        (0..self.out_dim)
            .map(|o| {
                (0..self.in_dim)
                    .map(|k| f64::from(self.at(slot, o, k)) * f64::from(x[k]))
                    .sum::<f64>() as f32
            })
            .collect()
    }

    pub(crate) fn upload(&self, gpu: &Gpu) -> [DeviceBuffer; 3] {
        [
            gpu.upload(&self.codes),
            gpu.upload(&self.scales),
            gpu.upload(&f32_bytes(&self.globals)),
        ]
    }
}

fn close(what: &str, got: &[f32], want: &[f32], rel: f32) {
    let mean = want.iter().map(|v| v.abs()).sum::<f32>() / want.len() as f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = rel * w.abs().max(mean).max(1e-12);
        assert!((g - w).abs() <= tol, "{what}: {i}: {g} vs {w}");
    }
}

pub(crate) fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A routing's column lists as the engine's router leaves them: columns
/// grouped by expert in column order (`sorted`), each expert's span
/// (`bounds`), and the experts with a nonempty span followed by their count
/// at index `n_expert` (`active`).
pub(crate) struct Route {
    pub(crate) sorted: Vec<i32>,
    pub(crate) bounds: Vec<i32>,
    pub(crate) active: Vec<i32>,
}

impl Route {
    /// `ids[col]` is column `col`'s expert; column `col` is token
    /// `col / n_used`'s.
    pub(crate) fn new(ids: &[i32], n_expert: usize) -> Self {
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
        Self {
            sorted,
            bounds,
            active,
        }
    }

    /// Distinct experts the routing touches.
    pub(crate) fn distinct(&self) -> usize {
        *self.active.last().expect("the count closes the list") as usize
    }

    pub(crate) fn upload(&self, gpu: &Gpu, map: &[u32], columns: usize) -> DevRoute {
        DevRoute {
            sorted: gpu.upload(&i32_bytes(&self.sorted)),
            bounds: gpu.upload(&i32_bytes(&self.bounds)),
            active: gpu.upload(&i32_bytes(&self.active)),
            map: gpu.upload(&map.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()),
            columns,
        }
    }
}

/// A [`Route`] and the expert-to-slot map on the device.
pub(crate) struct DevRoute {
    pub(crate) sorted: DeviceBuffer,
    pub(crate) bounds: DeviceBuffer,
    pub(crate) active: DeviceBuffer,
    pub(crate) map: DeviceBuffer,
    pub(crate) columns: usize,
}

/// One stage set: entries, launch geometry, and the tolerance against the
/// host product.
#[derive(Clone, Copy)]
pub(crate) struct Stages {
    gate_up: &'static str,
    down: &'static str,
    /// The gate-and-up grid's first axis at an intermediate width.
    gate_up_blocks: fn(u32) -> u32,
    /// The down grid's first axis at a hidden width.
    down_blocks: fn(u32) -> u32,
    threads: u32,
    /// Dynamic shared memory a block takes.
    smem: u32,
    rel: f32,
    /// Whether the gate-and-up entry writes the activated intermediate (one
    /// output) rather than the two rows.
    act: bool,
}

/// The warp-per-row step kernels: one row of gate and up a warp, four of
/// down, four warps a block (`Q4E_NVFP4_GU_ROWS`, `Q4E_NVFP4_DN_ROWS`,
/// `Q4E_NVFP4_WARPS`).
const STEP: Stages = Stages {
    gate_up: "q4e_nvfp4_experts_gate_up_kernel",
    down: "q4e_nvfp4_experts_down_kernel",
    gate_up_blocks: |out_dim| out_dim.div_ceil(4),
    down_blocks: |out_dim| out_dim.div_ceil(16),
    threads: 128,
    smem: 0,
    rel: 2e-4,
    act: false,
};

/// The step stages with the activation fused into gate and up.
const STEP_ACT: Stages = Stages {
    gate_up: "q4e_nvfp4_experts_gate_up_act_kernel",
    act: true,
    ..STEP
};

/// The activated stages holding a 2560-wide row in registers, a verify's.
const STEP_ACT_HELD: Stages = Stages {
    gate_up: "q4e_nvfp4_experts_gate_up_act_held_kernel",
    act: true,
    ..STEP
};

/// The tensor-core prompt kernels: 128 rows of gate and up, or 256 of down, a
/// block, each row tile over four blocks, sixteen warps and `Q4E_MG_SMEM`
/// bytes of dynamic shared memory a block (`Q4E_MG_HALF`, `Q4E_MG_BM`,
/// `Q4E_MG_SPLIT`, `Q4E_MG_THREADS`). Their weights decode exactly but their activations are
/// rounded to halves, which is the whole of the wider budget: eleven bits of
/// mantissa on every input of a 2,560-term sum.
pub(crate) const PROMPT: Stages = Stages {
    gate_up: "q4e_nvfp4_experts_gemm_gate_up_kernel",
    down: "q4e_nvfp4_experts_gemm_down_kernel",
    gate_up_blocks: |out_dim| 4 * out_dim.div_ceil(128),
    down_blocks: |out_dim| 4 * out_dim.div_ceil(256),
    threads: 512,
    smem: 92_416,
    rel: 4e-3,
    act: false,
};

/// The three device arrays of one bank.
pub(crate) type Parts<'a> = [DeviceView<'a>; 3];

/// One gate-and-up launch.
pub(crate) struct GateUp<'a> {
    pub(crate) out_gate: DeviceView<'a>,
    pub(crate) out_up: DeviceView<'a>,
    pub(crate) gate: Parts<'a>,
    pub(crate) up: Parts<'a>,
    pub(crate) x: DeviceView<'a>,
    pub(crate) route: &'a DevRoute,
    pub(crate) hidden: u32,
    pub(crate) inter: u32,
    pub(crate) n_expert: u32,
    pub(crate) n_used: u32,
}

impl GateUp<'_> {
    pub(crate) fn launch(&self, gpu: &Gpu, st: &Stages) {
        let r = self.route;
        let mut args = vec![
            KernelArg::Buffer(self.out_gate),
            KernelArg::Buffer(self.out_up),
            KernelArg::Buffer(self.gate[0]),
            KernelArg::Buffer(self.gate[1]),
            KernelArg::Buffer(self.gate[2]),
            KernelArg::Buffer(self.up[0]),
            KernelArg::Buffer(self.up[1]),
            KernelArg::Buffer(self.up[2]),
            KernelArg::Buffer(self.x),
            KernelArg::Buffer(r.sorted.view()),
            KernelArg::Buffer(r.bounds.view()),
            KernelArg::Buffer(r.active.view()),
            KernelArg::Buffer(r.map.view()),
            KernelArg::U32(self.hidden),
            KernelArg::U32(self.inter),
            KernelArg::U32(self.n_expert),
            KernelArg::U32(self.n_used),
        ];
        if st.act {
            // One output: the activated intermediate.
            args.remove(1);
        }
        let mut launch = LaunchArgs::new(
            Extent3::new(
                (st.gate_up_blocks)(self.inter),
                (r.columns as u32).min(self.n_expert),
                1,
            ),
            Extent3::linear(st.threads),
            &args,
        );
        launch.smem_bytes = st.smem;
        run(gpu, KernelRef::new(MODULE, st.gate_up), &launch);
    }
}

/// One down launch.
pub(crate) struct Down<'a> {
    pub(crate) out: DeviceView<'a>,
    pub(crate) weights: Parts<'a>,
    pub(crate) mid: DeviceView<'a>,
    pub(crate) route: &'a DevRoute,
    pub(crate) hidden: u32,
    pub(crate) inter: u32,
    pub(crate) n_expert: u32,
    pub(crate) n_used: u32,
}

impl Down<'_> {
    pub(crate) fn launch(&self, gpu: &Gpu, st: &Stages) {
        let r = self.route;
        let args = [
            KernelArg::Buffer(self.out),
            KernelArg::Buffer(self.weights[0]),
            KernelArg::Buffer(self.weights[1]),
            KernelArg::Buffer(self.weights[2]),
            KernelArg::Buffer(self.mid),
            KernelArg::Buffer(r.sorted.view()),
            KernelArg::Buffer(r.bounds.view()),
            KernelArg::Buffer(r.active.view()),
            KernelArg::Buffer(r.map.view()),
            KernelArg::U32(self.inter),
            KernelArg::U32(self.hidden),
            KernelArg::U32(self.n_expert),
            KernelArg::U32(self.n_used),
        ];
        let mut launch = LaunchArgs::new(
            Extent3::new(
                (st.down_blocks)(self.hidden),
                (r.columns as u32).min(self.n_expert),
                1,
            ),
            Extent3::linear(st.threads),
            &args,
        );
        launch.smem_bytes = st.smem;
        run(gpu, KernelRef::new(MODULE, st.down), &launch);
    }
}

pub(crate) fn views(bufs: &[DeviceBuffer; 3]) -> Parts<'_> {
    [bufs[0].view(), bufs[1].view(), bufs[2].view()]
}

/// Both NVFP4 expert stages over a sorted column list through a slot map that
/// is not the identity, at this family's widths: the warp-per-row step
/// kernels.
pub fn the_nvfp4_expert_stages_follow_the_slot_map(gpu: &Gpu) {
    stages(gpu, &STEP);
}

/// The same, through the tensor-core prompt kernels.
pub fn the_nvfp4_prompt_stages_follow_the_slot_map(gpu: &Gpu) {
    stages(gpu, &PROMPT);
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
    let x = signal(n_tok * hidden, 81);
    let (gb, ub, db) = (gate.upload(gpu), up.upload(gpu), down.upload(gpu));
    let xs = gpu.upload(&f32_bytes(&x));
    let cols = ids.len();
    let route = Route::new(&ids, n_expert).upload(gpu, &map, cols);
    let og = gpu.upload(&vec![0u8; cols * inter * 4]);
    let ou = gpu.upload(&vec![0u8; cols * inter * 4]);
    GateUp {
        out_gate: og.view(),
        out_up: ou.view(),
        gate: views(&gb),
        up: views(&ub),
        x: xs.view(),
        route: &route,
        hidden: hidden as u32,
        inter: inter as u32,
        n_expert: n_expert as u32,
        n_used: n_used as u32,
    }
    .launch(gpu, st);
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
    Down {
        out: od.view(),
        weights: views(&db),
        mid: mid_b.view(),
        route: &route,
        hidden: hidden as u32,
        inter: inter as u32,
        n_expert: n_expert as u32,
        n_used: n_used as u32,
    }
    .launch(gpu, st);
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

/// Runs both stages of `st` over one routing and returns every column's
/// gate, up and down rows, `[columns, width]` each. `mid` is the down
/// stage's input, `[columns, inter]`.
fn outputs(
    gpu: &Gpu,
    st: &Stages,
    banks: &[[DeviceBuffer; 3]; 3],
    case: &Case<'_>,
) -> [Vec<f32>; 3] {
    let (hidden, inter) = (case.hidden, case.inter);
    let cols = case.ids.len();
    let route = Route::new(case.ids, case.n_expert).upload(gpu, case.map, cols);
    let xs = gpu.upload(&f32_bytes(case.x));
    let mid = gpu.upload(&f32_bytes(case.mid));
    let og = gpu.upload(&vec![0u8; cols * inter * 4]);
    let ou = gpu.upload(&vec![0u8; cols * inter * 4]);
    let od = gpu.upload(&vec![0u8; cols * hidden * 4]);
    GateUp {
        out_gate: og.view(),
        out_up: ou.view(),
        gate: views(&banks[0]),
        up: views(&banks[1]),
        x: xs.view(),
        route: &route,
        hidden: hidden as u32,
        inter: inter as u32,
        n_expert: case.n_expert as u32,
        n_used: case.n_used as u32,
    }
    .launch(gpu, st);
    Down {
        out: od.view(),
        weights: views(&banks[2]),
        mid: mid.view(),
        route: &route,
        hidden: hidden as u32,
        inter: inter as u32,
        n_expert: case.n_expert as u32,
        n_used: case.n_used as u32,
    }
    .launch(gpu, st);
    drain(gpu);
    [gpu.read_f32(&og), gpu.read_f32(&ou), gpu.read_f32(&od)]
}

/// One routing of an invariance case.
struct Case<'a> {
    ids: &'a [i32],
    map: &'a [u32],
    x: &'a [f32],
    mid: &'a [f32],
    hidden: usize,
    inter: usize,
    n_expert: usize,
    n_used: usize,
}

/// A row of `st` computed alone and inside a wider pass must be the same
/// bits: the first token's columns, routed alone, and again as the first of
/// `n_tok` tokens whose other columns land on the same experts (and on
/// others), so a stage whose sums depended on how many columns share an
/// expert, or on where a column sits in its expert's span, shows here.
fn invariant(gpu: &Gpu, st: &Stages, n_tok: usize) {
    let (hidden, inter, n_expert, n_used, slots) = (2560usize, 640usize, 32usize, 4usize, 8usize);
    let banks = [
        Bank::new(slots, inter, hidden, 171).upload(gpu),
        Bank::new(slots, inter, hidden, 172).upload(gpu),
        Bank::new(slots, hidden, inter, 173).upload(gpu),
    ];
    let map: Vec<u32> = (0..n_expert as u32)
        .map(|e| (e * 5 + 3) % slots as u32)
        .collect();
    // Token 0 takes experts 3, 7, 11, 30; every later token shares two of them
    // and adds two of its own, so the shared experts' spans grow with the pass
    // and token 0's column is not always first in them.
    let mut ids = vec![3i32, 7, 11, 30];
    for t in 1..n_tok as i32 {
        ids.extend([(t * 13) % 32, 7, (t * 7 + 1) % 32, 3]);
    }
    // Duplicates inside a token are not a routing the router produces.
    for t in 1..n_tok {
        let row = &mut ids[t * n_used..(t + 1) * n_used];
        for i in 0..n_used {
            while row[..i].contains(&row[i]) {
                row[i] = (row[i] + 1) % 32;
            }
        }
    }
    let cols = ids.len();
    let x = signal(n_tok * hidden, 181);
    let mid = signal(cols * inter, 182);
    let wide = outputs(
        gpu,
        st,
        &banks,
        &Case {
            ids: &ids,
            map: &map,
            x: &x,
            mid: &mid,
            hidden,
            inter,
            n_expert,
            n_used,
        },
    );
    let alone = outputs(
        gpu,
        st,
        &banks,
        &Case {
            ids: &ids[..n_used],
            map: &map,
            x: &x[..hidden],
            mid: &mid[..n_used * inter],
            hidden,
            inter,
            n_expert,
            n_used,
        },
    );
    for (what, (w, a), width) in [
        ("gate", (&wide[0], &alone[0]), inter),
        ("up", (&wide[1], &alone[1]), inter),
        ("down", (&wide[2], &alone[2]), hidden),
    ] {
        for (i, (p, q)) in w[..n_used * width].iter().zip(a).enumerate() {
            assert_eq!(
                p.to_bits(),
                q.to_bits(),
                "{}: {what} {i}: {p} in a pass of {n_tok} tokens vs {q} alone",
                st.gate_up
            );
        }
    }
}

/// A verify row equals the single-step row bit for bit, through the step
/// kernels: speculative decoding accepts a drafted token only if the target
/// would have produced it alone.
pub fn the_nvfp4_step_rows_do_not_depend_on_the_pass(gpu: &Gpu) {
    for n_tok in [2, 5, 8] {
        invariant(gpu, &STEP, n_tok);
    }
}

/// A prompt column equals itself computed in a narrower chunk, bit for bit,
/// through the prompt kernels, at widths that fill one tile of an expert and
/// spill into several.
pub fn the_nvfp4_prompt_columns_do_not_depend_on_the_chunk(gpu: &Gpu) {
    for n_tok in [3, 40, 200] {
        invariant(gpu, &PROMPT, n_tok);
    }
}

impl Bank {
    /// Row `o` of slot `slot` against `x` in the step kernels' exact order:
    /// lane `l` takes groups `l, l + 32, ...`; a group is its two eight-code
    /// words, each a fused multiply-add chain from zero over its codes in
    /// order, the two added, then fused into the lane's sum by the group's
    /// `e4m3(scale) * global`; the lanes fold by the xor butterfly at 16, 8,
    /// 4, 2 and 1, and lane 0's value is the row.
    fn lane_order(&self, slot: usize, o: usize, x: &[f32]) -> f32 {
        let groups = self.in_dim / 16;
        let row = slot * self.out_dim + o;
        let mut lanes = [0f32; 32];
        for (l, lane) in lanes.iter_mut().enumerate() {
            let mut acc = 0f32;
            for g in (l..groups).step_by(32) {
                let word = |half: usize| {
                    let at = row * self.in_dim / 2 + g * 8 + half * 4;
                    let mut s = 0f32;
                    for j in 0..8 {
                        let byte = self.codes[at + j / 2];
                        let code = if j % 2 == 0 { byte & 0xF } else { byte >> 4 };
                        s = e2m1(code).mul_add(x[g * 16 + half * 8 + j], s);
                    }
                    s
                };
                let part = word(0) + word(1);
                let scale = e4m3(self.scales[row * groups + g]) * self.globals[slot];
                acc = part.mul_add(scale, acc);
            }
            *lane = acc;
        }
        for offset in [16, 8, 4, 2, 1] {
            let before = lanes;
            for (l, lane) in lanes.iter_mut().enumerate() {
                *lane = before[l] + before[l ^ offset];
            }
        }
        lanes[0]
    }
}

/// Asserts `got` holds `want`'s bits, naming the first element that differs.
fn same_bits(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: element count");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "{what}: element {i} is {g:e}, expected {w:e}"
        );
    }
}

/// The step stages' rows are, bit for bit, the host restatement of their
/// order ([`Bank::lane_order`]), at this family's widths over a routing that
/// shares experts between tokens: the contract that lets a kernel change its
/// loads, its blocking or its fusion and prove it moved no bit.
pub fn the_nvfp4_step_rows_are_the_lane_order_bit_for_bit(gpu: &Gpu) {
    let (hidden, inter, n_expert, n_used, slots) = (2560usize, 640usize, 32usize, 4usize, 8usize);
    let host = [
        Bank::new(slots, inter, hidden, 271),
        Bank::new(slots, inter, hidden, 272),
        Bank::new(slots, hidden, inter, 273),
    ];
    let banks = [
        host[0].upload(gpu),
        host[1].upload(gpu),
        host[2].upload(gpu),
    ];
    let map: Vec<u32> = (0..n_expert as u32)
        .map(|e| (e * 3 + 1) % slots as u32)
        .collect();
    let ids: Vec<i32> = vec![3, 7, 11, 30, 7, 2, 3, 19, 30, 7, 5, 12];
    let n_tok = ids.len() / n_used;
    let x = signal(n_tok * hidden, 281);
    let mid = signal(ids.len() * inter, 282);
    for st in [&STEP] {
        let [g, u, d] = outputs(
            gpu,
            st,
            &banks,
            &Case {
                ids: &ids,
                map: &map,
                x: &x,
                mid: &mid,
                hidden,
                inter,
                n_expert,
                n_used,
            },
        );
        for (col, &e) in ids.iter().enumerate() {
            let slot = map[e as usize] as usize;
            let xr = &x[(col / n_used) * hidden..(col / n_used + 1) * hidden];
            let mr = &mid[col * inter..(col + 1) * inter];
            for (what, got, bank, input, width) in [
                ("gate", &g, &host[0], xr, inter),
                ("up", &u, &host[1], xr, inter),
                ("down", &d, &host[2], mr, hidden),
            ] {
                let want: Vec<f32> = (0..width)
                    .map(|o| bank.lane_order(slot, o, input))
                    .collect();
                same_bits(
                    &format!("{} {what} column {col}", st.gate_up),
                    &got[col * width..(col + 1) * width],
                    &want,
                );
            }
        }
    }
}

/// The activated gate-and-up entry writes, bit for bit, what the activation
/// launch (`q4e_swiglu_kernel`) writes from the two rows the plain entry
/// stores, over the same routing as
/// [`the_nvfp4_step_rows_are_the_lane_order_bit_for_bit`].
///
/// # Panics
///
/// When an output differs, or the device refuses a launch or a copy.
pub fn the_activated_gate_up_is_the_activation_of_its_rows(gpu: &Gpu) {
    let (hidden, inter, n_expert, n_used, slots) = (2560usize, 640usize, 32usize, 4usize, 8usize);
    let banks = [
        Bank::new(slots, inter, hidden, 371).upload(gpu),
        Bank::new(slots, inter, hidden, 372).upload(gpu),
    ];
    let map: Vec<u32> = (0..n_expert as u32)
        .map(|e| (e * 3 + 1) % slots as u32)
        .collect();
    let ids: Vec<i32> = vec![3, 7, 11, 30, 7, 2, 3, 19, 30, 7, 5, 12];
    let n_tok = ids.len() / n_used;
    let cols = ids.len();
    let x = gpu.upload(&f32_bytes(&signal(n_tok * hidden, 381)));
    let route = Route::new(&ids, n_expert).upload(gpu, &map, cols);
    let n = cols * inter;
    let og = gpu.upload(&vec![0u8; n * 4]);
    let ou = gpu.upload(&vec![0u8; n * 4]);
    let want = gpu.upload(&f32_bytes(&vec![f32::NAN; n]));
    let got = gpu.upload(&f32_bytes(&vec![f32::NAN; n]));
    let gate_up = GateUp {
        out_gate: og.view(),
        out_up: ou.view(),
        gate: views(&banks[0]),
        up: views(&banks[1]),
        x: x.view(),
        route: &route,
        hidden: hidden as u32,
        inter: inter as u32,
        n_expert: n_expert as u32,
        n_used: n_used as u32,
    };
    gate_up.launch(gpu, &STEP);
    run(
        gpu,
        KernelRef::new("q4e_moe", "q4e_swiglu_kernel"),
        &LaunchArgs::new(
            Extent3::new((n as u32).div_ceil(256), 1, 1),
            Extent3::linear(256),
            &[
                KernelArg::Buffer(want.view()),
                KernelArg::Buffer(og.view()),
                KernelArg::Buffer(ou.view()),
                KernelArg::U64(n as u64),
            ],
        ),
    );
    drain(gpu);
    let want = gpu.read_f32(&want);
    for st in [&STEP_ACT, &STEP_ACT_HELD] {
        gpu.device()
            .copy_h2d(got.view(), &f32_bytes(&vec![f32::NAN; n]))
            .expect("host-to-device copy");
        GateUp {
            out_gate: got.view(),
            ..gate_up
        }
        .launch(gpu, st);
        drain(gpu);
        same_bits(st.gate_up, &gpu.read_f32(&got), &want);
    }
}

/// One NVFP4 bank's three host arrays, every slot back to back, as a caller
/// puts them on the device.
#[derive(Debug, Clone, Copy)]
pub struct BankArrays<'a> {
    /// Packed four-bit codes, `slots * out_dim * in_dim / 2` bytes.
    pub codes: &'a [u8],
    /// `e4m3` group scales, `slots * out_dim * in_dim / 16` bytes.
    pub scales: &'a [u8],
    /// One `f32` global per slot, little-endian.
    pub globals: &'a [u8],
    /// Matrices in the bank.
    pub slots: u32,
    /// Contraction width of one matrix.
    pub in_dim: u32,
    /// Output width of one matrix.
    pub out_dim: u32,
}

/// Both NVFP4 step stages over banks the caller places on the device - the
/// engine's resident bank type, in the CUDA driver - through the identity
/// expert-to-slot map a resident bank is read with, at four experts of this
/// family's widths, against the host restatement of the decode.
///
/// `place` puts one bank on the device; `parts` hands back its codes, group
/// scales and globals, in that order. The case is the one
/// [`the_nvfp4_expert_stages_follow_the_slot_map`] runs over plain buffers,
/// so a resident bank whose slots were laid out other than a streamed slot
/// pool's would read a neighbouring expert's rows or its global here.
pub fn the_nvfp4_stages_read_a_placed_bank<B>(
    gpu: &Gpu,
    place: impl Fn(&Gpu, BankArrays<'_>) -> B,
    parts: impl for<'a> Fn(&'a B) -> [DeviceView<'a>; 3],
) {
    let (hidden, inter, n_expert, n_used) = (2560usize, 640usize, 4usize, 2usize);
    let gate = Bank::new(n_expert, inter, hidden, 91);
    let up = Bank::new(n_expert, inter, hidden, 92);
    let down = Bank::new(n_expert, hidden, inter, 93);
    let placed = |bank: &Bank| {
        place(
            gpu,
            BankArrays {
                codes: &bank.codes,
                scales: &bank.scales,
                globals: &f32_bytes(&bank.globals),
                slots: n_expert as u32,
                in_dim: bank.in_dim as u32,
                out_dim: bank.out_dim as u32,
            },
        )
    };
    let (gate_b, up_b, down_b) = (placed(&gate), placed(&up), placed(&down));
    // Three tokens, two experts each, every expert used and the last slot
    // twice, so a stride error past slot 0 cannot hide.
    let ids: Vec<i32> = vec![3, 1, 0, 3, 2, 1];
    let n_tok = ids.len() / n_used;
    let map: Vec<u32> = (0..n_expert as u32).collect();
    let x = signal(n_tok * hidden, 97);
    let xs = gpu.upload(&f32_bytes(&x));
    let cols = ids.len();
    let route = Route::new(&ids, n_expert).upload(gpu, &map, cols);
    let og = gpu.upload(&vec![0u8; cols * inter * 4]);
    let ou = gpu.upload(&vec![0u8; cols * inter * 4]);
    GateUp {
        out_gate: og.view(),
        out_up: ou.view(),
        gate: parts(&gate_b),
        up: parts(&up_b),
        x: xs.view(),
        route: &route,
        hidden: hidden as u32,
        inter: inter as u32,
        n_expert: n_expert as u32,
        n_used: n_used as u32,
    }
    .launch(gpu, &STEP);
    drain(gpu);
    let got_g = gpu.read_f32(&og);
    let got_u = gpu.read_f32(&ou);
    let mut mid = vec![0f32; cols * inter];
    for (col, &e) in ids.iter().enumerate() {
        let xr = &x[(col / n_used) * hidden..(col / n_used + 1) * hidden];
        let g = gate.project(e as usize, xr);
        let u = up.project(e as usize, xr);
        close(
            "resident gate",
            &got_g[col * inter..(col + 1) * inter],
            &g,
            STEP.rel,
        );
        close(
            "resident up",
            &got_u[col * inter..(col + 1) * inter],
            &u,
            STEP.rel,
        );
        for i in 0..inter {
            mid[col * inter + i] = g[i] / (1.0 + (-g[i]).exp()) * u[i];
        }
    }
    let mid_b = gpu.upload(&f32_bytes(&mid));
    let od = gpu.upload(&vec![0u8; cols * hidden * 4]);
    Down {
        out: od.view(),
        weights: parts(&down_b),
        mid: mid_b.view(),
        route: &route,
        hidden: hidden as u32,
        inter: inter as u32,
        n_expert: n_expert as u32,
        n_used: n_used as u32,
    }
    .launch(gpu, &STEP);
    drain(gpu);
    let got_d = gpu.read_f32(&od);
    for (col, &e) in ids.iter().enumerate() {
        let want = down.project(e as usize, &mid[col * inter..(col + 1) * inter]);
        close(
            "resident down",
            &got_d[col * hidden..(col + 1) * hidden],
            &want,
            STEP.rel,
        );
    }
}

// --- bench -------------------------------------------------------------------

/// This family's routed shape: hidden width, expert intermediate width,
/// experts, experts a token.
pub(crate) const BENCH_SHAPE: (u32, u32, u32, u32) = (2560, 640, 512, 10);

/// Routings a step fixture cycles through, so consecutive launches touch
/// different experts and none finds its weights in the last-level cache.
const BENCH_STEP_ROUTINGS: u64 = 8;

/// A tiny deterministic generator for the bench routings.
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// A prompt-like routing: `n_used` distinct experts a token drawn from a
/// Zipf-like law (`p ∝ 1 / (rank + 20)`) over a shuffled expert order, with
/// eight experts never chosen. At 2,048 tokens of ten experts over 512 this
/// gives a median of about 22 columns an expert, a tenth below 12, and a few
/// above 300 - the spread a real prompt's router produces.
pub(crate) fn bench_routing(n_tok: u32, seed: u64) -> Vec<i32> {
    let (_, _, n_expert, n_used) = BENCH_SHAPE;
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut order: Vec<i32> = (0..n_expert as i32).collect();
    for i in (1..order.len()).rev() {
        let j = (xorshift(&mut state) % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let live = n_expert as usize - 8;
    let mut cdf = Vec::with_capacity(live);
    let mut total = 0.0f64;
    for rank in 0..live {
        total += 1.0 / (rank as f64 + 20.0);
        cdf.push(total);
    }
    let mut ids = Vec::with_capacity((n_tok * n_used) as usize);
    for _ in 0..n_tok {
        let start = ids.len();
        while ids.len() - start < n_used as usize {
            let u = (xorshift(&mut state) >> 11) as f64 / (1u64 << 53) as f64 * total;
            let rank = cdf.partition_point(|&c| c < u).min(live - 1);
            let e = order[rank];
            if !ids[start..].contains(&e) {
                ids.push(e);
            }
        }
    }
    ids
}

/// One bench bank of `slots` matrices on the device, filled from a repeated
/// pseudo-random pattern so no host copy of the whole bank is ever held.
pub(crate) fn bench_bank(
    gpu: &Gpu,
    slots: u32,
    out_dim: u32,
    in_dim: u32,
    seed: u32,
) -> [DeviceBuffer; 3] {
    let fill = |bytes: u64, salt: u32, scale_bytes: bool| {
        let buf = alloc(gpu.device(), bytes).expect("device allocation");
        let chunk_len = bytes.min(16 << 20);
        let chunk: Vec<u8> = signal(chunk_len as usize, seed ^ salt)
            .iter()
            .map(|v| {
                let b = (v.abs() * 9973.0) as u32;
                // Scales stay in a narrow normal band so the sums stay finite.
                if scale_bytes {
                    (0x30 + b % 0x10) as u8
                } else {
                    (b & 0xFF) as u8
                }
            })
            .collect();
        let mut at = 0;
        while at < bytes {
            let len = chunk_len.min(bytes - at);
            gpu.device()
                .copy_h2d(
                    buf.slice(at, len).expect("inside the buffer"),
                    &chunk[..len as usize],
                )
                .expect("host-to-device copy");
            at += len;
        }
        buf
    };
    let rows = u64::from(slots) * u64::from(out_dim);
    let globals: Vec<f32> = (0..slots)
        .map(|s| 1.0e-2 * (1.0 + (s % 7) as f32 * 0.1))
        .collect();
    [
        fill(rows * u64::from(in_dim) / 2, 0x11, false),
        fill(rows * u64::from(in_dim) / 16, 0x22, true),
        gpu.upload(&f32_bytes(&globals)),
    ]
}

/// A zeroed device buffer of `bytes`.
pub(crate) fn zeroed(gpu: &Gpu, bytes: u64) -> DeviceBuffer {
    let buf = alloc(gpu.device(), bytes).expect("device allocation");
    gpu.device().fill_zero(buf.view()).expect("zero fill");
    buf
}

/// Bytes of one matrix of a bank: codes and group scales.
pub(crate) const fn matrix_bytes(out_dim: u32, in_dim: u32) -> u64 {
    out_dim as u64 * in_dim as u64 / 2 + out_dim as u64 * in_dim as u64 / 16
}

/// Which stage a bench fixture times.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    GateUp,
    Down,
}

/// One stage of `st` over `n_tok` tokens of the bench routing, cycling
/// through `routings` routings. Bytes are the distinct experts' matrices (the
/// floor: each read once) plus the activations in and the outputs out; flops
/// are the routed columns' products.
fn bench_stage(gpu: &Gpu, st: Stages, stage: Stage, n_tok: u32, routings: u64) -> Fixture {
    let (hidden, inter, n_expert, n_used) = BENCH_SHAPE;
    let cols = n_tok * n_used;
    let map: Vec<u32> = (0..n_expert).collect();
    let routes: Vec<(Route, DevRoute)> = (0..routings)
        .map(|seed| {
            let route = Route::new(&bench_routing(n_tok, seed + 1), n_expert as usize);
            let dev = route.upload(gpu, &map, cols as usize);
            (route, dev)
        })
        .collect();
    let distinct = routes.iter().map(|(r, _)| r.distinct() as u64).sum::<u64>() / routings;
    let col_bytes = u64::from(cols) * 4;
    let next = std::cell::Cell::new(0usize);
    match stage {
        Stage::GateUp => {
            let gate = bench_bank(gpu, n_expert, inter, hidden, 0x6A7E);
            let up = bench_bank(gpu, n_expert, inter, hidden, 0x0B0B);
            let x = gpu.upload(&f32_bytes(&signal((n_tok * hidden) as usize, 0x5EED)));
            let og = zeroed(gpu, col_bytes * u64::from(inter));
            let ou = zeroed(gpu, col_bytes * u64::from(inter));
            let bytes = distinct * 2 * matrix_bytes(inter, hidden)
                + u64::from(n_tok * hidden) * 4
                + 2 * col_bytes * u64::from(inter);
            let flops = 2 * 2 * u64::from(cols) * u64::from(inter) * u64::from(hidden);
            Fixture::new(bytes, flops, move |gpu| {
                let (_, route) = &routes[next.get() % routes.len()];
                next.set(next.get().wrapping_add(1));
                GateUp {
                    out_gate: og.view(),
                    out_up: ou.view(),
                    gate: views(&gate),
                    up: views(&up),
                    x: x.view(),
                    route,
                    hidden,
                    inter,
                    n_expert,
                    n_used,
                }
                .launch(gpu, &st);
            })
        }
        Stage::Down => {
            let down = bench_bank(gpu, n_expert, hidden, inter, 0xD0D0);
            let mid = gpu.upload(&f32_bytes(&signal((cols * inter) as usize, 0x3EED)));
            let od = zeroed(gpu, col_bytes * u64::from(hidden));
            let bytes = distinct * matrix_bytes(hidden, inter)
                + col_bytes * u64::from(inter)
                + col_bytes * u64::from(hidden);
            let flops = 2 * u64::from(cols) * u64::from(inter) * u64::from(hidden);
            Fixture::new(bytes, flops, move |gpu| {
                let (_, route) = &routes[next.get() % routes.len()];
                next.set(next.get().wrapping_add(1));
                Down {
                    out: od.view(),
                    weights: views(&down),
                    mid: mid.view(),
                    route,
                    hidden,
                    inter,
                    n_expert,
                    n_used,
                }
                .launch(gpu, &st);
            })
        }
    }
}

/// The activated step gate-and-up stage at one decode token.
pub(crate) fn bench_step_gate_up_act_t1(gpu: &Gpu) -> Fixture {
    bench_stage(gpu, STEP_ACT, Stage::GateUp, 1, BENCH_STEP_ROUTINGS)
}
/// The activated step gate-and-up stage over a five-token verify.
pub(crate) fn bench_step_gate_up_act_t5(gpu: &Gpu) -> Fixture {
    bench_stage(gpu, STEP_ACT, Stage::GateUp, 5, BENCH_STEP_ROUTINGS)
}
/// The held activated step gate-and-up stage over a five-token verify.
pub(crate) fn bench_step_gate_up_act_held_t5(gpu: &Gpu) -> Fixture {
    bench_stage(gpu, STEP_ACT_HELD, Stage::GateUp, 5, BENCH_STEP_ROUTINGS)
}

/// The step gate-and-up stage at one decode token.
pub(crate) fn bench_step_gate_up_t1(gpu: &Gpu) -> Fixture {
    bench_stage(gpu, STEP, Stage::GateUp, 1, BENCH_STEP_ROUTINGS)
}

/// The step gate-and-up stage at a speculative verify of five tokens.
pub(crate) fn bench_step_gate_up_t5(gpu: &Gpu) -> Fixture {
    bench_stage(gpu, STEP, Stage::GateUp, 5, BENCH_STEP_ROUTINGS)
}

/// The step gate-and-up stage at eight tokens.
pub(crate) fn bench_step_gate_up_t8(gpu: &Gpu) -> Fixture {
    bench_stage(gpu, STEP, Stage::GateUp, 8, BENCH_STEP_ROUTINGS)
}

/// The step down stage at one decode token.
pub(crate) fn bench_step_down_t1(gpu: &Gpu) -> Fixture {
    bench_stage(gpu, STEP, Stage::Down, 1, BENCH_STEP_ROUTINGS)
}

/// The step down stage at a speculative verify of five tokens.
pub(crate) fn bench_step_down_t5(gpu: &Gpu) -> Fixture {
    bench_stage(gpu, STEP, Stage::Down, 5, BENCH_STEP_ROUTINGS)
}

/// The step down stage at eight tokens.
pub(crate) fn bench_step_down_t8(gpu: &Gpu) -> Fixture {
    bench_stage(gpu, STEP, Stage::Down, 8, BENCH_STEP_ROUTINGS)
}

/// The prompt gate-and-up stage over a 2,048-token chunk.
pub(crate) fn bench_prompt_gate_up_2048(gpu: &Gpu) -> Fixture {
    bench_stage(gpu, PROMPT, Stage::GateUp, 2048, 1)
}

/// The prompt down stage over a 2,048-token chunk.
pub(crate) fn bench_prompt_down_2048(gpu: &Gpu) -> Fixture {
    bench_stage(gpu, PROMPT, Stage::Down, 2048, 1)
}

/// The prompt gate-and-up stage over a 4,096-token chunk.
pub(crate) fn bench_prompt_gate_up_4096(gpu: &Gpu) -> Fixture {
    bench_stage(gpu, PROMPT, Stage::GateUp, 4096, 1)
}

/// The prompt down stage over a 4,096-token chunk.
pub(crate) fn bench_prompt_down_4096(gpu: &Gpu) -> Fixture {
    bench_stage(gpu, PROMPT, Stage::Down, 4096, 1)
}

/// The prompt gate-and-up stage over an 8,192-token chunk.
pub(crate) fn bench_prompt_gate_up_8192(gpu: &Gpu) -> Fixture {
    bench_stage(gpu, PROMPT, Stage::GateUp, 8192, 1)
}

/// The prompt down stage over an 8,192-token chunk.
pub(crate) fn bench_prompt_down_8192(gpu: &Gpu) -> Fixture {
    bench_stage(gpu, PROMPT, Stage::Down, 8192, 1)
}
