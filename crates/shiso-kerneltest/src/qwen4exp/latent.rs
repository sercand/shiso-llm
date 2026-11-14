//! The latent routed experts' kernels, graded against a host restatement of
//! the latent feed-forward's routed half.
//!
//! A latent expert contracts over a narrower row than the hidden one - the
//! token's activation is projected once into `n_groups` latent rows, and an
//! expert reads the row its label names. The gate-and-up entries are the plain
//! NVFP4 ones with that one indirection added, so beside the host grading one
//! case holds the other half of the contract: with a single group and every
//! label zero they are the plain entries, bit for bit. The combine entry sums a
//! token's routed outputs per group, in a fixed order the host repeats exactly.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "fixture dimensions and ids are small literals"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;
use crate::qwen4exp::nvfp4::{Bank, DevRoute, Route, i32_bytes, views};
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

/// The module the NVFP4 expert entries live in.
const MODULE: &str = "q4e_modelopt";
/// The module the combine entry lives in.
const MOE_MODULE: &str = "q4e_moe";

/// The latent pack's expert shape: contraction width of gate and up.
const IN: usize = 1792;
/// Width of gate's and up's output.
const INTER: usize = 640;
/// Latent groups a token's activation is split into.
const GROUPS: usize = 8;
/// Experts a fixture routes over.
const EXPERTS: usize = 16;
/// Experts a token uses.
const USED: usize = 10;

/// Relative budget of the step kernels against the host product, the plain
/// step entries' own (`2e-4`).
const STEP_REL: f32 = 2e-4;
/// Relative budget of the tensor-core prompt kernel: it rounds activations to
/// halves, the plain prompt entry's own (`4e-3`).
const PROMPT_REL: f32 = 4e-3;

/// A step entry: one output (the activated intermediate), 128 threads, no
/// dynamic shared memory, `ceil(inter / 4)` row blocks.
const STEP_THREADS: u32 = 128;
/// The prompt entry's block, and the dynamic shared memory it opts into
/// (`Q4E_MG_THREADS`, `Q4E_MG_SMEM`).
const PROMPT_THREADS: u32 = 512;
/// See [`PROMPT_THREADS`].
const PROMPT_SMEM: u32 = 92_416;

/// A generator for fixture routings and labels.
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// `USED` distinct experts a token, for `n_tok` tokens.
fn routing(n_tok: usize, seed: u64) -> Vec<i32> {
    let mut state = seed;
    let mut ids = Vec::with_capacity(n_tok * USED);
    for _ in 0..n_tok {
        let mut pool: Vec<i32> = (0..EXPERTS as i32).collect();
        for _ in 0..USED {
            let at = (xorshift(&mut state) % pool.len() as u64) as usize;
            ids.push(pool.swap_remove(at));
        }
    }
    ids
}

/// A random group below `n_groups` for each expert.
fn random_labels(n_groups: usize, seed: u64) -> Vec<u32> {
    let mut state = seed;
    (0..EXPERTS)
        .map(|_| (xorshift(&mut state) % n_groups as u64) as u32)
        .collect()
}

/// A fixture seed widened to the generator's, kept clear of a small seed's low
/// bits so the generator does not start near zero.
fn seed64(seed: u32) -> u64 {
    u64::from(seed) + 0x2_0000_0000
}

fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// An expert-to-slot map that is not the identity.
fn slot_map() -> Vec<u32> {
    (0..EXPERTS as u32)
        .map(|e| (e * 5 + 3) % EXPERTS as u32)
        .collect()
}

/// One bank's slot as a dense `[out, in]` matrix, decoded once.
fn dense(bank: &Bank, slot: usize) -> Vec<f32> {
    let mut m = Vec::with_capacity(bank.out_dim * bank.in_dim);
    for o in 0..bank.out_dim {
        for k in 0..bank.in_dim {
            m.push(bank.at(slot, o, k));
        }
    }
    m
}

/// `m [rows, k] · x`, summed in `f64` as the plain cases' host product is.
fn matvec(m: &[f32], x: &[f32]) -> Vec<f32> {
    m.chunks_exact(x.len())
        .map(|row| {
            row.iter()
                .zip(x)
                .map(|(w, v)| f64::from(*w) * f64::from(*v))
                .sum::<f64>() as f32
        })
        .collect()
}

fn close(what: &str, got: &[f32], want: &[f32], rel: f32) {
    let mean = want.iter().map(|v| v.abs()).sum::<f32>() / want.len() as f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = rel * w.abs().max(mean).max(1e-12);
        assert!((g - w).abs() <= tol, "{what}: {i}: {g} vs {w}");
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

/// Which gate-and-up entry a launch runs.
#[derive(Clone, Copy)]
struct Entry {
    name: &'static str,
    /// Whether it takes `labels, n_groups` after `n_used`.
    latent: bool,
    /// Whether it writes the activated intermediate (one output) rather than
    /// the gate and up rows.
    act: bool,
    /// Whether it is the tensor-core prompt kernel.
    prompt: bool,
}

const LATENT_ACT: Entry = Entry {
    name: "q4e_nvfp4_latent_gate_up_act_kernel",
    latent: true,
    act: true,
    prompt: false,
};
const LATENT_ACT_HELD: Entry = Entry {
    name: "q4e_nvfp4_latent_gate_up_act_held_kernel",
    latent: true,
    act: true,
    prompt: false,
};
const LATENT_GEMM: Entry = Entry {
    name: "q4e_nvfp4_latent_gemm_gate_up_kernel",
    latent: true,
    act: false,
    prompt: true,
};
const PLAIN_ACT: Entry = Entry {
    name: "q4e_nvfp4_experts_gate_up_act_kernel",
    latent: false,
    act: true,
    prompt: false,
};
const PLAIN_ACT_HELD: Entry = Entry {
    name: "q4e_nvfp4_experts_gate_up_act_held_kernel",
    latent: false,
    act: true,
    prompt: false,
};
const PLAIN_GEMM: Entry = Entry {
    name: "q4e_nvfp4_experts_gemm_gate_up_kernel",
    latent: false,
    act: false,
    prompt: true,
};

/// A fixture: gate and up banks on the device and host, a routing, and an
/// activation of `n_rows` rows of [`IN`].
struct Fixture {
    gate: Bank,
    up: Bank,
    gate_dev: [DeviceBuffer; 3],
    up_dev: [DeviceBuffer; 3],
    map: Vec<u32>,
    ids: Vec<i32>,
    route: DevRoute,
}

impl Fixture {
    fn new(gpu: &Gpu, n_tok: usize, seed: u32) -> Self {
        let gate = Bank::new(EXPERTS, INTER, IN, seed);
        let up = Bank::new(EXPERTS, INTER, IN, seed + 1);
        let map = slot_map();
        let ids = routing(n_tok, seed64(seed));
        let route = Route::new(&ids, EXPERTS).upload(gpu, &map, ids.len());
        Self {
            gate_dev: gate.upload(gpu),
            up_dev: up.upload(gpu),
            gate,
            up,
            map,
            ids,
            route,
        }
    }

    fn columns(&self) -> usize {
        self.ids.len()
    }

    /// Launches `entry` over `x` (`[rows, IN]`) and returns its outputs: the
    /// activated intermediate, or the gate rows then the up rows, each
    /// `[columns, INTER]`.
    fn launch(
        &self,
        gpu: &Gpu,
        entry: Entry,
        x: &[f32],
        labels: &[u32],
        n_groups: usize,
    ) -> Vec<Vec<f32>> {
        let cols = self.columns();
        let n = cols * INTER;
        let outs: Vec<DeviceBuffer> = (0..if entry.act { 1 } else { 2 })
            .map(|_| gpu.upload(&f32_bytes(&vec![f32::NAN; n])))
            .collect();
        let xs = gpu.upload(&f32_bytes(x));
        let lb = gpu.upload(&u32_bytes(labels));
        let r = &self.route;
        let mut args: Vec<KernelArg<'_>> =
            outs.iter().map(|o| KernelArg::Buffer(o.view())).collect();
        for part in views(&self.gate_dev).into_iter().chain(views(&self.up_dev)) {
            args.push(KernelArg::Buffer(part));
        }
        args.extend([
            KernelArg::Buffer(xs.view()),
            KernelArg::Buffer(r.sorted.view()),
            KernelArg::Buffer(r.bounds.view()),
            KernelArg::Buffer(r.active.view()),
            KernelArg::Buffer(r.map.view()),
            KernelArg::U32(IN as u32),
            KernelArg::U32(INTER as u32),
            KernelArg::U32(EXPERTS as u32),
            KernelArg::U32(USED as u32),
        ]);
        if entry.latent {
            args.push(KernelArg::Buffer(lb.view()));
            args.push(KernelArg::U32(n_groups as u32));
        }
        let (blocks, threads, smem) = if entry.prompt {
            (4 * INTER.div_ceil(128), PROMPT_THREADS, PROMPT_SMEM)
        } else {
            (INTER.div_ceil(4), STEP_THREADS, 0)
        };
        let mut launch = LaunchArgs::new(
            Extent3::new(blocks as u32, (cols as u32).min(EXPERTS as u32), 1),
            Extent3::linear(threads),
            &args,
        );
        launch.smem_bytes = smem;
        run(gpu, KernelRef::new(MODULE, entry.name), &launch);
        drain(gpu);
        outs.iter().map(|o| gpu.read_f32(o)).collect()
    }

    /// The host's gate and up rows for every column, each column against its
    /// own group's row of `x`: `[columns, INTER]` each.
    fn host(&self, x: &[f32], labels: &[u32], n_groups: usize) -> [Vec<f32>; 2] {
        let dense_banks: Vec<[Vec<f32>; 2]> = (0..EXPERTS)
            .map(|e| {
                let slot = self.map[e] as usize;
                [dense(&self.gate, slot), dense(&self.up, slot)]
            })
            .collect();
        let mut gate = Vec::with_capacity(self.columns() * INTER);
        let mut up = Vec::with_capacity(self.columns() * INTER);
        for (col, &e) in self.ids.iter().enumerate() {
            let e = e as usize;
            let row = (col / USED) * n_groups + labels[e] as usize;
            let xr = &x[row * IN..(row + 1) * IN];
            gate.extend(matvec(&dense_banks[e][0], xr));
            up.extend(matvec(&dense_banks[e][1], xr));
        }
        [gate, up]
    }
}

/// `silu(gate) * up` as the device writes it, in `f32`.
fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
        .collect()
}

/// An activated latent entry against the host: each column's `silu(gate) *
/// up` over its own label's row.
fn activated_matches_host(gpu: &Gpu, entry: Entry, n_tok: usize, seed: u32) {
    let fx = Fixture::new(gpu, n_tok, seed);
    let labels = random_labels(GROUPS, seed64(seed) + 1);
    let x = signal(n_tok * GROUPS * IN, seed + 7);
    let [out] = <[Vec<f32>; 1]>::try_from(fx.launch(gpu, entry, &x, &labels, GROUPS))
        .expect("an activated entry has one output");
    let [gate, up] = fx.host(&x, &labels, GROUPS);
    close(entry.name, &out, &swiglu(&gate, &up), STEP_REL);
}

/// The decode entry, one token.
///
/// # Panics
///
/// When an output differs from the host's, or the device refuses a launch or a
/// copy.
pub fn latent_gate_up_act_matches_host(gpu: &Gpu) {
    activated_matches_host(gpu, LATENT_ACT, 1, 1001);
}

/// The verify entry, four tokens, the activation row held in registers.
///
/// # Panics
///
/// When an output differs from the host's, or the device refuses a launch or a
/// copy.
pub fn latent_gate_up_act_held_matches_host(gpu: &Gpu) {
    activated_matches_host(gpu, LATENT_ACT_HELD, 4, 1101);
}

/// The prompt entry, sixty-four tokens: gate and up rows against the host.
///
/// # Panics
///
/// When an output differs from the host's, or the device refuses a launch or a
/// copy.
pub fn latent_gemm_gate_up_matches_host(gpu: &Gpu) {
    let n_tok = 64;
    let fx = Fixture::new(gpu, n_tok, 1201);
    let labels = random_labels(GROUPS, seed64(1201) + 1);
    let x = signal(n_tok * GROUPS * IN, 1208);
    let got = fx.launch(gpu, LATENT_GEMM, &x, &labels, GROUPS);
    let want = fx.host(&x, &labels, GROUPS);
    close("latent gate", &got[0], &want[0], PROMPT_REL);
    close("latent up", &got[1], &want[1], PROMPT_REL);
}

/// With one group and every label zero, each latent gate-and-up entry is its
/// plain twin, bit for bit: the group indirection adds a row choice and
/// nothing else.
///
/// # Panics
///
/// When an output differs from the host's, or the device refuses a launch or a
/// copy.
pub fn latent_labels_zero_equals_plain(gpu: &Gpu) {
    let n_tok = 4;
    let fx = Fixture::new(gpu, n_tok, 1301);
    let labels = vec![0u32; EXPERTS];
    let x = signal(n_tok * IN, 1308);
    for (latent, plain) in [
        (LATENT_ACT, PLAIN_ACT),
        (LATENT_ACT_HELD, PLAIN_ACT_HELD),
        (LATENT_GEMM, PLAIN_GEMM),
    ] {
        let got = fx.launch(gpu, latent, &x, &labels, 1);
        let want = fx.launch(gpu, plain, &x, &labels, 1);
        assert_eq!(got.len(), want.len(), "{}: output count", latent.name);
        for (g, w) in got.iter().zip(&want) {
            same_bits(latent.name, g, w);
        }
    }
}

/// The combine entry against a host sum in the same slot order, bit for bit,
/// at `grid_x` blocks across the width.
fn combine_case(gpu: &Gpu, ids: &[i32], labels: &[u32], width: usize, grid_x: u32) {
    let n_used = USED;
    let n_tok = ids.len() / n_used;
    let cols = ids.len();
    let down = signal(cols * width, 1401);
    let weights: Vec<f32> = signal(cols, 1402).iter().map(|v| v.abs()).collect();
    let mut want = vec![0f32; n_tok * GROUPS * width];
    for t in 0..n_tok {
        for g in 0..GROUPS {
            for i in 0..width {
                let mut acc = 0f32;
                for j in 0..n_used {
                    let col = t * n_used + j;
                    if ids[col] >= 0 && labels[ids[col] as usize] as usize == g {
                        acc = weights[col].mul_add(down[col * width + i], acc);
                    }
                }
                want[(t * GROUPS + g) * width + i] = acc;
            }
        }
    }
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; want.len()]));
    let (d, w, i, l) = (
        gpu.upload(&f32_bytes(&down)),
        gpu.upload(&f32_bytes(&weights)),
        gpu.upload(&i32_bytes(ids)),
        gpu.upload(&u32_bytes(labels)),
    );
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(d.view()),
        KernelArg::Buffer(w.view()),
        KernelArg::Buffer(i.view()),
        KernelArg::Buffer(l.view()),
        KernelArg::U32(width as u32),
        KernelArg::U32(GROUPS as u32),
        KernelArg::U32(n_used as u32),
    ];
    run(
        gpu,
        KernelRef::new(MOE_MODULE, "q4e_latent_combine_kernel"),
        &LaunchArgs::new(
            Extent3::new(grid_x, n_tok as u32, 1),
            Extent3::linear(256),
            &args,
        ),
    );
    drain(gpu);
    same_bits("q4e_latent_combine_kernel", &gpu.read_f32(&out), &want);
}

/// The latent combine over a routing built to hit its edges - a token with
/// three experts in one group, groups no expert is in, a slot with no expert -
/// and over a random one,
/// each at a full grid over the token's `GROUPS * width` elements and at a grid
/// narrower than that, so the stride is run.
///
/// # Panics
///
/// When an output differs from the host's, or the device refuses a launch or a
/// copy.
pub fn latent_combine_matches_host(gpu: &Gpu) {
    let width = IN;
    // Experts 0, 5 and 10 share group 0; groups 5 to 7 hold no expert at all.
    let edge_labels: Vec<u32> = (0..EXPERTS as u32).map(|e| e % 5).collect();
    let mut ids = vec![0, 5, 10, 1, 2, 3, 4, 6, 7, 8];
    ids.extend(routing(2, 77));
    // A slot the router left empty adds nothing to any group.
    ids[13] = -1;
    for grid_x in [(GROUPS * width).div_ceil(256) as u32, 2] {
        combine_case(gpu, &ids, &edge_labels, width, grid_x);
        let labels = random_labels(GROUPS, 91);
        combine_case(gpu, &routing(5, 93), &labels, width, grid_x);
    }
}

/// The module the block-format step projections live in.
const GGML_MODULE: &str = "q4e_ggml";

/// Activation rows one block of a step projection carries.
const STEP_COLS: usize = 8;

/// How a test's dense projection rows are stored.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Codec {
    /// `Q8_0`: a half-precision scale and 32 signed codes a block.
    Q8,
    /// `Q4_0`: a half-precision scale and 16 bytes of nibbles a block.
    Q4,
}

/// Random block rows of `codec`: `in_dim / 32` blocks a row, every scale
/// finite and small.
fn block_rows(codec: Codec, out_dim: usize, in_dim: usize, seed: u64) -> Vec<u8> {
    let payload = if codec == Codec::Q8 { 32 } else { 16 };
    let mut state = seed | 1;
    let mut bytes = Vec::with_capacity(out_dim * (in_dim / 32) * (2 + payload));
    for _ in 0..out_dim * (in_dim / 32) {
        // 0x2c00..0x2fff: positive, between 2^-4 and 2^-3.
        let scale = 0x2c00u16 | (xorshift(&mut state) & 0x3ff) as u16;
        bytes.extend(scale.to_le_bytes());
        bytes.extend((0..payload).map(|_| xorshift(&mut state) as u8));
    }
    bytes
}

/// Which of the latent stage's two step projections a launch is.
#[derive(Clone, Copy)]
enum Side {
    /// Into the latent rows: unused groups' output rows are left out.
    Down,
    /// Out of them: unused groups' contraction runs are left out.
    Up,
}

/// One step projection launch, plain or masked, as the host plans it: lanes a
/// row from the contraction width, four output rows a unit past two
/// activation rows.
fn step_projection(
    gpu: &Gpu,
    entry: (Codec, Option<Side>),
    dims: (usize, usize, usize),
    buffers: (&DeviceBuffer, &DeviceBuffer, &DeviceBuffer),
    masks: Option<(&DeviceBuffer, usize)>,
) {
    let (in_dim, out_dim, rows) = dims;
    let (out, w, x) = buffers;
    let lanes: u32 = if in_dim / 16 <= 32 { 32 } else { 128 };
    let unit: u32 = if rows > 2 { 4 } else { 1 };
    let name = match (entry.0, entry.1, unit == 4) {
        (Codec::Q8, None, false) => "q4e_g_step_matvec_q8_0_kernel",
        (Codec::Q8, None, true) => "q4e_g_step_matvec4_q8_0_kernel",
        (Codec::Q8, Some(Side::Down), false) => "q4e_g_step_latent_down_q8_0_kernel",
        (Codec::Q8, Some(Side::Down), true) => "q4e_g_step_latent_down4_q8_0_kernel",
        (Codec::Q8, Some(Side::Up), false) => "q4e_g_step_latent_up_q8_0_kernel",
        (Codec::Q8, Some(Side::Up), true) => "q4e_g_step_latent_up4_q8_0_kernel",
        (Codec::Q4, None, false) => "q4e_g_step_matvec_q4_0_kernel",
        (Codec::Q4, None, true) => "q4e_g_step_matvec4_q4_0_kernel",
        (Codec::Q4, Some(Side::Down), false) => "q4e_g_step_latent_down_q4_0_kernel",
        (Codec::Q4, Some(Side::Down), true) => "q4e_g_step_latent_down4_q4_0_kernel",
        (Codec::Q4, Some(Side::Up), false) => "q4e_g_step_latent_up_q4_0_kernel",
        (Codec::Q4, Some(Side::Up), true) => "q4e_g_step_latent_up4_q4_0_kernel",
    };
    let mut args = vec![
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(w.view()),
        KernelArg::Buffer(x.view()),
        KernelArg::U32(in_dim as u32),
        KernelArg::U32(out_dim as u32),
        KernelArg::U32(rows as u32),
        KernelArg::U32(lanes),
    ];
    if let Some((masks, width)) = masks {
        args.push(KernelArg::Buffer(masks.view()));
        args.push(KernelArg::U32(width as u32));
    }
    let per_block = 4 * 32 / lanes * unit;
    run(
        gpu,
        KernelRef::new(GGML_MODULE, name),
        &LaunchArgs::new(
            Extent3::new(
                (out_dim as u32).div_ceil(per_block),
                rows.div_ceil(STEP_COLS) as u32,
                1,
            ),
            Extent3::linear(128),
            &args,
        ),
    );
    drain(gpu);
}

/// The groups token `t` routes to, as a mask.
fn used_groups(ids: &[i32], labels: &[u32], t: usize) -> u32 {
    ids[t * USED..(t + 1) * USED]
        .iter()
        .filter(|&&e| e >= 0)
        .fold(0u32, |m, &e| m | 1 << labels[e as usize])
}

/// The mask launch over a routing, checked against a host fold: the device
/// buffer and the masks, one a block of eight rows.
fn group_masks(gpu: &Gpu, ids: &[i32], labels: &[u32]) -> (DeviceBuffer, Vec<u32>) {
    let n_tok = ids.len() / USED;
    let want: Vec<u32> = (0..n_tok.div_ceil(STEP_COLS))
        .map(|b| {
            (b * STEP_COLS..n_tok.min((b + 1) * STEP_COLS))
                .fold(0, |m, t| m | used_groups(ids, labels, t))
        })
        .collect();
    let (ids_dev, labels_dev) = (gpu.upload(&i32_bytes(ids)), gpu.upload(&u32_bytes(labels)));
    let masks = gpu.upload(&u32_bytes(&vec![u32::MAX; want.len()]));
    run(
        gpu,
        KernelRef::new(MOE_MODULE, "q4e_latent_groups_kernel"),
        &LaunchArgs::new(
            Extent3::new(want.len() as u32, 1, 1),
            Extent3::linear(32),
            &[
                KernelArg::Buffer(masks.view()),
                KernelArg::Buffer(ids_dev.view()),
                KernelArg::Buffer(labels_dev.view()),
                KernelArg::U32(n_tok as u32),
                KernelArg::U32(USED as u32),
            ],
        ),
    );
    drain(gpu);
    let got: Vec<u32> = gpu
        .read(&masks)
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(got, want, "q4e_latent_groups_kernel");
    (masks, want)
}

/// One shape and routing of [`latent_step_projections_skip_unused_groups`].
fn masked_case(
    gpu: &Gpu,
    codec: Codec,
    (hidden, width): (usize, usize),
    ids: &[i32],
    labels: &[u32],
    seed: u64,
) {
    let n_tok = ids.len() / USED;
    let span = GROUPS * width;
    let used = |t: usize| used_groups(ids, labels, t);
    let (masks, want_masks) = group_masks(gpu, ids, labels);

    // Down: `hidden` in, `span` out. A used group's rows are the plain
    // entry's; an unused group's are left as they were.
    let down_w = gpu.upload(&block_rows(codec, span, hidden, seed));
    let x = gpu.upload(&f32_bytes(&signal(n_tok * hidden, 1501)));
    let plain = gpu.upload(&f32_bytes(&vec![f32::NAN; n_tok * span]));
    let masked = gpu.upload(&f32_bytes(&vec![f32::NAN; n_tok * span]));
    step_projection(
        gpu,
        (codec, None),
        (hidden, span, n_tok),
        (&plain, &down_w, &x),
        None,
    );
    step_projection(
        gpu,
        (codec, Some(Side::Down)),
        (hidden, span, n_tok),
        (&masked, &down_w, &x),
        Some((&masks, width)),
    );
    let (plain, masked) = (gpu.read_f32(&plain), gpu.read_f32(&masked));
    let mut skipped = 0;
    for t in 0..n_tok {
        for g in 0..GROUPS {
            let at = (t * GROUPS + g) * width;
            if want_masks[t / STEP_COLS] >> g & 1 == 1 {
                same_bits(
                    "latent down, a used group",
                    &masked[at..at + width],
                    &plain[at..at + width],
                );
            } else {
                assert!(
                    masked[at..at + width].iter().all(|v| v.is_nan()),
                    "latent down wrote group {g} of token {t}, which no row of its block uses"
                );
                skipped += 1;
            }
        }
    }
    assert!(
        skipped > 0,
        "the routing leaves no group unused; the case tests nothing"
    );

    // Up: `span` in, `hidden` out, over sums that are zero in every group the
    // token does not use, as the combine leaves them. Every output is the
    // plain entry's.
    let mut sums = signal(n_tok * span, 1502);
    for t in 0..n_tok {
        for g in 0..GROUPS {
            if used(t) >> g & 1 == 0 {
                sums[(t * GROUPS + g) * width..][..width].fill(0.0);
            }
        }
    }
    let up_w = gpu.upload(&block_rows(codec, hidden, span, seed + 1));
    let sums = gpu.upload(&f32_bytes(&sums));
    let plain = gpu.upload(&f32_bytes(&vec![f32::NAN; n_tok * hidden]));
    let masked = gpu.upload(&f32_bytes(&vec![f32::NAN; n_tok * hidden]));
    step_projection(
        gpu,
        (codec, None),
        (span, hidden, n_tok),
        (&plain, &up_w, &sums),
        None,
    );
    step_projection(
        gpu,
        (codec, Some(Side::Up)),
        (span, hidden, n_tok),
        (&masked, &up_w, &sums),
        Some((&masks, width)),
    );
    same_bits("latent up", &gpu.read_f32(&masked), &gpu.read_f32(&plain));
}

/// Experts 0 to 9 in group 0, 10 to 12 in group 3, 13 to 15 in group 6.
fn disjoint_labels() -> Vec<u32> {
    (0..EXPERTS as u32)
        .map(|e| match e {
            0..=9 => 0,
            10..=12 => 3,
            _ => 6,
        })
        .collect()
}

/// Token `t`'s experts in the disjoint routing: A uses group 0 alone, B groups
/// 0 and 3, C groups 0 and 6, so a block mixing them has a mask wider than most
/// of its rows' own.
fn disjoint_routing(n_tok: usize) -> Vec<i32> {
    let mut ids = Vec::with_capacity(n_tok * USED);
    for t in 0..n_tok {
        match t % 3 {
            0 => ids.extend(0..10),
            1 => ids.extend((0..7).chain(10..13)),
            _ => ids.extend((0..7).chain(13..16)),
        }
    }
    ids
}

/// A row's result does not depend on which rows share its block: over the
/// disjoint routing, in passes of two, five and eleven rows, each row's down
/// rows (the groups it uses) and its up output equal a one-row launch of that
/// row with its own mask, bit for bit.
fn block_invariance_case(gpu: &Gpu, codec: Codec, (hidden, width): (usize, usize), n_tok: usize) {
    let labels = disjoint_labels();
    let ids = disjoint_routing(n_tok);
    let span = GROUPS * width;
    let (masks, _) = group_masks(gpu, &ids, &labels);
    let down_w = gpu.upload(&block_rows(codec, span, hidden, 301));
    let up_w = gpu.upload(&block_rows(codec, hidden, span, 302));
    let xs = signal(n_tok * hidden, 1601);
    let mut sums = signal(n_tok * span, 1602);
    for t in 0..n_tok {
        for g in 0..GROUPS {
            if used_groups(&ids, &labels, t) >> g & 1 == 0 {
                sums[(t * GROUPS + g) * width..][..width].fill(0.0);
            }
        }
    }
    let nan = |n: usize| gpu.upload(&f32_bytes(&vec![f32::NAN; n]));
    let (down, up) = (nan(n_tok * span), nan(n_tok * hidden));
    let (x_dev, sums_dev) = (gpu.upload(&f32_bytes(&xs)), gpu.upload(&f32_bytes(&sums)));
    step_projection(
        gpu,
        (codec, Some(Side::Down)),
        (hidden, span, n_tok),
        (&down, &down_w, &x_dev),
        Some((&masks, width)),
    );
    step_projection(
        gpu,
        (codec, Some(Side::Up)),
        (span, hidden, n_tok),
        (&up, &up_w, &sums_dev),
        Some((&masks, width)),
    );
    let (down, up) = (gpu.read_f32(&down), gpu.read_f32(&up));
    for t in 0..n_tok {
        let (own, _) = group_masks(gpu, &ids[t * USED..(t + 1) * USED], &labels);
        let x_one = gpu.upload(&f32_bytes(&xs[t * hidden..(t + 1) * hidden]));
        let sums_one = gpu.upload(&f32_bytes(&sums[t * span..(t + 1) * span]));
        let (down_one, up_one) = (nan(span), nan(hidden));
        step_projection(
            gpu,
            (codec, Some(Side::Down)),
            (hidden, span, 1),
            (&down_one, &down_w, &x_one),
            Some((&own, width)),
        );
        step_projection(
            gpu,
            (codec, Some(Side::Up)),
            (span, hidden, 1),
            (&up_one, &up_w, &sums_one),
            Some((&own, width)),
        );
        let down_one = gpu.read_f32(&down_one);
        for g in 0..GROUPS {
            if used_groups(&ids, &labels, t) >> g & 1 == 1 {
                same_bits(
                    "a row's used down rows, alone and in a block",
                    &down[(t * GROUPS + g) * width..][..width],
                    &down_one[g * width..][..width],
                );
            }
        }
        same_bits(
            "a row's up output, alone and in a block",
            &up[t * hidden..(t + 1) * hidden],
            &gpu.read_f32(&up_one),
        );
    }
}

/// A step pass's two latent projections against the plain step entries on the
/// same inputs, bit for bit, with the group masks checked against a host
/// fold: the down entry writes exactly the groups a block of rows uses, and
/// the up entry's skipped runs change no output.
///
/// Over one row (one output row a unit), five and eleven (four a unit, the
/// second with two blocks of rows whose masks differ), at a contraction a
/// block of lanes sums and at one a warp does.
///
/// # Panics
///
/// When a mask, a used group's row or an up output differs, an unused group
/// is written, or the device refuses a launch or a read.
pub fn latent_step_projections_skip_unused_groups(gpu: &Gpu) {
    for codec in [Codec::Q8, Codec::Q4] {
        skip_unused_groups(gpu, codec);
    }
}

/// [`latent_step_projections_skip_unused_groups`] over projections stored as
/// `codec`.
fn skip_unused_groups(gpu: &Gpu, codec: Codec) {
    // Experts land in groups 0 to 4 only, so 5 to 7 are never used.
    let edge: Vec<u32> = (0..EXPERTS as u32).map(|e| e % 5).collect();
    // Three groups in all.
    let few: Vec<u32> = (0..EXPERTS as u32)
        .map(|e| [1, 4, 6][e as usize % 3])
        .collect();
    for (hidden, width) in [(2560, 128), (256, 32)] {
        for (n_tok, seed) in [(1, 201), (5, 202), (11, 203)] {
            masked_case(
                gpu,
                codec,
                (hidden, width),
                &routing(n_tok, seed),
                &edge,
                seed,
            );
            masked_case(
                gpu,
                codec,
                (hidden, width),
                &routing(n_tok, seed + 50),
                &few,
                seed,
            );
        }
        // Slots the router left empty set no bit and leave nothing to compute.
        let mut sparse = routing(11, 204);
        for slot in [3, 14, 15, 60, 109] {
            sparse[slot] = -1;
        }
        masked_case(gpu, codec, (hidden, width), &sparse, &edge, 204);
        for n_tok in [2, 5, 11] {
            block_invariance_case(gpu, codec, (hidden, width), n_tok);
        }
    }
}
