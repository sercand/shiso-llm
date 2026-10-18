//! The routed block's fused step launch (`q4e_ffn`), graded against the
//! launches it replaces.
//!
//! `q4e_ffn_route_kernel` (one row) and `q4e_ffn_route_rows_kernel` (two to
//! eight) promise, from one block input, the bits of: the brain-float router
//! and the shared expert's gate logit (`q4e_bf16_gemv_kernel` /
//! `q4e_bf16_rows_kernel` at the engine's slicing), the tiny routed selection
//! and sort (`q4e_moe_route_sort_tiny_kernel`), the shared expert's MXFP8 gate
//! and up (the group launches) through `q4e_swiglu_kernel`, and its MXFP8 down
//! projection (`q4e_mxfp8_gemv_kernel` / `q4e_mxfp8_rows_kernel`). Every one
//! of those outputs is compared bit for bit, at every row count the launch
//! takes, and the logits are also held to a host `f64` product so the chain
//! itself is anchored.
//!
//! The launch orders its work items through four words it must find at zero
//! and leave at zero; every case reuses one word buffer across launches and
//! checks it after each, so a launch that left a word behind fails the next.
//!
//! The geometry is transcribed from `shiso_engine::sparse::{matmul, moe}`,
//! which this crate may not depend on; each constant names the one it
//! mirrors.
#![expect(
    clippy::cast_possible_truncation,
    reason = "fixture dimensions are model widths far below u32::MAX"
)]

use half::bf16;
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::bench::Fixture;
use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;
use crate::qwen4exp::mxfp8::{Device as Mx, Matrix as MxMatrix, same_bits, sentinel};

/// The fused launch's module, mirroring `kernels::MOD_FFN`.
const MODULE: &str = "q4e_ffn";
/// The one-row entry.
const ROUTE: &str = "q4e_ffn_route_kernel";
/// The two-to-eight-row entry.
const ROUTE_ROWS: &str = "q4e_ffn_route_rows_kernel";
/// Hidden width, experts, experts a token, shared expert width: the shipped
/// shape.
const HIDDEN: u32 = 2_560;
const EXPERTS: u32 = 512;
const USED: u32 = 10;
const INTER: u32 = 640;
/// The widest pass the launch takes, mirroring `moe::FUSED_ROUTE_MAX_TOKENS`.
const MAX_ROWS: u32 = 8;
/// Warps a fused block runs, mirroring `moe::FUSED_ROUTE_WARPS`.
const WARPS: u32 = 8;
/// Dynamic shared memory the launch reserves at up to [`THROTTLED`] rows,
/// mirroring `moe::FUSED_ROUTE_SMEM`.
const SMEM: u32 = 36_000;
/// Mirroring `moe::FUSED_ROUTE_THROTTLED`.
const THROTTLED: u32 = 2;

/// Whether a fused launch's argument list - `rows` last - takes [`SMEM`].
fn throttled(args: &[KernelArg<'_>]) -> bool {
    matches!(args.last(), Some(KernelArg::U32(rows)) if *rows <= THROTTLED)
}
/// Words the launch orders its items through, mirroring
/// `moe::FUSED_ROUTE_SYNC_WORDS`.
const SYNC_WORDS: usize = 4;

/// `matmul::bf16_split`, restated.
fn bf16_split(in_dim: u32, out_dim: u32) -> u32 {
    let mut split = 1u32;
    while split < 8 && out_dim * split < 2_048 && in_dim / (split * 2) >= 256 {
        split *= 2;
    }
    split
}

/// A brain-float matrix `[out, in]` on the host and the device.
struct Brain {
    values: Vec<f32>,
    buf: DeviceBuffer,
}

impl Brain {
    fn new(gpu: &Gpu, out_dim: u32, in_dim: u32, seed: u32) -> Self {
        let bits: Vec<u16> = signal(out_dim as usize * in_dim as usize, seed)
            .iter()
            .map(|v| bf16::from_f32(v * 0.05).to_bits())
            .collect();
        let values = bits.iter().map(|b| bf16::from_bits(*b).to_f32()).collect();
        let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        Self {
            values,
            buf: gpu.upload(&bytes),
        }
    }
}

/// One layer's weights.
struct Layer {
    router: Brain,
    gate: Brain,
    shared: [Mx; 3],
}

impl Layer {
    fn new(gpu: &Gpu, seed: u32) -> Self {
        let mx = |out_dim, in_dim, s: u32| {
            Mx::upload(gpu, &MxMatrix::new(out_dim, in_dim, u64::from(seed ^ s)))
        };
        Self {
            router: Brain::new(gpu, EXPERTS, HIDDEN, seed),
            gate: Brain::new(gpu, 1, HIDDEN, seed ^ 0x6A7E),
            shared: [
                mx(INTER, HIDDEN, 0x51),
                mx(INTER, HIDDEN, 0x52),
                mx(HIDDEN, INTER, 0x53),
            ],
        }
    }
}

/// Every buffer either path writes, filled with NaN so a value nobody wrote
/// fails the comparison.
struct Outs {
    logits: DeviceBuffer,
    gate_logit: DeviceBuffer,
    s_gate: DeviceBuffer,
    s_up: DeviceBuffer,
    mid: DeviceBuffer,
    shared_out: DeviceBuffer,
    ids: DeviceBuffer,
    weights: DeviceBuffer,
    bounds: DeviceBuffer,
    active: DeviceBuffer,
    sorted: DeviceBuffer,
    pos: DeviceBuffer,
}

impl Outs {
    fn new(gpu: &Gpu, rows: u32) -> Self {
        let r = rows as usize;
        let cols = (rows * USED) as usize;
        let e = EXPERTS as usize + 1;
        Self {
            logits: sentinel(gpu, r * EXPERTS as usize),
            gate_logit: sentinel(gpu, r),
            s_gate: sentinel(gpu, r * INTER as usize),
            s_up: sentinel(gpu, r * INTER as usize),
            mid: sentinel(gpu, r * INTER as usize),
            shared_out: sentinel(gpu, r * HIDDEN as usize),
            ids: sentinel(gpu, cols),
            weights: sentinel(gpu, cols),
            bounds: sentinel(gpu, e),
            active: sentinel(gpu, e),
            sorted: sentinel(gpu, cols),
            pos: sentinel(gpu, cols),
        }
    }
}

/// Launches `entry` of `module` over a one-dimensional grid.
fn launch(
    gpu: &Gpu,
    module: &'static str,
    entry: &'static str,
    blocks: u32,
    threads: u32,
    args: &[KernelArg<'_>],
) {
    let mut launch = LaunchArgs::new(
        Extent3::new(blocks, 1, 1),
        Extent3::new(threads, 1, 1),
        args,
    );
    if module == MODULE && throttled(args) {
        launch.smem_bytes = SMEM;
    }
    run(gpu, KernelRef::new(module, entry), &launch);
}

/// A brain-float step projection as `matmul::project_one` plans it.
fn bf16_step(gpu: &Gpu, out: &DeviceBuffer, w: &Brain, xs: &DeviceBuffer, out_dim: u32, rows: u32) {
    let splits = bf16_split(HIDDEN, out_dim);
    let (entry, per_warp) = if rows == 1 {
        ("q4e_bf16_gemv_kernel", 1)
    } else {
        ("q4e_bf16_rows_kernel", 2)
    };
    launch(
        gpu,
        "q4e_bf16",
        entry,
        out_dim.div_ceil(WARPS / splits * per_warp),
        WARPS * 32,
        &[
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(w.buf.view()),
            KernelArg::Buffer(xs.view()),
            KernelArg::U32(HIDDEN),
            KernelArg::U32(out_dim),
            KernelArg::U32(splits),
            KernelArg::U32(rows),
        ],
    );
}

/// The launches the fused one replaces, as the engine plans them for the
/// mixed export at `rows` rows.
fn separate(gpu: &Gpu, layer: &Layer, xs: &DeviceBuffer, out: &Outs, rows: u32) {
    bf16_step(gpu, &out.logits, &layer.router, xs, EXPERTS, rows);
    launch(
        gpu,
        "q4e_moe",
        "q4e_moe_route_sort_tiny_kernel",
        1,
        256,
        &[
            KernelArg::Buffer(out.ids.view()),
            KernelArg::Buffer(out.weights.view()),
            KernelArg::Buffer(out.logits.view()),
            KernelArg::Buffer(out.bounds.view()),
            KernelArg::Buffer(out.active.view()),
            KernelArg::Buffer(out.sorted.view()),
            KernelArg::Buffer(out.pos.view()),
            KernelArg::U32(rows),
            KernelArg::U32(EXPERTS),
            KernelArg::U32(USED),
        ],
    );
    // The shared gate and up as one group (`matmul::mxfp8_group_entry`).
    let (group, per_block) = if rows == 1 {
        ("q4e_mxfp8_gemv_group_narrow_kernel", 4)
    } else {
        ("q4e_mxfp8_rows_group_kernel", 16)
    };
    let [s_gate, s_up, s_down] = &layer.shared;
    let mut args = vec![
        KernelArg::Buffer(out.s_gate.view()),
        KernelArg::Buffer(s_gate.codes.view()),
        KernelArg::Buffer(s_gate.scales.view()),
        KernelArg::U32(INTER),
        KernelArg::Buffer(out.s_up.view()),
        KernelArg::Buffer(s_up.codes.view()),
        KernelArg::Buffer(s_up.scales.view()),
        KernelArg::U32(INTER),
    ];
    for _ in 0..2 {
        args.extend([
            KernelArg::Buffer(out.s_gate.view()),
            KernelArg::Buffer(s_gate.codes.view()),
            KernelArg::Buffer(s_gate.scales.view()),
            KernelArg::U32(0),
        ]);
    }
    args.extend([
        KernelArg::Buffer(xs.view()),
        KernelArg::U32(HIDDEN),
        KernelArg::U32(rows),
    ]);
    launch(
        gpu,
        "q4e_mxfp8",
        group,
        2 * INTER.div_ceil(per_block),
        128,
        &args,
    );
    let elems = u64::from(rows * INTER);
    launch(
        gpu,
        "q4e_moe",
        "q4e_swiglu_kernel",
        elems.div_ceil(256) as u32,
        256,
        &[
            KernelArg::Buffer(out.mid.view()),
            KernelArg::Buffer(out.s_gate.view()),
            KernelArg::Buffer(out.s_up.view()),
            KernelArg::U64(elems),
        ],
    );
    let (down, per_block) = if rows == 1 {
        ("q4e_mxfp8_gemv_kernel", 8)
    } else {
        ("q4e_mxfp8_rows_kernel", 16)
    };
    launch(
        gpu,
        "q4e_mxfp8",
        down,
        HIDDEN.div_ceil(per_block),
        128,
        &[
            KernelArg::Buffer(out.shared_out.view()),
            KernelArg::Buffer(s_down.codes.view()),
            KernelArg::Buffer(s_down.scales.view()),
            KernelArg::Buffer(out.mid.view()),
            KernelArg::U32(INTER),
            KernelArg::U32(HIDDEN),
            KernelArg::U32(rows),
        ],
    );
    bf16_step(gpu, &out.gate_logit, &layer.gate, xs, 1, rows);
}

/// Blocks the fused launch takes, mirroring `moe::fused_route`.
fn fused_blocks() -> u32 {
    INTER.div_ceil(WARPS)
        + EXPERTS * bf16_split(HIDDEN, EXPERTS) / WARPS
        + 1
        + HIDDEN.div_ceil(2 * WARPS)
}

/// The fused launch.
fn fused(gpu: &Gpu, layer: &Layer, xs: &DeviceBuffer, out: &Outs, sync: &DeviceBuffer, rows: u32) {
    let [s_gate, s_up, s_down] = &layer.shared;
    launch(
        gpu,
        MODULE,
        if rows == 1 { ROUTE } else { ROUTE_ROWS },
        fused_blocks(),
        WARPS * 32,
        &[
            KernelArg::Buffer(out.logits.view()),
            KernelArg::Buffer(out.gate_logit.view()),
            KernelArg::Buffer(out.mid.view()),
            KernelArg::Buffer(out.shared_out.view()),
            KernelArg::Buffer(out.ids.view()),
            KernelArg::Buffer(out.weights.view()),
            KernelArg::Buffer(out.bounds.view()),
            KernelArg::Buffer(out.active.view()),
            KernelArg::Buffer(out.sorted.view()),
            KernelArg::Buffer(out.pos.view()),
            KernelArg::Buffer(sync.view()),
            KernelArg::Buffer(layer.router.buf.view()),
            KernelArg::Buffer(layer.gate.buf.view()),
            KernelArg::Buffer(s_gate.codes.view()),
            KernelArg::Buffer(s_gate.scales.view()),
            KernelArg::Buffer(s_up.codes.view()),
            KernelArg::Buffer(s_up.scales.view()),
            KernelArg::Buffer(s_down.codes.view()),
            KernelArg::Buffer(s_down.scales.view()),
            KernelArg::Buffer(xs.view()),
            KernelArg::U32(HIDDEN),
            KernelArg::U32(EXPERTS),
            KernelArg::U32(USED),
            KernelArg::U32(INTER),
            KernelArg::U32(bf16_split(HIDDEN, EXPERTS)),
            KernelArg::U32(bf16_split(HIDDEN, 1)),
            KernelArg::U32(rows),
        ],
    );
}

/// A buffer's words as `i32`.
fn ints(gpu: &Gpu, buffer: &DeviceBuffer) -> Vec<i32> {
    gpu.read(buffer)
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Block inputs for `rows` rows: a scale like the post-norm block input's.
fn inputs(rows: u32, seed: u32) -> Vec<f32> {
    signal((rows * HIDDEN) as usize, seed)
        .iter()
        .map(|v| v * 2.0)
        .collect()
}

/// The router's logits held to the host product: single-precision
/// accumulation over 2560 terms and nothing else.
fn logits_close(router: &Brain, x: &[f32], got: &[f32], rows: u32) {
    for t in 0..rows as usize {
        let xr = &x[t * HIDDEN as usize..(t + 1) * HIDDEN as usize];
        for e in 0..EXPERTS as usize {
            let w = &router.values[e * HIDDEN as usize..(e + 1) * HIDDEN as usize];
            let (sum, mag) = w.iter().zip(xr).fold((0.0f64, 0.0f64), |(s, m), (a, b)| {
                let p = f64::from(*a) * f64::from(*b);
                (s + p, m + p.abs())
            });
            let g = f64::from(got[t * EXPERTS as usize + e]);
            assert!(
                (g - sum).abs() <= 1e-5 * mag.max(1e-12),
                "logit ({t}, {e}): {g} against the host's {sum}"
            );
        }
    }
}

/// The fused launch writes, bit for bit, what the launches it replaces write:
/// logits, the gate logit, the routed ids, weights, bounds, active list,
/// sorted columns and their inverse, the shared expert's activated rows and
/// its output. At every row count it takes, over two layers' weights, with
/// its words back at zero after every launch.
///
/// # Panics
///
/// When any output differs, or a word is left behind.
pub fn the_fused_route_is_the_launches_it_replaces_bit_for_bit(gpu: &Gpu) {
    let layers = [Layer::new(gpu, 0xF0), Layer::new(gpu, 0xF1)];
    let sync = gpu.upload(&[0u8; SYNC_WORDS * 4]);
    for rows in 1..=MAX_ROWS {
        for (l, layer) in layers.iter().enumerate() {
            let x = inputs(rows, 0x1D ^ rows ^ ((l as u32) << 8));
            let xs = gpu.upload(&f32_bytes(&x));
            let want = Outs::new(gpu, rows);
            let got = Outs::new(gpu, rows);
            separate(gpu, layer, &xs, &want, rows);
            fused(gpu, layer, &xs, &got, &sync, rows);
            drain(gpu);
            let what = |name: &str| format!("{name} at {rows} rows, layer {l}");
            for (name, a, b) in [
                ("logits", &got.logits, &want.logits),
                ("gate logit", &got.gate_logit, &want.gate_logit),
                ("weights", &got.weights, &want.weights),
                ("shared mid", &got.mid, &want.mid),
                ("shared out", &got.shared_out, &want.shared_out),
            ] {
                same_bits(&what(name), &gpu.read_f32(a), &gpu.read_f32(b));
            }
            let cols = (rows * USED) as usize;
            let want_active = ints(gpu, &want.active);
            let live = usize::try_from(want_active[EXPERTS as usize]).unwrap_or(0);
            assert!(live > 0 && live <= cols, "{}", what("active count"));
            for (name, a, b, n) in [
                ("ids", &got.ids, &want.ids, cols),
                ("bounds", &got.bounds, &want.bounds, EXPERTS as usize + 1),
                ("sorted", &got.sorted, &want.sorted, cols),
                ("pos", &got.pos, &want.pos, cols),
            ] {
                assert_eq!(ints(gpu, a)[..n], ints(gpu, b)[..n], "{}", what(name));
            }
            let got_active = ints(gpu, &got.active);
            assert_eq!(
                got_active[..live],
                want_active[..live],
                "{}",
                what("active")
            );
            assert_eq!(
                got_active[EXPERTS as usize],
                want_active[EXPERTS as usize],
                "{}",
                what("active count")
            );
            assert_eq!(ints(gpu, &sync), [0; SYNC_WORDS], "{}", what("sync words"));
            if l == 0 {
                logits_close(&layer.router, &x, &gpu.read_f32(&got.logits), rows);
            }
        }
    }
}

/// The fused close writes, bit for bit, what `q4e_moe_combine_kernel` then
/// `q4e_shared_add_kernel` write, at every row count, with gate logits of
/// both signs and a large one.
///
/// # Panics
///
/// When any output differs.
pub fn the_fused_close_is_the_combine_and_the_shared_add(gpu: &Gpu) {
    let n_embd = HIDDEN;
    for rows in 1..=MAX_ROWS {
        let cols = rows * USED;
        let down = gpu.upload(&f32_bytes(&signal((cols * n_embd) as usize, 0xC0 ^ rows)));
        let weights = gpu.upload(&f32_bytes(
            &signal(cols as usize, 0xC1 ^ rows)
                .iter()
                .map(|v| v.abs() * 0.2)
                .collect::<Vec<_>>(),
        ));
        let shared = gpu.upload(&f32_bytes(&signal((rows * n_embd) as usize, 0xC2 ^ rows)));
        let logit: Vec<f32> = (0..rows)
            .map(|t| [-3.5f32, 0.25, 12.0, -0.75][t as usize % 4])
            .collect();
        let logit = gpu.upload(&f32_bytes(&logit));
        let want = sentinel(gpu, (rows * n_embd) as usize);
        let got = sentinel(gpu, (rows * n_embd) as usize);
        let grid = Extent3::new(n_embd.div_ceil(256), rows, 1);
        let block = Extent3::new(256, 1, 1);
        run(
            gpu,
            KernelRef::new("q4e_moe", "q4e_moe_combine_kernel"),
            &LaunchArgs::new(
                grid,
                block,
                &[
                    KernelArg::Buffer(want.view()),
                    KernelArg::Buffer(down.view()),
                    KernelArg::Buffer(weights.view()),
                    KernelArg::U32(n_embd),
                    KernelArg::U32(USED),
                ],
            ),
        );
        run(
            gpu,
            KernelRef::new("q4e_moe", "q4e_shared_add_kernel"),
            &LaunchArgs::new(
                grid,
                block,
                &[
                    KernelArg::Buffer(want.view()),
                    KernelArg::Buffer(shared.view()),
                    KernelArg::Buffer(logit.view()),
                    KernelArg::U32(n_embd),
                ],
            ),
        );
        run(
            gpu,
            KernelRef::new(MODULE, "q4e_ffn_combine_kernel"),
            &LaunchArgs::new(
                grid,
                block,
                &[
                    KernelArg::Buffer(got.view()),
                    KernelArg::Buffer(down.view()),
                    KernelArg::Buffer(weights.view()),
                    KernelArg::Buffer(shared.view()),
                    KernelArg::Buffer(logit.view()),
                    KernelArg::U32(n_embd),
                    KernelArg::U32(USED),
                ],
            ),
        );
        drain(gpu);
        same_bits(
            &format!("the close at {rows} rows"),
            &gpu.read_f32(&got),
            &gpu.read_f32(&want),
        );
    }
}

// --- bench -------------------------------------------------------------------

/// Layers of weights a bench fixture cycles through: 24 x 7.7 MB, far past
/// the last-level cache, so every launch reads its weights from memory as a
/// decode step does.
const BENCH_LAYERS: u32 = 24;

/// The fused launch at `rows` rows over [`BENCH_LAYERS`] layers.
fn bench(gpu: &Gpu, rows: u32) -> Fixture {
    let layers: Vec<Layer> = (0..BENCH_LAYERS)
        .map(|l| Layer::new(gpu, 0xB0 + l))
        .collect();
    let x = gpu.upload(&f32_bytes(&inputs(rows, 0xBE)));
    let outs = Outs::new(gpu, rows);
    let sync = gpu.upload(&[0u8; SYNC_WORDS * 4]);
    let per_layer =
        u64::from(EXPERTS * HIDDEN * 2 + HIDDEN * 2) + 3 * u64::from(INTER * HIDDEN) * 33 / 32;
    let next = std::cell::Cell::new(0usize);
    Fixture::new(per_layer, 0, move |gpu| {
        let l = next.get() % layers.len();
        next.set(next.get().wrapping_add(1));
        fused(gpu, &layers[l], &x, &outs, &sync, rows);
    })
}

/// The fused launch at one decode row.
pub(crate) fn bench_route_t1(gpu: &Gpu) -> Fixture {
    bench(gpu, 1)
}

/// The fused launch at a five-row verify.
pub(crate) fn bench_route_t5(gpu: &Gpu) -> Fixture {
    bench(gpu, 5)
}
