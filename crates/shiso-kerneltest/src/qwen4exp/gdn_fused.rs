//! The fused prompt path of the sparse family's recurrent block
//! (`q4e_gdn_fused`) against the chain it replaces.
//!
//! The chain is what a prefill chunk ran before the fused pair: the parallel
//! convolution and its window rewrite, the two L2 norms, the gates, the
//! tensor-core prep and scan, and the output gate. The fused pair claims the
//! chain's bits - every gated output, the carried state, the convolution window
//! and the snapshot a later prompt resumes from - so these cases compare bits,
//! not values within a budget: a case that passed on "close" would let the two
//! paths drift apart one rounding at a time, and the engine picks between them
//! by which kernels a build carries, not by anything about the request.
//!
//! The shape is the real model's: 48 value heads reading 16 key heads, a
//! 128-wide head, a four-tap convolution over 10240 channels. Each launch takes
//! the grid, block and shared memory `shiso_engine::sparse::kernels` derives for
//! its entry, restated here because this crate does not depend on the engine.
//! The inputs are the fixture signal scaled to the ranges the projections
//! produce, with decay bases spanning `-e^{2.5} .. -1` so that some heads
//! forget within a run and others barely at all.
#![expect(
    clippy::cast_possible_truncation,
    reason = "kernel arguments are u32 while the fixtures index in usize; every \
              dimension here is a literal far below 2^32"
)]

use shiso_core::{DeviceBuffer, DeviceView, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::bench::Fixture;
use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// The fused pair's module, mirroring `kernels::MOD_GDN_FUSED`.
const FUSED: &str = "q4e_gdn_fused";
/// The chain's ported module, `kernels::MOD_GDN`.
const GDN: &str = "q4e_gdn";
/// The chain's tensor-core module, `kernels::MOD_GDN_TC`.
const TC: &str = "q4e_gdn_tc";
/// Value heads.
const V_HEADS: usize = 48;
/// Query/key heads.
const QK_HEADS: usize = 16;
/// Width of one head.
const HEAD_DIM: usize = 128;
/// Convolution taps.
const CONV_KERNEL: usize = 4;
/// Width of the query (or key) block.
const QK_DIM: usize = QK_HEADS * HEAD_DIM;
/// Channels of the fused q|k|v row.
const IN_DIM: usize = 2 * QK_DIM + V_HEADS * HEAD_DIM;
/// Width of the value (and output) block.
const V_DIM: usize = V_HEADS * HEAD_DIM;
/// Tokens of one run, `kernels::GDN_TC_RUN`.
const RUN: usize = 64;
/// The output norm's epsilon, the model's.
const EPS: f32 = 1e-6;
/// `kernels::BLOCK_LINEAR`.
const BLOCK_LINEAR: u32 = 256;
/// `kernels::BLOCK_NARROW`.
const BLOCK_NARROW: u32 = 128;
/// `kernels::gdn_tc_prep`'s shared memory.
const TC_PREP_SMEM: u32 = (2 * RUN * 136 * 2 + RUN * 68 * 4 + 3 * RUN * 4) as u32;
/// `kernels::gdn_tc_scan`'s shared memory.
const TC_SCAN_SMEM: u32 = (HEAD_DIM * 136 * 2 + 2 * RUN * 136 * 2 + 2 * RUN * 4) as u32;
/// `kernels::GDN_FUSED_PREP_SMEM`.
const FUSED_PREP_SMEM: u32 = (2 * RUN * 136 * 2 + 3 * RUN * 68 * 4 + 2 * 3 * RUN * 4) as u32;
/// `kernels::GDN_FUSED_SCAN_SMEM`.
const FUSED_SCAN_SMEM: u32 =
    (2 * RUN * HEAD_DIM * 4 + RUN * HEAD_DIM * 2 + 2 * RUN * RUN * 2 + 2 * RUN * 4) as u32;
/// The engine's absent pointer (`Operand::Null` binds as `KernelArg::U64(0)`).
const NULL: KernelArg<'static> = KernelArg::U64(0);

/// Runs of `rows` rows.
const fn runs(rows: usize) -> usize {
    rows.div_ceil(RUN)
}

/// Blocks of `per` covering `n`.
fn blocks_for(n: usize, per: u32) -> u32 {
    n.div_ceil(per as usize) as u32
}

/// One prompt's inputs to the block, everything after its projections.
struct Inputs {
    rows: usize,
    /// The fused projection's rows, `[rows][IN_DIM]`.
    x: DeviceBuffer,
    /// The convolution weight, `[IN_DIM][CONV_KERNEL]`.
    conv_w: DeviceBuffer,
    /// The decay and write-strength projections, `[rows][V_HEADS]`.
    alpha: DeviceBuffer,
    beta: DeviceBuffer,
    /// Decay bias and base, `[V_HEADS]`.
    dt_bias: DeviceBuffer,
    a: DeviceBuffer,
    /// The gate projection, `[rows][V_DIM]`, and the output norm's weight.
    z: DeviceBuffer,
    norm_w: DeviceBuffer,
    /// The session's window and state before the prompt.
    conv_state: Vec<f32>,
    state: Vec<f32>,
}

impl Inputs {
    fn new(gpu: &Gpu, rows: usize, seed: u32) -> Self {
        let scaled = |len: usize, s: u32, k: f32| -> Vec<f32> {
            signal(len, seed.wrapping_add(s))
                .iter()
                .map(|v| v * k)
                .collect()
        };
        let a: Vec<f32> = signal(V_HEADS, seed.wrapping_add(6))
            .iter()
            .map(|v| -(v.abs() * 2.5).exp())
            .collect();
        let norm_w: Vec<f32> = signal(HEAD_DIM, seed.wrapping_add(8))
            .iter()
            .map(|v| 1.0 + 0.5 * v)
            .collect();
        Self {
            rows,
            x: gpu.upload(&f32_bytes(&scaled(rows * IN_DIM, 1, 1.5))),
            conv_w: gpu.upload(&f32_bytes(&scaled(IN_DIM * CONV_KERNEL, 2, 0.5))),
            alpha: gpu.upload(&f32_bytes(&scaled(rows * V_HEADS, 3, 3.0))),
            beta: gpu.upload(&f32_bytes(&scaled(rows * V_HEADS, 4, 3.0))),
            dt_bias: gpu.upload(&f32_bytes(&scaled(V_HEADS, 5, 1.0))),
            a: gpu.upload(&f32_bytes(&a)),
            z: gpu.upload(&f32_bytes(&scaled(rows * V_DIM, 7, 4.0))),
            norm_w: gpu.upload(&f32_bytes(&norm_w)),
            conv_state: scaled((CONV_KERNEL - 1) * IN_DIM, 9, 1.5),
            state: scaled(V_HEADS * HEAD_DIM * HEAD_DIM, 10, 0.1),
        }
    }
}

/// What one path leaves behind.
#[derive(Debug)]
struct Outcome {
    gated: Vec<f32>,
    state: Vec<f32>,
    conv: Vec<f32>,
    /// The state and window snapshots, when one was asked for.
    snap: Option<(Vec<f32>, Vec<f32>)>,
}

/// The device side of one path's run: the session's window and state, the
/// gated output, and the snapshot slabs.
struct Session {
    conv: DeviceBuffer,
    state: DeviceBuffer,
    gated: DeviceBuffer,
    snap_state: DeviceBuffer,
    snap_conv: DeviceBuffer,
}

impl Session {
    fn new(gpu: &Gpu, inp: &Inputs) -> Self {
        Self {
            conv: gpu.upload(&f32_bytes(&inp.conv_state)),
            state: gpu.upload(&f32_bytes(&inp.state)),
            gated: gpu.upload(&f32_bytes(&vec![f32::NAN; inp.rows * V_DIM])),
            snap_state: gpu.upload(&f32_bytes(&vec![f32::NAN; inp.state.len()])),
            snap_conv: gpu.upload(&f32_bytes(&vec![f32::NAN; inp.conv_state.len()])),
        }
    }

    fn finish(&self, gpu: &Gpu, snapped: bool) -> Outcome {
        drain(gpu);
        Outcome {
            gated: gpu.read_f32(&self.gated),
            state: gpu.read_f32(&self.state),
            conv: gpu.read_f32(&self.conv),
            snap: snapped.then(|| {
                (
                    gpu.read_f32(&self.snap_state),
                    gpu.read_f32(&self.snap_conv),
                )
            }),
        }
    }
}

/// `width` floats a row of `buf`, from row `r0` for `n` rows.
fn rows_of(buf: &DeviceBuffer, width: usize, r0: usize, n: usize) -> DeviceView<'_> {
    buf.slice((r0 * width * 4) as u64, (n * width * 4) as u64)
        .expect("the rows lie inside the buffer")
}

/// One launch of `entry` in `module`.
fn launch(
    gpu: &Gpu,
    module: &'static str,
    entry: &'static str,
    geometry: (Extent3, u32, u32),
    args: &[KernelArg<'_>],
) {
    let (grid, block, smem_bytes) = geometry;
    run(
        gpu,
        KernelRef::new(module, entry),
        &LaunchArgs {
            grid,
            block: Extent3::linear(block),
            smem_bytes,
            args,
        },
    );
}

/// The window rewrite both paths end a chunk with (and, cut at a snapshot
/// row, the window snapshot): `q4e_gdn_conv_tail_kernel`.
fn conv_tail(gpu: &Gpu, window: DeviceView<'_>, x: DeviceView<'_>, n: usize) {
    launch(
        gpu,
        TC,
        "q4e_gdn_conv_tail_kernel",
        (
            Extent3::linear(blocks_for(IN_DIM, BLOCK_LINEAR)),
            BLOCK_LINEAR,
            0,
        ),
        &[
            KernelArg::Buffer(window),
            KernelArg::Buffer(x),
            KernelArg::U32(IN_DIM as u32),
            KernelArg::U32(CONV_KERNEL as u32),
            KernelArg::U32(n as u32),
        ],
    );
}

/// The chain's scratch, sized for its widest chunk.
struct ChainScratch {
    conv: DeviceBuffer,
    decay: DeviceBuffer,
    beta: DeviceBuffer,
    attn: DeviceBuffer,
    u: DeviceBuffer,
    w: DeviceBuffer,
    p: DeviceBuffer,
    g: DeviceBuffer,
}

impl ChainScratch {
    fn new(gpu: &Gpu, rows: usize) -> Self {
        let zeros = |bytes: usize| gpu.upload(&vec![0u8; bytes]);
        let slots = runs(rows) * V_HEADS;
        Self {
            conv: zeros(rows * IN_DIM * 4),
            decay: zeros(rows * V_HEADS * 4),
            beta: zeros(rows * V_HEADS * 4),
            attn: zeros(rows * V_DIM * 4),
            u: zeros(slots * RUN * HEAD_DIM * 2),
            w: zeros(slots * RUN * HEAD_DIM * 2),
            p: zeros(slots * RUN * RUN * 2),
            g: zeros(slots * RUN * 4),
        }
    }
}

/// One chunk of a prompt: rows `r0 .. r0 + n`, and the row a snapshot is
/// taken before, counted from the chunk's first row.
#[derive(Debug, Clone, Copy)]
struct Chunk {
    r0: usize,
    n: usize,
    snap_row: Option<usize>,
}

impl Chunk {
    /// The snapshot pointer and run a carry is handed: the session's slab and
    /// the snapshot row's run, or the null pair.
    fn snap(self, s: &Session) -> (KernelArg<'_>, u32) {
        match self.snap_row {
            Some(row) => (KernelArg::Buffer(s.snap_state.view()), (row / RUN) as u32),
            None => (NULL, u32::MAX),
        }
    }
}

/// Everything one launch of either path reads and writes.
struct Pass<'a, S> {
    inp: &'a Inputs,
    s: &'a Session,
    sc: &'a S,
    chunk: Chunk,
}

impl<S> Pass<'_, S> {
    /// The chunk's rows of the fused projection.
    fn x(&self) -> DeviceView<'_> {
        rows_of(&self.inp.x, IN_DIM, self.chunk.r0, self.chunk.n)
    }

    /// The chunk's rows of a per-head input `width` floats a row.
    fn rows<'b>(&self, buf: &'b DeviceBuffer, width: usize) -> DeviceView<'b> {
        rows_of(buf, width, self.chunk.r0, self.chunk.n)
    }

    /// The chunk's rows as a kernel argument.
    fn n(&self) -> KernelArg<'static> {
        KernelArg::U32(self.chunk.n as u32)
    }

    /// The window rewrite, and the window snapshot when one is asked for.
    fn conv_tail(&self, gpu: &Gpu) {
        conv_tail(gpu, self.s.conv.view(), self.x(), self.chunk.n);
        if let Some(row) = self.chunk.snap_row {
            conv_tail(gpu, self.s.snap_conv.view(), self.x(), row);
        }
    }
}

impl Pass<'_, ChainScratch> {
    fn conv_par(&self, gpu: &Gpu) {
        launch(
            gpu,
            TC,
            "q4e_gdn_conv_par_kernel",
            (
                Extent3::new(
                    blocks_for(IN_DIM, BLOCK_LINEAR),
                    runs(self.chunk.n) as u32,
                    1,
                ),
                BLOCK_LINEAR,
                0,
            ),
            &[
                KernelArg::Buffer(self.sc.conv.view()),
                KernelArg::Buffer(self.s.conv.view()),
                KernelArg::Buffer(self.x()),
                KernelArg::Buffer(self.inp.conv_w.view()),
                KernelArg::U32(IN_DIM as u32),
                KernelArg::U32(CONV_KERNEL as u32),
                self.n(),
            ],
        );
    }

    fn norms_and_gates(&self, gpu: &Gpu) {
        for offset in [0, QK_DIM] {
            launch(
                gpu,
                GDN,
                "q4e_gdn_l2norm_kernel",
                (
                    Extent3::new(QK_HEADS as u32, self.chunk.n as u32, 1),
                    BLOCK_NARROW,
                    0,
                ),
                &[
                    KernelArg::Buffer(self.sc.conv.view()),
                    KernelArg::U32(HEAD_DIM as u32),
                    KernelArg::U32(QK_HEADS as u32),
                    KernelArg::U32(IN_DIM as u32),
                    KernelArg::U32(offset as u32),
                ],
            );
        }
        launch(
            gpu,
            GDN,
            "q4e_gdn_gates_kernel",
            (
                Extent3::linear(blocks_for(V_HEADS * self.chunk.n, BLOCK_LINEAR)),
                BLOCK_LINEAR,
                0,
            ),
            &[
                KernelArg::Buffer(self.sc.decay.view()),
                KernelArg::Buffer(self.sc.beta.view()),
                KernelArg::Buffer(self.rows(&self.inp.alpha, V_HEADS)),
                KernelArg::Buffer(self.rows(&self.inp.beta, V_HEADS)),
                KernelArg::Buffer(self.inp.dt_bias.view()),
                KernelArg::Buffer(self.inp.a.view()),
                KernelArg::U32(V_HEADS as u32),
                self.n(),
            ],
        );
    }

    fn tc_prep(&self, gpu: &Gpu) {
        launch(
            gpu,
            TC,
            "q4e_gdn_tc_prep_kernel",
            (
                Extent3::new(V_HEADS as u32, runs(self.chunk.n) as u32, 1),
                512,
                TC_PREP_SMEM,
            ),
            &[
                KernelArg::Buffer(self.sc.u.view()),
                KernelArg::Buffer(self.sc.w.view()),
                KernelArg::Buffer(self.sc.p.view()),
                KernelArg::Buffer(self.sc.g.view()),
                KernelArg::Buffer(self.sc.conv.view()),
                KernelArg::Buffer(self.sc.decay.view()),
                KernelArg::Buffer(self.sc.beta.view()),
                KernelArg::U32(QK_HEADS as u32),
                KernelArg::U32(V_HEADS as u32),
                KernelArg::U32(QK_DIM as u32),
                KernelArg::U32(2 * QK_DIM as u32),
                KernelArg::U32(IN_DIM as u32),
                self.n(),
            ],
        );
    }

    fn tc_scan(&self, gpu: &Gpu) {
        let (snap, snap_run) = self.chunk.snap(self.s);
        launch(
            gpu,
            TC,
            "q4e_gdn_tc_scan_kernel",
            (Extent3::linear(V_HEADS as u32), 512, TC_SCAN_SMEM),
            &[
                KernelArg::Buffer(self.sc.attn.view()),
                KernelArg::Buffer(self.s.state.view()),
                KernelArg::Buffer(self.sc.conv.view()),
                KernelArg::Buffer(self.sc.u.view()),
                KernelArg::Buffer(self.sc.w.view()),
                KernelArg::Buffer(self.sc.p.view()),
                KernelArg::Buffer(self.sc.g.view()),
                KernelArg::U32(QK_HEADS as u32),
                KernelArg::U32(V_HEADS as u32),
                KernelArg::U32(QK_DIM as u32),
                KernelArg::U32(IN_DIM as u32),
                self.n(),
                snap,
                KernelArg::U32(snap_run),
            ],
        );
    }

    fn out_gate(&self, gpu: &Gpu) {
        launch(
            gpu,
            GDN,
            "q4e_gdn_out_gate_kernel",
            (
                Extent3::linear((self.chunk.n * V_HEADS) as u32),
                BLOCK_NARROW,
                0,
            ),
            &[
                KernelArg::Buffer(self.rows(&self.s.gated, V_DIM)),
                KernelArg::Buffer(self.sc.attn.view()),
                KernelArg::Buffer(self.rows(&self.inp.z, V_DIM)),
                KernelArg::Buffer(self.inp.norm_w.view()),
                KernelArg::U32(HEAD_DIM as u32),
                KernelArg::U32(V_HEADS as u32),
                KernelArg::F32(EPS),
            ],
        );
    }

    /// The chain over the chunk, in the engine's launch order.
    fn all(&self, gpu: &Gpu) {
        self.conv_par(gpu);
        self.conv_tail(gpu);
        self.norms_and_gates(gpu);
        self.tc_prep(gpu);
        self.tc_scan(gpu);
        self.out_gate(gpu);
    }
}

/// The fused pair's scratch, sized for its widest chunk.
struct FusedScratch {
    u: DeviceBuffer,
    t: DeviceBuffer,
    p: DeviceBuffer,
    ge: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
}

impl FusedScratch {
    fn new(gpu: &Gpu, rows: usize) -> Self {
        let zeros = |bytes: usize| gpu.upload(&vec![0u8; bytes]);
        let slots = runs(rows) * V_HEADS;
        let key_slots = runs(rows) * QK_HEADS;
        Self {
            u: zeros(slots * RUN * HEAD_DIM * 2),
            t: zeros(slots * RUN * RUN * 2),
            p: zeros(slots * RUN * RUN * 2),
            ge: zeros(slots * 2 * RUN * 4),
            q: zeros(key_slots * RUN * HEAD_DIM * 4),
            k: zeros(key_slots * RUN * HEAD_DIM * 4),
        }
    }
}

impl Pass<'_, FusedScratch> {
    fn prep(&self, gpu: &Gpu) {
        launch(
            gpu,
            FUSED,
            "q4e_gdf_prep_kernel",
            (
                Extent3::new(QK_HEADS as u32, runs(self.chunk.n) as u32, 1),
                512,
                FUSED_PREP_SMEM,
            ),
            &[
                KernelArg::Buffer(self.sc.u.view()),
                KernelArg::Buffer(self.sc.t.view()),
                KernelArg::Buffer(self.sc.p.view()),
                KernelArg::Buffer(self.sc.ge.view()),
                KernelArg::Buffer(self.sc.q.view()),
                KernelArg::Buffer(self.sc.k.view()),
                KernelArg::Buffer(self.x()),
                KernelArg::Buffer(self.s.conv.view()),
                KernelArg::Buffer(self.inp.conv_w.view()),
                KernelArg::Buffer(self.rows(&self.inp.alpha, V_HEADS)),
                KernelArg::Buffer(self.rows(&self.inp.beta, V_HEADS)),
                KernelArg::Buffer(self.inp.dt_bias.view()),
                KernelArg::Buffer(self.inp.a.view()),
                KernelArg::U32(QK_HEADS as u32),
                KernelArg::U32(V_HEADS as u32),
                KernelArg::U32(IN_DIM as u32),
                KernelArg::U32(0),
                self.n(),
            ],
        );
    }

    fn scan(&self, gpu: &Gpu) {
        let (snap, snap_run) = self.chunk.snap(self.s);
        launch(
            gpu,
            FUSED,
            "q4e_gdf_scan_kernel",
            (Extent3::linear(V_HEADS as u32), 256, FUSED_SCAN_SMEM),
            &[
                KernelArg::Buffer(self.rows(&self.s.gated, V_DIM)),
                KernelArg::Buffer(self.s.state.view()),
                KernelArg::Buffer(self.sc.u.view()),
                KernelArg::Buffer(self.sc.t.view()),
                KernelArg::Buffer(self.sc.p.view()),
                KernelArg::Buffer(self.sc.ge.view()),
                KernelArg::Buffer(self.sc.q.view()),
                KernelArg::Buffer(self.sc.k.view()),
                KernelArg::Buffer(self.rows(&self.inp.z, V_DIM)),
                KernelArg::Buffer(self.inp.norm_w.view()),
                KernelArg::U32(QK_HEADS as u32),
                KernelArg::U32(V_HEADS as u32),
                KernelArg::U32(0),
                self.n(),
                KernelArg::F32(EPS),
                snap,
                KernelArg::U32(snap_run),
            ],
        );
    }

    /// The fused pair over the chunk, then the window rewrite after it, in
    /// the engine's launch order.
    fn all(&self, gpu: &Gpu) {
        self.prep(gpu);
        self.scan(gpu);
        self.conv_tail(gpu);
    }
}

/// Which path a run takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Path {
    Chain,
    Fused,
}

/// The whole prompt through `path`, cut into chunks of the widths in `cuts`
/// (which sum to the prompt's rows), from the inputs' window and state. A
/// snapshot row is only meaningful on a prompt of one chunk.
fn run_path(
    gpu: &Gpu,
    inp: &Inputs,
    path: Path,
    cuts: &[usize],
    snap_row: Option<usize>,
) -> Outcome {
    assert_eq!(
        cuts.iter().sum::<usize>(),
        inp.rows,
        "the cuts cover the prompt"
    );
    assert!(
        snap_row.is_none() || cuts.len() == 1,
        "a snapshot is a one-chunk case"
    );
    let widest = cuts.iter().copied().max().unwrap_or(0);
    let s = Session::new(gpu, inp);
    let mut r0 = 0;
    match path {
        Path::Chain => {
            let sc = ChainScratch::new(gpu, widest);
            for &n in cuts {
                let chunk = Chunk { r0, n, snap_row };
                Pass {
                    inp,
                    s: &s,
                    sc: &sc,
                    chunk,
                }
                .all(gpu);
                r0 += n;
            }
        }
        Path::Fused => {
            let sc = FusedScratch::new(gpu, widest);
            for &n in cuts {
                let chunk = Chunk { r0, n, snap_row };
                Pass {
                    inp,
                    s: &s,
                    sc: &sc,
                    chunk,
                }
                .all(gpu);
                r0 += n;
            }
        }
    }
    s.finish(gpu, snap_row.is_some())
}

/// The first element whose bits differ, if any.
fn first_bit_difference(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    assert_eq!(a.len(), b.len(), "lengths");
    a.iter()
        .zip(b)
        .enumerate()
        .find(|(_, (x, y))| x.to_bits() != y.to_bits())
        .map(|(i, (&x, &y))| (i, x, y))
}

/// Requires the same bits from `got` as from `want` in every output.
fn same_bits(want: &Outcome, got: &Outcome, what: &str) {
    assert!(
        want.gated.iter().all(|v| v.is_finite()),
        "{what}: the reference wrote every gated output"
    );
    let mut pairs = vec![
        ("gated output", &want.gated, &got.gated),
        ("recurrent state", &want.state, &got.state),
        ("convolution window", &want.conv, &got.conv),
    ];
    if let (Some(a), Some(b)) = (&want.snap, &got.snap) {
        pairs.push(("state snapshot", &a.0, &b.0));
        pairs.push(("window snapshot", &a.1, &b.1));
    }
    for (name, a, b) in pairs {
        if let Some((i, x, y)) = first_bit_difference(a, b) {
            panic!("{what}: {name} element {i} is {y}, expected {x}");
        }
    }
}

/// The fused pair gives the chain's bits - every gated output, the carried
/// state, the window and the snapshot - over prompts of one partial run, of
/// whole runs, and of whole runs and a partial one, with the snapshot taken
/// at the chunk's first run, inside it and at its last.
///
/// # Panics
///
/// When any element of any output differs in its bits.
pub fn fused_prompt_matches_the_chain(gpu: &Gpu) {
    for (rows, snap_row, seed) in [
        (37, Some(0), 11),
        (128, Some(64), 13),
        (200, Some(192), 17),
        (777, Some(320), 19),
    ] {
        let inp = Inputs::new(gpu, rows, seed);
        let chain = run_path(gpu, &inp, Path::Chain, &[rows], snap_row);
        let fused = run_path(gpu, &inp, Path::Fused, &[rows], snap_row);
        same_bits(
            &chain,
            &fused,
            &format!("{rows} rows, snapshot at {snap_row:?}"),
        );
    }
}

/// Where a prompt is cut into chunks on the run grid moves no bit of what the
/// fused pair leaves: every cut gives the bits of one pass over the whole
/// prompt, and that pass gives the chain's.
///
/// # Panics
///
/// When any element of any output differs in its bits.
pub fn fused_prompt_is_chunk_width_invariant(gpu: &Gpu) {
    let rows = 650;
    let inp = Inputs::new(gpu, rows, 23);
    let whole = run_path(gpu, &inp, Path::Fused, &[rows], None);
    same_bits(
        &run_path(gpu, &inp, Path::Chain, &[rows], None),
        &whole,
        "one chunk against the chain",
    );
    for cuts in [
        &[64, 586][..],
        &[128, 192, 330],
        &[256, 384, 10],
        &[576, 64, 10],
        &[64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10],
    ] {
        same_bits(
            &whole,
            &run_path(gpu, &inp, Path::Fused, cuts, None),
            &format!("cut as {cuts:?}"),
        );
    }
}

// --- the benchmarks ---------------------------------------------------------

/// Rows of the benchmark chunk: the profiled prefill chunk.
const BENCH_ROWS: usize = 4_096;

/// The benchmark chunk, whole, with no snapshot.
const WHOLE: Chunk = Chunk {
    r0: 0,
    n: BENCH_ROWS,
    snap_row: None,
};

/// The least traffic either path must move over `rows` rows: the projection's
/// rows and the gate rows read, the gated rows written.
const fn bench_bytes(rows: usize) -> u64 {
    (rows * (IN_DIM + 2 * V_DIM) * 4) as u64
}

/// The chain over one 4096-row chunk, all eight launches.
pub(crate) fn bench_chain_4096(gpu: &Gpu) -> Fixture {
    let inp = Inputs::new(gpu, BENCH_ROWS, 29);
    let s = Session::new(gpu, &inp);
    let sc = ChainScratch::new(gpu, BENCH_ROWS);
    Fixture::new(bench_bytes(BENCH_ROWS), 0, move |gpu| {
        whole(&inp, &s, &sc).all(gpu);
    })
}

/// The fused pair over one 4096-row chunk, with the window rewrite.
pub(crate) fn bench_fused_4096(gpu: &Gpu) -> Fixture {
    let inp = Inputs::new(gpu, BENCH_ROWS, 29);
    let s = Session::new(gpu, &inp);
    let sc = FusedScratch::new(gpu, BENCH_ROWS);
    Fixture::new(bench_bytes(BENCH_ROWS), 0, move |gpu| {
        whole(&inp, &s, &sc).all(gpu);
    })
}

/// The benchmark chunk as one pass over these buffers.
const fn whole<'a, S>(inp: &'a Inputs, s: &'a Session, sc: &'a S) -> Pass<'a, S> {
    Pass {
        inp,
        s,
        sc,
        chunk: WHOLE,
    }
}

/// One launch of the chain, timed alone.
#[derive(Debug, Clone, Copy)]
enum ChainOne {
    ConvPar,
    TcPrep,
    TcScan,
    OutGate,
}

/// One launch of the fused pair, timed alone.
#[derive(Debug, Clone, Copy)]
enum FusedOne {
    Prep,
    Scan,
}

/// One launch of the chain over the benchmark chunk, after the whole chain has
/// run once so that whatever it reads has been written.
fn bench_chain_one(gpu: &Gpu, one: ChainOne) -> Fixture {
    let inp = Inputs::new(gpu, BENCH_ROWS, 29);
    let s = Session::new(gpu, &inp);
    let sc = ChainScratch::new(gpu, BENCH_ROWS);
    whole(&inp, &s, &sc).all(gpu);
    drain(gpu);
    Fixture::new(0, 0, move |gpu| {
        let p = whole(&inp, &s, &sc);
        match one {
            ChainOne::ConvPar => p.conv_par(gpu),
            ChainOne::TcPrep => p.tc_prep(gpu),
            ChainOne::TcScan => p.tc_scan(gpu),
            ChainOne::OutGate => p.out_gate(gpu),
        }
    })
}

/// One launch of the fused pair over the benchmark chunk, likewise.
fn bench_fused_one(gpu: &Gpu, one: FusedOne) -> Fixture {
    let inp = Inputs::new(gpu, BENCH_ROWS, 29);
    let s = Session::new(gpu, &inp);
    let sc = FusedScratch::new(gpu, BENCH_ROWS);
    whole(&inp, &s, &sc).all(gpu);
    drain(gpu);
    Fixture::new(0, 0, move |gpu| {
        let p = whole(&inp, &s, &sc);
        match one {
            FusedOne::Prep => p.prep(gpu),
            FusedOne::Scan => p.scan(gpu),
        }
    })
}

/// `q4e_gdn_conv_par_kernel` alone over the benchmark chunk.
pub(crate) fn bench_conv_par_4096(gpu: &Gpu) -> Fixture {
    bench_chain_one(gpu, ChainOne::ConvPar)
}

/// `q4e_gdn_tc_prep_kernel` alone over the benchmark chunk.
pub(crate) fn bench_tc_prep_4096(gpu: &Gpu) -> Fixture {
    bench_chain_one(gpu, ChainOne::TcPrep)
}

/// `q4e_gdn_tc_scan_kernel` alone over the benchmark chunk.
pub(crate) fn bench_tc_scan_4096(gpu: &Gpu) -> Fixture {
    bench_chain_one(gpu, ChainOne::TcScan)
}

/// `q4e_gdn_out_gate_kernel` alone over the benchmark chunk.
pub(crate) fn bench_out_gate_4096(gpu: &Gpu) -> Fixture {
    bench_chain_one(gpu, ChainOne::OutGate)
}

/// `q4e_gdf_prep_kernel` alone over the benchmark chunk.
pub(crate) fn bench_fused_prep_4096(gpu: &Gpu) -> Fixture {
    bench_fused_one(gpu, FusedOne::Prep)
}

/// `q4e_gdf_scan_kernel` alone over the benchmark chunk.
pub(crate) fn bench_fused_scan_4096(gpu: &Gpu) -> Fixture {
    bench_fused_one(gpu, FusedOne::Scan)
}
