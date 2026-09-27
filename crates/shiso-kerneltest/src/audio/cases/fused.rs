//! The decoder step's fused entries, each against the chain of plain entries
//! it replaces: the fused launch must give the chain's values exactly, so a
//! decoder may take either. The plain entries' own cases hold the chain to
//! its oracle.

use bytemuck::{Pod, Zeroable};
use shiso_core::{DeviceBuffer, Extent3, KernelArg};

use super::{
    ARGMAX_BLOCK, AxpyArgs, GemmArgs, KvAppendArgs, LINEAR_BLOCK, MATVEC_BLOCK, MATVEC_COLS,
    NORM_SMEM, NormArgs, RopeArgs, SiluMulArgs, blocks, decode_state, launch, weight_fixture,
    zeros,
};
use crate::{Gpu, drain, f32_bytes, oracle};

/// `audio_args_matvec3`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(super) struct Matvec3Args {
    pub(super) n0: u32,
    pub(super) n1: u32,
    pub(super) n2: u32,
    pub(super) k: u32,
}
const _: () = assert!(size_of::<Matvec3Args>() == 16);

/// `audio_args_qk_rope`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct QkRopeArgs {
    rows: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    cap: u32,
    eps: f32,
    theta: f32,
}
const _: () = assert!(size_of::<QkRopeArgs>() == 28);

/// `audio_args_argmax_split`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ArgmaxSplitArgs {
    cols: u32,
    span: u32,
    parts: u32,
}
const _: () = assert!(size_of::<ArgmaxSplitArgs>() == 12);

/// Columns one threadgroup of the split argmax scans, as the host cuts rows.
const ARGMAX_SPAN: u32 = 2048;

/// Equal value for value; the fused launch forms each one as the chain does,
/// so no tolerance. A zero's sign is the one thing either may pick.
pub(super) fn same(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let zero = |v: f32| v.to_bits().trailing_zeros() >= 31;
        assert!(
            g.to_bits() == w.to_bits() || (zero(g) && zero(w)) || (g.is_nan() && w.is_nan()),
            "{what}[{i}]: got {g}, the chain gave {w}"
        );
    }
}

fn norm(gpu: &Gpu, entry: &'static str, args: NormArgs, bufs: &[&DeviceBuffer]) {
    let mut kargs = vec![KernelArg::Bytes(bytemuck::bytes_of(&args))];
    kargs.extend(bufs.iter().map(|b| KernelArg::Buffer(b.view())));
    launch(
        gpu,
        entry,
        Extent3::linear(args.rows),
        Extent3::linear(LINEAR_BLOCK),
        NORM_SMEM,
        &kargs,
    );
}

/// The residual add and RMS norm in one launch equal the axpy at alpha 1
/// then the RMS norm: rows that are and are not a whole number of the
/// threads' eight-value runs, with and without the weight, the output bound
/// to the added rows.
pub fn add_rms_norm_is_the_axpy_and_rms_norm_it_replaces(gpu: &Gpu) {
    for &(rows, dim) in &[(3usize, 2048usize), (2, 1000)] {
        let x = oracle::signal(rows * dim, 301);
        let add = oracle::signal(rows * dim, 302);
        let w = oracle::signal(dim, 303);
        let wb = gpu.upload(&f32_bytes(&w));
        for flags in [1u32, 0] {
            let args = NormArgs {
                rows: rows as u32,
                dim: dim as u32,
                eps: 1e-6,
                flags,
            };
            let (x1, a1) = (gpu.upload(&f32_bytes(&x)), gpu.upload(&f32_bytes(&add)));
            let out1 = zeros(gpu, rows * dim);
            let axpy = AxpyArgs {
                count: (rows * dim) as u32,
                alpha: 1.0,
            };
            launch(
                gpu,
                "kernel_audio_axpy_f32",
                Extent3::linear(blocks(rows * dim, LINEAR_BLOCK)),
                Extent3::linear(LINEAR_BLOCK),
                0,
                &[
                    KernelArg::Bytes(bytemuck::bytes_of(&axpy)),
                    KernelArg::Buffer(x1.view()),
                    KernelArg::Buffer(a1.view()),
                ],
            );
            norm(gpu, "kernel_audio_rms_norm_f32", args, &[&out1, &x1, &wb]);
            let (x2, a2) = (gpu.upload(&f32_bytes(&x)), gpu.upload(&f32_bytes(&add)));
            norm(
                gpu,
                "kernel_audio_add_rms_norm_f32",
                args,
                &[&a2, &x2, &a2, &wb],
            );
            drain(gpu);
            let what = format!("add + rms norm {rows}x{dim} flags {flags}");
            same(&gpu.read_f32(&a2), &gpu.read_f32(&out1), &what);
            same(
                &gpu.read_f32(&x2),
                &gpu.read_f32(&x1),
                &format!("{what}: residual"),
            );
        }
    }
}

/// The shapes of one fused norm, rotation and append case.
struct QkCase {
    rows: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    cap: usize,
    theta: f32,
}

/// The plain chain the fused norm, rotation and append replaces: the query
/// rows, then the two cache planes it leaves.
fn qk_chain(
    gpu: &Gpu,
    c: &QkCase,
    [q, k, plane]: [&[f32]; 3],
    [qnb, knb, vb, state]: [&DeviceBuffer; 4],
) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer) {
    let QkCase {
        rows,
        heads,
        kv_heads,
        hd,
        cap,
        theta,
    } = *c;
    let (q1, k1) = (gpu.upload(&f32_bytes(q)), gpu.upload(&f32_bytes(k)));
    let (kc1, vc1) = (gpu.upload(&f32_bytes(plane)), gpu.upload(&f32_bytes(plane)));
    for (buf, w, n) in [(&q1, &qnb, heads), (&k1, &knb, kv_heads)] {
        let args = NormArgs {
            rows: (rows * n) as u32,
            dim: hd as u32,
            eps: 1e-6,
            flags: 1,
        };
        norm(gpu, "kernel_audio_rms_norm_f32", args, &[buf, buf, w]);
        let rope = RopeArgs {
            rows: rows as u32,
            heads: n as u32,
            head_dim: hd as u32,
            pos0: 0,
            theta,
        };
        launch(
            gpu,
            "kernel_audio_rope_half_state_f32",
            Extent3::linear(blocks(rows * n * hd / 2, LINEAR_BLOCK)),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&rope)),
                KernelArg::Buffer(buf.view()),
                KernelArg::Buffer(state.view()),
            ],
        );
    }
    let append = KvAppendArgs {
        rows: rows as u32,
        kv_heads: kv_heads as u32,
        head_dim: hd as u32,
        cap: cap as u32,
    };
    launch(
        gpu,
        "kernel_audio_kv_append_f32",
        Extent3::linear(blocks(rows * kv_heads * hd, LINEAR_BLOCK)),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&append)),
            KernelArg::Buffer(kc1.view()),
            KernelArg::Buffer(vc1.view()),
            KernelArg::Buffer(k1.view()),
            KernelArg::Buffer(vb.view()),
            KernelArg::Buffer(state.view()),
        ],
    );
    (q1, kc1, vc1)
}

/// The query and key norms, the rotation and the append in one launch equal
/// the RMS norm over each head, the state-positioned rotation and the append:
/// grouped heads, one row and two, and a second row past the cache's end
/// that neither writes.
pub fn qk_norm_rope_append_is_the_chain_it_replaces(gpu: &Gpu) {
    let (heads, kv_heads, hd, cap, theta) = (16usize, 8usize, 128usize, 20usize, 1.0e6f32);
    for &(rows, pos) in &[(1usize, 5u32), (2, 19)] {
        let q = oracle::signal(rows * heads * hd, 311);
        let k = oracle::signal(rows * kv_heads * hd, 312);
        let v = oracle::signal(rows * kv_heads * hd, 313);
        let (qn, kn) = (oracle::signal(hd, 314), oracle::signal(hd, 315));
        let plane = oracle::signal(kv_heads * cap * hd, 316);
        let (qnb, knb) = (gpu.upload(&f32_bytes(&qn)), gpu.upload(&f32_bytes(&kn)));
        let vb = gpu.upload(&f32_bytes(&v));
        let state = decode_state(gpu, pos, 0);
        let (q1, kc1, vc1) = qk_chain(
            gpu,
            &QkCase {
                rows,
                heads,
                kv_heads,
                hd,
                cap,
                theta,
            },
            [&q, &k, &plane],
            [&qnb, &knb, &vb, &state],
        );
        // The fused launch.
        let (q2, k2) = (gpu.upload(&f32_bytes(&q)), gpu.upload(&f32_bytes(&k)));
        let (kc2, vc2) = (
            gpu.upload(&f32_bytes(&plane)),
            gpu.upload(&f32_bytes(&plane)),
        );
        let args = QkRopeArgs {
            rows: rows as u32,
            heads: heads as u32,
            kv_heads: kv_heads as u32,
            head_dim: hd as u32,
            cap: cap as u32,
            eps: 1e-6,
            theta,
        };
        launch(
            gpu,
            "kernel_audio_qk_norm_rope_append_f32",
            Extent3::new((heads + kv_heads) as u32, rows as u32, 1),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(q2.view()),
                KernelArg::Buffer(k2.view()),
                KernelArg::Buffer(vb.view()),
                KernelArg::Buffer(qnb.view()),
                KernelArg::Buffer(knb.view()),
                KernelArg::Buffer(kc2.view()),
                KernelArg::Buffer(vc2.view()),
                KernelArg::Buffer(state.view()),
            ],
        );
        drain(gpu);
        let what = format!("{rows} rows at {pos}");
        same(
            &gpu.read_f32(&q2),
            &gpu.read_f32(&q1),
            &format!("{what}: q"),
        );
        same(
            &gpu.read_f32(&kc2),
            &gpu.read_f32(&kc1),
            &format!("{what}: k cache"),
        );
        same(
            &gpu.read_f32(&vc2),
            &gpu.read_f32(&vc1),
            &format!("{what}: v cache"),
        );
        same(
            &gpu.read_f32(&k2),
            &k,
            &format!("{what}: the keys are left alone"),
        );
    }
}

fn matvec(
    gpu: &Gpu,
    kind: &str,
    y: &DeviceBuffer,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    n: usize,
    k: usize,
) {
    let entry = match kind {
        "f32w" => "kernel_audio_matvec_f32w",
        "bf16w" => "kernel_audio_matvec_bf16w",
        "f16w" => "kernel_audio_matvec_f16w",
        _ => "kernel_audio_matvec_q8_0w",
    };
    let args = GemmArgs {
        m: 1,
        n: n as u32,
        k: k as u32,
        flags: 0,
        k_chunk: k as u32,
        splits: 1,
    };
    launch(
        gpu,
        entry,
        Extent3::new(blocks(n, MATVEC_COLS), 1, 1),
        Extent3::linear(MATVEC_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(y.view()),
            KernelArg::Buffer(x.view()),
            KernelArg::Buffer(w.view()),
            KernelArg::Buffer(x.view()),
        ],
    );
}

/// Three products in one launch equal three one-product launches, for every
/// weight type and column counts that end mid-threadgroup.
pub fn matvec3_is_three_matvecs(gpu: &Gpu) {
    let (k, ns) = (1152usize, [37usize, 16, 9]);
    let x = gpu.upload(&f32_bytes(&oracle::signal(k, 321)));
    for kind in ["f32w", "bf16w", "f16w", "q8_0w"] {
        let ws: Vec<DeviceBuffer> = ns
            .iter()
            .enumerate()
            .map(|(i, &n)| gpu.upload(&weight_fixture(kind, n * k, 322 + i as u32).0))
            .collect();
        let ones: Vec<DeviceBuffer> = ns.iter().map(|&n| zeros(gpu, n)).collect();
        for ((y, w), &n) in ones.iter().zip(&ws).zip(&ns) {
            matvec(gpu, kind, y, &x, w, n, k);
        }
        let threes: Vec<DeviceBuffer> = ns.iter().map(|&n| zeros(gpu, n)).collect();
        let args = Matvec3Args {
            n0: ns[0] as u32,
            n1: ns[1] as u32,
            n2: ns[2] as u32,
            k: k as u32,
        };
        let groups: usize = ns.iter().map(|n| n.div_ceil(MATVEC_COLS as usize)).sum();
        let entry = match kind {
            "f32w" => "kernel_audio_matvec3_f32w",
            "bf16w" => "kernel_audio_matvec3_bf16w",
            "f16w" => "kernel_audio_matvec3_f16w",
            _ => "kernel_audio_matvec3_q8_0w",
        };
        launch(
            gpu,
            entry,
            Extent3::linear(groups as u32),
            Extent3::linear(MATVEC_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(threes[0].view()),
                KernelArg::Buffer(threes[1].view()),
                KernelArg::Buffer(threes[2].view()),
                KernelArg::Buffer(x.view()),
                KernelArg::Buffer(ws[0].view()),
                KernelArg::Buffer(ws[1].view()),
                KernelArg::Buffer(ws[2].view()),
            ],
        );
        drain(gpu);
        for (i, (a, b)) in threes.iter().zip(&ones).enumerate() {
            same(
                &gpu.read_f32(a),
                &gpu.read_f32(b),
                &format!("matvec3 {kind} part {i}"),
            );
        }
    }
}

/// The gated pair in one launch equals the two products and the gated
/// product kernel, for every weight type, one row and two.
pub fn matvec_silu_is_the_products_and_silu_mul(gpu: &Gpu) {
    let (n, k) = (37usize, 1152usize);
    for rows in [1usize, 2] {
        let xs = oracle::signal(rows * k, 331);
        for kind in ["f32w", "bf16w", "f16w", "q8_0w"] {
            let wg = gpu.upload(&weight_fixture(kind, n * k, 332).0);
            let wu = gpu.upload(&weight_fixture(kind, n * k, 333).0);
            let mut want = Vec::new();
            for r in 0..rows {
                let x = gpu.upload(&f32_bytes(&xs[r * k..(r + 1) * k]));
                let (g, u) = (zeros(gpu, n), zeros(gpu, n));
                matvec(gpu, kind, &g, &x, &wg, n, k);
                matvec(gpu, kind, &u, &x, &wu, n, k);
                let args = SiluMulArgs { count: n as u32 };
                launch(
                    gpu,
                    "kernel_audio_silu_mul_f32",
                    Extent3::linear(blocks(n, LINEAR_BLOCK)),
                    Extent3::linear(LINEAR_BLOCK),
                    0,
                    &[
                        KernelArg::Bytes(bytemuck::bytes_of(&args)),
                        KernelArg::Buffer(g.view()),
                        KernelArg::Buffer(g.view()),
                        KernelArg::Buffer(u.view()),
                    ],
                );
                drain(gpu);
                want.extend(gpu.read_f32(&g));
            }
            let x = gpu.upload(&f32_bytes(&xs));
            let y = zeros(gpu, rows * n);
            let args = GemmArgs {
                m: rows as u32,
                n: n as u32,
                k: k as u32,
                flags: 0,
                k_chunk: k as u32,
                splits: 1,
            };
            let entry = match kind {
                "f32w" => "kernel_audio_matvec_silu_f32w",
                "bf16w" => "kernel_audio_matvec_silu_bf16w",
                "f16w" => "kernel_audio_matvec_silu_f16w",
                _ => "kernel_audio_matvec_silu_q8_0w",
            };
            launch(
                gpu,
                entry,
                Extent3::new(blocks(n, MATVEC_COLS), rows as u32, 1),
                Extent3::linear(MATVEC_BLOCK),
                0,
                &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(y.view()),
                    KernelArg::Buffer(x.view()),
                    KernelArg::Buffer(wg.view()),
                    KernelArg::Buffer(wu.view()),
                ],
            );
            drain(gpu);
            same(
                &gpu.read_f32(&y),
                &want,
                &format!("matvec_silu {kind} {rows} rows"),
            );
        }
    }
}

/// The two-launch argmax of a wide row answers what a strictly-greater scan
/// from index 0 answers: a tie across two threadgroups' spans keeps the lower
/// index, a NaN never wins, the maximum in the ragged last span is found, and
/// a row with nothing above negative infinity answers 0.
///
/// # Panics
///
/// When the device's answer is not the scan's.
pub fn argmax_wide_matches_the_row_argmax(gpu: &Gpu) {
    let cols = 20_000usize;
    let base = oracle::signal(cols, 341);
    let mut rows: Vec<Vec<f32>> = Vec::new();
    let mut tie = base.clone();
    tie[9000] = 50.0;
    tie[700] = 50.0;
    rows.push(tie);
    let mut nan = base.clone();
    nan[0] = f32::NAN;
    nan[4097] = f32::NAN;
    nan[cols - 1] = 60.0;
    rows.push(nan);
    rows.push(vec![f32::NEG_INFINITY; cols]);
    let mut inside = base;
    inside[2048 + 3] = 70.0;
    inside[2048 + 3 + ARGMAX_BLOCK as usize] = 70.0;
    rows.push(inside);
    let parts = cols.div_ceil(ARGMAX_SPAN as usize);
    for (r, row) in rows.iter().enumerate() {
        let input = gpu.upload(&f32_bytes(row));
        let scratch = zeros(gpu, 2 * parts);
        let out = zeros(gpu, 1);
        let args = ArgmaxSplitArgs {
            cols: cols as u32,
            span: ARGMAX_SPAN,
            parts: parts as u32,
        };
        launch(
            gpu,
            "kernel_audio_argmax_split_f32",
            Extent3::linear(parts as u32),
            Extent3::linear(ARGMAX_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(scratch.view()),
                KernelArg::Buffer(input.view()),
            ],
        );
        launch(
            gpu,
            "kernel_audio_argmax_merge_f32",
            Extent3::linear(1),
            Extent3::linear(ARGMAX_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(scratch.view()),
            ],
        );
        drain(gpu);
        let mut best = (0u32, f32::NEG_INFINITY);
        for (i, &v) in row.iter().enumerate() {
            if v > best.1 {
                best = (i as u32, v);
            }
        }
        let got = gpu.read_f32(&out)[0].to_bits();
        assert_eq!(got, best.0, "wide argmax row {r}");
    }
}
