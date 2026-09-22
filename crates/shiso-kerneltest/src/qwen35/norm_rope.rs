//! `q35` fused per-head RMS-norm + partial rope tests.
//!
//! Task 5 (`docs/superpowers/plans/tasks/task-05-qwen35-msl-kernels.md`), Step 9.
//! The fused kernel is compared with an unfused oracle (normalize, then rotate the
//! prefix neox pairs): each reading is obvious on its own, so agreement says the
//! fusion kept both the reduction shape and the pair layout.
//!
//! The `#[test]` wrappers live in each backend's own driver -
//! `crates/shiso-metal/tests/q35_norm_rope.rs`,
//! `crates/shiso-cuda/tests/q35_norm_rope.rs`.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "kernel argument structs take u32 dimensions while the fixtures are written in usize, and every fixture dimension here is a small literal"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use shiso_core::{Extent3, KernelArg, KernelRef, LaunchArgs};

/// Attention `head_dim`: this is the q/k path, not the 128-wide GDN heads.
const HD: usize = 256;
const N_ROT: usize = 64;
const N_HEAD: usize = 4;
const N_HEAD_KV: usize = 4;
const N_TOK: usize = 2;
const BASE: f32 = 1_000_000.0;
const EPS: f32 = 1e-6;
const TOL: f32 = 1e-5;
/// `f16` cache round-trip, per the plan's quantized/`f16` tolerance.
const TOL_F16: f32 = 1e-3;
/// `Q35_BLOCK_SUM_SLOTS` plus the staged row, in bytes.
const SMEM: u32 = ((33 + HD) * 4) as u32;

/// Positions the fixture names: token 0 sits at position 0 (the identity
/// rotation) and token 1 sits at position 3, so the cache slot and the token
// index disagree from the second token onward.
const POS: [i32; N_TOK] = [0, 3];

/// `q35_args_q_norm_rope` and `q35_args_k_norm_rope` share a shape: `head_dim`,
/// head count, `n_rot`, `n_tok`, then the two floats, in field order.
fn rope_args(n_heads: u32) -> Vec<u8> {
    let mut v = Vec::new();
    for field in [HD as u32, n_heads, N_ROT as u32, N_TOK as u32] {
        v.extend(field.to_le_bytes());
    }
    v.extend(BASE.to_le_bytes());
    v.extend(EPS.to_le_bytes());
    v
}

/// Multiples of a quarter: the fixture values are exact in binary, so a
/// difference is the kernel's arithmetic and not the fixture's own error.
fn fixture(len: usize) -> Vec<f32> {
    (0..len).map(|i| ((i % 17) as f32 - 8.0) * 0.25).collect()
}

/// Unfused oracle: RMS-normalize (weight applied), then rotate the first `n_rot`
/// dims as neox pairs.  The tail is copied, exactly as the kernel copies it.
fn norm_rope_oracle(row: &[f32], w: &[f32], pos: f32) -> Vec<f32> {
    let sum: f32 = row.iter().map(|v| v * v).sum();
    let scale = 1.0 / (sum / row.len() as f32 + EPS).sqrt();
    let mut out: Vec<f32> = row
        .iter()
        .zip(w)
        .map(|(v, weight)| v * scale * weight)
        .collect();
    let half = N_ROT / 2;
    for i in 0..half {
        let theta = pos * BASE.powf(-2.0 * (i as f32) / (N_ROT as f32));
        let (s, c) = (theta.sin(), theta.cos());
        let (x0, x1) = (out[i], out[i + half]);
        out[i] = x0 * c - x1 * s;
        out[i + half] = x0 * s + x1 * c;
    }
    out
}

/// `f16` bits to `f32`, for reading back the f16 cache the k path writes.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1F);
    let frac = f32::from(bits & 0x3FF);
    if exp == 0 {
        sign * frac * 2f32.powi(-24)
    } else if exp == 31 {
        if frac == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (frac / 1024.0 + 1.0) * 2f32.powi(exp - 15)
    }
}

/// The decoder above is only a valid oracle if its known answers are right,
/// including the subnormal and infinity edges the cache can hold.
pub fn f16_decoder_matches_known_patterns() {
    let cases: [(u16, f32); 8] = [
        (0x0000, 0.0),
        (0x8000, -0.0),
        (0x3C00, 1.0),
        (0xBC00, -1.0),
        (0x3800, 0.5),
        (0x3E00, 1.5),
        (0x7BFF, 65504.0),
        (0x0001, 5.960_464_5e-8),
    ];
    for (bits, want) in cases {
        assert_eq!(
            f16_to_f32(bits).to_bits(),
            want.to_bits(),
            "bits {bits:#06x}"
        );
    }
    assert!(
        f16_to_f32(0x7C00).is_infinite(),
        "the inf pattern decodes to inf"
    );
}

/// `f16` cache bytes to `f32`, in the order the kernel wrote them.
fn f16_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|word| f16_to_f32(u16::from_le_bytes([word[0], word[1]])))
        .collect()
}

/// `pos` as device `int32_t` values.
fn pos_bytes(values: &[i32]) -> Vec<u8> {
    let mut out = Vec::new();
    for v in values {
        out.extend(v.to_le_bytes());
    }
    out
}

/// The fused q path against the unfused oracle, plus the three properties the
/// fusion is easy to get wrong: pos 0 must be the identity rotation, the tail
/// dims must be copied, and the gate half must be a raw copy of the projection.
pub fn q_norm_rope_matches_the_unfused_oracle(gpu: &Gpu) {
    let weight = vec![1.0f32; HD];
    let qkv_host = fixture(N_TOK * N_HEAD * 2 * HD);
    let qkv = gpu.upload(&f32_bytes(&qkv_host));
    let w = gpu.upload(&f32_bytes(&weight));
    let pos = gpu.upload(&pos_bytes(&POS));
    let q_out = gpu.upload(&f32_bytes(&vec![0.0f32; N_TOK * N_HEAD * HD]));
    let gate = gpu.upload(&f32_bytes(&vec![0.0f32; N_TOK * N_HEAD * HD]));

    let args = rope_args(N_HEAD as u32);
    let params = [
        KernelArg::Bytes(&args),
        KernelArg::Buffer(q_out.view()),
        KernelArg::Buffer(gate.view()),
        KernelArg::Buffer(qkv.view()),
        KernelArg::Buffer(w.view()),
        KernelArg::Buffer(pos.view()),
    ];
    let launch = LaunchArgs {
        grid: Extent3::new(N_HEAD as u32, N_TOK as u32, 1),
        block: Extent3::linear(128),
        smem_bytes: SMEM,
        args: &params,
    };
    run(
        gpu,
        KernelRef::new("qwen35", "kernel_q35_q_norm_rope"),
        &launch,
    );
    drain(gpu);

    let got_q = gpu.read_f32(&q_out);
    let got_gate = gpu.read_f32(&gate);
    for (t, &pos) in POS.iter().enumerate().take(N_TOK) {
        for h in 0..N_HEAD {
            let src = (t * N_HEAD + h) * 2 * HD;
            let dst = (t * N_HEAD + h) * HD;
            let want = norm_rope_oracle(&qkv_host[src..src + HD], &weight, pos as f32);
            for i in 0..HD {
                let got = got_q[dst + i];
                assert!(
                    (got - want[i]).abs() <= TOL,
                    "token {t} head {h} dim {i}: got {got}, want {}",
                    want[i]
                );
            }
            for i in 0..HD {
                assert_eq!(
                    got_gate[dst + i].to_bits(),
                    qkv_host[src + HD + i].to_bits(),
                    "gate copy token {t} head {h} dim {i}"
                );
            }
        }
    }
}

/// The k path writes an f16 cache: normalized + rotated keys at the slot named by
/// `pos`, values copied raw beside them.  Slots the positions never name must stay
/// untouched, which is what proves the slot is the position rather than the token
/// index (the two differ as soon as a cache is reused).
pub fn k_norm_rope_writes_the_f16_cache_at_its_position_slot(gpu: &Gpu) {
    let weight = vec![1.0f32; HD];
    let k_host = fixture(N_TOK * N_HEAD_KV * HD);
    let v_host = fixture(N_TOK * N_HEAD_KV * HD);
    let k = gpu.upload(&f32_bytes(&k_host));
    let v = gpu.upload(&f32_bytes(&v_host));
    let kw = gpu.upload(&f32_bytes(&weight));
    let pos = gpu.upload(&pos_bytes(&[0, 3]));
    let pages = gpu.upload(&page_bytes(&identity_pages(1)));
    // Four slots, of which positions 0 and 3 are the only ones written.
    let slots = 4usize;
    let k_cache = gpu.upload(&vec![0u8; slots * N_HEAD_KV * HD * 2]);
    let v_cache = gpu.upload(&vec![0u8; slots * N_HEAD_KV * HD * 2]);

    let args = kv_rope_args(N_HEAD_KV as u32);
    let params = [
        KernelArg::Bytes(&args),
        KernelArg::Buffer(k_cache.view()),
        KernelArg::Buffer(v_cache.view()),
        KernelArg::Buffer(k.view()),
        KernelArg::Buffer(v.view()),
        KernelArg::Buffer(kw.view()),
        KernelArg::Buffer(pos.view()),
        KernelArg::Buffer(pages.view()),
    ];
    let launch = LaunchArgs {
        grid: Extent3::new(N_HEAD_KV as u32, N_TOK as u32, 1),
        block: Extent3::linear(128),
        smem_bytes: SMEM,
        args: &params,
    };
    run(
        gpu,
        KernelRef::new("qwen35", "kernel_q35_k_norm_rope_kv"),
        &launch,
    );
    drain(gpu);

    let k_cache = f16_slice(&gpu.read(&k_cache));
    let v_cache = f16_slice(&gpu.read(&v_cache));
    for (t, &position) in [0usize, 3].iter().enumerate() {
        for h in 0..N_HEAD_KV {
            let src = (t * N_HEAD_KV + h) * HD;
            let slot = (position * N_HEAD_KV + h) * HD;
            let want = norm_rope_oracle(&k_host[src..src + HD], &weight, position as f32);
            for i in 0..HD {
                let got = k_cache[slot + i];
                assert!(
                    (got - want[i]).abs() <= TOL_F16,
                    "k cache position {position} head {h} dim {i}: got {got}, want {}",
                    want[i]
                );
                // The fixture values are exact in f16, so the copy is exact too.
                assert_eq!(
                    v_cache[slot + i].to_bits(),
                    v_host[src + i].to_bits(),
                    "v cache position {position} head {h} dim {i}"
                );
            }
        }
    }
    for slot in [1usize, 2] {
        for h in 0..N_HEAD_KV {
            let base = (slot * N_HEAD_KV + h) * HD;
            assert!(
                k_cache[base..base + HD].iter().all(|v| *v == 0.0)
                    && v_cache[base..base + HD].iter().all(|v| *v == 0.0),
                "slot {slot} head {h} was written, so the slot is not the position"
            );
        }
    }
}

// The store kernel translates every position through a page table, so it needs a
// page size and a table even in a case where the table is the identity.
const PAGE_TOKENS: u32 = 4;

fn identity_pages(n: usize) -> Vec<i32> {
    (0..n)
        .map(|p| i32::try_from(p).expect("a fixture page index fits an i32"))
        .collect()
}

fn page_bytes(pages: &[i32]) -> Vec<u8> {
    pages.iter().copied().flat_map(i32::to_le_bytes).collect()
}

fn kv_rope_args(n_heads: u32) -> Vec<u8> {
    let mut v = rope_args(n_heads);
    v.extend(PAGE_TOKENS.to_le_bytes());
    // `page_stride`, the last field of the block: 0 means every row of the
    // launch reads the table the pointer names, which is what these
    // single-session fixtures want. Written rather than left off, because a
    // block one field short is read four bytes past its own end and the value
    // the kernel then strides by is whatever followed it on the host.
    v.extend(0u32.to_le_bytes());
    v
}

// --- the multimodal rotation ------------------------------------------------
//
// The multimodal entry points are the same normalization, the same gate split
// and the same cache write; only where the rotary angle's position comes from
// differs. Two things are worth pinning separately. The first is the section
// rule itself, against an oracle that deals the pairs out by index. The second
// is that three agreeing streams reproduce the one-dimensional rotation
// bit-for-bit, which is what makes "the text path is unchanged" a claim about
// the arithmetic rather than only about the dispatch.

/// The positions of the three streams for the fixture's two tokens.
///
/// Deliberately all different, and deliberately not a permutation of each other:
/// an implementation that read the wrong stream for a section would still agree
/// with the oracle on positions that happened to match.
const POS_T: [i32; N_TOK] = [0, 5];
const POS_H: [i32; N_TOK] = [0, 2];
const POS_W: [i32; N_TOK] = [0, 9];

/// Unfused oracle for the multimodal rotation: normalize exactly as the
/// one-dimensional oracle does, then rotate pair `i` by the position of section
/// `i % 3` -- temporal, row, column in that order.
fn norm_mrope_oracle(row: &[f32], w: &[f32], pos: [f32; 3]) -> Vec<f32> {
    let sum: f32 = row.iter().map(|v| v * v).sum();
    let scale = 1.0 / (sum / row.len() as f32 + EPS).sqrt();
    let mut out: Vec<f32> = row
        .iter()
        .zip(w)
        .map(|(v, weight)| v * scale * weight)
        .collect();
    let half = N_ROT / 2;
    for i in 0..half {
        let theta = pos[i % 3] * BASE.powf(-2.0 * (i as f32) / (N_ROT as f32));
        let (s, c) = (theta.sin(), theta.cos());
        let (x0, x1) = (out[i], out[i + half]);
        out[i] = x0 * c - x1 * s;
        out[i + half] = x0 * s + x1 * c;
    }
    out
}

/// Runs the query path at whichever entry point the streams describe and returns
/// the rotated query and the copied gate.
fn run_q(
    gpu: &Gpu,
    streams: Option<[&[i32]; 3]>,
    qkv_host: &[f32],
    weight: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let qkv = gpu.upload(&f32_bytes(qkv_host));
    let w = gpu.upload(&f32_bytes(weight));
    let q_out = gpu.upload(&f32_bytes(&vec![0.0f32; N_TOK * N_HEAD * HD]));
    let gate = gpu.upload(&f32_bytes(&vec![0.0f32; N_TOK * N_HEAD * HD]));
    let args = rope_args(N_HEAD as u32);
    let linear = gpu.upload(&pos_bytes(&POS));
    let uploaded: Vec<_> = streams
        .into_iter()
        .flatten()
        .map(|s| gpu.upload(&pos_bytes(s)))
        .collect();

    let mut params = vec![
        KernelArg::Bytes(&args),
        KernelArg::Buffer(q_out.view()),
        KernelArg::Buffer(gate.view()),
        KernelArg::Buffer(qkv.view()),
        KernelArg::Buffer(w.view()),
    ];
    let entry = if uploaded.is_empty() {
        params.push(KernelArg::Buffer(linear.view()));
        "kernel_q35_q_norm_rope"
    } else {
        for buffer in &uploaded {
            params.push(KernelArg::Buffer(buffer.view()));
        }
        "kernel_q35_q_norm_rope_mrope"
    };
    let launch = LaunchArgs {
        grid: Extent3::new(N_HEAD as u32, N_TOK as u32, 1),
        block: Extent3::linear(128),
        smem_bytes: SMEM,
        args: &params,
    };
    run(gpu, KernelRef::new("qwen35", entry), &launch);
    drain(gpu);
    (gpu.read_f32(&q_out), gpu.read_f32(&gate))
}

/// The section rule: pair `i` rotates by stream `i % 3`.
pub fn the_multimodal_query_rotation_deals_its_pairs_among_the_three_streams(gpu: &Gpu) {
    let weight = vec![1.0f32; HD];
    let qkv_host = fixture(N_TOK * N_HEAD * 2 * HD);
    let (got_q, got_gate) = run_q(gpu, Some([&POS_T, &POS_H, &POS_W]), &qkv_host, &weight);

    for t in 0..N_TOK {
        let pos = [POS_T[t] as f32, POS_H[t] as f32, POS_W[t] as f32];
        for h in 0..N_HEAD {
            let src = (t * N_HEAD + h) * 2 * HD;
            let dst = (t * N_HEAD + h) * HD;
            let want = norm_mrope_oracle(&qkv_host[src..src + HD], &weight, pos);
            for i in 0..HD {
                let got = got_q[dst + i];
                assert!(
                    (got - want[i]).abs() <= TOL,
                    "token {t} head {h} dim {i}: got {got}, want {}",
                    want[i]
                );
            }
            for i in 0..HD {
                assert_eq!(
                    got_gate[dst + i].to_bits(),
                    qkv_host[src + HD + i].to_bits(),
                    "the gate is still a raw copy: token {t} head {h} dim {i}"
                );
            }
        }
    }
}

/// Three streams carrying the same number rotate exactly as one number does.
///
/// Bit-for-bit, not within a tolerance: the two entry points share a body and
/// build their angle from the same expression, so anything short of equality
/// means the multimodal path took a different route through the arithmetic and
/// the text-only guarantee is no longer a statement about the rotation.
pub fn three_agreeing_streams_reproduce_the_one_dimensional_rotation_exactly(gpu: &Gpu) {
    let weight = vec![1.0f32; HD];
    let qkv_host = fixture(N_TOK * N_HEAD * 2 * HD);
    let (linear, _) = run_q(gpu, None, &qkv_host, &weight);
    let (mrope, _) = run_q(gpu, Some([&POS, &POS, &POS]), &qkv_host, &weight);
    assert_eq!(linear.len(), mrope.len());
    for (index, (a, b)) in linear.iter().zip(&mrope).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "element {index}: one-dimensional {a}, multimodal {b}",
        );
    }
}

/// The key path's two positions are two different questions. The cache row comes
/// from the LINEAR position; the angle comes from the three streams. Feeding the
/// temporal stream to the page table instead would fold a run onto a handful of
/// rows with every count in the request still correct, so the fixture makes the
/// two disagree and checks both answers separately.
pub fn the_multimodal_key_path_separates_the_cache_row_from_the_angle(gpu: &Gpu) {
    let weight = vec![1.0f32; HD];
    let k_host = fixture(N_TOK * N_HEAD_KV * HD);
    let v_host = fixture(N_TOK * N_HEAD_KV * HD);
    let k = gpu.upload(&f32_bytes(&k_host));
    let v = gpu.upload(&f32_bytes(&v_host));
    let kw = gpu.upload(&f32_bytes(&weight));
    // Linear positions 0 and 3; the streams name 0/5, 0/2 and 0/9, none of which
    // is 3. A cache row taken from any stream would land somewhere else.
    let linear = [0i32, 3];
    let pos = gpu.upload(&pos_bytes(&linear));
    let pos_t = gpu.upload(&pos_bytes(&POS_T));
    let pos_h = gpu.upload(&pos_bytes(&POS_H));
    let pos_w = gpu.upload(&pos_bytes(&POS_W));
    let pages = gpu.upload(&page_bytes(&identity_pages(1)));
    let slots = 4usize;
    let k_cache = gpu.upload(&vec![0u8; slots * N_HEAD_KV * HD * 2]);
    let v_cache = gpu.upload(&vec![0u8; slots * N_HEAD_KV * HD * 2]);

    let args = kv_rope_args(N_HEAD_KV as u32);
    let params = [
        KernelArg::Bytes(&args),
        KernelArg::Buffer(k_cache.view()),
        KernelArg::Buffer(v_cache.view()),
        KernelArg::Buffer(k.view()),
        KernelArg::Buffer(v.view()),
        KernelArg::Buffer(kw.view()),
        KernelArg::Buffer(pos.view()),
        KernelArg::Buffer(pages.view()),
        KernelArg::Buffer(pos_t.view()),
        KernelArg::Buffer(pos_h.view()),
        KernelArg::Buffer(pos_w.view()),
    ];
    let launch = LaunchArgs {
        grid: Extent3::new(N_HEAD_KV as u32, N_TOK as u32, 1),
        block: Extent3::linear(128),
        smem_bytes: SMEM,
        args: &params,
    };
    run(
        gpu,
        KernelRef::new("qwen35", "kernel_q35_k_norm_rope_kv_mrope"),
        &launch,
    );
    drain(gpu);

    let k_cache = f16_slice(&gpu.read(&k_cache));
    let v_cache = f16_slice(&gpu.read(&v_cache));
    for (t, &position) in linear.iter().enumerate() {
        let position = usize::try_from(position).expect("a fixture position is not negative");
        let angle = [POS_T[t] as f32, POS_H[t] as f32, POS_W[t] as f32];
        for h in 0..N_HEAD_KV {
            let src = (t * N_HEAD_KV + h) * HD;
            let slot = (position * N_HEAD_KV + h) * HD;
            let want = norm_mrope_oracle(&k_host[src..src + HD], &weight, angle);
            for i in 0..HD {
                let got = k_cache[slot + i];
                assert!(
                    (got - want[i]).abs() <= TOL_F16,
                    "k cache row {position} head {h} dim {i}: got {got}, want {}",
                    want[i]
                );
                assert_eq!(
                    v_cache[slot + i].to_bits(),
                    v_host[src + i].to_bits(),
                    "v cache row {position} head {h} dim {i}"
                );
            }
        }
    }
    // Rows 1 and 2 are named by no linear position. A stream reaching the page
    // table would have written one of them.
    for slot in [1usize, 2] {
        for h in 0..N_HEAD_KV {
            let base = (slot * N_HEAD_KV + h) * HD;
            assert!(
                k_cache[base..base + HD].iter().all(|v| *v == 0.0)
                    && v_cache[base..base + HD].iter().all(|v| *v == 0.0),
                "row {slot} head {h} was written, so a grid stream reached the page table"
            );
        }
    }
}
