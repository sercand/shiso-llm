//! `kernel_audio_window_attn_f32`: bidirectional attention over a chunked
//! window, for the head's encoder layers.
//!
//! A group is one head and a block of eight queries. Query `q` sees key `ka`
//! when the key's chunk is not after the query's and at most `left_chunks`
//! before it, inside `[k_abs0, min(k_end, k_abs0 + n_k))`. A query that sees
//! nothing writes zeros. The device streams keys through an online softmax;
//! here each query takes the exact two-pass softmax over its visible keys,
//! which is the same function with fewer roundings.
//!
//! The relative-position score, the content bias and the per-dim query scale
//! are not ported: the head never sends them, and a launch that does is
//! refused rather than computed without them.

use shiso_core::{CoreError, CoreResult};

use crate::dispatch::{ConstRule, GroupsDisjoint, KernelSpec, Launch, Role, Writes};
use crate::kernels::args::WindowAttnArgs;

use super::{extent, floats, is_linear};

/// Queries one group takes (`AUDIO_ATTN_QB`).
const QUERY_BLOCK: usize = 8;
/// The widest head the device keeps in registers (`32 lanes * 4 slots`).
const MAX_HEAD_DIM: usize = 128;

/// Destination, queries, keys, values, and the three buffers the head leaves
/// at the query alias (content bias, per-dim scale, relative scores).
pub(in crate::kernels) const WINDOW_ATTN: KernelSpec = KernelSpec {
    group,
    roles: &[
        Role::Out,
        Role::In,
        Role::In,
        Role::In,
        Role::In,
        Role::In,
        Role::In,
    ],
    consts: ConstRule::None,
    block: |b| is_linear(b, 32),
    disjoint: GroupsDisjoint::Yes,
};

fn unported(key: &'static str, value: u32) -> CoreError {
    CoreError::InvalidValue {
        key,
        value: format!("{value}: relative scores, content bias and query scale are not ported"),
    }
}

fn group(launch: &Launch<'_>, gid: [u32; 3], out: &mut Writes) -> CoreResult<()> {
    let args: WindowAttnArgs = launch.bytes()?;
    if args.n_d > 0 {
        return Err(unported("window_attn n_d", args.n_d));
    }
    if args.has_bias_uv != 0 {
        return Err(unported("window_attn has_bias_uv", args.has_bias_uv));
    }
    if args.has_q_dim_scale != 0 {
        return Err(unported(
            "window_attn has_q_dim_scale",
            args.has_q_dim_scale,
        ));
    }
    let (head, i0) = (
        extent(gid[0], "window_attn head")?,
        extent(gid[1], "window_attn block")? * QUERY_BLOCK,
    );
    let (heads, n_q, n_k) = (
        extent(args.heads, "window_attn heads")?,
        extent(args.n_q, "window_attn n_q")?,
        extent(args.n_k, "window_attn n_k")?,
    );
    if head >= heads || i0 >= n_q {
        return Ok(());
    }
    let hd = extent(args.head_dim, "window_attn head_dim")?;
    if hd == 0 || hd > MAX_HEAD_DIM || args.chunk == 0 {
        return Err(CoreError::InvalidValue {
            key: "window_attn head_dim/chunk",
            value: format!("head_dim {hd} (1..={MAX_HEAD_DIM}), chunk {}", args.chunk),
        });
    }
    let width = heads * hd;
    let hoff = head * hd;
    let nq = QUERY_BLOCK.min(n_q - i0);
    let (queries, keys, values) = (
        floats(launch.f32(1)?, 0, n_q * width, "window_attn queries")?,
        floats(launch.f32(2)?, 0, n_k * width, "window_attn keys")?,
        floats(launch.f32(3)?, 0, n_k * width, "window_attn values")?,
    );

    // Absolute positions can pass u32 in the intermediate products.
    let chunk = u64::from(args.chunk);
    let left = u64::from(args.left_chunks);
    let k_abs0 = u64::from(args.k_abs0);
    let hi_abs = u64::from(args.k_end).min(k_abs0 + n_k as u64);
    let mut vals = vec![0.0f32; nq * hd];
    let mut scores: Vec<(usize, f32)> = Vec::new();
    for row in 0..nq {
        let qabs = u64::from(args.q_abs0) + (i0 + row) as u64;
        let qc = qabs / chunk;
        let lo_abs = (qc.saturating_sub(left) * chunk).max(k_abs0);
        let query_vec = &queries[(i0 + row) * width + hoff..][..hd];
        scores.clear();
        for key_abs in lo_abs..hi_abs.min((qc + 1) * chunk) {
            let key_row = extent_u64(key_abs - k_abs0)?;
            let key_vec = &keys[key_row * width + hoff..][..hd];
            let mut dot = 0.0f32;
            for (a, b) in query_vec.iter().zip(key_vec) {
                dot += a * b;
            }
            let mut score = args.scale * (args.ac_scale * dot);
            if args.softcap > 0.0 {
                score = args.softcap * (score / args.softcap).tanh();
            }
            scores.push((key_row, score));
        }
        // No visible key: the row stays zero.
        let Some(max) = scores.iter().map(|score| score.1).reduce(f32::max) else {
            continue;
        };
        let mut total = 0.0f32;
        for score in &mut scores {
            score.1 = (score.1 - max).exp();
            total += score.1;
        }
        let acc = &mut vals[row * hd..][..hd];
        for &(key_row, prob) in &scores {
            let value_vec = &values[key_row * width + hoff..][..hd];
            for (a, value) in acc.iter_mut().zip(value_vec) {
                *a += prob * value;
            }
        }
        if total > 0.0 {
            for a in acc.iter_mut() {
                *a /= total;
            }
        }
    }
    for row in 0..nq {
        out.f32(0, (i0 + row) * width + hoff, &vals[row * hd..][..hd]);
    }
    Ok(())
}

/// A key offset as an index, refused when it does not fit.
fn extent_u64(value: u64) -> CoreResult<usize> {
    usize::try_from(value).map_err(|_| CoreError::InvalidValue {
        key: "window_attn key",
        value: value.to_string(),
    })
}
