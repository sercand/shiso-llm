//! `kernel_dense_attn_decode` and `kernel_dense_attn_prefill`: windowed,
//! bounded and scaled grouped-query attention over the paged f16 cache.
//!
//! What a query may read is a span of logical positions: the low end a sliding
//! window (`pos + 1 - window`, or 0 without one), the high end a per-token bound
//! the caller supplies. Neither is derivable from the other. The score scale is
//! an argument and is applied to the reduced score, so the number in the
//! argument block is the number the softmax sees.
//!
//! The device's two entries differ in how they tile work (a simdgroup per
//! position against a tile of sixteen queries); a sequential group has no tile
//! to amortise, so both run the same per-query routine and differ only in which
//! queries a group owns and in the page table they read, as on the device.

use half::f16;
use shiso_core::{CoreError, CoreResult, Extent3};

use crate::dispatch::{ConstRule, GroupsDisjoint, KernelSpec, Launch, Role, Writes};
use crate::kernels::args::DenseAttnArgs;

use super::{cache_row, position};

/// Queries per prefill group, as on the device.
const QT: u32 = 16;

/// Output, queries, key cache, value cache, positions, bounds, page table.
const ROLES: &[Role] = &[
    Role::Out,
    Role::In,
    Role::In,
    Role::In,
    Role::In,
    Role::In,
    Role::In,
];

/// One group per (head, token).
pub(in crate::kernels) const DECODE: KernelSpec = KernelSpec {
    group: decode_group,
    roles: ROLES,
    consts: ConstRule::None,
    block: |b: Extent3| b == Extent3::new(256, 1, 1),
    disjoint: GroupsDisjoint::Yes,
};

/// One group per (head, block of sixteen tokens).
pub(in crate::kernels) const PREFILL: KernelSpec = KernelSpec {
    group: prefill_group,
    roles: ROLES,
    consts: ConstRule::None,
    block: |b: Extent3| b == Extent3::new(256, 1, 1),
    disjoint: GroupsDisjoint::Yes,
};

/// The first key position a query at `pos` may read under `window`.
///
/// Written as `pos + 1 > window` so the subtraction never underflows.
fn span_lo(pos: u32, window: u32) -> u32 {
    if window != 0 && pos + 1 > window {
        pos + 1 - window
    } else {
        0
    }
}

fn invalid(key: &'static str, value: String) -> CoreError {
    CoreError::InvalidValue { key, value }
}

/// Everything per-launch the query routine needs, validated once per group.
struct Attn<'a> {
    args: DenseAttnArgs,
    q: &'a [f32],
    k: &'a [f16],
    v: &'a [f16],
    pos: &'a [i32],
    bound: &'a [i32],
    pages: &'a [i32],
}

impl<'a> Attn<'a> {
    fn new(launch: &Launch<'a>) -> CoreResult<Self> {
        let args: DenseAttnArgs = launch.bytes()?;
        if args.head_dim == 0 || args.n_head_kv == 0 || !args.n_head.is_multiple_of(args.n_head_kv)
        {
            return Err(invalid(
                "dense attn heads",
                format!(
                    "head_dim {}, n_head {}, n_head_kv {}",
                    args.head_dim, args.n_head, args.n_head_kv
                ),
            ));
        }
        Ok(Self {
            args,
            q: launch.f32(1)?,
            k: launch.f16(2)?,
            v: launch.f16(3)?,
            pos: launch.i32(4)?,
            bound: launch.i32(5)?,
            pages: launch.i32(6)?,
        })
    }

    /// Attention of query `t`, head `h`; `table_t` picks the page table row.
    ///
    /// Writes `head_dim` floats at the query's own row of the output. A span
    /// with no visible key writes zeros, which is what attention over no keys
    /// contributes; the softmax would otherwise divide by nothing.
    fn query(&self, t: u32, head: u32, table_t: u32, out: &mut Writes) -> CoreResult<()> {
        let dims = &self.args;
        let head_dim = dims.head_dim as usize;
        let n_kv = dims.n_head_kv as usize;
        let hkv = (head / (dims.n_head / dims.n_head_kv)) as usize;
        let qbase = (t as usize * dims.n_head as usize + head as usize) * head_dim;
        let read = |what: &'static str, data: &[i32]| {
            data.get(t as usize)
                .copied()
                .ok_or_else(|| invalid(what, format!("none for token {t}")))
                .and_then(|x| position(x, what))
        };
        let lo = span_lo(read("dense attn position", self.pos)?, dims.window);
        let hi = read("dense attn bound", self.bound)?;
        let query = self.q.get(qbase..qbase + head_dim).ok_or_else(|| {
            invalid(
                "dense attn queries",
                format!("row at {qbase} outside the buffer"),
            )
        })?;

        let mut rows = Vec::new();
        let mut scores = Vec::new();
        for p in lo..=hi {
            let row = cache_row(self.pages, dims.page_tokens, dims.page_stride, table_t, p)?;
            let kb = (row * n_kv + hkv) * head_dim;
            let key = self.k.get(kb..kb + head_dim).ok_or_else(|| {
                invalid(
                    "dense attn key cache",
                    format!("row {row} outside the buffer"),
                )
            })?;
            let mut dot = 0.0f32;
            for (&qv, &kv) in query.iter().zip(key) {
                dot += qv * kv.to_f32();
            }
            scores.push(dot * dims.scale);
            rows.push(kb);
        }
        let mut result = vec![0.0f32; head_dim];
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max != f32::NEG_INFINITY {
            let mut den = 0.0f32;
            for (&score, &kb) in scores.iter().zip(&rows) {
                let weight = (score - max).exp();
                den += weight;
                let val = self.v.get(kb..kb + head_dim).ok_or_else(|| {
                    invalid(
                        "dense attn value cache",
                        format!("at {kb} outside the buffer"),
                    )
                })?;
                for (acc, &x) in result.iter_mut().zip(val) {
                    *acc += weight * x.to_f32();
                }
            }
            let inv = 1.0 / den;
            for acc in &mut result {
                *acc *= inv;
            }
        }
        out.f32(0, qbase, &result);
        Ok(())
    }
}

fn decode_group(launch: &Launch<'_>, gid: [u32; 3], out: &mut Writes) -> CoreResult<()> {
    let attn = Attn::new(launch)?;
    let (h, t) = (gid[0], gid[1]);
    if h >= attn.args.n_head || t >= attn.args.n_tok {
        return Err(invalid(
            "dense attn decode grid",
            format!(
                "group ({h}, {t}) outside {} heads x {} tokens",
                attn.args.n_head, attn.args.n_tok
            ),
        ));
    }
    // This row's own page table: a batched step carries one row per session.
    attn.query(t, h, t, out)
}

fn prefill_group(launch: &Launch<'_>, gid: [u32; 3], out: &mut Writes) -> CoreResult<()> {
    let attn = Attn::new(launch)?;
    let a = &attn.args;
    // The device indexes lanes by `head_dim / 32` and leaves a ragged tail
    // unwritten; refuse the shape instead.
    if !a.head_dim.is_multiple_of(32) {
        return Err(invalid(
            "dense attn prefill head_dim",
            format!("{} is not a multiple of 32", a.head_dim),
        ));
    }
    let h = gid[0];
    if h >= a.n_head {
        return Err(invalid(
            "dense attn prefill grid",
            format!("head {h} outside {} heads", a.n_head),
        ));
    }
    let t0 = gid[1].saturating_mul(QT);
    if t0 >= a.n_tok {
        return Ok(());
    }
    // One table for the whole launch: a chunk is one session's consecutive
    // positions, so `page_stride` is not read here.
    for t in t0..a.n_tok.min(t0 + QT) {
        attn.query(t, h, 0, out)?;
    }
    Ok(())
}
