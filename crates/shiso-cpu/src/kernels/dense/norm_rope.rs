//! `kernel_dense_q_rope` and `kernel_dense_kv_rope_store`: rotate a head and
//! store it, with no norm.
//!
//! One group per (head, token), as on the device. The device stages the head in
//! threadgroup memory because a pair's two dimensions belong to different
//! lanes; a sequential group has no such hazard and rotates a copy of the row.

use half::f16;
use shiso_core::{CoreError, CoreResult, Extent3};

use crate::dispatch::{ConstRule, GroupsDisjoint, KernelSpec, Launch, Role, Writes};
use crate::kernels::args::{DenseKvNormRopeArgs, DenseQNormRopeArgs};

use super::{cache_row, position};

/// Output, input, positions.
pub(in crate::kernels) const Q_ROPE: KernelSpec = KernelSpec {
    group: q_rope_group,
    roles: &[Role::Out, Role::In, Role::In],
    consts: ConstRule::None,
    block: |b: Extent3| b == Extent3::new(128, 1, 1),
    disjoint: GroupsDisjoint::Yes,
};

/// Key cache, value cache, keys, values, positions, page table.
pub(in crate::kernels) const KV_ROPE_STORE: KernelSpec = KernelSpec {
    group: kv_rope_store_group,
    roles: &[
        Role::InOut,
        Role::InOut,
        Role::In,
        Role::In,
        Role::In,
        Role::In,
    ],
    consts: ConstRule::None,
    block: |b: Extent3| b == Extent3::new(128, 1, 1),
    disjoint: GroupsDisjoint::Yes,
};

/// The row `[(t * heads + h) * head_dim ..][..head_dim]` of `data`.
fn head_row<'a>(
    data: &'a [f32],
    t: usize,
    heads: usize,
    h: usize,
    head_dim: usize,
    what: &'static str,
) -> CoreResult<&'a [f32]> {
    let at = (t * heads + h) * head_dim;
    data.get(at..at + head_dim)
        .ok_or_else(|| CoreError::InvalidValue {
            key: what,
            value: format!("elements {at}..{} lie outside the buffer", at + head_dim),
        })
}

/// Rotary over the leading `n_rot` dims, the ladder derived over `freq_dims`.
///
/// Pair `i` is `(i, i + freq_dims / 2)` for `i < n_rot / 2` at the angle
/// `pos * base^(-2i / freq_dims)`, with the accurate `powf` and `sin_cos`: the
/// ladder spans six decades of base and a fast variant loses its far end. A
/// ladder narrower than the rotated slice would make a pair's partner another
/// pair's member, so it is refused.
#[expect(
    clippy::cast_precision_loss,
    reason = "the ladder index and width are head dimensions, and the position is the device's own `(float) uint` conversion"
)]
fn rope(v: &mut [f32], n_rot: usize, freq_dims: usize, base: f32, pos: u32) -> CoreResult<()> {
    let half_width = freq_dims / 2;
    let n_pair = n_rot / 2;
    if freq_dims < n_rot || n_pair + half_width > v.len() {
        return Err(CoreError::InvalidValue {
            key: "dense rope n_rot/freq_dims",
            value: format!(
                "n_rot {n_rot}, freq_dims {freq_dims} for a head of {}",
                v.len()
            ),
        });
    }
    for i in 0..n_pair {
        let theta = pos as f32 * base.powf(-2.0 * i as f32 / freq_dims as f32);
        let (s, c) = theta.sin_cos();
        let (x0, x1) = (v[i], v[i + half_width]);
        v[i] = x0 * c - x1 * s;
        v[i + half_width] = x0 * s + x1 * c;
    }
    Ok(())
}

fn q_rope_group(launch: &Launch<'_>, gid: [u32; 3], out: &mut Writes) -> CoreResult<()> {
    let args: DenseQNormRopeArgs = launch.bytes()?;
    let (h, t) = (gid[0] as usize, gid[1] as usize);
    let (head_dim, n_head) = (args.head_dim as usize, args.n_head as usize);
    if h >= n_head || t >= args.n_tok as usize {
        return Err(CoreError::InvalidValue {
            key: "dense q_rope grid",
            value: format!(
                "group ({h}, {t}) outside {n_head} heads x {} tokens",
                args.n_tok
            ),
        });
    }
    let pos = position(
        *launch
            .i32(2)?
            .get(t)
            .ok_or_else(|| CoreError::InvalidValue {
                key: "dense q_rope positions",
                value: format!("none for token {t}"),
            })?,
        "dense q_rope position",
    )?;
    let mut row = head_row(launch.f32(1)?, t, n_head, h, head_dim, "dense q_rope input")?.to_vec();
    rope(
        &mut row,
        args.n_rot as usize,
        args.freq_dims as usize,
        args.rope_base,
        pos,
    )?;
    out.f32(0, (t * n_head + h) * head_dim, &row);
    Ok(())
}

fn kv_rope_store_group(launch: &Launch<'_>, gid: [u32; 3], out: &mut Writes) -> CoreResult<()> {
    let args: DenseKvNormRopeArgs = launch.bytes()?;
    let (h, t) = (gid[0] as usize, gid[1]);
    let (head_dim, n_kv) = (args.head_dim as usize, args.n_head_kv as usize);
    if h >= n_kv || t >= args.n_tok {
        return Err(CoreError::InvalidValue {
            key: "dense kv_rope_store grid",
            value: format!(
                "group ({h}, {t}) outside {n_kv} heads x {} tokens",
                args.n_tok
            ),
        });
    }
    let pos = position(
        *launch
            .i32(4)?
            .get(t as usize)
            .ok_or_else(|| CoreError::InvalidValue {
                key: "dense kv_rope_store positions",
                value: format!("none for token {t}"),
            })?,
        "dense kv_rope_store position",
    )?;
    let row = cache_row(launch.i32(5)?, args.page_tokens, args.page_stride, t, pos)?;
    let slot = (row * n_kv + h) * head_dim;
    let mut key = head_row(
        launch.f32(2)?,
        t as usize,
        n_kv,
        h,
        head_dim,
        "dense key input",
    )?
    .to_vec();
    let value = head_row(
        launch.f32(3)?,
        t as usize,
        n_kv,
        h,
        head_dim,
        "dense value input",
    )?;
    rope(
        &mut key,
        args.n_rot as usize,
        args.freq_dims as usize,
        args.rope_base,
        pos,
    )?;
    // The value never takes the rotation: this entry has no value norm, so it is
    // stored raw (`v_norm` is deliberately unread).
    let key: Vec<f16> = key.iter().map(|&x| f16::from_f32(x)).collect();
    let value: Vec<f16> = value.iter().map(|&x| f16::from_f32(x)).collect();
    out.f16(0, slot, &key);
    out.f16(1, slot, &value);
    Ok(())
}
