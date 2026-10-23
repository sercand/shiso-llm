//! `kernel_mul_mv_ext_q8_0_f32_r1_{2,3,4,5}`: the small-batch matvec, `R`
//! tokens per group (Metal's `r1ptg` template argument).
//!
//! Group `(x, y, z)` owns rows `x * R .. (x + 1) * R` with `R = (32 / nxpsg) *
//! nsg` (each of the `nsg` simdgroups covers `32 / nxpsg` rows) and tokens
//! `R*y .. R*y + R` of plane `z`. Lanes within a row split `k`; the shuffle
//! reduction is one `q8_row_dot` here. `nsg` comes from `block.y` and `nxpsg`
//! from constant 601 (16 when absent). Rows past `ne01` and tokens past `ne11`
//! are not written.

use shiso_core::{CoreError, CoreResult, Extent3, FnConstValue};

use crate::dispatch::{ConstRule, GroupsDisjoint, KernelSpec, Launch, Role, Writes};
use crate::kernels::args::MulMvExtArgs;
use crate::q8::q8_row_dot;

use super::{bytes_at, count, floats_at, q8_row_bytes, ratio};

/// The kernel at `TOKENS` tokens per group. Weights, activations, destination.
const fn spec<const TOKENS: usize>() -> KernelSpec {
    KernelSpec {
        group: group::<TOKENS>,
        roles: &[Role::In, Role::In, Role::Out],
        consts: ConstRule::MulMvExt,
        block: |b: Extent3| b.x == 32 && b.y >= 1 && b.z == 1,
        disjoint: GroupsDisjoint::Yes,
    }
}

/// Two tokens per group.
pub(in crate::kernels) const R1_2: KernelSpec = spec::<2>();
/// Three tokens per group.
pub(in crate::kernels) const R1_3: KernelSpec = spec::<3>();
/// Four tokens per group.
pub(in crate::kernels) const R1_4: KernelSpec = spec::<4>();
/// Five tokens per group.
pub(in crate::kernels) const R1_5: KernelSpec = spec::<5>();

fn group<const TOKENS: usize>(
    launch: &Launch<'_>,
    gid: [u32; 3],
    out: &mut Writes,
) -> CoreResult<()> {
    let args: MulMvExtArgs = launch.bytes()?;
    let nxpsg = launch
        .consts
        .iter()
        .find_map(|c| match (c.index, c.value) {
            (601, FnConstValue::I16(v)) => usize::try_from(v).ok(),
            _ => None,
        })
        .unwrap_or(16);
    let rows_per_group = (32 / nxpsg) * launch.block.y as usize;
    let (ne01, ne11) = (
        count(args.ne01, "mul_mv_ext ne01")?,
        count(args.ne11, "mul_mv_ext ne11")?,
    );
    let ne12 = count(args.ne12, "mul_mv_ext ne12")?;
    if ne12 == 0 {
        return Err(CoreError::InvalidValue {
            key: "mul_mv_ext ne12",
            value: "0".into(),
        });
    }
    let row_bytes = q8_row_bytes(args.ne00, "mul_mv_ext ne00")?;
    let ne00 = count(args.ne00, "mul_mv_ext ne00")?;
    let (r2, r3) = (
        ratio(args.r2, "mul_mv_ext r2")?,
        ratio(args.r3, "mul_mv_ext r3")?,
    );

    let i01 = gid[0] as usize * rows_per_group;
    let i11 = gid[1] as usize * TOKENS;
    let im = gid[2] as usize;
    let (i12, i13) = ((im % ne12) as u64, (im / ne12) as u64);
    let rows = rows_per_group.min(ne01.saturating_sub(i01));
    let toks = TOKENS.min(ne11.saturating_sub(i11));
    if rows == 0 || toks == 0 {
        return Ok(());
    }

    let weights = launch.raw(0)?;
    let acts = launch.f32(1)?;
    let w_plane = (i12 / r2)
        .saturating_mul(args.nb02)
        .saturating_add((i13 / r3).saturating_mul(args.nb03));
    let x_plane = i12
        .saturating_mul(args.nb12)
        .saturating_add(i13.saturating_mul(args.nb13));
    let ne0 = count(args.ne0, "mul_mv_ext ne0")? as u64;
    let n_tok_out = count(args.ne1, "mul_mv_ext ne1")? as u64;

    let mut rows_w = Vec::with_capacity(rows);
    for row in 0..rows {
        let at = w_plane.saturating_add(((i01 + row) as u64).saturating_mul(args.nb01));
        rows_w.push(bytes_at(weights, at, row_bytes, "mul_mv_ext weights")?);
    }
    for t in 0..toks {
        let at = ((i11 + t) as u64)
            .saturating_mul(args.nb11)
            .saturating_add(x_plane);
        let x = floats_at(acts, at, ne00, "mul_mv_ext activations")?;
        let vals: Vec<f32> = rows_w.iter().map(|w| q8_row_dot(w, x)).collect();
        let dst = (im as u64)
            .saturating_mul(ne0)
            .saturating_mul(n_tok_out)
            .saturating_add(((i11 + t) as u64).saturating_mul(ne0))
            .saturating_add(i01 as u64);
        out.f32(2, usize::try_from(dst).unwrap_or(usize::MAX), &vals);
    }
    Ok(())
}
