//! `kernel_mul_mv_q8_0_f32`: the decode matvec over `Q8_0` weights.
//!
//! Group `(x, y, z)` owns `nr0` consecutive output rows `x * nr0 ..` of token
//! `y` in plane `z`, as the Metal threadgroup does. Its simdgroups split `k`
//! (each strides the blocks by `nsg * 8`), not the rows, and the final
//! reduction sums their partials into one value per row, so a group's `nsg`
//! "threads" collapse to one `q8_row_dot` per row here. Rows at or past
//! `ne01` are not written, which is what makes `ceil(out / nr0)` groups safe
//! for an odd row count.

use shiso_core::{CoreResult, Extent3};

use crate::dispatch::{ConstRule, GroupsDisjoint, KernelSpec, Launch, Role, Writes};
use crate::kernels::args::MulMvArgs;
use crate::q8::q8_row_dot;

use super::{bytes_at, count, floats_at, q8_row_bytes, ratio};

/// Weights, activations, destination.
pub(in crate::kernels) const Q8_0: KernelSpec = KernelSpec {
    group,
    roles: &[Role::In, Role::In, Role::Out],
    consts: ConstRule::MulMv,
    block: |b: Extent3| b.x == 32 && b.y >= 1 && b.z == 1,
    disjoint: GroupsDisjoint::Yes,
};

fn group(launch: &Launch<'_>, gid: [u32; 3], out: &mut Writes) -> CoreResult<()> {
    let args: MulMvArgs = launch.bytes()?;
    let ne01 = count(args.ne01, "mul_mv ne01")?;
    let nr0 = count(args.nr0, "mul_mv nr0")?;
    if nr0 == 0 {
        return Err(shiso_core::CoreError::InvalidValue {
            key: "mul_mv nr0",
            value: "0".into(),
        });
    }
    let ne0 = count(args.ne0, "mul_mv ne0")? as u64;
    let n_rows1 = count(args.ne1, "mul_mv ne1")? as u64;
    let row_bytes = q8_row_bytes(args.ne00, "mul_mv ne00")?;
    let ne00 = count(args.ne00, "mul_mv ne00")?;
    let ne12 = count(args.ne12, "mul_mv ne12")?;
    if ne12 == 0 {
        return Err(shiso_core::CoreError::InvalidValue {
            key: "mul_mv ne12",
            value: "0".into(),
        });
    }
    let (r2, r3) = (ratio(args.r2, "mul_mv r2")?, ratio(args.r3, "mul_mv r3")?);

    let r0 = gid[0] as usize * nr0;
    let r1 = u64::from(gid[1]);
    let im = gid[2] as usize;
    let (i12, i13) = ((im % ne12) as u64, (im / ne12) as u64);

    let weights = launch.raw(0)?;
    let acts = launch.f32(1)?;
    let x_at = r1
        .saturating_mul(args.nb11)
        .saturating_add(i12.saturating_mul(args.nb12))
        .saturating_add(i13.saturating_mul(args.nb13));
    let w_plane = (i12 / r2)
        .saturating_mul(args.nb02)
        .saturating_add((i13 / r3).saturating_mul(args.nb03));

    let rows = nr0.min(ne01.saturating_sub(r0));
    let mut vals = Vec::with_capacity(rows);
    if rows > 0 {
        let x = floats_at(acts, x_at, ne00, "mul_mv activations")?;
        for row in 0..rows {
            let at = w_plane.saturating_add(((r0 + row) as u64).saturating_mul(args.nb01));
            let w = bytes_at(weights, at, row_bytes, "mul_mv weights")?;
            vals.push(q8_row_dot(w, x));
        }
        let dst = (im as u64)
            .saturating_mul(ne0)
            .saturating_mul(n_rows1)
            .saturating_add(r1.saturating_mul(ne0));
        let dst = usize::try_from(dst).unwrap_or(usize::MAX);
        out.f32(2, dst.saturating_add(r0), &vals);
    }
    Ok(())
}
