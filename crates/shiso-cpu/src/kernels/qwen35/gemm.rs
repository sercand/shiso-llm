//! `kernel_mul_mm_q8_0_f32`: the prefill GEMM over `Q8_0` weights.
//!
//! Group `(x, y, z)` owns a 32-token by 64-row tile: tokens `32x ..`, rows
//! `64y ..`, plane `z`, clipped to `ne1` and `ne0` on both edges. The Metal
//! tile stages both operands as `half`, so each activation is narrowed to f16
//! before the shared dot. The weights are deliberately NOT narrowed: Metal's
//! dequantizer rounds `q * d` to half, but the CPU keeps it in f32. That is
//! at most one half-ulp (2^-11 relative) per weight, well inside the tolerance
//! the shared cases grade with, and the exact value is the better one for the
//! single Q8 dot every CPU kernel shares.
//!
//! The bounds-check function constants are accepted and ignored: a CPU group clips its tile unconditionally.

use half::f16;
use shiso_core::{CoreError, CoreResult, Extent3};

use crate::dispatch::{ConstRule, GroupsDisjoint, KernelSpec, Launch, Role, Writes};
use crate::kernels::args::MulMmArgs;
use crate::kernels::generic::{bytes_at, count, floats_at, q8_row_bytes, ratio};
use crate::q8::q8_row_dot;

/// Output rows per tile.
const NR0: usize = 64;
/// Tokens per tile.
const NR1: usize = 32;

/// Weights, activations, destination.
pub(in crate::kernels) const Q8_0: KernelSpec = KernelSpec {
    group,
    roles: &[Role::In, Role::In, Role::Out],
    consts: ConstRule::MulMm,
    block: |b: Extent3| b == Extent3::new(128, 1, 1),
    disjoint: GroupsDisjoint::Yes,
};

fn group(launch: &Launch<'_>, gid: [u32; 3], out: &mut Writes) -> CoreResult<()> {
    let args: MulMmArgs = launch.bytes()?;
    let ne0 = count(args.ne0, "mul_mm ne0")?;
    let ne1 = count(args.ne1, "mul_mm ne1")?;
    let ne12 = count(args.ne12, "mul_mm ne12")?;
    if ne12 == 0 {
        return Err(CoreError::InvalidValue {
            key: "mul_mm ne12",
            value: "0".into(),
        });
    }
    let row_bytes = q8_row_bytes(args.ne00, "mul_mm ne00")?;
    let ne00 = count(args.ne00, "mul_mm ne00")?;
    let (r2, r3) = (ratio(args.r2, "mul_mm r2")?, ratio(args.r3, "mul_mm r3")?);

    let r1 = gid[0] as usize * NR1;
    let r0 = gid[1] as usize * NR0;
    let im = gid[2] as usize;
    let nr0 = NR0.min(ne0.saturating_sub(r0));
    let nr1 = NR1.min(ne1.saturating_sub(r1));
    if nr0 == 0 || nr1 == 0 {
        return Ok(());
    }
    let (i12, i13) = ((im % ne12) as u64, (im / ne12) as u64);

    let weights = launch.raw(0)?;
    let acts = launch.f32(1)?;
    let w_plane = (i12 / r2)
        .saturating_mul(args.nb02)
        .saturating_add((i13 / r3).saturating_mul(args.nb03));
    let x_plane = i12
        .saturating_mul(args.nb12)
        .saturating_add(i13.saturating_mul(args.nb13));

    let mut rows_w = Vec::with_capacity(nr0);
    for row in 0..nr0 {
        let at = w_plane.saturating_add(((r0 + row) as u64).saturating_mul(args.nb01));
        rows_w.push(bytes_at(weights, at, row_bytes, "mul_mm weights")?);
    }
    let mut x = vec![0.0f32; ne00];
    let mut vals = vec![0.0f32; nr0];
    for t in 0..nr1 {
        let at = ((r1 + t) as u64)
            .saturating_mul(args.nb11)
            .saturating_add(x_plane);
        let src = floats_at(acts, at, ne00, "mul_mm activations")?;
        for (d, s) in x.iter_mut().zip(src) {
            *d = f16::from_f32(*s).to_f32();
        }
        for (v, w) in vals.iter_mut().zip(&rows_w) {
            *v = q8_row_dot(w, &x);
        }
        let dst = (im as u64)
            .saturating_mul(ne1 as u64)
            .saturating_mul(ne0 as u64)
            .saturating_add(((r1 + t) as u64).saturating_mul(ne0 as u64))
            .saturating_add(r0 as u64);
        out.f32(2, usize::try_from(dst).unwrap_or(usize::MAX), &vals);
    }
    Ok(())
}
