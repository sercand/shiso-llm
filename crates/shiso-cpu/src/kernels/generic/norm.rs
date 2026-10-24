//! The two RMS norms the dense pass uses.
//!
//! A group owns one activation row (`gid` is the row's `x, y, z`). The Metal
//! kernels reduce the sum of squares across a simdgroup tree; here it is a
//! plain f32 sum in index order, then `1 / sqrt(mean + eps)`.

use shiso_core::{CoreError, CoreResult, Extent3};

use crate::dispatch::{ConstRule, GroupsDisjoint, KernelSpec, Launch, Role, Writes};
use crate::kernels::args::NormArgs;

use super::{count, f32_index};

/// Activation, weight, the never-read second fuse operand, destination.
pub(in crate::kernels) const RMS_NORM_MUL: KernelSpec = KernelSpec {
    group: rms_norm_mul_group,
    roles: &[Role::In, Role::In, Role::In, Role::Out],
    consts: ConstRule::None,
    block: is_warp,
    disjoint: GroupsDisjoint::Yes,
};

/// Activation, addend, weight, sum destination, norm destination.
pub(in crate::kernels) const ADD_RMS_NORM_MUL: KernelSpec = KernelSpec {
    group: add_rms_norm_mul_group,
    roles: &[Role::In, Role::In, Role::In, Role::Out, Role::Out],
    consts: ConstRule::None,
    block: is_warp,
    disjoint: GroupsDisjoint::Yes,
};

fn is_warp(b: Extent3) -> bool {
    b == Extent3::new(32, 1, 1)
}

/// Width of one `float4` of the row.
const VEC: usize = 4;

/// A row's element offset for `gid` under strides `(z, y, x)` in bytes.
fn row_at(gid: [u32; 3], strides: [u64; 3], what: &'static str) -> CoreResult<usize> {
    let bytes = u64::from(gid[2])
        .saturating_mul(strides[0])
        .saturating_add(u64::from(gid[1]).saturating_mul(strides[1]))
        .saturating_add(u64::from(gid[0]).saturating_mul(strides[2]));
    f32_index(bytes, what)
}

/// The weight row for `gid`, broadcast by modulo over the weight extents.
fn weight_row<'a>(
    weight: &'a [f32],
    args: &NormArgs,
    gid: [u32; 3],
    n: usize,
) -> CoreResult<&'a [f32]> {
    let ext = |v: i32, what: &'static str| -> CoreResult<u64> {
        match u64::try_from(v) {
            Ok(v) if v > 0 => Ok(v),
            _ => Err(CoreError::InvalidValue {
                key: what,
                value: v.to_string(),
            }),
        }
    };
    let z = u64::from(gid[2]) % ext(args.nef3[1], "norm nef3")?;
    let y = u64::from(gid[1]) % ext(args.nef2[1], "norm nef2")?;
    let x = u64::from(gid[0]) % ext(args.nef1[1], "norm nef1")?;
    let bytes = z
        .saturating_mul(args.nbf3[1])
        .saturating_add(y.saturating_mul(args.nbf2[1]))
        .saturating_add(x.saturating_mul(args.nbf1[1]));
    let at = f32_index(bytes, "norm weight stride")?;
    window(weight, at, n, "norm weight")
}

fn window<'a>(data: &'a [f32], at: usize, n: usize, what: &'static str) -> CoreResult<&'a [f32]> {
    data.get(at..at.saturating_add(n))
        .ok_or_else(|| CoreError::InvalidValue {
            key: what,
            value: format!(
                "elements {at}..{} lie outside the buffer",
                at.saturating_add(n)
            ),
        })
}

/// `1 / sqrt(mean(x^2) + eps)`, the row scale.
#[expect(
    clippy::cast_precision_loss,
    reason = "the device converts the i32 element count to f32 for the mean, and a row is far below 2^24 elements"
)]
fn row_scale(row: &[f32], args: &NormArgs) -> CoreResult<f32> {
    let ne00 = count(args.ne00, "norm ne00")?;
    if ne00 == 0 {
        return Err(CoreError::InvalidValue {
            key: "norm ne00",
            value: "0".to_owned(),
        });
    }
    let mut sum = 0.0f32;
    for v in row {
        sum += v * v;
    }
    let mean = sum / args.ne00 as f32;
    Ok(1.0 / (mean + args.eps).sqrt())
}

fn rms_norm_mul_group(launch: &Launch<'_>, gid: [u32; 3], out: &mut Writes) -> CoreResult<()> {
    let args: NormArgs = launch.bytes()?;
    let n = count(args.ne00_t, "norm ne00_t")? * VEC;
    let src = launch.f32(0)?;
    let at = row_at(gid, [args.nbf3[0], args.nbf2[0], args.nbf1[0]], "norm nbf")?;
    let x = window(src, at, n, "norm activation")?;
    let w = weight_row(launch.f32(1)?, &args, gid, n)?;
    let scale = row_scale(x, &args)?;
    let y: Vec<f32> = x.iter().zip(w).map(|(&x, &w)| (x * scale) * w).collect();
    let dst = row_at(gid, [args.nb3, args.nb2, args.nb1], "norm nb")?;
    out.f32(3, dst, &y);
    Ok(())
}

fn add_rms_norm_mul_group(launch: &Launch<'_>, gid: [u32; 3], out: &mut Writes) -> CoreResult<()> {
    let args: NormArgs = launch.bytes()?;
    let n = count(args.ne00_t, "norm ne00_t")? * VEC;
    let at = row_at(gid, [args.nbf3[0], args.nbf2[0], args.nbf1[0]], "norm nbf")?;
    // The addend shares the activation's strides, as on the device.
    let a = window(launch.f32(0)?, at, n, "norm activation")?;
    let b = window(launch.f32(1)?, at, n, "norm addend")?;
    let w = weight_row(launch.f32(2)?, &args, gid, n)?;
    let sum: Vec<f32> = a.iter().zip(b).map(|(&a, &b)| a + b).collect();
    let scale = row_scale(&sum, &args)?;
    let norm: Vec<f32> = sum.iter().zip(w).map(|(&s, &w)| (s * scale) * w).collect();
    let dst = row_at(gid, [args.nb3, args.nb2, args.nb1], "norm nb")?;
    out.f32(3, dst, &sum);
    out.f32(4, dst, &norm);
    Ok(())
}
