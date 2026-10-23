//! `kernel_audio_unary_f32` and `kernel_audio_axpy_f32`.
//!
//! A group owns 256 consecutive elements, as the Metal threadgroup does, and
//! writes only those below `count`.

use shiso_core::{CoreError, CoreResult};

use crate::dispatch::{ConstRule, GroupsDisjoint, KernelSpec, Launch, Role, Writes};
use crate::kernels::args::{AxpyArgs, UnaryArgs};

use super::{LINEAR_BLOCK, extent, floats, is_linear};

/// Destination (which may be the input), input.
pub(in crate::kernels) const UNARY: KernelSpec = KernelSpec {
    group: unary_group,
    roles: &[Role::Out, Role::In],
    consts: ConstRule::None,
    block: |b| is_linear(b, LINEAR_BLOCK),
    disjoint: GroupsDisjoint::Yes,
};

/// Accumulator, addend. The accumulator is read as well as written; reads see
/// its pre-launch value, which is all `y[i] += ...` needs.
pub(in crate::kernels) const AXPY: KernelSpec = KernelSpec {
    group: axpy_group,
    roles: &[Role::InOut, Role::In],
    consts: ConstRule::None,
    block: |b| is_linear(b, LINEAR_BLOCK),
    disjoint: GroupsDisjoint::Yes,
};

/// The group's element range `[lo, hi)` of `count`.
fn span(gid: [u32; 3], count: u32) -> CoreResult<(usize, usize)> {
    let lo = extent(gid[0], "audio linear gid")?.saturating_mul(LINEAR_BLOCK as usize);
    let count = extent(count, "audio linear count")?;
    Ok((
        lo.min(count),
        lo.saturating_add(LINEAR_BLOCK as usize).min(count),
    ))
}

/// The logistic function.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The error function: the rational-times-Gaussian form of `audio_erf` in
/// `kernels/metal/audio/audio_common.h` (error below 1.5e-7), which the
/// shared oracle evaluates in f64. std has no stable `erf`, and using the
/// device's own approximation keeps GELU-erf the same function on both.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_8 + t * (-1.453_152 + t * 1.061_405_4))));
    sign * (1.0 - poly * (-a * a).exp())
}

fn unary_group(launch: &Launch<'_>, gid: [u32; 3], out: &mut Writes) -> CoreResult<()> {
    let args: UnaryArgs = launch.bytes()?;
    let (lo, hi) = span(gid, args.count)?;
    if lo == hi {
        return Ok(());
    }
    let input = floats(launch.f32(1)?, lo, hi - lo, "unary input")?;
    let (a, b) = (args.a, args.b);
    let f: fn(f32, f32, f32) -> f32 = match args.op {
        0 => |x, _, _| x.max(0.0),
        1 => |x, _, _| x * sigmoid(x),
        2 => |x, _, _| sigmoid(x),
        3 => |x, _, _| 0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2)),
        4 => |x, _, _| x.tanh(),
        5 => |x, a, b| x.clamp(a, b),
        6 => |x, a, _| x * a,
        // The device passes an unknown code through; a wrong code is a
        // caller bug, so it is refused.
        op => {
            return Err(CoreError::InvalidValue {
                key: "unary op",
                value: op.to_string(),
            });
        }
    };
    if args.op == 5 && a > b {
        // `f32::clamp` panics on this; the device's clamp is just wrong.
        return Err(CoreError::InvalidValue {
            key: "unary clamp",
            value: format!("min {a} exceeds max {b}"),
        });
    }
    let vals: Vec<f32> = input.iter().map(|&x| f(x, a, b)).collect();
    out.f32(0, lo, &vals);
    Ok(())
}

fn axpy_group(launch: &Launch<'_>, gid: [u32; 3], out: &mut Writes) -> CoreResult<()> {
    let args: AxpyArgs = launch.bytes()?;
    let (lo, hi) = span(gid, args.count)?;
    if lo == hi {
        return Ok(());
    }
    let y = floats(launch.f32(0)?, lo, hi - lo, "axpy y")?;
    let x = floats(launch.f32(1)?, lo, hi - lo, "axpy x")?;
    let vals: Vec<f32> = y.iter().zip(x).map(|(&y, &x)| y + args.alpha * x).collect();
    out.f32(0, lo, &vals);
    Ok(())
}
