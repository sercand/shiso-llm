//! `kernel_audio_layer_norm_f32`.
//!
//! A group owns one row. Everything is taken about the row's first value, as
//! the Metal kernel does: a row near 1e4 has an f32 ulp of 1e-3 there, so a
//! mean stored at that magnitude is already off by more than a deviation, and
//! no summation order recovers it. `x - pivot` is exact for values that close
//! together, and the shifted mean stays deviation-sized. The lane reductions
//! become plain sums in index order.
//!
//! The weight is always applied (the engine refuses a layer norm without one,
//! and the shared case passes it with the flag clear); only the bias is
//! flagged.

use shiso_core::{CoreError, CoreResult};

use crate::dispatch::{ConstRule, GroupsDisjoint, KernelSpec, Launch, Role, Writes};
use crate::kernels::args::AudioNormArgs;

use super::{extent, floats};

/// Bit of `flags` that says a bias is present.
const HAS_BIAS: u32 = 2;

/// Destination (which may be the input), input, weight, bias.
pub(in crate::kernels) const LAYER_NORM: KernelSpec = KernelSpec {
    group,
    roles: &[Role::Out, Role::In, Role::In, Role::In],
    consts: ConstRule::None,
    // The threads only stride the row on the device; a CPU group runs the row
    // itself, so any one-dimensional block up to the device's 1024 is fine.
    block: |b| (1..=1024).contains(&b.x) && b.y == 1 && b.z == 1,
    disjoint: GroupsDisjoint::Yes,
};

#[expect(
    clippy::cast_precision_loss,
    reason = "the device converts the u32 row width to f32 for the mean, and a row is far below 2^24 elements"
)]
fn group(launch: &Launch<'_>, gid: [u32; 3], out: &mut Writes) -> CoreResult<()> {
    let args: AudioNormArgs = launch.bytes()?;
    let r = extent(gid[0], "layer_norm row")?;
    if r >= extent(args.rows, "layer_norm rows")? {
        return Ok(());
    }
    let dim = extent(args.dim, "layer_norm dim")?;
    if dim == 0 {
        return Err(CoreError::InvalidValue {
            key: "layer_norm dim",
            value: "0".into(),
        });
    }
    let src = floats(launch.f32(1)?, r * dim, dim, "layer_norm input")?;
    let weight = floats(launch.f32(2)?, 0, dim, "layer_norm weight")?;
    let bias = if args.flags & HAS_BIAS != 0 {
        Some(floats(launch.f32(3)?, 0, dim, "layer_norm bias")?)
    } else {
        None
    };
    let pivot = src[0];
    let mut sum = 0.0f32;
    for &v in src {
        sum += v - pivot;
    }
    let width = dim as f32;
    let dmean = sum / width;
    let mut var = 0.0f32;
    for &v in src {
        let d = (v - pivot) - dmean;
        var += d * d;
    }
    let rstd = 1.0 / (var / width + args.eps).sqrt();
    let vals: Vec<f32> = (0..dim)
        .map(|i| {
            let y = ((src[i] - pivot) - dmean) * rstd * weight[i];
            bias.map_or(y, |b| y + b[i])
        })
        .collect();
    out.f32(0, r * dim, &vals);
    Ok(())
}
