//! `kernel_swiglu_flat_f32`: `silu(gate) * up * alpha` over a flat run.
//!
//! Group `g` owns elements `g * block.x ..` up to `ne0`, one per Metal thread.

use shiso_core::{CoreResult, Extent3};

use crate::dispatch::{ConstRule, GroupsDisjoint, KernelSpec, Launch, Role, Writes};
use crate::kernels::args::GluArgs;

use super::count;

/// Gate, up, destination.
pub(in crate::kernels) const SWIGLU_FLAT: KernelSpec = KernelSpec {
    group: swiglu_flat_group,
    roles: &[Role::In, Role::In, Role::Out],
    consts: ConstRule::None,
    block: |b: Extent3| b == Extent3::new(256, 1, 1),
    disjoint: GroupsDisjoint::Yes,
};

/// Below this the limit means "none", as on the device.
const LIMIT_FLOOR: f32 = 1.0e-6;

fn swiglu_flat_group(launch: &Launch<'_>, gid: [u32; 3], out: &mut Writes) -> CoreResult<()> {
    let args: GluArgs = launch.bytes()?;
    let ne0 = count(args.ne0, "swiglu ne0")?;
    let i00 = count(args.i00, "swiglu i00")?;
    let i10 = count(args.i10, "swiglu i10")?;
    let first = gid[0] as usize * launch.block.x as usize;
    if first >= ne0 {
        return Ok(());
    }
    let n = (ne0 - first).min(launch.block.x as usize);
    let gate = launch.f32(0)?;
    let up = launch.f32(1)?;
    let gate = slice(gate, i00 + first, n, "swiglu gate")?;
    let up = slice(up, i10 + first, n, "swiglu up")?;
    let clamp = args.limit > LIMIT_FLOOR;
    let row: Vec<f32> = gate
        .iter()
        .zip(up)
        .map(|(&g, &u)| {
            let (x0, x1) = if clamp {
                (g.min(args.limit), u.clamp(-args.limit, args.limit))
            } else {
                (g, u)
            };
            let silu = x0 / (1.0 + (-x0).exp());
            silu * x1 * args.alpha
        })
        .collect();
    out.f32(2, first, &row);
    Ok(())
}

fn slice<'a>(data: &'a [f32], at: usize, n: usize, what: &'static str) -> CoreResult<&'a [f32]> {
    data.get(at..at + n)
        .ok_or_else(|| shiso_core::CoreError::InvalidValue {
            key: what,
            value: format!("elements {at}..{} lie outside the buffer", at + n),
        })
}
