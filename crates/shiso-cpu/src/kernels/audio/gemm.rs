//! The `Q8_0`-weight products of the audio set: the 32-row and 64-row tiled
//! GEMMs, the matrix-vector kernel and the split-k closing sum.
//!
//! Every output is one [`q8_row_dot`] over the whole of its `k` slice, so the
//! tiled, the 64-row and the matrix-vector entries agree bit for bit; the
//! device's differing tile and lane arithmetic only reorders sums, which the
//! shared cases tolerate and this port does not need to imitate. A group owns
//! the same output tile as its Metal threadgroup.

use shiso_core::{CoreError, CoreResult, Extent3};

use crate::dispatch::{ConstRule, GroupsDisjoint, KernelSpec, Launch, Role, Writes};
use crate::kernels::args::AudioGemmArgs;
use crate::q8::{BLOCK_Q8_0_BYTES, QK8_0, q8_row_dot};

use super::{LINEAR_BLOCK, extent, floats, is_linear};

/// Bit of `flags` that says a bias is present.
const HAS_BIAS: u32 = 1;
/// Output columns of one tiled group.
const TILE_N: usize = 64;
/// Output rows of the 32-row and the 64-row tiled groups.
const TILE_M: usize = 32;
const TILE_M64: usize = 64;
/// Output columns and rows of one matrix-vector group: four simdgroups of four
/// columns, up to eight rows.
const GEMV_COLS: usize = 16;
const GEMV_ROWS: usize = 8;

/// Destination (the partial planes when split), activations, weights, bias.
pub(in crate::kernels) const GEMM: KernelSpec = KernelSpec {
    group: gemm_group,
    roles: &[Role::Out, Role::In, Role::In, Role::In],
    consts: ConstRule::None,
    block: |b: Extent3| is_linear(b, 128),
    disjoint: GroupsDisjoint::Yes,
};

/// As [`GEMM`], over 64-row tiles.
pub(in crate::kernels) const GEMM64: KernelSpec = KernelSpec {
    group: gemm64_group,
    ..GEMM
};

/// As [`GEMM`], for a few rows.
pub(in crate::kernels) const GEMV: KernelSpec = KernelSpec {
    group: gemv_group,
    ..GEMM
};

/// Destination, partial planes, bias.
pub(in crate::kernels) const SPLIT_SUM: KernelSpec = KernelSpec {
    group: split_sum_group,
    roles: &[Role::Out, Role::In, Role::In],
    consts: ConstRule::None,
    block: |b: Extent3| is_linear(b, LINEAR_BLOCK),
    disjoint: GroupsDisjoint::Yes,
};

/// The validated shape of one product.
struct Shape {
    m: usize,
    n: usize,
    k: usize,
    row_bytes: usize,
    bias: bool,
}

fn invalid(key: &'static str, value: String) -> CoreError {
    CoreError::InvalidValue { key, value }
}

/// Reads and checks the arguments. The dot silently drops a ragged tail in
/// release, so a `k` that is not whole blocks is refused here.
fn shape(args: &AudioGemmArgs) -> CoreResult<Shape> {
    let k = extent(args.k, "audio gemm k")?;
    if k == 0 || !k.is_multiple_of(QK8_0) {
        return Err(invalid(
            "audio gemm k",
            format!("{k} is not a positive multiple of {QK8_0}"),
        ));
    }
    Ok(Shape {
        m: extent(args.m, "audio gemm m")?,
        n: extent(args.n, "audio gemm n")?,
        k,
        row_bytes: k / QK8_0 * BLOCK_Q8_0_BYTES,
        bias: args.flags & HAS_BIAS != 0,
    })
}

/// One output: row `r` of `x` against weight row `c`, over `[kb, ke)`.
fn dot(
    x: &[f32],
    w: &[u8],
    sh: &Shape,
    (r, c): (usize, usize),
    (kb, ke): (usize, usize),
) -> CoreResult<f32> {
    let xs = floats(x, r * sh.k + kb, ke - kb, "audio gemm activations")?;
    let at = c * sh.row_bytes + kb / QK8_0 * BLOCK_Q8_0_BYTES;
    let len = (ke - kb) / QK8_0 * BLOCK_Q8_0_BYTES;
    let ws = at
        .checked_add(len)
        .and_then(|end| w.get(at..end))
        .ok_or_else(|| {
            invalid(
                "audio gemm weights",
                format!("{len} bytes at {at} lie outside the buffer"),
            )
        })?;
    Ok(q8_row_dot(ws, xs))
}

fn gemm_group(launch: &Launch<'_>, gid: [u32; 3], out: &mut Writes) -> CoreResult<()> {
    tiled(launch, gid, out, TILE_M)
}

fn gemm64_group(launch: &Launch<'_>, gid: [u32; 3], out: &mut Writes) -> CoreResult<()> {
    tiled(launch, gid, out, TILE_M64)
}

/// One `tile_m` x 64 output tile of plane `gid.z`, the plane's slice of `k`
/// being `[z * k_chunk, min(k, (z + 1) * k_chunk))`.
fn tiled(launch: &Launch<'_>, gid: [u32; 3], out: &mut Writes, tile_m: usize) -> CoreResult<()> {
    let args: AudioGemmArgs = launch.bytes()?;
    let sh = shape(&args)?;
    let z = extent(gid[2], "audio gemm z")?;
    let splits = extent(args.splits, "audio gemm splits")?;
    let k_chunk = extent(args.k_chunk, "audio gemm k_chunk")?;
    // A slice boundary inside a block would cut the weight's blocks.
    if k_chunk == 0 || !k_chunk.is_multiple_of(QK8_0) {
        return Err(invalid(
            "audio gemm k_chunk",
            format!("{k_chunk} is not a positive multiple of {QK8_0}"),
        ));
    }
    if splits == 0 || z >= splits {
        return Err(invalid(
            "audio gemm z",
            format!("plane {z} of {splits} splits"),
        ));
    }
    let kb = (z * k_chunk).min(sh.k);
    let ke = sh.k.min(kb + k_chunk);
    let (m0, n0) = (
        extent(gid[1], "audio gemm y")? * tile_m,
        extent(gid[0], "audio gemm x")? * TILE_N,
    );
    if m0 >= sh.m || n0 >= sh.n {
        return Ok(());
    }
    let (x, w) = (launch.f32(1)?, launch.raw(2)?);
    // A partial plane carries no bias: the split sum adds it once.
    let bias = if sh.bias && splits <= 1 {
        Some(floats(
            launch.f32(3)?,
            n0,
            TILE_N.min(sh.n - n0),
            "audio gemm bias",
        )?)
    } else {
        None
    };
    let cols = TILE_N.min(sh.n - n0);
    let mut vals = Vec::with_capacity(cols);
    for r in m0..sh.m.min(m0 + tile_m) {
        vals.clear();
        for j in 0..cols {
            let v = dot(x, w, &sh, (r, n0 + j), (kb, ke))?;
            vals.push(bias.map_or(v, |b| v + b[j]));
        }
        out.f32(0, z * sh.m * sh.n + r * sh.n + n0, &vals);
    }
    Ok(())
}

/// Up to eight rows of up to sixteen columns, each a whole-`k` dot.
fn gemv_group(launch: &Launch<'_>, gid: [u32; 3], out: &mut Writes) -> CoreResult<()> {
    let args: AudioGemmArgs = launch.bytes()?;
    let sh = shape(&args)?;
    let (n0, m0) = (
        extent(gid[0], "audio gemv x")? * GEMV_COLS,
        extent(gid[1], "audio gemv y")? * GEMV_ROWS,
    );
    if n0 >= sh.n || m0 >= sh.m {
        return Ok(());
    }
    let cols = GEMV_COLS.min(sh.n - n0);
    let (x, w) = (launch.f32(1)?, launch.raw(2)?);
    let bias = if sh.bias {
        Some(floats(launch.f32(3)?, n0, cols, "audio gemv bias")?)
    } else {
        None
    };
    let mut vals = Vec::with_capacity(cols);
    for r in m0..sh.m.min(m0 + GEMV_ROWS) {
        vals.clear();
        for j in 0..cols {
            let v = dot(x, w, &sh, (r, n0 + j), (0, sh.k))?;
            vals.push(bias.map_or(v, |b| v + b[j]));
        }
        out.f32(0, r * sh.n + n0, &vals);
    }
    Ok(())
}

/// `y[i] = sum_z parts[z][i] (+ bias[i % n])`, the planes summed in order.
fn split_sum_group(launch: &Launch<'_>, gid: [u32; 3], out: &mut Writes) -> CoreResult<()> {
    let args: AudioGemmArgs = launch.bytes()?;
    let (m, n) = (
        extent(args.m, "split_sum m")?,
        extent(args.n, "split_sum n")?,
    );
    let splits = extent(args.splits, "split_sum splits")?;
    let count = m * n;
    let lo = (extent(gid[0], "split_sum x")? * LINEAR_BLOCK as usize).min(count);
    let hi = (lo + LINEAR_BLOCK as usize).min(count);
    if lo == hi {
        return Ok(());
    }
    let parts = launch.f32(1)?;
    let bias = if args.flags & HAS_BIAS != 0 {
        Some(floats(launch.f32(2)?, 0, n, "split_sum bias")?)
    } else {
        None
    };
    let mut vals = vec![0.0f32; hi - lo];
    for z in 0..splits {
        let plane = floats(parts, z * count + lo, hi - lo, "split_sum parts")?;
        for (t, p) in vals.iter_mut().zip(plane) {
            *t += p;
        }
    }
    if let Some(b) = bias {
        for (i, t) in vals.iter_mut().enumerate() {
            *t += b[(lo + i) % n];
        }
    }
    out.f32(0, lo, &vals);
    Ok(())
}
