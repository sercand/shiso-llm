//! `kernel_get_rows_q8_0_f32`: one f32 row per token id, dequantized from a
//! `Q8_0` table.
//!
//! Group `(block, token)` owns the 32 outputs of one block of one token's row,
//! so the grid is `ceil(n_embd / 32)` by `n_tokens`.

use shiso_core::{CoreError, CoreResult, Extent3};

use crate::dispatch::{ConstRule, GroupsDisjoint, KernelSpec, Launch, Role, Writes};
use crate::kernels::args::GetRowsArgs;
use crate::q8::{BLOCK_Q8_0_BYTES, QK8_0, q8_dequant_block};

use super::{count, f32_index};

/// Table, ids, destination.
pub(in crate::kernels) const Q8_0: KernelSpec = KernelSpec {
    group: q8_0_group,
    roles: &[Role::In, Role::In, Role::Out],
    consts: ConstRule::None,
    block: |b: Extent3| b == Extent3::new(32, 1, 1),
    disjoint: GroupsDisjoint::Yes,
};

fn q8_0_group(launch: &Launch<'_>, gid: [u32; 3], out: &mut Writes) -> CoreResult<()> {
    let args: GetRowsArgs = launch.bytes()?;
    let n_embd = count(args.n_embd, "get_rows n_embd")?;
    let block = gid[0] as usize;
    let tok = gid[1] as usize;
    // The grid may be taller than the token count; the extra groups own nothing.
    if tok >= count(args.n_tokens, "get_rows n_tokens")? {
        return Ok(());
    }
    let ids = launch.raw(1)?;
    let at = (tok as u64).saturating_mul(args.token_stride);
    let id_bytes = usize::try_from(at)
        .ok()
        .and_then(|at| ids.get(at..at.checked_add(4)?))
        .ok_or_else(|| short("get_rows id", tok))?;
    let token = i32::from_le_bytes([id_bytes[0], id_bytes[1], id_bytes[2], id_bytes[3]]);
    // An id outside the table leaves its row untouched, as the device kernel
    // does: the engine pads a batch with ids it never reads back.
    let Some(token) = usize::try_from(token).ok().filter(|_| token < args.n_vocab) else {
        return Ok(());
    };
    let i0 = block * QK8_0;
    if i0 >= n_embd {
        return Ok(());
    }
    let table = launch.raw(0)?;
    let start = (token as u64)
        .saturating_mul(args.src_row_bytes)
        .saturating_add((block * BLOCK_Q8_0_BYTES) as u64);
    let src = usize::try_from(start)
        .ok()
        .and_then(|s| table.get(s..s.checked_add(BLOCK_Q8_0_BYTES)?))
        .ok_or_else(|| short("get_rows table", token))?;
    let mut values = [0.0f32; QK8_0];
    q8_dequant_block(src, &mut values);
    let take = QK8_0.min(n_embd - i0);
    let dst = (tok as u64).saturating_mul(args.dst_row_bytes);
    out.f32(
        2,
        f32_index(dst, "get_rows dst_row_bytes")? + i0,
        &values[..take],
    );
    Ok(())
}

fn short(what: &'static str, at: usize) -> CoreError {
    CoreError::InvalidValue {
        key: what,
        value: format!("row {at} lies outside its buffer"),
    }
}
