//! The qwen35 attention block: a fused query projection, a paged cache, and two
//! softmaxes.
//!
//! The kernels and their geometry come from `metal/qwen35/KERNELS.md`. Two things
//! about this block are unlike the recurrent one, and both show up in the order of
//! launches here: the query projection is twice as wide as the query because the
//! second half of every head's row is the output gate, and the cache is reached
//! through a session's page table rather than by row number.
//!
//! # Where a cache row comes from
//!
//! Nothing in this module computes a cache row. The three kernels that touch the
//! cache take absolute positions and the session's page table and translate a
//! position into a row themselves, so the engine never computes a cache offset and
//! therefore cannot get one wrong. What this module is responsible for is narrower
//! and still worth stating: the table handed to a kernel is the one the session
//! uploaded, and the rows a kernel writes are rows that session had claimed.
//!
//! # Why the gate is split out at the norm
//!
//! The query projection is twice as wide as the query because the second half of
//! every head's row is the output gate. The normalization kernel writes the
//! normalized, rotated query and the raw gate as two contiguous rows, which is
//! what lets `kernel_q35_attn_gate_mul` index both operands flat; carrying the
//! gate inside the query row would have made every later kernel carry a
//! `2 * head_dim` stride for no gain.
//!
//! # The order below is not free
//!
//! The key kernel writes the cache before the softmax runs, because the softmax
//! reads the cache: a chunk's first token has to see the keys the same chunk just
//! stored, so a launch order that deferred the cache write would silently drop
//! each token's own position out of its own attention.

use shiso_core::Extent3;
use shiso_core::backend::{Backend, KernelArg, KernelRef};
use shiso_core::buffer::{DeviceBuffer, DeviceView};
use shiso_core::error::{CoreError, CoreResult};

use shiso_engine::attn::{Attn, AttnCache, AttnGeometry, AttnScratch, Pages, Positions};
use shiso_engine::launch::{
    self, ATTN_DECODE_SMEM_BYTES, ATTN_PREFILL_SMEM_BYTES, ATTN_PREFILL_TILE, AttnArgs,
    AttnGateArgs, BLOCK_128, BLOCK_LINEAR, KNormRopeArgs, NORM_ROPE_SMEM_BYTES, Q35, QNormRopeArgs,
};
use shiso_engine::projection;

/// Normalizes and rotates the query, splitting the gate out of the projection row.
fn q_norm_rope(
    backend: &dyn Backend,
    attn: &Attn,
    geometry: &AttnGeometry,
    scratch: &AttnScratch,
    pos: Positions<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    let args = QNormRopeArgs {
        head_dim: geometry.head_dim,
        n_head: geometry.n_head,
        n_rot: geometry.n_rot,
        n_tok,
        rope_base: geometry.rope_base,
        eps: geometry.eps,
    };
    let mut binds = vec![
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(scratch.q.view()),
        KernelArg::Buffer(scratch.gate.view()),
        KernelArg::Buffer(scratch.qkv.view()),
        KernelArg::Buffer(attn.q_norm.view()),
    ];
    // The one-dimensional entry takes the sequence position; the multimodal one
    // takes the three grid streams and no sequence position at all, because
    // nothing in it addresses memory by one.
    let name = match pos {
        Positions::Linear(linear) => {
            binds.push(KernelArg::Buffer(linear));
            "kernel_q35_q_norm_rope"
        }
        Positions::Mrope { t, h, w, .. } => {
            binds.push(KernelArg::Buffer(t));
            binds.push(KernelArg::Buffer(h));
            binds.push(KernelArg::Buffer(w));
            "kernel_q35_q_norm_rope_mrope"
        }
    };
    launch::run(
        backend,
        KernelRef::new(Q35, name),
        Extent3::new(geometry.n_head, n_tok, 1),
        Extent3::new(BLOCK_128, 1, 1),
        NORM_ROPE_SMEM_BYTES,
        &binds,
    )
    .map_err(|err| launch::launch_failure(name, err))
}

/// Normalizes and rotates the keys and writes both halves of the cache.
///
/// The kernel writes the cache itself, in f16, at the row the session's table gives
/// the position. The value projection is handed to this kernel rather than copied
/// separately because a position's key and value land in the same row, and one
/// launch is what keeps the two halves of a row in step.
fn k_norm_rope_kv(
    backend: &dyn Backend,
    k_norm: &DeviceBuffer,
    geometry: &AttnGeometry,
    scratch: &AttnScratch,
    cache: &AttnCache<'_>,
    pos: Positions<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    let args = KNormRopeArgs {
        head_dim: geometry.head_dim,
        n_head_kv: geometry.n_head_kv,
        n_rot: geometry.n_rot,
        n_tok,
        rope_base: geometry.rope_base,
        eps: geometry.eps,
        page_tokens: geometry.page_tokens,
        page_stride: cache.pages.stride(),
    };
    // The sequence position is bound either way, and only it reaches the page
    // table. Handing the temporal stream to the translation instead would fold
    // a whole image onto a handful of cache rows with every count still right.
    let mut binds = vec![
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(cache.k),
        KernelArg::Buffer(cache.v),
        KernelArg::Buffer(scratch.k.view()),
        KernelArg::Buffer(scratch.v.view()),
        KernelArg::Buffer(k_norm.view()),
        KernelArg::Buffer(pos.linear()),
        KernelArg::Buffer(cache.pages.table()),
    ];
    let name = match pos {
        Positions::Linear(_) => "kernel_q35_k_norm_rope_kv",
        Positions::Mrope { t, h, w, .. } => {
            binds.push(KernelArg::Buffer(t));
            binds.push(KernelArg::Buffer(h));
            binds.push(KernelArg::Buffer(w));
            "kernel_q35_k_norm_rope_kv_mrope"
        }
    };
    launch::run(
        backend,
        KernelRef::new(Q35, name),
        Extent3::new(geometry.n_head_kv, n_tok, 1),
        Extent3::new(BLOCK_128, 1, 1),
        NORM_ROPE_SMEM_BYTES,
        &binds,
    )
    .map_err(|err| launch::launch_failure(name, err))
}

/// Runs the softmax over the session's cache.
///
/// The two entry points are one softmax over the same rows. The prefill one tiles
/// tokens so a chunk costs one launch instead of one per token, and below one tile
/// it would stage a tile for tokens a single decode threadgroup would have covered
/// in one pass, which is the whole reason the threshold exists.
fn softmax(
    backend: &dyn Backend,
    geometry: &AttnGeometry,
    scratch: &AttnScratch,
    cache: &AttnCache<'_>,
    pos: DeviceView<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    let args = AttnArgs {
        head_dim: geometry.head_dim,
        n_head: geometry.n_head,
        n_head_kv: geometry.n_head_kv,
        n_tok,
        page_tokens: geometry.page_tokens,
        page_stride: cache.pages.stride(),
    };
    let (name, grid, smem) = if n_tok >= ATTN_PREFILL_TILE {
        one_table_per_tile(cache.pages)?;
        (
            "kernel_q35_attn_prefill",
            Extent3::new(geometry.n_head, n_tok.div_ceil(ATTN_PREFILL_TILE), 1),
            ATTN_PREFILL_SMEM_BYTES,
        )
    } else {
        (
            "kernel_q35_attn_decode",
            Extent3::new(geometry.n_head, n_tok, 1),
            ATTN_DECODE_SMEM_BYTES,
        )
    };
    launch::run(
        backend,
        KernelRef::new(Q35, name),
        grid,
        Extent3::new(BLOCK_LINEAR, 1, 1),
        smem,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(scratch.attn.view()),
            KernelArg::Buffer(scratch.q.view()),
            KernelArg::Buffer(cache.k),
            KernelArg::Buffer(cache.v),
            KernelArg::Buffer(pos),
            KernelArg::Buffer(cache.pages.table()),
        ],
    )
    .map_err(|err| launch::launch_failure(name, err))
}

/// Refuses a per-row page table on an entry point that tiles rows.
///
/// The tiled softmax puts several query rows in one threadgroup and translates
/// each key ONCE for the whole tile, so it has no row index to select a table
/// with. A pass that reached it with per-row tables would silently read the
/// tile's first row's table for every row in it. That cannot happen today - a
/// decode batch is at most `MAX_DECODE_BATCH` rows and the tile threshold is
/// wider - but "cannot happen today" is exactly the kind of invariant a later
/// widening breaks in silence, so it is checked rather than commented.
fn one_table_per_tile(pages: Pages<'_>) -> CoreResult<()> {
    if pages.is_shared() {
        return Ok(());
    }
    Err(CoreError::Unsupported {
        what: "a per-row page table on the tiled attention entry, which translates \
               one key per tile and has no row index to select a table with",
        backend: "engine",
    })
}

/// Scales each head's output by its own sigmoid gate, in place.
fn gate_mul(
    backend: &dyn Backend,
    geometry: &AttnGeometry,
    scratch: &AttnScratch,
    n_tok: u32,
) -> CoreResult<()> {
    let args = AttnGateArgs {
        head_dim: geometry.head_dim,
        n_head: geometry.n_head,
        n_tok,
    };
    let elements = n_tok
        .saturating_mul(geometry.n_head)
        .saturating_mul(geometry.head_dim);
    launch::run(
        backend,
        KernelRef::new(Q35, "kernel_q35_attn_gate_mul"),
        Extent3::linear(elements.div_ceil(BLOCK_LINEAR)),
        Extent3::new(BLOCK_LINEAR, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(scratch.attn.view()),
            KernelArg::Buffer(scratch.gate.view()),
        ],
    )
    .map_err(|err| launch::launch_failure("kernel_q35_attn_gate_mul", err))
}

/// Runs one attention block over `n_tok` rows of `input`.
///
/// `pos` carries the chunk's `int32` positions and `cache` is the session's own:
/// the block writes the rows those positions name and reads everything the table
/// says the prefix occupies. `block_out` receives the block's contribution to the
/// residual stream and belongs to the caller, which is what adds it in.
#[expect(
    clippy::too_many_arguments,
    reason = "the positions and the cache runs are the session's, and a struct for \
              six views would only name the same arguments twice"
)]
/// # Errors
///
/// Whatever the projection family reports for a width or format it cannot carry,
/// and [`shiso_core::CoreError::Backend`] naming the kernel when a launch fails.
pub(super) fn run(
    backend: &dyn Backend,
    attn: &Attn,
    geometry: &AttnGeometry,
    scratch: &AttnScratch,
    input: DeviceView<'_>,
    pos: Positions<'_>,
    cache: &AttnCache<'_>,
    block_out: DeviceView<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    // Every qwen35 slot owns its cache, so all three are present; asking once
    // here is what makes the four launches below unconditional.
    let (k_proj, v_proj, k_norm) = attn.kv()?;
    projection::projection(backend, &attn.q, n_tok, input, scratch.qkv.view())?;
    projection::projection(backend, k_proj, n_tok, input, scratch.k.view())?;
    projection::projection(backend, v_proj, n_tok, input, scratch.v.view())?;
    q_norm_rope(backend, attn, geometry, scratch, pos, n_tok)?;
    k_norm_rope_kv(backend, k_norm, geometry, scratch, cache, pos, n_tok)?;
    // The softmax reads sequence positions only: what a query may see is a
    // question about order, and a placeholder token's grid coordinates say
    // nothing about it.
    softmax(backend, geometry, scratch, cache, pos.linear(), n_tok)?;
    gate_mul(backend, geometry, scratch, n_tok)?;
    projection::projection(backend, &attn.output, n_tok, scratch.attn.view(), block_out)
}
