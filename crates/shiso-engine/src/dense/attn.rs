//! The dense attention block: a query projection, a paged cache a slot may not
//! own, and two softmaxes over a span.
//!
//! The kernels and their geometry come from `metal/dense/KERNELS.md`. Three
//! things about this block are unlike qwen35's, and all three show up in the
//! launches below rather than in the kernels' names: the query projection is
//! exactly as wide as the query because this family has no output gate, the
//! rotation's frequency ladder has a width of its own, and what a query may see
//! is a span with two independent ends rather than "everything at or before me".
//!
//! # A slot that writes nothing
//!
//! [`AttnGeometry::owns_kv`] is false on a slot whose cache another slot filled.
//! Such a slot projects no keys and no values and launches no store: the runs in
//! its [`AttnCache`] are the owner's, written earlier in the same pass, and a
//! second store into them would overwrite a layer's keys with a different
//! layer's. That is why the three key weights are reached through [`Attn::kv`]
//! and only on the owning branch — a slot bound as an owner without them is a
//! loader bug, and it is refused here rather than half-launched.
//!
//! # Where a cache row comes from
//!
//! Nothing in this module computes a cache row. The kernels that touch the cache
//! take absolute positions and the session's page table and translate a position
//! themselves, so the engine never computes a cache offset and therefore cannot
//! get one wrong.
//!
//! # The order below is not free
//!
//! The store runs before the softmax, because the softmax reads the cache: a
//! chunk's first token has to see the keys the same chunk just stored, so a
//! launch order that deferred the cache write would silently drop each token's
//! own position out of its own attention.

use shiso_core::Extent3;
use shiso_core::backend::{Backend, KernelArg, KernelRef};
use shiso_core::buffer::{DeviceBuffer, DeviceView};
use shiso_core::error::{CoreError, CoreResult};

use crate::attn::{Attn, AttnCache, AttnGeometry, AttnScratch, Pages, Positions};
use crate::blocks::AttnIo;
use crate::launch::{
    self, BLOCK_128, BLOCK_LINEAR, DENSE, DENSE_ATTN_DECODE_SMEM_BYTES,
    DENSE_ATTN_PREFILL_SMEM_BYTES, DENSE_ATTN_PREFILL_TILE, DENSE_NORM_ROPE_SMEM_BYTES,
    DenseAttnArgs, DenseKvNormRopeArgs, DenseQNormRopeArgs,
};
use crate::projection;

/// The chunk's linear positions, or a refusal for a chunk carrying media.
///
/// Three-stream positions are a multimodal fact this kernel set has no entry
/// point for. Refusing here rather than silently rotating by the temporal stream
/// is the difference between a request that fails and a request that answers
/// wrongly.
fn linear_positions(pos: Positions<'_>) -> CoreResult<DeviceView<'_>> {
    match pos {
        Positions::Linear(linear) => Ok(linear),
        Positions::Mrope { .. } => Err(CoreError::Unsupported {
            what: "three-stream positions on the dense attention path",
            backend: "engine",
        }),
    }
}

/// Normalizes and rotates the query.
fn q_norm_rope(
    backend: &dyn Backend,
    attn: &Attn,
    geometry: &AttnGeometry,
    scratch: &AttnScratch,
    pos: DeviceView<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    const NAME: &str = "kernel_dense_q_norm_rope";
    let args = DenseQNormRopeArgs {
        head_dim: geometry.head_dim,
        n_head: geometry.n_head,
        n_rot: geometry.n_rot,
        freq_dims: geometry.freq_dims,
        n_tok,
        rope_base: geometry.rope_base,
        eps: geometry.eps,
    };
    launch::run(
        backend,
        KernelRef::new(DENSE, NAME),
        Extent3::new(geometry.n_head, n_tok, 1),
        Extent3::new(BLOCK_128, 1, 1),
        DENSE_NORM_ROPE_SMEM_BYTES,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(scratch.q.view()),
            // The projection wrote rows `n_head * head_dim` wide, packed: the
            // row stride is the weight's output width, not the scratch's. What
            // the gated width buys is slack at the END of the buffer, because
            // one scratch serves every slot of the stack and a family with no
            // gate simply leaves that tail alone.
            KernelArg::Buffer(scratch.qkv.view()),
            KernelArg::Buffer(attn.q_norm()?.view()),
            KernelArg::Buffer(pos),
        ],
    )
    .map_err(|err| launch::launch_failure(NAME, err))
}

/// Normalizes and rotates the keys, normalizes the values when the layer asks,
/// and writes both halves of the cache.
///
/// One launch for the two halves because a position's key and value land in the
/// same row, and one launch is what keeps them in step.
fn kv_norm_rope_store(
    backend: &dyn Backend,
    k_norm: &DeviceBuffer,
    geometry: &AttnGeometry,
    scratch: &AttnScratch,
    cache: &AttnCache<'_>,
    pos: DeviceView<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    const NAME: &str = "kernel_dense_kv_norm_rope_store";
    let args = DenseKvNormRopeArgs {
        head_dim: geometry.head_dim,
        n_head_kv: geometry.n_head_kv,
        n_rot: geometry.n_rot,
        freq_dims: geometry.freq_dims,
        n_tok,
        rope_base: geometry.rope_base,
        eps: geometry.eps,
        page_tokens: geometry.page_tokens,
        v_norm: u32::from(geometry.v_norm),
        page_stride: cache.pages.stride(),
    };
    launch::run(
        backend,
        KernelRef::new(DENSE, NAME),
        Extent3::new(geometry.n_head_kv, n_tok, 1),
        Extent3::new(BLOCK_128, 1, 1),
        DENSE_NORM_ROPE_SMEM_BYTES,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(cache.k),
            KernelArg::Buffer(cache.v),
            KernelArg::Buffer(scratch.k.view()),
            KernelArg::Buffer(scratch.v.view()),
            KernelArg::Buffer(k_norm.view()),
            KernelArg::Buffer(pos),
            KernelArg::Buffer(cache.pages.table()),
        ],
    )
    .map_err(|err| launch::launch_failure(NAME, err))
}

/// Writes keys and values into the cache as they came out of their
/// projections, for a slot that neither normalizes nor rotates them.
fn kv_store_plain(
    backend: &dyn Backend,
    geometry: &AttnGeometry,
    scratch: &AttnScratch,
    cache: &AttnCache<'_>,
    pos: DeviceView<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    const NAME: &str = "kernel_dense_kv_store_f32";
    let args = DenseKvNormRopeArgs {
        head_dim: geometry.head_dim,
        n_head_kv: geometry.n_head_kv,
        n_rot: 0,
        freq_dims: 0,
        n_tok,
        rope_base: 0.0,
        eps: 0.0,
        page_tokens: geometry.page_tokens,
        v_norm: 0,
        page_stride: cache.pages.stride(),
    };
    launch::run(
        backend,
        KernelRef::new(DENSE, NAME),
        Extent3::new(geometry.n_head_kv, n_tok, 1),
        Extent3::new(BLOCK_128, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(cache.k),
            KernelArg::Buffer(cache.v),
            KernelArg::Buffer(scratch.k.view()),
            KernelArg::Buffer(scratch.v.view()),
            KernelArg::Buffer(pos),
            KernelArg::Buffer(cache.pages.table()),
        ],
    )
    .map_err(|err| launch::launch_failure(NAME, err))
}

/// Projects the keys and values of `n_tok` rows and writes them into the
/// cache, through the store the slot's shape names.
fn project_and_store(
    backend: &dyn Backend,
    attn: &Attn,
    geometry: &AttnGeometry,
    scratch: &AttnScratch,
    cache: &AttnCache<'_>,
    rows: StoreRows<'_>,
) -> CoreResult<()> {
    if geometry.qk_norm {
        let (k_proj, v_proj, k_norm) = attn.kv()?;
        projection::projections(
            backend,
            rows.n_tok,
            rows.input,
            &[(k_proj, scratch.k.view()), (v_proj, scratch.v.view())],
        )?;
        kv_norm_rope_store(
            backend, k_norm, geometry, scratch, cache, rows.pos, rows.n_tok,
        )
    } else {
        let (k_proj, v_proj) = attn.kv_plain()?;
        projection::projections(
            backend,
            rows.n_tok,
            rows.input,
            &[(k_proj, scratch.k.view()), (v_proj, scratch.v.view())],
        )?;
        kv_store_plain(backend, geometry, scratch, cache, rows.pos, rows.n_tok)
    }
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

/// Runs the softmax over the session's cache.
///
/// The two entry points are one softmax over the same rows. The prefill one tiles
/// tokens so a chunk costs one launch instead of one per token, and below one tile
/// it would stage a key tile for tokens a single decode threadgroup would have
/// covered in one pass, which is the whole reason the threshold exists.
fn softmax(
    backend: &dyn Backend,
    geometry: &AttnGeometry,
    scratch: &AttnScratch,
    cache: &AttnCache<'_>,
    io: AttnIo<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    let args = DenseAttnArgs {
        head_dim: geometry.head_dim,
        n_head: geometry.n_head,
        n_head_kv: geometry.n_head_kv,
        n_tok,
        page_tokens: geometry.page_tokens,
        window: geometry.window,
        scale: geometry.scale,
        page_stride: cache.pages.stride(),
    };
    let (name, grid, smem) = if n_tok >= DENSE_ATTN_PREFILL_TILE {
        one_table_per_tile(cache.pages)?;
        (
            "kernel_dense_attn_prefill",
            Extent3::new(geometry.n_head, n_tok.div_ceil(DENSE_ATTN_PREFILL_TILE), 1),
            DENSE_ATTN_PREFILL_SMEM_BYTES,
        )
    } else {
        // A head dimension the slot count fits exactly takes its own
        // instantiation: the same arithmetic with the per-lane arrays sized at
        // compile time, which keeps them out of memory.
        let name = match geometry.head_dim {
            128 => "kernel_dense_attn_decode_hd128",
            256 => "kernel_dense_attn_decode_hd256",
            _ => "kernel_dense_attn_decode",
        };
        (
            name,
            Extent3::new(geometry.n_head, n_tok, 1),
            DENSE_ATTN_DECODE_SMEM_BYTES,
        )
    };
    launch::run(
        backend,
        KernelRef::new(DENSE, name),
        grid,
        Extent3::new(BLOCK_LINEAR, 1, 1),
        smem,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(scratch.attn.view()),
            KernelArg::Buffer(scratch.q.view()),
            KernelArg::Buffer(cache.k),
            KernelArg::Buffer(cache.v),
            // The window's low end is measured from the query's own position and
            // its high end is the bound the caller supplies. Neither is
            // derivable from the other, so both buffers are bound.
            KernelArg::Buffer(io.pos.linear()),
            KernelArg::Buffer(io.bound),
            KernelArg::Buffer(cache.pages.table()),
        ],
    )
    .map_err(|err| launch::launch_failure(name, err))
}

/// Writes `n_tok` rows of keys and values into the cache and nothing else.
///
/// The key and value halves of [`run`], for a caller whose rows are context
/// other queries will attend to later but that ask nothing themselves: a block
/// drafter's view of the target's committed positions. The launches are
/// [`run`]'s own, so a row stored here reads exactly as one [`run`] stored.
///
/// # Errors
///
/// As [`run`].
pub(crate) fn store(
    backend: &dyn Backend,
    attn: &Attn,
    geometry: &AttnGeometry,
    scratch: &AttnScratch,
    cache: &AttnCache<'_>,
    rows: StoreRows<'_>,
) -> CoreResult<()> {
    project_and_store(backend, attn, geometry, scratch, cache, rows)
}

/// The rows [`store`] writes: their inputs, their positions and how many.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StoreRows<'a> {
    /// `int32` positions, one a row.
    pub(crate) pos: DeviceView<'a>,
    /// The normalized rows the key and value projections read.
    pub(crate) input: DeviceView<'a>,
    /// Rows.
    pub(crate) n_tok: u32,
}

/// Runs one dense attention block over `n_tok` rows of `io.input`.
///
/// `io.pos` carries the chunk's `int32` positions and `cache` is the session's
/// own: a slot that owns its cache writes the rows those positions name, and
/// every slot reads whatever its span and the session's table say the prefix
/// occupies. `io.block_out` receives the block's contribution to the residual
/// stream and belongs to the caller, which is what adds it in.
///
/// # Errors
///
/// [`CoreError::Unsupported`] for a chunk carrying three-stream positions, which
/// this kernel set has no entry point for; [`CoreError::InvalidValue`] when a
/// slot that owns its cache was bound without key/value projections; whatever the
/// projection family reports for a width or format it cannot carry; and
/// [`CoreError::Backend`] naming the kernel when a launch fails.
pub fn run(
    backend: &dyn Backend,
    attn: &Attn,
    geometry: &AttnGeometry,
    scratch: &AttnScratch,
    io: AttnIo<'_>,
    cache: &AttnCache<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    let pos = linear_positions(io.pos)?;
    if geometry.qk_norm {
        projection::projection(backend, &attn.q, n_tok, io.input, scratch.qkv.view())?;
        q_norm_rope(backend, attn, geometry, scratch, pos, n_tok)?;
    } else {
        // Nothing to do to the query, so the projection writes where the
        // softmax reads.
        projection::projection(backend, &attn.q, n_tok, io.input, scratch.q.view())?;
    }
    // A shared-cache slot launches neither projection nor store: the runs its
    // `cache` names are the owner's, already filled earlier in this same pass.
    if geometry.owns_kv {
        project_and_store(
            backend,
            attn,
            geometry,
            scratch,
            cache,
            StoreRows {
                pos,
                input: io.input,
                n_tok,
            },
        )?;
    }
    softmax(backend, geometry, scratch, cache, io, n_tok)?;
    projection::projection(
        backend,
        &attn.output,
        n_tok,
        scratch.attn.view(),
        io.block_out,
    )
}
