//! The engine's half of the paged key/value cache.
//!
//! Pages, refcounts, victim choice, checkpoints and the disk tier belong to the
//! cache crate. What lives here is the part only the engine can do: turn a shape
//! profile into the byte sizes a cache is configured with, hold the device
//! memory those bytes live in, move a page's rows between pool pages, and keep
//! one session's page table on the device so a kernel can turn a token position
//! into the row that holds it.
//!
//! # The one addressing rule
//!
//! A token position reaches a cache row through its session's page table:
//! `pages[position / page_tokens] * page_tokens + position % page_tokens`.
//! Nothing else in this crate computes a cache row, and no kernel is handed a
//! row computed anywhere else. Two sessions that share a span agree about which
//! bytes are theirs because they read the same table.
//!
//! # Why one page is spread over several buffers
//!
//! A page covers every attention layer, because a token's key and value exist
//! once per layer and a span is only reusable when all of them are present. Each
//! attention layer therefore owns its own key and value allocation, and a page
//! is the run each of those buffers holds at one row range. Copying a page
//! copies all of those runs: a page is not one contiguous extent, and nothing
//! here pretends it is.
//!
//! A page is priced run by run, through [`KvLayout`], rather than as one run size
//! times a layer count. A family whose attention layers are not all the same width
//! has runs of different sizes and no single run size to multiply, so the byte
//! total, the allocation and the three page walks all read the same per-run list.

use std::sync::Arc;

use crate::shape::HybridShape;
use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, DeviceView, alloc};
use shiso_core::error::{CoreError, CoreResult};
pub(crate) use shiso_core::page::PAGE_TOKENS;
use shiso_kvstore::{Backing, CacheGeometry, Frontier, KvError, PageId};

/// Bytes per stored element: the cache is f16, because that is what the store
/// kernel writes and what both attention kernels read.
const CACHE_ELEMENT_BYTES: u64 = 2;

/// Bytes per element of the recurrent state, which is f32 end to end.
const STATE_ELEMENT_BYTES: u64 = 4;

/// Bytes of one page-table slot: the kernels index pages with `i32`.
///
/// The table is device memory a session uploads whenever its frontier grows,
/// and its width is a kernel-side fact, so it lives beside the kernels' layout
/// rather than beside the cache's byte accounting.
pub(crate) const PAGE_TABLE_ENTRY_BYTES: u64 = 4;

/// Seconds one prefill token is expected to cost, until a measurement exists.
///
/// The cache weighs a re-prefill against a re-use with this number, so it only
/// has to be within a factor of reality to choose the same victim. The bench
/// runner replaces it with a measured value once one is recorded.
const ESTIMATED_PREFILL_S_PER_TOKEN: f64 = 0.004;

/// How long a span stays interesting after it was last touched.
const UTILITY_DECAY_S: f64 = 600.0;

/// How many allocations one page is spread over inside one attention layer:
/// its key rows and its value rows.
pub(crate) const RUNS_PER_LAYER: u64 = 2;

/// The device runs one page is spread over, in allocation order.
///
/// One entry per key or value allocation; `run_bytes[i]` is the bytes one page
/// occupies in run `i`. Built from the shape by [`KvLayout::from_shape`].
///
/// `pub(crate)` rather than `pub` only because [`crate::kv`] is a private module:
/// the type is the engine's own vocabulary for what a page costs, and nothing
/// outside the crate configures a pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KvLayout {
    /// Bytes one page occupies in each run, in the order [`DeviceKv::open`]
    /// allocates them.
    pub(crate) run_bytes: Vec<u64>,
}

impl KvLayout {
    /// Key and value runs for every attention slot of `shape` that OWNS its
    /// cache, in slot order, each `PAGE_TOKENS * kv_heads * head_dim * 2` bytes.
    ///
    /// The slots come from the shape's own per-layer list rather than from an
    /// attention stride, because that list is what the loader cross-checked the
    /// stride against. Recomputing it from the stride here would let a disagreeing
    /// checkpoint through and then put key/value rows in the wrong layer's buffer.
    ///
    /// A slot that reads an earlier slot's runs contributes nothing: its keys and
    /// values are already in the pool, written once by the slot that owns them,
    /// and a second pair of runs for them would be a second copy of the same rows
    /// that nothing would ever write.
    pub(crate) fn from_shape(shape: &HybridShape) -> Self {
        let mut run_bytes = Vec::new();
        for (_, layer) in shape.kv_owning_attn() {
            let run = u64::from(PAGE_TOKENS)
                .saturating_mul(u64::from(layer.kv_heads))
                .saturating_mul(u64::from(layer.head_dim))
                .saturating_mul(CACHE_ELEMENT_BYTES);
            for _ in 0..RUNS_PER_LAYER {
                run_bytes.push(run);
            }
        }
        Self { run_bytes }
    }

    /// Key and value runs for ONE attention slot of the given width.
    ///
    /// The draft head's block is an attention block on a stack that is three
    /// recurrent slots to one attention slot, so its keys and values have
    /// nowhere to live among the target's: every run in that pool belongs to a
    /// slot that writes it. It gets a pool of its own, and this is what sizes
    /// it.
    ///
    /// A constructor here rather than the same three multiplications at the
    /// call site, because the element width is [`CACHE_ELEMENT_BYTES`] and
    /// that constant is private to this module. A caller that restated the
    /// arithmetic once used `F32_BYTES` instead and over-allocated the pool by
    /// two - a gibibyte at the server's default page count, half of it slack,
    /// and nothing failed because an over-large buffer is still in bounds.
    pub(crate) fn for_one_attn_slot(kv_heads: u32, head_dim: u32) -> Self {
        let run = u64::from(PAGE_TOKENS)
            .saturating_mul(u64::from(kv_heads))
            .saturating_mul(u64::from(head_dim))
            .saturating_mul(CACHE_ELEMENT_BYTES);
        Self {
            run_bytes: vec![run; usize::try_from(RUNS_PER_LAYER).unwrap_or(2)],
        }
    }

    /// Key and value runs for `slots` attention slots of one width: a block
    /// drafter's layers, which have a pool of their own for the reason
    /// [`KvLayout::for_one_attn_slot`] gives.
    pub(crate) fn for_attn_slots(slots: u32, kv_heads: u32, head_dim: u32) -> Self {
        let one = Self::for_one_attn_slot(kv_heads, head_dim);
        Self {
            run_bytes: (0..slots).flat_map(|_| one.run_bytes.clone()).collect(),
        }
    }

    /// Sum over runs: the cache's `page_bytes`.
    pub(crate) fn page_bytes(&self) -> u64 {
        self.run_bytes.iter().copied().fold(0, u64::saturating_add)
    }
}

/// Elements in one layer's recurrent state: one square matrix per value head,
/// for a state-space stack one `head_dim x state` matrix per head, and for a
/// short-convolution stack a single placeholder.
pub(crate) fn state_elements(shape: &HybridShape) -> u64 {
    // A short-convolution block keeps no state matrix, only its window. One
    // unread float stands in for the matrix rather than zero bytes, because
    // the record is allocated, cleared, checkpointed and copied by the same
    // paths every recurrent kind takes, and a zero-length device buffer is not
    // something every backend will hand out.
    if shape.short_conv.is_some() {
        return 1;
    }
    if let Some(ssm) = shape.ssm {
        return u64::from(ssm.heads)
            .saturating_mul(u64::from(ssm.head_dim))
            .saturating_mul(u64::from(ssm.state));
    }
    u64::from(shape.gdn.v_heads)
        .saturating_mul(u64::from(shape.gdn.head_dim))
        .saturating_mul(u64::from(shape.gdn.head_dim))
}

/// Channels the recurrent projection carries: the key heads, the query heads
/// and the value heads, one head-width each - or a state-space block's
/// convolved channels, or a short-convolution block's residual width.
pub(crate) fn conv_channels(shape: &HybridShape) -> u64 {
    if shape.short_conv.is_some() {
        return u64::from(shape.hidden);
    }
    if let Some(ssm) = shape.ssm {
        return u64::from(ssm.conv_channels());
    }
    let qk = u64::from(shape.gdn.qk_heads).saturating_mul(u64::from(shape.gdn.head_dim));
    let v = u64::from(shape.gdn.v_heads).saturating_mul(u64::from(shape.gdn.head_dim));
    qk.saturating_mul(2).saturating_add(v)
}

/// Bytes of one layer's convolution window, as the conv kernel lays it out.
pub(crate) fn conv_state_bytes(shape: &HybridShape) -> u64 {
    let kernel = match (shape.short_conv, shape.ssm) {
        (Some(short), _) => short.taps,
        (None, Some(ssm)) => ssm.conv_kernel,
        (None, None) => shape.gdn.conv_kernel,
    };
    u64::from(kernel.saturating_sub(1))
        .saturating_mul(conv_channels(shape))
        .saturating_mul(STATE_ELEMENT_BYTES)
}

/// Bytes of one layer's recurrent state.
pub(crate) fn state_bytes(shape: &HybridShape) -> u64 {
    state_elements(shape).saturating_mul(STATE_ELEMENT_BYTES)
}

/// The cache geometry a profile implies.
///
/// `page_tokens` is the core's constant rather than a tunable: it is a multiple
/// of the pooled-block, tile and score-block sizes the readers assume, so a page
/// never straddles a block any of them works on.
pub(crate) fn geometry(shape: &HybridShape) -> CacheGeometry {
    let gdn_layers = shape.recurrent_count();
    // One page is every attention layer's key and value rows for `PAGE_TOKENS`
    // positions, because a span is only reusable when all of them are present.
    // The total is the sum of the runs the pool allocates rather than a separate
    // derivation, so the cache cannot be priced for a page the pool does not hold.
    let page_bytes = KvLayout::from_shape(shape).page_bytes();
    // The blob is the recurrent stack alone - every layer's conv window and
    // state matrix. The prompt-final logits row is committed beside it, and the
    // store checks a submitted blob against this number, so folding that row in
    // here would make a correct snapshot the wrong size.
    let ckpt_bytes = conv_state_bytes(shape)
        .saturating_add(state_bytes(shape))
        .saturating_mul(u64::from(gdn_layers));
    CacheGeometry::new(
        PAGE_TOKENS,
        page_bytes,
        ckpt_bytes,
        ESTIMATED_PREFILL_S_PER_TOKEN,
        UTILITY_DECAY_S,
    )
}

/// The device memory behind a cache: one key and one value allocation per
/// attention layer, each deep enough to hold the configured pool.
///
/// The cache hands this type page ids and asks for whole pages; it never learns
/// how many layers there are or how wide a row is. That split is what lets one
/// cache serve a model whose layers are laid out differently.
#[derive(Debug, Clone)]
pub(crate) struct DeviceKv {
    pool: Arc<Pool>,
}

/// The pool itself.
///
/// The cache owns this through the [`Backing`] implementation and a forward pass
/// reads the same runs through a second handle, so the runs live behind an `Arc`
/// and both see one allocation. Cloning [`DeviceKv`] hands out that second handle;
/// it never duplicates a buffer.
#[derive(Debug)]
struct Pool {
    backend: Arc<dyn Backend>,
    runs: Vec<DeviceBuffer>,
    /// Bytes one page occupies in each allocation, index for index with `runs`.
    run_bytes: Vec<u64>,
}

impl DeviceKv {
    /// Allocates the pool: one allocation per entry of `layout`, deep enough for
    /// `pool_pages` pages of that entry's run. The order is the layout's, which
    /// is key then value for every attention slot that OWNS runs, in slot order,
    /// so a layer's buffer index is its rank among those owning slots rather
    /// than its slot number.
    ///
    /// A pool the caller asked for is allocated in full: a cache too small for
    /// one session's prompt silently evicts the span that session is standing
    /// on, which is a worse failure than refusing to start.
    pub(crate) fn open(
        backend: &Arc<dyn Backend>,
        layout: &KvLayout,
        pool_pages: u32,
    ) -> CoreResult<Self> {
        let mut runs = Vec::with_capacity(layout.run_bytes.len());
        for run_bytes in &layout.run_bytes {
            let bytes = u64::from(pool_pages)
                .checked_mul(*run_bytes)
                .ok_or_else(|| CoreError::InvalidValue {
                    key: "kv_pool_pages",
                    value: format!(
                        "{pool_pages} pages of {run_bytes} bytes overflow a 64-bit size"
                    ),
                })?;
            runs.push(alloc(backend, bytes)?);
        }
        Ok(Self {
            pool: Arc::new(Pool {
                backend: Arc::clone(backend),
                runs,
                run_bytes: layout.run_bytes.clone(),
            }),
        })
    }

    /// The two whole allocations one attention layer reads, as key then value.
    ///
    /// The three kernels that touch the cache take a layer's whole allocation and
    /// turn a position into a row themselves, so this hands out the entire buffer
    /// rather than a page: a kernel given one page could not reach the rest of the
    /// prefix it has to read. `layer_rank` is the rank among the attention slots
    /// that OWN runs, which is the order [`DeviceKv::open`] allocated in and is
    /// what [`crate::shape::HybridShape::kv_owner_rank`] answers with. A slot
    /// that reads an earlier slot's cache asks for the owner's rank, not its
    /// own, and there is no allocation under its own.
    pub(crate) fn layer_runs(
        &self,
        layer_rank: u32,
    ) -> CoreResult<(DeviceView<'_>, DeviceView<'_>)> {
        let base = u64::from(layer_rank).saturating_mul(RUNS_PER_LAYER);
        Ok((self.whole(base)?, self.whole(base.saturating_add(1))?))
    }

    /// One whole allocation by its index in the pool.
    ///
    /// The failure is reported in the terms the index is in: `offset` is the run
    /// index that was asked for and `buffer_bytes` is how many runs the pool has.
    /// `len` carries [`Self::page_bytes`], the bytes a whole page occupies across
    /// the pool, because a run that does not exist has no size of its own to name
    /// and a zero there would read as "asked for nothing".
    fn whole(&self, index: u64) -> CoreResult<DeviceView<'_>> {
        let slot = usize::try_from(index).unwrap_or(usize::MAX);
        self.pool
            .runs
            .get(slot)
            .map(DeviceBuffer::view)
            .ok_or(CoreError::OutOfBounds {
                offset: index,
                len: self.page_bytes(),
                buffer_bytes: self.pool.runs.len() as u64,
            })
    }

    /// The run one page occupies inside one layer's allocation, whose size is
    /// that allocation's own `run_bytes` rather than a pool-wide number.
    fn run(buffer: &DeviceBuffer, run_bytes: u64, page: PageId) -> CoreResult<DeviceView<'_>> {
        let offset =
            u64::from(u32::from(page))
                .checked_mul(run_bytes)
                .ok_or(CoreError::OutOfBounds {
                    offset: u64::from(u32::from(page)),
                    len: run_bytes,
                    buffer_bytes: buffer.len(),
                })?;
        buffer.slice(offset, run_bytes)
    }

    /// Bytes of one page inside one allocation, as a host-side length: the
    /// stride that run occupies in the host page buffer.
    fn stride(run_bytes: u64) -> Result<usize, KvError> {
        usize::try_from(run_bytes).map_err(|_| KvError::BufferLen {
            want: usize::MAX,
            got: 0,
        })
    }

    /// Bytes of a whole page across the pool: every run's size, summed.
    ///
    /// This is the cache's `page_bytes` arrived at a second way, which is the
    /// point: the cache hands over a buffer sized from
    /// [`KvLayout::page_bytes`], and a walk that disagreed with it would write
    /// one run's bytes over another's.
    fn page_bytes(&self) -> u64 {
        self.pool
            .run_bytes
            .iter()
            .copied()
            .fold(0, u64::saturating_add)
    }

    /// The same total as a host-side length, for the page buffer the cache passes.
    fn host_page_bytes(&self) -> Result<usize, KvError> {
        Self::stride(self.page_bytes())
    }
}

/// Presents a device failure to the cache.
///
/// The cache's vocabulary has one variant that can carry another error's
/// message, so a driver failure travels as an I/O failure on a named operation
/// rather than as a geometry complaint about a size that was fine.
fn transfer_failure(op: &str, source: CoreError) -> KvError {
    KvError::Io {
        path: std::path::PathBuf::from(op),
        source: std::io::Error::other(source),
    }
}

impl Backing for DeviceKv {
    fn copy_page(&mut self, dst: PageId, src: PageId) -> Result<(), KvError> {
        for (buffer, run_bytes) in self.pool.runs.iter().zip(self.pool.run_bytes.iter()) {
            let into =
                Self::run(buffer, *run_bytes, dst).map_err(|e| transfer_failure("copy_page", e))?;
            let from =
                Self::run(buffer, *run_bytes, src).map_err(|e| transfer_failure("copy_page", e))?;
            self.pool
                .backend
                .copy_d2d(into, from)
                .map_err(|e| transfer_failure("copy_page", e))?;
        }
        Ok(())
    }

    /// Writes the page's runs end to end, in buffer order: run 0's page bytes,
    /// then run 1's, and so on.
    ///
    /// The cache hands over exactly `page_bytes`, which is the sum of this
    /// type's run sizes, so the layout the writer uses is the layout the reader
    /// restores: both walk `runs` in the same order with the same per-run
    /// strides. The host offset runs along with the walk rather than being
    /// `index * stride`, because two runs need not be the same size.
    fn page_to_host(&self, page: PageId, buf: &mut [u8]) -> Result<(), KvError> {
        let want = self.host_page_bytes()?;
        if buf.len() < want {
            return Err(KvError::BufferLen {
                want,
                got: buf.len(),
            });
        }
        let mut start = 0usize;
        for (buffer, run_bytes) in self.pool.runs.iter().zip(self.pool.run_bytes.iter()) {
            let stride = Self::stride(*run_bytes)?;
            let run = Self::run(buffer, *run_bytes, page)
                .map_err(|e| transfer_failure("page_to_host", e))?;
            let end = start.saturating_add(stride);
            let Some(slot) = buf.get_mut(start..end) else {
                return Err(KvError::BufferLen {
                    want: end,
                    got: buf.len(),
                });
            };
            self.pool
                .backend
                .copy_d2h(run, slot)
                .map_err(|e| transfer_failure("page_to_host", e))?;
            start = end;
        }
        Ok(())
    }

    fn page_from_host(&mut self, page: PageId, buf: &[u8]) -> Result<(), KvError> {
        let want = self.host_page_bytes()?;
        if buf.len() < want {
            return Err(KvError::BufferLen {
                want,
                got: buf.len(),
            });
        }
        let mut start = 0usize;
        for (buffer, run_bytes) in self.pool.runs.iter().zip(self.pool.run_bytes.iter()) {
            let stride = Self::stride(*run_bytes)?;
            let run = Self::run(buffer, *run_bytes, page)
                .map_err(|e| transfer_failure("page_from_host", e))?;
            let end = start.saturating_add(stride);
            let Some(slot) = buf.get(start..end) else {
                return Err(KvError::BufferLen {
                    want: end,
                    got: buf.len(),
                });
            };
            self.pool
                .backend
                .copy_h2d(run, slot)
                .map_err(|e| transfer_failure("page_from_host", e))?;
            start = end;
        }
        Ok(())
    }
}

/// What a claim may evict to make room: the cache its pages come from, and
/// the time that cache prices its spans at.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Room<'a> {
    /// The cache whose idle spans may be dropped.
    pub(crate) cache: &'a shiso_kvstore::KvCache,
    /// Seconds on the stack's clock.
    pub(crate) now: f64,
}

/// One session's view of the cache: where its token positions live, and the
/// device copy of that table the kernels read.
///
/// The table is allocated once, wide enough for the session's context capacity,
/// and only its contents move. A launch therefore never races a reallocation. A
/// kernel may read past the used entries: the unused slots hold zero, which is a
/// page the session has no claim on, and no row it points at is ever read,
/// because the causal bound stops the loop before it.
#[derive(Debug)]
pub(crate) struct SessionKv {
    backend: Arc<dyn Backend>,
    frontier: Frontier,
    pages_dev: DeviceBuffer,
    uploaded: usize,
}

impl SessionKv {
    /// Wraps a cold frontier with a device table sized for `capacity` positions.
    pub(crate) fn open(
        backend: &Arc<dyn Backend>,
        frontier: Frontier,
        capacity: u32,
    ) -> CoreResult<Self> {
        let slots = shiso_core::page::pages_for(capacity);
        let bytes = u64::from(slots).checked_mul(PAGE_TABLE_ENTRY_BYTES).ok_or(
            CoreError::InvalidValue {
                key: "ctx",
                value: format!("{slots} page slots do not fit a device table"),
            },
        )?;
        let pages_dev = alloc(backend, bytes)?;
        let mut session = Self {
            backend: Arc::clone(backend),
            frontier,
            pages_dev,
            uploaded: 0,
        };
        session.upload_table()?;
        Ok(session)
    }

    /// The cache's view of this session's path.
    pub(crate) fn frontier(&self) -> &Frontier {
        &self.frontier
    }

    /// The path the next commit extends.
    pub(crate) fn frontier_mut(&mut self) -> &mut Frontier {
        &mut self.frontier
    }

    /// Takes `frontier` in place of the one this session was standing on, and puts
    /// the new page ids where the kernels read them.
    ///
    /// The old frontier is dropped here rather than returned, which is what releases
    /// its page references: a session that kept both would hold two claims on the
    /// same prefix and the pool would never see the first one back. The device table
    /// is re-uploaded in the same step, so there is no moment where the session's
    /// pages and the table the kernels index disagree.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when the table cannot be uploaded, and
    /// [`CoreError::InvalidValue`] when a page id does not fit the width a kernel
    /// indexes with.
    pub(crate) fn adopt(&mut self, frontier: Frontier) -> CoreResult<()> {
        self.frontier = frontier;
        self.uploaded = 0;
        self.upload_table()
    }

    /// The table a KV-touching kernel indexes. Every KV-touching launch takes
    /// this; none of them computes a row itself.
    pub(crate) fn view(&self) -> DeviceView<'_> {
        self.pages_dev.view()
    }

    /// Claims pages up to `positions` and re-uploads the table if that moved.
    ///
    /// Returns whether the table changed, which is what a caller uses to decide
    /// whether a launch that already captured the old contents must be redone.
    /// Claiming is the engine's only way to make a cache row writable: a page the
    /// frontier has not claimed may be handed to another session at any moment.
    ///
    /// When the pool is short, idle spans are evicted from `room` first - the
    /// cache keeps every finished conversation's pages as a reusable prefix
    /// until something needs them, so a pool that has served a few requests is
    /// full of them. A pool that still cannot supply the pages is a refusal.
    /// It used to be read as "nothing changed", and the pass then wrote its rows
    /// through table slots that named no page of this session's.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] when the pool cannot supply the pages even
    /// after eviction, and whatever the table upload reports.
    pub(crate) fn ensure_positions(&mut self, positions: u32, room: Room<'_>) -> CoreResult<bool> {
        let before = self.frontier.pages();
        let held = u32::try_from(before).unwrap_or(u32::MAX);
        let want = shiso_core::page::pages_for(positions).saturating_sub(held);
        if want > 0 {
            let report = room.cache.evict_to(want, room.now);
            if report.pages_freed > 0 {
                tracing::debug!(
                    want,
                    positions,
                    pages_freed = report.pages_freed,
                    spans = report.nodes_dropped,
                    checkpoints = report.ckpts_dropped,
                    "evicted idle spans for a claim"
                );
            }
        }
        let covered = self
            .frontier
            .grow_to(positions)
            .map_err(|e| CoreError::InvalidValue {
                key: "kv pool",
                value: format!("cannot claim pages for {positions} tokens: {e}"),
            })?;
        if !covered {
            return Err(CoreError::InvalidValue {
                key: "kv pool",
                value: format!("the pool has no free page for position {positions}"),
            });
        }
        if self.frontier.pages() != before || self.uploaded != self.frontier.pages() {
            self.upload_table()?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Copies the frontier's page ids into the device table as `i32` slots,
    /// which is the width the kernels index with.
    fn upload_table(&mut self) -> CoreResult<()> {
        let pages = self.frontier.table().as_slice();
        let mut bytes = Vec::with_capacity(pages.len().saturating_mul(4));
        for page in pages {
            let slot = i32::try_from(u32::from(*page)).map_err(|_| CoreError::InvalidValue {
                key: "page id",
                value: format!("{page:?} does not fit the int32 a kernel reads"),
            })?;
            bytes.extend(slot.to_le_bytes());
        }
        let len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        self.backend
            .copy_h2d(self.pages_dev.slice(0, len)?, &bytes)?;
        self.uploaded = pages.len();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{KvLayout, PAGE_TOKENS, geometry};
    use crate::shape::{
        Activation, AttnLayer, GdnShape, HybridShape, LayerKind, MediaPositions, NormScheme,
    };
    use shiso_core::ids::LayerIdx;

    /// A stack of nothing but attention slots, one per entry of `attn`.
    fn dense(attn: Vec<AttnLayer>) -> HybridShape {
        HybridShape {
            layers: vec![LayerKind::Attn; attn.len()],
            attn,
            hidden: 2048,
            vocab: 128,
            intermediate: 4096,
            gdn: GdnShape::new(0, 0, 0, 1),
            attn_output_gate: true,
            rms_eps: 1e-6,
            activation: Activation::SwiGlu,
            norms: NormScheme::PreOnly,
            embed_scale: None,
            final_softcap: None,
            layer_scalar: false,
            ple: None,
            media_positions: MediaPositions::Mrope3D,
            v_norm: false,
            moe: None,
            ssm: None,
            short_conv: None,
            no_ffn: Vec::new(),
        }
    }

    /// A stack of two attention slots and nothing else, with the widths the
    /// qwen35 path actually runs: four key/value heads of 256 channels.
    fn shape() -> HybridShape {
        dense(vec![AttnLayer::new(16, 4, 256, 128, 128, 10_000.0); 2])
    }

    #[test]
    fn a_page_is_one_key_and_one_value_run_per_attention_slot() {
        let layout = KvLayout::from_shape(&shape());
        // 256 positions x 4 key/value heads x 256 channels x 2 bytes.
        assert_eq!(
            layout.run_bytes,
            vec![524_288; 4],
            "two attention slots contribute two runs each, in slot order"
        );
        assert_eq!(u64::from(PAGE_TOKENS), 256);
    }

    #[test]
    fn a_page_costs_the_sum_of_its_runs() {
        let layout = KvLayout::from_shape(&shape());
        assert_eq!(layout.page_bytes(), 2_097_152);
        assert_eq!(
            geometry(&shape()).page_bytes,
            layout.page_bytes(),
            "the cache is priced with exactly the runs the pool allocates"
        );
    }

    #[test]
    fn only_a_slot_that_owns_its_cache_is_priced_a_run() {
        // Six attention slots: four own runs and two read an earlier slot's, and
        // one of the owners is twice as wide as the others. The pool therefore
        // holds eight runs of two different sizes, which is the whole reason a
        // page is priced run by run.
        let mut attn = vec![AttnLayer::new(8, 2, 256, 256, 256, 1e4); 6];
        attn[3] = AttnLayer::new(8, 2, 512, 128, 512, 1e6);
        attn[4] = attn[2].shared_from(LayerIdx::new(2));
        attn[5] = attn[3].shared_from(LayerIdx::new(3));
        let shape = dense(attn);
        shape.validate().expect("the shape is valid");
        let layout = KvLayout::from_shape(&shape);
        assert_eq!(
            layout.run_bytes,
            vec![
                262_144, 262_144, 262_144, 262_144, 262_144, 262_144, 524_288, 524_288
            ],
            "three 256-wide owners and one 512-wide one, key then value each"
        );
        assert_eq!(
            layout.run_bytes.len(),
            8,
            "the two slots that read another's runs add none of their own"
        );
    }

    #[test]
    fn a_stack_with_no_recurrent_slots_checkpoints_nothing() {
        // The checkpoint blob is the recurrent stack alone, so a family that has
        // none commits an empty one: the store has to accept that, and the pool
        // still has pages to price.
        assert_eq!(geometry(&shape()).ckpt_bytes, 0);
        assert_ne!(geometry(&shape()).page_bytes, 0);
    }
}
