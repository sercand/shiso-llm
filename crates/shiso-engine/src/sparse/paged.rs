//! The sparse stack's half of the paged key/value cache.
//!
//! Pages, refcounts, victim choice, checkpoints and the disk tier belong to the
//! cache crate; there is no second pool here and no second span tree. What
//! lives here is the part only a forward pass can do: say how many bytes a page
//! of *this* stack is, hold the device memory those bytes live in, move a
//! page's runs between pool pages, and keep one session's page table where a
//! kernel can read it.
//!
//! # A page is more than a key and a value here
//!
//! In the dense stack a page is each attention layer's key rows and value rows
//! for one run of positions. Here it is that plus the indexer's two planes: the
//! raw key row every position stores, and the pooled key row that stands for a
//! whole block of them. All four have to be present for a span to be reusable,
//! because an attention block that resumed on a span without its pooled keys
//! would select blocks from another session's rows.
//!
//! The pooled plane is the reason the page size is what it is: it is a multiple
//! of the pooling ratio, so a pooled block never straddles a page and a page's
//! pooled rows are exactly its own share of the pooled table.
//!
//! # Growth invalidates captured graphs
//!
//! A captured launch bakes in the address of the page table it was handed. A
//! session whose frontier grows past its table's end therefore cannot replay
//! anything captured against the old table, and [`SessionPages::ensure`]
//! reports that so the caller can drop the cache before it publishes the new
//! pointer. The table is allocated once at its context capacity here, so the
//! pointer in practice never moves - but the report is what makes that a
//! checked property rather than an assumption.

use std::sync::Arc;

use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, DeviceView, alloc_tagged};
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ledger::MemTag;
use shiso_core::page::{PAGE_TOKENS, pages_for};
use shiso_kvstore::{Backing, CacheGeometry, Frontier, KvError, PageId};

use crate::shape::LayerKind;
use crate::sparse::kernels::{F16_BYTES, F32_BYTES, I32_BYTES};
use crate::sparse::mtp::SpecDepth;
use crate::sparse::shape::SparseShape;

/// Seconds one prefill token is expected to cost, until a measurement exists.
///
/// The cache weighs a re-prefill against a re-use with this number, so it only
/// has to be within a factor of reality to choose the same victim.
const ESTIMATED_PREFILL_S_PER_TOKEN: f64 = 0.004;

/// How long a span stays interesting after it was last touched.
const UTILITY_DECAY_S: f64 = 600.0;

/// Which plane of a page one run holds.
///
/// Named rather than positional because the four have different widths: a
/// reader that walked them by index alone and got the order wrong would read
/// finite numbers out of the wrong plane, which is the failure this whole
/// module is arranged to prevent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Plane {
    /// Attention keys for one layer.
    Key,
    /// Attention values for one layer.
    Value,
    /// The indexer's raw key row, one per position.
    IndexerKey,
    /// The indexer's pooled key row, one per block of positions.
    IndexerPooled,
}

/// The four planes in the order a page lays them out, per attention layer.
///
/// One array rather than four call sites, because the order is the layout: the
/// allocator, the page copier and the host serializer all walk it, and they
/// agree because there is one list.
pub const PLANES: [Plane; 4] = [
    Plane::Key,
    Plane::Value,
    Plane::IndexerKey,
    Plane::IndexerPooled,
];

impl Plane {
    /// Bytes this plane occupies for one page of positions.
    #[must_use]
    pub fn page_bytes(self, shape: &SparseShape) -> u64 {
        let positions = u64::from(PAGE_TOKENS);
        match self {
            Self::Key | Self::Value => positions
                .saturating_mul(u64::from(shape.kv_dim()))
                .saturating_mul(F16_BYTES),
            Self::IndexerKey => positions
                .saturating_mul(u64::from(shape.indexer.head_dim))
                .saturating_mul(F16_BYTES),
            Self::IndexerPooled => {
                let compress = u64::from(shape.indexer.compress).max(1);
                (positions / compress)
                    .saturating_mul(u64::from(shape.indexer.head_dim))
                    .saturating_mul(F16_BYTES)
            }
        }
    }
}

/// Whether a stack's pages carry the indexer planes and the draft head's.
///
/// Two independent facts, and they are a struct rather than two `bool`
/// arguments because at a call site `page_bytes(shape, true, false)` reads
/// identically to `page_bytes(shape, false, true)` and means something else.
///
/// Deliberately NOT `#[non_exhaustive]`: a caller has to build one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PagePlanes {
    /// Whether the indexer's raw and pooled planes are part of a page.
    pub indexer: bool,
    /// Whether a draft head's own key, value and indexer planes are too.
    ///
    /// The draft head runs the same indexer over its own rows, so it
    /// contributes a full attention layer's worth of planes rather than only a
    /// key and a value.
    pub draft: bool,
}

impl PagePlanes {
    /// Both planes present, which is what a stack with an indexer and a draft
    /// head runs.
    #[must_use]
    pub const fn full() -> Self {
        Self {
            indexer: true,
            draft: true,
        }
    }

    /// The indexer planes only.
    #[must_use]
    pub const fn indexer_only() -> Self {
        Self {
            indexer: true,
            draft: false,
        }
    }
}

/// Bytes one pool page occupies across every cache buffer.
///
/// Derived from the geometry rather than stated: one page is [`PAGE_TOKENS`]
/// positions of every attention layer's keys and values, of the indexer's raw
/// and quarter-rate pooled keys, and - when a draft head is resident - of that
/// head's own four planes.
///
/// A page is *not* one contiguous extent. It is the run each of those buffers
/// holds at one row range, and nothing here pretends otherwise: copying a page
/// copies every run.
#[must_use]
pub fn page_bytes(shape: &SparseShape, planes: PagePlanes) -> u64 {
    let attn = u64::from(shape.count(LayerKind::Attn));
    let kv = Plane::Key
        .page_bytes(shape)
        .saturating_add(Plane::Value.page_bytes(shape));
    let idx = Plane::IndexerKey
        .page_bytes(shape)
        .saturating_add(Plane::IndexerPooled.page_bytes(shape));
    let mut bytes = attn.saturating_mul(kv);
    if planes.indexer {
        bytes = bytes.saturating_add(attn.saturating_mul(idx));
    }
    if planes.draft {
        bytes = bytes.saturating_add(kv);
        if planes.indexer {
            bytes = bytes.saturating_add(idx);
        }
    }
    bytes
}

/// Bytes one checkpoint blob occupies.
///
/// The recurrent stack is most of it: every recurrent slot's convolution window
/// and its state matrix. The rest is what a resume cannot reconstruct from the
/// cache either - the n-gram block's dilated window, and the pending draft row
/// a speculative step left staged.
///
/// The draft row is there because a resumed session that dropped it could not
/// draft on its first step, and the first step is the one a dropped row would
/// change: the head's proposal for it is the one that was computed from the
/// residual the original run produced.
///
/// The frontier logits row is deliberately NOT here. It is committed beside the
/// blob rather than inside it, and the store checks a submitted blob against
/// this number - so folding the row in would make a correct snapshot the wrong
/// size and every save would be refused.
///
/// `spec` is what decides whether the draft row is part of the blob at all, and
/// it has to: the store derives the same length from the stored layout's own
/// draft-head word, and a blob whose length did not follow that word would be
/// read back with its sections shifted.
#[must_use]
pub fn ckpt_bytes(shape: &SparseShape, spec: SpecDepth) -> u64 {
    let gdn = u64::from(shape.count(LayerKind::Gdn));
    let per_slot = conv_state_bytes(shape).saturating_add(state_bytes(shape));
    let draft = if spec.is_speculative() {
        draft_row_bytes(shape)
    } else {
        0
    };
    gdn.saturating_mul(per_slot)
        .saturating_add(ple_window_bytes(shape))
        .saturating_add(draft)
}

/// Bytes of the residual row a speculative step leaves staged for the draft
/// head.
///
/// One row of the whole hyper-connected residual, which is every stream's share
/// of the hidden width.
#[must_use]
pub fn draft_row_bytes(shape: &SparseShape) -> u64 {
    u64::from(shape.stream_dim()).saturating_mul(F32_BYTES)
}

/// Bytes of one recurrent slot's convolution window.
#[must_use]
pub fn conv_state_bytes(shape: &SparseShape) -> u64 {
    u64::from(shape.gdn.conv_kernel.saturating_sub(1))
        .saturating_mul(u64::from(shape.gdn_in_dim()))
        .saturating_mul(F32_BYTES)
}

/// Bytes of one recurrent slot's state matrix: one square matrix per value
/// head.
#[must_use]
pub fn state_bytes(shape: &SparseShape) -> u64 {
    u64::from(shape.gdn.v_heads)
        .saturating_mul(u64::from(shape.gdn.head_dim))
        .saturating_mul(u64::from(shape.gdn.head_dim))
        .saturating_mul(F32_BYTES)
}

/// Bytes of the n-gram block's dilated convolution window.
#[must_use]
pub fn ple_window_bytes(shape: &SparseShape) -> u64 {
    shape.ple.map_or(0, |ple| {
        u64::from(ple.history())
            .saturating_mul(u64::from(shape.stream_dim()))
            .saturating_mul(F32_BYTES)
    })
}

/// The cache geometry this shape implies.
#[must_use]
pub fn cache_geometry(shape: &SparseShape, planes: PagePlanes, spec: SpecDepth) -> CacheGeometry {
    CacheGeometry::new(
        PAGE_TOKENS,
        page_bytes(shape, planes),
        ckpt_bytes(shape, spec),
        ESTIMATED_PREFILL_S_PER_TOKEN,
        UTILITY_DECAY_S,
    )
}

/// One allocation behind the pool, and the bytes one page occupies inside it.
#[derive(Debug)]
struct Run {
    buffer: DeviceBuffer,
    stride: u64,
}

/// The pool itself.
///
/// The cache owns this through [`Backing`] and a forward pass reads the same
/// runs through a second handle, so the runs live behind an `Arc` and both see
/// one allocation. Cloning [`PagedKv`] hands out that second handle; it never
/// duplicates a buffer.
#[derive(Debug)]
struct Pool {
    backend: Arc<dyn Backend>,
    runs: Vec<Run>,
    planes: usize,
}

/// The device memory behind a sparse stack's cache.
///
/// The cache hands this type page ids and asks for whole pages; it never learns
/// how many layers there are or how wide a row is. That split is what lets one
/// cache serve a model whose planes are laid out differently.
#[derive(Debug, Clone)]
pub struct PagedKv {
    pool: Arc<Pool>,
}

impl PagedKv {
    /// Device bytes one page occupies across every run, which is what the
    /// pool was allocated at.
    #[must_use]
    pub fn page_bytes(&self) -> u64 {
        self.pool
            .runs
            .iter()
            .map(|run| run.stride)
            .fold(0, u64::saturating_add)
    }

    /// Allocates the pool: [`PLANES`] allocations per attention layer, in layer
    /// order, each deep enough for `pool_pages` pages.
    ///
    /// A pool the caller asked for is allocated in full. A cache too small for
    /// one session's prompt silently evicts the span that session is standing
    /// on, which is a worse failure than refusing to start.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] when the requested depth overflows a 64-bit
    /// size, and [`CoreError::Backend`] when an allocation fails.
    pub fn open(
        backend: &Arc<dyn Backend>,
        shape: &SparseShape,
        planes: PagePlanes,
        pool_pages: u32,
    ) -> CoreResult<Self> {
        // The draft head is one more attention slot's worth of planes, ranked
        // after every target slot. It is *not* a layer of the shape - the
        // target's pass must not run it - so the count comes from the planes a
        // page was sized for rather than from the layer list.
        let attn = shape
            .count(LayerKind::Attn)
            .saturating_add(u32::from(planes.draft));
        let mut runs = Vec::with_capacity(PLANES.len().saturating_mul(attn as usize));
        for layer in 0..attn {
            // The draft head's rank is the last one, past every target slot.
            let draft = planes.draft && layer + 1 == attn;
            for plane in PLANES {
                let tag = match (draft, plane) {
                    (true, _) => MemTag::KvDraft,
                    (false, Plane::Key | Plane::Value) => MemTag::KvAttention,
                    (false, Plane::IndexerKey | Plane::IndexerPooled) => MemTag::KvIndexer,
                };
                let stride = plane.page_bytes(shape);
                let bytes = u64::from(pool_pages).checked_mul(stride).ok_or_else(|| {
                    CoreError::InvalidValue {
                        key: "kv_pool_pages",
                        value: format!(
                            "{pool_pages} pages of {stride} bytes overflow a 64-bit size"
                        ),
                    }
                })?;
                runs.push(Run {
                    buffer: alloc_tagged(backend, tag, bytes)?,
                    stride,
                });
            }
        }
        Ok(Self {
            pool: Arc::new(Pool {
                backend: Arc::clone(backend),
                runs,
                planes: PLANES.len(),
            }),
        })
    }

    /// One attention layer's whole allocation for one plane.
    ///
    /// The kernels that touch the cache take a layer's whole allocation and
    /// turn a position into a row themselves, so this hands out the entire
    /// buffer rather than a page: a kernel given one page could not reach the
    /// rest of the prefix it has to read.
    ///
    /// `rank` is the layer's rank among attention slots, which is the order
    /// [`PagedKv::open`] allocated in.
    ///
    /// # Errors
    ///
    /// [`CoreError::OutOfBounds`] when no allocation carries that rank.
    pub fn plane(&self, rank: u32, plane: Plane) -> CoreResult<DeviceView<'_>> {
        let offset = PLANES.iter().position(|p| *p == plane).unwrap_or(0);
        let index = (rank as usize)
            .saturating_mul(self.pool.planes)
            .saturating_add(offset);
        self.pool
            .runs
            .get(index)
            .map(|run| run.buffer.view())
            .ok_or(CoreError::OutOfBounds {
                offset: index as u64,
                len: 1,
                buffer_bytes: self.pool.runs.len() as u64,
            })
    }

    /// The run one page occupies inside one allocation.
    fn run(run: &Run, page: PageId) -> CoreResult<DeviceView<'_>> {
        let offset =
            u64::from(u32::from(page))
                .checked_mul(run.stride)
                .ok_or(CoreError::OutOfBounds {
                    offset: u64::from(u32::from(page)),
                    len: run.stride,
                    buffer_bytes: run.buffer.len(),
                })?;
        run.buffer.slice(offset, run.stride)
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

impl Backing for PagedKv {
    fn copy_page(&mut self, dst: PageId, src: PageId) -> Result<(), KvError> {
        for run in &self.pool.runs {
            let into = Self::run(run, dst).map_err(|e| transfer_failure("copy_page", e))?;
            let from = Self::run(run, src).map_err(|e| transfer_failure("copy_page", e))?;
            self.pool
                .backend
                .copy_d2d(into, from)
                .map_err(|e| transfer_failure("copy_page", e))?;
        }
        Ok(())
    }

    /// Writes the page's runs end to end, in allocation order.
    ///
    /// The cache hands over exactly the page size this module reported, which
    /// is the sum of the run strides, so the layout the writer uses is the
    /// layout the reader restores: both walk `runs` in the same order.
    fn page_to_host(&self, page: PageId, buf: &mut [u8]) -> Result<(), KvError> {
        let mut at = 0usize;
        for run in &self.pool.runs {
            let width = usize::try_from(run.stride).unwrap_or(usize::MAX);
            let end = at.saturating_add(width);
            let Some(slot) = buf.get_mut(at..end) else {
                return Err(KvError::BufferLen {
                    want: end,
                    got: buf.len(),
                });
            };
            let view = Self::run(run, page).map_err(|e| transfer_failure("page_to_host", e))?;
            self.pool
                .backend
                .copy_d2h(view, slot)
                .map_err(|e| transfer_failure("page_to_host", e))?;
            at = end;
        }
        Ok(())
    }

    fn page_from_host(&mut self, page: PageId, buf: &[u8]) -> Result<(), KvError> {
        let mut at = 0usize;
        for run in &self.pool.runs {
            let width = usize::try_from(run.stride).unwrap_or(usize::MAX);
            let end = at.saturating_add(width);
            let Some(slot) = buf.get(at..end) else {
                return Err(KvError::BufferLen {
                    want: end,
                    got: buf.len(),
                });
            };
            let view = Self::run(run, page).map_err(|e| transfer_failure("page_from_host", e))?;
            self.pool
                .backend
                .copy_h2d(view, slot)
                .map_err(|e| transfer_failure("page_from_host", e))?;
            at = end;
        }
        Ok(())
    }
}

/// What a growth of a session's page table means for anything captured against
/// it.
///
/// A `bool` return would be a fact with no name at the call site; this one
/// says what the caller has to do about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TableChange {
    /// The table is byte for byte what it was, so a captured launch still names
    /// the rows it was captured over.
    Unchanged,
    /// New page ids reached the table. Every captured launch that reads it has
    /// to be discarded before the new contents are published, because a capture
    /// baked in the table it saw.
    Republished,
}

/// One session's view of the cache: where its positions live, and the device
/// copy of that table the kernels read.
///
/// The table is allocated once, wide enough for the session's context capacity,
/// and only its contents move - so a launch never races a reallocation. A
/// kernel may read past the used entries, and several do: a split attention
/// divides the key range into a fixed number of slices and a slice with no keys
/// still resolves a row before it finds that out. The unused slots therefore
/// have to hold zero - page zero, a row that exists and is masked out - and
/// [`SessionPages::open`] is what makes that true. A device allocator returns
/// raw pages, so without it the slots past the claim hold the *previous*
/// tenant's page ids, which resolve to rows inside the pool that some other
/// sequence wrote. The first session on a fresh allocation reads zeros anyway
/// and answers correctly; the third one reads a stranger's pages, every score
/// in the empty slices comes back enormous, and the softmax the slices fold
/// into is zero for every row.
#[derive(Debug)]
pub struct SessionPages {
    backend: Arc<dyn Backend>,
    frontier: Frontier,
    table: DeviceBuffer,
    uploaded: usize,
}

impl SessionPages {
    /// Wraps a cold frontier with a device table sized for `capacity`
    /// positions.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] when the capacity does not fit a table, and
    /// [`CoreError::Backend`] when the allocation or the first upload fails.
    pub fn open(backend: &Arc<dyn Backend>, frontier: Frontier, capacity: u32) -> CoreResult<Self> {
        let slots = pages_for(capacity);
        let bytes = u64::from(slots)
            .checked_mul(I32_BYTES)
            .ok_or(CoreError::InvalidValue {
                key: "ctx",
                value: format!("{slots} page slots do not fit a device table"),
            })?;
        let table = alloc_tagged(backend, MemTag::PageTables, bytes)?;
        // Every slot past the claim has to read as page zero; see the type's
        // own notes for what happens when it does not.
        backend.fill_zero(table.view())?;
        let mut session = Self {
            backend: Arc::clone(backend),
            frontier,
            table,
            uploaded: 0,
        };
        session.upload()?;
        Ok(session)
    }

    /// The cache's view of this session's path.
    #[must_use]
    pub const fn frontier(&self) -> &Frontier {
        &self.frontier
    }

    /// The path the next commit extends.
    pub const fn frontier_mut(&mut self) -> &mut Frontier {
        &mut self.frontier
    }

    /// The table a cache-touching kernel indexes. Every such launch takes this;
    /// none of them computes a row itself.
    #[must_use]
    pub fn view(&self) -> DeviceView<'_> {
        self.table.view()
    }

    /// Takes `frontier` in place of the one this session was standing on and
    /// republishes the table.
    ///
    /// The old frontier is dropped here rather than returned, which is what
    /// releases its page references: a session that kept both would hold two
    /// claims on the same prefix and the pool would never see the first one
    /// back.
    ///
    /// # Errors
    ///
    /// Whatever the upload reports.
    pub fn adopt(&mut self, frontier: Frontier) -> CoreResult<TableChange> {
        self.frontier = frontier;
        self.uploaded = 0;
        self.upload()?;
        Ok(TableChange::Republished)
    }

    /// Pages this session still has to claim to hold `positions` tokens.
    #[must_use]
    pub fn shortfall(&self, positions: u32) -> u32 {
        let held = u32::try_from(self.frontier.pages()).unwrap_or(u32::MAX);
        pages_for(positions).saturating_sub(held)
    }

    /// Claims pages up to `positions` and republishes the table if that moved.
    ///
    /// Claiming is the only way to make a cache row writable: a page the
    /// frontier has not claimed may be handed to another session at any moment,
    /// which is why this happens before the launches that fill those rows and
    /// not after.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] when the pool cannot supply the pages - which
    /// it reports by declining rather than by failing, so the refusal is turned
    /// into an error here rather than being read as success - and
    /// [`CoreError::Backend`] when the upload fails.
    pub fn ensure(&mut self, positions: u32) -> CoreResult<TableChange> {
        let before = self.frontier.pages();
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
            self.upload()?;
            return Ok(TableChange::Republished);
        }
        Ok(TableChange::Unchanged)
    }

    /// Copies the frontier's page ids into the device table as `int32` slots,
    /// which is the width the kernels index with.
    fn upload(&mut self) -> CoreResult<()> {
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
        self.backend.copy_h2d(self.table.slice(0, len)?, &bytes)?;
        self.uploaded = pages.len();
        Ok(())
    }
}
