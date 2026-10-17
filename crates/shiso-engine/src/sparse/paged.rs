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
//! for one run of positions. Here it is that plus the indexer's pooled key row,
//! one for each block of [`IndexerShape::compress`] positions. All three have
//! to be present for a span to be reusable, because an attention block that
//! resumed on a span without its pooled keys would select blocks from another
//! session's rows.
//!
//! The pooled plane is the reason the page size is what it is: it is a multiple
//! of the pooling ratio, so a pooled block never straddles a page and a page's
//! pooled rows are exactly its own share of the pooled table.
//!
//! # The raw indexer keys are not a plane
//!
//! A pooled row is the mean of its block's raw keys, and nothing else ever
//! reads a raw key: not the scorers, which read pooled rows, and not the
//! attention, which reads the incomplete block's tail out of the key and value
//! planes. So once a block is pooled its raw keys are dead, and the only ones a
//! session ever needs again are those of the block its frontier stands inside -
//! at most `compress - 1` of them. A plane that kept one per position spent a
//! quarter of a key's width on every cached token to hold three live rows per
//! session; those rows live in each session's raw key ring instead
//! ([`ring_bytes`]), and travel in its checkpoint blob rather than in its pages.
//!
//! [`IndexerShape::compress`]: crate::sparse::shape::IndexerShape::compress
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
use shiso_core::page::{KvElem, PAGE_TOKENS, pages_for};
use shiso_kvstore::{Backing, CacheGeometry, Frontier, KvError, PageId};

use crate::shape::LayerKind;
use crate::sparse::kernels::{F16_BYTES, F32_BYTES, I32_BYTES, IDX_RING};
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
/// Named rather than positional because the three have different widths: a
/// reader that walked them by index alone and got the order wrong would read
/// finite numbers out of the wrong plane, which is the failure this whole
/// module is arranged to prevent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Plane {
    /// Attention keys for one layer.
    Key,
    /// Attention values for one layer.
    Value,
    /// The indexer's pooled key row, one per block of positions.
    IndexerPooled,
}

/// The three planes in the order a page lays them out, per attention layer.
///
/// One array rather than three call sites, because the order is the layout:
/// the allocator, the page copier and the host serializer all walk it, and they
/// agree because there is one list.
pub const PLANES: [Plane; 3] = [Plane::Key, Plane::Value, Plane::IndexerPooled];

impl Plane {
    /// Bytes this plane occupies for one page of positions, with the attention
    /// planes stored as `kv`.
    ///
    /// Only the key and value planes take the cache element. The indexer's
    /// pooled plane stays f16 whatever it is: they decide which blocks attention reads
    /// at all, so an error there is a block never seen rather than a value seen
    /// slightly wrong.
    #[must_use]
    pub fn page_bytes(self, shape: &SparseShape, kv: KvElem) -> u64 {
        let positions = u64::from(PAGE_TOKENS);
        match self {
            Self::Key | Self::Value => positions
                .saturating_mul(u64::from(shape.kv_dim()))
                .saturating_mul(u64::from(kv.bytes())),
            Self::IndexerPooled => {
                let compress = u64::from(shape.indexer.compress).max(1);
                (positions / compress)
                    .saturating_mul(u64::from(shape.indexer.head_dim))
                    .saturating_mul(F16_BYTES)
            }
        }
    }
}

/// Whether a stack's pages carry the indexer planes and the draft head's, and
/// what element their key and value planes are stored as.
///
/// Independent facts, and they are a struct rather than positional arguments
/// because at a call site `page_bytes(shape, true, false)` reads identically to
/// `page_bytes(shape, false, true)` and means something else.
///
/// Deliberately NOT `#[non_exhaustive]`: a caller has to build one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PagePlanes {
    /// Whether the indexer's pooled plane is part of a page, and whether each
    /// session keeps a raw key ring beside it.
    pub indexer: bool,
    /// Whether a draft head's own key, value and pooled planes are too.
    ///
    /// The draft head runs the same indexer over its own rows, so it
    /// contributes a full attention layer's worth of planes rather than only a
    /// key and a value.
    pub draft: bool,
    /// The element every attention rank's key and value planes are stored as,
    /// the draft head's included: it runs the same attention kernels over its
    /// own planes, so it reads them the same way.
    pub kv: KvElem,
}

impl PagePlanes {
    /// Both planes present, which is what a stack with an indexer and a draft
    /// head runs.
    #[must_use]
    pub const fn full() -> Self {
        Self {
            indexer: true,
            draft: true,
            kv: KvElem::F16,
        }
    }

    /// The indexer planes only.
    #[must_use]
    pub const fn indexer_only() -> Self {
        Self {
            indexer: true,
            draft: false,
            kv: KvElem::F16,
        }
    }
}

/// Bytes one pool page occupies across every cache buffer.
///
/// Derived from the geometry rather than stated: one page is [`PAGE_TOKENS`]
/// positions of every attention layer's keys and values, of the indexer's
/// quarter-rate pooled keys, and - when a draft head is resident - of that
/// head's own three planes. The raw indexer keys are not here: see the
/// module's notes.
///
/// A page is *not* one contiguous extent. It is the run each of those buffers
/// holds at one row range, and nothing here pretends otherwise: copying a page
/// copies every run.
#[must_use]
pub fn page_bytes(shape: &SparseShape, planes: PagePlanes) -> u64 {
    let attn = u64::from(shape.count(LayerKind::Attn));
    let kv = Plane::Key
        .page_bytes(shape, planes.kv)
        .saturating_add(Plane::Value.page_bytes(shape, planes.kv));
    let idx = Plane::IndexerPooled.page_bytes(shape, planes.kv);
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
///
/// Last comes the session's raw indexer key ring, one section per rank the
/// pages carry (`planes`): the raw keys of the block the frontier stands
/// inside, which no page holds. It is the reason a blob from before the ring
/// is a different length - and so, with the page size, a different identity on
/// disk: a file written when the page still carried a raw key per position is
/// a miss, never a misread.
#[must_use]
pub fn ckpt_bytes(shape: &SparseShape, spec: SpecDepth, planes: PagePlanes) -> u64 {
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
        .saturating_add(ring_bytes(shape, planes))
}

/// Ranks a session keeps a raw indexer key ring for: every attention slot, and
/// the draft head's when its planes are part of a page - the same ranks the
/// pool allocates pooled planes for, in the same order.
#[must_use]
pub fn ring_ranks(shape: &SparseShape, planes: PagePlanes) -> u32 {
    if !planes.indexer {
        return 0;
    }
    shape
        .count(LayerKind::Attn)
        .saturating_add(u32::from(planes.draft))
}

/// Bytes of one rank's raw indexer key ring: [`IDX_RING`] f16 rows.
#[must_use]
pub fn ring_rank_bytes(shape: &SparseShape) -> u64 {
    u64::from(IDX_RING)
        .saturating_mul(u64::from(shape.indexer.head_dim))
        .saturating_mul(F16_BYTES)
}

/// Bytes of a session's whole raw indexer key ring, every rank's in rank order.
///
/// Per session, not per token: this is what replaced a raw key row for every
/// cached position.
#[must_use]
pub fn ring_bytes(shape: &SparseShape, planes: PagePlanes) -> u64 {
    u64::from(ring_ranks(shape, planes)).saturating_mul(ring_rank_bytes(shape))
}

/// Where the raw key ring starts in a checkpoint blob: past the draft row when
/// the stack speculates, past the n-gram window when it does not.
#[must_use]
pub fn snapshot_ring_offset(shape: &SparseShape, spec: SpecDepth) -> u64 {
    let draft = if spec.is_speculative() {
        draft_row_bytes(shape)
    } else {
        0
    };
    snapshot_draft_offset(shape).saturating_add(draft)
}

/// Where one recurrent slot's convolution window starts in a checkpoint blob,
/// and so in the stack's snapshot buffer, which is laid out as one: every slot
/// in ordinal order, each window followed by its state matrix, then the n-gram
/// window, then the draft row, then the raw indexer key ring. The order is
/// `ckpt::snapshot`'s.
#[must_use]
pub fn snapshot_conv_offset(shape: &SparseShape, slot: u32) -> u64 {
    u64::from(slot).saturating_mul(conv_state_bytes(shape).saturating_add(state_bytes(shape)))
}

/// Where one recurrent slot's state matrix starts in a checkpoint blob.
#[must_use]
pub fn snapshot_state_offset(shape: &SparseShape, slot: u32) -> u64 {
    snapshot_conv_offset(shape, slot).saturating_add(conv_state_bytes(shape))
}

/// Where the n-gram window starts in a checkpoint blob: past every slot.
#[must_use]
pub fn snapshot_window_offset(shape: &SparseShape) -> u64 {
    snapshot_conv_offset(shape, shape.count(LayerKind::Gdn))
}

/// Where the draft row starts in a speculating stack's checkpoint blob.
#[must_use]
pub fn snapshot_draft_offset(shape: &SparseShape) -> u64 {
    snapshot_window_offset(shape).saturating_add(ple_window_bytes(shape))
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

/// Bytes of one recurrent slot's rollback slab at `records` recorded rows.
///
/// The state matrix as a verify pass found it, then each recorded row's own
/// recurrence inputs - its projected row, its decay and its beta - from which
/// a rollback replays the kept rows (`q4e_gdn_replay_kernel`). A state matrix
/// a row instead was 3 MB a layer and a draft for the pass to write.
#[must_use]
pub fn state_rollback_bytes(shape: &SparseShape, records: u32) -> u64 {
    let row = u64::from(shape.gdn_in_dim()).saturating_add(2 * u64::from(shape.gdn.v_heads));
    state_bytes(shape).saturating_add(
        u64::from(records)
            .saturating_mul(row)
            .saturating_mul(F32_BYTES),
    )
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

/// Where a checkpoint blob's raw key ring section sits and what is live in it.
///
/// A checkpoint at `q` stands each target rank on frontier `q` and the draft
/// head's rank, the last, on `q - 1`: the head at position `p` reads the token
/// at `p + 1`, so it has only ever written up to the position before the
/// target's frontier. The rows a resume there reads are the raw keys of the
/// block each frontier stands inside, from the block's first position up to
/// the frontier - at most `compress - 1` per rank.
///
/// Every other row of the ring is stale: a key whose block was pooled long ago,
/// a draft the target rejected, or nothing at all. None of it is ever read, but
/// left in the blob it would make two checkpoints of the same state differ by
/// whatever the ring happened to hold, so [`RingLayout::canonicalize`] zeroes
/// it. Copied into a blob as it stands, then canonicalized, the section is a
/// function of the tokens alone.
///
/// Deliberately NOT `#[non_exhaustive]`: a caller builds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingLayout {
    /// Byte offset of the section in the blob.
    pub offset: u64,
    /// Ranks the section holds, in rank order.
    pub ranks: u32,
    /// Whether the last rank is the draft head's.
    pub draft: bool,
    /// Bytes of one raw key row.
    pub row_bytes: u64,
    /// Positions one pooled block averages over.
    pub compress: u32,
}

impl RingLayout {
    /// The section a blob of this geometry carries.
    #[must_use]
    pub fn of(shape: &SparseShape, spec: SpecDepth, planes: PagePlanes) -> Self {
        Self {
            offset: snapshot_ring_offset(shape, spec),
            ranks: ring_ranks(shape, planes),
            draft: planes.draft && planes.indexer,
            row_bytes: u64::from(shape.indexer.head_dim).saturating_mul(F16_BYTES),
            compress: shape.indexer.compress.max(1),
        }
    }

    /// The frontier rank `rank` stands on in a checkpoint at `q`.
    #[must_use]
    pub const fn frontier(&self, rank: u32, q: u32) -> u32 {
        if self.draft && rank + 1 == self.ranks {
            q.saturating_sub(1)
        } else {
            q
        }
    }

    /// Whether a resume at `q` reads ring slot `slot` of rank `rank`.
    #[must_use]
    pub const fn live(&self, rank: u32, slot: u32, q: u32) -> bool {
        let frontier = self.frontier(rank, q);
        let first = frontier - frontier % self.compress;
        let mut at = first;
        while at < frontier {
            if at % IDX_RING == slot {
                return true;
            }
            at += 1;
        }
        false
    }

    /// Zeroes every row of `blob`'s ring section a resume at `q` does not read.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] when the blob is too short to hold the
    /// section, which would mean it was cut for another geometry.
    pub fn canonicalize(&self, blob: &mut [u8], q: u32) -> CoreResult<()> {
        let row = usize::try_from(self.row_bytes).unwrap_or(usize::MAX);
        let start = usize::try_from(self.offset).unwrap_or(usize::MAX);
        let len = row
            .saturating_mul(IDX_RING as usize)
            .saturating_mul(self.ranks as usize);
        let end = start.saturating_add(len);
        let got = u64::try_from(blob.len()).unwrap_or(u64::MAX);
        let section = blob.get_mut(start..end).ok_or(CoreError::ShapeMismatch {
            field: "checkpoint ring section",
            expected: u64::try_from(end).unwrap_or(u64::MAX),
            got,
        })?;
        for (index, bytes) in section.chunks_exact_mut(row.max(1)).enumerate() {
            let rank = u32::try_from(index / IDX_RING as usize).unwrap_or(u32::MAX);
            let slot = u32::try_from(index % IDX_RING as usize).unwrap_or(u32::MAX);
            if !self.live(rank, slot, q) {
                bytes.fill(0);
            }
        }
        Ok(())
    }
}

/// The cache geometry this shape implies.
#[must_use]
pub fn cache_geometry(shape: &SparseShape, planes: PagePlanes, spec: SpecDepth) -> CacheGeometry {
    // A prompt pass is summed the same way wherever it is cut, provided every
    // cut is on the recurrence's run grid; a prompt-final checkpoint off it
    // answers its own prompt and nothing longer.
    CacheGeometry::new(
        PAGE_TOKENS,
        page_bytes(shape, planes),
        ckpt_bytes(shape, spec, planes),
        ESTIMATED_PREFILL_S_PER_TOKEN,
        UTILITY_DECAY_S,
    )
    .with_resume_grid(crate::sparse::kernels::GDN_CHUNK)
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
                    (false, Plane::IndexerPooled) => MemTag::KvIndexer,
                };
                let stride = plane.page_bytes(shape, planes.kv);
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

#[cfg(test)]
mod tests {
    use super::*;

    const LAYOUT: RingLayout = RingLayout {
        offset: 8,
        ranks: 3,
        draft: true,
        row_bytes: 4,
        compress: 4,
    };

    #[test]
    fn a_checkpoint_on_a_block_boundary_keeps_no_target_row_and_three_draft_rows() {
        // q = 64: every target rank starts a fresh block, and the draft head,
        // one position behind, stands after 60, 61 and 62.
        for slot in 0..IDX_RING {
            assert!(!LAYOUT.live(0, slot, 64));
            assert!(!LAYOUT.live(1, slot, 64));
            assert_eq!(
                LAYOUT.live(2, slot, 64),
                matches!(slot, 12..=14),
                "slot {slot}"
            );
        }
    }

    #[test]
    fn the_live_rows_are_the_frontier_blocks_positions_at_their_ring_slots() {
        // q = 1_027 = 4 * 256 + 3: targets need 1024, 1025, 1026 - slots 0, 1,
        // 2 - and the head, standing on 1026, needs 1024 and 1025.
        let live = |rank| {
            (0..IDX_RING)
                .filter(|slot| LAYOUT.live(rank, *slot, 1_027))
                .collect::<Vec<_>>()
        };
        assert_eq!(live(0), [0, 1, 2]);
        assert_eq!(live(2), [0, 1]);
        // And a position wrapping the ring lands on its own slot.
        assert_eq!(
            (0..IDX_RING)
                .filter(|slot| LAYOUT.live(0, *slot, 31))
                .collect::<Vec<_>>(),
            [12, 13, 14]
        );
    }

    #[test]
    fn canonicalizing_zeroes_every_dead_row_and_nothing_outside_the_section() {
        let rows = (LAYOUT.ranks * IDX_RING) as usize;
        let mut blob = vec![0xAB_u8; 8 + rows * 4 + 5];
        LAYOUT
            .canonicalize(&mut blob, 1_027)
            .expect("the blob holds the section");
        assert!(
            blob[..8].iter().all(|b| *b == 0xAB),
            "the bytes in front moved"
        );
        assert!(
            blob[8 + rows * 4..].iter().all(|b| *b == 0xAB),
            "the bytes behind moved"
        );
        let kept = blob[8..8 + rows * 4]
            .chunks_exact(4)
            .filter(|row| row.iter().all(|b| *b == 0xAB))
            .count();
        // Three rows for each target rank, two for the head.
        assert_eq!(kept, 3 + 3 + 2);
        assert!(
            LAYOUT.canonicalize(&mut [0u8; 16], 1).is_err(),
            "a blob cut for another geometry is refused"
        );
    }

    #[test]
    fn the_pages_carry_no_raw_key_and_the_ring_is_per_session() {
        let shape = SparseShape {
            layers: vec![LayerKind::Gdn, LayerKind::Attn],
            hidden: 2_560,
            vocab: 248_320,
            streams: 4,
            stream_rank: 320,
            gqa: crate::sparse::shape::GqaShape::new(24, 2, 256, 64),
            gdn: crate::shape::GdnShape::new(48, 16, 128, 4),
            moe: crate::sparse::shape::MoeShape::new(512, 10, 1, 640),
            indexer: crate::sparse::shape::IndexerShape::new(4, 128, 2_048, 4),
            ple: None,
            mrope: crate::sparse::shape::MropeSections::new(11, 11, 10),
            rms_eps: 1.0e-6,
            rope_theta: 1.0e7,
        };
        let tokens = u64::from(PAGE_TOKENS);
        // One attention rank plus the head: key and value at two heads of 256,
        // then one pooled f16 row of 128 per four positions.
        for (kv, per_rank) in [
            (KvElem::F16, 2 * 2 * 256 * 2 + 64),
            (KvElem::Fp8E4m3, 2 * 2 * 256 + 64),
        ] {
            let planes = PagePlanes {
                kv,
                ..PagePlanes::full()
            };
            assert_eq!(page_bytes(&shape, planes), 2 * per_rank * tokens);
        }
        assert_eq!(ring_ranks(&shape, PagePlanes::full()), 2);
        assert_eq!(ring_ranks(&shape, PagePlanes::indexer_only()), 1);
        assert_eq!(ring_ranks(&shape, PagePlanes::default()), 0);
        assert_eq!(ring_bytes(&shape, PagePlanes::full()), 2 * 16 * 128 * 2);
        let spec = SpecDepth::new(3);
        assert_eq!(
            ckpt_bytes(&shape, spec, PagePlanes::full())
                - ckpt_bytes(&shape, spec, PagePlanes::default()),
            ring_bytes(&shape, PagePlanes::full())
        );
        assert_eq!(
            snapshot_ring_offset(&shape, spec) + ring_bytes(&shape, PagePlanes::full()),
            ckpt_bytes(&shape, spec, PagePlanes::full()),
            "the ring is the blob's last section"
        );
    }
}
