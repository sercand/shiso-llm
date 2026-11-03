//! The n-gram embedding block, and the seam the streaming tier fills.
//!
//! Exactly one slot folds this in, before that slot's first stream mix. It takes
//! a row per (position, head) out of a hash table too large to hold in memory,
//! decodes it, turns it into a per-stream key and a value, gates the value
//! against the residual, runs a dilated causal convolution over the gated value,
//! and adds both branches back into the residual.
//!
//! # The gather is a handle, and that is the whole seam
//!
//! The rows come off disk on the host - a dozen cold reads a token - and the
//! useful thing to do with that latency is hide it behind layers that are
//! already enqueued. So the gather is *started* before the layer loop and
//! *awaited* at the block, and [`PleGather`] is what travels between those two
//! points.
//!
//! [`PleGather::issue`] is where the real reads begin. It hashes the session's
//! trailing grams into row identifiers, hands them to a [`PleStream`], and
//! leaves the blocking fetch on a thread of its own, so the reads overlap the
//! host-side enqueue of every island ahead of the block rather than only the
//! device time of the layers below it. What comes back is a [`PleRows`], and
//! awaiting one is the only way to reach the bytes - a gather that was never
//! issued has nothing to await, and that is a type error rather than a runtime
//! one.
//!
//! # Why the block reads more tokens than the chunk has
//!
//! A position's grams reach back up to [`PleShape::lookback`] tokens, so the
//! gather needs the whole prefix rather than the chunk. That is why the handle
//! carries an absolute start rather than a slice.
//!
//! [`PleShape::lookback`]: crate::sparse::shape::PleShape::lookback

use std::sync::Arc;

use shiso_core::DType;
use shiso_core::buffer::DeviceBuffer;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::TokenId;
use shiso_ple::{PleParams, PleRowType, PleStream, codec};

pub use crate::ple_rows::PleRows;
use crate::ple_rows::{fetch_rows, issue_rows};
use crate::sparse::bind::{Args, LayerRef, NgramViews, Operand, Role, WeightRef};
use crate::sparse::hyper;
use crate::sparse::kernels::{self, Entry, MOD_GDN_TC, MOD_MISC, MOD_MODELOPT, MOD_PLE};
use crate::sparse::matmul::{self, Bound, Device};
use crate::sparse::moe::RowOrigin;
use crate::sparse::mtp::SpecDepth;
use crate::sparse::plan::Plan;
use crate::sparse::scratch::Buf;
use crate::sparse::shape::{PleShape, SparseShape};
use crate::sparse::weights::NgramDesc;

/// The hash metadata a table on this path must carry.
///
/// The n-gram tier addresses its rows by hashing the session's trailing grams,
/// so every number the hash reads has to be there. A table opened for direct
/// addressing - a per-layer embedding table, say - has none of them, and there
/// is no default that would be anything but a wrong row.
fn hashed(table: &PleStream) -> CoreResult<&PleParams> {
    table.params().ok_or(CoreError::Unsupported {
        what: "a direct-addressed table on the n-gram path",
        backend: "engine",
    })
}

/// Elements one block-quantized n-gram row packs into eighteen bytes.
const NGRAM_BLOCK_ELEMS: u32 = 32;
/// Bytes one such block occupies: a half-precision scale and sixteen packed
/// pairs.
const NGRAM_BLOCK_BYTES: u32 = 18;
/// Weights one `e4m3` group scale of an NVFP4 n-gram row covers.
const NGRAM_NVFP4_GROUP: u32 = 16;

/// Bytes one gathered row occupies.
///
/// The two decode kernels read two different layouts, and the width is what
/// tells each of them where the next row starts, so it is derived from the row
/// format rather than passed in: a stride one byte out reads every row after
/// the first from the middle of its predecessor.
///
/// `None` for a head width the format cannot describe, which is a geometry
/// error rather than a decode this build is missing.
#[must_use]
pub fn row_bytes(row: DType, head_dim: u32) -> Option<u32> {
    // A tail-biting trellis of six bits a weight, behind one half-precision row
    // scale; everything else is the block-quantized layout, sixteen packed
    // pairs and a scale to a block.
    if row == DType::Exl3Ngram6 {
        return Some(2 + head_dim.checked_mul(6)? / 8);
    }
    // One FP8 byte a weight; the scale is the table's, not the row's.
    if row == DType::Fp8E4M3 {
        return Some(head_dim);
    }
    // NVFP4: codes at four bits, then one `e4m3` scale byte per sixteen
    // weights, gathered from two arrays into one row by the reader; the
    // table's own scale is not in the row.
    if row == DType::Nvfp4Gs16 {
        if !head_dim.is_multiple_of(NGRAM_NVFP4_GROUP) {
            return None;
        }
        return Some(head_dim / 2 + head_dim / NGRAM_NVFP4_GROUP);
    }
    if !head_dim.is_multiple_of(NGRAM_BLOCK_ELEMS) {
        return None;
    }
    // The affine rows: codes, then one brain-float scale and one bias per
    // group, gathered from three arrays into one row by the reader.
    if row == DType::MlxAffine4Gs32 {
        return Some(head_dim / 2 + head_dim / NGRAM_BLOCK_ELEMS * 4);
    }
    Some(head_dim / NGRAM_BLOCK_ELEMS * NGRAM_BLOCK_BYTES)
}

/// The streamed n-gram table a stack gathers its rows from.
///
/// Three things that only mean anything together: the element type one row
/// arrives in, the reader the rows come off disk through, and the per-head bias
/// the trellis codec finishes each weight with. They are one value because the
/// invariants between them are what a caller gets wrong - a reader opened for
/// one codec and a decode kernel chosen for the other read the same bytes two
/// different ways and both produce finite numbers - so the constructor is the
/// one place they are checked and every later reader is total.
///
/// It is deliberately not `Clone`: the bias is a device allocation and the
/// reader owns a descriptor, and two stacks believing they own either is the
/// failure the ownership rule exists to prevent. Two stacks that really do want
/// one table share the [`PleStream`] by handing in the same [`Arc`].
#[derive(Debug)]
pub struct NgramTable {
    row: DType,
    stream: Arc<PleStream>,
    bias: Option<DeviceBuffer>,
}

impl NgramTable {
    /// Binds a reader, its row format and its bias together.
    ///
    /// `bias` is the table's per-head bias on the device, head-major. It is
    /// required by the trellis codec and refused for the other: the
    /// block-quantized decode kernel takes no such pointer, so a bias supplied
    /// with one would be a tensor nothing reads.
    ///
    /// The two codecs with one scale for the whole table - FP8
    /// ([`DType::Fp8E4M3`]) and NVFP4 ([`DType::Nvfp4Gs16`], whose gathered
    /// rows are codes then group scales) - take that scale in `bias` instead:
    /// one `f32` on the device, uploaded by the loader from the row format's
    /// own.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] when the row format and the reader's own
    /// codec disagree, and [`CoreError::ConfigMissing`] when the trellis codec
    /// is missing its bias or a table-scaled codec its scale.
    pub fn new(row: DType, stream: Arc<PleStream>, bias: Option<DeviceBuffer>) -> CoreResult<Self> {
        let stored = hashed(&stream)?.row_type();
        let table_scaled = match row {
            DType::Fp8E4M3 => Some(matches!(stored, PleRowType::Fp8E4M3 { .. })),
            DType::Nvfp4Gs16 => Some(matches!(stored, PleRowType::Nvfp4 { .. })),
            _ => None,
        };
        if let Some(agrees) = table_scaled {
            if !agrees {
                return Err(CoreError::ShapeMismatch {
                    field: "n-gram row codec",
                    expected: 1,
                    got: 0,
                });
            }
            // The table scale travels in the bias slot: one float the decode
            // kernel reads, uploaded by the loader from the row format's own.
            if bias.is_none() {
                return Err(CoreError::ConfigMissing {
                    key: format!("the {} n-gram table's scale on the device", row.type_name()),
                });
            }
            return Ok(Self { row, stream, bias });
        }
        let trellis = row == DType::Exl3Ngram6;
        let want = match row {
            DType::Exl3Ngram6 => PleRowType::TrellisK6,
            DType::MlxAffine4Gs32 => PleRowType::MlxAffine4,
            _ => PleRowType::BlockNl4,
        };
        if want != stored {
            return Err(CoreError::ShapeMismatch {
                field: "n-gram row codec",
                expected: u64::from(trellis),
                got: u64::from(stored == PleRowType::TrellisK6),
            });
        }
        if trellis && bias.is_none() {
            return Err(CoreError::ConfigMissing {
                key: "the trellis n-gram codec's per-head bias".to_owned(),
            });
        }
        if !trellis && bias.is_some() {
            return Err(CoreError::Unsupported {
                what: "a per-head bias for the block-quantized n-gram codec, whose decode kernel \
                       takes no such pointer",
                backend: "engine",
            });
        }
        Ok(Self { row, stream, bias })
    }

    /// The element type one gathered row arrives in.
    #[must_use]
    pub const fn row(&self) -> DType {
        self.row
    }

    /// Host bytes the reader's row cache holds, counted the way the cache
    /// sizes itself: each slot's row plus its bookkeeping.
    #[must_use]
    pub fn cache_bytes(&self) -> u64 {
        /// A slot's identifier, CLOCK bit, pending bit and two index entries.
        const SLOT_OVERHEAD: u64 = 8 + 1 + 2 * 4;
        self.stream
            .cache_rows()
            .saturating_mul(u64::from(self.stream.row_bytes()).saturating_add(SLOT_OVERHEAD))
    }

    /// The reader the rows come off disk through.
    #[must_use]
    pub const fn stream(&self) -> &Arc<PleStream> {
        &self.stream
    }

    /// What a pass binds of this table, given where its rows were staged.
    #[must_use]
    pub fn views<'a>(&'a self, rows: shiso_core::buffer::DeviceView<'a>) -> NgramViews<'a> {
        NgramViews {
            rows,
            bias: self.bias.as_ref().map(DeviceBuffer::view),
        }
    }
}

/// A table and the staging its gathered rows land in.
///
/// The staging is allocated once for the widest chunk rather than per pass,
/// because a buffer whose address moves cannot be read by a captured launch.
/// The two are one value because neither is useful without the other: rows with
/// nowhere to land are a read nobody can bind, and a landing ground with no
/// table behind it is the buffer this tier exists to stop being read out of.
#[derive(Debug)]
pub struct NgramTier {
    table: NgramTable,
    rows: DeviceBuffer,
    stride: u32,
}

impl NgramTier {
    /// Allocates the staging for `table` at this geometry.
    ///
    /// Every cross-check between the reader and the geometry happens here, once:
    /// the head count a position gathers, and the byte width of one row. Both
    /// are silent when wrong - a stride one byte out reads every row after the
    /// first from the middle of its predecessor, and a head count out by a
    /// factor reads another head's row as this one's - and both are constant
    /// for the life of the stack, so a per-pass check would be the same
    /// comparison thousands of times.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a head count or a row width the reader
    /// and the geometry disagree about, and [`CoreError::Backend`] when the
    /// staging cannot be allocated.
    pub fn open(
        backend: &Arc<dyn shiso_core::backend::Backend>,
        ple: PleShape,
        table: NgramTable,
        rows: u32,
    ) -> CoreResult<Self> {
        let stride = fits(ple, &table)?;
        let rows = shiso_core::buffer::alloc_tagged(
            backend,
            shiso_core::ledger::MemTag::NgramStaging,
            u64::from(rows)
                .saturating_mul(u64::from(ple.heads()))
                .saturating_mul(u64::from(stride)),
        )?;
        Ok(Self {
            table,
            rows,
            stride,
        })
    }

    /// The table the rows are read from.
    #[must_use]
    pub const fn table(&self) -> &NgramTable {
        &self.table
    }

    /// What a pass binds of this tier.
    #[must_use]
    pub fn views(&self) -> NgramViews<'_> {
        self.table.views(self.rows.view())
    }

    /// Copies one gather's awaited bytes into the staging.
    ///
    /// The destination is the front of the staging and its length is the
    /// gather's own, not the whole buffer: the staging is sized for the widest
    /// chunk and a decode launch reads exactly `n_tok * heads` rows from its
    /// front, so a copy that filled the tail would be writing rows no launch of
    /// this pass looks at.
    ///
    /// # Errors
    ///
    /// [`CoreError::OutOfBounds`] when the gather is wider than the staging -
    /// which means a chunk wider than the one the stack was built for - and
    /// whatever the device reports for the copy.
    pub fn upload(
        &self,
        backend: &dyn shiso_core::backend::Backend,
        bytes: &[u8],
    ) -> CoreResult<()> {
        let len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        backend.copy_h2d(self.rows.slice(0, len)?, bytes)
    }

    /// Bytes one gathered row occupies.
    #[must_use]
    pub const fn stride(&self) -> u32 {
        self.stride
    }
}

/// The row stride a table and a geometry agree on, or the first thing they
/// disagree about.
///
/// Separate from the allocation it precedes so that the agreement can be
/// checked without a device, which is the only way it gets checked at all on a
/// machine that has none. Both disagreements are silent when they happen: a
/// head count out by a factor reads another head's row as this one's, and a
/// stride one byte out reads every row after the first from the middle of its
/// predecessor.
fn fits(ple: PleShape, table: &NgramTable) -> CoreResult<u32> {
    let params = hashed(table.stream())?;
    if params.n_heads() != ple.heads() {
        return Err(CoreError::ShapeMismatch {
            field: "n-gram hash heads",
            expected: u64::from(ple.heads()),
            got: u64::from(params.n_heads()),
        });
    }
    let stride = row_bytes(table.row(), ple.head_dim).ok_or(CoreError::ShapeMismatch {
        field: "n-gram row width",
        expected: u64::from(NGRAM_BLOCK_ELEMS),
        got: u64::from(ple.head_dim),
    })?;
    if stride != params.row_bytes() {
        return Err(CoreError::ShapeMismatch {
            field: "n-gram row bytes",
            expected: u64::from(stride),
            got: u64::from(params.row_bytes()),
        });
    }
    Ok(stride)
}

/// What a pass tells the n-gram block that its geometry does not.
///
/// One value rather than two arguments because they are the two facts about the
/// *pass* the block reads, and both of them decide the same thing twice over:
/// the draft depth says whether the dilated convolution records a window per
/// position, and the row origin says whether it walks one window forward or one
/// per row. A batched pass is never speculative - the batched kernels take no
/// checkpoint slab at all - so the pair also makes that combination visible
/// where it is refused rather than silently unrecorded.
///
/// Deliberately NOT `#[non_exhaustive]`: the planner builds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NgramContext {
    /// Drafts one step may carry.
    pub spec: SpecDepth,
    /// Whether the rows are one sequence's consecutive positions or one row per
    /// sequence.
    pub origin: RowOrigin,
}

/// A gather of the n-gram rows one chunk needs, in flight.
///
/// The handle exists so a caller can start the reads early and await them late.
/// It is deliberately opaque and deliberately not `Copy`: the rows it names are
/// a resource, and two callers holding the same handle would both believe they
/// own the staging buffer behind it.
#[derive(Debug)]
pub struct PleGather {
    first: u32,
    rows: u32,
    heads: u32,
    lookback: u32,
}

impl PleGather {
    /// Starts a gather for `rows` positions beginning at absolute position
    /// `first`.
    ///
    /// `available` is how many positions of the session's path the caller can
    /// supply tokens for - the prefix it has already written plus the chunk
    /// about to run - not the chunk alone, because a position's grams reach back
    /// behind the chunk's first token. The tokens themselves are not taken here:
    /// the reads they drive are the streaming tier's, and the handle exists so
    /// that tier can start them early and be awaited late.
    ///
    /// # Errors
    ///
    /// [`CoreError::OutOfBounds`] when the chunk reaches past the path,
    /// which would mean hashing grams from positions the session has not
    /// written.
    pub fn start(ple: PleShape, available: u32, first: u32, rows: u32) -> CoreResult<Self> {
        let end = u64::from(first).saturating_add(u64::from(rows));
        let have = u64::from(available);
        if end > have {
            return Err(CoreError::OutOfBounds {
                offset: u64::from(first),
                len: u64::from(rows),
                buffer_bytes: have,
            });
        }
        Ok(Self {
            first,
            rows,
            heads: ple.heads(),
            lookback: ple.lookback(),
        })
    }

    /// Absolute position of the first row this gather covers.
    #[must_use]
    pub const fn first(&self) -> u32 {
        self.first
    }

    /// Positions the gather covers.
    #[must_use]
    pub const fn rows(&self) -> u32 {
        self.rows
    }

    /// Rows the gather reads: one per (position, head).
    #[must_use]
    pub const fn row_count(&self) -> u64 {
        self.rows as u64 * self.heads as u64
    }

    /// The earliest position the gather has to read a token from.
    ///
    /// Not the chunk's first position: the grams of the chunk's first token
    /// reach back behind it, and a gather that started at the chunk would hash
    /// the wrong grams for its own first rows.
    #[must_use]
    pub const fn earliest(&self) -> u32 {
        self.first.saturating_sub(self.lookback)
    }

    /// The row identifiers this gather names, hashed from the session's path.
    ///
    /// `tokens` is the path by absolute position and has to reach the end of
    /// the gather, not merely the start of the chunk: the first row's grams
    /// read the tokens behind it.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] when the table's head count is not the one
    /// this stack's geometry says, which would mean the rows and the decode
    /// kernel disagree about how many rows a position has; and whatever the
    /// hash reports about a path too short for the gather.
    pub fn row_ids(&self, table: &PleStream, tokens: &[TokenId]) -> CoreResult<Vec<u64>> {
        let params = hashed(table)?;
        if params.n_heads() != self.heads {
            return Err(CoreError::ShapeMismatch {
                field: "n-gram hash heads",
                expected: u64::from(self.heads),
                got: u64::from(params.n_heads()),
            });
        }
        // A gather this host can address is a gather that fits a length. The
        // conversion can only fail on a pointer narrower than the identifier
        // count, and refusing there beats asking for an absurd allocation.
        let want = usize::try_from(self.row_count()).map_err(|_| CoreError::OutOfBounds {
            offset: u64::from(self.first),
            len: self.row_count(),
            buffer_bytes: 0,
        })?;
        let mut ids = vec![0u64; want];
        codec::row_ids(params, tokens, self.first, self.rows, &mut ids)?;
        Ok(ids)
    }

    /// Starts the reads on a thread of its own.
    ///
    /// The identifiers are hashed here, on the caller's thread, because that is
    /// pure arithmetic over tokens the caller already holds; only the blocking
    /// reads move. The gather is consumed: what comes back is the same gather
    /// with its rows in flight, and there is no way to ask for them twice.
    ///
    /// # Errors
    ///
    /// Whatever [`PleGather::row_ids`] reports, and [`CoreError::Io`] when the
    /// fetch thread cannot be spawned.
    pub fn issue(self, table: &Arc<PleStream>, tokens: &[TokenId]) -> CoreResult<PleRows> {
        let ids = self.row_ids(table, tokens)?;
        issue_rows(table, ids)
    }

    /// The same rows read on the caller's thread; see [`fetch_rows`].
    ///
    /// # Errors
    ///
    /// Whatever [`PleGather::row_ids`] reports.
    pub fn fetch(self, table: &PleStream, tokens: &[TokenId]) -> CoreResult<PleRows> {
        let ids = self.row_ids(table, tokens)?;
        Ok(fetch_rows(table, &ids))
    }
}

/// Warms the table for a chunk that has not been enqueued yet.
///
/// The rows of the chunk after this one are already known, and this one is
/// about to spend its time on the device, so reading them now is free. The
/// fetch that eventually asks for them hits, and nothing here waits.
///
/// # Errors
///
/// Whatever [`PleGather::row_ids`] reports, and whatever the reader reports
/// about starting its prefetch thread.
pub fn prefetch(table: &PleStream, gather: &PleGather, tokens: &[TokenId]) -> CoreResult<()> {
    let ids = gather.row_ids(table, tokens)?;
    table.prefetch(&ids)?;
    Ok(())
}

/// The decode kernel one gathered row format is read by.
///
/// The trellis decode is reachable at one head width only - the width is also
/// its block width - which is why this is a lookup that can fail rather than a
/// branch on the format alone.
#[must_use]
pub const fn decode_entry(row: DType, head_dim: u32) -> Option<Entry> {
    match row {
        DType::Exl3Ngram6 => {
            if head_dim == 160 {
                Some(Entry::plain(MOD_PLE, "q4e_ple_dequant_exl3_kernel"))
            } else {
                None
            }
        }
        DType::MlxAffine4Gs32 => Some(Entry::plain(MOD_PLE, "q4e_ple_dequant_affine_kernel")),
        DType::Fp8E4M3 => Some(Entry::plain(MOD_MODELOPT, "q4e_ple_dequant_fp8_kernel")),
        DType::Nvfp4Gs16 => Some(Entry::plain(MOD_MODELOPT, "q4e_ple_dequant_nvfp4_kernel")),
        _ => Some(Entry::plain(MOD_PLE, "q4e_ple_dequant_kernel")),
    }
}

/// One n-gram block, in launch order.
///
/// Three normalizations, and they are three different weights over three
/// different things: the keys, the residual the keys are gated against, and the
/// gated value the convolution reads. Collapsing any two of them would be a
/// different model.
///
/// Two of those normalizations write the buffer they read. That is not an
/// optimization: the key norm and the convolution's input norm are each the
/// only reader of what they overwrite, and giving either a destination of its
/// own would be a buffer nothing else ever looks at.
///
/// # Errors
///
/// [`CoreError::Unsupported`] when the gathered rows are in a format whose
/// decode kernel this head width cannot reach, and whatever
/// [`matmul::project`] reports.
#[expect(
    clippy::too_many_lines,
    reason = "one statement per kernel argument is the point: these bodies are \
              transcriptions of parameter lists, and a split that broke a list \
              across two functions would hide the order being transcribed"
)]
pub fn block(
    shape: &SparseShape,
    ple: PleShape,
    desc: &NgramDesc,
    layer: LayerRef,
    ctx: NgramContext,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let spec = ctx.spec;
    // The dilated window is per-position state exactly as the recurrent slots'
    // is, and a rollback that restored one without the other would leave the
    // n-gram block reading a window for a position the rest of the stack has
    // already left.
    let window_ckpt = if spec.is_speculative() {
        (
            Operand::NgramWindowCkpt,
            Operand::U32(spec.rollback_slots()),
        )
    } else {
        (Operand::Null, Operand::U32(0))
    };
    let decode = decode_entry(desc.row, ple.head_dim).ok_or(CoreError::Unsupported {
        what: "an n-gram row format with no decode kernel at this head width",
        backend: "engine",
    })?;
    let stride = row_bytes(desc.row, ple.head_dim).ok_or(CoreError::ShapeMismatch {
        field: "n-gram row width",
        expected: u64::from(NGRAM_BLOCK_ELEMS),
        got: u64::from(ple.head_dim),
    })?;
    let at = |role: Role| WeightRef::layer(layer, role);
    let heads = ple.heads();
    let rows = n_tok.saturating_mul(heads);
    let mut plan = Plan::new();
    // The trellis row decode takes one more pointer than the block-quantized
    // one: the hash head's bias, a tensor of the streamed table rather than of
    // the layer, which is why it is an operand of its own rather than a
    // seventh n-gram role. Its block is the row width, because the code's ring
    // wraps at that width and one thread decodes one weight.
    let trellis = decode.name() == "q4e_ple_dequant_exl3_kernel";
    // FP8 and NVFP4 rows share an argument list: the table's one scale is
    // the last pointer.
    let table_scaled = matches!(desc.row, DType::Fp8E4M3 | DType::Nvfp4Gs16);
    let args = if table_scaled {
        // `out, rows, head_dim, n_heads, row_bytes, n_rows, scale`
        Args::of(&[
            Operand::Scratch(Buf::PleEmbed),
            Operand::NgramRows,
            Operand::U32(ple.head_dim),
            Operand::U32(heads),
            Operand::U32(stride),
            Operand::U32(rows),
            Operand::NgramBias,
        ])
    } else if trellis {
        // `out, rows, bias, head_dim, n_heads, row_bytes, n_rows`
        Args::of(&[
            Operand::Scratch(Buf::PleEmbed),
            Operand::NgramRows,
            Operand::NgramBias,
            Operand::U32(ple.head_dim),
            Operand::U32(heads),
            Operand::U32(stride),
            Operand::U32(rows),
        ])
    } else {
        // `out, rows, head_dim, n_heads, row_bytes, n_rows`
        Args::of(&[
            Operand::Scratch(Buf::PleEmbed),
            Operand::NgramRows,
            Operand::U32(ple.head_dim),
            Operand::U32(heads),
            Operand::U32(stride),
            Operand::U32(rows),
        ])
    };
    // The affine decode takes the block-quantized decode's argument list and
    // the trellis decode's grid: one thread a weight, one block a row.
    let launch = if trellis || table_scaled || desc.row == DType::MlxAffine4Gs32 {
        kernels::ple_dequant_trellis(heads, n_tok, ple.head_dim)
    } else {
        kernels::ple_dequant(heads, n_tok)
    };
    plan.launch(decode, launch, args);
    matmul::project(
        &mut plan,
        desc.key,
        Bound::new(
            at(Role::NgramKey),
            Operand::Scratch(Buf::PleEmbed),
            Operand::Scratch(Buf::PleKey),
        ),
        n_tok,
        device,
    )?;
    matmul::project(
        &mut plan,
        desc.value,
        Bound::new(
            at(Role::NgramValue),
            Operand::Scratch(Buf::PleEmbed),
            Operand::Scratch(Buf::PleValue),
        ),
        n_tok,
        device,
    )?;
    plan.append(hyper::norm(
        shape,
        at(Role::NgramNormKey),
        Operand::Scratch(Buf::PleKey),
        Operand::Scratch(Buf::PleKey),
        n_tok,
    ));
    plan.append(hyper::norm(
        shape,
        at(Role::NgramNormQuery),
        Operand::Scratch(Buf::Residual),
        Operand::Scratch(Buf::PleQuery),
        n_tok,
    ));
    plan.launch(
        Entry::plain(MOD_PLE, "q4e_ple_gated_value_kernel"),
        kernels::ple_gated_value(shape.streams, n_tok),
        // `gv, key, query, value, n_embd, n_hc`
        Args::of(&[
            Operand::Scratch(Buf::PleGated),
            Operand::Scratch(Buf::PleKey),
            Operand::Scratch(Buf::PleQuery),
            Operand::Scratch(Buf::PleValue),
            Operand::U32(shape.hidden),
            Operand::U32(shape.streams),
        ]),
    );
    plan.append(hyper::norm(
        shape,
        at(Role::NgramNormConv),
        Operand::Scratch(Buf::PleGated),
        Operand::Scratch(Buf::PleConv),
        n_tok,
    ));
    if let RowOrigin::Segmented(runs) = ctx.origin {
        // One scalar walk per run, over that run's rows and from its own
        // session's window, recording that session's rollback windows.
        let width = u64::from(shape.stream_dim()) * 4;
        for run in 0..runs.count() {
            let first = u64::from(runs.start(run)) * width;
            let ckpt = if spec.is_speculative() {
                (
                    Operand::NgramWindowCkptOf(run),
                    Operand::U32(spec.rollback_slots()),
                )
            } else {
                (Operand::Null, Operand::U32(0))
            };
            plan.launch(
                Entry::plain(MOD_PLE, "q4e_ple_conv_kernel"),
                kernels::ple_conv(shape.stream_dim()),
                // `out, state, x, w, channels, kernel, dilation, n_tok, ckpt,
                //  ckpt_cap`
                Args::of(&[
                    Operand::ScratchFrom(Buf::PleConv, first),
                    Operand::NgramWindowOf(run),
                    Operand::ScratchFrom(Buf::PleConv, first),
                    Operand::Weight(at(Role::NgramConv1d)),
                    Operand::U32(shape.stream_dim()),
                    Operand::U32(ple.conv),
                    Operand::U32(ple.ngram),
                    Operand::U32(runs.len(run)),
                    ckpt.0,
                    ckpt.1,
                ]),
            );
        }
    } else if ctx.origin == RowOrigin::Unrelated {
        plan.launch(
            Entry::plain(MOD_PLE, "q4e_ple_conv_batch_kernel"),
            kernels::ple_conv_batch(shape.stream_dim(), n_tok),
            // `out, states, x, w, channels, kernel, dilation` - one dilated
            // window per row, dereferenced by `blockIdx.y`. No row count and no
            // checkpoint slab: a row is one token, so there is nothing to walk
            // and nothing to record.
            Args::of(&[
                Operand::Scratch(Buf::PleConv),
                Operand::NgramWindows,
                Operand::Scratch(Buf::PleConv),
                Operand::Weight(at(Role::NgramConv1d)),
                Operand::U32(shape.stream_dim()),
                Operand::U32(ple.conv),
                Operand::U32(ple.ngram),
            ]),
        );
    } else {
        // The window as it will stand before the snapshot row, taken first: the
        // convolution runs in place and its inputs are gone once it has. The
        // window is the last `history` inputs, which is the recurrent blocks'
        // tail copy with a kernel one wider than the history.
        if let Some(row) = device.role.snapshot() {
            plan.launch(
                Entry::plain(MOD_GDN_TC, "q4e_gdn_conv_tail_kernel"),
                kernels::gdn_conv(shape.stream_dim()),
                // `state, x, channels, kernel, n_tok`
                Args::of(&[
                    Operand::Snapshot(crate::sparse::paged::snapshot_window_offset(shape)),
                    Operand::Scratch(Buf::PleConv),
                    Operand::U32(shape.stream_dim()),
                    Operand::U32(ple.history() + 1),
                    Operand::U32(row),
                ]),
            );
        }
        plan.launch(
            Entry::plain(MOD_PLE, "q4e_ple_conv_kernel"),
            kernels::ple_conv(shape.stream_dim()),
            // `out, state, x, w, channels, kernel, dilation, n_tok, ckpt,
            //  ckpt_cap`
            Args::of(&[
                Operand::Scratch(Buf::PleConv),
                Operand::NgramWindow,
                Operand::Scratch(Buf::PleConv),
                Operand::Weight(at(Role::NgramConv1d)),
                Operand::U32(shape.stream_dim()),
                Operand::U32(ple.conv),
                Operand::U32(ple.ngram),
                Operand::U32(n_tok),
                window_ckpt.0,
                window_ckpt.1,
            ]),
        );
    }
    plan.launch(
        Entry::plain(MOD_MISC, "q4e_add2_kernel"),
        kernels::elementwise(u64::from(n_tok) * u64::from(shape.stream_dim())),
        // `res, a, b, n` - both branches go back into the residual, the gated
        // value ungated by the convolution and the convolution's own output.
        Args::of(&[
            Operand::Scratch(Buf::Residual),
            Operand::Scratch(Buf::PleGated),
            Operand::Scratch(Buf::PleConv),
            Operand::U64(u64::from(n_tok) * u64::from(shape.stream_dim())),
        ]),
    );
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shiso_core::ids::LayerIdx;

    const PLE: PleShape = PleShape {
        layer: LayerIdx::new(1),
        ngram: 3,
        heads_per_ngram: 8,
        head_dim: 160,
        conv: 4,
    };

    #[test]
    fn a_gather_reaches_behind_its_own_first_position() {
        let gather = PleGather::start(PLE, 64, 32, 8).expect("the path covers it");
        assert_eq!(gather.first(), 32);
        assert_eq!(gather.earliest(), 30, "two tokens of gram context");
        assert_eq!(gather.row_count(), 8 * 16, "sixteen heads a position");
    }

    #[test]
    fn a_gather_at_the_start_of_a_session_cannot_reach_below_zero() {
        let gather = PleGather::start(PLE, 4, 0, 4).expect("the path covers it");
        assert_eq!(gather.earliest(), 0);
    }

    #[test]
    fn a_gather_past_the_path_is_refused_rather_than_hashing_absent_tokens() {
        let err = PleGather::start(PLE, 4, 2, 8).expect_err("the path is four long");
        assert!(matches!(err, CoreError::OutOfBounds { .. }), "{err:?}");
    }

    #[test]
    fn the_trellis_row_decode_is_reachable_at_one_head_width_only() {
        assert!(decode_entry(DType::Exl3Ngram6, 160).is_some());
        assert!(decode_entry(DType::Exl3Ngram6, 128).is_none());
        // Every other format decodes at any width.
        assert!(decode_entry(DType::F16, 128).is_some());
    }

    #[test]
    fn an_nvfp4_row_is_its_codes_then_one_scale_byte_per_sixteen() {
        assert_eq!(row_bytes(DType::Nvfp4Gs16, 160), Some(80 + 10));
        assert_eq!(row_bytes(DType::Nvfp4Gs16, 40), None, "not whole groups");
        let entry = decode_entry(DType::Nvfp4Gs16, 160).expect("any whole-group width");
        assert_eq!(entry.name(), "q4e_ple_dequant_nvfp4_kernel");
    }

    #[test]
    fn an_nvfp4_table_needs_an_nvfp4_reader_and_its_scale() {
        // The block-quantized rows are ninety bytes at this width too, so
        // the stride cannot tell the two apart; only the codec check does.
        let (_fixture, stream) = table_of(PLE.heads_per_ngram, PleRowType::BlockNl4);
        let err = NgramTable::new(DType::Nvfp4Gs16, stream, None)
            .expect_err("the reader is block-quantized");
        assert!(
            matches!(
                err,
                CoreError::ShapeMismatch {
                    field: "n-gram row codec",
                    ..
                }
            ),
            "{err:?}"
        );
        let (_fixture, stream) = table_of(
            PLE.heads_per_ngram,
            PleRowType::Nvfp4 {
                global_bits: 1.0f32.to_bits(),
            },
        );
        assert_eq!(stream.row_bytes(), 90);
        let err = NgramTable::new(DType::Nvfp4Gs16, stream, None)
            .expect_err("the table scale is not on the device");
        assert!(matches!(err, CoreError::ConfigMissing { .. }), "{err:?}");
    }

    /// Rows the fixture table holds.
    const TABLE_ROWS: u64 = 4096;

    /// The byte a fixture row holds at one position.
    fn row_byte(row: u64, j: usize) -> u8 {
        u8::try_from((row.wrapping_mul(31) + j as u64 * 7 + 11) & 0xFF).expect("masked to a byte")
    }

    /// Table metadata matching [`PLE`]: sixteen heads of a hundred and sixty
    /// trellis-coded weights.
    fn table_params_of(heads_per_ngram: u32, row_type: PleRowType) -> PleParams {
        let heads = (PLE.ngram - 1) * heads_per_ngram;
        let mut offsets = Vec::new();
        let mut vocab = Vec::new();
        let mut at = 0u64;
        for h in 0..u64::from(heads) {
            offsets.push(at);
            let n = 128 + h;
            vocab.push(n);
            at += n;
        }
        PleParams::new(shiso_ple::PleParamsDesc {
            row_type,
            ngram_size: PLE.ngram,
            heads_per_ngram,
            head_dim: PLE.head_dim,
            eos_token: TokenId::new(248_044),
            multipliers: vec![
                0x9E37_79B9_7F4A_7C15,
                0xBF58_476D_1CE4_E5B9,
                0x94D0_49BB_1331_11EB,
            ],
            head_offsets: offsets,
            head_vocab: vocab,
            head_bias: Some(Arc::new(vec![0u16; heads as usize * PLE.head_dim as usize])),
        })
        .expect("the fixture parameters are valid")
    }

    /// A table on disk, and a stream over it.
    fn table(heads_per_ngram: u32) -> (tempfile::NamedTempFile, Arc<PleStream>) {
        table_of(heads_per_ngram, PleRowType::TrellisK6)
    }

    /// The same, for a table stored in either codec.
    fn table_of(
        heads_per_ngram: u32,
        row_type: PleRowType,
    ) -> (tempfile::NamedTempFile, Arc<PleStream>) {
        use std::io::Write as _;

        let params = table_params_of(heads_per_ngram, row_type);
        let stride = params.row_bytes() as usize;
        let mut file = tempfile::NamedTempFile::new().expect("a temporary file");
        let mut row = vec![0u8; stride];
        for id in 0..TABLE_ROWS {
            for (j, byte) in row.iter_mut().enumerate() {
                *byte = row_byte(id, j);
            }
            file.write_all(&row).expect("a row");
        }
        file.flush().expect("flush");
        file.as_file().sync_all().expect("sync");

        let handle = std::fs::File::open(file.path()).expect("reopen");
        let stream = PleStream::open(
            handle,
            0,
            TABLE_ROWS,
            params,
            1 << 20,
            &shiso_ple::PleConfig {
                io: shiso_ple::PleIo::Pool,
                ..shiso_ple::PleConfig::default()
            },
        )
        .expect("the fixture opens");
        (file, Arc::new(stream))
    }

    fn history(n: usize) -> Vec<TokenId> {
        (0..n)
            .map(|i| TokenId::new((u32::try_from(i).unwrap_or(0) * 7919 + 13) % 100_000))
            .collect()
    }

    #[test]
    fn a_table_whose_row_format_and_reader_disagree_about_the_codec_is_refused() {
        // Both codecs ship, and the checkpoint says which in the table tensor's
        // own type. A reader opened for one and a decode kernel chosen for the
        // other read the same bytes two different ways, and both produce finite
        // numbers - so the pairing is checked once, here, rather than never.
        let (_fixture, stream) = table(PLE.heads_per_ngram);
        let err = NgramTable::new(DType::F16, Arc::clone(&stream), None)
            .expect_err("the reader is trellis-coded and the row format is not");
        assert!(
            matches!(
                err,
                CoreError::ShapeMismatch {
                    field: "n-gram row codec",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_table_whose_head_count_is_not_this_geometrys_never_gets_a_staging() {
        // The staging is sized as rows times heads times the stride, and the
        // decode launch indexes it as (position, head). A table with half the
        // heads fills half of it and every head after the first reads another
        // head's row - which decodes, because the rows are all the same width.
        let (_fixture, stream) = table_of(4, PleRowType::BlockNl4);
        let bound =
            NgramTable::new(DType::F16, stream, None).expect("the codec and the reader agree");
        let err = fits(PLE, &bound).expect_err("eight heads against this geometry's sixteen");
        assert!(
            matches!(
                err,
                CoreError::ShapeMismatch {
                    field: "n-gram hash heads",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_table_whose_rows_are_not_the_width_this_geometry_reads_is_refused() {
        // The stride is what turns a row index into an offset, so a geometry
        // that reads a narrower row than the file stores would read every row
        // after the first from the middle of its predecessor - and decode all
        // of them to finite numbers.
        let (_fixture, stream) = table_of(PLE.heads_per_ngram, PleRowType::BlockNl4);
        assert_eq!(
            stream.row_bytes(),
            90,
            "a hundred and sixty four-bit weights in five eighteen-byte blocks"
        );
        let bound =
            NgramTable::new(DType::F16, stream, None).expect("the codec and the reader agree");
        // Only the width moves: the head count is what the gram table decides,
        // and it is checked separately above.
        let narrow = PleShape {
            head_dim: 128,
            ..PLE
        };
        assert_eq!(narrow.heads(), PLE.heads(), "only the width moved");
        let err = fits(narrow, &bound)
            .expect_err("a 128-wide head reads 72-byte rows out of a 90-byte table");
        assert!(
            matches!(
                err,
                CoreError::ShapeMismatch {
                    field: "n-gram row bytes",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_trellis_table_without_its_per_head_bias_is_refused() {
        // The trellis decode finishes every weight with that bias, and the
        // kernel takes it as a pointer. A table opened without one would hand
        // the launch an operand nothing bound.
        let (_fixture, stream) = table(PLE.heads_per_ngram);
        let err = NgramTable::new(DType::Exl3Ngram6, stream, None)
            .expect_err("the trellis codec has no bias here");
        assert!(matches!(err, CoreError::ConfigMissing { .. }), "{err:?}");
    }

    #[test]
    fn an_issued_gather_brings_back_one_row_per_position_and_head() {
        let (_fixture, stream) = table(PLE.heads_per_ngram);
        let tokens = history(64);
        let gather = PleGather::start(PLE, 64, 32, 8).expect("the path covers it");
        let ids = gather.row_ids(&stream, &tokens).expect("the hash runs");
        assert_eq!(ids.len(), 8 * 16, "eight positions of sixteen heads");

        let rows = gather.issue(&stream, &tokens).expect("the reads start");
        assert_eq!(rows.rows(), 8 * 16);
        assert_eq!(rows.bytes(), 8 * 16 * 122);
        let bytes = rows.await_rows().expect("the reads land");
        assert_eq!(bytes.len() as u64, 8 * 16 * 122);

        // The bytes are the rows the hash named, in order, which is what the
        // device buffer is about to be filled with.
        for (i, &id) in ids.iter().enumerate() {
            for j in 0..122 {
                assert_eq!(bytes[i * 122 + j], row_byte(id, j), "row {id} byte {j}");
            }
        }
    }

    #[test]
    fn a_table_whose_head_count_is_not_this_stacks_is_refused_before_any_read() {
        // Four heads per gram width against the stack's eight: the rows would
        // arrive at half the count the decode launch expects, and every head
        // after the first would read another head's row.
        let (_fixture, stream) = table(4);
        let tokens = history(16);
        let gather = PleGather::start(PLE, 16, 0, 4).expect("the path covers it");
        let err = gather
            .row_ids(&stream, &tokens)
            .expect_err("the head counts disagree");
        assert!(
            matches!(
                err,
                CoreError::ShapeMismatch {
                    field: "n-gram hash heads",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_gather_over_a_path_shorter_than_it_names_is_refused_rather_than_hashing_nothing() {
        let (_fixture, stream) = table(PLE.heads_per_ngram);
        let tokens = history(8);
        // `start` cannot catch this: it is told how long the path is, and here
        // the caller's claim and the slice disagree.
        let gather = PleGather::start(PLE, 64, 0, 16).expect("the caller claims sixty-four");
        let err = gather
            .row_ids(&stream, &tokens)
            .expect_err("only eight tokens exist");
        assert!(
            matches!(err, CoreError::Backend { backend: "ple", .. }),
            "{err:?}"
        );
    }

    #[test]
    fn a_prefetched_chunk_is_already_resident_when_its_gather_runs() {
        let (_fixture, stream) = table(PLE.heads_per_ngram);
        let tokens = history(64);
        let next = PleGather::start(PLE, 64, 32, 8).expect("the path covers it");
        prefetch(&stream, &next, &tokens).expect("the prefetch starts");
        // A second prefetch finishes the one in flight before queueing itself,
        // so this returning means the first batch has been read.
        prefetch(&stream, &next, &tokens).expect("the prefetch starts");

        let before = stream.stats();
        assert!(before.reads > 0, "the prefetch really read: {before:?}");
        let rows = next
            .issue(&stream, &tokens)
            .expect("the reads start")
            .await_rows()
            .expect("the reads land");
        assert_eq!(rows.len(), 8 * 16 * 122);
        let after = stream.stats();
        assert_eq!(after.reads, before.reads, "the gather needed no new read");
        assert!(after.hits >= 8 * 16, "{after:?}");
    }
}
