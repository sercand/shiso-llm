//! A model on a device that is a `HashMap`, built from a checkpoint this file
//! writes.
//!
//! Every test binary here needs the same thing before it can ask its own
//! question: a loaded model that runs on every machine, with no checkpoint on
//! disk to find and no device to talk to. The device under it records what it
//! was asked to do instead of doing it, and the buffers it hands out are byte
//! vectors a test can read directly.
//!
//! # Why the fixture is tiny and the pattern is not
//!
//! Every width is cut down - eight slots, a 32-wide stream, a 64-token
//! vocabulary - and every *structural* fact is kept: three recurrent slots then
//! an attention slot, four residual streams, an n-gram block on slot one, a
//! router over eight experts. The widths are what make the fixture a few
//! kilobytes; the structure is what the ordinals, the cache planes, the island
//! chain and the expert arithmetic are all derived from, so none of it can be
//! cut without the tests stopping being about the model.
//!
//! # What a fake device can prove
//!
//! Not a number a kernel would have produced: `run` writes nothing. What it can
//! prove is the bookkeeping - which buffers a session owns, when they come back,
//! what a chunk's plan is made of - and that is what every test here asks about.
#![allow(
    dead_code,
    reason = "the test binaries that include this fixture ask different questions of it; \
              an item one of them does not need is shared plumbing, not dead code"
)]

use std::collections::HashMap;
use std::io::{Seek as _, Write as _};
use std::num::NonZeroU64;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use shiso_core::ids::LayerIdx;
use shiso_core::{
    Backend, BlasBackend, CoopLaunch, CoreError, CoreResult, DeviceView, Extent3, GemmDesc,
    KernelArg, KernelRef, LaunchArgs, PointerPack, RawBuffer, TokenId,
};
use shiso_engine::EngineOptions;
use shiso_engine::shape::GdnShape;
use shiso_engine::sparse::kernels::{self, Elem, EntryTable, Instantiation, TemplateArgs};
use shiso_engine::sparse::mtp::SpecDepth;
use shiso_engine::sparse::shape::{
    GqaShape, IndexerShape, MoeShape, MropeSections, PleShape, SparseShape,
};
use shiso_engine::sparse::stack::SparseOptions;
use shiso_models::qwen4exp::{Qwen4ExpConfig, Qwen4ExpModel};

/// The byte a fresh allocation is filled with, so that reading one shows up.
pub(crate) const POISON: u8 = 0xA5;

// ---------------------------------------------------------------------------
// the fake device
// ---------------------------------------------------------------------------

/// One argument as the device received it.
///
/// A buffer keeps the three numbers that identify the range rather than the
/// bytes: what a launch is checked on here is which allocation it was pointed
/// at and where in it, not what was in it, because nothing here computes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Arg {
    /// A device range: allocation, byte offset, byte length.
    Buffer(u64, u64, u64),
    /// An unsigned 32-bit scalar.
    U32(u32),
    /// A signed 32-bit scalar.
    I32(i32),
    /// An unsigned 64-bit scalar, which is also how a null pointer arrives.
    U64(u64),
    /// A single-precision scalar.
    F32(f32),
    /// A by-value struct, by length.
    Bytes(usize),
}

/// One launch as the device received it.
#[derive(Debug, Clone)]
pub(crate) struct Recorded {
    /// The module the entry was fetched from.
    pub(crate) module: &'static str,
    /// The entry name.
    pub(crate) entry: &'static str,
    /// Blocks.
    pub(crate) grid: Extent3,
    /// Threads per block.
    pub(crate) block: Extent3,
    /// Dynamic shared memory.
    pub(crate) smem: u32,
    /// Arguments, in the order the launch handed them over.
    pub(crate) args: Vec<Arg>,
    /// Whether the whole grid was required to be resident.
    pub(crate) cooperative: bool,
}

/// A device that is a `HashMap` of byte vectors and a log of what it was asked.
#[derive(Debug, Default)]
pub(crate) struct FakeBackend {
    live: Mutex<HashMap<u64, Vec<u8>>>,
    allocated: Mutex<Vec<u64>>,
    released: Mutex<Vec<u64>>,
    launches: Mutex<Vec<Recorded>>,
    reservations: Mutex<Vec<u64>>,
    scratch: Mutex<Option<RawBuffer>>,
    next_id: AtomicU64,
    double_frees: AtomicU64,
    reads: AtomicU64,
    uploads: Mutex<Vec<Upload>>,
}

/// One host-to-device copy, and where it fell among the launches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Upload {
    /// Launches the device had received when the copy arrived: the copy is
    /// ordered after launch `launches - 1` and before launch `launches`.
    pub(crate) launches: usize,
    /// The allocation written.
    pub(crate) buffer: u64,
    /// Byte offset into it.
    pub(crate) offset: u64,
    /// Bytes written.
    pub(crate) len: u64,
}

impl FakeBackend {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// How many allocations have been made, which is the mark a caller takes
    /// before a call it wants to attribute buffers to.
    pub(crate) fn allocation_mark(&self) -> usize {
        self.allocated.lock().len()
    }

    /// The ids allocated since `mark`, in allocation order.
    pub(crate) fn allocated_since(&self, mark: usize) -> Vec<u64> {
        self.allocated.lock()[mark..].to_vec()
    }

    /// Whether `id` is still an allocation the device holds.
    pub(crate) fn is_live(&self, id: u64) -> bool {
        self.live.lock().contains_key(&id)
    }

    /// How many allocations the device is still holding.
    pub(crate) fn live_count(&self) -> usize {
        self.live.lock().len()
    }

    /// Every host-to-device copy the device received, in order, each with
    /// its place among the launches.
    pub(crate) fn uploads(&self) -> Vec<Upload> {
        self.uploads.lock().clone()
    }

    /// How many device-to-host reads the device has served.
    ///
    /// The one thing a snapshot is, from here: the recurrent stack copied back
    /// a slot at a time. A caller that wants to know whether a span was
    /// checkpointed asks this, because a snapshot changes no answer and is
    /// invisible to every assertion about output.
    pub(crate) fn read_count(&self) -> u64 {
        self.reads.load(Ordering::Relaxed)
    }

    /// How many launches reached the device.
    ///
    /// Zero, always, on this half of the port - which is itself worth
    /// asserting: a plan that quietly launched something would mean the refusal
    /// above it was not the only path.
    pub(crate) fn launch_count(&self) -> usize {
        self.launches.lock().len()
    }

    /// Every launch the device received, in order.
    pub(crate) fn launches(&self) -> Vec<Recorded> {
        self.launches.lock().clone()
    }

    /// The scratch sizes a caller reserved, in order.
    pub(crate) fn reservations(&self) -> Vec<u64> {
        self.reservations.lock().clone()
    }

    pub(crate) fn double_frees(&self) -> u64 {
        self.double_frees.load(Ordering::Relaxed)
    }

    /// Writes one launch into the log exactly as it arrived.
    fn record(&self, kernel: KernelRef, args: &LaunchArgs<'_>, cooperative: bool) {
        self.launches.lock().push(Recorded {
            module: kernel.module,
            entry: kernel.entry,
            grid: args.grid,
            block: args.block,
            smem: args.smem_bytes,
            args: args
                .args
                .iter()
                .map(|arg| match *arg {
                    KernelArg::Buffer(view) => {
                        Arg::Buffer(view.id().get(), view.offset(), view.len())
                    }
                    KernelArg::U32(value) => Arg::U32(value),
                    KernelArg::I32(value) => Arg::I32(value),
                    KernelArg::U64(value) => Arg::U64(value),
                    KernelArg::F32(value) => Arg::F32(value),
                    KernelArg::Bytes(bytes) => Arg::Bytes(bytes.len()),
                    // The argument set is open; an argument this fixture has no
                    // name for is recorded as an empty by-value block rather
                    // than dropped, so a test that expected one sees a
                    // mismatch instead of a shorter list.
                    _ => Arg::Bytes(0),
                })
                .collect(),
            cooperative,
        });
    }

    fn fail(op: &'static str, detail: String) -> CoreError {
        CoreError::Backend {
            backend: "fake",
            op,
            source: Box::new(std::io::Error::other(detail)),
        }
    }

    /// The host range one view names, checked against the live allocation.
    fn range_of(
        &self,
        view: DeviceView<'_>,
        len: u64,
        op: &'static str,
    ) -> CoreResult<Range<usize>> {
        if len != view.len() {
            return Err(CoreError::OutOfBounds {
                offset: view.offset(),
                len,
                buffer_bytes: view.len(),
            });
        }
        let start = usize::try_from(view.offset())
            .map_err(|_| Self::fail(op, "offset too large".into()))?;
        let width = usize::try_from(len).map_err(|_| Self::fail(op, "length too large".into()))?;
        let table = self.live.lock();
        let buffer = table
            .get(&view.id().get())
            .ok_or_else(|| Self::fail(op, format!("no live allocation {}", view.id())))?;
        let end = start
            .checked_add(width)
            .ok_or_else(|| Self::fail(op, "range wraps".into()))?;
        if end > buffer.len() {
            return Err(CoreError::OutOfBounds {
                offset: view.offset(),
                len,
                buffer_bytes: u64::try_from(buffer.len()).unwrap_or(u64::MAX),
            });
        }
        Ok(start..end)
    }
}

impl Backend for FakeBackend {
    fn backend_name(&self) -> &'static str {
        "fake"
    }

    fn alloc_raw(&self, bytes: u64) -> CoreResult<RawBuffer> {
        let width = usize::try_from(bytes)
            .map_err(|_| Self::fail("alloc_raw", format!("{bytes} bytes do not fit this host")))?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let id = NonZeroU64::new(id)
            .ok_or_else(|| Self::fail("alloc_raw", "id space exhausted".into()))?;
        // Poisoned, not zeroed. A device allocator hands back whatever the
        // last tenant left at that address, and a double that zeroes is a
        // double that cannot fail the way the device does: every read of an
        // uninitialised buffer would come back plausible. 0xA5 is a value no
        // arithmetic here produces by accident.
        self.live.lock().insert(id.get(), vec![POISON; width]);
        self.allocated.lock().push(id.get());
        Ok(RawBuffer::new(id, bytes))
    }

    fn free_raw(&self, raw: RawBuffer) -> CoreResult<()> {
        if self.live.lock().remove(&raw.id().get()).is_some() {
            self.released.lock().push(raw.id().get());
            return Ok(());
        }
        self.double_frees.fetch_add(1, Ordering::Relaxed);
        Err(Self::fail(
            "free_raw",
            format!("{} was already free", raw.id()),
        ))
    }

    fn copy_h2d(&self, dst: DeviceView<'_>, src: &[u8]) -> CoreResult<()> {
        let len = u64::try_from(src.len()).unwrap_or(u64::MAX);
        let range = self.range_of(dst, len, "copy_h2d")?;
        let launches = self.launches.lock().len();
        self.uploads.lock().push(Upload {
            launches,
            buffer: dst.id().get(),
            offset: dst.offset(),
            len,
        });
        self.live
            .lock()
            .get_mut(&dst.id().get())
            .expect("the range check found the allocation")
            .splice(range, src.iter().copied());
        Ok(())
    }

    fn copy_d2h(&self, src: DeviceView<'_>, dst: &mut [u8]) -> CoreResult<()> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        let len = u64::try_from(dst.len()).unwrap_or(u64::MAX);
        let range = self.range_of(src, len, "copy_d2h")?;
        let table = self.live.lock();
        let buffer = table
            .get(&src.id().get())
            .expect("the range check found the allocation");
        dst.copy_from_slice(&buffer[range]);
        Ok(())
    }

    fn copy_d2d(&self, dst: DeviceView<'_>, src: DeviceView<'_>) -> CoreResult<()> {
        if dst.len() != src.len() {
            return Err(CoreError::ShapeMismatch {
                field: "copy_d2d len",
                expected: dst.len(),
                got: src.len(),
            });
        }
        let source = self.range_of(src, src.len(), "copy_d2d")?;
        let copied = {
            let table = self.live.lock();
            table
                .get(&src.id().get())
                .expect("the range check found the allocation")[source]
                .to_vec()
        };
        let target = self.range_of(dst, dst.len(), "copy_d2d")?;
        self.live
            .lock()
            .get_mut(&dst.id().get())
            .expect("the range check found the allocation")
            .splice(target, copied);
        Ok(())
    }

    fn run(&self, kernel: KernelRef, args: &LaunchArgs<'_>) -> CoreResult<()> {
        self.record(kernel, args, false);
        Ok(())
    }

    fn sync(&self) -> CoreResult<()> {
        Ok(())
    }

    /// The fake device claims a matrix tier.
    ///
    /// Not a nicety: the widest projections in the model have no per-row kernel
    /// past a handful of rows, so a device with no tier here refuses a prefill
    /// chunk outright - and a fixture that refused would be testing the refusal
    /// rather than the pass.
    fn as_blas(&self) -> Option<&dyn BlasBackend> {
        Some(self)
    }

    /// The fake device hands out addresses, because the batched kernels take
    /// arrays of them and a device that could not would refuse a decode batch
    /// outright - and a fixture that refused would be testing the refusal.
    fn as_pointers(&self) -> Option<&dyn PointerPack> {
        Some(self)
    }

    /// The fake device claims cooperative launch so that the scratch
    /// reservation the routed expert stage makes is a call a test can see
    /// rather than a refusal it cannot get past.
    fn as_coop(&self) -> Option<&dyn CoopLaunch> {
        Some(self)
    }
}

/// The address this device would have handed out for one range.
///
/// A real address comes from an allocator and means nothing to a reader; this
/// one is a function of the handle and the offset, which is everything a test
/// needs of an address and everything a kernel needs of one: two live
/// allocations never share a value, and a range inside an allocation differs
/// from the whole by exactly its offset. It is the function the test that reads
/// a packed array back compares against, so a packer that wrote the first row's
/// pointer into every slot fails rather than passing.
pub(crate) fn fake_address(view: DeviceView<'_>) -> u64 {
    (view.id().get() << 32) | (view.offset() & 0xffff_ffff)
}

impl PointerPack for FakeBackend {
    /// Writes one address per source range, checking each against the live
    /// table first.
    ///
    /// The check is the part worth having: a view naming an allocation the
    /// device has already released is a failure here, exactly as it would be a
    /// fault there, rather than an address written into an array nobody can
    /// tell apart from a good one.
    fn pack_pointers(&self, dst: DeviceView<'_>, srcs: &[DeviceView<'_>]) -> CoreResult<()> {
        let want = (srcs.len() as u64) * shiso_core::DEVICE_POINTER_BYTES;
        if want > dst.len() {
            return Err(CoreError::OutOfBounds {
                offset: dst.offset(),
                len: want,
                buffer_bytes: dst.len(),
            });
        }
        let mut bytes = Vec::with_capacity(srcs.len() * 8);
        for src in srcs {
            self.range_of(*src, src.len(), "pack_pointers")?;
            bytes.extend_from_slice(&fake_address(*src).to_ne_bytes());
        }
        self.copy_h2d(dst.slice(0, want)?, &bytes)
    }
}

impl CoopLaunch for FakeBackend {
    fn max_co_resident_blocks(
        &self,
        _kernel: KernelRef,
        _block: Extent3,
        _smem: u32,
    ) -> CoreResult<u32> {
        Ok(CO_RESIDENT)
    }

    fn reserve_scratch(&self, bytes: u64) -> CoreResult<()> {
        self.reservations.lock().push(bytes);
        let held = self.scratch.lock().map(RawBuffer::bytes).unwrap_or(0);
        if held >= bytes {
            return Ok(());
        }
        let raw = self.alloc_raw(bytes.max(1))?;
        if let Some(old) = self.scratch.lock().replace(raw) {
            self.free_raw(old)?;
        }
        Ok(())
    }

    fn scratch(&self) -> Option<DeviceView<'_>> {
        let raw = (*self.scratch.lock())?;
        Some(DeviceView::from_raw(raw, 0, raw.bytes()))
    }

    fn run_cooperative(&self, kernel: KernelRef, args: &LaunchArgs<'_>) -> CoreResult<()> {
        self.record(kernel, args, true);
        Ok(())
    }
}

impl BlasBackend for FakeBackend {
    /// Records nothing and computes nothing: what a plan says about the tier is
    /// which calls it makes, and those are steps in the plan rather than
    /// launches on a device.
    fn gemm(&self, _desc: &GemmDesc<'_>) -> CoreResult<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// the fixture's geometry
// ---------------------------------------------------------------------------

/// Layer slots the fixture carries.
pub(crate) const N_LAYERS: u32 = 8;
/// Slots between attention slots.
pub(crate) const INTERVAL: u32 = 4;
/// Recurrent slots, which is how many states a session allocates.
pub(crate) const RECURRENT_SLOTS: usize = 6;
/// Attention slots, which is how many planes the cache allocates for each of
/// its four per-layer buffers.
pub(crate) const ATTN_SLOTS: usize = 2;
/// Width of one residual stream.
///
/// Not cut further: the trellis codec mixes 128 output columns together and
/// packs 256 weights into a block, so an expert stack narrower than this cannot
/// be stored in the format the shipping checkpoints use - and the expert stage
/// is the one stage that has no other path.
pub(crate) const HIDDEN: u32 = 256;
/// Residual streams.
pub(crate) const STREAMS: u32 = 4;
/// Inner width of the stream mix's low-rank pair.
pub(crate) const STREAM_RANK: u32 = 16;
/// Vocabulary.
pub(crate) const VOCAB: u32 = 64;
/// Attention query heads.
pub(crate) const Q_HEADS: u32 = 4;
/// Attention key and value heads.
pub(crate) const KV_HEADS: u32 = 2;
/// Attention head width.
///
/// The one width the selected-block kernels' slabs are sized for, kept at full
/// size because every sparse path's precondition is stated on it: cutting it
/// would silently take the fixture off those paths.
pub(crate) const HEAD_DIM: u32 = 256;
/// Rotated prefix of an attention head.
pub(crate) const N_ROT: u32 = 8;
/// Recurrent value heads.
pub(crate) const GDN_V_HEADS: u32 = 4;
/// Recurrent query and key heads.
pub(crate) const GDN_QK_HEADS: u32 = 2;
/// Recurrent head width.
pub(crate) const GDN_HEAD_DIM: u32 = 16;
/// Width of the recurrent convolution.
pub(crate) const GDN_CONV: u32 = 4;
/// Routed experts.
pub(crate) const EXPERTS: u32 = 8;
/// Experts one token reaches.
pub(crate) const EXPERTS_USED: u32 = 2;
/// Inner width of one expert, bounded from below for the same reason as the
/// stream width.
pub(crate) const EXPERT_INTERMEDIATE: u32 = 256;
/// Indexer heads.
pub(crate) const INDEXER_HEADS: u32 = 4;
/// Width of one indexer head.
pub(crate) const INDEXER_HEAD_DIM: u32 = 128;
/// Positions the indexer's selection may reach.
pub(crate) const INDEXER_TOP_K: u32 = 64;
/// Positions one pooled key block averages over.
pub(crate) const INDEXER_COMPRESS: u32 = 4;
/// The slot that folds in the n-gram block.
pub(crate) const PLE_LAYER: u32 = 1;
/// Gram widths the n-gram table is partitioned into.
pub(crate) const PLE_NGRAM: u32 = 3;
/// Heads each gram width contributes.
pub(crate) const PLE_HEADS_PER_NGRAM: u32 = 2;
/// Width of one n-gram head's row: four heads multiply out to the residual.
pub(crate) const PLE_HEAD_DIM: u32 = HIDDEN / 4;
/// Storage format the expert stack is written in.
///
/// The trellis codec, because it is the one format whose expert stage this
/// build can plan: the staged gate and up belong to a kernel set that is not
/// vendored here, and a fixture that used it would be asserting the refusal
/// rather than the pass.
const T_EXL3_K4: u32 = 68;
/// Width of the n-gram convolution.
pub(crate) const PLE_CONV: u32 = 4;
/// Hash heads one position gathers a row for: one per head per gram width, and
/// the shortest width is the token itself, which the ordinary embedding table
/// already carries.
pub(crate) const PLE_HEADS: u32 = (PLE_NGRAM - 1) * PLE_HEADS_PER_NGRAM;
/// Rows each hash head owns in the fixture's table.
pub(crate) const PLE_HEAD_ROWS: u64 = 128;
/// Rows the fixture's whole n-gram table holds.
pub(crate) const PLE_TABLE_ROWS: u64 = PLE_HEAD_ROWS * PLE_HEADS as u64;
/// Bytes one of its rows occupies: thirty-two four-bit weights and a shared
/// half-precision scale, per block.
pub(crate) const PLE_ROW_BYTES: u64 = PLE_HEAD_DIM as u64 / 32 * 18;

/// The byte the fixture's table holds at position `j` of row `id`.
///
/// A function of both, so that a row read from the wrong offset is a different
/// byte at every position rather than a plausible one.
pub(crate) fn ple_row_byte(id: u64, j: usize) -> u8 {
    u8::try_from((id.wrapping_mul(31) + j as u64 * 7 + 11) & 0xFF).expect("masked to a byte")
}

/// The fixture table's own numbers.
///
/// The heads tile the table without a gap, which is what makes every identifier
/// the hash computes in range, and every multiplier is odd, which is what keeps
/// the hash a bijection on the low bits.
pub(crate) fn tiny_ngram() -> shiso_models::qwen4exp::NgramHash {
    shiso_models::qwen4exp::NgramHash {
        multipliers: vec![
            0x9E37_79B9_7F4A_7C15,
            0xBF58_476D_1CE4_E5B9,
            0x94D0_49BB_1331_11EB,
        ],
        head_offsets: (0..u64::from(PLE_HEADS))
            .map(|h| h * PLE_HEAD_ROWS)
            .collect(),
        head_vocab: vec![PLE_HEAD_ROWS; PLE_HEADS as usize],
        eos_token: TokenId::new(VOCAB - 2),
    }
}
/// Tokens one session may hold.
pub(crate) const CTX: u32 = 9_216;
/// Pages the pool holds.
pub(crate) const POOL_PAGES: u32 = 40;
/// Blocks the fake device claims it can hold resident.
pub(crate) const CO_RESIDENT: u32 = 48;

/// The fixture's geometry.
pub(crate) fn tiny_shape() -> SparseShape {
    SparseShape {
        layers: shiso_models::qwen4exp::layer_kinds(N_LAYERS, INTERVAL),
        hidden: HIDDEN,
        vocab: VOCAB,
        streams: STREAMS,
        stream_rank: STREAM_RANK,
        gqa: GqaShape::new(Q_HEADS, KV_HEADS, HEAD_DIM, N_ROT),
        gdn: GdnShape::new(GDN_V_HEADS, GDN_QK_HEADS, GDN_HEAD_DIM, GDN_CONV),
        moe: MoeShape::new(EXPERTS, EXPERTS_USED, 1, EXPERT_INTERMEDIATE),
        indexer: IndexerShape::new(
            INDEXER_HEADS,
            INDEXER_HEAD_DIM,
            INDEXER_TOP_K,
            INDEXER_COMPRESS,
        ),
        ple: Some(PleShape::new(
            LayerIdx::new(PLE_LAYER),
            PLE_NGRAM,
            PLE_HEADS_PER_NGRAM,
            PLE_HEAD_DIM,
            PLE_CONV,
        )),
        mrope: MropeSections::new(2, 1, 1),
        rms_eps: 1.0e-6,
        rope_theta: 1.0e7,
    }
}

/// The fixture's config.
pub(crate) fn tiny_config() -> Qwen4ExpConfig {
    Qwen4ExpConfig::new(
        tiny_shape(),
        TokenId::new(VOCAB - 1),
        INTERVAL,
        tiny_ngram(),
    )
    .expect("the fixture geometry folds in an n-gram block")
}

/// One instantiation of the half-precision row projection.
const fn f16_rows(rows: u32) -> Instantiation {
    Instantiation::new(
        kernels::MOD_MATVEC,
        "q4e_matmul_f16_rows_kernel",
        TemplateArgs::RowsIo {
            rows,
            input: Elem::F32,
            output: Elem::F32,
        },
    )
}

/// One instantiation of a cooperative trellis projection.
const fn trellis(template: &'static str, bits: u32, shape: u8) -> Instantiation {
    Instantiation::new(
        kernels::MOD_EXL3,
        template,
        TemplateArgs::Trellis { bits, shape },
    )
}

/// One instantiation of a trellis kernel that takes only the codec width.
const fn trellis_bits(template: &'static str, bits: u32) -> Instantiation {
    Instantiation::new(kernels::MOD_EXL3, template, TemplateArgs::TrellisBits(bits))
}

/// One trellis-module entry with C++ linkage and no template arguments.
const fn untemplated(name: &'static str) -> Instantiation {
    Instantiation::new(kernels::MOD_EXL3, name, TemplateArgs::Untemplated)
}

/// The entry names a build would have captured off its compiled modules.
///
/// Stand-ins, not guesses at a real mangling: nothing here launches anything,
/// and what the table is for is that a dispatch resolves at all. A build with
/// the toolchain fills this in from the module's own symbol list, which is the
/// only place those names can honestly come from.
static CAPTURED: [(Instantiation, &str); 125] = [
    (
        Instantiation::new(
            kernels::MOD_HC,
            "q4e_hc_norm_kernel",
            TemplateArgs::Elem(Elem::F32),
        ),
        "captured_hc_norm_f32",
    ),
    (
        Instantiation::new(
            kernels::MOD_HC,
            "q4e_hc_collapse_kernel",
            TemplateArgs::Elem(Elem::F32),
        ),
        "captured_hc_collapse_f32",
    ),
    (
        Instantiation::new(
            kernels::MOD_HC,
            "q4e_hc_combine_norm_kernel",
            TemplateArgs::Elem(Elem::F32),
        ),
        "captured_hc_combine_norm_f32",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matvec_combine_t_kernel",
            TemplateArgs::Elem(Elem::F32),
        ),
        "captured_matvec_combine_f32",
    ),
    (f16_rows(1), "captured_f16_rows_1"),
    (f16_rows(2), "captured_f16_rows_2"),
    (f16_rows(3), "captured_f16_rows_3"),
    (f16_rows(4), "captured_f16_rows_4"),
    (f16_rows(5), "captured_f16_rows_5"),
    (f16_rows(6), "captured_f16_rows_6"),
    (f16_rows(7), "captured_f16_rows_7"),
    (f16_rows(8), "captured_f16_rows_8"),
    (f16_rows(9), "captured_f16_rows_9"),
    (f16_rows(10), "captured_f16_rows_10"),
    (f16_rows(11), "captured_f16_rows_11"),
    (f16_rows(12), "captured_f16_rows_12"),
    (f16_rows(13), "captured_f16_rows_13"),
    (f16_rows(14), "captured_f16_rows_14"),
    (f16_rows(15), "captured_f16_rows_15"),
    (f16_rows(16), "captured_f16_rows_16"),
    (f16_rows(17), "captured_f16_rows_17"),
    (f16_rows(18), "captured_f16_rows_18"),
    (f16_rows(19), "captured_f16_rows_19"),
    (f16_rows(20), "captured_f16_rows_20"),
    (f16_rows(21), "captured_f16_rows_21"),
    (f16_rows(22), "captured_f16_rows_22"),
    (f16_rows(23), "captured_f16_rows_23"),
    (f16_rows(24), "captured_f16_rows_24"),
    (f16_rows(25), "captured_f16_rows_25"),
    (f16_rows(26), "captured_f16_rows_26"),
    (f16_rows(27), "captured_f16_rows_27"),
    (f16_rows(28), "captured_f16_rows_28"),
    (f16_rows(29), "captured_f16_rows_29"),
    (f16_rows(30), "captured_f16_rows_30"),
    (f16_rows(31), "captured_f16_rows_31"),
    (f16_rows(32), "captured_f16_rows_32"),
    (
        trellis(kernels::EXL3_GEMM, 4, 1),
        "captured_exl3_gemm_k4_s1",
    ),
    (
        trellis(kernels::EXL3_MGEMM, 4, 1),
        "captured_exl3_mgemm_k4_s1",
    ),
    (
        trellis(kernels::EXL3_GEMM, 4, 2),
        "captured_exl3_gemm_k4_s2",
    ),
    (
        trellis(kernels::EXL3_MGEMM, 4, 2),
        "captured_exl3_mgemm_k4_s2",
    ),
    (
        trellis(kernels::EXL3_GEMM, 4, 3),
        "captured_exl3_gemm_k4_s3",
    ),
    (
        trellis(kernels::EXL3_MGEMM, 4, 3),
        "captured_exl3_mgemm_k4_s3",
    ),
    (
        trellis(kernels::EXL3_GEMM, 4, 4),
        "captured_exl3_gemm_k4_s4",
    ),
    (
        trellis(kernels::EXL3_MGEMM, 4, 4),
        "captured_exl3_mgemm_k4_s4",
    ),
    (
        trellis(kernels::EXL3_GEMM, 5, 1),
        "captured_exl3_gemm_k5_s1",
    ),
    (
        trellis(kernels::EXL3_MGEMM, 5, 1),
        "captured_exl3_mgemm_k5_s1",
    ),
    (
        trellis(kernels::EXL3_GEMM, 5, 2),
        "captured_exl3_gemm_k5_s2",
    ),
    (
        trellis(kernels::EXL3_MGEMM, 5, 2),
        "captured_exl3_mgemm_k5_s2",
    ),
    (
        trellis(kernels::EXL3_GEMM, 5, 3),
        "captured_exl3_gemm_k5_s3",
    ),
    (
        trellis(kernels::EXL3_MGEMM, 5, 3),
        "captured_exl3_mgemm_k5_s3",
    ),
    (
        trellis(kernels::EXL3_GEMM, 5, 4),
        "captured_exl3_gemm_k5_s4",
    ),
    (
        trellis(kernels::EXL3_MGEMM, 5, 4),
        "captured_exl3_mgemm_k5_s4",
    ),
    (
        trellis(kernels::EXL3_GEMM, 6, 1),
        "captured_exl3_gemm_k6_s1",
    ),
    (
        trellis(kernels::EXL3_MGEMM, 6, 1),
        "captured_exl3_mgemm_k6_s1",
    ),
    (
        trellis(kernels::EXL3_GEMM, 6, 2),
        "captured_exl3_gemm_k6_s2",
    ),
    (
        trellis(kernels::EXL3_MGEMM, 6, 2),
        "captured_exl3_mgemm_k6_s2",
    ),
    (
        trellis(kernels::EXL3_GEMM, 6, 3),
        "captured_exl3_gemm_k6_s3",
    ),
    (
        trellis(kernels::EXL3_MGEMM, 6, 3),
        "captured_exl3_mgemm_k6_s3",
    ),
    (
        trellis(kernels::EXL3_GEMM, 6, 4),
        "captured_exl3_gemm_k6_s4",
    ),
    (
        trellis(kernels::EXL3_MGEMM, 6, 4),
        "captured_exl3_mgemm_k6_s4",
    ),
    (trellis_bits(kernels::EXL3_MOE, 4), "captured_exl3_moe_k4"),
    (
        trellis_bits(kernels::EXL3_RECONSTRUCT, 4),
        "captured_exl3_reconstruct_k4",
    ),
    (trellis_bits(kernels::EXL3_MOE, 5), "captured_exl3_moe_k5"),
    (
        trellis_bits(kernels::EXL3_RECONSTRUCT, 5),
        "captured_exl3_reconstruct_k5",
    ),
    (trellis_bits(kernels::EXL3_MOE, 6), "captured_exl3_moe_k6"),
    (
        trellis_bits(kernels::EXL3_RECONSTRUCT, 6),
        "captured_exl3_reconstruct_k6",
    ),
    // The weight-stationary routed block a prefill chunk takes.
    (
        trellis_bits(kernels::EXL3_MOE_WS_GATEUP, 4),
        "captured_ws_gateup_k4",
    ),
    (
        trellis_bits(kernels::EXL3_MOE_WS_GATEUP, 5),
        "captured_ws_gateup_k5",
    ),
    (
        trellis_bits(kernels::EXL3_MOE_WS_GATEUP, 6),
        "captured_ws_gateup_k6",
    ),
    (
        trellis_bits(kernels::EXL3_MOE_WS_DOWN_H, 4),
        "captured_ws_down_h_k4",
    ),
    (
        trellis_bits(kernels::EXL3_MOE_WS_DOWN_H, 5),
        "captured_ws_down_h_k5",
    ),
    (
        trellis_bits(kernels::EXL3_MOE_WS_DOWN_H, 6),
        "captured_ws_down_h_k6",
    ),
    (
        untemplated(kernels::EXL3_MOE_WS_INVERT),
        "captured_ws_invert",
    ),
    (
        untemplated(kernels::EXL3_MOE_WS_PREP_TOK),
        "captured_ws_prep_tok",
    ),
    (
        untemplated(kernels::EXL3_MOE_WS_COMBINE),
        "captured_ws_combine",
    ),
    // The decode-shaped routed block and dense projection.
    (
        trellis_bits(kernels::EXL3_MOE_WSD_GATEUP, 4),
        "captured_wsd_gateup_k4",
    ),
    (
        trellis_bits(kernels::EXL3_MOE_WSD_DOWN, 4),
        "captured_wsd_down_k4",
    ),
    (
        Instantiation::new(
            kernels::MOD_EXL3,
            kernels::EXL3_DGEMV,
            TemplateArgs::TrellisTiles { bits: 4, tiles: 1 },
        ),
        "captured_dgemv_k4_t1",
    ),
    (
        Instantiation::new(
            kernels::MOD_EXL3,
            kernels::EXL3_DGEMV,
            TemplateArgs::TrellisTiles { bits: 4, tiles: 2 },
        ),
        "captured_dgemv_k4_t2",
    ),
    (
        trellis_bits(kernels::EXL3_MOE_WSD_GATEUP, 5),
        "captured_wsd_gateup_k5",
    ),
    (
        trellis_bits(kernels::EXL3_MOE_WSD_DOWN, 5),
        "captured_wsd_down_k5",
    ),
    (
        Instantiation::new(
            kernels::MOD_EXL3,
            kernels::EXL3_DGEMV,
            TemplateArgs::TrellisTiles { bits: 5, tiles: 1 },
        ),
        "captured_dgemv_k5_t1",
    ),
    (
        Instantiation::new(
            kernels::MOD_EXL3,
            kernels::EXL3_DGEMV,
            TemplateArgs::TrellisTiles { bits: 5, tiles: 2 },
        ),
        "captured_dgemv_k5_t2",
    ),
    (
        trellis_bits(kernels::EXL3_MOE_WSD_GATEUP, 6),
        "captured_wsd_gateup_k6",
    ),
    (
        trellis_bits(kernels::EXL3_MOE_WSD_DOWN, 6),
        "captured_wsd_down_k6",
    ),
    (
        Instantiation::new(
            kernels::MOD_EXL3,
            kernels::EXL3_DGEMV,
            TemplateArgs::TrellisTiles { bits: 6, tiles: 1 },
        ),
        "captured_dgemv_k6_t1",
    ),
    (
        Instantiation::new(
            kernels::MOD_EXL3,
            kernels::EXL3_DGEMV,
            TemplateArgs::TrellisTiles { bits: 6, tiles: 2 },
        ),
        "captured_dgemv_k6_t2",
    ),
    (untemplated(kernels::EXL3_MOE_WSD_PREP), "captured_wsd_prep"),
    (untemplated(kernels::EXL3_MOE_WSD_ACT), "captured_wsd_act"),
    (
        untemplated(kernels::EXL3_MOE_WSD_COMBINE),
        "captured_wsd_combine",
    ),
    (untemplated(kernels::EXL3_DGEMV_HAD), "captured_dgemv_had"),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(1),
        ),
        "captured_p12_rows_1",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(2),
        ),
        "captured_p12_rows_2",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(3),
        ),
        "captured_p12_rows_3",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(4),
        ),
        "captured_p12_rows_4",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(5),
        ),
        "captured_p12_rows_5",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(6),
        ),
        "captured_p12_rows_6",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(7),
        ),
        "captured_p12_rows_7",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(8),
        ),
        "captured_p12_rows_8",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(9),
        ),
        "captured_p12_rows_9",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(10),
        ),
        "captured_p12_rows_10",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(11),
        ),
        "captured_p12_rows_11",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(12),
        ),
        "captured_p12_rows_12",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(13),
        ),
        "captured_p12_rows_13",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(14),
        ),
        "captured_p12_rows_14",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(15),
        ),
        "captured_p12_rows_15",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(16),
        ),
        "captured_p12_rows_16",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(17),
        ),
        "captured_p12_rows_17",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(18),
        ),
        "captured_p12_rows_18",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(19),
        ),
        "captured_p12_rows_19",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(20),
        ),
        "captured_p12_rows_20",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(21),
        ),
        "captured_p12_rows_21",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(22),
        ),
        "captured_p12_rows_22",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(23),
        ),
        "captured_p12_rows_23",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(24),
        ),
        "captured_p12_rows_24",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(25),
        ),
        "captured_p12_rows_25",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(26),
        ),
        "captured_p12_rows_26",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(27),
        ),
        "captured_p12_rows_27",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(28),
        ),
        "captured_p12_rows_28",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(29),
        ),
        "captured_p12_rows_29",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(30),
        ),
        "captured_p12_rows_30",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(31),
        ),
        "captured_p12_rows_31",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(32),
        ),
        "captured_p12_rows_32",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_p12_expand_kernel",
            TemplateArgs::Elem(Elem::F16),
        ),
        "captured_p12_expand_f16",
    ),
    (
        Instantiation::new(
            kernels::MOD_MATVEC,
            "q4e_p12_expand_kernel",
            TemplateArgs::Elem(Elem::F32),
        ),
        "captured_p12_expand_f32",
    ),
];

/// The options every test loads with.
pub(crate) fn options() -> SparseOptions {
    let mut engine = EngineOptions::default();
    engine.ctx = CTX;
    engine.kv_pool_pages = POOL_PAGES;
    engine.kv_ckpt_slots = 2;
    SparseOptions {
        engine,
        co_resident_blocks: CO_RESIDENT,
        entries: EntryTable::new(&CAPTURED),
        ..SparseOptions::default()
    }
}

// ---------------------------------------------------------------------------
// the synthetic checkpoint
// ---------------------------------------------------------------------------

const MAGIC: u32 = 0x4655_4747;
const FORMAT_VERSION: u32 = 3;
const ALIGN: usize = 32;
/// Storage format ids, as the container spells them.
const T_F32: u32 = 0;
const T_F16: u32 = 1;
/// The block-quantized n-gram row codec: thirty-two four-bit weights and a
/// shared half-precision scale to a block.
const T_IQ4_NL: u32 = 20;

/// One tensor the fixture writes.
struct Spec {
    name: String,
    dims: Vec<u64>,
    type_id: u32,
}

fn spec(name: String, dims: Vec<u64>, type_id: u32) -> Spec {
    Spec {
        name,
        dims,
        type_id,
    }
}

fn payload_bytes(type_id: u32, dims: &[u64]) -> u64 {
    let elems: u64 = dims.iter().product();
    match type_id {
        T_F32 => elems * 4,
        T_F16 => elems * 2,
        T_IQ4_NL => elems / 32 * 18,
        // A trellis block is 256 weights in 128 bytes, plus the two fp16 scale
        // vectors the format appends inside the tensor's own span: one of the
        // contraction width and one of the output width.
        T_EXL3_K4 => {
            let k = dims.first().copied().unwrap_or(0);
            let n = dims.get(1).copied().unwrap_or(1);
            elems.div_ceil(256) * 128 + 2 * (k + n)
        }
        other => panic!("the fixture writes no tensors of format {other}"),
    }
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, text: &str) {
    put_u64(out, u64::try_from(text.len()).expect("a fixture name fits"));
    out.extend_from_slice(text.as_bytes());
}

/// Every tensor the binding asks this family for, at the fixture's widths.
///
/// Two formats, and the split is the shipping one: the experts are trellis,
/// because that is the only expert stage this build can plan at all, and
/// everything else is half precision, because that is the dense tier whose
/// whole ladder is vendored here. A plan built over this fixture therefore
/// reaches a real kernel choice at every projection rather than a refusal, and
/// the refusals get tests of their own, stated directly.
#[expect(
    clippy::too_many_lines,
    reason = "one statement per tensor role is the point: a fixture that folded \
              roles into a loop would stop being readable as the tensor set the \
              binding asks for"
)]
fn fixture_tensors() -> Vec<Spec> {
    let hidden = u64::from(HIDDEN);
    let stream = u64::from(HIDDEN) * u64::from(STREAMS);
    let rank = u64::from(STREAM_RANK);
    let q_dim = u64::from(Q_HEADS) * u64::from(HEAD_DIM);
    let kv_dim = u64::from(KV_HEADS) * u64::from(HEAD_DIM);
    let gdn_qk = u64::from(GDN_QK_HEADS) * u64::from(GDN_HEAD_DIM);
    let gdn_v = u64::from(GDN_V_HEADS) * u64::from(GDN_HEAD_DIM);
    let gdn_in = 2 * gdn_qk + gdn_v;
    let idx_q = u64::from(INDEXER_HEADS) * u64::from(INDEXER_HEAD_DIM);
    let idx_k = u64::from(INDEXER_HEAD_DIM);
    let inter = u64::from(EXPERT_INTERMEDIATE);
    let experts = u64::from(EXPERTS);

    let mut out = vec![
        spec(
            "token_embd.weight".into(),
            vec![hidden, u64::from(VOCAB)],
            T_F16,
        ),
        spec(
            "output.weight".into(),
            vec![hidden, u64::from(VOCAB)],
            T_F16,
        ),
        spec("output_hc_norm.weight".into(), vec![stream], T_F32),
        spec("output_hc_down.weight".into(), vec![stream, rank], T_F16),
        spec("output_hc_up.weight".into(), vec![rank, stream], T_F16),
    ];
    for il in 0..N_LAYERS {
        for side in ["attn", "ffn"] {
            out.push(spec(
                format!("blk.{il}.hc_{side}_norm.weight"),
                vec![stream],
                T_F32,
            ));
            out.push(spec(
                format!("blk.{il}.hc_{side}_down.weight"),
                vec![stream, rank],
                T_F16,
            ));
            out.push(spec(
                format!("blk.{il}.hc_{side}_up.weight"),
                vec![rank, stream],
                T_F16,
            ));
            out.push(spec(
                format!("blk.{il}.hc_{side}_inject.weight"),
                vec![stream, u64::from(STREAMS)],
                T_F16,
            ));
        }
        if shiso_models::qwen4exp::is_attention(il, INTERVAL) {
            // The query projection carries its own output gate in its rows,
            // which is why it is twice as tall as the query alone.
            out.push(spec(
                format!("blk.{il}.attn_q.weight"),
                vec![hidden, 2 * q_dim],
                T_F16,
            ));
            out.push(spec(
                format!("blk.{il}.attn_k.weight"),
                vec![hidden, kv_dim],
                T_F16,
            ));
            out.push(spec(
                format!("blk.{il}.attn_v.weight"),
                vec![hidden, kv_dim],
                T_F16,
            ));
            out.push(spec(
                format!("blk.{il}.attn_q_norm.weight"),
                vec![u64::from(HEAD_DIM)],
                T_F32,
            ));
            out.push(spec(
                format!("blk.{il}.attn_k_norm.weight"),
                vec![u64::from(HEAD_DIM)],
                T_F32,
            ));
            out.push(spec(
                format!("blk.{il}.attn_output.weight"),
                vec![q_dim, hidden],
                T_F16,
            ));
            out.push(spec(
                format!("blk.{il}.indexer.q_proj.weight"),
                vec![hidden, idx_q],
                T_F16,
            ));
            out.push(spec(
                format!("blk.{il}.indexer.k_proj.weight"),
                vec![hidden, idx_k],
                T_F16,
            ));
            out.push(spec(
                format!("blk.{il}.indexer.q_norm.weight"),
                vec![idx_k],
                T_F32,
            ));
            out.push(spec(
                format!("blk.{il}.indexer.k_norm.weight"),
                vec![idx_k],
                T_F32,
            ));
        } else {
            out.push(spec(
                format!("blk.{il}.attn_qkv.weight"),
                vec![hidden, gdn_in],
                T_F16,
            ));
            out.push(spec(
                format!("blk.{il}.attn_gate.weight"),
                vec![hidden, gdn_v],
                T_F16,
            ));
            out.push(spec(
                format!("blk.{il}.ssm_conv1d.weight"),
                vec![u64::from(GDN_CONV), gdn_in],
                T_F32,
            ));
            out.push(spec(
                format!("blk.{il}.ssm_a"),
                vec![u64::from(GDN_V_HEADS)],
                T_F32,
            ));
            out.push(spec(
                format!("blk.{il}.ssm_alpha.weight"),
                vec![hidden, u64::from(GDN_V_HEADS)],
                T_F16,
            ));
            out.push(spec(
                format!("blk.{il}.ssm_beta.weight"),
                vec![hidden, u64::from(GDN_V_HEADS)],
                T_F16,
            ));
            out.push(spec(
                format!("blk.{il}.ssm_dt.bias"),
                vec![u64::from(GDN_V_HEADS)],
                T_F32,
            ));
            out.push(spec(
                format!("blk.{il}.ssm_norm.weight"),
                vec![u64::from(GDN_HEAD_DIM)],
                T_F32,
            ));
            out.push(spec(
                format!("blk.{il}.ssm_out.weight"),
                vec![gdn_v, hidden],
                T_F16,
            ));
        }
        if il == PLE_LAYER {
            out.push(spec(
                format!("blk.{il}.ple_key.weight"),
                vec![hidden, stream],
                T_F16,
            ));
            out.push(spec(
                format!("blk.{il}.ple_value.weight"),
                vec![hidden, hidden],
                T_F16,
            ));
            out.push(spec(
                format!("blk.{il}.ple_norm_key.weight"),
                vec![stream],
                T_F32,
            ));
            out.push(spec(
                format!("blk.{il}.ple_norm_query.weight"),
                vec![stream],
                T_F32,
            ));
            out.push(spec(
                format!("blk.{il}.ple_norm_conv.weight"),
                vec![stream],
                T_F32,
            ));
            out.push(spec(
                format!("blk.{il}.ple_conv1d.weight"),
                vec![u64::from(PLE_CONV), stream],
                T_F32,
            ));
        }
        out.push(spec(
            format!("blk.{il}.ffn_gate_inp.weight"),
            vec![hidden, experts],
            T_F16,
        ));
        out.push(spec(
            format!("blk.{il}.ffn_gate_inp_shexp.weight"),
            vec![hidden, 1],
            T_F16,
        ));
        for role in ["gate", "up"] {
            out.push(spec(
                format!("blk.{il}.ffn_{role}_exps.weight"),
                vec![hidden, inter * experts],
                T_EXL3_K4,
            ));
            out.push(spec(
                format!("blk.{il}.ffn_{role}_shexp.weight"),
                vec![hidden, inter],
                T_EXL3_K4,
            ));
        }
        out.push(spec(
            format!("blk.{il}.ffn_down_exps.weight"),
            vec![inter, hidden * experts],
            T_EXL3_K4,
        ));
        out.push(spec(
            format!("blk.{il}.ffn_down_shexp.weight"),
            vec![inter, hidden],
            T_EXL3_K4,
        ));
    }
    // The streamed n-gram table, which is never made resident: what the load
    // takes from it is its offset, its row count and its codec. Its second
    // dimension is the row count, which has to be what the head vocabulary
    // sizes add up to.
    out.push(spec(
        shiso_models::qwen4exp::NGRAM_TABLE.to_owned(),
        vec![u64::from(PLE_HEAD_DIM), PLE_TABLE_ROWS],
        T_IQ4_NL,
    ));
    out
}

/// Writes the synthetic checkpoint and returns its path.
pub(crate) fn write_fixture(dir: &tempfile::TempDir) -> PathBuf {
    let tensors = fixture_tensors();
    let mut out = Vec::new();
    put_u32(&mut out, MAGIC);
    put_u32(&mut out, FORMAT_VERSION);
    put_u64(&mut out, u64::try_from(tensors.len()).expect("count fits"));
    put_u64(&mut out, 1);

    put_str(&mut out, "general.architecture");
    put_u32(&mut out, 8);
    put_str(&mut out, shiso_models::qwen4exp::ARCH);

    let mut payload = 0_u64;
    let mut table_at = 0_u64;
    for tensor in &tensors {
        put_str(&mut out, &tensor.name);
        put_u32(
            &mut out,
            u32::try_from(tensor.dims.len()).expect("rank fits"),
        );
        for dim in &tensor.dims {
            put_u64(&mut out, *dim);
        }
        put_u32(&mut out, tensor.type_id);
        put_u64(&mut out, payload);
        if tensor.name == shiso_models::qwen4exp::NGRAM_TABLE {
            table_at = payload;
        }
        payload += payload_bytes(tensor.type_id, &tensor.dims);
    }
    while out.len() % ALIGN != 0 {
        out.push(0);
    }

    let path = dir.path().join("fixture.gguf");
    let mut file = std::fs::File::create(&path).expect("the fixture file is created");
    file.write_all(&out).expect("the fixture header is written");
    let len = u64::try_from(out.len()).expect("header fits") + payload;
    file.set_len(len).expect("the fixture payload is sized");
    // Every other tensor is left as the zeros `set_len` gives, because nothing
    // reads their bytes here. The n-gram table is different: it is read off
    // disk by the streaming tier and the rows it returns are what a test checks
    // the device staging against, so a table of zeros could not tell a row that
    // arrived from a row that never did.
    let base = u64::try_from(out.len()).expect("header fits") + table_at;
    file.seek(std::io::SeekFrom::Start(base))
        .expect("the table's offset is inside the fixture");
    let stride = usize::try_from(PLE_ROW_BYTES).expect("a row fits this host");
    let mut rows = vec![0_u8; stride * usize::try_from(PLE_TABLE_ROWS).expect("the table fits")];
    for id in 0..PLE_TABLE_ROWS {
        let at = usize::try_from(id).expect("the table fits") * stride;
        for j in 0..stride {
            rows[at + j] = ple_row_byte(id, j);
        }
    }
    file.write_all(&rows).expect("the table's rows are written");
    file.flush().expect("the fixture is flushed");
    path
}

/// A loaded model over a fresh fake device.
///
/// The temporary directory comes back with it: the mapping the weights were
/// imported from lives as long as the buffers that name it, so a fixture that
/// deleted itself at the end of this function would pull the file out from under
/// the model.
pub(crate) fn loaded() -> (tempfile::TempDir, Arc<FakeBackend>, Qwen4ExpModel) {
    loaded_with(options())
}

/// The same model under settings a caller chose.
///
/// Separate from [`loaded`] rather than an argument to it, because almost every
/// test wants the defaults and a settings argument at every call site would be
/// the same struct spelled out two dozen times.
pub(crate) fn loaded_with(
    opts: SparseOptions,
) -> (tempfile::TempDir, Arc<FakeBackend>, Qwen4ExpModel) {
    let dir = tempfile::TempDir::new().expect("a temporary directory");
    let path = write_fixture(&dir);
    let fake = FakeBackend::new();
    let backend: Arc<dyn Backend> = Arc::<FakeBackend>::clone(&fake);
    let model = Qwen4ExpModel::load(&backend, &path, &tiny_config(), opts)
        .expect("the synthetic checkpoint loads");
    (dir, fake, model)
}

/// The defaults with speculation turned on to `drafts` per step.
pub(crate) fn speculative_options(drafts: u32) -> SparseOptions {
    SparseOptions {
        spec: SpecDepth::new(drafts),
        ..options()
    }
}

/// `count` distinct token ids, which is what makes two sessions' paths differ.
pub(crate) fn tokens(first: u32, count: u32) -> Vec<TokenId> {
    (0..count)
        .map(|i| TokenId::new((first + i) % VOCAB))
        .collect()
}
