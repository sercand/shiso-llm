//! The ownership and object-safety contract, proven against a fake device.
//!
//! `Arc<dyn Backend>` and `Box<dyn Model>` are load-bearing: an engine holds the
//! former and the scheduler thread owns the latter. Object safety is proven here
//! by *constructing* those trait objects rather than by inspection, and the
//! buffer rules that make a use-after-free a compile error are exercised against
//! a backend that counts every free.
//!
//! The same shape of fake is what the device backends and engines reuse for
//! their own tests.

use std::collections::HashMap;
use std::io;
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use shiso_core::{
    Backend, CoreError, CoreResult, DecodeStep, DeviceBuffer, DeviceView, Extent3, GraphBackend,
    GraphCapture, GraphKey, HostBytes, KernelArg, KernelRef, LaunchArgs, Logits, Model, ModelInfo,
    RawBuffer, SessionId, StopTokens, TokenId, alloc, import_host,
};
use shiso_core::{MemLedger, MemTag, alloc_tagged, ledger};

/// The same device as the trait object every engine holds.
fn dyn_backend<B: Backend + 'static>(backend: Arc<B>) -> Arc<dyn Backend> {
    let object: Arc<dyn Backend> = backend;
    object
}

/// Rows a [`FakeModel`] can produce in one decode step.
const MAX_ROWS: u32 = 4;
/// Tokens a [`FakeModel`] session may hold.
const CAPACITY: u32 = 1_024;

/// A device that is a `HashMap` of byte vectors.
///
/// Every counter a test needs is an atomic, so the same fake answers "was this
/// freed", "freed twice" and "how much was copied" without a mock framework.
#[derive(Debug, Default)]
struct FakeBackend {
    allocations: Mutex<HashMap<u64, Vec<u8>>>,
    next_id: AtomicU64,
    allocs: AtomicU64,
    frees: AtomicU64,
    double_frees: AtomicU64,
    launches: AtomicU64,
    syncs: AtomicU64,
    h2d_bytes: AtomicU64,
    d2h_bytes: AtomicU64,
    ledger: Arc<MemLedger>,
}

impl FakeBackend {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn allocs(&self) -> u64 {
        self.allocs.load(Ordering::Relaxed)
    }

    fn frees(&self) -> u64 {
        self.frees.load(Ordering::Relaxed)
    }

    fn double_frees(&self) -> u64 {
        self.double_frees.load(Ordering::Relaxed)
    }

    fn launches(&self) -> u64 {
        self.launches.load(Ordering::Relaxed)
    }

    fn syncs(&self) -> u64 {
        self.syncs.load(Ordering::Relaxed)
    }

    fn live(&self, id: NonZeroU64) -> bool {
        self.allocations.lock().contains_key(&id.get())
    }

    fn bytes(&self, id: NonZeroU64) -> Vec<u8> {
        self.allocations
            .lock()
            .get(&id.get())
            .cloned()
            .unwrap_or_else(|| panic!("no live allocation for id {}", id.get()))
    }

    fn fail(op: &'static str, detail: String) -> CoreError {
        CoreError::Backend {
            backend: "fake",
            op,
            source: Box::new(io::Error::other(detail)),
        }
    }

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
        let mut table = self.allocations.lock();
        let buffer = table
            .get_mut(&view.id().get())
            .ok_or_else(|| Self::fail(op, format!("no live allocation {}", view.id())))?;
        let end = start
            .checked_add(width)
            .ok_or_else(|| CoreError::OutOfBounds {
                offset: view.offset(),
                len,
                buffer_bytes: buffer.len() as u64,
            })?;
        if end > buffer.len() {
            return Err(CoreError::OutOfBounds {
                offset: view.offset(),
                len,
                buffer_bytes: buffer.len() as u64,
            });
        }
        Ok(start..end)
    }
}

impl Backend for FakeBackend {
    fn backend_name(&self) -> &'static str {
        "fake"
    }

    fn ledger(&self) -> Option<&Arc<MemLedger>> {
        Some(&self.ledger)
    }

    fn alloc_raw(&self, bytes: u64) -> CoreResult<RawBuffer> {
        let width = usize::try_from(bytes)
            .map_err(|_| Self::fail("alloc_raw", format!("{bytes} bytes do not fit this host")))?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let id = NonZeroU64::new(id)
            .ok_or_else(|| Self::fail("alloc_raw", "id space exhausted".into()))?;
        self.allocations.lock().insert(id.get(), vec![0u8; width]);
        self.allocs.fetch_add(1, Ordering::Relaxed);
        Ok(RawBuffer::new(id, bytes))
    }

    fn free_raw(&self, raw: RawBuffer) -> CoreResult<()> {
        let freed = self.allocations.lock().remove(&raw.id().get()).is_some();
        if freed {
            self.frees.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        self.double_frees.fetch_add(1, Ordering::Relaxed);
        Err(Self::fail(
            "free_raw",
            format!("{} was already free", raw.id()),
        ))
    }

    fn copy_h2d(&self, dst: DeviceView<'_>, src: &[u8]) -> CoreResult<()> {
        let range = self.range_of(
            dst,
            u64::try_from(src.len()).unwrap_or(u64::MAX),
            "copy_h2d",
        )?;
        self.allocations
            .lock()
            .get_mut(&dst.id().get())
            .expect("range_of checked the id")
            .splice(range, src.iter().copied());
        self.h2d_bytes.fetch_add(
            u64::try_from(src.len()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        Ok(())
    }

    fn copy_d2h(&self, src: DeviceView<'_>, dst: &mut [u8]) -> CoreResult<()> {
        let range = self.range_of(
            src,
            u64::try_from(dst.len()).unwrap_or(u64::MAX),
            "copy_d2h",
        )?;
        let table = self.allocations.lock();
        let buffer = table.get(&src.id().get()).expect("range_of checked the id");
        dst.copy_from_slice(&buffer[range]);
        self.d2h_bytes.fetch_add(
            u64::try_from(dst.len()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
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
            let table = self.allocations.lock();
            table.get(&src.id().get()).expect("range_of checked the id")[source].to_vec()
        };
        let target = self.range_of(dst, dst.len(), "copy_d2d")?;
        self.allocations
            .lock()
            .get_mut(&dst.id().get())
            .expect("checked")
            .splice(target, copied);
        Ok(())
    }

    fn run(&self, _kernel: KernelRef, _args: &LaunchArgs<'_>) -> CoreResult<()> {
        self.launches.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn sync(&self) -> CoreResult<()> {
        self.syncs.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// The same device with the graph capability turned on.
#[derive(Debug, Default)]
struct FakeGraphBackend {
    inner: FakeBackend,
    capturing: Mutex<Vec<u64>>,
    committed: Mutex<Vec<u64>>,
    aborts: AtomicU64,
    finishes: AtomicU64,
    replays: AtomicU64,
}

impl FakeGraphBackend {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn aborts(&self) -> u64 {
        self.aborts.load(Ordering::Relaxed)
    }

    fn finishes(&self) -> u64 {
        self.finishes.load(Ordering::Relaxed)
    }

    fn is_capturing(&self, key: GraphKey) -> bool {
        self.capturing.lock().contains(&key.get())
    }
}

impl Backend for FakeGraphBackend {
    fn backend_name(&self) -> &'static str {
        "fake-graph"
    }

    fn alloc_raw(&self, bytes: u64) -> CoreResult<RawBuffer> {
        self.inner.alloc_raw(bytes)
    }

    fn free_raw(&self, raw: RawBuffer) -> CoreResult<()> {
        self.inner.free_raw(raw)
    }

    fn copy_h2d(&self, dst: DeviceView<'_>, src: &[u8]) -> CoreResult<()> {
        self.inner.copy_h2d(dst, src)
    }

    fn copy_d2h(&self, src: DeviceView<'_>, dst: &mut [u8]) -> CoreResult<()> {
        self.inner.copy_d2h(src, dst)
    }

    fn copy_d2d(&self, dst: DeviceView<'_>, src: DeviceView<'_>) -> CoreResult<()> {
        self.inner.copy_d2d(dst, src)
    }

    fn run(&self, kernel: KernelRef, args: &LaunchArgs<'_>) -> CoreResult<()> {
        self.inner.run(kernel, args)
    }

    fn sync(&self) -> CoreResult<()> {
        self.inner.sync()
    }

    fn as_graph(&self) -> Option<&dyn GraphBackend> {
        Some(self)
    }
}

impl GraphBackend for FakeGraphBackend {
    fn capture(&self, key: GraphKey) -> CoreResult<GraphCapture<'_>> {
        self.capturing.lock().push(key.get());
        Ok(GraphCapture::started(self, key))
    }

    fn finish_capture(&self, key: GraphKey) -> CoreResult<()> {
        self.capturing.lock().retain(|slot| *slot != key.get());
        self.committed.lock().push(key.get());
        self.finishes.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn abort_capture(&self, key: GraphKey) -> CoreResult<()> {
        self.capturing.lock().retain(|slot| *slot != key.get());
        self.aborts.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn replay(&self, key: GraphKey) -> CoreResult<bool> {
        if self.committed.lock().contains(&key.get()) {
            self.replays.fetch_add(1, Ordering::Relaxed);
            return Ok(true);
        }
        Ok(false)
    }

    fn invalidate_all(&self) {
        self.committed.lock().clear();
        self.capturing.lock().clear();
    }
}

/// A model whose "forward pass" writes each step's token into its own row.
///
/// Enough of a `Model` to exercise the lifecycle the trait promises: sessions
/// minted from a counter, capacity enforced, rows returned in request order, and
/// an explicit teardown that a `Drop` cannot express.
#[derive(Debug)]
struct FakeModel {
    info: ModelInfo,
    backend: Arc<FakeBackend>,
    logits: Option<DeviceBuffer>,
    row_bytes: u64,
    next_session: NonZeroU64,
    positions: HashMap<SessionId, u32>,
    torn_down: bool,
}

impl FakeModel {
    fn new() -> Self {
        Self::with_backend(FakeBackend::new())
    }

    fn with_backend(backend: Arc<FakeBackend>) -> Self {
        // Numbers a scheduler would read, and nothing that says which model they
        // came from: that is the whole of what the trait promises about geometry.
        let info = ModelInfo {
            context_capacity: CAPACITY,
            vocab: 248_320,
            hidden: 2560,
            n_layers: 48,
            stop_tokens: StopTokens::one(TokenId::new(248_044)),
        };
        let row_bytes = info.logits_row_bytes();
        let device: Arc<dyn Backend> = dyn_backend(Arc::clone(&backend));
        let logits = alloc(&device, row_bytes * u64::from(MAX_ROWS)).expect("logits scratch");
        Self {
            info,
            backend,
            logits: Some(logits),
            row_bytes,
            next_session: NonZeroU64::MIN,
            positions: HashMap::new(),
            torn_down: false,
        }
    }

    fn backend(&self) -> &Arc<FakeBackend> {
        &self.backend
    }

    fn advance(&mut self, session: SessionId, tokens: usize) -> CoreResult<()> {
        let used = self
            .positions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        let next = u32::try_from(tokens)
            .unwrap_or(u32::MAX)
            .saturating_add(*used);
        if next > CAPACITY {
            return Err(CoreError::ContextOverflow {
                session,
                cap: CAPACITY,
            });
        }
        *used = next;
        Ok(())
    }

    /// The scratch block, until teardown releases it.
    fn scratch(&self) -> CoreResult<&DeviceBuffer> {
        self.logits.as_ref().ok_or(CoreError::Backend {
            backend: "fake",
            op: "scratch",
            source: Box::new(io::Error::other("teardown already released the scratch")),
        })
    }

    fn is_torn_down(&self) -> bool {
        self.torn_down
    }

    /// Writes `token` into the low bytes of `row` so a test can read it back.
    fn write_row(&mut self, row: u32, token: TokenId) -> CoreResult<()> {
        let offset = u64::from(row) * self.row_bytes;
        let view = self.scratch()?.slice(offset, 4)?;
        self.backend.copy_h2d(view, &token.get().to_le_bytes())
    }

    fn logits_of(&self, rows: u32) -> CoreResult<Logits<'_>> {
        let view = self.scratch()?.slice(0, u64::from(rows) * self.row_bytes)?;
        Logits::new(view, rows, self.info.vocab)
    }
}

impl Default for FakeModel {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for FakeModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn open_session(&mut self) -> CoreResult<SessionId> {
        let session = SessionId::next(&mut self.next_session).ok_or(CoreError::Backend {
            backend: "fake",
            op: "open_session",
            source: Box::new(io::Error::other("session id space exhausted")),
        })?;
        self.positions.insert(session, 0);
        Ok(session)
    }

    fn close_session(&mut self, session: SessionId) -> CoreResult<()> {
        self.positions
            .remove(&session)
            .map(|_used| ())
            .ok_or(CoreError::UnknownSession(session))
    }

    fn prefill_chunk(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<()> {
        self.advance(session, tokens.len())
    }

    fn prefill_last(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<Logits<'_>> {
        self.advance(session, tokens.len())?;
        let last = tokens.last().copied().unwrap_or_else(|| TokenId::new(0));
        self.write_row(0, last)?;
        self.logits_of(1)
    }

    fn decode_batch(&mut self, steps: &[DecodeStep]) -> CoreResult<Logits<'_>> {
        let rows = u32::try_from(steps.len()).unwrap_or(u32::MAX);
        if rows > MAX_ROWS {
            return Err(CoreError::Unsupported {
                what: "decode batch larger than the scratch block",
                backend: "fake",
            });
        }
        for step in steps {
            if !self.positions.contains_key(&step.session) {
                return Err(CoreError::UnknownSession(step.session));
            }
        }
        for (row, step) in steps.iter().enumerate() {
            let row = u32::try_from(row).expect("at most MAX_ROWS rows");
            self.write_row(row, step.token)?;
        }
        self.logits_of(rows)
    }

    fn teardown(&mut self) -> CoreResult<()> {
        self.torn_down = true;
        self.positions.clear();
        // Releasing the scratch block is what a real model does for every
        // tensor it holds; here that is the one buffer, and dropping it is the
        // observable free.
        let Some(scratch) = self.logits.take() else {
            return Ok(());
        };
        let id = scratch.view().id();
        // Dropping the buffer is the release; this checks the device let go, so
        // a teardown that silently leaked would report instead.
        drop(scratch);
        if self.backend.live(id) {
            return Err(CoreError::Backend {
                backend: "fake",
                op: "teardown",
                source: Box::new(io::Error::other(format!("device still holds {id}"))),
            });
        }
        Ok(())
    }
}

fn token(value: u32) -> TokenId {
    TokenId::new(value)
}

/// Reads the token a fake row was stamped with.
fn row_token(backend: &FakeBackend, logits: &Logits<'_>, row: u32) -> TokenId {
    let view = logits.row(row).expect("row exists");
    let head = view.slice(0, 4).expect("a row is at least four bytes");
    let mut bytes = [0u8; 4];
    backend.copy_d2h(head, &mut bytes).expect("read back");
    TokenId::new(u32::from_le_bytes(bytes))
}

#[test]
fn backend_and_model_trait_objects_compile() {
    let backend: Arc<dyn Backend> = dyn_backend(FakeBackend::new());
    let model: Box<dyn Model> = Box::new(FakeModel::new());
    assert_eq!(backend.backend_name(), "fake");
    assert_eq!(model.info().context_capacity, CAPACITY);
    assert_eq!(model.info().n_layers, 48);
}

#[test]
fn a_plain_backend_reports_no_capabilities() {
    let backend: Arc<dyn Backend> = dyn_backend(FakeBackend::new());
    assert!(backend.as_graph().is_none());
    assert!(backend.as_blas().is_none());
    assert!(backend.as_coop().is_none());
}

#[test]
fn a_graph_backend_reports_graph_and_nothing_else() {
    let backend: Arc<dyn Backend> = dyn_backend(FakeGraphBackend::new());
    assert!(backend.as_graph().is_some());
    assert!(backend.as_blas().is_none());
    assert!(backend.as_coop().is_none());
}

#[test]
fn host_device_host_round_trip_through_a_slice() {
    let backend = FakeBackend::new();
    let device = dyn_backend(Arc::clone(&backend));
    let buffer = alloc(&device, 64).expect("alloc");
    assert_eq!(buffer.len(), 64);
    assert!(!buffer.is_empty());

    let head = buffer.slice(8, 4).expect("in range");
    backend.copy_h2d(head, &[1, 2, 3, 4]).expect("h2d");
    let mut read_back = [0u8; 4];
    backend.copy_d2h(head, &mut read_back).expect("d2h");
    assert_eq!(read_back, [1, 2, 3, 4]);

    // The rest of the buffer is untouched, so the slice really was a sub-range.
    let whole = backend.bytes(buffer.view().id());
    assert_eq!(&whole[8..12], &[1, 2, 3, 4]);
    assert!(whole[..8].iter().all(|&byte| byte == 0));

    let mut via_view = [0u8; 4];
    let view = buffer.view();
    backend
        .copy_d2h(
            view.slice(8, 4).expect("sub-range of a view"),
            &mut via_view,
        )
        .expect("d2h");
    assert_eq!(via_view, [1, 2, 3, 4]);
    assert_eq!(view.len(), 64);
    assert_eq!(view.slice(8, 4).expect("sub-range").offset(), 8);
}

#[test]
fn a_slice_past_the_end_is_out_of_bounds_never_a_clamp() {
    let backend = FakeBackend::new();
    let device = dyn_backend(Arc::clone(&backend));
    let buffer = alloc(&device, 16).expect("alloc");

    for (offset, len) in [(12u64, 8u64), (16, 1), (17, 0), (u64::MAX, 2)] {
        let err = buffer
            .slice(offset, len)
            .expect_err("range runs past the buffer");
        match err {
            CoreError::OutOfBounds {
                offset: got_offset,
                len: got_len,
                buffer_bytes,
            } => {
                assert_eq!(got_offset, offset);
                assert_eq!(got_len, len);
                assert_eq!(buffer_bytes, 16);
            }
            other => panic!("expected OutOfBounds, got {other:?}"),
        }
    }
    assert_eq!(
        buffer.slice(16, 0).expect("empty range at the end").len(),
        0
    );

    let view = buffer.view().slice(4, 4).expect("in range");
    let err = view.slice(2, 4).expect_err("sub-range past the view");
    assert!(
        matches!(
            err,
            CoreError::OutOfBounds {
                buffer_bytes: 4,
                ..
            }
        ),
        "a view bounds against itself, not the buffer: {err:?}"
    );
}

#[test]
fn dropping_a_buffer_frees_it_exactly_once() {
    let backend = FakeBackend::new();
    let device = dyn_backend(Arc::clone(&backend));
    let id = {
        let buffer = alloc(&device, 32).expect("alloc");
        let id = buffer.view().id();
        assert!(backend.live(id), "live while owned");
        assert_eq!(backend.frees(), 0);
        id
    };
    assert_eq!(backend.frees(), 1);
    assert!(!backend.live(id), "the id is gone from the device table");
    assert_eq!(backend.double_frees(), 0);
}

#[test]
fn a_buffer_counts_itself_in_the_ledger_under_its_tag_until_it_drops() {
    let backend = FakeBackend::new();
    let device = dyn_backend(Arc::clone(&backend));
    let pool = alloc_tagged(&device, MemTag::KvAttention, 4096).expect("alloc");
    let weights = {
        let _tag = ledger::scope(MemTag::WeightsExperts);
        let host: Arc<dyn HostBytes> = Arc::new(vec![0u8; 256]);
        import_host(&device, host, 0..256).expect("import")
    };
    let loose = alloc(&device, 16).expect("alloc");
    assert_eq!(weights.tag(), MemTag::WeightsExperts);
    assert_eq!(
        loose.tag(),
        MemTag::Untagged,
        "the scope ended with its guard"
    );
    let book = &backend.ledger;
    assert_eq!(book.live(MemTag::KvAttention), 4096);
    assert_eq!(book.live(MemTag::WeightsExperts), 256);
    assert_eq!(book.live(MemTag::Untagged), 16);
    drop(pool);
    assert_eq!(book.live(MemTag::KvAttention), 0);
    assert_eq!(book.peak(MemTag::KvAttention), 4096);
    drop((weights, loose));
    assert_eq!(book.live(MemTag::WeightsExperts), 0);
    assert_eq!(book.live(MemTag::Untagged), 0);
}

#[test]
fn a_hundred_allocate_drop_cycles_never_double_free() {
    let backend = FakeBackend::new();
    let device = dyn_backend(Arc::clone(&backend));
    for round in 0..100 {
        let buffer = alloc(&device, 8).unwrap_or_else(|err| panic!("round {round}: {err}"));
        assert!(backend.live(buffer.view().id()));
        drop(buffer);
    }
    assert_eq!(backend.allocs(), 100);
    assert_eq!(backend.frees(), 100);
    assert_eq!(backend.double_frees(), 0);
}

#[test]
fn a_failing_free_is_reported_not_silently_ignored() {
    let backend = FakeBackend::new();
    let device = dyn_backend(Arc::clone(&backend));
    let buffer = alloc(&device, 8).expect("alloc");
    let id = buffer.view().id();
    // Pull the allocation out from under the buffer: the free that `Drop`
    // performs now fails, and the only channel left is the trace.
    backend
        .allocations
        .lock()
        .remove(&id.get())
        .expect("the allocation was live");
    drop(buffer);
    assert_eq!(backend.double_frees(), 1);
}

#[test]
fn import_host_keeps_the_mapping_alive_after_the_caller_releases_it() {
    let backend = FakeBackend::new();
    let device = dyn_backend(Arc::clone(&backend));
    let host = Arc::new(vec![0u8, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(Arc::strong_count(&host), 1);

    let pinned = Arc::clone(&host);
    let shared: Arc<dyn HostBytes> = pinned;
    let buffer = import_host(&device, shared, 2..6).expect("import");
    assert_eq!(Arc::strong_count(&host), 2);
    drop(host);

    // The buffer pins the whole mapping, not a copy of the requested range:
    // `range` says which bytes the device wrap covers, while the `Arc` is what
    // keeps the bytes themselves alive.
    let mapping = buffer
        .host()
        .expect("an imported buffer names its host bytes");
    assert_eq!(Arc::strong_count(mapping), 1, "only the buffer pins it now");
    assert_eq!(AsRef::<[u8]>::as_ref(&**mapping), &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(buffer.len(), 4, "the wrap is the requested range");

    let mut read_back = [0u8; 4];
    backend
        .copy_d2h(buffer.view(), &mut read_back)
        .expect("the device still sees the bytes");
    assert_eq!(read_back, [2, 3, 4, 5]);

    drop(buffer);
    assert_eq!(backend.frees(), 1);
}

#[test]
fn import_host_rejects_a_range_past_the_host_bytes() {
    let backend = FakeBackend::new();
    let device = dyn_backend(Arc::clone(&backend));
    let owned = Arc::new(vec![0u8; 8]);
    let host: Arc<dyn HostBytes> = owned;
    let err = import_host(&device, host, 4..16).expect_err("range exceeds the bytes");
    assert!(
        matches!(
            err,
            CoreError::OutOfBounds {
                offset: 4,
                len: 12,
                buffer_bytes: 8
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn device_to_device_copy_requires_equal_lengths() {
    let backend = FakeBackend::new();
    let device = dyn_backend(Arc::clone(&backend));
    let src = alloc(&device, 16).expect("alloc");
    let dst = alloc(&device, 16).expect("alloc");
    backend
        .copy_h2d(src.view(), &[9, 8, 7, 6])
        .expect_err("a short source is not a partial transfer");

    let head = src.slice(0, 4).expect("in range");
    let tail = dst.slice(4, 4).expect("in range");
    backend.copy_d2d(tail, head).expect("equal lengths");
    assert_eq!(&backend.bytes(dst.view().id())[4..8], &[0, 0, 0, 0]);

    let err = backend
        .copy_d2d(dst.slice(0, 8).expect("in range"), head)
        .expect_err("lengths differ");
    match err {
        CoreError::ShapeMismatch {
            field,
            expected,
            got,
        } => {
            assert_eq!(field, "copy_d2d len");
            assert_eq!(expected, 8);
            assert_eq!(got, 4);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

/// A kernel the fake accepts, as a const table entry rather than a name built at
/// launch time.
static KERNEL: KernelRef = KernelRef::new("fake", "probe");

#[test]
fn a_launch_carries_views_and_scalars_in_order() {
    let backend = FakeBackend::new();
    let device = dyn_backend(Arc::clone(&backend));
    let buffer = alloc(&device, 32).expect("alloc");
    let args = [
        KernelArg::U32(7),
        KernelArg::I32(-1),
        KernelArg::U64(64),
        KernelArg::F32(0.5),
        KernelArg::Buffer(buffer.slice(0, 16).expect("in range")),
        KernelArg::Bytes(&[1, 2, 3, 4]),
    ];
    let launch = LaunchArgs {
        grid: Extent3::linear(4),
        block: Extent3::new(32, 1, 1),
        smem_bytes: 1_024,
        args: &args,
    };
    backend.run(KERNEL, &launch).expect("launch");
    backend
        .run(
            KERNEL,
            &LaunchArgs::new(Extent3::linear(1), Extent3::linear(32), &[]),
        )
        .expect("launch");
    backend.sync().expect("sync");
    assert_eq!(backend.launches(), 2);
    assert_eq!(backend.syncs(), 1);
    assert_eq!(Extent3::linear(4).volume(), 4);
    assert_eq!(Extent3::new(2, 3, 4).volume(), 24);
}

#[test]
fn an_uncommitted_capture_aborts_on_drop() {
    let backend = FakeGraphBackend::new();
    let key = GraphKey::new(11);
    {
        let capture = backend.capture(key).expect("capture");
        assert_eq!(capture.key(), key);
        assert!(backend.is_capturing(key));
    }
    assert_eq!(backend.aborts(), 1);
    assert_eq!(backend.finishes(), 0);
    assert!(!backend.is_capturing(key), "the device left capture mode");
    assert!(!backend.replay(key).expect("replay query"));
}

#[test]
fn an_early_return_inside_a_captured_region_still_leaves_capture_mode() {
    let backend = FakeGraphBackend::new();
    let key = GraphKey::new(3);
    let failed: CoreResult<()> = (|| {
        let _capture = backend.capture(key)?;
        Err(CoreError::Backend {
            backend: "fake-graph",
            op: "run",
            source: Box::new(io::Error::other("the kernel was not found")),
        })
    })();
    assert!(failed.is_err());
    assert_eq!(backend.aborts(), 1);
    assert!(!backend.is_capturing(key));
}

#[test]
fn a_finished_capture_replays_and_is_never_aborted() {
    let backend = FakeGraphBackend::new();
    let key = GraphKey::new(5);
    let capture = backend.capture(key).expect("capture");
    capture.finish().expect("finish");
    assert_eq!(backend.finishes(), 1);
    assert_eq!(backend.aborts(), 0);
    assert!(backend.replay(key).expect("replay"));
    backend.invalidate_all();
    assert!(!backend.replay(key).expect("replay after invalidate"));
}

#[test]
fn model_lifecycle_returns_rows_in_request_order() {
    let mut model = FakeModel::new();
    let first = model.open_session().expect("session");
    let second = model.open_session().expect("session");
    assert_ne!(first, second, "ids are minted from a counter");

    // The logits borrow the model, which is the point: the rows cannot be
    // freed while a sampler still holds a view of them. Everything the reads
    // need is taken before the block is produced.
    let prompt = [token(10), token(11), token(12)];
    let backend = Arc::clone(model.backend());
    let vocab = model.info().vocab;
    model.prefill_chunk(first, &prompt[..2]).expect("prefill");
    let logits = model
        .prefill_last(first, &prompt[2..])
        .expect("prefill_last");
    assert_eq!(logits.rows(), 1);
    assert_eq!(logits.vocab(), vocab);
    assert_eq!(row_token(&backend, &logits, 0), token(12));

    let steps = [
        DecodeStep {
            session: first,
            token: token(101),
        },
        DecodeStep {
            session: second,
            token: token(202),
        },
        DecodeStep {
            session: first,
            token: token(203),
        },
    ];
    let batch = model.decode_batch(&steps).expect("decode");
    assert_eq!(batch.rows(), 3);
    for (row, step) in steps.iter().enumerate() {
        let row = u32::try_from(row).expect("three rows");
        assert_eq!(row_token(&backend, &batch, row), step.token);
    }
    let err = batch
        .row(u32::try_from(steps.len()).expect("three rows"))
        .expect_err("past the end");
    assert!(matches!(err, CoreError::OutOfBounds { .. }), "got {err:?}");

    let empty = model
        .decode_batch(&[])
        .expect("an empty batch is not an error");
    assert_eq!(empty.rows(), 0);

    model.close_session(second).expect("close");
    let err = model
        .close_session(second)
        .expect_err("closing twice is not a no-op");
    assert!(
        matches!(err, CoreError::UnknownSession(id) if id == second),
        "got {err:?}"
    );

    let unknown = DecodeStep {
        session: second,
        token: token(1),
    };
    let err = model
        .decode_batch(&[unknown])
        .expect_err("a closed session cannot decode");
    assert!(matches!(err, CoreError::UnknownSession(_)), "got {err:?}");

    model.teardown().expect("teardown");
    assert!(model.is_torn_down());
    assert_eq!(model.backend().frees(), 1, "the scratch block was released");
    assert!(
        matches!(
            model.decode_batch(&[]),
            Err(CoreError::Backend { op: "scratch", .. })
        ),
        "a torn-down model has no device state left"
    );
    model.teardown().expect("teardown is idempotent");
    assert_eq!(model.backend().frees(), 1, "nothing is freed twice");
}

#[test]
fn a_session_at_capacity_overflows_instead_of_growing() {
    let mut model = FakeModel::new();
    let session = model.open_session().expect("session");
    let chunk = vec![token(1); 512];
    model.prefill_chunk(session, &chunk).expect("first half");
    model.prefill_chunk(session, &chunk).expect("second half");
    let err = model
        .prefill_last(session, &chunk[..1])
        .expect_err("one token past the capacity");
    match err {
        CoreError::ContextOverflow { session: id, cap } => {
            assert_eq!(id, session);
            assert_eq!(cap, CAPACITY);
        }
        other => panic!("expected ContextOverflow, got {other:?}"),
    }
    // An id minted by a different counter is well-formed and still unknown:
    // validity comes from the model that minted it, not from the integer.
    let mut stranger_counter = NonZeroU64::new(9_999).expect("nonzero");
    let stranger = SessionId::next(&mut stranger_counter).expect("an id from elsewhere");
    let err = model
        .prefill_chunk(stranger, &chunk)
        .expect_err("this model never handed out that handle");
    assert!(
        matches!(err, CoreError::UnknownSession(id) if id == stranger),
        "got {err:?}"
    );
}

#[test]
fn session_ids_come_only_from_a_counter_and_never_name_zero() {
    let mut counter = NonZeroU64::MIN;
    let first = SessionId::next(&mut counter).expect("first id");
    let second = SessionId::next(&mut counter).expect("second id");
    assert_eq!(first.get(), NonZeroU64::MIN);
    assert_ne!(first, second);
    assert!(first.get().get() > 0, "zero is not a usable handle");

    // At the top of the range minting stops rather than wrapping onto an id
    // that may still name a live session.
    let mut near_the_top = NonZeroU64::new(u64::MAX - 1).expect("nonzero");
    let last = SessionId::next(&mut near_the_top).expect("the last minted id");
    assert_eq!(last.get().get(), u64::MAX - 1);
    let mut at_the_top = NonZeroU64::new(u64::MAX).expect("nonzero");
    assert_eq!(
        SessionId::next(&mut at_the_top),
        None,
        "the id space stops instead of reusing"
    );
    assert_eq!(at_the_top, NonZeroU64::new(u64::MAX).expect("nonzero"));
}

// ---------------------------------------------------------------------------------------
// Error chains
// ---------------------------------------------------------------------------------------

/// A `Display` that stops at the outer variant is the whole reason the chain
/// exists: the variant names the kernel, and the sentence that says why the
/// kernel refused is one level down.
#[test]
fn a_chain_carries_every_cause_and_a_display_carries_none() {
    let inner = CoreError::Backend {
        backend: "metal",
        op: "run",
        source: Box::new(io::Error::other("512 threads is outside 1..=448")),
    };
    let outer = CoreError::Backend {
        backend: "engine",
        op: "kernel_q35_gdn_recurrent",
        source: Box::new(inner),
    };

    assert_eq!(
        outer.to_string(),
        "engine backend failed during kernel_q35_gdn_recurrent",
    );
    assert_eq!(
        outer.chain().to_string(),
        "engine backend failed during kernel_q35_gdn_recurrent: \
         metal backend failed during run: 512 threads is outside 1..=448",
    );
}

/// A variant with no `source` chains to exactly itself, so a log site can format
/// through the chain unconditionally rather than branching on whether the
/// failure it caught happens to have a cause.
#[test]
fn a_chain_over_a_sourceless_error_is_its_own_display() {
    let error = CoreError::ShapeMismatch {
        field: "layer count",
        expected: 32,
        got: 24,
    };
    assert_eq!(error.chain().to_string(), error.to_string());
}

/// A backend that does not describe launch limits says so, rather than
/// inventing a number a caller would then trust.
#[test]
fn a_backend_with_no_opinion_reports_no_block_limit() {
    let backend = FakeBackend::new();
    let device = dyn_backend(Arc::clone(&backend));
    assert_eq!(
        device.max_block_threads(KernelRef::new("qwen35", "anything")),
        None,
    );
}

/// The free function reaches the errors this workspace logs but does not define.
#[test]
fn a_foreign_error_chains_through_the_free_function() {
    let error = io::Error::other("the device refused a 4 byte copy");
    assert_eq!(
        shiso_core::chain_of(&error).to_string(),
        "the device refused a 4 byte copy",
    );
}
