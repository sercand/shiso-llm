//! The routed expert stream's bookkeeping, on the host device: which experts a
//! fetch reads, where the map points, and what the pool holds afterwards.

#![expect(
    clippy::cast_possible_truncation,
    reason = "fixture sizes and ids are a few dozen; the byte pattern wraps on purpose"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use shiso_core::{
    Backend, CoreError, CoreResult, DeviceView, KernelRef, LaunchArgs, RawBuffer, alloc,
};
use shiso_cpu::{CpuBackend, CpuOptions};
use shiso_engine::moe_biased::{ExpertStack, RowIq4NlStack, StackStore, check_streams};
use shiso_engine::sparse::weights::WeightPart;
use shiso_engine::sparse::{ExpertSource, ExpertStream};

/// Bytes one expert's first and second matrix occupy.
const LAYOUT: [u64; 2] = [8, 12];

/// A two-matrix source whose bytes name their layer, expert, array and offset,
/// and which fails every read of one expert when told to.
#[derive(Debug)]
struct FakeSource {
    failing: Option<u32>,
}

fn byte(layer: u32, expert: u32, array: usize, i: usize) -> u8 {
    (layer * 64 + expert * 4 + array as u32 * 2 + i as u32) as u8
}

impl ExpertSource for FakeSource {
    fn parts(&self) -> &'static [WeightPart] {
        &[WeightPart::Whole]
    }

    fn matrices(&self) -> usize {
        2
    }

    fn layout(&self) -> Vec<u64> {
        LAYOUT.to_vec()
    }

    fn read(&self, layer: u32, expert: u32, array: usize, dst: &mut [u8]) -> CoreResult<()> {
        if self.failing == Some(expert) {
            return Err(CoreError::Io(std::io::Error::other("the disk went away")));
        }
        for (i, b) in dst.iter_mut().enumerate() {
            *b = byte(layer, expert, array, i);
        }
        Ok(())
    }
}

fn backend() -> Arc<dyn Backend> {
    CpuBackend::new(CpuOptions::default()).expect("the CPU device always opens")
}

fn ids(backend: &Arc<dyn Backend>, values: &[i32]) -> shiso_core::DeviceBuffer {
    let buf = alloc(backend, (values.len() * 4) as u64).expect("alloc");
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    backend.copy_h2d(buf.view(), &bytes).expect("h2d");
    buf
}

fn map_of(backend: &Arc<dyn Backend>, stream: &ExpertStream, layer: u32) -> Vec<u32> {
    let view = stream.map(layer).expect("map");
    let mut raw = vec![0u8; view.len() as usize];
    backend.copy_d2h(view, &mut raw).expect("d2h");
    raw.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn slot_bytes(
    backend: &Arc<dyn Backend>,
    stream: &ExpertStream,
    matrix: usize,
    slot: u32,
) -> Vec<u8> {
    let bytes = stream.slot_layout()[matrix];
    let pool = stream.pool(matrix, WeightPart::Whole).expect("pool");
    let view = pool.slice(u64::from(slot) * bytes, bytes).expect("slice");
    let mut raw = vec![0u8; bytes as usize];
    backend.copy_d2h(view, &mut raw).expect("d2h");
    raw
}

#[test]
fn stream_with_two_matrices_fetches_and_maps() {
    let backend = backend();
    let stream = ExpertStream::open(
        &backend,
        Arc::new(FakeSource { failing: None }),
        4,
        16,
        8,
        6,
    )
    .expect("open");
    assert_eq!(stream.slot_layout(), &LAYOUT);
    let routed = ids(&backend, &[3, 5, 3]);
    stream
        .fetch(backend.as_ref(), 1, routed.view(), 3)
        .expect("fetch");
    assert_eq!(stream.stats().reads, 2);
    let map = map_of(&backend, &stream, 1);
    assert_ne!(map[3], map[5]);
    for expert in [3u32, 5] {
        for (matrix, &bytes) in LAYOUT.iter().enumerate() {
            let want: Vec<u8> = (0..bytes as usize)
                .map(|i| byte(1, expert, matrix, i))
                .collect();
            assert_eq!(
                slot_bytes(&backend, &stream, matrix, map[expert as usize]),
                want
            );
        }
    }
}

#[test]
fn failed_read_leaves_map_consistent() {
    let backend = backend();
    let stream = ExpertStream::open(
        &backend,
        Arc::new(FakeSource { failing: Some(5) }),
        4,
        16,
        8,
        6,
    )
    .expect("open");
    let five = ids(&backend, &[5]);
    let err = stream
        .fetch(backend.as_ref(), 1, five.view(), 1)
        .expect_err("expert 5 cannot be read");
    assert!(matches!(err, CoreError::Io(_)), "{err:?}");
    // Neither map may claim expert 5 is in a slot: a stage would multiply by
    // whatever the slot last held.
    assert_eq!(map_of(&backend, &stream, 1)[5], u32::MAX);
    let again = stream
        .fetch(backend.as_ref(), 1, five.view(), 1)
        .expect_err("a second fetch reads again rather than taking a hit");
    assert!(matches!(again, CoreError::Io(_)), "{again:?}");
    assert_eq!(stream.stats().hits, 0);
    let three = ids(&backend, &[3]);
    stream
        .fetch(backend.as_ref(), 1, three.view(), 1)
        .expect("a readable expert still fetches");
    let map = map_of(&backend, &stream, 1);
    assert_ne!(map[3], u32::MAX);
    assert_eq!(map[5], u32::MAX);
}

#[test]
fn prefetched_experts_are_used_and_wrong_guesses_cost_nothing() {
    let backend = backend();
    let stream = ExpertStream::open(
        &backend,
        Arc::new(FakeSource { failing: None }),
        4,
        16,
        8,
        6,
    )
    .expect("open");
    stream.prefetch(2, &[1, 2]);
    let routed = ids(&backend, &[2, 7]);
    stream
        .fetch(backend.as_ref(), 2, routed.view(), 2)
        .expect("fetch");
    let stats = stream.stats();
    // Expert 2 came from the staged read; only 7 was read on the fetch path.
    assert_eq!(stats.reads, 1);
    assert_eq!(stats.prefetched, 1);
    assert_eq!(stats.prefetch_reads, 2);
    let map = map_of(&backend, &stream, 2);
    assert_eq!(map[1], u32::MAX, "an unused guess is never mapped");
    for expert in [2u32, 7] {
        for (matrix, &bytes) in LAYOUT.iter().enumerate() {
            let want: Vec<u8> = (0..bytes as usize)
                .map(|i| byte(2, expert, matrix, i))
                .collect();
            assert_eq!(
                slot_bytes(&backend, &stream, matrix, map[expert as usize]),
                want
            );
        }
    }
    // A guess for a layer whose fetch already ran is dropped, not kept: the
    // next fetch of that layer reads 1 itself.
    let one = ids(&backend, &[1]);
    stream
        .fetch(backend.as_ref(), 2, one.view(), 1)
        .expect("fetch");
    assert_eq!(stream.stats().reads, 2);
}

#[test]
fn a_pool_full_of_read_aheads_still_serves_the_next_fetch() {
    let backend = backend();
    let stream = ExpertStream::open(
        &backend,
        Arc::new(FakeSource { failing: None }),
        4,
        16,
        8,
        6,
    )
    .expect("open");
    // Read-aheads reserve every slot; only expert 1 of them is then wanted.
    stream.prefetch(2, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let routed = ids(&backend, &[1, 9]);
    stream
        .fetch(backend.as_ref(), 2, routed.view(), 2)
        .expect("the unwanted read-aheads give their slots back");
    let map = map_of(&backend, &stream, 2);
    for expert in [1u32, 9] {
        assert_ne!(map[expert as usize], u32::MAX, "expert {expert} unmapped");
        assert_eq!(
            slot_bytes(&backend, &stream, 0, map[expert as usize]),
            (0..LAYOUT[0] as usize)
                .map(|i| byte(2, expert, 0, i))
                .collect::<Vec<_>>()
        );
    }
}

/// A source whose first read of one expert fails - or panics - and whose
/// every other read answers as [`FakeSource`] does: a disk that hiccups once.
#[derive(Debug)]
struct FlakySource {
    expert: u32,
    panics: bool,
    fired: AtomicBool,
}

impl FlakySource {
    fn new(expert: u32, panics: bool) -> Self {
        Self {
            expert,
            panics,
            fired: AtomicBool::new(false),
        }
    }
}

impl ExpertSource for FlakySource {
    fn parts(&self) -> &'static [WeightPart] {
        &[WeightPart::Whole]
    }

    fn matrices(&self) -> usize {
        2
    }

    fn layout(&self) -> Vec<u64> {
        LAYOUT.to_vec()
    }

    fn read(&self, layer: u32, expert: u32, array: usize, dst: &mut [u8]) -> CoreResult<()> {
        if expert == self.expert && !self.fired.swap(true, Ordering::SeqCst) {
            assert!(!self.panics, "the reader fell over");
            return Err(CoreError::Io(std::io::Error::other("the disk hiccuped")));
        }
        FakeSource { failing: None }.read(layer, expert, array, dst)
    }
}

/// The host device, except that a plain [`Backend::write_host`] fails while
/// `failing` is set: what a device whose map write is refused looks like.
/// Live writes - the reads into slots - still go through.
#[derive(Debug)]
struct RefusingBackend {
    inner: Arc<dyn Backend>,
    failing: AtomicBool,
}

impl Backend for RefusingBackend {
    fn backend_name(&self) -> &'static str {
        "refusing"
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

    fn write_host(
        &self,
        dsts: &[DeviceView<'_>],
        fill: &mut dyn FnMut(&mut [&mut [u8]]) -> CoreResult<()>,
    ) -> CoreResult<()> {
        if self.failing.load(Ordering::SeqCst) {
            return Err(CoreError::Backend {
                backend: "refusing",
                op: "write_host",
                source: "the device refused the write".into(),
            });
        }
        self.inner.write_host(dsts, fill)
    }

    fn write_host_live(
        &self,
        dsts: &[DeviceView<'_>],
        fill: &mut dyn FnMut(&mut [&mut [u8]]) -> CoreResult<()>,
    ) -> CoreResult<()> {
        self.inner.write_host_live(dsts, fill)
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
}

/// Asserts that `layer`'s map sends each of `experts` to a slot holding that
/// expert's own bytes.
fn assert_resident(backend: &Arc<dyn Backend>, stream: &ExpertStream, layer: u32, experts: &[u32]) {
    let map = map_of(backend, stream, layer);
    for &expert in experts {
        assert_ne!(map[expert as usize], u32::MAX, "expert {expert} unmapped");
        for (matrix, &bytes) in LAYOUT.iter().enumerate() {
            let want: Vec<u8> = (0..bytes as usize)
                .map(|i| byte(layer, expert, matrix, i))
                .collect();
            assert_eq!(
                slot_bytes(backend, stream, matrix, map[expert as usize]),
                want,
                "expert {expert} matrix {matrix}"
            );
        }
    }
}

#[test]
fn a_failed_read_ahead_is_read_again_by_its_fetch() {
    let backend = backend();
    let stream = ExpertStream::open(&backend, Arc::new(FlakySource::new(2, false)), 4, 16, 1, 1)
        .expect("open");
    // The read-ahead holds the pool's only slot and fails; the fetch must give
    // the slot back and read expert 2 into it itself.
    stream.prefetch(2, &[2]);
    let two = ids(&backend, &[2]);
    stream
        .fetch(backend.as_ref(), 2, two.view(), 1)
        .expect("the fetch reads what the read-ahead could not");
    let stats = stream.stats();
    assert_eq!(stats.prefetch_reads, 1);
    assert_eq!(stats.prefetched, 0, "a failed read-ahead is never taken");
    assert_eq!(stats.reads, 1);
    assert_resident(&backend, &stream, 2, &[2]);
    // And it is a hit from then on.
    stream
        .fetch(backend.as_ref(), 2, two.view(), 1)
        .expect("fetch");
    assert_eq!(stream.stats().hits, 1);
}

#[test]
fn a_panicking_read_ahead_settles_its_landing() {
    let backend = backend();
    let stream = Arc::new(
        ExpertStream::open(&backend, Arc::new(FlakySource::new(2, true)), 4, 16, 8, 6)
            .expect("open"),
    );
    stream.prefetch(2, &[2]);
    // A landing the worker never settles would park the fetch forever; run it
    // on a thread of its own so a regression fails instead of hanging.
    let (done, wait) = mpsc::channel();
    {
        let backend = Arc::clone(&backend);
        let stream = Arc::clone(&stream);
        std::thread::spawn(move || {
            let two = ids(&backend, &[2]);
            let _ = done.send(stream.fetch(backend.as_ref(), 2, two.view(), 1));
        });
    }
    wait.recv_timeout(Duration::from_secs(30))
        .expect("the fetch returns rather than waiting on a dead worker")
        .expect("the fetch reads what the read-ahead could not");
    assert_eq!(stream.stats().prefetched, 0);
    assert_resident(&backend, &stream, 2, &[2]);
}

#[test]
fn a_failed_batch_unmaps_every_expert_and_a_retry_reads_both() {
    let backend = backend();
    let stream = ExpertStream::open(&backend, Arc::new(FlakySource::new(5, false)), 4, 16, 8, 6)
        .expect("open");
    let routed = ids(&backend, &[3, 5]);
    let err = stream
        .fetch(backend.as_ref(), 1, routed.view(), 2)
        .expect_err("expert 5's first read fails");
    assert!(matches!(err, CoreError::Io(_)), "{err:?}");
    // Expert 3 read fine, but it shared the batch: its entry is gone as well,
    // so no stage trusts a slot the failed batch half wrote.
    let map = map_of(&backend, &stream, 1);
    assert_eq!(map[3], u32::MAX);
    assert_eq!(map[5], u32::MAX);
    stream
        .fetch(backend.as_ref(), 1, routed.view(), 2)
        .expect("the retry reads both");
    let stats = stream.stats();
    assert_eq!(stats.hits, 0, "neither expert counted as resident");
    assert_eq!(stats.reads, 2);
    assert_resident(&backend, &stream, 1, &[3, 5]);
}

#[test]
fn a_refused_map_write_leaves_no_expert_claimed() {
    let backend = backend();
    let refusing = Arc::new(RefusingBackend {
        inner: Arc::clone(&backend),
        failing: AtomicBool::new(false),
    });
    let device: Arc<dyn Backend> = refusing.clone();
    let stream = ExpertStream::open(&device, Arc::new(FakeSource { failing: None }), 4, 16, 8, 6)
        .expect("open");
    // Expert 2 lands from a read-ahead; 7 is missing. The device then refuses
    // the map write that publishes the landed one.
    stream.prefetch(2, &[2]);
    let routed = ids(&device, &[2, 7]);
    refusing.failing.store(true, Ordering::SeqCst);
    let err = stream
        .fetch(device.as_ref(), 2, routed.view(), 2)
        .expect_err("the map write is refused");
    assert!(matches!(err, CoreError::Backend { .. }), "{err:?}");
    refusing.failing.store(false, Ordering::SeqCst);
    // Neither expert is on the device map, so neither may count as a hit:
    // the retry has to place both again.
    stream
        .fetch(device.as_ref(), 2, routed.view(), 2)
        .expect("the retry places both");
    assert_eq!(stream.stats().hits, 0);
    assert_resident(&device, &stream, 2, &[2, 7]);
}

fn streamed_stack(stream: &Arc<ExpertStream>, matrix: usize, layer: u32) -> ExpertStack {
    ExpertStack::Iq4Nl(RowIq4NlStack {
        store: StackStore::Streamed {
            stream: Arc::clone(stream),
            matrix,
            layer,
        },
        in_dim: 32,
        out_dim: 1,
    })
}

#[test]
fn a_layers_banks_must_stream_through_one_stream_layer() {
    let backend = backend();
    let open = || {
        Arc::new(
            ExpertStream::open(
                &backend,
                Arc::new(FakeSource { failing: None }),
                4,
                16,
                8,
                6,
            )
            .expect("open"),
        )
    };
    let (one, other) = (open(), open());
    check_streams(&streamed_stack(&one, 0, 2), &streamed_stack(&one, 1, 2))
        .expect("one stream layer serves both banks");
    // The fetch plans through the up bank's stream layer only; a down bank
    // elsewhere would be read through map entries no fetch wrote.
    let err = check_streams(&streamed_stack(&one, 0, 2), &streamed_stack(&one, 1, 3))
        .expect_err("another layer of the same stream");
    assert!(matches!(err, CoreError::ShapeMismatch { .. }), "{err:?}");
    let err = check_streams(&streamed_stack(&one, 0, 2), &streamed_stack(&other, 1, 2))
        .expect_err("another stream of the same shape");
    assert!(matches!(err, CoreError::Unsupported { .. }), "{err:?}");
}
