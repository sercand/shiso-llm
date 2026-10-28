//! The routed expert stream's bookkeeping, on the host device: which experts a
//! fetch reads, where the map points, and what the pool holds afterwards.

#![expect(
    clippy::cast_possible_truncation,
    reason = "fixture sizes and ids are a few dozen; the byte pattern wraps on purpose"
)]

use std::sync::Arc;

use shiso_core::{Backend, CoreError, CoreResult, alloc};
use shiso_cpu::{CpuBackend, CpuOptions};
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
