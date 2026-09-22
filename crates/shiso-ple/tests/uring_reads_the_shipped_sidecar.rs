//! The Linux `io_uring` reader, against the table a real checkpoint ships.
//!
//! `crates/shiso-ple/src/ring/uring.rs` is compiled out on macOS, and every
//! phase of this work up to now happened there - so the ring arm had never run
//! anywhere, on any fixture, while being the reader the Gemma 4 text path
//! selects by default (`PleIo::Auto` picks it where the platform has one). The
//! rest of this crate's suite runs it on a temporary file it wrote itself. This
//! case runs it on the 1.5 GB table the mlx checkpoint ships, which is the
//! thing that is actually read in anger, and which arrived here with a download
//! rather than being generated on this machine.
//!
//! Two claims, and the second is the one worth having:
//!
//! 1. The sidecar's header describes the file it is the header of. The check is
//!    exactly what should catch a table that does not belong to this
//!    checkpoint, and the constants come from
//!    `shiso_models::gemma4::ple_repack` rather than being written out again
//!    here - a header check whose expected values have drifted from the
//!    writer's is a header check that passes on the wrong file.
//! 2. Every row the ring returns is byte-identical to the same row read with a
//!    plain positional read. `PleIo::Uring` is asked for by name, not `Auto`,
//!    so a platform that quietly fell back to the thread pool fails here rather
//!    than reporting a pass for a reader that did not run.
//!
//! Run: `cargo test -p shiso-ple --test uring_reads_the_shipped_sidecar --
//! --ignored --nocapture`
#![cfg(target_os = "linux")]

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

use shiso_models::gemma4::ple_repack::{
    BITS, GROUP_SIZE, HEADER_BYTES, MAGIC, ROW_BYTES, SIDECAR_FILENAME, SLOTS, VERSION, VOCAB_ROWS,
};
use shiso_ple::{PleBackend, PleConfig, PleError, PleIo, PleStream};

/// The rows this case reads: the first two, one inside the first page-sized
/// run, one well past it, and the last row in the table. The last one is the
/// only row whose read can run past the end of the file, and a direct read has
/// to round its length up to the device's alignment to get it - so it is the
/// row a reader that forgot the tail returns short.
const ROWS: [u64; 5] = [0, 1, 17, 4095, VOCAB_ROWS - 1];

/// The sidecar beside the checkpoint `SHISO_TEST_MODEL_GEMMA4_SIDECAR` names,
/// or `None` with the reason on stderr.
///
/// `None` is "this machine does not have the checkpoint", which is a skip
/// rather than a failure; a checkpoint that is present but missing its sidecar
/// is a failure, because the file is written once beside the checkpoint and its
/// absence means the load path never ran.
#[expect(
    clippy::print_stderr,
    reason = "the reason a checkpoint case did not run belongs on stderr"
)]
fn sidecar() -> Option<PathBuf> {
    let Ok(dir) = std::env::var("SHISO_TEST_MODEL_GEMMA4_SIDECAR") else {
        eprintln!("note: SHISO_TEST_MODEL_GEMMA4_SIDECAR unset, skipping");
        return None;
    };
    let path = PathBuf::from(dir).join(SIDECAR_FILENAME);
    assert!(
        path.is_file(),
        "{} names a checkpoint with no {SIDECAR_FILENAME} beside it",
        path.display(),
    );
    Some(path)
}

/// Bytes of cache that hold `rows` rows, with room for the per-slot bookkeeping
/// the cache adds on top of each row.
///
/// A fetch cannot ask for more distinct rows than the cache has slots: every
/// miss claims a slot and holds it until the read lands, so a batch wider than
/// the cache has nothing left to claim. The stream refuses that with
/// [`PleError::CacheTooSmall`], which the second case below pins.
fn cache_for(stride: u32, rows: usize) -> u64 {
    (u64::from(stride) + 64) * rows as u64
}

/// Opens the shipped sidecar on the ring, by name rather than by `Auto`.
fn open_ring(path: &std::path::Path, stride: u32, cache_bytes: u64) -> PleStream {
    let shard = File::open(path).expect("the sidecar reopens for the stream");
    PleStream::open_direct(
        shard,
        HEADER_BYTES as u64,
        VOCAB_ROWS,
        stride,
        cache_bytes,
        &PleConfig {
            io: PleIo::Uring,
            ..PleConfig::DEFAULT
        },
    )
    .expect("the ring opens over the shipped table")
}

/// Reads and checks the 64-byte header, returning the row stride it declares.
///
/// Every field is compared against the writer's own constant, and the declared
/// geometry against the file's actual length, because the two ways this file
/// can be wrong are a stale format and a table from another checkpoint.
fn header_stride(path: &std::path::Path) -> u32 {
    let file = File::open(path).expect("the sidecar opens");
    let len = file.metadata().expect("the sidecar has a length").len();
    let mut header = [0u8; HEADER_BYTES];
    file.read_exact_at(&mut header, 0)
        .expect("the sidecar is at least a header long");

    assert_eq!(&header[0..4], MAGIC, "sidecar magic");
    let field32 =
        |at: usize| u32::from_le_bytes(header[at..at + 4].try_into().expect("four bytes"));
    let field64 =
        |at: usize| u64::from_le_bytes(header[at..at + 8].try_into().expect("eight bytes"));

    assert_eq!(field32(4), VERSION, "sidecar format version");
    assert_eq!(field64(8), VOCAB_ROWS, "declared row count");
    assert_eq!(field32(16), ROW_BYTES, "declared row stride");
    assert_eq!(field32(20), GROUP_SIZE, "declared quantization group size");
    assert_eq!(field32(24), BITS, "declared quantization bits");
    assert_eq!(field32(28), SLOTS, "declared layer slots per row");
    assert_eq!(
        len,
        HEADER_BYTES as u64 + VOCAB_ROWS * u64::from(ROW_BYTES),
        "the file's length is not the geometry its header declares",
    );
    ROW_BYTES
}

#[test]
#[ignore = "needs the gemma sidecar"]
#[expect(
    clippy::print_stdout,
    reason = "run by hand under --ignored; which reader actually served the rows is the result"
)]
fn the_uring_reader_and_a_plain_read_agree_on_every_row_tried() {
    let Some(path) = sidecar() else { return };
    let stride = header_stride(&path);

    let stream = open_ring(&path, stride, cache_for(stride, ROWS.len()));

    assert_eq!(
        stream.backend(),
        PleBackend::Uring,
        "the case asked for the ring by name; a fall back to the pool would make this a test of \
         the pool wearing the ring's name",
    );
    println!(
        "reader: {}, rows {}, stride {stride}",
        stream.backend().name(),
        stream.n_rows(),
    );

    let width = stride as usize;
    let mut got = vec![0u8; ROWS.len() * width];
    stream.fetch(&ROWS, &mut got).expect("the ring reads");

    let plain = File::open(&path).expect("the sidecar reopens for the plain read");
    let mut want = vec![0u8; width];
    for (index, &row) in ROWS.iter().enumerate() {
        let at = HEADER_BYTES as u64 + row * u64::from(stride);
        plain
            .read_exact_at(&mut want, at)
            .unwrap_or_else(|err| panic!("plain read of row {row} at {at}: {err}"));
        let mine = &got[index * width..(index + 1) * width];
        assert_eq!(
            mine,
            &want[..],
            "row {row} differs between the ring and a plain read at offset {at}",
        );
        assert!(
            want.iter().any(|&b| b != 0),
            "row {row} is entirely zero, so this case would agree with a reader that read nothing",
        );
    }

    let stats = stream.stats();
    assert_eq!(
        stats.lookups,
        ROWS.len() as u64,
        "every row asked for was looked up",
    );
    assert!(
        stats.reads > 0,
        "the reader did at least one read: {stats:?}"
    );
    println!("stats: {stats:?}");
    stream.close().expect("the ring tears down");
}

#[test]
#[ignore = "needs the gemma sidecar"]
fn a_batch_wider_than_the_cache_is_refused_rather_than_spun_on() {
    let Some(path) = sidecar() else { return };
    let stride = header_stride(&path);

    // One slot short of the batch. Every miss claims a slot and holds it until
    // its read lands, so the last row has nothing left to claim and no slot it
    // may evict. Before this was checked, the eviction hand walked an array in
    // which every entry was pending and never came back: one whole processor,
    // no I/O issued, no error, no completion. That is what this case is for,
    // and it is run against the shipped table rather than a fixture because a
    // direct-I/O stream is where the shape was found.
    let stream = open_ring(&path, stride, cache_for(stride, ROWS.len() - 1));
    let mut out = vec![0u8; ROWS.len() * stride as usize];
    let err = stream
        .fetch(&ROWS, &mut out)
        .expect_err("a batch wider than the cache cannot be served");
    assert!(
        matches!(err, PleError::CacheTooSmall { .. }),
        "the refusal must name the cache, not something downstream of it: {err}",
    );
    stream.close().expect("the ring tears down");
}
