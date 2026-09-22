//! The reader, against a real file.
//!
//! The fixture mirrors the shape the reference's own I/O test uses: twenty
//! thousand rows at a file offset that is deliberately not block aligned, each
//! byte a function of its row and its position, so a row read from the wrong
//! offset is not merely different but recognisably the wrong row. The bias
//! table is zero because a fetch returns raw quantized bytes and never decodes
//! - that is the codec's job and has its own vectors.
//!
//! Every test here pins the thread-pool reader explicitly. The ring is not
//! available on every host, and a test that took whatever it was given would
//! quietly stop testing anything the day the ring came up.

use std::fs::File;
use std::io::Write;
use std::sync::Arc;

use shiso_core::ids::TokenId;
use shiso_ple::{
    PleBackend, PleConfig, PleError, PleIo, PleParams, PleParamsDesc, PleRowType, PleStream,
};

/// Rows the fixture table holds.
const TABLE_ROWS: u64 = 20_000;
/// Where the table starts, chosen so it is aligned to nothing.
const FILE_OFFSET: u64 = 1234;
/// Heads, and so rows per position.
const HEADS: u32 = 16;

/// The byte a row holds at one position.
fn row_byte(row: u64, j: usize) -> u8 {
    u8::try_from((row.wrapping_mul(31) + j as u64 * 7 + 11) & 0xFF).expect("masked to a byte")
}

/// The fixture parameters: trellis rows, so a row is a hundred and twenty two
/// bytes, with a zero bias table.
fn params() -> PleParams {
    let mut offsets = Vec::with_capacity(HEADS as usize);
    let mut vocab = Vec::with_capacity(HEADS as usize);
    let mut at = 0u64;
    for h in 0..u64::from(HEADS) {
        offsets.push(at);
        let n = 1000 + h * 37;
        vocab.push(n);
        at += n;
    }
    PleParams::new(PleParamsDesc {
        row_type: PleRowType::TrellisK6,
        ngram_size: 3,
        heads_per_ngram: 8,
        head_dim: 160,
        eos_token: TokenId::new(248_044),
        multipliers: vec![
            0x9E37_79B9_7F4A_7C15,
            0xBF58_476D_1CE4_E5B9,
            0x94D0_49BB_1331_11EB,
        ],
        head_offsets: offsets,
        head_vocab: vocab,
        head_bias: Some(Arc::new(vec![0u16; HEADS as usize * 160])),
    })
    .expect("the fixture parameters are valid")
}

/// Writes the table and hands back the file, kept alive by the returned handle.
fn table(rows: u64, row_bytes: usize) -> tempfile::NamedTempFile {
    let mut file = tempfile::NamedTempFile::new().expect("a temporary file");
    file.write_all(&vec![
        0xAB;
        usize::try_from(FILE_OFFSET).expect("the padding fits")
    ])
    .expect("the pre-table padding");
    let mut row = vec![0u8; row_bytes];
    for id in 0..rows {
        for (j, byte) in row.iter_mut().enumerate() {
            *byte = row_byte(id, j);
        }
        file.write_all(&row).expect("a row");
    }
    file.flush().expect("flush");
    file.as_file().sync_all().expect("sync");
    file
}

/// A stream over a freshly written fixture, on the thread-pool reader.
fn open(cache_bytes: u64, workers: u32) -> (tempfile::NamedTempFile, PleStream) {
    let params = params();
    let row_bytes = params.row_bytes() as usize;
    let fixture = table(TABLE_ROWS, row_bytes);
    let file = File::open(fixture.path()).expect("reopen the fixture");
    let stream = PleStream::open(
        file,
        FILE_OFFSET,
        TABLE_ROWS,
        params,
        cache_bytes,
        &PleConfig {
            io: PleIo::Pool,
            workers,
            ..PleConfig::default()
        },
    )
    .expect("the fixture opens");
    (fixture, stream)
}

/// Checks that `out` holds exactly the rows asked for.
fn expect_rows(out: &[u8], rows: &[u64], stride: usize) {
    for (i, &row) in rows.iter().enumerate() {
        for j in 0..stride {
            assert_eq!(
                out[i * stride + j],
                row_byte(row, j),
                "entry {i} is not row {row} at byte {j}"
            );
        }
    }
}

/// The five shapes the reference's own I/O test uses.
fn shapes() -> Vec<(&'static str, Vec<u64>)> {
    vec![
        ("a single row", vec![7]),
        (
            "one decode step",
            (0..9).map(|i| i * 1301 % TABLE_ROWS).collect(),
        ),
        (
            "a step with repeats",
            vec![5, 9, 5, 12, 9, 5, 100, 12, 100, 5, 9, 12],
        ),
        (
            "a prefill chunk",
            (0..2048u64).map(|i| i * 7 % TABLE_ROWS).collect(),
        ),
        (
            "the tail of the table",
            (TABLE_ROWS - 4..TABLE_ROWS).collect(),
        ),
    ]
}

#[test]
fn every_shape_comes_back_byte_exact_on_the_thread_pool() {
    let (_fixture, stream) = open(8 << 20, 64);
    let stride = stream.row_bytes() as usize;
    assert_eq!(
        stream.backend(),
        PleBackend::Pool,
        "the test asked for the pool and must be told it got it"
    );
    for (name, rows) in shapes() {
        let mut out = vec![0u8; rows.len() * stride];
        stream.fetch(&rows, &mut out).unwrap_or_else(|err| {
            panic!("{name}: {err}");
        });
        expect_rows(&out, &rows, stride);
    }
    stream.close().expect("teardown");
}

#[test]
fn a_second_pass_over_a_shape_is_served_from_the_cache() {
    let (_fixture, stream) = open(8 << 20, 64);
    let stride = stream.row_bytes() as usize;
    let rows: Vec<u64> = (0..9).map(|i| i * 1301 % TABLE_ROWS).collect();
    let mut out = vec![0u8; rows.len() * stride];

    stream.fetch(&rows, &mut out).expect("first pass");
    let after_first = stream.stats();
    stream.fetch(&rows, &mut out).expect("second pass");
    let after_second = stream.stats();

    expect_rows(&out, &rows, stride);
    assert!(
        after_second.hits >= after_first.hits + rows.len() as u64,
        "{after_second:?} should have hit every row the second time"
    );
    assert_eq!(
        after_second.reads, after_first.reads,
        "no row was read twice"
    );
}

#[test]
fn repeats_inside_one_batch_are_read_once() {
    let (_fixture, stream) = open(8 << 20, 64);
    let stride = stream.row_bytes() as usize;
    let rows = vec![5u64, 9, 5, 12, 9, 5, 100, 12, 100, 5, 9, 12];
    let mut out = vec![0u8; rows.len() * stride];
    stream.fetch(&rows, &mut out).expect("fetch");
    expect_rows(&out, &rows, stride);

    let stats = stream.stats();
    assert_eq!(stats.lookups, rows.len() as u64);
    assert_eq!(stats.misses, 4, "four distinct rows");
    assert_eq!(stats.reads, 4, "and four reads");
}

#[test]
fn a_stream_with_no_cache_reads_every_entry_every_time() {
    let (_fixture, stream) = open(0, 64);
    assert_eq!(stream.cache_rows(), 0);
    let stride = stream.row_bytes() as usize;
    let rows = vec![3u64, 3, 4];
    let mut out = vec![0u8; rows.len() * stride];
    for _ in 0..2 {
        stream.fetch(&rows, &mut out).expect("fetch");
        expect_rows(&out, &rows, stride);
    }
    let stats = stream.stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.reads, 6);
}

#[test]
fn one_reader_and_many_readers_agree_byte_for_byte() {
    // The pool's work stealing must not reorder rows: entry `i` is row `i`
    // whichever thread happened to take it.
    let stride = params().row_bytes() as usize;
    let rows: Vec<u64> = (0..2048u64).map(|i| i * 13 % TABLE_ROWS).collect();

    let (_a, serial) = open(8 << 20, 1);
    let (_b, parallel) = open(8 << 20, 64);
    let mut one = vec![0u8; rows.len() * stride];
    let mut many = vec![0u8; rows.len() * stride];
    serial.fetch(&rows, &mut one).expect("one reader");
    parallel.fetch(&rows, &mut many).expect("many readers");
    assert_eq!(one, many);
    expect_rows(&one, &rows, stride);
}

#[test]
fn a_row_past_the_end_of_the_table_is_refused_by_name() {
    let (_fixture, stream) = open(8 << 20, 64);
    let stride = stream.row_bytes() as usize;
    let mut out = vec![0u8; 2 * stride];
    let err = stream
        .fetch(&[1, TABLE_ROWS], &mut out)
        .expect_err("the second row is past the table");
    assert!(
        matches!(
            err,
            PleError::RowOutOfRange {
                row: TABLE_ROWS,
                n_rows: TABLE_ROWS
            }
        ),
        "{err:?}"
    );
}

#[test]
fn a_table_that_claims_more_rows_than_the_file_holds_reports_the_short_read() {
    let params = params();
    let stride = params.row_bytes() as usize;
    // The file holds a hundred rows; the stream is told there are two hundred.
    let fixture = table(100, stride);
    let file = File::open(fixture.path()).expect("reopen");
    let stream = PleStream::open(
        file,
        FILE_OFFSET,
        200,
        params,
        1 << 20,
        &PleConfig {
            io: PleIo::Pool,
            ..PleConfig::default()
        },
    )
    .expect("open");
    let mut out = vec![0u8; stride];
    let err = stream
        .fetch(&[150], &mut out)
        .expect_err("row 150 is not there");
    assert!(
        matches!(err, PleError::ShortRead { row: 150, .. }),
        "{err:?}"
    );

    // And the failure left nothing behind: the rows that do exist still work.
    let mut out = vec![0u8; stride];
    stream.fetch(&[99], &mut out).expect("the last real row");
    expect_rows(&out, &[99], stride);
}

#[test]
fn a_prefetch_warms_the_cache_so_the_fetch_that_follows_hits() {
    let (_fixture, stream) = open(8 << 20, 64);
    let stride = stream.row_bytes() as usize;
    let first: Vec<u64> = (0..64u64).map(|i| i * 29 % TABLE_ROWS).collect();
    let second: Vec<u64> = (0..64u64).map(|i| 5000 + i * 31 % TABLE_ROWS).collect();

    stream.prefetch(&first).expect("start the first prefetch");
    // A second prefetch finishes the one in flight before queueing itself, so
    // by the time this returns the first batch has been read.
    stream.prefetch(&second).expect("start the second prefetch");
    let after = stream.stats();
    assert!(
        after.reads >= first.len() as u64,
        "the first prefetch had to finish before the second was accepted: {after:?}"
    );

    let mut out = vec![0u8; first.len() * stride];
    stream.fetch(&first, &mut out).expect("fetch");
    expect_rows(&out, &first, stride);
    let served = stream.stats();
    assert_eq!(
        served.reads, after.reads,
        "the prefetched rows needed no new read"
    );
    assert!(served.hits >= first.len() as u64, "{served:?}");
    stream.close().expect("teardown");
}

#[test]
fn a_prefetch_without_a_cache_is_a_no_op_rather_than_a_wasted_read() {
    let (_fixture, stream) = open(0, 64);
    stream.prefetch(&[1, 2, 3]).expect("accepted");
    assert_eq!(
        stream.stats().reads,
        0,
        "there is nowhere to warm, so nothing was read"
    );
    stream.close().expect("teardown");
}

#[test]
fn closing_a_stream_that_never_prefetched_is_still_a_clean_teardown() {
    let (_fixture, stream) = open(1 << 20, 4);
    stream.close().expect("teardown");
}

#[test]
fn a_queue_depth_or_worker_count_a_reader_cannot_use_is_refused_at_open() {
    let params = params();
    let stride = params.row_bytes() as usize;
    let fixture = table(8, stride);
    for cfg in [
        PleConfig {
            io: PleIo::Pool,
            qd: 0,
            workers: 4,
        },
        PleConfig {
            io: PleIo::Pool,
            qd: 8,
            workers: 0,
        },
        PleConfig {
            io: PleIo::Pool,
            qd: 8,
            workers: 1000,
        },
    ] {
        let file = File::open(fixture.path()).expect("reopen");
        let err = PleStream::open(file, FILE_OFFSET, 8, params.clone(), 0, &cfg)
            .err()
            .unwrap_or_else(|| panic!("{cfg:?} should be refused"));
        assert!(matches!(err, PleError::InvalidParams { .. }), "{err:?}");
    }
}

/// The ring reader, on a host that has one.
///
/// `[INFERENCE]` until this has run: the ring arm is type-checked against the
/// real bindings by cross-compiling, and its logic is transcribed from a
/// reference that is known to work, but no ring has been opened by this code.
#[cfg(target_os = "linux")]
mod ring {
    use super::{FILE_OFFSET, TABLE_ROWS, expect_rows, params, shapes, table};
    use shiso_ple::{PleBackend, PleConfig, PleIo, PleStream};
    use std::fs::File;

    fn open_ring(qd: u32) -> (tempfile::NamedTempFile, PleStream) {
        let params = params();
        let stride = params.row_bytes() as usize;
        let fixture = table(TABLE_ROWS, stride);
        let file = File::open(fixture.path()).expect("reopen");
        let stream = PleStream::open(
            file,
            FILE_OFFSET,
            TABLE_ROWS,
            params,
            8 << 20,
            &PleConfig {
                io: PleIo::Uring,
                qd,
                ..PleConfig::default()
            },
        )
        .expect("the ring opens on a kernel that has one");
        (fixture, stream)
    }

    #[test]
    #[ignore = "needs a Linux host with io_uring; runbook for the gate box"]
    fn every_shape_comes_back_byte_exact_on_the_ring() {
        let (_fixture, stream) = open_ring(256);
        assert_eq!(stream.backend(), PleBackend::Uring, "asked, not assumed");
        let stride = stream.row_bytes() as usize;
        for (name, rows) in shapes() {
            let mut out = vec![0u8; rows.len() * stride];
            stream
                .fetch(&rows, &mut out)
                .unwrap_or_else(|err| panic!("{name}: {err}"));
            expect_rows(&out, &rows, stride);
        }
        stream.close().expect("teardown");
    }

    #[test]
    #[ignore = "needs a Linux host with io_uring; runbook for the gate box"]
    fn the_ring_and_the_pool_read_the_same_bytes_at_every_queue_depth() {
        let rows: Vec<u64> = (0..2048u64).map(|i| i * 11 % TABLE_ROWS).collect();
        let stride = params().row_bytes() as usize;

        let fixture = table(TABLE_ROWS, stride);
        let file = File::open(fixture.path()).expect("reopen");
        let pool = PleStream::open(
            file,
            FILE_OFFSET,
            TABLE_ROWS,
            params(),
            8 << 20,
            &PleConfig {
                io: PleIo::Pool,
                ..PleConfig::default()
            },
        )
        .expect("open");
        let mut want = vec![0u8; rows.len() * stride];
        pool.fetch(&rows, &mut want).expect("pool fetch");

        // A depth of one exercises the requeue path hardest, because every read
        // is issued and reaped on its own.
        for qd in [1u32, 3, 256] {
            let (_f, ring) = open_ring(qd);
            let mut got = vec![0u8; rows.len() * stride];
            ring.fetch(&rows, &mut got).expect("ring fetch");
            assert_eq!(got, want, "queue depth {qd}");
            ring.close().expect("teardown");
        }
    }
}
