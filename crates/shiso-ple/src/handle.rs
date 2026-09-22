//! The table metadata, the open stream, and the file geometry both readers share.
//!
//! # Why the parameters are validated once, at construction
//!
//! Every number in [`PleParams`] comes out of the checkpoint's metadata, and
//! several of them have to agree with each other: the head offsets have to tile
//! the table exactly, the multipliers have to be odd, the trellis codec is
//! reachable at one row width only. A wrong one does not crash - it produces a
//! row identifier in range for a different head, which decodes to perfectly
//! finite weights. So the agreement is checked once, at the only place a
//! [`PleParams`] can come into existence, and after that the codec and the
//! reader can index without re-deciding whether they are allowed to.
//!
//! # Why the parameters are loaded and never derived
//!
//! The reference builds the multipliers and the per-head vocabulary sizes from
//! a seeded generator and a prime search. Reproducing that bit-exactly would be
//! risk with no upside, so a checkpoint always carries them and this crate
//! always reads them.
//!
//! # Where the file geometry lives
//!
//! One function turns a row identifier into the read to issue, and both
//! readers call it, so their idea of where a row sits cannot drift apart. It
//! lives here beside the descriptor and the alignment it reads, because those
//! are what it depends on.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use shiso_core::ids::TokenId;

use crate::cache::{RowCache, RowSource, Tally};
use crate::codec::{self, BLOCK_ELEMS, MAX_HEADS, MAX_NGRAM, TRELLIS_HEAD_DIM};
use crate::pool::Pool;
use crate::prefetch::Prefetch;
use crate::ring::uring::Ring;
use crate::{PleError, PleResult};

/// The environment knob that forces or refuses the ring reader.
pub(crate) const IO_ENV: &str = "SHISO_PLE_IO";

/// How a row is stored in the file.
///
/// The checkpoint decides, and says so in the table tensor's own type, so this
/// is a parameter rather than a build-time choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PleRowType {
    /// Thirty-two four-bit weights per block, sharing one half-precision scale.
    BlockNl4,
    /// A six-bit tail-biting trellis over the whole row, one half-precision row
    /// scale, and a per-head bias added after decoding.
    TrellisK6,
}

/// The table metadata a checkpoint supplies, before it has been checked.
///
/// A separate type from [`PleParams`] on purpose: this one is what a loader
/// fills in field by field out of metadata it has just parsed, and the
/// validated one is what everything downstream reads. Deliberately not
/// `#[non_exhaustive]` - a model crate has to build one, and a new field here
/// is a number some checkpoint family is not supplying yet, which should stop
/// that crate's compile rather than silently default.
#[derive(Debug, Clone)]
pub struct PleParamsDesc {
    /// How a row is stored.
    pub row_type: PleRowType,
    /// Trailing tokens the hash reads.
    pub ngram_size: u32,
    /// Rows each gram width contributes.
    pub heads_per_ngram: u32,
    /// Weights in one row.
    pub head_dim: u32,
    /// The table's own sequence boundary, which is not the chat end-of-turn
    /// token and is usually far outside the chat vocabulary.
    pub eos_token: TokenId,
    /// One multiplier per gram slot, all odd.
    pub multipliers: Vec<u64>,
    /// Where each head's rows start in the table.
    pub head_offsets: Vec<u64>,
    /// How many rows each head has.
    pub head_vocab: Vec<u64>,
    /// Half-precision bias, head-major, required by the trellis codec and
    /// unused by the other.
    ///
    /// Shared rather than owned because the decode path and the device upload
    /// read the same table and neither should copy it.
    pub head_bias: Option<Arc<Vec<u16>>>,
}

/// Checked table metadata.
///
/// Every accessor below is total: the checks that make it so ran once, in
/// [`PleParams::new`].
#[derive(Debug, Clone)]
pub struct PleParams {
    desc: PleParamsDesc,
    n_heads: u32,
    row_bytes: u32,
}

impl PleParams {
    /// Checks a checkpoint's metadata and freezes it.
    ///
    /// # Errors
    ///
    /// [`PleError::InvalidParams`] naming the first thing that cannot work: a
    /// gram width outside the supported range, a head count the tables cannot
    /// cover, a row width the block codec cannot tile, a trellis table without
    /// its bias or at the wrong width, an even multiplier, a head with no
    /// vocabulary, or head offsets that are not a contiguous prefix sum of the
    /// vocabulary sizes.
    pub fn new(desc: PleParamsDesc) -> PleResult<Self> {
        let bad = |what: &'static str, value: String| PleError::InvalidParams { what, value };

        if desc.ngram_size < 2 || desc.ngram_size > MAX_NGRAM {
            return Err(bad("gram width", desc.ngram_size.to_string()));
        }
        if desc.heads_per_ngram == 0 {
            return Err(bad("heads per gram width", "0".to_owned()));
        }
        let n_heads = (desc.ngram_size - 1) * desc.heads_per_ngram;
        if n_heads == 0 || n_heads > MAX_HEADS {
            return Err(bad("head count", n_heads.to_string()));
        }
        if desc.head_dim == 0 || !desc.head_dim.is_multiple_of(BLOCK_ELEMS) {
            return Err(bad("row width", desc.head_dim.to_string()));
        }
        if desc.multipliers.len() != desc.ngram_size as usize {
            return Err(bad(
                "multiplier count",
                format!(
                    "{} for {} gram slots",
                    desc.multipliers.len(),
                    desc.ngram_size
                ),
            ));
        }
        for (i, &m) in desc.multipliers.iter().enumerate() {
            // The reference builds them as twice something plus one, which is
            // what keeps the hash a bijection on the low bits; an even one
            // collapses half the table onto the other half.
            if m % 2 == 0 {
                return Err(bad("multiplier parity", format!("slot {i} is even")));
            }
        }
        if desc.head_offsets.len() != n_heads as usize || desc.head_vocab.len() != n_heads as usize
        {
            return Err(bad(
                "head table length",
                format!(
                    "{} offsets and {} vocabulary sizes for {n_heads} heads",
                    desc.head_offsets.len(),
                    desc.head_vocab.len()
                ),
            ));
        }
        // The heads must tile the table without gaps or overlap: the identifier
        // this crate computes for a head is only in range if they do.
        let mut rows = 0u64;
        for h in 0..n_heads as usize {
            if desc.head_offsets[h] != rows {
                return Err(bad(
                    "head offsets",
                    format!(
                        "head {h} starts at {} and not at {rows}",
                        desc.head_offsets[h]
                    ),
                ));
            }
            if desc.head_vocab[h] == 0 {
                return Err(bad("head vocabulary", format!("head {h} has no rows")));
            }
            rows = rows.saturating_add(desc.head_vocab[h]);
        }
        if desc.row_type == PleRowType::TrellisK6 {
            if desc.head_dim != TRELLIS_HEAD_DIM {
                return Err(bad(
                    "trellis row width",
                    format!("{} rather than {TRELLIS_HEAD_DIM}", desc.head_dim),
                ));
            }
            let want = n_heads as usize * desc.head_dim as usize;
            match desc.head_bias.as_deref() {
                None => return Err(bad("trellis bias", "absent".to_owned())),
                Some(bias) if bias.len() != want => {
                    return Err(bad(
                        "trellis bias length",
                        format!("{} rather than {want}", bias.len()),
                    ));
                }
                Some(_) => {}
            }
        }

        let mut params = Self {
            desc,
            n_heads,
            row_bytes: 0,
        };
        params.row_bytes = codec::row_bytes(&params);
        Ok(params)
    }

    /// How a row is stored.
    #[must_use]
    pub const fn row_type(&self) -> PleRowType {
        self.desc.row_type
    }

    /// Trailing tokens the hash reads.
    #[must_use]
    pub const fn ngram_size(&self) -> u32 {
        self.desc.ngram_size
    }

    /// Rows each gram width contributes.
    #[must_use]
    pub const fn heads_per_ngram(&self) -> u32 {
        self.desc.heads_per_ngram
    }

    /// Rows one position needs, which is one per hash head.
    ///
    /// Derived rather than stored beside the two numbers it comes from: the
    /// shortest gram width is the token itself, which the ordinary embedding
    /// table already carries, so it contributes no head here.
    #[must_use]
    pub const fn n_heads(&self) -> u32 {
        self.n_heads
    }

    /// Weights in one row.
    #[must_use]
    pub const fn head_dim(&self) -> u32 {
        self.desc.head_dim
    }

    /// The table's own sequence boundary token.
    #[must_use]
    pub const fn eos_token(&self) -> TokenId {
        self.desc.eos_token
    }

    /// Bytes one row occupies in the file.
    #[must_use]
    pub const fn row_bytes(&self) -> u32 {
        self.row_bytes
    }

    /// Rows the whole table holds, summed over the heads.
    #[must_use]
    pub fn total_rows(&self) -> u64 {
        self.desc.head_vocab.iter().copied().sum()
    }

    /// The multiplier for one gram slot.
    #[must_use]
    pub(crate) fn multiplier(&self, slot: u32) -> u64 {
        self.desc.multipliers[slot as usize]
    }

    /// Where one head's rows start.
    #[must_use]
    pub(crate) fn head_offset(&self, head: usize) -> u64 {
        self.desc.head_offsets[head]
    }

    /// How many rows one head has.
    #[must_use]
    pub(crate) fn head_vocab(&self, head: usize) -> u64 {
        self.desc.head_vocab[head]
    }

    /// One head's bias row, when the codec has one.
    #[must_use]
    pub(crate) fn head_bias_row(&self, head: u32) -> Option<&[u16]> {
        let width = self.desc.head_dim as usize;
        let at = head as usize * width;
        self.desc
            .head_bias
            .as_deref()
            .and_then(|bias| bias.get(at..at + width))
    }
}

/// Which reader a caller wants.
///
/// Deliberately not `#[non_exhaustive]`: a caller builds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PleIo {
    /// The ring where the platform has one, the pool otherwise.
    #[default]
    Auto,
    /// The ring, and an error if it cannot be had.
    Uring,
    /// The thread pool, whatever the platform offers.
    Pool,
}

/// Which reader a stream actually ended up with.
///
/// Ask rather than assume: the choice can fall back at open, and again at
/// runtime if the ring stops being accountable, so a test that asserts on the
/// backend it requested is asserting on nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PleBackend {
    /// The `io_uring` reader.
    Uring,
    /// The blocking thread pool.
    Pool,
}

impl PleBackend {
    /// The name this backend logs and reports under.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Uring => "uring",
            Self::Pool => "pool",
        }
    }
}

/// Per-stream reader settings.
///
/// These were process-wide environment knobs in the reference. They are
/// constructor parameters here because they are facts about one table on one
/// device, not about the process: two streams on two devices want two answers,
/// and an environment variable cannot give them.
///
/// Deliberately not `#[non_exhaustive]`: a caller builds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PleConfig {
    /// Which reader to use.
    pub io: PleIo,
    /// Reads the ring keeps in flight.
    pub qd: u32,
    /// Threads the pool may use, the submitting thread included.
    pub workers: u32,
}

impl PleConfig {
    /// The measured defaults.
    ///
    /// A fast device's random-read ceiling needs something over two hundred
    /// reads in flight to reach, which is where the queue depth comes from. The
    /// worker count is where a thread-per-read pool stops paying: past it the
    /// depth is worth buying with a ring instead, because the threads cost
    /// whole cores of kernel time to keep blocked.
    ///
    /// A constant as well as a [`Default`], because a caller composing it into
    /// its own `const` settings cannot call a trait method.
    pub const DEFAULT: Self = Self {
        io: PleIo::Auto,
        qd: 256,
        workers: 64,
    };
}

impl Default for PleConfig {
    /// [`PleConfig::DEFAULT`].
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// What the reader has done since the stream was opened.
///
/// Handed out, never built by a caller, so it is `#[non_exhaustive]`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[non_exhaustive]
pub struct PleStats {
    /// Rows asked for, repeats included.
    pub lookups: u64,
    /// Rows served from the cache.
    pub hits: u64,
    /// Rows that had to be read.
    pub misses: u64,
    /// Positional reads issued, after coalescing.
    pub reads: u64,
    /// Bytes actually asked of the device, block padding included.
    pub read_bytes: u64,
    /// Rows dropped from the cache to make room.
    pub evictions: u64,
    /// Time the reads took.
    ///
    /// The pool sums each reader's own latency; the ring reports the batch's
    /// wall time. The two are the same measurement only at queue depth one,
    /// which is why the backend is reported beside this.
    pub read_seconds: f64,
}

/// One shared counter.
#[derive(Debug, Default)]
pub(crate) struct Counter(AtomicU64);

impl Counter {
    /// Adds to the counter.
    pub(crate) fn bump(&self, by: u64) {
        self.0.fetch_add(by, Ordering::Relaxed);
    }

    /// Reads it.
    fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// The counters behind [`PleStats`].
///
/// Atomic rather than sheltered by the fetch lock, so reading the statistics
/// never waits on a gather that may be halfway through a prefill chunk.
#[derive(Debug, Default)]
pub(crate) struct Counters {
    pub(crate) lookups: Counter,
    pub(crate) hits: Counter,
    pub(crate) misses: Counter,
    pub(crate) reads: Counter,
    pub(crate) read_bytes: Counter,
    pub(crate) evictions: Counter,
    read_nanos: Counter,
}

impl Counters {
    /// Folds one reader's numbers in.
    pub(crate) fn merge(&self, tally: Tally) {
        self.reads.bump(tally.reads);
        self.read_bytes.bump(tally.bytes);
        self.read_nanos
            .bump(u64::try_from(tally.elapsed.as_nanos()).unwrap_or(u64::MAX));
    }

    /// The numbers as a caller sees them.
    pub(crate) fn snapshot(&self) -> PleStats {
        PleStats {
            lookups: self.lookups.get(),
            hits: self.hits.get(),
            misses: self.misses.get(),
            reads: self.reads.get(),
            read_bytes: self.read_bytes.get(),
            evictions: self.evictions.get(),
            read_seconds: Duration::from_nanos(self.read_nanos.get()).as_secs_f64(),
        }
    }
}

/// Where one row's read lands: the block to ask the device for, the row's
/// offset inside it, and how many bytes that is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RowRead {
    /// File offset to read from, aligned when the descriptor demands it.
    pub(crate) start: u64,
    /// Where the row begins inside what comes back.
    pub(crate) skew: usize,
    /// Bytes to ask for.
    pub(crate) len: usize,
}

/// The descriptor, and the arithmetic that turns a row into a read.
#[derive(Debug)]
pub(crate) struct RowReader {
    file: File,
    file_offset: u64,
    row_bytes: usize,
    direct: bool,
    align: usize,
}

impl RowReader {
    /// Takes the descriptor and decides whether reads can bypass the page cache.
    ///
    /// Direct reads keep a table tens of gibibytes wide out of the page cache,
    /// where it would otherwise compete with the resident weights for the same
    /// memory. The probe is a reopen of the same file through the process's own
    /// descriptor table; where that door does not exist the reads are buffered,
    /// which is correct and merely less considerate.
    fn new(shard: File, file_offset: u64, row_bytes: u32) -> Self {
        let (file, direct, align) = match reopen_direct(&shard) {
            Some((direct_file, align)) => (direct_file, true, align),
            None => (shard, false, 1),
        };
        Self {
            file,
            file_offset,
            row_bytes: row_bytes as usize,
            direct,
            align,
        }
    }

    /// Bytes one row occupies.
    pub(crate) const fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// The descriptor, for a reader that registers it with the kernel.
    #[cfg(target_os = "linux")]
    pub(crate) fn raw_fd(&self) -> std::os::fd::RawFd {
        std::os::fd::AsRawFd::as_raw_fd(&self.file)
    }

    /// Whether reads bypass the page cache, and so must be aligned.
    pub(crate) const fn is_direct(&self) -> bool {
        self.direct
    }

    /// The alignment reads and buffers must satisfy.
    pub(crate) const fn align(&self) -> usize {
        self.align
    }

    /// The read one row needs.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the skew is the remainder modulo the alignment, which is a usize, so it fits one"
    )]
    pub(crate) const fn geometry(&self, row: u64) -> RowRead {
        let off = self.file_offset + row * self.row_bytes as u64;
        if !self.is_direct() {
            return RowRead {
                start: off,
                skew: 0,
                len: self.row_bytes,
            };
        }
        let a = self.align();
        let start = off / a as u64 * a as u64;
        let skew = (off - start) as usize;
        RowRead {
            start,
            skew,
            len: (skew + self.row_bytes).div_ceil(a) * a,
        }
    }

    /// The widest block a single row can need.
    ///
    /// A row is not aligned to anything, so it can straddle two blocks; a
    /// reader that sized its bounce buffer at the row's own rounded length
    /// would come up one block short on exactly the rows that straddle.
    #[cfg_attr(
        all(not(target_os = "linux"), not(test)),
        expect(
            dead_code,
            reason = "only the ring arm sizes a shared bounce arena; the pool sizes each reader's own from the geometry"
        )
    )]
    pub(crate) const fn bounce_stride(&self) -> usize {
        if !self.is_direct() {
            return self.row_bytes;
        }
        let a = self.align();
        (a - 1 + self.row_bytes).div_ceil(a) * a
    }

    /// Reads one row into `dst`, through `bounce` when the descriptor is direct.
    ///
    /// # Errors
    ///
    /// [`PleError::Io`] for a failed read and [`PleError::ShortRead`] when the
    /// file ends before the row is covered. A read that stops short of the
    /// block it asked for but past the row it wanted is the file's last partial
    /// block and not an error.
    pub(crate) fn read_row(
        &self,
        row: u64,
        dst: &mut [u8],
        bounce: &mut Vec<u8>,
    ) -> PleResult<Tally> {
        let started = Instant::now();
        let geom = self.geometry(row);
        let mut done = 0usize;
        if self.is_direct() {
            let into = aligned_slice(bounce, geom.len, self.align());
            done = read_at(&self.file, into, geom.start, row, geom.skew, self.row_bytes)?;
            dst[..self.row_bytes].copy_from_slice(&into[geom.skew..geom.skew + self.row_bytes]);
        } else {
            done += read_at(
                &self.file,
                &mut dst[..geom.len],
                geom.start,
                row,
                geom.skew,
                self.row_bytes,
            )?;
        }
        debug_assert!(done <= geom.len);
        Ok(Tally {
            reads: 1,
            bytes: geom.len as u64,
            elapsed: started.elapsed(),
        })
    }
}

/// Fills `into` from `start`, retrying interrupts and partial reads.
fn read_at(
    file: &File,
    into: &mut [u8],
    start: u64,
    row: u64,
    skew: usize,
    row_bytes: usize,
) -> PleResult<usize> {
    let mut done = 0usize;
    while done < into.len() {
        match file.read_at(&mut into[done..], start + done as u64) {
            Ok(0) => {
                // The file's last row can sit inside a partial block, so a read
                // that covered the row and stopped is finished, not broken.
                // "Covered" is measured from the start of the block: the row
                // begins `skew` bytes into what was asked for, so the read has
                // to have reached `skew + row_bytes`, not `row_bytes - skew`.
                // Read the other way round, a direct read that returned nothing
                // at all came back as a covered row - so a row past the end of
                // the table was served out of the bounce buffer instead of
                // refused. It could only happen where the descriptor is direct,
                // which is why it survived: the platform this was written on
                // has no O_DIRECT and takes the `skew == 0` path.
                if done >= skew.saturating_add(row_bytes) {
                    break;
                }
                return Err(PleError::ShortRead {
                    row,
                    got: done.saturating_sub(skew) as u64,
                    want: row_bytes as u64,
                });
            }
            Ok(got) => done += got,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
            Err(err) => return Err(PleError::Io(err)),
        }
    }
    Ok(done)
}

/// A `len`-byte window inside `buf`, starting at an `align`-aligned address.
///
/// Direct reads want the destination aligned as well as the offset and the
/// length, and a `Vec<u8>` is only aligned to one byte. Over-allocating by one
/// alignment and slicing to the first aligned address inside it is the same
/// thing an aligned allocator would do, without leaving this crate's allocator
/// to itself.
pub(crate) fn aligned_slice(buf: &mut Vec<u8>, len: usize, align: usize) -> &mut [u8] {
    let need = len + align;
    if buf.len() < need {
        buf.resize(need, 0);
    }
    let at = buf.as_ptr().align_offset(align).min(align);
    &mut buf[at..at + len]
}

/// Reopens a descriptor with the page cache bypassed.
#[cfg(target_os = "linux")]
fn reopen_direct(shard: &File) -> Option<(File, usize)> {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::OpenOptionsExt;

    let path = format!("/proc/self/fd/{}", shard.as_raw_fd());
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .ok()
        // Four kibibytes covers every device this runs on, and the kernel
        // reports that far less is actually required.
        .map(|file| (file, 4096usize))
}

/// This target has no way to bypass the page cache, so reads are buffered.
#[cfg(not(target_os = "linux"))]
const fn reopen_direct(_shard: &File) -> Option<(File, usize)> {
    None
}

/// The two readers, and which of them a batch goes to.
#[derive(Debug)]
struct Io {
    /// Declared first so it is torn down before the descriptor it registered.
    ring: Option<Ring>,
    pool: Pool,
    reader: RowReader,
}

impl RowSource for Io {
    /// Runs a batch on the ring if one is up, otherwise on the pool.
    ///
    /// A ring that fails in a way that leaves reads unaccounted for is dropped
    /// rather than reused: its queue holds writes into buffers this crate is
    /// about to hand back, and a queue that cannot be accounted for cannot be
    /// reasoned about. The stream stays usable, on the pool.
    fn read_rows(&mut self, rows: &[u64], staging: &mut [u8], stats: &Counters) -> PleResult<()> {
        if let Some(ring) = self.ring.as_mut() {
            let outcome = ring.read_rows(&self.reader, rows, staging, stats);
            if ring.is_poisoned() {
                tracing::warn!(
                    "the n-gram ring reader left reads unaccounted for; falling back to the thread pool"
                );
                self.ring = None;
            }
            return outcome;
        }
        self.pool.read_rows(&self.reader, rows, staging, stats)
    }
}

/// The parts of a stream the prefetch thread shares with its owner.
#[derive(Debug)]
pub(crate) struct Shared {
    /// The hash metadata, for a table whose rows are addressed by one.
    ///
    /// `None` for a direct-addressed table, whose caller already knows which
    /// row it wants. Nothing below reads it: the reader is driven by
    /// [`Shared::row_bytes`], which both kinds of table carry.
    params: Option<PleParams>,
    n_rows: u64,
    /// Bytes one row occupies, which is [`PleParams::row_bytes`] for a hashed
    /// table and the caller's own width for a direct-addressed one.
    row_bytes: u32,
    stats: Counters,
    /// Fetches are serialized: the miss batch is per stream, and the prefetch
    /// thread fetches for a later chunk while the caller fetches for this one.
    guts: Mutex<Guts>,
}

/// Everything one fetch at a time may touch.
#[derive(Debug)]
struct Guts {
    cache: RowCache,
    io: Io,
}

impl Shared {
    /// Copies the raw bytes of `rows` into `out`.
    pub(crate) fn fetch(&self, rows: &[u64], out: &mut [u8]) -> PleResult<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut guts = self.guts.lock();
        let Guts { cache, io } = &mut *guts;
        cache.fetch(rows, out, self.n_rows, io, &self.stats)
    }

    /// Bytes one row occupies.
    pub(crate) const fn row_bytes(&self) -> usize {
        self.row_bytes as usize
    }
}

/// An open n-gram table.
///
/// Owns its descriptor, its cache and its reader; dropping it releases all
/// three. [`PleStream::close`] is the same teardown with the errors reported
/// rather than logged.
#[derive(Debug)]
pub struct PleStream {
    shared: Arc<Shared>,
    prefetch: Mutex<Option<Prefetch>>,
}

impl PleStream {
    /// Opens a table that starts `file_offset` bytes into `shard`.
    ///
    /// The file is taken by value because the stream reads from it for its
    /// whole life. Borrowing it would either push a lifetime through every
    /// consumer or invite a caller to close it underneath a ring that has
    /// registered it.
    ///
    /// `cache_bytes` of zero disables caching and reads every row through.
    ///
    /// # Errors
    ///
    /// [`PleError::InvalidParams`] for a queue depth or worker count outside
    /// what a reader can use, or for an unrecognised `SHISO_PLE_IO`;
    /// [`PleError::BackendRefused`] when the ring was insisted on and cannot be
    /// had. A ring that merely fails to come up when nobody insisted is logged
    /// and the stream opens on the pool.
    pub fn open(
        shard: File,
        file_offset: u64,
        n_rows: u64,
        params: PleParams,
        cache_bytes: u64,
        cfg: &PleConfig,
    ) -> PleResult<Self> {
        let row_bytes = params.row_bytes();
        Self::open_rows(
            shard,
            file_offset,
            n_rows,
            Some(params),
            row_bytes,
            cache_bytes,
            cfg,
        )
    }

    /// Opens a table whose rows are addressed by their own index.
    ///
    /// The other half of [`PleStream::open`]: the same reader, row cache, ring
    /// and thread pool over the same file geometry, with no hash metadata at
    /// all. A per-layer embedding table is addressed by token id, so there is
    /// nothing for the hash to compute and nothing for it to check - `row_bytes`
    /// is the whole of what the reader needs, and the caller supplies it.
    ///
    /// `cache_bytes` of zero disables caching and reads every row through.
    ///
    /// # Errors
    ///
    /// [`PleError::InvalidParams`] for a zero `row_bytes`, for a queue depth or
    /// worker count outside what a reader can use, or for an unrecognised
    /// `SHISO_PLE_IO`; [`PleError::BackendRefused`] when the ring was insisted
    /// on and cannot be had.
    pub fn open_direct(
        shard: File,
        file_offset: u64,
        n_rows: u64,
        row_bytes: u32,
        cache_bytes: u64,
        cfg: &PleConfig,
    ) -> PleResult<Self> {
        if row_bytes == 0 {
            return Err(PleError::InvalidParams {
                what: "row width",
                value: "0".to_owned(),
            });
        }
        Self::open_rows(
            shard,
            file_offset,
            n_rows,
            None,
            row_bytes,
            cache_bytes,
            cfg,
        )
    }

    /// The body both constructors share: the reader, the cache and the readers'
    /// backends, over a row width whichever of them decided it.
    fn open_rows(
        shard: File,
        file_offset: u64,
        n_rows: u64,
        params: Option<PleParams>,
        row_bytes: u32,
        cache_bytes: u64,
        cfg: &PleConfig,
    ) -> PleResult<Self> {
        if cfg.qd == 0 || cfg.qd > 4096 {
            return Err(PleError::InvalidParams {
                what: "queue depth",
                value: cfg.qd.to_string(),
            });
        }
        if cfg.workers == 0 || cfg.workers > 256 {
            return Err(PleError::InvalidParams {
                what: "worker count",
                value: cfg.workers.to_string(),
            });
        }

        let reader = RowReader::new(shard, file_offset, row_bytes);
        let cache = RowCache::new(row_bytes, n_rows, cache_bytes);
        let want = resolve_io(cfg.io)?;

        let mut ring = None;
        if want != PleIo::Pool {
            match Ring::open(&reader, cfg.qd) {
                Ok(opened) => ring = Some(opened),
                Err(err) if want == PleIo::Uring => return Err(err),
                Err(err) => tracing::warn!(
                    %err,
                    workers = cfg.workers,
                    "the n-gram ring reader is unavailable; falling back to the thread pool"
                ),
            }
        }
        let backend = if ring.is_some() {
            PleBackend::Uring
        } else {
            PleBackend::Pool
        };
        tracing::info!(
            backend = backend.name(),
            cache_mib = cache_bytes >> 20,
            cache_rows = cache.slots(),
            rows = n_rows,
            row_bytes,
            "opened the n-gram table"
        );

        Ok(Self {
            shared: Arc::new(Shared {
                params,
                n_rows,
                row_bytes,
                stats: Counters::default(),
                guts: Mutex::new(Guts {
                    cache,
                    io: Io {
                        ring,
                        pool: Pool::new(cfg.workers),
                        reader,
                    },
                }),
            }),
            prefetch: Mutex::new(None),
        })
    }

    /// The checked hash metadata this table was opened with.
    ///
    /// `None` means a direct-addressed table: one opened through
    /// [`PleStream::open_direct`], whose rows are indexed by the caller's own
    /// number rather than by a hash over trailing tokens, and which therefore
    /// carries no head offsets, no multipliers and no per-head vocabulary. A
    /// caller that needs any of those is looking at the wrong kind of table and
    /// should refuse rather than substitute a default.
    #[must_use]
    pub fn params(&self) -> Option<&PleParams> {
        self.shared.params.as_ref()
    }

    /// Bytes one row of this table occupies.
    ///
    /// Carried here rather than reached for through [`PleStream::params`],
    /// because a direct-addressed table has no metadata and still has a row
    /// width: it is what the reader strides the file by.
    #[must_use]
    pub fn row_bytes(&self) -> u32 {
        self.shared.row_bytes
    }

    /// Rows the table holds.
    #[must_use]
    pub fn n_rows(&self) -> u64 {
        self.shared.n_rows
    }

    /// Copies the raw quantized bytes of `rows` into `out`, in order.
    ///
    /// Thread-safe: fetches are serialized, and a row wanted by two callers at
    /// once is read once.
    ///
    /// # Errors
    ///
    /// [`PleError::BufferTooSmall`] when `out` is shorter than one row per
    /// identifier, [`PleError::RowOutOfRange`] for an identifier past the end
    /// of the table, and whatever the reader reports. On any of them the cache
    /// is left exactly as it was.
    pub fn fetch(&self, rows: &[u64], out: &mut [u8]) -> PleResult<()> {
        self.shared.fetch(rows, out)
    }

    /// Warms the cache with `rows` on a thread of its own.
    ///
    /// Returns without waiting; a prefetch still running is finished first, so
    /// two of these in a row cost the first one's remaining time and nothing
    /// else. A no-op without a cache, because there would be nowhere to put the
    /// rows.
    ///
    /// # Errors
    ///
    /// [`PleError::Io`] when the prefetch thread cannot be started. The rows
    /// themselves are read on that thread and their outcome is not reported
    /// here: a prefetch that fails costs a later fetch its hit and nothing
    /// more.
    pub fn prefetch(&self, rows: &[u64]) -> PleResult<()> {
        if rows.is_empty() || self.cache_rows() == 0 {
            return Ok(());
        }
        let mut slot = self.prefetch.lock();
        if slot.is_none() {
            *slot = Some(Prefetch::start(Arc::clone(&self.shared))?);
        }
        if let Some(prefetch) = slot.as_ref() {
            prefetch.submit(rows);
        }
        Ok(())
    }

    /// Which reader the last batch actually used.
    #[must_use]
    pub fn backend(&self) -> PleBackend {
        if self.shared.guts.lock().io.ring.is_some() {
            PleBackend::Uring
        } else {
            PleBackend::Pool
        }
    }

    /// What the reader has done since the stream was opened.
    #[must_use]
    pub fn stats(&self) -> PleStats {
        self.shared.stats.snapshot()
    }

    /// Rows the cache can hold.
    #[must_use]
    pub fn cache_rows(&self) -> u64 {
        self.shared.guts.lock().cache.slots()
    }

    /// Tears the stream down and reports what went wrong.
    ///
    /// The order matters: the prefetch thread is stopped before anything it
    /// fetches into is released. Dropping a stream does the same thing and logs
    /// instead of reporting, so forgetting to call this leaks nothing.
    ///
    /// # Errors
    ///
    /// [`PleError::Io`] when the prefetch thread ended by panicking, which is
    /// the only failure teardown can have: everything else it releases is
    /// memory and a descriptor.
    pub fn close(self) -> PleResult<()> {
        let taken = self.prefetch.lock().take();
        match taken {
            Some(prefetch) => prefetch.stop(),
            None => Ok(()),
        }
    }
}

impl Drop for PleStream {
    /// Stops the prefetch thread before the buffers it writes into go away.
    fn drop(&mut self) {
        let taken = self.prefetch.lock().take();
        if let Some(prefetch) = taken
            && let Err(err) = prefetch.stop()
        {
            tracing::warn!(%err, "the n-gram prefetch thread did not stop cleanly");
        }
    }
}

/// Combines the caller's choice with the process-wide override.
///
/// The environment wins, because it is the operator's knob and its whole point
/// is to force a backend a build would not have chosen. An unrecognised value
/// is refused rather than ignored: silently running the default when an
/// operator asked for something else is how a benchmark measures the wrong
/// reader for a week.
fn resolve_io(requested: PleIo) -> PleResult<PleIo> {
    match std::env::var(IO_ENV) {
        Err(_) => Ok(requested),
        Ok(value) => match value.as_str() {
            "uring" => Ok(PleIo::Uring),
            "pool" => Ok(PleIo::Pool),
            other => Err(PleError::InvalidParams {
                what: IO_ENV,
                value: other.to_owned(),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc() -> PleParamsDesc {
        PleParamsDesc {
            row_type: PleRowType::BlockNl4,
            ngram_size: 3,
            heads_per_ngram: 2,
            head_dim: 64,
            eos_token: TokenId::new(99),
            multipliers: vec![3, 5, 7],
            head_offsets: vec![0, 10, 30, 60],
            head_vocab: vec![10, 20, 30, 40],
            head_bias: None,
        }
    }

    #[test]
    fn the_head_count_is_derived_and_cannot_disagree_with_the_tables() {
        let params = PleParams::new(desc()).expect("valid");
        assert_eq!(params.n_heads(), 4, "two gram widths of two heads each");
        assert_eq!(params.total_rows(), 100);
        assert_eq!(params.row_bytes(), 64 / 32 * 18);
    }

    #[test]
    fn an_even_multiplier_is_refused_because_the_hash_would_stop_being_one() {
        let mut d = desc();
        d.multipliers[1] = 4;
        let err = PleParams::new(d).expect_err("even multiplier");
        assert!(
            matches!(
                err,
                PleError::InvalidParams {
                    what: "multiplier parity",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn head_offsets_that_are_not_a_prefix_sum_of_the_vocabularies_are_refused() {
        let mut d = desc();
        d.head_offsets[2] = 31;
        let err = PleParams::new(d).expect_err("gap between heads");
        assert!(
            matches!(
                err,
                PleError::InvalidParams {
                    what: "head offsets",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_head_with_no_rows_is_refused_because_the_remainder_would_divide_by_zero() {
        let mut d = desc();
        d.head_vocab = vec![10, 0, 30, 40];
        d.head_offsets = vec![0, 10, 10, 40];
        let err = PleParams::new(d).expect_err("empty head");
        assert!(
            matches!(
                err,
                PleError::InvalidParams {
                    what: "head vocabulary",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn the_trellis_codec_needs_its_bias_table_and_its_one_row_width() {
        let mut d = desc();
        d.row_type = PleRowType::TrellisK6;
        d.head_dim = 160;
        let err = PleParams::new(d.clone()).expect_err("no bias");
        assert!(
            matches!(
                err,
                PleError::InvalidParams {
                    what: "trellis bias",
                    ..
                }
            ),
            "{err:?}"
        );
        d.head_bias = Some(Arc::new(vec![0u16; 4 * 160]));
        let params = PleParams::new(d.clone()).expect("bias supplied");
        assert_eq!(params.row_bytes(), 122);

        d.head_dim = 128;
        d.head_bias = Some(Arc::new(vec![0u16; 4 * 128]));
        let err = PleParams::new(d).expect_err("wrong width");
        assert!(
            matches!(
                err,
                PleError::InvalidParams {
                    what: "trellis row width",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_gram_width_outside_the_supported_range_is_refused() {
        for ngram in [0u32, 1, 5] {
            let mut d = desc();
            d.ngram_size = ngram;
            d.multipliers = vec![3; ngram.max(1) as usize];
            assert!(PleParams::new(d).is_err(), "gram width {ngram}");
        }
    }

    #[test]
    fn buffered_row_geometry_is_the_row_itself_at_its_own_offset() {
        let file = tempfile::tempfile().expect("a temporary file");
        let reader = RowReader::new(file, 1234, 122);
        assert!(!reader.is_direct() || cfg!(target_os = "linux"));
        if !reader.is_direct() {
            let geom = reader.geometry(7);
            assert_eq!(geom.start, 1234 + 7 * 122);
            assert_eq!(geom.skew, 0);
            assert_eq!(geom.len, 122);
            assert_eq!(reader.bounce_stride(), 122);
        }
    }

    #[test]
    fn a_direct_addressed_table_reads_rows_by_their_own_index_and_carries_no_hash() {
        use std::io::Write as _;

        // A thousand rows of twenty-four bytes, each filled with a byte derived
        // from its own index: a row read at the wrong offset is a row of the
        // wrong byte, which no tolerance hides.
        const ROWS: u64 = 1000;
        const WIDTH: u32 = 24;

        let mut file = tempfile::NamedTempFile::new().expect("a temporary file");
        for row in 0..ROWS {
            let byte = u8::try_from(row % 251).expect("a remainder below 251 is a byte");
            file.write_all(&vec![byte; WIDTH as usize])
                .expect("a row lands");
        }
        file.flush().expect("flush");
        file.as_file().sync_all().expect("sync");

        let handle = File::open(file.path()).expect("reopen");
        let stream = PleStream::open_direct(
            handle,
            0,
            ROWS,
            WIDTH,
            4096,
            &PleConfig {
                io: PleIo::Pool,
                ..PleConfig::DEFAULT
            },
        )
        .expect("a direct-addressed table opens");

        assert!(
            stream.params().is_none(),
            "a direct-addressed table has no hash metadata"
        );
        assert_eq!(stream.row_bytes(), WIDTH);
        assert_eq!(stream.n_rows(), ROWS);

        let wanted = [999u64, 0, 17, 17];
        let mut out = vec![0u8; wanted.len() * WIDTH as usize];
        stream.fetch(&wanted, &mut out).expect("the rows land");
        for (slot, &row) in wanted.iter().enumerate() {
            let byte = u8::try_from(row % 251).expect("a remainder below 251 is a byte");
            let at = slot * WIDTH as usize;
            assert_eq!(
                out[at..at + WIDTH as usize],
                vec![byte; WIDTH as usize][..],
                "slot {slot} should hold row {row}"
            );
        }
        // The repeat is served out of the cache rather than read again, which is
        // the whole reason a row cache sits in front of the reader.
        assert_eq!(stream.stats().reads, 3, "{:?}", stream.stats());
    }

    #[test]
    fn a_direct_addressed_table_of_no_width_is_refused() {
        let file = tempfile::tempfile().expect("a temporary file");
        let err = PleStream::open_direct(file, 0, 4, 0, 0, &PleConfig::DEFAULT)
            .expect_err("a row of no bytes is not a row");
        assert!(
            matches!(
                err,
                PleError::InvalidParams {
                    what: "row width",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn an_aligned_window_starts_at_an_aligned_address_and_is_the_length_asked_for() {
        let mut buf = Vec::new();
        for align in [512usize, 4096] {
            let slice = aligned_slice(&mut buf, 300, align);
            assert_eq!(slice.len(), 300);
            assert_eq!(slice.as_ptr().addr() % align, 0, "align {align}");
        }
    }
}
