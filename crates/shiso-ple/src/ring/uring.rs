//! The ring reader.
//!
//! Same contract as the thread pool - fill the staging buffer with one row per
//! identifier - from a single thread. The reads are the same reads; what
//! changes is what they cost to issue. A registered file and registered bounce
//! buffers mean the kernel neither looks a descriptor up nor pins pages per
//! read, and one enter call carries a whole batch, so the same queue depth
//! costs roughly a quarter of the processor a thread-per-read pool needs for
//! it. The gather is already off the critical path, so that processor time is
//! the entire point.
//!
//! # Short reads are normal, not exceptional
//!
//! A read can come back with less than the block it asked for, so a slot tracks
//! how much of its row has landed and asks for the rest. The completion test is
//! whether the row is covered, not whether the block is: a read that stops
//! short of the block but past the row is the file's last partial block, which
//! is where the table's final rows live.
//!
//! # One ring per stream is enough
//!
//! Every fetch is serialized by the stream's own lock, so the ring is never
//! entered from two threads. The pool stays behind it as the fallback for a
//! kernel that will not give a ring, and for a ring that stops being
//! accountable partway through a batch.

use std::io;
use std::time::Instant;

use io_uring::{IoUring, opcode, types};

use crate::cache::Tally;
use crate::handle::{Counters, RowReader};
use crate::{PleError, PleResult};

/// One outstanding read.
#[derive(Debug, Clone, Copy, Default)]
struct Slot {
    /// Which entry of the batch this serves.
    entry: usize,
    /// Aligned file offset being read.
    start: u64,
    /// Where the row begins inside the block.
    skew: usize,
    /// Bytes the read asked for.
    len: usize,
    /// Bytes received so far.
    done: usize,
}

/// A ring, its registered bounce arena, and the slot bookkeeping over both.
pub(crate) struct Ring {
    /// Declared first so the ring is torn down before the buffers it registered
    /// and before the descriptor those reads target.
    queues: IoUring,
    bounce: Vec<u8>,
    /// Where the aligned arena starts inside `bounce`.
    base: usize,
    /// Bytes per slot.
    stride: usize,
    qd: usize,
    slots: Vec<Slot>,
    /// Stack of idle slots.
    free: Vec<usize>,
    /// Stack of slots waiting for a submission entry.
    pend: Vec<usize>,
    poisoned: bool,
}

impl std::fmt::Debug for Ring {
    /// Names the ring by its geometry.
    ///
    /// The kernel's queues are not `Debug`, and they would say nothing useful
    /// if they were: what a reader of a log wants to know is how deep the ring
    /// is, how wide a slot is, and whether it is still trusted.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ring")
            .field("qd", &self.qd)
            .field("stride", &self.stride)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

impl Ring {
    /// Brings up a ring with its file and bounce buffers registered.
    ///
    /// Every failure path here leaves the stream usable on the pool, so the
    /// caller can warn or insist depending on how the reader was chosen.
    ///
    /// # Errors
    ///
    /// [`PleError::RingUnavailable`] naming the setup step the kernel refused.
    pub(crate) fn open(reader: &RowReader, qd: u32) -> PleResult<Self> {
        let qd = qd as usize;
        let stride = reader.bounce_stride();
        // Direct reads want the buffer aligned as well as the offset and the
        // length. Four kibibytes covers every device this runs on, and a
        // descriptor that asks for more says so.
        let align = if reader.is_direct() {
            reader.align().max(4096)
        } else {
            4096
        };

        let ring = IoUring::new(u32::try_from(qd).map_err(|_| PleError::InvalidParams {
            what: "queue depth",
            value: qd.to_string(),
        })?)
        .map_err(|source| PleError::RingUnavailable {
            op: "queue setup",
            source,
        })?;

        let mut bounce = vec![0u8; qd * stride + align];
        let base = bounce.as_ptr().align_offset(align).min(align);

        let iovecs: Vec<libc::iovec> = (0..qd)
            .map(|i| libc::iovec {
                // SAFETY: `base + i * stride + stride` is at most
                // `align + qd * stride`, which is the length just allocated, so
                // every offset stays inside the one allocation.
                iov_base: unsafe { bounce.as_mut_ptr().add(base + i * stride) }.cast(),
                iov_len: stride,
            })
            .collect();

        // SAFETY: the buffers are the `bounce` allocation this struct owns and
        // never resizes, so they stay valid and at a fixed address for as long
        // as the registration does. `ring` is declared before `bounce`, so the
        // ring - and with it the registration - is dropped first.
        unsafe { ring.submitter().register_buffers(&iovecs) }.map_err(|source| {
            PleError::RingUnavailable {
                op: "buffer registration",
                source,
            }
        })?;
        ring.submitter()
            .register_files(&[reader.raw_fd()])
            .map_err(|source| PleError::RingUnavailable {
                op: "file registration",
                source,
            })?;

        Ok(Self {
            queues: ring,
            bounce,
            base,
            stride,
            qd,
            slots: vec![Slot::default(); qd],
            free: vec![0; qd],
            pend: Vec::with_capacity(qd),
            poisoned: false,
        })
    }

    /// Whether this ring left reads unaccounted for and must not be reused.
    pub(crate) const fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Runs the whole batch through the ring, `qd` reads in flight.
    ///
    /// # Errors
    ///
    /// The first failure the batch hit, as [`PleError::Io`] for a kernel error
    /// or [`PleError::ShortRead`] for a row the file could not cover. An error
    /// stops the batch rather than reading the rest of a list whose result is
    /// about to be discarded; whatever is still outstanding is waited for
    /// first, because those reads write into buffers the unwind is about to
    /// hand back.
    pub(crate) fn read_rows(
        &mut self,
        reader: &RowReader,
        rows: &[u64],
        staging: &mut [u8],
        stats: &Counters,
    ) -> PleResult<()> {
        let started = Instant::now();
        let mut batch = Batch {
            nfree: self.qd,
            ..Batch::default()
        };
        for (i, slot) in self.free.iter_mut().enumerate() {
            *slot = i;
        }
        self.pend.clear();

        while batch.completed < rows.len() && batch.failure.is_none() {
            let submitted = self.issue(reader, rows, &mut batch);
            if submitted == 0 && batch.inflight == 0 {
                batch.failure = Some(PleError::Io(io::Error::other(
                    "the n-gram ring made no progress",
                )));
                break;
            }
            match self.queues.submit_and_wait(usize::from(batch.inflight > 0)) {
                Ok(_) => {}
                Err(err)
                    if matches!(
                        err.kind(),
                        io::ErrorKind::Interrupted | io::ErrorKind::WouldBlock
                    ) => {}
                Err(err) => {
                    batch.failure = Some(PleError::Io(err));
                    break;
                }
            }
            self.reap(rows, staging, reader.row_bytes(), &mut batch);
        }

        let outcome = self.drain(&mut batch);
        stats.merge(Tally {
            // Unlike the pool, which sums each reader's own latency, this is
            // the batch's wall time: with one thread issuing them the two are
            // the same measurement only at a queue depth of one.
            elapsed: started.elapsed(),
            ..batch.tally
        });
        outcome
    }

    /// Hands idle slots the next rows and submits as many as the queue takes.
    ///
    /// Returns how many entries went in. Zero with nothing in flight means the
    /// ring is stuck, which the caller turns into a failure rather than a spin.
    fn issue(&mut self, reader: &RowReader, rows: &[u64], batch: &mut Batch) -> usize {
        while batch.issued < rows.len() && batch.nfree > 0 {
            batch.nfree -= 1;
            let si = self.free[batch.nfree];
            let geom = reader.geometry(rows[batch.issued]);
            self.slots[si] = Slot {
                entry: batch.issued,
                start: geom.start,
                skew: geom.skew,
                len: geom.len,
                done: 0,
            };
            batch.issued += 1;
            self.pend.push(si);
        }

        let (base, stride) = (self.base, self.stride);
        let buf = self.bounce.as_mut_ptr();
        let mut submitted = 0usize;
        let mut sq = self.queues.submission();
        while let Some(&si) = self.pend.last() {
            let slot = self.slots[si];
            let at = base + si * stride + slot.done;
            let entry = opcode::ReadFixed::new(
                types::Fixed(0),
                // SAFETY: `at + (len - done)` is at most `base + (si + 1) *
                // stride`, which is inside the arena allocated at open; the
                // kernel writes only the slot this batch handed to this read,
                // and no other read holds that slot.
                unsafe { buf.add(at) },
                u32::try_from(slot.len - slot.done).unwrap_or(u32::MAX),
                u16::try_from(si).unwrap_or(u16::MAX),
            )
            .offset(slot.start + slot.done as u64)
            .build()
            .user_data(si as u64);
            // SAFETY: the entry names a registered buffer index and the
            // registered file, both of which outlive the read - the ring is
            // drained before `read_rows` returns, and the arena is owned by
            // this struct. No other submission queue exists: the stream
            // serializes fetches.
            if unsafe { sq.push(&entry) }.is_err() {
                break; // the queue is full; reap before pushing more
            }
            self.pend.pop();
            batch.inflight += 1;
            submitted += 1;
        }
        submitted
    }

    /// Copies out every read that has landed, requeueing the short ones.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the user data is a slot index this code wrote, and the result is cast only on the branch that has already established it is not negative"
    )]
    fn reap(&mut self, rows: &[u64], staging: &mut [u8], row_bytes: usize, batch: &mut Batch) {
        let (base, stride) = (self.base, self.stride);
        let buf = self.bounce.as_ptr();
        for cqe in self.queues.completion() {
            let si = cqe.user_data() as usize;
            batch.inflight -= 1;
            batch.tally.reads += 1;
            let result = cqe.result();
            if result < 0 {
                if batch.failure.is_none() {
                    batch.failure = Some(PleError::Io(io::Error::from_raw_os_error(-result)));
                }
                self.free[batch.nfree] = si;
                batch.nfree += 1;
                continue;
            }
            let got = result as usize;
            batch.tally.bytes += got as u64;
            let slot = &mut self.slots[si];
            slot.done += got;
            // The block asked for always covers the row, so covering the row is
            // the only completion test needed.
            if slot.done >= slot.skew + row_bytes {
                let from = base + si * stride + slot.skew;
                // SAFETY: the read into this slot has completed, so the kernel
                // is no longer writing there, and `from + row_bytes` is inside
                // that slot's own stride of the arena.
                let src = unsafe { std::slice::from_raw_parts(buf.add(from), row_bytes) };
                let at = slot.entry * row_bytes;
                staging[at..at + row_bytes].copy_from_slice(src);
                batch.completed += 1;
                self.free[batch.nfree] = si;
                batch.nfree += 1;
            } else if got == 0 {
                if batch.failure.is_none() {
                    batch.failure = Some(PleError::ShortRead {
                        row: rows[slot.entry],
                        got: (slot.done + slot.skew) as u64,
                        want: row_bytes as u64,
                    });
                }
                self.free[batch.nfree] = si;
                batch.nfree += 1;
            } else {
                self.pend.push(si); // short read: ask for the rest
            }
        }
    }

    /// Waits for whatever is still outstanding, then reports the batch.
    ///
    /// An error can leave the loop with reads in flight, and those write into
    /// the arena and, through the staging buffer, into cache slots the unwind
    /// is about to hand back. So they are waited for. If even that fails the
    /// ring's state is no longer known, and the stream falls back to the pool
    /// rather than to a queue it cannot account for.
    fn drain(&mut self, batch: &mut Batch) -> PleResult<()> {
        while batch.inflight > 0 {
            match self.queues.submit_and_wait(1) {
                Ok(_) => {}
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(err) => {
                    self.poisoned = true;
                    return Err(batch.failure.take().unwrap_or(PleError::Io(err)));
                }
            }
            for _ in self.queues.completion() {
                batch.inflight -= 1;
                batch.tally.reads += 1;
            }
        }
        match batch.failure.take() {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

/// One batch's position in the ring.
///
/// A struct rather than six locals because three methods advance it together,
/// and six `&mut` parameters is not a signature anyone can read.
#[derive(Debug, Default)]
struct Batch {
    /// Idle slots on the free stack.
    nfree: usize,
    /// Rows handed to a slot so far.
    issued: usize,
    /// Rows copied out so far.
    completed: usize,
    /// Reads the kernel has not answered yet.
    inflight: usize,
    /// The first failure, kept so the drain still runs.
    failure: Option<PleError>,
    tally: Tally,
}
