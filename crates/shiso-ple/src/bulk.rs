//! Whole-file reads with several in flight: the other access pattern.
//!
//! The row reader is built for a scattered gather of hundred-byte rows out of
//! one huge file. A cache restore is the opposite shape - a few hundred files
//! of a few mebibytes each, every byte of each wanted - and what it needs from
//! the device is queue depth: one read at a time leaves a solid-state drive idle
//! between requests. This module keeps up to `depth` whole-file reads
//! outstanding through the same kernel interface, and falls back to one
//! blocking read per file where there is no ring.
//!
//! # Buffers are the caller's
//!
//! Each read lands in a `Vec` the caller hands in and gets back filled, so a
//! restore can recycle a handful of page-sized buffers instead of allocating
//! (and faulting in) a fresh one per page.
//!
//! # A failed file does not fail the batch
//!
//! Every file answers for itself. A cache treats one unreadable file as a miss
//! on that entry and wants to know which, so the batch returns one result per
//! file rather than stopping at the first.

use std::fs::File;
use std::io::{self, Read as _};
use std::path::PathBuf;

/// One file to read whole.
#[derive(Debug)]
pub struct FileRead {
    /// The file.
    pub path: PathBuf,
    /// Where its bytes land. Resized to the file's length; its previous
    /// contents are discarded.
    pub buf: Vec<u8>,
    /// How the read went, filled in by [`BulkReader::read`].
    pub result: io::Result<()>,
}

impl FileRead {
    /// A read of `path` into `buf`.
    #[must_use]
    pub fn new(path: PathBuf, buf: Vec<u8>) -> Self {
        Self {
            path,
            buf,
            result: Ok(()),
        }
    }
}

/// Reads batches of whole files, `depth` at a time.
#[derive(Debug)]
pub struct BulkReader {
    #[cfg(target_os = "linux")]
    ring: Option<arm::BulkRing>,
    depth: u32,
}

impl BulkReader {
    /// A reader keeping up to `depth` reads in flight.
    ///
    /// Tries for a ring and quietly keeps the blocking path when the kernel
    /// will not give one: a restore that is slower is still a restore, and
    /// unlike the n-gram reader nobody measures which path ran.
    #[must_use]
    pub fn new(depth: u32) -> Self {
        let depth = depth.max(1);
        Self {
            #[cfg(target_os = "linux")]
            ring: match arm::BulkRing::open(depth) {
                Ok(ring) => Some(ring),
                Err(error) => {
                    tracing::debug!(%error, "no ring for bulk reads; reading one file at a time");
                    None
                }
            },
            depth,
        }
    }

    /// Which path this reader takes: `"uring"` or `"blocking"`.
    #[must_use]
    pub fn backend(&self) -> &'static str {
        #[cfg(target_os = "linux")]
        if self.ring.is_some() {
            return "uring";
        }
        "blocking"
    }

    /// Reads every file in `files` whole, filling each one's `result`.
    pub fn read(&mut self, files: &mut [FileRead]) {
        #[cfg(target_os = "linux")]
        if let Some(mut ring) = self.ring.take() {
            let depth = usize::try_from(self.depth).unwrap_or(1);
            let mut healthy = true;
            for batch in files.chunks_mut(depth) {
                if healthy {
                    match ring.read(batch) {
                        Ok(()) => continue,
                        Err(error) => {
                            // The ring's own bookkeeping failed, not a file: this
                            // batch and every later one take the blocking path,
                            // and so does the rest of this reader's life.
                            tracing::warn!(%error, "the bulk ring failed; reading one file at a time");
                            healthy = false;
                        }
                    }
                }
                for file in batch.iter_mut() {
                    file.result = read_blocking(file);
                }
            }
            if healthy {
                self.ring = Some(ring);
            }
            return;
        }
        for file in files.iter_mut() {
            file.result = read_blocking(file);
        }
    }
}

/// One file, one blocking read.
fn read_blocking(file: &mut FileRead) -> io::Result<()> {
    let mut handle = File::open(&file.path)?;
    let len = usize::try_from(handle.metadata()?.len()).unwrap_or(usize::MAX);
    file.buf.clear();
    file.buf.resize(len, 0);
    handle.read_exact(&mut file.buf)
}

#[cfg(target_os = "linux")]
mod arm {
    //! The ring arm: every file of a batch is opened, sized and submitted at
    //! once, and short reads are resubmitted for the remainder until each file
    //! is covered or fails.

    use std::fs::File;
    use std::io;
    use std::os::fd::AsRawFd as _;

    use io_uring::{IoUring, opcode, types};

    use super::FileRead;

    /// A ring sized for one batch.
    pub(super) struct BulkRing {
        ring: IoUring,
    }

    impl std::fmt::Debug for BulkRing {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("BulkRing").finish_non_exhaustive()
        }
    }

    /// One file's progress through the ring.
    struct Slot {
        file: Option<File>,
        len: usize,
        done: usize,
        inflight: bool,
    }

    impl BulkRing {
        /// A ring with room for `depth` submissions.
        pub(super) fn open(depth: u32) -> io::Result<Self> {
            Ok(Self {
                ring: IoUring::new(depth.max(1))?,
            })
        }

        /// Reads one batch, no larger than the ring's depth.
        ///
        /// An `Err` means the ring itself is in an unknown state; every file's
        /// own failure is reported in its `result` instead.
        pub(super) fn read(&mut self, files: &mut [FileRead]) -> io::Result<()> {
            let mut slots: Vec<Slot> = files
                .iter_mut()
                .map(|file| {
                    let opened = File::open(&file.path).and_then(|handle| {
                        let len = usize::try_from(handle.metadata()?.len()).unwrap_or(usize::MAX);
                        Ok((handle, len))
                    });
                    match opened {
                        Ok((handle, len)) => {
                            file.buf.clear();
                            file.buf.resize(len, 0);
                            file.result = Ok(());
                            Slot {
                                file: Some(handle),
                                len,
                                done: 0,
                                inflight: false,
                            }
                        }
                        Err(error) => {
                            file.result = Err(error);
                            Slot {
                                file: None,
                                len: 0,
                                done: 0,
                                inflight: false,
                            }
                        }
                    }
                })
                .collect();
            let mut inflight = 0usize;
            loop {
                inflight += self.submit(files, &mut slots);
                if inflight == 0 {
                    return Ok(());
                }
                match self.ring.submit_and_wait(1) {
                    Ok(_) => {}
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                    Err(error) => {
                        self.drain(inflight);
                        return Err(error);
                    }
                }
                inflight -= self.reap(files, &mut slots);
            }
        }

        /// Pushes a read for every slot that has bytes left and none in flight.
        fn submit(&mut self, files: &mut [FileRead], slots: &mut [Slot]) -> usize {
            let mut pushed = 0usize;
            let mut sq = self.ring.submission();
            for (index, (slot, file)) in slots.iter_mut().zip(files.iter_mut()).enumerate() {
                let Some(handle) = slot.file.as_ref() else {
                    continue;
                };
                if slot.inflight || slot.done >= slot.len || file.result.is_err() {
                    continue;
                }
                let remaining = slot.len - slot.done;
                let entry = opcode::Read::new(
                    types::Fd(handle.as_raw_fd()),
                    // SAFETY: `done + remaining == len == buf.len()`, so the
                    // pointer and length stay inside the buffer, which is not
                    // touched again until this read completes: `read` does not
                    // return while any read is in flight, and `drain` waits them
                    // all out on the failure path.
                    unsafe { file.buf.as_mut_ptr().add(slot.done) },
                    u32::try_from(remaining).unwrap_or(u32::MAX),
                )
                .offset(u64::try_from(slot.done).unwrap_or(u64::MAX))
                .build()
                .user_data(u64::try_from(index).unwrap_or(u64::MAX));
                // SAFETY: the entry names a descriptor held open in `slots` and a
                // buffer owned by `files`, both of which outlive the read for the
                // reason above; this ring has no other submitter.
                if unsafe { sq.push(&entry) }.is_err() {
                    break;
                }
                slot.inflight = true;
                pushed += 1;
            }
            pushed
        }

        /// Accounts every completion; returns how many reads finished.
        #[expect(
            clippy::cast_sign_loss,
            reason = "the result is cast only on the branch that has established it is not negative"
        )]
        fn reap(&mut self, files: &mut [FileRead], slots: &mut [Slot]) -> usize {
            let mut finished = 0usize;
            for cqe in self.ring.completion() {
                finished += 1;
                let Ok(index) = usize::try_from(cqe.user_data()) else {
                    continue;
                };
                let (Some(slot), Some(file)) = (slots.get_mut(index), files.get_mut(index)) else {
                    continue;
                };
                slot.inflight = false;
                let result = cqe.result();
                if result < 0 {
                    file.result = Err(io::Error::from_raw_os_error(-result));
                } else if result == 0 && slot.done < slot.len {
                    file.result = Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "the file ended before its length",
                    ));
                } else {
                    slot.done += result as usize;
                }
            }
            finished
        }

        /// Waits out every read still in flight, so no buffer is handed back
        /// while the kernel may still write into it.
        fn drain(&mut self, mut inflight: usize) {
            while inflight > 0 {
                match self.ring.submit_and_wait(1) {
                    Ok(_) => {}
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    // Unknowable from here; the reader is dropped by its caller
                    // before any buffer is reused.
                    Err(_) => return,
                }
                for _ in self.ring.completion() {
                    inflight = inflight.saturating_sub(1);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BulkReader, FileRead};

    #[test]
    fn every_file_comes_back_whole_and_a_missing_one_answers_for_itself() {
        let dir = tempfile::tempdir().expect("a directory");
        let mut files = Vec::new();
        for n in 0..5u8 {
            let path = dir.path().join(format!("f{n}"));
            let body: Vec<u8> = (0..(1000 + usize::from(n) * 4099))
                .map(|i| u8::try_from(i % 251).expect("fits") ^ n)
                .collect();
            std::fs::write(&path, &body).expect("written");
            files.push((path, body));
        }
        let mut reads: Vec<FileRead> = files
            .iter()
            .map(|(path, _)| FileRead::new(path.clone(), Vec::new()))
            .collect();
        reads.push(FileRead::new(dir.path().join("absent"), Vec::new()));
        let mut reader = BulkReader::new(3);
        reader.read(&mut reads);
        for (read, (_, body)) in reads.iter().zip(files.iter()) {
            assert!(read.result.is_ok(), "{:?}", read.result);
            assert_eq!(&read.buf, body);
        }
        assert!(reads.last().expect("the missing one").result.is_err());
    }
}
