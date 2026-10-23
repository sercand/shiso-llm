//! The host-CPU device: safe-Rust kernels behind [`shiso_core::Backend`].
//!
//! The engine reaches a device only through the trait, so this crate is one
//! more implementation of it, with host RAM as "device" memory. It is written
//! for correctness and portability first: auto-vectorised loops, no `std::arch`
//! and no `unsafe` (`#![forbid(unsafe_code)]`), and kernels that follow the
//! arithmetic of the Metal ones so a case graded on one device is graded on
//! this one by the same fixture.
//!
//! # One stream
//!
//! Every launch runs to completion before [`Backend::run`] returns, on the
//! caller's thread plus scoped workers that have all joined by then. There is
//! no queue, so [`Backend::sync`] has nothing to wait for, and a launch after
//! another always sees its writes. Launches are serialised by one lock over the
//! allocations, which is what makes that promise cheap to keep.
//!
//! # Imports copy
//!
//! [`Backend::import_raw`] keeps its default: it copies the host bytes into a
//! fresh allocation. Wrapping a memory-mapped weight file in place would hand
//! kernels bytes with no alignment promise and tie a buffer's life to the
//! mapping; the copy costs one pass at load and nothing afterwards.
//!
//! # Measured speed
//!
//! Apple M1 Max (10 cores), 2026-10-23, `--release`, SmolLM2-135M Clef `Q8_0`
//! (`target/release/shiso serve --device cpu|metal`, one request at a time,
//! warm). The 417-token `/v1/systemone` request (the `support` reference case)
//! and decode tok/s from `/v1/completions` (`max_tokens` 128, greedy, the
//! server's own `decode_tokens_per_second`). Completions was timed because the
//! hand-built serve borrowed its config and tokenizer from another model home,
//! so the chat template was not this model's:
//!
//! | Device | `/v1/systemone`, 417 tokens | Decode |
//! |---|---|---|
//! | cpu | 4.45 s | 31.8 tok/s |
//! | metal | 0.050 s | 287 tok/s |
//!
//! The parallel-launch thresholds were swept and left alone; the numbers sit
//! beside them in the `dispatch` module.
//! The Linux `x86_64` and GB10 figures are not recorded here yet.
//!
//! Kernel arguments and the launch-time aliasing model are described in the
//! `dispatch` module docs.

#![forbid(unsafe_code)]

mod dispatch;
mod kernels;
mod memory;
mod q8;

use std::fmt;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, MutexGuard};

use shiso_core::{
    Backend, CoreError, CoreResult, DeviceView, KernelRef, LaunchArgs, MemLedger, MemoryInfo,
    RawBuffer,
};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

use crate::memory::Slabs;

/// Knobs of the CPU device. For tests: the server has no flag for them.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuOptions {
    /// Worker threads a launch may use. `None` asks the host
    /// (`available_parallelism`, or one thread if it cannot say).
    pub threads: Option<NonZeroUsize>,
}

/// The host CPU as a device.
pub struct CpuBackend {
    slabs: Mutex<Slabs>,
    threads: usize,
    ledger: Arc<MemLedger>,
}

impl fmt::Debug for CpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuBackend")
            .field("threads", &self.threads)
            .finish_non_exhaustive()
    }
}

impl CpuBackend {
    /// Opens the CPU device.
    ///
    /// # Errors
    ///
    /// None today; the signature is fallible so a device that can fail to
    /// open and this one are called the same way.
    pub fn new(options: CpuOptions) -> CoreResult<Arc<Self>> {
        let threads = options.threads.map_or_else(
            || std::thread::available_parallelism().map_or(1, NonZeroUsize::get),
            NonZeroUsize::get,
        );
        Ok(Arc::new(Self {
            slabs: Mutex::new(Slabs::new()),
            threads,
            // Host RAM is the one pool, so the ledger reads it as unified.
            ledger: Arc::new(MemLedger::new(true)),
        }))
    }

    /// The allocations, locked. A poisoned lock means a kernel panicked
    /// mid-launch; the slabs themselves are still whole bytes, and refusing
    /// every later call would turn one bug into a dead device.
    fn slabs(&self) -> MutexGuard<'_, Slabs> {
        self.slabs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Whether the CPU device has a kernel for `(module, entry)`.
#[must_use]
pub fn has_entry(module: &str, entry: &str) -> bool {
    kernels::TABLE
        .iter()
        .any(|(m, e, _)| *m == module && *e == entry)
}

impl Backend for CpuBackend {
    fn backend_name(&self) -> &'static str {
        "cpu"
    }

    fn numerics_id(&self) -> String {
        "cpu-v1".to_owned()
    }

    fn alloc_raw(&self, bytes: u64) -> CoreResult<RawBuffer> {
        self.slabs().alloc(bytes)
    }

    fn free_raw(&self, raw: RawBuffer) -> CoreResult<()> {
        // An unknown handle was already released, which is idempotent rather
        // than a failure: `Drop` cannot retry anyway.
        if !self.slabs().free(raw.id()) {
            tracing::debug!(id = raw.id().get(), "CPU handle was already freed");
        }
        Ok(())
    }

    fn memory_info(&self) -> Option<MemoryInfo> {
        let system = System::new_with_specifics(
            RefreshKind::nothing().with_memory(MemoryRefreshKind::nothing().with_ram()),
        );
        let total = system.total_memory();
        (total > 0).then(|| MemoryInfo {
            free: system.available_memory().min(total),
            total,
        })
    }

    fn ledger(&self) -> Option<&Arc<MemLedger>> {
        Some(&self.ledger)
    }

    fn copy_h2d(&self, dst: DeviceView<'_>, src: &[u8]) -> CoreResult<()> {
        if dst.len() != src.len() as u64 {
            return Err(CoreError::ShapeMismatch {
                field: "copy_h2d len",
                expected: dst.len(),
                got: src.len() as u64,
            });
        }
        self.slabs().bytes_mut(dst)?.copy_from_slice(src);
        Ok(())
    }

    fn copy_d2h(&self, src: DeviceView<'_>, dst: &mut [u8]) -> CoreResult<()> {
        if src.len() != dst.len() as u64 {
            return Err(CoreError::ShapeMismatch {
                field: "copy_d2h len",
                expected: src.len(),
                got: dst.len() as u64,
            });
        }
        dst.copy_from_slice(self.slabs().bytes(src)?);
        Ok(())
    }

    fn fill_zero(&self, dst: DeviceView<'_>) -> CoreResult<()> {
        self.slabs().bytes_mut(dst)?.fill(0);
        Ok(())
    }

    fn copy_d2d(&self, dst: DeviceView<'_>, src: DeviceView<'_>) -> CoreResult<()> {
        if dst.len() != src.len() {
            return Err(CoreError::ShapeMismatch {
                field: "copy_d2d len",
                expected: src.len(),
                got: dst.len(),
            });
        }
        self.slabs().copy(dst, src)
    }

    fn run(&self, kernel: KernelRef, args: &LaunchArgs<'_>) -> CoreResult<()> {
        // The lock is held for the whole launch: that is the one-stream
        // promise, and it keeps a concurrent copy from changing the bytes a
        // kernel snapshotted.
        let mut slabs = self.slabs();
        dispatch::execute(kernels::TABLE, &mut slabs, kernel, args, self.threads)
    }

    fn sync(&self) -> CoreResult<()> {
        Ok(())
    }
}
