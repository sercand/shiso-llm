//! A backend that allocates, records launches and refuses everything else.
//!
//! Test-only, and shared by every host-side dispatch case in this crate.
//! `HANDOFF.md` at the repository root is twelve defects found bringing this
//! engine up on CUDA, **eleven of which failed silently**, and the shape of
//! most of them was the host handing a correct kernel the wrong argument: a
//! page count where a row count was wanted, both outputs where one was, a
//! score row strided by the budget instead of the context. None of those is
//! visible to a kernel and none of them fails a launch.
//!
//! So the launches are recorded and asserted: which entry, which grid, which
//! block, how many buffers, and what the by-value argument block actually
//! contains. It is also the smallest thing that can hold a real
//! [`shiso_core::DeviceView`], which has no public constructor on purpose.

use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use shiso_core::backend::{Backend, KernelRef, LaunchArgs};
use shiso_core::{CoreError, CoreResult, DeviceView, Extent3, RawBuffer};

/// One recorded launch.
#[derive(Debug, Clone)]
pub(crate) struct Launch {
    pub(crate) module: &'static str,
    pub(crate) entry: &'static str,
    pub(crate) grid: Extent3,
    pub(crate) block: Extent3,
    /// The first argument, which is the entry's by-value struct.
    pub(crate) args: Vec<u8>,
    /// Dynamic shared memory the launch asked for.
    pub(crate) smem: u32,
    /// The buffers that followed the by-value block, in the order the launch
    /// listed them.
    ///
    /// The ids and not just the count, because the count only catches a launch
    /// that is short an operand. A launch with the right number of buffers in
    /// the wrong ORDER - two views of the same slab swapped - is the same
    /// class of defect and is invisible to a count.
    pub(crate) ids: Vec<NonZeroU64>,
}

impl Launch {
    /// How many buffers followed the by-value block.
    pub(crate) fn buffers(&self) -> usize {
        self.ids.len()
    }
}

/// A backend that allocates, records launches, and refuses everything else.
#[derive(Debug)]
pub(crate) struct Recorder {
    launches: Mutex<Vec<Launch>>,
    next: Mutex<u64>,
}

impl Recorder {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            launches: Mutex::new(Vec::new()),
            next: Mutex::new(1),
        })
    }

    pub(crate) fn launches(&self) -> Vec<Launch> {
        self.launches.lock().expect("no panics here").clone()
    }
}

impl Backend for Recorder {
    fn backend_name(&self) -> &'static str {
        "recorder"
    }

    fn alloc_raw(&self, bytes: u64) -> CoreResult<RawBuffer> {
        let mut next = self.next.lock().expect("no panics here");
        let id = NonZeroU64::new(*next).expect("the counter starts at one");
        *next += 1;
        Ok(RawBuffer::new(id, bytes))
    }

    fn free_raw(&self, _raw: RawBuffer) -> CoreResult<()> {
        Ok(())
    }

    fn copy_h2d(&self, _dst: DeviceView<'_>, _src: &[u8]) -> CoreResult<()> {
        Ok(())
    }

    fn copy_d2h(&self, _src: DeviceView<'_>, _dst: &mut [u8]) -> CoreResult<()> {
        Err(CoreError::Unsupported {
            what: "a read-back",
            backend: "recorder",
        })
    }

    fn copy_d2d(&self, _dst: DeviceView<'_>, _src: DeviceView<'_>) -> CoreResult<()> {
        Ok(())
    }

    fn run(&self, kernel: KernelRef, launch: &LaunchArgs<'_>) -> CoreResult<()> {
        let mut args = Vec::new();
        let mut ids = Vec::new();
        for arg in launch.args {
            match arg {
                shiso_core::KernelArg::Bytes(bytes) if args.is_empty() => {
                    args.extend_from_slice(bytes);
                }
                shiso_core::KernelArg::Buffer(view) => ids.push(view.id()),
                _ => {}
            }
        }
        self.launches.lock().expect("no panics here").push(Launch {
            module: kernel.module,
            entry: kernel.entry,
            grid: launch.grid,
            block: launch.block,
            args,
            smem: launch.smem_bytes,
            ids,
        });
        Ok(())
    }

    fn sync(&self) -> CoreResult<()> {
        Ok(())
    }
}

/// Host bytes a weight is imported from.
///
/// [`shiso_core::HostBytes`] is blanket-implemented for everything that is
/// `AsRef<[u8]>`, so this is one impl and no trait method.
#[derive(Debug)]
pub(crate) struct Bytes(pub(crate) Vec<u8>);

impl AsRef<[u8]> for Bytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
