//! A backend that exists only so unit tests can hold a real device view.
//!
//! [`shiso_core::DeviceView`] has no public constructor — it is only ever
//! borrowed from a [`shiso_core::DeviceBuffer`], which is only ever produced by
//! a [`Backend`]. That is the whole point of the type, and it means the parts
//! of this crate that *consume* views — the buffer table, the argument packer,
//! the matrix-multiply validator — cannot be tested without a backend on the
//! other end.
//!
//! This is that backend and nothing more. It hands out handles from a real
//! [`BufferTable`] against invented addresses, so the table under test is the
//! table the views name, and every other method is a refusal. It has no device
//! and makes no driver call, which is exactly why it can run here.

use std::sync::Arc;

use parking_lot::{Mutex, MutexGuard};
use shiso_core::{Backend, CoreError, CoreResult, DeviceView, KernelRef, LaunchArgs, RawBuffer};

use crate::device::BufferTable;
use crate::sys::DevicePtr;

/// Address the first allocation is pretended to live at.
const FIRST_ADDRESS: u64 = 0x1_0000;

/// Distance between consecutive pretend allocations.
///
/// Wide enough that an offset-arithmetic bug lands outside the allocation it
/// belongs to rather than plausibly inside the next one.
const ADDRESS_STRIDE: u64 = 0x1_0000;

/// A backend whose only capability is handing out buffer handles.
#[derive(Debug)]
pub(crate) struct TableBackend {
    table: Mutex<BufferTable>,
    next_address: Mutex<u64>,
}

impl TableBackend {
    /// A backend with an empty table.
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            table: Mutex::new(BufferTable::new()),
            next_address: Mutex::new(FIRST_ADDRESS),
        })
    }

    /// The table the handles this backend issued are keyed in.
    pub(crate) fn table(&self) -> MutexGuard<'_, BufferTable> {
        self.table.lock()
    }

    /// The address the first allocation is given.
    pub(crate) const fn first_address() -> u64 {
        FIRST_ADDRESS
    }
}

/// The refusal every capability this backend does not have answers with.
fn unsupported() -> CoreError {
    CoreError::Unsupported {
        what: "this operation",
        backend: "test",
    }
}

impl Backend for TableBackend {
    fn backend_name(&self) -> &'static str {
        "test"
    }

    fn alloc_raw(&self, bytes: u64) -> CoreResult<RawBuffer> {
        let mut address = self.next_address.lock();
        let ptr = DevicePtr::from_raw(*address);
        *address += ADDRESS_STRIDE;
        Ok(self.table.lock().insert(ptr, bytes))
    }

    fn free_raw(&self, raw: RawBuffer) -> CoreResult<()> {
        self.table.lock().remove(raw.id());
        Ok(())
    }

    fn copy_h2d(&self, _dst: DeviceView<'_>, _src: &[u8]) -> CoreResult<()> {
        Err(unsupported())
    }

    fn copy_d2h(&self, _src: DeviceView<'_>, _dst: &mut [u8]) -> CoreResult<()> {
        Err(unsupported())
    }

    fn copy_d2d(&self, _dst: DeviceView<'_>, _src: DeviceView<'_>) -> CoreResult<()> {
        Err(unsupported())
    }

    fn run(&self, _kernel: KernelRef, _args: &LaunchArgs<'_>) -> CoreResult<()> {
        Err(unsupported())
    }

    fn sync(&self) -> CoreResult<()> {
        Ok(())
    }
}
