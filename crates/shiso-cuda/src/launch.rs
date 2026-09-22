//! Argument packing and the launch itself.
//!
//! # The argument array
//!
//! A launch hands the driver an array of pointers, one per declared kernel
//! parameter, each addressing a value of that parameter's type. The driver
//! reads as many bytes from each as the kernel's own signature says, so what
//! this module has to get right is the *order* and the *contents* of the slots,
//! not their declared widths.
//!
//! Slots live in one `Vec<u64>` rather than one allocation each. That gives
//! every slot eight-byte alignment for free — a struct passed by value may
//! contain a 64-bit field, and a pointer into a byte vector would not be
//! aligned for it — and it means the whole argument list costs one allocation
//! per launch instead of one per argument. The pointer array is built from the
//! word buffer at the moment of the call, after the buffer has stopped growing,
//! because a pointer taken before a reallocation would address freed memory.
//!
//! # Geometry goes through untouched
//!
//! `grid` is a block count and `block` is threads per block, exactly as the
//! caller computed them. Nothing here divides, rounds up, or reinterprets
//! either. A backend that silently adjusted a grid would make every kernel's
//! own index arithmetic a guess, and the kernels this loads were written
//! against launch geometry that the host is expected to have got right.

use std::ffi::c_void;

use shiso_core::{CoreError, CoreResult, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::CudaBackend;
use crate::device::BufferTable;
use crate::sys::host_error;
use crate::sys::nvidia;

/// Which of the two launch entry points a dispatch uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LaunchMode {
    /// Blocks run whenever the device has room; the ordinary case.
    Ordinary,
    /// Every block must be resident at once, because they synchronise against
    /// each other. Refused outright by the driver when the grid does not fit.
    Cooperative,
}

/// One packed kernel argument: where its bytes start and how many there are.
#[derive(Debug, Clone, Copy)]
struct Slot {
    /// Index into the word buffer where this argument's bytes begin.
    start: usize,
    /// Bytes the argument occupies, which may be fewer than the words reserved.
    bytes: usize,
}

/// The packed argument list for one launch.
///
/// Built fresh per launch. It borrows nothing from the buffer table: a device
/// address is copied into a slot while the table lock is held, so the list is
/// self-contained by the time the driver sees it.
pub(crate) struct Argv {
    /// Slot storage. Eight-byte aligned by construction, which is the alignment
    /// every parameter type this backend passes needs.
    words: Vec<u64>,
    slots: Vec<Slot>,
    /// Rebuilt by [`Argv::params`] from `words`, never kept across a push.
    params: Vec<*mut c_void>,
}

impl Argv {
    /// An empty list, sized for `capacity` arguments.
    fn with_capacity(capacity: usize) -> Self {
        Self {
            words: Vec::with_capacity(capacity),
            slots: Vec::with_capacity(capacity),
            params: Vec::with_capacity(capacity),
        }
    }

    /// Appends one argument's bytes as a new slot.
    fn push(&mut self, bytes: &[u8]) {
        let start = self.words.len();
        for chunk in bytes.chunks(size_of::<u64>()) {
            let mut word = [0_u8; size_of::<u64>()];
            word[..chunk.len()].copy_from_slice(chunk);
            // Native byte order: the slot is a copy of the value's in-memory
            // image, and the device shares this host's byte order.
            self.words.push(u64::from_ne_bytes(word));
        }
        if bytes.is_empty() {
            // A zero-length by-value argument still occupies a parameter slot,
            // so it gets a word to point at rather than an address that would
            // collide with the next argument's.
            self.words.push(0);
        }
        self.slots.push(Slot {
            start,
            bytes: bytes.len(),
        });
    }

    /// Number of parameter slots.
    pub(crate) fn len(&self) -> usize {
        self.slots.len()
    }

    /// The bytes of one slot, for a caller checking what was packed.
    #[cfg(test)]
    pub(crate) fn slot(&self, index: usize) -> Option<Vec<u8>> {
        let slot = *self.slots.get(index)?;
        let words = slot.bytes.div_ceil(size_of::<u64>()).max(1);
        let mut bytes = Vec::with_capacity(words * size_of::<u64>());
        for word in self.words.get(slot.start..slot.start + words)? {
            bytes.extend_from_slice(&word.to_ne_bytes());
        }
        bytes.truncate(slot.bytes);
        Some(bytes)
    }

    /// The pointer array the driver takes.
    ///
    /// Built here rather than during packing: `words` grows while arguments are
    /// appended, and a pointer taken before a reallocation would address memory
    /// the vector no longer owns. The returned slice borrows `self`, so the
    /// pointers cannot outlive the storage they address.
    pub(crate) fn params(&mut self) -> &mut [*mut c_void] {
        let base = self.words.as_mut_ptr();
        self.params.clear();
        for slot in &self.slots {
            // SAFETY: `slot.start` is an index into `words` that `push`
            // produced and `words` has only grown since, so the offset stays
            // inside the same allocation.
            let ptr = unsafe { base.add(slot.start) };
            self.params.push(ptr.cast::<c_void>());
        }
        &mut self.params
    }
}

impl std::fmt::Debug for Argv {
    /// The shape of the argument list, not its contents.
    ///
    /// Slot contents are device addresses and raw struct images; what a reader
    /// of a trace wants is how many parameters the launch carried and how wide
    /// they were, which is also what tells a mismatched kernel signature apart
    /// from a mismatched value.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bytes: usize = self.slots.iter().map(|slot| slot.bytes).sum();
        f.debug_struct("Argv")
            .field("slots", &self.slots.len())
            .field("bytes", &bytes)
            .finish_non_exhaustive()
    }
}

/// Packs one launch's arguments, resolving every device range against `table`.
///
/// One slot per argument, in the order the caller gave them, which is the order
/// the kernel declares its parameters in. A buffer argument becomes the address
/// of its first byte; a scalar becomes its own bytes; a by-value struct becomes
/// its bytes verbatim, because the layout it was built from is the struct
/// definition and re-deriving it here could only get it wrong.
///
/// # Errors
///
/// [`CoreError::OutOfBounds`] when a range runs past its allocation,
/// [`CoreError::Backend`] when a range names an allocation that is not live,
/// and [`CoreError::Unsupported`] for an argument kind this backend has no
/// representation for — never a silently skipped argument, which would shift
/// every later parameter by one.
pub(crate) fn pack_args(table: &BufferTable, args: &[KernelArg<'_>]) -> CoreResult<Argv> {
    let mut packed = Argv::with_capacity(args.len());
    for arg in args {
        match arg {
            KernelArg::Buffer(view) => {
                let ptr = table.resolve(*view)?;
                packed.push(&ptr.raw().to_ne_bytes());
            }
            KernelArg::U32(value) => packed.push(&value.to_ne_bytes()),
            KernelArg::I32(value) => packed.push(&value.to_ne_bytes()),
            KernelArg::U64(value) => packed.push(&value.to_ne_bytes()),
            KernelArg::F32(value) => packed.push(&value.to_ne_bytes()),
            KernelArg::Bytes(bytes) => packed.push(bytes),
            // A kind added to the core vocabulary must be packed here rather
            // than skipped: a missing slot does not drop one argument, it
            // shifts every argument after it.
            _ => {
                return Err(CoreError::Unsupported {
                    what: "this kernel argument kind",
                    backend: crate::sys::BACKEND,
                });
            }
        }
    }
    Ok(packed)
}

/// Refuses a kernel reference that carries specialization constants.
///
/// There is no mechanism on this device to give a compiled entry point a
/// constant at launch time; a template variant is a separate entry point in the
/// generated registry, with its own name. A non-empty constant table therefore
/// means the caller expected a specialization this backend cannot apply, and
/// ignoring the field would launch the wrong kernel and say nothing.
///
/// # Errors
///
/// [`CoreError::Unsupported`] when the table is non-empty.
pub(crate) fn reject_function_constants(kernel: KernelRef) -> CoreResult<()> {
    if kernel.consts.is_empty() {
        return Ok(());
    }
    Err(CoreError::Unsupported {
        what: "function constants",
        backend: crate::sys::BACKEND,
    })
}

/// Refuses a launch geometry the device cannot run.
///
/// A zero in any dimension launches nothing, and the driver reports it as an
/// invalid value with no hint about which of the six numbers was wrong. Caught
/// here, the message names the dimension.
///
/// # Errors
///
/// [`CoreError::Backend`] when any component of either extent is zero.
pub(crate) fn check_geometry(grid: Extent3, block: Extent3) -> CoreResult<()> {
    for (label, extent) in [("grid", grid), ("block", block)] {
        if extent.x == 0 || extent.y == 0 || extent.z == 0 {
            return Err(host_error(
                "run",
                format!(
                    "{label} {}x{}x{} has a zero dimension, so the launch would run nothing",
                    extent.x, extent.y, extent.z
                ),
            ));
        }
    }
    Ok(())
}

/// Resolves, packs and queues one launch.
///
/// # Errors
///
/// [`CoreError::Unsupported`] for a specialized kernel reference or an argument
/// kind with no representation here, and [`CoreError::Backend`] for an unknown
/// module or entry point, a refused shared-memory request, or a refused launch.
pub(crate) fn dispatch(
    backend: &CudaBackend,
    kernel: KernelRef,
    launch: &LaunchArgs<'_>,
    mode: LaunchMode,
) -> CoreResult<()> {
    reject_function_constants(kernel)?;
    check_geometry(launch.grid, launch.block)?;
    backend.device.bind()?;

    // Resolved before the state lock is taken, so the registry's own lock is
    // never held while the buffer table's is and the two can never deadlock
    // against each other.
    let func = backend.modules.function(kernel)?;
    backend
        .modules
        .ensure_shared_memory(func, launch.smem_bytes)?;

    let state = backend.state.lock();
    let mut argv = pack_args(&state.buffers, launch.args)?;
    let stream = state.graphs.stream(&backend.device);
    tracing::trace!(
        module = kernel.module,
        entry = kernel.entry,
        args = argv.len(),
        smem_bytes = launch.smem_bytes,
        cooperative = mode == LaunchMode::Cooperative,
        "queueing a launch",
    );
    let params = argv.params();
    // SAFETY: one slot per argument was packed in declaration order, each
    // addressing storage `argv` owns for the whole call; every device address
    // in those slots was proved to lie inside a live allocation, and the state
    // lock is still held so none of them can be freed under the launch. The
    // entry point and the stream are owned by this backend and outlive the call.
    let queued = match mode {
        // SAFETY: as stated above the match.
        LaunchMode::Ordinary => unsafe {
            nvidia::launch_kernel(
                func,
                launch.grid,
                launch.block,
                launch.smem_bytes,
                stream,
                params,
            )
        },
        // SAFETY: as stated above the match, and the caller reached this arm
        // through `run_cooperative`, which is where the co-residency obligation
        // is documented.
        LaunchMode::Cooperative => unsafe {
            nvidia::launch_cooperative(
                func,
                launch.grid,
                launch.block,
                launch.smem_bytes,
                stream,
                params,
            )
        },
    };
    queued?;
    Ok(())
}

/// Largest grid of `kernel` the device can keep entirely resident.
///
/// The product of the multiprocessor count and the blocks of this entry point
/// that fit on one of them. Both halves come from the device; neither is used
/// to choose a kernel or an algorithm, only to size a grid whose blocks
/// synchronise against each other and would deadlock if some of them were not
/// running yet.
///
/// # Errors
///
/// [`CoreError::Unsupported`] for a specialized kernel reference, and
/// [`CoreError::Backend`] when the block is larger than a launch can be or the
/// device cannot answer for this entry point.
pub(crate) fn max_co_resident_blocks(
    backend: &CudaBackend,
    kernel: KernelRef,
    block: Extent3,
    smem: u32,
) -> CoreResult<u32> {
    reject_function_constants(kernel)?;
    backend.device.bind()?;
    let threads = u32::try_from(block.volume()).map_err(|_| {
        host_error(
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
            format!(
                "a block of {} threads is larger than any launch can be",
                block.volume()
            ),
        )
    })?;
    let func = backend.modules.function(kernel)?;
    backend.modules.ensure_shared_memory(func, smem)?;
    // SAFETY: `func` was resolved from a module this backend owns and has not
    // unloaded.
    let per_multiprocessor = unsafe { nvidia::occupancy_max_active_blocks(func, threads, smem) }?;
    Ok(per_multiprocessor.saturating_mul(backend.device.multiprocessors()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use shiso_core::{
        Backend, CoreError, Extent3, FnConst, FnConstValue, KernelArg, KernelRef, LaunchArgs,
    };

    use super::{Argv, check_geometry, pack_args, reject_function_constants};
    use crate::device::BufferTable;
    use crate::sys::DevicePtr;
    use crate::testing::TableBackend;

    /// Every slot the driver is handed must be eight-byte aligned, because a
    /// by-value struct argument may contain a 64-bit field.
    fn assert_slots_aligned(argv: &mut Argv) {
        for param in argv.params() {
            assert_eq!(
                (*param as usize) % align_of::<u64>(),
                0,
                "an argument slot was not aligned for a 64-bit field",
            );
        }
    }

    #[test]
    fn arguments_get_one_slot_each_in_the_order_they_were_given() {
        let table = BufferTable::new();
        let list = [
            KernelArg::U32(7),
            KernelArg::I32(-3),
            KernelArg::U64(1 << 40),
            KernelArg::F32(0.5),
        ];
        let mut argv = pack_args(&table, &list).expect("packing");
        assert_eq!(argv.len(), 4);
        assert_eq!(argv.slot(0), Some(7_u32.to_ne_bytes().to_vec()));
        assert_eq!(argv.slot(1), Some((-3_i32).to_ne_bytes().to_vec()));
        assert_eq!(argv.slot(2), Some((1_u64 << 40).to_ne_bytes().to_vec()));
        assert_eq!(argv.slot(3), Some(0.5_f32.to_ne_bytes().to_vec()));
        assert_slots_aligned(&mut argv);
    }

    #[test]
    fn scalar_slots_carry_their_own_width_and_nothing_more() {
        let table = BufferTable::new();
        let list = [KernelArg::U32(1), KernelArg::U64(2)];
        let argv = pack_args(&table, &list).expect("packing");
        assert_eq!(argv.slot(0).map(|s| s.len()), Some(4));
        assert_eq!(argv.slot(1).map(|s| s.len()), Some(8));
    }

    #[test]
    fn a_buffer_argument_becomes_the_address_of_its_first_byte() {
        let backend = TableBackend::new();
        let handle: Arc<dyn Backend> = backend.clone();
        let buffer = shiso_core::alloc(&handle, 512).expect("allocation");
        let base = TableBackend::first_address();
        let tail = buffer.slice(128, 64).expect("sub-range");

        let table = backend.table();
        let list = [KernelArg::Buffer(buffer.view()), KernelArg::Buffer(tail)];
        let mut argv = pack_args(&table, &list).expect("packing");
        assert_eq!(argv.slot(0), Some(base.to_ne_bytes().to_vec()));
        assert_eq!(argv.slot(1), Some((base + 128).to_ne_bytes().to_vec()));
        assert_slots_aligned(&mut argv);
    }

    #[test]
    fn a_buffer_argument_whose_range_left_its_allocation_is_out_of_bounds() {
        let backend = TableBackend::new();
        let handle: Arc<dyn Backend> = backend.clone();
        let buffer = shiso_core::alloc(&handle, 64).expect("allocation");
        let view = buffer.view();
        let mut shorter = BufferTable::new();
        shorter.insert(DevicePtr::from_raw(0xdead_0000), 16);

        match pack_args(&shorter, &[KernelArg::Buffer(view)]) {
            Err(CoreError::OutOfBounds { len, .. }) => assert_eq!(len, 64),
            other => panic!("expected an out-of-bounds error, got {other:?}"),
        }
    }

    #[test]
    fn a_by_value_struct_is_packed_verbatim() {
        let table = BufferTable::new();
        let blob: [u8; 13] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
        let list = [KernelArg::Bytes(&blob)];
        let mut argv = pack_args(&table, &list).expect("packing");
        assert_eq!(argv.slot(0), Some(blob.to_vec()));
        assert_slots_aligned(&mut argv);
    }

    #[test]
    fn a_launch_with_no_arguments_packs_an_empty_array() {
        let table = BufferTable::new();
        let mut argv = pack_args(&table, &[]).expect("packing");
        assert_eq!(argv.len(), 0);
        assert!(argv.params().is_empty());
        assert_eq!(argv.slot(0), None);
    }

    #[test]
    fn every_argument_keeps_a_distinct_slot_address() {
        let table = BufferTable::new();
        let list = [KernelArg::U32(1), KernelArg::U32(2), KernelArg::U32(3)];
        let mut argv = pack_args(&table, &list).expect("packing");
        let params: Vec<usize> = argv.params().iter().map(|p| *p as usize).collect();
        let mut sorted = params.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            params.len(),
            "two arguments sharing a slot would make the kernel read one twice",
        );
    }

    #[test]
    fn a_specialized_kernel_reference_is_unsupported_rather_than_silently_ignored() {
        const CONSTS: [FnConst; 1] = [FnConst {
            index: 0,
            value: FnConstValue::U32(4),
        }];
        let kernel = KernelRef::specialized("q4e_hc", "kernel", &CONSTS);
        match reject_function_constants(kernel) {
            Err(CoreError::Unsupported { what, backend }) => {
                assert_eq!(what, "function constants");
                assert_eq!(backend, "cuda");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        reject_function_constants(KernelRef::new("q4e_hc", "kernel"))
            .expect("an unspecialized reference is accepted");
    }

    #[test]
    fn a_zero_dimension_in_either_extent_is_refused_with_the_dimension_named() {
        let ok = Extent3::new(2, 3, 4);
        check_geometry(ok, ok).expect("a fully positive geometry is accepted");
        for (grid, block, expect) in [
            (Extent3::new(0, 1, 1), ok, "grid"),
            (ok, Extent3::new(1, 0, 1), "block"),
            (ok, Extent3::new(1, 1, 0), "block"),
        ] {
            match check_geometry(grid, block) {
                Err(CoreError::Backend { source, .. }) => {
                    let text = source.to_string();
                    assert!(text.contains(expect), "{text}");
                }
                other => panic!("expected a refusal, got {other:?}"),
            }
        }
    }

    #[test]
    fn launch_geometry_reaches_the_packer_exactly_as_the_caller_wrote_it() {
        // The packer does not see the geometry, which is the point: nothing
        // between the caller and the driver may rescale a grid, so the only
        // check worth making here is that `LaunchArgs` carries the numbers
        // through unchanged.
        let grid = Extent3::new(37, 2, 1);
        let block = Extent3::new(128, 1, 1);
        let launch = LaunchArgs::new(grid, block, &[]);
        assert_eq!(launch.grid, grid);
        assert_eq!(launch.block, block);
        assert_eq!(launch.smem_bytes, 0);
    }
}
