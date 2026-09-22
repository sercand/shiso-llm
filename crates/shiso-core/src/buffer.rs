//! Device memory that frees itself.
//!
//! A [`DeviceBuffer`] owns a device allocation and releases it when dropped, so
//! there is no `free` to forget and no way to double-free. Kernels never take a
//! buffer: they take a [`DeviceView`], a borrowed range that carries the
//! buffer's lifetime. A view that outlives its buffer is a compile error, which
//! is the whole reason this module exists: with handle-and-integer GPU APIs the
//! same mistake is a silent read of recycled memory.

use std::fmt;
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Arc;

use crate::backend::Backend;
use crate::error::{CoreError, CoreResult};

/// Opaque, backend-local allocation handle.
///
/// Fields are private and the only way to obtain one is through a [`Backend`],
/// so a handle can never be forged, aliased or pointed at memory the caller
/// does not own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RawBuffer {
    id: NonZeroU64,
    bytes: u64,
}

impl RawBuffer {
    /// Builds a handle. Only a backend calls this: the id must be one that its
    /// own `free_raw` will accept.
    #[must_use]
    #[doc(hidden)]
    pub const fn new(id: NonZeroU64, bytes: u64) -> Self {
        Self { id, bytes }
    }

    /// The backend's handle for this allocation.
    #[must_use]
    pub const fn id(self) -> NonZeroU64 {
        self.id
    }

    /// Size of the allocation in bytes.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }
}

/// Host memory a backend may wrap without copying, typically a file mapping.
///
/// `DeviceBuffer` holds this by `Arc`, so a zero-copy wrap cannot outlive the
/// bytes the device is reading. That is what makes [`import_host`] safe.
pub trait HostBytes: AsRef<[u8]> + Send + Sync + fmt::Debug + 'static {}

impl<T: AsRef<[u8]> + Send + Sync + fmt::Debug + 'static> HostBytes for T {}

/// A borrowed, `Copy` range of a [`DeviceBuffer`].
///
/// The lifetime parameter is the entire point: a view cannot outlive the buffer
/// it names, so the use-after-free that a raw handle makes easy here is a
/// compile error. Views are only produced by [`DeviceBuffer::view`] and
/// [`DeviceBuffer::slice`], never by a caller.
///
/// # The lifetime is the guarantee
///
/// The following does not compile, and that is the point: the view cannot be
/// returned, stored or read past the buffer it was taken from, because the
/// buffer's owner has already dropped it.
///
/// ```compile_fail
/// use std::sync::Arc;
///
/// use shiso_core::{Backend, alloc};
///
/// fn first_length(backend: &Arc<dyn Backend>) -> u64 {
///     let buffer = alloc(backend, 64).expect("allocation");
///     let view = buffer.view();
///     drop(buffer);
///     view.len()
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DeviceView<'a> {
    id: NonZeroU64,
    offset: u64,
    len: u64,
    _buf: PhantomData<&'a DeviceBuffer>,
}

impl DeviceView<'_> {
    /// The backend handle of the buffer this range lives in.
    #[must_use]
    pub const fn id(self) -> NonZeroU64 {
        self.id
    }

    /// Byte offset of the range within the buffer.
    #[must_use]
    pub const fn offset(self) -> u64 {
        self.offset
    }

    /// Length of the range in bytes.
    #[must_use]
    pub const fn len(self) -> u64 {
        self.len
    }

    /// Whether the range is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// An owning device allocation.
///
/// Dropping it frees the device memory. There is no `free` to forget, no way to
/// double-free, and no constructor a caller can reach: buffers come from
/// [`alloc`] or [`import_host`], both of which go through a [`Backend`].
#[derive(Debug)]
pub struct DeviceBuffer {
    raw: RawBuffer,
    owner: Arc<dyn Backend>,
    /// `Some` for [`import_host`] buffers: keeps the host mapping alive for
    /// exactly as long as the device can see it.
    host: Option<Arc<dyn HostBytes>>,
}

impl DeviceView<'_> {
    /// A view over a range of memory the caller owns.
    ///
    /// Public because a backend lives in its own crate: it has to be able to hand
    /// out a view over an allocation it made, and there is no other door. The two
    /// that needed one are the default `import_raw`, which allocates and copies,
    /// and any backend holding a slab of its own - cooperative scratch, a staging
    /// ring - that the engine addresses through the ordinary view type.
    ///
    /// Safe to call, and that is not an oversight: a view is three numbers and
    /// constructing one dereferences nothing. A range that does not name real
    /// memory fails when something tries to use it, which is where the backend
    /// checks it against the allocation the id belongs to.
    #[must_use]
    pub const fn from_raw(raw: RawBuffer, offset: u64, len: u64) -> Self {
        Self {
            id: raw.id(),
            offset,
            len,
            _buf: PhantomData,
        }
    }
}

impl DeviceBuffer {
    pub(crate) fn new(raw: RawBuffer, owner: Arc<dyn Backend>) -> Self {
        Self {
            raw,
            owner,
            host: None,
        }
    }

    pub(crate) fn wrapped_host(
        raw: RawBuffer,
        owner: Arc<dyn Backend>,
        host: Arc<dyn HostBytes>,
    ) -> Self {
        Self {
            raw,
            owner,
            host: Some(host),
        }
    }

    /// The backend this allocation belongs to.
    #[must_use]
    pub fn backend(&self) -> &Arc<dyn Backend> {
        &self.owner
    }

    /// Size of the allocation in bytes.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.raw.bytes()
    }

    /// Whether the allocation is zero bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A view of the whole buffer.
    #[must_use]
    pub fn view(&self) -> DeviceView<'_> {
        DeviceView::from_raw(self.raw, 0, self.len())
    }

    /// Bounds-checked sub-range.
    ///
    /// `offset + len` past the end is [`CoreError::OutOfBounds`]; the range is
    /// never silently clamped, because a kernel that wrote less than it was
    /// asked to is worse than one that failed.
    ///
    /// # Errors
    ///
    /// [`CoreError::OutOfBounds`] when `offset + len` runs past the end of the
    /// buffer.
    pub fn slice(&self, offset: u64, len: u64) -> CoreResult<DeviceView<'_>> {
        let total = self.len();
        let end =
            offset
                .checked_add(len)
                .filter(|&end| end <= total)
                .ok_or(CoreError::OutOfBounds {
                    offset,
                    len,
                    buffer_bytes: total,
                })?;
        Ok(DeviceView::from_raw(self.raw, offset, end - offset))
    }
}
impl DeviceBuffer {
    /// The host mapping behind an [`import_host`] buffer, if any.
    ///
    /// Exposed so a caller can assert the mapping is still pinned rather than
    /// trust it; dropping the last `DeviceBuffer` is what releases it.
    #[must_use]
    pub fn host(&self) -> Option<&Arc<dyn HostBytes>> {
        self.host.as_ref()
    }
}

impl Drop for DeviceBuffer {
    /// Releases the allocation through the backend that made it.
    ///
    /// A `Drop` cannot return an error, so a failed free is logged at error
    /// level instead of being swallowed: leaking device memory is visible in
    /// the trace even when it cannot be handled.
    fn drop(&mut self) {
        if let Err(err) = self.owner.free_raw(self.raw) {
            tracing::error!(
                id = %self.raw.id(),
                bytes = self.raw.bytes(),
                error = %err,
                "device free failed",
            );
        }
    }
}

/// Allocates device memory through `backend`.
///
/// This is the only way to create a `DeviceBuffer` outside a backend, and the
/// returned value releases itself.
///
/// # Errors
///
/// [`CoreError::Backend`] when the device cannot reserve `bytes`, and
/// [`CoreError::OutOfBounds`] when the byte count would overflow the
/// allocation size.
pub fn alloc(backend: &Arc<dyn Backend>, bytes: u64) -> CoreResult<DeviceBuffer> {
    let raw = backend.alloc_raw(bytes)?;
    if raw.bytes() < bytes {
        return Err(CoreError::Backend {
            backend: backend.backend_name(),
            op: "alloc_raw",
            source: Box::new(ShortAllocation {
                requested: bytes,
                granted: raw.bytes(),
            }),
        });
    }
    Ok(DeviceBuffer::new(raw, Arc::clone(backend)))
}

/// Wraps `host[range]` as device memory without copying where the backend can,
/// and allocates plus copies where it cannot.
///
/// Safe in both cases: the `Arc` keeps the host bytes alive for exactly as long
/// as the `DeviceBuffer` that names them, so a file mapping cannot be unmapped
/// while the device still reads it.
///
/// # Errors
///
/// [`CoreError::OutOfBounds`] when `range` exceeds the host bytes, and
/// whatever [`Backend::import_raw`] reports.
pub fn import_host(
    backend: &Arc<dyn Backend>,
    host: Arc<dyn HostBytes>,
    range: Range<usize>,
) -> CoreResult<DeviceBuffer> {
    // Explicit `AsRef` target: `Arc::as_ref` would otherwise win and hand
    // back `&dyn HostBytes`.
    let all: &[u8] = AsRef::<[u8]>::as_ref(&*host);
    let len = range.end.saturating_sub(range.start);
    if range.end > all.len() {
        return Err(CoreError::OutOfBounds {
            offset: range.start as u64,
            len: len as u64,
            buffer_bytes: all.len() as u64,
        });
    }
    let raw = backend.import_raw(host.as_ref(), range)?;
    Ok(DeviceBuffer::wrapped_host(raw, Arc::clone(backend), host))
}

/// A backend that granted less memory than it claimed.
#[derive(Debug)]
struct ShortAllocation {
    requested: u64,
    granted: u64,
}

impl fmt::Display for ShortAllocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "backend granted {} of {} requested bytes",
            self.granted, self.requested
        )
    }
}

impl std::error::Error for ShortAllocation {}

impl<'a> DeviceView<'a> {
    /// A sub-range of this range.
    ///
    /// Bounds are checked against this view, not the underlying buffer, so a
    /// caller holding a sub-range cannot reach the parts of the buffer it was
    /// not given.
    ///
    /// # Errors
    ///
    /// [`CoreError::OutOfBounds`] when `offset + len` runs past the end of
    /// this view.
    pub fn slice(&self, offset: u64, len: u64) -> CoreResult<DeviceView<'a>> {
        let end = offset
            .checked_add(len)
            .filter(|&end| end <= self.len)
            .ok_or(CoreError::OutOfBounds {
                offset,
                len,
                buffer_bytes: self.len,
            })?;
        Ok(Self {
            id: self.id,
            offset: self.offset + offset,
            len: end - offset,
            _buf: PhantomData,
        })
    }
}
