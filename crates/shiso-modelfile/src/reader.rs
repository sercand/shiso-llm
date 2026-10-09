//! Header, metadata table, tensor directory and the file mapping.
//!
//! The reader is a *describer*: it walks the header, the key/value table and
//! the tensor directory, and records where every tensor lives. Weight bytes are
//! never copied, never realigned and never decoded here - a [`TensorView`] is a
//! borrow into the mapping, and the same mapping is what a device backend wraps
//! through [`GgufFile::mapping`].
//!
//! Every parse failure becomes a [`GgufError`] that names the tensor or key
//! involved, because a malformed model is ordinary runtime input. There is no
//! "log a warning and keep going" path: a tensor whose format is unknown, or
//! whose declared bytes run past the end of the file, stops the open.

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::mem::{align_of, size_of};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;
use shiso_core::DType;
use shiso_core::HostBytes;

use crate::metadata::GgufMetadata;
use crate::quant::{self, QuantBlock};

/// The fourcc every model file starts with, little endian.
pub const GGUF_MAGIC: u32 = 0x4655_4747;
/// Largest tensor rank the on-disk format can describe.
pub const MAX_DIMS: usize = 8;

/// Smallest file that can hold a header; anything shorter is not a model.
const MIN_FILE_BYTES: u64 = 32;
/// Header version this reader accepts, and the only one that exists.
const SUPPORTED_VERSION: u32 = 3;
/// Tensor payload alignment used when the metadata omits `general.alignment`.
const DEFAULT_ALIGNMENT: u64 = 32;
/// Largest accepted `general.alignment`, which is a page.
const MAX_ALIGNMENT: u32 = 4096;
/// `split.count` above which a model is a multi-file split, which is refused.
const SINGLE_FILE: u32 = 1;

/// Everything that can go wrong while opening or reading a model file.
///
/// The variants naming a tensor carry the name rather than an index, because
/// the only thing a caller can do with a load failure is report it to someone
/// who can look at the file.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GgufError {
    /// The magic word did not match.
    #[error("file is not a model file (bad magic)")]
    NotGguf,
    /// The header named a version this reader does not implement.
    #[error("only model format version {SUPPORTED_VERSION} is supported, got {got}")]
    Version {
        /// Version the header declared.
        got: u32,
    },
    /// The file ended where the format required more bytes.
    #[error("file is truncated at byte {pos}")]
    Truncated {
        /// Offset at which the read ran past the end.
        pos: u64,
    },
    /// Metadata arrays nested deeper than the accepted cap.
    #[error("metadata array nesting is deeper than the supported limit")]
    NestingTooDeep,
    /// A tensor uses a format id this crate cannot size.
    #[error("tensor {tensor} uses unsupported format id {dtype_id}")]
    UnsupportedType {
        /// Name of the offending tensor.
        tensor: Box<str>,
        /// The raw format id from the directory.
        dtype_id: u32,
    },
    /// A tensor declared more dimensions than [`MAX_DIMS`].
    #[error("tensor {tensor} declares {ndim} dimensions, at most {MAX_DIMS} are supported")]
    DimLimit {
        /// Name of the offending tensor.
        tensor: Box<str>,
        /// The declared rank.
        ndim: u32,
    },
    /// A rank outside `1..=MAX_DIMS` reached [`TensorShape::new`].
    #[error("tensor rank {ndim} is outside 1..={MAX_DIMS}")]
    BadRank {
        /// The declared rank.
        ndim: usize,
    },
    /// A zero-length dimension reached [`TensorShape::new`].
    #[error("dimension {index} is zero")]
    ZeroDim {
        /// Index of the zero dimension.
        index: usize,
    },
    /// The product of a tensor's dimensions overflowed `u64`.
    #[error("tensor element count overflows a 64-bit integer")]
    ElementCountOverflow,
    /// A tensor's byte count overflowed `u64`.
    #[error("tensor {tensor} is too large to describe")]
    TensorTooLarge {
        /// Name of the offending tensor.
        tensor: Box<str>,
    },
    /// A tensor's declared span is not inside the file.
    #[error("tensor {tensor} points outside the file")]
    OutOfBounds {
        /// Name of the offending tensor.
        tensor: Box<str>,
    },
    /// The model is one shard of a multi-file split.
    ///
    /// A split needs several files stitched into one address range, which a
    /// single memory mapping cannot express. Refusing is deliberate: silently
    /// reading only the first shard would produce a model that loads and then
    /// answers with garbage.
    #[error("model is split into {count} files; shard 00001-of-{count:05}.gguf is only one part")]
    SplitUnsupported {
        /// Number of shards the metadata declared.
        count: u32,
    },
    /// A required tensor is absent.
    #[error("model is missing tensor {name}")]
    MissingTensor {
        /// Name that was looked for.
        name: Box<str>,
    },
    /// A tensor is present but has the wrong rank, dimensions or format.
    #[error("tensor {tensor} has {got}, expected {want}")]
    LayoutMismatch {
        /// Name of the offending tensor.
        tensor: Box<str>,
        /// Human-readable description of what was required.
        want: Box<str>,
        /// Human-readable description of what the file held.
        got: Box<str>,
    },
    /// A metadata key disagrees with the shape profile it was checked against.
    #[error("model metadata {key} is {got}, the shape profile says {want}")]
    ProfileMismatch {
        /// The metadata key that disagreed.
        key: Box<str>,
        /// Value the shape profile requires.
        want: u64,
        /// Value the file declared.
        got: u64,
    },
    /// A block cast was asked for a format the tensor is not stored in.
    #[error("tensor {tensor} is {got:?}, not {want:?}")]
    TypeMismatch {
        /// Name of the offending tensor.
        tensor: Box<str>,
        /// Format the caller asked for.
        want: DType,
        /// Format the tensor declares.
        got: DType,
    },
    /// A block cast needs an alignment the payload does not have.
    #[error("tensor {tensor} is not {want_align}-byte aligned")]
    Misaligned {
        /// Name of the offending tensor.
        tensor: Box<str>,
        /// Required alignment in bytes.
        want_align: usize,
    },
    /// A payload is not a whole number of blocks.
    #[error("tensor {tensor} holds {got} bytes, not a whole number of {want}-byte blocks")]
    SizeMismatch {
        /// Name of the offending tensor.
        tensor: Box<str>,
        /// Block size in bytes.
        want: usize,
        /// Payload length in bytes.
        got: usize,
    },
    /// The file's architecture is not the one the caller asked to bind.
    #[error("model architecture is {got}, expected {want}")]
    ArchMismatch {
        /// Architecture the caller required.
        want: &'static str,
        /// Architecture the file declared, empty when it declared none.
        got: Box<str>,
    },
    /// A text field was not valid UTF-8.
    #[error("{what} is not valid UTF-8")]
    NotUtf8 {
        /// What kind of field failed to decode.
        what: &'static str,
    },
    /// A metadata array index was out of range.
    #[error("metadata array index {index} is out of range for {len} entries")]
    ArrayIndex {
        /// The rejected index.
        index: usize,
        /// Number of entries in the array.
        len: usize,
    },
    /// An I/O failure while opening or sizing the file.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Result alias for this crate.
///
/// Deliberately not named `Result`: the crate root is glob-imported by
/// consumers, and a bare `Result` in scope would shadow the standard one.
pub type GgufResult<T> = Result<T, GgufError>;

/// Tensor dimensions, in the order the format stores them.
///
/// `dims[0]` is the fastest-varying dimension, which is also the inner
/// (reduction) dimension of the matrix multiply a weight takes part in. A
/// two-dimensional tensor is therefore indexed `y * dims[0] + x`.
///
/// Fields are private so the rank and the dimension array can never disagree,
/// which is the one way a hand-filled struct could make a kernel read the wrong
/// stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorShape {
    dims: [u64; MAX_DIMS],
    ndim: u8,
}

impl TensorShape {
    /// Builds a shape from `dims`.
    ///
    /// # Errors
    ///
    /// [`GgufError::BadRank`] if `dims` is empty or longer than [`MAX_DIMS`],
    /// [`GgufError::ZeroDim`] if any dimension is zero.
    pub fn new(dims: &[u64]) -> GgufResult<Self> {
        if dims.is_empty() || dims.len() > MAX_DIMS {
            return Err(GgufError::BadRank { ndim: dims.len() });
        }
        let mut out = [0_u64; MAX_DIMS];
        for (index, &dim) in dims.iter().enumerate() {
            if dim == 0 {
                return Err(GgufError::ZeroDim { index });
            }
            out[index] = dim;
        }
        Ok(Self {
            dims: out,
            // The length was checked against MAX_DIMS, so it fits in a u8.
            ndim: u8::try_from(dims.len()).unwrap_or(u8::MAX),
        })
    }

    /// The dimensions, with exactly [`TensorShape::ndim`] entries.
    #[must_use]
    pub fn dims(&self) -> &[u64] {
        &self.dims[..usize::from(self.ndim)]
    }

    /// Number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        usize::from(self.ndim)
    }

    /// One dimension, `None` past the last one.
    #[must_use]
    pub fn dim(&self, i: usize) -> Option<u64> {
        self.dims().get(i).copied()
    }

    /// Product of the dimensions.
    ///
    /// # Errors
    ///
    /// [`GgufError::ElementCountOverflow`] if the product overflows `u64`. A
    /// directory claiming such a tensor is rejected rather than wrapped.
    pub fn elements(&self) -> GgufResult<u64> {
        self.dims()
            .iter()
            .try_fold(1_u64, |acc, &d| acc.checked_mul(d))
            .ok_or(GgufError::ElementCountOverflow)
    }
}

/// One tensor's directory entry.
///
/// `bytes` is the whole payload, including the scale vectors the trellis
/// formats append after their tiles. There is no element-count field: it is
/// [`TensorShape::elements`], and storing it twice would let the two disagree.
#[derive(Debug, Clone)]
pub struct TensorDesc {
    /// Tensor name, the key the binding looks up.
    pub name: Box<str>,
    /// Storage format.
    pub dtype: DType,
    /// Dimensions, in format order.
    pub shape: TensorShape,
    /// Offset of the payload within the file mapping.
    pub abs_offset: u64,
    /// Payload length in bytes.
    pub bytes: u64,
}

/// A borrowed tensor payload.
///
/// The lifetime is the mapping's, so a view cannot outlive the file it points
/// into, and reading a weight costs no copy.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    /// Tensor name.
    pub name: &'a str,
    /// Storage format.
    pub dtype: DType,
    /// Raw payload, exactly `desc.bytes` long.
    pub data: &'a [u8],
    /// Dimensions, in format order.
    pub shape: TensorShape,
}

impl<'a> TensorView<'a> {
    /// Reinterprets the payload as whole blocks of `B`, with no copy.
    ///
    /// This is the intended way to read quantized weights: the format check and
    /// the alignment and length checks all happen here, so no caller does
    /// pointer arithmetic on a payload.
    ///
    /// # Errors
    ///
    /// [`GgufError::TypeMismatch`] if `B::DTYPE` is not this tensor's format,
    /// [`GgufError::Misaligned`] if the payload is not block-aligned,
    /// [`GgufError::SizeMismatch`] if it is not a whole number of blocks.
    pub fn blocks<B: QuantBlock>(&self) -> GgufResult<&'a [B]> {
        if B::DTYPE != self.dtype {
            return Err(GgufError::TypeMismatch {
                tensor: self.name.into(),
                want: B::DTYPE,
                got: self.dtype,
            });
        }
        bytemuck::try_cast_slice(self.data).map_err(|err| match err {
            bytemuck::PodCastError::AlignmentMismatch => GgufError::Misaligned {
                tensor: self.name.into(),
                want_align: align_of::<B>(),
            },
            _ => GgufError::SizeMismatch {
                tensor: self.name.into(),
                want: size_of::<B>(),
                got: self.data.len(),
            },
        })
    }

    /// The payload as `f32` weights.
    ///
    /// # Errors
    ///
    /// [`GgufError::TypeMismatch`] unless the format is [`DType::F32`], plus the
    /// alignment and length errors of [`TensorView::blocks`].
    pub fn as_f32(&self) -> GgufResult<&'a [f32]> {
        self.typed_slice::<f32>(DType::F32)
    }

    /// The payload as `f16` weights.
    ///
    /// # Errors
    ///
    /// [`GgufError::TypeMismatch`] unless the format is [`DType::F16`], plus the
    /// alignment and length errors of [`TensorView::blocks`].
    pub fn as_f16(&self) -> GgufResult<&'a [half::f16]> {
        self.typed_slice::<half::f16>(DType::F16)
    }

    /// Shared body of the two typed accessors: format check, then a checked
    /// zero-copy cast.
    fn typed_slice<T: bytemuck::Pod>(&self, want: DType) -> GgufResult<&'a [T]> {
        if self.dtype != want {
            return Err(GgufError::TypeMismatch {
                tensor: self.name.into(),
                want,
                got: self.dtype,
            });
        }
        bytemuck::try_cast_slice(self.data).map_err(|err| match err {
            bytemuck::PodCastError::AlignmentMismatch => GgufError::Misaligned {
                tensor: self.name.into(),
                want_align: align_of::<T>(),
            },
            _ => GgufError::SizeMismatch {
                tensor: self.name.into(),
                want: size_of::<T>(),
                got: self.data.len(),
            },
        })
    }
}

/// A read-only mapping of a model file.
///
/// Private to the crate: the only way to obtain one is through
/// [`GgufFile::open`], and the only way to hand one outside is
/// [`GgufFile::mapping`], which gives out `Arc<dyn HostBytes>` rather than the
/// concrete type.
pub(crate) struct Mapping {
    path: PathBuf,
    map: Mmap,
    /// Kept open so the page cache can be advised about the mapping's pages.
    file: File,
}

impl Mapping {
    /// Maps `path` read-only. `shared` selects a file-backed shared mapping
    /// (the one a device backend can wrap without copying); a private mapping
    /// cannot be wrapped that way.
    fn open(path: &Path, shared: bool) -> GgufResult<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        if len < MIN_FILE_BYTES {
            return Err(GgufError::Truncated { pos: 0 });
        }
        // SAFETY: the mapping is read-only and taken over a file this function
        // opened read-only, so nothing in this process can write through it.
        // The remaining hazard is outside the language: if another process
        // truncates or overwrites the file while the mapping is alive, every
        // &[u8] handed out from here becomes a window onto changed or unmapped
        // memory, which is undefined behaviour no type system can prevent. The
        // invariant is therefore a caller-side one, stated on GgufFile::open
        // and open_shared: a model file must not be mutated or truncated for as
        // long as the GgufFile that mapped it is alive.
        let map = unsafe {
            if shared {
                Mmap::map(&file)?
            } else {
                memmap2::MmapOptions::new().map_copy_read_only(&file)?
            }
        };
        Ok(Self {
            path: path.to_path_buf(),
            map,
            file,
        })
    }

    /// Unmaps `range` from this process and drops its pages from the page
    /// cache, where the kernel will. See [`release_pages`].
    fn release(&self, range: Range<usize>) {
        release_pages(&self.map, &self.file, range);
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.map
    }

    fn len(&self) -> u64 {
        u64::try_from(self.map.len()).unwrap_or(u64::MAX)
    }
}

/// Unmaps `range` of `map` from this process and drops its pages from the
/// page cache, where the kernel will.
///
/// Both halves are needed: the cache keeps a page any process still maps, so
/// advising the file alone leaves every page this process faulted in. Advice
/// only - the bytes stay readable, and a later read faults them back in from
/// the file - so a failure is not reported. `file` is the descriptor `map` was
/// taken over.
#[cfg(target_os = "linux")]
pub(crate) fn release_pages(map: &Mmap, file: &File, range: Range<usize>) {
    use std::os::fd::AsRawFd;
    const PAGE: usize = 4096;
    let Some(bytes) = map.get(range.clone()) else {
        return;
    };
    let start = bytes.as_ptr() as usize;
    let aligned = start.next_multiple_of(PAGE);
    let end = (start + bytes.len()) & !(PAGE - 1);
    if end > aligned {
        // SAFETY: the range lies inside a read-only shared file mapping, on
        // whole pages that hold nothing but bytes of `range`. `MADV_DONTNEED`
        // on such a mapping only drops the page-table entries: the next access
        // faults the same file bytes back in, so every slice handed out from
        // the mapping stays valid and unchanged.
        let _ = unsafe {
            libc::madvise(
                aligned as *mut libc::c_void,
                end - aligned,
                libc::MADV_DONTNEED,
            )
        };
    }
    let offset = libc::off_t::try_from(range.start).unwrap_or(libc::off_t::MAX);
    let len = libc::off_t::try_from(bytes.len()).unwrap_or(0);
    let fd = file.as_raw_fd();
    // SAFETY: advice on a file descriptor the caller owns; it reads and writes
    // no memory of this process.
    let _ = unsafe { libc::posix_fadvise(fd, offset, len, libc::POSIX_FADV_DONTNEED) };
}

/// Nothing to release on a target without the advice calls.
#[cfg(not(target_os = "linux"))]
pub(crate) const fn release_pages(_map: &Mmap, _file: &File, _range: Range<usize>) {}

impl AsRef<[u8]> for Mapping {
    fn as_ref(&self) -> &[u8] {
        &self.map
    }
}

impl fmt::Debug for Mapping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mapping")
            .field("path", &self.path)
            .field("len", &self.map.len())
            .finish_non_exhaustive()
    }
}

/// An open model file: the mapping, the metadata table and the tensor
/// directory.
///
/// Cloning is not cheap and is not needed - the mapping is shared behind an
/// [`Arc`], and [`GgufFile::mapping`] hands that same `Arc` to a device backend
/// so imported weights keep the file alive without a lifetime dance.
///
/// # Mutating the file
///
/// The bytes handed out by [`GgufFile::get`] borrow this mapping. Replacing,
/// truncating or rewriting the file while any of them is alive is undefined
/// behaviour, so treat the path as immutable for as long as the `GgufFile` (or
/// anything imported from its [`GgufFile::mapping`]) exists.
#[derive(Debug)]
pub struct GgufFile {
    mapping: Arc<Mapping>,
    version: u32,
    alignment: u64,
    tensor_data_pos: u64,
    metadata: GgufMetadata,
    tensors: Vec<TensorDesc>,
    index: HashMap<Box<str>, usize>,
}

impl GgufFile {
    /// Opens a model file with a private read-only mapping.
    ///
    /// This is the right choice for host-side reading. A device backend needs
    /// [`GgufFile::open_shared`] instead, because a private mapping cannot be
    /// wrapped by the device without a copy.
    ///
    /// # Mutating the file
    ///
    /// Do not point this at a file another process is writing, and do not
    /// truncate or replace the file while the returned value (or anything
    /// borrowed from it) is alive. The mapping is a window onto the file, and
    /// the borrow checker cannot see that the file has other writers.
    ///
    /// # Errors
    ///
    /// [`GgufError::Io`] if the file cannot be opened or sized;
    /// [`GgufError::Truncated`], [`GgufError::NotGguf`] or
    /// [`GgufError::Version`] for a short, foreign or wrong-version header;
    /// [`GgufError::SplitUnsupported`] for a multi-file split; any other variant
    /// for a malformed metadata table or tensor directory.
    pub fn open(path: impl AsRef<Path>) -> GgufResult<Self> {
        Self::open_with(path.as_ref(), false)
    }

    /// Opens a model file with a *shared* read-only mapping, the form a device
    /// backend can wrap without copying.
    ///
    /// # Mutating the file
    ///
    /// Same invariant as [`GgufFile::open`], and stronger in practice: a device
    /// buffer may still be reading the mapping after the code that created it
    /// has moved on. Never mutate or replace a model file that is loaded.
    ///
    /// # Errors
    ///
    /// Same as [`GgufFile::open`].
    pub fn open_shared(path: impl AsRef<Path>) -> GgufResult<Self> {
        Self::open_with(path.as_ref(), true)
    }

    /// The file's `general.architecture`, read without parsing the rest.
    ///
    /// For a caller asking only what a file is: [`GgufFile::open`] parses the
    /// whole metadata table and tensor directory, which for a large model's
    /// vocabulary arrays is real work for a yes-or-no answer. This maps the
    /// file and walks the table only as far as the key. `Ok(None)` when the
    /// file states no architecture.
    ///
    /// # Errors
    ///
    /// [`GgufError::Io`] if the file cannot be opened or mapped, and the
    /// header and metadata errors of [`GgufFile::open`] for whatever part of
    /// the table it reads.
    pub fn read_architecture(path: impl AsRef<Path>) -> GgufResult<Option<Box<str>>> {
        let mapping = Mapping::open(path.as_ref(), false)?;
        let mut cursor = Cursor::new(mapping.bytes());
        let header = read_header(&mut cursor)?;
        crate::metadata::find_str(&mut cursor, header.n_kv, "general.architecture")
    }

    /// Shared body of the two openers.
    fn open_with(path: &Path, shared: bool) -> GgufResult<Self> {
        let mapping = Arc::new(Mapping::open(path, shared)?);
        let file_len = mapping.len();

        let mut cursor = Cursor::new(mapping.bytes());
        let header = read_header(&mut cursor)?;

        let metadata = GgufMetadata::parse(&mut cursor, &mapping, header.n_kv)?;
        let alignment = declared_alignment(&metadata)?;

        if let Some(count) = metadata.get_u32("split.count")
            && count > SINGLE_FILE
        {
            return Err(GgufError::SplitUnsupported { count });
        }

        let (tensors, tensor_data_pos) =
            parse_tensor_directory(&mut cursor, file_len, alignment, header.n_tensors)?;

        let index = build_index(&tensors);
        tracing::debug!(
            path = %path.display(),
            version = header.version,
            tensors = tensors.len(),
            metadata = header.n_kv,
            alignment,
            "opened model file"
        );

        Ok(Self {
            mapping,
            version: header.version,
            alignment,
            tensor_data_pos,
            metadata,
            tensors,
            index,
        })
    }

    /// Header format version, always 3 for a value that opened successfully,
    /// since that is the only version [`GgufFile::open`] accepts.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Payload alignment declared by `general.alignment`, or 32 when absent.
    #[must_use]
    pub const fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Offset where tensor payloads begin, after the directory.
    #[must_use]
    pub const fn tensor_data_pos(&self) -> u64 {
        self.tensor_data_pos
    }

    /// The metadata table.
    #[must_use]
    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    /// Every tensor, in directory order.
    #[must_use]
    pub fn tensors(&self) -> &[TensorDesc] {
        &self.tensors
    }

    /// Directory entry for `name`, from a hash index built once at open time.
    #[must_use]
    pub fn find(&self, name: &str) -> Option<&TensorDesc> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    /// Borrow `name`'s payload.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<TensorView<'_>> {
        let desc = self.find(name)?;
        let range = self.clamped_range(desc);
        self.mapping.bytes().get(range).map(|data| TensorView {
            name: &desc.name,
            dtype: desc.dtype,
            data,
            shape: desc.shape,
        })
    }

    /// The whole mapping as shareable host bytes, for zero-copy weight
    /// registration with a device backend.
    ///
    /// Cloning the returned `Arc` is what keeps the file mapped while the
    /// device is still reading through it.
    #[must_use]
    pub fn mapping(&self) -> Arc<dyn HostBytes> {
        let shared: Arc<Mapping> = Arc::clone(&self.mapping);
        shared
    }

    /// Hands the pages of `range` back to the kernel once their bytes have been
    /// copied somewhere else.
    ///
    /// A loader whose device copies every weight calls this after each import:
    /// the file is larger than the host memory left beside the copies, so pages
    /// kept cached are pages the device's next allocation has to reclaim. The
    /// bytes stay readable - a later access reads them from the file again.
    pub fn release_cached(&self, range: Range<usize>) {
        self.mapping.release(range);
    }

    /// Byte range `desc` occupies within [`GgufFile::mapping`].
    ///
    /// The range is clamped to the mapping, so a descriptor taken from a
    /// different model yields a truncated or empty range instead of a panic.
    #[must_use]
    pub fn range_of(&self, desc: &TensorDesc) -> Range<usize> {
        self.clamped_range(desc)
    }

    /// Directory entry and borrowed payload together.
    ///
    /// # Errors
    ///
    /// [`GgufError::MissingTensor`] if `name` is absent.
    pub fn require(&self, name: &str) -> GgufResult<TensorView<'_>> {
        self.get(name)
            .ok_or_else(|| GgufError::MissingTensor { name: name.into() })
    }

    fn clamped_range(&self, desc: &TensorDesc) -> Range<usize> {
        let len = self.mapping.bytes().len();
        let start = usize::try_from(desc.abs_offset).unwrap_or(len);
        let end = start
            .saturating_add(usize::try_from(desc.bytes).unwrap_or(0))
            .min(len);
        start..end
    }
}

/// Builds the name to directory-index map behind [`GgufFile::find`].
///
/// A model of this size has hundreds of tensors and a forward pass looks up
/// hundreds of names per layer, so a linear scan per lookup is quadratic work
/// done at the worst possible time.
fn build_index(tensors: &[TensorDesc]) -> HashMap<Box<str>, usize> {
    let mut index = HashMap::with_capacity(tensors.len());
    for (i, desc) in tensors.iter().enumerate() {
        // A duplicate name is a corrupt directory, but the first entry is the
        // one the payload offsets were validated for, so keep the first.
        index.entry(desc.name.clone()).or_insert(i);
    }
    index
}

/// Reads `general.alignment`, defaulting to 32.
///
/// The value has to be a power of two no larger than a page: the payload base
/// is derived from it, and the assumption that it divides a page is what lets a
/// device backend slice the mapping at tensor boundaries. An out-of-range value
/// is reported rather than clamped, because clamping would read every tensor at
/// the wrong offset.
fn declared_alignment(metadata: &GgufMetadata) -> GgufResult<u64> {
    let Some(raw) = metadata.get_u32("general.alignment") else {
        return Ok(DEFAULT_ALIGNMENT);
    };
    if raw == 0 || !raw.is_power_of_two() || raw > MAX_ALIGNMENT {
        return Err(GgufError::LayoutMismatch {
            tensor: Box::from("general.alignment"),
            want: Box::from(format!("a power of two in 1..={MAX_ALIGNMENT}")),
            got: Box::from(raw.to_string()),
        });
    }
    Ok(u64::from(raw))
}

/// The four header fields every parse step depends on.
struct Header {
    version: u32,
    n_tensors: u64,
    n_kv: u64,
}

/// Reads and validates the fixed-size header.
fn read_header(cursor: &mut Cursor<'_>) -> GgufResult<Header> {
    let magic = cursor.u32()?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::NotGguf);
    }
    let version = cursor.u32()?;
    if version != SUPPORTED_VERSION {
        return Err(GgufError::Version { got: version });
    }
    let n_tensors = cursor.u64()?;
    let n_kv = cursor.u64()?;
    Ok(Header {
        version,
        n_tensors,
        n_kv,
    })
}

/// Reads the tensor directory and rebases relative payload offsets.
///
/// Returns the descriptors and the offset at which payloads begin. Every
/// descriptor is checked to lie inside the file here, which is what lets
/// [`GgufFile::get`] hand out slices without re-validating.
fn parse_tensor_directory(
    cursor: &mut Cursor<'_>,
    file_len: u64,
    alignment: u64,
    n_tensors: u64,
) -> GgufResult<(Vec<TensorDesc>, u64)> {
    // Every directory entry costs at least a length word plus a rank, a format
    // id and an offset, so a count larger than the bytes remaining cannot be
    // real. Checking first keeps a tiny file from asking for a huge allocation.
    if n_tensors > cursor.remaining() {
        return Err(GgufError::Truncated {
            pos: cursor.position(),
        });
    }

    let mut descs = Vec::with_capacity(usize::try_from(n_tensors).unwrap_or(0));
    for _ in 0..n_tensors {
        let name = cursor.string("tensor name")?;
        let ndim = cursor.u32()?;
        let rank = usize::try_from(ndim).unwrap_or(usize::MAX);
        if rank == 0 || rank > MAX_DIMS {
            return Err(GgufError::DimLimit { tensor: name, ndim });
        }
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            dims.push(cursor.u64()?);
        }
        // The rank was checked above, so the only failure left is a declared
        // zero dimension - reported as a layout problem because it names the
        // tensor, which a bare rank error would not.
        let shape = TensorShape::new(&dims).map_err(|err| match err {
            GgufError::ZeroDim { index } => GgufError::LayoutMismatch {
                tensor: name.clone(),
                want: Box::from("no zero dimensions"),
                got: Box::from(format!("dimension {index} is zero")),
            },
            other => other,
        })?;
        let dtype_id = cursor.u32()?;
        let dtype = DType::from_gguf_id(dtype_id).ok_or(GgufError::UnsupportedType {
            tensor: name.clone(),
            dtype_id,
        })?;
        let rel_offset = cursor.u64()?;
        let bytes = quant::block_bytes_for(dtype, &shape).ok_or(GgufError::TensorTooLarge {
            tensor: name.clone(),
        })?;
        descs.push(TensorDesc {
            name,
            dtype,
            shape,
            // Relative for now; rebased once the payload base is known.
            abs_offset: rel_offset,
            bytes,
        });
    }

    let data_pos = align_up(cursor.position(), alignment);
    for desc in &mut descs {
        desc.abs_offset =
            data_pos
                .checked_add(desc.abs_offset)
                .ok_or(GgufError::TensorTooLarge {
                    tensor: desc.name.clone(),
                })?;
        let end = desc
            .abs_offset
            .checked_add(desc.bytes)
            .ok_or(GgufError::OutOfBounds {
                tensor: desc.name.clone(),
            })?;
        if end > file_len {
            return Err(GgufError::OutOfBounds {
                tensor: desc.name.clone(),
            });
        }
    }

    Ok((descs, data_pos))
}

/// Rounds `value` up to a multiple of `alignment`.
///
/// `alignment` is validated as a power of two no larger than a page when the
/// metadata is read, so the rounding can never reach `u64::MAX` and saturating
/// here is a formality rather than a behaviour.
fn align_up(value: u64, alignment: u64) -> u64 {
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value.saturating_add(alignment - rem)
    }
}

/// A bounds-checked little-endian reader over a mapping.
///
/// Reads never panic and never return a partial value: the first failure is a
/// [`GgufError::Truncated`] carrying the offset it happened at, which is the
/// only diagnostic that makes a truncated model actionable.
pub(crate) struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Current read offset. `usize` never exceeds `u64::MAX` on any supported
    /// platform, so the fallback is unreachable but keeps the conversion from
    /// becoming a panic path.
    pub(crate) fn position(&self) -> u64 {
        u64::try_from(self.pos).unwrap_or(u64::MAX)
    }

    /// Bytes left in the mapping.
    pub(crate) fn remaining(&self) -> u64 {
        u64::try_from(self.bytes.len() - self.pos).unwrap_or(u64::MAX)
    }

    fn take(&mut self, n: usize) -> GgufResult<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(GgufError::Truncated {
            pos: self.position(),
        })?;
        let slice = self.bytes.get(self.pos..end).ok_or(GgufError::Truncated {
            pos: self.position(),
        })?;
        self.pos = end;
        Ok(slice)
    }

    pub(crate) fn skip(&mut self, n: usize) -> GgufResult<()> {
        self.take(n).map(|_| ())
    }

    fn int<const N: usize>(&mut self) -> GgufResult<[u8; N]> {
        let bytes = self.take(N)?;
        let mut out = [0_u8; N];
        out.copy_from_slice(bytes);
        Ok(out)
    }

    pub(crate) fn u8(&mut self) -> GgufResult<u8> {
        Ok(self.int::<1>()?[0])
    }

    pub(crate) fn i8(&mut self) -> GgufResult<i8> {
        Ok(i8::from_le_bytes(self.int::<1>()?))
    }

    pub(crate) fn u16(&mut self) -> GgufResult<u16> {
        Ok(u16::from_le_bytes(self.int()?))
    }

    pub(crate) fn i16(&mut self) -> GgufResult<i16> {
        Ok(i16::from_le_bytes(self.int()?))
    }

    pub(crate) fn u32(&mut self) -> GgufResult<u32> {
        Ok(u32::from_le_bytes(self.int()?))
    }

    pub(crate) fn i32(&mut self) -> GgufResult<i32> {
        Ok(i32::from_le_bytes(self.int()?))
    }

    pub(crate) fn u64(&mut self) -> GgufResult<u64> {
        Ok(u64::from_le_bytes(self.int()?))
    }

    pub(crate) fn i64(&mut self) -> GgufResult<i64> {
        Ok(i64::from_le_bytes(self.int()?))
    }

    pub(crate) fn f32(&mut self) -> GgufResult<f32> {
        Ok(f32::from_le_bytes(self.int()?))
    }

    pub(crate) fn f64(&mut self) -> GgufResult<f64> {
        Ok(f64::from_le_bytes(self.int()?))
    }

    /// A length-prefixed UTF-8 string.
    pub(crate) fn string(&mut self, what: &'static str) -> GgufResult<Box<str>> {
        let len = self.u64()?;
        let len = usize::try_from(len).map_err(|_| GgufError::Truncated {
            pos: self.position(),
        })?;
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes)
            .map(Into::into)
            .map_err(|_| GgufError::NotUtf8 { what })
    }
}
