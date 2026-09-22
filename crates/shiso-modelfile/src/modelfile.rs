//! One handle over both checkpoint containers.
//!
//! A model on disk is either a single self-describing file or a directory of
//! shards plus a config, and an engine that has to branch on which one it got
//! ends up branching in more places than it expected: opening, describing,
//! reading bytes, and deciding whether those bytes can be handed to a device
//! without a copy. [`ModelFile`] collapses the first three into one type and
//! makes the fourth an explicit question with a one-line answer.
//!
//! # The one difference that cannot be hidden
//!
//! A single-file checkpoint's weights are already in a format the kernels read,
//! so its bytes are a window onto a file mapping and a device can wrap them
//! in place. A directory checkpoint's weights are packed, so its bytes are a
//! buffer this crate had to build, and wrapping them in place would hand the
//! device memory that is about to be freed.
//!
//! That is what [`ModelFile::is_mmap_backed`] reports and what
//! [`ModelFile::tensor_bytes`] shows in its return type. Hiding the difference
//! behind a uniform "give me the bytes" call would not make it go away; it
//! would move it to whoever eventually frees the buffer.
//!
//! # The one caller that does not have to care
//!
//! A loader moving a whole model onto a device is the exception, and
//! [`ModelFile::tensor_host`] is why. A device import already takes shareable
//! host memory plus a range, and those two together express both cases exactly:
//! a window onto the shared mapping for one container, a buffer of its own for
//! the other, with ownership carried by the `Arc` either way. So the loader asks
//! one question, gets one answer type, and the branch stays here - which is the
//! only place that can still say which of the two it took.

use std::borrow::Cow;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use shiso_core::HostBytes;

use crate::name_map::{IdentityNames, NameMap};
use crate::reader::{GgufError, GgufFile, GgufResult, TensorDesc, TensorView};
use crate::safetensors::{AffineTriple, SafetensorsFile};
use crate::source::TensorSource;

/// An open checkpoint, whichever container it came in.
///
/// The enum is `#[non_exhaustive]` because the container list grows: match on
/// [`ModelFile::is_mmap_backed`] rather than on the variant unless the variant
/// is genuinely what you need.
#[derive(Debug)]
#[non_exhaustive]
pub enum ModelFile {
    /// A single self-describing checkpoint file.
    Gguf(GgufFile),
    /// A checkpoint directory of shards, a config, and packed weights.
    Safetensors(SafetensorsFile),
}

impl ModelFile {
    /// Opens whichever container `path` names.
    ///
    /// A directory is a sharded checkpoint, anything else is a single file.
    /// The choice is made on the path rather than by sniffing content: a
    /// directory cannot be a single-file checkpoint and a file cannot be a
    /// directory of shards, so there is nothing to guess and no way for the
    /// guess to be wrong on a truncated download.
    ///
    /// The single-file path takes a *shared* mapping, which is the form a
    /// device backend can wrap without copying.
    ///
    /// # Mutating the files
    ///
    /// Everything this opens stays mapped for as long as the returned value
    /// lives. Do not rewrite, truncate or replace a loaded checkpoint; see
    /// [`GgufFile::open_shared`] and [`SafetensorsFile::open`].
    ///
    /// Opens whichever container `path` names, using `names` to translate
    /// tensor names in a sharded checkpoint directory.
    ///
    /// # Errors
    ///
    /// Whatever the chosen container reports: see [`GgufFile::open_shared`] and
    /// [`SafetensorsFile::open_with`].
    pub fn open_with(path: impl AsRef<Path>, names: Arc<dyn NameMap>) -> GgufResult<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            tracing::debug!(path = %path.display(), "opening a sharded checkpoint");
            Ok(Self::Safetensors(SafetensorsFile::open_with(path, names)?))
        } else {
            tracing::debug!(path = %path.display(), "opening a single-file checkpoint");
            Ok(Self::Gguf(GgufFile::open_shared(path)?))
        }
    }

    /// Opens whichever container `path` names with the default identity name map.
    ///
    /// # Errors
    ///
    /// Whatever the chosen container reports: see [`GgufFile::open_shared`] and
    /// [`SafetensorsFile::open`].
    pub fn open(path: impl AsRef<Path>) -> GgufResult<Self> {
        Self::open_with(path, Arc::new(IdentityNames))
    }

    /// Returns the packed affine triple for `name`, when this container is a
    /// safetensors export holding one.
    ///
    /// # Errors
    ///
    /// Shard I/O or bounds errors; returns `Ok(None)` for single-file GGUF checkpoints,
    /// or when `name` is not a packed affine triple in a safetensors checkpoint.
    pub fn affine_triple(&self, name: &str) -> GgufResult<Option<AffineTriple<'_>>> {
        match self {
            Self::Gguf(_) => Ok(None),
            Self::Safetensors(file) => file.affine_triple(name),
        }
    }

    /// Whether tensor bytes borrow a file mapping.
    ///
    /// `true` means [`ModelFile::tensor_bytes`] hands back a window onto mapped
    /// pages, which a device backend can wrap without copying and which stays
    /// valid for as long as this value does. `false` means every tensor is a
    /// buffer this crate synthesised, so a device has to copy it before the
    /// buffer is dropped.
    #[must_use]
    pub fn is_mmap_backed(&self) -> bool {
        matches!(self, Self::Gguf(_))
    }

    /// Bytes for one tensor, in the format its descriptor declares.
    ///
    /// Borrowed straight from the mapping when the checkpoint already stores
    /// the tensor that way, owned when it had to be widened or unpacked. The
    /// `Cow` is the whole point: an out-parameter would force every caller to
    /// own a scratch buffer whether or not one was needed, and would hide which
    /// of the two happened from the caller that has to decide whether the
    /// device may wrap the bytes in place.
    ///
    /// # Errors
    ///
    /// [`GgufError::MissingTensor`] when no tensor has that name, plus the
    /// decode errors of [`SafetensorsFile::tensor_bytes`] for a sharded
    /// checkpoint.
    pub fn tensor_bytes(&self, name: &str) -> GgufResult<Cow<'_, [u8]>> {
        match self {
            Self::Gguf(file) => {
                let view = file
                    .get(name)
                    .ok_or_else(|| GgufError::MissingTensor { name: name.into() })?;
                Ok(Cow::Borrowed(view.data))
            }
            Self::Safetensors(file) => file.tensor_bytes(name),
        }
    }

    /// Host bytes a device may import one tensor from, and the range of them it
    /// occupies.
    ///
    /// This is [`ModelFile::tensor_bytes`] shaped for the one caller that has to
    /// hand the bytes to a device rather than read them: a device import takes
    /// shareable host memory plus a range, because the range is what the device
    /// wraps and the `Arc` is what keeps the memory alive underneath it for as
    /// long as the device buffer lives.
    ///
    /// Both containers answer it, and the difference between them survives in
    /// what comes back rather than in a branch the caller writes:
    ///
    /// - a single-file checkpoint returns *the whole mapping* and a range inside
    ///   it, so every tensor of a model shares one `Arc` and the device wraps
    ///   mapped pages in place. Nothing is copied and nothing is read until the
    ///   device touches it - a gigabytes-large checkpoint is served out of the
    ///   page cache;
    /// - a directory checkpoint returns *one decoded buffer per tensor* and its
    ///   whole range, because the packed weights are in no format a kernel reads
    ///   and the unpacked bytes exist only because this call made them.
    ///
    /// # Why a descriptor and not a name
    ///
    /// The mapped case needs the payload's offset and length, which live on the
    /// descriptor the caller already holds; looking them up again from a name
    /// would be a second directory search that can fail for a reason the caller
    /// has already ruled out. The decoded case needs only the name, and takes it
    /// from the same descriptor.
    ///
    /// # Errors
    ///
    /// Nothing for a mapped checkpoint: a descriptor from another file yields a
    /// range clamped to the mapping rather than a failure, which is what
    /// [`GgufFile::range_of`] promises. Otherwise the decode errors of
    /// [`ModelFile::tensor_bytes`].
    pub fn tensor_host(&self, desc: &TensorDesc) -> GgufResult<(Arc<dyn HostBytes>, Range<usize>)> {
        match self {
            Self::Gguf(file) => Ok((file.mapping(), file.range_of(desc))),
            Self::Safetensors(file) => {
                // `into_owned` rather than a match on the `Cow`: the borrowed
                // arm is the handful of vectors the checkpoint already stores as
                // `f32`, which are kilobytes, and an `Arc<dyn HostBytes>` cannot
                // borrow anyway. Paying a copy on those to keep one code path is
                // the right trade; paying one on a weight matrix would not be,
                // and the decode already had to allocate those.
                let bytes = file.tensor_bytes(&desc.name)?.into_owned();
                let range = 0..bytes.len();
                Ok((Arc::new(bytes), range))
            }
        }
    }
}

impl TensorSource for ModelFile {
    fn architecture(&self) -> Option<&str> {
        match self {
            Self::Gguf(file) => file.metadata().architecture(),
            Self::Safetensors(file) => file.architecture(),
        }
    }

    fn arch_u32(&self, suffix: &str) -> Option<u32> {
        match self {
            Self::Gguf(file) => file.metadata().arch_u32(suffix),
            Self::Safetensors(file) => file.arch_u32(suffix),
        }
    }

    fn find(&self, name: &str) -> Option<&TensorDesc> {
        match self {
            Self::Gguf(file) => file.find(name),
            Self::Safetensors(file) => file.find(name),
        }
    }

    fn get(&self, name: &str) -> Option<TensorView<'_>> {
        match self {
            Self::Gguf(file) => file.get(name),
            Self::Safetensors(file) => file.get(name),
        }
    }
}
