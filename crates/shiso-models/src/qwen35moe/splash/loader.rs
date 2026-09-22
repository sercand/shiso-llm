//! A Splash `.bin` into device buffers, one section at a time.
//!
//! The file is read once and every section is a span of that one reading, so a
//! layer's forty-odd weights cost one read and no per-tensor host copy: the
//! spans are handed to [`shiso_core::import_host`], which on a unified-memory
//! device wraps the pages the file is already in. The package pads every
//! section start to `format.section_alignment_bytes` - 16,384 in the shipped
//! one, which is also the page size on Apple silicon - so the wrap is a wrap
//! rather than a copy for every section that is not the last in its file.
//!
//! A span that does not fit its file is refused by section name rather than
//! truncated, and a file whose length is not the length the layout declares is
//! refused before any section is looked at. Both are the same failure seen from
//! two distances: a truncated weight is a model that answers.
//!
//! # What this module deliberately does not do
//!
//! It does not decode, reshape, or interpret a single byte. A section's dtype,
//! its group size, the padded row width it was written at and the stride from
//! one expert's slab to the next are carried through untouched on
//! [`SectionBuffer::section`], because what they mean is the codec's business
//! and not the reader's. In particular nothing here reads `shape[0]` as a
//! width: on every quantized section in the shipped package that number is the
//! padded width and [`Section::out_dim`] is the live one.

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::{Backend, CoreError, DeviceBuffer, HostBytes, import_host};

use super::layout::{FileLayout, Section};

/// What a section could not be read as, naming it.
///
/// Its own error rather than [`CoreError::ShapeMismatch`] for one reason:
/// `ShapeMismatch::field` is a `&'static str`, and the only way to put a
/// section name read out of a document into one is to leak it. This is the
/// shape of [`super::VerifyError`], next door, for the same reason that one has
/// it - a refusal about a named thing should carry the name.
#[derive(Debug, thiserror::Error)]
pub enum SectionError {
    /// The file could not be read.
    #[error("{path}: {source}")]
    Io {
        /// The file the layout names.
        path: PathBuf,
        /// The underlying failure.
        source: std::io::Error,
    },
    /// The file is not the length the layout declares.
    #[error("{path}: the layout declares {want} bytes, the file holds {got}")]
    Size {
        /// The file the layout names.
        path: PathBuf,
        /// Declared length.
        want: u64,
        /// Observed length.
        got: u64,
    },
    /// The span runs past the end of its file.
    #[error("{name}: declared {want} bytes, {got} remain in the file")]
    OutOfBounds {
        /// The section's name, as the package spells it.
        name: String,
        /// Bytes the section declared.
        want: u64,
        /// Bytes actually left after the offset.
        got: u64,
    },
    /// The device would not take the span.
    #[error("{name}: {source}")]
    Device {
        /// The section's name, as the package spells it.
        name: String,
        /// What the backend said.
        source: CoreError,
    },
}

/// The bytes of `section` within `file`.
///
/// # Errors
///
/// [`SectionError::OutOfBounds`] naming the section when the span does not fit.
pub fn slice_section<'a>(file: &'a [u8], section: &Section) -> Result<&'a [u8], SectionError> {
    let span = span_of(file.len(), section)?;
    Ok(&file[span])
}

/// One section's bytes on the device, with what the package said about them.
///
/// The [`Section`] rides along rather than being consumed here because every
/// number a codec needs - the dtype, the group size, the padded row width, the
/// live width, the stride from one expert's slab to the next - is on it, and a
/// buffer that arrived without them is a slab of bytes nobody can decode.
#[derive(Debug)]
pub struct SectionBuffer {
    /// What the layout says this span is.
    pub section: Section,
    /// The span, on the device.
    pub buffer: DeviceBuffer,
}

/// Every section of one `.bin`, on the device.
#[derive(Debug)]
pub struct LayerBuffers {
    /// The file this came from, as the layout spells it: relative to the
    /// package root.
    pub path: String,
    /// One entry per section, in the layout's order.
    pub sections: Vec<SectionBuffer>,
}

impl LayerBuffers {
    /// The section called `name`.
    ///
    /// `None` is an answer rather than a failure, the same way
    /// [`FileLayout::section`] is: a recurrent layer file carries no
    /// `attention_input`, and asking is how a caller finds that out.
    #[must_use]
    pub fn section(&self, name: &str) -> Option<&SectionBuffer> {
        self.sections.iter().find(|held| held.section.name == name)
    }

    /// Bytes this file put on the device, summed over its sections.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.sections
            .iter()
            .map(|held| held.section.bytes)
            .fold(0, u64::saturating_add)
    }
}

/// Reads one `.bin` of a Splash package onto `backend`.
///
/// `root` is the package root and `file.path` is relative to it, which is how
/// `layout.json` writes it. Through a Hugging Face cache every one of those
/// paths is a symlink into the blob store; the read follows it.
///
/// # Errors
///
/// [`SectionError::Io`] naming the file when it cannot be read,
/// [`SectionError::Size`] when it is not the declared length,
/// [`SectionError::OutOfBounds`] naming the first section that does not fit,
/// and [`SectionError::Device`] naming the first section the backend refused.
pub fn load_layer(
    backend: &Arc<dyn Backend>,
    root: &Path,
    file: &FileLayout,
) -> Result<LayerBuffers, SectionError> {
    let path = root.join(&file.path);
    let bytes = std::fs::read(&path).map_err(|source| SectionError::Io {
        path: path.clone(),
        source,
    })?;
    let got = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if got != file.bytes {
        return Err(SectionError::Size {
            path,
            want: file.bytes,
            got,
        });
    }
    let len = bytes.len();
    // One `Arc` for the whole file, cloned into every buffer taken from it:
    // that is what keeps the host pages alive for exactly as long as the device
    // can still read any span of them.
    let host: Arc<dyn HostBytes> = Arc::new(bytes);
    let mut sections = Vec::with_capacity(file.sections.len());
    for section in &file.sections {
        let span = span_of(len, section)?;
        let buffer = import_host(backend, Arc::clone(&host), span).map_err(|source| {
            SectionError::Device {
                name: section.name.clone(),
                source,
            }
        })?;
        sections.push(SectionBuffer {
            section: section.clone(),
            buffer,
        });
    }
    Ok(LayerBuffers {
        path: file.path.clone(),
        sections,
    })
}

/// The section's byte range within a file of `file_len` bytes.
///
/// Saturating rather than wrapping on both conversions: a 64-bit offset that
/// does not fit a `usize` is a span no reading of this file can satisfy, and
/// `usize::MAX` takes it to the same refusal as a span that simply runs off the
/// end rather than to a wrapped offset that lands inside the file.
fn span_of(file_len: usize, section: &Section) -> Result<Range<usize>, SectionError> {
    let start = usize::try_from(section.offset).unwrap_or(usize::MAX);
    let len = usize::try_from(section.bytes).unwrap_or(usize::MAX);
    let end = start.saturating_add(len);
    if end > file_len {
        return Err(SectionError::OutOfBounds {
            name: section.name.clone(),
            want: section.bytes,
            got: u64::try_from(file_len.saturating_sub(start)).unwrap_or(u64::MAX),
        });
    }
    Ok(start..end)
}

#[cfg(test)]
#[path = "loader_tests.rs"]
mod tests;
