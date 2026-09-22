//! The safetensors container, below any producer's conventions.
//!
//! A safetensors file is eight little-endian length bytes, that many bytes of
//! JSON naming every tensor with its format, its shape and its payload span,
//! and then the payloads. That much is the same whoever wrote it. What differs
//! is everything above: which formats appear, what the names mean, which arrays
//! belong together, and what `__metadata__.format` says.
//!
//! This module is the part that is the same, and it exists because two
//! containers in this crate now need it. [`crate::safetensors`] reads the
//! MLX/OptiQ export, where a packed weight is a `.weight`/`.scales`/`.biases`
//! triple of `u32` and `bf16`; [`crate::modelopt`] reads the modelopt
//! mixed-precision export, where a weight is `F8_E4M3` or nibble-packed `U8`
//! with its own companions. Neither is the other, and their translation layers
//! share nothing.
//!
//! **What they do share is the part that has historically been wrong.** The
//! repository's own account of bringing this engine up - `HANDOFF.md` at the
//! repository root - is twelve defects of which eleven failed silently, and
//! three of them were a span computed against the wrong base. A payload offset,
//! a bounds check and a shape-against-byte-count comparison have one
//! implementation here rather than one per container, so a fix to either is a
//! fix to both.

use std::fmt;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde::Deserialize;

use crate::reader::{GgufError, GgufResult};

/// Key the header reserves for its own metadata rather than a tensor.
const HEADER_METADATA_KEY: &str = "__metadata__";
/// Bytes of little-endian length that precede the header JSON.
const HEADER_LEN_BYTES: usize = 8;
/// Largest header this loader will read.
///
/// Well past any real one, and "any real one" is now a measured number rather
/// than a guess: the largest shard of the 94,393-tensor checkpoint this crate
/// reads carries a 6.6 MB header.
const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;

/// Storage formats a shard header may declare.
///
/// Parsed into an enum rather than carried as a string so that a format no
/// container in this crate can decode is rejected once, at header-parse time,
/// instead of surprising a decode loop one tensor at a time.
///
/// The set is the union over both containers, and a container is responsible
/// for refusing a format it does not itself read. That is deliberate: the
/// alternative is a per-container dtype enum and two parsers, and then a
/// spelling accepted by one and mis-parsed by the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StDType {
    /// Brain float: the top sixteen bits of an `f32`.
    Bf16,
    /// IEEE half precision.
    F16,
    /// IEEE single precision.
    F32,
    /// Unsigned 32-bit words, which is how the MLX export packs weights.
    U32,
    /// Raw bytes. The modelopt export stores NVFP4 weights this way, two
    /// four-bit codes to a byte, so an element here is a *byte* and not a
    /// weight - which is why [`StDType::size`] answers 1 and the shape a header
    /// declares for such a tensor is half the matrix's width.
    U8,
    /// FP8 with four exponent bits and three mantissa bits, no infinities.
    ///
    /// The modelopt export uses it twice over, for two unrelated things: an
    /// FP8 target's weights, and an NVFP4 target's per-group scales.
    F8E4M3,
}

impl StDType {
    /// Parses the header's spelling of a format.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "BF16" => Some(Self::Bf16),
            "F16" => Some(Self::F16),
            "F32" => Some(Self::F32),
            "U32" => Some(Self::U32),
            "U8" => Some(Self::U8),
            "F8_E4M3" => Some(Self::F8E4M3),
            _ => None,
        }
    }

    /// Bytes one element occupies.
    #[must_use]
    pub const fn size(self) -> u64 {
        match self {
            Self::U8 | Self::F8E4M3 => 1,
            Self::Bf16 | Self::F16 => 2,
            Self::F32 | Self::U32 => 4,
        }
    }
}

/// Every spelling [`StDType::parse`] accepts, for the error a refusal carries.
const SPELLINGS: &str = "one of BF16, F16, F32, U32, U8, F8_E4M3";

/// One tensor as the shard header describes it, before translation.
#[derive(Debug, Deserialize)]
pub struct RawTensor {
    /// Storage format, as a header spelling.
    pub dtype: String,
    /// Dimensions, slowest-varying axis first.
    pub shape: Vec<u64>,
    /// Start and end of the payload, relative to the payload base.
    pub data_offsets: [u64; 2],
}

impl RawTensor {
    /// The parsed format, or the refusal naming the spelling.
    ///
    /// # Errors
    ///
    /// [`GgufError::LayoutMismatch`] naming the tensor and the spelling.
    pub fn parsed_dtype(&self, name: &str) -> GgufResult<StDType> {
        StDType::parse(&self.dtype).ok_or_else(|| GgufError::LayoutMismatch {
            tensor: name.into(),
            want: Box::from(SPELLINGS),
            got: Box::from(self.dtype.as_str()),
        })
    }

    /// Elements the declared shape holds.
    ///
    /// A rank-zero shape - which this checkpoint writes for every per-tensor
    /// scalar scale - is one element rather than zero, because that is what its
    /// payload span says and an empty product is one.
    ///
    /// # Errors
    ///
    /// [`GgufError::ElementCountOverflow`] when the axes multiply past `u64`.
    pub fn elements(&self) -> GgufResult<u64> {
        self.shape
            .iter()
            .try_fold(1_u64, |acc, &dim| acc.checked_mul(dim))
            .ok_or(GgufError::ElementCountOverflow)
    }
}

/// The header metadata object, of which one key matters.
#[derive(Debug, Deserialize)]
struct HeaderMetadata {
    /// Producer tag; each container implements one producer's conventions.
    format: String,
}

/// A read-only mapping of one shard.
///
/// # Mutating the file
///
/// The mapping stays alive for as long as this value does. Replacing,
/// truncating or rewriting the file in the meantime is undefined behaviour;
/// the containers that build on this state that invariant on their own `open`.
pub struct Shard {
    /// Path the mapping was taken over, kept for error messages.
    path: PathBuf,
    /// The mapping itself.
    map: Mmap,
    /// First byte of tensor payloads, just past the header.
    payload_base: u64,
}

impl Shard {
    /// Maps `path` read-only and parses its tensor header.
    ///
    /// `want_format` is the producer tag the header must carry. A container
    /// that implements one producer's conventions names it, and a header
    /// written by anyone else is refused here rather than decoded into finite
    /// numbers by the wrong rules.
    ///
    /// Returns the shard and its tensor table; the table is consumed by the
    /// caller and not stored, because after translation the raw names are not
    /// something any later lookup should be able to reach.
    ///
    /// # Errors
    ///
    /// [`GgufError::Io`] when the file cannot be opened or its header is not
    /// JSON, [`GgufError::ArchMismatch`] for the wrong producer tag,
    /// [`GgufError::Truncated`] or [`GgufError::OutOfBounds`] for a span that
    /// does not fit the file, [`GgufError::LayoutMismatch`] for a format no
    /// container reads, and [`GgufError::SizeMismatch`] when a declared shape
    /// and a payload span disagree.
    pub fn open(
        path: &Path,
        want_format: &'static str,
    ) -> GgufResult<(Self, Vec<(String, RawTensor)>)> {
        Self::open_tagged(path, Some(want_format))
    }

    /// Maps `path` read-only and parses its header without requiring a
    /// producer tag.
    ///
    /// The tag is how a container refuses a file written by somebody else's
    /// rules, and every checkpoint shard in this tree carries one. A *sidecar*
    /// need not: `optiq/mtp.safetensors`, the 29-tensor draft head this
    /// family's `config.json` names under `mtp_file`, has no `__metadata__`
    /// object at all. Refusing it on a missing tag would mean refusing the only
    /// draft head the dense family ships, so a caller that already knows which
    /// document named the file - and therefore whose rules it was written under
    /// - opens it through here instead.
    ///
    /// # Mutating the file
    ///
    /// As [`Shard::open`].
    ///
    /// # Errors
    ///
    /// As [`Shard::open`], less [`GgufError::ArchMismatch`]: nothing here
    /// compares a tag.
    pub fn open_untagged(path: &Path) -> GgufResult<(Self, Vec<(String, RawTensor)>)> {
        Self::open_tagged(path, None)
    }

    /// The body of both openers, with the tag requirement as an argument.
    fn open_tagged(
        path: &Path,
        want_format: Option<&'static str>,
    ) -> GgufResult<(Self, Vec<(String, RawTensor)>)> {
        let file = File::open(path).map_err(|err| missing_file(path, &err))?;
        // SAFETY: the mapping is read-only, taken over a file this function
        // opened read-only, so nothing in this process writes through it. The
        // remaining hazard is outside the language: another process that
        // truncates or rewrites this file while the mapping is alive turns
        // every &[u8] handed out from here into a window onto changed or
        // unmapped memory. That is a filesystem-level invariant no signature
        // can express, and it is stated on every public `open` that reaches
        // this code.
        let map = unsafe { Mmap::map(&file)? };
        let (payload_base, tensors) = parse_header(&map, path, want_format)?;
        Ok((
            Self {
                path: path.to_path_buf(),
                map,
                payload_base,
            },
            tensors,
        ))
    }

    /// Payload bytes for a span expressed relative to the payload base.
    ///
    /// # Errors
    ///
    /// [`GgufError::OutOfBounds`] when the span is not inside the mapping.
    pub fn payload(&self, start: u64, end: u64) -> GgufResult<&[u8]> {
        let range = usize::try_from(self.payload_base.saturating_add(start))
            .ok()
            .zip(usize::try_from(self.payload_base.saturating_add(end)).ok())
            .filter(|&(from, to)| from <= to && to <= self.map.len());
        let (from, to) = range.ok_or_else(|| GgufError::OutOfBounds {
            tensor: Box::from(self.path.to_string_lossy().as_ref()),
        })?;
        Ok(&self.map[from..to])
    }

    /// Path this shard was mapped from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Bytes the mapping covers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the mapping is empty, which a real shard never is.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl fmt::Debug for Shard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Shard")
            .field("path", &self.path)
            .field("len", &self.map.len())
            .field("payload_base", &self.payload_base)
            .finish()
    }
}

/// Parses a shard header, returning the payload base and the tensor table.
fn parse_header(
    map: &[u8],
    path: &Path,
    want_format: Option<&'static str>,
) -> GgufResult<(u64, Vec<(String, RawTensor)>)> {
    let len_bytes: [u8; HEADER_LEN_BYTES] = map
        .get(..HEADER_LEN_BYTES)
        .and_then(|head| head.try_into().ok())
        .ok_or(GgufError::Truncated { pos: 0 })?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > MAX_HEADER_BYTES {
        return Err(GgufError::Truncated { pos: header_len });
    }
    let base = u64::try_from(HEADER_LEN_BYTES).unwrap_or(u64::MAX);
    let payload_base = base
        .checked_add(header_len)
        .ok_or(GgufError::Truncated { pos: header_len })?;
    let end =
        usize::try_from(payload_base).map_err(|_| GgufError::Truncated { pos: header_len })?;
    let json = map
        .get(HEADER_LEN_BYTES..end)
        .ok_or(GgufError::Truncated { pos: payload_base })?;

    let mut table: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(json)
        .map_err(|err| {
            GgufError::Io(io::Error::other(format!(
                "{}: tensor header does not parse: {err}",
                path.display()
            )))
        })?;

    let tag = table.remove(HEADER_METADATA_KEY);
    if let Some(want_format) = want_format {
        let metadata = tag.ok_or_else(|| GgufError::ArchMismatch {
            want: want_format,
            got: Box::from("a header with no producer tag"),
        })?;
        let metadata: HeaderMetadata = serde_json::from_value(metadata).map_err(|err| {
            GgufError::Io(io::Error::other(format!(
                "{}: header metadata does not parse: {err}",
                path.display()
            )))
        })?;
        if metadata.format != want_format {
            return Err(GgufError::ArchMismatch {
                want: want_format,
                got: Box::from(metadata.format.as_str()),
            });
        }
    }

    let mut tensors = Vec::with_capacity(table.len());
    for (name, value) in table {
        let raw: RawTensor = serde_json::from_value(value).map_err(|err| {
            GgufError::Io(io::Error::other(format!(
                "{}: entry {name} does not parse: {err}",
                path.display()
            )))
        })?;
        check_raw(&name, &raw, map.len(), payload_base)?;
        tensors.push((name, raw));
    }
    Ok((payload_base, tensors))
}

/// Checks that one header entry's span is inside the shard and matches its
/// declared shape.
///
/// The shape-against-bytes comparison is the load-bearing half. A span inside
/// the file that is the wrong length for the shape beside it is exactly the
/// failure that produces finite numbers from the wrong offset, and it is
/// checked here for every tensor of every container.
fn check_raw(name: &str, raw: &RawTensor, file_len: usize, payload_base: u64) -> GgufResult<()> {
    let dtype = raw.parsed_dtype(name)?;
    let [start, end] = raw.data_offsets;
    let last = payload_base
        .checked_add(end)
        .filter(|&last| start <= end && last <= u64::try_from(file_len).unwrap_or(u64::MAX));
    if last.is_none() {
        return Err(GgufError::OutOfBounds {
            tensor: name.into(),
        });
    }
    let want =
        raw.elements()?
            .checked_mul(dtype.size())
            .ok_or_else(|| GgufError::TensorTooLarge {
                tensor: name.into(),
            })?;
    let got = end - start;
    if got != want {
        return Err(GgufError::SizeMismatch {
            tensor: name.into(),
            want: usize::try_from(want).unwrap_or(usize::MAX),
            got: usize::try_from(got).unwrap_or(usize::MAX),
        });
    }
    Ok(())
}

/// Reports a file a checkpoint needs and does not have, naming the path.
pub(crate) fn missing_file(path: &Path, err: &io::Error) -> GgufError {
    GgufError::Io(io::Error::new(
        err.kind(),
        format!("{}: {err}", path.display()),
    ))
}

#[cfg(test)]
#[path = "st_header_tests.rs"]
mod tests;
