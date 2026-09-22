//! The vision tower shard that sits beside a checkpoint's language model.
//!
//! # Why this is not the checkpoint-directory loader
//!
//! The model-file crate's directory loader deliberately follows the shard index
//! and maps ONLY the shards the language model's tensors live in - the vision
//! tower and the speculative head beside them are never opened at all, which is
//! the difference between ignoring hundreds of megabytes and reading them. That
//! is the right behaviour for loading a language model, and it is exactly why
//! it cannot be pointed at the tower: the tower is one of the files it refuses
//! to map, and it is a bare shard with no config of its own next to it.
//!
//! So the header is parsed here. It is a small format - a little-endian length,
//! a JSON table of name to `{dtype, shape, data_offsets}`, then the payloads -
//! and the alternative was to widen a loader whose narrowness is a documented
//! feature. The second consumer this was waiting for has arrived: the Qwen
//! family's tower loader in `shiso-models` reads its shard through here, so
//! the type is exposed rather than crate-private.
//!
//! # What it does not do
//!
//! Nothing here unpacks a quantized weight. The tower ships in a float format
//! the device reads directly, and a tower that did not would be a different
//! loading problem rather than a bigger one - so an unrecognized format is
//! refused by name rather than guessed at.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use shiso_core::DType;

use crate::error::{ShardFault, VisionError};

/// Bytes of little-endian length that precede the header.
const HEADER_LEN_BYTES: usize = 8;

/// Largest header this reader will accept, far past any real one. A shard
/// declaring more than this is refused before the length is used to size an
/// allocation, which is the reason the bound exists at all.
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

/// Key the header reserves for its own metadata rather than for a tensor.
const METADATA_KEY: &str = "__metadata__";

/// One tensor's entry in the shard header.
#[derive(Debug, Clone)]
pub struct ShardEntry {
    /// Storage format, already narrowed to one this loader can upload.
    pub dtype: DType,
    /// Dimensions, slowest-varying axis first.
    pub shape: Vec<u64>,
    /// Payload start, relative to the end of the header.
    start: u64,
    /// Payload end, relative to the end of the header.
    end: u64,
}

impl ShardEntry {
    /// Payload length in bytes.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }
}

/// An open tower shard: its header, parsed, and the file to read payloads from.
///
/// The payloads are read on demand rather than mapped. A tower is uploaded once
/// at boot and the device owns the bytes afterwards, so a mapping would buy a
/// copy avoided at the cost of a file that must stay untouched for the life of
/// the process - a bad trade for the one caller this has.
#[derive(Debug)]
pub struct VisionShard {
    path: PathBuf,
    file: File,
    entries: HashMap<String, ShardEntry>,
    data_start: u64,
}

impl VisionShard {
    /// Opens a shard and parses its header.
    ///
    /// # Errors
    ///
    /// [`VisionError::TowerIo`] when the file cannot be read and
    /// [`VisionError::ShardHeader`] for a header this reader cannot trust,
    /// with the reason as a typed fault rather than a message.
    pub fn open(path: &Path) -> Result<Self, VisionError> {
        let mut file = File::open(path).map_err(|source| VisionError::TowerIo {
            path: path.to_path_buf(),
            source,
        })?;

        let mut length = [0u8; HEADER_LEN_BYTES];
        read_exact(&mut file, &mut length, path, ShardFault::Truncated)?;
        let header_bytes = u64::from_le_bytes(length);
        if header_bytes == 0 || header_bytes > MAX_HEADER_BYTES {
            return Err(VisionError::ShardHeader {
                path: path.to_path_buf(),
                fault: ShardFault::HeaderLength,
            });
        }
        let width = usize::try_from(header_bytes).map_err(|_| VisionError::ShardHeader {
            path: path.to_path_buf(),
            fault: ShardFault::HeaderLength,
        })?;
        let mut header = vec![0u8; width];
        read_exact(&mut file, &mut header, path, ShardFault::Truncated)?;

        let table: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header)
            .map_err(|_| VisionError::ShardHeader {
                path: path.to_path_buf(),
                fault: ShardFault::HeaderJson,
            })?;

        let data_start =
            header_bytes.saturating_add(u64::try_from(HEADER_LEN_BYTES).unwrap_or(u64::MAX));
        let file_bytes = file
            .metadata()
            .map_err(|source| VisionError::TowerIo {
                path: path.to_path_buf(),
                source,
            })?
            .len();

        let mut entries = HashMap::with_capacity(table.len());
        for (name, value) in table {
            if name == METADATA_KEY {
                continue;
            }
            let entry = parse_entry(&value, path)?;
            // Offsets are relative to the payload region, so the bound is the
            // file's length minus where that region starts. Checked here, once,
            // rather than at each read: a range that escapes the file must not
            // be reachable from a value that already exists.
            let end = data_start.saturating_add(entry.end);
            if entry.end < entry.start || end > file_bytes {
                return Err(VisionError::ShardHeader {
                    path: path.to_path_buf(),
                    fault: ShardFault::OffsetsOutOfRange,
                });
            }
            entries.insert(name, entry);
        }

        tracing::debug!(
            path = %path.display(),
            tensors = entries.len(),
            "opened a vision tower shard"
        );
        Ok(Self {
            path: path.to_path_buf(),
            file,
            entries,
            data_start,
        })
    }

    /// Tensors the shard declares.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the shard declares no tensors at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Header entry for `name`.
    #[must_use]
    pub fn entry(&self, name: &str) -> Option<&ShardEntry> {
        self.entries.get(name)
    }

    /// Reads one tensor's payload.
    ///
    /// # Errors
    ///
    /// [`VisionError::MissingTensor`] when the shard declares no such tensor,
    /// and [`VisionError::TowerIo`] when the read fails.
    pub fn read(&self, name: &str) -> Result<Vec<u8>, VisionError> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| VisionError::MissingTensor {
                name: name.to_owned(),
            })?;
        let width = usize::try_from(entry.bytes()).map_err(|_| VisionError::ShardHeader {
            path: self.path.clone(),
            fault: ShardFault::OffsetsOutOfRange,
        })?;
        let mut buffer = vec![0u8; width];
        // `&File` reads through its own cursor, so the seek and the read are
        // one operation from this function's point of view. The loader is the
        // only caller and it is single-threaded; a concurrent reader would
        // need its own handle.
        let mut handle = &self.file;
        handle
            .seek(SeekFrom::Start(self.data_start.saturating_add(entry.start)))
            .map_err(|source| VisionError::TowerIo {
                path: self.path.clone(),
                source,
            })?;
        handle
            .read_exact(&mut buffer)
            .map_err(|source| VisionError::TowerIo {
                path: self.path.clone(),
                source,
            })?;
        Ok(buffer)
    }
}

/// Reads exactly `buffer.len()` bytes, reporting a short file as a header fault
/// rather than as a generic read error - a truncated download is the common
/// cause and it deserves to say so.
fn read_exact(
    file: &mut File,
    buffer: &mut [u8],
    path: &Path,
    fault: ShardFault,
) -> Result<(), VisionError> {
    file.read_exact(buffer)
        .map_err(|_| VisionError::ShardHeader {
            path: path.to_path_buf(),
            fault,
        })
}

/// Parses one header entry.
fn parse_entry(value: &serde_json::Value, path: &Path) -> Result<ShardEntry, VisionError> {
    let fault = |fault: ShardFault| VisionError::ShardHeader {
        path: path.to_path_buf(),
        fault,
    };
    let dtype = value
        .get("dtype")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| fault(ShardFault::HeaderJson))?;
    let dtype = match dtype {
        "BF16" => DType::BF16,
        "F16" => DType::F16,
        "F32" => DType::F32,
        _ => return Err(fault(ShardFault::UnsupportedDType)),
    };
    let shape: Vec<u64> = value
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| fault(ShardFault::MissingShape))?
        .iter()
        .map(|dim| dim.as_u64().ok_or_else(|| fault(ShardFault::MissingShape)))
        .collect::<Result<_, _>>()?;
    if shape.is_empty() || shape.contains(&0) {
        return Err(fault(ShardFault::MissingShape));
    }
    let offsets = value
        .get("data_offsets")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| fault(ShardFault::HeaderJson))?;
    let (Some(start), Some(end), 2) = (
        offsets.first().and_then(serde_json::Value::as_u64),
        offsets.get(1).and_then(serde_json::Value::as_u64),
        offsets.len(),
    ) else {
        return Err(fault(ShardFault::OffsetsOutOfRange));
    };
    Ok(ShardEntry {
        dtype,
        shape,
        start,
        end,
    })
}
