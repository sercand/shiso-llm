//! Named tensors from either container an audio checkpoint ships in.
//!
//! Audio checkpoints arrive as a GGUF file or as one safetensors file beside a
//! JSON config, and a family reads the same roles from both. The store hides
//! the difference that matters to a loader: a GGUF directory lists dimensions
//! fastest first, a safetensors header slowest first, and every shape handed
//! out here is in the slowest-first order the reference framework prints, so
//! a family writes its expected shapes once.
//!
//! Payloads are borrowed from the mapping, never copied; [`Stored::host`]
//! hands the mapping itself to a device import.

use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use shiso_audio::AudioError;
use shiso_core::{DType, HostBytes};
use shiso_modelfile::GgufFile;
use shiso_modelfile::st_header::{RawTensor, Shard};

/// How a stored tensor's bytes encode its elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoredType {
    /// IEEE single.
    F32,
    /// IEEE half.
    F16,
    /// Brain float.
    Bf16,
    /// GGUF `Q8_0` blocks.
    Q8_0,
    /// GGUF `Q4_K` super-blocks.
    Q4K,
    /// Packed 32-bit words (an affine-quantised weight).
    U32,
    /// Raw bytes.
    U8,
    /// Anything else, by its container's own name.
    Other(String),
}

/// One tensor, borrowed from the store.
#[derive(Debug, Clone)]
pub struct Stored<'a> {
    /// Name as the container spells it.
    pub name: &'a str,
    /// Encoding.
    pub ty: StoredType,
    /// Dimensions, slowest-varying first.
    pub shape: Vec<u64>,
    /// The payload.
    pub data: &'a [u8],
    /// Where the payload sits in [`Stored::host`].
    pub range: Range<usize>,
    /// The mapping the payload belongs to, for a device import.
    pub host: Arc<dyn HostBytes>,
    /// Which file of a sharded checkpoint holds it (0 for a single file).
    pub shard: usize,
}

impl Stored<'_> {
    /// Elements the shape holds.
    #[must_use]
    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

enum Backing {
    Gguf(Arc<GgufFile>),
    /// One or more safetensors files; `index` maps a name to its file.
    Safetensors {
        shards: Vec<(Arc<Shard>, HashMap<String, RawTensor>)>,
        index: HashMap<String, usize>,
    },
}

/// The named tensors of one checkpoint file.
pub struct TensorStore {
    backing: Backing,
    /// Names a test pretends the file does not hold.
    #[cfg(test)]
    hidden: Vec<String>,
}

impl std::fmt::Debug for TensorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.backing {
            Backing::Gguf(_) => f.write_str("TensorStore(gguf)"),
            Backing::Safetensors { index, shards } => {
                write!(
                    f,
                    "TensorStore(safetensors, {} tensors in {} files)",
                    index.len(),
                    shards.len()
                )
            }
        }
    }
}

fn checkpoint(what: impl Into<String>) -> AudioError {
    AudioError::Checkpoint { what: what.into() }
}

fn gguf_type(dtype: DType) -> StoredType {
    match dtype {
        DType::F32 => StoredType::F32,
        DType::F16 => StoredType::F16,
        DType::BF16 => StoredType::Bf16,
        DType::Q8_0 => StoredType::Q8_0,
        DType::Q4_K => StoredType::Q4K,
        other => StoredType::Other(format!("{other:?}")),
    }
}

fn safetensors_type(spelling: &str) -> StoredType {
    match spelling {
        "F32" => StoredType::F32,
        "F16" => StoredType::F16,
        "BF16" => StoredType::Bf16,
        "U32" => StoredType::U32,
        "U8" => StoredType::U8,
        other => StoredType::Other(other.to_owned()),
    }
}

impl TensorStore {
    /// Opens a GGUF file.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] when the file cannot be read as GGUF.
    pub fn open_gguf(path: &Path) -> Result<Self, AudioError> {
        let file = GgufFile::open_shared(path)
            .map_err(|e| checkpoint(format!("{}: {e}", path.display())))?;
        Ok(Self {
            backing: Backing::Gguf(Arc::new(file)),
            #[cfg(test)]
            hidden: Vec::new(),
        })
    }

    /// Opens one safetensors file.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] when the file cannot be read as safetensors.
    pub fn open_safetensors(path: &Path) -> Result<Self, AudioError> {
        Self::open_shards(&[path.to_path_buf()])
    }

    /// Opens a checkpoint split over several safetensors files.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] when a file cannot be read, or when two files
    /// both hold a tensor of the same name.
    pub fn open_shards(paths: &[std::path::PathBuf]) -> Result<Self, AudioError> {
        let mut shards = Vec::with_capacity(paths.len());
        let mut index = HashMap::new();
        for (i, path) in paths.iter().enumerate() {
            let (shard, table) = Shard::open_untagged(path)
                .map_err(|e| checkpoint(format!("{}: {e}", path.display())))?;
            let tensors: HashMap<String, RawTensor> = table.into_iter().collect();
            for name in tensors.keys() {
                if index.insert(name.clone(), i).is_some() {
                    return Err(checkpoint(format!("{name} is in more than one file")));
                }
            }
            shards.push((Arc::new(shard), tensors));
        }
        Ok(Self {
            backing: Backing::Safetensors { shards, index },
            #[cfg(test)]
            hidden: Vec::new(),
        })
    }

    /// The tensor called `name`, if the file holds one.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Stored<'_>> {
        #[cfg(test)]
        if self.hidden.iter().any(|h| h == name) {
            return None;
        }
        match &self.backing {
            Backing::Gguf(file) => {
                let desc = file.find(name)?;
                let view = file.get(name)?;
                let mut shape = desc.shape.dims().to_vec();
                shape.reverse();
                Some(Stored {
                    name: view.name,
                    ty: gguf_type(desc.dtype),
                    shape,
                    data: view.data,
                    range: file.range_of(desc),
                    host: file.mapping(),
                    shard: 0,
                })
            }
            Backing::Safetensors { shards, index } => {
                let at = *index.get(name)?;
                let (shard, tensors) = &shards[at];
                let (key, raw) = tensors.get_key_value(name)?;
                let [start, end] = raw.data_offsets;
                let host: Arc<Shard> = Arc::clone(shard);
                Some(Stored {
                    name: key.as_str(),
                    ty: safetensors_type(&raw.dtype),
                    shape: raw.shape.clone(),
                    data: shard.payload(start, end).ok()?,
                    range: shard.payload_range(start, end).ok()?,
                    host,
                    shard: at,
                })
            }
        }
    }

    /// The tensor called `name`.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] naming the tensor when the file holds none.
    pub fn require(&self, name: &str) -> Result<Stored<'_>, AudioError> {
        self.get(name)
            .ok_or_else(|| checkpoint(format!("missing tensor {name}")))
    }

    /// The same store with `name` treated as absent.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn hiding(mut self, name: &str) -> Self {
        self.hidden.push(name.to_owned());
        self
    }

    /// The GGUF file, for its metadata.
    #[must_use]
    pub fn gguf(&self) -> Option<&GgufFile> {
        match &self.backing {
            Backing::Gguf(file) => Some(file),
            Backing::Safetensors { .. } => None,
        }
    }
}

fn f16_bits(b: &[u8]) -> f32 {
    half::f16::from_le_bytes([b[0], b[1]]).to_f32()
}

fn bf16_bits(b: &[u8]) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)
}

/// Every element of `t` as f32, for the dense types and `Q8_0`.
///
/// # Errors
///
/// [`AudioError::Checkpoint`] naming the tensor for any other type, or for a
/// payload that is not the size its shape implies.
pub fn dequant(t: &Stored<'_>) -> Result<Vec<f32>, AudioError> {
    let n = usize::try_from(t.elements())
        .map_err(|_| checkpoint(format!("{} is too large", t.name)))?;
    let short = || {
        checkpoint(format!(
            "{} holds {} bytes, short of its shape {:?}",
            t.name,
            t.data.len(),
            t.shape
        ))
    };
    let out = match t.ty {
        StoredType::F32 => {
            let bytes = t.data.get(..n * 4).ok_or_else(short)?;
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        StoredType::F16 => t
            .data
            .get(..n * 2)
            .ok_or_else(short)?
            .chunks_exact(2)
            .map(f16_bits)
            .collect(),
        StoredType::Bf16 => t
            .data
            .get(..n * 2)
            .ok_or_else(short)?
            .chunks_exact(2)
            .map(bf16_bits)
            .collect(),
        StoredType::Q8_0 => {
            if !n.is_multiple_of(32) {
                return Err(checkpoint(format!(
                    "{} is not a whole number of Q8_0 blocks",
                    t.name
                )));
            }
            let bytes = t.data.get(..n / 32 * 34).ok_or_else(short)?;
            let mut out = Vec::with_capacity(n);
            for block in bytes.chunks_exact(34) {
                let d = f16_bits(&block[..2]);
                out.extend(
                    block[2..]
                        .iter()
                        .map(|q| d * f32::from(i8::from_le_bytes([*q]))),
                );
            }
            out
        }
        _ => {
            return Err(checkpoint(format!(
                "{} is stored as {:?}, which host dequantisation does not read",
                t.name, t.ty
            )));
        }
    };
    Ok(out)
}

/// An affine-quantised matrix as f32: `words` `[out][in*bits/32]` with element
/// `j` of a word in its low bits first, bf16 `scales` and `biases`
/// `[out][in/group]`, and `w = q * scale + bias`.
///
/// Dequantised in f32 rather than through the half-precision table the file
/// readers produce, because the result is an oracle and a host table both, and
/// half precision's 5e-4 is visible in either.
///
/// # Errors
///
/// [`AudioError::Checkpoint`] naming the tensor for a layout the three arrays
/// do not agree on.
pub fn dequant_affine(
    words: &Stored<'_>,
    scales: &Stored<'_>,
    biases: &Stored<'_>,
    bits: u32,
    group: usize,
) -> Result<Vec<f32>, AudioError> {
    let bad = |why: &str| checkpoint(format!("{}: {why}", words.name));
    let (rows, packed) = match words.shape.as_slice() {
        [r, p] => (
            usize::try_from(*r).map_err(|_| bad("too many rows"))?,
            usize::try_from(*p).map_err(|_| bad("too many words"))?,
        ),
        _ => return Err(bad("not [out][words]")),
    };
    if bits != 4 && bits != 8 {
        return Err(bad("only 4- and 8-bit packing exists"));
    }
    let per_word = 32 / usize::try_from(bits).map_err(|_| bad("bits"))?;
    let in_dim = packed * per_word;
    if group == 0 || !in_dim.is_multiple_of(group) {
        return Err(bad("the group does not divide the row"));
    }
    let groups = in_dim / group;
    let side_shape = [
        u64::try_from(rows).unwrap_or(0),
        u64::try_from(groups).unwrap_or(0),
    ];
    for side in [scales, biases] {
        if side.ty != StoredType::Bf16 || side.shape != side_shape {
            return Err(bad("the scales and biases are not bf16 [out][in/group]"));
        }
    }
    if words.data.len() < rows * packed * 4
        || scales.data.len() < rows * groups * 2
        || biases.data.len() < rows * groups * 2
    {
        return Err(bad("the words, scales and biases disagree on the layout"));
    }
    let mask = (1u32 << bits) - 1;
    let mut out = Vec::with_capacity(rows * in_dim);
    for r in 0..rows {
        for i in 0..in_dim {
            let w = &words.data[(r * packed + i / per_word) * 4..][..4];
            let word = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
            let shift = u32::try_from(i % per_word).map_err(|_| bad("shift"))? * bits;
            let q = (word >> shift) & mask;
            let g = r * groups + i / group;
            let q = f32::from(u16::try_from(q).map_err(|_| bad("quant"))?);
            out.push(q * bf16_bits(&scales.data[g * 2..]) + bf16_bits(&biases.data[g * 2..]));
        }
    }
    Ok(out)
}

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;
