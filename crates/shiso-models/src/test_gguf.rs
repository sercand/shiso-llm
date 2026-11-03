//! A synthetic GGUF writer for the family readers' tests: metadata of every
//! kind a reader asks for, and tensors with real bytes, so a test can name the
//! exact header it means instead of needing a checkpoint.

use std::path::Path;

/// One metadata value.
pub(crate) enum Kv {
    /// A string.
    Str(&'static str),
    /// An unsigned 32-bit integer.
    U32(u32),
    /// A 32-bit float.
    F32(f32),
    /// A boolean.
    Bool(bool),
    /// An array of strings.
    Strs(Vec<&'static str>),
    /// An array of signed 32-bit integers.
    I32s(Vec<i32>),
}

/// One tensor: its name, its ggml type id, its dimensions innermost first,
/// and its bytes.
pub(crate) struct Tensor {
    /// The tensor's name.
    pub name: String,
    /// The ggml type id (0 `f32`, 8 `q8_0`, 20 `iq4_nl`, 30 `bf16`, ...).
    pub ggml_type: u32,
    /// Dimensions, innermost first.
    pub dims: Vec<u64>,
    /// The payload.
    pub data: Vec<u8>,
}

/// The data section's alignment, which is the GGUF default.
const ALIGN: usize = 32;

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&u64::try_from(s.len()).expect("fits").to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// Writes a GGUF v3 file holding `kvs` and `tensors`.
pub(crate) fn write_gguf(path: &Path, kvs: &[(&str, Kv)], tensors: &[Tensor]) {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&u64::try_from(tensors.len()).expect("fits").to_le_bytes());
    out.extend_from_slice(&u64::try_from(kvs.len()).expect("fits").to_le_bytes());
    for (key, value) in kvs {
        put_str(&mut out, key);
        match value {
            Kv::Str(s) => {
                out.extend_from_slice(&8u32.to_le_bytes());
                put_str(&mut out, s);
            }
            Kv::U32(v) => {
                out.extend_from_slice(&4u32.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            Kv::F32(v) => {
                out.extend_from_slice(&6u32.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            Kv::Bool(v) => {
                out.extend_from_slice(&7u32.to_le_bytes());
                out.push(u8::from(*v));
            }
            Kv::Strs(list) => {
                out.extend_from_slice(&9u32.to_le_bytes());
                out.extend_from_slice(&8u32.to_le_bytes());
                out.extend_from_slice(&u64::try_from(list.len()).expect("fits").to_le_bytes());
                for s in list {
                    put_str(&mut out, s);
                }
            }
            Kv::I32s(list) => {
                out.extend_from_slice(&9u32.to_le_bytes());
                out.extend_from_slice(&5u32.to_le_bytes());
                out.extend_from_slice(&u64::try_from(list.len()).expect("fits").to_le_bytes());
                for v in list {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    let mut offset = 0usize;
    let mut offsets = Vec::with_capacity(tensors.len());
    for t in tensors {
        offsets.push(offset);
        offset = (offset + t.data.len()).next_multiple_of(ALIGN);
    }
    for (t, at) in tensors.iter().zip(&offsets) {
        put_str(&mut out, &t.name);
        out.extend_from_slice(&u32::try_from(t.dims.len()).expect("fits").to_le_bytes());
        for d in &t.dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&t.ggml_type.to_le_bytes());
        out.extend_from_slice(&u64::try_from(*at).expect("fits").to_le_bytes());
    }
    out.resize(out.len().next_multiple_of(ALIGN), 0);
    let base = out.len();
    for (t, at) in tensors.iter().zip(&offsets) {
        out.resize(base + at, 0);
        out.extend_from_slice(&t.data);
    }
    std::fs::write(path, out).expect("writes");
}
