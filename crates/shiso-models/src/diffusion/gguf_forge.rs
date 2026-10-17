//! A synthetic GGUF writer for the loader and detection tests.
//!
//! Writes a version-3 file holding string metadata and tensors whose payload
//! the caller supplies, so a refusal test can forge exactly the header it
//! needs without a checkpoint on disk.

use std::path::Path;

/// One tensor of a forged file.
pub(crate) struct ForgedTensor<'a> {
    /// Tensor name.
    pub name: &'a str,
    /// Dimensions in GGUF `ne` order, fastest-varying first.
    pub ne: &'a [u64],
    /// The GGUF type id: 0 for `F32`, 8 for `Q8_0`, 10 to 14 for `Q2_K` to `Q6_K`.
    pub type_id: u32,
    /// The tensor's payload bytes.
    pub bytes: &'a [u8],
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&u64::try_from(s.len()).expect("fits").to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// Writes a GGUF v3 file with the string metadata `kvs` and `tensors`.
pub(crate) fn write_gguf(path: &Path, kvs: &[(&str, &str)], tensors: &[ForgedTensor<'_>]) {
    const ALIGN: usize = 32;
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&u64::try_from(tensors.len()).expect("fits").to_le_bytes());
    out.extend_from_slice(&u64::try_from(kvs.len()).expect("fits").to_le_bytes());
    for (key, value) in kvs {
        put_str(&mut out, key);
        out.extend_from_slice(&8u32.to_le_bytes());
        put_str(&mut out, value);
    }
    let mut offset = 0usize;
    for t in tensors {
        put_str(&mut out, t.name);
        out.extend_from_slice(&u32::try_from(t.ne.len()).expect("fits").to_le_bytes());
        for d in t.ne {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&t.type_id.to_le_bytes());
        out.extend_from_slice(&u64::try_from(offset).expect("fits").to_le_bytes());
        offset = (offset + t.bytes.len()).next_multiple_of(ALIGN);
    }
    out.resize(out.len().next_multiple_of(ALIGN), 0);
    for t in tensors {
        out.extend_from_slice(t.bytes);
        out.resize(out.len().next_multiple_of(ALIGN), 0);
    }
    std::fs::write(path, out).expect("writes the forged file");
}
