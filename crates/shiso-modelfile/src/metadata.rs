//! The key/value metadata table.
//!
//! Scalars are decoded once, at open time, because they are few and every
//! caller wants them repeatedly. Arrays are **not**: an array value records its
//! element type, its length and the byte range it occupies, and decodes an
//! entry only when asked. That distinction is the difference between a cheap
//! open and an expensive one - a vocabulary array holds a quarter of a million
//! strings, so materialising it eagerly would cost tens of megabytes and
//! hundreds of thousands of allocations on every open, for data the inference
//! path never reads.
//!
//! The typed getters widen and narrow deliberately: producers do not agree on
//! which integer width a key gets, so `get_u32` accepts any unsigned or signed
//! width that fits, and `get_f32` accepts `f64`, `u32` and `i32` as well. The
//! accepted set per getter is documented on the getter, because a silently
//! missing value and a rejected value look the same to a caller that has to
//! fall back to a default.

use std::collections::HashMap;
use std::sync::Arc;

use crate::reader::{Cursor, GgufError, GgufResult, Mapping};

/// Type id of a metadata value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaType {
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
    /// 32-bit float.
    F32,
    /// Boolean, stored as one byte.
    Bool,
    /// Length-prefixed UTF-8 string.
    Str,
    /// Array: an element type, a count, then that many elements.
    Array,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 64-bit integer.
    I64,
    /// 64-bit float.
    F64,
}

impl MetaType {
    /// Maps an on-disk type id.
    #[must_use]
    pub const fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::U8),
            1 => Some(Self::I8),
            2 => Some(Self::U16),
            3 => Some(Self::I16),
            4 => Some(Self::U32),
            5 => Some(Self::I32),
            6 => Some(Self::F32),
            7 => Some(Self::Bool),
            8 => Some(Self::Str),
            9 => Some(Self::Array),
            10 => Some(Self::U64),
            11 => Some(Self::I64),
            12 => Some(Self::F64),
            _ => None,
        }
    }

    /// The on-disk type id.
    #[must_use]
    pub const fn id(self) -> u32 {
        match self {
            Self::U8 => 0,
            Self::I8 => 1,
            Self::U16 => 2,
            Self::I16 => 3,
            Self::U32 => 4,
            Self::I32 => 5,
            Self::F32 => 6,
            Self::Bool => 7,
            Self::Str => 8,
            Self::Array => 9,
            Self::U64 => 10,
            Self::I64 => 11,
            Self::F64 => 12,
        }
    }

    /// Byte width of a fixed-size scalar, `None` for the two variable-length
    /// types.
    #[must_use]
    pub const fn scalar_size(self) -> Option<usize> {
        match self {
            Self::U8 | Self::I8 | Self::Bool => Some(1),
            Self::U16 | Self::I16 => Some(2),
            Self::U32 | Self::I32 | Self::F32 => Some(4),
            Self::U64 | Self::I64 | Self::F64 => Some(8),
            Self::Str | Self::Array => None,
        }
    }
}

/// One metadata value.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MetaValue {
    /// Unsigned 8-bit integer.
    U8(u8),
    /// Signed 8-bit integer.
    I8(i8),
    /// Unsigned 16-bit integer.
    U16(u16),
    /// Signed 16-bit integer.
    I16(i16),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Signed 32-bit integer.
    I32(i32),
    /// 32-bit float.
    F32(f32),
    /// Boolean.
    Bool(bool),
    /// String.
    Str(Box<str>),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Signed 64-bit integer.
    I64(i64),
    /// 64-bit float.
    F64(f64),
    /// Array, kept as a lazy range into the mapping.
    Array(MetaArray),
}

/// A metadata array: a range into the mapping plus its element type.
///
/// Entries decode on demand, so holding one costs three words and an `Arc`
/// clone no matter how long the array is.
#[derive(Clone)]
pub struct MetaArray {
    key: Box<str>,
    elem: MetaType,
    len: usize,
    /// Offset of the first element within the mapping.
    data_pos: u64,
    map: Arc<Mapping>,
}

impl std::fmt::Debug for MetaArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetaArray")
            .field("key", &self.key)
            .field("elem", &self.elem)
            .field("len", &self.len)
            .field("data_pos", &self.data_pos)
            .finish_non_exhaustive()
    }
}

impl MetaArray {
    /// Number of entries.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the array is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Element type.
    #[must_use]
    pub const fn elem_type(&self) -> MetaType {
        self.elem
    }

    /// The key this array was read from.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Entry `i`.
    ///
    /// Fixed-width elements are read by direct offset; variable-length elements
    /// are walked from the start, which is linear in `i`.
    ///
    /// # Errors
    ///
    /// [`GgufError::ArrayIndex`] past the end, [`GgufError::Truncated`] if the
    /// declared range runs off the end of the mapping, [`GgufError::NotUtf8`]
    /// for a malformed string element.
    pub fn get(&self, i: usize) -> GgufResult<MetaValue> {
        if i >= self.len {
            return Err(GgufError::ArrayIndex {
                index: i,
                len: self.len,
            });
        }
        let mut cursor = self.body()?;
        if let Some(size) = self.elem.scalar_size() {
            cursor.skip(i.saturating_mul(size))?;
            return decode(&mut cursor, self.elem, 0, &self.key);
        }
        for _ in 0..i {
            skip(&mut cursor, self.elem, 0, &self.key)?;
        }
        decode(&mut cursor, self.elem, 0, &self.key)
    }

    /// Iterates the entries, decoding each once.
    pub fn iter(&self) -> impl Iterator<Item = GgufResult<MetaValue>> + '_ {
        MetaArrayIter {
            array: self,
            cursor: self.body().ok(),
            left: self.len,
        }
    }

    /// Collects a string array.
    ///
    /// This is the fast path the tokenizer tables want: one pass over the
    /// range, no per-entry re-seeking.
    ///
    /// # Errors
    ///
    /// [`GgufError::LayoutMismatch`] if the element type is not [`MetaType::Str`],
    /// plus the errors of [`MetaArray::iter`].
    pub fn strings(&self) -> GgufResult<Vec<Box<str>>> {
        if self.elem != MetaType::Str {
            return Err(GgufError::LayoutMismatch {
                tensor: self.key.clone(),
                want: Box::from("an array of strings"),
                got: Box::from(format!("an array of {:?}", self.elem)),
            });
        }
        self.iter()
            .map(|value| match value? {
                MetaValue::Str(s) => Ok(s),
                other => Err(GgufError::LayoutMismatch {
                    tensor: self.key.clone(),
                    want: Box::from("an array of strings"),
                    got: Box::from(format!("an array containing {other:?}")),
                }),
            })
            .collect()
    }

    /// The array body, as a cursor over the mapping.
    fn body(&self) -> GgufResult<Cursor<'_>> {
        let start = usize::try_from(self.data_pos)
            .map_err(|_| GgufError::Truncated { pos: self.data_pos })?;
        let bytes = self
            .map
            .bytes()
            .get(start..)
            .ok_or(GgufError::Truncated { pos: self.data_pos })?;
        Ok(Cursor::new(bytes))
    }
}

/// Forward iterator over a [`MetaArray`].
///
/// The cursor is `Option` so a truncated range yields exactly one error and
/// then stops, rather than re-reporting the same truncation `len` times.
struct MetaArrayIter<'a> {
    array: &'a MetaArray,
    cursor: Option<Cursor<'a>>,
    left: usize,
}

impl Iterator for MetaArrayIter<'_> {
    type Item = GgufResult<MetaValue>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.left == 0 {
            return None;
        }
        self.left -= 1;
        if let Some(cursor) = self.cursor.as_mut() {
            return Some(decode(cursor, self.array.elem, 0, &self.array.key));
        }
        // The range did not even start inside the mapping: report that once
        // instead of pretending the array is shorter than it is.
        self.left = 0;
        Some(Err(self.array.body().err().unwrap_or(
            GgufError::Truncated {
                pos: self.array.data_pos,
            },
        )))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}

/// The metadata table: entries in file order plus a name index.
#[derive(Debug)]
pub struct GgufMetadata {
    entries: Vec<(Box<str>, MetaValue)>,
    index: HashMap<Box<str>, usize>,
}

impl GgufMetadata {
    /// Reads `n_kv` entries, leaving `cursor` at the end of the table.
    ///
    /// Scalars and strings are decoded here; arrays become [`MetaArray`] values
    /// that point back into `map`.
    ///
    /// # Errors
    ///
    /// [`GgufError::Truncated`] if the declared count cannot fit in the bytes
    /// left, [`GgufError::LayoutMismatch`] for an unknown value type id,
    /// [`GgufError::NestingTooDeep`] for an over-nested array.
    pub(crate) fn parse(
        cursor: &mut Cursor<'_>,
        map: &Arc<Mapping>,
        n_kv: u64,
    ) -> GgufResult<Self> {
        // Each entry costs at least a length word plus a type id, so a count
        // larger than the remaining bytes is not real. Checking first keeps a
        // truncated file from asking for an enormous allocation.
        if n_kv > cursor.remaining() {
            return Err(GgufError::Truncated {
                pos: cursor.position(),
            });
        }

        let capacity = usize::try_from(n_kv).unwrap_or(0);
        let mut entries = Vec::with_capacity(capacity);
        let mut index = HashMap::with_capacity(capacity);

        for _ in 0..n_kv {
            let key = cursor.string("metadata key")?;
            let type_id = cursor.u32()?;
            let value_type =
                MetaType::from_id(type_id).ok_or_else(|| GgufError::LayoutMismatch {
                    tensor: key.clone(),
                    want: Box::from("a known metadata type id"),
                    got: Box::from(type_id.to_string()),
                })?;

            let value = if value_type == MetaType::Array {
                let elem =
                    MetaType::from_id(cursor.u32()?).ok_or_else(|| GgufError::LayoutMismatch {
                        tensor: key.clone(),
                        want: Box::from("a known array element type id"),
                        got: Box::from(type_id.to_string()),
                    })?;
                let raw_len = cursor.u64()?;
                let len = usize::try_from(raw_len).map_err(|_| GgufError::Truncated {
                    pos: cursor.position(),
                })?;
                let data_pos = cursor.position();
                skip_body(cursor, elem, raw_len, 1, &key)?;
                MetaValue::Array(MetaArray {
                    key: key.clone(),
                    elem,
                    len,
                    data_pos,
                    map: Arc::clone(map),
                })
            } else {
                decode(cursor, value_type, 0, &key)?
            };

            index.entry(key.clone()).or_insert(entries.len());
            entries.push((key, value));
        }

        Ok(Self { entries, index })
    }

    /// Number of key/value entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Entries in file order.
    #[must_use]
    pub fn entries(&self) -> &[(Box<str>, MetaValue)] {
        &self.entries
    }

    /// The declared architecture, `None` when the file never declared one.
    ///
    /// There is deliberately no default here. A fallback architecture would let
    /// an architecture check pass on a file that declared nothing, which is the
    /// opposite of what the check is for.
    #[must_use]
    pub fn architecture(&self) -> Option<&str> {
        self.get_str("general.architecture")
    }

    /// Value stored under `key`.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.index.get(key).map(|&i| &self.entries[i].1)
    }

    /// String value, `None` for any other type.
    #[must_use]
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.get(key)? {
            MetaValue::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Integer value, widened from `u16`, `i16` and `i32` as well.
    ///
    /// Producers disagree on the width of split and count keys, so any width
    /// that fits is accepted; a negative signed value is rejected rather than
    /// wrapped.
    #[must_use]
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match *self.get(key)? {
            MetaValue::U32(v) => Some(v),
            MetaValue::U16(v) => Some(u32::from(v)),
            MetaValue::I16(v) => u32::try_from(v).ok(),
            MetaValue::I32(v) => u32::try_from(v).ok(),
            _ => None,
        }
    }

    /// Integer value, widened from `u32` as well.
    #[must_use]
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match *self.get(key)? {
            MetaValue::U64(v) => Some(v),
            MetaValue::U32(v) => Some(u64::from(v)),
            _ => None,
        }
    }

    /// Float value, narrowed from `f64` and widened from `u32` and `i32`.
    ///
    /// The float and integer conversions are this getter's documented job:
    /// producers write `f64` for keys the kernels consume as `f32`, and a
    /// rejected value would push every caller into inventing its own default.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "documented narrowing: an f64 metadata value becomes f32"
    )]
    #[expect(
        clippy::cast_precision_loss,
        reason = "documented widening: integer metadata values become f32"
    )]
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match *self.get(key)? {
            MetaValue::F32(v) => Some(v),
            MetaValue::F64(v) => Some(v as f32),
            MetaValue::U32(v) => Some(v as f32),
            MetaValue::I32(v) => Some(v as f32),
            _ => None,
        }
    }

    /// Boolean value, `None` for any other type.
    #[must_use]
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match *self.get(key)? {
            MetaValue::Bool(v) => Some(v),
            _ => None,
        }
    }

    /// Array value, `None` for any other type.
    #[must_use]
    pub fn get_array(&self, key: &str) -> Option<&MetaArray> {
        match self.get(key)? {
            MetaValue::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Value of `"<architecture>.<suffix>"`, `None` when no architecture was
    /// declared or the key is absent.
    #[must_use]
    pub fn arch_u32(&self, suffix: &str) -> Option<u32> {
        self.get_u32(&arch_key(self.architecture()?, suffix))
    }

    /// [`GgufMetadata::arch_u32`] widened to 64 bits.
    #[must_use]
    pub fn arch_u64(&self, suffix: &str) -> Option<u64> {
        self.get_u64(&arch_key(self.architecture()?, suffix))
    }

    /// [`GgufMetadata::get_f32`] under the architecture prefix.
    #[must_use]
    pub fn arch_f32(&self, suffix: &str) -> Option<f32> {
        self.get_f32(&arch_key(self.architecture()?, suffix))
    }

    /// [`GgufMetadata::get_str`] under the architecture prefix.
    #[must_use]
    pub fn arch_str(&self, suffix: &str) -> Option<&str> {
        self.get_str(&arch_key(self.architecture()?, suffix))
    }
}

/// Walks `n_kv` entries for the string stored under `want`, stopping there.
///
/// The probe [`crate::GgufFile::read_architecture`] runs: it decodes nothing
/// past the key it is after, so a writer that puts the architecture first -
/// which they do - costs one entry rather than the vocabulary's arrays.
/// `Ok(None)` when no entry has that key or its value is not a string.
pub(crate) fn find_str(
    cursor: &mut Cursor<'_>,
    n_kv: u64,
    want: &str,
) -> GgufResult<Option<Box<str>>> {
    for _ in 0..n_kv {
        let key = cursor.string("metadata key")?;
        let type_id = cursor.u32()?;
        let ty = MetaType::from_id(type_id).ok_or_else(|| GgufError::LayoutMismatch {
            tensor: key.clone(),
            want: Box::from("a known metadata type id"),
            got: Box::from(type_id.to_string()),
        })?;
        if ty == MetaType::Array {
            let elem =
                MetaType::from_id(cursor.u32()?).ok_or_else(|| GgufError::LayoutMismatch {
                    tensor: key.clone(),
                    want: Box::from("a known array element type id"),
                    got: Box::from(type_id.to_string()),
                })?;
            let len = cursor.u64()?;
            skip_body(cursor, elem, len, 1, &key)?;
            if &*key == want {
                return Ok(None);
            }
            continue;
        }
        let value = decode(cursor, ty, 0, &key)?;
        if &*key == want {
            return Ok(match value {
                MetaValue::Str(s) => Some(s),
                _ => None,
            });
        }
    }
    Ok(None)
}

/// Builds an architecture-prefixed key.
fn arch_key(arch: &str, suffix: &str) -> String {
    format!("{arch}.{suffix}")
}

/// Deepest array nesting accepted, matching the reader's fixed cap.
const MAX_DEPTH: u32 = 8;

/// Decodes one value. `depth` counts nested arrays, `key` names the metadata
/// entry being decoded so a failure says which key went bad.
fn decode(cursor: &mut Cursor<'_>, ty: MetaType, depth: u32, key: &str) -> GgufResult<MetaValue> {
    if depth > MAX_DEPTH {
        return Err(GgufError::NestingTooDeep);
    }
    Ok(match ty {
        MetaType::U8 => MetaValue::U8(cursor.u8()?),
        MetaType::I8 => MetaValue::I8(cursor.i8()?),
        MetaType::U16 => MetaValue::U16(cursor.u16()?),
        MetaType::I16 => MetaValue::I16(cursor.i16()?),
        MetaType::U32 => MetaValue::U32(cursor.u32()?),
        MetaType::I32 => MetaValue::I32(cursor.i32()?),
        MetaType::F32 => MetaValue::F32(cursor.f32()?),
        MetaType::Bool => MetaValue::Bool(cursor.u8()? != 0),
        MetaType::Str => MetaValue::Str(cursor.string("metadata string")?),
        MetaType::U64 => MetaValue::U64(cursor.u64()?),
        MetaType::I64 => MetaValue::I64(cursor.i64()?),
        MetaType::F64 => MetaValue::F64(cursor.f64()?),
        // A nested array element cannot be represented: `MetaValue` has no
        // subtree variant, and inventing one for a shape no reader in this
        // crate consumes would be API nobody calls. Such a file is rejected
        // with the key that caused it.
        MetaType::Array => {
            return Err(GgufError::LayoutMismatch {
                tensor: key.into(),
                want: Box::from("a scalar or string array element"),
                got: Box::from("a nested array"),
            });
        }
    })
}

/// Advances past one value without building it.
fn skip(cursor: &mut Cursor<'_>, ty: MetaType, depth: u32, key: &str) -> GgufResult<()> {
    if depth > MAX_DEPTH {
        return Err(GgufError::NestingTooDeep);
    }
    if let Some(size) = ty.scalar_size() {
        return cursor.skip(size);
    }
    if ty == MetaType::Str {
        let len = cursor.u64()?;
        let len = usize::try_from(len).map_err(|_| GgufError::Truncated {
            pos: cursor.position(),
        })?;
        return cursor.skip(len);
    }
    let elem = MetaType::from_id(cursor.u32()?).ok_or_else(|| unknown_type(key))?;
    let len = cursor.u64()?;
    skip_body(cursor, elem, len, depth + 1, key)
}

/// Advances past `len` elements of type `elem`, whose header was already read.
fn skip_body(
    cursor: &mut Cursor<'_>,
    elem: MetaType,
    len: u64,
    depth: u32,
    key: &str,
) -> GgufResult<()> {
    if depth > MAX_DEPTH {
        return Err(GgufError::NestingTooDeep);
    }
    if let Some(size) = elem.scalar_size() {
        let total = u64::try_from(size).unwrap_or(0).saturating_mul(len);
        let total = usize::try_from(total).map_err(|_| GgufError::Truncated {
            pos: cursor.position(),
        })?;
        return cursor.skip(total);
    }
    for _ in 0..len {
        skip(cursor, elem, depth, key)?;
    }
    Ok(())
}

/// Error for a type id that is not one of the thirteen defined ones.
fn unknown_type(key: &str) -> GgufError {
    GgufError::LayoutMismatch {
        tensor: key.into(),
        want: Box::from("a known metadata type id"),
        got: Box::from("an unassigned type id"),
    }
}
