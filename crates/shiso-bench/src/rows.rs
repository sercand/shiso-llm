//! Full logits rows on disk, in the layout the independent reference writes.
//!
//! A `.vec` records the top of a distribution, which is enough to say whether
//! two engines pick the same tokens and not enough to say how far apart their
//! distributions are: an exact KL needs the whole row, tail included, because a
//! quarter of a million near-zero probabilities is where a quantized build
//! loses its mass. So a capture can also write the row itself, once per case,
//! in the one shape both sides of the comparison agreed on:
//!
//! - `<dir>/<case-id>.f16.npy`: a numpy `.npy` version 1.0 file, dtype `<f2`,
//!   shape `[vocab]`, holding the raw pre-softmax logits;
//! - `<dir>/index.json`: case id to `{"prompt_file", "frontier",
//!   "prompt_tokens"}`.
//!
//! `frontier` in the index is the **0-based position of the row**: the logits
//! read after consuming tokens `0..=frontier`, which is one less than the
//! token count a `.vec` case calls its frontier. The index uses the position
//! because that is what a teacher-forced reference pass indexes its logits by,
//! and the two sides must name the same row. `prompt_tokens` is how many tokens
//! the whole prompt file encodes to, so a reader can tell a row at the end of a
//! prompt from one in its middle.
//!
//! The header is written here by hand rather than through a crate: the format
//! is ten bytes and a Python dict literal, and a dependency for that would be a
//! dependency for nothing.
//!
//! # Writes once
//!
//! Like a fixture capture, a row dump refuses to overwrite: every file is
//! opened `create_new`, and [`RowDump::create`] checks the index up front so a
//! run pointed at a used directory fails before it loads a checkpoint, not
//! after an hour of prefill.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use half::f16;

/// The magic prefix every `.npy` file opens with.
const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

/// The alignment numpy pads the header to, so the data starts on it.
///
/// numpy itself has used 64 since 1.14; readers accept any multiple of 16.
const NPY_ALIGN: usize = 64;

/// The file name of a dump's index.
pub const INDEX_FILE: &str = "index.json";

/// What can go wrong writing or reading a row.
#[derive(Debug, thiserror::Error)]
pub enum RowError {
    /// A file exists where the dump would write one.
    #[error(
        "{path} already exists; a row dump refuses to replace one, so move the old one aside \
         deliberately"
    )]
    Exists {
        /// The file.
        path: PathBuf,
    },
    /// The filesystem refused.
    #[error("{path}: {source}")]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// The operating system's reason.
        #[source]
        source: std::io::Error,
    },
    /// A value is not finite in half precision.
    ///
    /// Refused rather than written: a NaN in the row makes every KL over it
    /// NaN, and a logit past the half-precision range would be written as an
    /// infinity that the reader could not tell from a real one. Either way the
    /// row is the finding, and it should be looked at as an `f32` first.
    #[error("{case}: {count} of {len} logits are not finite in f16 (first at index {first})")]
    NotFinite {
        /// The case the row belongs to.
        case: String,
        /// How many values were refused.
        count: usize,
        /// The row's length.
        len: usize,
        /// The first offending index.
        first: usize,
    },
    /// A case id would not make a plain file name.
    #[error("case id {id:?} is not usable as a file name")]
    BadId {
        /// The id.
        id: String,
    },
    /// The same case was dumped twice in one run.
    #[error("case {id} was already dumped in this run")]
    Duplicate {
        /// The id.
        id: String,
    },
    /// A file does not have the header this module writes.
    #[error("not an .npy v1 row: {detail}")]
    BadHeader {
        /// What was wrong.
        detail: String,
    },
    /// The index would not serialize.
    #[error("serializing the index: {0}")]
    Json(#[from] serde_json::Error),
}

/// The header of a one-dimensional `.npy` row, as [`read_npy_header`] finds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpyHeader {
    /// The dtype descriptor, e.g. `<f2`.
    pub descr: String,
    /// Whether the file claims Fortran order.
    pub fortran_order: bool,
    /// The shape, outermost first.
    pub shape: Vec<usize>,
    /// Where the data starts.
    pub data_offset: usize,
}

/// Encodes `row` as a `.npy` v1 file of dtype `<f2` and shape `[row.len()]`.
///
/// The conversion is `f16::from_f32`, which rounds to nearest, ties to even.
///
/// # Errors
///
/// [`RowError::NotFinite`] when any value is NaN or infinite after conversion.
pub fn encode_f16_npy(case: &str, row: &[f32]) -> Result<Vec<u8>, RowError> {
    let halves: Vec<f16> = row.iter().copied().map(f16::from_f32).collect();
    let bad: Vec<usize> = halves
        .iter()
        .enumerate()
        .filter(|(_, h)| !h.is_finite())
        .map(|(i, _)| i)
        .collect();
    if let Some(&first) = bad.first() {
        return Err(RowError::NotFinite {
            case: case.to_owned(),
            count: bad.len(),
            len: row.len(),
            first,
        });
    }

    let mut dict = format!(
        "{{'descr': '<f2', 'fortran_order': False, 'shape': ({},), }}",
        row.len()
    );
    // Magic (6) + version (2) + header length (2), then the dict padded with
    // spaces and closed by a newline so the data lands on the alignment.
    let unpadded = NPY_MAGIC.len() + 2 + 2 + dict.len() + 1;
    let pad = (NPY_ALIGN - unpadded % NPY_ALIGN) % NPY_ALIGN;
    dict.extend(std::iter::repeat_n(' ', pad));
    dict.push('\n');
    let header_len = u16::try_from(dict.len()).map_err(|_| RowError::BadHeader {
        detail: format!("a {}-byte header does not fit version 1.0", dict.len()),
    })?;

    let mut out = Vec::with_capacity(NPY_MAGIC.len() + 4 + dict.len() + halves.len() * 2);
    out.extend_from_slice(NPY_MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    for h in &halves {
        out.extend_from_slice(&h.to_le_bytes());
    }
    Ok(out)
}

/// Parses the header of a one-dimensional `.npy` v1 file.
///
/// Only as general as what [`encode_f16_npy`] and a numpy `np.save` of a 1-D
/// array write; it exists so a test can read back what this module wrote
/// without a second implementation in another language.
///
/// # Errors
///
/// [`RowError::BadHeader`] for anything else.
pub fn read_npy_header(bytes: &[u8]) -> Result<NpyHeader, RowError> {
    let bad = |detail: &str| RowError::BadHeader {
        detail: detail.to_owned(),
    };
    let rest = bytes
        .strip_prefix(NPY_MAGIC)
        .ok_or_else(|| bad("no magic"))?;
    let [major, minor, lo, hi, dict @ ..] = rest else {
        return Err(bad("truncated before the header length"));
    };
    if (*major, *minor) != (1, 0) {
        return Err(bad(&format!("version {major}.{minor}, expected 1.0")));
    }
    let len = usize::from(u16::from_le_bytes([*lo, *hi]));
    let text = dict.get(..len).ok_or_else(|| bad("truncated header"))?;
    let text = std::str::from_utf8(text).map_err(|_| bad("header is not text"))?;
    let data_offset = NPY_MAGIC.len() + 4 + len;
    if !text.ends_with('\n') {
        return Err(bad("header does not end in a newline"));
    }

    let field = |key: &str| -> Result<&str, RowError> {
        let at = text
            .find(&format!("'{key}': "))
            .ok_or_else(|| bad(&format!("no {key}")))?;
        Ok(&text[at + key.len() + 4..])
    };
    let descr = field("descr")?
        .strip_prefix('\'')
        .and_then(|s| s.split('\'').next())
        .ok_or_else(|| bad("descr is not a string"))?
        .to_owned();
    let fortran = field("fortran_order")?;
    let fortran_order = if fortran.starts_with("False") {
        false
    } else if fortran.starts_with("True") {
        true
    } else {
        return Err(bad("fortran_order is not a bool"));
    };
    let shape_text = field("shape")?
        .strip_prefix('(')
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| bad("shape is not a tuple"))?;
    let shape = shape_text
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| bad("shape entry is not a count"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(NpyHeader {
        descr,
        fortran_order,
        shape,
        data_offset,
    })
}

/// One case's line in `index.json`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct IndexEntry {
    /// The prompt file, as the case names it.
    pub prompt_file: String,
    /// The 0-based position of the row: tokens `0..=frontier` were consumed.
    pub frontier: u32,
    /// How many tokens the whole prompt file encodes to.
    pub prompt_tokens: u32,
}

/// A directory being filled with rows.
#[derive(Debug)]
pub struct RowDump {
    dir: PathBuf,
    index: BTreeMap<String, IndexEntry>,
}

impl RowDump {
    /// Opens `dir` for a dump, creating it if needed.
    ///
    /// # Errors
    ///
    /// [`RowError::Exists`] when the directory already holds an index, and
    /// [`RowError::Io`] when it cannot be created.
    pub fn create(dir: &Path) -> Result<Self, RowError> {
        std::fs::create_dir_all(dir).map_err(|source| RowError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let index = dir.join(INDEX_FILE);
        if index.exists() {
            return Err(RowError::Exists { path: index });
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            index: BTreeMap::new(),
        })
    }

    /// Where the row of case `id` goes.
    ///
    /// # Errors
    ///
    /// [`RowError::BadId`] when `id` would not be a plain file name.
    pub fn row_path(&self, id: &str) -> Result<PathBuf, RowError> {
        let plain = !id.is_empty()
            && id != "."
            && id != ".."
            && !id.contains(['/', '\\', '\0'])
            && !id.starts_with('.');
        if !plain {
            return Err(RowError::BadId { id: id.to_owned() });
        }
        Ok(self.dir.join(format!("{id}.f16.npy")))
    }

    /// Refuses up front if any of `ids` already has a row on disk.
    ///
    /// # Errors
    ///
    /// [`RowError::Exists`] naming the first one, or [`RowError::BadId`].
    pub fn check_free<'a>(&self, ids: impl IntoIterator<Item = &'a str>) -> Result<(), RowError> {
        for id in ids {
            let path = self.row_path(id)?;
            if path.exists() {
                return Err(RowError::Exists { path });
            }
        }
        Ok(())
    }

    /// Writes one case's row and remembers its index entry.
    ///
    /// # Errors
    ///
    /// Whatever [`encode_f16_npy`] refuses, [`RowError::Duplicate`],
    /// [`RowError::Exists`] when the file appeared since, and [`RowError::Io`].
    pub fn write(&mut self, id: &str, row: &[f32], entry: IndexEntry) -> Result<(), RowError> {
        if self.index.contains_key(id) {
            return Err(RowError::Duplicate { id: id.to_owned() });
        }
        let path = self.row_path(id)?;
        let bytes = encode_f16_npy(id, row)?;
        write_new(&path, &bytes)?;
        self.index.insert(id.to_owned(), entry);
        Ok(())
    }

    /// Writes `index.json` and returns its path.
    ///
    /// # Errors
    ///
    /// [`RowError::Exists`] when an index appeared since [`RowDump::create`],
    /// and [`RowError::Io`].
    pub fn finish(self) -> Result<PathBuf, RowError> {
        let path = self.dir.join(INDEX_FILE);
        let mut text = serde_json::to_string_pretty(&self.index)?;
        text.push('\n');
        write_new(&path, text.as_bytes())?;
        Ok(path)
    }
}

/// Writes `bytes` to a file that must not exist yet.
fn write_new(path: &Path, bytes: &[u8]) -> Result<(), RowError> {
    let io = |source| RowError::Io {
        path: path.to_path_buf(),
        source,
    };
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| {
            if source.kind() == std::io::ErrorKind::AlreadyExists {
                RowError::Exists {
                    path: path.to_path_buf(),
                }
            } else {
                io(source)
            }
        })?;
    file.write_all(bytes).map_err(io)?;
    file.sync_all().map_err(io)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "shiso-rows-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        drop(std::fs::remove_dir_all(&dir));
        dir
    }

    #[test]
    fn header_round_trips_and_data_is_aligned() {
        let row: Vec<f32> = (0..248_320_u32)
            .map(|i| f32::from(u16::try_from(i % 97).expect("below 97")) * 0.25 - 12.0)
            .collect();
        let bytes = encode_f16_npy("c", &row).expect("finite row");
        let header = read_npy_header(&bytes).expect("own header");
        assert_eq!(header.descr, "<f2");
        assert!(!header.fortran_order);
        assert_eq!(header.shape, vec![248_320]);
        assert_eq!(header.data_offset % NPY_ALIGN, 0);
        assert_eq!(bytes.len(), header.data_offset + 2 * row.len());
        let second =
            f16::from_le_bytes([bytes[header.data_offset + 2], bytes[header.data_offset + 3]]);
        assert_eq!(second.to_bits(), f16::from_f32(-11.75).to_bits());
    }

    #[test]
    fn conversion_rounds_to_nearest_even() {
        // 1 + 2^-11 is exactly half an f16 ulp above 1.0: ties go to the even
        // mantissa, 1.0. Three halves of an ulp goes up to 1 + 2^-9.
        let tie = 1.0 + 2f32.powi(-11);
        let above = 1.0 + 3.0 * 2f32.powi(-11);
        let bytes = encode_f16_npy("c", &[tie, above, 65504.0]).expect("finite");
        let off = read_npy_header(&bytes).expect("header").data_offset;
        let at = |i: usize| f16::from_le_bytes([bytes[off + 2 * i], bytes[off + 2 * i + 1]]);
        assert_eq!(at(0).to_bits(), 0x3c00);
        assert_eq!(at(1).to_bits(), 0x3c02);
        assert_eq!(at(2).to_bits(), 0x7bff);
    }

    #[test]
    fn refuses_values_that_are_not_finite_in_half() {
        let err = encode_f16_npy("c", &[0.0, 70_000.0, f32::NAN]).expect_err("overflow");
        assert!(matches!(
            err,
            RowError::NotFinite {
                count: 2,
                first: 1,
                ..
            }
        ));
    }

    #[test]
    fn dump_writes_rows_and_index_and_refuses_to_overwrite() {
        let dir = scratch("dump");
        let mut dump = RowDump::create(&dir).expect("fresh dir");
        let entry = IndexEntry {
            prompt_file: "p.txt".to_owned(),
            frontier: 14,
            prompt_tokens: 15,
        };
        dump.write("case_a", &[1.0, -2.0, 3.5], entry.clone())
            .expect("first write");
        assert!(matches!(
            dump.write("case_a", &[0.0], entry.clone()),
            Err(RowError::Duplicate { .. })
        ));
        assert!(matches!(dump.row_path("../x"), Err(RowError::BadId { .. })));
        let index = dump.finish().expect("index");

        let bytes = std::fs::read(dir.join("case_a.f16.npy")).expect("row");
        assert_eq!(read_npy_header(&bytes).expect("header").shape, vec![3]);
        let parsed: BTreeMap<String, IndexEntry> =
            serde_json::from_str(&std::fs::read_to_string(&index).expect("index")).expect("json");
        assert_eq!(parsed.get("case_a"), Some(&entry));

        assert!(matches!(
            RowDump::create(&dir),
            Err(RowError::Exists { .. })
        ));
        std::fs::remove_file(&index).expect("clear index");
        let again = RowDump::create(&dir).expect("no index now");
        assert!(matches!(
            again.check_free(["case_a"]),
            Err(RowError::Exists { .. })
        ));
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }
}
