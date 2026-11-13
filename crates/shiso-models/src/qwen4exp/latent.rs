//! The latent pack: a checkpoint of this family whose routed experts live in
//! flat files rather than in a container's tensors.
//!
//! A pack is a directory of
//!
//! - `experts.bin`, `layers * experts` records of one fixed size and no header,
//!   the record of (layer, expert) at `(layer * experts + expert) * record`;
//! - `native_experts.txt`, the index that says so (and what the records are);
//! - `labels.bin`, the latent group of each expert;
//! - `latent_dense_q8.bin`, per layer the down projection into the latent
//!   space and the up projection out of it, in `Q8_0` rows;
//! - `model-skeleton.gguf`, everything else.
//!
//! This module is the host side only: it parses and checks the index files,
//! turns one stored record into the nine arrays an [`ExpertStream`] slot holds
//! (the same layout [`super::nvfp4::Nvfp4Experts`] fills), and lays the up
//! projection out as one matrix.
//!
//! # A record is a permutation of the stream's bytes, never a decode
//!
//! A record stores its three projections as NVFP4 blocks of 64 weights: four
//! `e4m3` scales (one per sixteen weights) then 32 code bytes whose nibbles are
//! split-half within each sixteen. The stream wants pairwise nibbles (the
//! earlier weight in the low nibble) and the scales apart from the codes. The
//! `e2m1` codes and `e4m3` scales are the same values in both, so the move is
//! lossless and no weight is ever widened; the tests compare every weight
//! before and after.
//!
//! The scale bytes are checked on the way past, because a stored scale with
//! its sign bit set, or the all-ones code, is not a value this format
//! produces: it means the file is not what the index says it is, and decoding
//! it would yield a plausible-looking wrong weight.
//!
//! [`ExpertStream`]: shiso_engine::sparse::stream::ExpertStream

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use shiso_core::error::{CoreError, CoreResult};
use shiso_engine::sparse::stream::ExpertSource;
use shiso_engine::sparse::weights::{WeightPart, nvfp4_bytes};
use shiso_modelfile::st_header::open_uncached;

use crate::qwen4exp::config::{EXPERT_INTERMEDIATE, EXPERTS, HIDDEN, N_LAYERS};

/// The index of the routed records.
const INDEX: &str = "native_experts.txt";
/// The records themselves.
const EXPERTS_FILE: &str = "experts.bin";
/// Each expert's latent group.
const LABELS: &str = "labels.bin";
/// The dense latent projections, `Q8_0`.
const PROJECTIONS_Q8: &str = "latent_dense_q8.bin";
/// The same projections at four bits, which this reader does not take.
const PROJECTIONS_Q4: &str = "latent_dense_q4.bin";
/// The pack's non-expert tensors.
pub(crate) const SKELETON: &str = "model-skeleton.gguf";

/// The index's type code for NVFP4 blocks in the record.
const NVFP4_TYPE: u32 = 40;
/// Weights in one stored NVFP4 block.
const BLOCK_WEIGHTS: u32 = 64;
/// Bytes of one stored NVFP4 block: four scales, then 32 code bytes.
const BLOCK_BYTES: usize = 36;
/// Scales at the head of a block.
const BLOCK_SCALES: usize = 4;
/// Code bytes in a block, which is also the pairwise bytes of 64 weights.
const BLOCK_CODES: usize = 32;
/// Weights one `Q8_0` block covers.
const Q8_WEIGHTS: u32 = 32;
/// Bytes of one `Q8_0` block: an `f16` scale and 32 `i8`.
const Q8_BLOCK_BYTES: usize = 34;
/// Bytes of the record's tail: three `f32` global scales and a zero.
const TAIL_BYTES: u64 = 16;
/// Bytes of one global scale.
const GLOBAL_BYTES: usize = 4;

/// Arrays one projection of one expert is stored as: codes, group scales,
/// global scale.
const PARTS: [WeightPart; 3] = [
    WeightPart::Whole,
    WeightPart::GroupScales,
    WeightPart::GlobalScale,
];

/// The sizes a pack's files are checked against.
///
/// The family's fixed shape for a real pack; a smaller one lets the tests
/// build a pack in a temp directory.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    layers: u32,
    experts: u32,
    hidden: u32,
    inter: u32,
}

/// The shape of every pack this family ships.
const FAMILY: Geometry = Geometry {
    layers: N_LAYERS,
    experts: EXPERTS,
    hidden: HIDDEN,
    inter: EXPERT_INTERMEDIATE,
};

/// Bytes of one expert's record at latent width `width` and inner width
/// `inter`: gate, up and down as NVFP4 blocks, then the tail.
#[must_use]
pub fn record_bytes(width: u32, inter: u32) -> u64 {
    let (width, inter) = (u64::from(width), u64::from(inter));
    let block = BLOCK_BYTES as u64;
    let gate = inter * (width / u64::from(BLOCK_WEIGHTS)) * block;
    let down = width * (inter / u64::from(BLOCK_WEIGHTS)) * block;
    2 * gate + down + TAIL_BYTES
}

/// Whether `dir` holds a latent pack: the index, the records, the labels and
/// the skeleton are all there.
#[must_use]
pub fn is_latent_pack(dir: &Path) -> bool {
    [INDEX, EXPERTS_FILE, LABELS, SKELETON]
        .iter()
        .all(|name| dir.join(name).is_file())
}

/// One refusal, naming the file it is about.
fn refuse(file: &'static str, why: impl std::fmt::Display) -> CoreError {
    CoreError::InvalidValue {
        key: file,
        value: why.to_string(),
    }
}

/// A file's bytes, or the absence of it named.
fn read_file(dir: &Path, name: &'static str) -> CoreResult<Vec<u8>> {
    match std::fs::read(dir.join(name)) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(CoreError::ConfigMissing {
            key: name.to_owned(),
        }),
        other => other.map_err(CoreError::Io),
    }
}

/// A file's length, or the absence of it named.
fn file_len(dir: &Path, name: &'static str) -> CoreResult<u64> {
    match std::fs::metadata(dir.join(name)) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(CoreError::ConfigMissing {
            key: name.to_owned(),
        }),
        other => other.map(|m| m.len()).map_err(CoreError::Io),
    }
}

/// The number after `key` in the index's header line.
fn header_value(line: &str, key: &str) -> Option<u32> {
    let mut tokens = line.split(|c: char| c.is_whitespace() || matches!(c, ',' | '(' | ')'));
    tokens.find(|t| *t == key)?;
    tokens.find(|t| !t.is_empty())?.parse().ok()
}

/// A latent pack's index files, parsed and checked against each other and
/// against the files' sizes.
#[derive(Debug)]
pub struct LatentPack {
    /// Width of one latent group: the contraction of gate and up, the output
    /// of down.
    pub width: u32,
    /// Latent groups, which `latent_dense_q8.bin`'s size fixes.
    pub groups: u32,
    /// Bytes of one expert's record.
    pub record_bytes: u64,
    geometry: Geometry,
    experts: PathBuf,
    projections: PathBuf,
    labels: Vec<u32>,
}

impl LatentPack {
    /// Opens the pack in `dir`.
    ///
    /// # Errors
    ///
    /// [`CoreError::ConfigMissing`] naming an absent file, and
    /// [`CoreError::InvalidValue`] naming the file whose contents or size
    /// disagree with the index or the family's shape.
    pub fn open(dir: &Path) -> CoreResult<Self> {
        Self::open_with(dir, FAMILY)
    }

    /// [`Self::open`] against `geometry`.
    fn open_with(dir: &Path, geometry: Geometry) -> CoreResult<Self> {
        let (width, record) = parse_index(dir, geometry)?;
        let experts_len = file_len(dir, EXPERTS_FILE)?;
        let want = u64::from(geometry.layers) * u64::from(geometry.experts) * record;
        if experts_len != want {
            return Err(refuse(
                EXPERTS_FILE,
                format_args!("{experts_len} bytes, the index describes {want}"),
            ));
        }
        let (groups, projections) = projection_groups(dir, geometry, width)?;
        let labels = parse_labels(dir, geometry, groups)?;
        Ok(Self {
            width,
            groups,
            record_bytes: record,
            geometry,
            experts: dir.join(EXPERTS_FILE),
            projections,
            labels,
        })
    }

    /// The group of each expert, layer-major: `labels()[layer * experts + e]`.
    #[must_use]
    pub fn labels(&self) -> &[u32] {
        &self.labels
    }

    /// The records file.
    #[must_use]
    pub fn experts_path(&self) -> &Path {
        &self.experts
    }

    /// The dense projections file.
    #[must_use]
    pub fn projection_path(&self) -> &Path {
        &self.projections
    }

    /// Bytes of one dense projection, down or up, of one layer.
    #[must_use]
    pub fn projection_bytes(&self) -> u64 {
        projection_bytes(self.groups, self.width, self.geometry.hidden)
    }

    /// Where layer `layer`'s down and up projections start in the dense
    /// projections file.
    #[must_use]
    pub fn projection_offsets(&self, layer: u32) -> (u64, u64) {
        let each = self.projection_bytes();
        (
            u64::from(layer) * 2 * each,
            (u64::from(layer) * 2 + 1) * each,
        )
    }
}

/// Bytes of one layer's down (or up) projection: `groups * width` `Q8_0` rows
/// of `hidden` inputs, or the same count of bytes laid out the other way.
fn projection_bytes(groups: u32, width: u32, hidden: u32) -> u64 {
    u64::from(groups) * u64::from(width) * u64::from(hidden / Q8_WEIGHTS) * Q8_BLOCK_BYTES as u64
}

/// Parses `native_experts.txt`: the latent width from its header, and one
/// line a layer, each of the expected type, size and (contiguous) offset.
/// Returns the width and the record's size.
fn parse_index(dir: &Path, geometry: Geometry) -> CoreResult<(u32, u64)> {
    let bytes = read_file(dir, INDEX)?;
    let text = String::from_utf8(bytes).map_err(|_| refuse(INDEX, "not UTF-8 text"))?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .filter(|line| line.starts_with('#'))
        .ok_or_else(|| refuse(INDEX, "no header line"))?;
    let width = header_value(header, "expert_embd")
        .ok_or_else(|| refuse(INDEX, "the header names no expert_embd"))?;
    let inter = header_value(header, "expert_ff")
        .ok_or_else(|| refuse(INDEX, "the header names no expert_ff"))?;
    if inter != geometry.inter {
        return Err(refuse(
            INDEX,
            format_args!("expert_ff {inter}, the family has {}", geometry.inter),
        ));
    }
    if width == 0 || !width.is_multiple_of(BLOCK_WEIGHTS) || !inter.is_multiple_of(BLOCK_WEIGHTS) {
        return Err(refuse(
            INDEX,
            format_args!("expert_embd {width} is not whole blocks of {BLOCK_WEIGHTS}"),
        ));
    }
    let record = record_bytes(width, inter);
    let stride = u64::from(geometry.experts) * record;
    let mut seen = 0u32;
    for (n, line) in lines.filter(|l| !l.trim().is_empty()).enumerate() {
        let fields: Vec<u64> = line
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()
            .map_err(|_| refuse(INDEX, format_args!("line {} is not five integers", n + 2)))?;
        let [layer, gu, down, offset, blob] = fields[..] else {
            return Err(refuse(
                INDEX,
                format_args!("line {} is not five integers", n + 2),
            ));
        };
        if layer != n as u64 {
            return Err(refuse(
                INDEX,
                format_args!("line {} names layer {layer}, expected {n}", n + 2),
            ));
        }
        if gu != u64::from(NVFP4_TYPE) || down != u64::from(NVFP4_TYPE) {
            return Err(refuse(
                INDEX,
                format_args!("layer {layer} has types {gu}/{down}, only {NVFP4_TYPE} is read"),
            ));
        }
        if offset != layer * stride {
            return Err(refuse(
                INDEX,
                format_args!(
                    "layer {layer} starts at {offset}, contiguous records put it at {}",
                    layer * stride
                ),
            ));
        }
        if blob != record {
            return Err(refuse(
                INDEX,
                format_args!("layer {layer} records are {blob} bytes, expected {record}"),
            ));
        }
        seen += 1;
    }
    if seen != geometry.layers {
        return Err(refuse(
            INDEX,
            format_args!("{seen} layers, the family has {}", geometry.layers),
        ));
    }
    Ok((width, record))
}

/// The group count the dense projections file's size implies, and its path.
///
/// Only the `Q8_0` file is read; a pack that ships the four-bit one alone is
/// refused by its name rather than reported as missing a file it was never
/// meant to have.
fn projection_groups(dir: &Path, geometry: Geometry, width: u32) -> CoreResult<(u32, PathBuf)> {
    if !dir.join(PROJECTIONS_Q8).is_file() && dir.join(PROJECTIONS_Q4).is_file() {
        return Err(refuse(
            PROJECTIONS_Q4,
            "only the Q8_0 projections are supported",
        ));
    }
    if !geometry.hidden.is_multiple_of(Q8_WEIGHTS) {
        return Err(refuse(
            PROJECTIONS_Q8,
            format_args!("hidden size {} is not whole Q8_0 blocks", geometry.hidden),
        ));
    }
    let len = file_len(dir, PROJECTIONS_Q8)?;
    let per_group = 2 * u64::from(geometry.layers) * projection_bytes(1, width, geometry.hidden);
    let groups = len / per_group;
    if groups == 0 || groups * per_group != len {
        return Err(refuse(
            PROJECTIONS_Q8,
            format_args!(
                "{len} bytes is not a whole number of groups ({per_group} bytes each, \
                 down and up, every layer)"
            ),
        ));
    }
    let groups = u32::try_from(groups)
        .map_err(|_| refuse(PROJECTIONS_Q8, format_args!("{groups} groups")))?;
    Ok((groups, dir.join(PROJECTIONS_Q8)))
}

/// Parses `labels.bin`: one little-endian `i32` per expert, each a group.
fn parse_labels(dir: &Path, geometry: Geometry, groups: u32) -> CoreResult<Vec<u32>> {
    let bytes = read_file(dir, LABELS)?;
    let count = u64::from(geometry.layers) * u64::from(geometry.experts);
    if bytes.len() as u64 != count * 4 {
        return Err(refuse(
            LABELS,
            format_args!(
                "{} bytes, expected {} (one i32 per expert)",
                bytes.len(),
                count * 4
            ),
        ));
    }
    bytes
        .chunks_exact(4)
        .enumerate()
        .map(|(i, chunk)| {
            let raw = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            u32::try_from(raw)
                .ok()
                .filter(|g| *g < groups)
                .ok_or_else(|| {
                    refuse(
                        LABELS,
                        format_args!("expert {i} is in group {raw}, the pack has {groups}"),
                    )
                })
        })
        .collect()
}

/// Rows and row bytes of a stored section of `in_dim` inputs.
fn section_rows(len: usize, in_dim: u32) -> (usize, usize) {
    assert!(
        in_dim.is_multiple_of(BLOCK_WEIGHTS) && in_dim > 0,
        "a stored row is whole blocks of {BLOCK_WEIGHTS}, got {in_dim}"
    );
    let row_bytes = in_dim as usize / BLOCK_WEIGHTS as usize * BLOCK_BYTES;
    assert!(
        len.is_multiple_of(row_bytes),
        "a stored section of {len} bytes is not whole rows of {row_bytes}"
    );
    (len / row_bytes, row_bytes)
}

/// Rewrites one stored projection's code bytes as the stream's pairwise
/// codes: `dst` is `rows * in_dim / 2` bytes, the earlier weight of each byte
/// in its low nibble.
///
/// A stored sixteen-weight sub-block holds weight `j` in the low nibble of
/// byte `j` and weight `8 + j` in the high nibble; this puts weight `2k` and
/// `2k + 1` into byte `k`. Every nibble is moved, none is changed.
///
/// # Panics
///
/// When `in_dim` is not whole blocks of 64, or `src` or `dst` is not the size
/// that `in_dim` implies for a whole number of rows.
pub fn repack_codes(src: &[u8], in_dim: u32, dst: &mut [u8]) {
    let (rows, _) = section_rows(src.len(), in_dim);
    assert_eq!(
        dst.len(),
        rows * in_dim as usize / 2,
        "the repacked codes of {rows} rows of {in_dim}"
    );
    for (block, out) in src
        .chunks_exact(BLOCK_BYTES)
        .zip(dst.chunks_exact_mut(BLOCK_CODES))
    {
        for (qs, out) in block[BLOCK_SCALES..]
            .chunks_exact(8)
            .zip(out.chunks_exact_mut(8))
        {
            for k in 0..4 {
                let (a, b) = (qs[2 * k], qs[2 * k + 1]);
                out[k] = (a & 0x0F) | (b << 4);
                out[4 + k] = (a >> 4) | (b & 0xF0);
            }
        }
    }
}

/// Copies one stored projection's group scales out of its blocks into
/// `dst`, `rows * in_dim / 16` bytes row-major, refusing a byte the format
/// does not produce.
///
/// # Errors
///
/// [`CoreError::InvalidValue`] naming `experts.bin` for a scale with its sign
/// bit set (which covers the negative NaN) or the positive NaN, `0x7F`.
///
/// # Panics
///
/// As [`repack_codes`], for `src` and `dst`.
pub fn split_scales(src: &[u8], in_dim: u32, dst: &mut [u8]) -> CoreResult<()> {
    let (rows, _) = section_rows(src.len(), in_dim);
    assert_eq!(
        dst.len(),
        rows * in_dim as usize / 16,
        "the scales of {rows} rows of {in_dim}"
    );
    for (block, out) in src
        .chunks_exact(BLOCK_BYTES)
        .zip(dst.chunks_exact_mut(BLOCK_SCALES))
    {
        let scales = &block[..BLOCK_SCALES];
        if let Some(bad) = scales.iter().find(|b| **b & 0x80 != 0 || **b == 0x7F) {
            return Err(refuse(
                EXPERTS_FILE,
                format_args!("scale byte {bad:#04x} is not a value the format stores"),
            ));
        }
        out.copy_from_slice(scales);
    }
    Ok(())
}

/// Lays the up projection out as one matrix.
///
/// `src` is `groups` blocks of `hidden` `Q8_0` rows of `width` inputs each, the
/// stored order; the result is `hidden` rows of `groups * width` inputs, row
/// `h` the `groups` rows `h` end to end, group 0 first. Whole `Q8_0` blocks are
/// moved, so no weight changes.
///
/// # Panics
///
/// When `width` is not whole `Q8_0` blocks or `src` is not the size the three
/// counts imply.
#[must_use]
pub fn interleave_up(src: &[u8], groups: u32, hidden: u32, width: u32) -> Vec<u8> {
    assert!(
        width.is_multiple_of(Q8_WEIGHTS),
        "a Q8_0 row is whole blocks of {Q8_WEIGHTS}, got {width}"
    );
    let row = (width / Q8_WEIGHTS) as usize * Q8_BLOCK_BYTES;
    let (groups, hidden) = (groups as usize, hidden as usize);
    assert_eq!(
        src.len(),
        groups * hidden * row,
        "{groups} groups of {hidden} rows of {row} bytes"
    );
    let mut out = Vec::with_capacity(src.len());
    for h in 0..hidden {
        for g in 0..groups {
            let at = (g * hidden + h) * row;
            out.extend_from_slice(&src[at..at + row]);
        }
    }
    out
}

/// The routed experts of a latent pack, one record read per array.
#[derive(Debug)]
pub struct LatentExperts {
    file: File,
    experts: u32,
    record: u64,
    width: u32,
    inter: u32,
    layout: Vec<u64>,
    /// Bytes of one stored gate (or up) section, which is also the down's.
    section: usize,
}

impl LatentExperts {
    /// Opens the pack's records file, bypassing the page cache.
    ///
    /// # Errors
    ///
    /// [`CoreError::Io`] when the file cannot be opened, and
    /// [`CoreError::InvalidValue`] for a record too large to address.
    pub fn new(pack: &LatentPack) -> CoreResult<Self> {
        let (width, inter) = (pack.width, pack.geometry.inter);
        let mut layout = Vec::with_capacity(9);
        for (in_dim, out_dim) in [(width, inter), (width, inter), (inter, width)] {
            let (codes, scales) = nvfp4_bytes(in_dim, out_dim);
            layout.extend([codes, scales, GLOBAL_BYTES as u64]);
        }
        let section = usize::try_from((pack.record_bytes - TAIL_BYTES) / 3)
            .map_err(|_| refuse(EXPERTS_FILE, "a record section larger than memory"))?;
        Ok(Self {
            file: open_uncached(&pack.experts).map_err(CoreError::Io)?,
            experts: pack.geometry.experts,
            record: pack.record_bytes,
            width,
            inter,
            layout,
            section,
        })
    }
}

impl ExpertSource for LatentExperts {
    fn parts(&self) -> &'static [WeightPart] {
        &PARTS
    }

    fn layout(&self) -> Vec<u64> {
        self.layout.clone()
    }

    fn read(&self, layer: u32, expert: u32, array: usize, dst: &mut [u8]) -> CoreResult<()> {
        let want = *self.layout.get(array).ok_or(CoreError::OutOfBounds {
            offset: array as u64,
            len: 1,
            buffer_bytes: self.layout.len() as u64,
        })?;
        if dst.len() as u64 != want {
            return Err(CoreError::ShapeMismatch {
                field: "latent expert array bytes",
                expected: want,
                got: dst.len() as u64,
            });
        }
        let (proj, kind) = (array / 3, array % 3);
        let base = (u64::from(layer) * u64::from(self.experts) + u64::from(expert)) * self.record;
        if kind == 2 {
            let mut bytes = [0u8; GLOBAL_BYTES];
            self.file
                .read_exact_at(
                    &mut bytes,
                    base + 3 * self.section as u64 + (proj * GLOBAL_BYTES) as u64,
                )
                .map_err(CoreError::Io)?;
            if !f32::from_le_bytes(bytes).is_finite() {
                return Err(refuse(
                    EXPERTS_FILE,
                    format_args!("layer {layer} expert {expert} has a non-finite global scale"),
                ));
            }
            dst.copy_from_slice(&bytes);
            return Ok(());
        }
        // Gate and up contract over the latent width; down over the inner width.
        let in_dim = if proj == 2 { self.inter } else { self.width };
        let mut stored = vec![0u8; self.section];
        self.file
            .read_exact_at(&mut stored, base + (proj * self.section) as u64)
            .map_err(CoreError::Io)?;
        if kind == 0 {
            repack_codes(&stored, in_dim, dst);
            Ok(())
        } else {
            split_scales(&stored, in_dim, dst)
        }
    }
}

#[cfg(test)]
#[path = "latent_tests.rs"]
mod tests;
