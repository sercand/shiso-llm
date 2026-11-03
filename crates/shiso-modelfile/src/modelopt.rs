//! The modelopt mixed-precision checkpoint directory: three shards, 94,393
//! arrays, and three storage schemes that share one file.
//!
//! The third container in this crate, and the first whose weights are not all
//! stored the same way. `crates/shiso-modelfile/src/reader.rs` reads a GGUF file
//! whose blocks the kernels read in place; [`crate::safetensors`] reads an
//! MLX/OptiQ directory where every quantized weight is one affine triple. This
//! reads what `modelopt` writes for `quant_algo: MIXED_PRECISION`, where three
//! schemes sit side by side in one shard and which one a tensor is under is a
//! fact about that tensor:
//!
//! | scheme | `weight` | companions |
//! |---|---|---|
//! | plain | `BF16` / `F16` / `F32` | none |
//! | FP8 | `F8_E4M3` `[out, in]` | `weight_scale` `f32`, `input_scale` `f32` |
//! | NVFP4 | `U8` `[out, in/2]` | `weight_scale` `F8_E4M3` `[out, in/16]`, `weight_scale_2` `f32` |
//! | MXFP8 | `F8_E4M3` `[out, in]` | `weight_scale` `U8` `[out, in/32]` (`ue8m0`) |
//!
//! FP8 and MXFP8 share a weight format and differ in the scale beside it - one
//! `f32` for the tensor, or one power of two per 32 weights - so which of the
//! two a tensor is under is read off that scale's own format and shape.
//!
//! # Describing is eager, decoding is not
//!
//! Opening parses every shard header, groups every array with its companions,
//! checks each group's shapes against each other, and reads the per-tensor
//! scalar scales - which are four bytes each and are needed to describe a
//! tensor at all. Nothing else is read. A 23.4 GB checkpoint opens without
//! touching a weight.
//!
//! # The scalar scales are read at open, and that is deliberate
//!
//! A tensor's `weight_scale_2` is a single `f32` with a rank-zero shape, and
//! there are 30,841 of them. They are read here rather than at decode because
//! dropping one is the failure NVFP4 punishes hardest: it rescales a whole
//! tensor by about 31,458x, and the next normalization very nearly hides it.
//! A scale that is part of the *description* cannot be forgotten by a decode
//! path; one fetched on demand can.
//!
//! # What is checked against what
//!
//! Every group is checked for internal agreement, which is the only check a
//! container can make without a config: an NVFP4 weight of `[r, c]` bytes must
//! have a scale of `[r, c * 2 / 16]`, because those two numbers come from the
//! same matrix and a file where they disagree is a file whose halves were
//! written at different times. What each tensor *means* - which layer, which
//! role, how many of them there should be - is the family's business, above
//! this module.
//!
//! # Mutating the files
//!
//! The shards stay mapped for as long as an open file lives. Replacing,
//! truncating or rewriting any of them in the meantime is undefined behaviour.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::nvfp4::{GROUP, Nvfp4Layout, PER_BYTE};
use crate::reader::{GgufError, GgufResult};
use crate::st_header::{RawTensor, Shard, StDType, missing_file};

/// Producer tag the shard headers carry, which is what a `torch` export writes.
const HEADER_FORMAT: &str = "pt";
/// Optional map from tensor name to shard, authoritative when present.
const SHARD_INDEX_FILE: &str = "model.safetensors.index.json";
/// Shard a single-file export writes.
const PRIMARY_SHARD: &str = "model.safetensors";
/// Trailing component of the weight array of any scheme.
const WEIGHT_SUFFIX: &str = ".weight";
/// Trailing component of the per-group or per-tensor weight scale.
const WEIGHT_SCALE_SUFFIX: &str = ".weight_scale";
/// Trailing component of NVFP4's per-tensor scale, above the per-group one.
const GLOBAL_SCALE_SUFFIX: &str = ".weight_scale_2";
/// Trailing component of an FP8 target's activation scale.
const INPUT_SCALE_SUFFIX: &str = ".input_scale";

/// How one tensor's values are stored.
///
/// Three variants and each carries what a decode of that scheme needs and
/// nothing more, so a caller that matches on this cannot reach for a scale the
/// scheme does not have.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Stored {
    /// Stored at a float format the reader widens, with no scale at all.
    Plain {
        /// The format as the header declares it.
        dtype: StDType,
    },
    /// FP8 E4M3 weights with FP8 activations: W8A8.
    Fp8 {
        /// The one `f32` every weight of this tensor is multiplied by.
        weight_scale: f32,
        /// The scale the activations were calibrated at, when the export
        /// states one.
        ///
        /// `None` is not "one": it is a target the export quantized the
        /// weights of and not the activations, which is a different pass and
        /// has to be a different decision at the kernel.
        input_scale: Option<f32>,
    },
    /// NVFP4: four-bit codes, per-group `e4m3` scales, one `f32` above both.
    Nvfp4 {
        /// The whole tensor's scale, which the group scales are relative to.
        global: f32,
    },
    /// MXFP8: `e4m3` codes under one `ue8m0` power of two per 32 weights
    /// along a row ([`crate::mxfp8`]). No tensor-wide scale.
    Mxfp8,
}

impl Stored {
    /// A short name for this scheme, for a census or a log line.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Plain { .. } => "plain",
            Self::Fp8 { .. } => "fp8",
            Self::Nvfp4 { .. } => "nvfp4",
            Self::Mxfp8 => "mxfp8",
        }
    }
}

/// Where one source array lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span {
    /// Index into the open shard list.
    shard: usize,
    /// Payload start, relative to that shard's payload base.
    start: u64,
    /// Payload end, relative to that shard's payload base.
    end: u64,
}

impl Span {
    /// The span one header entry declares.
    const fn of(shard: usize, raw: &RawTensor) -> Self {
        Self {
            shard,
            start: raw.data_offsets[0],
            end: raw.data_offsets[1],
        }
    }

    /// Bytes the span covers.
    const fn len(self) -> u64 {
        self.end.saturating_sub(self.start)
    }
}

/// One tensor as this container describes it.
///
/// The name is the *base* name - `...q_proj`, not `...q_proj.weight` - for
/// every scheme including the plain one, so a caller never has to know whether
/// the array it wants had companions.
#[derive(Debug)]
pub struct ModeloptTensor {
    /// Base name, without the `.weight` suffix.
    name: Box<str>,
    /// Rows of the logical matrix, the slowest-varying axis.
    ///
    /// The LOGICAL shape, not the stored one: an NVFP4 weight stored as
    /// `[512, 1024]` bytes is a `[512, 2048]` matrix, and a caller that read
    /// the header's own shape would size every buffer at half.
    rows: u64,
    /// Values per row of the logical matrix.
    cols: u64,
    /// The shape the header declared, whole.
    ///
    /// [`ModeloptTensor::rows`] and [`ModeloptTensor::cols`] fold everything
    /// past the first axis together, which is what a projection wants and what
    /// a rank-3 or rank-5 tensor is not: this checkpoint's recurrent
    /// convolution is `[8192, 1, 4]` and its vision tower's patch projection
    /// is `[1152, 3, 2, 16, 16]`, and a loader that checks a manifest against a
    /// declared shape needs the shape rather than its product.
    declared: Vec<u64>,
    /// How its values are stored.
    stored: Stored,
    /// The weight array.
    weight: Span,
    /// The per-group scale array, for NVFP4 and MXFP8.
    group_scales: Option<Span>,
}

impl ModeloptTensor {
    /// Base name, without the `.weight` suffix.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Rows of the logical matrix.
    #[must_use]
    pub const fn rows(&self) -> u64 {
        self.rows
    }

    /// Values per row of the logical matrix.
    #[must_use]
    pub const fn cols(&self) -> u64 {
        self.cols
    }

    /// The shape the header declared, slowest axis first.
    ///
    /// The STORED shape, not the logical one: an NVFP4 weight's last axis is
    /// half the matrix's width here, exactly as the file writes it.
    #[must_use]
    pub fn declared(&self) -> &[u64] {
        &self.declared
    }

    /// Logical values the tensor holds.
    #[must_use]
    pub const fn elements(&self) -> u64 {
        self.rows.saturating_mul(self.cols)
    }

    /// How its values are stored.
    #[must_use]
    pub const fn stored(&self) -> Stored {
        self.stored
    }
}

/// The three arrays an NVFP4 tensor is, and the layout that reads them.
///
/// A named struct rather than a tuple, on the precedent of
/// [`crate::safetensors::AffineTriple`]: two same-typed byte slices in a return
/// position are two slices a caller can transpose, and transposing these reads
/// the scales as codes and the codes as scales.
#[derive(Debug, Clone, Copy)]
pub struct Nvfp4Parts<'a> {
    /// Packed four-bit codes, two to a byte, low nibble first.
    pub weights: &'a [u8],
    /// One `e4m3` byte per group of [`GROUP`] weights along a row.
    pub scales: &'a [u8],
    /// The logical geometry and the per-tensor scale above both arrays.
    pub layout: Nvfp4Layout,
}

/// The two arrays an MXFP8 tensor is, and its layout.
///
/// Named fields for the reason [`Nvfp4Parts`] has them: two byte slices in a
/// tuple are two a caller can transpose.
#[derive(Debug, Clone, Copy)]
pub struct Mxfp8Parts<'a> {
    /// One `e4m3` byte per weight, row-major.
    pub codes: &'a [u8],
    /// One `ue8m0` byte per group of 32 weights along a row.
    pub scales: &'a [u8],
    /// The logical geometry.
    pub layout: crate::mxfp8::Mxfp8Layout,
}

/// The shard index, when the directory has one.
#[derive(Debug, Deserialize)]
struct ShardIndex {
    /// Tensor name to the shard file holding it, relative to the directory.
    weight_map: HashMap<String, String>,
}

/// How many tensors of each scheme a directory holds.
///
/// A census rather than a length, because "291 quantized targets" and "94,393
/// arrays" and "31,333 tensors" are three different true statements about the
/// same directory and a caller asserting one of them should have to say which.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct Census {
    /// Header entries across every shard that was mapped, companions included.
    pub arrays: usize,
    /// Tensors kept after the name filter, companions folded in.
    pub tensors: usize,
    /// Of those, stored at a float format with no scale.
    pub plain: usize,
    /// Of those, stored as FP8 E4M3 with an `f32` scale.
    pub fp8: usize,
    /// Of those, stored as NVFP4 under three levels of scale.
    pub nvfp4: usize,
    /// Of those, stored as MXFP8 under a power of two per 32 weights.
    pub mxfp8: usize,
}

/// An open modelopt checkpoint directory.
///
/// # Mutating the files
///
/// The shards stay mapped for as long as this value lives. Replacing,
/// truncating or rewriting any of them in the meantime is undefined behaviour,
/// so treat the directory as immutable while it is open.
#[derive(Debug)]
pub struct ModeloptFile {
    /// Directory the checkpoint was opened from.
    dir: PathBuf,
    /// Mapped shards, in the order they were opened.
    shards: Vec<Shard>,
    /// Described tensors, sorted by name.
    tensors: Vec<ModeloptTensor>,
    /// Base name to index into `tensors`.
    index: HashMap<Box<str>, usize>,
    /// What was found, counted by scheme.
    census: Census,
}

impl ModeloptFile {
    /// Opens the directory at `dir`, keeping the tensors whose names start with
    /// one of `prefixes`.
    ///
    /// The filter is a caller's argument rather than this module's table
    /// because a directory of this kind is several models: a language model, a
    /// vision tower and a draft head, each with its own loader above. A caller
    /// that wants the language model passes its prefixes and the draft head's
    /// 785 arrays are described for nobody. An empty slice keeps everything.
    ///
    /// # Mutating the files
    ///
    /// Do not point this at a directory something else is writing, and do not
    /// replace or truncate a shard while the returned value is alive. The
    /// mappings are windows onto those files and the borrow checker cannot see
    /// the other writer.
    ///
    /// # Errors
    ///
    /// [`GgufError::Io`] for a missing file or a header that is not JSON,
    /// [`GgufError::ArchMismatch`] for a shard from another producer,
    /// [`GgufError::Truncated`], [`GgufError::OutOfBounds`] or
    /// [`GgufError::SizeMismatch`] for a span that does not fit its file or its
    /// shape, [`GgufError::LayoutMismatch`] for a group whose arrays do not
    /// describe one matrix, and [`GgufError::MissingTensor`] when a scheme's
    /// required companion is absent.
    pub fn open(dir: impl AsRef<Path>, prefixes: &[&str]) -> GgufResult<Self> {
        let dir = dir.as_ref();
        let mut shards = Vec::new();
        let mut arrays: HashMap<String, (usize, RawTensor)> = HashMap::new();
        let mut array_count = 0usize;
        for name in shard_files(dir)? {
            let path = dir.join(&name);
            let (shard, tensors) = Shard::open(&path, HEADER_FORMAT)?;
            let slot = shards.len();
            shards.push(shard);
            array_count += tensors.len();
            for (tensor, raw) in tensors {
                arrays.insert(tensor, (slot, raw));
            }
        }

        let tensors = describe(&arrays, &shards, prefixes)?;
        let mut census = Census {
            arrays: array_count,
            tensors: tensors.len(),
            ..Census::default()
        };
        for tensor in &tensors {
            match tensor.stored {
                Stored::Plain { .. } => census.plain += 1,
                Stored::Fp8 { .. } => census.fp8 += 1,
                Stored::Nvfp4 { .. } => census.nvfp4 += 1,
                Stored::Mxfp8 => census.mxfp8 += 1,
            }
        }
        let index = tensors
            .iter()
            .enumerate()
            .map(|(at, tensor)| (tensor.name.clone(), at))
            .collect();
        tracing::debug!(
            dir = %dir.display(),
            shards = shards.len(),
            arrays = census.arrays,
            tensors = census.tensors,
            plain = census.plain,
            fp8 = census.fp8,
            nvfp4 = census.nvfp4,
            mxfp8 = census.mxfp8,
            "opened a modelopt checkpoint directory"
        );
        Ok(Self {
            dir: dir.to_path_buf(),
            shards,
            tensors,
            index,
            census,
        })
    }

    /// The directory this checkpoint was opened from.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// What was found, counted by scheme.
    #[must_use]
    pub const fn census(&self) -> Census {
        self.census
    }

    /// Shards that were mapped.
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Every described tensor, sorted by name.
    #[must_use]
    pub fn tensors(&self) -> impl ExactSizeIterator<Item = &ModeloptTensor> {
        self.tensors.iter()
    }

    /// The tensor with this base name, or `None`.
    ///
    /// A trailing `.weight` is accepted and stripped, so a caller holding a
    /// name out of the checkpoint's own config document does not have to know
    /// which form this container keeps.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ModeloptTensor> {
        let base = name.strip_suffix(WEIGHT_SUFFIX).unwrap_or(name);
        self.index.get(base).and_then(|&at| self.tensors.get(at))
    }

    /// The weight array's bytes, exactly as the shard stores them.
    ///
    /// # Errors
    ///
    /// [`GgufError::OutOfBounds`] when the span is not inside its shard, which
    /// `open` has already checked - so a failure here is a shard that changed
    /// underneath the mapping.
    pub fn weight_bytes(&self, tensor: &ModeloptTensor) -> GgufResult<&[u8]> {
        self.span(tensor.weight)
    }

    /// The per-group `e4m3` scale bytes and the layout that decodes them,
    /// for an NVFP4 tensor.
    ///
    /// `Ok(None)` for a tensor stored under any other scheme, so a caller can
    /// ask without matching first.
    ///
    /// # Errors
    ///
    /// Whatever [`ModeloptFile::weight_bytes`] reports.
    pub fn nvfp4_parts(&self, tensor: &ModeloptTensor) -> GgufResult<Option<Nvfp4Parts<'_>>> {
        let Stored::Nvfp4 { global } = tensor.stored else {
            return Ok(None);
        };
        let scales = tensor
            .group_scales
            .ok_or_else(|| GgufError::MissingTensor {
                name: format!("{}{WEIGHT_SCALE_SUFFIX}", tensor.name).into_boxed_str(),
            })?;
        Ok(Some(Nvfp4Parts {
            weights: self.span(tensor.weight)?,
            scales: self.span(scales)?,
            layout: Nvfp4Layout::new(
                usize::try_from(tensor.rows).unwrap_or(usize::MAX),
                usize::try_from(tensor.cols).unwrap_or(usize::MAX),
                global,
            ),
        }))
    }

    /// The `e4m3` codes and `ue8m0` group scales of an MXFP8 tensor, and its
    /// layout, for [`crate::mxfp8::dequant_mxfp8`] or a kernel.
    ///
    /// `Ok(None)` for a tensor stored under any other scheme.
    ///
    /// # Errors
    ///
    /// Whatever [`ModeloptFile::weight_bytes`] reports.
    pub fn mxfp8_parts(&self, tensor: &ModeloptTensor) -> GgufResult<Option<Mxfp8Parts<'_>>> {
        let (Stored::Mxfp8, Some(scales)) = (tensor.stored, tensor.group_scales) else {
            return Ok(None);
        };
        Ok(Some(Mxfp8Parts {
            codes: self.span(tensor.weight)?,
            scales: self.span(scales)?,
            layout: crate::mxfp8::Mxfp8Layout {
                rows: usize::try_from(tensor.rows).unwrap_or(usize::MAX),
                cols: usize::try_from(tensor.cols).unwrap_or(usize::MAX),
            },
        }))
    }

    /// Bytes for one span.
    fn span(&self, span: Span) -> GgufResult<&[u8]> {
        let shard = self
            .shards
            .get(span.shard)
            .ok_or_else(|| GgufError::OutOfBounds {
                tensor: Box::from(self.dir.to_string_lossy().as_ref()),
            })?;
        shard.payload(span.start, span.end)
    }
}

/// Groups the raw arrays into tensors and checks each group.
///
/// Every lookup below is a hash lookup on a name built from the base, so the
/// whole pass is linear in the array count. That is stated because it is not
/// free to get wrong at this size: the sibling container resolves a tensor's
/// quantization parameters by scanning its per-tensor table when the direct
/// lookup misses, which at 94,393 arrays would be a quadratic open.
fn describe(
    arrays: &HashMap<String, (usize, RawTensor)>,
    shards: &[Shard],
    prefixes: &[&str],
) -> GgufResult<Vec<ModeloptTensor>> {
    let mut tensors = Vec::new();
    for (name, (shard, raw)) in arrays {
        if is_companion(name, arrays) {
            continue;
        }
        if !prefixes.is_empty() && !prefixes.iter().any(|prefix| name.starts_with(prefix)) {
            continue;
        }
        // The `.weight` suffix is stripped so a quantized tensor is named the
        // way its companions and the checkpoint's own quantization document
        // name it. Everything else keeps the name it has: this checkpoint
        // carries 226 arrays that are not anybody's `.weight` - every bias in
        // the vision tower, and the recurrent block's `A_log` and `dt_bias` -
        // and an earlier draft of this function skipped all of them. That is
        // the exact failure this task exists to prevent, caught by the census
        // below rather than by a recurrent block with no decay.
        let base = name.strip_suffix(WEIGHT_SUFFIX).unwrap_or(name.as_str());
        tensors.push(describe_one(base, *shard, raw, arrays, shards)?);
    }
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    if let Some(pair) = tensors.windows(2).find(|pair| pair[0].name == pair[1].name) {
        // `x` and `x.weight` both present would collapse to one index entry and
        // hide one of them. It does not happen in any export seen here, and a
        // silent loss is what this whole function is guarding against.
        return Err(GgufError::LayoutMismatch {
            tensor: pair[0].name.clone(),
            want: Box::from("one array per name"),
            got: Box::from("a bare name and its own .weight, which describe to the same tensor"),
        });
    }
    Ok(tensors)
}

/// Whether `name` is some weight's scale rather than a tensor of its own.
///
/// The test is not the suffix alone. A tensor genuinely called `...input_scale`
/// with no `...weight` beside it is a tensor; this checkpoint has none, and
/// deciding by suffix alone would be a rule that silently drops one in a
/// checkpoint that does.
fn is_companion(name: &str, arrays: &HashMap<String, (usize, RawTensor)>) -> bool {
    [GLOBAL_SCALE_SUFFIX, WEIGHT_SCALE_SUFFIX, INPUT_SCALE_SUFFIX]
        .iter()
        .any(|suffix| {
            name.strip_suffix(suffix)
                .is_some_and(|base| arrays.contains_key(&format!("{base}{WEIGHT_SUFFIX}")))
        })
}

/// Describes one tensor from its weight array and whatever sits beside it.
///
/// The scheme is read off the weight's own format rather than off a config,
/// and then the companions that scheme requires are demanded by name. A
/// checkpoint whose config and whose bytes disagree is therefore refused by
/// the bytes, which are the thing the kernels will actually read.
fn describe_one(
    base: &str,
    shard: usize,
    raw: &RawTensor,
    arrays: &HashMap<String, (usize, RawTensor)>,
    shards: &[Shard],
) -> GgufResult<ModeloptTensor> {
    let dtype = raw.parsed_dtype(base)?;
    let weight = Span::of(shard, raw);
    match dtype {
        StDType::Bf16 | StDType::F16 | StDType::F32 => {
            let (rows, cols) = leading_and_rest(base, &raw.shape)?;
            Ok(ModeloptTensor {
                name: base.into(),
                rows,
                cols,
                declared: raw.shape.clone(),
                stored: Stored::Plain { dtype },
                weight,
                group_scales: None,
            })
        }
        StDType::F8E4M3 => {
            let (rows, cols) = two_dims(base, &raw.shape)?;
            if let Some(tensor) = describe_mxfp8(base, rows, cols, &raw.shape, weight, arrays)? {
                return Ok(tensor);
            }
            let weight_scale = scalar(base, WEIGHT_SCALE_SUFFIX, arrays, shards)?;
            let input_scale = optional_scalar(base, INPUT_SCALE_SUFFIX, arrays, shards)?;
            Ok(ModeloptTensor {
                name: base.into(),
                rows,
                cols,
                declared: raw.shape.clone(),
                stored: Stored::Fp8 {
                    weight_scale,
                    input_scale,
                },
                weight,
                group_scales: None,
            })
        }
        StDType::U8 => {
            let (rows, packed) = two_dims(base, &raw.shape)?;
            let cols =
                packed
                    .checked_mul(PER_BYTE as u64)
                    .ok_or_else(|| GgufError::TensorTooLarge {
                        tensor: base.into(),
                    })?;
            let (scale_shard, scale_raw) = companion(base, WEIGHT_SCALE_SUFFIX, arrays)?;
            let scale_dtype = scale_raw.parsed_dtype(base)?;
            let (scale_rows, groups) = two_dims(base, &scale_raw.shape)?;
            // The two arrays come from one matrix, so they have to say the same
            // thing about it. A scale array a group short reads every later
            // group's scale from the row below, which decodes to finite numbers
            // everywhere and to the wrong matrix.
            if scale_dtype != StDType::F8E4M3
                || scale_rows != rows
                || groups.checked_mul(GROUP as u64) != Some(cols)
            {
                return Err(GgufError::LayoutMismatch {
                    tensor: base.into(),
                    want: Box::from("e4m3 scales, one per group of 16 along each packed row"),
                    got: Box::from(format!(
                        "{} {:?} beside a [{rows}, {packed}] packed weight",
                        scale_raw.dtype, scale_raw.shape
                    )),
                });
            }
            let global = scalar(base, GLOBAL_SCALE_SUFFIX, arrays, shards)?;
            Ok(ModeloptTensor {
                name: base.into(),
                rows,
                cols,
                declared: raw.shape.clone(),
                stored: Stored::Nvfp4 { global },
                weight,
                group_scales: Some(Span::of(scale_shard, scale_raw)),
            })
        }
        StDType::U32 => Err(GgufError::LayoutMismatch {
            tensor: base.into(),
            want: Box::from("a format this container stores"),
            got: Box::from("packed u32 words, which are the MLX container's"),
        }),
        StDType::I64 | StDType::I16 | StDType::I32 => Err(GgufError::LayoutMismatch {
            tensor: base.into(),
            want: Box::from("a format this container stores"),
            got: Box::from("64-bit integers, which no weight of this container is"),
        }),
        StDType::I8 => Err(GgufError::LayoutMismatch {
            tensor: base.into(),
            want: Box::from("a format this container stores"),
            got: Box::from("signed byte codes, which are the per-row INT8 image export's"),
        }),
    }
}

/// Describes an `e4m3` weight as MXFP8 when the scale beside it is a `U8`
/// matrix of one byte per 32 weights, and returns `None` when it is not one,
/// leaving the tensor to the per-tensor FP8 arm.
///
/// A `U8` scale of any other shape is refused here rather than handed on: it
/// is not the scalar the FP8 arm wants either, and the FP8 arm's refusal would
/// name the wrong scheme.
fn describe_mxfp8(
    base: &str,
    rows: u64,
    cols: u64,
    declared: &[u64],
    weight: Span,
    arrays: &HashMap<String, (usize, RawTensor)>,
) -> GgufResult<Option<ModeloptTensor>> {
    let Ok((scale_shard, scale_raw)) = companion(base, WEIGHT_SCALE_SUFFIX, arrays) else {
        return Ok(None);
    };
    if scale_raw.parsed_dtype(base)? != StDType::U8 {
        return Ok(None);
    }
    let group = crate::mxfp8::GROUP as u64;
    if scale_raw.shape != [rows, cols / group] || !cols.is_multiple_of(group) {
        return Err(GgufError::LayoutMismatch {
            tensor: base.into(),
            want: Box::from("ue8m0 scales, one byte per group of 32 along each row"),
            got: Box::from(format!(
                "U8 {:?} beside a [{rows}, {cols}] e4m3 weight",
                scale_raw.shape
            )),
        });
    }
    Ok(Some(ModeloptTensor {
        name: base.into(),
        rows,
        cols,
        declared: declared.to_vec(),
        stored: Stored::Mxfp8,
        weight,
        group_scales: Some(Span::of(scale_shard, scale_raw)),
    }))
}

/// Finds one of a tensor's companion arrays.
fn companion<'a>(
    base: &str,
    suffix: &str,
    arrays: &'a HashMap<String, (usize, RawTensor)>,
) -> GgufResult<(usize, &'a RawTensor)> {
    let name = format!("{base}{suffix}");
    arrays
        .get(&name)
        .map(|(shard, raw)| (*shard, raw))
        .ok_or_else(|| GgufError::MissingTensor {
            name: name.into_boxed_str(),
        })
}

/// Reads a companion that is a single `f32`.
///
/// The checkpoint writes these with a rank-zero shape, which is one element
/// and four bytes. Both are checked, because a scale read from a four-byte
/// window of the wrong array is a finite number.
fn scalar(
    base: &str,
    suffix: &str,
    arrays: &HashMap<String, (usize, RawTensor)>,
    shards: &[Shard],
) -> GgufResult<f32> {
    let (shard, raw) = companion(base, suffix, arrays)?;
    read_scalar(base, suffix, shard, raw, shards)
}

/// The same, for a companion a scheme does not require.
///
/// Returns `Ok(None)` when the export names no such companion, which is a
/// fact about the target rather than a refusal: an FP8 tensor with no
/// `input_scale` is one whose ACTIVATIONS were not quantized, and that is a
/// different launch rather than a missing file. `Stored::Fp8` carries the
/// distinction up unchanged for the same reason.
fn optional_scalar(
    base: &str,
    suffix: &str,
    arrays: &HashMap<String, (usize, RawTensor)>,
    shards: &[Shard],
) -> GgufResult<Option<f32>> {
    let name = format!("{base}{suffix}");
    let Some((shard, raw)) = arrays.get(&name) else {
        return Ok(None);
    };
    read_scalar(base, suffix, *shard, raw, shards).map(Some)
}

/// Reads four bytes of one array as an `f32`.
fn read_scalar(
    base: &str,
    suffix: &str,
    shard: usize,
    raw: &RawTensor,
    shards: &[Shard],
) -> GgufResult<f32> {
    let name = format!("{base}{suffix}");
    let dtype = raw.parsed_dtype(&name)?;
    let span = Span::of(shard, raw);
    if dtype != StDType::F32 || raw.elements()? != 1 || span.len() != 4 {
        return Err(GgufError::LayoutMismatch {
            tensor: name.into_boxed_str(),
            want: Box::from("one f32"),
            got: Box::from(format!("{} {:?}", raw.dtype, raw.shape)),
        });
    }
    let shard = shards.get(shard).ok_or_else(|| GgufError::OutOfBounds {
        tensor: Box::from(base),
    })?;
    let bytes: [u8; 4] = shard
        .payload(span.start, span.end)?
        .try_into()
        .map_err(|_| GgufError::OutOfBounds {
            tensor: Box::from(base),
        })?;
    Ok(f32::from_le_bytes(bytes))
}

/// Reads a rank-2 shape, which is the only rank a quantized weight can have.
fn two_dims(tensor: &str, dims: &[u64]) -> GgufResult<(u64, u64)> {
    match dims {
        [rows, cols] => Ok((*rows, *cols)),
        _ => Err(GgufError::LayoutMismatch {
            tensor: tensor.into(),
            want: Box::from("a two-dimensional array"),
            got: Box::from(format!("{dims:?}")),
        }),
    }
}

/// Splits any shape into its leading axis and the values under it.
///
/// A plain tensor here may be rank 1 (a norm), rank 2 (an embedding table) or
/// rank 3 (the recurrent convolution, `[channels, 1, taps]`), so unlike
/// [`two_dims`] this folds everything past the first axis into one length.
fn leading_and_rest(name: &str, dims: &[u64]) -> GgufResult<(u64, u64)> {
    let Some((&rows, rest)) = dims.split_first() else {
        return Ok((1, 1));
    };
    let mut cols: u64 = 1;
    for &dim in rest {
        cols = cols
            .checked_mul(dim)
            .ok_or_else(|| GgufError::TensorTooLarge {
                tensor: name.into(),
            })?;
    }
    Ok((rows, cols))
}

/// Names the shards to map, relative to `dir`.
///
/// Every shard the index names is mapped. Unlike the MLX container this does
/// not narrow the shard set by tensor name: this checkpoint's three shards each
/// hold language-model tensors, so narrowing would map all three anyway and the
/// filter that does save work is the per-tensor one in [`describe`].
fn shard_files(dir: &Path) -> GgufResult<Vec<String>> {
    let index_path = dir.join(SHARD_INDEX_FILE);
    if !index_path.is_file() {
        let primary = dir.join(PRIMARY_SHARD);
        if !primary.is_file() {
            return Err(missing_file(
                &primary,
                &io::Error::from(io::ErrorKind::NotFound),
            ));
        }
        return Ok(vec![PRIMARY_SHARD.to_string()]);
    }
    let text =
        std::fs::read_to_string(&index_path).map_err(|err| missing_file(&index_path, &err))?;
    let index: ShardIndex = serde_json::from_str(&text).map_err(|err| {
        GgufError::Io(io::Error::other(format!(
            "{}: shard index does not parse: {err}",
            index_path.display()
        )))
    })?;
    let mut files: Vec<String> = Vec::new();
    for shard in index.weight_map.values() {
        if !files.contains(shard) {
            files.push(shard.clone());
        }
    }
    if files.is_empty() {
        return Err(GgufError::MissingTensor {
            name: Box::from("a shard index naming no shard"),
        });
    }
    files.sort();
    Ok(files)
}

#[cfg(test)]
#[path = "modelopt_tests.rs"]
mod tests;
