//! The rotation a rotated checkpoint of this family was folded under, read
//! from whichever container states it and checked before anything loads.
//!
//! Both of this family's rotated exports state the same contract, under the
//! same keys: the GGUF carries it as metadata, an MLX directory as
//! `hadamard.json`. Version 1 of that contract says:
//!
//! - every matrix named in `weight_names` had `H(x * signs)` folded into its
//!   input axis, where `H` is the normalized Sylvester-ordered Walsh-Hadamard
//!   transform over each consecutive `block_size` inputs and `signs` is the
//!   vector for the matrix's input width;
//! - every table named in `inverse_weight_names` stores rotated rows, which a
//!   gather takes back with `signs * H(row)`;
//! - `sign_widths` lists the widths that have a vector and `sign_values` holds
//!   the vectors back to back;
//! - `gdn_v_grouped` says the recurrent output projection's columns were
//!   folded in grouped value-head order, while the activation this family's
//!   kernels produce is tiled. See [`HeadOrder`].
//!
//! Anything else is refused. A rotated file read without its rotation, or with
//! a different one, does not fail: it produces fluent nonsense, which is the
//! failure this module exists to turn into an error.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde::Deserialize;
use shiso_core::error::{CoreError, CoreResult};
use shiso_engine::rotation::HeadOrder;
use shiso_modelfile::metadata::{GgufMetadata, MetaValue};

use crate::qwen35::st_names::{LM_PREFIX, map_role};

/// The contract version this reader implements.
const VERSION: u32 = 1;
/// The transform version 1 names.
const TRANSFORM: &str = "normalized-sylvester-walsh-hadamard";
/// The axis version 1 rotates.
const AXIS: &str = "input-last-dimension";
/// The only sign mode this reader accepts: vectors stated in the file.
const SIGN_MODE: &str = "explicit";
/// The block sizes a rotation may use.
const BLOCKS: [u32; 4] = [512, 1024, 2048, 4096];
/// Prefix of every key of the contract.
const PREFIX: &str = "prism.hadamard.";
/// The MLX directory's statement of the contract.
pub const MLX_FILE: &str = "hadamard.json";

/// How one stored tensor relates to the rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rotated {
    /// A matrix whose input passes through the forward rotation.
    Forward,
    /// A table whose gathered rows pass through the inverse rotation.
    Inverse,
}

/// The checked contract, keyed by this family's GGUF tensor names.
#[derive(Debug, Clone)]
pub struct RotationPlan {
    block: u32,
    signs: BTreeMap<u32, Vec<f32>>,
    roles: HashMap<String, Rotated>,
    grouped_value_heads: bool,
}

/// The contract as either container states it, before any check.
#[derive(Debug, Default, Deserialize)]
struct Raw {
    #[serde(rename = "prism.hadamard.version")]
    version: Option<u32>,
    #[serde(rename = "prism.hadamard.block_size")]
    block: Option<u32>,
    #[serde(rename = "prism.hadamard.transform")]
    transform: Option<String>,
    #[serde(rename = "prism.hadamard.axis")]
    axis: Option<String>,
    #[serde(rename = "prism.hadamard.sign_mode")]
    sign_mode: Option<String>,
    #[serde(rename = "prism.hadamard.weight_names", default)]
    forward: Vec<String>,
    #[serde(rename = "prism.hadamard.inverse_weight_names", default)]
    inverse: Vec<String>,
    #[serde(rename = "prism.hadamard.sign_widths", default)]
    widths: Vec<i64>,
    #[serde(rename = "prism.hadamard.sign_values", default)]
    values: Vec<f64>,
    #[serde(rename = "prism.hadamard.gdn_v_grouped")]
    grouped: Option<bool>,
}

/// An error naming the contract key at fault.
fn refuse(key: &'static str, value: String) -> CoreError {
    CoreError::InvalidValue { key, value }
}

impl RotationPlan {
    /// Reads the contract from a GGUF's metadata.
    ///
    /// `None` means the file states no rotation at all: no key under the
    /// contract's prefix. A file with some keys and not others is an error.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] naming the first key that is missing,
    /// malformed or outside what version 1 allows.
    pub fn from_gguf(meta: &GgufMetadata) -> CoreResult<Option<Self>> {
        if !meta.entries().iter().any(|(k, _)| k.starts_with(PREFIX)) {
            return Ok(None);
        }
        let strings = |key: &'static str| -> CoreResult<Vec<String>> {
            match meta.get_array(key) {
                None => Ok(Vec::new()),
                Some(array) => array
                    .strings()
                    .map(|v| v.into_iter().map(String::from).collect())
                    .map_err(|e| refuse(key, e.to_string())),
            }
        };
        let numbers = |key: &'static str| -> CoreResult<Vec<f64>> {
            let Some(array) = meta.get_array(key) else {
                return Ok(Vec::new());
            };
            array
                .iter()
                .map(|v| match v.map_err(|e| refuse(key, e.to_string()))? {
                    MetaValue::I8(x) => Ok(f64::from(x)),
                    MetaValue::U8(x) => Ok(f64::from(x)),
                    MetaValue::I16(x) => Ok(f64::from(x)),
                    MetaValue::U16(x) => Ok(f64::from(x)),
                    MetaValue::I32(x) => Ok(f64::from(x)),
                    MetaValue::U32(x) => Ok(f64::from(x)),
                    MetaValue::F32(x) => Ok(f64::from(x)),
                    MetaValue::F64(x) => Ok(x),
                    other => Err(refuse(key, format!("{other:?} is not a number"))),
                })
                .collect()
        };
        let widths = numbers("prism.hadamard.sign_widths")?
            .into_iter()
            .map(|w| {
                if w.fract() == 0.0 && (0.0..=f64::from(u32::MAX)).contains(&w) {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "checked whole and in range"
                    )]
                    Ok(w as i64)
                } else {
                    Err(refuse("prism.hadamard.sign_widths", format!("{w}")))
                }
            })
            .collect::<CoreResult<Vec<i64>>>()?;
        let raw = Raw {
            version: meta.get_u32("prism.hadamard.version"),
            block: meta.get_u32("prism.hadamard.block_size"),
            transform: meta.get_str("prism.hadamard.transform").map(String::from),
            axis: meta.get_str("prism.hadamard.axis").map(String::from),
            sign_mode: meta.get_str("prism.hadamard.sign_mode").map(String::from),
            forward: strings("prism.hadamard.weight_names")?,
            inverse: strings("prism.hadamard.inverse_weight_names")?,
            widths,
            values: numbers("prism.hadamard.sign_values")?,
            grouped: meta.get_bool("prism.hadamard.gdn_v_grouped"),
        };
        Self::check(raw, |name| Some(name.to_owned())).map(Some)
    }

    /// Reads the contract from an MLX directory's `hadamard.json`, mapping
    /// its tensor names onto this family's GGUF names.
    ///
    /// # Errors
    ///
    /// [`CoreError::Io`] when the file cannot be read, and
    /// [`CoreError::InvalidValue`] for a document that is not the contract or
    /// names a tensor this family does not have.
    pub fn from_mlx_dir(dir: &Path) -> CoreResult<Self> {
        let path = dir.join(MLX_FILE);
        let text = std::fs::read_to_string(&path)?;
        let raw: Raw =
            serde_json::from_str(&text).map_err(|e| refuse("hadamard.json", e.to_string()))?;
        Self::check(raw, |name| {
            let module = name.strip_prefix(LM_PREFIX).unwrap_or(name);
            let module = module.strip_suffix(".weight").unwrap_or(module);
            map_role(module).map(|role| role.gguf)
        })
    }

    /// Checks `raw` against version 1, naming tensors through `rename`.
    fn check(raw: Raw, rename: impl Fn(&str) -> Option<String>) -> CoreResult<Self> {
        let version = raw
            .version
            .ok_or_else(|| refuse("prism.hadamard.version", "missing".into()))?;
        if version != VERSION {
            return Err(refuse(
                "prism.hadamard.version",
                format!("{version}, not {VERSION}"),
            ));
        }
        let expect = |key: &'static str, got: Option<String>, want: &str| -> CoreResult<()> {
            match got {
                Some(v) if v == want => Ok(()),
                other => Err(refuse(key, format!("{other:?}, not {want:?}"))),
            }
        };
        expect("prism.hadamard.transform", raw.transform, TRANSFORM)?;
        expect("prism.hadamard.axis", raw.axis, AXIS)?;
        expect("prism.hadamard.sign_mode", raw.sign_mode, SIGN_MODE)?;
        let block = raw
            .block
            .filter(|b| BLOCKS.contains(b))
            .ok_or_else(|| refuse("prism.hadamard.block_size", format!("{:?}", raw.block)))?;

        let signs = sign_vectors(block, raw.widths, &raw.values)?;

        let mut roles = HashMap::new();
        for (names, role, key) in [
            (raw.forward, Rotated::Forward, "prism.hadamard.weight_names"),
            (
                raw.inverse,
                Rotated::Inverse,
                "prism.hadamard.inverse_weight_names",
            ),
        ] {
            for name in names {
                let gguf = rename(&name)
                    .ok_or_else(|| refuse(key, format!("{name} is not a tensor of this family")))?;
                if roles.insert(gguf, role).is_some() {
                    return Err(refuse(key, format!("{name} is listed twice")));
                }
            }
        }
        if roles.is_empty() {
            return Err(refuse(
                "prism.hadamard.weight_names",
                "no tensor is rotated".into(),
            ));
        }
        Ok(Self {
            block,
            signs,
            roles,
            grouped_value_heads: raw.grouped.unwrap_or(false),
        })
    }

    /// The transform's block size.
    #[must_use]
    pub const fn block(&self) -> u32 {
        self.block
    }

    /// How the tensor this family's loader calls `gguf_name` was stored.
    #[must_use]
    pub fn role(&self, gguf_name: &str) -> Option<Rotated> {
        self.roles.get(gguf_name).copied()
    }

    /// How many tensors take each direction: `(forward, inverse)`.
    #[must_use]
    pub fn counts(&self) -> (usize, usize) {
        let forward = self
            .roles
            .values()
            .filter(|r| **r == Rotated::Forward)
            .count();
        (forward, self.roles.len() - forward)
    }

    /// The widest input any forward rotation reads.
    #[must_use]
    pub fn widest(&self) -> u32 {
        self.signs.keys().copied().max().unwrap_or(0)
    }

    /// The sign vector for inputs of `width`.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] when the contract states none for it.
    pub fn signs(&self, width: u32) -> CoreResult<&[f32]> {
        self.signs.get(&width).map(Vec::as_slice).ok_or_else(|| {
            refuse(
                "prism.hadamard.sign_widths",
                format!("no vector for width {width}"),
            )
        })
    }

    /// The head permutation the recurrent output projection's input is read
    /// through, for a geometry of `key_heads` key heads, `value_heads` value
    /// heads of `head_dim`.
    ///
    /// `None` is not a refusal: it means the value heads do not repeat the key
    /// heads, so tiled and grouped order are the same order and there is
    /// nothing to permute.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] when the value heads repeat the key heads
    /// and the contract does not say the projection was folded in grouped
    /// order: this reader does not know what order an ungrouped fold assumed,
    /// and guessing is how a rotated file produces nonsense.
    pub fn value_heads(
        &self,
        head_dim: u32,
        key_heads: u32,
        value_heads: u32,
    ) -> CoreResult<Option<HeadOrder>> {
        if value_heads == key_heads {
            return Ok(None);
        }
        if !self.grouped_value_heads || key_heads == 0 || !value_heads.is_multiple_of(key_heads) {
            return Err(refuse(
                "prism.hadamard.gdn_v_grouped",
                format!(
                    "{value_heads} value heads over {key_heads} key heads need a grouped fold, \
                     and the file does not state one"
                ),
            ));
        }
        Ok(Some(HeadOrder {
            head_dim,
            key_heads,
            group: value_heads / key_heads,
        }))
    }
}

/// The sign vectors `widths` and `values` state, back to back, checked.
fn sign_vectors(
    block: u32,
    widths: Vec<i64>,
    values: &[f64],
) -> CoreResult<BTreeMap<u32, Vec<f32>>> {
    let mut signs = BTreeMap::new();
    let mut offset = 0_usize;
    for width in widths {
        let w = u32::try_from(width)
            .ok()
            .filter(|w| *w > 0 && w.is_multiple_of(block))
            .ok_or_else(|| {
                refuse(
                    "prism.hadamard.sign_widths",
                    format!("{width} is not whole blocks of {block}"),
                )
            })?;
        let len = usize::try_from(w).unwrap_or(usize::MAX);
        let end = offset
            .checked_add(len)
            .filter(|end| *end <= values.len())
            .ok_or_else(|| {
                refuse(
                    "prism.hadamard.sign_values",
                    "shorter than sign_widths".into(),
                )
            })?;
        let vector = values[offset..end]
            .iter()
            .map(|&v| {
                if v.to_bits() == 1.0_f64.to_bits() {
                    Ok(1.0_f32)
                } else if v.to_bits() == (-1.0_f64).to_bits() {
                    Ok(-1.0_f32)
                } else {
                    Err(refuse(
                        "prism.hadamard.sign_values",
                        format!("{v} is neither +1 nor -1"),
                    ))
                }
            })
            .collect::<CoreResult<Vec<f32>>>()?;
        if signs.insert(w, vector).is_some() {
            return Err(refuse(
                "prism.hadamard.sign_widths",
                format!("{w} listed twice"),
            ));
        }
        offset = end;
    }
    if offset != values.len() {
        return Err(refuse(
            "prism.hadamard.sign_values",
            format!("{} values past the last width", values.len() - offset),
        ));
    }
    Ok(signs)
}

#[cfg(test)]
#[path = "rotation_tests.rs"]
mod tests;
