//! Hugging Face config parsing.
//!
//! Parsing happens in two stages. The first pass reads only the architecture
//! name and keeps the rest of the document as an unparsed value, so an unknown
//! architecture produces a typed "unknown architecture" error instead of a
//! confusing complaint about a field that happens to be absent. The second stage
//! deserializes the architecture-specific subtree once the architecture is known
//! to be one this build understands.
//!
//! Every dimension is then checked against the value this build was written
//! for. A config that claims a different head width would otherwise produce
//! wrong logits with no diagnostic at all.
//!
//! # No architecture is named here
//!
//! This module reads a config *document*: it finds the architecture name, the
//! declared heads, and the vision tower, all of which every checkpoint spells
//! the same way. Turning a named architecture's subtree into geometry is that
//! architecture's own business, and it reaches the subtree through
//! [`HfConfig::section`]. A build that supports a third family therefore adds a
//! crate rather than a match arm here.

use std::path::Path;
use std::str::FromStr;

use serde::Deserialize;

use crate::error::{CoreError, CoreResult};
use crate::ids::TokenId;

/// Activation used inside the vision tower's feed-forward blocks.
///
/// Parsed once, at load: an unknown activation is a load error rather than a
/// string carried into the vision kernels to fail on later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HiddenAct {
    /// GELU approximation via `tanh`, the default in these checkpoints.
    GeluPytorchTanh,
    /// Exact GELU via the error function.
    GeluErf,
    /// Swish/SiLU.
    Silu,
}

impl FromStr for HiddenAct {
    type Err = CoreError;

    /// Maps the config spelling onto the activation the kernels implement.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "gelu_pytorch_tanh" => Ok(Self::GeluPytorchTanh),
            "gelu" | "gelu_erf" => Ok(Self::GeluErf),
            "silu" => Ok(Self::Silu),
            other => Err(CoreError::InvalidValue {
                key: "hidden_act",
                value: other.to_string(),
            }),
        }
    }
}
/// Geometry and token ids of the vision tower, for the vision tower and the
/// media-token plumbing around it.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct VisionConfig {
    /// Number of vision transformer blocks.
    pub depth: u32,
    /// Width inside the tower.
    pub hidden: u32,
    /// Attention heads in the tower.
    pub num_heads: u32,
    /// Feed-forward inner width.
    pub intermediate_size: u32,
    /// Width of the vectors handed to the language model, which is the text
    /// model's hidden width.
    pub out_hidden: u32,
    /// Spatial patch edge in pixels.
    pub patch_size: u32,
    /// Frames combined into one patch.
    pub temporal_patch_size: u32,
    /// Side length of the patch grid merged into one output token.
    pub spatial_merge_size: u32,
    /// Rows in the learned positional table.
    pub num_position_embeddings: u32,
    /// Extra output taps on the tower. Empty means a single tower.
    pub deepstack_visual_indexes: Vec<u32>,
    /// Feed-forward activation.
    pub hidden_act: HiddenAct,
    /// Placeholder token replaced by image features.
    pub image_token_id: TokenId,
    /// Placeholder token replaced by video features.
    pub video_token_id: TokenId,
    /// Token opening a media span.
    pub vision_start_token_id: TokenId,
    /// Token closing a media span.
    pub vision_end_token_id: TokenId,
    /// Multimodal rope axis sections; they sum to half the rotated width.
    pub mrope_section: [u32; 3],
    /// Whether the sections are applied interleaved rather than in blocks.
    pub mrope_interleaved: bool,
}

/// Rope block of a text config.
#[derive(Debug, Deserialize)]
struct RopeParameters {
    #[serde(default = "default_mrope_section")]
    mrope_section: Vec<u32>,
    #[serde(default)]
    mrope_interleaved: bool,
}

/// Multimodal rope sections that transformers assumes when a config leaves them
/// out, so an omitted block means "the default layout" rather than a load error.
fn default_mrope_section() -> Vec<u32> {
    vec![16, 24, 24]
}

/// Vision tower geometry as written in the config, before validation.
#[derive(Debug, Deserialize)]
struct VisionRaw {
    depth: u32,
    hidden_size: u32,
    num_heads: u32,
    intermediate_size: u32,
    out_hidden_size: u32,
    patch_size: u32,
    temporal_patch_size: u32,
    spatial_merge_size: u32,
    num_position_embeddings: u32,
    #[serde(default)]
    deepstack_visual_indexes: Vec<u32>,
    #[serde(default = "default_hidden_act")]
    hidden_act: String,
}

fn default_hidden_act() -> String {
    "gelu_pytorch_tanh".to_string()
}
/// The part of the document that is read before the architecture is known.
///
/// Deliberately tiny: parsing the whole document into a typed struct would turn
/// "this architecture is not supported" into a confusing complaint about a
/// missing field that only exists on the supported ones.
#[derive(Debug, Deserialize)]
struct Head {
    model_type: String,
    #[serde(default)]
    architectures: Vec<String>,
}

/// A parsed config document.
#[derive(Debug, Clone)]
pub struct HfConfig {
    model_type: String,
    architectures: Vec<String>,
    document: serde_json::Value,
}

impl HfConfig {
    /// Reads and parses a config file.
    ///
    /// # Errors
    ///
    /// [`CoreError::Io`] when the file cannot be read and [`CoreError::Json`]
    /// when it does not parse.
    pub fn from_file(path: &Path) -> CoreResult<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_json_str(&text)
    }

    /// Parses a config that is already in memory.
    ///
    /// # Errors
    ///
    /// [`CoreError::Json`] when the text is not valid JSON or has no
    /// `model_type`.
    pub fn from_json_str(text: &str) -> CoreResult<Self> {
        let document: serde_json::Value = serde_json::from_str(text)?;
        let head: Head = serde_json::from_value(document.clone())?;
        Ok(Self {
            model_type: head.model_type,
            architectures: head.architectures,
            document,
        })
    }

    /// The `model_type` string that selects the architecture.
    #[must_use]
    pub fn model_type(&self) -> &str {
        &self.model_type
    }

    /// The declared `architectures`, for logging and for rejecting a checkpoint
    /// whose head does not match its `model_type`.
    #[must_use]
    pub fn architectures(&self) -> &[String] {
        &self.architectures
    }

    /// One top-level subtree of the document, unparsed.
    ///
    /// This is how a crate that knows an architecture reads that
    /// architecture's own geometry without this module having to know the
    /// architecture exists. The value is handed over as it was written, so the
    /// crate that understands it decides which keys are required and what each
    /// one has to equal.
    ///
    /// # Errors
    ///
    /// [`CoreError::ConfigMissing`] naming `key` when the document has no such
    /// subtree.
    pub fn section(&self, key: &str) -> CoreResult<&serde_json::Value> {
        self.document
            .get(key)
            .ok_or_else(|| CoreError::ConfigMissing {
                key: key.to_string(),
            })
    }

    /// Checks one parsed dimension against the value the build was written for.
    ///
    /// Shape constants are asserted rather than trusted: a config that claimed a
    /// different head width would silently produce wrong logits, so the mismatch
    /// is reported with the field name and both values. Public because the crate
    /// that owns an architecture does the asserting, and every such crate should
    /// report a deviation the same way.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] when the two disagree.
    pub fn expect(field: &'static str, got: u64, want: u64) -> CoreResult<()> {
        if got == want {
            Ok(())
        } else {
            Err(CoreError::ShapeMismatch {
                field,
                expected: want,
                got,
            })
        }
    }
}

/// Reads one required media token id from the top of the document.
///
/// The placeholder ids live at the top of the document rather than inside
/// either sub-config. A tower whose placeholder is unknown cannot splice
/// features into the token stream, so a missing one names the key and fails.
impl HfConfig {
    /// Vision tower configuration, if this checkpoint has one.
    ///
    /// `Ok(None)` means the document describes a text-only model. A
    /// `vision_config` object that is present but unusable is an error rather
    /// than `None`: degrading to text-only would accept an image and answer
    /// about it with unrelated tokens.
    ///
    /// # Errors
    ///
    /// [`CoreError::Json`] when the `vision_config` object does not have the
    /// expected shape, [`CoreError::InvalidValue`] for an activation spelling
    /// this build does not implement, [`CoreError::Unsupported`] for a tower
    /// with deepstack taps, [`CoreError::ConfigMissing`] for an absent or
    /// unusable media token, and [`CoreError::ShapeMismatch`] for a zero
    /// dimension.
    pub fn vision_config(&self) -> CoreResult<Option<VisionConfig>> {
        let Some(raw) = self.document.get("vision_config") else {
            return Ok(None);
        };
        let tower: VisionRaw = serde_json::from_value(raw.clone())?;
        let hidden_act = tower.hidden_act.parse::<HiddenAct>()?;
        expect_nonzero("vision.depth", tower.depth)?;
        expect_nonzero("vision.hidden_size", tower.hidden_size)?;
        expect_nonzero("vision.num_heads", tower.num_heads)?;
        expect_nonzero("vision.intermediate_size", tower.intermediate_size)?;
        expect_nonzero("vision.out_hidden_size", tower.out_hidden_size)?;
        expect_nonzero("vision.patch_size", tower.patch_size)?;
        expect_nonzero("vision.temporal_patch_size", tower.temporal_patch_size)?;
        expect_nonzero("vision.spatial_merge_size", tower.spatial_merge_size)?;
        expect_nonzero(
            "vision.num_position_embeddings",
            tower.num_position_embeddings,
        )?;
        if !tower.deepstack_visual_indexes.is_empty() {
            return Err(CoreError::Unsupported {
                what: "deepstack vision towers",
                backend: "core",
            });
        }
        let rope = self.rope_parameters()?;
        Ok(Some(VisionConfig {
            depth: tower.depth,
            hidden: tower.hidden_size,
            num_heads: tower.num_heads,
            intermediate_size: tower.intermediate_size,
            out_hidden: tower.out_hidden_size,
            patch_size: tower.patch_size,
            temporal_patch_size: tower.temporal_patch_size,
            spatial_merge_size: tower.spatial_merge_size,
            num_position_embeddings: tower.num_position_embeddings,
            deepstack_visual_indexes: tower.deepstack_visual_indexes,
            hidden_act,
            image_token_id: self.media_token("image_token_id")?,
            video_token_id: self.media_token("video_token_id")?,
            vision_start_token_id: self.media_token("vision_start_token_id")?,
            vision_end_token_id: self.media_token("vision_end_token_id")?,
            mrope_section: rope.section,
            mrope_interleaved: rope.interleaved,
        }))
    }
}

impl HfConfig {
    /// Reads one required media token id from the top of the document.
    ///
    /// These live outside both sub-configs, and a tower whose placeholder is
    /// unknown cannot splice features into the token stream, so a missing one
    /// is a load error naming the key rather than a default.
    fn media_token(&self, key: &str) -> CoreResult<TokenId> {
        let raw = self
            .document
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| CoreError::ConfigMissing {
                key: key.to_string(),
            })?;
        TokenId::try_from(raw).map_err(|_| CoreError::ConfigMissing {
            key: key.to_string(),
        })
    }

    /// The rope block of `text_config`, which is where the multimodal rope
    /// sections live. An absent block means the transformer defaults rather than
    /// a load error; a block that is present with a malformed section is
    /// [`CoreError::ConfigMissing`], because the tower cannot place its features
    /// on any axis without three sections.
    fn rope_parameters(&self) -> CoreResult<Mrope> {
        let rope = match self
            .document
            .get("text_config")
            .and_then(|text| text.get("rope_parameters"))
        {
            Some(raw) => serde_json::from_value::<RopeParameters>(raw.clone())?,
            None => RopeParameters {
                mrope_section: default_mrope_section(),
                mrope_interleaved: false,
            },
        };
        let [t, h, w] =
            <[u32; 3]>::try_from(rope.mrope_section).map_err(|_| CoreError::ConfigMissing {
                key: "text_config.rope_parameters.mrope_section".to_string(),
            })?;
        Ok(Mrope {
            section: [t, h, w],
            interleaved: rope.mrope_interleaved,
        })
    }
}

/// Multimodal rope settings, read from the text config but consumed by the
/// vision tower, which is what assigns positions to patches.
struct Mrope {
    section: [u32; 3],
    interleaved: bool,
}

/// Rejects a zero dimension.
///
/// Every use of these values is a divisor, a stride or a buffer extent, so a
/// zero would be a division panic or a zero-sized allocation deep inside a
/// kernel launch. `expected: 1` reads as "at least one".
fn expect_nonzero(field: &'static str, value: u32) -> CoreResult<()> {
    if value == 0 {
        Err(CoreError::ShapeMismatch {
            field,
            expected: 1,
            got: 0,
        })
    } else {
        Ok(())
    }
}
