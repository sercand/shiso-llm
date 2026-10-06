//! Recipe manifests, index catalog, and newtypes.

use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::HubError;
use crate::host::Backend;

/// Current schema version produced and consumed by this build.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// Canonical model name or alias.
///
/// Must match `^[a-z0-9][a-z0-9._-]*$` and be at most 64 characters.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelName(String);

impl ModelName {
    /// Validates and constructs a new `ModelName`.
    ///
    /// # Errors
    /// Returns [`HubError::Validation`] if `name` is invalid.
    pub fn parse(s: &str) -> Result<Self, HubError> {
        if s.is_empty() || s.len() > 64 {
            return Err(HubError::Validation {
                reason: format!("model name '{s}' must be between 1 and 64 characters"),
            });
        }
        let first = s.as_bytes()[0];
        if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
            return Err(HubError::Validation {
                reason: format!("model name '{s}' must start with lowercase letter or digit"),
            });
        }
        for b in s.bytes() {
            if !b.is_ascii_lowercase() && !b.is_ascii_digit() && b != b'.' && b != b'_' && b != b'-'
            {
                return Err(HubError::Validation {
                    reason: format!(
                        "model name '{s}' contains invalid character '{}'",
                        b as char
                    ),
                });
            }
        }
        Ok(Self(s.to_string()))
    }

    /// Returns the underlying string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for ModelName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for ModelName {
    type Err = HubError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl Serialize for ModelName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ModelName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::parse(&s).map_err(serde::de::Error::custom)
    }
}

/// Key identifying a specific variant within a model recipe (e.g. `metal-mlx4`).
///
/// Must match `^[a-z0-9_-]+$`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VariantKey(String);

impl VariantKey {
    /// Validates and constructs a new `VariantKey`.
    ///
    /// # Errors
    /// Returns [`HubError::Validation`] if `s` is invalid.
    pub fn parse(s: &str) -> Result<Self, HubError> {
        if s.is_empty() || s.len() > 64 {
            return Err(HubError::Validation {
                reason: format!("variant key '{s}' must be between 1 and 64 characters"),
            });
        }
        for b in s.bytes() {
            if !b.is_ascii_lowercase() && !b.is_ascii_digit() && b != b'_' && b != b'-' {
                return Err(HubError::Validation {
                    reason: format!(
                        "variant key '{s}' contains invalid character '{}'",
                        b as char
                    ),
                });
            }
        }
        Ok(Self(s.to_string()))
    }

    /// Returns the underlying string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for VariantKey {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for VariantKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for VariantKey {
    type Err = HubError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl Serialize for VariantKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for VariantKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::parse(&s).map_err(serde::de::Error::custom)
    }
}

/// A 40-character hexadecimal Git commit SHA.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CommitSha(String);

impl CommitSha {
    /// Validates and constructs a `CommitSha`.
    ///
    /// # Errors
    /// Returns [`HubError::Validation`] if `s` is not exactly 40 lowercase hex characters.
    pub fn parse(s: &str) -> Result<Self, HubError> {
        if s.len() != 40
            || !s
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            return Err(HubError::Validation {
                reason: format!(
                    "commit SHA '{s}' must be exactly 40 lowercase hexadecimal characters"
                ),
            });
        }
        Ok(Self(s.to_string()))
    }

    /// Returns the underlying string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for CommitSha {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CommitSha {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for CommitSha {
    type Err = HubError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl Serialize for CommitSha {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for CommitSha {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::parse(&s).map_err(serde::de::Error::custom)
    }
}

/// A 64-character lowercase hexadecimal SHA-256 digest.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sha256(String);

impl Sha256 {
    /// Validates and constructs a `Sha256`.
    ///
    /// # Errors
    /// Returns [`HubError::Validation`] if `s` is not exactly 64 lowercase hex characters.
    pub fn parse(s: &str) -> Result<Self, HubError> {
        if s.len() != 64
            || !s
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            return Err(HubError::Validation {
                reason: format!(
                    "SHA-256 digest '{s}' must be exactly 64 lowercase hexadecimal characters"
                ),
            });
        }
        Ok(Self(s.to_string()))
    }

    /// Returns the underlying string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for Sha256 {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Sha256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for Sha256 {
    type Err = HubError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl Serialize for Sha256 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for Sha256 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::parse(&s).map_err(serde::de::Error::custom)
    }
}

/// A model reference formatted as `name[:variant]`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelRef {
    /// The model name or alias.
    pub model: ModelName,
    /// The optional variant key.
    pub variant: Option<VariantKey>,
}

impl ModelRef {
    /// Parses a model reference string (e.g. `qwen3.5-9b` or `qwen3.5-9b:metal-mlx4`).
    ///
    /// # Errors
    /// Returns [`HubError::Validation`] if either part is invalid.
    pub fn parse(s: &str) -> Result<Self, HubError> {
        if let Some((m, v)) = s.split_once(':') {
            let model = ModelName::parse(m)?;
            let variant = Some(VariantKey::parse(v)?);
            Ok(Self { model, variant })
        } else {
            let model = ModelName::parse(s)?;
            Ok(Self {
                model,
                variant: None,
            })
        }
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.variant {
            Some(v) => write!(f, "{}:{}", self.model, v),
            None => write!(f, "{}", self.model),
        }
    }
}

impl FromStr for ModelRef {
    type Err = HubError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl Serialize for ModelRef {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for ModelRef {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::parse(&s).map_err(serde::de::Error::custom)
    }
}

/// The component role within a model variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Language model base weights (`--model`).
    Base,
    /// Speculative draft model checkpoint (`--draft-model`).
    Draft,
    /// Speculative draft model external vocabulary (`--draft-vocab`).
    DraftVocab,
    /// Vision projector and tower weights (`--vision`).
    Vision,
    /// Speech recognition model (`--asr-model`).
    Asr,
    /// Speech recognition audio tower (`--asr-mmproj`).
    AsrMmproj,
    /// Speaker diarization model (`--diarization-model`).
    Diarization,
    /// Image generation model (`--image-model`): a checkpoint file or directory.
    Image,
    /// Shared image assets directory (`--image-assets-dir`).
    ///
    /// For an image checkpoint that ships its transformer alone, such as an
    /// FP8 export: the directory holds the `text_encoder`, `vae`, `tokenizer`
    /// and `transformer/config.json` the generator reads beside the weights.
    ImageAssets,
    /// Per-layer n-gram embedding table (`--ngram`).
    ///
    /// For a checkpoint that keeps the table in a file of its own rather than
    /// inside its shards; the loader reads it from the flag, never by name.
    Ngram,
    /// Architecture config document (`--config`).
    ///
    /// For a checkpoint whose weights file carries no `config.json` beside it,
    /// such as a converted single-file container. The server reads the
    /// tokenizer, chat template and generation config from beside this file,
    /// so the component ships those too.
    Config,
}

impl Role {
    /// Returns the CLI flag name corresponding to this role.
    #[must_use]
    pub const fn flag_name(self) -> &'static str {
        match self {
            Self::Base => "--model",
            Self::Draft => "--draft-model",
            Self::DraftVocab => "--draft-vocab",
            Self::Vision => "--vision",
            Self::Asr => "--asr-model",
            Self::AsrMmproj => "--asr-mmproj",
            Self::Diarization => "--diarization-model",
            Self::Image => "--image-model",
            Self::ImageAssets => "--image-assets-dir",
            Self::Ngram => "--ngram",
            Self::Config => "--config",
        }
    }

    /// Returns the string representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Draft => "draft",
            Self::DraftVocab => "draft_vocab",
            Self::Vision => "vision",
            Self::Asr => "asr",
            Self::AsrMmproj => "asr_mmproj",
            Self::Diarization => "diarization",
            Self::Image => "image",
            Self::ImageAssets => "image_assets",
            Self::Ngram => "ngram",
            Self::Config => "config",
        }
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for Role {
    type Err = HubError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "base" => Ok(Self::Base),
            "draft" => Ok(Self::Draft),
            "draft_vocab" => Ok(Self::DraftVocab),
            "vision" => Ok(Self::Vision),
            "asr" => Ok(Self::Asr),
            "asr_mmproj" => Ok(Self::AsrMmproj),
            "diarization" => Ok(Self::Diarization),
            "image" => Ok(Self::Image),
            "image_assets" => Ok(Self::ImageAssets),
            "ngram" => Ok(Self::Ngram),
            "config" => Ok(Self::Config),
            _ => Err(HubError::Validation {
                reason: format!("unknown role '{s}'"),
            }),
        }
    }
}

/// An API task served by a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    /// Chat / text completion API.
    Chat,
    /// Speech-to-text transcription.
    Transcription,
    /// Speaker diarization.
    Diarization,
    /// Image generation.
    ImageGeneration,
    /// Text or multimodal embeddings.
    Embedding,
    /// Schema decisions answered by a decision head.
    Decision,
}

impl Task {
    /// Returns the string representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Transcription => "transcription",
            Self::Diarization => "diarization",
            Self::ImageGeneration => "image_generation",
            Self::Embedding => "embedding",
            Self::Decision => "decision",
        }
    }
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for Task {
    type Err = HubError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "chat" => Ok(Self::Chat),
            "transcription" => Ok(Self::Transcription),
            "diarization" => Ok(Self::Diarization),
            "image_generation" => Ok(Self::ImageGeneration),
            "embedding" => Ok(Self::Embedding),
            "decision" => Ok(Self::Decision),
            _ => Err(HubError::Validation {
                reason: format!("unknown task '{s}'"),
            }),
        }
    }
}

/// Input/output modality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    /// Text.
    Text,
    /// Static image.
    Image,
    /// Video clip.
    Video,
    /// Audio stream or clip.
    Audio,
}

impl Modality {
    /// Returns the string representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Video => "video",
            Self::Audio => "audio",
        }
    }
}

impl fmt::Display for Modality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for Modality {
    type Err = HubError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "text" => Ok(Self::Text),
            "image" => Ok(Self::Image),
            "video" => Ok(Self::Video),
            "audio" => Ok(Self::Audio),
            _ => Err(HubError::Validation {
                reason: format!("unknown modality '{s}'"),
            }),
        }
    }
}

/// Input and output modality declarations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModalCapabilities {
    /// Input modalities accepted by the model.
    #[serde(default)]
    pub inputs: Vec<Modality>,
    /// Output modalities produced by the model.
    #[serde(default)]
    pub outputs: Vec<Modality>,
}

/// Feature capabilities declared by a recipe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecipeFeatures {
    /// Whether the model supports tool calls.
    #[serde(default)]
    pub tools: bool,
    /// Whether the model supports reasoning traces.
    #[serde(default)]
    pub reasoning: bool,
    /// Supported reasoning effort labels.
    #[serde(default)]
    pub reasoning_efforts: Vec<String>,
    /// Default reasoning effort label.
    #[serde(default)]
    pub default_reasoning_effort: Option<String>,
    /// Whether structured output (JSON schema) is supported.
    #[serde(default)]
    pub structured_output: bool,
}

/// A single file in a component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileEntry {
    /// Relative file path within the component (may contain subdirs, no `..`, no leading `/`).
    pub name: String,
    /// File size in bytes.
    pub size: u64,
    /// Expected SHA-256 hexadecimal digest.
    pub sha256: Sha256,
    /// Another role whose flag this file is passed to.
    ///
    /// Lets one repository ship several roles' files - a checkpoint, its draft
    /// head and its vision tower side by side - without the recipe repeating
    /// the repository once per role. The file is downloaded into its own
    /// component's directory; only the flag it feeds changes. A role is
    /// supplied by a component or by one bound file, never both.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binds: Option<Role>,
}

/// Largest file the hub hosts for a recipe: one mebibyte. Hub files are side
/// files (a draft vocabulary, a template), not weights.
pub const HUB_FILE_MAX_BYTES: u64 = 1_048_576;

/// Files the registry build writes into a model's directory, which a hosted
/// file sharing that directory may not be named.
pub const RESERVED_MODEL_FILES: [&str; 2] = ["index.json", "benchmarks.json"];

/// Whether `name` can be a hub-hosted file: `[A-Za-z0-9][A-Za-z0-9._-]*`,
/// not one of the build's own files.
///
/// Narrower than a path segment on purpose. The name becomes a URL path
/// segment as-is (so no `?`, `#`, `%` or spaces), a path `git status` prints
/// unquoted and whitespace-free (so the dirty check can see it), and a file
/// the static host serves (so no leading dot).
#[must_use]
pub fn hub_file_name_ok(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphanumeric())
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        && !RESERVED_MODEL_FILES.contains(&name)
}

/// Where a component's files are downloaded from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provider {
    /// A Hugging Face repository at a pinned commit.
    #[serde(rename = "huggingface")]
    HuggingFace,
    /// The shiso hub itself: files committed beside the model's manifest in
    /// the registry repository, at most [`HUB_FILE_MAX_BYTES`] each.
    #[serde(rename = "shiso")]
    Shiso,
}

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::HuggingFace => "huggingface",
            Self::Shiso => "shiso",
        })
    }
}

/// A component of a variant (e.g. `base`, `vision`, `draft`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Component {
    /// Where the files come from.
    pub provider: Provider,
    /// Provider repository (e.g. `"shiso-community/qwen3.5-9b-mlx-4bit"`);
    /// required for Hugging Face, absent for the hub.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    /// Pinned commit SHA; required for Hugging Face, absent for the hub.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<CommitSha>,
    /// Which file the component flag points at (absent = component directory).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry: Option<String>,
    /// List of all files making up this component.
    pub files: Vec<FileEntry>,
}

impl Component {
    /// The Hugging Face repository and commit, when this component has both.
    #[must_use]
    pub fn hf_source(&self) -> Option<(&str, &CommitSha)> {
        match (self.provider, self.repo.as_deref(), self.revision.as_ref()) {
            (Provider::HuggingFace, Some(repo), Some(rev)) => Some((repo, rev)),
            _ => None,
        }
    }
}

/// Hardware requirements for running a variant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Requirements {
    /// Minimum host RAM in bytes.
    pub min_ram_bytes: u64,
    /// Recommended host RAM in bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recommended_ram_bytes: Option<u64>,
    /// Minimum device memory in bytes: CUDA VRAM, or the Metal device's
    /// recommended working set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_vram_bytes: Option<u64>,
    /// Disk space required by all files in this variant in bytes.
    pub disk_bytes: u64,
}

/// Hardware constraints for variant execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct HostConstraints {
    /// Minimum CUDA compute capability (e.g. 89 for Ada, 100 for Blackwell).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_cuda_sm: Option<u32>,
    /// Minimum Apple GPU family (7 = M1, 8 = M2, 9 = M3/M4, 10 = M5).
    ///
    /// The family, not a chip list: a Pro, Max or Ultra shares its base
    /// chip's family and differs only in memory, which `min_vram_bytes` and
    /// the RAM requirements already express.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_metal_family: Option<u32>,
}

/// Typed server defaults allowlist (§3.4).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerDefaults {
    /// Context window size (`--ctx`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctx: Option<usize>,
    /// Maximum number of concurrent sequences (`--max-num-seqs`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_num_seqs: Option<usize>,
    /// Speculation depth (`--spec-k`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec_k: Option<usize>,
    /// Turn replay tokens cap (`--turn-replay-tokens`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_replay_tokens: Option<usize>,
    /// Vision maximum pixels (`--vision-max-pixels`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision_max_pixels: Option<usize>,
    /// Video frames per second (`--video-fps`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_fps: Option<f32>,
    /// Video maximum frames (`--video-max-frames`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_max_frames: Option<usize>,
    /// ASR lookahead frames (`--asr-lookahead`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asr_lookahead: Option<usize>,
    /// ASR batch size (`--asr-batch`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asr_batch: Option<usize>,
    /// Expert residency (`--expert-residency`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_residency: Option<ExpertResidency>,
    /// Working-set gibibytes the automatic residency leaves free
    /// (`--expert-reserve-gb`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_reserve_gb: Option<u64>,
    /// Expert slots when the banks are streamed (`--expert-slots`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_slots: Option<u32>,
    /// Prompt-lookup speculation depth (`--spec-lookup`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec_lookup: Option<u32>,
    /// Capture verify passes as device graphs (`--capture-verify`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_verify: Option<bool>,
    /// Capture multi-session passes as device graphs (`--capture-batch`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_batch: Option<bool>,
    /// Fraction of device memory the server may claim
    /// (`--gpu-memory-utilization`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_memory_utilization: Option<f64>,
    /// Prompt rows per pass when one session prefills alone
    /// (`--solo-prefill-chunk`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solo_prefill_chunk: Option<usize>,
}

/// Where a sparse checkpoint's routed expert banks live (`--expert-residency`).
///
/// Typed rather than a free string so that a recipe naming a choice this build
/// does not have fails to parse instead of being skipped with a warning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExpertResidency {
    /// Resident when every weight fits the working set, streamed otherwise.
    Auto,
    /// Imported beside the other weights.
    Resident,
    /// Streamed from disk into a pool of expert slots.
    #[serde(alias = "streamed")]
    Stream,
}

/// A specific backend and quantization variant of a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Variant {
    /// Unique key within the recipe (e.g. `metal-mlx4`, `cuda-fp8`).
    pub key: VariantKey,
    /// Backends this variant's files run on, in author preference order.
    ///
    /// A list because one set of files can serve on several backends - a
    /// K-quant container runs on Metal and CUDA alike. The legacy single-string
    /// `backend` spelling is read as a one-element list.
    #[serde(alias = "backend", deserialize_with = "one_or_many_backends")]
    pub backends: Vec<Backend>,
    /// Quantization label (e.g. `"mlx-4bit"`, `"fp8"`).
    pub quant: String,
    /// Optional host execution constraints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires: Option<HostConstraints>,
    /// Resource requirements for running this variant.
    pub requirements: Requirements,
    /// Components that make up this variant.
    pub components: BTreeMap<Role, Component>,
    /// Optional server defaults allowlist.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_defaults: Option<ServerDefaults>,
}

impl Variant {
    /// Every role this variant supplies a flag for, with the component
    /// directory the path lives under and the file inside it (`None` is the
    /// directory itself).
    ///
    /// A component supplies its own role through `entry`; a file with `binds`
    /// supplies the role it names.
    ///
    /// # Errors
    /// Returns [`HubError::Validation`] when two places supply the same role,
    /// or a file binds the role of the component that holds it.
    pub fn role_paths(&self) -> Result<BTreeMap<Role, RolePath<'_>>, HubError> {
        let mut paths = BTreeMap::new();
        for (role, comp) in &self.components {
            paths.insert(
                *role,
                RolePath {
                    dir: *role,
                    file: comp.entry.as_deref(),
                },
            );
        }
        for (role, comp) in &self.components {
            for file in &comp.files {
                let Some(bound) = file.binds else { continue };
                if bound == *role {
                    return Err(HubError::Validation {
                        reason: format!(
                            "variant '{}' role '{role}' file '{}' binds its own role; use 'entry' instead",
                            self.key, file.name
                        ),
                    });
                }
                let path = RolePath {
                    dir: *role,
                    file: Some(file.name.as_str()),
                };
                if paths.insert(bound, path).is_some() {
                    return Err(HubError::Validation {
                        reason: format!(
                            "variant '{}' supplies role '{bound}' twice (file '{}' in '{role}' binds it)",
                            self.key, file.name
                        ),
                    });
                }
            }
        }
        Ok(paths)
    }
}

/// Where one role's flag points inside an installed variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RolePath<'a> {
    /// The component whose directory holds the path.
    pub dir: Role,
    /// The file inside that directory, or `None` for the directory itself.
    pub file: Option<&'a str>,
}

/// Reads `"metal"` or `["metal", "cuda"]` as a list of backends.
fn one_or_many_backends<'de, D>(deserializer: D) -> Result<Vec<Backend>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(Backend),
        Many(Vec<Backend>),
    }
    Ok(match OneOrMany::deserialize(deserializer)? {
        OneOrMany::One(b) => vec![b],
        OneOrMany::Many(list) => list,
    })
}

/// Descriptive metadata about a model, shown on the hub site.
///
/// Nothing here takes part in variant selection or serving: a client that
/// ignores the whole block pulls and serves exactly the same files.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Who trained or published the weights (e.g. `"Google"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub publisher: Option<String>,
    /// License identifier (an SPDX id where one exists, e.g. `"apache-2.0"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Project or model homepage (`https://` only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub homepage: Option<String>,
    /// Repository whose README is the model card.
    ///
    /// Absent, the card is read from the first variant's `base` component. A
    /// quantized re-export often ships a stub README, so pointing this at the
    /// original repository gives the page the publisher's own card.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub card: Option<CardSource>,
    /// Total parameter count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<u64>,
    /// Parameters active per token, for sparse (mixture-of-experts) models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_parameters: Option<u64>,
    /// Longest context the model was trained for, in tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u64>,
}

/// A pinned repository revision whose `README.md` is a model's card.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CardSource {
    /// Remote provider (only `"huggingface"` today).
    pub provider: String,
    /// Provider repository (`owner/name`).
    pub repo: String,
    /// Pinned commit, so the card cannot change under a published manifest.
    pub revision: CommitSha,
}

/// What a quality score was measured on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityScope {
    /// The unquantized model the variants were made from; it says nothing
    /// about what a quantization lost.
    BaseModel,
    /// One variant's own files, named by [`QualityScore::variant`].
    Variant,
}

/// One published quality score, with where it came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityScore {
    /// Who published the score.
    pub source: String,
    /// The metric's name as the source spells it.
    pub metric: String,
    /// The score.
    pub value: f64,
    /// Top of the metric's scale, when it has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    /// Where the score can be checked (`https://` only).
    pub url: String,
    /// Date the score was read, `YYYY-MM-DD`; sources revise their numbers.
    pub as_of: String,
    /// What was measured.
    pub scope: QualityScope,
    /// The variant measured; required for [`QualityScope::Variant`], refused
    /// otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<VariantKey>,
    /// The scored model as the source names it, when that differs from ours.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
}

/// A model recipe manifest describing how to fetch and serve a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recipe {
    /// Manifest schema version (starts at 1).
    pub schema_version: u32,
    /// Canonical model name.
    pub name: ModelName,
    /// Alternate names / aliases.
    #[serde(default)]
    pub aliases: Vec<ModelName>,
    /// Family name as recognized by `shiso-models`.
    pub family: String,
    /// Oldest `shiso` version that can serve all variants.
    pub min_shiso_version: semver::Version,
    /// Short description.
    pub description: String,
    /// Free-form search tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// API surfaces served by this model.
    #[serde(default)]
    pub tasks: Vec<Task>,
    /// Modalities supported.
    pub modalities: ModalCapabilities,
    /// Feature capabilities.
    pub features: RecipeFeatures,
    /// Available variants in author preference order.
    pub variants: Vec<Variant>,
    /// Descriptive metadata for the hub site.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub info: Option<ModelInfo>,
    /// Quality scores published for the model or one of its variants.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub quality: Vec<QualityScore>,
    /// Git-derived publication timestamp (compiled manifests only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub published_at: Option<String>,
    /// Git-derived last update timestamp (compiled manifests only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
}

/// An entry for one variant inside the compact registry index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexVariantEntry {
    /// Variant key.
    pub key: VariantKey,
    /// Backends the variant runs on.
    #[serde(alias = "backend", deserialize_with = "one_or_many_backends")]
    pub backends: Vec<Backend>,
    /// Quantization label.
    pub quant: String,
    /// Minimum host RAM in bytes.
    pub min_ram_bytes: u64,
    /// Disk space required in bytes.
    pub disk_bytes: u64,
    /// Minimum CUDA compute capability, if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_cuda_sm: Option<u32>,
    /// Minimum Apple GPU family, if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_metal_family: Option<u32>,
}

/// An entry for one model inside the compact registry index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexModelEntry {
    /// Canonical model name.
    pub name: ModelName,
    /// Alternate names.
    #[serde(default)]
    pub aliases: Vec<ModelName>,
    /// Model family name.
    pub family: String,
    /// Oldest compatible `shiso` version.
    pub min_shiso_version: semver::Version,
    /// One-line description.
    pub description: String,
    /// Search tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Tasks served.
    #[serde(default)]
    pub tasks: Vec<Task>,
    /// Input modalities.
    #[serde(default)]
    pub inputs: Vec<Modality>,
    /// Output modalities.
    #[serde(default)]
    pub outputs: Vec<Modality>,
    /// Last update timestamp.
    pub updated_at: String,
    /// Path of the compiled manifest, relative to the registry base URL
    /// (`models/<name>/index.json`).
    ///
    /// Carried rather than derived so the layout can change without every
    /// client hard-coding it; the client refuses anything that is not a plain
    /// relative path.
    pub manifest: String,
    /// SHA-256 digest of the compiled manifest named by `manifest`.
    pub manifest_sha256: Sha256,
    /// Variants available for this model.
    pub variants: Vec<IndexVariantEntry>,
}

/// The compact registry index (`index.json`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Index {
    /// Index schema version.
    pub schema_version: u32,
    /// ISO-8601 generation timestamp.
    pub generated_at: String,
    /// Models in the catalog.
    pub models: Vec<IndexModelEntry>,
}

/// Record of an installed file inside `installed.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledFileEntry {
    /// Component role.
    pub role: Role,
    /// Relative path inside the role directory.
    pub name: String,
    /// File size in bytes.
    pub size: u64,
    /// SHA-256 digest of the blob.
    pub sha256: Sha256,
}

/// Record stored in `models/<name>/<variant>/installed.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledMetadata {
    /// SHA-256 digest of the manifest this variant was installed from.
    pub manifest_sha256: Sha256,
    /// ISO-8601 installation timestamp.
    pub installed_at: String,
    /// Verified files present in this variant installation.
    pub files: Vec<InstalledFileEntry>,
}
