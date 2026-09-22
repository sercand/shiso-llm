//! `manifest.json`: every artifact and its digest, and what the package was
//! packed to.
//!
//! The digests are verified before a buffer is allocated. A partial download is
//! the failure that catches, and it is worth catching early: the alternative is
//! a model that loads, runs and answers from bytes that are half another file.
//!
//! The shipped package declares 56 artifacts - 40 target layers, a head, an
//! embedding table, seven draft files, the tower, five tokenizer files and
//! `layout.json` itself - and one [`Manifest::artifact_set_sha256`] over the
//! set. That set digest is read and kept but not checked: it is a digest of the
//! list, and the list is what is already in hand; the digests that can be wrong
//! are the per-file ones, and those are checked.
//!
//! Each file is hashed as it is read rather than after: the tower alone is
//! 901 MB and the whole package is ~20 GB, and there is no reason for the peak
//! to be a whole artifact when it can be one buffer. The declared size is
//! checked first, from the directory entry, so a truncated download is named
//! as a length before minutes are spent hashing it.
//!
//! # Why the whole top level is typed and not just the artifact list
//!
//! Because the rest of it is the only usable statement of what the converter
//! did. [`Format`] carries the two constants the packed bytes were written
//! under - `storage_n` 256 and `quant_group_size` 64 - which a codec otherwise
//! has to assume; [`TargetGeometry`] carries the layer roll, the expert count
//! and both of the recurrent widths, padded and live. A key that is dropped
//! silently is the same class of bug as one that is refused, and quieter:
//! nothing fails, and the reader simply proceeds on a value it made up.
//!
//! # Two documents describe this model, and they are not the same document
//!
//! - **This one** describes the package *as converted*: what was packed, at
//!   which widths, with which magics, by which converter revision.
//! - **`tokenizer/config.json`** - despite its path - is the full upstream
//!   Hugging Face config of `mlx-community/Qwen3.6-35B-A3B-4bit`, carried along
//!   unchanged: `text_config`, `vision_config`, `quantization`. It describes
//!   the *upstream checkpoint*, before this converter touched it.
//!
//! They overlap (both state a hidden size, both state a layer roll) and they
//! are not reconciled here. Nothing in this module reads the HF config, and
//! nothing here should grow a type that spans both: which of the two a loader
//! should believe, and where, is a separate ruling.
//!
//! # Refusing a schema version
//!
//! [`SCHEMA_VERSION`] is the version this reader was written for, and any other
//! is refused by name and number at parse. A reader that silently accepts a
//! version it does not know is how a format change becomes a wrong answer
//! instead of an error - and every field below is a version-4 field, so
//! accepting a version 5 document would be reading version 4's meanings off
//! version 5's keys.

use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

/// The `schema_version` this reader was written for.
pub const SCHEMA_VERSION: u32 = 4;

/// One file the package declares.
#[derive(Debug, Clone, Deserialize)]
pub struct Artifact {
    /// Path relative to the package root.
    pub path: String,
    /// Lowercase hex SHA-256 of the file's bytes.
    pub sha256: String,
    /// Size in bytes.
    pub size: u64,
}

/// How the bytes in every `.bin` were written.
///
/// These are the constants the packer used, and they are the ones a codec on
/// the other side has to agree with rather than assume.
#[derive(Debug, Clone, Deserialize)]
pub struct Format {
    /// The packing this build reads, `splash-packed-q4-moe`.
    pub name: String,
    /// Codes per quantization group, 64 - one scale and one bias each.
    pub quant_group_size: u32,
    /// The width every quantized row is padded to, 256. The live width is the
    /// section's own `actual_output`; this is what it was padded *to*.
    pub storage_n: u32,
    /// The boundary each section starts on inside its file.
    pub section_alignment_bytes: u64,
    /// Bits per code in a `q4_affine` section.
    pub q4_bits: u32,
    /// Bits per code in a `q8_affine` section.
    pub q8_bits: u32,
    /// The magic at the head of a target layer file.
    pub target_layer_magic: String,
    /// The magic at the head of a draft layer file.
    pub draft_layer_magic: String,
    /// The magic at the head of the vision tower file.
    pub vision_magic: String,
}

/// What a layer of the target stack is, as `manifest.json` spells it.
///
/// A second spelling of [`shiso_engine::shape::LayerKind`] rather than that
/// type itself, because the engine's is not a `serde` type and should not
/// become one to suit one package format. The conversion is the [`From`] impl
/// below, and it is the only place the two spellings meet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum LayerKind {
    /// A gated delta-net layer: the recurrent one.
    #[serde(rename = "gdn")]
    Gdn,
    /// A full-attention layer, every fourth slot.
    #[serde(rename = "attention")]
    Attention,
}

impl From<LayerKind> for shiso_engine::shape::LayerKind {
    fn from(kind: LayerKind) -> Self {
        match kind {
            LayerKind::Gdn => Self::Gdn,
            LayerKind::Attention => Self::Attn,
        }
    }
}

/// The target model's geometry, as the converter packed it.
///
/// This is the converted package's account of the model. The upstream
/// checkpoint's own account is the Hugging Face config carried in
/// `tokenizer/config.json`; see the module docs on why the two are not
/// reconciled here.
#[derive(Debug, Clone, Deserialize)]
pub struct TargetGeometry {
    /// The upstream architecture string, `qwen3_5_moe`.
    pub architecture: String,
    /// The residual stream width.
    pub hidden_size: u32,
    /// How many layers the stack has.
    pub layers: u32,
    /// Routed experts per layer.
    pub experts: u32,
    /// How many of those the router picks per token.
    pub experts_per_token: u32,
    /// One routed expert's inner width.
    pub moe_intermediate_size: u32,
    /// The always-on expert's inner width.
    pub shared_expert_intermediate_size: u32,
    /// Rows in the embedding table and the logits projection.
    pub vocabulary_size: u32,
    /// What each layer is, in order. One entry per layer.
    pub layer_types: Vec<LayerKind>,
    /// The live width of the recurrent input projection, 12352.
    pub gdn_actual_width: u32,
    /// The width that projection was padded to, 12544 - 192 rows of padding
    /// that are not weights.
    pub gdn_packed_width: u32,
    /// The width the attention input projection was padded to.
    pub attention_packed_width: u32,
}

/// The draft model's geometry.
///
/// Kept because the draft is part of this package rather than a second model
/// fetched from somewhere else, and its shape is stated nowhere else in it.
#[derive(Debug, Clone, Deserialize)]
pub struct DraftGeometry {
    /// The draft architecture's own name.
    pub architecture: String,
    /// Tokens the draft proposes in one block.
    pub block_size: u32,
    /// Channels per dynamic-convolution group.
    pub dynamic_conv_group_size: u32,
    /// Taps in the dynamic convolution.
    pub dynamic_conv_kernel_size: u32,
    /// The draft's residual stream width.
    pub hidden_size: u32,
    /// The draft's feed-forward width.
    pub intermediate_size: u32,
    /// Whether the draft attends causally.
    pub is_causal: bool,
    /// How many layers the draft has.
    pub layers: u32,
    /// Rank of the successor/predecessor selector.
    pub selector_rank: u32,
    /// Candidates the selector keeps.
    pub selector_top_k: u32,
    /// The draft's attention window.
    pub sliding_window: u32,
    /// Which target layers the draft reads hidden states from.
    pub target_capture_layers: Vec<u32>,
}

/// The sizes the converter expects this package to be run at.
///
/// Advice rather than law - the engine's own budgets win - but it is the
/// packer's statement of what the widths were chosen for, and a run far outside
/// it is worth knowing about.
#[derive(Debug, Clone, Deserialize)]
pub struct ExecutionGeometry {
    /// Bytes per device allocation extent.
    pub allocation_extent_target_bytes: u64,
    /// Tokens per attention history group.
    pub attention_history_group: u32,
    /// Tokens the draft proposes per step.
    pub draft_proposal_tokens: u32,
    /// Query rows the draft runs at once.
    pub draft_query_rows: u32,
    /// The draft's attention window at run time.
    pub draft_sliding_window: u32,
    /// Sequences in flight.
    pub maximum_batch_width: u32,
    /// Tokens per prefill chunk.
    pub prefill_token_budget: u32,
    /// Tokens per target KV block.
    pub target_kv_block_tokens: u32,
    /// Rows the target verifies per step.
    pub target_verify_rows: u32,
}

/// What produced this package.
///
/// Provenance, not configuration: nothing here changes how a byte is read. It
/// is kept because "which converter wrote these bytes" is the first question a
/// package that decodes to nonsense raises, and the answer costs five strings.
#[derive(Debug, Clone, Deserialize)]
pub struct Converter {
    /// Path of the conversion script within its own repository.
    pub source_path: String,
    /// The commit that script was at.
    pub source_git_commit: String,
    /// Digest of the script.
    pub source_sha256: String,
    /// Digest of the adapter the script ran with.
    pub adapter_sha256: String,
    /// The MLX version the conversion ran under.
    pub mlx_version: String,
}

/// One repository a part of this package was converted from.
#[derive(Debug, Clone, Deserialize)]
pub struct UpstreamRef {
    /// The Hugging Face repository id.
    pub repo_id: String,
    /// The exact revision taken.
    pub revision: String,
}

/// Where each part of the package came from. Provenance, like [`Converter`].
#[derive(Debug, Clone, Deserialize)]
pub struct Upstream {
    /// The target model's source.
    pub target: UpstreamRef,
    /// The draft model's source.
    pub draft: UpstreamRef,
    /// The tokenizer's source.
    pub tokenizer: UpstreamRef,
    /// The vision tower's source.
    pub vision: UpstreamRef,
}

/// A package's whole `manifest.json`.
///
/// Every top-level key of the shipped document has a field here. That is
/// deliberate: a key read past silently is a fact the package stated and the
/// reader ignored, which fails no test and shows up later as a value somebody
/// assumed instead.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    /// The document version. Refused at parse unless it is [`SCHEMA_VERSION`].
    #[serde(deserialize_with = "schema_version")]
    pub schema_version: u32,
    /// The model this package is, as the converter named it.
    pub model: String,
    /// How the bytes were packed.
    pub format: Format,
    /// The target stack's geometry.
    pub target: TargetGeometry,
    /// The draft stack's geometry.
    pub draft: DraftGeometry,
    /// The sizes the converter expects it to be run at.
    pub execution_geometry: ExecutionGeometry,
    /// What produced the package.
    pub converter: Converter,
    /// Where its parts came from.
    pub upstream: Upstream,
    /// A digest over the artifact list. Kept, and deliberately not checked:
    /// see the module docs.
    pub artifact_set_sha256: String,
    /// Every file, in the order the package declares them.
    pub artifacts: Vec<Artifact>,
}

impl Manifest {
    /// Reads a `manifest.json`.
    ///
    /// # Errors
    ///
    /// [`serde_json::Error`] naming the field - and, for a document written to
    /// another schema, naming `schema_version` and both numbers.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }
}

/// Reads `schema_version`, refusing any version but [`SCHEMA_VERSION`].
///
/// The refusal is at parse rather than after it because every other field in
/// [`Manifest`] is a version-4 field: reading them out of a version-5 document
/// would be reading version 4's meanings off version 5's keys, which is the
/// failure this check exists to prevent rather than to report.
fn schema_version<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let got = u32::deserialize(deserializer)?;
    if got == SCHEMA_VERSION {
        Ok(got)
    } else {
        Err(serde::de::Error::custom(format!(
            "schema_version {got}: this reader was written for splash manifest \
             schema_version {SCHEMA_VERSION}"
        )))
    }
}

/// What a verification found wrong, naming the file.
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    /// The file could not be read.
    #[error("{path}: {source}")]
    Io {
        /// The artifact.
        path: PathBuf,
        /// The underlying failure.
        source: std::io::Error,
    },
    /// The file is the wrong length.
    #[error("{path}: declared {want} bytes, found {got}")]
    Size {
        /// The artifact.
        path: PathBuf,
        /// Declared size.
        want: u64,
        /// Observed size.
        got: u64,
    },
    /// The digest disagrees.
    #[error("{path}: declared sha256 {want}, computed {got}")]
    Digest {
        /// The artifact.
        path: PathBuf,
        /// Declared digest.
        want: String,
        /// Computed digest.
        got: String,
    },
}

impl VerifyError {
    /// The artifact this refusal is about.
    #[must_use]
    pub fn path(&self) -> &Path {
        match self {
            Self::Io { path, .. } | Self::Size { path, .. } | Self::Digest { path, .. } => path,
        }
    }
}

/// Verifies every artifact against `root`.
///
/// A package downloaded through the Hugging Face cache stores each file as a
/// symlink into the blob store; opening the path follows it, and both the
/// length and the bytes are the blob's.
///
/// # Errors
///
/// [`VerifyError`] naming the first artifact that disagrees.
pub fn verify(manifest: &Manifest, root: &Path) -> Result<(), VerifyError> {
    for artifact in &manifest.artifacts {
        let path = root.join(&artifact.path);
        let io = |source| VerifyError::Io {
            path: path.clone(),
            source,
        };
        let mut file = std::fs::File::open(&path).map_err(io)?;
        let got = file.metadata().map_err(io)?.len();
        if got != artifact.size {
            return Err(VerifyError::Size {
                path,
                want: artifact.size,
                got,
            });
        }
        let digest = format!("{:x}", hash(&mut file).map_err(io)?);
        if digest != artifact.sha256 {
            return Err(VerifyError::Digest {
                path,
                want: artifact.sha256.clone(),
                got: digest,
            });
        }
    }
    Ok(())
}

/// SHA-256 of everything left in `reader`, a buffer at a time.
///
/// 1 MiB is the buffer because the cost here is the hashing and not the read:
/// a larger one buys nothing measurable and a whole-file read would make the
/// peak an artifact rather than a page.
fn hash(reader: &mut impl Read) -> std::io::Result<sha2::digest::Output<Sha256>> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            return Ok(hasher.finalize());
        }
        hasher.update(&buffer[..read]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped manifest's whole top level, artifact list and digests
    /// abbreviated. Every other key and value is the real document's, because
    /// the point of these cases is that the reader keeps them.
    const DOCUMENT: &str = r#"{
      "artifact_set_sha256": "0c46",
      "artifacts": [
        {"path": "target/layer-0.bin", "sha256": "6e49", "size": 34324480}
      ],
      "converter": {
        "adapter_sha256": "d8a3", "mlx_version": "0.32.2",
        "source_git_commit": "d441", "source_path": "install/convert.py",
        "source_sha256": "dd4f"},
      "draft": {
        "architecture": "DFlash2DraftModel", "block_size": 8,
        "dynamic_conv_group_size": 16, "dynamic_conv_kernel_size": 2,
        "hidden_size": 2048, "intermediate_size": 6144, "is_causal": false,
        "layers": 6, "selector_rank": 256, "selector_top_k": 16,
        "sliding_window": 2048,
        "target_capture_layers": [1, 6, 11, 16, 22, 27, 32, 37]},
      "execution_geometry": {
        "allocation_extent_target_bytes": 134217728,
        "attention_history_group": 128, "draft_proposal_tokens": 7,
        "draft_query_rows": 8, "draft_sliding_window": 2048,
        "maximum_batch_width": 4, "prefill_token_budget": 2048,
        "target_kv_block_tokens": 32, "target_verify_rows": 8},
      "format": {
        "draft_layer_magic": "MDFD0004", "name": "splash-packed-q4-moe",
        "q4_bits": 4, "q8_bits": 8, "quant_group_size": 64,
        "section_alignment_bytes": 16384, "storage_n": 256,
        "target_layer_magic": "MDFM0001", "vision_magic": "MDFV0001"},
      "model": "Qwen3.6-35B-A3B",
      "schema_version": 4,
      "target": {
        "architecture": "qwen3_5_moe", "attention_packed_width": 9216,
        "experts": 256, "experts_per_token": 8, "gdn_actual_width": 12352,
        "gdn_packed_width": 12544, "hidden_size": 2048,
        "layer_types": ["gdn", "gdn", "gdn", "attention"],
        "layers": 40, "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512, "vocabulary_size": 248320},
      "upstream": {
        "draft": {"repo_id": "a/b", "revision": "8e71"},
        "target": {"repo_id": "c/d", "revision": "3874"},
        "tokenizer": {"repo_id": "c/d", "revision": "3874"},
        "vision": {"repo_id": "c/d", "revision": "3874"}}}"#;

    /// The document above with `artifacts` replaced, so a verification case can
    /// point at files it just wrote without restating the top level.
    fn manifest_of(artifacts: Vec<Artifact>) -> Manifest {
        Manifest {
            artifacts,
            ..Manifest::from_json(DOCUMENT).expect("the fixture parses")
        }
    }

    #[test]
    fn a_manifest_lists_every_artifact_with_its_digest() {
        let m = Manifest::from_json(DOCUMENT).expect("manifest parses");
        assert_eq!(m.artifacts.len(), 1);
        assert_eq!(m.artifacts[0].size, 34_324_480);
        assert_eq!(m.artifact_set_sha256, "0c46");
    }

    #[test]
    fn the_whole_top_level_is_kept_and_not_read_past() {
        let m = Manifest::from_json(DOCUMENT).expect("manifest parses");
        assert_eq!(m.schema_version, SCHEMA_VERSION);
        assert_eq!(m.model, "Qwen3.6-35B-A3B");
        // The two constants a codec on the other side has to agree with rather
        // than assume.
        assert_eq!(m.format.storage_n, 256);
        assert_eq!(m.format.quant_group_size, 64);
        assert_eq!(m.format.name, "splash-packed-q4-moe");
        assert_eq!(m.format.section_alignment_bytes, 16_384);
        assert_eq!(m.format.target_layer_magic, "MDFM0001");
        // Both recurrent widths, which is the whole padding question stated by
        // the package itself rather than inferred from a section.
        assert_eq!(m.target.gdn_actual_width, 12_352);
        assert_eq!(m.target.gdn_packed_width, 12_544);
        assert_eq!(m.target.gdn_packed_width - m.target.gdn_actual_width, 192);
        assert_eq!(m.target.experts, 256);
        assert_eq!(m.target.experts_per_token, 8);
        assert_eq!(m.target.vocabulary_size, 248_320);
        assert_eq!(m.target.layer_types[3], LayerKind::Attention);
        assert_eq!(
            shiso_engine::shape::LayerKind::from(m.target.layer_types[0]),
            shiso_engine::shape::LayerKind::Gdn
        );
        assert_eq!(m.draft.layers, 6);
        assert_eq!(m.draft.target_capture_layers.len(), 8);
        assert!(!m.draft.is_causal);
        assert_eq!(m.execution_geometry.target_kv_block_tokens, 32);
        assert_eq!(m.converter.mlx_version, "0.32.2");
        assert_eq!(m.upstream.target.repo_id, "c/d");
    }

    #[test]
    fn a_schema_this_reader_was_not_written_for_is_refused_by_number() {
        let bad = DOCUMENT.replace("\"schema_version\": 4", "\"schema_version\": 5");
        assert_ne!(bad, DOCUMENT, "the fixture still states a version");
        let err = Manifest::from_json(&bad).expect_err("version 5 has no reader");
        let text = format!("{err}");
        assert!(text.contains("schema_version"), "{text}");
        assert!(text.contains('5') && text.contains('4'), "{text}");
    }

    #[test]
    fn the_version_this_reader_was_written_for_is_accepted() {
        let m = Manifest::from_json(DOCUMENT).expect("version 4 parses");
        assert_eq!(m.schema_version, 4);
    }

    #[test]
    fn a_layer_role_this_build_cannot_run_is_refused_by_name() {
        let bad = DOCUMENT.replace("\"gdn\", \"gdn\", \"gdn\"", "\"gdn\", \"mamba\", \"gdn\"");
        let err = Manifest::from_json(&bad).expect_err("mamba is not a layer here");
        assert!(
            format!("{err}").contains("mamba"),
            "the error names the role"
        );
    }

    #[test]
    fn a_digest_that_does_not_match_is_refused_by_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("target")).expect("mkdir");
        std::fs::write(dir.path().join("target/layer-0.bin"), b"not the bytes").expect("write");
        let m = manifest_of(vec![Artifact {
            path: "target/layer-0.bin".into(),
            sha256: "0".repeat(64),
            size: 13,
        }]);
        let err = verify(&m, dir.path()).expect_err("digest disagrees");
        assert!(format!("{err}").contains("target/layer-0.bin"));
        assert!(matches!(err, VerifyError::Digest { .. }), "{err}");
    }

    #[test]
    fn a_truncated_file_is_refused_as_a_length_before_it_is_hashed() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("half.bin"), b"half").expect("write");
        let m = manifest_of(vec![Artifact {
            path: "half.bin".into(),
            sha256: "0".repeat(64),
            size: 8,
        }]);
        let err = verify(&m, dir.path()).expect_err("the file is short");
        match &err {
            VerifyError::Size { want, got, .. } => {
                assert_eq!((*want, *got), (8, 4));
            }
            other => panic!("expected a length refusal, got {other}"),
        }
        assert!(err.path().ends_with("half.bin"));
    }

    #[test]
    fn an_artifact_that_is_not_there_is_refused_by_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let m = manifest_of(vec![Artifact {
            path: "target/layer-0.bin".into(),
            sha256: "0".repeat(64),
            size: 13,
        }]);
        let err = verify(&m, dir.path()).expect_err("nothing is there");
        assert!(matches!(err, VerifyError::Io { .. }), "{err}");
        assert!(format!("{err}").contains("target/layer-0.bin"));
    }

    #[test]
    fn a_file_that_agrees_passes_over_more_than_one_buffer() {
        // Two full buffers and a tail, so the streaming hash is exercised
        // rather than the single-read case that a small fixture would take.
        let bytes: Vec<u8> = (0..(2 * (1 << 20) + 7_usize))
            .map(|i| u8::try_from(i % 251).unwrap_or(0))
            .collect();
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("big.bin"), &bytes).expect("write");
        let m = manifest_of(vec![Artifact {
            path: "big.bin".into(),
            sha256: format!("{:x}", Sha256::digest(&bytes)),
            size: u64::try_from(bytes.len()).expect("the fixture's length"),
        }]);
        verify(&m, dir.path()).expect("the bytes are the declared ones");
    }
}
