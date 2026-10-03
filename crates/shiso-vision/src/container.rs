//! The doors a vision tower's weights arrive through.
//!
//! What is here is the shape of the two containers and nothing of the tower
//! inside them: which file a checkpoint names as its tower shard, and how a
//! standalone container is opened. Deciding what the tensors *mean* — the
//! manifest they must match, the axis order they are stored in, the contract
//! that judges their dimensions — belongs to the family that reads them, and
//! each family's loader composes these helpers into its own upload loop.
//!
//! A family that ships its tower as a plain GGUF next to the language model
//! opens it with [`open_gguf`] and never touches the shard path at all; the
//! shard door exists for checkpoints that pack the tower into a safetensors
//! block beside the rest of the weights.

use std::path::{Path, PathBuf};

use shiso_core::HfConfig;
use shiso_modelfile::GgufFile;

use crate::error::VisionError;

/// Where a vision tower's weights come from.
///
/// Not an `Option` anywhere: "this deployment has no vision tower" is the
/// absence of a source at the call site, so a result of `Option` here could
/// only ever mean "the source existed but produced nothing", which is a load
/// error rather than a configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisionSource {
    /// A checkpoint directory. The tower rides in a shard beside the language
    /// model and its dimensions come from the directory's own config.
    OptiqDir(PathBuf),
    /// A standalone tower file, named by the operator. Its dimensions come from
    /// its own metadata and are then judged against the family's contract.
    SidecarGguf(PathBuf),
    /// A `modelopt` checkpoint directory, whose tower rides in the SAME shard
    /// set as the language model rather than in a shard of its own.
    ///
    /// The distinction from [`VisionSource::OptiqDir`] is where the bytes are,
    /// not what they mean: an `OptiQ` directory carries a separate
    /// `optiq_vision.safetensors` beside the language model, and a `modelopt`
    /// one carries `model.visual.*` interleaved through
    /// `model-0000N-of-0000M.safetensors` with everything else. The tower is
    /// the same tower and reaches the same manifest.
    ModeloptDir(PathBuf),
    /// A trellis-quantized tower file inside a checkpoint directory, whose
    /// dimensions come from the directory's own config. The path is the
    /// tower file.
    TrellisShard(PathBuf),
}

impl VisionSource {
    /// The path the operator named.
    #[must_use]
    pub fn path(&self) -> &Path {
        match self {
            Self::OptiqDir(path)
            | Self::SidecarGguf(path)
            | Self::ModeloptDir(path)
            | Self::TrellisShard(path) => path,
        }
    }
}

/// Config file naming the architecture, the tower and where its shard lives.
const CONFIG_FILE: &str = "config.json";

/// Key holding the tower shard's path, relative to the checkpoint directory.
const TOWER_BLOCK_KEY: &str = "optiq_vision";

/// Sub-key under [`TOWER_BLOCK_KEY`] naming the shard.
const TOWER_PATH_KEY: &str = "sidecar";

/// Where the tower shard sits when the config does not say.
const DEFAULT_TOWER_SHARD: &str = "optiq/optiq_vision.safetensors";

/// Reads a checkpoint's config and resolves the path of its tower shard.
///
/// The shard is named, not opened: a checkpoint whose config carries no vision
/// block has no tower to load, and the family that reads that refusal must
/// answer it before a file the tower would live in is reached for.
///
/// One read, two readers: the config parser knows the model and nothing about
/// where the tower shard lives, and the shard path is one key the parser does
/// not model. Re-reading the file for the second would let the two disagree
/// about which config they read.
///
/// # Errors
///
/// [`VisionError::TowerIo`] when the config cannot be read and
/// [`VisionError::TowerConfig`] when it does not parse.
pub fn open_checkpoint(dir: &Path) -> Result<(HfConfig, PathBuf), VisionError> {
    let config_path = dir.join(CONFIG_FILE);
    let text = std::fs::read_to_string(&config_path).map_err(|source| VisionError::TowerIo {
        path: config_path,
        source,
    })?;
    let config = HfConfig::from_json_str(&text).map_err(|source| VisionError::TowerConfig {
        path: dir.join(CONFIG_FILE),
        source,
    })?;
    Ok((config, dir.join(shard_relative_path(&text))))
}

/// Where the tower shard sits, according to the checkpoint's own config.
///
/// Falls back to the conventional location rather than failing: the key is a
/// convenience, and a checkpoint that lays its tower out conventionally but
/// omits the key is not broken. A shard that is missing at both is reported by
/// the open that follows, with the path in it.
fn shard_relative_path(config_text: &str) -> PathBuf {
    let declared = serde_json::from_str::<serde_json::Value>(config_text)
        .ok()
        .and_then(|document| {
            document
                .get(TOWER_BLOCK_KEY)?
                .get(TOWER_PATH_KEY)?
                .as_str()
                .map(PathBuf::from)
        });
    declared.unwrap_or_else(|| PathBuf::from(DEFAULT_TOWER_SHARD))
}

/// Opens a standalone container the operator named.
///
/// # Errors
///
/// [`VisionError::TowerContainer`] when the file is not a readable container.
pub fn open_gguf(path: &Path) -> Result<GgufFile, VisionError> {
    GgufFile::open(path).map_err(|source| VisionError::TowerContainer {
        path: path.to_path_buf(),
        source,
    })
}
