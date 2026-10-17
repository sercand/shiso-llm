//! The one place a parsed config becomes a running model.
//!
//! # Why this is a match and not a registry
//!
//! Large serving stacks reach for a dynamic registry - a table of architecture
//! strings mapped to constructors, populated at startup - because they carry a
//! hundred architectures and cannot afford a recompile per addition. This port
//! carries one family today and a second one later, and for two entries a
//! registry buys nothing and costs the thing that matters most here: a
//! compile-time list of what is served. Adding a family means adding an arm,
//! and the compiler is what tells you that the arm is missing.
//!
//! # Why the dispatch is on the config and not on a shape enum
//!
//! There used to be a closed `ShapeProfile` enum in the core crate with one
//! variant per family, and matching on it was exactly this. That enum is gone,
//! because a crate underneath every model cannot know which models exist
//! without the whole stack being coupled to a fixed list of them. What replaced
//! it is this: each family crate owns a reader that either recognises a config
//! document or reports that it describes something else, and the dispatch here
//! is the ordered list of those readers. The closed set did not disappear - it
//! moved up here, to the one crate that is allowed to know the whole list.
//!
//! # Why the loaded model is `Checkpointed` and not `Model`
//!
//! Every family this build serves can store and resume a prefix, and the disk
//! tier is switched on *after* loading - only when a directory was given - so a
//! caller that wants a model without touching the disk still gets one, which is
//! what makes a boot test possible at all. Returning the plain contract here
//! would make that impossible: the checkpointing methods would be gone by the
//! time anybody could ask for them, and there is no way back down a trait
//! object. Going the other way is free, because a `&mut dyn Checkpointed`
//! upcasts to the `&mut dyn Model` every forward pass takes.
//!
//! A family that genuinely cannot checkpoint is a reason to widen this into an
//! enum of the two, not a reason to narrow every arm to the weaker contract.

use std::path::Path;
use std::sync::Arc;

use shiso_audio::{AudioError, AudioTower};
use shiso_core::{Backend, CoreError, HfConfig};
use shiso_engine::EngineOptions;
use shiso_engine::ckpt::Checkpointed;
use shiso_engine::sparse::capture;
use shiso_engine::sparse::mtp::SpecDepth;
use shiso_engine::sparse::stack::SparseOptions;
use shiso_models::gemma4::{
    Gemma4AudioTower, Gemma4Config, Gemma4Model, Gemma4Tower, PleOptions, holds_audio_tower,
};
use shiso_models::lfm2::{Lfm2Config, Lfm2Model, Lfm2Tower};
use shiso_models::nemotron_h::{NemotronHConfig, NemotronHModel};
use shiso_models::qwen4exp::{Qwen4ExpConfig, Qwen4ExpModel};
use shiso_models::qwen35::mtp::{self, AffineQuant, MtpManifest};
use shiso_models::qwen35::{Qwen35Config, Qwen35Model, Qwen35Tower};
use shiso_models::qwen35moe::{Qwen35MoeConfig, Qwen35MoeModel};
use shiso_vision::{MediaPlacement, VisionError, VisionTower};

use crate::cli::ServeArgs;
use crate::tokenizer::render_prompt::{EffortSpelling, PromptRenderer};
use shiso_parsers::{ReasoningScheme, ToolCallScheme};

/// Why a checkpoint could not be turned into a model.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FactoryError {
    /// The config document describes no family this build serves.
    ///
    /// Carries the spelling as it appeared, because that is the one thing an
    /// operator can compare against what they meant to serve.
    #[error("no model in this build serves the architecture {model_type:?}")]
    UnknownArchitecture {
        /// The `model_type` as the config spelled it.
        model_type: String,
    },
    /// The family was recognised but the checkpoint would not load.
    #[error("the checkpoint could not be loaded")]
    Load(#[source] CoreError),
    /// The scheduler's prefill width is wider than this model's widest pass.
    ///
    /// A solo read wider than a pass is split into passes underneath, but a
    /// batched scoring pass is planned at that width and must be one pass:
    /// past it every batch would be refused call by call. Refused at boot, by
    /// the flags that set the width, rather than discovered per request.
    #[error(
        "--solo-prefill-chunk / --mixed-prefill-quantum (or --max-num-batched-tokens) ask for \
         {widest} prompt rows a step, wider than this model's widest pass of {rows}"
    )]
    PrefillTooWide {
        /// The widest prefill width the scheduler was configured with.
        widest: usize,
        /// The rows one pass of this model reads at most.
        rows: u32,
    },
}

/// How a family's vision tower is opened, when the operator names one.
///
/// A function pointer rather than a trait object because there is nothing to
/// capture: the family is already decided by the time one of these is chosen,
/// and the two things an opener needs - a device and a path - are the two
/// arguments. A family that ships no tower supplies a refusal rather than a
/// `None`, so that every arm answers the same question and the boot path has
/// one message to print instead of two shapes to match on.
///
/// # Errors
///
/// Whatever the family's loader refuses the source for, and
/// [`VisionError::Unsupported`] for a family this build serves no tower for.
pub type TowerOpener = fn(&Arc<dyn Backend>, &Path) -> Result<Box<dyn VisionTower>, VisionError>;

/// How a family's audio tower is opened from the container `--vision` names.
///
/// The same shape as [`TowerOpener`], and the same path: a family that hears
/// ships its audio tower in the container that carries its vision tower.
///
/// # Errors
///
/// Whatever the family's loader refuses the container for.
pub type AudioTowerOpener = fn(&Arc<dyn Backend>, &Path) -> Result<AudioTowerFound, AudioError>;

/// What a family's audio opener found in the tower container.
#[derive(Debug)]
pub enum AudioTowerFound {
    /// The tower, loaded.
    Tower(Arc<dyn AudioTower>),
    /// No audio tower to open - an export without one, or a family that does
    /// not hear - which turns audio input off rather than failing boot.
    Absent,
}

/// What a family says about its own output, for the server to default to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FamilyDefaults {
    /// The default reasoning scheme.
    pub reasoning: ReasoningScheme,
    /// The default tool-call scheme.
    pub tools: ToolCallScheme,
    /// Whether thinking is enabled by default.
    pub thinking_default: bool,
    /// How this family's template spells each effort rung.
    ///
    /// Templates raise on a word outside their list, so a family that borrowed
    /// another's spelling turned a valid request into a refusal.
    pub effort: EffortSpelling,
    /// Whether this family's arm honours `--spec-k` at all.
    ///
    /// **This field exists to close a hole that a relaxation opened.** The
    /// command line used to refuse `--spec-k` without `--draft-model` for
    /// every family; that made the dense family's own default - the
    /// `mtp_file` its sidecar names - unreachable, so the rule moved to the
    /// arms. The arms that read `args.spec_k` then refused correctly and the
    /// one that does not read it at all, Gemma 4's, served with no
    /// speculation and no error: exactly "a server that quietly decodes one
    /// token a step while an operator believes it is speculating".
    ///
    /// It lives HERE, on the per-family constants, for two reasons that a
    /// check written inside one arm cannot give:
    ///
    /// - [`FamilyDefaults::for_config`] runs *before* any arm, so the refusal
    ///   happens before several gigabytes of weights move.
    /// - Every family's defaults are a named constant built field by field, so
    ///   **a family added later cannot inherit this silently** - the compiler
    ///   demands an answer. And a family that gains speculation flips its own
    ///   constant in the same place, so the answer cannot go stale in the
    ///   other direction either, which a hard-coded list of non-speculating
    ///   families would.
    ///
    /// `true` does not mean a head is present. It means the arm is responsible
    /// for finding one and for refusing by name when it cannot, which is what
    /// [`draft_manifest`] does for the dense family and what the sparse arm
    /// does for itself.
    pub speculates: bool,
}

impl FamilyDefaults {
    /// Family defaults for Qwen 3 models: Qwen 3's reasoning tags and its JSON
    /// tool-call body.
    ///
    /// No family [`FamilyDefaults::for_config`] recognises selects this today;
    /// the sparse family did until it gained [`FamilyDefaults::QWEN4EXP`].
    pub const QWEN3: Self = Self {
        reasoning: ReasoningScheme::Qwen3,
        tools: ToolCallScheme::Qwen3,
        thinking_default: true,
        effort: EffortSpelling::STANDARD,
        speculates: true,
    };

    /// Family defaults for the sparse hybrid checkpoint.
    ///
    /// A constant of its own rather than `QWEN3`: this family's template writes
    /// tool calls as the Qwen 3.5 XML body and accepts its own effort words, and a
    /// family that borrows another's constant inherits every fact that differs.
    ///
    /// The tool scheme is the one that fact decided. The template writes a call
    /// as `<function=NAME>` wrapping one `<parameter=KEY>` block per argument;
    /// while this family borrowed Qwen 3's JSON reader, every such call failed to
    /// decode and went back to the client as prose, with nothing failing.
    pub const QWEN4EXP: Self = Self {
        reasoning: ReasoningScheme::Qwen3,
        tools: ToolCallScheme::Qwen35,
        thinking_default: true,
        // Its template accepts only low, medium and xhigh and raises on high.
        effort: EffortSpelling::XHIGH_TOP,
        // The sparse family drafts with a head named by `--draft-model`; its
        // arm refuses by name when the depth is asked for without one.
        speculates: true,
    };

    /// Family defaults for Qwen 3.5 models.
    ///
    /// The reasoning tags are Qwen 3's and the tool-call body is not: the
    /// template writes a call as `<function=NAME>` wrapping one
    /// `<parameter=KEY>` block per argument, and a JSON reader hands every one
    /// of them back to the client as prose.
    pub const QWEN35: Self = Self {
        reasoning: ReasoningScheme::Qwen3,
        tools: ToolCallScheme::Qwen35,
        thinking_default: true,
        effort: EffortSpelling::STANDARD,
        // The dense family drafts with the head its sidecar names under
        // `mtp_file`; `draft_manifest` resolves it and refuses by name.
        speculates: true,
    };

    /// Family defaults for the Qwen 3.5 mixture-of-experts checkpoint.
    ///
    /// The same two schemes the dense hybrid emits, because it is the same
    /// template: this checkpoint ships Qwen 3's reasoning tags and the Qwen 3.5
    /// XML tool-call body, and a JSON reader over the latter hands every call
    /// back as prose.
    pub const QWEN35MOE: Self = Self {
        reasoning: ReasoningScheme::Qwen3,
        tools: ToolCallScheme::Qwen35,
        thinking_default: true,
        effort: EffortSpelling::STANDARD,
        // **True since this family gained a head it can run.** It was false,
        // and the reason was sound at the time: the `modelopt` export's own MTP
        // head is BF16 and this build has no BF16 matvec, so there was no head
        // to find and a `true` would have meant an arm responsible for finding
        // one that did not exist.
        //
        // What changed is a second head, published as a directory of its own -
        // `Qwen3.6-35B-A3B-MTP-4bit`, MLX affine 4-bit group 64 - which this
        // build does decode, and which drafts at 0.860 acceptance. So the
        // question moves to the arm, where it belongs, and the arm answers it
        // for BOTH containers: see `load_qwen35moe`, which refuses `--spec-k`
        // by name for the directory container that still has no runnable head,
        // and requires `--draft-model` for the GGUF that does.
        //
        // That split is the whole reason this is not simply `true` and left
        // there. An arm that does not read `args.spec_k` answers it by silence,
        // which is how Gemma 4 once served with no speculation and no error.
        speculates: true,
    };

    /// Family defaults for the hybrid state-space checkpoint.
    ///
    /// Its template wraps reasoning in Qwen 3's tags and writes a tool call as
    /// the Qwen 3.5 XML body, `<function=NAME>` around `<parameter=KEY>`
    /// blocks, and takes no effort words.
    pub const NEMOTRON_H: Self = Self {
        reasoning: ReasoningScheme::Qwen3,
        tools: ToolCallScheme::Qwen35,
        thinking_default: true,
        effort: EffortSpelling::STANDARD,
        // No draft head is bound for this family yet; the export's own is not
        // read. Flip this with the arm that binds one.
        speculates: false,
    };

    /// Family defaults for the LFM2 hybrid.
    ///
    /// Its template wraps reasoning in Qwen 3's `<think>` tags but opens no
    /// reasoning block of its own, so thinking defaults off. It writes a tool
    /// call as a bracketed Python call list between `<|tool_call_start|>` and
    /// `<|tool_call_end|>`, which no reader in this build parses, so a call
    /// comes back as prose rather than being misread as another family's.
    pub const LFM2: Self = Self {
        reasoning: ReasoningScheme::Qwen3,
        tools: ToolCallScheme::None,
        thinking_default: false,
        effort: EffortSpelling::STANDARD,
        // No draft head exists for this family.
        speculates: false,
    };

    /// Family defaults for Gemma 4 models.
    pub const GEMMA4: Self = Self {
        reasoning: ReasoningScheme::Gemma4,
        tools: ToolCallScheme::Gemma4,
        thinking_default: false,
        effort: EffortSpelling::STANDARD,
        // This family ships no draft head and its arm binds none. Flip this
        // the day it does, and the refusal below stops firing for it - which
        // is why the answer lives on the constant rather than in a list of
        // families somebody would have to remember to edit.
        speculates: false,
    };

    /// Resolves the family defaults for the architecture described by `config`.
    ///
    /// This is the one table, and [`load_model`] reads it rather than naming a
    /// constant per arm: the schemes a family emits are one fact, and a second
    /// copy of it is a copy that can be wrong while every test still passes.
    ///
    /// The order is the order the readers are tried in [`load_model`], and each
    /// entry asks the same question its reader asks - whether the *document*
    /// names the family - so the two cannot disagree about which arm a
    /// checkpoint takes.
    ///
    /// # Errors
    /// Returns [`FactoryError::UnknownArchitecture`] if the config document is not recognised.
    pub fn for_config(config: &HfConfig) -> Result<Self, FactoryError> {
        if !matches!(
            Qwen35Config::from_hf(config),
            Err(CoreError::UnknownArch { .. })
        ) {
            return Ok(Self::QWEN35);
        }
        // After the dense reader, and the order is load-bearing in one
        // direction only. The dense reader recognises `qwen3_5` and
        // `qwen3_5_text` and this one recognises `qwen3_5_moe` and
        // `qwen3_5_moe_text` - two disjoint sets - so neither can claim the
        // other's document whichever way round they are asked. What the order
        // preserves is the account above: this list is the order `load_model`
        // tries its arms in, and the two must not be able to disagree.
        if Qwen35MoeConfig::describes(config) {
            return Ok(Self::QWEN35MOE);
        }
        if Qwen4ExpConfig::describes(config) {
            return Ok(Self::QWEN4EXP);
        }
        if Gemma4Config::describes(config) {
            return Ok(Self::GEMMA4);
        }
        if NemotronHConfig::describes_type(config.model_type()) {
            return Ok(Self::NEMOTRON_H);
        }
        if Lfm2Config::describes(config) {
            return Ok(Self::LFM2);
        }
        Err(FactoryError::UnknownArchitecture {
            model_type: config.model_type().to_owned(),
        })
    }
}

impl FamilyDefaults {
    /// Applies this family's rendering facts to a renderer.
    ///
    /// The one place the boot path and the parity tests share, so a fact added
    /// here reaches both - a test that configured its own renderer would pass
    /// while production rendered with a different default.
    #[must_use]
    pub fn configure(self, renderer: PromptRenderer) -> PromptRenderer {
        renderer
            .with_thinking_default(self.thinking_default)
            .with_effort_spelling(self.effort)
    }
}

/// A loaded checkpoint and the two family facts the server needs beside it.
///
/// The placement is here rather than on the tower alone because a server
/// started WITHOUT `--vision` still renders prompts: the checkpoint declares a
/// vision geometry, the template still writes a media marker, and a renderer
/// that did not know the family's marker would silently leave it unexpanded.
/// So the family answers "where does media go" at load time, and the tower - if
/// one is ever opened - answers it again from the same constant.
pub struct Loaded {
    /// The model, ready to be given to a scheduler.
    pub model: Box<dyn Checkpointed>,
    /// How this family's tower is opened, when the operator names one.
    pub open_tower: TowerOpener,
    /// How this family's audio tower is opened from the same container.
    pub open_audio: AudioTowerOpener,
    /// Where this family's template puts media.
    pub placement: MediaPlacement,
    /// What this family defaults to for output parsing and thinking.
    pub defaults: FamilyDefaults,
}

impl std::fmt::Debug for Loaded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Loaded")
            .field("placement", &self.placement)
            .field("defaults", &self.defaults)
            .finish_non_exhaustive()
    }
}

/// The LFM2 family's opener.
fn open_lfm2_tower(
    backend: &Arc<dyn Backend>,
    path: &Path,
) -> Result<Box<dyn VisionTower>, VisionError> {
    Ok(Box::new(Lfm2Tower::open(backend, path)?))
}

/// The Qwen family's opener.
fn open_qwen35_tower(
    backend: &Arc<dyn Backend>,
    path: &Path,
) -> Result<Box<dyn VisionTower>, VisionError> {
    Ok(Box::new(Qwen35Tower::open(backend, path)?))
}

/// The opener of a family that ships no vision tower: `--vision` is refused.
fn open_no_tower(
    _backend: &Arc<dyn Backend>,
    _path: &Path,
) -> Result<Box<dyn VisionTower>, VisionError> {
    Err(VisionError::Unsupported {
        what: "a vision tower for this family",
    })
}

/// The opener of a family that does not hear: audio input stays off.
#[expect(
    clippy::unnecessary_wraps,
    reason = "every family's opener has the one signature the boot path calls"
)]
fn no_audio_tower(
    _backend: &Arc<dyn Backend>,
    _path: &Path,
) -> Result<AudioTowerFound, AudioError> {
    Ok(AudioTowerFound::Absent)
}

/// The Gemma 4 family's audio opener: absent when the container was exported
/// without the tower.
fn open_gemma4_audio(
    backend: &Arc<dyn Backend>,
    path: &Path,
) -> Result<AudioTowerFound, AudioError> {
    if !holds_audio_tower(path)? {
        return Ok(AudioTowerFound::Absent);
    }
    Ok(AudioTowerFound::Tower(Arc::new(Gemma4AudioTower::open(
        backend, path,
    )?)))
}

/// The Gemma 4 family's opener.
fn open_gemma4_tower(
    backend: &Arc<dyn Backend>,
    path: &Path,
) -> Result<Box<dyn VisionTower>, VisionError> {
    Ok(Box::new(Gemma4Tower::open(backend, path)?))
}

/// Loads the checkpoint at `path` as whichever family `config` describes.
///
/// Takes `&ServeArgs` whole rather than a merged options struct: the factory
/// lives in the crate that owns the command line, so each arm reads the knobs
/// its own architecture has, and no union-of-all-models configuration type has
/// to exist for arms to share.
///
/// # Errors
///
/// [`FactoryError::UnknownArchitecture`] when no arm recognises the config, and
/// [`FactoryError::Load`] when the family was recognised and the checkpoint
/// still would not load.
pub fn load_model(
    backend: &Arc<dyn Backend>,
    path: &Path,
    config: &HfConfig,
    args: &ServeArgs,
) -> Result<Loaded, FactoryError> {
    let opts = engine_options(args)?;
    // Resolved once, from the same table every other caller reads, rather than
    // written out again per arm. Two copies of "what this family emits" drift,
    // and the copy that drifts is the one no running server reads: an arm that
    // named Qwen 3's schemes for a Qwen 3.5 checkpoint passed every test of
    // `for_config` and still served the wrong parser.
    //
    // Resolving it before the arms rather than inside them is safe because
    // every reader below recognises a family from the document alone and
    // consults the container only for geometry afterwards, which is exactly
    // what `for_config` does. A document no reader recognises fails here with
    // the same error the fallthrough at the end would have produced.
    let defaults = FamilyDefaults::for_config(config)?;
    refuse_kv_dtype(args, config)?;

    // Before any arm, because this is the one speculative question that can be
    // answered from the document alone - and because the answer is a refusal
    // that must land before several gigabytes of weights do.
    //
    // `ServeArgs::validate` deliberately no longer asks it: whether a depth
    // needs an explicit `--draft-model` depends on the family, and pushing
    // that to the arms is what made the dense family's sidecar default
    // reachable. What the arms cannot be trusted with is the *prior* question
    // - whether the family speculates at all - because an arm that does not
    // read `args.spec_k` answers it by silence. Gemma 4's did, and served
    // with no speculation and no error.
    if args.spec_k > 0 && !defaults.speculates {
        return Err(FactoryError::Load(CoreError::InvalidValue {
            key: "--spec-k",
            value: format!(
                "{} was asked for, and this build binds no draft head for the {:?} family - \
                 use --spec-k 0 to decode one token a step",
                args.spec_k,
                config.model_type()
            ),
        }));
    }

    // One arm per family crate, in the order they are tried. `UnknownArch` is
    // the only failure that means "not mine"; every other failure from a reader
    // means the family matched and the document is wrong, and passing that
    // through as "unknown architecture" would hide a real config error behind a
    // misleading one.
    match Qwen35Config::from_hf(config) {
        Ok(family) => {
            // A block drafter is a directory of its own, not a sidecar head.
            if let Some(dir) = block_drafter_dir(args) {
                return load_qwen35_block(backend, path, &family, args, dir, defaults);
            }
            // Resolved before the load and refused before a byte moves: a head
            // that cannot be found is an operator's mistake, and finding it out
            // after several gigabytes of weights is finding it out too late.
            let head = draft_manifest(args).map_err(FactoryError::Load)?;
            let model = Qwen35Model::load_with_draft(
                backend,
                path,
                &family,
                dense_options(args)?,
                head.as_ref(),
            )
            .map_err(FactoryError::Load)?;
            return Ok(Loaded {
                model: Box::new(model),
                open_tower: open_qwen35_tower,
                open_audio: no_audio_tower,
                placement: Qwen35Tower::PLACEMENT,
                defaults,
            });
        }
        Err(CoreError::UnknownArch { .. }) => {}
        Err(other) => return Err(FactoryError::Load(other)),
    }

    // The mixture-of-experts family, whose `--model` is a DIRECTORY rather than
    // a container. Its body is a function of its own because it is the one arm
    // that has to say what kind of path it was handed before it can read
    // anything; see [`load_qwen35moe`].
    if Qwen35MoeConfig::describes(config) {
        return load_qwen35moe(backend, path, args, defaults);
    }

    // The sparse family's directory containers - MLX shards, the reference
    // export, the published trellis export - carry their geometry in the
    // document rather than in a container's metadata, so they take an arm of
    // their own ahead of the single-file one.
    if Qwen4ExpConfig::describes(config) && path.is_dir() {
        return load_qwen4exp_dir(backend, path, config, args, opts, defaults);
    }

    // The sparse family's reader takes the checkpoint as well as the document,
    // because this family carries no geometry in a config document: the reader
    // recognises the family by name and then reads every dimension out of the
    // container's own metadata. See `Qwen4ExpConfig::from_hf`.
    match Qwen4ExpConfig::from_hf(config, path) {
        Ok(family) => {
            // This family publishes its head as a separate container and its
            // config names no `mtp_file`, so there is nothing to default to.
            // Refused by name here because `ServeArgs::validate` no longer
            // does it for every family: the dense arm has a default and this
            // one does not, and a silent fallback would be a server that
            // decodes one token a step while an operator believes it is
            // speculating.
            if args.spec_k > 0 && args.draft_model.is_none() {
                return Err(FactoryError::Load(CoreError::ConfigMissing {
                    key: format!(
                        "--draft-model: --spec-k {} was asked for and this family's config \
                         names no mtp_file to default to",
                        args.spec_k
                    ),
                }));
            }
            let model = Qwen4ExpModel::load_with(
                backend,
                path,
                args.draft_model.as_deref(),
                draft_vocab_of(args),
                &family,
                sparse_options(args, opts, backend)?,
            )
            .map_err(FactoryError::Load)?;
            // This family shares the Qwen template's markers AND its tower:
            // the checkpoint carries one under the same `vision_tower` prefix,
            // at the same 27 blocks of width 1152, so it opens through the
            // dense hybrid's loader rather than through one of its own. The
            // seam the rows are spliced through is this family's stack's, and
            // `Qwen4ExpModel` answers `media_mut` with itself.
            return Ok(Loaded {
                model: Box::new(model),
                open_tower: open_qwen35_tower,
                open_audio: no_audio_tower,
                placement: Qwen35Tower::PLACEMENT,
                defaults,
            });
        }
        Err(CoreError::UnknownArch { .. }) => {}
        Err(other) => return Err(FactoryError::Load(other)),
    }

    // Gemma 4 reads its geometry from the document's `text_config` when it has
    // one and from the container's own metadata when it does not, so its reader
    // takes the checkpoint too. The per-layer table is the one knob this family
    // has that no other does, and it is read here rather than folded into the
    // shared options for the reason the sparse family's depth is.
    match Gemma4Config::from_hf(config, path) {
        Ok(family) => {
            let ple = PleOptions {
                cache_bytes: args.ple_cache_mb.saturating_mul(1 << 20),
                resident: args.ple_resident,
            };
            let model =
                Gemma4Model::load(backend, path, &family, opts, ple).map_err(FactoryError::Load)?;
            // The placement is this family's own - a bracketed run of one
            // marker - so a text-only server over this checkpoint still renders
            // the prompt the model was trained on. The opener loads this
            // family's tower from the mmproj the operator points `--vision` at.
            return Ok(Loaded {
                model: Box::new(model),
                open_tower: open_gemma4_tower,
                open_audio: open_gemma4_audio,
                placement: MediaPlacement::GEMMA4,
                defaults,
            });
        }
        Err(CoreError::UnknownArch { .. }) => {}
        Err(other) => return Err(FactoryError::Load(other)),
    }

    load_last_arms(backend, path, config, opts, defaults)
}

/// Refuses `--kv-cache-dtype` other than f16 for every family but the sparse
/// one, whose stack is the only one that stores anything else.
///
/// Asked from the document alone, before any arm, for the reason the
/// speculative question in [`load_model`] is: every other stack refuses the
/// element too, but only once its weights are resident.
///
/// # Errors
///
/// [`FactoryError::Load`] carrying [`CoreError::InvalidValue`] naming the
/// flag.
fn refuse_kv_dtype(args: &ServeArgs, config: &HfConfig) -> Result<(), FactoryError> {
    if args.kv_cache_dtype == crate::cli::KvCacheDtype::F16 || Qwen4ExpConfig::describes(config) {
        return Ok(());
    }
    Err(FactoryError::Load(CoreError::InvalidValue {
        key: "--kv-cache-dtype",
        value: format!(
            "{:?} was asked for, and the {:?} family's cache is f16 only - use \
             --kv-cache-dtype f16",
            args.kv_cache_dtype,
            config.model_type()
        ),
    }))
}

/// The arms tried after the Gemma 4 one, and the refusal when none of them
/// recognises the document either.
///
/// # Errors
///
/// [`FactoryError::UnknownArchitecture`] when no arm recognises the config,
/// and [`FactoryError::Load`] when one did and the checkpoint would not load.
fn load_last_arms(
    backend: &Arc<dyn Backend>,
    path: &Path,
    config: &HfConfig,
    opts: EngineOptions,
    defaults: FamilyDefaults,
) -> Result<Loaded, FactoryError> {
    // The hybrid state-space family: a directory of MLX shards whose geometry
    // is its own `config.json`, read again here because the parsed document
    // carries no typed view of it.
    if NemotronHConfig::describes_type(config.model_type()) {
        return load_nemotron_h(backend, path, opts, defaults);
    }

    // The LFM2 hybrid reads every width out of its container and checks the
    // document's against them, so its reader takes the checkpoint too.
    if Lfm2Config::describes(config) {
        return load_lfm2(backend, path, config, opts, defaults);
    }

    Err(FactoryError::UnknownArchitecture {
        model_type: config.model_type().to_owned(),
    })
}

/// The hybrid state-space arm: a directory of MLX shards and its config.
///
/// # Errors
///
/// [`FactoryError::Load`] for whatever the config or the checkpoint refuses.
fn load_nemotron_h(
    backend: &Arc<dyn Backend>,
    path: &Path,
    opts: EngineOptions,
    defaults: FamilyDefaults,
) -> Result<Loaded, FactoryError> {
    let family = NemotronHConfig::from_dir(path).map_err(FactoryError::Load)?;
    let model = NemotronHModel::load(backend, path, &family, opts).map_err(FactoryError::Load)?;
    Ok(Loaded {
        model: Box::new(model),
        open_tower: open_no_tower,
        open_audio: no_audio_tower,
        // The template writes no media marker; the placement is only what a
        // renderer is configured with.
        placement: MediaPlacement::QWEN35,
        defaults,
    })
}

/// The LFM2 arm: a GGUF container whose geometry is its own metadata.
///
/// # Errors
///
/// [`FactoryError::Load`] for whatever the config or the checkpoint refuses.
fn load_lfm2(
    backend: &Arc<dyn Backend>,
    path: &Path,
    config: &HfConfig,
    opts: EngineOptions,
    defaults: FamilyDefaults,
) -> Result<Loaded, FactoryError> {
    let family = Lfm2Config::from_hf(config, path).map_err(FactoryError::Load)?;
    let model = Lfm2Model::load(backend, path, &family, opts).map_err(FactoryError::Load)?;
    Ok(Loaded {
        model: Box::new(model),
        open_tower: open_lfm2_tower,
        open_audio: no_audio_tower,
        placement: Lfm2Tower::PLACEMENT,
        defaults,
    })
}

/// The `--draft-model` directory when it holds a block drafter: the target
/// then loads plain, at the asked depth, and the drafter binds beside it.
fn block_drafter_dir(args: &ServeArgs) -> Option<&Path> {
    args.draft_model
        .as_deref()
        .filter(|dir| shiso_models::qwen35::block_draft::is_block_drafter(dir))
}

/// The dense arm with a block drafter beside it: the target at the asked depth,
/// the drafter in `dir` bound to it.
///
/// # Errors
///
/// [`FactoryError::Load`] for whatever the target or the drafter refuses.
fn load_qwen35_block(
    backend: &Arc<dyn Backend>,
    path: &Path,
    family: &Qwen35Config,
    args: &ServeArgs,
    dir: &Path,
    defaults: FamilyDefaults,
) -> Result<Loaded, FactoryError> {
    let mut model = Qwen35Model::load(backend, path, family, dense_options(args)?)
        .map_err(FactoryError::Load)?;
    model
        .bind_block_drafter(backend, dir)
        .map_err(FactoryError::Load)?;
    Ok(Loaded {
        model: Box::new(model),
        open_tower: open_qwen35_tower,
        open_audio: no_audio_tower,
        placement: Qwen35Tower::PLACEMENT,
        defaults,
    })
}

/// The mixture-of-experts arm: a checkpoint DIRECTORY, read as two documents.
///
/// Three safetensors shards, a `config.json` and the `hf_quant_config.json`
/// that says how each of its 31,333 tensors is stored. Every other arm of
/// [`load_model`] is handed a container, which is why the reader here takes the
/// directory rather than the document [`load_model`] already parsed: the
/// quantization document is the second half of this family's geometry, it lives
/// beside the weights, and it has no flag of its own.
///
/// # Why the two path checks, and why `draft_manifest` keeps its opposite one
///
/// [`FamilyDefaults::for_config`] has already recognised the family from the
/// document alone, so arriving here with a path that is not a directory is an
/// operator pointing `--model` at the wrong thing. It is refused here by name,
/// as `exists` and then `is_dir` rather than as a bare `is_dir`, so that the
/// message distinguishes "nothing is there" from "a file where a directory
/// belongs" - two different mistakes with two different fixes, and `from_dir`
/// would report both of them as a missing `config.json` one layer down.
///
/// [`draft_manifest`] keeps its own `is_file` check unchanged, and the two are
/// not in tension: a draft head IS a container, so a directory there is the
/// same mistake the other way round. The rule is that each check asks for the
/// kind of thing its own consumer opens.
///
/// # Errors
///
/// [`FactoryError::Load`] naming the path when it is absent or is a file, and
/// whatever [`Qwen35MoeConfig::from_dir`] and [`Qwen35MoeModel::load`] report.
fn load_qwen35moe(
    backend: &Arc<dyn Backend>,
    path: &Path,
    args: &ServeArgs,
    defaults: FamilyDefaults,
) -> Result<Loaded, FactoryError> {
    if !path.exists() {
        return Err(FactoryError::Load(CoreError::ConfigMissing {
            key: format!("--model {}: nothing is there", path.display()),
        }));
    }
    // Two containers, and the path says which. A DIRECTORY is the `modelopt`
    // safetensors export - shards, `config.json` and `hf_quant_config.json`
    // together - and a FILE is a GGUF, which carries every width in its own
    // metadata and needs no document beside it for geometry.
    //
    // The config document is still read, and still decides the family: it is
    // what routed the request to this arm and what `FamilyDefaults` reads the
    // chat template and the parsers from. Only the GEOMETRY comes from the
    // container. That is the split the sparse family already makes, and this
    // arm's header says why it is the one that has to look at the path.
    let model: Box<dyn Checkpointed> = if path.is_dir() {
        // The `modelopt` export's own head is BF16 and this build has no BF16
        // matvec, so there is nothing to bind. Refused by name rather than
        // served unspeculated, which is the failure `FamilyDefaults::QWEN35MOE`
        // used to prevent by refusing the whole family.
        if args.spec_k > 0 {
            return Err(FactoryError::Load(CoreError::InvalidValue {
                key: "--spec-k",
                value: format!(
                    "{} is the safetensors export, whose own MTP head is BF16 and which this \
                     build binds no matvec for - serve the GGUF with --draft-model to \
                     speculate, or pass --spec-k 0",
                    path.display()
                ),
            }));
        }
        let family = Qwen35MoeConfig::from_dir(path).map_err(FactoryError::Load)?;
        Box::new(
            Qwen35MoeModel::load(backend, path, &family, dense_options(args)?)
                .map_err(FactoryError::Load)?,
        )
    } else {
        // The head is a DIRECTORY of its own for this family, which is how it
        // is published, so `--draft-model` names one rather than a container.
        // It is ignored unless `--spec-k` asks for a depth, and the loader
        // refuses a head whose widths are not this target's.
        //
        // A depth with no head named is refused here rather than served
        // silently: this container ships no head beside it and there is no
        // `mtp_file` in its metadata to default to, so there is nothing to fall
        // back to and a fallback would be a server that decodes one token a
        // step while an operator believes it is speculating.
        if args.spec_k > 0 && args.draft_model.is_none() {
            return Err(FactoryError::Load(CoreError::ConfigMissing {
                key: format!(
                    "--draft-model: --spec-k {} was asked for and a GGUF of this family \
                     carries no head beside it to default to",
                    args.spec_k
                ),
            }));
        }
        Box::new(
            Qwen35MoeModel::load_gguf_with_draft(
                backend,
                path,
                dense_options(args)?,
                args.draft_model.as_deref(),
            )
            .map_err(FactoryError::Load)?,
        )
    };
    // The tower is the dense hybrid's, unchanged - 27 blocks of width 1152
    // projecting to 2048 - and this checkpoint writes it under the same prefix,
    // interleaved through its own shards rather than in a shard of its own.
    // That is a container arm inside that family's loader and nothing here, so
    // `--vision` names this same directory and the dense opener reads it. The
    // placement is the same family's too, because it is the same chat template.
    Ok(Loaded {
        model,
        open_tower: open_qwen35_tower,
        open_audio: no_audio_tower,
        placement: Qwen35Tower::PLACEMENT,
        defaults,
    })
}

/// The engine settings every family shares.
///
/// `--spec-k` is absent here and read by the two arms that implement
/// speculation, because it is what sizes the per-row rollback records on every
/// session: a family with no draft head that took this field would allocate a
/// state matrix per drafted row per recurrent slot for records nothing writes.
///
/// The pool is `--kv-cache-tokens` when given, and a share of the device the
/// stack resolves once it is resident when `--gpu-memory-utilization` is; the
/// two are exclusive on the command line.
///
/// # Errors
///
/// [`FactoryError::Load`] carrying [`CoreError::InvalidValue`] for a pool or a
/// share [`ServeArgs::validate`] would have refused.
fn engine_options(args: &ServeArgs) -> Result<EngineOptions, FactoryError> {
    let mut opts = EngineOptions::default();
    opts.ctx = args.ctx;
    let pages = args.kv_pool_pages().map_err(|error| {
        FactoryError::Load(CoreError::InvalidValue {
            key: "kv_cache_tokens",
            value: error.to_string(),
        })
    })?;
    if let Some(pages) = pages {
        opts.kv_pool_pages = pages;
    }
    opts.kv_ckpt_slots = args.kv_checkpoints;
    // A resume always lands on a checkpoint, so without one no request reuses
    // a prefix. Legitimate - an operator measuring cold prefill asks for it -
    // but not something to find out from the throughput.
    if opts.kv_ckpt_slots == 0 {
        tracing::warn!("--kv-checkpoints 0: no prefix is reused, every request prefills from zero");
    }
    opts.kv_fit = args.kv_fit().map_err(FactoryError::Load)?;
    opts.kv_check = args.kv_check().map_err(FactoryError::Load)?;
    opts.kv_elem = args.kv_cache_dtype.into();
    Ok(opts)
}

/// The dense family's settings: the shared ones, plus the depth it drafts at.
///
/// # Errors
///
/// As [`engine_options`].
fn dense_options(args: &ServeArgs) -> Result<EngineOptions, FactoryError> {
    let mut opts = engine_options(args)?;
    opts.spec = SpecDepth::new(args.spec_k);
    Ok(opts)
}

/// The draft head the command line asks the dense arm for, if any.
///
/// `None` when no depth was asked for, which is every build until an operator
/// says otherwise. `--draft-model` names the head's container directly;
/// without it the sidecar's own `mtp_file` is the default, which is where this
/// family has always published it and means the flag is only needed to
/// override.
///
/// # Why this refuses rather than falling back
///
/// A server started with `--spec-k 3` and no head would answer correctly and
/// slowly, and the operator would have no way to see which of the two they
/// got. So a depth with no head is an error, and the error names the path it
/// looked at.
///
/// # Errors
///
/// [`CoreError::ConfigMissing`] naming the head when the depth was asked for
/// and no head can be found, and whatever the sidecar's document reports when
/// it cannot be read.
pub fn draft_manifest(args: &ServeArgs) -> Result<Option<MtpManifest>, CoreError> {
    if !SpecDepth::new(args.spec_k).is_speculative() {
        return Ok(None);
    }
    let sidecar = args.config_path();
    let sidecar = sidecar.parent().unwrap_or(Path::new("."));
    if let Some(named) = args.draft_model.as_deref() {
        if !named.is_file() {
            return Err(CoreError::ConfigMissing {
                key: format!(
                    "--draft-model names {}, which is not a file",
                    named.display()
                ),
            });
        }
        // The codec is the sidecar's statement about its own head, and an
        // operator pointing at a copy of that head elsewhere has not changed
        // it. A sidecar that states nothing gets the family's default, which
        // is the only codec this family has ever shipped a head in.
        let quant = HfConfig::from_file(&args.config_path())
            .ok()
            .and_then(|document| mtp::read_manifest_from(sidecar, &document).ok().flatten())
            .map_or_else(AffineQuant::family_default, |named| named.quant);
        return Ok(Some(MtpManifest {
            file: named.to_path_buf(),
            layers: 1,
            quant,
        }));
    }
    let document = HfConfig::from_file(&args.config_path())?;
    mtp::read_manifest_from(sidecar, &document)?.map_or_else(
        || {
            Err(CoreError::ConfigMissing {
                key: format!(
                    "a draft head: --spec-k {} was asked for, {} names no mtp_file, and \
                     --draft-model was not given",
                    args.spec_k,
                    args.config_path().display()
                ),
            })
        },
        |found| Ok(Some(found)),
    )
}

/// The sparse family's settings: the shared ones, the speculative depth that
/// only this family reads, and the two facts only the device can answer.
///
/// The entry table and the cooperative grid ceiling are *measured*, not
/// configured. Both defaults are the conservative answers - an empty table,
/// which refuses a template dispatch by name rather than guessing one, and a
/// single-block grid, which is correct and slow - and both are what a backend
/// with no compiled kernels or no cooperative launch still gets back here,
/// because each probe reports its own absence rather than inventing a number.
///
/// Order matters: the ceiling is read off the trellis entry points, so it can
/// only be asked once their names are known.
///
/// `tiled_attention`, `tensor_core` and the chunked recurrence are on exactly
/// when the device has kernels for them (see [`capture::optional_paths`]): a
/// backend whose kernel set leaves one out - the Metal port has neither the
/// matrix-unit nor the large-shared-memory layouts - gets the always-correct
/// fallback rather than a launch of an entry that does not exist.
///
/// # Errors
///
/// [`FactoryError::PrefillTooWide`] for a prefill width past the widest pass
/// the stack can be built with.
/// The draft vocabulary a load narrows the draft head with, said once in the
/// log: nothing else shows the narrowing except the speed.
fn draft_vocab_of(args: &ServeArgs) -> Option<&Path> {
    let path = args.draft_vocab.as_deref();
    if let Some(path) = path {
        tracing::info!(path = %path.display(), "narrowed the draft head's vocabulary");
    }
    path
}

fn sparse_options(
    args: &ServeArgs,
    engine: EngineOptions,
    backend: &Arc<dyn Backend>,
) -> Result<SparseOptions, FactoryError> {
    // Before the weights: a device without the cache element's kernels would
    // otherwise be found out only once the stack is built over them.
    capture::check_kv_elem(backend.as_ref(), engine.kv_elem).map_err(FactoryError::Load)?;
    let prefill_activations = args.prefill_activations.into();
    capture::check_prefill_activations(backend.as_ref(), prefill_activations)
        .map_err(FactoryError::Load)?;
    let entries = capture::capture(backend.as_ref());
    let co_resident_blocks = capture::co_resident_blocks(backend.as_ref(), entries);
    let paths = capture::optional_paths(backend.as_ref());
    tracing::info!(
        entries = entries.len(),
        co_resident_blocks,
        "device facts the sparse stack was built with",
    );
    // The scratch is allocated at the widest pass the scheduler will ask
    // for, which is what lets the prefill flags trade speed for pool. The
    // widths were validated at boot, so the fallback is never taken.
    let widest = args
        .prefill_widths()
        .map_or(shiso_engine::sparse::MAX_PREFILL_CHUNK as usize, |widths| {
            widths.solo_chunk.max(widths.mixed_quantum)
        });
    let prefill_rows = sparse_prefill_rows(widest)?;
    // The lookup proposes past the head's chain, so the depth the records are
    // sized for is the deeper of the two; the head still chains its own.
    let depth = SpecDepth::new(args.spec_k.max(args.spec_lookup));
    Ok(SparseOptions {
        engine,
        spec: depth,
        head_drafts: args.spec_k,
        pack_halves: !args.no_pack_halves,
        lookup_drafts: if args.spec_k > 0 { args.spec_lookup } else { 0 },
        graphs: shiso_engine::sparse::island::GraphPolicy {
            spec_depth: depth.get(),
            capture_verify: args.capture_verify,
            capture_batch: args.capture_batch,
        },
        entries,
        co_resident_blocks,
        prefill_rows,
        tensor_core: paths.tensor_core,
        tiled_attention: paths.tiled_attention,
        prompt_matmul: paths.prompt_matmul,
        prefill_activations,
        route_sort: paths.route_sort,
        chunk: shiso_engine::sparse::recurrent::ChunkPolicy {
            available: paths.chunked_recurrence,
            fused: paths.fused_recurrence,
            ..shiso_engine::sparse::recurrent::ChunkPolicy::new()
        },
        ..SparseOptions::default()
    })
}

/// The rows one sparse pass is built for when the scheduler's widest prefill
/// is `widest`.
///
/// # Errors
///
/// [`FactoryError::PrefillTooWide`] when that is narrower than `widest`, which
/// only a width past the stack's ceiling makes it.
fn sparse_prefill_rows(widest: usize) -> Result<u32, FactoryError> {
    let rows = SparseOptions::prefill_rows_for(widest);
    if widest > usize::try_from(rows).unwrap_or(usize::MAX) {
        return Err(FactoryError::PrefillTooWide { widest, rows });
    }
    Ok(rows)
}

/// Whether the checkpoint at `path` reads `--ngram`: only the sparse family's
/// trellis export keeps its n-gram table in a file of its own.
#[must_use]
pub fn takes_ngram(config: &HfConfig, path: &Path) -> bool {
    Qwen4ExpConfig::describes(config)
        && path.is_dir()
        && shiso_models::qwen4exp::exl3::is_exl3_dir(path)
}

/// The sparse family's directory: the trellis export when its shard index
/// names one, the MLX or reference export otherwise.
fn load_qwen4exp_dir(
    backend: &Arc<dyn Backend>,
    path: &Path,
    config: &HfConfig,
    args: &ServeArgs,
    opts: EngineOptions,
    defaults: FamilyDefaults,
) -> Result<Loaded, FactoryError> {
    if shiso_models::qwen4exp::exl3::is_exl3_dir(path) {
        load_qwen4exp_exl3(backend, path, config, args, opts, defaults)
    } else if shiso_models::qwen4exp::modelopt_mixed::is_export(path) {
        // Asked before both arms below, whose tests this directory may also
        // pass: it carries the reference export's names, and only its
        // `config.json` says how each matrix is stored.
        load_qwen4exp_streamed(backend, path, config, args, opts, defaults, true)
    } else if shiso_models::qwen4exp::nvfp4::is_nvfp4_export(path) {
        // Before the MLX arm, whose reference-export test this directory also
        // passes: the two share every name and differ in the experts' codec.
        load_qwen4exp_streamed(backend, path, config, args, opts, defaults, false)
    } else {
        load_qwen4exp_mlx(backend, path, config, args, opts, defaults)
    }
}

/// NVIDIA's two `modelopt` exports of the sparse family, both with NVFP4
/// routed experts, always streamed: the first (`mixed: false`) with
/// brain-float dense weights, the mixed-precision one (`mixed: true`) with
/// MXFP8 projections, CSF-scaled experts and an NVFP4 n-gram table and draft
/// head.
fn load_qwen4exp_streamed(
    backend: &Arc<dyn Backend>,
    path: &Path,
    config: &HfConfig,
    args: &ServeArgs,
    opts: EngineOptions,
    defaults: FamilyDefaults,
    mixed: bool,
) -> Result<Loaded, FactoryError> {
    let reserve = args.expert_reserve_gb.saturating_mul(1 << 30);
    // The head is in the directory; a second one named beside it would be a
    // head nobody can say which of the two is serving.
    if args.draft_model.is_some() {
        return Err(FactoryError::Load(CoreError::InvalidValue {
            key: "--draft-model",
            value: format!(
                "{} carries its own draft head under mtp.*; drop --draft-model and set --spec-k",
                path.display()
            ),
        }));
    }
    let slots = match args.expert_residency {
        crate::cli::ExpertResidencyChoice::Resident => {
            Some(shiso_models::qwen4exp::EXPERTS * shiso_models::qwen4exp::N_LAYERS)
        }
        _ if args.expert_slots > 0 => Some(args.expert_slots),
        _ => None,
    };
    let load = if mixed {
        Qwen4ExpModel::load_modelopt_mixed
    } else {
        Qwen4ExpModel::load_nvfp4
    };
    let model = load(
        backend,
        path,
        config,
        sparse_options(args, opts, backend)?,
        &shiso_models::qwen4exp::StreamedLoad {
            slots,
            reserve,
            draft: args.spec_k > 0,
            counts: draft_vocab_of(args),
        },
    )
    .map_err(FactoryError::Load)?;
    Ok(Loaded {
        model: Box::new(model),
        open_tower: open_qwen35_tower,
        open_audio: no_audio_tower,
        placement: Qwen35Tower::PLACEMENT,
        defaults,
    })
}

/// The sparse family's trellis export, as published: a directory that carries
/// its own draft head, loaded when speculation is asked for.
fn load_qwen4exp_exl3(
    backend: &Arc<dyn Backend>,
    path: &Path,
    config: &HfConfig,
    args: &ServeArgs,
    opts: EngineOptions,
    defaults: FamilyDefaults,
) -> Result<Loaded, FactoryError> {
    // The head is in the directory; a second one named beside it would be a
    // head nobody can say which of the two is serving.
    if args.draft_model.is_some() {
        return Err(FactoryError::Load(CoreError::InvalidValue {
            key: "--draft-model",
            value: format!(
                "{} carries its own draft head under mtp.*; drop --draft-model and set --spec-k",
                path.display()
            ),
        }));
    }
    let ngram = args.ngram.as_deref().ok_or_else(|| {
        FactoryError::Load(CoreError::ConfigMissing {
            key: format!(
                "--ngram: {} keeps its n-gram table in a file of its own; name it",
                path.display()
            ),
        })
    })?;
    let model = Qwen4ExpModel::load_exl3(
        backend,
        path,
        ngram,
        config,
        args.spec_k > 0,
        draft_vocab_of(args),
        sparse_options(args, opts, backend)?,
    )
    .map_err(FactoryError::Load)?;
    Ok(Loaded {
        model: Box::new(model),
        open_tower: open_qwen35_tower,
        open_audio: no_audio_tower,
        placement: Qwen35Tower::PLACEMENT,
        defaults,
    })
}

/// The sparse family's MLX directory: the geometry from its `config.json`, the
/// routed banks resident or streamed as `--expert-residency` says.
fn load_qwen4exp_mlx(
    backend: &Arc<dyn Backend>,
    path: &Path,
    config: &HfConfig,
    args: &ServeArgs,
    opts: EngineOptions,
    defaults: FamilyDefaults,
) -> Result<Loaded, FactoryError> {
    use shiso_models::qwen4exp::mlx::ExpertResidency;
    if args.spec_k > 0 {
        return Err(FactoryError::Load(CoreError::Unsupported {
            what: "speculation on the MLX container, which carries no draft head",
            backend: "qwen4exp",
        }));
    }
    let residency = match args.expert_residency {
        crate::cli::ExpertResidencyChoice::Auto if args.expert_slots == 0 => None,
        crate::cli::ExpertResidencyChoice::Resident => Some(ExpertResidency::Resident),
        crate::cli::ExpertResidencyChoice::Auto | crate::cli::ExpertResidencyChoice::Stream => {
            let slots = if args.expert_slots == 0 {
                shiso_models::qwen4exp::EXPERTS * shiso_models::qwen4exp::N_LAYERS
            } else {
                args.expert_slots
            };
            Some(ExpertResidency::Streamed { slots })
        }
    };
    let reserve = args.expert_reserve_gb.saturating_mul(1 << 30);
    if shiso_models::qwen4exp::mlx::MlxCheckpoint::is_reference_export(path) {
        // The reference export streams its FP8 experts whatever the device;
        // `resident` asks for a pool that holds every one of them.
        let slots = match args.expert_residency {
            crate::cli::ExpertResidencyChoice::Resident => {
                Some(shiso_models::qwen4exp::EXPERTS * shiso_models::qwen4exp::N_LAYERS)
            }
            _ if args.expert_slots > 0 => Some(args.expert_slots),
            _ => None,
        };
        let model = Qwen4ExpModel::load_reference(
            backend,
            path,
            config,
            sparse_options(args, opts, backend)?,
            slots,
            reserve,
        )
        .map_err(FactoryError::Load)?;
        return Ok(Loaded {
            model: Box::new(model),
            open_tower: open_qwen35_tower,
            open_audio: no_audio_tower,
            placement: Qwen35Tower::PLACEMENT,
            defaults,
        });
    }
    let model = Qwen4ExpModel::load_mlx(
        backend,
        path,
        config,
        sparse_options(args, opts, backend)?,
        residency,
        reserve,
    )
    .map_err(FactoryError::Load)?;
    Ok(Loaded {
        model: Box::new(model),
        open_tower: open_qwen35_tower,
        open_audio: no_audio_tower,
        placement: Qwen35Tower::PLACEMENT,
        defaults,
    })
}

/// Whether this build recognises a config document at all.
///
/// Separated from the load so that a caller can refuse an unservable checkpoint
/// before paying for a device, and so that the recognition rule can be tested
/// without a backend or a weights file.
#[must_use]
pub fn recognises(config: &HfConfig) -> bool {
    !matches!(
        Qwen35Config::from_hf(config),
        Err(CoreError::UnknownArch { .. })
    ) || Qwen35MoeConfig::describes(config)
        || Qwen4ExpConfig::describes(config)
        || Gemma4Config::describes(config)
        || Lfm2Config::describes(config)
}

#[cfg(test)]
mod tests {
    use super::{FactoryError, sparse_prefill_rows};

    #[test]
    fn a_prefill_width_past_the_widest_sparse_pass_is_refused_at_boot() {
        let ceiling = shiso_engine::sparse::MAX_PREFILL_CHUNK;
        let ceiling_width = usize::try_from(ceiling).expect("a host width");
        assert_eq!(sparse_prefill_rows(4_096).expect("fits"), 4_096);
        assert_eq!(sparse_prefill_rows(3_000).expect("fits"), 3_072);
        assert_eq!(sparse_prefill_rows(ceiling_width).expect("fits"), ceiling);
        let error = sparse_prefill_rows(ceiling_width + 512).expect_err("one pass cannot hold it");
        assert!(
            matches!(error, FactoryError::PrefillTooWide { widest, rows }
                if widest == ceiling_width + 512 && rows == ceiling),
            "{error:?}"
        );
        assert!(
            error.to_string().contains("--solo-prefill-chunk"),
            "{error}"
        );
    }
}
