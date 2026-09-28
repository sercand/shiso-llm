//! What checkpoint this is, and the numbers it runs with.
//!
//! One architecture ships in three containers: an HF
//! directory (bf16, one or two safetensors shards), an MLX directory (the
//! decoder's linears 8-bit affine, the tower bf16), and a GGUF pair - a text
//! file holding the decoder and an mmproj holding the audio tower. The HF and
//! MLX configs state the geometry of both halves; the GGUF pair states it in
//! two key sets (`qwen3vl.*` in the text file, `clip.audio.*` in the mmproj),
//! and the mmproj omits the window sizes, the frontend and the token ids
//! Those are constants here, each saying where its value comes from, and an
//! HF config that states a different value is refused rather than
//! half-honoured.
//!
//! The text GGUF does not say which mmproj belongs to it. `--asr-mmproj`
//! names it; without the flag the one `mmproj-*.gguf` beside the text file
//! whose quantisation suffix matches is taken, and anything else - none, or
//! several - is refused naming the candidates, because a tower from another
//! quantisation or another checkpoint loads without complaint and transcribes
//! wrong.

use std::path::{Path, PathBuf};

use serde_json::Value;
use shiso_audio::{
    AudioError, Filterbank, Framing, LogScale, MelScale, MelSpec, PieceVocab, Preemphasis,
    Spectrum, Window,
};
use shiso_modelfile::{GgufFile, GgufMetadata};

use super::names::{Names, Stem};
use crate::audio_ops::store::TensorStore;

/// Mel frames per conv-stem chunk, `n_window * 2`: the tower cuts its input
/// into 1 s chunks before the conv stem.
pub const CHUNK_FRAMES: usize = 100;
/// Tower tokens one full chunk yields: three stride-2 convs over 100 frames
/// (`100 -> 50 -> 25 -> 13`). This is also the number of position-table rows
/// ever used, because the positions restart in every chunk.
pub const TOKENS_PER_CHUNK: usize = 13;
/// Tokens per bidirectional attention window in the tower,
/// `13 * (n_window_infer / (n_window * 2))`: eight whole chunks, 8 s of audio.
pub const WINDOW_TOKENS: usize = 104;
/// `n_window` the constants above are derived from, as the HF config states it.
const N_WINDOW: u64 = 50;
/// `n_window_infer` the constants above are derived from, as the HF config
/// states it.
const N_WINDOW_INFER: u64 = 800;
/// Mel bins the frontend computes.
pub const MEL_BINS: usize = 128;
/// The tower's layer-norm epsilon: the framework default, which the HF config
/// does not state and the mmproj states as this.
const AUDIO_NORM_EPS: f32 = 1e-5;

/// `<|endoftext|>`, the pad and a stop token (the tokenizer config's ids,
/// as every id below).
pub const ENDOFTEXT: u32 = 151_643;
/// `<|im_start|>`.
pub const IM_START: u32 = 151_644;
/// `<|im_end|>`, the end of a turn and a stop token.
pub const IM_END: u32 = 151_645;
/// `<|audio_start|>`.
pub const AUDIO_START: u32 = 151_669;
/// `<|audio_end|>`.
pub const AUDIO_END: u32 = 151_670;
/// `<|audio_pad|>`: each is replaced by one tower row, in order.
pub const AUDIO_PAD: u32 = 151_676;
/// `<asr_text>`, an added token that is not special, so it survives decoding
/// and the transcript is split on it as text.
pub const ASR_TEXT: u32 = 151_704;
/// Decoding stops on either: the generation config's end ids.
pub const STOP_TOKENS: [u32; 2] = [IM_END, ENDOFTEXT];
/// The first id past the byte-level alphabet: everything from here up is an
/// added token whose text is used literally.
pub const FIRST_SPECIAL: u32 = ENDOFTEXT;

/// The languages the model can be told to transcribe as: an ISO 639 code and
/// the name the model writes after `language ` (the config's
/// `support_languages`, in its order). Filipino has no two-letter code of its
/// own, so it takes the three-letter one, as Cantonese does.
const LANGUAGES: [(&str, &str); 30] = [
    ("zh", "Chinese"),
    ("en", "English"),
    ("yue", "Cantonese"),
    ("ar", "Arabic"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ko", "Korean"),
    ("ru", "Russian"),
    ("th", "Thai"),
    ("vi", "Vietnamese"),
    ("ja", "Japanese"),
    ("tr", "Turkish"),
    ("hi", "Hindi"),
    ("ms", "Malay"),
    ("nl", "Dutch"),
    ("sv", "Swedish"),
    ("da", "Danish"),
    ("fi", "Finnish"),
    ("pl", "Polish"),
    ("cs", "Czech"),
    ("fil", "Filipino"),
    ("fa", "Persian"),
    ("el", "Greek"),
    ("ro", "Romanian"),
    ("hu", "Hungarian"),
    ("mk", "Macedonian"),
];

/// Every language the model supports, as `(ISO code, the model's name)`.
#[must_use]
pub fn languages() -> &'static [(&'static str, &'static str)] {
    &LANGUAGES
}

/// The model's name for the language `tag` selects, by its primary subtag and
/// ignoring case (`en`, `en-US` and `EN_gb` all name English); `None` for a
/// language the model does not support.
#[must_use]
pub fn language_name(tag: &str) -> Option<&'static str> {
    let primary = tag.split(['-', '_']).next()?;
    LANGUAGES
        .iter()
        .find(|(code, _)| code.eq_ignore_ascii_case(primary))
        .map(|(_, name)| *name)
}

/// The ISO code of the language the model names `name` (in any case), the
/// inverse of [`language_name`]; `None` for a name outside the model's list.
#[must_use]
pub fn language_code(name: &str) -> Option<&'static str> {
    LANGUAGES
        .iter()
        .find(|(_, n)| n.eq_ignore_ascii_case(name))
        .map(|(code, _)| *code)
}

/// Whether `id` is an added token the tokenizer marks special - the chat
/// markers, the end ids, the vision and audio placeholders, the TTS markers
/// and the blanks - or an id past the checkpoint's own table (the GGUF
/// vocabulary pads to 151936). Never text. The added tokens that are not
/// special (`<asr_text>`, `<think>`, the tool and FIM markers, `<non_speech>`)
/// are text the model may write, and are not in this set (the
/// `special` flags of `tokenizer_config.json`'s `added_tokens_decoder`,
/// identical in every export of both checkpoints).
#[must_use]
pub fn is_special(id: u32) -> bool {
    matches!(id, 151_643..=151_656 | 151_669..=151_674 | 151_676..=151_703) || id > ASR_TEXT
}

/// The log-mel frontend, as the reference feature extractor computes it: a
/// periodic Hann window of 400 over a 400-point FFT every 160 samples,
/// reflect-padded centre framing, the power spectrum through 128 Slaney mels from 0 to 8 kHz, and the whole-clip
/// `log10` floor and rescale.
#[must_use]
pub fn mel_spec() -> MelSpec {
    MelSpec {
        sample_rate: 16_000,
        n_fft: 400,
        win_length: 400,
        hop_length: 160,
        n_mels: MEL_BINS,
        window: Window::HannPeriodic,
        framing: Framing::CentredReflect,
        preemphasis: Preemphasis::None,
        spectrum: Spectrum::Power,
        filterbank: Filterbank::Computed {
            scale: MelScale::Slaney,
            slaney_norm: true,
            f_min: 0.0,
            f_max: 8_000.0,
        },
        log_scale: LogScale::Log10MaxFloor,
    }
}

/// Which container a checkpoint path is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flavour {
    /// The reference framework's safetensors, in bf16.
    Hf,
    /// The MLX export: the decoder's linears 8-bit affine in groups of 64,
    /// everything else bf16.
    Mlx,
    /// A text GGUF (architecture `qwen3vl`) with its mmproj (`clip`).
    Gguf,
}

/// A checkpoint path recognised as this family.
#[derive(Debug)]
pub struct Located {
    /// The container.
    pub flavour: Flavour,
    /// The directory the export lives in.
    pub dir: PathBuf,
    /// The files holding the decoder: the safetensors shards in order, or the
    /// one text GGUF. For a directory export they hold the tower too.
    pub weights: Vec<PathBuf>,
    /// The mmproj holding the tower, for a GGUF export.
    pub mmproj: Option<PathBuf>,
    /// Elements per affine group, for the MLX export.
    affine_group: Option<usize>,
    config: Option<Value>,
}

/// The opened tensor files of one export.
#[derive(Debug)]
pub struct Stores {
    text: TensorStore,
    mmproj: Option<TensorStore>,
}

impl Stores {
    /// The store holding the decoder.
    #[must_use]
    pub fn text(&self) -> &TensorStore {
        &self.text
    }

    /// The store holding the audio tower: the mmproj, or the same store as
    /// [`Stores::text`] for a directory export.
    #[must_use]
    pub fn tower(&self) -> &TensorStore {
        self.mmproj.as_ref().unwrap_or(&self.text)
    }

    /// The same stores with `name` treated as absent in each.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn hiding(self, name: &str) -> Self {
        Self {
            text: self.text.hiding(name),
            mmproj: self.mmproj.map(|m| m.hiding(name)),
        }
    }
}

/// The audio tower's geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioConfig {
    /// Mel bins in.
    pub mels: usize,
    /// Width of the transformer.
    pub d_model: usize,
    /// Transformer layers.
    pub layers: usize,
    /// Attention heads.
    pub heads: usize,
    /// Width of one head.
    pub head_dim: usize,
    /// Feed-forward width.
    pub ffn: usize,
    /// Channels of each conv in the stem.
    pub conv_channels: usize,
    /// Width of the output head, which must be the decoder's.
    pub output_dim: usize,
    /// Layer-norm epsilon.
    pub norm_eps: f32,
}

/// The decoder's geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct TextConfig {
    /// Residual width.
    pub hidden: usize,
    /// Decoder layers.
    pub layers: usize,
    /// Query heads.
    pub heads: usize,
    /// Key and value heads.
    pub kv_heads: usize,
    /// Width of one head.
    pub head_dim: usize,
    /// MLP width.
    pub ffn: usize,
    /// Rows of the (tied) embedding.
    pub vocab: usize,
    /// RMS-norm epsilon.
    pub rms_eps: f32,
    /// Rotary base.
    pub rope_theta: f32,
    /// Positions the decoder was trained for.
    pub max_positions: usize,
}

/// The numbers one checkpoint runs with.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3AsrConfig {
    /// The audio tower.
    pub audio: AudioConfig,
    /// The decoder.
    pub text: TextConfig,
}

fn checkpoint(what: impl Into<String>) -> AudioError {
    AudioError::Checkpoint { what: what.into() }
}

fn read_json(path: &Path) -> Result<Value, AudioError> {
    let text = std::fs::read_to_string(path).map_err(|source| AudioError::Io {
        what: "reading a checkpoint config",
        source,
    })?;
    serde_json::from_str(&text).map_err(|e| checkpoint(format!("{}: {e}", path.display())))
}

/// Recognises `path` (a text `.gguf`, a directory of one, or an export
/// directory) as this family. `mmproj` is `--asr-mmproj`: the tower file for a
/// text GGUF, overriding the search beside it.
///
/// # Errors
///
/// [`AudioError::Checkpoint`] naming what is broken in an export of this
/// family: an MLX quantisation other than 8-bit groups of 64, a text GGUF
/// whose mmproj is missing, ambiguous or not an audio tower, a directory
/// holding several text GGUFs, or `mmproj` given for a directory export.
// None: the path is not this family's (another family's, or no checkpoint at
// all), and the registry tries the next one.
pub fn locate(path: &Path, mmproj: Option<&Path>) -> Result<Option<Located>, AudioError> {
    if path.is_file() {
        if !is_text_gguf(path) {
            return Ok(None);
        }
        return located_gguf(path, mmproj).map(Some);
    }
    if !path.is_dir() {
        return Ok(None);
    }
    let config_path = path.join("config.json");
    if !config_path.exists() {
        let mut texts = gguf_files(path)?
            .into_iter()
            .filter(|p| is_text_gguf(p))
            .collect::<Vec<_>>();
        return match texts.len() {
            0 => Ok(None),
            1 => located_gguf(&texts.remove(0), mmproj).map(Some),
            _ => Err(checkpoint(format!(
                "{} holds several text GGUFs of this family ({}); name the file, not the directory",
                path.display(),
                names_of(&texts)
            ))),
        };
    }
    let config = read_json(&config_path)?;
    if config.get("model_type").and_then(Value::as_str) != Some("qwen3_asr") {
        return Ok(None);
    }
    if let Some(given) = mmproj {
        return Err(checkpoint(format!(
            "--asr-mmproj names {}, but {} is a directory export that carries its own audio tower",
            given.display(),
            path.display()
        )));
    }
    let (flavour, affine_group) = match config.get("quantization") {
        None => (Flavour::Hf, None),
        Some(q) => {
            let bits = q.get("bits").and_then(Value::as_u64);
            let group = q.get("group_size").and_then(Value::as_u64);
            let mode = q.get("mode").and_then(Value::as_str).unwrap_or("affine");
            if bits != Some(8) || group != Some(64) || mode != "affine" {
                return Err(checkpoint(format!(
                    "{}: the MLX export is quantised as bits {}, group_size {}, mode {mode}; \
                     only 8-bit affine groups of 64 are read",
                    config_path.display(),
                    bits.map_or_else(|| "(none)".into(), |b| b.to_string()),
                    group.map_or_else(|| "(none)".into(), |g| g.to_string()),
                )));
            }
            (Flavour::Mlx, Some(64))
        }
    };
    Ok(Some(Located {
        flavour,
        dir: path.to_path_buf(),
        weights: shards(path)?,
        mmproj: None,
        affine_group,
        config: Some(config),
    }))
}

/// Whether `path` is a GGUF whose architecture key is this family's text
/// model. A file that is not GGUF at all is simply not one.
fn is_text_gguf(path: &Path) -> bool {
    GgufFile::open(path).is_ok_and(|f| f.metadata().architecture() == Some("qwen3vl"))
}

/// The `.gguf` files directly inside `dir`, sorted by name.
fn gguf_files(dir: &Path) -> Result<Vec<PathBuf>, AudioError> {
    let entries = std::fs::read_dir(dir).map_err(|source| AudioError::Io {
        what: "listing a checkpoint directory",
        source,
    })?;
    let mut out = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| AudioError::Io {
            what: "listing a checkpoint directory",
            source,
        })?;
        let p = entry.path();
        if p.is_file()
            && p.extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}

fn file_name(p: &Path) -> String {
    p.file_name().map_or_else(
        || p.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

fn names_of(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|p| file_name(p))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The safetensors files of a directory export, in shard order: those its
/// index names, or the one `model.safetensors`.
fn shards(dir: &Path) -> Result<Vec<PathBuf>, AudioError> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let json = read_json(&index)?;
        let map = json
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| checkpoint(format!("{} has no weight_map", index.display())))?;
        let mut files: Vec<&str> = map.values().filter_map(Value::as_str).collect();
        files.sort_unstable();
        files.dedup();
        if files.is_empty() {
            return Err(checkpoint(format!("{} names no files", index.display())));
        }
        return Ok(files.into_iter().map(|f| dir.join(f)).collect());
    }
    let single = dir.join("model.safetensors");
    if !single.exists() {
        return Err(checkpoint(format!(
            "{} has neither model.safetensors nor model.safetensors.index.json",
            dir.display()
        )));
    }
    Ok(vec![single])
}

fn located_gguf(text: &Path, mmproj: Option<&Path>) -> Result<Located, AudioError> {
    let tower = match mmproj {
        Some(given) => given.to_path_buf(),
        None => find_mmproj(text)?,
    };
    check_mmproj(&tower)?;
    Ok(Located {
        flavour: Flavour::Gguf,
        dir: dir_of(text).to_path_buf(),
        weights: vec![text.to_path_buf()],
        mmproj: Some(tower),
        affine_group: None,
        config: None,
    })
}

/// The quantisation suffix of a GGUF file name: what follows its last `-`,
/// `Q8_0` for `Model-Q8_0.gguf`.
fn quant_suffix(p: &Path) -> Option<String> {
    let stem = p.file_stem()?.to_str()?;
    stem.rsplit_once('-').map(|(_, s)| s.to_owned())
}

/// The directory `path` sits in: its parent, or the working directory for a
/// bare file name, whose parent is the empty path (and `read_dir("")` fails).
fn dir_of(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

/// The one `mmproj-*.gguf` beside `text` with the same quantisation suffix.
fn find_mmproj(text: &Path) -> Result<PathBuf, AudioError> {
    let dir = dir_of(text);
    let candidates: Vec<PathBuf> = gguf_files(dir)?
        .into_iter()
        .filter(|p| file_name(p).starts_with("mmproj-"))
        .collect();
    let suffix = quant_suffix(text);
    let mut matching: Vec<PathBuf> = candidates
        .iter()
        .filter(|p| {
            matches!((&suffix, quant_suffix(p)), (Some(want), Some(got)) if want.eq_ignore_ascii_case(&got))
        })
        .cloned()
        .collect();
    let wanted = suffix.as_deref().map_or_else(
        || "the same quantisation".to_owned(),
        |s| format!("-{s}.gguf"),
    );
    match matching.len() {
        1 => Ok(matching.remove(0)),
        0 if candidates.is_empty() => Err(checkpoint(format!(
            "{} needs its audio tower, and no mmproj-*.gguf is beside it in {}; \
             name one with --asr-mmproj (SHISO_ASR_MMPROJ)",
            text.display(),
            dir.display()
        ))),
        0 => Err(checkpoint(format!(
            "{} needs its audio tower, and no mmproj beside it ends in {wanted} \
             (found {}); name one with --asr-mmproj (SHISO_ASR_MMPROJ)",
            text.display(),
            names_of(&candidates)
        ))),
        _ => Err(checkpoint(format!(
            "{} needs its audio tower, and several mmproj files beside it end in {wanted} \
             ({}); name one with --asr-mmproj (SHISO_ASR_MMPROJ)",
            text.display(),
            names_of(&matching)
        ))),
    }
}

/// Refuses a tower file that is not this family's audio mmproj, naming it.
fn check_mmproj(path: &Path) -> Result<(), AudioError> {
    let file = GgufFile::open(path).map_err(|e| {
        checkpoint(format!(
            "the mmproj {} does not open as GGUF: {e}",
            path.display()
        ))
    })?;
    let meta = file.metadata();
    let arch = meta.architecture();
    let audio = meta.get_bool("clip.has_audio_encoder") == Some(true);
    let projector = meta.get_str("clip.audio.projector_type");
    if arch != Some("clip") || !audio || projector != Some("qwen3a") {
        return Err(checkpoint(format!(
            "{} is not this family's audio mmproj (architecture {}, audio encoder {audio}, \
             projector {}); name the right one with --asr-mmproj (SHISO_ASR_MMPROJ)",
            path.display(),
            arch.unwrap_or("(none)"),
            projector.unwrap_or("(none)")
        )));
    }
    Ok(())
}

impl Located {
    /// Elements per affine group of an export that quantises its linears that
    /// way; `None` for an export that does not.
    #[must_use]
    pub fn affine_group(&self) -> Option<usize> {
        self.affine_group
    }

    /// Opens the tensor files.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] when one cannot be read.
    pub fn stores(&self) -> Result<Stores, AudioError> {
        match self.flavour {
            Flavour::Hf | Flavour::Mlx => Ok(Stores {
                text: TensorStore::open_shards(&self.weights)?,
                mmproj: None,
            }),
            Flavour::Gguf => {
                let text = self
                    .weights
                    .first()
                    .ok_or_else(|| checkpoint("a GGUF export with no text file"))?;
                let tower = self
                    .mmproj
                    .as_ref()
                    .ok_or_else(|| checkpoint("a GGUF export with no mmproj"))?;
                Ok(Stores {
                    text: TensorStore::open_gguf(text)?,
                    mmproj: Some(TensorStore::open_gguf(tower)?),
                })
            }
        }
    }

    fn json(&self, pointer: &str) -> Result<&Value, AudioError> {
        self.config
            .as_ref()
            .and_then(|c| c.pointer(pointer))
            .ok_or_else(|| {
                checkpoint(format!(
                    "{}/config.json has no {pointer}",
                    self.dir.display()
                ))
            })
    }

    fn json_u64(&self, pointer: &str) -> Result<u64, AudioError> {
        self.json(pointer)?
            .as_u64()
            .ok_or_else(|| checkpoint(format!("config.json {pointer} is not a count")))
    }

    fn json_usize(&self, pointer: &str) -> Result<usize, AudioError> {
        usize::try_from(self.json_u64(pointer)?)
            .map_err(|_| checkpoint(format!("config.json {pointer} does not fit")))
    }

    fn json_f32(&self, pointer: &str) -> Result<f32, AudioError> {
        self.json(pointer)?
            .as_f64()
            .map(narrow)
            .ok_or_else(|| checkpoint(format!("config.json {pointer} is not a number")))
    }

    fn text_name(&self) -> String {
        self.weights.first().map_or_else(
            || self.dir.display().to_string(),
            |p| p.display().to_string(),
        )
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the kernels read these constants as f32"
)]
fn narrow(v: f64) -> f32 {
    v as f32
}

fn meta_of(store: &TensorStore) -> Result<&GgufMetadata, AudioError> {
    store
        .gguf()
        .map(GgufFile::metadata)
        .ok_or_else(|| checkpoint("a GGUF export's store is not GGUF"))
}

fn gguf_usize(meta: &GgufMetadata, key: &str, file: &Path) -> Result<usize, AudioError> {
    meta.get_u32(key)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| checkpoint(format!("{} has no {key}", file.display())))
}

fn gguf_f32(meta: &GgufMetadata, key: &str, file: &Path) -> Result<f32, AudioError> {
    meta.get_f32(key)
        .ok_or_else(|| checkpoint(format!("{} has no {key}", file.display())))
}

impl Qwen3AsrConfig {
    /// Reads the numbers of the checkpoint at `located`.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] naming the missing key or tensor, a tower
    /// whose output width is not the decoder's (naming both files and
    /// `--asr-mmproj` for a GGUF pair), a window or token id that differs from
    /// the constants this family is written for, or a dimension the kernels
    /// are not written for.
    pub fn read(located: &Located, stores: &Stores) -> Result<Self, AudioError> {
        let config = match located.flavour {
            Flavour::Hf | Flavour::Mlx => Self::from_json(located)?,
            Flavour::Gguf => Self::from_gguf(located, stores)?,
        };
        config.validate()?;
        Ok(config)
    }

    fn from_json(l: &Located) -> Result<Self, AudioError> {
        let a = "/thinker_config/audio_config";
        let t = "/thinker_config/text_config";
        // The windows and token ids are constants of the kernels and the
        // prompt; an export that states others is a different model.
        for (key, want) in [
            (format!("{a}/n_window"), N_WINDOW),
            (format!("{a}/n_window_infer"), N_WINDOW_INFER),
            (
                "/thinker_config/audio_token_id".to_owned(),
                u64::from(AUDIO_PAD),
            ),
            (
                "/thinker_config/audio_start_token_id".to_owned(),
                u64::from(AUDIO_START),
            ),
            (
                "/thinker_config/audio_end_token_id".to_owned(),
                u64::from(AUDIO_END),
            ),
        ] {
            let got = l.json_u64(&key)?;
            if got != want {
                return Err(checkpoint(format!(
                    "{}/config.json {key} is {got}; this family is written for {want}",
                    l.dir.display()
                )));
            }
        }
        if l.json(&format!("{t}/tie_word_embeddings"))?.as_bool() != Some(true) {
            return Err(checkpoint(format!(
                "{}/config.json does not tie the output head to the embedding, \
                 and the loader reads the head from the embedding",
                l.dir.display()
            )));
        }
        let d_model = l.json_usize(&format!("{a}/d_model"))?;
        let heads = l.json_usize(&format!("{a}/encoder_attention_heads"))?;
        let audio = AudioConfig {
            mels: l.json_usize(&format!("{a}/num_mel_bins"))?,
            d_model,
            layers: l.json_usize(&format!("{a}/encoder_layers"))?,
            heads,
            head_dim: d_model.checked_div(heads).unwrap_or(0),
            ffn: l.json_usize(&format!("{a}/encoder_ffn_dim"))?,
            conv_channels: l.json_usize(&format!("{a}/downsample_hidden_size"))?,
            output_dim: l.json_usize(&format!("{a}/output_dim"))?,
            norm_eps: AUDIO_NORM_EPS,
        };
        let text = TextConfig {
            hidden: l.json_usize(&format!("{t}/hidden_size"))?,
            layers: l.json_usize(&format!("{t}/num_hidden_layers"))?,
            heads: l.json_usize(&format!("{t}/num_attention_heads"))?,
            kv_heads: l.json_usize(&format!("{t}/num_key_value_heads"))?,
            head_dim: l.json_usize(&format!("{t}/head_dim"))?,
            ffn: l.json_usize(&format!("{t}/intermediate_size"))?,
            vocab: l.json_usize(&format!("{t}/vocab_size"))?,
            rms_eps: l.json_f32(&format!("{t}/rms_norm_eps"))?,
            rope_theta: l.json_f32(&format!("{t}/rope_theta"))?,
            max_positions: l.json_usize(&format!("{t}/max_position_embeddings"))?,
        };
        if audio.output_dim != text.hidden {
            return Err(checkpoint(format!(
                "{}/config.json: the audio tower projects to {} but the decoder is {} wide",
                l.dir.display(),
                audio.output_dim,
                text.hidden
            )));
        }
        Ok(Self { audio, text })
    }

    fn from_gguf(l: &Located, stores: &Stores) -> Result<Self, AudioError> {
        let text_path = PathBuf::from(l.text_name());
        let tower_path = l.mmproj.clone().unwrap_or_default();
        let tm = meta_of(stores.text())?;
        let am = meta_of(stores.tower())?;
        let k = |key: &str| gguf_usize(tm, key, &text_path);
        let vocab = tm
            .get_array("tokenizer.ggml.tokens")
            .map(shiso_modelfile::MetaArray::len)
            .ok_or_else(|| {
                checkpoint(format!(
                    "{} has no tokenizer.ggml.tokens",
                    text_path.display()
                ))
            })?;
        let text = TextConfig {
            hidden: k("qwen3vl.embedding_length")?,
            layers: k("qwen3vl.block_count")?,
            heads: k("qwen3vl.attention.head_count")?,
            kv_heads: k("qwen3vl.attention.head_count_kv")?,
            head_dim: k("qwen3vl.attention.key_length")?,
            ffn: k("qwen3vl.feed_forward_length")?,
            vocab,
            rms_eps: gguf_f32(tm, "qwen3vl.attention.layer_norm_rms_epsilon", &text_path)?,
            rope_theta: gguf_f32(tm, "qwen3vl.rope.freq_base", &text_path)?,
            max_positions: k("qwen3vl.context_length")?,
        };
        if k("qwen3vl.attention.value_length")? != text.head_dim {
            return Err(checkpoint(format!(
                "{}: key and value heads differ in width",
                text_path.display()
            )));
        }
        let a = |key: &str| gguf_usize(am, key, &tower_path);
        let output_dim = a("clip.audio.projection_dim")?;
        // Checked before anything else is read from the tower: this is the
        // mismatch a wrongly paired mmproj shows first.
        if output_dim != text.hidden {
            return Err(checkpoint(format!(
                "the mmproj {} projects to {output_dim}, but the text model {} is {} wide; \
                 name the mmproj that belongs to it with --asr-mmproj (SHISO_ASR_MMPROJ)",
                tower_path.display(),
                text_path.display(),
                text.hidden
            )));
        }
        let d_model = a("clip.audio.embedding_length")?;
        let heads = a("clip.audio.attention.head_count")?;
        let mels = a("clip.audio.num_mel_bins")?;
        let layers = a("clip.audio.block_count")?;
        let ffn = a("clip.audio.feed_forward_length")?;
        let norm_eps = gguf_f32(am, "clip.audio.attention.layer_norm_epsilon", &tower_path)?;
        // The mmproj states no stem width; the first conv's rows are it.
        let conv = Names::new(Flavour::Gguf).weight(Stem::Conv(1));
        let conv_channels = stores
            .tower()
            .require(&conv)?
            .shape
            .first()
            .and_then(|d| usize::try_from(*d).ok())
            .unwrap_or(0);
        let audio = AudioConfig {
            mels,
            d_model,
            layers,
            heads,
            head_dim: d_model.checked_div(heads).unwrap_or(0),
            ffn,
            conv_channels,
            output_dim,
            norm_eps,
        };
        Ok(Self { audio, text })
    }

    fn validate(&self) -> Result<(), AudioError> {
        let fail = |what: &str| {
            Err(checkpoint(format!(
                "{what} is not something this family's kernels are written for"
            )))
        };
        let (a, t) = (&self.audio, &self.text);
        if a.mels != MEL_BINS {
            return fail("a mel count other than 128");
        }
        if a.heads == 0 || a.heads * a.head_dim != a.d_model {
            return fail("an audio width that is not whole heads");
        }
        if t.heads == 0 || t.kv_heads == 0 || !t.heads.is_multiple_of(t.kv_heads) {
            return fail("query heads that are not whole groups of key heads");
        }
        for head in [a.head_dim, t.head_dim] {
            if head > 128 || !head.is_multiple_of(2) {
                return fail("a head wider than 128 or odd");
            }
        }
        if a.layers == 0 || t.layers == 0 || a.conv_channels == 0 {
            return fail("an empty stack");
        }
        if usize::try_from(ASR_TEXT).map_or(true, |id| t.vocab <= id) {
            return fail("a vocabulary without the prompt's token ids");
        }
        Ok(())
    }
}

/// The checkpoint's byte-level vocabulary: the byte alphabet, then the added
/// tokens (`<|im_end|>`, `<|audio_pad|>`, `<asr_text>`, ...) as literal text.
///
/// # Errors
///
/// [`AudioError::Checkpoint`], [`AudioError::Io`] or [`AudioError::Vocab`] for
/// a vocabulary that is missing, not dense, or holds a piece the byte table
/// cannot decode.
pub fn read_vocab(located: &Located, stores: &Stores) -> Result<PieceVocab, AudioError> {
    match located.flavour {
        Flavour::Hf | Flavour::Mlx => {
            let vocab = read_json(&located.dir.join("vocab.json"))?;
            let map = vocab
                .as_object()
                .ok_or_else(|| checkpoint("vocab.json is not an object"))?;
            let mut by_id: Vec<Option<String>> = vec![None; map.len()];
            for (piece, id) in map {
                let slot = id
                    .as_u64()
                    .and_then(|i| usize::try_from(i).ok())
                    .and_then(|i| by_id.get_mut(i))
                    .ok_or_else(|| {
                        checkpoint(format!("vocab.json gives {piece:?} an id outside its size"))
                    })?;
                *slot = Some(piece.clone());
            }
            let tokens = by_id
                .into_iter()
                .enumerate()
                .map(|(id, p)| p.ok_or_else(|| checkpoint(format!("vocab.json has no id {id}"))))
                .collect::<Result<Vec<_>, _>>()?;
            let config = read_json(&located.dir.join("tokenizer_config.json"))?;
            let added = config
                .get("added_tokens_decoder")
                .and_then(Value::as_object)
                .ok_or_else(|| checkpoint("tokenizer_config.json has no added_tokens_decoder"))?;
            let mut specials = added
                .iter()
                .map(|(id, entry)| {
                    let id: u32 = id.parse().map_err(|_| {
                        checkpoint(format!("added token id {id:?} is not a number"))
                    })?;
                    let text = entry
                        .get("content")
                        .and_then(Value::as_str)
                        .ok_or_else(|| checkpoint(format!("added token {id} has no content")))?;
                    Ok((id, text.to_owned()))
                })
                .collect::<Result<Vec<_>, AudioError>>()?;
            specials.sort_unstable();
            PieceVocab::from_byte_level(tokens, &specials)
        }
        Flavour::Gguf => {
            let meta = meta_of(stores.text())?;
            if meta.get_str("tokenizer.ggml.model") != Some("gpt2") {
                return Err(checkpoint(format!(
                    "{}: the tokenizer is not byte-level BPE (tokenizer.ggml.model is not gpt2)",
                    located.text_name()
                )));
            }
            let all = meta
                .get_array("tokenizer.ggml.tokens")
                .ok_or_else(|| checkpoint("the text GGUF has no tokenizer.ggml.tokens"))?
                .strings()
                .map_err(|e| checkpoint(format!("tokenizer.ggml.tokens: {e}")))?;
            let first = usize::try_from(FIRST_SPECIAL).unwrap_or(usize::MAX);
            let mut tokens = Vec::with_capacity(first);
            let mut specials = Vec::new();
            for (id, piece) in all.into_iter().enumerate() {
                if id < first {
                    tokens.push(String::from(piece));
                } else {
                    let id = u32::try_from(id).map_err(|_| checkpoint("a token id past u32"))?;
                    specials.push((id, String::from(piece)));
                }
            }
            PieceVocab::from_byte_level(tokens, &specials)
        }
    }
}

/// The checkpoint's BPE merges, `"left right"` in rank order: `merges.txt`
/// (its `#version` header dropped) for HF and MLX exports,
/// `tokenizer.ggml.merges` for GGUF.
///
/// # Errors
///
/// [`AudioError::Checkpoint`] or [`AudioError::Io`] for merges that are
/// missing or unreadable: a checkpoint that cannot tokenize a context
/// refuses at load rather than tokenizing one wrong.
pub fn read_merges(located: &Located, stores: &Stores) -> Result<Vec<String>, AudioError> {
    match located.flavour {
        Flavour::Hf | Flavour::Mlx => {
            let path = located.dir.join("merges.txt");
            let text = std::fs::read_to_string(&path).map_err(|source| AudioError::Io {
                what: "reading merges.txt",
                source,
            })?;
            Ok(text
                .lines()
                .filter(|line| !line.starts_with("#version") && !line.is_empty())
                .map(str::to_owned)
                .collect())
        }
        Flavour::Gguf => {
            let meta = meta_of(stores.text())?;
            Ok(meta
                .get_array("tokenizer.ggml.merges")
                .ok_or_else(|| checkpoint("the text GGUF has no tokenizer.ggml.merges"))?
                .strings()
                .map_err(|e| checkpoint(format!("tokenizer.ggml.merges: {e}")))?
                .into_iter()
                .map(String::from)
                .collect())
        }
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
pub(crate) mod tests;
