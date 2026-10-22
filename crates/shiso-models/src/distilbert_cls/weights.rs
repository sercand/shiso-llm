//! The classifier's tensors, by the names the checkpoint gives them.
//!
//! What stays off the device, and why:
//!
//! - **The word and position tables.** An input reads a few dozen rows of a
//!   30,522-row table; its rows are decoded on the host straight out of the
//!   mapping.
//! - **The head.** Two small products on one row per input, which cost less
//!   than a launch.
//!
//! The layer linears and the `LayerNorm` vectors are imported to the device.
//! [`validate`] checks every tensor's presence, shape and encoding on the
//! host first, so a wrong file is refused by name before anything is
//! uploaded, and without a device.

use std::sync::Arc;

use shiso_audio::AudioError;
use shiso_core::{DeviceBuffer, HostBytes};

use super::ClassifyError;
use super::config::{ClassifierConfig, NUM_LABELS};
use crate::audio_ops::load::{LinearSource, f32_param, linear};
use crate::audio_ops::store::{StoredType, TensorStore, dequant};
use crate::audio_ops::{Linear, Ops};

const WORD: &str = "distilbert.embeddings.word_embeddings.weight";
const POSITION: &str = "distilbert.embeddings.position_embeddings.weight";
const EMBED_LN: &str = "distilbert.embeddings.LayerNorm";
const PRE_CLASSIFIER: &str = "pre_classifier";
const CLASSIFIER: &str = "classifier";

/// A `LayerNorm`'s scale and shift, on the device.
#[derive(Debug)]
pub struct NormWeights {
    /// Scale, `[dim]`.
    pub weight: DeviceBuffer,
    /// Shift, `[dim]`.
    pub bias: DeviceBuffer,
}

/// One transformer layer's resident tensors.
#[derive(Debug)]
pub struct LayerWeights {
    /// Query projection, `dim -> dim`.
    pub q: Linear,
    /// Key projection, `dim -> dim`.
    pub k: Linear,
    /// Value projection, `dim -> dim`.
    pub v: Linear,
    /// Attention output projection, `dim -> dim`.
    pub out: Linear,
    /// Norm after the attention residual.
    pub attn_norm: NormWeights,
    /// Feed-forward up projection, `dim -> hidden_dim`.
    pub ffn_in: Linear,
    /// Feed-forward down projection, `hidden_dim -> dim`.
    pub ffn_out: Linear,
    /// Norm after the feed-forward residual.
    pub ffn_norm: NormWeights,
}

/// A `[rows][width]` table read on the host.
#[derive(Debug)]
pub struct HostTable {
    name: String,
    host: Arc<dyn HostBytes>,
    start: usize,
    encoding: RowEncoding,
    width: usize,
    rows: usize,
}

/// How one table row is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowEncoding {
    F32,
    F16,
    Bf16,
}

impl RowEncoding {
    fn of(ty: &StoredType) -> Option<Self> {
        match ty {
            StoredType::F32 => Some(Self::F32),
            StoredType::F16 => Some(Self::F16),
            StoredType::Bf16 => Some(Self::Bf16),
            _ => None,
        }
    }

    fn element_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }
}

impl HostTable {
    /// Rows in the table.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Elements per row.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Writes row `id`, decoded, into the first `width` elements of `out`.
    ///
    /// # Errors
    ///
    /// [`ClassifyError::BadTensor`] for an `out` shorter than a row, an id
    /// past the table or a row that runs past the mapping; a caller that tokenizes with this model's own
    /// vocabulary never sees either.
    pub fn row(&self, id: usize, out: &mut [f32]) -> Result<(), ClassifyError> {
        let why = |why: String| ClassifyError::BadTensor {
            name: self.name.clone(),
            why,
        };
        if out.len() < self.width {
            return Err(why(format!(
                "row buffer holds {} values, a row is {}",
                out.len(),
                self.width
            )));
        }
        if id >= self.rows {
            return Err(why(format!("row {id} is past its {} rows", self.rows)));
        }
        let len = self.width * self.encoding.element_bytes();
        let at = self.start + id * len;
        let bytes = self
            .host
            .as_ref()
            .as_ref()
            .get(at..at + len)
            .ok_or_else(|| why(format!("row {id} runs past the mapping")))?;
        match self.encoding {
            RowEncoding::F32 => {
                for (o, c) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                    *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
            }
            RowEncoding::F16 => {
                for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                    *o = half::f16::from_le_bytes([c[0], c[1]]).to_f32();
                }
            }
            RowEncoding::Bf16 => {
                for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                    *o = half::bf16::from_le_bytes([c[0], c[1]]).to_f32();
                }
            }
        }
        Ok(())
    }
}

/// A dense linear layer on the host, `[out][inp]` row-major plus its bias.
#[derive(Debug, Clone, PartialEq)]
pub struct HostLinear {
    /// Weight, `[out][inp]` row-major.
    pub weight: Vec<f32>,
    /// Bias, `[out]`.
    pub bias: Vec<f32>,
}

/// Everything the classifier reads.
#[derive(Debug)]
pub struct ClassifierWeights {
    /// Word table, `[vocab][dim]`.
    pub word: HostTable,
    /// Position table, `[max_positions][dim]`.
    pub position: HostTable,
    /// The embedding `LayerNorm`, applied on the device to the summed rows.
    pub embed_norm: NormWeights,
    /// The layers, in order.
    pub layers: Vec<LayerWeights>,
    /// First head layer, `dim -> dim`, followed by a `ReLU`.
    pub pre_classifier: HostLinear,
    /// Last head layer, `dim -> labels`.
    pub classifier: HostLinear,
}

/// One tensor the checkpoint must hold: its name and expected shape.
type Expected = (String, Vec<u64>);

fn dim64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

fn linear_names(prefix: &str, out: usize, inp: usize, into: &mut Vec<Expected>) {
    into.push((format!("{prefix}.weight"), vec![dim64(out), dim64(inp)]));
    into.push((format!("{prefix}.bias"), vec![dim64(out)]));
}

fn norm_names(prefix: &str, dim: usize, into: &mut Vec<Expected>) {
    for part in ["weight", "bias"] {
        into.push((format!("{prefix}.{part}"), vec![dim64(dim)]));
    }
}

/// Every tensor `cfg` implies, with its shape.
pub(super) fn expected(cfg: &ClassifierConfig) -> Vec<Expected> {
    let (d, f) = (cfg.dim, cfg.hidden_dim);
    let mut all = vec![
        (WORD.to_owned(), vec![dim64(cfg.vocab), dim64(d)]),
        (
            POSITION.to_owned(),
            vec![dim64(cfg.max_positions), dim64(d)],
        ),
    ];
    norm_names(EMBED_LN, d, &mut all);
    for i in 0..cfg.layers {
        let block = format!("distilbert.transformer.layer.{i}");
        for lin in ["q_lin", "k_lin", "v_lin", "out_lin"] {
            linear_names(&format!("{block}.attention.{lin}"), d, d, &mut all);
        }
        norm_names(&format!("{block}.sa_layer_norm"), d, &mut all);
        linear_names(&format!("{block}.ffn.lin1"), f, d, &mut all);
        linear_names(&format!("{block}.ffn.lin2"), d, f, &mut all);
        norm_names(&format!("{block}.output_layer_norm"), d, &mut all);
    }
    linear_names(PRE_CLASSIFIER, d, d, &mut all);
    linear_names(CLASSIFIER, NUM_LABELS, d, &mut all);
    all
}

fn bad_tensor(name: &str, why: String) -> ClassifyError {
    ClassifyError::BadTensor {
        name: name.to_owned(),
        why,
    }
}

/// Checks, on the host, that `store` holds every tensor `cfg` implies, in the
/// shape `cfg` implies and an encoding this family reads (`F32`, `F16` or
/// `BF16`).
///
/// # Errors
///
/// [`ClassifyError::MissingTensor`] naming the first tensor that is absent;
/// [`ClassifyError::BadTensor`] naming the first one that is misshapen,
/// stored in another encoding, or shorter than its shape.
pub fn validate(store: &TensorStore, cfg: &ClassifierConfig) -> Result<(), ClassifyError> {
    for (name, shape) in expected(cfg) {
        let t = store
            .get(&name)
            .ok_or_else(|| ClassifyError::MissingTensor(name.clone()))?;
        if t.shape != shape {
            return Err(bad_tensor(
                &name,
                format!("has shape {:?}, not {shape:?}", t.shape),
            ));
        }
        let Some(encoding) = RowEncoding::of(&t.ty) else {
            return Err(bad_tensor(
                &name,
                format!("is stored as {:?}, not F32, F16 or BF16", t.ty),
            ));
        };
        let elements = usize::try_from(t.elements()).unwrap_or(usize::MAX);
        if t.data.len() < elements.saturating_mul(encoding.element_bytes()) {
            return Err(bad_tensor(&name, "is shorter than its shape".to_owned()));
        }
    }
    Ok(())
}

fn device(name: &str, error: AudioError) -> ClassifyError {
    match error {
        AudioError::Checkpoint { what } => ClassifyError::BadTensor {
            name: name.to_owned(),
            why: format!("could not be imported: {what}"),
        },
        other => ClassifyError::Device(other.to_string()),
    }
}

pub(super) fn host_table(
    store: &TensorStore,
    name: &str,
    width: usize,
) -> Result<HostTable, ClassifyError> {
    let t = store
        .get(name)
        .ok_or_else(|| ClassifyError::MissingTensor(name.to_owned()))?;
    let encoding = RowEncoding::of(&t.ty)
        .ok_or_else(|| bad_tensor(name, format!("is stored as {:?}", t.ty)))?;
    let rows = t
        .shape
        .first()
        .and_then(|r| usize::try_from(*r).ok())
        .ok_or_else(|| bad_tensor(name, "has no rows".to_owned()))?;
    Ok(HostTable {
        name: name.to_owned(),
        host: Arc::clone(&t.host),
        start: t.range.start,
        encoding,
        width,
        rows,
    })
}

fn host_vec(store: &TensorStore, name: &str) -> Result<Vec<f32>, ClassifyError> {
    let t = store
        .get(name)
        .ok_or_else(|| ClassifyError::MissingTensor(name.to_owned()))?;
    dequant(&t).map_err(|e| device(name, e))
}

fn host_linear(store: &TensorStore, prefix: &str) -> Result<HostLinear, ClassifyError> {
    Ok(HostLinear {
        weight: host_vec(store, &format!("{prefix}.weight"))?,
        bias: host_vec(store, &format!("{prefix}.bias"))?,
    })
}

impl ClassifierWeights {
    /// Imports every tensor `cfg` says the classifier has.
    ///
    /// Runs [`validate`] first, so a wrong checkpoint is refused by name
    /// before any upload.
    ///
    /// # Errors
    ///
    /// Whatever [`validate`] returns; [`ClassifyError::Device`] when an
    /// upload fails.
    pub fn load(
        ops: &Ops,
        store: &TensorStore,
        cfg: &ClassifierConfig,
    ) -> Result<Self, ClassifyError> {
        validate(store, cfg)?;
        let d = cfg.dim;
        let norm = |prefix: &str| -> Result<NormWeights, ClassifyError> {
            Ok(NormWeights {
                weight: f32_param(ops, store, &format!("{prefix}.weight"), d)
                    .map_err(|e| device(&format!("{prefix}.weight"), e))?
                    .0,
                bias: f32_param(ops, store, &format!("{prefix}.bias"), d)
                    .map_err(|e| device(&format!("{prefix}.bias"), e))?
                    .0,
            })
        };
        let lin = |prefix: &str, out: usize, inp: usize| -> Result<Linear, ClassifyError> {
            let weight = format!("{prefix}.weight");
            let bias = format!("{prefix}.bias");
            linear(
                ops,
                store,
                &LinearSource {
                    weight: &weight,
                    bias: Some(&bias),
                    affine_group: None,
                },
                out,
                inp,
            )
            .map_err(|e| device(&weight, e))
        };
        let layers = (0..cfg.layers)
            .map(|i| {
                let block = format!("distilbert.transformer.layer.{i}");
                let attn = |name: &str| lin(&format!("{block}.attention.{name}"), d, d);
                Ok(LayerWeights {
                    q: attn("q_lin")?,
                    k: attn("k_lin")?,
                    v: attn("v_lin")?,
                    out: attn("out_lin")?,
                    attn_norm: norm(&format!("{block}.sa_layer_norm"))?,
                    ffn_in: lin(&format!("{block}.ffn.lin1"), cfg.hidden_dim, d)?,
                    ffn_out: lin(&format!("{block}.ffn.lin2"), d, cfg.hidden_dim)?,
                    ffn_norm: norm(&format!("{block}.output_layer_norm"))?,
                })
            })
            .collect::<Result<Vec<_>, ClassifyError>>()?;
        Ok(Self {
            word: host_table(store, WORD, d)?,
            position: host_table(store, POSITION, d)?,
            embed_norm: norm(EMBED_LN)?,
            layers,
            pre_classifier: host_linear(store, PRE_CLASSIFIER)?,
            classifier: host_linear(store, CLASSIFIER)?,
        })
    }
}
