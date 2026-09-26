//! Where each role's tensor sits, per export.
//!
//! Four exports of one checkpoint spell the same tensors three ways (the two
//! GGUF exports differ only in where the head sits). The table lives here,
//! once, so the loader asks for a role and never for a string.

use super::config::Flavour;

/// A norm, a linear or a conv: the stem its `.weight` and `.bias` hang off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stem {
    /// The stacked-feature projection.
    Proj,
    /// The norm before the first layer.
    InNorm,
    /// The norm after the last layer.
    FinalNorm,
    /// Layer `n`'s norm before attention.
    Norm1(usize),
    /// Layer `n`'s norm before the feed-forward.
    Norm2(usize),
    /// Layer `n`'s output projection.
    O(usize),
    /// Layer `n`'s feed-forward up.
    Fc1(usize),
    /// Layer `n`'s feed-forward down.
    Fc2(usize),
    /// The encoder-to-head projection.
    EncProj,
    /// The subpixel upsampling conv.
    Upsample,
    /// The head's hidden linear.
    Dense,
    /// The head's speaker logits.
    Out,
}

/// How one layer stores its query, key and value projections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Qkv {
    /// One `[3 d][d]` matrix, rows in query, key, value order.
    Fused(String),
    /// Three `[d][d]` matrices.
    Split([String; 3]),
}

/// One export's spellings.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Names {
    flavour: Flavour,
}

impl Names {
    /// The table for `flavour`.
    pub(crate) const fn new(flavour: Flavour) -> Self {
        Self { flavour }
    }

    /// The stem of `role`.
    pub(crate) fn stem(self, role: Stem) -> String {
        let hf = self.flavour == Flavour::Hf;
        let layer = |n: usize, hf_rest: &str, rest: &str| {
            if hf {
                format!("model.audio_tower.layers.{n}.{hf_rest}")
            } else {
                format!("encoder.layers.{n}.{rest}")
            }
        };
        // The head's spelling: the reference framework's, the one prefix both
        // the MLX and audio-framework exports share, or the `sortformer`
        // GGUF's own.
        let head = |hf_name: &str, rest: &str, sortformer: &str| match self.flavour {
            Flavour::Hf => hf_name.to_owned(),
            Flavour::Mlx | Flavour::GgufAudioFramework => format!("sortformer_modules.{rest}"),
            Flavour::GgufSortformer => sortformer.to_owned(),
        };
        match role {
            Stem::Proj if hf => "model.audio_tower.embedder.projection".to_owned(),
            Stem::Proj => "encoder.pre_encode.proj".to_owned(),
            Stem::InNorm if hf => "model.audio_tower.input_layer_norm".to_owned(),
            Stem::InNorm => "encoder.embed_norm".to_owned(),
            Stem::FinalNorm if hf => "model.audio_tower.layer_norm".to_owned(),
            Stem::FinalNorm => "encoder.final_norm".to_owned(),
            Stem::Norm1(n) => layer(n, "layer_norm1", "norm1"),
            Stem::Norm2(n) => layer(n, "layer_norm2", "norm2"),
            Stem::O(n) => layer(n, "self_attn.o_proj", "attn.out_proj"),
            Stem::Fc1(n) => match self.flavour {
                Flavour::Mlx => layer(n, "", "ffn.linear1"),
                _ => layer(n, "mlp.fc1", "ffn.net.0"),
            },
            Stem::Fc2(n) => match self.flavour {
                Flavour::Mlx => layer(n, "", "ffn.linear2"),
                _ => layer(n, "mlp.fc2", "ffn.net.3"),
            },
            Stem::EncProj => head("model.proj", "encoder_proj", "encoder_proj"),
            Stem::Upsample => head(
                "model.upsampler.conv",
                "subpixel_upsample",
                "subpixel_upsample",
            ),
            Stem::Dense => head(
                "classifier.dense",
                "first_hidden_to_hidden",
                "head.first_hidden_to_hidden",
            ),
            Stem::Out => head(
                "classifier.out_proj",
                "single_hidden_to_spks",
                "head.single_hidden_to_spks",
            ),
        }
    }

    /// `role`'s weight tensor.
    pub(crate) fn weight(self, role: Stem) -> String {
        format!("{}.weight", self.stem(role))
    }

    /// `role`'s bias tensor.
    pub(crate) fn bias(self, role: Stem) -> String {
        format!("{}.bias", self.stem(role))
    }

    /// Layer `n`'s query, key and value projections.
    pub(crate) fn qkv(self, n: usize) -> Qkv {
        if self.flavour == Flavour::Hf {
            let p = format!("model.audio_tower.layers.{n}.self_attn");
            Qkv::Split([
                format!("{p}.q_proj.weight"),
                format!("{p}.k_proj.weight"),
                format!("{p}.v_proj.weight"),
            ])
        } else {
            Qkv::Fused(format!("encoder.layers.{n}.attn.w_qkv.weight"))
        }
    }

    /// The learned silence embedding the speaker cache pads with.
    pub(crate) fn silence(self) -> &'static str {
        match self.flavour {
            Flavour::Hf => "silence_embeds",
            Flavour::Mlx | Flavour::GgufAudioFramework => "sortformer_modules.learnable_sil_emb",
            Flavour::GgufSortformer => "learnable_sil_emb",
        }
    }

    /// Whether the upsampling conv is stored `[out][k][in]` rather than the
    /// reference's `[out][in][k]`.
    pub(crate) fn upsample_taps_first(self) -> bool {
        self.flavour == Flavour::Mlx
    }
}
