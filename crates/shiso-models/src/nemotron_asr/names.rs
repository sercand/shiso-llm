//! Where each role's tensor sits, per container flavour.
//!
//! Four exports of one checkpoint spell the same tensors four ways. The table
//! lives here, once, so the loader asks for a role and never for a string.

use super::config::Flavour;

/// A layer norm, a linear or a conv: the stem its `.weight` and `.bias` hang off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stem {
    /// The first subsampling conv (1 -> C, full).
    Conv0,
    /// Subsampling stage `s` in 1..=2, its depthwise conv.
    SubDw(usize),
    /// Subsampling stage `s` in 1..=2, its pointwise conv.
    SubPw(usize),
    /// The subsampling output linear.
    SubOut,
    /// Block `n`'s norm before feed-forward `i` (1 or 2).
    NormFf(usize, u8),
    /// Block `n`'s norm before attention.
    NormAtt(usize),
    /// Block `n`'s norm before the conv module.
    NormConv(usize),
    /// Block `n`'s closing norm.
    NormOut(usize),
    /// Block `n`, feed-forward `i` (1 or 2), linear `j` (1 or 2).
    Ff(usize, u8, u8),
    /// Block `n`'s query projection.
    Q(usize),
    /// Block `n`'s key projection.
    K(usize),
    /// Block `n`'s value projection.
    V(usize),
    /// Block `n`'s output projection.
    O(usize),
    /// Block `n`'s relative-position projection.
    Pos(usize),
    /// Block `n`'s first pointwise conv (to the GLU).
    Pw1(usize),
    /// Block `n`'s depthwise conv.
    Dw(usize),
    /// Block `n`'s norm after the depthwise conv.
    ConvNorm(usize),
    /// Block `n`'s second pointwise conv.
    Pw2(usize),
    /// The prediction network's embedding.
    Embed,
    /// The projection of the prediction network's output.
    DecProj,
    /// The projection of the encoder's (prompt-fused) output.
    EncProj,
    /// The joint network's output layer.
    JointOut,
    /// The language prompt's first linear.
    Prompt1,
    /// The language prompt's second linear.
    Prompt2,
}

/// How a flavour stores one LSTM layer's biases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LstmBias {
    /// One tensor holding `b_ih + b_hh`.
    Folded(String),
    /// The input and recurrent biases apart.
    Separate(String, String),
}

/// One flavour's spellings.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Names {
    flavour: Flavour,
}

impl Names {
    /// The table for `flavour`.
    pub(crate) fn new(flavour: Flavour) -> Self {
        Self { flavour }
    }

    fn block(self, n: usize) -> String {
        match self.flavour {
            Flavour::GgufParakeet => format!("enc.blocks.{n}."),
            _ => format!("encoder.layers.{n}."),
        }
    }

    /// The stem of `role`.
    pub(crate) fn stem(self, role: Stem) -> String {
        use Flavour::{GgufAsr, GgufParakeet, Hf, Mlx};
        let f = self.flavour;
        let staged = |i: usize| match f {
            GgufParakeet => format!("enc.pre_encode.conv.{i}"),
            _ => format!("encoder.pre_encode.conv.{i}"),
        };
        match role {
            Stem::Conv0 => match f {
                Hf => "encoder.subsampling.conv_in".into(),
                _ => staged(0),
            },
            Stem::SubDw(s) => match f {
                Hf => format!("encoder.subsampling.layers.{}.depthwise_conv", s - 1),
                _ => staged(3 * s - 1),
            },
            Stem::SubPw(s) => match f {
                Hf => format!("encoder.subsampling.layers.{}.pointwise_conv", s - 1),
                _ => staged(3 * s),
            },
            Stem::SubOut => match f {
                Hf => "encoder.subsampling.linear".into(),
                GgufParakeet => "enc.pre_encode.out".into(),
                Mlx | GgufAsr => "encoder.pre_encode.out".into(),
            },
            Stem::NormFf(n, i) => match f {
                GgufParakeet => format!("{}norm_ff{i}", self.block(n)),
                _ => format!("{}norm_feed_forward{i}", self.block(n)),
            },
            Stem::NormAtt(n) => match f {
                GgufParakeet => format!("{}norm_attn", self.block(n)),
                _ => format!("{}norm_self_att", self.block(n)),
            },
            Stem::NormConv(n) => format!("{}norm_conv", self.block(n)),
            Stem::NormOut(n) => format!("{}norm_out", self.block(n)),
            Stem::Ff(n, i, j) => match f {
                GgufParakeet => format!("{}ff{i}.linear{j}", self.block(n)),
                _ => format!("{}feed_forward{i}.linear{j}", self.block(n)),
            },
            Stem::Q(n) => self.attn(n, "q_proj", "linear_q"),
            Stem::K(n) => self.attn(n, "k_proj", "linear_k"),
            Stem::V(n) => self.attn(n, "v_proj", "linear_v"),
            Stem::O(n) => self.attn(n, "o_proj", "linear_out"),
            Stem::Pos(n) => self.attn(n, "relative_k_proj", "linear_pos"),
            Stem::Pw1(n) => self.conv(n, "pointwise_conv1", "pointwise1"),
            Stem::Dw(n) => self.conv(n, "depthwise_conv", "depthwise"),
            Stem::ConvNorm(n) => match f {
                Hf => format!("{}conv.norm", self.block(n)),
                GgufParakeet => format!("{}conv.bn", self.block(n)),
                Mlx | GgufAsr => format!("{}conv.batch_norm", self.block(n)),
            },
            Stem::Pw2(n) => self.conv(n, "pointwise_conv2", "pointwise2"),
            Stem::Embed => match f {
                Hf => "decoder.embedding".into(),
                GgufParakeet => "pred.embed".into(),
                Mlx | GgufAsr => "decoder.prediction.embed".into(),
            },
            Stem::DecProj => match f {
                Hf => "decoder.decoder_projector".into(),
                _ => "joint.pred".into(),
            },
            Stem::EncProj => match f {
                Hf => "encoder_projector".into(),
                _ => "joint.enc".into(),
            },
            Stem::JointOut => match f {
                Hf => "joint.head".into(),
                GgufParakeet => "joint.out".into(),
                Mlx | GgufAsr => "joint.joint_net.2".into(),
            },
            Stem::Prompt1 => match f {
                Hf => "prompt_projector.linear_1".into(),
                GgufParakeet => "prompt.mlp.0".into(),
                Mlx | GgufAsr => "prompt_kernel.0".into(),
            },
            Stem::Prompt2 => match f {
                Hf => "prompt_projector.linear_2".into(),
                GgufParakeet => "prompt.mlp.2".into(),
                Mlx | GgufAsr => "prompt_kernel.2".into(),
            },
        }
    }

    fn attn(self, n: usize, long: &str, short: &str) -> String {
        match self.flavour {
            Flavour::Hf => format!("{}self_attn.{long}", self.block(n)),
            Flavour::GgufParakeet => format!("{}attn.{short}", self.block(n)),
            Flavour::Mlx | Flavour::GgufAsr => format!("{}self_attn.{short}", self.block(n)),
        }
    }

    fn conv(self, n: usize, long: &str, short: &str) -> String {
        match self.flavour {
            Flavour::GgufParakeet => format!("{}conv.{short}", self.block(n)),
            _ => format!("{}conv.{long}", self.block(n)),
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

    /// Block `n`'s content and position biases, each `[heads][head_dim]`.
    pub(crate) fn pos_biases(self, n: usize) -> (String, String) {
        match self.flavour {
            Flavour::Hf => (
                format!("{}self_attn.bias_u", self.block(n)),
                format!("{}self_attn.bias_v", self.block(n)),
            ),
            Flavour::GgufParakeet => (
                format!("{}attn.pos_bias_u", self.block(n)),
                format!("{}attn.pos_bias_v", self.block(n)),
            ),
            Flavour::Mlx | Flavour::GgufAsr => (
                format!("{}self_attn.pos_bias_u", self.block(n)),
                format!("{}self_attn.pos_bias_v", self.block(n)),
            ),
        }
    }

    /// LSTM layer `l`'s input and recurrent weights, each `[4 * hidden][in]`.
    pub(crate) fn lstm_weights(self, l: usize) -> (String, String) {
        match self.flavour {
            Flavour::Hf => (
                format!("decoder.lstm.weight_ih_l{l}"),
                format!("decoder.lstm.weight_hh_l{l}"),
            ),
            Flavour::Mlx => (
                format!("decoder.prediction.dec_rnn.lstm.{l}.Wx"),
                format!("decoder.prediction.dec_rnn.lstm.{l}.Wh"),
            ),
            Flavour::GgufAsr => (
                format!("decoder.prediction.dec_rnn.lstm.ih_l{l}.weight"),
                format!("decoder.prediction.dec_rnn.lstm.hh_l{l}.weight"),
            ),
            Flavour::GgufParakeet => (format!("pred.lstm.{l}.Wx"), format!("pred.lstm.{l}.Wh")),
        }
    }

    /// LSTM layer `l`'s biases.
    pub(crate) fn lstm_bias(self, l: usize) -> LstmBias {
        match self.flavour {
            Flavour::Hf => LstmBias::Separate(
                format!("decoder.lstm.bias_ih_l{l}"),
                format!("decoder.lstm.bias_hh_l{l}"),
            ),
            Flavour::Mlx => LstmBias::Folded(format!("decoder.prediction.dec_rnn.lstm.{l}.bias")),
            Flavour::GgufAsr => LstmBias::Separate(
                format!("decoder.prediction.dec_rnn.lstm.ih_l{l}.bias"),
                format!("decoder.prediction.dec_rnn.lstm.hh_l{l}.bias"),
            ),
            Flavour::GgufParakeet => LstmBias::Folded(format!("pred.lstm.{l}.bias")),
        }
    }
}
