//! Where each role's tensor sits, per container flavour.
//!
//! Three containers spell the same tensors three ways:
//! the HF export under a `thinker.` prefix, the MLX export without it, and the
//! GGUF pair in its own short names, the tower in the mmproj and the decoder in
//! the text file. The table lives here, once, so the loader asks for a role and
//! never for a string. The GGUF q/k weights keep the checkpoint's row order
//! (the converter does not permute them for this architecture), so no role
//! needs a reordering.

use super::config::Flavour;

/// A norm, a linear or a conv: the stem its `.weight` and `.bias` hang off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stem {
    /// Tower conv `i` in 1..=3.
    Conv(u8),
    /// The tower's flattened-conv projection, 7680 -> 1024, no bias.
    ConvOut,
    /// Tower layer `n`'s norm before attention.
    AttnNorm(usize),
    /// Tower layer `n`'s query projection.
    Q(usize),
    /// Tower layer `n`'s key projection.
    K(usize),
    /// Tower layer `n`'s value projection.
    V(usize),
    /// Tower layer `n`'s output projection.
    O(usize),
    /// Tower layer `n`'s norm before the feed-forward.
    FfnNorm(usize),
    /// Tower layer `n`'s feed-forward up.
    Fc1(usize),
    /// Tower layer `n`'s feed-forward down.
    Fc2(usize),
    /// The tower's closing norm.
    PostLn,
    /// The output head's first linear.
    Proj1,
    /// The output head's second linear, into the decoder's width.
    Proj2,
    /// The token embedding, also the tied output head.
    Embed,
    /// The decoder's final norm.
    Norm,
    /// Decoder layer `n`'s norm before attention.
    DecAttnNorm(usize),
    /// Decoder layer `n`'s query projection.
    DecQ(usize),
    /// Decoder layer `n`'s key projection.
    DecK(usize),
    /// Decoder layer `n`'s value projection.
    DecV(usize),
    /// Decoder layer `n`'s output projection.
    DecO(usize),
    /// Decoder layer `n`'s per-head query norm.
    QNorm(usize),
    /// Decoder layer `n`'s per-head key norm.
    KNorm(usize),
    /// Decoder layer `n`'s norm before the MLP.
    DecFfnNorm(usize),
    /// Decoder layer `n`'s MLP gate.
    Gate(usize),
    /// Decoder layer `n`'s MLP up.
    Up(usize),
    /// Decoder layer `n`'s MLP down.
    Down(usize),
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

    /// The stem of `role`.
    pub(crate) fn stem(self, role: Stem) -> String {
        match self.flavour {
            Flavour::Hf => format!("thinker.{}", safetensors(role)),
            Flavour::Mlx => safetensors(role),
            Flavour::Gguf => gguf(role),
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
}

/// The safetensors spelling of the HF and MLX exports, `thinker.` prefix
/// excluded.
fn safetensors(role: Stem) -> String {
    let tower = |n: usize, rest: &str| format!("audio_tower.layers.{n}.{rest}");
    let text = |n: usize, rest: &str| format!("model.layers.{n}.{rest}");
    match role {
        Stem::Conv(i) => format!("audio_tower.conv2d{i}"),
        Stem::ConvOut => "audio_tower.conv_out".into(),
        Stem::AttnNorm(n) => tower(n, "self_attn_layer_norm"),
        Stem::Q(n) => tower(n, "self_attn.q_proj"),
        Stem::K(n) => tower(n, "self_attn.k_proj"),
        Stem::V(n) => tower(n, "self_attn.v_proj"),
        Stem::O(n) => tower(n, "self_attn.out_proj"),
        Stem::FfnNorm(n) => tower(n, "final_layer_norm"),
        Stem::Fc1(n) => tower(n, "fc1"),
        Stem::Fc2(n) => tower(n, "fc2"),
        Stem::PostLn => "audio_tower.ln_post".into(),
        Stem::Proj1 => "audio_tower.proj1".into(),
        Stem::Proj2 => "audio_tower.proj2".into(),
        Stem::Embed => "model.embed_tokens".into(),
        Stem::Norm => "model.norm".into(),
        Stem::DecAttnNorm(n) => text(n, "input_layernorm"),
        Stem::DecQ(n) => text(n, "self_attn.q_proj"),
        Stem::DecK(n) => text(n, "self_attn.k_proj"),
        Stem::DecV(n) => text(n, "self_attn.v_proj"),
        Stem::DecO(n) => text(n, "self_attn.o_proj"),
        Stem::QNorm(n) => text(n, "self_attn.q_norm"),
        Stem::KNorm(n) => text(n, "self_attn.k_norm"),
        Stem::DecFfnNorm(n) => text(n, "post_attention_layernorm"),
        Stem::Gate(n) => text(n, "mlp.gate_proj"),
        Stem::Up(n) => text(n, "mlp.up_proj"),
        Stem::Down(n) => text(n, "mlp.down_proj"),
    }
}

/// The GGUF spelling: `a.*` and `mm.a.*` in the mmproj, the rest in the
/// text file.
fn gguf(role: Stem) -> String {
    match role {
        Stem::Conv(i) => format!("a.conv2d.{i}"),
        Stem::ConvOut => "a.conv_out".into(),
        Stem::AttnNorm(n) => format!("a.blk.{n}.ln1"),
        Stem::Q(n) => format!("a.blk.{n}.attn_q"),
        Stem::K(n) => format!("a.blk.{n}.attn_k"),
        Stem::V(n) => format!("a.blk.{n}.attn_v"),
        Stem::O(n) => format!("a.blk.{n}.attn_out"),
        Stem::FfnNorm(n) => format!("a.blk.{n}.ln2"),
        Stem::Fc1(n) => format!("a.blk.{n}.ffn_up"),
        Stem::Fc2(n) => format!("a.blk.{n}.ffn_down"),
        Stem::PostLn => "a.post_ln".into(),
        Stem::Proj1 => "mm.a.mlp.1".into(),
        Stem::Proj2 => "mm.a.mlp.2".into(),
        Stem::Embed => "token_embd".into(),
        Stem::Norm => "output_norm".into(),
        Stem::DecAttnNorm(n) => format!("blk.{n}.attn_norm"),
        Stem::DecQ(n) => format!("blk.{n}.attn_q"),
        Stem::DecK(n) => format!("blk.{n}.attn_k"),
        Stem::DecV(n) => format!("blk.{n}.attn_v"),
        Stem::DecO(n) => format!("blk.{n}.attn_output"),
        Stem::QNorm(n) => format!("blk.{n}.attn_q_norm"),
        Stem::KNorm(n) => format!("blk.{n}.attn_k_norm"),
        Stem::DecFfnNorm(n) => format!("blk.{n}.ffn_norm"),
        Stem::Gate(n) => format!("blk.{n}.ffn_gate"),
        Stem::Up(n) => format!("blk.{n}.ffn_up"),
        Stem::Down(n) => format!("blk.{n}.ffn_down"),
    }
}
