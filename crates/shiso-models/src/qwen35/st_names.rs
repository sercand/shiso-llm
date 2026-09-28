//! Tensor names and roles for the Qwen 3.5 model family in safetensors exports.

use shiso_modelfile::{HeadAxis, NameMap, Role, ValueOp};

/// Prefix the multimodal wrapper puts on every language-model tensor.
pub(crate) const LM_PREFIX: &str = "language_model.";

/// Qwen 3.5 safetensors name mapping implementing [`NameMap`].
#[derive(Debug, Default, Clone, Copy)]
pub struct Qwen35Names;

impl NameMap for Qwen35Names {
    fn map_role(&self, name: &str) -> Option<Role> {
        let name = name.strip_prefix(LM_PREFIX).unwrap_or(name);
        map_role(name)
    }

    fn expected_tensors(&self, n_layers: u32, full_attention_interval: u32) -> Option<usize> {
        Some(expected_tensor_count(n_layers, full_attention_interval))
    }

    /// The output projection: the 4B checkpoint ties it to the embedding and
    /// ships none, the 27B stores its own. Leaving it unmapped read the 27B's
    /// logits off its embedding - fluent-looking nonsense, since both matrices
    /// have the vocabulary's shape.
    fn optional_tensors(&self) -> &'static [&'static str] {
        &["output.weight"]
    }
}

/// Tensors every layer slot carries, whichever block kind it holds.
const SHARED_PER_LAYER: usize = 5;
/// Additional tensors a recurrent layer slot carries.
const GDN_PER_LAYER: usize = 9;
/// Additional tensors an attention layer slot carries.
const ATTN_PER_LAYER: usize = 6;
/// Tensors that live outside any layer: the embedding and the final norm.
const MODEL_LEVEL: usize = 2;

/// How many tensors a complete export of this family translates to.
///
/// The count is derived rather than hard-coded so that it stays a real check:
/// it fails when a layer's worth of tensors is missing, and it also fails when
/// the layer-kind pattern is not the one the rest of the loader assumed.
///
/// `interval` of zero means "no attention layers", which is how a purely
/// recurrent variant would be described.
fn expected_tensor_count(n_layers: u32, interval: u32) -> usize {
    let mut total = MODEL_LEVEL;
    for layer in 0..n_layers {
        let attention = interval != 0 && (layer + 1).is_multiple_of(interval);
        total += SHARED_PER_LAYER
            + if attention {
                ATTN_PER_LAYER
            } else {
                GDN_PER_LAYER
            };
    }
    total
}

/// Translates one tensor name, or reports that this loader does not know it.
pub(crate) fn map_role(name: &str) -> Option<Role> {
    match name {
        "model.embed_tokens" => return Some(Role::plain("token_embd.weight")),
        "model.norm.weight" => return Some(Role::plain("output_norm.weight")),
        "lm_head" | "lm_head.weight" => return Some(Role::plain("output.weight")),
        _ => {}
    }

    let rest = name.strip_prefix("model.layers.")?;
    let (index, role) = rest.split_once('.')?;
    let layer: u32 = index.parse().ok()?;

    let (suffix, op, heads) = match role {
        // Norms, shared by both block kinds.
        "input_layernorm.weight" => ("attn_norm.weight", ValueOp::Keep, None),
        "post_attention_layernorm.weight" => ("post_attention_norm.weight", ValueOp::Keep, None),

        // Feed-forward.
        "mlp.gate_proj" => ("ffn_gate.weight", ValueOp::Keep, None),
        "mlp.up_proj" => ("ffn_up.weight", ValueOp::Keep, None),
        "mlp.down_proj" => ("ffn_down.weight", ValueOp::Keep, None),

        // Recurrent block.
        "linear_attn.in_proj_qkv" => (
            "attn_qkv.weight",
            ValueOp::Keep,
            Some(HeadAxis::QkvTailRows),
        ),
        "linear_attn.in_proj_z" => ("attn_gate.weight", ValueOp::Keep, Some(HeadAxis::ValueRows)),
        "linear_attn.in_proj_a" => (
            "ssm_alpha.weight",
            ValueOp::Keep,
            Some(HeadAxis::PerHeadRows),
        ),
        "linear_attn.in_proj_b" => (
            "ssm_beta.weight",
            ValueOp::Keep,
            Some(HeadAxis::PerHeadRows),
        ),
        "linear_attn.out_proj" => (
            "ssm_out.weight",
            ValueOp::Keep,
            Some(HeadAxis::ValueColumns),
        ),
        "linear_attn.conv1d.weight" => (
            "ssm_conv1d.weight",
            ValueOp::Keep,
            Some(HeadAxis::QkvTailRows),
        ),
        "linear_attn.A_log" => ("ssm_a", ValueOp::NegExp, Some(HeadAxis::PerHeadRows)),
        "linear_attn.dt_bias" => ("ssm_dt.bias", ValueOp::Keep, Some(HeadAxis::PerHeadRows)),
        "linear_attn.norm.weight" => ("ssm_norm.weight", ValueOp::Keep, None),

        // Attention block.
        "self_attn.q_proj" => ("attn_q.weight", ValueOp::Keep, None),
        "self_attn.k_proj" => ("attn_k.weight", ValueOp::Keep, None),
        "self_attn.v_proj" => ("attn_v.weight", ValueOp::Keep, None),
        "self_attn.o_proj" => ("attn_output.weight", ValueOp::Keep, None),
        "self_attn.q_norm.weight" => ("attn_q_norm.weight", ValueOp::Keep, None),
        "self_attn.k_norm.weight" => ("attn_k_norm.weight", ValueOp::Keep, None),

        _ => return None,
    };

    Some(Role {
        gguf: format!("blk.{layer}.{suffix}"),
        op,
        heads,
    })
}
