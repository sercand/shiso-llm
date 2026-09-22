//! Gemma 4 tensor names and binding for MLX safetensors checkpoints.
#![expect(clippy::too_many_lines, reason = "binds all 42 layers and model roles")]

use shiso_core::LayerIdx;
use shiso_engine::shape::KvSource;
use shiso_modelfile::name_map::{NameMap, Role};
use shiso_modelfile::reader::{GgufError, GgufResult, TensorDesc};
use shiso_modelfile::safetensors::SafetensorsFile;
use shiso_modelfile::source::TensorSource;

use super::binding::{
    Gemma4AttnRoles, Gemma4DescBinding, Gemma4FfnRoles, Gemma4LayerRoles, Gemma4PleRoles,
    Gemma4Roles,
};
use super::config::Gemma4Config;

/// Prefix the MLX export puts on language-model tensors.
pub const LM_PREFIX: &str = "language_model.model.";

/// Gemma 4 MLX safetensors name mapping implementing [`NameMap`].
#[derive(Debug, Default, Clone, Copy)]
pub struct Gemma4Names;

impl NameMap for Gemma4Names {
    fn map_role(&self, name: &str) -> Option<Role> {
        let name = name.strip_prefix(LM_PREFIX).unwrap_or(name);
        map_role(name)
    }
}

/// Translates one MLX tensor name into its GGUF role name.
pub(crate) fn map_role(name: &str) -> Option<Role> {
    match name {
        "embed_tokens" | "embed_tokens.weight" => return Some(Role::plain("token_embd.weight")),
        "norm.weight" | "norm" => return Some(Role::plain("output_norm.weight")),
        "per_layer_model_projection" | "per_layer_model_projection.weight" => {
            return Some(Role::plain("per_layer_model_proj.weight"));
        }
        "per_layer_projection_norm.weight" | "per_layer_projection_norm" => {
            return Some(Role::plain("per_layer_proj_norm.weight"));
        }
        "embed_tokens_per_layer" | "embed_tokens_per_layer.weight" => {
            return Some(Role::plain("per_layer_token_embd.weight"));
        }
        _ => {}
    }

    let rest = name.strip_prefix("layers.")?;
    let (index, role) = rest.split_once('.')?;
    let layer: u32 = index.parse().ok()?;

    let role = role.strip_suffix(".weight").unwrap_or(role);

    let suffix = match role {
        "input_layernorm" => "attn_norm.weight",
        "post_attention_layernorm" => "post_attention_norm.weight",
        "pre_feedforward_layernorm" => "ffn_norm.weight",
        "post_feedforward_layernorm" => "post_ffw_norm.weight",
        "post_per_layer_input_norm" => "post_norm.weight",
        "layer_scalar" => "layer_output_scale.weight",
        "self_attn.q_proj" => "attn_q.weight",
        // Named whatever slot they sit on. Which slots may carry a key or a
        // value is the geometry's answer, not a name map's, and the binder
        // refuses a sharing slot that ships one - so translating these
        // unconditionally is what lets that refusal see them at all.
        "self_attn.k_proj" => "attn_k.weight",
        "self_attn.v_proj" => "attn_v.weight",
        "self_attn.o_proj" => "attn_output.weight",
        "self_attn.q_norm" => "attn_q_norm.weight",
        "self_attn.k_norm" => "attn_k_norm.weight",
        "mlp.gate_proj" => "ffn_gate.weight",
        "mlp.up_proj" => "ffn_up.weight",
        "mlp.down_proj" => "ffn_down.weight",
        "per_layer_input_gate" => "inp_gate.weight",
        "per_layer_projection" => "proj.weight",
        _ => return None,
    };
    Some(Role::plain(format!("blk.{layer}.{suffix}")))
}

/// Helper to lookup a tensor descriptor from a `SafetensorsFile`.
fn find_tensor(src: &SafetensorsFile, name: &str, want_shape: &[u64]) -> GgufResult<TensorDesc> {
    let desc = src
        .find(name)
        .ok_or_else(|| GgufError::MissingTensor { name: name.into() })?
        .clone();

    // Check dimensions
    let got_dims = desc.shape.dims();
    if got_dims != want_shape {
        return Err(GgufError::LayoutMismatch {
            tensor: name.into(),
            want: Box::from(format!("{want_shape:?}")),
            got: Box::from(format!("{got_dims:?}")),
        });
    }
    Ok(desc)
}

/// Binds every tensor Gemma 4 needs from an MLX safetensors checkpoint.
///
/// # Errors
///
/// [`GgufError`] on missing tensors or dimension mismatches.
pub fn bind_gemma4_mlx_desc(
    src: &SafetensorsFile,
    config: &Gemma4Config,
) -> GgufResult<Gemma4DescBinding> {
    let shape = config.shape();
    let hidden = u64::from(shape.hidden);
    let vocab = u64::from(shape.vocab);
    let intermediate = u64::from(shape.intermediate);
    let n_layers = shape.n_layers();

    let ple = shape.ple.ok_or_else(|| GgufError::LayoutMismatch {
        tensor: Box::from("shape profile"),
        want: Box::from("a per-layer embedding table"),
        got: Box::from("none"),
    })?;
    let ple_dim = u64::from(ple.dim);
    let ple_rows = u64::from(n_layers) * ple_dim;

    let token_embd = find_tensor(src, "token_embd.weight", &[hidden, vocab])?;
    let output_norm = find_tensor(src, "output_norm.weight", &[hidden])?;
    let ple_model_proj = find_tensor(src, "per_layer_model_proj.weight", &[hidden, ple_rows])?;
    let ple_proj_norm = find_tensor(src, "per_layer_proj_norm.weight", &[ple_dim])?;
    let ple_table = find_tensor(src, "per_layer_token_embd.weight", &[ple_rows, vocab])?;

    let mut layers = Vec::with_capacity(n_layers as usize);
    for il in 0..n_layers {
        let prefix = format!("blk.{il}");
        let input_norm = find_tensor(src, &format!("{prefix}.attn_norm.weight"), &[hidden])?;
        let post_attn_norm = find_tensor(
            src,
            &format!("{prefix}.post_attention_norm.weight"),
            &[hidden],
        )?;
        let pre_ffn_norm = find_tensor(src, &format!("{prefix}.ffn_norm.weight"), &[hidden])?;
        let post_ffn_norm = find_tensor(src, &format!("{prefix}.post_ffw_norm.weight"), &[hidden])?;
        let post_norm = find_tensor(src, &format!("{prefix}.post_norm.weight"), &[hidden])?;
        let layer_scale = find_tensor(src, &format!("{prefix}.layer_output_scale.weight"), &[1])?;
        let Some(slot) = shape.attn_layer(LayerIdx::new(il)) else {
            return Err(GgufError::LayoutMismatch {
                tensor: Box::from(format!("blk.{il}")),
                want: Box::from("an attention slot"),
                got: Box::from("none"),
            });
        };
        let head = u64::from(slot.head_dim);
        let q_out = u64::from(slot.q_heads) * head;
        let kv_out = u64::from(slot.kv_heads) * head;
        let owns_kv = slot.kv == KvSource::Own;

        let q = find_tensor(src, &format!("{prefix}.attn_q.weight"), &[hidden, q_out])?;
        let output = find_tensor(
            src,
            &format!("{prefix}.attn_output.weight"),
            &[q_out, hidden],
        )?;
        let q_norm = find_tensor(src, &format!("{prefix}.attn_q_norm.weight"), &[head])?;

        let (k, v, k_norm) = if owns_kv {
            let k = find_tensor(src, &format!("{prefix}.attn_k.weight"), &[hidden, kv_out])?;
            let v = find_tensor(src, &format!("{prefix}.attn_v.weight"), &[hidden, kv_out])?;
            let k_norm = find_tensor(src, &format!("{prefix}.attn_k_norm.weight"), &[head])?;
            (Some(k), Some(v), Some(k_norm))
        } else {
            // A sharing slot is not merely allowed to omit its key and value:
            // it is required to. Binding whatever is there and letting the
            // geometry decide what to read would accept an export whose
            // schedule disagrees with the config and quietly never read one of
            // its tensors.
            for role in ["attn_k.weight", "attn_v.weight", "attn_k_norm.weight"] {
                if src.find(&format!("{prefix}.{role}")).is_some() {
                    return Err(GgufError::LayoutMismatch {
                        tensor: format!("{prefix}.{role}").into(),
                        want: Box::from("absent on a sharing slot"),
                        got: Box::from("present"),
                    });
                }
            }
            (None, None, None)
        };

        let attn = Gemma4AttnRoles {
            q,
            k,
            v,
            output,
            q_norm,
            k_norm,
        };

        let ffn = Gemma4FfnRoles {
            gate: find_tensor(
                src,
                &format!("{prefix}.ffn_gate.weight"),
                &[hidden, intermediate],
            )?,
            up: find_tensor(
                src,
                &format!("{prefix}.ffn_up.weight"),
                &[hidden, intermediate],
            )?,
            down: find_tensor(
                src,
                &format!("{prefix}.ffn_down.weight"),
                &[intermediate, hidden],
            )?,
        };

        let ple = Gemma4PleRoles {
            gate: find_tensor(
                src,
                &format!("{prefix}.inp_gate.weight"),
                &[hidden, ple_dim],
            )?,
            proj: find_tensor(src, &format!("{prefix}.proj.weight"), &[ple_dim, hidden])?,
            post_norm,
        };

        layers.push(Gemma4LayerRoles {
            input_norm,
            post_attn_norm,
            pre_ffn_norm,
            post_ffn_norm,
            layer_scale,
            attn,
            ffn,
            ple,
        });
    }

    Ok(Gemma4Roles {
        token_embd,
        output_norm,
        ple_model_proj,
        ple_proj_norm,
        ple_table,
        layers,
    })
}
