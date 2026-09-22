//! Every tensor this family's forward pass reads, named once.
//!
//! The list is derived from the geometry rather than written out: 40 layers of
//! 256 experts is 31,333 names, and a table of them would be a table nobody
//! reads and nothing checks. [`expected_tensors`] walks the shape and produces
//! them in the order the pass reads them.
//!
//! # What this buys, and why a count would not have bought it
//!
//! [`shiso_modelfile::NameMap::expected_tensors`] - the sibling container's version
//! of this - answers a *count*, and the loader refuses a directory that
//! translates to any other number. That turns "every tensor we looked for was
//! there" into "there was nothing else there we should have looked for", which
//! is most of the value. It does not catch a checkpoint with the right number
//! of the wrong tensors, and at 31,333 names built from a template a
//! transposed index is exactly the mistake that keeps the count.
//!
//! So [`check_export`] compares names, not counts, in both directions, and
//! then compares each tensor's storage scheme against what the checkpoint's own
//! `hf_quant_config.json` claims for it. That last one is the check with no
//! sibling anywhere in this tree: the config says `mlp.gate` is unquantized and
//! the bytes say `BF16`, and a router whose logits were decoded from nibbles
//! sends every token to the wrong eight experts while every shape still
//! matches.

use shiso_core::error::{CoreError, CoreResult};
use shiso_engine::shape::{HybridShape, LayerKind};
use shiso_modelfile::modelopt::{ModeloptFile, Stored};

use super::{QuantAlgo, Qwen35MoeConfig};

/// Prefix every language-model tensor carries.
pub const TEXT_PREFIX: &str = "model.language_model.";

/// Prefix the vision tower's tensors carry, which this module does not name.
pub const VISION_PREFIX: &str = "model.visual.";

/// The output projection, which sits outside both towers.
pub const LM_HEAD: &str = "lm_head";

/// The prefixes a language-model load asks the container for.
///
/// The vision tower is described by its own loader and the draft head by
/// nothing, so neither is in this list and neither costs a description.
pub const TEXT_ROOTS: [&str; 2] = [TEXT_PREFIX, LM_HEAD];

/// Every tensor the forward pass reads, in the order it reads them.
///
/// The order is the pass's rather than the checkpoint's: the embedding table,
/// then each slot's norms, block and feed-forward, then the final norm and the
/// output projection. A loader that walks this list touches each shard's
/// mapping roughly in order, which is the difference between a sequential read
/// of 23.4 GB and a random one.
#[must_use]
pub fn expected_tensors(shape: &HybridShape) -> Vec<String> {
    let moe = shape.moe;
    let experts = moe.map_or(0, |moe| moe.experts);
    let mut names = Vec::new();
    names.push(format!("{TEXT_PREFIX}embed_tokens"));
    for (slot, kind) in shape.layers.iter().enumerate() {
        let at = format!("{TEXT_PREFIX}layers.{slot}");
        names.push(format!("{at}.input_layernorm"));
        names.push(format!("{at}.post_attention_layernorm"));
        match kind {
            LayerKind::Gdn => {
                // `A_log` and `dt_bias` carry no `.weight`, which is a fact
                // about the export and the reason the container names a tensor
                // by its array rather than by a stripped base.
                for role in [
                    "A_log",
                    "dt_bias",
                    "conv1d",
                    "in_proj_a",
                    "in_proj_b",
                    "in_proj_qkv",
                    "in_proj_z",
                    "norm",
                    "out_proj",
                ] {
                    names.push(format!("{at}.linear_attn.{role}"));
                }
            }
            LayerKind::Attn => {
                for role in ["q_norm", "k_norm", "q_proj", "k_proj", "v_proj", "o_proj"] {
                    names.push(format!("{at}.self_attn.{role}"));
                }
            }
        }
        // The router itself, the shared expert's own scalar gate, the shared
        // expert, and then one tensor per expert per projection - which is what
        // makes this a different loader from the sparse family's, where an
        // expert is a stride into one stacked slab.
        names.push(format!("{at}.mlp.gate"));
        names.push(format!("{at}.mlp.shared_expert_gate"));
        for role in ["gate_proj", "up_proj", "down_proj"] {
            names.push(format!("{at}.mlp.shared_expert.{role}"));
        }
        for expert in 0..experts {
            for role in ["gate_proj", "up_proj", "down_proj"] {
                names.push(format!("{at}.mlp.experts.{expert}.{role}"));
            }
        }
    }
    names.push(format!("{TEXT_PREFIX}norm"));
    names.push(LM_HEAD.to_string());
    names
}

/// The scheme a tensor's declared quantization implies.
///
/// The two vocabularies are different documents' - `hf_quant_config.json`'s
/// `quant_algo` and the container's reading of the bytes - and this is the one
/// place they are put beside each other.
fn scheme_of(algo: QuantAlgo) -> &'static str {
    match algo {
        QuantAlgo::Fp8 => "fp8",
        QuantAlgo::W4A16Nvfp4 { .. } => "nvfp4",
        QuantAlgo::None => "plain",
    }
}

/// Checks an open checkpoint against the geometry and the quantization
/// document, before a weight is read.
///
/// Three things, and each is a wrong answer rather than a crash if it goes
/// unchecked: a tensor the pass will read that is not there; a language-model
/// tensor that is there and that nothing will read, which means the list above
/// and the export disagree about what this family is; and a tensor whose bytes
/// are stored under a scheme other than the one its own config claims.
///
/// # Errors
///
/// [`CoreError::ShapeMismatch`] naming `tensor_count` when the two lists differ
/// in length, [`CoreError::InvalidValue`] naming the first tensor that is
/// missing, unexpected, or stored under the wrong scheme, and
/// [`CoreError::ConfigMissing`] when `config` was read from a container that
/// carries no quantization document to check the export against.
pub fn check_export(file: &ModeloptFile, config: &Qwen35MoeConfig) -> CoreResult<()> {
    let want = expected_tensors(config.shape());
    // A modelopt export is checked against the document that describes it. A
    // config read from a container that carries no such document cannot say
    // what any tensor here was declared as, and passing every tensor because
    // the map is empty is how an export whose halves disagree gets in.
    let quant = config.quant().ok_or(CoreError::ConfigMissing {
        key: "hf_quant_config.json".to_string(),
    })?;
    for name in &want {
        let Some(tensor) = file.get(name) else {
            return Err(CoreError::InvalidValue {
                key: "missing tensor",
                value: name.clone(),
            });
        };
        // The container read the scheme off the weight's own format; the
        // document claims one. A disagreement means the two halves of this
        // export were written at different times, and the bytes are what the
        // kernels will read.
        let claimed = scheme_of(quant.algo_for(name));
        let stored = tensor.stored().kind();
        if claimed != stored {
            return Err(CoreError::InvalidValue {
                key: "storage scheme",
                value: format!("{name} is stored {stored} and declared {claimed}"),
            });
        }
    }
    // The other direction. A checkpoint carrying a language-model tensor this
    // list does not name is a checkpoint of a family this module has not read,
    // and binding it would be binding most of a different model.
    let found = file
        .tensors()
        .filter(|tensor| {
            TEXT_ROOTS
                .iter()
                .any(|root| tensor.name().starts_with(root))
        })
        .count();
    if found != want.len() {
        return Err(CoreError::ShapeMismatch {
            field: "tensor_count",
            expected: u64::try_from(want.len()).unwrap_or(u64::MAX),
            got: u64::try_from(found).unwrap_or(u64::MAX),
        });
    }
    Ok(())
}

/// Whether a tensor of this family is stored as its scheme expects.
///
/// Public because the census cases assert over it and because a loader that
/// wants to match on the scheme should ask the container rather than re-derive
/// it from a name.
#[must_use]
pub const fn is_routed_expert(stored: Stored) -> bool {
    matches!(stored, Stored::Nvfp4 { .. })
}

#[cfg(test)]
#[path = "binding_tests.rs"]
mod tests;
