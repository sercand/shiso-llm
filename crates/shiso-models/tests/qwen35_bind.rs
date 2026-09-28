//! Binds the real `Q4_K_M` checkpoint to layer roles and checks the hybrid block
//! pattern, every role's dimensions, and the formats the checkpoint actually
//! uses for them.
//!
//! Skips when no local model is present. The dimension expectations are derived
//! from the shipped architecture config rather than hardcoded per tensor, so a
//! profile change moves the expectations with it; the concrete numbers are
//! asserted once each for a recurrent and an attention layer, which is where a
//! wrong formula would show up.

#[path = "common/qwen35.rs"]
mod common;

use std::path::{Path, PathBuf};

use shiso_core::{DType, HfConfig, LayerIdx};
use shiso_engine::shape::LayerKind;
use shiso_modelfile::{GgufFile, TensorView};
use shiso_models::qwen35::WEIGHT_TYPES;
use shiso_models::qwen35::{BoundLayer, Qwen35Config, bind_qwen35};

/// Layer slots in the test checkpoint.
const N_LAYERS: usize = 32;
/// Residual-stream width.
const HIDDEN: u64 = 2560;
/// Feed-forward inner width.
const INTERMEDIATE: u64 = 9216;
/// Recurrent value heads.
const V_HEADS: u64 = 32;
/// Combined query/key/value width of a recurrent block.
const GDN_QKV: u64 = 8192;
/// Output-gate width of a recurrent block.
const GDN_GATE: u64 = 4096;
/// Width of one attention head.
const ATTN_HEAD: u64 = 256;
/// Query projection width of an attention block, gate included.
const ATTN_Q: u64 = 8192;
/// Key or value projection width of an attention block.
const ATTN_KV: u64 = 1024;
/// Output projection width of an attention block.
const ATTN_OUT: u64 = 4096;
/// Causal convolution width of a recurrent block.
const CONV_KERNEL: u64 = 4;
/// Attention block stride.
const INTERVAL: u32 = 4;

/// The architecture fixture that ships with the workspace.
fn fixture_config() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/loader/qwen35-config.json")
}

/// Opens the model and derives the profile from the shipped config fixture.
fn model_and_profile() -> Option<(GgufFile, Qwen35Config)> {
    let file = common::open_model()?;
    let config = HfConfig::from_file(&fixture_config()).expect("config fixture parses");
    let profile = Qwen35Config::from_hf(&config).expect("qwen35 config");
    Some((file, profile))
}

#[test]
fn profile_is_24_recurrent_and_8_attention_layers() {
    let Some((_file, profile)) = model_and_profile() else {
        common::skip("model not found");
        return;
    };
    let shape = profile.shape();

    assert_eq!(shape.n_layers() as usize, N_LAYERS);
    assert_eq!(profile.full_attention_interval(), INTERVAL);
    let attn = (0..shape.n_layers())
        .filter(|&il| shape.layer_kind(LayerIdx::new(il)) == Some(LayerKind::Attn))
        .collect::<Vec<_>>();
    let want = (0..shape.n_layers())
        .filter(|&il| (il + 1).is_multiple_of(INTERVAL))
        .collect::<Vec<_>>();
    assert_eq!(attn, want);
}

#[test]
fn binding_reproduces_the_hybrid_pattern() {
    let Some((file, profile)) = model_and_profile() else {
        common::skip("model not found");
        return;
    };
    let binding = bind_qwen35(&file, &profile).expect("the real model binds");

    assert_eq!(binding.layers.len(), N_LAYERS);
    let mut gdn = 0;
    let mut attn = 0;
    for (il, layer) in binding.layers.iter().enumerate() {
        let expected = if u32::try_from(il).expect("layer index fits") % INTERVAL == INTERVAL - 1 {
            LayerKind::Attn
        } else {
            LayerKind::Gdn
        };
        assert_eq!(layer.kind, expected, "layer {il}");
        assert!(
            layer.gdn.is_some() != layer.attn.is_some(),
            "layer {il} must hold exactly one block kind"
        );
        match expected {
            LayerKind::Gdn => gdn += 1,
            LayerKind::Attn => attn += 1,
            LayerKind::Ssm => panic!("layer {il}: this family has no {expected:?} slot"),
        }
    }
    assert_eq!((gdn, attn), (24, 8));
}

#[test]
fn globals_match_and_logits_are_tied() {
    let Some((file, profile)) = model_and_profile() else {
        common::skip("model not found");
        return;
    };
    let binding = bind_qwen35(&file, &profile).expect("the real model binds");

    assert_eq!(binding.token_embd.dtype, DType::Q6_K);
    assert_eq!(binding.token_embd.shape.dims(), &[HIDDEN, 248_320]);
    assert_eq!(binding.output_norm.dtype, DType::F32);
    assert_eq!(binding.output_norm.shape.dims(), &[HIDDEN]);
    assert!(binding.lm_head.is_none(), "this checkpoint ties its logits");
}

#[test]
fn recurrent_block_roles_have_the_expected_geometry() {
    let Some((file, profile)) = model_and_profile() else {
        common::skip("model not found");
        return;
    };
    let binding = bind_qwen35(&file, &profile).expect("the real model binds");

    for (il, layer) in binding.layers.iter().enumerate() {
        let Some(gdn) = &layer.gdn else {
            continue;
        };
        check_dims(&gdn.qkv, &[HIDDEN, GDN_QKV], il, "qkv");
        check_dims(&gdn.gate, &[HIDDEN, GDN_GATE], il, "gate");
        check_dims(&gdn.conv1d, &[CONV_KERNEL, GDN_QKV], il, "conv1d");
        check_dims(&gdn.a, &[V_HEADS], il, "a");
        check_dims(&gdn.alpha, &[HIDDEN, V_HEADS], il, "alpha");
        check_dims(&gdn.beta, &[HIDDEN, V_HEADS], il, "beta");
        check_dims(&gdn.dt_bias, &[V_HEADS], il, "dt_bias");
        check_dims(&gdn.norm, &[128], il, "norm");
        check_dims(&gdn.out, &[GDN_GATE, HIDDEN], il, "out");
        assert_eq!(gdn.conv1d.dtype, DType::F32, "layer {il} conv1d");
        assert_eq!(gdn.a.dtype, DType::F32, "layer {il} a");
        assert_eq!(gdn.dt_bias.dtype, DType::F32, "layer {il} dt_bias");
        assert_eq!(gdn.norm.dtype, DType::F32, "layer {il} norm");
        for weight in [&gdn.qkv, &gdn.gate, &gdn.alpha, &gdn.beta, &gdn.out] {
            assert!(
                WEIGHT_TYPES.contains(&weight.dtype),
                "layer {il} {} has unusable format {}",
                weight.name,
                weight.dtype.type_name()
            );
        }
    }

    let first = binding
        .layers
        .iter()
        .find_map(|layer| layer.gdn.as_ref())
        .expect("at least one recurrent layer");
    assert_eq!(first.qkv.dtype, DType::Q5_K);
    assert_eq!(first.gate.dtype, DType::Q4_K);
    assert_eq!(first.alpha.dtype, DType::Q8_0);
    assert_eq!(first.out.dtype, DType::Q5_K);
}

#[test]
fn attention_block_roles_have_the_expected_geometry() {
    let Some((file, profile)) = model_and_profile() else {
        common::skip("model not found");
        return;
    };
    let binding = bind_qwen35(&file, &profile).expect("the real model binds");

    let mut checked = 0;
    for (il, layer) in binding.layers.iter().enumerate() {
        let Some(attn) = &layer.attn else {
            continue;
        };
        check_dims(&attn.q, &[HIDDEN, ATTN_Q], il, "q");
        check_dims(&attn.k, &[HIDDEN, ATTN_KV], il, "k");
        check_dims(&attn.v, &[HIDDEN, ATTN_KV], il, "v");
        check_dims(&attn.q_norm, &[ATTN_HEAD], il, "q_norm");
        check_dims(&attn.k_norm, &[ATTN_HEAD], il, "k_norm");
        check_dims(&attn.output, &[ATTN_OUT, HIDDEN], il, "output");
        assert_eq!(attn.q_norm.dtype, DType::F32, "layer {il} q_norm");
        assert_eq!(attn.k_norm.dtype, DType::F32, "layer {il} k_norm");
        checked += 1;
    }
    assert_eq!(checked, 8);

    let first = binding
        .layers
        .iter()
        .find_map(|layer| layer.attn.as_ref())
        .expect("at least one attention layer");
    assert_eq!(first.q.dtype, DType::Q4_K);
    assert_eq!(first.k.dtype, DType::Q4_K);
    assert_eq!(first.v.dtype, DType::Q6_K);
    assert_eq!(first.output.dtype, DType::Q4_K);
}

#[test]
fn shared_roles_and_payloads_are_consistent() {
    let Some((file, profile)) = model_and_profile() else {
        common::skip("model not found");
        return;
    };
    let binding = bind_qwen35(&file, &profile).expect("the real model binds");
    let mapping = file.mapping();
    let bytes: &[u8] = (*mapping).as_ref();
    let mapping_len = u64::try_from(bytes.len()).expect("mapping length fits in u64");

    for (il, layer) in binding.layers.iter().enumerate() {
        check_dims(&layer.input_norm, &[HIDDEN], il, "input_norm");
        check_dims(&layer.post_norm, &[HIDDEN], il, "post_norm");
        assert_eq!(layer.input_norm.dtype, DType::F32, "layer {il} input_norm");
        assert_eq!(layer.post_norm.dtype, DType::F32, "layer {il} post_norm");
        check_dims(&layer.ffn.gate, &[HIDDEN, INTERMEDIATE], il, "ffn_gate");
        check_dims(&layer.ffn.up, &[HIDDEN, INTERMEDIATE], il, "ffn_up");
        check_dims(&layer.ffn.down, &[INTERMEDIATE, HIDDEN], il, "ffn_down");

        for view in layer_views(layer) {
            assert!(!view.data.is_empty(), "layer {il} {} is empty", view.name);
            let desc = file
                .find(view.name)
                .unwrap_or_else(|| panic!("{} missing from the directory", view.name));
            assert_eq!(
                view.data.len(),
                usize::try_from(desc.bytes).expect("bytes fit")
            );
            assert!(
                desc.abs_offset + desc.bytes <= mapping_len,
                "{} ends past the file",
                desc.name
            );
        }
    }
}

/// Every tensor a bound layer owns.
fn layer_views<'a>(layer: &BoundLayer<'a>) -> Vec<TensorView<'a>> {
    let mut views = vec![
        layer.input_norm,
        layer.post_norm,
        layer.ffn.gate,
        layer.ffn.up,
        layer.ffn.down,
    ];
    if let Some(gdn) = &layer.gdn {
        views.extend([
            gdn.qkv,
            gdn.gate,
            gdn.conv1d,
            gdn.a,
            gdn.alpha,
            gdn.beta,
            gdn.dt_bias,
            gdn.norm,
            gdn.out,
        ]);
    }
    if let Some(attn) = &layer.attn {
        views.extend([
            attn.q,
            attn.k,
            attn.v,
            attn.q_norm,
            attn.k_norm,
            attn.output,
        ]);
    }
    views
}

/// Asserts a view's dimensions, naming the layer and role when they differ.
fn check_dims(view: &TensorView<'_>, want: &[u64], il: usize, role: &str) {
    assert_eq!(
        view.shape.dims(),
        want,
        "layer {il} role {role} ({} bytes, {})",
        view.data.len(),
        view.dtype.type_name()
    );
}
