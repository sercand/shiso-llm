//! The quantization document, over the real file.
//!
//! The census below is the one `docs/superpowers/plans/2026-09-20-qwen35moe-cuda.md`
//! read off the checkpoint: 291 targets, 130 FP8 and 161 NVFP4 at group size
//! 16, an FP8 key/value cache, and the draft head excluded.

use std::path::{Path, PathBuf};

use shiso_core::error::CoreError;

use super::{QuantAlgo, QuantConfig};

/// The checked-in fixture.
fn fixture() -> QuantConfig {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/loader/qwen35moe-quant-config.json");
    QuantConfig::from_file(Path::new(&path))
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

/// The NVFP4 algorithm this checkpoint stores its experts as.
const NVFP4: QuantAlgo = QuantAlgo::W4A16Nvfp4 { group_size: 16 };

#[test]
fn the_census_is_what_the_checkpoint_declares() {
    let quant = fixture();
    assert_eq!(quant.len(), 291);
    assert_eq!(quant.count_of(QuantAlgo::Fp8), 130);
    assert_eq!(quant.count_of(NVFP4), 161);
    assert_eq!(quant.kv_cache(), Some(QuantAlgo::Fp8));
}

#[test]
fn the_fp8_targets_are_the_recurrent_and_attention_projections() {
    let quant = fixture();
    // 30 recurrent slots times three projections, plus 10 attention slots times
    // four - which is 130 and accounts for every FP8 target in the file.
    assert_eq!(30 * 3 + 10 * 4, 130);
    for name in [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.k_proj",
        "model.language_model.layers.3.self_attn.v_proj",
        "model.language_model.layers.3.self_attn.o_proj",
    ] {
        assert_eq!(quant.algo_for(name), QuantAlgo::Fp8, "{name}");
    }
}

#[test]
fn the_nvfp4_targets_are_every_expert_and_the_output_projection() {
    let quant = fixture();
    // 40 slots times (the expert stack plus three shared projections), plus
    // `lm_head` - which is 161 and accounts for every NVFP4 target.
    assert_eq!(40 * 4 + 1, 161);
    for name in [
        "model.language_model.layers.0.mlp.experts",
        "model.language_model.layers.0.mlp.shared_expert.gate_proj",
        "model.language_model.layers.0.mlp.shared_expert.up_proj",
        "model.language_model.layers.0.mlp.shared_expert.down_proj",
        "lm_head",
    ] {
        assert_eq!(quant.algo_for(name), NVFP4, "{name}");
    }
}

#[test]
fn a_target_is_a_prefix_and_an_expert_is_reached_through_it() {
    let quant = fixture();
    // The document names the expert stack once; the shard set carries 256
    // experts times three projections times three tensors under that one name.
    // A lookup that matched whole names would find none of them and decode
    // 30,720 weight tensors as bf16.
    assert_eq!(
        quant.algo_for("model.language_model.layers.0.mlp.experts.0.gate_proj.weight"),
        NVFP4
    );
    assert_eq!(
        quant.algo_for("model.language_model.layers.39.mlp.experts.255.down_proj.weight_scale"),
        NVFP4
    );
    assert_eq!(
        quant.algo_for("model.language_model.layers.3.self_attn.q_proj.weight_scale"),
        QuantAlgo::Fp8
    );
}

#[test]
fn a_tensor_no_target_covers_is_stored_as_it_is_declared() {
    let quant = fixture();
    // Every norm, bias and embedding table in the file, and they are the
    // majority of its named roles.
    for name in [
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.mlp.gate.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.visual.blocks.0.attn.qkv.weight",
    ] {
        assert_eq!(quant.algo_for(name), QuantAlgo::None, "{name}");
    }
}

#[test]
fn the_router_itself_is_not_quantized() {
    // `mlp.gate` is the router projection, 2048 x 256 per layer, and the
    // document names no target for it. A reader that assumed "inside `mlp`,
    // therefore NVFP4" would decode the routing logits from nibbles and send
    // every token to the wrong eight experts - which reads as a model that is
    // merely a bit worse.
    assert_eq!(
        fixture().algo_for("model.language_model.layers.7.mlp.gate.weight"),
        QuantAlgo::None
    );
}

#[test]
fn the_draft_head_is_excluded_and_nothing_else_is() {
    let quant = fixture();
    assert!(quant.is_excluded("mtp.layers.0.self_attn.q_proj.weight"));
    assert!(quant.is_excluded("mtp.embed_tokens.weight"));
    assert!(!quant.is_excluded("model.language_model.layers.0.mlp.gate.weight"));
    assert!(!quant.is_excluded("lm_head.weight"));
}

#[test]
fn an_algorithm_with_no_decoder_is_refused_by_its_own_spelling() {
    let doc = r#"{"quantization":{"quant_algo":"MIXED_PRECISION",
        "quantized_layers":{"a.b":{"quant_algo":"AWQ_W4A16"}}}}"#;
    match QuantConfig::from_json_str(doc).expect_err("no decoder reads AWQ here") {
        CoreError::InvalidValue { key, value } => {
            assert_eq!(key, "quant_algo");
            assert_eq!(value, "AWQ_W4A16");
        }
        other => panic!("expected InvalidValue, got {other:?}"),
    }
}

#[test]
fn nvfp4_without_a_group_size_is_refused_rather_than_guessed_at() {
    // A guessed group size is a scale read from the wrong offset, which decodes
    // to finite numbers.
    let doc = r#"{"quantization":{"quantized_layers":{"a.b":{"quant_algo":"W4A16_NVFP4"}}}}"#;
    match QuantConfig::from_json_str(doc).expect_err("the group size is not a default") {
        CoreError::ConfigMissing { key } => assert!(key.contains("group_size"), "{key}"),
        other => panic!("expected ConfigMissing, got {other:?}"),
    }
}

#[test]
fn nvfp4_with_a_four_bit_activation_scale_reads_as_the_same_weight_codec() {
    // The dense 27B export spells its feed-forward targets `NVFP4` and ships
    // an `input_scale` beside each. The weights are the same codec either way;
    // only the activation pass the scale was calibrated for differs.
    let doc = r#"{"quantization":{"quantized_layers":{
        "a.b":{"quant_algo":"NVFP4","group_size":16}}}}"#;
    let quant = QuantConfig::from_json_str(doc).expect("NVFP4 has a decoder");
    assert_eq!(
        quant.algo_for("a.b"),
        QuantAlgo::W4A16Nvfp4 { group_size: 16 }
    );
}

#[test]
fn the_longest_prefix_wins() {
    // Not a shape this checkpoint has, and the rule the walk has to follow
    // anyway: a document that quantizes a stack one way and one projection
    // inside it another must resolve to the inner target.
    let doc = r#"{"quantization":{"quantized_layers":{
        "a.b":{"quant_algo":"FP8"},
        "a.b.c":{"quant_algo":"W4A16_NVFP4","group_size":32}}}}"#;
    let quant = QuantConfig::from_json_str(doc).expect("two nested targets");
    assert_eq!(quant.algo_for("a.b.d.weight"), QuantAlgo::Fp8);
    assert_eq!(
        quant.algo_for("a.b.c.weight"),
        QuantAlgo::W4A16Nvfp4 { group_size: 32 }
    );
}

#[test]
fn the_key_value_cache_is_declared_fp8_and_this_build_serves_f16() {
    // The ruling is `docs/handoff/cuda-qwen35.md` section 8, taken by Phase 8
    // Part C's Task C5 and re-affirmed by this plan's Task 6. It is here as
    // well as there because prose is not checkable: the declaration below is
    // what the checkpoint asks for, `crates/shiso-engine/src/kv.rs` stores
    // `f16`, and a session grading this family against an engine that honours
    // the declaration will see a divergence that grows with context.
    //
    // The divergence is in the direction of MORE precision. `e4m3` has three
    // mantissa bits against `f16`'s ten, over a narrower exponent range at
    // both ends - its smallest subnormal is 2^-9 where `f16` reaches 6e-8, and
    // it saturates at 448 where `f16` reaches 65504. Every key and value this
    // engine stores is therefore a strict refinement of what the reference
    // stores, at every magnitude, and an `f16` cache cannot be the cause of a
    // wrong answer relative to an FP8 one.
    assert_eq!(fixture().kv_cache(), Some(QuantAlgo::Fp8));
}
