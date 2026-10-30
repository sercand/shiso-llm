//! The GGUF reader against synthetic headers: the layer pattern the per-block
//! arrays spell, the trailing prediction block, and the switches it refuses.

use std::path::Path;

use shiso_core::CoreError;
use shiso_modelfile::GgufFile;

use super::{Mixer, NemotronHConfig, pair};
use crate::test_gguf::{Kv, Tensor, write_gguf};

/// The checkpoint's 52 decoder layers, `M` state-space, `E` routed, `A`
/// attention, as its own config states them.
const PATTERN: &str = "MEMEMAEMEMEMAEMEMEMAEMEMEMAEMEMEMAEMEMEMEMAEMEMEMEME";

fn mixers() -> Vec<Mixer> {
    PATTERN
        .chars()
        .map(|c| match c {
            'M' => Mixer::Mamba,
            'E' => Mixer::Moe,
            _ => Mixer::Attention,
        })
        .collect()
}

fn tensor(name: &str) -> Tensor {
    Tensor {
        name: name.to_owned(),
        ggml_type: 0,
        dims: vec![1],
        data: vec![0; 4],
    }
}

/// A header with the checkpoint's metadata; `extra` tensors beside the
/// prediction block's projection, and `shared` shared experts. The speech
/// pairing test reads its 2688-wide residual stream too.
pub(crate) fn write(path: &Path, extra: &[&str], shared: u32) {
    let mut kv_heads: Vec<i32> = PATTERN
        .chars()
        .map(|c| if c == 'A' { 2 } else { 0 })
        .collect();
    let mut ffn: Vec<i32> = PATTERN
        .chars()
        .map(|c| if c == 'E' { 1856 } else { 0 })
        .collect();
    // The prediction block: attention and a routed feed-forward in one.
    kv_heads.push(2);
    ffn.push(1856);
    let kvs = [
        ("general.architecture", Kv::Str("nemotron_h_moe")),
        ("nemotron_h_moe.block_count", Kv::U32(53)),
        ("nemotron_h_moe.embedding_length", Kv::U32(2688)),
        ("nemotron_h_moe.feed_forward_length", Kv::I32s(ffn)),
        ("nemotron_h_moe.attention.head_count", Kv::U32(32)),
        ("nemotron_h_moe.attention.head_count_kv", Kv::I32s(kv_heads)),
        ("nemotron_h_moe.attention.key_length", Kv::U32(128)),
        (
            "nemotron_h_moe.attention.layer_norm_rms_epsilon",
            Kv::F32(1e-5),
        ),
        ("nemotron_h_moe.expert_count", Kv::U32(128)),
        ("nemotron_h_moe.expert_used_count", Kv::U32(6)),
        ("nemotron_h_moe.expert_group_count", Kv::U32(1)),
        ("nemotron_h_moe.expert_group_used_count", Kv::U32(1)),
        ("nemotron_h_moe.vocab_size", Kv::U32(131_072)),
        ("nemotron_h_moe.rope.dimension_count", Kv::U32(84)),
        ("nemotron_h_moe.ssm.conv_kernel", Kv::U32(4)),
        ("nemotron_h_moe.ssm.state_size", Kv::U32(128)),
        ("nemotron_h_moe.ssm.group_count", Kv::U32(8)),
        ("nemotron_h_moe.ssm.inner_size", Kv::U32(4096)),
        ("nemotron_h_moe.ssm.time_step_rank", Kv::U32(64)),
        ("nemotron_h_moe.expert_feed_forward_length", Kv::U32(1856)),
        (
            "nemotron_h_moe.expert_shared_feed_forward_length",
            Kv::U32(3712),
        ),
        ("nemotron_h_moe.expert_shared_count", Kv::U32(shared)),
        ("nemotron_h_moe.expert_weights_norm", Kv::Bool(true)),
        ("nemotron_h_moe.expert_weights_scale", Kv::F32(2.5)),
        ("nemotron_h_moe.nextn_predict_layers", Kv::U32(1)),
        ("tokenizer.ggml.eos_token_id", Kv::U32(11)),
    ];
    let mut tensors = vec![tensor("blk.52.nextn.eh_proj.weight")];
    tensors.extend(extra.iter().map(|n| tensor(n)));
    write_gguf(path, &kvs, &tensors);
}

fn read(extra: &[&str], shared: u32) -> Result<NemotronHConfig, CoreError> {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("decoder.gguf");
    write(&path, extra, shared);
    let file = GgufFile::open(&path).expect("opens");
    NemotronHConfig::from_gguf(&file)
}

#[test]
fn gguf_pattern_pairs_like_the_hf_config() {
    let config = read(&[], 1).expect("reads");
    assert_eq!(config.mixers(), mixers().as_slice());
    assert_eq!(config.mtp_block(), Some(52));
    assert_eq!(config.slots(), pair(&mixers()).expect("pairs").as_slice());
    let shape = config.shape();
    assert_eq!(shape.n_layers(), 29);
    assert_eq!(
        (shape.hidden, shape.vocab, shape.intermediate),
        (2688, 131_072, 3712)
    );
    let moe = shape.moe.expect("routed");
    assert_eq!((moe.experts, moe.used, moe.intermediate), (128, 6, 1856));
    let ssm = shape.ssm.expect("state-space");
    assert_eq!(
        (ssm.heads, ssm.head_dim, ssm.state, ssm.groups),
        (64, 64, 128, 8)
    );
    assert!((config.routed_scale() - 2.5).abs() < f32::EPSILON);
    assert!(config.renormalize());
    assert_eq!(config.stop_tokens()[0].get(), 11);
}

#[test]
fn gguf_with_gated_experts_is_refused() {
    let err = read(&["blk.1.ffn_gate_exps.weight"], 1).expect_err("gated experts");
    assert!(err.to_string().contains("ffn_gate_exps"), "{err}");
}

#[test]
fn gguf_with_two_shared_experts_is_refused() {
    let err = read(&[], 2).expect_err("two shared experts");
    assert!(err.to_string().contains("expert_shared_count"), "{err}");
}
