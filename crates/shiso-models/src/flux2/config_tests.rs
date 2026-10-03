//! Checkpoint discovery tests against forged checkpoints: the refusals a
//! real loader must not paper over (two exports in one directory, a config
//! that contradicts or cannot describe the tensors, a config that is absent
//! or unparsable) are pinned against files this test writes itself, so they
//! run on any machine with no checkpoint and no device.

use std::path::{Path, PathBuf};

use super::config::*;
use super::weights::StStore;

#[test]
fn count_blocks_from_tensor_map() {
    let names = [
        "double_blocks.0.img_attn.qkv.weight",
        "double_blocks.4.img_attn.qkv.weight",
        "single_blocks.19.linear1.weight",
    ];

    let depth = count_blocks_internal(names.iter().copied(), "double_blocks.").expect("depth");
    assert_eq!(depth, 5);

    let single_depth =
        count_blocks_internal(names.iter().copied(), "single_blocks.").expect("depth");
    assert_eq!(single_depth, 20);
}

/// Bytes one element of a header dtype spelling occupies.
fn dtype_size(dtype: &str) -> u64 {
    match dtype {
        "F32" => 4,
        "BF16" => 2,
        "F8_E4M3" | "U8" => 1,
        other => panic!("test writes no sizes for dtype {other}"),
    }
}

/// Writes a minimal safetensors file: the eight length bytes, the header
/// JSON, and a zero-filled payload sized from each tensor's shape. Discovery
/// and the store index read only headers and spans, so the bytes themselves
/// never matter to these tests.
fn write_safetensors(path: &Path, tensors: &[(&str, &str, &[u64])]) {
    use std::fmt::Write as _;
    let mut entries = String::from("{");
    let mut offset = 0u64;
    let mut payload_len = 0usize;
    for (i, (name, dtype, shape)) in tensors.iter().enumerate() {
        let elements: u64 = shape.iter().product();
        let bytes = elements * dtype_size(dtype);
        let shape_text = shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        if i > 0 {
            entries.push(',');
        }
        write!(
            entries,
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape_text}],\
             \"data_offsets\":[{offset},{}]}}",
            offset + bytes
        )
        .expect("header text always formats");
        offset += bytes;
        payload_len += usize::try_from(bytes).expect("test payloads are tiny");
    }
    entries.push('}');
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    bytes.extend_from_slice(entries.as_bytes());
    bytes.resize(8 + entries.len() + payload_len, 0);
    std::fs::write(path, bytes).expect("write safetensors");
}

/// The signature tensors plus two double blocks and one single block, in
/// `BF16`, under the given file name.
fn write_transformer(dir: &Path, file_name: &str, dtype: &str) -> PathBuf {
    let path = dir.join(file_name);
    write_safetensors(
        &path,
        &[
            ("img_in.weight", dtype, &[4, 2]),
            ("txt_in.weight", dtype, &[4, 3]),
            ("double_blocks.0.img_attn.qkv.weight", dtype, &[12, 4]),
            ("double_blocks.1.img_attn.qkv.weight", dtype, &[12, 4]),
            ("single_blocks.0.linear1.weight", dtype, &[18, 4]),
        ],
    );
    path
}

/// A transformer config describing the `write_transformer` tensors: hidden
/// 4 over 2 heads of 2, with the block depths the caller names.
fn write_transformer_config(dir: &Path, num_layers: usize, num_single_layers: usize) {
    let transformer = dir.join("transformer");
    std::fs::create_dir_all(&transformer).expect("create transformer dir");
    std::fs::write(
        transformer.join("config.json"),
        format!(
            "{{\"attention_head_dim\":2,\"axes_dims_rope\":[1,1,1,1],\
             \"mlp_ratio\":3.0,\"num_attention_heads\":2,\"num_layers\":{num_layers},\
             \"num_single_layers\":{num_single_layers},\"rope_theta\":2000.0}}"
        ),
    )
    .expect("write config");
}

fn desc_for(model_path: &Path) -> ImageModelDesc {
    ImageModelDesc {
        model_path: model_path.to_path_buf(),
        text_encoder_path: None,
        vae_path: None,
        tokenizer_path: None,
        assets_dir: None,
    }
}

#[test]
fn a_second_export_beside_the_selected_weights_is_left_out_not_merged() {
    // A BF16 file beside an FP8 one: both carry the signature tensor, so
    // discovery must settle on one file and hand the store exactly that
    // file - indexing the directory would resolve every shared name to
    // whichever shard was read last, and the dtype scan would call the
    // BF16 weights FP8.
    let dir = tempfile::tempdir().expect("tempdir");
    let bf16 = write_transformer(dir.path(), "model-bf16.safetensors", "BF16");
    write_transformer(dir.path(), "model-fp8.safetensors", "F8_E4M3");
    write_transformer_config(dir.path(), 2, 1);
    for asset in ["text_encoder", "vae", "tokenizer"] {
        std::fs::create_dir(dir.path().join(asset)).expect("create asset dir");
    }

    let located = locate_and_inspect(&desc_for(dir.path())).expect("selection");
    assert_eq!(located.weights_path, bf16);
    assert_eq!(located.weight_files, vec![bf16]);
    assert!(
        !located.params.is_fp8,
        "the FP8 sibling must not flip the selected file's dtype"
    );
    assert_eq!(located.params.depth, 2);
    assert_eq!(located.params.depth_single_blocks, 1);
}

#[test]
fn disjoint_shards_of_one_export_all_join_the_selection() {
    // The other half of the rule: a second file with no name in common is
    // a shard, not a second export, and must reach the store.
    let dir = tempfile::tempdir().expect("tempdir");
    write_transformer(dir.path(), "model-00001-of-00002.safetensors", "BF16");
    write_safetensors(
        &dir.path().join("model-00002-of-00002.safetensors"),
        &[("double_blocks.2.img_attn.qkv.weight", "BF16", &[12, 4])],
    );
    write_transformer_config(dir.path(), 3, 1);
    for asset in ["text_encoder", "vae", "tokenizer"] {
        std::fs::create_dir(dir.path().join(asset)).expect("create asset dir");
    }

    let located = locate_and_inspect(&desc_for(dir.path())).expect("selection");
    assert_eq!(located.weight_files.len(), 2);
    assert_eq!(located.params.depth, 3);
}

#[test]
fn a_config_naming_fewer_blocks_than_the_tensors_carry_is_refused() {
    // The silent truncation this cross-check exists to stop: the config
    // names one double block, the tensors carry two.
    let dir = tempfile::tempdir().expect("tempdir");
    write_transformer(dir.path(), "model.safetensors", "BF16");
    write_transformer_config(dir.path(), 1, 1);

    let err = locate_and_inspect(&desc_for(dir.path())).expect_err("depth mismatch refuses");
    let msg = err.to_string();
    assert!(
        msg.contains("double_blocks") && msg.contains("refusing"),
        "{msg}"
    );
}

#[test]
fn a_checkpoint_with_no_reachable_transformer_config_is_refused_by_value() {
    // No config anywhere: the head split, RoPE parameterization and MLP
    // ratio are not derivable from shapes, so the refusal must name them
    // rather than guess.
    let dir = tempfile::tempdir().expect("tempdir");
    write_transformer(dir.path(), "model.safetensors", "BF16");

    let err = locate_and_inspect(&desc_for(dir.path())).expect_err("missing config refuses");
    let msg = err.to_string();
    for value in [
        "num_attention_heads",
        "axes_dims_rope",
        "rope_theta",
        "mlp_ratio",
    ] {
        assert!(msg.contains(value), "refusal must name {value}: {msg}");
    }
}

#[test]
fn an_unparsable_transformer_config_is_refused_not_guessed_past() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_transformer(dir.path(), "model.safetensors", "BF16");
    let transformer = dir.path().join("transformer");
    std::fs::create_dir_all(&transformer).expect("create transformer dir");
    std::fs::write(transformer.join("config.json"), "{ not json").expect("write config");

    let err = locate_and_inspect(&desc_for(dir.path())).expect_err("broken config refuses");
    assert!(err.to_string().contains("parsing"), "{err}");
}

#[test]
fn the_config_is_found_through_assets_dir_when_the_model_dir_has_none() {
    // The FP8 export layout: weights in one directory, the transformer
    // config only in the BF16 tree named by `assets_dir`.
    let dir = tempfile::tempdir().expect("tempdir");
    let weights = dir.path().join("fp8");
    std::fs::create_dir(&weights).expect("create fp8 dir");
    write_transformer(&weights, "model.safetensors", "F8_E4M3");
    let assets = dir.path().join("bf16");
    write_transformer_config(&assets, 2, 1);
    for asset in ["text_encoder", "vae", "tokenizer"] {
        std::fs::create_dir(weights.join(asset)).expect("create asset dir");
    }

    let mut desc = desc_for(&weights);
    desc.assets_dir = Some(assets);
    let located = locate_and_inspect(&desc).expect("assets_dir config");
    assert_eq!(located.params.num_heads, 2);
    assert!(located.params.is_fp8);
}

#[test]
fn a_tensor_name_two_shards_share_is_refused_naming_tensor_and_files() {
    // The store-level backstop: even if a caller hands `open_files` two
    // exports side by side, the shared name is refused by name, with both
    // files, instead of resolving to the last one indexed.
    let dir = tempfile::tempdir().expect("tempdir");
    let first = dir.path().join("a.safetensors");
    let second = dir.path().join("b.safetensors");
    write_safetensors(&first, &[("img_in.weight", "BF16", &[4, 2])]);
    write_safetensors(&second, &[("img_in.weight", "F8_E4M3", &[4, 2])]);

    let err = StStore::open_files(&[first.clone(), second.clone()])
        .expect_err("a duplicate name across shards refuses");
    let msg = err.to_string();
    assert!(msg.contains("img_in.weight"), "must name the tensor: {msg}");
    assert!(
        msg.contains("a.safetensors") && msg.contains("b.safetensors"),
        "{msg}"
    );
}
