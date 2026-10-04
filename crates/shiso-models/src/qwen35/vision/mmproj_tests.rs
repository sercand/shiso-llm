use super::*;

#[test]
fn every_manifest_role_has_a_projector_name() {
    let dims = VisionDims::try_from(&crate::qwen35::vision::testing::vision_config())
        .expect("the recorded tower has dimensions");
    let manifest = super::super::manifest::TowerManifest::build(SIDECAR_PREFIX, dims);
    for name in manifest.names() {
        if is_patch_weight(name) {
            continue;
        }
        let stored = gguf_name(name).expect("every role maps");
        assert!(
            stored.starts_with("v.") || stored.starts_with("mm."),
            "{name} -> {stored}"
        );
    }
}

#[test]
fn block_roles_map_to_the_projector_spelling() {
    let at = |role: &str| gguf_name(&format!("{SIDECAR_PREFIX}.{role}")).expect("maps");
    assert_eq!(at("blocks.7.attn.qkv.weight"), "v.blk.7.attn_qkv.weight");
    assert_eq!(at("blocks.7.attn.proj.bias"), "v.blk.7.attn_out.bias");
    assert_eq!(at("blocks.0.norm1.weight"), "v.blk.0.ln1.weight");
    assert_eq!(at("blocks.23.norm2.bias"), "v.blk.23.ln2.bias");
    assert_eq!(
        at("blocks.3.mlp.linear_fc1.weight"),
        "v.blk.3.ffn_up.weight"
    );
    assert_eq!(
        at("blocks.3.mlp.linear_fc2.weight"),
        "v.blk.3.ffn_down.weight"
    );
    assert_eq!(at("merger.norm.weight"), "v.post_ln.weight");
    assert_eq!(at("merger.linear_fc1.weight"), "mm.0.weight");
    assert_eq!(at("merger.linear_fc2.bias"), "mm.2.bias");
    assert_eq!(at("pos_embed.weight"), "v.position_embd.weight");
}

#[test]
fn a_role_the_table_does_not_know_is_refused() {
    let error =
        gguf_name(&format!("{SIDECAR_PREFIX}.blocks.0.attn.q.weight")).expect_err("no such role");
    assert!(
        matches!(error, VisionError::MissingTensor { .. }),
        "{error}"
    );
    let error = gguf_name("vision_tower.pos_embed.weight").expect_err("wrong prefix");
    assert!(
        matches!(error, VisionError::MissingTensor { .. }),
        "{error}"
    );
}

#[test]
fn single_precision_narrows_to_the_upload_width() {
    let data: Vec<u8> = [1.0f32, -2.5, 65504.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let half = convert("t", DType::F32, DType::F16, &data).expect("narrows");
    let back: Vec<f32> = half
        .chunks_exact(2)
        .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();
    assert_eq!(back, vec![1.0, -2.5, 65504.0]);
    let same = convert("t", DType::F16, DType::F16, &half).expect("borrows");
    assert!(matches!(same, Cow::Borrowed(_)));
    let error = convert("t", DType::Q8_0, DType::F16, &data).expect_err("quantized");
    assert!(matches!(error, VisionError::TensorDType { .. }), "{error}");
}
