//! Family detection against forged checkpoints: header-only files written by
//! the test, so the dispatch is pinned on any machine with no checkpoint and
//! no device.

use std::path::Path;

use shiso_vision::generator::ImageGenError;

use crate::image_registry::{ImageFamily, ImageModelDesc, detect_image_family};

/// A safetensors file declaring each name as a one-element `F32` tensor.
fn write_names(path: &Path, names: &[&str]) {
    use std::fmt::Write as _;
    let mut header = String::from("{");
    for (i, name) in names.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        let start = i * 4;
        write!(
            header,
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[{start},{}]}}",
            start + 4
        )
        .expect("header text always formats");
    }
    header.push('}');
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.resize(8 + header.len() + names.len() * 4, 0);
    std::fs::write(path, bytes).expect("write safetensors");
}

fn desc_for(path: &Path) -> ImageModelDesc {
    ImageModelDesc {
        model_path: path.to_path_buf(),
        text_encoder_path: None,
        vae_path: None,
        tokenizer_path: None,
        assets_dir: None,
        residency: crate::image_registry::ImageResidency::Auto,
        reserve_bytes: crate::image_registry::DEFAULT_IMAGE_RESERVE_BYTES,
        draft_steps: 0,
        final_reuse: None,
    }
}

fn detect(names: &[&str]) -> Result<ImageFamily, ImageGenError> {
    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join("model.safetensors");
    write_names(&file, names);
    detect_image_family(&desc_for(&file))
}

#[test]
fn flux2_names_select_flux2() {
    let family = detect(&[
        "img_in.weight",
        "txt_in.weight",
        "double_blocks.0.img_attn.qkv.weight",
        "single_blocks.0.linear1.weight",
    ])
    .expect("flux2");
    assert_eq!(family, ImageFamily::Flux2);
}

#[test]
fn a_directory_of_shards_is_read_as_one_checkpoint() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_names(&dir.path().join("a.safetensors"), &["img_in.weight"]);
    write_names(
        &dir.path().join("b.safetensors"),
        &["single_blocks.0.linear1.weight"],
    );
    assert_eq!(
        detect_image_family(&desc_for(dir.path())).expect("flux2"),
        ImageFamily::Flux2
    );
}

#[test]
fn qwen_names_select_qwen_with_and_without_the_prefix() {
    for prefix in ["", "model.diffusion_model."] {
        let q = format!("{prefix}transformer_blocks.0.attn.to_q.weight");
        let m = format!("{prefix}transformer_blocks.0.img_mod.1.weight");
        let m2 = format!("{prefix}transformer_blocks.0.modulation.1.weight");
        // The shipped export: one modulation shared by every block.
        let shared = format!("{prefix}modulation.1.weight");
        for modulation in [&m, &m2, &shared] {
            let family = detect(&[&q, modulation]).expect("qwen");
            assert_eq!(family, ImageFamily::QwenImage21, "prefix {prefix:?}");
        }
    }
}

#[test]
fn an_unknown_checkpoint_is_refused_naming_both_signatures() {
    let error = detect(&["img_in.weight", "something.else"]).expect_err("unknown");
    let ImageGenError::Checkpoint(message) = error else {
        panic!("expected a checkpoint error");
    };
    assert!(message.contains("double_blocks"), "{message}");
    assert!(
        message.contains("transformer_blocks.0.attn.to_q.weight"),
        "{message}"
    );
}

#[test]
fn a_checkpoint_matching_both_is_refused() {
    let error = detect(&[
        "img_in.weight",
        "double_blocks.0.x",
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.modulation.1.weight",
    ])
    .expect_err("ambiguous");
    assert!(matches!(error, ImageGenError::Checkpoint(_)));
}

#[test]
fn detects_flux3_from_tensor_names() {
    for prefix in ["", "model.diffusion_model."] {
        let image = format!("{prefix}content_mode_blocks.image.0.q_proj.weight");
        let text = format!("{prefix}txt_mode_blocks.0.q_proj.weight");
        assert_eq!(
            detect(&[
                &image,
                &text,
                "single_blocks.0.q_proj.weight",
                "txt_in.weight"
            ])
            .expect("flux3"),
            ImageFamily::Flux3,
            "prefix {prefix:?}"
        );
    }
}

#[test]
fn flux3_and_flux2_signatures_do_not_overlap() {
    // FLUX.3 keeps `single_blocks.*` and `txt_in.weight`, the names the older
    // family also uses, so neither may claim the other's checkpoint.
    let flux3 = [
        "content_mode_blocks.image.0.q_proj.weight",
        "txt_mode_blocks.0.q_proj.weight",
        "single_blocks.0.q_proj.weight",
        "txt_in.weight",
        "emb_in.image.weight",
    ];
    assert_eq!(detect(&flux3).expect("flux3"), ImageFamily::Flux3);
    let flux2 = [
        "img_in.weight",
        "txt_in.weight",
        "double_blocks.0.img_attn.qkv.weight",
        "single_blocks.0.linear1.weight",
    ];
    assert_eq!(detect(&flux2).expect("flux2"), ImageFamily::Flux2);
    // Only half of the flux3 signature is not a family.
    assert!(
        detect(&[
            "content_mode_blocks.image.0.q_proj.weight",
            "single_blocks.0.q_proj.weight"
        ])
        .is_err()
    );
    // Both signatures at once are refused, naming both.
    let mut both = flux3.to_vec();
    both.extend(flux2);
    let ImageGenError::Checkpoint(message) = detect(&both).expect_err("ambiguous") else {
        panic!("expected a checkpoint error");
    };
    assert!(
        message.contains("flux2") && message.contains("flux3"),
        "{message}"
    );
}
