//! The two readers over the real `Qwen3.6-35B-A3B-Splash` package.
//!
//! Gated on `SHISO_TEST_MODEL_QWEN35MOE_SPLASH`, the way
//! `crates/shiso-models/src/qwen35moe/binding_tests.rs` gates on
//! `SHISO_TEST_MODEL_QWEN35MOE`: the package is ~20 GB and is not in the tree.
//! Every skip says which variable was wanted, because a case that skips and a
//! case that passes print the same colour otherwise.
//!
//! The split between the three cases is deliberate. Parsing `layout.json` is
//! 250 KB of JSON and runs in milliseconds, so it runs whenever the package is
//! there. Verifying the manifest hashes every artifact - 20.9 GB, measured at
//! 69 s here - which is the only reason it is `#[ignore]`d; it is not
//! optional, and it is the case that says the bytes on this machine are the
//! packer's.

use std::path::PathBuf;

use super::{LayerKind, Layout, Manifest, SCHEMA_VERSION, SectionDtype, verify};
use crate::qwen35moe::Qwen35MoeConfig;

/// The variable that points at an unpacked Splash snapshot.
const ENV: &str = "SHISO_TEST_MODEL_QWEN35MOE_SPLASH";

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// The real package, or `None` when it is not on this machine.
fn package() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var(ENV).ok()?);
    path.join("layout.json").is_file().then_some(path)
}

#[test]
fn the_shipped_layout_reads_as_the_package_wrote_it() {
    let Some(root) = package() else {
        skip(&format!("{ENV} is unset or names no layout.json"));
        return;
    };
    let text = std::fs::read_to_string(root.join("layout.json")).expect("layout.json");
    let layout = Layout::from_json(&text).expect("the shipped layout parses");

    // 40 layers, a head and an embedding table; six draft layers and a draft
    // model; one tower, written as a bare object rather than a list of one.
    assert_eq!(layout.target.len(), 42);
    assert_eq!(layout.draft.len(), 7);
    assert_eq!(layout.vision.len(), 1);
    assert_eq!(layout.vision[0].path, "vision/model.bin");

    let layer0 = &layout.target[0];
    assert_eq!(layer0.path, "target/layer-0.bin");
    let gdn = layer0.section("gdn_input").expect("gdn_input");
    assert_eq!(gdn.shape, vec![12544, 2048]);
    assert_eq!(gdn.actual_output, Some(12352));
    assert_eq!(gdn.out_dim(), 12352, "the padded width is not the live one");
    let experts = layer0.section("experts.gate_proj").expect("gate_proj");
    assert_eq!(experts.experts(), Some(256));
    assert_eq!(experts.expert_stride_bytes, Some(589_824));

    // Layer 3 is the first attention slot: one in four, and the recurrent
    // sections are absent from it rather than zeroed.
    let layer3 = &layout.target[3];
    assert!(layer3.section("attention_input").is_some());
    assert!(layer3.section("gdn_input").is_none());

    // The parse above already says every dtype in the package has a reader,
    // since an unknown one is refused by name rather than skipped. This is the
    // one file whose dtype is its own and the only section anywhere that states
    // a row order, so it is the one worth naming.
    let embedding = layout
        .target
        .iter()
        .find(|f| f.path == "target/embedding.bin")
        .expect("embedding.bin");
    let codes = embedding.section("embedding.weights").expect("weights");
    assert_eq!(codes.dtype, SectionDtype::Q4AffineWeights);
    assert_eq!(codes.layout.as_deref(), Some("vocabulary_row_major"));

    // Nothing overlaps and nothing runs off the end of its file: the layout is
    // a set of byte ranges before it is anything else.
    for file in layout
        .target
        .iter()
        .chain(&layout.draft)
        .chain(&layout.vision)
    {
        let mut spans: Vec<(u64, u64)> = file
            .sections
            .iter()
            .map(|s| (s.offset, s.offset + s.bytes))
            .collect();
        spans.sort_unstable();
        for pair in spans.windows(2) {
            assert!(
                pair[0].1 <= pair[1].0,
                "{}: sections overlap at {:?}",
                file.path,
                pair
            );
        }
        if let Some(last) = spans.last() {
            assert!(
                last.1 <= file.bytes,
                "{}: a section runs past the end",
                file.path
            );
        }
    }
}

/// Reads `manifest.json` without hashing anything, and checks its account of
/// the package against `layout.json`'s. Two documents written by the same
/// converter disagreeing is a defect neither one can show on its own, and
/// finding it costs a few hundred KB of JSON rather than 20 GB of SHA-256.
#[test]
fn the_shipped_manifest_states_the_geometry_the_layout_was_packed_to() {
    let Some(root) = package() else {
        skip(&format!("{ENV} is unset or names no layout.json"));
        return;
    };
    let manifest = Manifest::from_json(
        &std::fs::read_to_string(root.join("manifest.json")).expect("manifest.json"),
    )
    .expect("the shipped manifest parses");
    let layout =
        Layout::from_json(&std::fs::read_to_string(root.join("layout.json")).expect("layout.json"))
            .expect("the shipped layout parses");

    assert_eq!(manifest.schema_version, SCHEMA_VERSION);
    assert_eq!(manifest.model, "Qwen3.6-35B-A3B");
    assert_eq!(manifest.format.name, "splash-packed-q4-moe");
    assert_eq!(manifest.format.storage_n, 256);
    assert_eq!(manifest.format.quant_group_size, 64);

    // The layer roll is one entry per layer, and the layout has one file per
    // layer plus the head and the embedding table.
    assert_eq!(manifest.target.layers, 40);
    assert_eq!(
        manifest.target.layer_types.len(),
        manifest.target.layers as usize
    );
    assert_eq!(layout.target.len(), manifest.target.layers as usize + 2);

    // Every section the packer padded was padded to the width `format` states,
    // and every quantized section was grouped at the size it states. This is
    // the cross-check the typed top level exists for: without it both numbers
    // are assumptions a codec makes.
    for file in layout
        .target
        .iter()
        .chain(&layout.draft)
        .chain(&layout.vision)
    {
        for section in &file.sections {
            if let Some(storage_n) = section.storage_n {
                assert_eq!(
                    storage_n, manifest.format.storage_n,
                    "{}/{}: padded to a width the manifest does not state",
                    file.path, section.name
                );
            }
            if let Some(group) = section.group_size {
                assert_eq!(
                    group, manifest.format.quant_group_size,
                    "{}/{}: grouped at a size the manifest does not state",
                    file.path, section.name
                );
            }
            assert_eq!(
                section.offset % manifest.format.section_alignment_bytes,
                0,
                "{}/{}: not on the stated alignment",
                file.path,
                section.name
            );
        }
    }

    // The manifest's two recurrent widths are the section's own, stated the
    // other way round: `gdn_packed_width` is what `shape[0]` says, and
    // `gdn_actual_width` is what `actual_output` says.
    let gdn = layout.target[0].section("gdn_input").expect("gdn_input");
    assert_eq!(gdn.shape[0], manifest.target.gdn_packed_width);
    assert_eq!(gdn.actual_output, Some(manifest.target.gdn_actual_width));
    assert_eq!(gdn.out_dim(), manifest.target.gdn_actual_width);

    // And the roll agrees with which sections each layer file actually holds.
    for (index, kind) in manifest.target.layer_types.iter().enumerate() {
        let file = &layout.target[index];
        match kind {
            LayerKind::Gdn => {
                assert!(file.section("gdn_input").is_some(), "{}", file.path);
                assert!(file.section("attention_input").is_none(), "{}", file.path);
            }
            LayerKind::Attention => {
                assert!(file.section("attention_input").is_some(), "{}", file.path);
                assert!(file.section("gdn_input").is_none(), "{}", file.path);
                let attn = file.section("attention_input").expect("attention_input");
                assert_eq!(attn.shape[0], manifest.target.attention_packed_width);
            }
        }
    }

    // The expert stack is the manifest's expert count at the manifest's inner
    // width, and the shared expert is its own.
    let experts = layout.target[0]
        .section("experts.gate_proj")
        .expect("gate_proj");
    assert_eq!(experts.experts(), Some(manifest.target.experts));
    assert_eq!(experts.out_dim(), manifest.target.moe_intermediate_size);
    let shared = layout.target[0]
        .section("shared_expert.gate_proj")
        .expect("shared gate_proj");
    assert_eq!(
        shared.out_dim(),
        manifest.target.shared_expert_intermediate_size
    );

    // The vocabulary is one number in three places.
    let head = layout
        .target
        .iter()
        .find(|f| f.path == "target/head.bin")
        .expect("head.bin");
    assert_eq!(
        head.section("logits").expect("logits").out_dim(),
        manifest.target.vocabulary_size
    );

    // The draft's geometry matches the draft files the layout lists: six layer
    // files and the model file.
    assert_eq!(layout.draft.len(), manifest.draft.layers as usize + 1);
    assert_eq!(manifest.draft.hidden_size, manifest.target.hidden_size);
}

/// The manifest's account of the model against the upstream checkpoint's own,
/// which is the cross-check [`Qwen35MoeConfig::from_splash`] exists for. Two
/// documents written by two different programs, and nothing but this says they
/// still describe the same model.
#[test]
fn the_shipped_manifest_and_the_upstream_config_state_the_same_model() {
    let Some(root) = package() else {
        skip(&format!("{ENV} is unset or names no layout.json"));
        return;
    };
    // The model config is at `tokenizer/config.json` - despite the path, the
    // whole upstream HF config - and the package root has no `config.json`.
    assert!(!root.join("config.json").exists());
    assert!(root.join("tokenizer").join("config.json").is_file());

    let cfg = Qwen35MoeConfig::from_splash_dir(&root)
        .expect("the manifest and the upstream config agree");
    let shape = cfg.shape();
    assert_eq!(shape.hidden, 2048);
    assert_eq!(shape.n_layers(), 40);
    assert_eq!(shape.vocab, 248_320);
    assert_eq!(cfg.moe().experts, 256);
    assert_eq!(cfg.moe().used, 8);
    assert_eq!(cfg.moe().intermediate, 512);
    assert_eq!(shape.intermediate, 512);
    assert_eq!(cfg.full_attention_interval(), 4);
    // Everything above came out of both documents. This came out of the
    // upstream one alone, because the manifest does not carry it.
    assert_eq!(shape.attn[0].head_dim, 256);
    assert_eq!(shape.attn[0].n_rot, 64);
    assert_eq!(shape.gdn.v_heads, 32);
    assert_eq!(cfg.eos_token().get(), 248_044);
    // And a Splash package carries no quantization document at all.
    assert!(cfg.quant().is_none());
}

#[test]
#[ignore = "hashes the whole 20.9 GB package - 69 s on an M-series laptop, and \
            longer on anything reading from a spinning disk. Run it with \
            --ignored and SHISO_TEST_MODEL_QWEN35MOE_SPLASH set."]
fn the_shipped_bytes_are_the_ones_the_manifest_declares() {
    let Some(root) = package() else {
        skip(&format!("{ENV} is unset or names no layout.json"));
        return;
    };
    let text = std::fs::read_to_string(root.join("manifest.json")).expect("manifest.json");
    let manifest = Manifest::from_json(&text).expect("the shipped manifest parses");
    // 42 target files, 7 draft, the tower, five tokenizer files - and
    // `layout.json` itself, which the manifest covers because a layout nobody
    // checked is a set of byte ranges taken on trust.
    assert_eq!(manifest.artifacts.len(), 56);
    verify(&manifest, &root).expect("every artifact matches its declared digest");
}
