//! Parses the real `Q4_K_M` checkpoint and pins everything a downstream crate
//! depends on: header geometry, the metadata table, the per-format tensor mix,
//! and the whole tensor directory against a checked-in fixture.
//!
//! Skips when no local model is present. The fixture in
//! `test-vectors/loader/qwen35-q4km-manifest.tsv` is the directory as parsed, so this
//! is a regression gate on the parser rather than a restatement of it: the
//! numbers below were read off the file, not assumed.

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use shiso_core::DType;
use shiso_modelfile::TensorDesc;
use shiso_modelfile::quant::{BlockQ4K, BlockQ5K, BlockQ6K};
use shiso_modelfile::reader::GgufError;

/// Expected payload alignment of the test checkpoint.
const ALIGNMENT: u64 = 32;
/// Expected header format version.
const VERSION: u32 = 3;
/// Expected tensor count.
const N_TENSORS: usize = 426;
/// Expected metadata key count.
const N_KV: usize = 46;
/// Expected vocabulary size.
const VOCAB: u64 = 248_320;
/// Expected residual-stream width.
const HIDDEN: u64 = 2560;

/// The checked-in directory listing.
fn manifest_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/loader/qwen35-q4km-manifest.tsv")
}

#[test]
fn header_and_directory_geometry() {
    let Some(file) = common::open_model() else {
        common::skip("model not found");
        return;
    };

    assert_eq!(file.version(), VERSION);
    assert_eq!(file.alignment(), ALIGNMENT);
    assert_eq!(file.tensors().len(), N_TENSORS);
    assert_eq!(file.metadata().len(), N_KV);
    assert_eq!(file.metadata().architecture(), Some("qwen35"));

    // The payload base sits past the directory, and the first tensor starts
    // exactly there: nothing is packed between the directory and its payloads.
    let first = &file.tensors()[0];
    assert_eq!(first.abs_offset, file.tensor_data_pos());
    assert!(file.tensor_data_pos() > 0);
}

#[test]
fn format_mix_is_the_q4_k_m_mix() {
    let Some(file) = common::open_model() else {
        common::skip("model not found");
        return;
    };

    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for desc in file.tensors() {
        *counts.entry(desc.dtype.type_name()).or_default() += 1;
    }

    let want: BTreeMap<&'static str, usize> = [
        ("f32", 177),
        ("q8_0", 48),
        ("q4_k", 131),
        ("q5_k", 48),
        ("q6_k", 22),
    ]
    .into_iter()
    .collect();
    assert_eq!(counts, want);
    assert_eq!(counts.values().sum::<usize>(), N_TENSORS);
}

#[test]
fn embeddings_are_tied_and_six_bit() {
    let Some(file) = common::open_model() else {
        common::skip("model not found");
        return;
    };

    let embd = file.find("token_embd.weight").expect("embedding tensor");
    assert_eq!(embd.dtype, DType::Q6_K);
    assert_eq!(embd.shape.dims(), &[HIDDEN, VOCAB]);

    let norm = file.find("output_norm.weight").expect("output norm");
    assert_eq!(norm.dtype, DType::F32);
    assert_eq!(norm.shape.dims(), &[HIDDEN]);

    // This checkpoint ties its logits to the embedding, so there is no separate
    // output projection to bind.
    assert!(file.find("output.weight").is_none());
}

#[test]
fn every_view_matches_its_descriptor() {
    let Some(file) = common::open_model() else {
        common::skip("model not found");
        return;
    };

    let path = common::model_path().expect("the caller already checked the model exists");
    let on_disk = std::fs::metadata(&path).expect("model statable").len();

    let mapping = file.mapping();
    let bytes: &[u8] = (*mapping).as_ref();
    let len = bytes.len();
    assert_eq!(
        u64::try_from(len).expect("mapping length fits in u64"),
        on_disk,
        "the mapping must be the whole file"
    );
    for desc in file.tensors() {
        let view = file
            .get(&desc.name)
            .unwrap_or_else(|| panic!("{} vanished from the index", desc.name));
        assert_eq!(view.name, &*desc.name);
        assert_eq!(view.dtype, desc.dtype);
        assert_eq!(view.shape, desc.shape);
        assert_eq!(
            view.data.len(),
            usize::try_from(desc.bytes).expect("descriptor fits in usize"),
            "{} payload length disagrees with its descriptor",
            desc.name
        );

        let range = file.range_of(desc);
        assert_eq!(
            range.start,
            usize::try_from(desc.abs_offset).expect("offset fits")
        );
        assert!(
            range.end <= len,
            "{} ends at {} past the {}-byte mapping",
            desc.name,
            range.end,
            len
        );
        assert_eq!(&bytes[range], view.data, "{} range mismatch", desc.name);
    }
}

#[test]
fn directory_matches_the_checked_in_manifest() {
    let Some(file) = common::open_model() else {
        common::skip("model not found");
        return;
    };

    let path = manifest_path();
    assert!(
        path.is_file(),
        "missing fixture {}; regenerate it with the dump_manifest example",
        path.display()
    );
    let fixture = std::fs::read_to_string(&path).expect("fixture readable");

    let base = file.tensor_data_pos();
    let actual = file
        .tensors()
        .iter()
        .map(|desc| manifest_line(desc, base))
        .collect::<Vec<_>>()
        .join("\n");
    let expected = fixture.trim_end_matches('\n');

    assert_eq!(actual.lines().count(), N_TENSORS);
    assert_eq!(actual, expected, "manifest differs from {}", path.display());
}

#[test]
fn tokenizer_arrays_stay_lazy() {
    let Some(file) = common::open_model() else {
        common::skip("model not found");
        return;
    };

    // The API shape is the assertion: the vocabulary is reachable as a count
    // without a single per-token allocation having happened.
    let tokens = file
        .metadata()
        .get_array("tokenizer.ggml.tokens")
        .expect("vocabulary array");
    assert_eq!(tokens.len(), usize::try_from(VOCAB).expect("vocab fits"));
    assert!(!tokens.is_empty());

    let merges = file
        .metadata()
        .get_array("tokenizer.ggml.merges")
        .expect("merges array");
    assert!(merges.len() > 200_000);

    // And an element decodes on demand.
    let first = tokens.get(0).expect("first token");
    assert!(matches!(first, shiso_modelfile::MetaValue::Str(_)));
}

#[test]
fn typed_block_access_works_on_real_payloads() {
    let Some(file) = common::open_model() else {
        common::skip("model not found");
        return;
    };

    // The embedding is a whole number of 256-element blocks, so the cast is
    // exact and the block count is derivable from the declared shape.
    let embd = file.require("token_embd.weight").expect("embedding");
    let blocks: &[BlockQ6K] = embd.blocks().expect("q6_k blocks");
    let elems = embd.shape.elements().expect("element product");
    assert_eq!(
        blocks.len(),
        usize::try_from(elems.div_ceil(256)).expect("count fits")
    );
    assert_eq!(
        size_of_val(blocks),
        embd.data.len(),
        "the payload is exactly whole blocks"
    );

    // A K-quant projection, and the wrong format is refused rather than cast.
    let qkv = file.require("blk.0.attn_qkv.weight").expect("qkv");
    let qkv_blocks: &[BlockQ5K] = qkv.blocks().expect("q5_k blocks");
    assert!(!qkv_blocks.is_empty());
    assert!(matches!(
        qkv.blocks::<BlockQ4K>(),
        Err(GgufError::TypeMismatch { .. })
    ));

    // A normalization vector reads as plain f32, and as nothing else.
    let norm = file.require("output_norm.weight").expect("output norm");
    let weights = norm.as_f32().expect("f32 weights");
    assert_eq!(
        u64::try_from(weights.len()).expect("length fits in u64"),
        HIDDEN
    );
    assert!(norm.as_f16().is_err());
}

/// Renders one tensor the way `examples/dump_manifest.rs` does.
fn manifest_line(desc: &TensorDesc, base: u64) -> String {
    let dims = desc
        .shape
        .dims()
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let rel = desc.abs_offset.saturating_sub(base);
    format!(
        "{}\t{}\t{}\t{}\t{rel}",
        desc.name,
        desc.shape.ndim(),
        dims,
        desc.dtype.gguf_id(),
    )
}
