//! The container's own arithmetic, over headers written here.
//!
//! Nothing in this file needs a checkpoint: a safetensors header is eight
//! length bytes and some JSON, which a test can write in full. The cases that
//! need real shards are the modelopt container's, and they are there.

use std::io::Write;

use super::{RawTensor, Shard, StDType};
use crate::reader::GgufError;

/// Writes a shard with `header` as its JSON table and `payload` behind it.
fn shard(header: &str, payload: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let path = dir.path().join("model.safetensors");
    let mut file = std::fs::File::create(&path).expect("create");
    let bytes = header.as_bytes();
    file.write_all(&u64::try_from(bytes.len()).expect("fits").to_le_bytes())
        .expect("length");
    file.write_all(bytes).expect("header");
    file.write_all(payload).expect("payload");
    (dir, path)
}

#[test]
fn every_spelling_both_containers_meet_parses_to_its_own_width() {
    assert_eq!(StDType::parse("BF16"), Some(StDType::Bf16));
    assert_eq!(StDType::parse("F16"), Some(StDType::F16));
    assert_eq!(StDType::parse("F32"), Some(StDType::F32));
    assert_eq!(StDType::parse("U32"), Some(StDType::U32));
    assert_eq!(StDType::parse("U8"), Some(StDType::U8));
    assert_eq!(StDType::parse("F8_E4M3"), Some(StDType::F8E4M3));
    assert_eq!(StDType::parse("F8_E5M2"), None);
    assert_eq!(StDType::parse("I64"), None);

    // A byte is a byte whatever it means. An `F8_E4M3` element is one byte and
    // so is a `U8`, which for NVFP4 holds TWO weights - so a header's element
    // count for such a tensor is half the matrix's width, and the shape it
    // declares is the packed one.
    assert_eq!(StDType::U8.size(), 1);
    assert_eq!(StDType::F8E4M3.size(), 1);
    assert_eq!(StDType::Bf16.size(), 2);
    assert_eq!(StDType::F32.size(), 4);
    assert_eq!(StDType::U32.size(), 4);
}

#[test]
fn a_rank_zero_shape_is_one_element_and_not_none() {
    // This checkpoint writes `[]` for every per-tensor scalar scale - 30,841 of
    // them - and the payload span beside each is four bytes. A reader that made
    // an empty product zero would refuse every one of them on a size mismatch.
    let raw = RawTensor {
        dtype: "F32".to_string(),
        shape: vec![],
        data_offsets: [0, 4],
    };
    assert_eq!(raw.elements().expect("an empty product is one"), 1);
}

#[test]
fn a_header_from_another_producer_is_refused_by_its_own_tag() {
    let (_dir, path) = shard(r#"{"__metadata__":{"format":"pt"}}"#, &[]);
    match Shard::open(&path, "mlx").expect_err("pt is not mlx") {
        GgufError::ArchMismatch { want, got } => {
            assert_eq!(want, "mlx");
            assert_eq!(&*got, "pt");
        }
        other => panic!("expected ArchMismatch, got {other:?}"),
    }
    // And the same file opens for the container that asked for that tag.
    let (shard, tensors) = Shard::open(&path, "pt").expect("pt is pt");
    assert!(tensors.is_empty());
    // A header-only shard: the mapping covers the length prefix and the JSON
    // and nothing else, so it is not empty and has no payload.
    assert_eq!(shard.len(), 8 + r#"{"__metadata__":{"format":"pt"}}"#.len());
}

#[test]
fn a_header_with_no_producer_tag_is_refused() {
    let (_dir, path) = shard(
        r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        &[0; 4],
    );
    match Shard::open(&path, "pt").expect_err("no tag at all") {
        GgufError::ArchMismatch { got, .. } => assert!(got.contains("no producer tag"), "{got}"),
        other => panic!("expected ArchMismatch, got {other:?}"),
    }
}

#[test]
fn a_shape_that_disagrees_with_its_payload_span_is_refused_by_name() {
    // The load-bearing check. A span inside the file that is the wrong length
    // for the shape beside it reads finite numbers from the wrong offset, which
    // is the failure this repository's own handoff records eleven times.
    let (_dir, path) = shard(
        r#"{"__metadata__":{"format":"pt"},
            "w":{"dtype":"F8_E4M3","shape":[4,4],"data_offsets":[0,8]}}"#,
        &[0; 8],
    );
    match Shard::open(&path, "pt").expect_err("sixteen bytes, not eight") {
        GgufError::SizeMismatch { tensor, want, got } => {
            assert_eq!(&*tensor, "w");
            assert_eq!((want, got), (16, 8));
        }
        other => panic!("expected SizeMismatch, got {other:?}"),
    }
}

#[test]
fn a_span_past_the_end_of_the_file_is_refused() {
    let (_dir, path) = shard(
        r#"{"__metadata__":{"format":"pt"},
            "w":{"dtype":"U8","shape":[64],"data_offsets":[0,64]}}"#,
        &[0; 8],
    );
    match Shard::open(&path, "pt").expect_err("sixty-four bytes are not there") {
        GgufError::OutOfBounds { tensor } => assert_eq!(&*tensor, "w"),
        other => panic!("expected OutOfBounds, got {other:?}"),
    }
}

#[test]
fn a_format_no_container_reads_is_refused_naming_the_spelling() {
    let (_dir, path) = shard(
        r#"{"__metadata__":{"format":"pt"},
            "w":{"dtype":"F8_E5M2","shape":[8],"data_offsets":[0,8]}}"#,
        &[0; 8],
    );
    match Shard::open(&path, "pt").expect_err("no decoder reads e5m2") {
        GgufError::LayoutMismatch { tensor, got, .. } => {
            assert_eq!(&*tensor, "w");
            assert_eq!(&*got, "F8_E5M2");
        }
        other => panic!("expected LayoutMismatch, got {other:?}"),
    }
}

#[test]
fn the_payload_base_is_past_the_header_and_spans_are_relative_to_it() {
    let (_dir, path) = shard(
        r#"{"__metadata__":{"format":"pt"},
            "w":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},
            "x":{"dtype":"U8","shape":[4],"data_offsets":[4,8]}}"#,
        &[1, 2, 3, 4, 5, 6, 7, 8],
    );
    let (shard, tensors) = Shard::open(&path, "pt").expect("a well-formed shard");
    assert_eq!(tensors.len(), 2);
    assert_eq!(shard.payload(0, 4).expect("in bounds"), &[1, 2, 3, 4]);
    assert_eq!(shard.payload(4, 8).expect("in bounds"), &[5, 6, 7, 8]);
    shard.payload(4, 12).expect_err("past the payload");
}
