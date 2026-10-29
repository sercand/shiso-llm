//! The pairing against synthetic decoder headers.

use std::path::Path;

use shiso_core::CoreError;

use super::{SPEECH_MODEL_KEY, speech_encoder_for};
use crate::test_gguf::{Kv, write_gguf};

fn decoder(dir: &Path, encoder: Option<&'static str>) -> std::path::PathBuf {
    let path = dir.join("decoder.gguf");
    let mut kvs = vec![("general.architecture", Kv::Str("nemotron_h_moe"))];
    if let Some(name) = encoder {
        kvs.push((SPEECH_MODEL_KEY, Kv::Str(name)));
    }
    write_gguf(&path, &kvs, &[]);
    path
}

#[test]
fn missing_encoder_is_refused_before_load() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = decoder(dir.path(), Some("encoder-f16.gguf"));
    let refused = speech_encoder_for(&path, None);
    let Err(CoreError::InvalidValue { value, .. }) = refused else {
        panic!("a missing encoder is refused, got {refused:?}");
    };
    assert!(
        value.contains(&dir.path().join("encoder-f16.gguf").display().to_string()),
        "the refusal names the path: {value}"
    );
}

#[test]
fn the_encoder_beside_the_decoder_is_found() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = decoder(dir.path(), Some("encoder-f16.gguf"));
    std::fs::write(dir.path().join("encoder-f16.gguf"), b"x").expect("writes");
    assert_eq!(
        speech_encoder_for(&path, None).expect("found"),
        Some(dir.path().join("encoder-f16.gguf"))
    );
}

#[test]
fn an_explicit_encoder_wins_and_must_exist() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = decoder(dir.path(), Some("encoder-f16.gguf"));
    let other = dir.path().join("elsewhere.gguf");
    assert!(speech_encoder_for(&path, Some(&other)).is_err());
    std::fs::write(&other, b"x").expect("writes");
    assert_eq!(
        speech_encoder_for(&path, Some(&other)).expect("found"),
        Some(other)
    );
}

#[test]
fn a_decoder_naming_no_encoder_pairs_with_none() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = decoder(dir.path(), None);
    assert_eq!(speech_encoder_for(&path, None).expect("reads"), None);
}
