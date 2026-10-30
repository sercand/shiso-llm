//! The pairing against synthetic decoder headers.

use std::path::Path;

use shiso_core::CoreError;

use shiso_modelfile::GgufFile;

use super::{SPEECH_MODEL_KEY, check_pairing, speech_encoder_for};
use crate::nemotron_h::NemotronHConfig;
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

#[test]
fn an_encoder_as_wide_as_the_decoder_pairs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let decoder = dir.path().join("decoder.gguf");
    crate::nemotron_h::config::gguf_tests::write(&decoder, &[], 1);
    let config =
        NemotronHConfig::from_gguf(&GgufFile::open(&decoder).expect("opens")).expect("reads");
    let encoder = dir.path().join("encoder.gguf");
    crate::nemotron_transcribe::config::tests::write(&encoder, "pee", 2688);
    check_pairing(&encoder, &config).expect("2688 pairs with 2688");
    crate::nemotron_transcribe::config::tests::write(&encoder, "pee", 4096);
    let refused = check_pairing(&encoder, &config);
    let Err(CoreError::InvalidValue { value, .. }) = refused else {
        panic!("a 4096-wide encoder is refused, got {refused:?}");
    };
    assert!(value.contains("4096") && value.contains("2688"), "{value}");
    // An encoder that is not one at all is refused the same way.
    crate::nemotron_transcribe::config::tests::write(&encoder, "clip", 2688);
    assert!(matches!(
        check_pairing(&encoder, &config),
        Err(CoreError::InvalidValue { .. })
    ));
}
