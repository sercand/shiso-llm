//! The metadata reader against synthetic headers.

use std::path::Path;

use shiso_modelfile::GgufFile;

use super::PeeConfig;
use crate::test_gguf::{Kv, write_gguf};

fn write(path: &Path, arch: &'static str, proj_dim: u32) {
    let block = |p: &'static str, d: u32, h: u32, l: u32, ff: u32, rot: u32, qk: bool| {
        let key = |s: &str| -> &'static str { Box::leak(format!("{p}.{s}").into_boxed_str()) };
        vec![
            (key("d_model"), Kv::U32(d)),
            (key("n_heads"), Kv::U32(h)),
            (key("n_layers"), Kv::U32(l)),
            (key("ff_hidden"), Kv::U32(ff)),
            (key("rotary_dims"), Kv::U32(rot)),
            (key("qk_norm"), Kv::Bool(qk)),
        ]
    };
    let mut kvs = vec![("general.architecture", Kv::Str(arch))];
    kvs.extend(block("pee.asr", 1280, 10, 32, 5120, 64, true));
    kvs.extend(block("pee.diar", 512, 8, 31, 2048, 64, false));
    kvs.extend(block("ctc", 1280, 16, 2, 640, 40, true));
    kvs.extend([
        ("pee.asr.subsampling_factor", Kv::U32(8)),
        ("pee.diar.subsampling_factor", Kv::U32(8)),
        ("pee.diar.num_speakers", Kv::U32(8)),
        ("pee.diar.tf_d_model", Kv::U32(192)),
        ("pee.fusion.proj_dim", Kv::U32(proj_dim)),
        ("pee.fusion.speaker_activity_threshold", Kv::F32(0.5)),
        ("pee.online.inference_length", Kv::U32(500)),
        ("pee.online.chunk_left_context", Kv::U32(50)),
        ("pee.preprocessor.sample_rate", Kv::U32(16_000)),
        ("pee.preprocessor.window_size", Kv::F32(0.025)),
        ("pee.preprocessor.window_stride", Kv::F32(0.01)),
        ("pee.preprocessor.n_fft", Kv::U32(512)),
        ("pee.preprocessor.features", Kv::U32(128)),
        ("ctc.vocab_size", Kv::U32(23_863)),
        ("ctc.upsampling_factor", Kv::U32(2)),
        ("ctc.residual_scale", Kv::F32(0.1)),
    ]);
    write_gguf(path, &kvs, &[]);
}

fn read(arch: &'static str, proj_dim: u32) -> Result<PeeConfig, shiso_audio::AudioError> {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("encoder.gguf");
    write(&path, arch, proj_dim);
    PeeConfig::from_gguf(&GgufFile::open(&path).expect("opens"))
}

#[test]
fn pee_config_reads_the_spec_dims() {
    let c = read("pee", 2688).expect("reads");
    assert_eq!(
        (
            c.asr.layers,
            c.asr.d_model,
            c.asr.heads,
            c.asr.head_dim,
            c.asr.ff,
            c.asr.rot_dims
        ),
        (32, 1280, 10, 128, 5120, 64)
    );
    assert!(c.asr.qk_norm);
    assert_eq!(
        (c.diar.layers, c.diar.d_model, c.diar.head_dim, c.speakers),
        (31, 512, 64, 8)
    );
    assert!(!c.diar.qk_norm);
    assert_eq!(
        (c.ctc.head_dim, c.ctc.rot_dims, c.ctc_vocab),
        (80, 40, 23_863)
    );
    assert_eq!(c.fusion_dim, 2688);
    assert_eq!(
        (c.win_length, c.hop_length, c.n_fft, c.mels),
        (400, 160, 512, 128)
    );
    assert_eq!((c.window_core, c.window_context), (500, 50));
}

#[test]
fn fusion_dim_mismatch_is_refused() {
    let c = read("pee", 4096).expect("reads");
    let err = c.check_decoder(2688).expect_err("4096 is not 2688");
    assert!(err.to_string().contains("4096"), "{err}");
}

#[test]
fn foreign_arch_is_refused() {
    let err = read("clip", 2688).expect_err("not pee");
    assert!(err.to_string().contains("clip"), "{err}");
}
