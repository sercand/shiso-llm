#![expect(clippy::indexing_slicing, reason = "test fixtures index known shapes")]

//! The latent pack's loader seams on synthetic files: the n-gram container's
//! refusals, the half-precision widening, and the skeleton binding.

use std::path::Path;

use shiso_core::TokenId;
use shiso_core::error::CoreError;
use shiso_modelfile::GgufFile;

use super::{ROW_ROUNDING, check_rows, read_table, widen_half};
use crate::qwen4exp::config::{FULL_ATTENTION_INTERVAL, N_LAYERS, shape};
use crate::qwen4exp::{NgramHash, Qwen4ExpConfig, bind_qwen4exp, bind_skeleton};
use crate::test_gguf::{Kv, Tensor, write_gguf};

const FILE: &str = "ple-fp8.gguf";

/// A table of `rows` rows with the given type id and row width.
fn table(ggml_type: u32, width: u64, rows: u64) -> Tensor {
    let elements = width * rows;
    let bytes = match ggml_type {
        0 => elements * 4,
        _ => elements,
    };
    Tensor {
        name: "per_layer_token_embd.weight".to_owned(),
        ggml_type,
        dims: vec![width, rows],
        data: vec![0x38; usize::try_from(bytes).expect("small")],
    }
}

fn good_kvs() -> Vec<(&'static str, Kv)> {
    vec![
        ("general.architecture", Kv::Str("strata-ple")),
        ("strata.ple.format", Kv::Str("f8_e4m3")),
        ("strata.ple.scale", Kv::F32(0.5)),
    ]
}

fn open(dir: &Path) -> Result<(), CoreError> {
    read_table(dir).map(|_| ())
}

#[test]
fn a_well_formed_table_opens() {
    let dir = tempfile::TempDir::new().expect("a directory");
    write_gguf(&dir.path().join(FILE), &good_kvs(), &[table(24, 160, 4)]);
    let found = read_table(dir.path()).expect("it opens");
    assert_eq!((found.rows, found.scale.to_bits()), (4, 0.5_f32.to_bits()));
}

#[test]
fn a_table_the_reader_cannot_decode_is_refused_by_name() {
    let dir = tempfile::TempDir::new().expect("a directory");
    let put = |kvs: &[(&str, Kv)], tensors: &[Tensor]| {
        write_gguf(&dir.path().join(FILE), kvs, tensors);
        open(dir.path())
    };
    // The codec name.
    let mut kvs = good_kvs();
    kvs[1].1 = Kv::Str("e5m2");
    assert!(matches!(
        put(&kvs, &[table(24, 160, 4)]),
        Err(CoreError::InvalidValue { key: FILE, .. })
    ));
    // The tensor's type.
    assert!(matches!(
        put(&good_kvs(), &[table(0, 160, 4)]),
        Err(CoreError::InvalidValue { key: FILE, .. })
    ));
    // The row width.
    assert!(matches!(
        put(&good_kvs(), &[table(24, 159, 4)]),
        Err(CoreError::ShapeMismatch {
            field: "n-gram row bytes",
            expected: 160,
            got: 159
        })
    ));
    // The scale: missing, doubled, zero, negative, not a number.
    let without: Vec<_> = good_kvs().into_iter().take(2).collect();
    assert!(matches!(
        put(&without, &[table(24, 160, 4)]),
        Err(CoreError::ConfigMissing { .. })
    ));
    let mut twice = good_kvs();
    twice.push(("other.ple.scale", Kv::F32(0.25)));
    assert!(matches!(
        put(&twice, &[table(24, 160, 4)]),
        Err(CoreError::ConfigMissing { .. })
    ));
    for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let mut kvs = good_kvs();
        kvs[2].1 = Kv::F32(bad);
        assert!(
            matches!(
                put(&kvs, &[table(24, 160, 4)]),
                Err(CoreError::InvalidValue { key: FILE, .. })
            ),
            "scale {bad}"
        );
    }
    // The tensor, absent.
    assert!(matches!(
        put(&good_kvs(), &[]),
        Err(CoreError::ConfigMissing { .. })
    ));
    // No file at all.
    std::fs::remove_file(dir.path().join(FILE)).expect("removed");
    assert!(open(dir.path()).is_err());
}

#[test]
fn a_table_shorter_than_its_rows_is_refused() {
    let dir = tempfile::TempDir::new().expect("a directory");
    let path = dir.path().join(FILE);
    write_gguf(&path, &good_kvs(), &[table(24, 160, 4)]);
    let whole = std::fs::metadata(&path).expect("stat").len();
    // One row's bytes short: the header still says four rows.
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("open")
        .set_len(whole - 160)
        .expect("truncated");
    assert!(open(dir.path()).is_err(), "the table is refused");
}

#[test]
fn only_the_writers_rounding_may_follow_the_last_head() {
    assert!(check_rows(1000, 1000).is_ok());
    assert!(check_rows(1000 + ROW_ROUNDING - 1, 1000).is_ok());
    // The real pack: 320,001,536 stored, 320,001,446 addressable.
    assert!(check_rows(320_001_536, 320_001_446).is_ok());
    for stored in [999, 1000 + ROW_ROUNDING, 2000] {
        assert!(
            matches!(
                check_rows(stored, 1000),
                Err(CoreError::ShapeMismatch {
                    field: "n-gram table rows",
                    ..
                })
            ),
            "{stored} rows"
        );
    }
}

#[test]
fn half_widening_is_exact_and_refuses_the_wrong_length() {
    // 1.0, -2.5, the smallest subnormal 2^-24, the largest subnormal, the
    // largest normal, and negative zero.
    let halves: [u16; 6] = [0x3C00, 0xC100, 0x0001, 0x03FF, 0x7BFF, 0x8000];
    let bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
    let wide = widen_half(&bytes, 6).expect("widens");
    let floats: Vec<f32> = wide
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect();
    assert_eq!(floats[0].to_bits(), 1.0_f32.to_bits());
    assert_eq!(floats[1].to_bits(), (-2.5_f32).to_bits());
    assert_eq!(floats[2].to_bits(), (2.0_f32.powi(-24)).to_bits());
    assert_eq!(
        floats[3].to_bits(),
        (1023.0_f32 * 2.0_f32.powi(-24)).to_bits()
    );
    assert_eq!(floats[4].to_bits(), 65_504.0_f32.to_bits());
    assert_eq!(floats[5].to_bits(), (-0.0_f32).to_bits());
    // Odd, short and long.
    let long = [bytes.clone(), vec![0, 0]].concat();
    for (given, elements) in [
        (&bytes[..11], 6),
        (&bytes[..10], 6),
        (&long[..], 6),
        (&bytes[..0], 1),
    ] {
        assert!(
            matches!(
                widen_half(given, elements),
                Err(CoreError::ShapeMismatch {
                    field: "half-precision vector bytes",
                    ..
                })
            ),
            "{} bytes for {elements} elements",
            given.len()
        );
    }
    assert!(widen_half(&[], 0).expect("empty is fine").is_empty());
}

/// Every tensor name `bind_qwen4exp` asks for, with or without the stacked
/// routed banks.
fn names(stacked: bool) -> Vec<String> {
    let mut names = vec!["token_embd.weight".to_owned()];
    for part in ["norm", "down", "up"] {
        names.push(format!("output_hc_{part}.weight"));
    }
    for il in 0..N_LAYERS {
        for side in ["attn", "ffn"] {
            for part in ["norm", "down", "up", "inject"] {
                names.push(format!("blk.{il}.hc_{side}_{part}.weight"));
            }
        }
        let block: &[&str] = if (il + 1) % FULL_ATTENTION_INTERVAL == 0 {
            &[
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "attn_output.weight",
                "indexer.q_proj.weight",
                "indexer.k_proj.weight",
                "indexer.q_norm.weight",
                "indexer.k_norm.weight",
            ]
        } else {
            &[
                "attn_qkv.weight",
                "attn_gate.weight",
                "ssm_conv1d.weight",
                "ssm_a",
                "ssm_alpha.weight",
                "ssm_beta.weight",
                "ssm_dt.bias",
                "ssm_norm.weight",
                "ssm_out.weight",
            ]
        };
        names.extend(block.iter().map(|n| format!("blk.{il}.{n}")));
        let mut routed = vec![
            "ffn_gate_inp.weight",
            "ffn_gate_inp_shexp.weight",
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
        ];
        if stacked {
            routed.extend([
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            ]);
        }
        names.extend(routed.iter().map(|n| format!("blk.{il}.{n}")));
        if il == 1 {
            for part in [
                "ple_key.weight",
                "ple_value.weight",
                "ple_norm_key.weight",
                "ple_norm_query.weight",
                "ple_norm_conv.weight",
                "ple_conv1d.weight",
            ] {
                names.push(format!("blk.{il}.{part}"));
            }
        }
    }
    names
}

/// A container holding `names` as one-element tensors: the binding looks
/// tensors up by name and reads no shape.
fn container(dir: &Path, names: &[String]) -> GgufFile {
    let tensors: Vec<Tensor> = names
        .iter()
        .map(|name| Tensor {
            name: name.clone(),
            ggml_type: 0,
            dims: vec![1],
            data: vec![0; 4],
        })
        .collect();
    let path = dir.join("skeleton.gguf");
    write_gguf(
        &path,
        &[("general.architecture", Kv::Str("qwen4exp"))],
        &tensors,
    );
    GgufFile::open_shared(&path).expect("the container opens")
}

fn config() -> Qwen4ExpConfig {
    let ngram = NgramHash {
        multipliers: vec![1, 3, 5],
        head_offsets: vec![0],
        head_vocab: vec![1],
        eos_token: TokenId::new(1),
    };
    Qwen4ExpConfig::new(shape(), TokenId::new(1), FULL_ATTENTION_INTERVAL, ngram)
        .expect("the family's shape folds in an n-gram block")
}

#[test]
fn a_skeleton_binds_without_banks_and_everything_else_is_still_required() {
    let dir = tempfile::TempDir::new().expect("a directory");
    let config = config();

    // No stacked banks: the skeleton binds, the full binding refuses by name.
    let file = container(dir.path(), &names(false));
    let bound = bind_skeleton(&file, &config).expect("a skeleton binds");
    assert!(
        bound.layers.iter().all(|l| l.experts.gate.is_none()
            && l.experts.up.is_none()
            && l.experts.down.is_none())
    );
    match bind_qwen4exp(&file, &config) {
        Err(CoreError::ConfigMissing { key }) => assert_eq!(key, "blk.0.ffn_gate_exps.weight"),
        other => panic!("{:?}", other.map(|_| ())),
    }
    drop(bound);
    drop(file);

    // With them, both bind and the skeleton still sees the banks.
    let file = container(dir.path(), &names(true));
    bind_qwen4exp(&file, &config).expect("a full container binds");
    assert!(bind_skeleton(&file, &config).is_ok());
    drop(file);

    // A missing non-routed tensor is refused by the skeleton binding.
    for gone in [
        "blk.5.attn_qkv.weight",
        "blk.3.attn_q.weight",
        "token_embd.weight",
    ] {
        let kept: Vec<String> = names(false).into_iter().filter(|n| n != gone).collect();
        let file = container(dir.path(), &kept);
        match bind_skeleton(&file, &config) {
            Err(CoreError::ConfigMissing { key }) => assert_eq!(key, gone),
            other => panic!("{gone}: {:?}", other.map(|_| ())),
        }
    }
}
