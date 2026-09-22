//! Parses the real sharded checkpoint and pins everything downstream depends
//! on: the raw header, the translated tensor stream, and the bound role tree.
//!
//! Skips when no local checkpoint is present. The fixture in
//! `test-vectors/loader/qwen35-optiq-layout.json` was produced from this very file by
//! `examples/dump_optiq_layout.rs`, so these assertions are a regression gate on
//! the loader rather than a restatement of it: every number below was read off
//! the checkpoint, not assumed.
//!
//! The header assertions deliberately bypass the loader and re-read the raw
//! JSON. A test that asked the loader what the loader found would agree with a
//! loader that had stopped reading half the file.

mod optiq;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use std::sync::Arc;

use shiso_core::{DType, HfConfig, LayerIdx};
use shiso_engine::shape::LayerKind;
use shiso_modelfile::{ModelFile, SafetensorsFile, TensorSource};
use shiso_models::qwen35::Qwen35Names;
use shiso_models::qwen35::{Qwen35Config, bind_qwen35_desc};
/// Shard the language model lives in.
const SHARD: &str = "model.safetensors";
/// Bytes of little-endian length in front of the header JSON.
const HEADER_LEN_BYTES: usize = 8;
/// Bits in one packed storage word.
const WORD_BITS: u64 = 32;

/// The checked-in layout fixture.
fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/loader/qwen35-optiq-layout.json")
}

/// The fixture, parsed.
fn fixture() -> serde_json::Value {
    let text = std::fs::read_to_string(fixture_path()).expect("the fixture is checked in");
    serde_json::from_str(&text).expect("the fixture is valid JSON")
}

/// One raw header entry, before any translation.
#[derive(Debug, Clone)]
struct RawEntry {
    /// Storage format as the header spells it.
    dtype: String,
    /// Dimensions, slowest-varying axis first.
    shape: Vec<u64>,
}

/// Reads the shard's header without going through the loader.
fn raw_header(dir: &Path) -> (String, BTreeMap<String, RawEntry>) {
    let bytes = std::fs::read(dir.join(SHARD)).expect("the shard is readable");
    let len_bytes: [u8; HEADER_LEN_BYTES] = bytes[..HEADER_LEN_BYTES]
        .try_into()
        .expect("the shard is longer than its length prefix");
    let end = HEADER_LEN_BYTES + usize::try_from(u64::from_le_bytes(len_bytes)).unwrap_or(0);
    let table: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[HEADER_LEN_BYTES..end]).expect("the header is valid JSON");

    let mut format = String::new();
    let mut entries = BTreeMap::new();
    for (name, value) in table {
        if name == "__metadata__" {
            format = value
                .get("format")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string();
            continue;
        }
        let dtype = value
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .expect("every entry names a format")
            .to_string();
        let shape: Vec<u64> = value
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .expect("every entry has a shape")
            .iter()
            .filter_map(serde_json::Value::as_u64)
            .collect();
        entries.insert(name, RawEntry { dtype, shape });
    }
    (format, entries)
}

/// Renders the header as the digest's input: one sorted row per tensor.
fn digest_rows(entries: &BTreeMap<String, RawEntry>) -> String {
    use std::fmt::Write as _;

    let mut text = String::new();
    for (name, entry) in entries {
        let dims: Vec<String> = entry.shape.iter().map(u64::to_string).collect();
        writeln!(text, "{name}|{}|{}", entry.dtype, dims.join(","))
            .expect("writing into a String cannot fail");
    }
    text
}

/// Hex `sha256` of `text`, or `None` when this machine has no hashing tool.
///
/// The digest is fixture bookkeeping rather than a runtime feature, so it is
/// computed by the system tool instead of pulling a hash implementation into
/// the crate's dependency tree for the benefit of one assertion.
fn sha256(text: &str) -> Option<String> {
    let path = std::env::temp_dir().join("shiso-optiq-header-rows.txt");
    std::fs::write(&path, text).ok()?;
    let out = Command::new("shasum")
        .args(["-a", "256"])
        .arg(&path)
        .output()
        .ok()?;
    drop(std::fs::remove_file(&path));
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8(out.stdout).ok()?;
    stdout.split_whitespace().next().map(str::to_string)
}

/// The shape profile the checkpoint's own config describes.
fn profile(dir: &Path) -> Qwen35Config {
    Qwen35Config::from_hf(
        &HfConfig::from_file(&dir.join("config.json")).expect("the config parses"),
    )
    .expect("the config describes a supported architecture")
}

#[test]
fn the_header_is_the_one_the_fixture_records() {
    let Some(dir) = optiq::checkpoint_dir() else {
        optiq::skip("checkpoint directory not found");
        return;
    };
    let fixture = fixture();
    let (format, entries) = raw_header(&dir);

    assert_eq!(format, "mlx", "the shard names its producer");
    assert_eq!(
        u64::try_from(entries.len()).unwrap_or(u64::MAX),
        fixture["n_tensors"]
            .as_u64()
            .expect("fixture records a count"),
    );

    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    for entry in entries.values() {
        *counts.entry(entry.dtype.clone()).or_default() += 1;
    }
    let want = fixture["dtype_counts"]
        .as_object()
        .expect("fixture records the format mix");
    for (dtype, count) in want {
        assert_eq!(
            counts.get(dtype.as_str()).copied().unwrap_or_default(),
            count.as_u64().expect("a count is a number"),
            "format {dtype} count"
        );
    }
    assert_eq!(counts.len(), want.len(), "no unexpected formats");

    let Some(digest) = sha256(&digest_rows(&entries)) else {
        optiq::skip("no sha256 tool on this machine; the header digest was not checked");
        return;
    };
    assert_eq!(
        digest,
        fixture["header_sha256"]
            .as_str()
            .expect("fixture records a digest"),
        "the tensor table is not the one the fixture was taken from"
    );
}

#[test]
fn every_packed_tensor_agrees_with_the_config_about_its_width() {
    let Some(dir) = optiq::checkpoint_dir() else {
        optiq::skip("checkpoint directory not found");
        return;
    };
    let fixture = fixture();
    let (_, entries) = raw_header(&dir);
    let quant = fixture["quantization_config"]
        .as_object()
        .expect("fixture records the quantization config");
    let default_group = quant["group_size"].as_u64().unwrap_or(0);

    let mut histogram: BTreeMap<u64, u64> = BTreeMap::new();
    for (name, entry) in &entries {
        if entry.dtype != "U32" {
            continue;
        }
        let base = name
            .strip_suffix(".weight")
            .expect("a packed array is a weight");
        let scales = entries
            .get(&format!("{base}.scales"))
            .expect("a packed weight has scales");
        let group = quant
            .get(base)
            .and_then(|entry| entry.get("group_size"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(default_group);
        let packed_last = *entry.shape.last().expect("a packed array has a last axis");
        let scales_last = *scales.shape.last().expect("scales have a last axis");
        // Bits are a consequence of the shapes, not something a sidecar file
        // gets to claim: a row of `p` words over `g * group_size` weights can
        // only be `p * 32 / (g * group_size)` bits wide.
        let bits = packed_last * WORD_BITS / (scales_last * group);
        assert_eq!(
            quant
                .get(base)
                .and_then(|entry| entry.get("bits"))
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_else(|| quant["bits"].as_u64().unwrap_or(0)),
            bits,
            "the config and the packed shape disagree about {base}"
        );
        *histogram.entry(bits).or_default() += 1;
    }

    let want = fixture["bits_histogram"]
        .as_object()
        .expect("fixture records the width mix");
    for (bits, count) in want {
        let bits: u64 = bits.parse().expect("a width is a number");
        assert_eq!(
            histogram.get(&bits).copied().unwrap_or_default(),
            count.as_u64().expect("a count is a number"),
            "{bits}-bit tensor count"
        );
    }
    assert_eq!(histogram.len(), want.len(), "no unexpected widths");
}

#[test]
fn the_translated_stream_is_a_complete_model() {
    let Some(file) = optiq::open_checkpoint() else {
        optiq::skip("checkpoint directory not found");
        return;
    };
    let fixture = fixture();
    let want = &fixture["gguf_tensors"];

    let mut total = 0_u64;
    let mut f16 = 0_u64;
    let mut f32 = 0_u64;
    for desc in file.tensors() {
        total += 1;
        match desc.dtype {
            DType::F16 => f16 += 1,
            DType::F32 => f32 += 1,
            other => panic!("tensor {} decoded to {}", desc.name, other.type_name()),
        }
    }
    assert_eq!(
        total,
        want["total"].as_u64().expect("fixture records a total")
    );
    assert_eq!(
        f16,
        want["f16"].as_u64().expect("fixture records the f16 count")
    );
    assert_eq!(
        f32,
        want["f32"].as_u64().expect("fixture records the f32 count")
    );
}

#[test]
fn the_translated_names_bind_as_a_qwen35_model() {
    let Some(dir) = optiq::checkpoint_dir() else {
        optiq::skip("checkpoint directory not found");
        return;
    };
    let file =
        SafetensorsFile::open_with(&dir, Arc::new(Qwen35Names)).expect("the checkpoint opens");
    let profile = profile(&dir);
    let shape = profile.shape();

    let binding = bind_qwen35_desc(&file, &profile).expect("the checkpoint binds");
    assert_eq!(
        binding.layers.len(),
        usize::try_from(shape.n_layers()).unwrap_or(0)
    );
    assert!(
        binding.lm_head.is_none(),
        "the output projection is tied to the embedding"
    );

    let interval = file
        .arch_u32("full_attention_interval")
        .expect("the loader reports the attention stride");
    let mut gdn = 0_u32;
    let mut attn = 0_u32;
    for (il, layer) in binding.layers.iter().enumerate() {
        let il = u32::try_from(il).unwrap_or(0);
        let want = if (il + 1).is_multiple_of(interval) {
            LayerKind::Attn
        } else {
            LayerKind::Gdn
        };
        assert_eq!(layer.kind, want, "layer {il}");
        assert_eq!(layer.gdn.is_some(), want == LayerKind::Gdn, "layer {il}");
        assert_eq!(layer.attn.is_some(), want == LayerKind::Attn, "layer {il}");
        assert_eq!(
            shape.layer_kind(LayerIdx::new(il)),
            Some(want),
            "the profile and the checkpoint agree about layer {il}"
        );
        match want {
            LayerKind::Gdn => gdn += 1,
            LayerKind::Attn => attn += 1,
        }
    }
    assert_eq!((gdn, attn), (24, 8));
}

#[test]
fn the_spot_shapes_are_the_ones_the_kernels_expect() {
    let Some(file) = optiq::open_checkpoint() else {
        optiq::skip("checkpoint directory not found");
        return;
    };

    // The reversed-axis rule and the squeezed unit axis, both visible here: the
    // convolution is exported as [channels, kernel, 1] and must arrive as
    // [kernel, channels], and nothing is transposed on the way.
    let cases: &[(&str, DType, &[u64])] = &[
        ("token_embd.weight", DType::F16, &[2560, 248_320]),
        ("output_norm.weight", DType::F32, &[2560]),
        ("blk.0.ssm_conv1d.weight", DType::F32, &[4, 8192]),
        ("blk.0.ssm_a", DType::F32, &[32]),
        ("blk.0.ssm_dt.bias", DType::F32, &[32]),
        ("blk.0.ssm_norm.weight", DType::F32, &[128]),
        ("blk.0.attn_qkv.weight", DType::F16, &[2560, 8192]),
        ("blk.0.ffn_gate.weight", DType::F16, &[2560, 9216]),
        ("blk.0.ffn_down.weight", DType::F16, &[9216, 2560]),
        ("blk.3.attn_q.weight", DType::F16, &[2560, 8192]),
        ("blk.3.attn_k.weight", DType::F16, &[2560, 1024]),
        ("blk.3.attn_output.weight", DType::F16, &[4096, 2560]),
        ("blk.3.attn_q_norm.weight", DType::F32, &[256]),
    ];
    for &(name, dtype, dims) in cases {
        let desc = file
            .find(name)
            .unwrap_or_else(|| panic!("{name} is present"));
        assert_eq!(desc.dtype, dtype, "{name} format");
        assert_eq!(desc.shape.dims(), dims, "{name} shape");
        assert_eq!(
            desc.bytes,
            desc.shape.elements().expect("the shape is sane")
                * u64::from(dtype.block_bytes().get()),
            "{name} byte count"
        );
    }
}

#[test]
fn the_recurrence_coefficient_is_negated_and_exponentiated() {
    let Some(file) = optiq::open_checkpoint() else {
        optiq::skip("checkpoint directory not found");
        return;
    };
    // The checkpoint stores the logarithm; the kernels want the negated
    // exponential. Every value must therefore be strictly negative and finite,
    // which the logarithms themselves are not.
    let view = file.get("blk.0.ssm_a").expect("the coefficient is present");
    let values = view.as_f32().expect("it decoded to f32");
    assert_eq!(values.len(), 32);
    for &value in values {
        assert!(value < 0.0 && value.is_finite(), "ssm_a holds {value}");
    }
}

#[test]
fn decoding_one_weight_yields_the_declared_byte_count() {
    let Some(file) = optiq::open_checkpoint() else {
        optiq::skip("checkpoint directory not found");
        return;
    };
    let name = "blk.0.ffn_gate.weight";
    let desc = file.find(name).expect("the weight is present").clone();
    let bytes = file.tensor_bytes(name).expect("the weight decodes");
    assert_eq!(
        u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        9216 * 2560 * 2
    );
    assert_eq!(u64::try_from(bytes.len()).unwrap_or(u64::MAX), desc.bytes);
    // A decoded weight is synthesised, never a window onto the shard, so the
    // caller is told to copy it.
    assert!(matches!(bytes, std::borrow::Cow::Owned(_)));
}

#[test]
fn the_directory_dispatches_to_the_sharded_loader_and_binds_through_it() {
    let Some(dir) = optiq::checkpoint_dir() else {
        optiq::skip("checkpoint directory not found");
        return;
    };
    let file = ModelFile::open_with(&dir, Arc::new(Qwen35Names)).expect("the directory opens");
    assert!(
        matches!(file, ModelFile::Safetensors(_)),
        "a directory is a sharded checkpoint"
    );
    // Decoded weights are synthesised, so a device has to copy them; this is
    // the one question a loader asks before wrapping bytes in place.
    assert!(!file.is_mmap_backed());

    let binding = bind_qwen35_desc(&file, &profile(&dir)).expect("the checkpoint binds");
    assert_eq!(binding.layers.len(), 32);
    assert_eq!(file.architecture(), Some("qwen35"));
    assert_eq!(
        u64::try_from(
            file.tensor_bytes(&binding.layers[0].ffn.gate.name)
                .expect("the weight decodes")
                .len()
        )
        .unwrap_or(u64::MAX),
        binding.layers[0].ffn.gate.bytes
    );
}

#[test]
fn a_tensor_this_loader_does_not_define_is_named_in_the_error() {
    let Some(file) = optiq::open_checkpoint() else {
        optiq::skip("checkpoint directory not found");
        return;
    };
    let err = file
        .tensor_bytes("blk.0.not_a_role.weight")
        .expect_err("there is no such role");
    assert!(
        err.to_string().contains("blk.0.not_a_role.weight"),
        "the error should name the tensor, got {err}"
    );
}
