//! Checks that need no model file, so they run on every machine.
//!
//! The synthetic-model builder at the bottom writes just enough of the format
//! to exercise the metadata getters, the error paths and the binding's name and
//! dimension rules. Building a *complete* synthetic model is not possible
//! without a multi-gigabyte file, so the positive full-model assertions live in
//! the model-dependent tests; what is checked here is that every rejection
//! fires for the right reason, which is the half a real model cannot show.

use std::io::Write;
use std::path::{Path, PathBuf};

use shiso_core::{DType, HfConfig, LayerIdx};
use shiso_engine::shape::LayerKind;
use shiso_modelfile::quant::{
    self, BlockIq1M, BlockIq1S, BlockIq2S, BlockIq2Xs, BlockIq2Xxs, BlockIq3S, BlockIq3Xxs,
    BlockIq4Nl, BlockIq4Xs, BlockMxfp4, BlockPq2_0, BlockPtq1_0, BlockQ2K, BlockQ4K, BlockQ5_1,
    BlockQ5K, BlockQ6K, BlockQ8_0, BlockQ8K, QuantBlock,
};
use shiso_modelfile::reader::{GgufError, GgufFile, MAX_DIMS, TensorShape};
use shiso_modelfile::{MetaType, MetaValue, TensorSource};
use shiso_models::qwen35::Qwen35Config;
use shiso_models::qwen35::bind_qwen35;
use tempfile::NamedTempFile;

const MAGIC: u32 = 0x4655_4747;
const VERSION: u32 = 3;
const ALIGN: usize = 32;

/// The shape fixture that ships with the workspace.
fn fixture_config() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/loader/qwen35-config.json")
}

// ---------------------------------------------------------------------------
// (a) block sizing
// ---------------------------------------------------------------------------

#[test]
fn block_bytes_matches_format_geometry() {
    let cases: [(DType, &[u64], u64); 4] = [
        // 100 whole blocks of 256 elements, 144 bytes each.
        (DType::Q4_K, &[25_600], 14_400),
        // 10 blocks of 32, 34 bytes each.
        (DType::Q8_0, &[320], 340),
        // One microscaling block: 17 bytes, not 18.
        (DType::MXFP4, &[32], 17),
        // Five 4-bit non-linear blocks at 18 bytes.
        (DType::IQ4_NL, &[160], 90),
    ];
    for (dtype, dims, want) in cases {
        let shape = TensorShape::new(dims).expect("fixture shape");
        assert_eq!(
            quant::block_bytes_for(dtype, &shape),
            Some(want),
            "{dtype:?} {dims:?}"
        );
    }
}

#[test]
fn partial_trailing_block_rounds_up() {
    let shape = TensorShape::new(&[257]).expect("fixture shape");
    assert_eq!(
        quant::block_bytes_for(DType::Q4_K, &shape),
        Some(2 * 144),
        "257 elements is two blocks, not one"
    );
}

#[test]
fn trellis_payload_is_tiles_plus_scale_vectors() {
    assert_eq!(quant::exl3::k_bits(DType::Exl3K4), Some(4));
    assert_eq!(quant::exl3::k_bits(DType::Exl3K5), Some(5));
    assert_eq!(quant::exl3::k_bits(DType::Exl3K6), Some(6));
    assert_eq!(quant::exl3::k_bits(DType::Exl3K8), Some(8));
    assert_eq!(quant::exl3::k_bits(DType::Q4_K), None);
    assert_eq!(quant::exl3::tile_bytes(4), 128);
    assert_eq!(quant::exl3::tile_bytes(5), 160);
    assert_eq!(quant::exl3::tile_bytes(6), 192);
    assert_eq!(quant::exl3::tile_bytes(8), 256);

    // [2560, 640]: 6400 tiles of 192 bytes, then two fp16 vectors of 2560 and
    // 640 words, which is 2 * (2560 + 640) = 6400 bytes.
    let shape = TensorShape::new(&[2560, 640]).expect("fixture shape");
    assert_eq!(quant::exl3::scale_bytes(&shape), 6_400);
    assert_eq!(
        quant::block_bytes_for(DType::Exl3K6, &shape),
        Some(6_400 * 192 + 6_400)
    );

    // A third dimension is the expert count, and each expert gets its own pair.
    let three_d = TensorShape::new(&[2560, 640, 4]).expect("fixture shape");
    assert_eq!(quant::exl3::scale_bytes(&three_d), 4 * 6_400);
}

#[test]
fn block_structs_agree_with_the_format_table() {
    macro_rules! check {
        ($ty:ty, $elems:expr, $bytes:expr) => {{
            assert_eq!(<$ty as QuantBlock>::ELEMS, $elems);
            assert_eq!(
                <$ty as QuantBlock>::DTYPE.block_elems().get() as usize,
                <$ty as QuantBlock>::ELEMS,
                stringify!($ty)
            );
            assert_eq!(
                <$ty as QuantBlock>::DTYPE.block_bytes().get() as usize,
                std::mem::size_of::<$ty>(),
                stringify!($ty)
            );
            assert_eq!(std::mem::size_of::<$ty>(), $bytes, stringify!($ty));
        }};
    }
    check!(BlockQ8_0, 32, 34);
    check!(BlockQ2K, 256, 84);
    check!(BlockQ4K, 256, 144);
    check!(BlockQ5K, 256, 176);
    check!(BlockQ6K, 256, 210);
    check!(BlockQ8K, 256, 292);
    check!(BlockIq2Xxs, 256, 66);
    check!(BlockIq2Xs, 256, 74);
    check!(BlockIq3Xxs, 256, 98);
    check!(BlockIq1S, 256, 50);
    check!(BlockIq3S, 256, 110);
    check!(BlockIq2S, 256, 82);
    check!(BlockIq4Xs, 256, 136);
    check!(BlockIq1M, 256, 56);
    check!(BlockMxfp4, 32, 17);
    check!(BlockQ5_1, 32, 24);
    check!(BlockIq4Nl, 32, 18);
    check!(BlockPq2_0, 128, 34);
    check!(BlockPtq1_0, 128, 28);
}

#[test]
fn shape_rejects_ranks_and_zero_dimensions() {
    assert!(TensorShape::new(&[]).is_err());
    assert!(TensorShape::new(&[1; MAX_DIMS + 1]).is_err());
    assert!(TensorShape::new(&[4, 0]).is_err());
    let shape = TensorShape::new(&[3, 4]).expect("fixture shape");
    assert_eq!(shape.ndim(), 2);
    assert_eq!(shape.dims(), &[3, 4]);
    assert_eq!(shape.dim(1), Some(4));
    assert_eq!(shape.dim(2), None);
    assert_eq!(shape.elements().expect("product"), 12);
}

// ---------------------------------------------------------------------------
// (b) metadata getters, against a hand-built key/value table
// ---------------------------------------------------------------------------

#[test]
fn metadata_getters_widen_and_narrow_as_documented() {
    let kvs: Vec<(&str, Kv)> = vec![
        ("test.u16", Kv::U16(7)),
        ("test.i16.pos", Kv::I16(5)),
        ("test.i16.neg", Kv::I16(-3)),
        ("test.u32", Kv::U32(9)),
        ("test.i32.pos", Kv::I32(11)),
        ("test.i32.neg", Kv::I32(-11)),
        ("test.u64", Kv::U64(1 << 40)),
        ("test.f32", Kv::F32(1.5)),
        ("test.f64", Kv::F64(2.25)),
        ("test.bool", Kv::Bool(true)),
        ("test.str", Kv::Str("hello")),
        ("test.arr", Kv::StrArray(&["a", "b", "c"])),
        ("test.i32arr", Kv::I32Array(&[1, 2, 3, 4])),
    ];
    let file = write_model(&kvs, &[]);
    let model = GgufFile::open(file.path()).expect("synthetic model opens");
    let md = model.metadata();

    assert_eq!(md.architecture(), None, "no architecture key was written");
    assert_eq!(md.len(), kvs.len());

    // get_u32 widens u16 and accepts non-negative i16/i32.
    assert_eq!(md.get_u32("test.u16"), Some(7));
    assert_eq!(md.get_u32("test.i16.pos"), Some(5));
    assert_eq!(md.get_u32("test.i16.neg"), None, "a negative value wraps");
    assert_eq!(md.get_u32("test.u32"), Some(9));
    assert_eq!(md.get_u32("test.i32.pos"), Some(11));
    assert_eq!(md.get_u32("test.i32.neg"), None);
    assert_eq!(
        md.get_u32("test.u64"),
        None,
        "64 bits is not silently truncated"
    );

    // get_u64 widens u32 only.
    assert_eq!(md.get_u64("test.u64"), Some(1 << 40));
    assert_eq!(md.get_u64("test.u32"), Some(9));
    assert_eq!(md.get_u64("test.i32.pos"), None);

    // get_f32 narrows f64 and widens the integer widths.
    assert_eq!(md.get_f32("test.f32"), Some(1.5));
    assert_eq!(md.get_f32("test.f64"), Some(2.25));
    assert_eq!(md.get_f32("test.u32"), Some(9.0));
    assert_eq!(md.get_f32("test.i32.pos"), Some(11.0));
    assert_eq!(md.get_f32("test.str"), None);

    assert_eq!(md.get_bool("test.bool"), Some(true));
    assert_eq!(md.get_bool("test.u32"), None);
    assert_eq!(md.get_str("test.str"), Some("hello"));
    assert_eq!(md.get_str("test.u32"), None);
    assert!(md.get("missing").is_none());

    // Arrays stay lazy until read.
    let arr = md.get_array("test.arr").expect("array key");
    assert_eq!(arr.elem_type(), MetaType::Str);
    assert_eq!(arr.len(), 3);
    assert!(!arr.is_empty());
    assert!(matches!(arr.get(1).expect("entry"), MetaValue::Str(ref v) if &**v == "b"));
    assert!(matches!(
        arr.get(3),
        Err(GgufError::ArrayIndex { index: 3, len: 3 })
    ));
    assert_eq!(
        arr.strings().expect("string array"),
        vec![Box::<str>::from("a"), "b".into(), "c".into()]
    );
    let ints: Vec<MetaValue> = md
        .get_array("test.i32arr")
        .expect("int array")
        .iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("entries decode");
    assert_eq!(ints.len(), 4);
    assert!(matches!(ints[3], MetaValue::I32(4)));
    assert!(matches!(
        md.get_array("test.i32arr").expect("array").strings(),
        Err(GgufError::LayoutMismatch { .. })
    ));
}

#[test]
fn architecture_prefixed_lookup() {
    let kvs: Vec<(&str, Kv)> = vec![
        ("general.architecture", Kv::Str("qwen35")),
        ("qwen35.block_count", Kv::U32(32)),
        ("qwen35.rope.freq_base", Kv::F64(1.0e7)),
    ];
    let file = write_model(&kvs, &[]);
    let model = GgufFile::open(file.path()).expect("synthetic model opens");
    assert_eq!(model.metadata().architecture(), Some("qwen35"));
    assert_eq!(model.metadata().arch_u32("block_count"), Some(32));
    assert_eq!(model.metadata().arch_f32("rope.freq_base"), Some(1.0e7));
    assert_eq!(model.metadata().arch_u32("absent"), None);
}

#[test]
fn the_architecture_probe_reads_the_key_wherever_it_sits() {
    // After an array and a scalar, so the walk has to skip both to reach it.
    let kvs: Vec<(&str, Kv)> = vec![
        ("tokenizer.tokens", Kv::StrArray(&["a", "b", "c"])),
        ("test.u32", Kv::U32(7)),
        ("general.architecture", Kv::Str("dflash")),
    ];
    let file = write_model(&kvs, &[]);
    let arch = GgufFile::read_architecture(file.path()).expect("the probe reads the header");
    assert_eq!(arch.as_deref(), Some("dflash"));

    let none = write_model(&[("test.u32", Kv::U32(7))], &[]);
    let arch = GgufFile::read_architecture(none.path()).expect("the probe reads the header");
    assert_eq!(arch, None);

    let foreign = NamedTempFile::new().expect("temporary file");
    std::fs::write(foreign.path(), [0_u8; 32]).expect("foreign bytes written");
    assert!(matches!(
        GgufFile::read_architecture(foreign.path()),
        Err(GgufError::NotGguf)
    ));
}

// ---------------------------------------------------------------------------
// (c) malformed input
// ---------------------------------------------------------------------------

#[test]
fn short_and_foreign_files_are_rejected() {
    assert!(matches!(
        open_bytes(&[]),
        Err(GgufError::Truncated { pos: 0 })
    ));
    assert!(matches!(
        open_bytes(&[0u8; 16]),
        Err(GgufError::Truncated { pos: 0 })
    ));
    // Magic alone is not a header.
    let mut magic = Vec::new();
    magic.extend_from_slice(&MAGIC.to_le_bytes());
    assert!(matches!(
        open_bytes(&magic),
        Err(GgufError::Truncated { .. })
    ));

    let mut foreign = vec![0u8; 64];
    foreign[0..4].copy_from_slice(b"XXXX");
    assert!(matches!(open_bytes(&foreign), Err(GgufError::NotGguf)));

    let mut old = vec![0u8; 64];
    old[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    old[4..8].copy_from_slice(&2u32.to_le_bytes());
    assert!(matches!(
        open_bytes(&old),
        Err(GgufError::Version { got: 2 })
    ));
}

#[test]
fn header_without_directory_is_truncated() {
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&1u64.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    // The header claims one tensor and the file ends immediately after it, so
    // the directory read is the first thing to run out of bytes.
    out.resize(32, 0);
    assert!(matches!(open_bytes(&out), Err(GgufError::Truncated { .. })));
}

#[test]
fn directory_errors_name_the_tensor() {
    // Unknown format id.
    let err = open_model(&[], &[spec("blk.0.weird", &[8, 8], 999)]).expect_err("id 999");
    assert!(
        matches!(&err, GgufError::UnsupportedType { tensor, dtype_id } if **tensor == *"blk.0.weird" && *dtype_id == 999),
        "{err:?}"
    );

    // Rank above the limit.
    let wide = vec![2u64; MAX_DIMS + 1];
    let err = open_model(&[], &[spec("blk.0.wide", &wide, 0)]).expect_err("nine dims");
    assert!(
        matches!(&err, GgufError::DimLimit { tensor, ndim } if **tensor == *"blk.0.wide" && *ndim == 9),
        "{err:?}"
    );

    // A payload that runs past the end of the file.
    let err = open_model_truncated(&[], &[spec("blk.0.big", &[1_000_000, 1_000_000], 0)])
        .expect_err("declared payload exceeds the file");
    assert!(
        matches!(&err, GgufError::OutOfBounds { tensor } if **tensor == *"blk.0.big"),
        "{err:?}"
    );
}

#[test]
fn split_models_are_refused_not_misread() {
    let kvs: Vec<(&str, Kv)> = vec![("split.count", Kv::U32(4)), ("split.no", Kv::U32(0))];
    let err = open_model(&kvs, &[]).expect_err("a four-shard split");
    assert!(
        matches!(err, GgufError::SplitUnsupported { count: 4 }),
        "{err:?}"
    );
    // The message names the shard a caller would reach for first, so the
    // refusal points at the file set rather than at a bare count.
    assert!(
        err.to_string().contains("00001-of-00004"),
        "unhelpful message: {err}"
    );
}

#[test]
fn a_nonsense_alignment_is_reported() {
    let kvs: Vec<(&str, Kv)> = vec![("general.alignment", Kv::U32(3))];
    let err = open_model(&kvs, &[]).expect_err("alignment must divide a page");
    assert!(
        matches!(&err, GgufError::LayoutMismatch { tensor, .. } if **tensor == *"general.alignment"),
        "{err:?}"
    );
}

// ---------------------------------------------------------------------------
// (d) binding: role names, dimension rules, profile cross-checks
// ---------------------------------------------------------------------------

#[test]
fn fixture_profile_is_a_hybrid_of_24_recurrent_and_8_attention_layers() {
    let config = HfConfig::from_file(&fixture_config()).expect("fixture config parses");
    let profile = Qwen35Config::from_hf(&config).expect("fixture shape profile");
    assert_eq!(profile.shape().n_layers(), 32);

    let mut attn = 0;
    let mut gdn = 0;
    for il in 0..profile.shape().n_layers() {
        let kind = profile
            .shape()
            .layer_kind(LayerIdx::new(il))
            .expect("layer index");
        assert_eq!(
            kind == LayerKind::Attn,
            (il + 1).is_multiple_of(4),
            "layer {il}"
        );
        match kind {
            LayerKind::Attn => attn += 1,
            LayerKind::Gdn => gdn += 1,
            LayerKind::Ssm | LayerKind::ShortConv => {
                panic!("layer {il}: this family has no {kind:?} slot")
            }
        }
    }
    assert_eq!(attn, 8);
    assert_eq!(gdn, 24);
}

/// The binder refuses a file that is not this family's.
///
/// It used to also refuse a *profile* from another family, and that case is gone
/// because it can no longer be written: the binder takes this family's own config
/// type, so handing it another family's geometry is a compile error rather than a
/// runtime refusal. That is the better half of the check, and it is why the
/// remaining half is only about the file.
#[test]
fn binding_refuses_a_file_that_declares_another_architecture() {
    let config = HfConfig::from_file(&fixture_config()).expect("fixture config parses");
    let profile = Qwen35Config::from_hf(&config).expect("fixture shape profile");

    // A file that declares a different architecture.
    let kvs: Vec<(&str, Kv)> = vec![
        ("general.architecture", Kv::Str("otherarch")),
        ("otherarch.block_count", Kv::U32(32)),
    ];
    let file = write_model(&kvs, &[]);
    let model = GgufFile::open(file.path()).expect("synthetic model opens");
    let err = bind_qwen35(&model, &profile).expect_err("wrong architecture");
    assert!(
        matches!(&err, GgufError::ArchMismatch { want, got } if *want == "qwen35" && **got == *"otherarch"),
        "{err:?}"
    );

    // A file that declares no architecture at all: the empty name in the error
    // is what distinguishes "absent" from "wrong".
    let file = write_model(&[("general.architecture", Kv::Str(""))], &[]);
    let model = GgufFile::open(file.path()).expect("synthetic model opens");
    assert!(matches!(
        bind_qwen35(&model, &profile),
        Err(GgufError::ArchMismatch { .. })
    ));
}

#[test]
fn binding_reports_a_layer_count_that_contradicts_the_profile() {
    let config = HfConfig::from_file(&fixture_config()).expect("fixture config parses");
    let profile = Qwen35Config::from_hf(&config).expect("fixture shape profile");
    let kvs: Vec<(&str, Kv)> = vec![
        ("general.architecture", Kv::Str("qwen35")),
        ("qwen35.block_count", Kv::U32(31)),
    ];
    let file = write_model(&kvs, &[]);
    let model = GgufFile::open(file.path()).expect("synthetic model opens");
    assert!(matches!(
        bind_qwen35(&model, &profile),
        Err(GgufError::ProfileMismatch {
            want: 32,
            got: 31,
            ..
        })
    ));
}

#[test]
fn binding_reports_an_attention_interval_that_contradicts_the_profile() {
    let config = HfConfig::from_file(&fixture_config()).expect("fixture config parses");
    let profile = Qwen35Config::from_hf(&config).expect("fixture shape profile");
    // An interval of 3 would make layer 2 an attention block, which the
    // profile's own layer list says is recurrent.
    let kvs: Vec<(&str, Kv)> = vec![
        ("general.architecture", Kv::Str("qwen35")),
        ("qwen35.block_count", Kv::U32(32)),
        ("qwen35.full_attention_interval", Kv::U32(3)),
    ];
    let file = write_model(&kvs, &[]);
    let model = GgufFile::open(file.path()).expect("synthetic model opens");
    assert!(matches!(
        bind_qwen35(&model, &profile),
        Err(GgufError::ProfileMismatch { .. })
    ));
}

#[test]
fn binding_names_the_first_missing_role() {
    let config = HfConfig::from_file(&fixture_config()).expect("fixture config parses");
    let profile = Qwen35Config::from_hf(&config).expect("fixture shape profile");
    let kvs: Vec<(&str, Kv)> = vec![
        ("general.architecture", Kv::Str("qwen35")),
        ("qwen35.block_count", Kv::U32(32)),
        ("qwen35.full_attention_interval", Kv::U32(4)),
    ];
    let file = write_model(&kvs, &[]);
    let model = GgufFile::open(file.path()).expect("synthetic model opens");
    let err = bind_qwen35(&model, &profile).expect_err("an empty directory binds nothing");
    assert!(
        matches!(&err, GgufError::MissingTensor { name } if **name == *"token_embd.weight"),
        "{err:?}"
    );
}

#[test]
fn binding_rejects_a_wrong_dimension_under_the_right_name() {
    let config = HfConfig::from_file(&fixture_config()).expect("fixture config parses");
    let profile = Qwen35Config::from_hf(&config).expect("fixture shape profile");
    let kvs: Vec<(&str, Kv)> = vec![
        ("general.architecture", Kv::Str("qwen35")),
        ("qwen35.block_count", Kv::U32(32)),
    ];
    // Right name, wrong width: the embedding is the first thing the binding
    // asks for, so this is the cheapest way to reach the dimension check.
    let file = write_model(&kvs, &[spec("token_embd.weight", &[8, 8], 0)]);
    let model = GgufFile::open(file.path()).expect("synthetic model opens");
    let err = bind_qwen35(&model, &profile).expect_err("wrong embedding width");
    assert!(
        matches!(&err, GgufError::LayoutMismatch { tensor, .. } if **tensor == *"token_embd.weight"),
        "{err:?}"
    );
}

#[test]
fn binding_rejects_a_quantized_norm_vector() {
    let config = HfConfig::from_file(&fixture_config()).expect("fixture config parses");
    let profile = Qwen35Config::from_hf(&config).expect("fixture shape profile");
    let kvs: Vec<(&str, Kv)> = vec![
        ("general.architecture", Kv::Str("qwen35")),
        ("qwen35.block_count", Kv::U32(32)),
    ];
    let tensors = [
        spec("token_embd.weight", &[2560, 248_320], 8),
        spec("output_norm.weight", &[2560], 12),
    ];
    let file = write_model(&kvs, &tensors);
    let model = GgufFile::open(file.path()).expect("synthetic model opens");
    let err = bind_qwen35(&model, &profile).expect_err("a norm vector cannot be a K-quant");
    assert!(
        matches!(&err, GgufError::LayoutMismatch { tensor, .. } if **tensor == *"output_norm.weight"),
        "{err:?}"
    );
}

#[test]
fn source_trait_sees_the_same_directory_as_the_inherent_api() {
    let kvs: Vec<(&str, Kv)> = vec![("general.architecture", Kv::Str("qwen35"))];
    let file = write_model(&kvs, &[spec("output_norm.weight", &[2560], 0)]);
    let model = GgufFile::open(file.path()).expect("synthetic model opens");
    let src: &dyn TensorSource = &model;
    assert_eq!(src.architecture(), Some("qwen35"));
    let desc = src.find("output_norm.weight").expect("indexed lookup");
    assert_eq!(desc.dtype, DType::F32);
    assert_eq!(desc.shape.dims(), &[2560]);
    let view = src.get("output_norm.weight").expect("payload");
    assert_eq!(
        view.data.len(),
        usize::try_from(desc.bytes).expect("fixture size")
    );
    assert!(src.find("nope").is_none());
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Metadata values the builder can write.
enum Kv {
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(&'static str),
    StrArray(&'static [&'static str]),
    I32Array(&'static [i32]),
}

/// One tensor directory entry.
#[derive(Clone)]
struct Spec {
    name: &'static str,
    dims: Vec<u64>,
    type_id: u32,
}

fn spec(name: &'static str, dims: &[u64], type_id: u32) -> Spec {
    Spec {
        name,
        dims: dims.to_vec(),
        type_id,
    }
}

fn put_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    put_u64(
        out,
        u64::try_from(s.len()).expect("a test string fits in u64"),
    );
    out.extend_from_slice(s.as_bytes());
}

fn write_kv(out: &mut Vec<u8>, key: &str, value: &Kv) {
    put_str(out, key);
    match *value {
        Kv::U16(v) => {
            put_u32(out, 2);
            put_u16(out, v);
        }
        Kv::I16(v) => {
            put_u32(out, 3);
            out.extend_from_slice(&v.to_le_bytes());
        }
        Kv::U32(v) => {
            put_u32(out, 4);
            put_u32(out, v);
        }
        Kv::I32(v) => {
            put_u32(out, 5);
            out.extend_from_slice(&v.to_le_bytes());
        }
        Kv::U64(v) => {
            put_u32(out, 10);
            put_u64(out, v);
        }
        Kv::F32(v) => {
            put_u32(out, 6);
            put_u32(out, v.to_bits());
        }
        Kv::F64(v) => {
            put_u32(out, 12);
            put_u64(out, v.to_bits());
        }
        Kv::Bool(v) => {
            put_u32(out, 7);
            out.push(u8::from(v));
        }
        Kv::Str(s) => {
            put_u32(out, 8);
            put_str(out, s);
        }
        Kv::StrArray(items) => {
            put_u32(out, 9);
            put_u32(out, 8);
            put_u64(
                out,
                u64::try_from(items.len()).expect("test array fits in u64"),
            );
            for item in items {
                put_str(out, item);
            }
        }
        Kv::I32Array(items) => {
            put_u32(out, 9);
            put_u32(out, 5);
            put_u64(
                out,
                u64::try_from(items.len()).expect("test array fits in u64"),
            );
            for item in items {
                put_u32(out, item.cast_unsigned());
            }
        }
    }
}

/// Payload size for a format, computed here rather than imported, so the
/// reader is checked against a second statement of the same geometry.
fn payload_bytes(type_id: u32, dims: &[u64]) -> u64 {
    let elems: u64 = dims.iter().product();
    // An id the reader does not know is written with an empty payload: the
    // reader rejects the id itself, long before it looks at any payload.
    let geometry = match type_id {
        0 => Some((1, 4)),
        1 => Some((1, 2)),
        8 => Some((32, 34)),
        12 => Some((256, 144)),
        13 => Some((256, 176)),
        14 => Some((256, 210)),
        _ => None,
    };
    let Some((block_elems, block_bytes)) = geometry else {
        return 0;
    };
    elems.div_ceil(block_elems) * block_bytes
}

/// Serializes a model up to and including its tensor directory, and reports how
/// many payload bytes the directory claims follow it.
fn model_bytes(kvs: &[(&str, Kv)], tensors: &[Spec]) -> (Vec<u8>, u64) {
    let mut out = Vec::new();
    put_u32(&mut out, MAGIC);
    put_u32(&mut out, VERSION);
    put_u64(
        &mut out,
        u64::try_from(tensors.len()).expect("test tensor count fits"),
    );
    put_u64(
        &mut out,
        u64::try_from(kvs.len()).expect("test key count fits"),
    );

    for (key, value) in kvs {
        write_kv(&mut out, key, value);
    }

    let mut payload_total = 0_u64;
    for tensor in tensors {
        put_str(&mut out, tensor.name);
        put_u32(
            &mut out,
            u32::try_from(tensor.dims.len()).expect("test rank fits"),
        );
        for &dim in &tensor.dims {
            put_u64(&mut out, dim);
        }
        put_u32(&mut out, tensor.type_id);
        put_u64(&mut out, payload_total);
        payload_total += payload_bytes(tensor.type_id, &tensor.dims);
    }

    while out.len() % ALIGN != 0 {
        out.push(0);
    }
    (out, payload_total)
}

/// Writes a model to a temporary file that deletes itself.
///
/// The payload region is allocated with `set_len` rather than written, so a
/// model whose directory describes a full-size checkpoint costs a few kilobytes
/// of disk instead of gigabytes. Nothing in these tests reads payload bytes -
/// the reader only ever checks that they are inside the file.
fn write_model(kvs: &[(&str, Kv)], tensors: &[Spec]) -> NamedTempFile {
    let (bytes, payload) = model_bytes(kvs, tensors);
    let mut file = NamedTempFile::new().expect("temporary file");
    file.write_all(&bytes).expect("test model written");
    let len = u64::try_from(bytes.len()).expect("test header fits in u64") + payload;
    file.as_file().set_len(len).expect("test payload allocated");
    file.flush().expect("test model flushed");
    file
}

/// Opens a model straight from memory.
fn open_bytes(bytes: &[u8]) -> Result<GgufFile, GgufError> {
    let mut file = NamedTempFile::new().expect("temporary file");
    file.write_all(bytes).expect("test bytes written");
    file.flush().expect("test bytes flushed");
    GgufFile::open(file.path())
}

/// Builds a model whose file is shorter than its directory claims.
///
/// [`write_model`] allocates the whole payload region, so a model that declares
/// more bytes than the file holds has to be cut short instead. Only a slice of
/// the payload region exists on disk.
fn write_model_truncated(kvs: &[(&str, Kv)], tensors: &[Spec]) -> NamedTempFile {
    let (bytes, _payload) = model_bytes(kvs, tensors);
    let mut file = NamedTempFile::new().expect("temporary file");
    file.write_all(&bytes).expect("test model written");
    let header = u64::try_from(bytes.len()).expect("test header fits in u64");
    let slice = u64::try_from(ALIGN).expect("test alignment fits in u64");
    let len = header + slice;
    file.as_file().set_len(len).expect("test payload truncated");
    file.flush().expect("test model flushed");
    file
}

/// Builds and opens a synthetic model, expecting it to fail.
fn open_model(kvs: &[(&str, Kv)], tensors: &[Spec]) -> Result<GgufFile, GgufError> {
    let file = write_model(kvs, tensors);
    GgufFile::open(file.path())
}

/// Like [`open_model`], but the file is truncated inside the payload region.
fn open_model_truncated(kvs: &[(&str, Kv)], tensors: &[Spec]) -> Result<GgufFile, GgufError> {
    let file = write_model_truncated(kvs, tensors);
    GgufFile::open(file.path())
}
