//! The modelopt container, over synthetic shards and over the real one.
//!
//! The synthetic cases run everywhere and are what pin the arithmetic: which
//! scheme a weight's own format selects, how a packed shape becomes a logical
//! one, and which disagreement between a weight and its scales is refused. The
//! real-checkpoint case runs only where the 23.4 GB snapshot is, and says so
//! out loud when it does not - the plan this work comes from records three
//! lanes on this branch that were found silently skipping, and a case that
//! prints nothing is how that happens.

use std::io::Write;
use std::path::PathBuf;

use super::{ModeloptFile, Stored};
use crate::reader::GgufError;
use crate::st_header::StDType;

/// Writes a one-shard directory whose header is `header` and payload `payload`.
fn checkpoint(header: &str, payload: &[u8]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let mut file = std::fs::File::create(dir.path().join("model.safetensors")).expect("create");
    let bytes = header.as_bytes();
    file.write_all(&u64::try_from(bytes.len()).expect("fits").to_le_bytes())
        .expect("length");
    file.write_all(bytes).expect("header");
    file.write_all(payload).expect("payload");
    dir
}

/// Four little-endian bytes of `value`.
fn f32_bytes(value: f32) -> [u8; 4] {
    value.to_le_bytes()
}

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// Prints a measurement; the plan asks for one and a number nobody sees is not
/// a measurement.
#[expect(clippy::print_stdout, reason = "the measurement is this test's output")]
fn report(what: &str) {
    println!("{what}");
}

/// The real checkpoint, or `None` when it is not on this machine.
fn snapshot() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_QWEN35MOE").ok()?);
    path.join("model.safetensors.index.json")
        .is_file()
        .then_some(path)
}

/// One NVFP4 tensor of `[2, 32]` weights: 32 bytes packed, 4 group scales.
fn nvfp4_shard() -> tempfile::TempDir {
    let header = r#"{"__metadata__":{"format":"pt"},
        "w.weight":{"dtype":"U8","shape":[2,16],"data_offsets":[0,32]},
        "w.weight_scale":{"dtype":"F8_E4M3","shape":[2,2],"data_offsets":[32,36]},
        "w.weight_scale_2":{"dtype":"F32","shape":[],"data_offsets":[36,40]}}"#;
    let mut payload = vec![0x21_u8; 32];
    payload.extend_from_slice(&[0x38, 0x38, 0x38, 0x38]);
    payload.extend_from_slice(&f32_bytes(0.25));
    checkpoint(header, &payload)
}

#[test]
fn a_packed_shape_is_half_the_matrix_and_the_logical_one_is_kept() {
    let dir = nvfp4_shard();
    let file = ModeloptFile::open(dir.path(), &[]).expect("one nvfp4 tensor");
    let tensor = file.get("w").expect("by base name");
    // The header says [2, 16] bytes. The matrix is [2, 32] weights, and a
    // caller that sized a buffer from the header would size it at half.
    assert_eq!((tensor.rows(), tensor.cols()), (2, 32));
    assert_eq!(tensor.elements(), 64);
    assert_eq!(tensor.stored(), Stored::Nvfp4 { global: 0.25 });
    // The same tensor is reachable under the name the config document uses.
    assert_eq!(file.get("w.weight").expect("with the suffix").name(), "w");
}

#[test]
fn the_three_nvfp4_arrays_come_back_together_with_their_layout() {
    let dir = nvfp4_shard();
    let file = ModeloptFile::open(dir.path(), &[]).expect("one nvfp4 tensor");
    let tensor = file.get("w").expect("by base name");
    let parts = file
        .nvfp4_parts(tensor)
        .expect("in bounds")
        .expect("an nvfp4 tensor");
    assert_eq!(parts.weights.len(), 32);
    assert_eq!(parts.scales.len(), 4);
    assert_eq!(parts.layout.out_rows, 2);
    assert_eq!(parts.layout.in_dim, 32);
    // The third level, which is the one that is easy to drop.
    assert!((parts.layout.global - 0.25).abs() < f32::EPSILON);
}

#[test]
fn a_scale_array_that_does_not_cover_the_packed_row_is_refused() {
    // One group short. Every later group would read its scale from the row
    // below, which decodes to finite numbers everywhere.
    let header = r#"{"__metadata__":{"format":"pt"},
        "w.weight":{"dtype":"U8","shape":[2,16],"data_offsets":[0,32]},
        "w.weight_scale":{"dtype":"F8_E4M3","shape":[2,1],"data_offsets":[32,34]},
        "w.weight_scale_2":{"dtype":"F32","shape":[],"data_offsets":[34,38]}}"#;
    let mut payload = vec![0u8; 34];
    payload.extend_from_slice(&f32_bytes(1.0));
    let dir = checkpoint(header, &payload);
    match ModeloptFile::open(dir.path(), &[]).expect_err("two groups, not one") {
        GgufError::LayoutMismatch { tensor, got, .. } => {
            assert_eq!(&*tensor, "w");
            assert!(got.contains("[2, 1]"), "{got}");
        }
        other => panic!("expected LayoutMismatch, got {other:?}"),
    }
}

#[test]
fn nvfp4_scales_that_are_not_e4m3_are_refused() {
    let header = r#"{"__metadata__":{"format":"pt"},
        "w.weight":{"dtype":"U8","shape":[2,16],"data_offsets":[0,32]},
        "w.weight_scale":{"dtype":"BF16","shape":[2,2],"data_offsets":[32,40]},
        "w.weight_scale_2":{"dtype":"F32","shape":[],"data_offsets":[40,44]}}"#;
    let mut payload = vec![0u8; 40];
    payload.extend_from_slice(&f32_bytes(1.0));
    let dir = checkpoint(header, &payload);
    match ModeloptFile::open(dir.path(), &[]).expect_err("e4m3, not bf16") {
        GgufError::LayoutMismatch { got, .. } => assert!(got.contains("BF16"), "{got}"),
        other => panic!("expected LayoutMismatch, got {other:?}"),
    }
}

#[test]
fn an_nvfp4_weight_with_no_per_tensor_scale_is_refused_by_name() {
    // The level that is easy to drop, dropped. Its absence has to be a refusal
    // naming the array, because a decode without it is wrong by about 31,458x
    // and the next normalization very nearly hides that.
    let header = r#"{"__metadata__":{"format":"pt"},
        "w.weight":{"dtype":"U8","shape":[2,16],"data_offsets":[0,32]},
        "w.weight_scale":{"dtype":"F8_E4M3","shape":[2,2],"data_offsets":[32,36]}}"#;
    let dir = checkpoint(header, &[0u8; 36]);
    match ModeloptFile::open(dir.path(), &[]).expect_err("no weight_scale_2") {
        GgufError::MissingTensor { name } => assert_eq!(&*name, "w.weight_scale_2"),
        other => panic!("expected MissingTensor, got {other:?}"),
    }
}

#[test]
fn an_fp8_weight_carries_both_of_its_scales_and_an_activation_scale_is_optional() {
    let header = r#"{"__metadata__":{"format":"pt"},
        "a.weight":{"dtype":"F8_E4M3","shape":[2,4],"data_offsets":[0,8]},
        "a.weight_scale":{"dtype":"F32","shape":[],"data_offsets":[8,12]},
        "a.input_scale":{"dtype":"F32","shape":[],"data_offsets":[12,16]},
        "b.weight":{"dtype":"F8_E4M3","shape":[2,4],"data_offsets":[16,24]},
        "b.weight_scale":{"dtype":"F32","shape":[],"data_offsets":[24,28]}}"#;
    let mut payload = vec![0u8; 8];
    payload.extend_from_slice(&f32_bytes(0.5));
    payload.extend_from_slice(&f32_bytes(2.0));
    payload.extend_from_slice(&[0u8; 8]);
    payload.extend_from_slice(&f32_bytes(0.125));
    let dir = checkpoint(header, &payload);
    let file = ModeloptFile::open(dir.path(), &[]).expect("two fp8 tensors");
    assert_eq!(
        file.get("a").expect("a").stored(),
        Stored::Fp8 {
            weight_scale: 0.5,
            input_scale: Some(2.0),
        }
    );
    // `None` is a target whose activations were not quantized, which is a
    // different decision at the kernel from "an activation scale of one".
    assert_eq!(
        file.get("b").expect("b").stored(),
        Stored::Fp8 {
            weight_scale: 0.125,
            input_scale: None,
        }
    );
}

#[test]
fn an_fp8_weight_with_no_weight_scale_is_refused_by_name() {
    let header = r#"{"__metadata__":{"format":"pt"},
        "a.weight":{"dtype":"F8_E4M3","shape":[2,4],"data_offsets":[0,8]}}"#;
    let dir = checkpoint(header, &[0u8; 8]);
    match ModeloptFile::open(dir.path(), &[]).expect_err("no weight_scale") {
        GgufError::MissingTensor { name } => assert_eq!(&*name, "a.weight_scale"),
        other => panic!("expected MissingTensor, got {other:?}"),
    }
}

#[test]
fn a_scalar_scale_that_is_not_one_f32_is_refused() {
    let header = r#"{"__metadata__":{"format":"pt"},
        "a.weight":{"dtype":"F8_E4M3","shape":[2,4],"data_offsets":[0,8]},
        "a.weight_scale":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}"#;
    let dir = checkpoint(header, &[0u8; 16]);
    match ModeloptFile::open(dir.path(), &[]).expect_err("one f32, not two") {
        GgufError::LayoutMismatch { tensor, want, .. } => {
            assert_eq!(&*tensor, "a.weight_scale");
            assert_eq!(&*want, "one f32");
        }
        other => panic!("expected LayoutMismatch, got {other:?}"),
    }
}

#[test]
fn a_plain_tensor_keeps_its_rank_and_has_no_scale() {
    // The recurrent convolution is rank 3 - `[channels, 1, taps]` - and the
    // norms are rank 1, so the plain arm folds everything past the first axis
    // rather than insisting on two.
    let header = r#"{"__metadata__":{"format":"pt"},
        "conv1d.weight":{"dtype":"BF16","shape":[4,1,4],"data_offsets":[0,32]},
        "norm.weight":{"dtype":"BF16","shape":[4],"data_offsets":[32,40]}}"#;
    let dir = checkpoint(header, &[0u8; 40]);
    let file = ModeloptFile::open(dir.path(), &[]).expect("two plain tensors");
    let conv = file.get("conv1d").expect("conv1d");
    assert_eq!((conv.rows(), conv.cols()), (4, 4));
    assert_eq!(
        conv.stored(),
        Stored::Plain {
            dtype: StDType::Bf16
        }
    );
    assert_eq!(file.get("norm").expect("norm").elements(), 4);
}

#[test]
fn the_mlx_container_s_packed_words_are_refused_here() {
    // The two containers are both safetensors directories and both quantized,
    // and reading one with the other's rules decodes to finite numbers. So a
    // `U32` weight is a refusal naming the other container rather than a
    // fall-through.
    let header = r#"{"__metadata__":{"format":"pt"},
        "a.weight":{"dtype":"U32","shape":[2,4],"data_offsets":[0,32]}}"#;
    let dir = checkpoint(header, &[0u8; 32]);
    match ModeloptFile::open(dir.path(), &[]).expect_err("u32 is the mlx container's") {
        GgufError::LayoutMismatch { got, .. } => assert!(got.contains("MLX"), "{got}"),
        other => panic!("expected LayoutMismatch, got {other:?}"),
    }
}

#[test]
fn the_prefix_filter_describes_nothing_it_was_not_asked_for() {
    let header = r#"{"__metadata__":{"format":"pt"},
        "model.a.weight":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},
        "mtp.b.weight":{"dtype":"BF16","shape":[4],"data_offsets":[8,16]}}"#;
    let dir = checkpoint(header, &[0u8; 16]);
    let all = ModeloptFile::open(dir.path(), &[]).expect("no filter keeps everything");
    assert_eq!(all.census().tensors, 2);
    let kept = ModeloptFile::open(dir.path(), &["model."]).expect("one prefix");
    assert_eq!(kept.census().tensors, 1);
    assert!(kept.get("mtp.b").is_none());
    // The arrays are still counted: a filter narrows what is described, not
    // what was found, so a caller can notice a shard it did not expect.
    assert_eq!(kept.census().arrays, 2);
}

#[test]
fn the_real_checkpoint_s_census_is_what_the_plan_measured() {
    let Some(dir) = snapshot() else {
        skip("SHISO_TEST_MODEL_QWEN35MOE names no checkpoint directory");
        return;
    };
    let started = std::time::Instant::now();
    let file = ModeloptFile::open(&dir, &[]).expect("the real checkpoint");
    let elapsed = started.elapsed();
    let census = file.census();
    report(&format!(
        "modelopt open: {} shards, {} arrays, {} tensors in {:.3} s",
        file.shard_count(),
        census.arrays,
        census.tensors,
        elapsed.as_secs_f64()
    ));

    // The array census, which is the checkpoint as the headers declare it.
    assert_eq!(file.shard_count(), 3);
    assert_eq!(census.arrays, 94_393);

    // The tensor census, which is the same file with each scheme's companions
    // folded into the weight they belong to. 30,841 NVFP4 targets: 40 layers
    // times (256 experts plus one shared expert) times three projections, plus
    // `lm_head`. 130 FP8: 30 recurrent slots times three projections plus 10
    // attention slots times four.
    assert_eq!(40 * (256 + 1) * 3 + 1, 30_841);
    assert_eq!(30 * 3 + 10 * 4, 130);
    assert_eq!(census.nvfp4, 30_841);
    assert_eq!(census.fp8, 130);
    assert_eq!(census.plain + census.fp8 + census.nvfp4, census.tensors);

    // Every array is accounted for, and this is the assertion that matters:
    // each NVFP4 tensor is three arrays, each FP8 tensor here is three, and
    // each plain tensor is one. An earlier draft of the describe pass kept
    // only arrays ending in `.weight`, which silently dropped 226 of them -
    // every bias in the vision tower, and the recurrent block's `A_log` and
    // `dt_bias`, without which the delta rule has no decay at all. Nothing
    // else in this file would have noticed.
    let fp8_with_input = file
        .tensors()
        .filter(|tensor| {
            matches!(
                tensor.stored(),
                Stored::Fp8 {
                    input_scale: Some(_),
                    ..
                }
            )
        })
        .count();
    assert_eq!(fp8_with_input, 130);
    assert_eq!(
        census.nvfp4 * 3 + census.fp8 * 3 + census.plain,
        census.arrays
    );
    assert_eq!(census.tensors, 32_451);

    // The two recurrent parameters that carry no `.weight`, by name.
    for name in [
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.dt_bias",
    ] {
        assert!(file.get(name).is_some(), "{name}");
    }
}

#[test]
fn the_real_checkpoint_s_expert_geometry_is_what_the_router_will_index() {
    let Some(dir) = snapshot() else {
        skip("SHISO_TEST_MODEL_QWEN35MOE names no checkpoint directory");
        return;
    };
    let file = ModeloptFile::open(&dir, &["model.language_model.", "lm_head"])
        .expect("the language model");

    // One tensor per expert, which is what makes this a different loader from
    // the sparse family's: that one addresses an expert by a stride into one
    // stacked slab.
    let experts = file
        .tensors()
        .filter(|tensor| tensor.name().contains(".mlp.experts."))
        .count();
    assert_eq!(experts, 40 * 256 * 3);

    // The widths the routed stage will be launched over. `gate_proj` is
    // hidden -> moe_intermediate and `down_proj` comes back.
    let gate = file
        .get("model.language_model.layers.0.mlp.experts.0.gate_proj")
        .expect("expert zero's gate");
    assert_eq!((gate.rows(), gate.cols()), (512, 2048));
    let down = file
        .get("model.language_model.layers.0.mlp.experts.0.down_proj")
        .expect("expert zero's down");
    assert_eq!((down.rows(), down.cols()), (2048, 512));

    // The output projection is NVFP4 too, which is the one target outside the
    // feed-forward that is.
    let head = file.get("lm_head").expect("the output projection");
    assert_eq!((head.rows(), head.cols()), (248_320, 2048));
    assert!(matches!(head.stored(), Stored::Nvfp4 { .. }));

    // And the router itself is not quantized at all.
    let router = file
        .get("model.language_model.layers.0.mlp.gate")
        .expect("the router projection");
    assert_eq!((router.rows(), router.cols()), (256, 2048));
    assert_eq!(
        router.stored(),
        Stored::Plain {
            dtype: StDType::Bf16
        }
    );
}
