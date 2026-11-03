//! The per-row INT8 layout and the `ConvRot` rotation, on the host and on the
//! device.
//!
//! The format refusals run against files this test forges, on any machine.
//! The real-weight checks read the Qwen-Image-2.1 Turbo INT8 export named by
//! `SHISO_TEST_MODEL_QWEN_IMAGE21` (the `turbo-int8` directory) and grade it
//! against `test-vectors/references/qwen_image21/convrot_*`, which an
//! independent implementation captured: `y_int8` is that implementation's
//! rotate-then-dequantized product and `y_bf16` the official BF16 weight's
//! unrotated one. Without the variable those cases say so on stderr and
//! return, which looks like a pass - check stderr before reporting one.
#![expect(
    clippy::cast_precision_loss,
    reason = "fixture indices converted to floats are small, exact in f32"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "the f64 oracle sums are narrowed to the f32 the fixtures store"
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::Backend;
use shiso_vision::generator::ImageGenError;

use super::lin::Lin;
use super::ops::{GemmCall, LinWeight, LinearResidualCall, Ops, WeightKind};
use super::quant::{RotGroup, parse_quant_spec, read_int8_layout, unrotate_row_f32};
use super::weights::StStore;

const DOC: &str = r#"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}"#;

// --- the format document -----------------------------------------------------

#[test]
fn the_exported_document_asks_for_the_256_rotation() {
    assert_eq!(
        parse_quant_spec("w", DOC.as_bytes()).expect("the export's own document"),
        Some(RotGroup::Hadamard256)
    );
    let plain = r#"{"format": "int8_tensorwise"}"#;
    assert_eq!(
        parse_quant_spec("w", plain.as_bytes()).expect("plain"),
        None
    );
    let off = r#"{"format": "int8_tensorwise", "convrot": false, "convrot_groupsize": 256}"#;
    assert_eq!(parse_quant_spec("w", off.as_bytes()).expect("off"), None);
}

/// Asserts `result` is a refusal naming `blk.w` and `needle`; `Ok(None)`
/// (a plain unrotated weight) fails it like any other answer.
fn refusal(result: Result<Option<RotGroup>, ImageGenError>, needle: &str) {
    match result {
        Err(ImageGenError::Checkpoint(message)) => {
            assert!(message.contains("'blk.w'"), "names the tensor: {message}");
            assert!(message.contains(needle), "says why ({needle}): {message}");
        }
        other => panic!("expected a checkpoint refusal mentioning {needle}, got {other:?}"),
    }
}

#[test]
fn a_document_this_build_does_not_implement_is_refused_naming_the_tensor() {
    refusal(
        parse_quant_spec("blk.w", br#"{"format": "float8_e4m3fn"}"#),
        "float8_e4m3fn",
    );
    refusal(
        parse_quant_spec(
            "blk.w",
            br#"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 128}"#,
        ),
        "128",
    );
    refusal(
        parse_quant_spec(
            "blk.w",
            br#"{"format": "int8_tensorwise", "convrot": true}"#,
        ),
        "convrot_groupsize",
    );
    refusal(parse_quant_spec("blk.w", b"\x00\x01"), "does not parse");
}

// --- the loader, against forged files ----------------------------------------

/// One forged tensor: header spelling, shape and payload bytes.
struct Forged<'a> {
    name: &'a str,
    dtype: &'a str,
    shape: &'a [u64],
    bytes: Vec<u8>,
}

fn write_safetensors(path: &Path, tensors: &[Forged<'_>]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    let mut payload = Vec::new();
    for t in tensors {
        header.insert(
            t.name.to_string(),
            serde_json::json!({
                "dtype": t.dtype,
                "shape": t.shape,
                "data_offsets": [offset, offset + t.bytes.len()],
            }),
        );
        offset += t.bytes.len();
        payload.extend_from_slice(&t.bytes);
    }
    let text = serde_json::Value::Object(header).to_string();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
    bytes.extend_from_slice(text.as_bytes());
    bytes.extend_from_slice(&payload);
    std::fs::write(path, bytes).expect("write safetensors");
}

fn f32_le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A forged `blk.w` linear: `out x in` codes, the given scale tensor and
/// (optionally) the format document.
fn forge(
    out: u64,
    input: u64,
    scale: Option<(&[u64], &[f32])>,
    doc: Option<&str>,
) -> (tempfile::TempDir, StStore) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("w.safetensors");
    let elements = usize::try_from(out * input).expect("tiny");
    let shape = [out, input];
    let doc_shape = [doc.map_or(0, |d| d.len() as u64)];
    let mut tensors = vec![Forged {
        name: "blk.w.weight",
        dtype: "I8",
        shape: &shape,
        bytes: vec![1u8; elements],
    }];
    if let Some((shape, values)) = scale {
        tensors.push(Forged {
            name: "blk.w.weight_scale",
            dtype: "F32",
            shape,
            bytes: f32_le(values),
        });
    }
    if let Some(doc) = doc {
        tensors.push(Forged {
            name: "blk.w.comfy_quant",
            dtype: "U8",
            shape: &doc_shape,
            bytes: doc.as_bytes().to_vec(),
        });
    }
    write_safetensors(&path, &tensors);
    let store = StStore::open(&path).expect("the forged file opens");
    (dir, store)
}

fn load_refusal(store: &StStore, needle: &str) {
    match read_int8_layout(store, "blk.w") {
        Err(ImageGenError::Checkpoint(message)) => {
            assert!(message.contains("blk.w"), "names the tensor: {message}");
            assert!(message.contains(needle), "says why ({needle}): {message}");
        }
        other => panic!("expected a refusal mentioning {needle}, got {other:?}"),
    }
}

#[test]
fn the_loader_reads_a_per_row_scale_and_broadcasts_a_scalar_one() {
    let rows: Vec<f32> = (0..4).map(|i| 0.5 + i as f32).collect();
    let (_dir, store) = forge(4, 256, Some((&[4, 1], &rows)), Some(DOC));
    let layout = read_int8_layout(&store, "blk.w").expect("a well-formed linear");
    assert_eq!(layout.rotate, Some(RotGroup::Hadamard256));
    assert_eq!(layout.row_scale, rows);

    for shape in [&[][..], &[1][..]] {
        let (_dir, store) = forge(4, 256, Some((shape, &[0.25])), Some(DOC));
        let layout = read_int8_layout(&store, "blk.w").expect("a scalar scale");
        assert_eq!(layout.row_scale, vec![0.25; 4], "scale shape {shape:?}");
    }
}

#[test]
fn the_loader_refuses_codes_whose_layout_it_would_have_to_guess() {
    let rows = vec![1.0f32; 4];
    let (_dir, store) = forge(4, 256, Some((&[4, 1], &rows)), None);
    load_refusal(&store, "comfy_quant");
    let (_dir, store) = forge(4, 256, None, Some(DOC));
    load_refusal(&store, "weight_scale");
    let (_dir, store) = forge(4, 256, Some((&[2, 2], &rows)), Some(DOC));
    load_refusal(&store, "[2, 2]");
    let (_dir, store) = forge(4, 255, Some((&[4, 1], &rows)), Some(DOC));
    load_refusal(&store, "multiple of the convrot group");
    let other = r#"{"format": "int4_blockwise"}"#;
    let (_dir, store) = forge(4, 256, Some((&[4, 1], &rows)), Some(other));
    load_refusal(&store, "int4_blockwise");
}

// --- the host rotation ---------------------------------------------------------

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/qwen_image21")
}

fn read_f32_file(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

/// The explicit group matrix, built from the Kronecker definition.
fn hadamard_matrix() -> Vec<f64> {
    const H4: [[f64; 4]; 4] = [
        [1.0, 1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0, 1.0],
    ];
    let mut h = vec![0f64; 256 * 256];
    for i in 0..256 {
        for j in 0..256 {
            let mut v = 1.0 / 16.0;
            for d in 0..4 {
                v *= H4[(i >> (2 * d)) & 3][(j >> (2 * d)) & 3];
            }
            h[i * 256 + j] = v;
        }
    }
    h
}

#[test]
fn the_host_rotation_is_the_explicit_matrix_and_the_reference_first_row() {
    let h = hadamard_matrix();
    let mut row: Vec<f32> = (0..512)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let input = row.clone();
    unrotate_row_f32(&mut row, 256).expect("two whole groups");
    for g in 0..2 {
        for j in 0..256 {
            let want: f64 = (0..256)
                .map(|i| f64::from(input[g * 256 + i]) * h[i * 256 + j])
                .sum();
            let got = f64::from(row[g * 256 + j]);
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                "group {g} col {j}"
            );
        }
    }
    // H's first row is e0 . H; the reference's own construction wrote it.
    let row0 = read_f32_file(&fixture_dir().join("convrot_hadamard_row0.f32"));
    let mut e0 = vec![0f32; 256];
    e0[0] = 1.0;
    unrotate_row_f32(&mut e0, 256).expect("one group");
    assert_eq!(e0, row0, "H row 0 against the reference");
    // Self-inverse: the same call recovers the input.
    unrotate_row_f32(&mut row, 256).expect("two whole groups");
    for (a, b) in row.iter().zip(&input) {
        assert!((a - b).abs() <= 1e-6 * b.abs().max(1.0));
    }
}

#[test]
fn the_host_rotation_refuses_a_group_it_does_not_implement() {
    let mut row = vec![0f32; 512];
    assert!(unrotate_row_f32(&mut row, 128).is_err());
    let mut ragged = vec![0f32; 300];
    assert!(unrotate_row_f32(&mut ragged, 256).is_err());
}

// --- the real export -----------------------------------------------------------

/// The three block-0 linears the reference captured: tensor base, fixture
/// stem, rows of `x`.
const LAYERS: [(&str, &str); 3] = [
    ("transformer_blocks.0.attn.to_q", "convrot_attn_to_q"),
    (
        "transformer_blocks.0.img_mlp.gate_up",
        "convrot_img_mlp_gate_up",
    ),
    ("transformer_blocks.0.img_mlp.out", "convrot_img_mlp_out"),
];
/// Rows of the reference's seeded input.
const ROWS: usize = 8;

#[expect(
    clippy::print_stderr,
    reason = "a skipped real-weight case must say why on stderr, or it reads as a pass"
)]
fn transformer_store() -> Option<StStore> {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_QWEN_IMAGE21") else {
        eprintln!(
            "note: SHISO_TEST_MODEL_QWEN_IMAGE21 is unset, skipping the INT8 real-weight case"
        );
        return None;
    };
    let dir = PathBuf::from(root).join("image/diffusion_models");
    Some(StStore::open(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())))
}

/// Per-row cosine of `got` against `want`, rows `width` wide.
fn row_cosines(got: &[f32], want: &[f32], width: usize) -> Vec<f64> {
    got.chunks_exact(width)
        .zip(want.chunks_exact(width))
        .map(|(g, w)| {
            let (mut gw, mut gg, mut ww) = (0f64, 0f64, 0f64);
            for (&a, &b) in g.iter().zip(w) {
                gw += f64::from(a) * f64::from(b);
                gg += f64::from(a) * f64::from(a);
                ww += f64::from(b) * f64::from(b);
            }
            gw / (gg.sqrt() * ww.sqrt())
        })
        .collect()
}

/// Largest `|got - want|` over the set's RMS: one number for the report.
fn rel_err(got: &[f32], want: &[f32]) -> f64 {
    let rms = (want.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (f64::from(g) - f64::from(w)).abs())
        .fold(0f64, f64::max)
        / rms
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the measured agreement is the case's report"
)]
fn the_real_int8_linears_match_the_reference_on_the_host() {
    let Some(store) = transformer_store() else {
        return;
    };
    let dir = fixture_dir();
    for (base, stem) in LAYERS {
        let layout = read_int8_layout(&store, base).expect("the export's layout reads");
        assert_eq!(layout.rotate, Some(RotGroup::Hadamard256), "{base}");
        let entry = store.require(&format!("{base}.weight")).expect("weight");
        let (out, input) = (
            usize::try_from(entry.shape[0]).expect("fits"),
            usize::try_from(entry.shape[1]).expect("fits"),
        );
        let codes: Vec<i8> = store
            .bytes(&format!("{base}.weight"))
            .expect("codes")
            .iter()
            .map(|&b| i8::from_le_bytes([b]))
            .collect();
        let corner = read_f32_file(&dir.join(format!("{stem}_q64.f32")));
        for r in 0..64 {
            for c in 0..64 {
                // The corner holds integers, exact in f32: compare their bits.
                assert_eq!(
                    f32::from(codes[r * input + c]).to_bits(),
                    corner[r * 64 + c].to_bits(),
                    "{base} q[{r},{c}]"
                );
            }
        }
        let scale64 = read_f32_file(&dir.join(format!("{stem}_scale64.f32")));
        assert_eq!(&layout.row_scale[..64], &scale64[..], "{base} scales");

        let mut x = read_f32_file(&dir.join(format!("{stem}_x.f32")));
        for row in x.chunks_exact_mut(input) {
            unrotate_row_f32(row, 256).expect("whole groups");
        }
        let mut y = vec![0f32; ROWS * out];
        for r in 0..ROWS {
            let xr = &x[r * input..(r + 1) * input];
            for o in 0..out {
                let qr = &codes[o * input..(o + 1) * input];
                let dot: f64 = xr
                    .iter()
                    .zip(qr)
                    .map(|(&a, &q)| f64::from(a) * f64::from(q))
                    .sum();
                y[r * out + o] = (dot * f64::from(layout.row_scale[o])) as f32;
            }
        }
        let y_int8 = read_f32_file(&dir.join(format!("{stem}_y_int8.f32")));
        let y_bf16 = read_f32_file(&dir.join(format!("{stem}_y_bf16.f32")));
        let err = rel_err(&y, &y_int8);
        let cos = row_cosines(&y, &y_bf16, out);
        let min_cos = cos.iter().copied().fold(1f64, f64::min);
        eprintln!(
            "host {base}: max |y - y_int8| / rms = {err:.2e}, min row cosine vs BF16 = {min_cos:.6}"
        );
        assert!(err <= 1e-5, "{base}: host rotate+dequant vs y_int8 {err:e}");
        assert!(min_cos >= 0.9999, "{base}: cosine vs BF16 {cos:?}");
    }
}

#[expect(
    clippy::print_stderr,
    reason = "the device cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn device() -> Option<Arc<dyn Backend>> {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the INT8 device case");
        return None;
    }
    let metal = shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");
    let backend: Arc<dyn Backend> = metal;
    Some(backend)
}

#[test]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_QWEN_IMAGE21"]
#[expect(
    clippy::print_stderr,
    reason = "the measured agreement is the case's report"
)]
fn the_real_int8_linears_match_the_reference_on_the_device() {
    let Some(store) = transformer_store() else {
        return;
    };
    let Some(backend) = device() else { return };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let dir = fixture_dir();
    for (base, stem) in LAYERS {
        let lin = Lin::load(&store, &ops, base).expect("the INT8 linear loads");
        assert_eq!(lin.kind, WeightKind::Int8Row);
        assert_eq!(lin.rotate, Some(RotGroup::Hadamard256));
        let (out, input) = (lin.out_dim, lin.in_dim);
        let x = read_f32_file(&dir.join(format!("{stem}_x.f32")));
        let d_x = ops.upload_f32("x", &x).expect("upload x");
        let y_int8 = read_f32_file(&dir.join(format!("{stem}_y_int8.f32")));
        let y_bf16 = read_f32_file(&dir.join(format!("{stem}_y_bf16.f32")));

        // The plain linear entry.
        let d_y = ops.alloc_owned_f32("y", ROWS * out).expect("alloc y");
        ops.gemm(&GemmCall {
            step: "int8 real gemm",
            out: d_y.view(),
            input: d_x.view(),
            weight: lin.weight(),
            m: ROWS,
            n: out,
            k: input,
        })
        .expect("gemm");
        backend.sync().expect("sync");
        let y = ops.download("y", d_y.view(), ROWS * out).expect("download");
        // The caller's input is never rotated in place.
        let x_after = ops.download("x", d_x.view(), x.len()).expect("download x");
        assert_eq!(x_after, x, "{base}: the input buffer was written");

        // The fused residual entry with a zero stream and a unit gate is the
        // same product, through the rotation pass that forms the row itself.
        let d_res = ops
            .upload_f32("res", &vec![0f32; ROWS * out])
            .expect("zeros");
        let d_gate = ops.upload_f32("gate", &vec![1f32; out]).expect("ones");
        ops.linear_residual(&LinearResidualCall {
            step: "int8 real fused",
            residual: d_res.view(),
            gate: d_gate.view(),
            dense: Some(d_x.view()),
            dense_cols: input,
            swiglu: None,
            weight: lin.weight(),
            m: ROWS,
            n: out,
        })
        .expect("linear_residual");
        backend.sync().expect("sync");
        let y_fused = ops
            .download("y", d_res.view(), ROWS * out)
            .expect("download");

        for (what, got) in [("gemm", &y), ("linear_residual", &y_fused)] {
            let err = rel_err(got, &y_int8);
            let cos = row_cosines(got, &y_bf16, out);
            let min_cos = cos.iter().copied().fold(1f64, f64::min);
            eprintln!(
                "device {what} {base}: max |y - y_int8| / rms = {err:.2e}, min row cosine vs BF16 = {min_cos:.6}"
            );
            assert!(err <= 1e-4, "{base} {what}: device vs y_int8 {err:e}");
            assert!(min_cos >= 0.9999, "{base} {what}: cosine vs BF16 {cos:?}");
        }
    }
}

/// Only the INT8 export stores a rotated weight, so a rotation riding any
/// other kind is a loader bug: both entries refuse it rather than rotate a
/// BF16/FP8/F32 product's input and return a different answer.
#[test]
fn a_rotation_on_a_weight_that_is_not_int8_is_refused() {
    let Some(backend) = device() else { return };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let (m, n, k) = (2, 4, 256);
    let weight = ops.upload_f32("w", &vec![0.5f32; n * k]).expect("w");
    let input = ops.upload_f32("x", &vec![1f32; m * k]).expect("x");
    let out = ops.alloc_owned_f32("out", m * n).expect("out");
    let gate = ops.upload_f32("gate", &vec![1f32; n]).expect("gate");
    let rotated = LinWeight::f32(weight.view()).with_rotation(Some(RotGroup::Hadamard256));
    let refused = |result: Result<(), ImageGenError>, what: &str| match result {
        Err(ImageGenError::Backend(message)) => {
            assert!(message.contains("rotation"), "{what}: {message}");
        }
        other => panic!("{what}: expected a refusal, got {other:?}"),
    };
    refused(
        ops.gemm(&GemmCall {
            step: "refuse",
            out: out.view(),
            input: input.view(),
            weight: rotated,
            m,
            n,
            k,
        }),
        "gemm",
    );
    refused(
        ops.linear_residual(&LinearResidualCall {
            step: "refuse",
            residual: out.view(),
            gate: gate.view(),
            dense: Some(input.view()),
            dense_cols: k,
            swiglu: None,
            weight: rotated,
            m,
            n,
        }),
        "linear_residual",
    );
}

// --- GGUF block formats ------------------------------------------------------

mod gguf_formats {
    use super::super::gguf_forge::{ForgedTensor, write_gguf};
    use super::super::ops::check_reduction;
    use super::super::weights::{BlockFormat, StStore};
    use super::{ImageGenError, WeightKind};
    use shiso_modelfile::st_header::StDType;

    const ALL: [(BlockFormat, usize, usize); 6] = [
        (BlockFormat::Q2K, 84, 256),
        (BlockFormat::Q3K, 110, 256),
        (BlockFormat::Q4K, 144, 256),
        (BlockFormat::Q5K, 176, 256),
        (BlockFormat::Q6K, 210, 256),
        (BlockFormat::Q8_0, 34, 32),
    ];

    #[test]
    fn row_bytes_per_format() {
        for (format, bytes, elems) in ALL {
            assert_eq!(format.block_bytes(), bytes, "{format:?}");
            assert_eq!(format.block_elems(), elems, "{format:?}");
            assert_eq!(format.row_bytes(768), 768 / elems * bytes, "{format:?}");
        }
    }

    #[test]
    fn gguf_open_maps_every_type() {
        // (name, GGUF type id, bytes per row of 256 weights, expected)
        let cases: [(&str, u32, usize, Option<BlockFormat>, StDType); 9] = [
            ("f32", 0, 1024, None, StDType::F32),
            ("q8_0", 8, 34 * 8, Some(BlockFormat::Q8_0), StDType::U8),
            ("q2k", 10, 84, Some(BlockFormat::Q2K), StDType::U8),
            ("q3k", 11, 110, Some(BlockFormat::Q3K), StDType::U8),
            ("q4k", 12, 144, Some(BlockFormat::Q4K), StDType::U8),
            ("q5k", 13, 176, Some(BlockFormat::Q5K), StDType::U8),
            ("q6k", 14, 210, Some(BlockFormat::Q6K), StDType::U8),
            ("bf16", 30, 512, None, StDType::Bf16),
            ("f16", 1, 512, None, StDType::F16),
        ];
        let payloads: Vec<Vec<u8>> = cases.iter().map(|c| vec![0u8; c.2]).collect();
        let tensors: Vec<ForgedTensor<'_>> = cases
            .iter()
            .zip(&payloads)
            .map(|(c, bytes)| ForgedTensor {
                name: c.0,
                ne: &[256, 1],
                type_id: c.1,
                bytes,
            })
            .collect();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("m.gguf");
        write_gguf(&path, &[("general.architecture", "flux")], &tensors);
        let store = StStore::open(&path).expect("every supported type opens");
        for (name, _, _, block, dtype) in cases {
            let entry = store.get(name).expect(name);
            assert_eq!(entry.block, block, "{name}");
            assert_eq!(entry.dtype, dtype, "{name}");
            assert_eq!(
                entry.shape,
                vec![1, 256],
                "{name}: shape is reversed to (out, in)"
            );
        }
    }

    #[test]
    fn unsupported_gguf_type_names_the_type() {
        // IQ4_XS: id 23, 136 bytes per 256 weights.
        let bytes = vec![0u8; 136];
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("m.gguf");
        write_gguf(
            &path,
            &[],
            &[ForgedTensor {
                name: "blk.0.attn.weight",
                ne: &[256, 1],
                type_id: 23,
                bytes: &bytes,
            }],
        );
        match StStore::open(&path) {
            Err(ImageGenError::Checkpoint(message)) => {
                assert!(message.contains("iq4_xs"), "names the type: {message}");
                assert!(
                    message.contains("blk.0.attn.weight"),
                    "names the tensor: {message}"
                );
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn q8_0_needs_k_multiple_of_32_others_256() {
        check_reduction("step.a", WeightKind::Q8_0, 96).expect("96 is three Q8_0 blocks");
        for kind in [
            WeightKind::Q2K,
            WeightKind::Q3K,
            WeightKind::Q4K,
            WeightKind::Q5K,
            WeightKind::Q6K,
        ] {
            check_reduction("step.a", kind, 768).expect("768 is three k-blocks");
            for k in [96, 272] {
                match check_reduction("step.a", kind, k) {
                    Err(ImageGenError::Backend(message)) => {
                        assert!(message.contains("step.a"), "names the step: {message}");
                        assert!(message.contains(&k.to_string()), "names k: {message}");
                        assert!(message.contains("256"), "says the multiple: {message}");
                    }
                    other => panic!("{kind:?} k={k}: expected a refusal, got {other:?}"),
                }
            }
        }
        match check_reduction("step.b", WeightKind::Q8_0, 48) {
            Err(ImageGenError::Backend(message)) => {
                assert!(
                    message.contains("step.b") && message.contains("48"),
                    "{message}"
                );
                assert!(message.contains("32"), "{message}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        check_reduction("step.c", WeightKind::Bf16, 17).expect("plain kinds have no block");
    }
}

#[test]
fn the_document_a_real_export_writes_parses() {
    // Verbatim from a published per-row ConvRot export: the format name says
    // tensorwise, `per_row` marks the [out, 1] scale.
    let doc = br#"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256, "per_row": true}"#;
    assert_eq!(
        parse_quant_spec("blk.w.weight", doc).expect("parses"),
        Some(RotGroup::Hadamard256)
    );
}
