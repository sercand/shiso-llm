//! The tensor store over both containers, and host dequantisation.

use std::io::Write;

use super::{StoredType, TensorStore, dequant, dequant_affine};

/// Writes a safetensors file holding `tensors` (name, dtype spelling, shape,
/// little-endian bytes) and returns it.
fn safetensors(tensors: &[(&str, &str, Vec<u64>, Vec<u8>)]) -> tempfile::NamedTempFile {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = payload.len();
        payload.extend_from_slice(bytes);
        header.insert(
            (*name).to_owned(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()]}),
        );
    }
    let text = serde_json::to_vec(&serde_json::Value::Object(header)).expect("serialises");
    let mut file = tempfile::NamedTempFile::new().expect("a temp file");
    file.write_all(&u64::try_from(text.len()).expect("small").to_le_bytes())
        .expect("writes");
    file.write_all(&text).expect("writes");
    file.write_all(&payload).expect("writes");
    file.flush().expect("flushes");
    file
}

fn le_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn safetensors_tensors_keep_torch_shapes_and_dequantise() {
    let bf16: Vec<u8> = [1.5f32, -2.0, 0.25]
        .iter()
        .flat_map(|v| {
            u16::try_from(v.to_bits() >> 16)
                .expect("the high half")
                .to_le_bytes()
        })
        .collect();
    let file = safetensors(&[
        (
            "a.weight",
            "F32",
            vec![2, 3],
            le_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ),
        ("b", "BF16", vec![3], bf16),
    ]);
    let store = TensorStore::open_safetensors(file.path()).expect("opens");
    let a = store.require("a.weight").expect("present");
    assert_eq!(
        (a.ty.clone(), a.shape.clone()),
        (StoredType::F32, vec![2, 3])
    );
    assert_eq!(
        dequant(&a).expect("f32"),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    let b = store.require("b").expect("present");
    assert_eq!(b.ty, StoredType::Bf16);
    assert_eq!(dequant(&b).expect("bf16"), vec![1.5, -2.0, 0.25]);
    let missing = store.require("nope").expect_err("absent");
    assert!(missing.to_string().contains("nope"), "{missing}");
}

#[test]
fn q8_0_blocks_dequantise_to_scale_times_byte() {
    let mut bytes = half::f16::from_f32(0.5).to_le_bytes().to_vec();
    bytes.extend((0..32i8).map(|q| (q - 16).to_le_bytes()[0]));
    let file = safetensors(&[("q", "U8", vec![34], bytes.clone())]);
    let store = TensorStore::open_safetensors(file.path()).expect("opens");
    let mut t = store.require("q").expect("present");
    // Present the raw bytes as a Q8_0 row of 32, as a GGUF directory would.
    t.ty = StoredType::Q8_0;
    t.shape = vec![32];
    let v = dequant(&t).expect("q8_0");
    assert_eq!(v.len(), 32);
    assert!((v[0] - (-16.0 * 0.5)).abs() < 1e-6 && (v[31] - 7.5).abs() < 1e-6);
}

#[test]
fn affine_rows_unpack_low_bits_first() {
    // One row of 64 weights at 8 bits: 16 words; element i is byte i.
    let q: Vec<u8> = (0..64u8).collect();
    let (scale, bias) = (0.5f32, -1.0f32);
    let bf = |v: f32| {
        u16::try_from(v.to_bits() >> 16)
            .expect("high half")
            .to_le_bytes()
            .to_vec()
    };
    let file = safetensors(&[
        ("w.weight", "U32", vec![1, 16], q.clone()),
        ("w.scales", "BF16", vec![1, 1], bf(scale)),
        ("w.biases", "BF16", vec![1, 1], bf(bias)),
    ]);
    let store = TensorStore::open_safetensors(file.path()).expect("opens");
    let words = store.require("w.weight").expect("words");
    let scales = store.require("w.scales").expect("scales");
    let biases = store.require("w.biases").expect("biases");
    let values = dequant_affine(&words, &scales, &biases, 8, 64).expect("dequantises");
    for (i, x) in values.iter().enumerate() {
        let want = f32::from(u8::try_from(i).expect("small")) * scale + bias;
        assert!(
            (x - want).abs() <= 1e-2 * want.abs().max(1.0),
            "[{i}] {x} vs {want}"
        );
    }
}

#[test]
fn a_gguf_tensor_reports_its_torch_shape() {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_ASR") else {
        return skip("SHISO_TEST_MODEL_ASR is unset");
    };
    let path = std::path::Path::new(&root).join("gguf/nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf");
    if !path.exists() {
        return skip("the parakeet Q4_K_M GGUF is absent");
    }
    let store = TensorStore::open_gguf(&path).expect("opens");
    let conv = store
        .require("enc.pre_encode.conv.0.weight")
        .expect("present");
    assert_eq!(
        (conv.ty.clone(), conv.shape.clone()),
        (StoredType::F32, vec![256, 1, 3, 3])
    );
    assert!(store.gguf().is_some());
}

#[expect(clippy::print_stdout, reason = "a skipped case says why on stdout")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}
