//! Checks the affine decode against the array framework's own implementation.
//!
//! The unit tests pin the decode against hand-written words, which catches a
//! reversed field order and a misplaced group boundary. They cannot catch the
//! decode being a self-consistent misreading of what the exporter actually
//! wrote - for that the answer has to come from the framework that wrote the
//! file. So this test asks it, on a real tensor, and compares.
//!
//! Ignored by default: it needs a Python environment with the array framework
//! and its checkpoint reader installed, which is not something a build should
//! require. Run it with `cargo test -p shiso-modelfile -- --ignored`.
//!
//! The oracle casts scales and biases to `f32` *before* dequantizing. Left as
//! `bf16`, the framework would produce a `bf16` result whose eight-bit mantissa
//! carries more rounding error than the tolerance being checked, and the test
//! would be measuring the oracle rather than the decode.

mod optiq;

use std::path::Path;
use std::process::Command;

use half::f16;

/// Tensor the oracle and the loader both decode.
const TENSOR: &str = "blk.0.ffn_gate.weight";
/// The same tensor under the checkpoint's own spelling.
const SOURCE_TENSOR: &str = "language_model.model.layers.0.mlp.gate_proj";
/// Rows compared; the oracle dumps exactly this many.
const ROWS: usize = 64;
/// Weights per row of this tensor.
const ROW_WIDTH: usize = 2560;
/// Weights sharing a scale and bias.
const GROUP_SIZE: usize = 64;
/// Bits per weight in this tensor.
const BITS: u32 = 4;
/// Absolute tolerance, and the coefficient of the relative one.
const TOLERANCE: f32 = 1e-3;

/// The script that dumps the oracle, with the tensor and geometry filled in.
fn oracle_script() -> String {
    format!(
        r#"
import sys, mlx.core as mx
from safetensors import safe_open
with safe_open(sys.argv[1], framework="mlx") as f:
    q = f.get_tensor("{SOURCE_TENSOR}.weight")
    s = f.get_tensor("{SOURCE_TENSOR}.scales").astype(mx.float32)
    b = f.get_tensor("{SOURCE_TENSOR}.biases").astype(mx.float32)
mx.dequantize(q, s, b, group_size={GROUP_SIZE}, bits={BITS})[0:{ROWS}].tofile(sys.argv[2])
"#
    )
}

/// Whether this machine has the Python packages the oracle needs.
fn oracle_available() -> bool {
    Command::new("python3")
        .args(["-c", "import mlx.core, safetensors"])
        .output()
        .is_ok_and(|out| out.status.success())
}

/// Runs the oracle and reads back its `f32` dump.
fn oracle(dir: &Path) -> Option<Vec<f32>> {
    let out_path = std::env::temp_dir().join("shiso-optiq-oracle-gate0.bin");
    let status = Command::new("python3")
        .arg("-")
        .arg(dir.join("model.safetensors"))
        .arg(&out_path)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .ok()
        .and_then(|mut child| {
            use std::io::Write as _;
            child
                .stdin
                .as_mut()?
                .write_all(oracle_script().as_bytes())
                .ok()?;
            child.wait().ok()
        })?;
    if !status.success() {
        return None;
    }
    let bytes = std::fs::read(&out_path).ok()?;
    drop(std::fs::remove_file(&out_path));
    Some(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|quad| f32::from_le_bytes(*quad))
            .collect(),
    )
}

#[test]
#[ignore = "needs a Python environment with the array framework installed"]
fn the_decode_matches_the_framework_that_wrote_the_file() {
    let Some(dir) = optiq::checkpoint_dir() else {
        optiq::skip("checkpoint directory not found");
        return;
    };
    if !oracle_available() {
        optiq::skip(
            "python3 has no array framework or checkpoint reader; install them to run this",
        );
        return;
    }
    let Some(file) = optiq::open_checkpoint() else {
        return;
    };
    let Some(want) = oracle(&dir) else {
        optiq::skip("the oracle script did not produce a dump");
        return;
    };
    assert_eq!(
        want.len(),
        ROWS * ROW_WIDTH,
        "the oracle dumped {ROWS} rows"
    );

    let bytes = file.tensor_bytes(TENSOR).expect("the weight decodes");
    let got: Vec<f32> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .take(ROWS * ROW_WIDTH)
        .map(|pair| f16::from_le_bytes(*pair))
        .map(f16::to_f32)
        .collect();
    assert_eq!(got.len(), want.len());

    // Mixed absolute and relative: the absolute term covers the weights near
    // zero, where a relative bound would be unmeetable at half precision, and
    // the relative term covers the large ones, where an absolute bound would be
    // vacuous.
    let mut worst = 0.0_f32;
    let mut worst_at = 0_usize;
    for (at, (&got, &want)) in got.iter().zip(want.iter()).enumerate() {
        let slack = TOLERANCE + TOLERANCE * want.abs();
        let error = (got - want).abs();
        if error / slack > worst {
            worst = error / slack;
            worst_at = at;
        }
    }
    assert!(
        worst <= 1.0,
        "weight {worst_at} of {TENSOR} is {} against the oracle's {}, \
         which is {worst:.3} times the allowed slack",
        got[worst_at],
        want[worst_at]
    );
}
