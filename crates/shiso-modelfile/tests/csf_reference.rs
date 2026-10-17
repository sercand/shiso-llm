//! The CSF decode against matrices the reference decoded with its own kernels.
//!
//! `test-vectors/references/qwen4exp-csf/` holds one routed expert of the
//! NVIDIA modelopt export (`model.language_model.layers.5.mlp.experts.137`,
//! gate and down projections): the raw `nvfp4_csf_fixed` and
//! `nvfp4_csf_exceptions` arrays as the shard stores them, and the plain
//! `e4m3` scale matrix the reference's storage-conversion kernels expanded
//! them to, run on CPU in the reference container. The decode has to match
//! that matrix byte for byte.

use std::path::{Path, PathBuf};

use shiso_modelfile::csf::{CsfLayout, decode_csf};

/// Where the fixtures live.
fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/qwen4exp-csf")
}

/// The payload and shape of a `.npy` file (format 1.0 or 2.0, C order).
fn npy(name: &str) -> (Vec<u8>, Vec<usize>) {
    let bytes = std::fs::read(dir().join(name)).expect("the fixture is on this machine");
    assert_eq!(&bytes[..6], b"\x93NUMPY", "{name}: not a .npy file");
    let (len, start) = if bytes[6] == 1 {
        (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10)
    } else {
        let word = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        (usize::try_from(word).expect("header fits"), 12)
    };
    let header = std::str::from_utf8(&bytes[start..start + len]).expect("ascii header");
    assert!(
        header.contains("'fortran_order': False"),
        "{name}: {header}"
    );
    let shape_text = header
        .split("'shape': (")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .expect("a shape");
    let shape = shape_text
        .split(',')
        .map(str::trim)
        .filter(|dim| !dim.is_empty())
        .map(|dim| dim.parse().expect("a dimension"))
        .collect();
    (bytes[start + len..].to_vec(), shape)
}

fn replay(projection: &str, layout: CsfLayout) {
    let (fixed, fixed_shape) = npy(&format!("golden_l5_e137_{projection}_csf_fixed.npy"));
    let (exceptions, _) = npy(&format!("golden_l5_e137_{projection}_csf_exceptions.npy"));
    let (want, want_shape) = npy(&format!("golden_l5_e137_{projection}_scales_e4m3.npy"));
    assert_eq!(fixed_shape, [layout.rows / 16, layout.record_bytes()]);
    assert_eq!(want_shape, [layout.rows, layout.scale_cols()]);
    let mut got = vec![0u8; layout.scale_bytes()];
    decode_csf(projection, layout, &fixed, &exceptions, &mut got).expect("decodes");
    let first = got.iter().zip(&want).position(|(g, w)| g != w);
    assert_eq!(
        first, None,
        "{projection}: first differing scale at {first:?}"
    );
    assert!(
        !exceptions.is_empty(),
        "{projection}: the fixture exercises exceptions"
    );
}

#[test]
fn the_gate_projection_decodes_to_the_references_scales() {
    replay(
        "gate_proj",
        CsfLayout {
            rows: 640,
            cols: 2560,
        },
    );
}

#[test]
fn the_down_projection_decodes_to_the_references_scales() {
    replay(
        "down_proj",
        CsfLayout {
            rows: 2560,
            cols: 640,
        },
    );
}
