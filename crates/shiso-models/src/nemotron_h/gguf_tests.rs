//! The expert source against a synthetic export: every slice it reads is the
//! one the mapping holds, and a file cut short under it is an I/O error.

use std::path::Path;

use shiso_core::CoreError;
use shiso_engine::sparse::ExpertSource;
use shiso_modelfile::GgufFile;

use super::{GgufExperts, iq4nl_expert_bytes};
use crate::test_gguf::{Kv, Tensor, write_gguf};

const EXPERTS: u32 = 3;
const HIDDEN: u32 = 64;
const WIDTH: u32 = 32;

/// Two routed blocks, 1 and 3, each with an up and a down bank of
/// distinguishable bytes.
fn write(path: &Path) {
    let bank = |block: u32, role: u32, in_dim: u32, out_dim: u32| {
        let bytes = iq4nl_expert_bytes(out_dim, in_dim) * u64::from(EXPERTS);
        Tensor {
            name: format!(
                "blk.{block}.{}.weight",
                if role == 0 {
                    "ffn_up_exps"
                } else {
                    "ffn_down_exps"
                }
            ),
            ggml_type: 20,
            dims: vec![u64::from(in_dim), u64::from(out_dim), u64::from(EXPERTS)],
            data: (0..bytes)
                .map(|i| {
                    u8::try_from((i * 7 + u64::from(block * 31 + role * 13)) % 251).expect("byte")
                })
                .collect(),
        }
    };
    write_gguf(
        path,
        &[("general.architecture", Kv::Str("nemotron_h_moe"))],
        &[
            bank(1, 0, HIDDEN, WIDTH),
            bank(1, 1, WIDTH, HIDDEN),
            bank(3, 0, HIDDEN, WIDTH),
            bank(3, 1, WIDTH, HIDDEN),
        ],
    );
}

#[test]
fn expert_read_equals_the_mapped_slice() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("decoder.gguf");
    write(&path);
    let file = GgufFile::open(&path).expect("opens");
    let experts = GgufExperts::open(&path, &file, &[1, 3], EXPERTS).expect("locates");
    let per = iq4nl_expert_bytes(WIDTH, HIDDEN);
    assert_eq!(experts.layout(), vec![per, per]);
    assert_eq!(experts.layers(), 2);
    for (layer, block) in [1u32, 3].iter().enumerate() {
        for (array, role) in ["ffn_up_exps", "ffn_down_exps"].iter().enumerate() {
            let mapped = file
                .get(&format!("blk.{block}.{role}.weight"))
                .expect("present")
                .data;
            for expert in 0..EXPERTS {
                let mut got = vec![0u8; usize::try_from(per).expect("fits")];
                experts
                    .read(u32::try_from(layer).expect("fits"), expert, array, &mut got)
                    .expect("reads");
                let at = usize::try_from(u64::from(expert) * per).expect("fits");
                assert_eq!(
                    got,
                    mapped[at..at + got.len()],
                    "block {block} {role} expert {expert}"
                );
            }
        }
    }
}

#[test]
fn short_file_read_is_an_io_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("decoder.gguf");
    write(&path);
    let experts = {
        let file = GgufFile::open(&path).expect("opens");
        GgufExperts::open(&path, &file, &[1, 3], EXPERTS).expect("locates")
    };
    // The export is cut short while the source holds it open: the last
    // bank's last expert is no longer on disk.
    let len = std::fs::metadata(&path).expect("stat").len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("opens for write")
        .set_len(len - 100)
        .expect("truncates");
    let mut buf = vec![0u8; usize::try_from(iq4nl_expert_bytes(HIDDEN, WIDTH)).expect("fits")];
    let err = experts
        .read(1, EXPERTS - 1, 1, &mut buf)
        .expect_err("past the end");
    assert!(matches!(err, CoreError::Io(_)), "{err:?}");
}
