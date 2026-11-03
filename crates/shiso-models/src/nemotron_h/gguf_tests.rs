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

/// A tiny export of the whole layout: block 0 state-space, 1 routed, 2
/// attention, 3 routed, and the prediction block 4 (attention and routed in
/// one), every tensor the loader binds at its real element count.
mod tiny {
    use std::path::Path;
    use std::sync::Arc;

    use shiso_core::CoreError;
    use shiso_core::backend::Backend;
    use shiso_cpu::{CpuBackend, CpuOptions};
    use shiso_engine::EngineOptions;
    use shiso_engine::layers::Block;
    use shiso_engine::sparse::SpecDepth;
    use shiso_modelfile::GgufFile;

    use super::super::{ExpertPlacement, draft_parts, iq4nl_expert_bytes, weights_gguf};
    use crate::nemotron_h::{GgufResidency, NemotronHConfig, NemotronHModel};
    use crate::test_gguf::{Kv, Tensor, write_gguf};

    const HIDDEN: u64 = 64;
    const VOCAB: u64 = 32;
    const EXPERTS: u64 = 4;
    const EXPERT_WIDTH: u64 = 32;
    const SHARED_WIDTH: u64 = 64;
    const Q: u64 = 64;
    const KV: u64 = 32;
    const INNER: u64 = 128;
    const SSM_HEADS: u64 = 4;
    const STATE: u64 = 16;
    const GROUPS: u64 = 2;
    const TAPS: u64 = 4;
    const CHANNELS: u64 = INNER + 2 * GROUPS * STATE;

    fn floats(name: &str, dims: &[u64], value: f32) -> Tensor {
        let n: u64 = dims.iter().product();
        Tensor {
            name: name.to_owned(),
            ggml_type: 0,
            dims: dims.to_vec(),
            data: (0..n).flat_map(|_| value.to_le_bytes()).collect(),
        }
    }

    fn matrix(name: &str, in_dim: u64, out_dim: u64) -> Tensor {
        floats(name, &[in_dim, out_dim], 0.0)
    }

    fn bank(name: &str, in_dim: u64, out_dim: u64) -> Tensor {
        let bytes = iq4nl_expert_bytes(
            u32::try_from(out_dim).expect("fits"),
            u32::try_from(in_dim).expect("fits"),
        ) * EXPERTS;
        Tensor {
            name: name.to_owned(),
            ggml_type: 20,
            dims: vec![in_dim, out_dim, EXPERTS],
            data: vec![0; usize::try_from(bytes).expect("fits")],
        }
    }

    fn routed(block: u32, router_rows: u64) -> Vec<Tensor> {
        let at = |role: &str| format!("blk.{block}.{role}");
        vec![
            floats(&at("attn_norm.weight"), &[HIDDEN], 1.0),
            matrix(&at("ffn_up_shexp.weight"), HIDDEN, SHARED_WIDTH),
            matrix(&at("ffn_down_shexp.weight"), SHARED_WIDTH, HIDDEN),
            floats(&at("ffn_gate_inp.weight"), &[HIDDEN, router_rows], 0.0),
            floats(&at("exp_probs_b.bias"), &[EXPERTS], 0.0),
            bank(&at("ffn_up_exps.weight"), HIDDEN, EXPERT_WIDTH),
            bank(&at("ffn_down_exps.weight"), EXPERT_WIDTH, HIDDEN),
        ]
    }

    fn attention(block: u32) -> Vec<Tensor> {
        let at = |role: &str| format!("blk.{block}.{role}");
        vec![
            floats(&at("attn_norm.weight"), &[HIDDEN], 1.0),
            matrix(&at("attn_q.weight"), HIDDEN, Q),
            matrix(&at("attn_k.weight"), HIDDEN, KV),
            matrix(&at("attn_v.weight"), HIDDEN, KV),
            matrix(&at("attn_output.weight"), Q, HIDDEN),
        ]
    }

    /// What to leave out or get wrong.
    #[derive(Default)]
    struct Faults {
        /// Tensor names not written.
        skip: &'static [&'static str],
        /// Router rows of block 3, `EXPERTS` when sound.
        router_rows: Option<u64>,
        /// Elements of block 0's group norm, `INNER` when sound.
        ssm_norm: Option<u64>,
        /// Whether the prediction block is in the export at all.
        no_head: bool,
    }

    fn write(path: &Path, faults: &Faults) {
        let head = !faults.no_head;
        let mut kv_heads = vec![0, 0, 1, 0];
        let mut ffn = vec![0, 32, 0, 32];
        if head {
            kv_heads.push(1);
            ffn.push(32);
        }
        let blocks = u32::try_from(kv_heads.len()).expect("fits");
        let kvs = [
            ("general.architecture", Kv::Str("nemotron_h_moe")),
            ("nemotron_h_moe.block_count", Kv::U32(blocks)),
            ("nemotron_h_moe.embedding_length", Kv::U32(64)),
            ("nemotron_h_moe.feed_forward_length", Kv::I32s(ffn)),
            ("nemotron_h_moe.attention.head_count", Kv::U32(2)),
            ("nemotron_h_moe.attention.head_count_kv", Kv::I32s(kv_heads)),
            ("nemotron_h_moe.attention.key_length", Kv::U32(32)),
            (
                "nemotron_h_moe.attention.layer_norm_rms_epsilon",
                Kv::F32(1e-5),
            ),
            ("nemotron_h_moe.expert_count", Kv::U32(4)),
            ("nemotron_h_moe.expert_used_count", Kv::U32(2)),
            ("nemotron_h_moe.expert_group_count", Kv::U32(1)),
            ("nemotron_h_moe.expert_group_used_count", Kv::U32(1)),
            ("nemotron_h_moe.vocab_size", Kv::U32(32)),
            ("nemotron_h_moe.ssm.conv_kernel", Kv::U32(4)),
            ("nemotron_h_moe.ssm.state_size", Kv::U32(16)),
            ("nemotron_h_moe.ssm.group_count", Kv::U32(2)),
            ("nemotron_h_moe.ssm.inner_size", Kv::U32(128)),
            ("nemotron_h_moe.ssm.time_step_rank", Kv::U32(4)),
            ("nemotron_h_moe.expert_feed_forward_length", Kv::U32(32)),
            (
                "nemotron_h_moe.expert_shared_feed_forward_length",
                Kv::U32(64),
            ),
            ("nemotron_h_moe.expert_shared_count", Kv::U32(1)),
            ("nemotron_h_moe.expert_weights_norm", Kv::Bool(true)),
            ("nemotron_h_moe.expert_weights_scale", Kv::F32(2.5)),
            (
                "nemotron_h_moe.nextn_predict_layers",
                Kv::U32(u32::from(head)),
            ),
            ("tokenizer.ggml.eos_token_id", Kv::U32(1)),
        ];
        let mut tensors = vec![
            matrix("token_embd.weight", HIDDEN, VOCAB),
            floats("output_norm.weight", &[HIDDEN], 1.0),
            matrix("output.weight", HIDDEN, VOCAB),
            // Block 0: the state-space mixer, its vectors in the container's
            // own shapes (unit axes and the grouped norm included).
            floats("blk.0.attn_norm.weight", &[HIDDEN], 1.0),
            matrix("blk.0.ssm_in.weight", HIDDEN, INNER + CHANNELS + SSM_HEADS),
            floats("blk.0.ssm_conv1d.weight", &[TAPS, CHANNELS], 0.0),
            floats("blk.0.ssm_conv1d.bias", &[CHANNELS], 0.0),
            floats("blk.0.ssm_dt.bias", &[SSM_HEADS], 0.0),
            floats("blk.0.ssm_a", &[1, SSM_HEADS], -1.0),
            floats("blk.0.ssm_d", &[1, SSM_HEADS], 1.0),
            floats(
                "blk.0.ssm_norm.weight",
                &[faults.ssm_norm.unwrap_or(INNER) / GROUPS, GROUPS],
                1.0,
            ),
            matrix("blk.0.ssm_out.weight", INNER, HIDDEN),
        ];
        tensors.extend(routed(1, EXPERTS));
        tensors.extend(attention(2));
        tensors.extend(routed(3, faults.router_rows.unwrap_or(EXPERTS)));
        if head {
            // The prediction block carries both the attention's input norm
            // and the feed-forward's, beside its own projection and norms.
            tensors.extend(attention(4));
            tensors.extend(
                routed(4, EXPERTS)
                    .into_iter()
                    .filter(|t| t.name != "blk.4.attn_norm.weight"),
            );
            tensors.push(floats("blk.4.post_attention_norm.weight", &[HIDDEN], 1.0));
            tensors.push(matrix("blk.4.nextn.eh_proj.weight", 2 * HIDDEN, HIDDEN));
            for role in ["enorm", "hnorm", "shared_head_norm"] {
                tensors.push(floats(
                    &format!("blk.4.nextn.{role}.weight"),
                    &[HIDDEN],
                    1.0,
                ));
            }
        }
        tensors.retain(|t| !faults.skip.contains(&t.name.as_str()));
        write_gguf(path, &kvs, &tensors);
    }

    fn cpu() -> Arc<dyn Backend> {
        CpuBackend::new(CpuOptions::default()).expect("cpu device")
    }

    struct Export {
        _dir: tempfile::TempDir,
        path: std::path::PathBuf,
        file: GgufFile,
        config: NemotronHConfig,
    }

    fn export(faults: &Faults) -> Export {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("decoder.gguf");
        write(&path, faults);
        let file = GgufFile::open(&path).expect("opens");
        let config = NemotronHConfig::from_gguf(&file).expect("reads");
        Export {
            _dir: dir,
            path,
            file,
            config,
        }
    }

    #[test]
    fn every_tensor_binds_at_its_count() {
        let e = export(&Faults::default());
        let backend = cpu();
        let weights =
            weights_gguf(&backend, &e.file, &e.config, &ExpertPlacement::Resident).expect("binds");
        assert_eq!(weights.layers.len(), 2);
        assert!(matches!(weights.layers[0].block, Block::Ssm(_)));
        assert!(matches!(weights.layers[1].block, Block::Attn(_)));
        assert!(weights.layers.iter().all(|l| l.moe.is_some()));
        let block = e.config.mtp_block().expect("head");
        let parts = draft_parts(
            &backend,
            &e.file,
            &e.config,
            &ExpertPlacement::Resident,
            block,
            2,
        )
        .expect("head binds");
        assert!(parts.moe.is_some());
        assert!(parts.hidden_prenorm.is_some());
    }

    #[test]
    fn a_vector_of_the_wrong_count_is_refused() {
        let e = export(&Faults {
            ssm_norm: Some(INNER - GROUPS),
            ..Faults::default()
        });
        let err = weights_gguf(&cpu(), &e.file, &e.config, &ExpertPlacement::Resident)
            .expect_err("short norm");
        assert!(
            matches!(
                err,
                CoreError::ShapeMismatch {
                    expected: 128,
                    got: 126,
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_router_wider_than_its_bias_is_refused() {
        let e = export(&Faults {
            router_rows: Some(EXPERTS + 1),
            ..Faults::default()
        });
        let err = weights_gguf(&cpu(), &e.file, &e.config, &ExpertPlacement::Resident)
            .expect_err("router rows");
        assert!(
            matches!(
                err,
                CoreError::ShapeMismatch {
                    expected: 256,
                    got: 320,
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_head_missing_a_norm_is_refused() {
        let e = export(&Faults {
            skip: &["blk.4.post_attention_norm.weight"],
            ..Faults::default()
        });
        let err = draft_parts(&cpu(), &e.file, &e.config, &ExpertPlacement::Resident, 4, 2)
            .expect_err("no ffn norm");
        assert!(
            err.to_string().contains("blk.4.post_attention_norm.weight"),
            "{err}"
        );
    }

    #[test]
    fn speculating_without_a_head_is_refused() {
        let e = export(&Faults {
            no_head: true,
            ..Faults::default()
        });
        assert_eq!(e.config.mtp_block(), None);
        let mut opts = EngineOptions::new();
        opts.spec = SpecDepth::new(2);
        let err = NemotronHModel::load_gguf(
            &cpu(),
            &e.path,
            &e.config,
            opts,
            GgufResidency::Resident,
            false,
        )
        .expect_err("no head");
        assert!(err.to_string().contains("--spec-k"), "{err}");
    }
}
