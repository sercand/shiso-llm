//! What the qwen35 binder accepts from a mixed i-quant checkpoint.
//!
//! The Saluki export does not store a role pair in one format: its feed-forward
//! gate and up projection are siblings of the same width in different codecs
//! (`iq1_s` beside `iq1_m`), and the binding is the place that decision is made
//! or refused. The cases run against a synthesised tensor directory rather than
//! a file, for the same reason the gemma4 binder's refusal cases do: the shape
//! of the question is "what does the binder do when one role's format differs
//! from its neighbour's", and that is an edit to a directory, not to a
//! gigabyte-scale checkpoint on disk.

use std::collections::HashMap;

use shiso_core::{DType, TokenId};
use shiso_engine::shape::{
    Activation, AttnLayer, GdnShape, HybridShape, LayerKind, MediaPositions, NormScheme,
};
use shiso_modelfile::source::TensorSource;
use shiso_modelfile::{GgufError, TensorDesc, TensorShape, TensorView};
use shiso_models::qwen35::{Qwen35Config, bind_qwen35_desc};

/// Layer slots: one recurrent then one attention, the pattern the binder
/// cross-checks the interval against.
const N_LAYERS: u32 = 2;
/// Attention block stride.
const INTERVAL: u32 = 2;
/// Residual-stream width.
const HIDDEN: u64 = 64;
/// Vocabulary, which is also the embedding table's height.
const VOCAB: u64 = 128;
/// Feed-forward inner width.
const INTERMEDIATE: u64 = 128;
/// Recurrent value heads.
const V_HEADS: u32 = 2;
/// Recurrent query/key heads.
const QK_HEADS: u32 = 1;
/// Recurrent head width.
const GDN_HEAD: u32 = 32;
/// Convolution window.
const CONV_KERNEL: u32 = 4;
/// Attention query heads.
const Q_HEADS: u32 = 2;
/// Attention key/value heads.
const KV_HEADS: u32 = 1;
/// Attention head width.
const ATTN_HEAD: u32 = 32;

/// The nine mixed i-quant formats of the Saluki export.
const MIXED_FORMATS: [DType; 9] = [
    DType::Q2_K,
    DType::IQ2_XXS,
    DType::IQ2_XS,
    DType::IQ2_S,
    DType::IQ3_XXS,
    DType::IQ3_S,
    DType::IQ1_S,
    DType::IQ1_M,
    DType::IQ4_XS,
];

/// A two-slot qwen35 profile small enough to bind from memory.
///
/// The widths are the fixture's, not the shipped model's: the binder derives
/// every expected dimension from the shape it is handed, so what these cases
/// test is the binder's reaction to the formats, and the widths only have to
/// be stated once on both sides.
fn profile() -> Qwen35Config {
    let shape = HybridShape {
        layers: vec![LayerKind::Gdn, LayerKind::Attn],
        attn: vec![AttnLayer::new(Q_HEADS, KV_HEADS, ATTN_HEAD, 16, 16, 1.0e7)],
        hidden: u32::try_from(HIDDEN).expect("the fixture width fits"),
        vocab: u32::try_from(VOCAB).expect("the fixture vocabulary fits"),
        intermediate: u32::try_from(INTERMEDIATE).expect("the fixture width fits"),
        gdn: GdnShape::new(V_HEADS, QK_HEADS, GDN_HEAD, CONV_KERNEL),
        attn_output_gate: false,
        rms_eps: 1e-6,
        activation: Activation::SwiGlu,
        norms: NormScheme::PreOnly,
        embed_scale: None,
        final_softcap: None,
        layer_scalar: false,
        ple: None,
        media_positions: MediaPositions::Mrope3D,
        v_norm: false,
        moe: None,
        ssm: None,
        no_ffn: Vec::new(),
    };
    Qwen35Config::new(shape, TokenId::new(0), INTERVAL)
}

/// One directory entry, with the offset and length a binding never reads.
fn desc(name: &str, dtype: DType, dims: &[u64]) -> TensorDesc {
    TensorDesc {
        name: Box::from(name),
        dtype,
        shape: TensorShape::new(dims).unwrap_or_else(|err| panic!("{name} has a shape: {err}")),
        abs_offset: 0,
        bytes: 0,
    }
}

/// A tensor directory with no file under it.
///
/// It answers the four questions [`TensorSource`] asks, which is the whole of
/// what a binding is allowed to know about a container.
struct FakeSource {
    /// Architecture-prefixed counts, keyed by the suffix `arch_u32` is given.
    meta: HashMap<String, u32>,
    /// The directory itself.
    tensors: HashMap<String, TensorDesc>,
}

impl FakeSource {
    /// The directory a correct two-slot checkpoint carries, with the layer-zero
    /// gate in `gate_format` and the layer-zero up projection in `up_format`.
    fn mixed(gate_format: DType, up_format: DType) -> Self {
        let gdn_qkv = u64::from(2 * QK_HEADS * GDN_HEAD + V_HEADS * GDN_HEAD);
        let gdn_v = u64::from(V_HEADS * GDN_HEAD);
        let attn_q = u64::from(Q_HEADS * ATTN_HEAD);
        let attn_kv = u64::from(KV_HEADS * ATTN_HEAD);
        let mut tensors = HashMap::new();
        let mut put = |entry: TensorDesc| {
            tensors.insert(entry.name.to_string(), entry);
        };
        put(desc("token_embd.weight", DType::Q8_0, &[HIDDEN, VOCAB]));
        put(desc("output_norm.weight", DType::F32, &[HIDDEN]));
        for slot in 0..N_LAYERS {
            let at = |suffix: &str| format!("blk.{slot}.{suffix}");
            let gate = if slot == 0 { gate_format } else { DType::Q8_0 };
            let up = if slot == 0 { up_format } else { DType::Q8_0 };
            put(desc(&at("attn_norm.weight"), DType::F32, &[HIDDEN]));
            put(desc(
                &at("post_attention_norm.weight"),
                DType::F32,
                &[HIDDEN],
            ));
            put(desc(&at("ffn_gate.weight"), gate, &[HIDDEN, INTERMEDIATE]));
            put(desc(&at("ffn_up.weight"), up, &[HIDDEN, INTERMEDIATE]));
            put(desc(
                &at("ffn_down.weight"),
                DType::Q8_0,
                &[INTERMEDIATE, HIDDEN],
            ));
            if slot + 1 == N_LAYERS {
                put(desc(&at("attn_q.weight"), DType::Q8_0, &[HIDDEN, attn_q]));
                put(desc(&at("attn_k.weight"), DType::Q8_0, &[HIDDEN, attn_kv]));
                put(desc(&at("attn_v.weight"), DType::Q8_0, &[HIDDEN, attn_kv]));
                put(desc(
                    &at("attn_q_norm.weight"),
                    DType::F32,
                    &[u64::from(ATTN_HEAD)],
                ));
                put(desc(
                    &at("attn_k_norm.weight"),
                    DType::F32,
                    &[u64::from(ATTN_HEAD)],
                ));
                put(desc(
                    &at("attn_output.weight"),
                    DType::Q8_0,
                    &[attn_q, HIDDEN],
                ));
            } else {
                put(desc(
                    &at("attn_qkv.weight"),
                    DType::Q8_0,
                    &[HIDDEN, gdn_qkv],
                ));
                put(desc(&at("attn_gate.weight"), DType::Q8_0, &[HIDDEN, gdn_v]));
                put(desc(
                    &at("ssm_conv1d.weight"),
                    DType::F32,
                    &[u64::from(CONV_KERNEL), gdn_qkv],
                ));
                put(desc(&at("ssm_a"), DType::F32, &[u64::from(V_HEADS)]));
                put(desc(
                    &at("ssm_alpha.weight"),
                    DType::Q8_0,
                    &[HIDDEN, u64::from(V_HEADS)],
                ));
                put(desc(
                    &at("ssm_beta.weight"),
                    DType::Q8_0,
                    &[HIDDEN, u64::from(V_HEADS)],
                ));
                put(desc(&at("ssm_dt.bias"), DType::F32, &[u64::from(V_HEADS)]));
                put(desc(
                    &at("ssm_norm.weight"),
                    DType::F32,
                    &[u64::from(GDN_HEAD)],
                ));
                put(desc(&at("ssm_out.weight"), DType::Q8_0, &[gdn_v, HIDDEN]));
            }
        }

        let meta = [
            ("block_count".to_owned(), N_LAYERS),
            ("full_attention_interval".to_owned(), INTERVAL),
        ]
        .into_iter()
        .collect();
        Self { meta, tensors }
    }
}

impl TensorSource for FakeSource {
    fn architecture(&self) -> Option<&str> {
        Some("qwen35")
    }

    fn arch_u32(&self, suffix: &str) -> Option<u32> {
        self.meta.get(suffix).copied()
    }

    fn find(&self, name: &str) -> Option<&TensorDesc> {
        self.tensors.get(name)
    }

    fn get(&self, name: &str) -> Option<TensorView<'_>> {
        let entry = self.tensors.get(name)?;
        Some(TensorView {
            name: &entry.name,
            dtype: entry.dtype,
            // The directory states a zero-length payload, so the empty slice
            // is the whole of it rather than a stub.
            data: &[],
            shape: entry.shape,
        })
    }
}

#[test]
fn a_gate_and_up_pair_of_different_formats_binds_with_its_own_formats() {
    // The review focus: the Saluki file's `blk.0.ffn_gate` is `iq1_s` and its
    // `blk.0.ffn_up` is `iq1_m`. The binder must keep each role's stored
    // format, because the loader dispatches each projection off its own
    // descriptor and a binding that unified the pair would launch one of them
    // against the other codec's kernel.
    let cfg = profile();
    let src = FakeSource::mixed(DType::IQ1_S, DType::IQ1_M);
    let binding = bind_qwen35_desc(&src, &cfg)
        .expect("a gate and up pair of different accepted formats binds");

    assert_eq!(binding.layers[0].ffn.gate.dtype, DType::IQ1_S);
    assert_eq!(binding.layers[0].ffn.up.dtype, DType::IQ1_M);
    assert_eq!(
        binding.layers[0].ffn.gate.name.as_ref(),
        "blk.0.ffn_gate.weight"
    );
    assert_eq!(
        binding.layers[0].ffn.up.name.as_ref(),
        "blk.0.ffn_up.weight"
    );
}

#[test]
fn every_mixed_i_quant_binds_as_a_weight_matrix() {
    let cfg = profile();
    for dtype in MIXED_FORMATS {
        let src = FakeSource::mixed(dtype, dtype);
        let binding = bind_qwen35_desc(&src, &cfg)
            .unwrap_or_else(|err| panic!("a gate stored in {} binds: {err}", dtype.type_name()));
        assert_eq!(binding.layers[0].ffn.gate.dtype, dtype);
    }
}

#[test]
fn a_format_outside_the_accept_set_is_refused() {
    // `iq4_nl` has no matvec in this kernel set, and the accept list is exactly
    // the formats that do: a binder that grew a tenth entry on a hope would
    // turn a launch-time refusal into a load that runs the wrong decode.
    let cfg = profile();
    let src = FakeSource::mixed(DType::IQ4_NL, DType::IQ4_NL);
    match bind_qwen35_desc(&src, &cfg) {
        Err(GgufError::LayoutMismatch { tensor, got, .. }) => {
            assert_eq!(tensor.as_ref(), "blk.0.ffn_gate.weight");
            assert_eq!(got.as_ref(), "iq4_nl");
        }
        other => panic!("expected the unlisted format to be refused, got {other:?}"),
    }
}
