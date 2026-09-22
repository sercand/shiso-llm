//! What the Gemma 4 role binding accepts, and the two shapes of file it must
//! refuse.
//!
//! The first three cases run against a synthesised directory rather than the
//! real container, for one reason: a refusal case has to be produced by taking
//! a tensor away or putting one back, and neither can be done to a five-gigabyte
//! file on disk. The directory is built from the same [`Gemma4Config`] the
//! binder is handed, so the names and widths it states are the ones the binder
//! derives - what the cases test is the binder's reaction to an edit, not
//! whether two hand-written tables agree.
//!
//! The fourth case is the one that checks the table itself, and it needs the
//! real file: a synthesised directory can only ever confirm that the binder
//! agrees with the test's idea of the names.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use shiso_core::config::HfConfig;
use shiso_core::{DType, LayerIdx};
use shiso_engine::shape::KvSource;
use shiso_modelfile::source::TensorSource;
use shiso_modelfile::{GgufError, GgufFile, TensorDesc, TensorShape, TensorView};
use shiso_models::gemma4::{Gemma4Config, bind_gemma4_desc};

/// Slots the E4B stack holds.
const N_LAYERS: u64 = 42;
/// Residual-stream width.
const HIDDEN: u64 = 2560;
/// Feed-forward inner width.
const INTERMEDIATE: u64 = 10240;
/// Vocabulary, which is also the per-layer table's row count.
const VOCAB: u64 = 262_144;
/// Width of one slot's per-layer embedding row.
const PLE_DIM: u64 = 256;
/// Rows of the per-layer table, which is every slot's row concatenated.
const PLE_ROWS: u64 = N_LAYERS * PLE_DIM;

/// The slot count as a list length, which is what the bound tree is measured in.
fn slots() -> usize {
    usize::try_from(N_LAYERS).expect("forty-two fits a pointer")
}

/// The checked-in fixture directory, reached from this crate's manifest.
fn vectors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors")
}

/// The E4B geometry, read from the shipped document.
///
/// The document carries `text_config`, so the path handed in is never opened.
fn config() -> Gemma4Config {
    let path = vectors().join("loader/gemma4-config.json");
    let document =
        HfConfig::from_file(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    Gemma4Config::from_hf(&document, Path::new("/nonexistent")).expect("the e4b geometry")
}

/// The local container, overridable, or `None` when it is not on this machine.
fn checkpoint() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_GEMMA4").ok()?);
    path.exists().then_some(path)
}

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
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
    /// What the container declares itself to be.
    arch: String,
    /// Architecture-prefixed counts, keyed by the suffix `arch_u32` is given.
    meta: HashMap<String, u32>,
    /// The directory itself.
    tensors: HashMap<String, TensorDesc>,
}

impl FakeSource {
    /// The directory a correct E4B container carries, for the geometry `cfg`
    /// states.
    fn e4b(cfg: &Gemma4Config) -> Self {
        let shape = cfg.shape();
        let mut tensors = HashMap::new();
        let mut put = |entry: TensorDesc| {
            tensors.insert(entry.name.to_string(), entry);
        };
        put(desc("token_embd.weight", DType::Q6_K, &[HIDDEN, VOCAB]));
        put(desc(
            "per_layer_token_embd.weight",
            DType::Q6_K,
            &[PLE_ROWS, VOCAB],
        ));
        put(desc(
            "per_layer_model_proj.weight",
            DType::F16,
            &[HIDDEN, PLE_ROWS],
        ));
        put(desc("per_layer_proj_norm.weight", DType::F32, &[PLE_DIM]));
        put(desc("output_norm.weight", DType::F32, &[HIDDEN]));

        for il in 0..shape.n_layers() {
            let slot = shape
                .attn_layer(LayerIdx::new(il))
                .expect("every e4b slot holds attention");
            let head = u64::from(slot.head_dim);
            let q_out = u64::from(slot.q_heads) * head;
            let kv_out = u64::from(slot.kv_heads) * head;
            let at = |suffix: &str| format!("blk.{il}.{suffix}");
            for norm in [
                "attn_norm.weight",
                "post_attention_norm.weight",
                "ffn_norm.weight",
                "post_ffw_norm.weight",
                "post_norm.weight",
            ] {
                put(desc(&at(norm), DType::F32, &[HIDDEN]));
            }
            put(desc(&at("layer_output_scale.weight"), DType::F32, &[1]));
            put(desc(&at("attn_q.weight"), DType::Q4_0, &[HIDDEN, q_out]));
            put(desc(
                &at("attn_output.weight"),
                DType::Q4_0,
                &[q_out, HIDDEN],
            ));
            put(desc(&at("attn_q_norm.weight"), DType::F32, &[head]));
            if slot.kv == KvSource::Own {
                put(desc(&at("attn_k.weight"), DType::Q4_0, &[HIDDEN, kv_out]));
                put(desc(&at("attn_v.weight"), DType::Q4_0, &[HIDDEN, kv_out]));
                put(desc(&at("attn_k_norm.weight"), DType::F32, &[head]));
            }
            put(desc(
                &at("ffn_gate.weight"),
                DType::Q4_0,
                &[HIDDEN, INTERMEDIATE],
            ));
            put(desc(
                &at("ffn_up.weight"),
                DType::Q4_0,
                &[HIDDEN, INTERMEDIATE],
            ));
            put(desc(
                &at("ffn_down.weight"),
                DType::Q4_0,
                &[INTERMEDIATE, HIDDEN],
            ));
            put(desc(
                &at("inp_gate.weight"),
                DType::Q4_0,
                &[HIDDEN, PLE_DIM],
            ));
            put(desc(&at("proj.weight"), DType::Q4_0, &[PLE_DIM, HIDDEN]));
        }

        let meta = [
            ("block_count", 42),
            ("attention.shared_kv_layers", 18),
            ("embedding_length_per_layer_input", 256),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect();

        Self {
            arch: "gemma4".to_owned(),
            meta,
            tensors,
        }
    }
}

impl TensorSource for FakeSource {
    fn architecture(&self) -> Option<&str> {
        Some(&self.arch)
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
            // The directory states a zero-length payload, so the empty slice is
            // the whole of it rather than a stub.
            data: &[],
            shape: entry.shape,
        })
    }
}

#[test]
fn the_e4b_directory_binds_to_its_roles() {
    let cfg = config();
    let src = FakeSource::e4b(&cfg);
    let binding = bind_gemma4_desc(&src, &cfg).expect("a correct e4b directory binds");

    assert_eq!(binding.layers.len(), slots());
    // Twenty-four slots own key/value runs and eighteen read an earlier slot's,
    // and a sharer holds no key, value or key norm of its own.
    assert!(binding.layers[23].attn.k.is_some());
    assert!(binding.layers[23].attn.v.is_some());
    assert!(binding.layers[23].attn.k_norm.is_some());
    assert!(binding.layers[24].attn.k.is_none());
    assert!(binding.layers[24].attn.v.is_none());
    assert!(binding.layers[24].attn.k_norm.is_none());
    let owning = binding
        .layers
        .iter()
        .filter(|layer| layer.attn.k.is_some())
        .count();
    assert_eq!(owning, 24);

    assert_eq!(binding.ple_table.shape.dims(), [PLE_ROWS, VOCAB]);
    assert_eq!(binding.token_embd.shape.dims(), [HIDDEN, VOCAB]);
    assert_eq!(binding.ple_model_proj.shape.dims(), [HIDDEN, PLE_ROWS]);
    assert_eq!(binding.ple_proj_norm.shape.dims(), [PLE_DIM]);
    assert_eq!(binding.output_norm.shape.dims(), [HIDDEN]);

    // A sliding slot and a full one, which differ in head width and in nothing
    // else the binding reads.
    assert_eq!(binding.layers[4].attn.q.shape.dims(), [HIDDEN, 8 * 256]);
    assert_eq!(binding.layers[4].attn.q_norm.shape.dims(), [256]);
    assert_eq!(binding.layers[5].attn.q.shape.dims(), [HIDDEN, 8 * 512]);
    assert_eq!(binding.layers[5].attn.q_norm.shape.dims(), [512]);
    assert_eq!(
        binding.layers[5].attn.output.shape.dims(),
        [8 * 512, HIDDEN]
    );

    let layer = &binding.layers[0];
    assert_eq!(layer.input_norm.name.as_ref(), "blk.0.attn_norm.weight");
    assert_eq!(
        layer.post_attn_norm.name.as_ref(),
        "blk.0.post_attention_norm.weight"
    );
    assert_eq!(layer.pre_ffn_norm.name.as_ref(), "blk.0.ffn_norm.weight");
    assert_eq!(
        layer.post_ffn_norm.name.as_ref(),
        "blk.0.post_ffw_norm.weight"
    );
    assert_eq!(
        layer.layer_scale.name.as_ref(),
        "blk.0.layer_output_scale.weight"
    );
    assert_eq!(layer.layer_scale.shape.dims(), [1]);
    assert_eq!(layer.ple.gate.name.as_ref(), "blk.0.inp_gate.weight");
    assert_eq!(layer.ple.proj.name.as_ref(), "blk.0.proj.weight");
    assert_eq!(layer.ple.post_norm.name.as_ref(), "blk.0.post_norm.weight");
    assert_eq!(layer.ffn.down.shape.dims(), [INTERMEDIATE, HIDDEN]);
}

#[test]
fn an_absent_per_layer_gate_is_named() {
    let cfg = config();
    let mut src = FakeSource::e4b(&cfg);
    src.tensors
        .remove("blk.7.inp_gate.weight")
        .expect("the directory had one");

    match bind_gemma4_desc(&src, &cfg) {
        Err(GgufError::MissingTensor { name }) => {
            assert_eq!(name.as_ref(), "blk.7.inp_gate.weight");
        }
        other => panic!("expected the missing gate to be named, got {other:?}"),
    }
}

#[test]
fn a_shared_slot_that_carries_a_key_is_refused() {
    let cfg = config();
    let mut src = FakeSource::e4b(&cfg);
    // Slot 30 reads slot 23's runs. A key projection of its own means the file
    // and the geometry disagree about which slots write the cache, and a stack
    // built from the geometry would leave the tensor unread rather than fail.
    src.tensors.insert(
        "blk.30.attn_k.weight".to_owned(),
        desc("blk.30.attn_k.weight", DType::Q4_0, &[HIDDEN, 2 * 512]),
    );

    let err = bind_gemma4_desc(&src, &cfg).expect_err("the stray key is refused");
    match &err {
        GgufError::LayoutMismatch { tensor, want, got } => {
            assert_eq!(tensor.as_ref(), "blk.30.attn_k.weight");
            // The variant renders as "tensor {tensor} has {got}, expected
            // {want}", so `got` has to describe the file and `want` the rule.
            // Swapped, it reads backwards, which no assertion on the name alone
            // would have caught - hence both halves, and the sentence itself.
            assert_eq!(got.as_ref(), "present");
            assert_eq!(
                want.as_ref(),
                "absent: a shared slot reads an earlier slot's key/value runs"
            );
        }
        other => panic!("expected the stray key to be refused, got {other:?}"),
    }
    assert_eq!(
        err.to_string(),
        "tensor blk.30.attn_k.weight has present, expected absent: a shared slot reads an \
         earlier slot's key/value runs"
    );
}

#[test]
fn a_slot_count_the_geometry_disagrees_with_is_refused() {
    let cfg = config();
    let mut src = FakeSource::e4b(&cfg);
    src.meta.insert("block_count".to_owned(), 41);

    match bind_gemma4_desc(&src, &cfg) {
        Err(GgufError::ProfileMismatch { key, want, got }) => {
            assert_eq!(key.as_ref(), "gemma4.block_count");
            assert_eq!((want, got), (N_LAYERS, 41));
        }
        other => panic!("expected the slot count to be refused, got {other:?}"),
    }
}

#[test]
fn an_absent_slot_count_is_refused() {
    let cfg = config();
    let mut src = FakeSource::e4b(&cfg);
    src.meta
        .remove("block_count")
        .expect("the container stated one");

    // Absent rather than wrong, which is a different branch: nothing downstream
    // reads this key, so a container that omits it would otherwise bind with one
    // of the three cross-checks silently skipped.
    match bind_gemma4_desc(&src, &cfg) {
        Err(GgufError::LayoutMismatch { tensor, want, got }) => {
            assert_eq!(tensor.as_ref(), "gemma4.block_count");
            assert_eq!(want.as_ref(), "42");
            assert_eq!(got.as_ref(), "absent");
        }
        other => panic!("expected the absent slot count to be refused, got {other:?}"),
    }
}

#[test]
#[ignore = "needs the local GGUF"]
fn the_real_container_binds() {
    let Some(path) = checkpoint() else {
        skip("the gemma 4 E4B container is not on this machine");
        return;
    };
    let file = GgufFile::open_shared(&path).expect("the container opens");
    let cfg = Gemma4Config::from_gguf(&file).expect("the container states the e4b shape");
    let binding = bind_gemma4_desc(&file, &cfg).expect("the real container binds");

    assert_eq!(binding.layers.len(), slots());
    assert_eq!(binding.token_embd.dtype, DType::Q6_K);
    assert_eq!(binding.layers[0].attn.q.dtype, DType::Q4_0);
    assert_eq!(binding.ple_model_proj.dtype, DType::F16);
    assert_eq!(binding.ple_table.dtype, DType::Q6_K);
    assert_eq!(binding.ple_table.shape.dims(), [PLE_ROWS, VOCAB]);
    assert!(binding.layers[24].attn.k.is_none());
    assert!(binding.layers[23].attn.k.is_some());
}
