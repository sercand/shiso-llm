//! Config parsing against the checked-in fixture.
//!
//! Every case starts from `shiso/test-vectors/loader/qwen35-config.json`, copied from
//! the local model directory, and edits it as text. Nothing here hand-writes a
//! config: a fixture typed out by hand would encode wrong constants into every
//! assertion that reads it.

use std::path::{Path, PathBuf};

use shiso_core::config::{HfConfig, HiddenAct};
use shiso_core::error::CoreError;
use shiso_core::ids::LayerIdx;
use shiso_engine::shape::{Activation, AttnScale, KvSource, LayerKind, MediaPositions, NormScheme};
use shiso_models::qwen35::Qwen35Config;

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/loader/qwen35-config.json")
        .normalize()
}

fn fixture_text() -> String {
    let path = fixture_path();
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

/// Lexically normalizes a path so a missing fixture says so at the right place.
trait Normalize {
    fn normalize(self) -> PathBuf;
}

impl Normalize for PathBuf {
    fn normalize(self) -> PathBuf {
        let mut out = PathBuf::new();
        for part in self.components() {
            match part {
                std::path::Component::ParentDir => {
                    out.pop();
                }
                other => out.push(other.as_os_str()),
            }
        }
        out
    }
}

/// Replaces one exact substring, failing loudly when the fixture moved.
fn replace(text: &str, old: &str, new: &str) -> String {
    assert!(
        text.contains(old),
        "fixture no longer contains {old:?}; the test is asserting nothing"
    );
    text.replace(old, new)
}

/// Removes a top-level entry from the document.
///
/// The fixture is machine-formatted with two-space indentation, so an entry can
/// be found by its indented key and its value cut by brace matching. Doing this
/// as text keeps the test free of a JSON library of its own.
fn remove_top_level(text: &str, key: &str) -> String {
    let needle = format!("\n  \"{key}\": ");
    let start = text
        .find(&needle)
        .unwrap_or_else(|| panic!("fixture has no top-level {key}"));
    let key_start = start + 1;
    let value_start = start + needle.len();
    let bytes = text.as_bytes();
    let mut depth = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut end = None;
    for (index, &byte) in bytes.iter().enumerate().skip(value_start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(index + 1);
                    break;
                }
            }
            b',' if depth == 0 => {
                end = Some(index);
                break;
            }
            _ => {}
        }
    }
    let mut stop = end.unwrap_or_else(|| panic!("unterminated value for top-level {key}"));
    if bytes.get(stop) == Some(&b',') {
        stop += 1;
    }
    let mut out = String::with_capacity(text.len());
    out.push_str(&text[..key_start]);
    out.push_str(&text[stop..]);
    out
}

#[test]
fn fixture_describes_the_multimodal_wrapper() {
    let config = HfConfig::from_file(&fixture_path()).expect("fixture parses");
    assert_eq!(config.model_type(), "qwen3_5");
    let architectures: Vec<&str> = config.architectures().iter().map(String::as_str).collect();
    assert_eq!(architectures, ["Qwen3_5ForConditionalGeneration"]);
}

#[expect(
    clippy::float_cmp,
    reason = "two JSON constants narrowed from f64 to f32, compared bit-exact on purpose"
)]
#[test]
fn fixture_parses_to_the_pinned_qwen35_shape() {
    let config = HfConfig::from_file(&fixture_path()).expect("fixture parses");
    let profile = Qwen35Config::from_hf(&config).expect("qwen35 shape");
    let shape = profile.shape();

    assert_eq!(shape.n_layers(), 32);
    assert_eq!(shape.hidden, 2560);
    assert_eq!(shape.vocab, 248_320);
    // One descriptor per attention slot, all eight of them identical.
    assert_eq!(shape.attn.len(), 8);
    let attn = shape.attn.first().expect("an attention slot");
    assert!(shape.attn.iter().all(|layer| layer == attn));
    assert_eq!(attn.q_heads, 16);
    assert_eq!(attn.kv_heads, 4);
    // The attention head is 256 wide with a quarter of it rotated; the 128
    // belongs to the recurrent heads, not these.
    assert_eq!(attn.head_dim, 256);
    assert_eq!(attn.n_rot, 64);
    // The frequency ladder is the rotated width itself, and nothing narrows the
    // slot: no window, its own cache, the usual inverse square root.
    assert_eq!(attn.freq_dims, 64);
    assert_eq!(attn.window, None);
    assert_eq!(attn.kv, KvSource::Own);
    assert_eq!(attn.scale, AttnScale::InvSqrtHeadDim);
    assert_eq!(shape.gdn.v_heads, 32);
    assert_eq!(shape.gdn.qk_heads, 16);
    assert_eq!(shape.gdn.head_dim, 128);
    assert_eq!(shape.gdn.conv_kernel, 4);
    assert_eq!(shape.intermediate, 9216);
    assert!(shape.attn_output_gate);
    assert_eq!(profile.full_attention_interval(), 4);
    assert_eq!(profile.eos_token().get(), 248_044);
    // Both are a fixed JSON constant narrowed from f64 to f32, so the bits are
    // exact and a tolerance would hide a narrowing that changed them.
    assert_eq!(shape.rms_eps, 1.0e-6);
    assert_eq!(attn.rope_theta, 1.0e7);

    // The family switches, which are what a Gemma-like checkpoint fills in
    // differently and this one does not.
    assert_eq!(shape.activation, Activation::SwiGlu);
    assert_eq!(shape.norms, NormScheme::PreOnly);
    assert_eq!(shape.embed_scale, None);
    assert_eq!(shape.final_softcap, None);
    assert!(!shape.layer_scalar);
    assert_eq!(shape.ple, None);
    assert_eq!(shape.media_positions, MediaPositions::Mrope3D);
    assert!(!shape.v_norm);
    shape.validate().expect("the parsed shape is valid");

    assert_eq!(profile.shape().n_layers(), 32);
    assert_eq!(profile.shape().hidden, 2560);
    assert_eq!(profile.shape().vocab, 248_320);
    assert_eq!(profile.eos_token().get(), 248_044);

    // Eight repeats of three recurrent slots and one attention slot, checked
    // per index against the interval rule rather than as one whole-list equal.
    assert_eq!(shape.layers.len(), 32);
    for (index, kind) in shape.layers.iter().enumerate() {
        let slot = u32::try_from(index).expect("32 slots fit a u32");
        let expected = if (slot + 1) % 4 == 0 {
            LayerKind::Attn
        } else {
            LayerKind::Gdn
        };
        assert_eq!(*kind, expected, "slot {slot}");
        assert_eq!(
            profile.shape().layer_kind(LayerIdx::new(slot)),
            Some(expected)
        );
    }
    assert_eq!(shape.layers[3], LayerKind::Attn);
    assert_eq!(shape.layers[31], LayerKind::Attn);
    assert_eq!(profile.shape().layer_kind(LayerIdx::new(32)), None);
}

#[test]
fn fixture_vision_tower_is_the_pinned_contract() {
    let config = HfConfig::from_file(&fixture_path()).expect("fixture parses");
    let tower = config
        .vision_config()
        .expect("vision config parses")
        .expect("fixture has a vision tower");

    assert_eq!(tower.depth, 24);
    assert_eq!(tower.hidden, 1024);
    assert_eq!(tower.num_heads, 16);
    assert_eq!(tower.intermediate_size, 4096);
    assert_eq!(tower.out_hidden, 2560);
    assert_eq!(tower.patch_size, 16);
    assert_eq!(tower.temporal_patch_size, 2);
    assert_eq!(tower.spatial_merge_size, 2);
    assert_eq!(tower.num_position_embeddings, 2304);
    assert_eq!(tower.deepstack_visual_indexes, Vec::<u32>::new());
    assert_eq!(tower.hidden_act, HiddenAct::GeluPytorchTanh);
    assert_eq!(tower.image_token_id.get(), 248_056);
    assert_eq!(tower.video_token_id.get(), 248_057);
    assert_eq!(tower.vision_start_token_id.get(), 248_053);
    assert_eq!(tower.vision_end_token_id.get(), 248_054);
    assert_eq!(tower.mrope_section, [11, 11, 10]);
    assert!(tower.mrope_interleaved);
}

#[test]
fn text_only_config_has_no_tower_and_still_parses() {
    let text = remove_top_level(&fixture_text(), "vision_config");
    let config = HfConfig::from_json_str(&text).expect("edited fixture parses");
    assert_eq!(config.vision_config().expect("no error"), None);
    let profile = Qwen35Config::from_hf(&config).expect("shape parse unaffected");
    assert_eq!(profile.shape().n_layers(), 32);
    assert_eq!(profile.shape().vocab, 248_320);
}

#[test]
fn a_missing_media_token_names_itself() {
    let text = remove_top_level(&fixture_text(), "image_token_id");
    let config = HfConfig::from_json_str(&text).expect("edited fixture parses");
    let err = config
        .vision_config()
        .expect_err("a tower with an unknown placeholder cannot load");
    match err {
        CoreError::ConfigMissing { key } => assert_eq!(key, "image_token_id"),
        other => panic!("expected ConfigMissing, got {other:?}"),
    }
}

#[test]
fn an_unknown_activation_is_a_load_error_not_a_string() {
    let text = replace(
        &fixture_text(),
        "\"hidden_act\": \"gelu_pytorch_tanh\"",
        "\"hidden_act\": \"mystery\"",
    );
    let config = HfConfig::from_json_str(&text).expect("edited fixture parses");
    let err = config
        .vision_config()
        .expect_err("an unimplemented activation cannot be carried to the kernels");
    match err {
        CoreError::InvalidValue { key, value } => {
            assert_eq!(key, "hidden_act");
            assert_eq!(value, "mystery");
        }
        other => panic!("expected InvalidValue, got {other:?}"),
    }
}

#[test]
fn a_tampered_layer_count_is_a_shape_mismatch() {
    let text = replace(
        &fixture_text(),
        "\"num_hidden_layers\": 32",
        "\"num_hidden_layers\": 31",
    );
    let config = HfConfig::from_json_str(&text).expect("edited fixture parses");
    let err = Qwen35Config::from_hf(&config).expect_err("a 31-layer checkpoint is not this shape");
    match err {
        CoreError::ShapeMismatch {
            field,
            expected,
            got,
        } => {
            assert_eq!(field, "num_hidden_layers");
            assert_eq!(expected, 32);
            assert_eq!(got, 31);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn a_layer_list_disagreeing_with_the_interval_is_rejected() {
    // The config states the layer pattern twice. Flipping one slot of the list
    // while leaving the interval at 4 would put an attention block where a
    // recurrent one belongs, with every individual dimension still correct.
    // The final group is the only one that ends the array, so this edit lands
    // on exactly one slot: slot 31 claims a recurrent block where the interval
    // rule says attention.
    let text = replace(
        &fixture_text(),
        "\"linear_attention\",\n      \"linear_attention\",\n      \"linear_attention\",\n      \"full_attention\"\n    ]",
        "\"linear_attention\",\n      \"linear_attention\",\n      \"linear_attention\",\n      \"linear_attention\"\n    ]",
    );
    let config = HfConfig::from_json_str(&text).expect("edited fixture parses");
    let err =
        Qwen35Config::from_hf(&config).expect_err("the two statements of the pattern must agree");
    match err {
        CoreError::ShapeMismatch { field, got, .. } => {
            assert_eq!(field, "layer_types.interval_mismatch_count");
            assert_eq!(got, 1);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn an_unknown_layer_kind_names_its_slot() {
    let text = replace(&fixture_text(), "\"linear_attention\"", "\"mamba\"");
    let config = HfConfig::from_json_str(&text).expect("edited fixture parses");
    let err =
        Qwen35Config::from_hf(&config).expect_err("an unrecognised layer kind cannot be guessed");
    match err {
        CoreError::ConfigMissing { key } => assert_eq!(key, "text_config.layer_types[0]"),
        other => panic!("expected ConfigMissing, got {other:?}"),
    }
}

#[test]
fn an_unknown_architecture_is_typed_not_a_serde_error() {
    let config =
        HfConfig::from_json_str(r#"{"model_type":"llama","architectures":["LlamaForCausalLM"]}"#)
            .expect("the head parses without the fields a supported config would have");
    let err = Qwen35Config::from_hf(&config).expect_err("this build cannot run a llama");
    match err {
        CoreError::UnknownArch { model_type } => assert_eq!(model_type, "llama"),
        other => panic!("expected UnknownArch, got {other:?}"),
    }
}

#[test]
fn a_config_without_a_model_type_is_a_json_error() {
    let err = HfConfig::from_json_str(r#"{"architectures":[]}"#)
        .expect_err("the architecture selector is required");
    assert!(matches!(err, CoreError::Json(_)), "got {err:?}");
}

// The sparse-family constant table used to be asserted here, because it lived in
// the shared crate beside this family's parser. It does not any more: a family's
// geometry belongs to the crate that can run it, so those two cases move with the
// table when the sparse family gets its crate. Leaving a case behind that
// referenced a type this crate does not own was the alternative, and it would have
// made this file the reason the table could not move.

/// The ids that end a turn, which are two and not one.
///
/// `text_config.eos_token_id` is 248044, `<|endoftext|>`: the id a base model
/// stops on. A chat turn ends on `<|im_end|>` (248046), which is what the
/// template writes after every assistant turn, what `tokenizer_config.json`
/// names as `eos_token`, and what the GGUF export declares as
/// `tokenizer.ggml.eos_token_id`. A server that stopped only on the document's
/// number would decode `<|im_end|>` as text and keep generating past the end of
/// the answer, which is exactly what it did.
#[test]
fn a_turn_ends_on_im_end_as_well_as_on_the_documents_eos() {
    let profile = Qwen35Config::from_file(&fixture_path()).expect("the fixture parses");
    assert_eq!(profile.eos_token().get(), 248_044);
    let stop: Vec<u32> = profile.stop_tokens().iter().map(|id| id.get()).collect();
    assert_eq!(stop, vec![248_046, 248_044]);
}

/// A document that already names the chat end-of-turn id does not get it twice.
#[test]
fn a_document_that_names_im_end_lists_it_once() {
    let text = fixture_text().replace("\"eos_token_id\": 248044", "\"eos_token_id\": 248046");
    let config = HfConfig::from_json_str(&text).expect("the edited fixture parses");
    let profile = Qwen35Config::from_hf(&config).expect("the edited fixture loads");
    let stop: Vec<u32> = profile.stop_tokens().iter().map(|id| id.get()).collect();
    assert_eq!(stop, vec![248_046]);
}
