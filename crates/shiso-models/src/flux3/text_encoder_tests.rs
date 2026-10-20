//! The padded text encoder: the prompt arithmetic against a toy vocabulary,
//! and the masked prefill against the unpadded one on a tiny synthetic
//! decoder (the device cases are ignored and say so when no device exists).
#![expect(
    clippy::cast_precision_loss,
    reason = "the synthetic weight noise is a 24-bit integer scaled into [-1, 1], exact in f32"
)]

use std::path::Path;
use std::sync::Arc;

use half::bf16;
use shiso_core::Backend;
use shiso_vision::generator::ImageGenError;

use super::text_encoder::{PREFIXES, TAP_LAYERS, detect_prefix, encode, load, tokenize};
use crate::diffusion::ops::Ops;
use crate::diffusion::prompt::{PromptTokenizer, load_chat_template};
use crate::diffusion::weights::StStore;

pub(super) const CHAT_TEMPLATE: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

/// A directory with a word-level tokenizer, its config and the chat
/// template. `pad` is the `pad_token` the config names, if any.
pub(super) fn toy_tokenizer(dir: &Path, pad: Option<&str>) {
    let added = |id: u32, content: &str| {
        serde_json::json!({
            "id": id, "content": content, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true,
        })
    };
    let tokenizer = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            added(4, "<|im_start|>"),
            added(5, "<|im_end|>"),
            added(6, "<|endoftext|>"),
        ],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": { "[UNK]": 0, "user": 1, "assistant": 2, "word": 3 },
            "unk_token": "[UNK]",
        },
    });
    std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).expect("tokenizer");
    let config = match pad {
        Some(pad) => serde_json::json!({ "pad_token": pad, "chat_template": CHAT_TEMPLATE }),
        None => serde_json::json!({ "chat_template": CHAT_TEMPLATE }),
    };
    std::fs::write(dir.join("tokenizer_config.json"), config.to_string()).expect("config");
}

/// The head and tail of the toy template are five ids around the prompt.
const TEMPLATE_IDS: usize = 5;

fn open(pad: Option<&str>) -> (tempfile::TempDir, PromptTokenizer, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    toy_tokenizer(dir.path(), pad);
    let tok = PromptTokenizer::open(dir.path()).expect("opens");
    let template = load_chat_template(Some(dir.path())).expect("template");
    (dir, tok, template)
}

fn words(n: usize) -> String {
    vec!["word"; n].join(" ")
}

#[test]
fn encode_pads_to_multiple_of_80_with_pad_token() {
    let (_dir, tok, template) = open(Some("<|endoftext|>"));
    let pad = tok.pad_id().expect("the config names a pad token");
    for (prompt_words, want_len) in [
        (1, 80),
        (80 - TEMPLATE_IDS, 80),
        (81 - TEMPLATE_IDS, 160),
        (160 - TEMPLATE_IDS, 160),
    ] {
        let padded = tokenize(&tok, &template, &words(prompt_words)).expect("tokenizes");
        assert_eq!(padded.real, prompt_words + TEMPLATE_IDS);
        assert_eq!(padded.ids.len(), want_len, "{prompt_words} words");
        assert!(
            padded.ids[padded.real..].iter().all(|&id| id == pad),
            "{prompt_words} words: every row past the real ones is the pad id"
        );
        assert!(
            padded.ids[..padded.real].iter().all(|&id| id != pad),
            "{prompt_words} words: no real row is the pad id"
        );
    }
    // The template ids frame the prompt: control ids around its words.
    let padded = tokenize(&tok, &template, "word").expect("tokenizes");
    assert_eq!(padded.ids.len(), 80);
    assert_eq!(padded.real, TEMPLATE_IDS + 1);
}

#[test]
fn encode_rejects_overlong_and_empty() {
    let (_dir, tok, template) = open(Some("<|endoftext|>"));
    assert!(matches!(
        tokenize(&tok, &template, ""),
        Err(ImageGenError::EmptyPrompt)
    ));
    assert!(matches!(
        tokenize(&tok, &template, "  \n "),
        Err(ImageGenError::EmptyPrompt)
    ));
    let longest = tokenize(&tok, &template, &words(4096 - TEMPLATE_IDS)).expect("4096 is allowed");
    assert_eq!(longest.ids.len(), 4096);
    assert_eq!(longest.real, 4096);
    assert!(matches!(
        tokenize(&tok, &template, &words(4097 - TEMPLATE_IDS)),
        Err(ImageGenError::PromptTooLong {
            tokens: 4097,
            max: 4096
        })
    ));
}

#[test]
fn a_tokenizer_config_without_a_pad_token_is_refused() {
    let (_dir, tok, template) = open(None);
    match tokenize(&tok, &template, "word") {
        Err(ImageGenError::Checkpoint(message)) => {
            assert!(message.contains("pad_token"), "{message}");
        }
        other => panic!("expected a pad refusal, got {other:?}"),
    }
    let (_dir, tok, template) = open(Some("<|nope|>"));
    assert!(
        tokenize(&tok, &template, "word").is_err(),
        "a pad token the vocabulary does not hold is no pad id"
    );
}

#[test]
fn the_taps_are_the_reference_hidden_states_4_to_32() {
    // hidden_states[k] counts the embedding output as 0, so it is the stream
    // after 0-indexed layer k - 1.
    let reference = [4usize, 8, 12, 16, 20, 24, 28, 32];
    let want: Vec<usize> = reference.iter().map(|k| k - 1).collect();
    assert_eq!(TAP_LAYERS.to_vec(), want);
    assert_eq!(
        TAP_LAYERS.len() * 2560,
        20_480,
        "eight taps of 2560 are the DiT's text width"
    );
}

// --- a tiny synthetic decoder ------------------------------------------------

const HIDDEN: usize = 16;
const HEAD_DIM: usize = 8;
const Q_HEADS: usize = 2;
const KV_HEADS: usize = 1;
const INTER: usize = 32;
const VOCAB: usize = 64;

/// A deterministic stream in `[-scale, scale]`.
struct Noise(u32);

impl Noise {
    fn next(&mut self, scale: f32) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
    }
}

/// Writes a BF16 safetensors file holding a `layers`-deep Qwen3-style
/// decoder under `prefix`, and the flat config beside it.
fn write_decoder(dir: &Path, prefix: &str, layers: usize) -> std::path::PathBuf {
    write_decoder_sized(dir, prefix, layers, HIDDEN)
}

/// [`write_decoder`] with a residual width of `hidden`: 2560 makes eight
/// taps the transformer's 20480.
#[expect(
    clippy::too_many_lines,
    reason = "one flat table of the decoder's tensors and their shapes"
)]
pub(super) fn write_decoder_sized(
    dir: &Path,
    prefix: &str,
    layers: usize,
    hidden: usize,
) -> std::path::PathBuf {
    let mut noise = Noise(7);
    // Keeps a wide residual stream from growing with the fan-in; 1.0 at the
    // 16-wide size the other cases use.
    #[expect(
        clippy::cast_precision_loss,
        reason = "a width of a few thousand is exact in f32"
    )]
    let width_scale = (16.0 / hidden as f32).sqrt();
    let mut header = serde_json::Map::new();
    let mut payload: Vec<u8> = Vec::new();
    let mut add = |name: String, shape: Vec<usize>, base: f32, scale: f32| {
        let count: usize = shape.iter().product();
        let start = payload.len();
        for _ in 0..count {
            payload.extend_from_slice(
                &bf16::from_f32(base + noise.next(scale * width_scale))
                    .to_bits()
                    .to_le_bytes(),
            );
        }
        header.insert(
            name,
            serde_json::json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [start, payload.len()],
            }),
        );
    };
    add(
        format!("{prefix}embed_tokens.weight"),
        vec![VOCAB, hidden],
        0.0,
        0.5,
    );
    for l in 0..layers {
        let b = format!("{prefix}layers.{l}.");
        add(format!("{b}input_layernorm.weight"), vec![hidden], 1.0, 0.1);
        add(
            format!("{b}post_attention_layernorm.weight"),
            vec![hidden],
            1.0,
            0.1,
        );
        add(
            format!("{b}self_attn.q_proj.weight"),
            vec![Q_HEADS * HEAD_DIM, hidden],
            0.0,
            0.3,
        );
        add(
            format!("{b}self_attn.k_proj.weight"),
            vec![KV_HEADS * HEAD_DIM, hidden],
            0.0,
            0.3,
        );
        add(
            format!("{b}self_attn.v_proj.weight"),
            vec![KV_HEADS * HEAD_DIM, hidden],
            0.0,
            0.3,
        );
        add(
            format!("{b}self_attn.o_proj.weight"),
            vec![hidden, Q_HEADS * HEAD_DIM],
            0.0,
            0.3,
        );
        add(
            format!("{b}self_attn.q_norm.weight"),
            vec![HEAD_DIM],
            1.0,
            0.1,
        );
        add(
            format!("{b}self_attn.k_norm.weight"),
            vec![HEAD_DIM],
            1.0,
            0.1,
        );
        add(
            format!("{b}mlp.gate_proj.weight"),
            vec![INTER, hidden],
            0.0,
            0.3,
        );
        add(
            format!("{b}mlp.up_proj.weight"),
            vec![INTER, hidden],
            0.0,
            0.3,
        );
        add(
            format!("{b}mlp.down_proj.weight"),
            vec![hidden, INTER],
            0.0,
            0.3,
        );
    }
    let text = serde_json::Value::Object(header).to_string();
    let mut file = Vec::new();
    file.extend_from_slice(&(text.len() as u64).to_le_bytes());
    file.extend_from_slice(text.as_bytes());
    file.extend_from_slice(&payload);
    let path = dir.join("model.safetensors");
    std::fs::write(&path, file).expect("writes the decoder");
    std::fs::write(
        dir.join("config.json"),
        r#"{"rope_theta": 10000.0, "rms_norm_eps": 1e-6, "num_hidden_layers": 32}"#,
    )
    .expect("config");
    path
}

#[test]
fn the_decoder_prefix_is_found_from_the_names() {
    for prefix in PREFIXES {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_decoder(dir.path(), prefix, 1);
        let store = StStore::open_files(std::slice::from_ref(&path)).expect("opens");
        assert_eq!(detect_prefix(&store, &[path]).expect("found"), prefix);
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_decoder(dir.path(), "decoder.", 1);
    let store = StStore::open_files(std::slice::from_ref(&path)).expect("opens");
    match detect_prefix(&store, &[path]) {
        Err(ImageGenError::Checkpoint(message)) => {
            assert!(message.contains("embed_tokens"), "{message}");
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[expect(
    clippy::print_stderr,
    reason = "the device cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn device() -> Option<Arc<dyn Backend>> {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the text encoder device case");
        return None;
    }
    let metal = shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");
    let backend: Arc<dyn Backend> = metal;
    Some(backend)
}

fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-3 * w.abs().max(0.05),
            "{what}[{i}]: {g} vs {w}"
        );
    }
}

#[test]
#[ignore = "requires a Metal device"]
fn tap_width_is_20480() {
    let Some(backend) = device() else { return };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_decoder(dir.path(), "model.language_model.", 32);
    let enc = load(
        std::slice::from_ref(&path),
        &dir.path().join("config.json"),
        &ops,
    )
    .expect("a 32-layer decoder loads through the last tap");
    // The real decoder is 2560 wide, so eight taps are the DiT's 20480.
    assert_eq!(enc.context_width(), 8 * HIDDEN);
    let rows = enc.prefill(&ops, &[1, 2, 3]).expect("prefill");
    assert_eq!(rows.len(), 3 * 8 * HIDDEN);

    // Eight taps need 32 layers: a decoder that stops short is refused.
    let short = tempfile::tempdir().expect("tempdir");
    let path = write_decoder(short.path(), "model.language_model.", 31);
    let refused = load(
        std::slice::from_ref(&path),
        &short.path().join("config.json"),
        &ops,
    );
    match refused {
        Err(ImageGenError::Checkpoint(message)) => {
            assert!(message.contains("31 layers"), "{message}");
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

#[test]
#[ignore = "requires a Metal device"]
fn prefill_padded_matches_unpadded_on_real_rows() {
    let Some(backend) = device() else { return };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_decoder(dir.path(), "model.language_model.", 32);
    let enc = load(
        std::slice::from_ref(&path),
        &dir.path().join("config.json"),
        &ops,
    )
    .expect("loads");
    let width = enc.context_width();
    let (real, len) = (7usize, 12usize);
    let mut ids: Vec<u32> = (0..real)
        .map(|i| u32::try_from(5 + 3 * i).expect("small"))
        .collect();
    ids.resize(len, 2);

    let unpadded = enc.prefill(&ops, &ids[..real]).expect("unpadded");
    let padded = enc.prefill_padded(&ops, &ids, real).expect("padded");
    assert_eq!(padded.len(), len * width);
    close(&padded[..real * width], &unpadded, "real rows");

    // Every pad row has the same token, position 1 and keys, so they agree
    // with each other, and they are not the causal rows a plain prefill
    // would have given them (which would also see the pad keys before them).
    let pad_rows: Vec<&[f32]> = padded[real * width..].chunks(width).collect();
    for row in &pad_rows[1..] {
        close(row, pad_rows[0], "pad rows are identical");
    }
    let causal = enc
        .prefill(&ops, &ids)
        .expect("plain causal over the padded ids");
    let differs = padded[real * width..]
        .iter()
        .zip(&causal[real * width..])
        .any(|(a, b)| (a - b).abs() > 1e-3);
    assert!(
        differs,
        "the masked pad rows must differ from the unmasked causal ones"
    );

    assert!(matches!(
        enc.prefill_padded(&ops, &ids, 0),
        Err(ImageGenError::EmptyPrompt)
    ));
    assert!(enc.prefill_padded(&ops, &ids, len + 1).is_err());
}

#[test]
#[ignore = "requires a Metal device"]
fn encode_returns_padded_rows_with_the_real_count() {
    let Some(backend) = device() else { return };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_decoder(dir.path(), "model.language_model.", 32);
    let enc = load(
        std::slice::from_ref(&path),
        &dir.path().join("config.json"),
        &ops,
    )
    .expect("loads");
    let (_tok_dir, tok, template) = open(Some("<|endoftext|>"));
    let encoded = encode(&enc, &tok, &template, &ops, "word word").expect("encodes");
    assert_eq!(encoded.real, TEMPLATE_IDS + 2);
    assert_eq!(encoded.len, 80);
    assert_eq!(encoded.tokens.len(), 80 * 8 * HIDDEN);
    assert!(encoded.tokens.iter().all(|v| v.is_finite()));
    assert!(matches!(
        encode(&enc, &tok, &template, &ops, " "),
        Err(ImageGenError::EmptyPrompt)
    ));
}
