//! The Qwen-Image prompt against the reference pipeline's own token ids.
//!
//! `test-vectors/references/qwen_image21/tokenizer.json` holds the ids the
//! official pipeline hands its text encoder for five prompts. The vocabulary
//! is the checkpoint's `image_assets/processor/tokenizer.json` under
//! `SHISO_TEST_MODEL_QWEN_IMAGE21` (the `turbo-int8` directory); without the
//! variable every case says so on stderr and returns, which looks like a
//! pass - check stderr before reporting one.

use std::path::{Path, PathBuf};

use shiso_vision::generator::ImageGenError;

use super::prompt::{DROP_IDX, MAX_PROMPT_TOKENS, QwenImagePrompt};

fn fixture() -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/qwen_image21/tokenizer.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("the tokenizer fixture parses")
}

fn ids(value: &serde_json::Value) -> Vec<u32> {
    value
        .as_array()
        .expect("an id list")
        .iter()
        .map(|v| u32::try_from(v.as_u64().expect("an id")).expect("ids fit u32"))
        .collect()
}

#[expect(
    clippy::print_stderr,
    reason = "a skipped real-vocabulary case must say why on stderr, or it reads as a pass"
)]
fn processor_dir() -> Option<PathBuf> {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_QWEN_IMAGE21") else {
        eprintln!(
            "note: SHISO_TEST_MODEL_QWEN_IMAGE21 is unset, skipping the Qwen-Image prompt case"
        );
        return None;
    };
    Some(PathBuf::from(root).join("image_assets/processor"))
}

fn prompt() -> Option<QwenImagePrompt> {
    let dir = processor_dir()?;
    Some(QwenImagePrompt::load(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())))
}

#[test]
fn plain_prompts_encode_exactly_as_the_reference_pipeline_does() {
    let Some(prompt) = prompt() else { return };
    let fixture = fixture();
    assert_eq!(fixture["drop_idx"].as_u64(), Some(DROP_IDX as u64));
    for case in ["cat", "empty", "scene60", "cjk"] {
        let want = &fixture["cases"][case];
        let text = want["prompt"].as_str().expect("a prompt");
        let got = prompt
            .tokenize(text)
            .unwrap_or_else(|e| panic!("{case}: {e}"));
        assert_eq!(got.ids, ids(&want["ids"]), "{case}: full ids");
        assert_eq!(got.drop, DROP_IDX, "{case}: drop");
        assert_eq!(
            got.kept(),
            ids(&want["kept_ids"]).as_slice(),
            "{case}: kept ids"
        );
    }
}

#[test]
fn a_spelled_control_token_stays_words() {
    let Some(prompt) = prompt() else { return };
    let fixture = fixture();
    let case = &fixture["cases"]["special_literal"];
    let text = case["prompt"].as_str().expect("a prompt");
    let reference = ids(&case["ids"]);
    let im_start =
        u32::try_from(fixture["special_ids"]["<|im_start|>"].as_u64().expect("id")).expect("fits");
    let image_pad = u32::try_from(
        fixture["special_ids"]["<|image_pad|>"]
            .as_u64()
            .expect("id"),
    )
    .expect("fits");
    // The reference reads the spellings as control ids; this encoding is
    // the deliberate deviation, so the two must differ.
    assert!(reference.contains(&image_pad));
    let got = prompt.tokenize(text).expect("the prompt encodes");
    assert_ne!(got.ids, reference);

    // The user span is everything between "<|im_start|>user\n" and the
    // closing "<|im_end|>\n<|im_start|>assistant\n": the template contributes
    // three tokens before it and five after it.
    let kept = got.kept();
    let user = &kept[3..kept.len() - 5];
    assert!(
        !user.contains(&im_start),
        "user span {user:?} holds <|im_start|>"
    );
    assert!(
        !user.contains(&image_pad),
        "user span {user:?} holds <|image_pad|>"
    );
    // The template around it is unchanged: same head and tail as the
    // reference's, and the reference's own ids before the caller's text.
    assert_eq!(&got.ids[..DROP_IDX + 3], &reference[..DROP_IDX + 3]);
    assert_eq!(&kept[kept.len() - 5..], &reference[reference.len() - 5..]);
    // Each spelling became several word tokens instead of one control id.
    assert!(
        user.len() > 6,
        "user span {user:?} is too short to hold the spellings as words"
    );
}

#[test]
fn whitespace_only_prompts_are_sent_as_one_space() {
    let Some(prompt) = prompt() else { return };
    let empty = prompt.tokenize("").expect("encodes");
    assert_eq!(prompt.tokenize("   ").expect("encodes"), empty);
    assert_eq!(prompt.tokenize("\n\t").expect("encodes"), empty);
}

#[test]
fn an_over_long_prompt_is_refused_with_its_count() {
    let Some(prompt) = prompt() else { return };
    // " cat" is one token, so 600 of them plus the eight template tokens
    // kept around the user text is well over the cap.
    let long = "cat".to_string() + &" cat".repeat(599);
    match prompt.tokenize(&long) {
        Err(ImageGenError::PromptTooLong { tokens, max }) => {
            assert_eq!(max as usize, MAX_PROMPT_TOKENS);
            assert_eq!(tokens, 600 + 8, "the count names the kept tokens");
        }
        other => panic!("expected PromptTooLong, got {other:?}"),
    }
    // Exactly at the cap is accepted.
    let at_cap = "cat".to_string() + &" cat".repeat(MAX_PROMPT_TOKENS - 8 - 1);
    let ok = prompt
        .tokenize(&at_cap)
        .expect("a prompt at the cap encodes");
    assert_eq!(ok.kept().len(), MAX_PROMPT_TOKENS);
}

/// `--image-tokenizer` (and the hub's `image_tokenizer` role) names a lone
/// `tokenizer.json` file: copied into an empty directory, with no
/// `tokenizer_config.json` or chat template beside it, it loads and
/// encodes every fixture prompt exactly as the processor directory does.
#[test]
fn a_lone_tokenizer_file_encodes_like_the_processor_directory() {
    let Some(dir) = processor_dir() else { return };
    let lone = std::env::temp_dir().join(format!("shiso-qwen-tokenizer-{}", std::process::id()));
    std::fs::create_dir_all(&lone).expect("mkdir");
    let file = lone.join("tokenizer.json");
    std::fs::copy(dir.join("tokenizer.json"), &file).expect("copy");
    let from_dir = QwenImagePrompt::load(&dir).expect("the processor directory loads");
    let from_file = QwenImagePrompt::load(&file).expect("the lone file loads");
    let fixture = fixture();
    for case in ["cat", "empty", "scene60", "cjk", "special_literal"] {
        let text = fixture["cases"][case]["prompt"].as_str().expect("a prompt");
        assert_eq!(
            from_file.tokenize(text).expect("encodes").ids,
            from_dir.tokenize(text).expect("encodes").ids,
            "{case}"
        );
    }
    std::fs::remove_dir_all(&lone).expect("cleanup");
}
