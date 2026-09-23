//! The qwen4exp chat path against the reference renderer, byte for byte.
//!
//! Every fixture under `test-vectors/references/chat/qwen4exp/` holds a wire body
//! and what the reference implementation's own `apply_chat_template` produced for
//! the conversation the official serving path would hand it. This renders the
//! body through the server's real path - wire parse, IR, template, tokenizer -
//! and demands the same text and the same ids. A difference is the server
//! showing the model a prompt it was not trained on.
//!
//! The template and tokenizer come from the checkpoint's sidecar directory,
//! named by `SHISO_TEST_SIDECAR_QWEN4EXP`; without it every case skips and says
//! so.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;
use shiso_server::model_factory::FamilyDefaults;
use shiso_server::tokenizer::Vocabulary;
use shiso_server::tokenizer::render_prompt::PromptRenderer;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/chat/qwen4exp")
}

#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

fn renderer(sidecar: &Path) -> PromptRenderer {
    let template = std::fs::read_to_string(sidecar.join("chat_template.jinja"))
        .expect("the sidecar has a chat template");
    let vocabulary = Arc::new(
        Vocabulary::from_file(&sidecar.join("tokenizer.json"))
            .expect("the sidecar vocabulary loads"),
    );
    let renderer = PromptRenderer::new(&template, shiso_server::SERVER_TOOL_JSON_STYLE, vocabulary)
        .expect("the template compiles");
    FamilyDefaults::QWEN4EXP.configure(renderer)
}

fn request(case: &Value) -> Result<shiso_ir::Request, String> {
    let body = &case["body"];
    match case["surface"].as_str() {
        Some("openai") => shiso_server::openai::chat_to_ir::to_ir(
            &serde_json::from_value(body.clone()).map_err(|e| format!("wire: {e}"))?,
            "qwen3.8-flash",
        )
        .map_err(|e| format!("to_ir: {e}")),
        Some("anthropic") => shiso_server::anthropic::messages_to_ir::to_ir(
            &serde_json::from_value(body.clone()).map_err(|e| format!("wire: {e}"))?,
            "qwen3.8-flash",
        )
        .map_err(|e| format!("to_ir: {e}")),
        other => Err(format!("unknown surface {other:?}")),
    }
}

/// Every fixture, rendered by the server, equals the reference.
///
/// All cases run before the assertion, so one run lists every mismatch rather
/// than stopping at the first - the list is the defect inventory.
#[test]
fn every_fixture_renders_like_the_reference() {
    let Some(sidecar) = std::env::var_os("SHISO_TEST_SIDECAR_QWEN4EXP").map(PathBuf::from) else {
        skip("SHISO_TEST_SIDECAR_QWEN4EXP is unset; the qwen4exp chat parity cases did not run");
        return;
    };
    let renderer = renderer(&sidecar);
    let mut failures = Vec::new();
    let mut cases: Vec<PathBuf> = std::fs::read_dir(fixtures())
        .expect("the fixture directory exists")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.to_string_lossy().ends_with(".request.json"))
        .collect();
    cases.sort();
    assert!(
        !cases.is_empty(),
        "no fixtures found under {}",
        fixtures().display()
    );
    for path in cases {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        let stem = name.trim_end_matches(".request.json");
        let case: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("the fixture is readable"))
                .expect("the fixture is JSON");
        let expected_stem = case["expected_from"].as_str().unwrap_or(stem);
        let want_text = std::fs::read_to_string(fixtures().join(format!("{expected_stem}.txt")))
            .expect("the expected render exists; run render_chat_fixtures.py");
        let want_ids: Vec<u32> =
            std::fs::read_to_string(fixtures().join(format!("{expected_stem}.ids")))
                .expect("the expected ids exist")
                .lines()
                .map(|line| line.parse().expect("an id per line"))
                .collect();
        let request = match request(&case) {
            Ok(request) => request,
            Err(error) => {
                failures.push(format!("{stem}: refused before rendering: {error}"));
                continue;
            }
        };
        let got = match renderer.render(&request) {
            Ok(got) => got,
            Err(error) => {
                failures.push(format!("{stem}: refused: {error}"));
                continue;
            }
        };
        if got.text != want_text {
            let at = got
                .text
                .bytes()
                .zip(want_text.bytes())
                .take_while(|(a, b)| a == b)
                .count();
            let lo = floor_char(&got.text, at.saturating_sub(40));
            let lo_want = floor_char(&want_text, at.saturating_sub(40));
            failures.push(format!(
                "{stem}: text differs at byte {at}\n  got:  {:?}\n  want: {:?}",
                got.text
                    .get(lo..floor_char(&got.text, (at + 80).min(got.text.len())))
                    .unwrap_or(""),
                want_text
                    .get(lo_want..floor_char(&want_text, (at + 80).min(want_text.len())))
                    .unwrap_or(""),
            ));
            continue;
        }
        let got_ids: Vec<u32> = got.tokens.iter().map(|t| t.get()).collect();
        if got_ids != want_ids {
            failures.push(format!(
                "{stem}: text equal but ids differ (got {} ids, want {}; first 4 got {:?} want {:?})",
                got_ids.len(),
                want_ids.len(),
                &got_ids[..got_ids.len().min(4)],
                &want_ids[..want_ids.len().min(4)],
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} case(s) differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The largest char boundary at or below `at`, so a diff window never splits
/// a multi-byte character and hides the very bytes that differ.
fn floor_char(text: &str, mut at: usize) -> usize {
    while at > 0 && !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}
