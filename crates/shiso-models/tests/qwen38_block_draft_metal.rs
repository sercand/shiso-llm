//! The dense 27B checkpoint drafting with its block drafter, on Metal.
//!
//! Three claims, each against something this build did not produce:
//!
//! - **Exactness.** A greedy continuation decoded speculatively is the one
//!   decoded a token at a time, id for id. Drafting changes how many tokens a
//!   step commits, never which.
//! - **The drafter is the drafter.** Its first block behind a prompt agrees with
//!   the reference implementation's (`references/qwen38-27b-dflash2-block.json`,
//!   captured by `scripts/reference/block_draft_reference.py`) at most positions.
//!   Not all: both sides quantize to 4 bits by the same rule from different
//!   float roundings, and a drafted token is an argmax.
//! - **It pays.** Tokens committed per round, printed beside the reference's.
//!
//! ```text
//! SHISO_TEST_MODEL_QWEN38_27B=<MLX export> SHISO_TEST_DRAFT_BLOCK=<drafter dir> \
//!   cargo test -p shiso-models --release --features shiso-metal/metal \
//!   --test qwen38_block_draft_metal -- --ignored --nocapture --test-threads=1
//! ```

#![expect(
    clippy::cast_precision_loss,
    reason = "token counts of a 64-token run, turned into rates"
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use shiso_core::HfConfig;
use shiso_core::backend::Backend;
use shiso_core::ids::TokenId;
use shiso_core::model::{DecodeStep, Model};
use shiso_engine::mtp::SpecDepth;
use shiso_engine::{EngineOptions, argmax};
use shiso_metal::{MetalBackend, metal_available};
use shiso_models::qwen35::{Qwen35Config, Qwen35Model};

/// Tokens each continuation runs to.
const STEPS: usize = 64;

#[expect(
    clippy::print_stdout,
    reason = "the skip report and the measurements are the test's output"
)]
fn say(what: &str) {
    println!("{what}");
}

fn dirs() -> Option<(PathBuf, PathBuf)> {
    let model = PathBuf::from(std::env::var("SHISO_TEST_MODEL_QWEN38_27B").ok()?);
    let draft = PathBuf::from(std::env::var("SHISO_TEST_DRAFT_BLOCK").ok()?);
    (model.is_dir() && draft.is_dir()).then_some((model, draft))
}

fn reference() -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/qwen38-27b-dflash2-block.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("the reference is present"))
        .expect("the reference parses")
}

fn ids(v: &serde_json::Value) -> Vec<TokenId> {
    v.as_array()
        .expect("an id array")
        .iter()
        .map(|x| TokenId::new(u32::try_from(x.as_u64().expect("an id")).expect("a u32 id")))
        .collect()
}

fn greedy(
    backend: &Arc<dyn Backend>,
    row: shiso_core::buffer::DeviceView<'_>,
    vocab: usize,
) -> TokenId {
    let mut bytes = vec![0u8; vocab * 4];
    backend
        .copy_d2h(row, &mut bytes)
        .expect("the row reads back");
    let floats: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    argmax(&floats)
}

fn load(backend: &Arc<dyn Backend>, model: &Path, spec: bool) -> Qwen35Model {
    let config =
        Qwen35Config::from_hf(&HfConfig::from_file(&model.join("config.json")).expect("config"))
            .expect("the 27B is this family");
    let mut opts = EngineOptions::default();
    opts.ctx = 2048;
    opts.kv_pool_pages = 64;
    if spec {
        opts.spec = SpecDepth::new(7);
    }
    Qwen35Model::load(backend, model, &config, opts).expect("the export loads")
}

#[test]
#[ignore = "needs the local 27B export, its block drafter and a Metal device"]
fn speculative_greedy_decoding_is_serial_greedy_decoding() {
    let Some((model_dir, draft_dir)) = dirs() else {
        say("SKIP: SHISO_TEST_MODEL_QWEN38_27B or SHISO_TEST_DRAFT_BLOCK names no directory");
        return;
    };
    if !metal_available() {
        say("SKIP: no Metal device");
        return;
    }
    let reference = reference();
    let backend: Arc<dyn Backend> = MetalBackend::new().expect("a Metal device");
    for record in reference["records"].as_array().expect("records") {
        let name = record["prompt"].as_str().expect("a prompt name");
        let prompt = ids(&record["prompt_tokens"]);

        // Serial.
        let mut model = load(&backend, &model_dir, false);
        let vocab = model.info().vocab as usize;
        let session = model.open_session().expect("a session");
        let row = model
            .prefill_last(session, &prompt)
            .expect("the prompt runs")
            .row(0)
            .expect("row");
        let mut token = greedy(&backend, row, vocab);
        let mut serial = vec![token];
        let clock = Instant::now();
        while serial.len() < STEPS {
            let out = model
                .decode_batch(&[DecodeStep { session, token }])
                .expect("a step");
            token = greedy(&backend, out.row(0).expect("row"), vocab);
            serial.push(token);
        }
        let serial_tps = (STEPS - 1) as f64 / clock.elapsed().as_secs_f64();
        model.close_session(session).expect("close");
        model.teardown().expect("teardown");
        drop(model);

        // Speculative.
        let mut model = load(&backend, &model_dir, true);
        model
            .bind_block_drafter(&backend, &draft_dir)
            .expect("the drafter binds");
        let session = model.open_session().expect("a session");
        let row = model
            .prefill_last(session, &prompt)
            .expect("the prompt runs")
            .row(0)
            .expect("row");
        let mut token = greedy(&backend, row, vocab);
        assert_eq!(
            token.get(),
            u32::try_from(record["first"].as_u64().expect("first")).expect("u32"),
            "{name}: the pick after the prompt is not the reference's"
        );
        let mut drafted = vec![token];
        let mut rounds = 0usize;
        let clock = Instant::now();
        while drafted.len() < STEPS {
            let spec = model
                .spec_mut()
                .expect("the drafter makes this model speculative");
            let step = spec.speculate(session, token).expect("a speculative step");
            rounds += 1;
            drafted.extend_from_slice(&step.produced);
            token = *step
                .produced
                .last()
                .expect("a step produces its bonus token");
        }
        let draft_tps = (drafted.len() - 1) as f64 / clock.elapsed().as_secs_f64();
        model.close_session(session).expect("close");
        model.teardown().expect("teardown");
        drafted.truncate(STEPS);

        let tokens_per_round = (drafted.len() - 1) as f64 / rounds as f64;
        say(&format!(
            "{name}: serial {serial_tps:.2} tok/s, drafted {draft_tps:.2} tok/s, \
             {tokens_per_round:.2} tokens/round (reference {:.2})",
            record["tokens_per_round"].as_f64().unwrap_or(0.0)
        ));
        let agree = serial
            .iter()
            .zip(&drafted)
            .take_while(|(a, b)| a == b)
            .count();
        assert_eq!(
            agree, STEPS,
            "{name}: speculative decoding parted from serial decoding at {agree}: \
             serial {serial:?}, drafted {drafted:?}"
        );
    }
}
