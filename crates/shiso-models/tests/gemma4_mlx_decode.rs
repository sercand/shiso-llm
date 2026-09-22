//! Tests greedy decoding from the Gemma 4 MLX checkpoint, comparing streamed vs resident PLE.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::HfConfig;
use shiso_core::backend::Backend;
use shiso_core::ids::TokenId;
use shiso_core::model::{DecodeStep, Model};
use shiso_engine::{EngineOptions, argmax};
use shiso_metal::{MetalBackend, metal_available};
use shiso_models::gemma4::{Gemma4Config, Gemma4Model, PleOptions};

const PROMPT: &str = "<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n";
const STEPS: usize = 16;
const VOCAB: u32 = 262_144;

#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

#[expect(clippy::print_stdout, reason = "the decoded answer is a test's output")]
fn report(what: &str) {
    println!("{what}");
}

fn mlx_dir() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_GEMMA4_MLX").ok()?);
    path.is_dir().then_some(path)
}

fn profile() -> Gemma4Config {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/loader/gemma4-config.json");
    let document =
        HfConfig::from_file(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    Gemma4Config::from_hf(&document, Path::new("/nonexistent")).expect("the e4b geometry")
}

fn options() -> EngineOptions {
    let mut opts = EngineOptions::default();
    opts.ctx = 1024;
    opts.kv_pool_pages = 64;
    opts
}

fn encode(tokenizer: &tokenizers::Tokenizer) -> Vec<TokenId> {
    let encoded = tokenizer.encode(PROMPT, true).expect("the prompt encodes");
    let ids: Vec<u32> = encoded.get_ids().to_vec();
    assert_eq!(ids.first().copied(), Some(2));
    ids.into_iter().map(TokenId::new).collect()
}

fn greedy(row: &[u8], scratch: &mut [f32]) -> TokenId {
    for (slot, bytes) in scratch.iter_mut().zip(row.as_chunks::<4>().0) {
        *slot = f32::from_le_bytes(*bytes);
    }
    argmax(scratch)
}

struct Run {
    tokens: Vec<TokenId>,
    first_row: Vec<u8>,
}

fn generate(backend: &Arc<dyn Backend>, dir: &Path, prompt: &[TokenId], ple: PleOptions) -> Run {
    let mut model =
        Gemma4Model::load(backend, dir, &profile(), options(), ple).expect("load model");
    let session = model.open_session().expect("open session");
    let row_bytes = usize::try_from(model.info().logits_row_bytes()).expect("fits usize");

    let mut scratch = vec![0.0f32; VOCAB as usize];
    let mut tokens = Vec::with_capacity(STEPS);
    let mut first_row = vec![0u8; row_bytes];

    let mut next = {
        let logits = model
            .prefill_last(session, prompt)
            .expect("prefill succeeds");
        backend
            .copy_d2h(logits.row(0).expect("first row"), &mut first_row)
            .expect("first row readback");
        greedy(&first_row, &mut scratch)
    };
    tokens.push(next);

    for _ in 1..STEPS {
        let step = DecodeStep {
            session,
            token: next,
        };
        let logits = model.decode_batch(&[step]).expect("decode step");
        let mut row = vec![0u8; row_bytes];
        backend
            .copy_d2h(logits.row(0).expect("decode row"), &mut row)
            .expect("decode row readback");
        next = greedy(&row, &mut scratch);
        tokens.push(next);
    }

    model.close_session(session).expect("close session");
    model.teardown().expect("teardown");
    Run { tokens, first_row }
}

#[test]
#[ignore = "needs local Gemma 4 MLX checkpoint, tokenizer, and Metal device"]
fn gemma4_mlx_answers_the_paris_prompt() {
    let Some(dir) = mlx_dir() else {
        skip("no local Gemma 4 MLX directory");
        return;
    };
    let tokenizer_file = dir.join("tokenizer.json");
    if !tokenizer_file.is_file() {
        skip("no tokenizer.json in MLX directory");
        return;
    }
    if !metal_available() {
        skip("no Metal device");
        return;
    }

    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_file).expect("load tokenizer");
    let prompt = encode(&tokenizer);
    let backend: Arc<dyn Backend> = MetalBackend::new().expect("Metal device");

    let run = generate(&backend, &dir, &prompt, PleOptions::default());
    let text = tokenizer
        .decode(
            &run.tokens.iter().map(|t| t.get()).collect::<Vec<_>>(),
            true,
        )
        .expect("decode tokens");
    report(&format!("gemma4 mlx answer: {text:?}"));
    assert!(
        text.contains("Paris"),
        "the answer must contain 'Paris', got {text:?}"
    );
}

#[test]
#[ignore = "needs local Gemma 4 MLX checkpoint, tokenizer, and Metal device"]
fn gemma4_mlx_streamed_equals_resident() {
    let Some(dir) = mlx_dir() else {
        skip("no local Gemma 4 MLX directory");
        return;
    };
    let tokenizer_file = dir.join("tokenizer.json");
    if !tokenizer_file.is_file() {
        skip("no tokenizer.json in MLX directory");
        return;
    }
    if !metal_available() {
        skip("no Metal device");
        return;
    }

    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_file).expect("load tokenizer");
    let prompt = encode(&tokenizer);
    let backend: Arc<dyn Backend> = MetalBackend::new().expect("Metal device");

    let streamed = generate(
        &backend,
        &dir,
        &prompt,
        PleOptions {
            resident: false,
            cache_bytes: 512 * 1024 * 1024,
        },
    );
    let resident = generate(
        &backend,
        &dir,
        &prompt,
        PleOptions {
            resident: true,
            cache_bytes: 0,
        },
    );
    let s_f32 = bytemuck::cast_slice::<u8, f32>(&streamed.first_row);
    let r_f32 = bytemuck::cast_slice::<u8, f32>(&resident.first_row);

    assert_eq!(
        streamed.tokens, resident.tokens,
        "streamed and resident tokens must be identical"
    );
    let differing = s_f32
        .iter()
        .zip(r_f32)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    report(&format!(
        "streamed vs resident: {differing} differing logits"
    ));
    assert_eq!(differing, 0, "first logits row must be bit-identical");
}
