//! What speculation must not change, and what it is allowed to change.
//!
//! One case, and it is the whole correctness claim of the feature: **a greedy
//! speculative run is byte-identical to a greedy plain one.** A speedup that
//! changes an answer is not a speedup, and the failure it guards against is
//! silent - a build that accepts one token too many answers fluently from a
//! state the target never computed, and nothing downstream can tell.
//!
//! It runs here rather than in `shiso-models` for the reason
//! `q35moe_load.rs`'s header gives: that crate dev-depends on the Metal
//! backend, so a case written there answers `None` for a device on the only
//! machine that has this checkpoint and a GPU, and a skip prints the same
//! colour as a pass.
//!
//! ```text
//! SHISO_SKIP_BUILD=0 SHISO_CUDA_COMPILE=1 \
//!   SHISO_TEST_MODEL_OPTIQ=/path/to/Qwen3.5-4B/mlx \
//!   cargo test -p shiso-cuda --test qwen35_spec -- --ignored --nocapture
//! ```
//!
//! # Why the dense checkpoint and not Qwen3.6-35B-A3B
//!
//! The NVFP4 Qwen3.6-35B-A3B checkpoint emits one token and then the same CJK token for ever, so a head
//! drafting for it would accept nearly every guess and report an acceptance
//! rate that means nothing. The dense model's output is healthy, so its
//! acceptance rate is a measurement rather than an artefact - and the identity
//! this case asserts is only interesting over a continuation that actually
//! moves.
#![cfg(target_os = "linux")]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use shiso_core::HfConfig;
use shiso_core::backend::Backend;
use shiso_core::ids::TokenId;
use shiso_core::model::{DecodeStep, Model};
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_engine::mtp::SpecDepth;
use shiso_engine::{EngineOptions, argmax};
use shiso_models::qwen35::{Qwen35Config, Qwen35Model, mtp};

/// The prompt the two runs continue.
const PROMPT: &str = "The capital of France is";

/// Tokens each run produces.
///
/// Sixty-four rather than a handful: a divergence between the two runs shows up
/// at the first step where the head guesses right and the rollback is wrong,
/// which on a healthy continuation is several tokens in.
const STEPS: usize = 64;

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// Reports what a run produced, which is what a reader grades acceptance by.
#[expect(clippy::print_stdout, reason = "the measurement is a test's output")]
fn report(what: &str) {
    println!("{what}");
}

/// The sidecar directory, which carries both the checkpoint and its head.
fn sidecar() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("SHISO_TEST_MODEL_OPTIQ").ok()?);
    dir.join("config.json").is_file().then_some(dir)
}

/// The highest-scoring id of one logits row.
fn greedy(row: &[u8], scratch: &mut [f32]) -> TokenId {
    for (slot, bytes) in scratch.iter_mut().zip(row.as_chunks::<4>().0) {
        *slot = f32::from_le_bytes(*bytes);
    }
    argmax(scratch)
}

/// What one run produced and, when it speculated, how much of it was drafted.
struct Run {
    /// The ids, in order.
    ids: Vec<u32>,
    /// Guesses the target agreed with.
    accepted: u32,
    /// Guesses the head made.
    drafted: u32,
    /// Wall-clock seconds spent decoding, prompt excluded.
    ///
    /// Timed here rather than in `shiso-bench`, because the number that matters
    /// for speculation is tokens per second over a *generation* - a step that
    /// commits three tokens and a step that commits one are not comparable per
    /// step, and the bench rig measures per step.
    seconds: f64,
}

impl Run {
    /// Accepted guesses per guess offered, which is the number Task 3.6
    /// reports.
    fn rate(&self) -> f64 {
        if self.drafted == 0 {
            return 0.0;
        }
        f64::from(self.accepted) / f64::from(self.drafted)
    }

    /// Tokens a second over the whole generation.
    fn rate_per_second(&self) -> f64 {
        if self.seconds <= 0.0 {
            return 0.0;
        }
        f64::from(u32::try_from(self.ids.len()).unwrap_or(u32::MAX)) / self.seconds
    }
}

/// Runs the prompt and at least `STEPS` greedy tokens, at draft depth `k`.
///
/// `k` of zero is plain decode: no head is bound, no session keeps a trail,
/// and the loop is the one `decode_batch` has always run.
fn decode(dir: &Path, prompt: &[TokenId], k: u32) -> Run {
    let document = HfConfig::from_file(&dir.join("config.json")).expect("the sidecar's document");
    let profile = Qwen35Config::from_hf(&document).expect("the document is this family's");
    let manifest = mtp::read_manifest_from(dir, &document).expect("the head's manifest reads");
    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");
    let mut opts = EngineOptions::default();
    opts.ctx = 1024;
    opts.kv_pool_pages = 64;
    opts.spec = SpecDepth::new(k);
    let mut model = Qwen35Model::load_with_draft(
        &backend,
        dir,
        &profile,
        opts,
        manifest.as_ref().filter(|_| k > 0),
    )
    .expect("the checkpoint loads");

    let vocab = usize::try_from(profile.shape().vocab).expect("a vocabulary fits a host index");
    let mut row = vec![0u8; vocab * 4];
    let mut scratch = vec![0f32; vocab];

    let session = model.open_session().expect("a session opens");
    let last = model
        .prefill_last(session, prompt)
        .expect("the prompt runs");
    backend
        .copy_d2h(last.row(0).expect("the prompt's row"), &mut row)
        .expect("the prompt's row reads back");
    let mut token = greedy(&row, &mut scratch);

    let mut ids = vec![token.get()];
    let mut accepted = 0;
    let mut drafted = 0;
    let clock = Instant::now();
    while ids.len() < STEPS {
        if k > 0 {
            let step = model
                .spec_mut()
                .expect("a build with a head answers the speculative path")
                .speculate(session, token)
                .expect("a speculative step runs");
            accepted += step.accepted;
            drafted += step.drafted;
            // `produced` is the guesses the target kept and then its own next
            // pick, which is the token the next step commits. Everything but
            // that last one is an answer token; the last one is both.
            for id in &step.produced {
                ids.push(id.get());
            }
            token = *step.produced.last().expect("a step produces at least one");
            continue;
        }
        let out = model
            .decode_batch(&[DecodeStep { session, token }])
            .expect("a decode step runs");
        backend
            .copy_d2h(out.row(0).expect("the step's row"), &mut row)
            .expect("the step's row reads back");
        token = greedy(&row, &mut scratch);
        ids.push(token.get());
    }
    let seconds = clock.elapsed().as_secs_f64();
    model.close_session(session).expect("the session closes");
    model.teardown().expect("the model tears down");
    ids.truncate(STEPS);
    Run {
        ids,
        accepted,
        drafted,
        seconds,
    }
}

/// A greedy speculative run is byte-identical to a greedy plain one.
///
/// This is the whole correctness claim of speculation. The two depths are run
/// against the same plain baseline rather than against each other, because a
/// shared bug in the acceptance loop would make two speculative runs agree with
/// one another and with nothing the model would have said.
///
/// The model is loaded once per depth rather than once for all three: this
/// machine holds one model-loading process at a time and the checkpoint is
/// small enough that three loads cost less than the risk of carrying three
/// stacks at once.
#[test]
#[ignore = "requires a CUDA device, the sidecar checkpoint and its 86.7 MB head"]
fn speculating_does_not_change_the_answer() {
    let Some(dir) = sidecar() else {
        skip("SHISO_TEST_MODEL_OPTIQ names no sidecar directory");
        return;
    };
    let vocabulary = dir.join("tokenizer.json");
    if !vocabulary.is_file() {
        skip("no tokenizer beside the sidecar");
        return;
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&vocabulary).expect("the vocabulary loads");
    let encoded = tokenizer.encode(PROMPT, false).expect("the prompt encodes");
    let prompt: Vec<TokenId> = encoded
        .get_ids()
        .iter()
        .copied()
        .map(TokenId::new)
        .collect();

    let plain = decode(&dir, &prompt, 0);
    report(&format!(
        "plain: {:?}",
        tokenizer
            .decode(&plain.ids, true)
            .expect("the answer decodes")
    ));
    report(&format!("k=0: {:.2} tok/s", plain.rate_per_second()));
    for k in [2u32, 3, 4] {
        let spec = decode(&dir, &prompt, k);
        report(&format!(
            "k={k}: accepted {}/{} = {:.3}, {:.2} tok/s",
            spec.accepted,
            spec.drafted,
            spec.rate(),
            spec.rate_per_second()
        ));
        assert_eq!(
            plain.ids, spec.ids,
            "k={k} must produce the same ids as plain decode"
        );
    }
}
