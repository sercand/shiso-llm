//! What batching must not change: a row's answer.
//!
//! One case, and it is the whole correctness claim of Phase 4. `decode_batch`
//! used to be a loop of single-row passes, so every row trivially got the ids it
//! would have got alone. It is now one pass over B rows, which means every row
//! shares a scratch, a logits block and a launch with rows belonging to other
//! conversations - and the ways that goes wrong are all silent. A row reading
//! another session's page table answers fluently out of the wrong prefix; a row
//! folding into another session's recurrent state answers fluently out of the
//! wrong prefix and keeps doing so; a logits row written at the wrong index
//! hands one caller another's answer. None of them fails a launch.
//!
//! So the assertion is exact and the budget is zero: **the same four prompts,
//! decoded four-at-a-time, must produce the same ids they produce alone.** Not
//! within a tolerance, because the quantity being compared is a token id and a
//! token id is a choice, not a measurement.
//!
//! # The floats behind those ids are *not* promised to be identical
//!
//! An earlier version of this header said batching "changes which rows are
//! issued together, not the arithmetic any one row does". That is no longer
//! true and it is worth saying exactly why, because the assertion below still
//! holds and the reason is not the one that sentence gave.
//!
//! `shiso_engine::launch::ext_nxpsg` splits the extended matvec's inner
//! dimension across **16 lanes when `n_tok < 3` and 8 otherwise**, and `nxpsg`
//! is the reduction association of the dot product - the kernel folds the
//! partial sums over an `nxpsg`-wide shuffle tree
//! (`kernels/cuda/generic/mul_mv_ext.cuh`). Floating-point addition is not
//! associative, so two lane counts are two sums. Before Phase 4 that boundary
//! was unreachable from decode, because a decode step was always one row; with
//! `GEMM_MIN_TOKENS = 9` routing decode through that same family, a batch of
//! one or two folds over 16 lanes and a batch of three to eight folds over 8.
//!
//! **So a session's decode logits do depend on how many other sessions share
//! its batch**, in the last bits. Every row is individually correct and no row
//! reads another's state; what is not promised is bit-identity across batch
//! widths. The assertion below is on the argmax, which is what a caller
//! receives, and it is exact - a last-bit difference that flipped a greedy
//! pick would fail this case, which is the right outcome and the reason the
//! budget stays at zero rather than becoming a tolerance.
//!
//! It runs here rather than in `shiso-models` for the reason `qwen35_spec.rs`'s
//! header gives: that crate dev-depends on the Metal backend, so a case written
//! there answers `None` for a device on the only machine that has this
//! checkpoint and a GPU, and a skip prints the same colour as a pass.
//!
//! ```text
//! SHISO_SKIP_BUILD=0 SHISO_CUDA_COMPILE=1 \
//!   SHISO_TEST_MODEL_OPTIQ=/path/to/Qwen3.5-4B/mlx \
//!   cargo test -p shiso-cuda --test qwen35_batch -- --ignored --nocapture
//! ```
//!
//! **`scripts/parity_gate.sh` runs it**, as the `batch` lane. It is
//! `#[ignore]`d AND env-gated, so `cargo test --workspace` does not reach it,
//! and for a while no script named it either - which left the case its own
//! header calls "the whole correctness claim of Phase 4" executed only when
//! somebody typed its name. That is the same finding the gate's `spec` lane
//! exists to close, and it is closed the same way.
#![cfg(target_os = "linux")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::HfConfig;
use shiso_core::backend::Backend;
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::model::{DecodeStep, Model};
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_engine::{EngineOptions, argmax};
use shiso_models::qwen35::{Qwen35Config, Qwen35Model};

/// Sessions the batch carries. Four, which is the stream count the phase's
/// target is stated at.
const ROWS: usize = 4;

/// Tokens each session decodes.
///
/// Long enough for a divergence to show: two sessions that started from the
/// same page-table entry stay in step for a few tokens even when one is reading
/// the other's prefix, because a short continuation of a short prompt is mostly
/// decided by the last token.
const STEPS: usize = 24;

/// Prompt lengths, in tokens, one per row.
///
/// They differ, and one of them crosses a cache page (256 positions), which is
/// the case a shared page table passes and a per-row one is needed for: with
/// every row inside its own first page, every row reads entry zero of its own
/// table, and a defect that handed row zero's table to the batch is already
/// visible; with one row past the boundary it reads entry one as well, which is
/// the second thing that can be wrong.
const LENGTHS: [u32; ROWS] = [7, 33, 300, 91];

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// The sidecar directory, which carries the checkpoint and its config.
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

/// Row `row`'s prompt: `LENGTHS[row]` ids, none of them shared with another
/// row's.
///
/// Arithmetic rather than text, because what the rows have to be is *different*
/// and *inside the vocabulary*, and a sentence would tie the case to a
/// tokenizer file it does not otherwise need.
fn prompt(row: usize, vocab: u32) -> Vec<TokenId> {
    let len = LENGTHS.get(row).copied().unwrap_or(1);
    let salt = u32::try_from(row).unwrap_or(0).saturating_add(1);
    (0..len)
        .map(|i| {
            TokenId::new((i.saturating_mul(97).saturating_add(salt.saturating_mul(31))) % vocab)
        })
        .collect()
}

/// A model on the device, with the widths a caller needs to read a row back.
struct Rig {
    backend: Arc<dyn Backend>,
    model: Qwen35Model,
    row: Vec<u8>,
    scratch: Vec<f32>,
    vocab: u32,
}

impl Rig {
    /// Loads the sidecar checkpoint on device 0.
    fn open(dir: &Path) -> Self {
        let document = HfConfig::from_file(&dir.join("config.json")).expect("the sidecar document");
        let profile = Qwen35Config::from_hf(&document).expect("the document is this family's");
        let backend: Arc<dyn Backend> =
            CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");
        let mut opts = EngineOptions::default();
        opts.ctx = 1024;
        // Eight sessions of up to 1024 positions each, at 256 positions a page.
        opts.kv_pool_pages = 128;
        let model = Qwen35Model::load(&backend, dir, &profile, opts).expect("the checkpoint loads");
        let vocab = profile.shape().vocab;
        let width = usize::try_from(vocab).expect("a vocabulary fits a host index");
        Self {
            backend,
            model,
            row: vec![0u8; width * 4],
            scratch: vec![0f32; width],
            vocab,
        }
    }

    /// Opens a session on row `row`'s prompt and answers the token that follows
    /// it.
    fn stand_at(&mut self, row: usize) -> (SessionId, TokenId) {
        let ids = prompt(row, self.vocab);
        let session = self.model.open_session().expect("a session opens");
        let last = self
            .model
            .prefill_last(session, &ids)
            .expect("the prompt runs");
        self.backend
            .copy_d2h(last.row(0).expect("the prompt's row"), &mut self.row)
            .expect("the prompt's row reads back");
        (session, greedy(&self.row, &mut self.scratch))
    }
}

/// `STEPS` greedy ids for row `row`, decoded as the only member of its batch.
fn alone(rig: &mut Rig, row: usize) -> Vec<u32> {
    let (session, mut token) = rig.stand_at(row);
    let mut ids = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        ids.push(token.get());
        let out = rig
            .model
            .decode_batch(&[DecodeStep { session, token }])
            .expect("a decode step runs");
        rig.backend
            .copy_d2h(out.row(0).expect("the step's row"), &mut rig.row)
            .expect("the step's row reads back");
        token = greedy(&rig.row, &mut rig.scratch);
    }
    rig.model
        .close_session(session)
        .expect("the session closes");
    ids
}

/// `STEPS` greedy ids for every row, decoded four at a time.
fn batched(rig: &mut Rig) -> Vec<Vec<u32>> {
    let mut sessions = Vec::with_capacity(ROWS);
    let mut tokens = Vec::with_capacity(ROWS);
    for row in 0..ROWS {
        let (session, token) = rig.stand_at(row);
        sessions.push(session);
        tokens.push(token);
    }
    let mut ids: Vec<Vec<u32>> = (0..ROWS).map(|_| Vec::with_capacity(STEPS)).collect();
    // One buffer per row: the sampler cannot run while the logits block is
    // still borrowed, so every row is copied out before any of them is picked.
    let mut host = vec![vec![0u8; rig.row.len()]; ROWS];
    for _ in 0..STEPS {
        let steps: Vec<DecodeStep> = sessions
            .iter()
            .zip(&tokens)
            .map(|(session, token)| DecodeStep {
                session: *session,
                token: *token,
            })
            .collect();
        for (row, token) in tokens.iter().enumerate() {
            ids[row].push(token.get());
        }
        {
            let out = rig.model.decode_batch(&steps).expect("the batch decodes");
            for (row, dst) in host.iter_mut().enumerate() {
                let at = u32::try_from(row).expect("four fits");
                rig.backend
                    .copy_d2h(out.row(at).expect("the row exists"), dst)
                    .expect("the row reads back");
            }
        }
        for (token, bytes) in tokens.iter_mut().zip(&host) {
            rig.row.copy_from_slice(bytes);
            *token = greedy(&rig.row, &mut rig.scratch);
        }
    }
    for session in sessions {
        rig.model
            .close_session(session)
            .expect("the session closes");
    }
    ids
}

/// Four sessions decoded together give each row exactly the ids it gets alone.
#[test]
#[ignore = "requires a CUDA device and the sidecar checkpoint"]
fn a_batched_step_gives_each_row_the_ids_it_gets_alone() {
    let Some(dir) = sidecar() else {
        skip("SHISO_TEST_MODEL_OPTIQ names no sidecar directory");
        return;
    };
    let mut rig = Rig::open(&dir);
    // The lone runs first, on sessions of their own: the batch's sessions are
    // opened afterwards and are a fresh set, so nothing a lone run left behind
    // is what the batch reads.
    let solo: Vec<Vec<u32>> = (0..ROWS).map(|row| alone(&mut rig, row)).collect();
    let together = batched(&mut rig);
    rig.model.teardown().expect("the model tears down");

    for (row, (want, got)) in solo.iter().zip(&together).enumerate() {
        assert_eq!(
            got, want,
            "row {row} decoded in a batch of {ROWS} must produce the ids it produced alone",
        );
    }
}
