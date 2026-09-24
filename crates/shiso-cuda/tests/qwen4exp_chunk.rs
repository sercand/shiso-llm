//! Where a prompt is cut into prefill chunks moves no bit of what it leaves
//! behind.
//!
//! A prompt reaches the model in pieces whose widths nobody chose for the
//! answer's sake: the solo chunk width, the quantum a prompt gets while other
//! sequences decode, the point a resume stood it on. So every piece must leave
//! the session exactly as one pass over the whole prompt would - every
//! recurrent slot's window and state, the n-gram window, and through the
//! attention caches the next row's logits. Close is not enough: a greedy answer
//! follows the same text until some later pick is a near tie, and then leaves
//! it for good, with nothing failing.
//!
//! The cuts are held to the recurrence's 64-position run grid, which is what
//! the scheduler and the stack keep them on; a cut between two grid points
//! splits a run, and that is a different sum by construction rather than a
//! defect. The prompt is past the indexer's budget, so the selected-block
//! attention is inside the claim, and one cut leaves a final piece narrow
//! enough to take the weight-stationary expert launch.
//!
//! It runs here rather than in `shiso-models` for the reason `qwen35_spec.rs`
//! gives: that crate names no device, so a case written there skips on the one
//! machine that could run it.
//!
//! ```text
//! SHISO_CUDA_COMPILE=1 \
//!   SHISO_TEST_MODEL_QWEN4EXP=/path/to/checkpoint.gguf \
//!   cargo test --release -p shiso-cuda --test qwen4exp_chunk -- --ignored --nocapture
//! ```
#![cfg(target_os = "linux")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::DeviceView;
use shiso_core::backend::Backend;
use shiso_core::ids::TokenId;
use shiso_core::model::Model;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_engine::EngineOptions;
use shiso_engine::sparse::capture;
use shiso_engine::sparse::stack::SparseOptions;
use shiso_modelfile::GgufFile;
use shiso_models::qwen4exp::{Qwen4ExpConfig, Qwen4ExpModel};

/// Context the model is loaded with.
const CTX: u32 = 4_096;

/// The prompt: past the indexer budget, and 30 positions past a grid point so
/// that a cut there leaves a final piece of 29 rows before the last token.
const PROMPT: u32 = 2_590;

/// The cut every other is compared with: the whole prompt in one pass.
const WHOLE: &[usize] = &[4_096];

/// Cuts on the grid, each a list of piece widths whose last entry repeats:
/// a pass per grid run; runs of mixed widths; a final piece narrow enough for
/// the weight-stationary expert launch; the scheduler's mixed quantum.
const CUTS: &[&[usize]] = &[&[64], &[1_024, 128, 1_408, 64], &[2_560], &[128], &[576]];

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// A path named by `var`, when it names a file.
fn file_from(var: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var(var).ok()?);
    path.is_file().then_some(path)
}

/// `count` token ids from a fixed pseudo-random walk over ordinary text ids.
fn prompt(count: u32) -> Vec<TokenId> {
    let mut state = 0x2545_f491_u32;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            TokenId::new(1_000 + (state >> 8) % 60_000)
        })
        .collect()
}

/// A device range, read back.
fn read(backend: &Arc<dyn Backend>, view: DeviceView<'_>) -> Vec<u8> {
    let mut bytes = vec![0_u8; usize::try_from(view.len()).expect("a host index")];
    backend
        .copy_d2h(view, &mut bytes)
        .expect("the range reads back");
    bytes
}

/// Loads the checkpoint with no draft head: a draft proposes, and what a
/// greedy verify keeps is the target's own pick, so it is outside the claim.
fn load(backend: &Arc<dyn Backend>, path: &Path) -> Qwen4ExpModel {
    let file = GgufFile::open(path).expect("the checkpoint opens");
    let config = Qwen4ExpConfig::from_metadata(file.metadata()).expect("the metadata agrees");
    drop(file);
    let mut engine = EngineOptions::default();
    engine.ctx = CTX;
    let entries = capture::capture(backend.as_ref());
    let opts = SparseOptions {
        engine,
        co_resident_blocks: capture::co_resident_blocks(backend.as_ref(), entries),
        entries,
        prefill_rows: SparseOptions::prefill_rows_for(4_096),
        ..SparseOptions::default()
    };
    Qwen4ExpModel::load_with(backend, path, None, None, &config, opts)
        .expect("the checkpoint loads")
}

/// What a prefill leaves behind, named so a mismatch says which part moved.
struct Left {
    /// Each recurrent slot's convolution window and state matrix.
    slots: Vec<(Vec<u8>, Vec<u8>)>,
    /// The n-gram block's window.
    window: Vec<u8>,
    /// The logits of the prompt's last token, run alone after the cut pieces:
    /// it reads every attention cache the pieces wrote.
    logits: Vec<u8>,
}

/// Prefills all but the prompt's last token in pieces of `cut`, then the last
/// token alone, and reads back what the session holds.
fn prefill(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    tokens: &[TokenId],
    cut: &[usize],
) -> Left {
    let session = model.open_session().expect("a session opens");
    let (body, last) = tokens.split_at(tokens.len() - 1);
    let mut rest = body;
    for piece in 0.. {
        if rest.is_empty() {
            break;
        }
        let width = cut[piece.min(cut.len() - 1)].min(rest.len());
        model
            .prefill_chunk(session, &rest[..width])
            .expect("a piece prefills");
        rest = &rest[width..];
    }
    let live = model.stack().session(session).expect("the session is open");
    let slots = (0..u32::try_from(live.recurrent_slots()).expect("a few slots"))
        .map(|slot| {
            let (conv, state) = live.recurrent(slot).expect("the slot exists");
            (read(backend, conv), read(backend, state))
        })
        .collect();
    let window = live
        .ngram_window()
        .map(|view| read(backend, view))
        .unwrap_or_default();
    let logits = {
        let rows = model
            .prefill_last(session, last)
            .expect("the last token runs");
        read(backend, rows.row(0).expect("one row"))
    };
    model.close_session(session).expect("the session closes");
    Left {
        slots,
        window,
        logits,
    }
}

#[test]
#[ignore = "needs a CUDA device and a checkpoint of this family"]
fn a_prompt_cut_anywhere_on_the_grid_leaves_the_bits_one_pass_leaves() {
    let Some(path) = file_from("SHISO_TEST_MODEL_QWEN4EXP") else {
        skip("no checkpoint named by SHISO_TEST_MODEL_QWEN4EXP");
        return;
    };
    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");
    let mut model = load(&backend, &path);
    let tokens = prompt(PROMPT);
    let whole = prefill(&backend, &mut model, &tokens, WHOLE);
    for cut in CUTS {
        let left = prefill(&backend, &mut model, &tokens, cut);
        let moved: Vec<usize> = (0..whole.slots.len())
            .filter(|&slot| whole.slots[slot] != left.slots[slot])
            .collect();
        assert!(
            moved.is_empty(),
            "cut {cut:?}: recurrent slots {moved:?} moved"
        );
        assert!(
            whole.window == left.window,
            "cut {cut:?}: the n-gram window moved"
        );
        assert!(
            whole.logits == left.logits,
            "cut {cut:?}: the next row's logits moved"
        );
    }
    model.teardown().expect("the model tears down");
}
