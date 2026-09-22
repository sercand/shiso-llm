//! Storing a session's prefix and standing a fresh one back up on it.
//!
//! The tier's policy - the floor, the prefix scan, the refusal vocabulary - is
//! the engine's and is tested there. What is tested here is the part that is
//! this family's own: that a blob carries everything a sparse resume cannot
//! reconstruct, that a resumed session is a byte image of the one that was
//! saved, and that a build which speculates and one which does not refuse to
//! read each other's files.
//!
//! # What the fake device makes possible
//!
//! The fixture's backend is a `HashMap` of byte vectors, so the device-to-host
//! reads a snapshot makes and the host-to-device writes a resume makes really
//! move bytes. A test can therefore write a recognisable pattern into a
//! session's state, store it, and assert that a *different* session standing up
//! from the file holds the same bytes - which is the only assertion that
//! distinguishes a resume from a zeroed buffer that answers anyway.
//!
//! # Nothing here has run on a GPU
//!
//! The bytes these cases move are bytes a test wrote, not bytes a kernel
//! produced: no kernel in this family has ever executed. What is asserted is the
//! bookkeeping around them.

#[path = "common/qwen4exp_fixture.rs"]
mod fixture;

use shiso_core::Model as _;
use shiso_core::ids::{SessionId, TokenId};
use shiso_engine::ckpt::{Checkpointed as _, CkptConfig, CkptOptions, SaveOutcome};
use shiso_engine::sparse::mtp::SpecDepth;
use shiso_engine::sparse::stack::SparseOptions;
use shiso_models::qwen4exp::{QWEN4EXP_MODEL_ID, Qwen4ExpModel};

/// Tokens the cases prefill, comfortably over the floor they lower.
const PROMPT: u32 = 64;

/// The floor the cases run under.
///
/// Lowered from the shipped default so a case can prefill a prompt in one chunk
/// on a fixture whose whole context is a thousand tokens. Lowering a floor is
/// not weakening a check: the floor is a cost decision about which files are
/// worth writing, and every other gate is left exactly as it ships.
fn config(dir: &std::path::Path) -> CkptConfig {
    let mut options = CkptOptions::new();
    options.min_tokens = 8;
    CkptConfig::new(dir.to_path_buf()).with_options(options)
}

/// Tokens `0..n`, which is a path no other case writes.
fn path_of(n: u32) -> Vec<TokenId> {
    (0..n).map(TokenId::new).collect()
}

/// Fills every recurrent slot, the n-gram window and - when the session has one -
/// the staged draft row with `value`.
fn paint(model: &Qwen4ExpModel, session: SessionId, value: u8) {
    let stack = model.stack();
    let live = stack.session(session).expect("the session is open");
    let mut views = Vec::new();
    for ordinal in 0..u32::try_from(live.recurrent_slots()).expect("a few slots") {
        let (conv, state) = live.recurrent(ordinal).expect("the slot exists");
        views.push(conv);
        views.push(state);
    }
    views.extend(live.ngram_window());
    views.extend(live.pending_draft());
    for view in views {
        let bytes = vec![value; usize::try_from(view.len()).expect("a fixture range fits")];
        stack
            .backend()
            .copy_h2d(view, &bytes)
            .expect("the fake device accepts a write");
    }
}

/// Whether every piece of restorable state holds `want` end to end.
fn state_is(model: &Qwen4ExpModel, session: SessionId, want: u8) -> bool {
    let stack = model.stack();
    let live = stack.session(session).expect("the session is open");
    let mut views = Vec::new();
    for ordinal in 0..u32::try_from(live.recurrent_slots()).expect("a few slots") {
        let (conv, state) = live.recurrent(ordinal).expect("the slot exists");
        views.push(conv);
        views.push(state);
    }
    views.extend(live.ngram_window());
    views.extend(live.pending_draft());
    views.into_iter().all(|view| {
        let mut bytes = vec![0_u8; usize::try_from(view.len()).expect("a fixture range fits")];
        stack
            .backend()
            .copy_d2h(view, &mut bytes)
            .expect("the fake device answers a read");
        bytes.iter().all(|byte| *byte == want)
    })
}

/// A loaded model with the tier open over `dir`, plus one prefilled session
/// whose state has been painted with `value`.
fn stored(
    dir: &tempfile::TempDir,
    opts: SparseOptions,
    value: u8,
    text: &[u8],
) -> (tempfile::TempDir, Qwen4ExpModel, SessionId) {
    let (weights, _backend, mut model) = fixture::loaded_with(opts);
    model
        .kv_enable(&config(dir.path()))
        .expect("the tier opens over a fresh directory");
    let session = model.open_session().expect("a session opens");
    let tokens = path_of(PROMPT);
    model
        .stack_mut()
        .reserve(session, PROMPT)
        .expect("the pool has room for the prompt");
    if opts.spec.is_speculative() {
        model
            .stack_mut()
            .stage_draft_row(session, 0)
            .expect("a speculative session stages its row");
    }
    paint(&model, session, value);
    // `true` is what says this chunk ended a prompt, and `Some(0)` names the
    // row a resume from here would sample: a commit without that row cannot be
    // resumed from without re-running a step, and a commit that claims to end a
    // prompt without ending one is a snapshot per decoded token.
    model
        .stack_mut()
        .advance(session, &tokens, true, Some(0))
        .expect("the span is registered");
    let outcome = model
        .kv_save(session, &tokens, text, shiso_engine::Reason::Cold)
        .expect("the store accepted the save");
    assert!(
        outcome.stored().is_some(),
        "the prefix was not stored: {outcome:?}"
    );
    (weights, model, session)
}

#[test]
fn a_stored_prefix_stands_a_fresh_session_up_on_the_bytes_it_was_saved_with() {
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let text = b"the prompt these bytes belong to";
    let (_weights, model, _session) = stored(&dir, fixture::options(), 0xA7, text);
    drop(model);

    let (_weights, _backend, mut fresh) = fixture::loaded_with(fixture::options());
    fresh
        .kv_enable(&config(dir.path()))
        .expect("the tier opens over the directory the first model wrote");
    let session = fresh.open_session().expect("a session opens");
    // A different pattern first, so a resume that wrote nothing would be visible
    // as the pattern this session started with rather than as zeros.
    paint(&fresh, session, 0x11);

    let reused = {
        let info = fresh
            .kv_resume(session, text)
            .expect("the resume did not fail")
            .expect("the stored prefix was found");
        assert_eq!(info.tokens, path_of(PROMPT), "the token path came back");
        assert_eq!(info.tokens_reused, PROMPT);
        assert_eq!(
            info.text_bytes,
            u64::try_from(text.len()).expect("a short prompt"),
            "the entry covers the whole prompt it was keyed by"
        );
        info.tokens_reused
    };
    assert_eq!(reused, PROMPT);
    assert!(
        state_is(&fresh, session, 0xA7),
        "the resumed session holds the bytes the saved one held, not the ones it \
         started with"
    );
}

#[test]
fn a_resume_restores_the_staged_draft_row_and_not_only_the_recurrent_stack() {
    // The row the draft head reads on its first step. A resume that dropped it
    // would answer, and would answer differently on exactly that step - which is
    // the one a stored prefix exists to make cheap.
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let text = b"a prompt a speculating build stored";
    let opts = fixture::speculative_options(4);
    let (_weights, model, _session) = stored(&dir, opts, 0x5E, text);
    drop(model);

    let (_weights, _backend, mut fresh) = fixture::loaded_with(opts);
    fresh
        .kv_enable(&config(dir.path()))
        .expect("the tier opens");
    let session = fresh.open_session().expect("a session opens");
    assert!(
        fresh
            .stack()
            .session(session)
            .expect("open")
            .pending_draft()
            .is_none(),
        "a fresh session has staged nothing"
    );
    fresh
        .kv_resume(session, text)
        .expect("the resume did not fail")
        .expect("the stored prefix was found");
    let stack = fresh.stack();
    let row = stack
        .session(session)
        .expect("open")
        .pending_draft()
        .expect("the resumed session can draft on its first step");
    let mut bytes = vec![0_u8; usize::try_from(row.len()).expect("a fixture range fits")];
    stack
        .backend()
        .copy_d2h(row, &mut bytes)
        .expect("the fake device answers a read");
    assert!(bytes.iter().all(|byte| *byte == 0x5E));
}

#[test]
fn a_build_that_speculates_will_not_read_a_file_written_by_one_that_does_not() {
    // The two blobs are different lengths, because the draft row is in one and
    // not the other. Reading either as the other would shift every section after
    // the first, and every shifted section still decodes to finite numbers.
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let text = b"a prompt a plain build stored";
    let (_weights, model, _session) = stored(&dir, fixture::options(), 0x22, text);
    drop(model);

    let (_weights, _backend, mut speculating) =
        fixture::loaded_with(fixture::speculative_options(4));
    speculating
        .kv_enable(&config(dir.path()))
        .expect("the tier opens");
    let session = speculating.open_session().expect("a session opens");
    assert!(
        speculating
            .kv_resume(session, text)
            .expect("a refusal to read is a cold start, not a failure")
            .is_none(),
        "a file whose state section is a different length was read anyway"
    );
}

#[test]
fn a_session_that_is_not_standing_on_the_tokens_it_offers_is_not_stored_under_their_key() {
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let (_weights, _backend, mut model) = fixture::loaded_with(fixture::options());
    model
        .kv_enable(&config(dir.path()))
        .expect("the tier opens");
    let session = model.open_session().expect("a session opens");
    model
        .stack_mut()
        .reserve(session, PROMPT)
        .expect("the pool has room");
    model
        .stack_mut()
        .advance(session, &path_of(PROMPT), true, Some(0))
        .expect("the span is registered");

    // The right number of the wrong tokens. Without the path comparison this
    // files one conversation's state under another conversation's key.
    let wrong = (100..100 + PROMPT).map(TokenId::new).collect::<Vec<_>>();
    let outcome = model
        .kv_save(session, &wrong, b"some text", shiso_engine::Reason::Cold)
        .expect("the store was asked");
    assert!(
        matches!(outcome, SaveOutcome::NotAtFrontier { .. }),
        "expected a frontier refusal, got {outcome:?}"
    );
}

#[test]
fn nothing_is_stored_until_a_directory_is_opened() {
    let (_weights, _backend, mut model) = fixture::loaded_with(fixture::options());
    assert!(!model.kv_checkpoint_enabled());
    let session = model.open_session().expect("a session opens");
    assert_eq!(
        model
            .kv_save(
                session,
                &path_of(PROMPT),
                b"text",
                shiso_engine::Reason::Cold
            )
            .expect("asking is not an error"),
        SaveOutcome::Disabled,
        "a process that was never told where to put a cache must not invent a place"
    );
    assert!(
        model
            .kv_resume(session, b"text")
            .expect("asking is not an error")
            .is_none()
    );
}

#[test]
fn the_tier_is_chosen_before_anything_stands_on_a_session() {
    // A session holds page references into the pool the old cache owns, and
    // turning the tier on builds a new one.
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let (_weights, _backend, mut model) = fixture::loaded_with(fixture::options());
    let _session = model.open_session().expect("a session opens");
    let error = model
        .kv_enable(&config(dir.path()))
        .expect_err("the tier cannot open under a live session");
    assert!(
        format!("{error:?}").contains("sessions are already running"),
        "the refusal names why: {error:?}"
    );
}

#[test]
fn a_shutdown_sweep_stores_every_session_it_is_given_text_for() {
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let (_weights, _backend, mut model) = fixture::loaded_with(fixture::options());
    model
        .kv_enable(&config(dir.path()))
        .expect("the tier opens");
    let mut sessions = Vec::new();
    for _ in 0..2u32 {
        let session = model.open_session().expect("a session opens");
        model
            .stack_mut()
            .reserve(session, PROMPT)
            .expect("the pool has room");
        model
            .stack_mut()
            .advance(session, &path_of(PROMPT), true, Some(0))
            .expect("the span is registered");
        sessions.push(session);
    }
    // One session's text is withheld, which is how a scheduler says it has none:
    // that session is skipped and the other is still stored.
    let first = *sessions.first().expect("two sessions");
    let written = model.kv_shutdown_sweep(&mut |id| {
        (id == first).then(|| (path_of(PROMPT), b"the first session's text".to_vec()))
    });
    assert_eq!(written, 1);
}

#[test]
fn this_family_tags_its_files_with_its_own_identity() {
    // The tag is what keeps a shared cache directory unambiguous, and the
    // quantization is read off the weights rather than stated by a caller.
    let (_weights, _backend, model) = fixture::loaded_with(fixture::options());
    let identity = model.ckpt_identity();
    assert_eq!(identity.model_id, QWEN4EXP_MODEL_ID);
    assert_eq!(
        identity.quant_bits,
        shiso_models::qwen4exp::quant_bits_of(model.expert_gate_dtype())
    );
    assert_ne!(
        SpecDepth::NONE.get(),
        1,
        "a sanity check on the depth newtype"
    );
}

/// A decode step produces a logits row and ends no prompt, and the difference
/// is not one the commit can work out for itself.
///
/// A snapshot is the whole recurrent stack read back to the host, one
/// device-to-host copy per slot. Reading "this span is a boundary worth
/// resuming from" off "this span produced a logits row" makes every decoded
/// token take one - and because a snapshot changes no answer, nothing that
/// asserts on output can see it. So the assertion is on the reads.
#[test]
fn a_span_that_ends_no_prompt_is_not_snapshotted() {
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let (_weights, backend, mut model) = fixture::loaded_with(SparseOptions::new());
    model
        .kv_enable(&config(dir.path()))
        .expect("the tier opens over a fresh directory");
    let session = model.open_session().expect("a session opens");
    model
        .stack_mut()
        .reserve(session, PROMPT)
        .expect("the pool has room for the prompt");
    model
        .stack_mut()
        .advance(session, &path_of(PROMPT), true, Some(0))
        .expect("the prompt's last chunk is registered");
    let after_prompt = backend.read_count();
    assert!(
        after_prompt > 0,
        "a prompt's last chunk took no snapshot at all"
    );

    // Four steps, each with the row the sampler reads and none of them a
    // boundary a later request would resume from.
    for step in 0..4u32 {
        model
            .stack_mut()
            .reserve(session, 1)
            .expect("the pool has room for a row");
        model
            .stack_mut()
            .advance(session, &[TokenId::new(PROMPT + step)], false, Some(0))
            .expect("the step is registered");
    }
    assert_eq!(
        backend.read_count(),
        after_prompt,
        "a decode step read the session's state back off the device"
    );
}
