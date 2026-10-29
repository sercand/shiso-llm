//! What a decode batch costs, counted in launches rather than in milliseconds.
//!
//! [`HybridStack::decode_batch`] used to be a loop: one whole forward pass per
//! member, with the rows stitched together afterwards. Nothing about that is
//! visible in an answer - every row got exactly the ids it would have got
//! alone - and nothing about it is visible in a unit test that asserts on
//! numbers. What it is visible in is the *launch count*: four rows dispatched
//! four passes, so every projection read the whole weight matrix four times,
//! and aggregate throughput over four concurrent streams was the same as over
//! one.
//!
//! So the property is asserted through [`crate::recorder::Recorder`] and not
//! through a clock: how many launches a batch issues is deterministic, is the
//! thing that actually changed, and runs on a machine with no device.
//!
//! The stack here is attention-only, because [`crate::dense::DenseKernels`] is
//! the only block launcher that lives in this crate - a recurrent slot needs a
//! family's kernels, which live a crate up. That is enough for what this file
//! asserts: the projections are what a batch amortizes, and every one of them
//! is issued by the attention block and the feed-forward beside it.

use std::sync::Arc;

use shiso_core::backend::Backend;
use shiso_core::error::CoreError;
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::model::DecodeStep;
use shiso_core::{DType, DeviceBuffer, HostBytes, alloc};

use crate::dense::DenseKernels;
use crate::draft::{DraftHead, DraftParts};
use crate::layers::{Block, LayerWeights, Weights};
use crate::mtp::SpecDepth;
use crate::options::EngineOptions;
use crate::projection::Weight;
use crate::recorder::{Bytes, Recorder};
use crate::shape::{
    Activation, AttnLayer, AttnScale, GdnShape, HybridShape, LayerKind, MediaPositions, NormScheme,
};
use crate::stack::HybridStack;

/// Residual width. A multiple of four, because the norm strides in `float4`.
const HIDDEN: u32 = 32;
/// Feed-forward inner width.
const INTER: u32 = 64;
/// Vocabulary, which is also the logits row width.
const VOCAB: u32 = 64;
/// Query and key/value heads of every attention slot.
const Q_HEADS: u32 = 2;
const KV_HEADS: u32 = 1;
/// Head width. A whole number of simdgroups, which is what
/// [`DenseKernels::validate`] refuses anything else for.
const HEAD_DIM: u32 = 32;
/// Rotated prefix, and the ladder it is derived from.
const N_ROT: u32 = 16;
const FREQ_DIMS: u32 = 32;
/// Tokens each fixture session is prefilled with.
const PROMPT: u32 = 5;

/// Two full-causal attention slots, which is the smallest stack with a layer
/// loop that runs more than once.
fn shape() -> HybridShape {
    let base = AttnLayer::new(Q_HEADS, KV_HEADS, HEAD_DIM, N_ROT, FREQ_DIMS, 1.0e4)
        .with_scale(AttnScale::One);
    HybridShape {
        layers: vec![LayerKind::Attn; 2],
        attn: vec![base, base],
        hidden: HIDDEN,
        vocab: VOCAB,
        intermediate: INTER,
        gdn: GdnShape::new(0, 0, 0, 1),
        attn_output_gate: false,
        rms_eps: 1e-6,
        activation: Activation::SwiGlu,
        norms: NormScheme::PreOnly,
        embed_scale: None,
        final_softcap: None,
        layer_scalar: false,
        ple: None,
        media_positions: MediaPositions::Linear,
        v_norm: false,
        moe: None,
        ssm: None,
        short_conv: None,
        no_ffn: Vec::new(),
    }
}

/// A weight of the declared shape, over host bytes of exactly its size.
fn weight(backend: &Arc<dyn Backend>, in_dim: u32, out_dim: u32) -> Weight {
    let bytes = usize::try_from(u64::from(in_dim) * u64::from(out_dim) * 2)
        .expect("a fixture weight fits this host");
    let host: Arc<dyn HostBytes> = Arc::new(Bytes(
        (0..bytes)
            .map(|i| u8::try_from(i % 251).unwrap_or(0))
            .collect(),
    ));
    Weight::import(backend, DType::F16, in_dim, out_dim, host, 0..bytes)
        .expect("a fixture weight imports")
}

fn norm(backend: &Arc<dyn Backend>, width: u32) -> DeviceBuffer {
    alloc(backend, u64::from(width) * 4).expect("a fixture norm allocates")
}

/// One attention slot with its feed-forward.
fn slot(backend: &Arc<dyn Backend>, ordinal: u32) -> LayerWeights {
    LayerWeights {
        input_norm: norm(backend, HIDDEN),
        post_attn_norm: None,
        ffn_norm: Some(norm(backend, HIDDEN)),
        post_ffn_norm: None,
        ffn: Some(crate::ffn::Ffn {
            gate: Some(weight(backend, HIDDEN, INTER)),
            up: weight(backend, HIDDEN, INTER),
            down: weight(backend, INTER, HIDDEN),
        }),
        moe: None,
        block: Block::Attn(crate::attn::Attn {
            q: weight(backend, HIDDEN, Q_HEADS * HEAD_DIM),
            k: Some(weight(backend, HIDDEN, KV_HEADS * HEAD_DIM)),
            v: Some(weight(backend, HIDDEN, KV_HEADS * HEAD_DIM)),
            output: weight(backend, Q_HEADS * HEAD_DIM, HIDDEN),
            q_norm: Some(norm(backend, HEAD_DIM)),
            k_norm: Some(norm(backend, HEAD_DIM)),
        }),
        ordinal,
        kv_rank: Some(ordinal),
        layer_scalar: 1.0,
        ple: None,
    }
}

fn weights(backend: &Arc<dyn Backend>) -> Weights {
    Weights {
        token_embd: weight(backend, HIDDEN, VOCAB),
        output_norm: norm(backend, HIDDEN),
        lm_head: None,
        ple_model_proj: None,
        ple_proj_norm: None,
        layers: vec![slot(backend, 0), slot(backend, 1)],
    }
}

/// A stack over a recording device, with `count` sessions already prefilled.
///
/// The prompts differ by session, so the sessions stand at different frontiers
/// over different pages. A fixture whose members were interchangeable could not
/// tell a batch that ran every row from one that ran one row four times.
fn fixture_with_sessions(count: u32) -> (Arc<Recorder>, HybridStack, Vec<SessionId>) {
    fixture_with_options(count, EngineOptions::new())
}

/// [`fixture_with_sessions`], with the engine options spelled out.
///
/// Separate only so the speculative case can open its sessions under a
/// non-zero [`SpecDepth`]: the depth sizes every session's rollback records, so
/// it has to be set before `open_session` rather than after.
fn fixture_with_options(
    count: u32,
    opts: EngineOptions,
) -> (Arc<Recorder>, HybridStack, Vec<SessionId>) {
    let recorder = Recorder::new();
    let backend: Arc<dyn Backend> = recorder.clone();
    let mut stack = HybridStack::new(
        &backend,
        shape(),
        weights(&backend),
        DType::F16,
        Arc::new(DenseKernels),
        None,
        opts,
    )
    .expect("the fixture stack builds");
    let mut sessions = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
    for member in 0..count {
        let session = stack.open_session().expect("a session opens");
        let prompt: Vec<TokenId> = (0..PROMPT)
            .map(|i| TokenId::new((i + member) % VOCAB))
            .collect();
        // `prefill_chunk`, not `prefill_last`: the last chunk of a prompt
        // projects a logits row and offers the cache a checkpoint, and a
        // checkpoint is a read-back the recording device refuses by design.
        // Nothing this file asserts needs the prompt's own logits row.
        stack
            .prefill_chunk(session, &prompt)
            .expect("the fixture prefills");
        sessions.push(session);
    }
    (recorder, stack, sessions)
}

/// Launches one `decode_batch` over `steps` issues.
fn count_launches(recorder: &Recorder, stack: &mut HybridStack, steps: &[DecodeStep]) -> usize {
    let before = recorder.launches().len();
    stack.decode_batch(steps).expect("the batch decodes");
    recorder.launches().len() - before
}

/// A batch of four costs one pass, not four.
///
/// The property, not the timing: a batched step must issue the projection
/// launches once for the batch rather than once per row. Counted through the
/// recorder rather than the clock, so the case is deterministic and runs
/// anywhere.
#[test]
fn a_batch_of_four_issues_one_set_of_projection_launches() {
    let (recorder, mut stack, sessions) = fixture_with_sessions(4);
    let steps: Vec<DecodeStep> = sessions
        .iter()
        .map(|session| DecodeStep {
            session: *session,
            token: TokenId::new(1),
        })
        .collect();

    let single = count_launches(&recorder, &mut stack, &steps[..1]);
    let quad = count_launches(&recorder, &mut stack, &steps);

    assert!(
        quad < single * 2,
        "four rows must not cost four passes: one row issued {single}, four issued {quad}"
    );
}

/// And the number, so that a regression says how far it regressed.
///
/// A four-row batch costs one pass and three more launches than a one-row
/// one, and they are worth naming because they are not the page table: the
/// per-row page table costs a device copy per row, not a launch. The three
/// are the logits rows. A row of the block is a row of the block, so the
/// vocabulary projection stays per row whatever else batches. The activation
/// is one flat launch over every row of the chunk.
///
/// What this pins is the shape: a small additive cost on top of ONE pass,
/// rather than a multiple of it. A stack with recurrent slots pays more, since
/// a slot's convolution and recurrence are per session and cost a launch a row
/// ([`crate::blocks::GdnStates::PerRow`]); this stack is attention-only,
/// because `DenseKernels` is the only block launcher this crate carries.
#[test]
fn a_batch_of_four_costs_one_pass_and_a_handful_of_launches() {
    let (recorder, mut stack, sessions) = fixture_with_sessions(4);
    let steps: Vec<DecodeStep> = sessions
        .iter()
        .map(|session| DecodeStep {
            session: *session,
            token: TokenId::new(1),
        })
        .collect();

    let single = count_launches(&recorder, &mut stack, &steps[..1]);
    let quad = count_launches(&recorder, &mut stack, &steps);

    // The logits projection is the one launch that is still per row, because a
    // row of the block is a row of the block. Everything else is one launch for
    // the batch.
    assert_eq!(
        quad,
        single + 3,
        "a batch of four should cost one pass plus three logits rows: one row \
         issued {single}, four issued {quad}"
    );
}

/// An empty batch is zero rows and zero launches, not a pass over nothing.
///
/// It was true by construction while `decode_batch` was a loop: the loop did
/// not turn. One pass over `steps` is a real pass whatever `steps` is, so the
/// emptiness is now a guard, and this is what says the guard is there. A pass
/// at `n_tok = 0` dispatches every launch of the stack at a zero width, which
/// is not a refusal anywhere: the norm would divide by the width it was handed.
#[test]
fn an_empty_batch_runs_nothing() {
    let (recorder, mut stack, _sessions) = fixture_with_sessions(1);

    let before = recorder.launches().len();
    let logits = stack.decode_batch(&[]).expect("an empty batch is Ok");

    assert_eq!(logits.rows(), 0, "an empty batch produces no rows");
    assert_eq!(
        recorder.launches().len(),
        before,
        "an empty batch issues no launch"
    );
}

/// A batch naming one session twice is refused, and refuses before it runs.
///
/// The two rows would share one recurrent state and one row of the packed page
/// table: the second would fold into a state the first had already moved past
/// and translate its position through a table written for the first's. Whether
/// that happens to come out right today is not the question - nothing states
/// it, so it is refused, and a caller that wants two tokens of one session
/// wants a chunk.
///
/// The refusal is checked beside the membership one, so nothing has advanced:
/// the session is still standing where it was and no launch went out.
#[test]
fn a_batch_naming_one_session_twice_is_refused_before_anything_runs() {
    let (recorder, mut stack, sessions) = fixture_with_sessions(2);
    let twice = [
        DecodeStep {
            session: sessions[0],
            token: TokenId::new(1),
        },
        DecodeStep {
            session: sessions[1],
            token: TokenId::new(2),
        },
        DecodeStep {
            session: sessions[0],
            token: TokenId::new(3),
        },
    ];

    let before = recorder.launches().len();
    let error = stack
        .decode_batch(&twice)
        .expect_err("a session named twice is refused");

    assert!(
        matches!(error, CoreError::Unsupported { .. }),
        "a duplicate member is refused by name, not by a device failure: {error}"
    );
    assert_eq!(
        recorder.launches().len(),
        before,
        "the refusal comes before the pass, so nothing was advanced"
    );
    // And the batch minus the duplicate still runs, so what was refused is the
    // duplicate and not the members.
    stack
        .decode_batch(&twice[..2])
        .expect("the same two members without the duplicate decode");
}

/// A resumed session does not read the residual row of the conversation it
/// replaced.
///
/// `Session::adopt` swaps the frontier, the token path and the position count,
/// and the caller restores the recurrent stack over the top. Nothing restores
/// the *scratch*: the two residual buffers hold whatever the last pass of
/// whatever session left in them, and the row index a pre-resume `PassRow`
/// carries indexes a prefix this session no longer stands on. A draft head
/// reading it would open its chain from the hidden state of the conversation
/// the resume replaced - fluently, at the right shape, with only the
/// acceptance rate to show for it, and with `SPEC_K = 0` nothing in the suite
/// would ever notice.
///
/// So the record is dropped, and this asserts both halves of that: the record
/// is gone after a resume, and a pass on the resumed prefix puts a fresh one
/// back rather than leaving the session refusing for ever.
#[test]
fn a_resumed_session_drops_the_pass_record_of_the_prefix_it_replaced() {
    let (_recorder, mut stack, sessions) = fixture_with_sessions(1);
    let session = sessions[0];
    let capacity = stack.opts.ctx;

    let before = stack
        .sessions
        .get(&session)
        .expect("the fixture's session")
        .last_pass()
        .expect("the fixture prefilled, so a pass has run");
    assert_eq!(
        before.row(),
        PROMPT - 1,
        "a chunk's record names its last row"
    );

    // A cold frontier and a different prefix: this is what a resume onto
    // somebody else's stored span looks like from the session's side.
    let frontier = stack
        .resident()
        .expect("the fixture is resident")
        .cache
        .cold_frontier();
    let elsewhere: Vec<TokenId> = (0..3).map(|i| TokenId::new(i + 17)).collect();
    stack
        .sessions
        .get_mut(&session)
        .expect("the fixture's session")
        .adopt(frontier, elsewhere, capacity)
        .expect("the session adopts the prefix");

    assert!(
        stack
            .sessions
            .get(&session)
            .expect("the fixture's session")
            .last_pass()
            .is_none(),
        "a resume drops the record naming a row of the prefix it replaced"
    );

    // And the session is not refused for ever: the next pass records its own.
    stack
        .decode_batch(&[DecodeStep {
            session,
            token: TokenId::new(1),
        }])
        .expect("the resumed session decodes");
    let after = stack
        .sessions
        .get(&session)
        .expect("the fixture's session")
        .last_pass()
        .expect("a pass on the resumed prefix records its own row");
    assert_eq!(
        after.row(),
        0,
        "a one-row decode step's record names row zero"
    );
}

/// A fixture stack with a draft head bound and `count` sessions prefilled.
///
/// The head is a second model over the same residual stream, so every width it
/// is built at is the fixture's own: `fc` takes the pair of normed terms at
/// `2 * HIDDEN` and lands on `HIDDEN`, and the block behind it is the same
/// attention slot and feed-forward the target's layers use.
fn fixture_with_draft(count: u32) -> (HybridStack, Vec<SessionId>) {
    let opts = EngineOptions {
        spec: SpecDepth::new(2),
        ..EngineOptions::new()
    };
    let (recorder, mut stack, sessions) = fixture_with_options(count, opts);
    let backend: Arc<dyn Backend> = recorder;
    let geometry =
        crate::layers::Geometry::new(&shape(), DType::F16).expect("the fixture geometry derives");
    let parts = DraftParts {
        fc: weight(&backend, 2 * HIDDEN, HIDDEN),
        norm_embed: norm(&backend, HIDDEN),
        norm_hidden: norm(&backend, HIDDEN),
        input_norm: norm(&backend, HIDDEN),
        ffn_norm: norm(&backend, HIDDEN),
        output_norm: norm(&backend, HIDDEN),
        head_norm: crate::draft::HeadNorm::Rms,
        hidden_prenorm: None,
        attn: crate::attn::Attn {
            q: weight(&backend, HIDDEN, Q_HEADS * HEAD_DIM),
            k: Some(weight(&backend, HIDDEN, KV_HEADS * HEAD_DIM)),
            v: Some(weight(&backend, HIDDEN, KV_HEADS * HEAD_DIM)),
            output: weight(&backend, Q_HEADS * HEAD_DIM, HIDDEN),
            q_norm: Some(norm(&backend, HEAD_DIM)),
            k_norm: Some(norm(&backend, HEAD_DIM)),
        },
        ffn: crate::ffn::Ffn {
            gate: Some(weight(&backend, HIDDEN, INTER)),
            up: weight(&backend, HIDDEN, INTER),
            down: weight(&backend, INTER, HIDDEN),
        },
        // A DENSE head, which is what the `qwen35` sidecar ships and what every
        // case in this file is about. The routed arm is the `qwen35moe` head's
        // and is exercised where that head is loaded.
        moe: None,
    };
    let head = DraftHead::new(&backend, parts, &geometry, 4, PROMPT)
        .expect("the fixture draft head is resident");
    stack.bind_draft(head).expect("the head binds");
    (stack, sessions)
}

/// Two speculative steps in a row, for two sessions, restage the head.
///
/// The head is one staged row shared by every session on the stack, and the
/// flag saying whether that row may be reused used to be a plain boolean.
/// `speculate` sets it at the end of every step and a verify pass does not
/// clear it - by design, because a verify is exactly the pass whose residual
/// the next chain wants - so `speculate(A)` followed by `speculate(B)` with no
/// plain or batched pass in between skipped staging altogether and B drafted
/// its first guess from A's hidden state.
///
/// That schedule is not hypothetical: the scheduler takes the speculative path
/// whenever exactly one row is runnable, so two sessions whose ticks serialise
/// alternate single-row speculative steps with nothing between them. The
/// target verifies every guess, so the symptom is an acceptance rate going
/// quietly to nothing rather than a wrong answer - which is why it is asserted
/// here rather than left to a golden.
///
/// Only the latch is exercised, because only the latch is reachable without a
/// device: every other step of a speculative round reads logits back, and the
/// recording backend refuses a read-back by design.
#[test]
fn a_second_session_restages_the_head_with_no_pass_in_between() {
    let (mut stack, sessions) = fixture_with_draft(2);
    let (first, second) = (sessions[0], sessions[1]);

    stack
        .stage_hidden_for_test(first)
        .expect("the first session stages");
    assert_eq!(
        stack.staged_session_for_test(),
        Some(first),
        "staging names the session it read the residual row from"
    );

    // No pass at all between the two, which is the case a boolean latch got
    // wrong: nothing has cleared the flag, and the row is still the first
    // session's.
    stack
        .stage_hidden_for_test(second)
        .expect("the second session stages");
    assert_eq!(
        stack.staged_session_for_test(),
        Some(second),
        "a second session's chain must restage rather than inherit the first's row"
    );

    // And the reuse the latch exists for still happens: asking twice for the
    // same session does not copy twice.
    stack
        .stage_hidden_for_test(second)
        .expect("the second session stages again");
    assert_eq!(
        stack.staged_session_for_test(),
        Some(second),
        "a repeat for the same session keeps the row it already holds"
    );
}

/// Any pass that is not a verify clears the latch for every session.
#[test]
fn a_plain_decode_step_stales_the_staged_row() {
    let (mut stack, sessions) = fixture_with_draft(1);
    let session = sessions[0];

    stack
        .stage_hidden_for_test(session)
        .expect("the session stages");
    assert_eq!(stack.staged_session_for_test(), Some(session));

    stack
        .decode_batch(&[DecodeStep {
            session,
            token: TokenId::new(1),
        }])
        .expect("the session decodes");
    assert_eq!(
        stack.staged_session_for_test(),
        None,
        "a pass that is not a verify leaves a residual of its own, so the staged row is stale"
    );
}
