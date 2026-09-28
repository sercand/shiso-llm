//! The generalized layer loop, driven over a Gemma-shaped stack on a device
//! that is a `HashMap`.
//!
//! The engine's loop now reads the shape: which norms a slot carries, which
//! activation the feed-forward takes, whether the embedding is scaled, whether
//! each slot's output carries a scalar, and whether the logits are capped. None
//! of that is arithmetic a fake device can settle — what it can settle is the
//! *dispatch*, and for this change the dispatch IS the question. A family switch
//! read the wrong way round does not fail: it launches a different sequence of
//! kernels that all succeed, and the answer is fluent and wrong.
//!
//! So this pins the sequence. It also pins the two things about a shared-cache
//! slot that no width can express: that it projects no keys and no values and
//! writes no cache row, and that the rows it reads are the rows the slot it
//! shares with wrote.
//!
//! It lives beside the model crates rather than in the engine because the
//! recording device it runs on is the qwen35 fixture's, and that fixture is
//! here. Nothing in this file is qwen35: the stack is built from the engine's
//! own [`DenseKernels`].

#[path = "common/qwen35_fixture.rs"]
mod fixture;

use std::io::Write as _;
use std::sync::Arc;

use shiso_core::{Backend, DType, DeviceBuffer, HostBytes, LayerIdx, TokenId, alloc};
use shiso_engine::dense::DenseKernels;
use shiso_engine::layers::{Block, LayerWeights, PerLayerWeights, PleTable, Weights};
use shiso_engine::projection::Weight;
use shiso_engine::shape::{
    Activation, AttnLayer, AttnScale, GdnShape, HybridShape, LayerKind, MediaPositions, NormScheme,
    PerLayerShape,
};
use shiso_engine::vision::{PadTokens, VisionGrid, VisionPrefill};
use shiso_engine::{EngineOptions, HybridStack, MAX_PREFILL_CHUNK};
use shiso_ple::{PleConfig, PleIo, PleStream};

use fixture::{FakeBackend, Launch, Span};

// ---------------------------------------------------------------------------
// a tiny Gemma-shaped stack
// ---------------------------------------------------------------------------

/// Residual width. A multiple of four, because the norm strides in `float4`.
const HIDDEN: u32 = 32;
/// Feed-forward inner width.
const INTER: u32 = 64;
/// Vocabulary, which is also the logits row width.
const VOCAB: u32 = 64;
/// Query heads and key/value heads of every attention slot.
const Q_HEADS: u32 = 2;
const KV_HEADS: u32 = 1;
/// Head width. A multiple of the simdgroup, because the dense prefill store
/// covers a row with 32 lanes and `DenseKernels::validate` refuses anything
/// else — the fixture is held to the same rule the checkpoint is.
const HEAD_DIM: u32 = 32;
/// Rotated prefix, and the width of the ladder it is derived from.
const N_ROT: u32 = 16;
const FREQ_DIMS: u32 = 32;
/// Positions the windowed slot may look back over.
const WINDOW: u32 = 4;
/// Tokens the prompt carries. Below the dense prefill tile, so the softmax is
/// the decode entry.
const PROMPT: u32 = 5;
/// The projection entry a chunk of [`PROMPT`] rows dispatches.
///
/// Named once rather than spelled out, because these cases pin the SHAPE of a
/// layer and not which projection family a width happens to route to.
/// `projection::GEMM_MIN_TOKENS` is one past the widest decode batch - the
/// tiled GEMM measured at 17 GB/s against the extended matvec's 120 on this
/// device, so a decode batch must not take it - which leaves a five-row chunk
/// on a matvec. `f16` has no extended instantiation on either backend, so it
/// is the plain one; only a chunk wider than a decode batch reaches
/// `kernel_mul_mm_f16_f32`.
const PROJ: &str = "kernel_mul_mv_f16_f32_4";
/// The placeholder token a media prompt is rendered with.
const IMG: TokenId = TokenId::new(60);
/// Layer slots, which is also how many `q6_K` blocks one per-layer row carries.
const SLOTS: usize = 3;
/// Width of one slot's per-layer row. Fixed at 256 by the prepare kernel: the
/// row IS a `q6_K` block, and the engine refuses any other width.
const PLE_DIM: u32 = 256;
/// Bytes of one `q6_K` block.
const Q6K_BYTES: usize = 210;
/// Bytes one token's whole per-layer row occupies.
const PLE_ROW_BYTES: usize = SLOTS * Q6K_BYTES;
/// Rows the per-layer table holds, which is a vocabulary of its own and
/// deliberately not the model's: a bound checked against the wrong one of the
/// two would pass every case where they agree.
const PLE_VOCAB: u32 = 32;

/// Three attention slots: one full-causal, one windowed, and one that reads the
/// windowed slot's cache instead of writing its own.
///
/// `ple` says whether the stack states a per-layer embedding table, which is the
/// switch the loop's per-layer branch is chosen by.
fn shape(ple: bool) -> HybridShape {
    let base = AttnLayer::new(Q_HEADS, KV_HEADS, HEAD_DIM, N_ROT, FREQ_DIMS, 1.0e4)
        .with_scale(AttnScale::One);
    HybridShape {
        layers: vec![LayerKind::Attn; 3],
        attn: vec![
            base,
            base.with_window(WINDOW),
            base.with_window(WINDOW).shared_from(LayerIdx::new(1)),
        ],
        hidden: HIDDEN,
        vocab: VOCAB,
        intermediate: INTER,
        // A stack with no recurrent slot still states the recurrent widths; the
        // geometry allocates none of it, which is what the zero-byte assertion
        // below is about.
        gdn: GdnShape::new(0, 0, 0, 1),
        attn_output_gate: false,
        rms_eps: 1e-6,
        activation: Activation::GeGluTanh,
        norms: NormScheme::PrePost,
        embed_scale: Some(5.656_854),
        final_softcap: Some(30.0),
        layer_scalar: true,
        ple: ple.then(|| PerLayerShape::new(PLE_DIM, PLE_VOCAB)),
        media_positions: MediaPositions::Linear,
        v_norm: true,
        moe: None,
        ssm: None,
        no_ffn: Vec::new(),
    }
}

/// A weight of the declared shape, over host bytes of exactly its size.
///
/// The bytes are a pattern rather than zeros so that a launch handed the wrong
/// buffer would at least be handed different numbers; nothing here runs a
/// kernel, so what the test reads is the binding and not the product.
fn weight(backend: &Arc<dyn Backend>, in_dim: u32, out_dim: u32) -> Weight {
    let bytes = usize::try_from(u64::from(in_dim) * u64::from(out_dim) * 2)
        .expect("a fixture weight fits this host");
    let host: Arc<dyn HostBytes> = Arc::new(
        (0..bytes)
            .map(|i| u8::try_from(i % 251).unwrap_or(0))
            .collect::<Vec<u8>>(),
    );
    Weight::import(backend, DType::F16, in_dim, out_dim, host, 0..bytes)
        .expect("a fixture weight imports")
}

fn norm(backend: &Arc<dyn Backend>, width: u32) -> DeviceBuffer {
    alloc(backend, u64::from(width) * 4).expect("a fixture norm allocates")
}

/// One layer slot: the four norms this family carries, its feed-forward, and an
/// attention block that owns its cache or reads another slot's.
fn slot(
    backend: &Arc<dyn Backend>,
    ordinal: u32,
    kv_rank: u32,
    owns: bool,
    layer_scalar: f32,
    ple: bool,
) -> LayerWeights {
    let kv = owns.then(|| weight(backend, HIDDEN, KV_HEADS * HEAD_DIM));
    LayerWeights {
        input_norm: norm(backend, HIDDEN),
        post_attn_norm: Some(norm(backend, HIDDEN)),
        ffn_norm: Some(norm(backend, HIDDEN)),
        post_ffn_norm: Some(norm(backend, HIDDEN)),
        ffn: Some(shiso_engine::ffn::Ffn {
            gate: Some(weight(backend, HIDDEN, INTER)),
            up: weight(backend, HIDDEN, INTER),
            down: weight(backend, INTER, HIDDEN),
        }),
        moe: None,
        block: Block::Attn(shiso_engine::attn::Attn {
            q: weight(backend, HIDDEN, Q_HEADS * HEAD_DIM),
            k: kv
                .is_some()
                .then(|| weight(backend, HIDDEN, KV_HEADS * HEAD_DIM)),
            v: kv,
            output: weight(backend, Q_HEADS * HEAD_DIM, HIDDEN),
            q_norm: Some(norm(backend, HEAD_DIM)),
            k_norm: owns.then(|| norm(backend, HEAD_DIM)),
        }),
        ordinal,
        kv_rank: Some(kv_rank),
        layer_scalar,
        ple: ple.then(|| PerLayerWeights {
            gate: weight(backend, HIDDEN, PLE_DIM),
            proj: weight(backend, PLE_DIM, HIDDEN),
            post_norm: norm(backend, HIDDEN),
        }),
    }
}

fn weights(backend: &Arc<dyn Backend>, ple: bool) -> Weights {
    Weights {
        token_embd: weight(backend, HIDDEN, VOCAB),
        output_norm: norm(backend, HIDDEN),
        lm_head: None,
        ple_model_proj: ple.then(|| {
            weight(
                backend,
                HIDDEN,
                PLE_DIM * u32::try_from(SLOTS).expect("three slots fit a u32"),
            )
        }),
        ple_proj_norm: ple.then(|| norm(backend, PLE_DIM)),
        layers: vec![
            slot(backend, 0, 0, true, 1.0, ple),
            slot(backend, 1, 1, true, 0.5, ple),
            // The shared slot: its own ordinal, the windowed slot's run rank.
            slot(backend, 2, 1, false, 0.25, ple),
        ],
    }
}

/// A stack over a fresh recording device, plus the device itself.
fn stack() -> (Arc<FakeBackend>, HybridStack) {
    stack_with(None)
}

/// The same stack over a per-layer embedding table, or over none.
fn stack_with(ple: Option<PleTable>) -> (Arc<FakeBackend>, HybridStack) {
    let fake = FakeBackend::new();
    let backend: Arc<dyn Backend> = Arc::<FakeBackend>::clone(&fake);
    let carries = ple.is_some();
    let weights = weights(&backend, carries);
    let stack = HybridStack::new(
        &backend,
        shape(carries),
        weights,
        DType::F16,
        Arc::new(DenseKernels),
        ple,
        EngineOptions::new(),
    )
    .expect("the fixture stack builds");
    (fake, stack)
}

/// Runs one prompt and hands back every launch it dispatched, in order.
///
/// The stack comes back with them: a buffer a launch names is only readable
/// while the device still holds it, and dropping the stack here would free every
/// row before a caller could look at one.
fn prefill() -> (Arc<FakeBackend>, HybridStack, Vec<Launch>) {
    let (fake, mut stack) = stack();
    let session = stack.open_session().expect("a session opens");
    let tokens: Vec<TokenId> = (0..PROMPT).map(TokenId::new).collect();
    let mark = fake.launch_mark();
    stack
        .prefill_last(session, &tokens)
        .expect("a prefill runs");
    let seen = fake.launches_since(mark);
    (fake, stack, seen)
}

/// The entry points, in dispatch order.
fn names(launches: &[Launch]) -> Vec<&'static str> {
    launches.iter().map(|launch| launch.entry).collect()
}

/// The position row and the bound row a softmax launch was handed, `width`
/// tokens of each.
///
/// Read out of the device rather than out of the staging: these are the bytes a
/// kernel would have read, and the buffers are named by the launch itself, so
/// nothing here has to know which allocation the scratch's rows are.
/// `kernel_dense_attn_decode` binds `out, q, k, v, pos, bound, pages`.
fn position_rows(fake: &FakeBackend, launches: &[Launch], width: usize) -> (Vec<i32>, Vec<i32>) {
    position_rows_of(fake, launches, "kernel_dense_attn_decode", width)
}

/// The same two rows off the first launch of `entry`.
///
/// The two softmax entries bind the same eight arguments in the same order - the
/// tile is the only difference between them - so a chunk wide enough to reach
/// the prefill entry is read exactly the same way.
fn position_rows_of(
    fake: &FakeBackend,
    launches: &[Launch],
    entry: &str,
    width: usize,
) -> (Vec<i32>, Vec<i32>) {
    let softmax = launches
        .iter()
        .find(|launch| launch.entry == entry)
        .expect("the chunk reached the softmax");
    let row = |span: Span| -> Vec<i32> {
        assert!(
            span.offset == 0 && span.len >= 4 * width as u64,
            "a position row is bound whole: {span:?}"
        );
        fake.contents(span.id)[..width * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| i32::from_le_bytes(*word))
            .collect()
    };
    let (pos, bound) = (softmax.spans[4], softmax.spans[5]);
    assert_ne!(
        pos.id, bound.id,
        "the two rows are two buffers, so comparing them means something"
    );
    (row(pos), row(bound))
}

/// Where each layer's launches begin.
///
/// A layer ends with the scaled close, which is the one launch only the end of a
/// slot dispatches. Splitting on it is how a per-layer assertion below names a
/// slot without counting kernels.
fn layer_bounds(launches: &[Launch]) -> Vec<usize> {
    launches
        .iter()
        .enumerate()
        .filter(|(_, launch)| launch.entry == "kernel_dense_add_scale_rms_norm_mul_f32_4")
        .map(|(at, _)| at)
        .collect()
}

// ---------------------------------------------------------------------------
// the loop
// ---------------------------------------------------------------------------

/// The whole of a Gemma-shaped slot, launch for launch.
///
/// Written out rather than spot-checked, because every switch this change added
/// is invisible in any other way: an embedding that is not scaled, a post-norm
/// that never runs, a `SwiGlu` where a gated GELU belongs and a close that
/// forgot its scalar are four different wrong answers that all dispatch
/// successfully.
#[test]
fn the_first_layer_dispatches_the_shape_it_was_given() {
    let (_fake, _stack, seen) = prefill();
    let entries = names(&seen);
    let first = layer_bounds(&seen)
        .first()
        .copied()
        .expect("a layer closes with the scaled close");

    let mut expected = vec![
        // The embedding, scaled by the shape's own multiplier, and the only
        // norm in the pass that has no residual to fold in.
        "kernel_get_rows_f16_f32",
        "kernel_dense_scale_f32",
        "kernel_rms_norm_mul_f32_4",
        // The block: the query, then the keys and values, then the softmax.
        PROJ,
        "kernel_dense_q_norm_rope",
        PROJ,
        PROJ,
        "kernel_dense_kv_norm_rope_store",
        "kernel_dense_attn_decode",
        PROJ,
        // The post-attention norm, and the close that folds the normalized
        // block output into the stream and normalizes for the feed-forward.
        "kernel_rms_norm_mul_f32_4",
        "kernel_add_rms_norm_mul_f32_4",
        // The feed-forward, on the gated GELU this family takes.
        PROJ,
        PROJ,
    ];
    // The activation is flat over one token's inner row, so a chunk costs one
    // launch per token.
    expected.extend(std::iter::repeat_n(
        "kernel_dense_geglu_flat_f32",
        usize::try_from(PROMPT).expect("the prompt fits this host"),
    ));
    expected.extend([
        PROJ,
        // The post-feed-forward norm, and the close that carries the slot's own
        // scalar.
        "kernel_rms_norm_mul_f32_4",
        "kernel_dense_add_scale_rms_norm_mul_f32_4",
    ]);
    assert_eq!(&entries[..=first], &expected[..], "layer 0");
}

/// The logits row is capped, and the cap is the last thing the pass does.
#[test]
fn the_pass_ends_by_capping_the_logits() {
    let (_fake, _stack, seen) = prefill();
    let entries = names(&seen);
    assert_eq!(
        entries.last().copied(),
        Some("kernel_dense_softcap_f32"),
        "the softcap is applied to the row the sampler reads: {entries:?}"
    );
    assert_eq!(
        entries
            .iter()
            .filter(|entry| **entry == "kernel_dense_softcap_f32")
            .count(),
        1,
        "once, over the one row a prompt produces"
    );
}

/// Every slot runs the shape's activation, and no slot runs the other family's.
#[test]
fn no_slot_reaches_the_other_activation_or_the_other_close() {
    let (_fake, _stack, seen) = prefill();
    let entries = names(&seen);
    assert!(
        !entries.contains(&"kernel_swiglu_flat_f32"),
        "a gated-GELU stack must not reach the gated SiLU: {entries:?}"
    );
    assert_eq!(
        entries
            .iter()
            .filter(|entry| **entry == "kernel_dense_add_scale_rms_norm_mul_f32_4")
            .count(),
        3,
        "one scaled close per slot"
    );
    assert_eq!(
        entries
            .iter()
            .filter(|entry| **entry == "kernel_rms_norm_mul_f32_4")
            .count(),
        1 + 2 * 3,
        "the opening norm, plus a post-attention and a post-feed-forward norm per slot"
    );
}

// ---------------------------------------------------------------------------
// the slot that writes nothing
// ---------------------------------------------------------------------------

/// A shared-cache slot projects no keys and no values and stores no cache row.
///
/// Nothing downstream could notice if it did: the projections would succeed, the
/// store would write rows of the right width, and the layer that owns those rows
/// would then attend over another layer's keys.
#[test]
fn the_shared_slot_projects_no_keys_and_stores_nothing() {
    let (_fake, _stack, seen) = prefill();
    let bounds = layer_bounds(&seen);
    assert_eq!(bounds.len(), 3, "three slots, three closes");
    let last = &seen[bounds[1] + 1..=bounds[2]];
    let entries = names(last);
    assert!(
        !entries.contains(&"kernel_dense_kv_norm_rope_store"),
        "the shared slot must not write a cache row: {entries:?}"
    );
    // Two projections fewer than an owning slot: the query, the output, and the
    // feed-forward's three.
    let owning = names(&seen[bounds[0] + 1..=bounds[1]]);
    let projections = |set: &[&'static str]| set.iter().filter(|e| **e == PROJ).count();
    assert_eq!(
        projections(&entries) + 2,
        projections(&owning),
        "a shared slot is an owning slot less the key and value projections: {entries:?}"
    );
}

/// And it attends over exactly the rows its owner wrote.
///
/// The buffer identities are the whole of this: a shared slot handed its own
/// rank would read a run nobody filled, which is zeros, which is a fluent
/// answer.
#[test]
fn the_shared_slot_reads_the_runs_its_owner_wrote() {
    let (_fake, _stack, seen) = prefill();
    let bounds = layer_bounds(&seen);
    let store = |layer: usize| -> Vec<Span> {
        let span = if layer == 0 {
            &seen[..=bounds[0]]
        } else {
            &seen[bounds[layer - 1] + 1..=bounds[layer]]
        };
        span.iter()
            .find(|launch| launch.entry == "kernel_dense_kv_norm_rope_store")
            .unwrap_or_else(|| panic!("slot {layer} stored its cache"))
            .spans
            .clone()
    };
    let softmax = |layer: usize| -> Vec<Span> {
        let span = if layer == 0 {
            &seen[..=bounds[0]]
        } else {
            &seen[bounds[layer - 1] + 1..=bounds[layer]]
        };
        span.iter()
            .find(|launch| launch.entry == "kernel_dense_attn_decode")
            .unwrap_or_else(|| panic!("slot {layer} attended"))
            .spans
            .clone()
    };

    // `kernel_dense_kv_norm_rope_store` binds the key cache then the value
    // cache; `kernel_dense_attn_decode` binds the output and the query first.
    let owner = store(1);
    let shared = softmax(2);
    assert_eq!(
        shared[2], owner[0],
        "the shared slot reads its owner's keys"
    );
    assert_eq!(
        shared[3], owner[1],
        "and its owner's values, at the same offset and length"
    );
    let other = store(0);
    assert_ne!(
        shared[2], other[0],
        "and not the run the first slot filled — the two ranks are different runs"
    );
}

// ---------------------------------------------------------------------------
// a picture positioned the way text is
// ---------------------------------------------------------------------------

/// A prompt carrying media reaches the SAME rotation a text prompt does.
///
/// This family positions a picture the way it positions text, so there are no
/// three-stream entry points to reach and the dense set has none. The bound row
/// does not distinguish the span either - attention in the language model is
/// causal over a picture exactly as it is over text - so the rotation is the
/// whole of what separates the two paths. A chunk that handed this path three
/// streams would be refused outright, which is what this case would catch.
#[test]
fn a_media_prompt_keeps_the_one_dimensional_rotation() {
    let (fake, mut stack) = stack();
    let backend: Arc<dyn Backend> = Arc::<FakeBackend>::clone(&fake);
    let grids = [VisionGrid::new(1, 2, 2)];
    let rows: u32 = 4;
    let block = alloc(&backend, u64::from(rows) * u64::from(HIDDEN) * 4)
        .expect("an encoder block allocates");
    let pads = PadTokens::new(Some(IMG), None);
    let media =
        VisionPrefill::new(block.view(), &grids, pads, HIDDEN).expect("the block matches the item");

    // Text, the item's four placeholders, text.
    let mut tokens = vec![TokenId::new(1)];
    tokens.extend(std::iter::repeat_n(
        IMG,
        usize::try_from(rows).expect("fits"),
    ));
    tokens.push(TokenId::new(2));

    let session = stack.open_session().expect("a session opens");
    let mark = fake.launch_mark();
    stack
        .prefill_last_media(session, &tokens, &media)
        .expect("a multimodal prefill runs");
    let seen = names(&fake.launches_since(mark));
    assert!(
        seen.contains(&"kernel_vit_splice_rows"),
        "the encoder rows have to reach the residual stream: {seen:?}"
    );
    assert!(
        seen.contains(&"kernel_dense_q_norm_rope"),
        "and the rotation is the one-dimensional one: {seen:?}"
    );
    assert!(
        !seen.iter().any(|entry| entry.ends_with("_mrope")),
        "there is no three-stream entry point on this path: {seen:?}"
    );

    // And the bound is each token's own position, not the item's last. The
    // placeholders sit at 1..=4 and each reads only up to itself: an earlier
    // revision widened them to the item's last position, which forced
    // bidirectional attention over the span and corrupted the states this path
    // produces. The checkpoint's config carries no `use_bidirectional_attention`.
    let (pos, bound) = position_rows(&fake, &fake.launches_since(mark), tokens.len());
    assert_eq!(pos, vec![0, 1, 2, 3, 4, 5], "one position per token");
    assert_eq!(
        bound,
        vec![0, 1, 2, 3, 4, 5],
        "every token, placeholder or text alike, reads up to its own position and no further"
    );
}

/// An item wider than the whole prefill chunk still never names a cache row the
/// chunk has not written.
///
/// [`MAX_PREFILL_CHUNK`] is 512 and this item is 576 placeholders, so no cut
/// clears it: the walk's split falls back to the width and the item is served
/// over more than one chunk. The one that carries its first 512 placeholders
/// states the item's own last position, 576, as every one of their bounds - and
/// positions 513 upwards are cache rows that pass has not written, holding
/// whatever the previous occupant of those pages left. The clamp in `run_chunk`
/// is what stands between that and a softmax reading them.
#[test]
fn an_item_wider_than_the_chunk_is_bounded_by_what_the_chunk_wrote() {
    let (fake, mut stack) = stack();
    let backend: Arc<dyn Backend> = Arc::<FakeBackend>::clone(&fake);
    // 24 x 24 post-merge, which is 576 placeholder tokens against a 512-token
    // chunk.
    let grids = [VisionGrid::new(1, 24, 24)];
    let rows = grids[0].pad_count();
    assert!(
        rows > MAX_PREFILL_CHUNK,
        "the item has to outgrow the chunk"
    );
    let block = alloc(&backend, u64::from(rows) * u64::from(HIDDEN) * 4)
        .expect("an encoder block allocates");
    let pads = PadTokens::new(Some(IMG), None);
    let media =
        VisionPrefill::new(block.view(), &grids, pads, HIDDEN).expect("the block matches the item");

    // One text token, the whole item, one text token. The first chunk is that
    // leading text token alone, because the split cannot cut into the item and
    // so stops before it; the second is the item's first 512 placeholders; the
    // third is its last 64 and the trailing text token.
    let mut tokens = vec![TokenId::new(1)];
    tokens.extend(std::iter::repeat_n(
        IMG,
        usize::try_from(rows).expect("fits"),
    ));
    tokens.push(TokenId::new(2));

    let session = stack.open_session().expect("a session opens");
    let mark = fake.launch_mark();
    stack
        .prefill_last_media(session, &tokens, &media)
        .expect("a multimodal prefill runs");

    // Every chunk uploads into the same scratch row, and the last one rewrote
    // only its own 65 entries: past them the row still holds what the 512-wide
    // chunk's softmax was handed. That tail is the assertion. Without the clamp
    // it reads 576 - a key 64 positions past anything that chunk wrote.
    let width = usize::try_from(MAX_PREFILL_CHUNK).expect("the chunk fits this host");
    let last = tokens.len() - 1 - width;
    let (_, bound) = position_rows_of(
        &fake,
        &fake.launches_since(mark),
        "kernel_dense_attn_prefill",
        width,
    );
    // Attention is causal, so within a chunk the bound is the token's own
    // sequence position: the final chunk covers 513..=577, and past its 65
    // entries the row still holds what the 512-wide chunk was handed, 66..=512.
    //
    // Note what this no longer proves. A causal bound cannot exceed its own
    // chunk, so the tail's ceiling of 512 is arithmetic rather than a guard
    // doing work - which is why the clamp that used to run at the upload was
    // removed and replaced by a `debug_assert!` in `HybridStack` and by
    // `mrope_tests`' `every_bound_a_chunk_emits_is_inside_that_chunk`. What
    // this case still pins is that a split item's two halves are positioned
    // and bounded per token across the cut.
    let head =
        i32::try_from(tokens.len() - last).expect("the last chunk's first position fits an i32");
    let mut expected: Vec<i32> = (0..last)
        .map(|i| head + i32::try_from(i).expect("an offset into the chunk fits an i32"))
        .collect();
    expected.extend(
        (last..width).map(|i| i32::try_from(i + 1).expect("a previous chunk position fits an i32")),
    );
    assert_eq!(
        bound, expected,
        "each chunk's tokens read up to their own positions across the cut, and the tail the \
         512-wide chunk left behind stops at the 512 positions it wrote"
    );
}

/// A text chunk is causal, and the row that says so is the one the block is
/// handed.
///
/// The bound row is uploaded on every path, so the text path uploads the
/// positions again. That upload is the only thing standing between a causal
/// chunk and a block reading whatever the previous request left in the buffer -
/// which for a shorter prompt is a bound PAST the end of this one.
#[test]
fn a_text_chunk_bounds_every_token_at_its_own_position() {
    let (fake, _stack, seen) = prefill();
    let width = usize::try_from(PROMPT).expect("the prompt fits this host");
    let (pos, bound) = position_rows(&fake, &seen, width);
    assert_eq!(pos, (0..5).collect::<Vec<_>>());
    assert_eq!(bound, pos, "a text token reads up to itself and no further");
}

// ---------------------------------------------------------------------------
// what a stack with no recurrent slot allocates
// ---------------------------------------------------------------------------

/// Nothing asks the device for a buffer of no bytes.
///
/// A stack with no recurrent slot has no recurrent geometry and no recurrent
/// scratch, and the alternative - allocating the widths a `GdnShape::new(0, 0,
/// 0, 1)` implies - is a run of zero-byte allocations that some backends refuse
/// and the rest hand back a buffer no launch may touch.
#[test]
fn no_allocation_is_empty() {
    let (fake, _stack, _seen) = prefill();
    let empty: Vec<usize> = fake
        .allocation_sizes()
        .into_iter()
        .enumerate()
        .filter(|(_, bytes)| *bytes == 0)
        .map(|(at, _)| at)
        .collect();
    assert!(
        empty.is_empty(),
        "allocations {empty:?} were requested with no bytes"
    );
}

// ---------------------------------------------------------------------------
// the per-layer embedding branch
// ---------------------------------------------------------------------------

/// The byte a fixture table row holds at one position.
///
/// Derived from both the row and the offset, so a row read at the wrong index
/// or at the wrong stride is a row of the wrong bytes.
fn table_byte(row: u64, at: usize) -> u8 {
    u8::try_from((row.wrapping_mul(37) + at as u64 * 11 + 3) % 251).expect("masked below 251")
}

/// The whole fixture table, `PLE_VOCAB` rows of `SLOTS` `q6_K` blocks.
fn table_bytes() -> Vec<u8> {
    let mut bytes = vec![0u8; usize::try_from(PLE_VOCAB).expect("fits") * PLE_ROW_BYTES];
    for (at, slot) in bytes.iter_mut().enumerate() {
        *slot = table_byte((at / PLE_ROW_BYTES) as u64, at % PLE_ROW_BYTES);
    }
    bytes
}

/// A resident table: the same bytes, imported as a `q6_K` weight the prepare
/// kernel indexes by token id.
fn resident_table(backend: &Arc<dyn Backend>) -> PleTable {
    let host: Arc<dyn HostBytes> = Arc::new(table_bytes());
    let bytes = usize::try_from(PLE_VOCAB).expect("fits") * PLE_ROW_BYTES;
    PleTable::Resident {
        table: Weight::import(
            backend,
            DType::Q6_K,
            PLE_DIM * u32::try_from(SLOTS).expect("three slots fit a u32"),
            PLE_VOCAB,
            host,
            0..bytes,
        )
        .expect("the fixture table imports"),
    }
}

/// A table weight of a declared format and extent, over bytes of exactly its
/// size.
///
/// Separate from [`resident_table`] so the refusal cases can build one that is
/// the wrong shape on purpose.
fn resident_weight(backend: &Arc<dyn Backend>, dtype: DType, in_dim: u32, out_dim: u32) -> Weight {
    let per_row = usize::try_from(
        u64::from(in_dim) / u64::from(dtype.block_elems().get())
            * u64::from(dtype.block_bytes().get()),
    )
    .expect("a fixture row fits this host");
    let bytes = per_row * usize::try_from(out_dim).expect("fits");
    let host: Arc<dyn HostBytes> = Arc::new(
        (0..bytes)
            .map(|i| u8::try_from(i % 251).unwrap_or(0))
            .collect::<Vec<u8>>(),
    );
    Weight::import(backend, dtype, in_dim, out_dim, host, 0..bytes)
        .expect("a fixture table imports")
}

/// A streamed table: the same bytes on disk, behind a direct-addressed reader.
///
/// The temporary file comes back with it, because dropping it would unlink the
/// table the reader is part way through.
fn streamed_table() -> (tempfile::NamedTempFile, PleTable) {
    streamed_table_of(u32::try_from(PLE_ROW_BYTES).expect("a row width fits a u32"))
}

/// The same, at a row width the caller chooses, so a narrow one can be refused.
fn streamed_table_of(row_bytes: u32) -> (tempfile::NamedTempFile, PleTable) {
    let mut file = tempfile::NamedTempFile::new().expect("a temporary file");
    file.write_all(&table_bytes()).expect("the table lands");
    file.flush().expect("flush");
    file.as_file().sync_all().expect("sync");
    let handle = std::fs::File::open(file.path()).expect("reopen");
    let stream = PleStream::open_direct(
        handle,
        0,
        u64::from(PLE_VOCAB),
        row_bytes,
        1 << 16,
        &PleConfig {
            io: PleIo::Pool,
            ..PleConfig::DEFAULT
        },
    )
    .expect("the fixture table opens");
    (
        file,
        PleTable::Streamed {
            stream: Arc::new(stream),
            row_dtype: DType::Q6_K,
        },
    )
}

/// Runs one prompt over `ple` and hands back every launch it dispatched.
fn prefill_ple(ple: PleTable) -> (Arc<FakeBackend>, HybridStack, Vec<Launch>) {
    let (fake, mut stack) = stack_with(Some(ple));
    let session = stack.open_session().expect("a session opens");
    let tokens: Vec<TokenId> = (0..PROMPT).map(TokenId::new).collect();
    let mark = fake.launch_mark();
    stack
        .prefill_last(session, &tokens)
        .expect("a prefill runs");
    let seen = fake.launches_since(mark);
    (fake, stack, seen)
}

/// The whole of a per-layer slot, launch for launch.
///
/// The branch is four launches and a norm between the feed-forward's own norm
/// and the close, and every one of them is invisible in any other way: a stack
/// that folded the feed-forward in at the close rather than before the gate, or
/// that gated against the wrong slot's row, dispatches successfully and answers
/// fluently.
fn assert_per_layer_sequence(seen: &[Launch]) {
    let entries = names(seen);
    let first = layer_bounds(seen)
        .first()
        .copied()
        .expect("a layer closes with the scaled close");

    let mut expected = vec![
        "kernel_get_rows_f16_f32",
        "kernel_dense_scale_f32",
        // Once per chunk, before the first norm: the model's own contribution to
        // every slot's per-layer input, and the mix with the table's row.
        PROJ,
        "kernel_dense_ple_prepare_q6_K",
        "kernel_rms_norm_mul_f32_4",
        PROJ,
        "kernel_dense_q_norm_rope",
        PROJ,
        PROJ,
        "kernel_dense_kv_norm_rope_store",
        "kernel_dense_attn_decode",
        PROJ,
        "kernel_rms_norm_mul_f32_4",
        "kernel_add_rms_norm_mul_f32_4",
        PROJ,
        PROJ,
    ];
    expected.extend(std::iter::repeat_n(
        "kernel_dense_geglu_flat_f32",
        usize::try_from(PROMPT).expect("the prompt fits this host"),
    ));
    expected.extend([
        PROJ,
        "kernel_rms_norm_mul_f32_4",
        // The branch: the feed-forward folded into the stream, the gate the
        // per-layer row multiplies, that row's slot, the way back to the
        // residual width and its own norm.
        "kernel_add2_f32",
        PROJ,
        "kernel_dense_ple_gate_mul",
        PROJ,
        "kernel_rms_norm_mul_f32_4",
        "kernel_dense_add_scale_rms_norm_mul_f32_4",
    ]);
    assert_eq!(&entries[..=first], &expected[..], "layer 0");

    // Once per chunk, and once per slot: the prepare is not in the loop and the
    // gate multiply is.
    assert_eq!(
        entries
            .iter()
            .filter(|e| **e == "kernel_dense_ple_prepare_q6_K")
            .count(),
        1,
    );
    assert_eq!(
        entries
            .iter()
            .filter(|e| **e == "kernel_dense_ple_gate_mul")
            .count(),
        SLOTS,
    );
}

/// A resident table dispatches the branch, and the prepare reads the table
/// itself.
#[test]
fn a_resident_table_dispatches_the_per_layer_branch() {
    let fake = FakeBackend::new();
    let backend: Arc<dyn Backend> = Arc::<FakeBackend>::clone(&fake);
    let table = resident_table(&backend);
    let weights = weights(&backend, true);
    let mut stack = HybridStack::new(
        &backend,
        shape(true),
        weights,
        DType::F16,
        Arc::new(DenseKernels),
        Some(table),
        EngineOptions::new(),
    )
    .expect("the fixture stack builds");
    let session = stack.open_session().expect("a session opens");
    let tokens: Vec<TokenId> = (0..PROMPT).map(TokenId::new).collect();
    let mark = fake.launch_mark();
    stack
        .prefill_last(session, &tokens)
        .expect("a prefill runs");
    let seen = fake.launches_since(mark);
    assert_per_layer_sequence(&seen);

    // The rows the prepare was pointed at are the whole table, not a chunk's
    // staging: a resident table is indexed by the token id, so the launch has
    // to be able to reach every row.
    let prepare = seen
        .iter()
        .find(|launch| launch.entry == "kernel_dense_ple_prepare_q6_K")
        .expect("the chunk prepared its per-layer block");
    assert_eq!(
        prepare.spans[3].len,
        u64::from(PLE_VOCAB) * PLE_ROW_BYTES as u64,
        "the whole table is bound"
    );
    assert_eq!(
        fake.contents(prepare.spans[3].id)[..PLE_ROW_BYTES],
        table_bytes()[..PLE_ROW_BYTES],
        "and it is the table's own bytes"
    );
}

/// A streamed table dispatches the same branch, over the chunk's staged rows.
///
/// The bytes are the whole of the second half: a reader that read the wrong row,
/// or a staging the upload never reached, leaves the prepare pointed at zeros,
/// which decodes and answers.
#[test]
fn a_streamed_table_stages_each_tokens_row_and_dispatches_the_same_branch() {
    let (_fixture, table) = streamed_table();
    let (fake, _stack, seen) = prefill_ple(table);
    assert_per_layer_sequence(&seen);

    let prepare = seen
        .iter()
        .find(|launch| launch.entry == "kernel_dense_ple_prepare_q6_K")
        .expect("the chunk prepared its per-layer block");
    let staged = fake.contents(prepare.spans[3].id);
    let rows = usize::try_from(PROMPT).expect("the prompt fits this host");
    assert!(
        staged.len() >= rows * PLE_ROW_BYTES,
        "the staging holds the chunk: {} bytes",
        staged.len()
    );
    // Token `t` is id `t` here, so row `t` of the table is what slot `t` of the
    // staging should hold — in that order, because the prepare indexes the
    // staging by the token's own position.
    for t in 0..rows {
        for at in 0..PLE_ROW_BYTES {
            assert_eq!(
                staged[t * PLE_ROW_BYTES + at],
                table_byte(t as u64, at),
                "token {t} byte {at}",
            );
        }
    }
}

/// A shape and a table that disagree are refused at construction.
#[test]
fn a_shape_and_a_table_that_disagree_never_build_a_stack() {
    // A table the shape does not describe: the loop would never dispatch the
    // branch and the rows would be read by nothing.
    let fake = FakeBackend::new();
    let backend: Arc<dyn Backend> = Arc::<FakeBackend>::clone(&fake);
    let table = resident_table(&backend);
    let err = HybridStack::new(
        &backend,
        shape(false),
        weights(&backend, false),
        DType::F16,
        Arc::new(DenseKernels),
        Some(table),
        EngineOptions::new(),
    )
    .expect_err("a table the shape states nothing about");
    assert!(
        matches!(
            err,
            shiso_core::CoreError::ShapeMismatch {
                field: "per-layer table",
                ..
            }
        ),
        "{err:?}"
    );

    // And the other way round: a shape that states a table with none supplied
    // would dispatch the branch over a buffer nobody filled.
    let fake = FakeBackend::new();
    let backend: Arc<dyn Backend> = Arc::<FakeBackend>::clone(&fake);
    let err = HybridStack::new(
        &backend,
        shape(true),
        weights(&backend, true),
        DType::F16,
        Arc::new(DenseKernels),
        None,
        EngineOptions::new(),
    )
    .expect_err("a shape that states a table it was not given");
    assert!(
        matches!(
            err,
            shiso_core::CoreError::ShapeMismatch {
                field: "per-layer table",
                ..
            }
        ),
        "{err:?}"
    );
}

/// A stack with no per-layer table allocates none of the branch's rows and
/// dispatches none of its launches.
///
/// This is the regression guarantee restated for this change: the text path is
/// pinned to a recorded vector, and a new allocation or a new launch on it would
/// be a different instruction stream for the same model.
#[test]
fn a_stack_without_a_table_dispatches_nothing_of_the_branch() {
    let (_fake, _stack, seen) = prefill();
    let entries = names(&seen);
    for entry in [
        "kernel_dense_ple_prepare_q6_K",
        "kernel_dense_ple_gate_mul",
        "kernel_add2_f32",
    ] {
        assert!(
            !entries.contains(&entry),
            "a stack with no per-layer table must not reach {entry}: {entries:?}"
        );
    }
    // The four rows the branch would need are four allocations this stack never
    // made. Counted against a stack that differs from it in the TABLE alone —
    // the same weights on both sides — so the difference is the scratch and
    // nothing else.
    let bill = |carries: bool| -> usize {
        let fake = FakeBackend::new();
        let backend: Arc<dyn Backend> = Arc::<FakeBackend>::clone(&fake);
        let held = carries.then(|| resident_table(&backend));
        let weights = weights(&backend, true);
        let mark = fake.allocation_mark();
        let stack = HybridStack::new(
            &backend,
            shape(carries),
            weights,
            DType::F16,
            Arc::new(DenseKernels),
            held,
            EngineOptions::new(),
        )
        .expect("the fixture stack builds");
        let made = fake.allocated_since(mark).len();
        drop(stack);
        made
    };
    assert_eq!(
        bill(false) + 4,
        bill(true),
        "the per-layer rows are four allocations a stack without the table never makes"
    );
}

// ---------------------------------------------------------------------------
// what the construction refuses
// ---------------------------------------------------------------------------

/// Builds a stack over `ple` at the per-layer shape, and hands back the refusal.
///
/// Every case below is a disagreement the pass cannot notice: the prepare kernel
/// strides a row identifier by `SLOTS * 210` bytes and decodes what it finds as
/// `q6_K`, so a table that is narrow, in another format or shorter than the
/// shape says is read as if it were none of those and produces finite weights at
/// every element.
fn refuse(build: impl FnOnce(&Arc<dyn Backend>) -> PleTable) -> shiso_core::CoreError {
    let fake = FakeBackend::new();
    let backend: Arc<dyn Backend> = Arc::<FakeBackend>::clone(&fake);
    // The table is built on the stack's OWN device, so a resident one is a
    // weight this stack could really have been handed.
    let ple = Some(build(&backend));
    let weights = weights(&backend, true);
    HybridStack::new(
        &backend,
        shape(true),
        weights,
        DType::F16,
        Arc::new(DenseKernels),
        ple,
        EngineOptions::new(),
    )
    .expect_err("the table disagrees with the shape")
}

/// A streamed table one block narrow is refused rather than read at the wrong
/// stride.
#[test]
fn a_streamed_table_whose_rows_are_not_this_shapes_width_is_refused() {
    let (_fixture, table) =
        streamed_table_of(u32::try_from((SLOTS - 1) * Q6K_BYTES).expect("a row width fits a u32"));
    let err = refuse(|_| table);
    assert!(
        matches!(
            err,
            shiso_core::CoreError::ShapeMismatch {
                field: "per-layer row bytes",
                expected: 630,
                got: 420,
            }
        ),
        "{err:?}"
    );
}

/// A resident table in another format is refused rather than decoded as if it
/// were `q6_K`.
#[test]
fn a_resident_table_in_another_format_is_refused() {
    // The extents are right and only the format is wrong, so nothing but the
    // format check can be what refuses this.
    let err = refuse(|backend| PleTable::Resident {
        table: resident_weight(
            backend,
            DType::F16,
            PLE_DIM * u32::try_from(SLOTS).expect("fits"),
            PLE_VOCAB,
        ),
    });
    assert!(
        matches!(err, shiso_core::CoreError::Unsupported { .. }),
        "{err:?}"
    );
}

/// A resident table of the wrong extent is refused, in either dimension.
#[test]
fn a_resident_table_of_the_wrong_extent_is_refused() {
    let slots = u32::try_from(SLOTS).expect("fits");

    // One slot too wide: every row after the first would start inside its
    // predecessor, and the last would run off the end.
    let err = refuse(|backend| PleTable::Resident {
        table: resident_weight(backend, DType::Q6_K, PLE_DIM * (slots + 1), PLE_VOCAB),
    });
    assert!(
        matches!(
            err,
            shiso_core::CoreError::ShapeMismatch {
                field: "per-layer row width",
                expected: 768,
                got: 1024,
            }
        ),
        "{err:?}"
    );

    // The right width and half the rows: the id bound would then admit ids the
    // table has no row for.
    let err = refuse(|backend| PleTable::Resident {
        table: resident_weight(backend, DType::Q6_K, PLE_DIM * slots, PLE_VOCAB / 2),
    });
    assert!(
        matches!(
            err,
            shiso_core::CoreError::ShapeMismatch {
                field: "per-layer table rows",
                ..
            }
        ),
        "{err:?}"
    );
}

/// A streamed table with the right width and the wrong row count is refused too.
#[test]
fn a_streamed_table_of_the_wrong_row_count_is_refused() {
    let mut file = tempfile::NamedTempFile::new().expect("a temporary file");
    file.write_all(&table_bytes()).expect("the table lands");
    file.flush().expect("flush");
    file.as_file().sync_all().expect("sync");
    let handle = std::fs::File::open(file.path()).expect("reopen");
    let stream = PleStream::open_direct(
        handle,
        0,
        u64::from(PLE_VOCAB) / 2,
        u32::try_from(PLE_ROW_BYTES).expect("fits"),
        1 << 16,
        &PleConfig {
            io: PleIo::Pool,
            ..PleConfig::DEFAULT
        },
    )
    .expect("the reader opens over half the table");
    let err = refuse(|_| PleTable::Streamed {
        stream: Arc::new(stream),
        row_dtype: DType::Q6_K,
    });
    assert!(
        matches!(
            err,
            shiso_core::CoreError::ShapeMismatch {
                field: "per-layer table rows",
                expected: 32,
                got: 16,
            }
        ),
        "{err:?}"
    );
}

/// A token id past the per-layer table's own vocabulary is refused, and the
/// session it was handed to is left exactly as it was.
///
/// The kernel indexes a resident table by the id and cannot refuse one: it would
/// read whatever follows the table and decode it to finite weights. Note the
/// bound is the PER-LAYER vocabulary, which here is half the model's, so an id
/// the model can gather an embedding for is still refused by this.
#[test]
fn a_token_past_the_per_layer_vocabulary_is_refused_in_resident_mode() {
    let fake = FakeBackend::new();
    let backend: Arc<dyn Backend> = Arc::<FakeBackend>::clone(&fake);
    let table = resident_table(&backend);
    let weights = weights(&backend, true);
    let mut stack = HybridStack::new(
        &backend,
        shape(true),
        weights,
        DType::F16,
        Arc::new(DenseKernels),
        Some(table),
        EngineOptions::new(),
    )
    .expect("the fixture stack builds");
    let session = stack.open_session().expect("a session opens");

    // The id below is inside the model's vocabulary and outside the table's, so
    // the bound this asserts on can only be the table's.
    const { assert!(PLE_VOCAB < VOCAB) };
    let mark = fake.launch_mark();
    let err = stack
        .prefill_last(session, &[TokenId::new(0), TokenId::new(PLE_VOCAB)])
        .expect_err("the table has no row for that id");
    assert!(
        matches!(
            err,
            shiso_core::CoreError::OutOfBounds {
                offset: 32,
                buffer_bytes: 32,
                ..
            }
        ),
        "{err:?}"
    );
    // Refused before anything ran, so the session can still be used.
    assert!(
        fake.launches_since(mark).is_empty(),
        "the refusal came before the first launch"
    );
    stack
        .prefill_last(session, &[TokenId::new(1)])
        .expect("a prompt the table can answer for still runs");
}

/// A per-layer table on a stack that does not normalize its blocks' outputs is
/// refused at construction, not at the first prefill.
///
/// The branch projects into the post-norm scratch, which only a `PrePost` stack
/// allocates. A stack that cannot run its own shape should not become a value
/// that says it can.
#[test]
fn a_per_layer_table_needs_a_stack_that_normalizes_its_blocks_outputs() {
    let fake = FakeBackend::new();
    let backend: Arc<dyn Backend> = Arc::<FakeBackend>::clone(&fake);
    let table = resident_table(&backend);
    let mut shape = shape(true);
    shape.norms = NormScheme::PreOnly;
    let err = HybridStack::new(
        &backend,
        shape,
        weights(&backend, true),
        DType::F16,
        Arc::new(DenseKernels),
        Some(table),
        EngineOptions::new(),
    )
    .expect_err("the branch has nowhere to project into");
    assert!(
        matches!(err, shiso_core::CoreError::Unsupported { .. }),
        "{err:?}"
    );
}
