//! The multimodal prefill, proven against a device that is a `HashMap`.
//!
//! It lives beside the qwen35 family rather than in the engine because what it
//! asserts is which *qwen35* entry points a chunk reaches, and those launches
//! are this crate's: the engine supplies the residual stream, the splice and
//! the position rows, and [`Qwen35Kernels`] supplies the rotation.
//!
//! What a fake device settles is the *dispatch*, and dispatch is most of what
//! the multimodal path is. Whether the rotation is arithmetically right is
//! settled on a real device in `shiso-metal`'s kernel cases; what cannot be
//! settled there is whether the engine reaches the multimodal entry points when
//! a request carries media, leaves them alone when it does not, and refuses the
//! one mismatch that would otherwise produce a fluent answer about the wrong
//! picture.
//!
//! Every case here runs on every machine. Nothing is skipped and nothing needs a
//! checkpoint: the weights are byte vectors of the right length, because no
//! kernel runs.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use shiso_core::{
    Backend, CoreError, CoreResult, DType, DeviceBuffer, DeviceView, HostBytes, KernelArg,
    KernelRef, LaunchArgs, RawBuffer, TokenId, alloc,
};
use shiso_engine::gdn::GdnGeometry;
use shiso_engine::layers::{Block, LayerWeights, Weights};
use shiso_engine::projection::Weight;
use shiso_engine::shape::{
    Activation, AttnLayer, GdnShape, HybridShape, LayerKind, MediaPositions, NormScheme,
};
use shiso_engine::vision::{PadTokens, VisionGrid, VisionPrefill};
use shiso_engine::{EngineOptions, HybridStack};
use shiso_models::qwen35::Qwen35Kernels;

// ---------------------------------------------------------------------------
// the fake device
// ---------------------------------------------------------------------------

/// A device that hands out byte vectors and writes down what it was asked to
/// dispatch. Nothing computes; the launch log is the whole instrument.
#[derive(Debug, Default)]
struct FakeBackend {
    live: Mutex<HashMap<u64, Vec<u8>>>,
    launches: Mutex<Vec<&'static str>>,
    next_id: AtomicU64,
}

impl FakeBackend {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn launch_mark(&self) -> usize {
        self.launches.lock().len()
    }

    /// The entry points dispatched since `mark`, in order.
    fn launches_since(&self, mark: usize) -> Vec<&'static str> {
        self.launches.lock()[mark..].to_vec()
    }

    fn fail(op: &'static str, detail: String) -> CoreError {
        CoreError::Backend {
            backend: "fake",
            op,
            source: Box::new(std::io::Error::other(detail)),
        }
    }

    /// The host range a view names, checked against the live allocation.
    fn range_of(
        &self,
        view: DeviceView<'_>,
        len: u64,
        op: &'static str,
    ) -> CoreResult<(u64, usize, usize)> {
        let start = usize::try_from(view.offset())
            .map_err(|_| Self::fail(op, "offset too large".into()))?;
        let width = usize::try_from(len).map_err(|_| Self::fail(op, "length too large".into()))?;
        let table = self.live.lock();
        let buffer = table
            .get(&view.id().get())
            .ok_or_else(|| Self::fail(op, format!("no live allocation {}", view.id())))?;
        let end = start
            .checked_add(width)
            .ok_or_else(|| Self::fail(op, "range wraps".into()))?;
        if end > buffer.len() {
            return Err(CoreError::OutOfBounds {
                offset: view.offset(),
                len,
                buffer_bytes: u64::try_from(buffer.len()).unwrap_or(u64::MAX),
            });
        }
        Ok((view.id().get(), start, end))
    }
}

impl Backend for FakeBackend {
    fn backend_name(&self) -> &'static str {
        "fake"
    }

    fn alloc_raw(&self, bytes: u64) -> CoreResult<RawBuffer> {
        let width = usize::try_from(bytes)
            .map_err(|_| Self::fail("alloc_raw", format!("{bytes} bytes do not fit this host")))?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let id = NonZeroU64::new(id)
            .ok_or_else(|| Self::fail("alloc_raw", "id space exhausted".into()))?;
        self.live.lock().insert(id.get(), vec![0_u8; width]);
        Ok(RawBuffer::new(id, bytes))
    }

    fn free_raw(&self, raw: RawBuffer) -> CoreResult<()> {
        if self.live.lock().remove(&raw.id().get()).is_some() {
            return Ok(());
        }
        Err(Self::fail(
            "free_raw",
            format!("{} was already free", raw.id()),
        ))
    }

    fn copy_h2d(&self, dst: DeviceView<'_>, src: &[u8]) -> CoreResult<()> {
        let len = u64::try_from(src.len()).unwrap_or(u64::MAX);
        let (id, start, end) = self.range_of(dst, len, "copy_h2d")?;
        self.live
            .lock()
            .get_mut(&id)
            .expect("the range check found the allocation")[start..end]
            .copy_from_slice(src);
        Ok(())
    }

    fn copy_d2h(&self, src: DeviceView<'_>, dst: &mut [u8]) -> CoreResult<()> {
        let len = u64::try_from(dst.len()).unwrap_or(u64::MAX);
        let (id, start, end) = self.range_of(src, len, "copy_d2h")?;
        let table = self.live.lock();
        dst.copy_from_slice(
            &table
                .get(&id)
                .expect("the range check found the allocation")[start..end],
        );
        Ok(())
    }

    fn copy_d2d(&self, dst: DeviceView<'_>, src: DeviceView<'_>) -> CoreResult<()> {
        if dst.len() != src.len() {
            return Err(CoreError::ShapeMismatch {
                field: "copy_d2d len",
                expected: dst.len(),
                got: src.len(),
            });
        }
        let (from, start, end) = self.range_of(src, src.len(), "copy_d2d")?;
        let copied = {
            let table = self.live.lock();
            table
                .get(&from)
                .expect("the range check found the allocation")[start..end]
                .to_vec()
        };
        let (to, start, end) = self.range_of(dst, dst.len(), "copy_d2d")?;
        self.live
            .lock()
            .get_mut(&to)
            .expect("the range check found the allocation")[start..end]
            .copy_from_slice(&copied);
        Ok(())
    }

    fn run(&self, kernel: KernelRef, args: &LaunchArgs<'_>) -> CoreResult<()> {
        // Every buffer a launch names has to be one the device still holds:
        // that is what makes "the splice was dispatched" a statement about a
        // real range rather than about a name.
        for arg in args.args {
            if let KernelArg::Buffer(view) = arg {
                self.range_of(*view, view.len(), "run")?;
            }
        }
        self.launches.lock().push(kernel.entry);
        Ok(())
    }

    fn sync(&self) -> CoreResult<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// a stack with one attention slot
// ---------------------------------------------------------------------------

/// Residual width. A multiple of four, because the norm strides in `float4`.
const HIDDEN: u32 = 64;
/// Vocabulary, which is also the logits row width.
const VOCAB: u32 = 128;
/// Feed-forward inner width.
const INTER: u32 = 128;
/// Attention head width, and the rotated prefix inside it.
const HEAD_DIM: u32 = 32;
const N_ROT: u32 = 16;
/// The still-image placeholder this fixture renders with.
const IMG: TokenId = TokenId::new(100);
/// The clip placeholder.
const VID: TokenId = TokenId::new(101);

fn shape() -> HybridShape {
    HybridShape {
        // A recurrent slot and an attention slot. The recurrent one reads no
        // positions and adds nothing to the question here, but a stack with no
        // recurrent state has no checkpoint blob and the cache refuses to open
        // over a zero-byte one, so the pattern is the real one's in miniature.
        layers: vec![LayerKind::Gdn, LayerKind::Attn],
        hidden: HIDDEN,
        vocab: VOCAB,
        intermediate: INTER,
        attn: vec![AttnLayer::new(4, 2, HEAD_DIM, N_ROT, N_ROT, 10_000.0)],
        gdn: GdnShape::new(2, 1, HEAD_DIM, 4),
        attn_output_gate: true,
        rms_eps: 1e-6,
        activation: Activation::SwiGlu,
        norms: NormScheme::PreOnly,
        embed_scale: None,
        final_softcap: None,
        layer_scalar: false,
        ple: None,
        media_positions: MediaPositions::Mrope3D,
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
    let host: Arc<dyn HostBytes> = Arc::new(vec![0_u8; bytes]);
    Weight::import(backend, DType::F16, in_dim, out_dim, host, 0..bytes)
        .expect("a fixture weight imports")
}

fn norm(backend: &Arc<dyn Backend>, width: u32) -> DeviceBuffer {
    alloc(backend, u64::from(width) * 4).expect("a fixture norm allocates")
}

/// A layer slot's two norms and its feed-forward, which every slot has.
fn slot(backend: &Arc<dyn Backend>, block: Block, ordinal: u32) -> LayerWeights {
    // Every attention slot here owns its cache, so its run rank is its ordinal;
    // a recurrent slot reads no runs and has none.
    let kv_rank = matches!(block, Block::Attn(_)).then_some(ordinal);
    LayerWeights {
        input_norm: norm(backend, HIDDEN),
        // The pre-only scheme: nothing normalizes a block's output, and the
        // residual sum carries no scalar.
        post_attn_norm: None,
        ffn_norm: Some(norm(backend, HIDDEN)),
        post_ffn_norm: None,
        ffn: Some(shiso_engine::ffn::Ffn {
            gate: Some(weight(backend, HIDDEN, INTER)),
            up: weight(backend, HIDDEN, INTER),
            down: weight(backend, INTER, HIDDEN),
        }),
        moe: None,
        block,
        ordinal,
        ple: None,
        kv_rank,
        layer_scalar: 1.0,
    }
}

fn weights(backend: &Arc<dyn Backend>) -> Weights {
    let shape = shape();
    let gqa = *shape
        .attn
        .first()
        .expect("the fixture has an attention slot");
    let gdn = shape.gdn;
    let q_width = gqa.q_heads * gqa.head_dim * 2;
    let kv_width = gqa.kv_heads * gqa.head_dim;
    let mix = (2 * gdn.qk_heads + gdn.v_heads) * gdn.head_dim;
    let gdn_width = gdn.v_heads * gdn.head_dim;
    let recurrent = Block::Gdn(shiso_engine::gdn::Gdn {
        qkv: weight(backend, HIDDEN, mix),
        gate: weight(backend, HIDDEN, gdn_width),
        alpha: weight(backend, HIDDEN, gdn.v_heads),
        beta: weight(backend, HIDDEN, gdn.v_heads),
        out: weight(backend, gdn_width, HIDDEN),
        conv1d: norm(backend, mix * gdn.conv_kernel),
        a: norm(backend, gdn.v_heads),
        dt_bias: norm(backend, gdn.v_heads),
        norm: norm(backend, gdn.head_dim),
    });
    let attention = Block::Attn(shiso_engine::attn::Attn {
        q: weight(backend, HIDDEN, q_width),
        k: Some(weight(backend, HIDDEN, kv_width)),
        v: Some(weight(backend, HIDDEN, kv_width)),
        output: weight(backend, gqa.q_heads * gqa.head_dim, HIDDEN),
        q_norm: Some(norm(backend, gqa.head_dim)),
        k_norm: Some(norm(backend, gqa.head_dim)),
    });
    Weights {
        token_embd: weight(backend, HIDDEN, VOCAB),
        output_norm: norm(backend, HIDDEN),
        lm_head: None,
        ple_model_proj: None,
        ple_proj_norm: None,
        layers: vec![slot(backend, recurrent, 0), slot(backend, attention, 0)],
    }
}

/// A stack over the fake device, plus the device itself.
fn stack() -> (Arc<FakeBackend>, HybridStack) {
    let fake = FakeBackend::new();
    let backend: Arc<dyn Backend> = fake.clone();
    let weights = weights(&backend);
    // The defaults, unmodified: the option struct is one a crate hands out
    // rather than one another crate fills in, so there is no field-by-field
    // literal to write here.
    let geometry = GdnGeometry::new(&shape().gdn).expect("the fixture recurrent widths fit");
    let kernels =
        Qwen35Kernels::choose(backend.as_ref(), &geometry).expect("a recurrent shape is chosen");
    let stack = HybridStack::new(
        &backend,
        shape(),
        weights,
        DType::Q8_0,
        Arc::new(kernels),
        None,
        EngineOptions::new(),
    )
    .expect("the fixture stack builds");
    (fake, stack)
}

/// An encoder output block of `rows` rows at the residual width.
fn encoder_rows(backend: &Arc<dyn Backend>, rows: u32) -> DeviceBuffer {
    alloc(backend, u64::from(rows) * u64::from(HIDDEN) * 4).expect("an encoder block allocates")
}

fn pads() -> PadTokens {
    PadTokens::new(Some(IMG), Some(VID))
}

/// The prompt shape the renderer produces: text, a placeholder run, text.
fn prompt(run: usize, pad: TokenId) -> Vec<TokenId> {
    let mut tokens = vec![TokenId::new(1), TokenId::new(2)];
    tokens.extend(std::iter::repeat_n(pad, run));
    tokens.push(TokenId::new(3));
    tokens
}

// ---------------------------------------------------------------------------
// the row-count contract
// ---------------------------------------------------------------------------

#[test]
fn an_encoder_block_that_matches_its_items_is_accepted() {
    let fake = FakeBackend::new();
    let backend: Arc<dyn Backend> = fake;
    let grids = [VisionGrid::new(1, 2, 3), VisionGrid::new(2, 2, 2)];
    let rows = 2 * 3 + 2 * 2 * 2;
    let block = encoder_rows(&backend, rows);
    let prefill = VisionPrefill::new(block.view(), &grids, pads(), HIDDEN)
        .expect("a block whose rows match its items");
    assert_eq!(prefill.row_count(), rows);
    assert!(!prefill.is_empty());
}

#[test]
fn an_encoder_block_with_the_wrong_row_count_is_refused() {
    let fake = FakeBackend::new();
    let backend: Arc<dyn Backend> = fake;
    let grids = [VisionGrid::new(1, 2, 3)];
    // One row short of the six the item claims. Every buffer is still a valid
    // buffer; this is the only place the disagreement is visible.
    let block = encoder_rows(&backend, 5);
    match VisionPrefill::new(block.view(), &grids, pads(), HIDDEN) {
        Err(CoreError::ShapeMismatch {
            field,
            expected,
            got,
        }) => {
            assert_eq!(field, "vision embedding rows");
            assert_eq!(expected, 6);
            assert_eq!(got, 5);
        }
        other => panic!("expected a row-count mismatch, got {other:?}"),
    }
}

#[test]
fn media_with_no_placeholder_in_the_vocabulary_is_refused() {
    let fake = FakeBackend::new();
    let backend: Arc<dyn Backend> = fake;
    let grids = [VisionGrid::new(1, 2, 2)];
    let block = encoder_rows(&backend, 4);
    assert!(matches!(
        VisionPrefill::new(block.view(), &grids, PadTokens::default(), HIDDEN),
        Err(CoreError::Unsupported { .. })
    ));
}

#[test]
fn an_empty_item_list_is_an_empty_prefill_rather_than_an_error() {
    let fake = FakeBackend::new();
    let backend: Arc<dyn Backend> = fake;
    let block = encoder_rows(&backend, 0);
    let prefill = VisionPrefill::new(block.view(), &[], PadTokens::default(), HIDDEN)
        .expect("no items and no rows agree");
    assert!(prefill.is_empty());
    assert_eq!(prefill.row_count(), 0);
}

// ---------------------------------------------------------------------------
// which entry points a prefill reaches
// ---------------------------------------------------------------------------

#[test]
fn a_text_only_prefill_takes_the_one_dimensional_rotation_and_no_splice() {
    let (fake, mut stack) = stack();
    let session = stack.open_session().expect("a session opens");
    let mark = fake.launch_mark();
    stack
        .prefill_last(
            session,
            &[TokenId::new(1), TokenId::new(2), TokenId::new(3)],
        )
        .expect("a text prefill runs");
    let seen = fake.launches_since(mark);

    assert!(
        seen.contains(&"kernel_q35_q_norm_rope") && seen.contains(&"kernel_q35_k_norm_rope_kv"),
        "the one-dimensional rotation is what a text prompt dispatches: {seen:?}",
    );
    assert!(
        !seen.iter().any(|entry| entry.ends_with("_mrope")),
        "a text prompt must not reach the multimodal rotation: {seen:?}",
    );
    assert!(
        !seen.contains(&"kernel_vit_splice_rows"),
        "a text prompt has nothing to splice: {seen:?}",
    );
}

#[test]
fn a_prefill_carrying_media_splices_and_takes_the_multimodal_rotation() {
    let (fake, mut stack) = stack();
    let backend: Arc<dyn Backend> = fake.clone();
    let grids = [VisionGrid::new(1, 2, 3)];
    let block = encoder_rows(&backend, 6);
    let media = VisionPrefill::new(block.view(), &grids, pads(), HIDDEN).expect("a matching block");

    let session = stack.open_session().expect("a session opens");
    let mark = fake.launch_mark();
    stack
        .prefill_last_media(session, &prompt(6, IMG), &media)
        .expect("a multimodal prefill runs");
    let seen = fake.launches_since(mark);

    assert!(
        seen.contains(&"kernel_vit_splice_rows"),
        "the encoder rows have to reach the residual stream: {seen:?}",
    );
    assert!(
        seen.contains(&"kernel_q35_q_norm_rope_mrope")
            && seen.contains(&"kernel_q35_k_norm_rope_kv_mrope"),
        "a prompt with media takes the multimodal rotation: {seen:?}",
    );
    assert!(
        !seen.contains(&"kernel_q35_q_norm_rope") && !seen.contains(&"kernel_q35_k_norm_rope_kv"),
        "and takes it instead of the one-dimensional one, not as well as: {seen:?}",
    );
    // The splice runs once per chunk, right after the gather, and before the
    // first norm reads the rows it rewrote.
    let splice = seen
        .iter()
        .position(|entry| *entry == "kernel_vit_splice_rows")
        .expect("the splice was dispatched");
    let first_norm = seen
        .iter()
        .position(|entry| entry.starts_with("kernel_rms_norm_mul"))
        .expect("the first norm was dispatched");
    assert!(
        splice < first_norm,
        "the splice must land before anything reads the rows: {seen:?}",
    );
}

/// An item list that carries no items is the text path, not a degenerate
/// multimodal one. A caller that builds one request shape for every request gets
/// the old launches back by handing over nothing.
#[test]
fn an_empty_prefill_leaves_a_request_on_the_text_path() {
    let (fake, mut stack) = stack();
    let backend: Arc<dyn Backend> = fake.clone();
    let block = encoder_rows(&backend, 0);
    let media = VisionPrefill::new(block.view(), &[], pads(), HIDDEN).expect("an empty prefill");

    let session = stack.open_session().expect("a session opens");
    let mark = fake.launch_mark();
    stack
        .prefill_last_media(session, &[TokenId::new(1), TokenId::new(2)], &media)
        .expect("the prefill runs");
    let seen = fake.launches_since(mark);
    assert!(
        seen.contains(&"kernel_q35_q_norm_rope") && !seen.contains(&"kernel_vit_splice_rows"),
        "no items means the text path, launch for launch: {seen:?}",
    );
}

// ---------------------------------------------------------------------------
// the prompt and the tower have to describe the same request
// ---------------------------------------------------------------------------

/// A prompt whose placeholder run is shorter than the items claim is refused.
///
/// This is the mismatch that the pad expansion at the edge exists to prevent and
/// that nothing downstream can see: the encoder rows are all there, the tokens
/// are all there, and the splice simply never reaches the last of the rows.
#[test]
fn a_prompt_whose_placeholder_run_is_short_is_refused() {
    let (fake, mut stack) = stack();
    let backend: Arc<dyn Backend> = fake.clone();
    let grids = [VisionGrid::new(1, 2, 3)];
    let block = encoder_rows(&backend, 6);
    let media = VisionPrefill::new(block.view(), &grids, pads(), HIDDEN).expect("a matching block");

    let session = stack.open_session().expect("a session opens");
    // One placeholder, as a chat template emits it, with the expansion missing.
    match stack.prefill_last_media(session, &prompt(1, IMG), &media) {
        Err(CoreError::ShapeMismatch {
            field,
            expected,
            got,
        }) => {
            assert_eq!(field, "vision placeholder tokens");
            assert_eq!(expected, 6);
            assert_eq!(got, 0, "a run shorter than the item consumes no row at all");
        }
        other => panic!("expected a placeholder-count mismatch, got {other:?}"),
    }
}

/// Both placeholders are consumed identically. A clip rendered with the clip
/// placeholder must be spliced exactly as a still rendered with the image one,
/// which is the bug that made a video a featureless field.
#[test]
fn a_clip_placeholder_is_spliced_like_an_image_one() {
    let (fake, mut stack) = stack();
    let backend: Arc<dyn Backend> = fake.clone();
    let grids = [VisionGrid::new(3, 2, 2)];
    let block = encoder_rows(&backend, 12);
    let media = VisionPrefill::new(block.view(), &grids, pads(), HIDDEN).expect("a matching block");

    let session = stack.open_session().expect("a session opens");
    let mark = fake.launch_mark();
    stack
        .prefill_last_media(session, &prompt(12, VID), &media)
        .expect("a clip prefill runs");
    let seen = fake.launches_since(mark);
    assert!(
        seen.contains(&"kernel_vit_splice_rows"),
        "the clip's rows reached the residual stream: {seen:?}",
    );
}

/// A prompt wider than the engine's own chunk is split internally, and the split
/// has to be invisible: the item straddling it keeps its base position and its
/// encoder rows, so the completed prefill still accounts for every row.
#[test]
fn a_prompt_wider_than_one_chunk_still_accounts_for_every_row() {
    let (fake, mut stack) = stack();
    let backend: Arc<dyn Backend> = fake.clone();
    // 20 by 20 merged positions is 400 placeholders; with the surrounding text
    // the prompt is under the context but the run straddles a chunk boundary
    // only if the chunk is narrower - so the case is written at a width that
    // exceeds the chunk whatever it is set to.
    let side = 24;
    let grids = [VisionGrid::new(1, side, side)];
    let rows = side * side;
    let block = encoder_rows(&backend, rows);
    let media = VisionPrefill::new(block.view(), &grids, pads(), HIDDEN).expect("a matching block");

    let session = stack.open_session().expect("a session opens");
    stack
        .prefill_last_media(session, &prompt(rows as usize, IMG), &media)
        .expect("a prefill wider than one chunk runs and accounts for every row");
}
