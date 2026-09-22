//! A model on a device that is a `HashMap`, built from a checkpoint this file writes.
//!
//! Two test binaries need the same thing before they can ask their own question: a
//! loaded model that runs on every machine, with no checkpoint on disk to find and no
//! GPU to talk to. The device under it records what it was asked to do instead of
//! doing it, and the buffers it hands out are byte vectors a test can read and write
//! directly - which is what lets byte-equality be a meaningful assertion about state
//! that would otherwise be unreachable.
//!
//! # Why a fake device can prove anything at all
//!
//! A no-op `run` writes no logits, so nothing here can check a number a kernel would
//! have produced. What it can check is the *dispatch*: the device records every buffer
//! range each launch was handed, so the row a step wrote is the offset the projection
//! was pointed at, and the session a step ran against is whose recurrent state the
//! launches before it touched. It can also check every byte the host itself moves,
//! which is the whole of the checkpoint path.
//!
//! # Why the model is tiny
//!
//! The geometry is the real one's shape with every width cut down: four layer slots, a
//! 64-wide residual stream, a 128-token vocabulary. The layer *pattern* is kept -
//! three recurrent slots then an attention slot, which is the interval the binding
//! cross-checks - because that pattern is what the ordinals and the cache's per-layer
//! runs are derived from. The widths are not, so the whole fixture is a few hundred
//! kilobytes and the tests are instant.
#![allow(
    dead_code,
    reason = "the two binaries that include this fixture ask different questions of it; an item one of them does not need is shared plumbing, not dead code"
)]

use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::num::NonZeroU64;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use tempfile::TempDir;

use shiso_core::{
    Backend, CoreError, CoreResult, DeviceView, HfConfig, KernelArg, KernelRef, LaunchArgs,
    RawBuffer, TokenId,
};
use shiso_engine::EngineOptions;
use shiso_engine::shape::{AttnLayer, LayerKind};
use shiso_models::qwen35::{Qwen35Config, Qwen35Model};

// ---------------------------------------------------------------------------
// the fake device
// ---------------------------------------------------------------------------

/// One buffer range a launch was handed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Span {
    pub(crate) id: u64,
    pub(crate) offset: u64,
    pub(crate) len: u64,
}

/// One recorded launch: what was dispatched and over which ranges.
///
/// The scalar and by-value arguments are dropped on purpose. What this test
/// argues about is which memory a launch was pointed at, and a kernel's argument
/// block is a statement about widths that the engine's own unit tests already
/// check.
#[derive(Debug, Clone)]
pub(crate) struct Launch {
    pub(crate) entry: &'static str,
    pub(crate) spans: Vec<Span>,
}

impl Launch {
    /// The range this launch was handed inside `buffer`, if any.
    pub(crate) fn span_in(&self, buffer: u64) -> Option<Span> {
        self.spans.iter().copied().find(|span| span.id == buffer)
    }

    /// Whether this launch touched any of `buffers`.
    pub(crate) fn touches_any(&self, buffers: &HashSet<u64>) -> bool {
        self.spans.iter().any(|span| buffers.contains(&span.id))
    }
}

/// A device that is a `HashMap` of byte vectors and a log of what it was asked.
///
/// Allocation and release are recorded as ordered lists rather than as counters,
/// because the two questions this test asks about memory are both about order:
/// which buffers one call created, and whether teardown released them from the
/// top down.
#[derive(Debug, Default)]
pub(crate) struct FakeBackend {
    live: Mutex<HashMap<u64, Vec<u8>>>,
    allocated: Mutex<Vec<u64>>,
    /// Bytes each allocation asked for, in allocation order and beside
    /// `allocated`. The width alone, because the question it answers is whether
    /// anything asked for a buffer no launch could touch.
    sizes: Mutex<Vec<u64>>,
    released: Mutex<Vec<u64>>,
    launches: Mutex<Vec<Launch>>,
    next_id: AtomicU64,
    double_frees: AtomicU64,
}

impl FakeBackend {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// How many allocations have been made, which is the mark a caller takes
    /// before a call it wants to attribute buffers to.
    pub(crate) fn allocation_mark(&self) -> usize {
        self.allocated.lock().len()
    }

    /// The ids allocated since `mark`, in allocation order.
    pub(crate) fn allocated_since(&self, mark: usize) -> Vec<u64> {
        self.allocated.lock()[mark..].to_vec()
    }

    /// The size every allocation asked for, in allocation order.
    pub(crate) fn allocation_sizes(&self) -> Vec<u64> {
        self.sizes.lock().clone()
    }

    /// How many launches have been recorded.
    pub(crate) fn launch_mark(&self) -> usize {
        self.launches.lock().len()
    }

    /// The launches recorded since `mark`, in dispatch order.
    pub(crate) fn launches_since(&self, mark: usize) -> Vec<Launch> {
        self.launches.lock()[mark..].to_vec()
    }

    /// Whether `id` is still an allocation the device holds.
    pub(crate) fn is_live(&self, id: u64) -> bool {
        self.live.lock().contains_key(&id)
    }

    /// How many allocations the device is still holding.
    pub(crate) fn live_count(&self) -> usize {
        self.live.lock().len()
    }

    /// Where `id` sits in the sequence of releases.
    ///
    /// Panics rather than returning an option: every caller is asserting an
    /// ordering, and an id that was never released has no place in one.
    pub(crate) fn release_rank(&self, id: u64) -> usize {
        self.released
            .lock()
            .iter()
            .position(|freed| *freed == id)
            .unwrap_or_else(|| panic!("buffer {id} was never released"))
    }

    pub(crate) fn double_frees(&self) -> u64 {
        self.double_frees.load(Ordering::Relaxed)
    }

    pub(crate) fn fail(op: &'static str, detail: String) -> CoreError {
        CoreError::Backend {
            backend: "fake",
            op,
            source: Box::new(std::io::Error::other(detail)),
        }
    }

    /// The host range one view names, checked against the live allocation.
    pub(crate) fn range_of(
        &self,
        view: DeviceView<'_>,
        len: u64,
        op: &'static str,
    ) -> CoreResult<Range<usize>> {
        if len != view.len() {
            return Err(CoreError::OutOfBounds {
                offset: view.offset(),
                len,
                buffer_bytes: view.len(),
            });
        }
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
        Ok(start..end)
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
        self.allocated.lock().push(id.get());
        self.sizes.lock().push(bytes);
        Ok(RawBuffer::new(id, bytes))
    }

    fn free_raw(&self, raw: RawBuffer) -> CoreResult<()> {
        if self.live.lock().remove(&raw.id().get()).is_some() {
            self.released.lock().push(raw.id().get());
            return Ok(());
        }
        self.double_frees.fetch_add(1, Ordering::Relaxed);
        Err(Self::fail(
            "free_raw",
            format!("{} was already free", raw.id()),
        ))
    }

    fn copy_h2d(&self, dst: DeviceView<'_>, src: &[u8]) -> CoreResult<()> {
        let len = u64::try_from(src.len()).unwrap_or(u64::MAX);
        let range = self.range_of(dst, len, "copy_h2d")?;
        self.live
            .lock()
            .get_mut(&dst.id().get())
            .expect("the range check found the allocation")
            .splice(range, src.iter().copied());
        Ok(())
    }

    fn copy_d2h(&self, src: DeviceView<'_>, dst: &mut [u8]) -> CoreResult<()> {
        let len = u64::try_from(dst.len()).unwrap_or(u64::MAX);
        let range = self.range_of(src, len, "copy_d2h")?;
        let table = self.live.lock();
        let buffer = table
            .get(&src.id().get())
            .expect("the range check found the allocation");
        dst.copy_from_slice(&buffer[range]);
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
        let source = self.range_of(src, src.len(), "copy_d2d")?;
        let copied = {
            let table = self.live.lock();
            table
                .get(&src.id().get())
                .expect("the range check found the allocation")[source]
                .to_vec()
        };
        let target = self.range_of(dst, dst.len(), "copy_d2d")?;
        self.live
            .lock()
            .get_mut(&dst.id().get())
            .expect("the range check found the allocation")
            .splice(target, copied);
        Ok(())
    }

    fn run(&self, kernel: KernelRef, args: &LaunchArgs<'_>) -> CoreResult<()> {
        let spans = args
            .args
            .iter()
            .filter_map(|arg| match arg {
                KernelArg::Buffer(view) => Some(Span {
                    id: view.id().get(),
                    offset: view.offset(),
                    len: view.len(),
                }),
                _ => None,
            })
            .collect();
        self.launches.lock().push(Launch {
            entry: kernel.entry,
            spans,
        });
        Ok(())
    }

    fn sync(&self) -> CoreResult<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// the synthetic checkpoint
// ---------------------------------------------------------------------------

/// The shape fixture that ships with the workspace, which is the only way in to a
/// profile: the shape struct is sealed against outside construction on purpose,
/// so a test builds one the way a server does and then narrows it.
pub(crate) fn fixture_config() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/loader/qwen35-config.json")
}

/// Layer slots the fixture carries.
pub(crate) const N_LAYERS: u32 = 4;
/// Slots between attention slots, which puts attention in the last one.
pub(crate) const INTERVAL: u32 = 4;
/// Residual-stream width. A multiple of four, because the norm strides in
/// `float4`.
pub(crate) const HIDDEN: u32 = 64;
/// Vocabulary, which is also the height of the embedding table.
pub(crate) const VOCAB: u32 = 128;
/// Feed-forward inner width.
pub(crate) const INTERMEDIATE: u32 = 128;
/// Attention query heads.
pub(crate) const Q_HEADS: u32 = 2;
/// Attention key/value heads.
pub(crate) const KV_HEADS: u32 = 1;
/// Attention head width.
pub(crate) const ATTN_HEAD: u32 = 32;
/// Rotated prefix of an attention head.
pub(crate) const N_ROT: u32 = 16;
/// Recurrent value heads.
pub(crate) const V_HEADS: u32 = 2;
/// Recurrent query/key heads.
pub(crate) const QK_HEADS: u32 = 1;
/// Recurrent head width.
pub(crate) const GDN_HEAD: u32 = 32;
/// Convolution window, one more than the state the session carries.
pub(crate) const CONV_KERNEL: u32 = 4;
/// Recurrent slots, which is how many states a session allocates.
pub(crate) const RECURRENT_SLOTS: usize = 3;

/// Tokens a session may hold in these tests.
pub(crate) const CTX: u32 = 64;
/// Pages the pool is built with: enough that no test evicts.
pub(crate) const POOL_PAGES: u32 = 8;
/// Checkpoint slots, likewise.
pub(crate) const CKPT_SLOTS: u32 = 4;

/// Bytes of one logits row.
pub(crate) const LOGITS_ROW_BYTES: u64 = VOCAB as u64 * 4;

pub(crate) const MAGIC: u32 = 0x4655_4747;
pub(crate) const FORMAT_VERSION: u32 = 3;
pub(crate) const ALIGN: usize = 32;
/// Storage format ids, as the container spells them.
pub(crate) const T_F32: u32 = 0;
pub(crate) const T_F16: u32 = 1;
pub(crate) const T_Q8_0: u32 = 8;

/// The options every test loads with.
///
/// Built from the default and narrowed field by field, because the struct is
/// sealed: a test cannot state the whole set, and asking it to would mean a new
/// field defaulted to nothing rather than to the value the server would get.
pub(crate) fn options() -> EngineOptions {
    let mut opts = EngineOptions::default();
    opts.ctx = CTX;
    opts.kv_pool_pages = POOL_PAGES;
    opts.kv_ckpt_slots = CKPT_SLOTS;
    opts
}

/// The real profile, narrowed to a geometry a synthetic file can carry.
///
/// The config parser checks every width against the one this build was written
/// for, so a small `config.json` would be refused before it produced a shape. The
/// full config is therefore parsed and then cut down through the shape's own
/// public fields, which keeps the one thing the binding cross-checks - the layer
/// pattern - stated the same way the parser states it.
pub(crate) fn tiny_profile() -> Qwen35Config {
    let config = HfConfig::from_file(&fixture_config()).expect("the fixture config parses");
    let full = Qwen35Config::from_hf(&config).expect("the fixture config has a shape");
    let mut shape = full.shape().clone();
    shape.hidden = HIDDEN;
    shape.vocab = VOCAB;
    shape.intermediate = INTERMEDIATE;
    shape.layers = (0..N_LAYERS)
        .map(|slot| {
            if (slot + 1).is_multiple_of(INTERVAL) {
                LayerKind::Attn
            } else {
                LayerKind::Gdn
            }
        })
        .collect();
    // One descriptor per attention slot, keeping the rotary base the real
    // config states: the slots of this family are all the same slot.
    let rope_theta = shape.attn.first().map_or(1.0e7, |layer| layer.rope_theta);
    shape.attn = vec![
        AttnLayer::new(Q_HEADS, KV_HEADS, ATTN_HEAD, N_ROT, N_ROT, rope_theta);
        usize::try_from(shape.count(LayerKind::Attn)).unwrap_or(0)
    ];
    shape.gdn.v_heads = V_HEADS;
    shape.gdn.qk_heads = QK_HEADS;
    shape.gdn.head_dim = GDN_HEAD;
    shape.gdn.conv_kernel = CONV_KERNEL;
    Qwen35Config::new(shape, full.eos_token(), INTERVAL)
}

/// One tensor directory entry.
pub(crate) struct Spec {
    name: String,
    dims: Vec<u64>,
    type_id: u32,
}

pub(crate) fn spec(name: String, dims: &[u64], type_id: u32) -> Spec {
    Spec {
        name,
        dims: dims.to_vec(),
        type_id,
    }
}

/// Payload size of one tensor, computed here rather than imported so the file the
/// reader is handed is described by a second statement of the same geometry.
pub(crate) fn payload_bytes(type_id: u32, dims: &[u64]) -> u64 {
    let elems: u64 = dims.iter().product();
    let (block_elems, block_bytes) = match type_id {
        T_F32 => (1, 4),
        T_F16 => (1, 2),
        T_Q8_0 => (32, 34),
        other => panic!("the fixture writes no tensors of format {other}"),
    };
    elems.div_ceil(block_elems) * block_bytes
}

pub(crate) fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn put_str(out: &mut Vec<u8>, text: &str) {
    put_u64(out, u64::try_from(text.len()).expect("a fixture name fits"));
    out.extend_from_slice(text.as_bytes());
}

/// Every tensor the binding asks a four-slot dense-hybrid model for.
///
/// The widths are derived from the constants above rather than written out, so a
/// fixture that disagrees with the profile is a compile-time edit in one place
/// instead of a load error nobody can attribute.
pub(crate) fn fixture_tensors() -> Vec<Spec> {
    let hidden = u64::from(HIDDEN);
    let inter = u64::from(INTERMEDIATE);
    let qk_width = u64::from(QK_HEADS) * u64::from(GDN_HEAD);
    let v_width = u64::from(V_HEADS) * u64::from(GDN_HEAD);
    let mix = 2 * qk_width + v_width;
    let fused_q = 2 * u64::from(Q_HEADS) * u64::from(ATTN_HEAD);
    let kv_width = u64::from(KV_HEADS) * u64::from(ATTN_HEAD);
    let attn_width = u64::from(Q_HEADS) * u64::from(ATTN_HEAD);

    // The embedding is the one tensor whose format is not free: the row gather has
    // an entry point for three quantized formats and for nothing else, and the
    // output projection is this same table read the other way round.
    let mut out = vec![
        spec(
            "token_embd.weight".to_owned(),
            &[hidden, u64::from(VOCAB)],
            T_Q8_0,
        ),
        spec("output_norm.weight".to_owned(), &[hidden], T_F32),
    ];
    for slot in 0..N_LAYERS {
        let at = |suffix: &str| format!("blk.{slot}.{suffix}");
        out.push(spec(at("attn_norm.weight"), &[hidden], T_F32));
        out.push(spec(at("post_attention_norm.weight"), &[hidden], T_F32));
        out.push(spec(at("ffn_gate.weight"), &[hidden, inter], T_F16));
        out.push(spec(at("ffn_up.weight"), &[hidden, inter], T_F16));
        out.push(spec(at("ffn_down.weight"), &[inter, hidden], T_F16));
        if (slot + 1).is_multiple_of(INTERVAL) {
            out.push(spec(at("attn_q.weight"), &[hidden, fused_q], T_F16));
            out.push(spec(at("attn_k.weight"), &[hidden, kv_width], T_F16));
            out.push(spec(at("attn_v.weight"), &[hidden, kv_width], T_F16));
            out.push(spec(
                at("attn_q_norm.weight"),
                &[u64::from(ATTN_HEAD)],
                T_F32,
            ));
            out.push(spec(
                at("attn_k_norm.weight"),
                &[u64::from(ATTN_HEAD)],
                T_F32,
            ));
            out.push(spec(at("attn_output.weight"), &[attn_width, hidden], T_F16));
        } else {
            out.push(spec(at("attn_qkv.weight"), &[hidden, mix], T_F16));
            out.push(spec(at("attn_gate.weight"), &[hidden, v_width], T_F16));
            out.push(spec(
                at("ssm_conv1d.weight"),
                &[u64::from(CONV_KERNEL), mix],
                T_F32,
            ));
            out.push(spec(at("ssm_a"), &[u64::from(V_HEADS)], T_F32));
            out.push(spec(
                at("ssm_alpha.weight"),
                &[hidden, u64::from(V_HEADS)],
                T_F16,
            ));
            out.push(spec(
                at("ssm_beta.weight"),
                &[hidden, u64::from(V_HEADS)],
                T_F16,
            ));
            out.push(spec(at("ssm_dt.bias"), &[u64::from(V_HEADS)], T_F32));
            out.push(spec(at("ssm_norm.weight"), &[u64::from(GDN_HEAD)], T_F32));
            out.push(spec(at("ssm_out.weight"), &[v_width, hidden], T_F16));
        }
    }
    out
}

/// Writes a checkpoint into `dir` and returns its path.
///
/// The payload region is sized rather than written: nothing in this test reads a
/// weight byte, and a file the loader only has to find inside costs kilobytes
/// instead of the sum of its tensors.
pub(crate) fn write_fixture(dir: &TempDir) -> PathBuf {
    let tensors = fixture_tensors();
    let mut out = Vec::new();
    put_u32(&mut out, MAGIC);
    put_u32(&mut out, FORMAT_VERSION);
    put_u64(&mut out, u64::try_from(tensors.len()).expect("count fits"));
    put_u64(&mut out, 3);

    put_str(&mut out, "general.architecture");
    put_u32(&mut out, 8);
    put_str(&mut out, "qwen35");
    put_str(&mut out, "qwen35.block_count");
    put_u32(&mut out, 4);
    put_u32(&mut out, N_LAYERS);
    put_str(&mut out, "qwen35.full_attention_interval");
    put_u32(&mut out, 4);
    put_u32(&mut out, INTERVAL);

    let mut payload = 0_u64;
    for tensor in &tensors {
        put_str(&mut out, &tensor.name);
        put_u32(
            &mut out,
            u32::try_from(tensor.dims.len()).expect("rank fits"),
        );
        for dim in &tensor.dims {
            put_u64(&mut out, *dim);
        }
        put_u32(&mut out, tensor.type_id);
        put_u64(&mut out, payload);
        payload += payload_bytes(tensor.type_id, &tensor.dims);
    }
    while out.len() % ALIGN != 0 {
        out.push(0);
    }

    let path = dir.path().join("fixture.gguf");
    let mut file = std::fs::File::create(&path).expect("the fixture file is created");
    file.write_all(&out).expect("the fixture header is written");
    let len = u64::try_from(out.len()).expect("header fits") + payload;
    file.set_len(len).expect("the fixture payload is sized");
    file.flush().expect("the fixture is flushed");
    path
}

/// A loaded model over a fresh fake device.
///
/// The temporary directory comes back with it: the mapping the weights were
/// imported from lives as long as the buffers that name it, so a fixture that
/// deleted itself at the end of this function would pull the file out from under
/// the model.
pub(crate) fn loaded() -> (TempDir, Arc<FakeBackend>, Qwen35Model) {
    loaded_with(options())
}

/// The same, at options a test chose.
pub(crate) fn loaded_with(opts: EngineOptions) -> (TempDir, Arc<FakeBackend>, Qwen35Model) {
    let dir = TempDir::new().expect("a temporary directory");
    let path = write_fixture(&dir);
    let profile = tiny_profile();
    let fake = FakeBackend::new();
    let backend: Arc<dyn Backend> = Arc::<FakeBackend>::clone(&fake);
    let model =
        Qwen35Model::load(&backend, &path, &profile, opts).expect("the synthetic checkpoint loads");
    (dir, fake, model)
}

/// `count` distinct token ids, which is what makes two sessions' paths differ.
pub(crate) fn tokens(first: u32, count: u32) -> Vec<TokenId> {
    (0..count)
        .map(|i| TokenId::new((first + i) % VOCAB))
        .collect()
}

impl FakeBackend {
    /// Overwrites a live allocation with a seeded pseudo-random pattern and hands the
    /// bytes back.
    ///
    /// The checkpoint path is a byte image: a buffer full of zeros would round-trip
    /// through it perfectly whether or not a single byte was actually carried. Filling
    /// the source with a pattern nothing else could produce is what makes
    /// "byte-equal after a resume" a statement about the payload rather than about
    /// zero.
    pub(crate) fn scribble(&self, id: u64, seed: u64) -> Vec<u8> {
        let mut table = self.live.lock();
        let buffer = table.get_mut(&id).expect("the buffer is live");
        let mut state = seed | 1;
        for byte in buffer.iter_mut() {
            // A plain xorshift: deterministic, cheap, and nothing like a run of zeros.
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = u8::try_from(state & 0xff).unwrap_or(0);
        }
        buffer.clone()
    }

    /// The current contents of a live allocation.
    pub(crate) fn contents(&self, id: u64) -> Vec<u8> {
        self.live
            .lock()
            .get(&id)
            .expect("the buffer is live")
            .clone()
    }

    /// The ids of every allocation the device still holds, in allocation order.
    pub(crate) fn live_ids(&self) -> Vec<u64> {
        let live = self.live.lock();
        self.allocated
            .lock()
            .iter()
            .copied()
            .filter(|id| live.contains_key(id))
            .collect()
    }
}
