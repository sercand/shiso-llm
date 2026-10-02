//! Reference image Key-Value caching for multi-reference editing.
//!
//! Step 0 runs the reference tokens through the blocks and keeps their
//! post-RoPE fused qkv rows; steps 1.. reuse them as extra attention keys and
//! values without re-running projection or MLP on the reference tokens. The
//! cache therefore holds *device* buffers - a host vector could not be read
//! by the attention kernel - and keeps the fused row width so the sdpa kernel
//! reads K and V out of the same stride it read the live tokens from.

use shiso_core::DeviceBuffer;

/// Per-layer cached fused qkv rows for reference image tokens.
#[derive(Debug, Default)]
pub(crate) struct LayerKvCache {
    /// Cached post-RoPE qkv rows: (`num_ref_tokens`, `row_stride`).
    pub kv_ref: Option<DeviceBuffer>,
    /// Elements per cached row (the block's fused qkv row width).
    pub row_stride: usize,
}

/// Complete KV cache container holding double and single stream block caches.
#[derive(Debug)]
pub(crate) struct Flux2KvCache {
    /// Caches for double-stream blocks.
    pub double_caches: Vec<LayerKvCache>,
    /// Caches for single-stream blocks.
    pub single_caches: Vec<LayerKvCache>,
    /// Number of reference image tokens cached.
    pub num_ref_tokens: usize,
}

impl Flux2KvCache {
    /// Creates an empty cache for the specified layer counts.
    #[must_use]
    pub(crate) fn new(num_double: usize, num_single: usize) -> Self {
        Self {
            double_caches: (0..num_double).map(|_| LayerKvCache::default()).collect(),
            single_caches: (0..num_single).map(|_| LayerKvCache::default()).collect(),
            num_ref_tokens: 0,
        }
    }

    /// Clears cached buffers.
    #[expect(
        dead_code,
        reason = "the reference-image KV path is implemented per plan; no wire format carries image input yet, so nothing clears a cache mid-session"
    )]
    pub(crate) fn clear(&mut self) {
        for c in &mut self.double_caches {
            c.kv_ref = None;
            c.row_stride = 0;
        }
        for c in &mut self.single_caches {
            c.kv_ref = None;
            c.row_stride = 0;
        }
        self.num_ref_tokens = 0;
    }
}
