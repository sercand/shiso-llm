//! Oracle tests for the fused-row attention and `RoPE` offsets.
//!
//! The transformer binds one fused qkv buffer as q, k and v (and as the q and
//! k of `RoPE`) with only the section offsets telling the head sections apart,
//! the exact seam a host twin can get wrong while every shape still checks
//! out: reading k or v from the q offset silently computes `softmax(q q^T) q`,
//! and rotating the q section twice leaves k unrotated. Both failures produce
//! plausible image-shaped numbers, so these tests pin the arithmetic itself: a
//! tiny fused-qkv buffer goes through the host twins of the transformer's own
//! call shapes (called directly - the production `Ops` path launches kernels
//! and never falls back), and the result is compared against naive
//! per-element reference math written here from the upstream formulas. Each
//! test also asserts its data discriminates: the buggy reading and the
//! correct one must differ, so a future simplification of the expected values
//! cannot quietly make the test vacuous.

#![expect(
    clippy::cast_precision_loss,
    reason = "test dimensions are single-digit element counts, exact in f32"
)]

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use shiso_core::{Backend, CoreError, CoreResult, DeviceView, KernelRef, LaunchArgs, RawBuffer};

use super::rope::rope_4d_freqs;
use crate::diffusion::ops::{Ops, Rope4dCall, SdpaCall};

/// Two heads of eight dims: the smallest head that the 4-axis interleaved
/// `RoPE` can rotate at all (one frequency pair per axis), so a wrong
/// section offset moves real values.
const HEADS: usize = 2;
const HEAD_DIM: usize = 8;
/// `heads * head_dim`: the width of one q/k/v section inside a fused row.
const HIDDEN: usize = HEADS * HEAD_DIM;
/// A fused qkv row: q, then k, then v.
const FUSED: usize = 3 * HIDDEN;
/// One frequency per axis (axis dim 2), so the table is four entries and the
/// rotation of every head covers all eight dims exactly once.
const AXES_DIM: [usize; 4] = [2, 2, 2, 2];
const THETA: f32 = 2000.0;

/// A device that is a table of byte vectors with working copies: the host
/// fallback downloads and writes back through them, and its bounds checks
/// turn an operand over-read into a test failure.
#[derive(Debug, Default)]
struct TableBackend {
    allocations: Mutex<HashMap<u64, Vec<u8>>>,
    next_id: AtomicU64,
}

impl TableBackend {
    fn fail(op: &'static str, detail: &str) -> CoreError {
        CoreError::Backend {
            backend: "table",
            op,
            source: Box::new(std::io::Error::other(detail.to_owned())),
        }
    }

    fn range_of(view: DeviceView<'_>, len: u64) -> CoreResult<Range<usize>> {
        let start =
            usize::try_from(view.offset()).map_err(|_| Self::fail("range", "offset too large"))?;
        let width = usize::try_from(len).map_err(|_| Self::fail("range", "length too large"))?;
        Ok(start..start.saturating_add(width))
    }

    fn live(
        table: &Mutex<HashMap<u64, Vec<u8>>>,
        id: u64,
        op: &'static str,
    ) -> CoreResult<Vec<u8>> {
        table
            .lock()
            .get(&id)
            .cloned()
            .ok_or_else(|| Self::fail(op, "no live allocation"))
    }
}

impl Backend for TableBackend {
    fn backend_name(&self) -> &'static str {
        "table"
    }

    fn alloc_raw(&self, bytes: u64) -> CoreResult<RawBuffer> {
        let width =
            usize::try_from(bytes).map_err(|_| Self::fail("alloc_raw", "the size does not fit"))?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let id = NonZeroU64::new(id).ok_or_else(|| Self::fail("alloc_raw", "id space empty"))?;
        self.allocations.lock().insert(id.get(), vec![0u8; width]);
        Ok(RawBuffer::new(id, bytes))
    }

    fn free_raw(&self, raw: RawBuffer) -> CoreResult<()> {
        self.allocations.lock().remove(&raw.id().get());
        Ok(())
    }

    fn copy_h2d(&self, dst: DeviceView<'_>, src: &[u8]) -> CoreResult<()> {
        let len = u64::try_from(src.len()).unwrap_or(u64::MAX);
        let range = Self::range_of(dst, len)?;
        let mut table = self.allocations.lock();
        let buffer = table
            .get_mut(&dst.id().get())
            .ok_or_else(|| Self::fail("copy_h2d", "no live allocation"))?;
        if range.end > buffer.len() {
            return Err(CoreError::OutOfBounds {
                offset: dst.offset(),
                len,
                buffer_bytes: u64::try_from(buffer.len()).unwrap_or(u64::MAX),
            });
        }
        buffer.splice(range, src.iter().copied());
        Ok(())
    }

    fn copy_d2h(&self, src: DeviceView<'_>, dst: &mut [u8]) -> CoreResult<()> {
        let len = u64::try_from(dst.len()).unwrap_or(u64::MAX);
        let range = Self::range_of(src, len)?;
        let buffer = Self::live(&self.allocations, src.id().get(), "copy_d2h")?;
        let Some(bytes) = buffer.get(range) else {
            return Err(CoreError::OutOfBounds {
                offset: src.offset(),
                len,
                buffer_bytes: u64::try_from(buffer.len()).unwrap_or(u64::MAX),
            });
        };
        dst.copy_from_slice(bytes);
        Ok(())
    }

    fn copy_d2d(&self, dst: DeviceView<'_>, src: DeviceView<'_>) -> CoreResult<()> {
        if dst.len() != src.len() {
            return Err(Self::fail("copy_d2d", "mismatched lengths"));
        }
        let src_range = Self::range_of(src, src.len())?;
        let dst_range = Self::range_of(dst, dst.len())?;
        let bytes = Self::live(&self.allocations, src.id().get(), "copy_d2d")?;
        let Some(bytes) = bytes.get(src_range).map(<[u8]>::to_vec) else {
            return Err(Self::fail("copy_d2d", "source range out of bounds"));
        };
        let mut table = self.allocations.lock();
        let buffer = table
            .get_mut(&dst.id().get())
            .ok_or_else(|| Self::fail("copy_d2d", "no live allocation"))?;
        if dst_range.end > buffer.len() {
            return Err(CoreError::OutOfBounds {
                offset: dst.offset(),
                len: dst.len(),
                buffer_bytes: u64::try_from(buffer.len()).unwrap_or(u64::MAX),
            });
        }
        buffer.splice(dst_range, bytes);
        Ok(())
    }

    fn run(&self, _kernel: KernelRef, _args: &LaunchArgs<'_>) -> CoreResult<()> {
        Err(Self::fail("run", "the table device never launches"))
    }

    fn sync(&self) -> CoreResult<()> {
        Ok(())
    }
}

/// A driver on a device without flux2 kernels, for the host twins only: the
/// twins download and write back through the table's byte vectors, and its
/// bounds checks turn an operand over-read into a test failure. Production
/// cannot build this driver - `Ops::new` refuses a device without the set.
fn host_ops() -> Ops {
    let backend: Arc<dyn Backend> = Arc::new(TableBackend::default());
    Ops::for_host_twins(&backend).expect("the table device backs the host twins")
}

/// Distinct deterministic values, so a section read from the wrong offset
/// lands on different numbers instead of a coincidental match.
fn fused_values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let step = (i * 37 + seed * 11) % 101;
            f32::from(u16::try_from(step).unwrap_or(u16::MAX)) * 0.125 - 6.25
        })
        .collect()
}

/// Naive interleaved-pair rotation of the listed row sections, written from
/// the reference's rotary embedding: pair `j` of axis `a` turns by
/// `pos_a * freq_a` (one frequency per axis in this geometry).
fn expect_rope(
    x: &[f32],
    positions: &[f32],
    freqs: &[f32],
    tokens: usize,
    sections: &[usize],
) -> Vec<f32> {
    let mut y = x.to_vec();
    for &section in sections {
        for t in 0..tokens {
            for h in 0..HEADS {
                let row = t * FUSED + section + h * HEAD_DIM;
                for a in 0..4 {
                    let angle = positions[t * 4 + a] * freqs[a];
                    let (cos, sin) = (angle.cos(), angle.sin());
                    let i0 = row + a * 2;
                    let (x0, x1) = (y[i0], y[i0 + 1]);
                    y[i0] = x0 * cos - x1 * sin;
                    y[i0 + 1] = x0 * sin + x1 * cos;
                }
            }
        }
    }
    y
}

/// Naive joint attention over fused rows: every operand reads its own
/// section, softmax over all joint keys, scale `1 / sqrt(head_dim)`, output
/// packed `(heads * head_dim)` per row.
fn expect_sdpa(rows: &[&[f32]], k_sections: &[usize], v_sections: &[usize]) -> Vec<f32> {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut out = vec![0f32; rows.len() * HIDDEN];
    for h in 0..HEADS {
        for (t, qrow) in rows.iter().enumerate() {
            let q = HEAD_DIM * h;
            let mut scores = Vec::with_capacity(rows.len());
            for (s, srow) in rows.iter().enumerate() {
                let mut dot = 0f32;
                for d in 0..HEAD_DIM {
                    dot += qrow[q + d] * srow[k_sections[s] + HEAD_DIM * h + d];
                }
                scores.push(dot * scale);
            }
            let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0f32;
            for v in &mut scores {
                *v = (*v - max).exp();
                sum += *v;
            }
            for (s, &p) in scores.iter().enumerate() {
                for d in 0..HEAD_DIM {
                    out[t * HIDDEN + HEAD_DIM * h + d] +=
                        p * rows[s][v_sections[s] + HEAD_DIM * h + d];
                }
            }
            for d in 0..HEAD_DIM {
                out[t * HIDDEN + HEAD_DIM * h + d] /= sum;
            }
        }
    }
    out
}

fn assert_close(got: &[f32], expected: &[f32], what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: length");
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert!(
            (g - e).abs() <= 1e-4 * e.abs().max(1.0),
            "{what}: element {i} got {g}, expected {e}"
        );
    }
}

/// The two candidate readings must differ, or the test proves nothing.
fn assert_differs(a: &[f32], b: &[f32], what: &str) {
    let discriminates = a
        .iter()
        .zip(b)
        .any(|(&x, &y)| (x - y).abs() > 1e-3 * y.abs().max(1.0));
    assert!(
        discriminates,
        "{what}: test data does not distinguish the two readings"
    );
}

/// Critical 2: on the transformer's call shape (`q == k ==` the fused view),
/// the q section rotates by θ exactly once, the k section rotates by θ exactly
/// once, and the v section is never touched.
#[test]
fn rope_4d_rotates_each_fused_section_exactly_once() {
    let ops = host_ops();
    let tokens = 3;
    let fused = fused_values(tokens * FUSED, 0);
    let positions = vec![
        0.5, 1.5, -2.0, 3.25, 1.0, -0.5, 2.0, 0.25, -1.75, 3.0, 0.5, 1.25,
    ];
    let freqs = rope_4d_freqs(&AXES_DIM, THETA);
    assert_eq!(freqs.len(), 4, "one frequency per axis in this geometry");

    let fused_buf = ops.upload_f32("test fused", &fused).expect("upload");
    let pos_buf = ops.upload_f32("test pos", &positions).expect("upload");
    let freqs_buf = ops.upload_f32("test freqs", &freqs).expect("upload");

    ops.rope_4d_host(&Rope4dCall {
        step: "test rope",
        q: fused_buf.view(),
        k: fused_buf.view(),
        positions: pos_buf.view(),
        freqs: freqs_buf.view(),
        tokens,
        heads: HEADS,
        head_dim: HEAD_DIM,
        freqs_per_axis: 1,
        row_stride: FUSED,
        q_offset: 0,
        k_offset: HIDDEN,
    })
    .expect("the host rope twin runs");

    let after = ops
        .download("test rope", fused_buf.view(), tokens * FUSED)
        .expect("download");

    let expected = expect_rope(&fused, &positions, &freqs, tokens, &[0, HIDDEN]);
    // The old bug: both passes rotated the q section and neither the k.
    let double_q = expect_rope(&fused, &positions, &freqs, tokens, &[0, 0]);
    assert_differs(&expected, &double_q, "rope: rotating q twice vs once");

    assert_close(&after, &expected, "rope: fused row after rotation");
    for t in 0..tokens {
        let v_start = t * FUSED + 2 * HIDDEN;
        assert_close(
            &after[v_start..v_start + HIDDEN],
            &fused[v_start..v_start + HIDDEN],
            "rope: v section must be untouched",
        );
    }
}

/// Critical 1: with one fused buffer bound as q, k and v, attention must read
/// k and v from their own sections, not the q section.
#[test]
fn sdpa_reads_q_k_and_v_from_their_own_sections() {
    let ops = host_ops();
    let txt = 2;
    let img = 1;
    let txt_rows = fused_values(txt * FUSED, 0);
    let img_rows = fused_values(img * FUSED, 7);

    let txt_buf = ops.upload_f32("test txt", &txt_rows).expect("upload");
    let img_buf = ops.upload_f32("test img", &img_rows).expect("upload");
    let out_txt = ops.alloc_f32("test out txt", txt * HIDDEN).expect("alloc");
    let out_img = ops.alloc_f32("test out img", img * HIDDEN).expect("alloc");

    ops.sdpa_host(&SdpaCall {
        step: "test sdpa",
        out_txt: out_txt.view(),
        out_img: out_img.view(),
        q_txt: txt_buf.view(),
        k_txt: txt_buf.view(),
        v_txt: txt_buf.view(),
        q_img: img_buf.view(),
        k_img: img_buf.view(),
        v_img: img_buf.view(),
        q_offset: 0,
        k_offset: HIDDEN,
        v_offset: 2 * HIDDEN,
        cache_k: ops.dummy_view(),
        cache_v: ops.dummy_view(),
        cache_stride: FUSED,
        txt_tokens: txt,
        img_tokens: img,
        cache_tokens: 0,
        heads: HEADS,
        head_dim: HEAD_DIM,
        txt_stride: FUSED,
        img_stride: FUSED,
        causal: false,
    })
    .expect("the host sdpa twin runs");

    let joint: Vec<&[f32]> = vec![&txt_rows[..FUSED], &txt_rows[FUSED..], &img_rows];
    let sections = joint.iter().map(|_| 0usize).collect::<Vec<_>>();
    let true_sections = joint.iter().map(|_| HIDDEN).collect::<Vec<_>>();
    let v_true = joint.iter().map(|_| 2 * HIDDEN).collect::<Vec<_>>();
    let expected = expect_sdpa(&joint, &true_sections, &v_true);
    // The old bug: k and v read from the q section (softmax(q q^T) q).
    let buggy = expect_sdpa(&joint, &sections, &sections);
    assert_differs(&expected, &buggy, "sdpa: true sections vs q section");

    let got_txt = ops
        .download("test sdpa", out_txt.view(), txt * HIDDEN)
        .expect("download");
    let got_img = ops
        .download("test sdpa", out_img.view(), img * HIDDEN)
        .expect("download");
    assert_close(&got_txt, &expected[..txt * HIDDEN], "sdpa: txt rows");
    assert_close(&got_img, &expected[txt * HIDDEN..], "sdpa: img rows");
}

/// The cached reference rows are fused qkv rows too: their k and v must come
/// from the same section offsets as the live rows.
#[test]
fn sdpa_cached_rows_use_the_section_offsets() {
    let ops = host_ops();
    let txt = 2;
    let txt_rows = fused_values(txt * FUSED, 0);
    let cache_row = fused_values(FUSED, 13);

    let txt_buf = ops.upload_f32("test txt", &txt_rows).expect("upload");
    let cache_buf = ops.upload_f32("test cache", &cache_row).expect("upload");
    let out_txt = ops.alloc_f32("test out txt", txt * HIDDEN).expect("alloc");

    ops.sdpa_host(&SdpaCall {
        step: "test sdpa cache",
        out_txt: out_txt.view(),
        out_img: ops.dummy_view(),
        q_txt: txt_buf.view(),
        k_txt: txt_buf.view(),
        v_txt: txt_buf.view(),
        q_img: ops.dummy_view(),
        k_img: ops.dummy_view(),
        v_img: ops.dummy_view(),
        q_offset: 0,
        k_offset: HIDDEN,
        v_offset: 2 * HIDDEN,
        cache_k: cache_buf.view(),
        cache_v: cache_buf.view(),
        cache_stride: FUSED,
        txt_tokens: txt,
        img_tokens: 0,
        cache_tokens: 1,
        heads: HEADS,
        head_dim: HEAD_DIM,
        txt_stride: FUSED,
        img_stride: FUSED,
        causal: false,
    })
    .expect("the host sdpa twin runs");

    let joint: Vec<&[f32]> = vec![&txt_rows[..FUSED], &txt_rows[FUSED..], &cache_row];
    let mut k_sections = vec![HIDDEN; 3];
    let mut v_sections = vec![2 * HIDDEN; 3];
    let expected = expect_sdpa(&joint, &k_sections, &v_sections);
    // The cache row read as if it were a q section.
    k_sections[2] = 0;
    v_sections[2] = 0;
    let buggy = expect_sdpa(&joint, &k_sections, &v_sections);
    assert_differs(&expected, &buggy, "sdpa cache: true sections vs q section");

    let got = ops
        .download("test sdpa cache", out_txt.view(), txt * HIDDEN)
        .expect("download");
    // `expect_sdpa` scores every joint row as a query; only the txt rows are
    // queries here, the cache row contributes keys and values only.
    assert_close(
        &got,
        &expected[..txt * HIDDEN],
        "sdpa: txt rows with cached keys",
    );
}

/// The production constructor refuses a device that does not carry the flux2
/// kernel set - naming the missing entries - instead of accepting it and
/// running the host twins silently. The table device resolves no entry, so
/// every required kernel is named.
#[test]
fn ops_new_refuses_a_device_without_the_flux2_kernels() {
    let backend: Arc<dyn Backend> = Arc::new(TableBackend::default());
    let err = Ops::new(&backend).expect_err("a device with no flux2 kernels is refused at load");
    let msg = err.to_string();
    // A sentinel from each end of the required list, so a probe that only
    // checked one entry (the old bug) cannot pass.
    assert!(msg.contains("kernel_dit_gemm_bf16"), "{msg}");
    assert!(msg.contains("kernel_dit_embed_bf16_f32"), "{msg}");
    assert!(msg.contains("kernel_dit_sdpa_f32"), "{msg}");
}
