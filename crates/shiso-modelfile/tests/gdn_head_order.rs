//! The two containers agree about the recurrent block's value heads.
//!
//! This is the assertion that was missing. The checkpoint directory and the
//! converted single file hold the same model, and the loader's job is to hand
//! the binding layer the same weights from either one. For most of the model
//! that is guaranteed by the shapes: a matrix loaded with its axes confused is
//! a matrix of the wrong size, and the binding layer says so.
//!
//! The recurrent block's value heads are the exception. The two containers
//! order them differently - the directory interleaves the pair that shares a
//! query/key head, the converter de-interleaves them - and neither order
//! changes a single dimension. A checkpoint loaded in the wrong one binds, runs
//! at full speed, and generates fluent, confident, unrelated text, because
//! every value head is reading a query/key head that is not its own. Nothing
//! short of comparing the two checkpoints catches that, so this compares them.
//!
//! What is compared is what can be compared exactly. The converted file stores
//! the per-head decay, the time-step bias and the short convolution
//! unquantized, so those are equality assertions; the alpha and beta gates are
//! `q8_0` there and affine-quantized here, so those are compared by direction,
//! which is far more agreement than a mismatched head could produce. The fused
//! projections and the output projection are `K`-quantized in the converted
//! file and this crate has no `K`-quant decoder; they carry the same head order
//! as the short convolution, which indexes the very same channels.
//!
//! Skips when either checkpoint is absent. No device is involved.

mod common;
mod optiq;

use half::f16;
use shiso_modelfile::quant::BlockQ8_0;
use shiso_modelfile::{GgufFile, SafetensorsFile, TensorSource};

/// Value heads in a recurrent block of this model.
const VALUE_HEADS: usize = 32;
/// Query and key heads, each shared by a pair of value heads.
const KEY_HEADS: usize = 16;
/// Channels in one value head.
const HEAD_DIM: usize = 128;
/// Channels the fused projection spends on queries and keys together.
const QK_WIDTH: usize = 2 * KEY_HEADS * HEAD_DIM;
/// Layer slots in this model.
const LAYERS: u32 = 32;
/// Every fourth slot holds attention rather than recurrence.
const ATTENTION_INTERVAL: u32 = 4;
/// Cosine similarity below which two quantizations of one row disagree.
///
/// Two independent quantizations of the same row agree to about `0.995`; two
/// different rows of the same matrix land near `0.2`. Anything in between
/// would be a new phenomenon worth failing over.
const SAME_ROW: f32 = 0.98;

/// Recurrent layer slots, which are the ones with a `linear_attn` block.
fn gdn_layers() -> impl Iterator<Item = u32> {
    (0..LAYERS).filter(|layer| !(layer + 1).is_multiple_of(ATTENTION_INTERVAL))
}

/// Both checkpoints, or `None` when this machine has neither.
fn both() -> Option<(GgufFile, SafetensorsFile)> {
    let converted = common::open_model()?;
    let directory = optiq::open_checkpoint()?;
    Some((converted, directory))
}

/// One tensor's `f32` values from whichever container it came from.
fn f32s(source: &impl TensorSource, name: &str) -> Vec<f32> {
    let view = source
        .get(name)
        .unwrap_or_else(|| panic!("{name} is in this checkpoint"));
    view.as_f32()
        .unwrap_or_else(|err| panic!("{name} decodes to f32: {err}"))
        .to_vec()
}

/// One tensor's `f16` values, widened, from the checkpoint directory.
fn f16s(source: &SafetensorsFile, name: &str) -> Vec<f32> {
    let view = source
        .get(name)
        .unwrap_or_else(|| panic!("{name} is in this checkpoint"));
    view.as_f16()
        .unwrap_or_else(|err| panic!("{name} decodes to f16: {err}"))
        .iter()
        .copied()
        .map(f16::to_f32)
        .collect()
}

/// One `q8_0` tensor from the converted file, decoded.
///
/// Thirty-four bytes per thirty-two weights: one `fp16` scale and the signed
/// weights it scales. Written out here rather than borrowed from a kernel
/// because this is the only place in the crate that needs it.
fn q8_0(file: &GgufFile, name: &str) -> Vec<f32> {
    let view = file
        .get(name)
        .unwrap_or_else(|| panic!("{name} is in the converted file"));
    let blocks: &[BlockQ8_0] = view
        .blocks()
        .unwrap_or_else(|err| panic!("{name} is q8_0: {err}"));
    let mut out = Vec::with_capacity(blocks.len() * 32);
    for block in blocks {
        let scale = f16::from_bits(block.d).to_f32();
        out.extend(block.qs.iter().map(|&q| scale * f32::from(q)));
    }
    out
}

/// Cosine similarity, which is what survives two different quantizations.
fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let dot: f32 = left.iter().zip(right).map(|(a, b)| a * b).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(left) * norm(right) + f32::EPSILON)
}

/// The value head the checkpoint directory holds at `index` before the loader
/// reorders it: the pair sharing query/key head `k` sits at `2k` and `2k + 1`.
fn interleaved(index: usize) -> usize {
    (index % KEY_HEADS) * 2 + index / KEY_HEADS
}

/// Two values that came from the same source through different arithmetic.
///
/// A relative ulp, not a tolerance: the only slack this needs is the last
/// place of an `f32`, because the decay is `-exp` of a stored logarithm and the
/// converter's `exp` is not this platform's. Anything looser would start to
/// admit a different head.
fn close(want: f32, got: f32) -> bool {
    (want - got).abs() <= f32::EPSILON * want.abs().max(f32::MIN_POSITIVE)
}

#[test]
fn per_head_scalars_agree_with_the_converted_checkpoint() {
    let Some((converted, directory)) = both() else {
        common::skip("one of the two checkpoints is missing");
        return;
    };

    for layer in gdn_layers() {
        for role in ["ssm_a", "ssm_dt.bias"] {
            let name = format!("blk.{layer}.{role}");
            let want = f32s(&converted, &name);
            let got = f32s(&directory, &name);
            assert_eq!(want.len(), VALUE_HEADS, "{name} is one value per head");
            assert_eq!(got.len(), VALUE_HEADS, "{name} is one value per head");

            // Both containers store these unquantized, from the same `bf16`
            // source. The time-step bias is therefore bit-identical; the decay
            // is `-exp` of it, and the two `exp` implementations that computed
            // it disagree in the last place on a few heads, so it is compared
            // to a relative ulp rather than to the bit.
            for (head, (&a, &b)) in want.iter().zip(&got).enumerate() {
                assert!(
                    close(a, b),
                    "{name}: head {head} disagrees between the two checkpoints, {a} against {b}"
                );
            }

            // And the order the directory stores them in is a different one,
            // which is what makes the comparison above worth making.
            let stale = (0..VALUE_HEADS).any(|h| !close(want[h], got[interleaved(h)]));
            assert!(stale, "{name} would pass without being reordered");
        }
    }
}

#[test]
fn the_short_convolution_agrees_channel_for_channel() {
    let Some((converted, directory)) = both() else {
        common::skip("one of the two checkpoints is missing");
        return;
    };

    for layer in gdn_layers() {
        let name = format!("blk.{layer}.ssm_conv1d.weight");
        let want = f32s(&converted, &name);
        let got = f32s(&directory, &name);
        let kernel = want.len() / (QK_WIDTH + VALUE_HEADS * HEAD_DIM);
        assert_eq!(
            want.len(),
            (QK_WIDTH + VALUE_HEADS * HEAD_DIM) * kernel,
            "{name} is one kernel per fused channel"
        );

        // The query and key channels are in the same order in both containers,
        // so a difference here would mean the reordering had reached past the
        // value channels it belongs to.
        assert_eq!(
            got[..QK_WIDTH * kernel],
            want[..QK_WIDTH * kernel],
            "{name}: the query and key channels moved"
        );
        assert_eq!(
            got[QK_WIDTH * kernel..],
            want[QK_WIDTH * kernel..],
            "{name}: the value channels disagree"
        );
    }
}

#[test]
fn the_per_head_gates_agree_row_for_row() {
    let Some((converted, directory)) = both() else {
        common::skip("one of the two checkpoints is missing");
        return;
    };

    for layer in gdn_layers() {
        for role in ["ssm_alpha.weight", "ssm_beta.weight"] {
            let name = format!("blk.{layer}.{role}");
            let want = q8_0(&converted, &name);
            let got = f16s(&directory, &name);
            assert_eq!(want.len(), got.len(), "{name} has the same element count");
            let width = want.len() / VALUE_HEADS;

            for head in 0..VALUE_HEADS {
                let row = |v: &[f32]| v[head * width..(head + 1) * width].to_vec();
                let matched = cosine(&row(&want), &row(&got));
                assert!(
                    matched > SAME_ROW,
                    "{name}: head {head} does not match the converted file ({matched})"
                );

                // The row the directory holds at this position before the
                // reordering belongs to another head, and is nothing like it.
                let source = interleaved(head);
                if source != head {
                    let unmoved = cosine(&row(&want), &got[source * width..(source + 1) * width]);
                    assert!(
                        unmoved < SAME_ROW,
                        "{name}: head {head} also matches head {source} ({unmoved}), \
                         so this test cannot tell the two orders apart"
                    );
                }
            }
        }
    }
}

#[test]
fn the_head_norm_is_shared_and_is_not_reordered() {
    let Some((converted, directory)) = both() else {
        common::skip("one of the two checkpoints is missing");
        return;
    };

    // One gain per channel of a head, shared by every head - so it carries no
    // value-head index and must come through in the order it was written. It
    // is the one tensor in the recurrent block that looks per-head and is not.
    for layer in gdn_layers() {
        let name = format!("blk.{layer}.ssm_norm.weight");
        let want = f32s(&converted, &name);
        let got = f32s(&directory, &name);
        assert_eq!(want.len(), HEAD_DIM, "{name} is one gain per head channel");
        assert_eq!(got.len(), HEAD_DIM, "{name} is one gain per head channel");
        // The directory stores this one as `bf16` and the converted file kept
        // `f32`, so the two differ by one rounding of the low bits.
        for (at, (a, b)) in want.iter().zip(&got).enumerate() {
            assert!(
                (a - b).abs() <= 2.0_f32.powi(-8) * a.abs().max(1.0),
                "{name}[{at}]: {a} against {b}"
            );
        }
    }
}
