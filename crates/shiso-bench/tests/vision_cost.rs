//! What a picture costs a prefill, measured on this machine.
//!
//! This is the perf side of the crate, so it records durations and never a
//! logit — the rule in the crate docs, applied. Everything below is timed;
//! nothing below is compared against a recorded answer.
//!
//! Run it with:
//!
//! ```text
//! SHISO_TEST_MODEL_OPTIQ=/path/to/mlx \
//!   cargo test -p shiso-bench --release --test vision_cost -- --ignored --nocapture
//! ```
//!
//! # What it measures, and what it deliberately does not
//!
//! A multimodal prefill costs two things that a text-only prefill does not:
//!
//! 1. the tower runs once per media item, turning patch rows into merged
//!    embedding rows;
//! 2. the prompt grows by one placeholder token per merged row, and those
//!    tokens are prefilled like any other — plus a splice and a three-stream
//!    rotation over the chunk they fall in.
//!
//! Only the first of those is measured here, and deliberately so: the tower
//! half is a per-item cost with no seam to hide behind, and this file drives a
//! [`shiso_models::qwen35::vision::VisionEncoder`] directly to report it. The
//! second half belongs to the engine's forward pass, whose text-only prefill
//! cost the other benches already record; the placeholder rows this tower
//! returns join that prefill one per prompt token, so the budget each image
//! size adds is reported beside the encode rather than a second prefill timing
//! being invented. Timing a text-only prefill of the right *length* and calling
//! it the multimodal number would omit the splice and the three-stream
//! rotation entirely, and a fabricated baseline would be a gate that passed
//! because it was written to pass.
//!
//! What this file therefore reports is the encoder half, per item, at two image
//! sizes, plus the placeholder budget each size adds to the prompt. The prefill
//! half is named in the report rather than guessed at.
//!
//! # The protocol
//!
//! [`shiso_bench::measure`] is the only place a median or a spread is computed
//! in this tree, and this case computes neither itself. It runs
//! [`Protocol::standard`]: one whole repetition discarded, one encode per
//! repetition discarded, a median over the timed encodes within a repetition,
//! and a median over the repetitions with the slowest-over-fastest spread
//! beside it. A spread past the noise policy fails the case rather than
//! producing a number, for the same reason the recorder refuses to write one:
//! a figure the next run cannot reproduce is worse than no figure.
//!
//! # Why the readback is inside the timed window
//!
//! The same reason the prefill and decode phases put it inside theirs. The
//! copy back to the host is the only thing in the phase that forces the device
//! to have finished; stopping the clock at submission would reward a backend
//! for queueing more deeply rather than for computing faster.
//!
//! # There is no baseline for this target
//!
//! Nothing here is graded. [`shiso_bench::baseline`] has no vision target, so
//! there is no committed record to compare against and none is invented — a
//! fabricated baseline would be a gate that passed because it was written to
//! pass. What recording one would take is listed at the bottom of this file.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use shiso_bench::measure::{Protocol, SPREAD_MAX, Summary, median};
use shiso_core::Backend;
use shiso_metal::{MetalBackend, metal_available};
use shiso_models::qwen35::vision::{VisionEncoder, VisionWeights, load_vision_weights};
use shiso_vision::{VisionItem, VisionSource};

/// Image edges the cost is reported at, in pixels on a side.
///
/// Two rather than one, because a single figure cannot say whether the cost is
/// dominated by the per-item overhead or by the grid. The larger edge covers
/// twice the area of the smaller, and the attention inside the tower is over
/// the full patch grid with no causal mask, so a reader can see from the pair
/// whether the number they care about scales with the grid or faster.
///
/// Both edges are small, and the reason is a measured one rather than
/// impatience. The protocol runs `(warmup + timed) * (warmup reps + reps)`
/// encodes per edge, and this tower is expensive: an edge of 448 measures at
/// about 1.7 seconds an encode on this device, which is four unbroken minutes
/// of GPU work for that edge alone. It does not survive it. On a quiesced
/// machine the first five repetitions of that edge agreed to three hundredths
/// of a per cent — 1696.9, 1696.6, 1696.8, 1696.7, 1697.1 ms — and the last
/// two came back 1718.2 and 1730.8, a monotone climb with the shape of a
/// device losing its clocks to heat rather than of a noisy neighbour. The
/// spread lands just past the noise policy and the case refuses to report,
/// which is the protocol working: quiescing the machine cannot fix a phase
/// that heats the device itself, so the phase is sized to what the device
/// holds. Both edges here do, at a spread of about five ten-thousandths.
const EDGES: [usize; 2] = [224, 320];

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// Prints a measurement, which is recorded as well as compared.
#[expect(clippy::print_stdout, reason = "the measurement is a test's output")]
fn report(what: &str) {
    println!("{what}");
}

/// The checkpoint directory holding the tower, or `None` when this box has none.
fn checkpoint_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("SHISO_TEST_MODEL_OPTIQ").ok()?);
    dir.is_dir().then_some(dir)
}

/// One item covering a square image of `edge` pixels at the tower's geometry.
///
/// The pixel values are generated rather than read from a file, and that is a
/// statement about what is being measured: the tower's cost is a function of
/// the grid and the widths, not of what the picture is of. A committed PNG
/// would add a decode and a resize to the timed path and change no number the
/// device produces.
///
/// The generator is a wrapping integer walk mapped into `[-1, 1]`. Cheap, but
/// not a constant: a buffer of identical values is the one input that could
/// flatter a reduction, and this is a timing case, so the input stays varied.
fn item_for(weights: &VisionWeights, edge: usize) -> VisionItem {
    let dims = weights.dims();
    let geom = dims.geometry();
    let side = edge / geom.patch_size();
    // The merger consumes whole blocks, so a grid that is not a multiple of the
    // merge side has no pad count and no output rows. Rounded down rather than
    // asserted: an edge that does not divide is a fixture choice, not a defect
    // for this file to diagnose.
    let side = (side / geom.spatial_merge_size()) * geom.spatial_merge_size();
    let row_len = geom.patch_dim();
    let rows = side * side;
    let mut pixels = Vec::with_capacity(rows * row_len);
    let mut state: u32 = 0x9E37_79B9;
    for _ in 0..rows * row_len {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        pixels.push(f32::from(state.to_le_bytes()[3]) / 127.5 - 1.0);
    }
    VisionItem::new(pixels.into_boxed_slice(), row_len, 1, side, side)
        .expect("the generated buffer matches the grid it was sized from")
}

/// One encode, from patch rows to merged rows back on the host, in milliseconds.
fn encode_once(
    backend: &Arc<dyn Backend>,
    encoder: VisionEncoder,
    weights: &VisionWeights,
    item: &VisionItem,
    host: &mut Vec<u8>,
) -> f64 {
    let started = Instant::now();
    let out = encoder
        .encode(backend, weights, item)
        .expect("the tower encodes the item");
    let len = usize::try_from(out.view().len()).expect("an output that fits in this address space");
    host.resize(len, 0);
    backend
        .copy_d2h(out.view(), host)
        .expect("the merged rows read back");
    started.elapsed().as_secs_f64() * 1000.0
}

/// The per-repetition medians for one item, under `protocol`.
fn repetitions(
    backend: &Arc<dyn Backend>,
    encoder: VisionEncoder,
    weights: &VisionWeights,
    item: &VisionItem,
    protocol: Protocol,
) -> Vec<f64> {
    let mut host = Vec::new();
    let one_rep = |host: &mut Vec<u8>| {
        for _ in 0..protocol.warmup_prefills {
            encode_once(backend, encoder, weights, item, host);
        }
        let mut timed = Vec::with_capacity(usize::try_from(protocol.timed_prefills).unwrap_or(0));
        for _ in 0..protocol.timed_prefills {
            timed.push(encode_once(backend, encoder, weights, item, host));
        }
        median(&timed).expect("the protocol asks for at least one timed encode")
    };

    for _ in 0..protocol.warmup_reps {
        one_rep(&mut host);
    }
    (0..protocol.reps).map(|_| one_rep(&mut host)).collect()
}

/// The encoder half of a multimodal prefill, at two image sizes.
///
/// Asserts three things, and reports the rest:
///
/// - the tower hands back exactly one row per placeholder token the prompt will
///   carry for the item. The whole cost figure is per merged token, so a row
///   count that disagreed with the pad run would make the number describe
///   nothing;
/// - every repetition produced a statistic;
/// - the repetitions agreed inside the noise policy. A machine that will not
///   hold that is quiesced and re-run; the policy is never widened to make a
///   run land.
#[test]
#[ignore = "needs the local checkpoint directory and a Metal device"]
fn vision_prefill_cost() {
    let Some(dir) = checkpoint_dir() else {
        skip("no local checkpoint directory");
        return;
    };
    if !metal_available() {
        skip("no Metal device");
        return;
    }
    let backend: Arc<dyn Backend> = MetalBackend::new().expect("the Metal device opens");
    let weights = match load_vision_weights(&backend, &VisionSource::OptiqDir(dir)) {
        Ok(weights) => weights,
        Err(why) => {
            skip(&format!("the tower would not load: {why}"));
            return;
        }
    };
    let dims = weights.dims();
    let merge = dims.geometry().spatial_merge_size();
    let encoder = VisionEncoder::new();
    let protocol = Protocol::standard();

    report(&format!(
        "tower: depth {} hidden {} heads {} out_hidden {} patch {} merge {} ({} bytes resident)",
        dims.depth(),
        dims.hidden(),
        dims.heads(),
        dims.out_hidden(),
        dims.geometry().patch_size(),
        merge,
        weights.device_bytes(),
    ));
    report(&format!(
        "protocol: {} warmup rep(s), {} warmup + {} timed encode(s) per rep, {} rep(s), \
         statistic {}",
        protocol.warmup_reps,
        protocol.warmup_prefills,
        protocol.timed_prefills,
        protocol.reps,
        protocol.statistic(),
    ));

    for edge in EDGES {
        let item = item_for(&weights, edge);
        let (_, grid_h, grid_w) = item.grid();
        let pads = item.pad_count(merge);

        // The row count is the contract the splice leans on and the divisor of
        // every figure below, so it is checked before anything is timed.
        let out = encoder
            .encode(&backend, &weights, &item)
            .expect("the tower encodes the item");
        let row_bytes = u64::try_from(dims.out_hidden() * size_of::<f32>())
            .expect("a row width that fits in a device length");
        assert_eq!(
            out.view().len() / row_bytes,
            u64::try_from(pads).expect("a pad count that fits in a device length"),
            "the tower produced a row count the prompt's placeholder run does not match",
        );
        drop(out);

        let reps = repetitions(&backend, encoder, &weights, &item, protocol);
        let summary = Summary::over(&reps).expect("every repetition produced a statistic");
        let per_pad = summary.value / f64::from(u32::try_from(pads).unwrap_or(u32::MAX));
        report(&format!(
            "{edge}x{edge}px: grid {grid_h}x{grid_w} patches, {pads} placeholder tokens, \
             encode {:.3} ms ({:.4} ms per merged token), spread {:.4}, reps {:?}",
            summary.value, per_pad, summary.spread, summary.reps,
        ));
        assert!(
            summary.steady(),
            "the repetitions for {edge}x{edge} disagreed by {:.4}, past the {SPREAD_MAX} noise \
             policy; quiesce the machine and re-run rather than widening the policy",
            summary.spread,
        );
    }

    report(
        "NOT MEASURED: the prompt-side half of a multimodal prefill - the placeholder rows, the \
         splice that replaces them, and the three-stream rotation. That path now exists and runs \
         (the Model contract carries media and the scheduler encodes and prefills it), so this is \
         a gap in what this case times rather than a gap in the build. Timing a text-only prefill \
         of the same length would omit exactly the launches that are new.",
    );
    report(
        "NOT RECORDED: shiso_bench::baseline has no vision target, so these numbers are printed \
         and not filed. Filing them needs a Target variant with its own baseline filename, a \
         recorder arm that writes the record, the gate arithmetic to read it, and a committed \
         baseline promoted from a record taken on a quiesced pinned machine.",
    );
}
