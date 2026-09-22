//! The two routes `projection::projection` can take for a prefill chunk,
//! graded against each other **through that function** rather than beside it.
//!
//! This case lives here and not in `shiso-kerneltest` for two reasons, and
//! only the first is about backends. Metal has no vendor matrix tier, so there
//! is no second answer for it to compare and a shared case would skip there -
//! and a skip prints the same colour as a pass. The second is that grading the
//! shipped path means calling the shipped entry point, which needs
//! `shiso-engine`; the shared crate depends on `shiso-core` alone, by design.
//!
//! # Why it goes through `projection` and not through two hand-built launches
//!
//! An earlier version of this case built both launches itself and multiplied
//! at 512 columns. It would have caught a broken `dequant_plane` or a wrong
//! `GemmDesc`, and nothing else: not a wrong band offset, not a missing
//! zero-fill of the padded rows, not a wrong eligibility predicate, and not
//! `PREFILL_BLAS_PAD` changing underneath it. All four are the parts of this
//! route that were hard to get right, so the case drives the real function and
//! chooses the route the way a caller does - by how long its buffers are.
//!
//! **The budget is a measurement, not a dial.** The two paths reduce in
//! different orders over the same inner dimension, so a nonzero disagreement
//! is the correct outcome and the number below is what it was observed to be.
//! Widening it to make a later change pass is the one thing this case must
//! never be used for; a later change that moves it is re-recorded here, in the
//! commit that moves it, with the new measurement in the message.
//!
//! GPU case: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --release --test
//! q35_prefill -- --ignored --nocapture`.
//!
//! **`scripts/parity_gate.sh` runs it**, as the `prefill` lane. It is
//! `#[ignore]`d, so `cargo test --workspace` does not reach it, and for a
//! while no script named it either - which left the case carrying Phase 5's
//! correctness claim executed only when somebody typed its name. That is the
//! same finding the gate's `spec` lane exists to close, and it is closed the
//! same way.
#![cfg(target_os = "linux")]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "the fixture dimensions are small literals and the fixture values are built from indices"
)]

use std::sync::Arc;

use shiso_core::{Backend, DeviceBuffer, alloc};
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_engine::MAX_PREFILL_CHUNK;
use shiso_engine::projection::{Weight, projection};
use shiso_kerneltest::oracle;

/// Output rows the fixture projects to: this family's residual width, which is
/// the shape both `o_proj` and `down_proj` are.
const OUT: u32 = 2560;
/// Inner dimension: ten `q4_K` super-blocks per row.
const IN: u32 = 2560;
/// Tokens in the chunk under test.
///
/// **Deliberately not a multiple of `PREFILL_BLAS_PAD`, and deliberately more
/// than one band.** At 384 the vendor route runs two multiplies of 256
/// columns, the second of which is half padding - so this one width exercises
/// the band offset, the partial last band and the zero-fill of the rows past
/// the chunk. A width equal to the pad would exercise none of the three.
const TOK: u32 = 384;
/// Elements in one `q4_K` super-block.
const QK_K: usize = 256;
/// Bytes in one, which is the loader's `BlockQ4K`.
const Q4K_BYTES: usize = 144;

/// What the two routes were measured to differ by, in absolute terms, at this
/// shape on this device.
///
/// **0.19140625, to the digit this case printed** when it was first run with
/// the budget at `0.0`, on a GB10 on 2026-09-22 - at 384 tokens, through
/// `projection`, with the vendor route running two bands of 256. It is
/// written down rather than
/// chosen, and the `<=` below is deliberately not a margin: both routes are
/// deterministic at a fixed shape, so a run that lands anywhere else is a
/// change to one of them and should be looked at rather than absorbed.
///
/// It is **not** evidence that one route is more faithful than the other. The
/// tile stages both operands as `f16`; the vendor route hands the library f32
/// and the library, with `CudaOptions::tf32` on - which is the default this
/// backend opens its handle with - truncates both operands to a ten-bit
/// mantissa, the same mantissa width `f16` carries. The two round their inputs
/// to the same precision and differ in exponent range and in reduction order.
/// This number is that difference and is not attributable to either side.
const PREFILL_GEMM_BUDGET: f32 = 0.191_406_25;

/// Opens device 0, or says on stderr why the case did not run.
#[expect(
    clippy::print_stderr,
    reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn open() -> Option<Arc<CudaBackend>> {
    match CudaBackend::new(0, CudaOptions::default()) {
        Ok(cuda) => Some(cuda),
        Err(err) => {
            eprintln!("note: no CUDA device on this machine, skipping the GPU cases: {err}");
            None
        }
    }
}

/// A device buffer holding `bytes`.
fn upload(device: &Arc<dyn Backend>, bytes: &[u8]) -> DeviceBuffer {
    let buffer = alloc(device, bytes.len() as u64).expect("device allocation");
    device
        .copy_h2d(buffer.view(), bytes)
        .expect("host-to-device copy");
    buffer
}

/// Reads a whole device buffer back as f32.
fn read_f32(device: &Arc<dyn Backend>, buffer: &DeviceBuffer) -> Vec<f32> {
    let mut raw = vec![0u8; usize::try_from(buffer.len()).expect("length fits usize")];
    device
        .copy_d2h(buffer.view(), &mut raw)
        .expect("device-to-host copy");
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect()
}

/// The vendor route and the hand-written tile agree on a prefill chunk.
///
/// Which route runs is chosen the way a caller chooses it: by how long the
/// activation and output buffers are. `projection::blas_projection` takes the
/// vendor route only for a caller whose buffers hold [`MAX_PREFILL_CHUNK`]
/// rows, so the short pair below gets the tile and the long pair gets the
/// vendor route, and neither call names a kernel.
#[test]
#[ignore = "requires a CUDA device"]
fn the_vendor_route_agrees_with_the_tile_through_the_projection_entry_point() {
    let Some(cuda) = open() else { return };
    let device: Arc<dyn Backend> = cuda.clone();

    let weight = Weight::from_buffer(
        shiso_core::DType::Q4_K,
        IN,
        OUT,
        upload(&device, &q4_k_plane()),
    );
    let acts = deterministic_acts();

    // The short pair: exactly the chunk, which is shorter than the cap, so the
    // eligibility predicate refuses and the tile runs.
    let short_src = upload(&device, &f32_bytes(&acts));
    let short_dst = upload(&device, &poison_bytes(u64::from(OUT) * u64::from(TOK)));
    projection(
        device.as_ref(),
        &weight,
        TOK,
        short_src.view(),
        short_dst.view(),
    )
    .expect("the tile route runs");

    // The long pair: the cap, which is what the hybrid stack's own scratch is.
    // The rows past the chunk are filled with a value no correct result can
    // contain, so a missing zero-fill shows up as a padded output row that is
    // not zero rather than as a number nobody looks at.
    let mut wide = acts.clone();
    wide.resize((IN * MAX_PREFILL_CHUNK) as usize, POISON);
    let long_src = upload(&device, &f32_bytes(&wide));
    let long_dst = upload(
        &device,
        &poison_bytes(u64::from(OUT) * u64::from(MAX_PREFILL_CHUNK)),
    );
    projection(
        device.as_ref(),
        &weight,
        TOK,
        long_src.view(),
        long_dst.view(),
    )
    .expect("the vendor route runs");

    device.sync().expect("the queue drains");
    let tiled = read_f32(&device, &short_dst);
    let vendor = read_f32(&device, &long_dst);

    // The bands the chunk does not fill were zeroed before the multiply read
    // them, so their output is exactly zero. This is also what proves the long
    // call took the vendor route at all: the tile writes `n_tok` columns and
    // would have left the poison there.
    let padded = &vendor[(OUT * TOK) as usize..];
    assert_eq!(
        padded.len(),
        ((MAX_PREFILL_CHUNK - TOK) * OUT) as usize,
        "the padded region is the rows between the chunk and the cap"
    );
    let stray = padded.iter().position(|v| *v != 0.0);
    assert!(
        stray.is_none(),
        "the vendor route left {:?} in padded output row {}, so either it did \
         not run or the rows past the chunk were not zeroed before the multiply \
         read them",
        stray.map(|i| padded[i]),
        stray.map_or(0, |i| TOK as usize + i / OUT as usize),
    );

    let (delta, at) = max_abs_delta(&vendor[..(OUT * TOK) as usize], &tiled);
    // Zero would mean both calls took the same route and this case graded
    // nothing - which is exactly what a broken eligibility predicate looks
    // like from the outside.
    assert!(
        delta > 0.0,
        "the two calls agreed to the bit, so both took the same route and this \
         case compared a result with itself"
    );
    assert!(
        delta <= PREFILL_GEMM_BUDGET,
        "the vendor route and the tile disagree by {delta} at element {at} \
         (token {}, output {}), past the recorded {PREFILL_GEMM_BUDGET}. This \
         budget is a record of a measurement: if a deliberate change moved it, \
         re-record it here with the new number in the commit message.",
        at / OUT as usize,
        at % OUT as usize,
    );
}

/// A value no correct product of this fixture can contain.
const POISON: f32 = -1.234_567_9e30;

/// `count` f32 poison values, as bytes.
fn poison_bytes(count: u64) -> Vec<u8> {
    POISON
        .to_le_bytes()
        .repeat(usize::try_from(count).expect("count fits usize"))
}

/// f32 values as little-endian bytes.
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The largest absolute disagreement between two readbacks, and where it was.
fn max_abs_delta(left: &[f32], right: &[f32]) -> (f32, usize) {
    let mut worst = 0.0f32;
    let mut at = 0;
    for (i, (&l, &r)) in left.iter().zip(right).enumerate() {
        let delta = (l - r).abs();
        if delta > worst {
            worst = delta;
            at = i;
        }
    }
    (worst, at)
}

/// A `q4_K` weight plane whose nibbles, scales and minima all vary by row.
///
/// Varying rather than fixed, and deliberately *not* exact in binary: a
/// fixture whose every product is representable would make both routes agree
/// to the last bit and the budget above would record nothing. What is being
/// measured is the disagreement between two reduction orders, and a fixture
/// that cannot disagree cannot measure it.
fn q4_k_plane() -> Vec<u8> {
    let blocks = IN as usize / QK_K;
    let mut out = Vec::with_capacity(OUT as usize * blocks * Q4K_BYTES);
    for row in 0..OUT as usize {
        for block in 0..blocks {
            let seed = row * 31 + block * 17;
            let mut values = [0u8; QK_K];
            for (e, slot) in values.iter_mut().enumerate() {
                *slot = ((seed + e * 7 + (e * e) / 13) % 16) as u8;
            }
            let mut scales = [0u8; 12];
            for (i, slot) in scales.iter_mut().enumerate() {
                *slot = ((seed + i * 11) % 64) as u8;
            }
            let packed = oracle::Q4K {
                // 0.0113 and 0.0071 as f16 bit patterns: two scales that are
                // not powers of two, so no product is exact.
                d: 0x21C9,
                dmin: 0x1F45,
                scales,
                qs: oracle::pack_q4_k_dense(&values),
            };
            out.extend_from_slice(&packed.d.to_le_bytes());
            out.extend_from_slice(&packed.dmin.to_le_bytes());
            out.extend_from_slice(&packed.scales);
            out.extend_from_slice(&packed.qs);
        }
    }
    out
}

/// Activations that span three orders of magnitude, in a fixed order.
///
/// A spread rather than a constant, because a reduction order only shows up
/// when the addends differ in size: summing equal values in any order gives
/// the same answer.
fn deterministic_acts() -> Vec<f32> {
    (0..(u64::from(TOK) * u64::from(IN)) as usize)
        .map(|i| {
            let phase = ((i % 251) as f32) / 251.0 - 0.5;
            let decade = 1.0 + ((i % 7) as f32) * 14.0;
            phase * decade
        })
        .collect()
}
