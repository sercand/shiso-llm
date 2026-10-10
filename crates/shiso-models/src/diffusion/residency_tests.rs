//! The residency planner's decisions at real sizes, and the span helpers.
//!
//! The component sizes are the Qwen-Image-2.1 INT8 install's (encoder
//! layers 6.95 GB, transformer 7.26 GB, the decoder widened to f32
//! 1.04 GB); the 16 GB-class working set is the one a 16 GB M4 reports
//! (`recommendedMaxWorkingSetSize` = 12 713 115 648 bytes), the 48 GB one
//! the same fraction of 48 GB.

use std::path::PathBuf;

use super::residency::{
    FileSpan, Footprint, JobPeaks, PlanDesc, Residency, ResidencyMode, Shortfall, coalesce, plan,
    split_at, total_bytes,
};

const GIB: u64 = 1 << 30;
const M4_16GB: u64 = 12_713_115_648;
const RESERVE: u64 = 3 * GIB / 2;

const FOOTPRINT: Footprint = Footprint {
    encoder: 6_952_600_000,
    denoiser: 7_257_000_000,
    decoder: 1_040_000_000,
};

/// The pipeline's estimates at a square side with a 526-token prompt
/// (the 512-token cap plus the system turn's 14 rows on the encoder side).
fn peaks(side: u64) -> JobPeaks {
    let t = (side / 16) * (side / 16);
    let p = 512;
    let row = (7 * 3072 + 2 * 9216 + 128) * 4;
    JobPeaks {
        encode: 526 * (8 * 4096 + 3 * 12288) * 4,
        denoise: 2 * 32 * p * 3072 * 4 + (t + p) * row,
        // The decoder's working-set estimate grows with the pixel count:
        // 1128 MiB at 512x512.
        decode: 1128 * (1 << 20) * (side * side) / (512 * 512),
    }
}

fn desc(mode: ResidencyMode, working_set: u64, side: u64) -> PlanDesc {
    PlanDesc {
        mode,
        working_set,
        reserve: RESERVE,
        footprint: FOOTPRINT,
        peaks: peaks(side),
    }
}

#[test]
fn a_16gb_host_swaps_and_keeps_the_denoiser_only_while_the_decode_fits_beside_it() {
    let table = [
        (
            512,
            Ok(Residency::Swap {
                keep_denoiser: true,
            }),
        ),
        (
            1024,
            Ok(Residency::Swap {
                keep_denoiser: false,
            }),
        ),
    ];
    for (side, want) in table {
        assert_eq!(
            plan(&desc(ResidencyMode::Auto, M4_16GB, side)),
            want,
            "{side}x{side}"
        );
        assert_eq!(
            plan(&desc(ResidencyMode::Swap, M4_16GB, side)),
            want,
            "{side}x{side} forced swap"
        );
    }
}

#[test]
fn a_2048_image_is_refused_on_a_16gb_host_with_its_decode_as_the_need() {
    let got = plan(&desc(ResidencyMode::Auto, M4_16GB, 2048));
    let budget = M4_16GB - RESERVE;
    assert_eq!(
        got,
        Err(Shortfall {
            needed: FOOTPRINT.decoder + peaks(2048).decode,
            budget
        })
    );
}

#[test]
fn forcing_residency_on_a_16gb_host_is_refused_with_the_resident_need() {
    let got = plan(&desc(ResidencyMode::Resident, M4_16GB, 512));
    let p = peaks(512);
    let need = FOOTPRINT.encoder
        + FOOTPRINT.denoiser
        + FOOTPRINT.decoder
        + p.encode.max(p.denoise).max(p.decode);
    assert_eq!(
        got,
        Err(Shortfall {
            needed: need,
            budget: M4_16GB - RESERVE
        })
    );
}

#[test]
fn a_48gb_host_keeps_everything_resident_up_to_2048() {
    let ws = M4_16GB * 3;
    assert_eq!(
        plan(&desc(ResidencyMode::Auto, ws, 512)),
        Ok(Residency::Resident)
    );
    assert_eq!(
        plan(&desc(ResidencyMode::Auto, ws, 1024)),
        Ok(Residency::Resident)
    );
    // 2048x2048 decodes in ~17.6 GiB, which still fits beside all three
    // components in a 35 GiB budget.
    assert_eq!(
        plan(&desc(ResidencyMode::Auto, ws, 2048)),
        Ok(Residency::Resident)
    );
    assert_eq!(
        plan(&desc(ResidencyMode::Swap, ws, 512)),
        Ok(Residency::Swap {
            keep_denoiser: true
        })
    );
}

#[test]
fn a_reserve_over_the_working_set_refuses_everything() {
    let mut d = desc(ResidencyMode::Auto, M4_16GB, 256);
    d.reserve = M4_16GB + 1;
    assert!(matches!(plan(&d), Err(Shortfall { budget: 0, .. })));
}

fn span(path: &str, start: u64, end: u64) -> FileSpan {
    FileSpan {
        path: PathBuf::from(path),
        start,
        end,
    }
}

#[test]
fn spans_coalesce_across_small_gaps_within_a_file_only() {
    let got = coalesce(vec![
        span("b", 0, 10),
        span("a", 6_000_000, 7_000_000),
        span("a", 0, 100),
        span("a", 200, 4_000_000),
    ]);
    assert_eq!(
        got,
        vec![
            span("a", 0, 4_000_000),
            span("a", 6_000_000, 7_000_000),
            span("b", 0, 10)
        ]
    );
}

#[test]
fn split_cuts_inside_the_span_that_crosses_the_budget() {
    let spans = vec![span("a", 0, 100), span("a", 200, 300), span("b", 0, 50)];
    let (head, tail) = split_at(&spans, 150);
    assert_eq!(head, vec![span("a", 0, 100), span("a", 200, 250)]);
    assert_eq!(tail, vec![span("a", 250, 300), span("b", 0, 50)]);
    assert_eq!(total_bytes(&head) + total_bytes(&tail), total_bytes(&spans));
    let (all, none) = split_at(&spans, 1_000);
    assert_eq!((all.len(), none.len()), (3, 0));
}
