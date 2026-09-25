//! The sparse family's one-launch routed selection and sort, graded against
//! the four launches it stands in for.
//!
//! `q4e_moe_route_sort_small_kernel` promises the same ids, weights, bounds
//! and active list as `q4e_moe_route_kernel` followed by the count, scan and
//! scatter of the three-pass sort, bit for bit - a prompt pass narrow enough
//! to take it and a wider one that does not have to route a row identically.
//! Inside one expert's span the fused sort orders columns by rank where the
//! scatter's atomic leaves the order to chance, so the spans are compared as
//! sets, and the inverse is checked against the fused sort itself.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "kernel arguments are u32/i32 while the fixtures index in usize; every size and \
              index here is a small literal"
)]
#![expect(
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    reason = "the case asserts by panicking, and running the four launches beside the one reads \
              best as one sequence"
)]

use shiso_core::{Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// The routing kernels' module, mirroring `kernels::MOD_MOE`.
const MOD_MOE: &str = "q4e_moe";
/// The shipped bank and selection width.
const EXPERTS: usize = 512;
/// Experts each token is routed to.
const USED: usize = 10;

/// Launches one entry and waits for it.
fn launch(
    gpu: &Gpu,
    entry: &'static str,
    grid: u32,
    block: u32,
    smem: u32,
    args: &[KernelArg<'_>],
) {
    run(
        gpu,
        KernelRef::new(MOD_MOE, entry),
        &LaunchArgs {
            grid: Extent3::new(grid, 1, 1),
            block: Extent3::new(block, 1, 1),
            smem_bytes: smem,
            args,
        },
    );
    drain(gpu);
}

/// A buffer's words as `i32`.
fn ints(gpu: &Gpu, buffer: &shiso_core::DeviceBuffer) -> Vec<i32> {
    gpu.read_f32(buffer)
        .iter()
        .map(|v| v.to_bits() as i32)
        .collect()
}

/// The fused launch and the four it replaces, over the same logits, agree.
pub fn the_fused_route_sort_is_the_four_launch_route_and_sort(gpu: &Gpu) {
    for n_tok in [1usize, 5, 32] {
        let rows = n_tok * USED;
        // Logits with ties on purpose: the lower id has to win both ways.
        let mut logits = signal(n_tok * EXPERTS, 0x5eed ^ n_tok as u32);
        for t in 0..n_tok {
            logits[t * EXPERTS + 7] = logits[t * EXPERTS + 300];
        }
        let lg = gpu.upload(&f32_bytes(&logits));
        let zeros = |n: usize| gpu.upload(&vec![0u8; n * 4]);

        // The four launches.
        let (ids, w, counts, bounds, active, sorted) = (
            zeros(rows),
            zeros(rows),
            zeros(EXPERTS + 1),
            zeros(EXPERTS + 1),
            zeros(EXPERTS + 1),
            zeros(rows),
        );
        launch(
            gpu,
            "q4e_moe_route_kernel",
            n_tok as u32,
            256,
            (EXPERTS * 4) as u32,
            &[
                KernelArg::Buffer(ids.view()),
                KernelArg::Buffer(w.view()),
                KernelArg::Buffer(lg.view()),
                KernelArg::U32(EXPERTS as u32),
                KernelArg::U32(USED as u32),
            ],
        );
        let flat = (rows as u32).div_ceil(256);
        launch(
            gpu,
            "q4e_moe_sort_count_kernel",
            flat,
            256,
            0,
            &[
                KernelArg::Buffer(counts.view()),
                KernelArg::Buffer(ids.view()),
                KernelArg::U32(rows as u32),
                KernelArg::U32(EXPERTS as u32),
            ],
        );
        launch(
            gpu,
            "q4e_moe_sort_scan_kernel",
            1,
            32,
            0,
            &[
                KernelArg::Buffer(bounds.view()),
                KernelArg::Buffer(active.view()),
                KernelArg::Buffer(counts.view()),
                KernelArg::U32(EXPERTS as u32),
            ],
        );
        let cursor = zeros(EXPERTS + 1);
        launch(
            gpu,
            "q4e_moe_sort_scatter_kernel",
            flat,
            256,
            0,
            &[
                KernelArg::Buffer(sorted.view()),
                KernelArg::Buffer(cursor.view()),
                KernelArg::Buffer(bounds.view()),
                KernelArg::Buffer(ids.view()),
                KernelArg::U32(rows as u32),
                KernelArg::U32(EXPERTS as u32),
            ],
        );

        // The one launch.
        let (f_ids, f_w, f_bounds, f_active, f_sorted, f_pos) = (
            zeros(rows),
            zeros(rows),
            zeros(EXPERTS + 1),
            zeros(EXPERTS + 1),
            zeros(rows),
            zeros(rows),
        );
        launch(
            gpu,
            "q4e_moe_route_sort_small_kernel",
            1,
            32 * 32,
            0,
            &[
                KernelArg::Buffer(f_ids.view()),
                KernelArg::Buffer(f_w.view()),
                KernelArg::Buffer(lg.view()),
                KernelArg::Buffer(f_bounds.view()),
                KernelArg::Buffer(f_active.view()),
                KernelArg::Buffer(f_sorted.view()),
                KernelArg::Buffer(f_pos.view()),
                KernelArg::U32(n_tok as u32),
                KernelArg::U32(EXPERTS as u32),
                KernelArg::U32(USED as u32),
            ],
        );

        let want_w = gpu.read_f32(&w);
        let got_w = gpu.read_f32(&f_w);
        assert_eq!(ints(gpu, &f_ids), ints(gpu, &ids), "{n_tok} tokens: ids");
        assert!(
            got_w
                .iter()
                .zip(&want_w)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{n_tok} tokens: weights {got_w:?} vs {want_w:?}"
        );
        let bounds_v = ints(gpu, &bounds);
        assert_eq!(ints(gpu, &f_bounds), bounds_v, "{n_tok} tokens: bounds");
        let active_v = ints(gpu, &active);
        let n_active = active_v[EXPERTS] as usize;
        let f_active_v = ints(gpu, &f_active);
        assert_eq!(
            f_active_v[EXPERTS], active_v[EXPERTS],
            "{n_tok} tokens: active count"
        );
        assert_eq!(
            f_active_v[..n_active],
            active_v[..n_active],
            "{n_tok} tokens: active"
        );
        let (want_s, got_s, pos) = (ints(gpu, &sorted), ints(gpu, &f_sorted), ints(gpu, &f_pos));
        for e in 0..EXPERTS {
            let span = bounds_v[e] as usize..bounds_v[e + 1] as usize;
            let mut a = want_s[span.clone()].to_vec();
            let mut b = got_s[span].to_vec();
            a.sort_unstable();
            assert!(
                b.windows(2).all(|p| p[0] < p[1]),
                "{n_tok} tokens: expert {e} span is not by rank"
            );
            b.sort_unstable();
            assert_eq!(a, b, "{n_tok} tokens: expert {e}'s columns");
        }
        for (col, &at) in pos.iter().enumerate() {
            assert_eq!(
                got_s[at as usize], col as i32,
                "{n_tok} tokens: inverse of column {col}"
            );
        }
    }
}
