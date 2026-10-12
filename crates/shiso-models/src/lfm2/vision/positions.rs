//! The learned position grid, resampled onto one picture's patch grid.
//!
//! The tower learns one square grid of position vectors (16 x 16 for the
//! shipped checkpoint) and a picture's patch grid is whatever its aspect ratio
//! produced, from 8 x 32 to 32 x 8 and everything between. The reference
//! resizes the grid with a bilinear filter that is *antialiased*: when an axis
//! shrinks, the filter widens with the ratio and averages every table row it
//! covers, rather than sampling the two nearest. A plain bilinear sample - the
//! Qwen tower's - agrees with it whenever an axis grows and disagrees whenever
//! one shrinks, and the shipped band shrinks one axis of every picture whose
//! aspect ratio passes 1:4. The difference is not noise: it is a different
//! position vector on every patch of that axis, and the model answers fluently
//! about a picture whose layout it misreads.
//!
//! So the filter is written out here in the reference's own terms and run on
//! the host. It is two small passes over a table of a few hundred rows against
//! a tower that is about to run 27 blocks over the result, and keeping it in
//! plain Rust keeps the easy-to-get-silently-wrong part in code a unit test
//! reads.
//!
//! # The weights are the reference's, operation for operation
//!
//! For output index `i` of an axis resized from `input` to `output` values,
//! with `scale = input / output`:
//!
//! ```text
//! support  = scale if scale >= 1 else 1        (the triangle's half width)
//! invscale = 1 / scale if scale >= 1 else 1
//! center   = scale * (i + 0.5)
//! first    = max(trunc(center - support + 0.5), 0)
//! last     = min(trunc(center + support + 0.5), input)        (exclusive)
//! w[j]     = max(0, 1 - |(j - center + 0.5) * invscale|),  j in first..last
//! ```
//!
//! and the weights are divided by their sum. Everything is single precision,
//! as the reference's float path is, and the width axis is resized before the
//! height axis, as the reference's separable kernel does; an axis whose size
//! does not change is copied, not filtered.

/// One axis's resampling taps: for each output index, the first input index
/// and the normalized weights from there on.
#[derive(Debug, Clone, PartialEq)]
struct Taps {
    first: Vec<usize>,
    weights: Vec<Vec<f32>>,
}

impl Taps {
    /// The antialiased triangle filter's taps for `input` values resized to
    /// `output`.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "the axes are a few dozen values; the truncations are the reference's own int64 casts of non-negative sums"
    )]
    fn new(input: usize, output: usize) -> Self {
        let scale = input as f32 / output as f32;
        let (support, invscale) = if scale >= 1.0 {
            (scale, 1.0 / scale)
        } else {
            (1.0, 1.0)
        };
        let mut first = Vec::with_capacity(output);
        let mut weights = Vec::with_capacity(output);
        for i in 0..output {
            let center = scale * (i as f32 + 0.5);
            // `trunc`, as the reference's cast to an integer is, and clamped at
            // zero after: the lower edge can go negative near the first index.
            let lo = (center - support + 0.5).trunc().max(0.0) as usize;
            let hi = ((center + support + 0.5).trunc().max(0.0) as usize).min(input);
            let mut w: Vec<f32> = (lo..hi)
                .map(|j| {
                    let x = (j as f32 - center + 0.5) * invscale;
                    (1.0 - x.abs()).max(0.0)
                })
                .collect();
            let total: f32 = w.iter().sum();
            if total != 0.0 {
                for value in &mut w {
                    *value /= total;
                }
            }
            first.push(lo);
            weights.push(w);
        }
        Self { first, weights }
    }
}

/// Resizes `table`, `[side][side][width]` row-major (row `y * side + x` is
/// the vector of grid cell `(y, x)`), to `[height][width_cells][width]`.
///
/// Returns `height * width_cells` rows of `width` values, row-major over the
/// target grid - the order the patch rows arrive in.
#[must_use]
pub(super) fn resample(
    table: &[f32],
    side: usize,
    width: usize,
    height: usize,
    width_cells: usize,
) -> Vec<f32> {
    // Width first: `[side][width_cells][width]`.
    let across = if width_cells == side {
        table.to_vec()
    } else {
        let taps = Taps::new(side, width_cells);
        let mut out = vec![0.0f32; side * width_cells * width];
        for y in 0..side {
            for x in 0..width_cells {
                let dst = (y * width_cells + x) * width;
                for (k, &w) in taps.weights[x].iter().enumerate() {
                    let src = (y * side + taps.first[x] + k) * width;
                    for d in 0..width {
                        out[dst + d] += w * table[src + d];
                    }
                }
            }
        }
        out
    };
    // Then height: `[height][width_cells][width]`.
    if height == side {
        return across;
    }
    let taps = Taps::new(side, height);
    let mut out = vec![0.0f32; height * width_cells * width];
    for y in 0..height {
        for (k, &w) in taps.weights[y].iter().enumerate() {
            let src_row = taps.first[y] + k;
            for x in 0..width_cells {
                let dst = (y * width_cells + x) * width;
                let src = (src_row * width_cells + x) * width;
                for d in 0..width {
                    out[dst + d] += w * across[src + d];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{Taps, resample};

    #[test]
    fn halving_an_axis_averages_four_neighbours() {
        // A 2x reduction widens the triangle to cover four inputs at
        // 1/8, 3/8, 3/8, 1/8 - the antialiasing a plain bilinear sample
        // would not do - and the first output clips its left neighbour.
        let taps = Taps::new(16, 8);
        assert_eq!(taps.first[3], 5);
        assert_eq!(taps.weights[3], vec![0.125, 0.375, 0.375, 0.125]);
        assert_eq!(taps.first[0], 0);
        let w0 = &taps.weights[0];
        assert_eq!(w0.len(), 3);
        assert!((w0[0] - 3.0 / 7.0).abs() < 1e-6, "{w0:?}");
    }

    #[test]
    fn growing_an_axis_is_plain_bilinear_with_clamped_edges() {
        // 16 -> 32: every output sits a quarter step off an input centre.
        let taps = Taps::new(16, 32);
        assert_eq!(taps.first[0], 0);
        assert_eq!(taps.weights[0], vec![1.0]);
        assert_eq!(taps.first[5], 2);
        assert_eq!(taps.weights[5], vec![0.75, 0.25]);
        assert_eq!(*taps.weights.last().expect("taps"), vec![1.0]);
    }

    #[test]
    fn an_unchanged_grid_is_the_table() {
        let table: Vec<f32> = (0..16u16 * 16 * 3).map(|v| f32::from(v) * 0.5).collect();
        assert_eq!(resample(&table, 16, 3, 16, 16), table);
    }

    /// One channel, `table[y][x] = sin(0.37 y) + cos(0.21 x) + 0.01 x y`,
    /// resized by transformers 5.19.0 / torch 2.14.1's own
    /// `F.interpolate(mode="bilinear", align_corners=False, antialias=True)`
    /// on the CPU in f32. The four target grids cover both axes shrinking,
    /// both growing, and one of each.
    #[test]
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "a 16x16 fixture's indices; the angles are formed in double and narrowed, as the script that produced the reference did"
    )]
    fn resample_matches_the_reference_filter() {
        let mut table = vec![0.0f32; 256];
        for y in 0..16 {
            for x in 0..16 {
                let (yf, xf) = (y as f32, x as f32);
                let a = (0.37f64 * y as f64) as f32;
                let b = (0.21f64 * x as f64) as f32;
                table[y * 16 + x] = a.sin() + b.cos() + 0.01 * xf * yf;
            }
        }
        // (height, width, [first 3 values, last value]) from the reference.
        let cases: [(usize, usize, [f32; 4]); 4] = REFERENCE;
        for (h, w, want) in cases {
            let got = resample(&table, 16, 1, h, w);
            assert_eq!(got.len(), h * w);
            let picks = [got[0], got[1], got[2], got[h * w - 1]];
            for (g, e) in picks.iter().zip(want) {
                assert!((g - e).abs() < 2e-6, "{h}x{w}: {picks:?} vs {want:?}");
            }
        }
    }

    /// The reference's values; see the case above.
    #[expect(
        clippy::excessive_precision,
        clippy::unreadable_literal,
        reason = "the reference's printed values, kept verbatim"
    )]
    const REFERENCE: [(usize, usize, [f32; 4]); 4] = [
        (
            8,
            32,
            [
                1.2513049840927124,
                1.24759840965271,
                1.240185260772705,
                0.3313348591327667,
            ],
        ),
        (
            32,
            8,
            [
                0.9781688451766968,
                0.8511053919792175,
                0.5761182904243469,
                0.4942421615123749,
            ],
        ),
        (
            12,
            20,
            [
                1.1084846258163452,
                1.0952062606811523,
                1.058544635772705,
                0.4687731862068176,
            ],
        ),
        (
            24,
            24,
            [
                1.0,
                0.9890154600143433,
                0.9672072529792786,
                0.580795407295227,
            ],
        ),
    ];
}
