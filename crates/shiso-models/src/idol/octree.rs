//! The octree descent: how many anchors land in each cell.
//!
//! The decoder answers, for every cell at a level, how the points inside it
//! split among its eight children. The split is a **systematic resampling**,
//! not a multinomial draw: one uniform offset in `[0, 1/n)` is drawn per cell,
//! and the `n` stratified points `offset + k/n` are each placed in the child
//! whose cumulative probability covers them. That is what makes a cell with
//! probability `0.5` and ten points receive exactly five, every time, which a
//! multinomial draw would not.

use super::draws::Draws;

/// Children per cell.
pub const CHILDREN: usize = 8;

/// The child probabilities of every cell at one level, as the decoder answers
/// them, or the decoder's failure. A named type because the closure is passed
/// across a module boundary.
pub type ProbsAt<'a> = dyn FnMut(&[[u32; 3]], u32) -> Result<Vec<[f32; CHILDREN]>, String> + 'a;

/// One cell of the octree: its integer coordinates at its own level, and how
/// many of the points still inside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    /// Coordinates at this level, each in `0..2^level`.
    pub coords: [u32; 3],
    /// Points that landed in this cell.
    pub count: u32,
}

/// The child offsets, in the reference's order: `i` fastest, then `j`, then
/// `k`, so child index `i + 2*j + 4*k`.
const CHILD_OFFSETS: [[u32; 3]; CHILDREN] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

/// Splits `count` points among the eight children of one cell.
///
/// `probs` are the child probabilities as the decoder answered them: clamped at
/// zero and normalised here, with an all-zero row falling back to uniform,
/// which is the reference's rule for a cell whose logits underflowed.
///
/// The draws are written into `counts`, which is zeroed first.
#[expect(
    clippy::cast_precision_loss,
    reason = "the point count is the anchor budget, which the request bounds well below 2^24"
)]
pub fn systematic_sample(
    probs: &[f32; CHILDREN],
    count: u32,
    draws: &mut dyn Draws,
    counts: &mut [u32; 8],
) {
    counts.fill(0);
    if count == 0 {
        return;
    }
    let mut normalised = [0.0f32; CHILDREN];
    let mut total = 0.0f32;
    for (slot, probability) in normalised.iter_mut().zip(probs) {
        *slot = probability.max(0.0);
        total += *slot;
    }
    if total <= 0.0 {
        normalised = [1.0 / CHILDREN as f32; CHILDREN];
    } else {
        for slot in &mut normalised {
            *slot /= total;
        }
    }
    // The cumulative distribution, clamped just below one so the last
    // stratified point can never fall past the final bin.
    let mut cdf = [0.0f32; CHILDREN];
    let mut running = 0.0f32;
    for (slot, probability) in cdf.iter_mut().zip(&normalised) {
        running += *probability;
        *slot = running.min(1.0 - 1e-12);
    }

    let count_f = count as f32;
    let offset = draws.uniform() / count_f;
    for k in 0..count {
        let point = (offset + k as f32 / count_f).min(1.0 - 1e-12);
        // The first bin whose cumulative probability covers the point.
        let child = cdf
            .iter()
            .position(|edge| *edge >= point)
            .unwrap_or(CHILDREN - 1);
        counts[child] += 1;
    }
}

/// Descends the octree from one root cell to `level`, placing `num_anchors`
/// points, and returns their positions in `[0, 1]^3`.
///
/// `probs_at` answers the child probabilities for every cell at a level, in the
/// order the cells are handed to it. It is a closure rather than a tensor
/// because the probabilities come from the device and the descent is host
/// arithmetic: the two are joined here, not in the graph.
///
/// Each point ends at a uniformly random position inside its leaf cell, which
/// is the reference's last step and the reason the output is continuous rather
/// than a lattice.
///
/// # Errors
///
/// The decoder's failure, or a decoder that answers a different number of
/// distributions than there are cells.
#[expect(
    clippy::cast_precision_loss,
    reason = "the anchor count and the leaf coordinates are bounded by the request's Gaussian \
              ceiling and the octree's eight levels"
)]
pub fn place_anchors(
    num_anchors: u32,
    level: u32,
    draws: &mut dyn Draws,
    probs_at: &mut ProbsAt<'_>,
) -> Result<Vec<[f32; 3]>, String> {
    let mut cells = vec![Cell {
        coords: [0, 0, 0],
        count: num_anchors,
    }];
    for depth in 1..=level {
        let coords: Vec<[u32; 3]> = cells.iter().map(|cell| cell.coords).collect();
        let probabilities = probs_at(&coords, depth)?;
        if probabilities.len() != cells.len() {
            return Err(format!(
                "the octree decoder answered {} distributions for {} cells at level {depth}",
                probabilities.len(),
                cells.len()
            ));
        }
        let mut next = Vec::with_capacity(cells.len() * CHILDREN);
        for (cell, probs) in cells.iter().zip(&probabilities) {
            let mut counts = [0u32; CHILDREN];
            systematic_sample(probs, cell.count, draws, &mut counts);
            for (child, count) in counts.into_iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let offset = CHILD_OFFSETS[child];
                next.push(Cell {
                    coords: [
                        cell.coords[0] * 2 + offset[0],
                        cell.coords[1] * 2 + offset[1],
                        cell.coords[2] * 2 + offset[2],
                    ],
                    count,
                });
            }
        }
        cells = next;
    }

    let side = (1u32 << level) as f32;
    let mut anchors = Vec::with_capacity(num_anchors as usize);
    for cell in &cells {
        for _ in 0..cell.count {
            anchors.push([
                (cell.coords[0] as f32 + draws.uniform()) / side,
                (cell.coords[1] as f32 + draws.uniform()) / side,
                (cell.coords[2] as f32 + draws.uniform()) / side,
            ]);
        }
    }
    Ok(anchors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idol::draws::SeededDraws;
    use shiso_core::ids::Seed;

    fn rng() -> SeededDraws {
        SeededDraws::new(Seed::new(7))
    }

    #[test]
    fn all_the_mass_in_one_child_puts_every_point_there() {
        let mut counts = [0u32; 8];
        let mut probs = [0.0f32; 8];
        probs[3] = 1.0;
        systematic_sample(&probs, 10, &mut rng(), &mut counts);
        assert_eq!(counts[3], 10);
        assert_eq!(counts.iter().sum::<u32>(), 10);
    }

    #[test]
    fn an_even_split_is_exact_not_a_draw() {
        // Systematic resampling is deterministic given the offset: half the
        // points go to each of the two equal children, whatever the offset.
        for seed in 0..8 {
            let mut counts = [0u32; 8];
            let mut probs = [0.0f32; 8];
            probs[0] = 0.5;
            probs[1] = 0.5;
            systematic_sample(
                &probs,
                10,
                &mut SeededDraws::new(Seed::new(seed)),
                &mut counts,
            );
            assert_eq!((counts[0], counts[1]), (5, 5), "seed {seed}");
        }
    }

    #[test]
    fn a_skewed_split_is_proportional_to_within_one() {
        let mut counts = [0u32; 8];
        let mut probs = [0.0f32; 8];
        probs[0] = 0.7;
        probs[1] = 0.3;
        systematic_sample(&probs, 100, &mut rng(), &mut counts);
        assert!(
            counts[0].abs_diff(70) <= 1,
            "70 of 100 should land in the first child, got {}",
            counts[0]
        );
        assert_eq!(counts.iter().sum::<u32>(), 100);
    }

    #[test]
    fn an_all_zero_row_falls_back_to_uniform() {
        let mut counts = [0u32; 8];
        systematic_sample(&[0.0; 8], 8, &mut rng(), &mut counts);
        assert!(counts.iter().all(|count| *count == 1), "{counts:?}");
    }

    #[test]
    fn negative_probabilities_are_clamped_not_subtracted() {
        let mut counts = [0u32; 8];
        let mut probs = [0.0f32; 8];
        probs[0] = -1.0;
        probs[1] = 1.0;
        systematic_sample(&probs, 5, &mut rng(), &mut counts);
        assert_eq!(counts[1], 5);
    }

    #[test]
    fn a_zero_count_draws_nothing() {
        let mut counts = [9u32; 8];
        systematic_sample(&[0.125; 8], 0, &mut rng(), &mut counts);
        assert_eq!(counts, [0; 8]);
    }

    #[test]
    fn the_descent_lands_every_anchor_in_the_cell_the_mass_names() {
        // Every level puts all the mass in child 7, the top corner, so every
        // anchor must land inside the top corner cell of the leaf level.
        let mut probs_at = |coords: &[[u32; 3]], _level: u32| {
            let mut probs = [0.0f32; 8];
            probs[7] = 1.0;
            Ok(vec![probs; coords.len()])
        };
        let anchors = place_anchors(16, 4, &mut rng(), &mut probs_at).expect("places");
        assert_eq!(anchors.len(), 16);
        let side = 16.0;
        for anchor in &anchors {
            for axis in anchor {
                assert!(
                    *axis >= 15.0 / side && *axis < 1.0,
                    "anchor {axis} is outside the top corner cell"
                );
            }
        }
    }

    #[test]
    fn the_descent_keeps_the_anchor_count_exact() {
        // A uniform split at every level: the counts must still sum to the
        // request, with no point lost to rounding.
        let mut probs_at = |coords: &[[u32; 3]], _level: u32| Ok(vec![[0.125f32; 8]; coords.len()]);
        let anchors = place_anchors(37, 3, &mut rng(), &mut probs_at).expect("places");
        assert_eq!(anchors.len(), 37);
        for anchor in &anchors {
            for axis in anchor {
                assert!(
                    (0.0..1.0).contains(axis),
                    "anchor {axis} is outside the cube"
                );
            }
        }
    }

    #[test]
    fn the_jitter_keeps_points_inside_their_cell() {
        // Two children at the first level, then all the mass in child 0: an
        // anchor is in the first quarter or the third, never the second or
        // fourth, which is exactly what "inside its leaf cell" means here.
        let mut probs_at = |coords: &[[u32; 3]], level: u32| {
            let mut probs = [0.0f32; 8];
            if level == 1 {
                probs[0] = 0.5;
                probs[1] = 0.5;
            } else {
                probs[0] = 1.0;
            }
            Ok(vec![probs; coords.len()])
        };
        let anchors = place_anchors(20, 2, &mut rng(), &mut probs_at).expect("places");
        assert_eq!(anchors.len(), 20);
        for anchor in &anchors {
            let x = anchor[0];
            assert!(
                (0.0..0.25).contains(&x) || (0.5..0.75).contains(&x),
                "anchor x {x} left the leaf cell it was assigned to"
            );
        }
    }
}
