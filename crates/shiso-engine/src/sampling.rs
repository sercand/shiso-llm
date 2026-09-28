//! Token selection over one row of logits: the greedy path and the stage
//! dispatcher that applies top-k, top-p and min-p.
//!
//! Three invariants hold at every entry point here, and the integration tests
//! assert each one:
//!
//! * A stochastic sample consumes exactly one random draw, whichever stage
//!   produced it. A greedy sample consumes none at all, so a greedy caller
//!   cannot push a stochastic caller that shares its stream out of step.
//! * Ties resolve to the lower vocabulary id, both in the argmax scan and in
//!   every stage that orders candidates by probability.
//! * A non-finite logit is skipped by the stochastic stages but is never
//!   removed from the row, so the argmax fallback still names a token.
//!
//! Probabilities are always accumulated in index order and never renormalized
//! by a per-element division, because both the running sums and the reciprocal
//! multiply are part of the observable result.

use std::cmp::Ordering;

use shiso_core::{SamplingParams, Seed, TokenId};

/// The scan seed value: below every finite logit yet itself finite, so a row of
/// all-nonfinite logits still yields id zero rather than a sentinel.
pub(crate) const NEG_MAX: f32 = -1.0e30_f32;

/// Hard ceiling on the top-k stage, which is also the size of its scratch.
pub(crate) const TOP_K_LIMIT: usize = 1_024;

/// Largest candidate set the bounded-heap stage keeps.
const HEAP_LIMIT: usize = 512;

/// One candidate's contribution, kept together so a stage cannot mix a
/// probability with the id of a different candidate.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Candidate {
    /// Vocabulary id, already narrowed to a `usize` index.
    pub(crate) id: usize,
    /// The raw logit, which is what ordering uses.
    pub(crate) logit: f32,
    /// Unnormalized softmax weight against the row maximum.
    pub(crate) prob: f32,
}

/// Strict total order over candidates: higher logit first, lower id on a tie.
///
/// The tie rule is what makes the order total rather than merely a partial one,
/// so a sort of candidates has exactly one correct output sequence.
pub(crate) fn candidate_gt(a: Candidate, b: Candidate) -> bool {
    match a.logit.partial_cmp(&b.logit) {
        Some(ordering) => ordering.is_gt(),
        None => false,
    }
}

/// Restores the heap property below `idx`, exactly as the insertion path needs
/// it: the root is the weakest candidate the stage still holds.
pub(crate) fn sift_up(heap: &mut [Candidate], idx: usize) {
    let mut idx = idx;
    while idx > 0 {
        let parent = (idx - 1) / 2;
        if !candidate_gt(heap[parent], heap[idx]) {
            break;
        }
        heap.swap(parent, idx);
        idx = parent;
    }
}

/// Restores the heap property above `idx` for a heap of `n` entries.
pub(crate) fn sift_down(heap: &mut [Candidate], n: usize, idx: usize) {
    let mut idx = idx;
    loop {
        let left = idx * 2 + 1;
        let right = left + 1;
        let mut weakest = idx;
        if left < n && candidate_gt(heap[weakest], heap[left]) {
            weakest = left;
        }
        if right < n && candidate_gt(heap[weakest], heap[right]) {
            weakest = right;
        }
        if weakest == idx {
            break;
        }
        heap.swap(idx, weakest);
        idx = weakest;
    }
}

/// Greedy scan of `logits[begin..end]` against an incumbent.
///
/// Eight lanes let the comparisons overlap; they are merged in lane order
/// afterwards, and a lane wins only on a strict improvement or on an equal value
/// with a lower id. The result is therefore the lowest id that attains the
/// maximum, whichever lane happened to see it. A non-finite logit cannot beat an
/// incumbent, so it loses every comparison it takes part in.
fn argmax_lanes(logits: &[f32], begin: usize, end: usize, best: &mut usize, best_v: &mut f32) {
    let mut lane_id = [*best; 8];
    let mut lane_v = [*best_v; 8];
    let mut i = begin;
    while i + 8 <= end {
        for (k, v) in logits[i..i + 8].iter().copied().enumerate() {
            if v > lane_v[k] {
                lane_v[k] = v;
                lane_id[k] = i + k;
            }
        }
        i += 8;
    }
    for (id, v) in lane_id.iter().copied().zip(lane_v) {
        // The equality test is written as a pair of comparisons rather than `==`
        // so that a tie between `+0.0` and `-0.0` resolves to the lower id, as it
        // does in C. `total_cmp` would order those two values apart.
        #[expect(
            clippy::double_comparisons,
            reason = "the pair mirrors the C \
              `>=`/`<=` tie test that the port copies; `==` is the same value \
              test but this file keeps the source's operators"
        )]
        let wins = v > *best_v || (v >= *best_v && v <= *best_v && id < *best);
        if wins {
            *best_v = v;
            *best = id;
        }
    }
    while i < end {
        let v = logits[i];
        if v > *best_v {
            *best_v = v;
            *best = i;
        }
        i += 1;
    }
}

/// The greedy scan as an index, for the stages that fall back to argmax.
pub(crate) fn argmax_index(logits: &[f32]) -> usize {
    let mut best = 0_usize;
    let mut best_v = NEG_MAX;
    argmax_lanes(logits, 0, logits.len(), &mut best, &mut best_v);
    best
}

/// The greedy token: the lowest id attaining the highest finite logit.
///
/// An empty row and a row whose every entry is non-finite both yield id zero,
/// because the incumbent starts at id zero with a sentinel below every finite
/// logit. A caller that samples with `temperature <= 0` lands here, which is why
/// this path never touches the rng.
#[must_use]
pub fn argmax(logits: &[f32]) -> TokenId {
    TokenId::new(u32::try_from(argmax_index(logits)).unwrap_or(u32::MAX))
}

/// The row's largest finite logit, the lowest id attaining it, and how many
/// entries are finite.
///
/// Counting the finite entries is what lets a stage tell "this row has nothing
/// to sample" apart from "this row is sparse", and the strict comparison is what
/// makes the id the lowest one at the maximum.
fn row_summary(logits: &[f32]) -> (f32, usize, usize) {
    let mut max_logit = NEG_MAX;
    let mut best = 0_usize;
    let mut finite = 0_usize;
    for (idx, v) in logits.iter().copied().enumerate() {
        if !v.is_finite() {
            continue;
        }
        finite += 1;
        if v > max_logit {
            max_logit = v;
            best = idx;
        }
    }
    (max_logit, best, finite)
}

/// One row plus the knobs a stage reads, bundled because both stochastic stages
/// need the whole set and a tenth positional argument helps nobody.
struct Stage<'a> {
    /// The logits, in vocabulary order.
    logits: &'a [f32],
    /// How many entries of the row are finite.
    finite: usize,
    /// The largest finite logit.
    max_logit: f32,
    /// The lowest id attaining `max_logit`, which is the fallback token.
    best: usize,
    /// Softmax divisor, already known to be positive.
    temperature: f32,
    /// Cumulative-mass cutoff, already clamped into `(0, 1]`.
    top_p: f32,
    /// Floor as a fraction of the top probability; never negative.
    min_p: f32,
}

impl<'a> Stage<'a> {
    /// Scans the row once and pairs it with the knobs, which the caller has
    /// already clamped.
    fn new(logits: &'a [f32], temperature: f32, top_p: f32, min_p: f32) -> Self {
        let (max_logit, best, finite) = row_summary(logits);
        Self {
            logits,
            finite,
            max_logit,
            best,
            temperature,
            top_p,
            min_p,
        }
    }

    /// The min-p floor as a non-negative fraction, so a negative or non-finite
    /// floor degrades to no floor instead of rejecting everything.
    fn min_rel(&self) -> f32 {
        if self.min_p > 0.0 { self.min_p } else { 0.0 }
    }
}

/// The order `qsort` imposes on a candidate set: higher logit first, lower id on
/// a tie.
///
/// The comparator is total, so an unstable sort lands on the same sequence the C
/// build's `qsort` does; a partial one would leave the order of equal-probability
/// candidates up to the sort's internals.
pub(crate) fn cmp_desc(a: &Candidate, b: &Candidate) -> Ordering {
    match a.logit.partial_cmp(&b.logit) {
        Some(ordering) if ordering.is_ne() => ordering.reverse(),
        _ => a.id.cmp(&b.id),
    }
}

/// The bounded stage: keeps the strongest `HEAP_LIMIT` weights in a heap, so a
/// row whose mass is concentrated never sorts the whole vocabulary.
///
/// `None` means the stage cannot prove the answer and the caller must fall back
/// to the full-row stage. That happens when the heap holds fewer candidates than
/// the row and the mass it holds does not yet cover `top_p`, or when the floor
/// stopped inside the heap while the weakest entry still held could have cleared
/// it. Otherwise exactly one draw is consumed, and the returned id is the first
/// candidate whose running mass covers the draw.
fn fast_top_p(stage: &Stage<'_>, rng: &mut Seed) -> Option<usize> {
    if stage.finite > HEAP_LIMIT && stage.top_p >= 0.999 {
        return None;
    }
    let cap = stage.finite.min(HEAP_LIMIT);
    let mut heap: Vec<Candidate> = Vec::with_capacity(cap);
    let mut n = 0_usize;
    let mut sum = 0_f32;
    let mut heap_sum = 0_f32;
    let min_p = stage.min_rel();
    for (idx, v) in stage.logits.iter().copied().enumerate() {
        if !v.is_finite() {
            continue;
        }
        let p = ((v - stage.max_logit) / stage.temperature).exp();
        sum += p;
        let cand = Candidate {
            id: idx,
            logit: v,
            prob: p,
        };
        if n < cap {
            heap.push(cand);
            heap_sum += p;
            sift_up(&mut heap, n);
            n += 1;
        } else if candidate_gt(cand, heap[0]) {
            heap_sum -= heap[0].prob;
            heap[0] = cand;
            heap_sum += p;
            sift_down(&mut heap, n, 0);
        }
    }
    if sum <= 0.0 || !sum.is_finite() {
        return Some(stage.best);
    }
    if n < stage.finite && heap_sum < stage.top_p * sum {
        return None;
    }
    heap.sort_unstable_by(cmp_desc);
    let min_prob = (heap[0].prob / sum) * min_p;
    let min_prob_raw = heap[0].prob * min_p;
    let mut filtered_sum = 0_f32;
    let mut filtered = 0_usize;
    let mut stopped_by_min_p = false;
    for (i, cand) in heap.iter().enumerate() {
        let p = cand.prob / sum;
        if i > 0 && p < min_prob {
            stopped_by_min_p = true;
            break;
        }
        filtered_sum += cand.prob;
        filtered += 1;
        if filtered_sum / sum >= stage.top_p {
            break;
        }
    }
    if n < stage.finite && stopped_by_min_p && min_p > 0.0 && heap[n - 1].prob >= min_prob_raw {
        return None;
    }
    if filtered == 0 {
        return Some(stage.best);
    }
    let mut r = rng_f32(rng) * filtered_sum;
    for cand in heap.iter().take(filtered) {
        r -= cand.prob;
        if r <= 0.0 {
            return Some(cand.id);
        }
    }
    Some(heap[filtered - 1].id)
}

/// The scaled logit at which the `min_rel` floor is already certain to fail.
///
/// The floor is `exp(scaled) >= min_rel` and `exp` is monotonic, so the boundary
/// is `ln(min_rel)` nudged down until its own `exp` lands strictly below
/// `min_rel`: an entry at or below the nudged value cannot survive the floor, and
/// skipping it spares a wide vocabulary one `exp` per entry. The nudge, rather
/// than trust in `ln`, is what keeps the boundary on the safe side of rounding -
/// a boundary one unit too high would drop a survivor. `(NEG_MAX, false)` means no
/// boundary is usable and every entry pays the full test.
fn reject_boundary(min_rel: f32) -> (f32, bool) {
    if min_rel <= 0.0 || !min_rel.is_finite() {
        return (NEG_MAX, false);
    }
    let mut cutoff = min_rel.ln();
    for _ in 0..8 {
        if !cutoff.is_finite() {
            break;
        }
        cutoff = cutoff.next_down();
        if cutoff.exp() < min_rel {
            return (cutoff, true);
        }
    }
    (NEG_MAX, false)
}

/// The row's candidates and their total mass, or `None` when the row carries no
/// mass worth drawing from.
///
/// A `min_p` floor is applied to `p / sum`, so the normalization cancels and the
/// test is equivalent to `p >= min_p` - but the whole softmax sum is still
/// accumulated in index order first, because the draw is scaled by that same sum
/// and a different sum names a different token. Without a floor every finite
/// entry is a candidate, and the sum is accumulated in the same pass.
fn candidates(stage: &Stage<'_>) -> Option<(Vec<Candidate>, f32)> {
    let mut sum = 0_f32;
    let mut cand: Vec<Candidate> = Vec::new();
    if stage.min_p > 0.0 && stage.min_p <= 1.0 {
        let mut scratch = vec![-1.0_f32; stage.logits.len()];
        for (idx, v) in stage.logits.iter().copied().enumerate() {
            if !v.is_finite() {
                continue;
            }
            let p = ((v - stage.max_logit) / stage.temperature).exp();
            scratch[idx] = p;
            sum += p;
        }
        if sum <= 0.0 || !sum.is_finite() {
            return None;
        }
        let min_prob = (1.0 / sum) * stage.min_p;
        let survivors = scratch
            .iter()
            .filter(|p| **p >= 0.0 && **p / sum >= min_prob)
            .count();
        if survivors == 0 {
            return None;
        }
        cand.reserve(survivors);
        for (idx, p) in scratch.iter().copied().enumerate() {
            if p < 0.0 || p / sum < min_prob {
                continue;
            }
            cand.push(Candidate {
                id: idx,
                logit: stage.logits[idx],
                prob: p,
            });
        }
    } else {
        cand.reserve(stage.finite);
        for (idx, v) in stage.logits.iter().copied().enumerate() {
            if !v.is_finite() {
                continue;
            }
            let p = ((v - stage.max_logit) / stage.temperature).exp();
            cand.push(Candidate {
                id: idx,
                logit: v,
                prob: p,
            });
            sum += p;
        }
    }
    Some((cand, sum))
}

/// The full-row stage: every finite entry of the row takes part, in vocabulary
/// order.
///
/// `top_p >= 1` never sorts. With no nucleus to cut the row, min-p is the only
/// filter left, so the stage walks the row once in log space and keeps a
/// rejection boundary derived from the same `exp` the probabilities use; a value
/// at or below that boundary is guaranteed to fail the floor, which spares a wide
/// vocabulary an `exp` per entry. Otherwise the bounded stage has already declined
/// and the row is sorted in full.
fn full_vocab(stage: &Stage<'_>, rng: &mut Seed) -> usize {
    if stage.finite == 0 {
        return argmax_index(stage.logits);
    }
    if stage.top_p < 1.0
        && let Some(id) = fast_top_p(stage, rng)
    {
        return id;
    }
    let min_rel = stage.min_rel();
    if stage.top_p >= 1.0 {
        if min_rel > 1.0 {
            return stage.best;
        }
        let (reject_scaled, have_reject) = reject_boundary(min_rel);
        let mut scratch = vec![-1.0_f32; stage.logits.len()];
        let mut sum = 0_f32;
        for (idx, v) in stage.logits.iter().copied().enumerate() {
            if !v.is_finite() {
                continue;
            }
            let scaled = (v - stage.max_logit) / stage.temperature;
            if have_reject && scaled <= reject_scaled {
                continue;
            }
            let p = scaled.exp();
            if p < min_rel {
                continue;
            }
            scratch[idx] = p;
            sum += p;
        }
        if sum <= 0.0 || !sum.is_finite() {
            return stage.best;
        }
        let mut r = rng_f32(rng) * sum;
        for (idx, p) in scratch.iter().copied().enumerate() {
            if p < 0.0 {
                continue;
            }
            r -= p;
            if r <= 0.0 {
                return idx;
            }
        }
        return stage.best;
    }

    let Some((mut cand, sum)) = candidates(stage) else {
        return stage.best;
    };
    if sum <= 0.0 || !sum.is_finite() {
        return stage.best;
    }
    cand.sort_unstable_by(cmp_desc);
    let min_prob = (cand[0].prob / sum) * min_rel;
    let mut filtered_sum = 0_f32;
    let mut filtered = 0_usize;
    for (i, cand) in cand.iter().enumerate() {
        let p = cand.prob / sum;
        if i > 0 && p < min_prob {
            break;
        }
        filtered_sum += cand.prob;
        filtered += 1;
        if filtered_sum / sum >= stage.top_p {
            break;
        }
    }
    if filtered == 0 {
        return stage.best;
    }
    let mut r = rng_f32(rng) * filtered_sum;
    for cand in cand.iter().take(filtered) {
        r -= cand.prob;
        if r <= 0.0 {
            return cand.id;
        }
    }
    cand[filtered - 1].id
}

/// The top-k stage: the strongest `top_k` logits are kept in a descending
/// insertion array, and the softmax runs over that set alone.
///
/// The array is sorted, which is what lets the stage drop a candidate by
/// comparing it against the last entry alone, and it is why `max_logit` here is
/// `vals[0]` rather than the row maximum: the temperature is applied to the kept
/// set, so the row's largest entry outside the set cannot shift the distribution.
fn sample_top_k(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    rng: &mut Seed,
) -> usize {
    let mut ids = vec![0_usize; top_k];
    let mut vals = vec![NEG_MAX; top_k];
    let mut n = 0_usize;
    for (idx, v) in logits.iter().copied().enumerate() {
        if !v.is_finite() {
            continue;
        }
        if n == top_k && v <= vals[n - 1] {
            continue;
        }
        let mut j = if n < top_k {
            n += 1;
            n - 1
        } else {
            n - 1
        };
        while j > 0 && vals[j - 1] < v {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
            j -= 1;
        }
        vals[j] = v;
        ids[j] = idx;
    }
    if n == 0 {
        return argmax_index(logits);
    }
    let max_logit = vals[0];
    let mut probs = vec![0_f32; n];
    let mut sum = 0_f32;
    for (prob, val) in probs.iter_mut().zip(vals.iter()).take(n) {
        *prob = ((val - max_logit) / temperature).exp();
        sum += *prob;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return ids[0];
    }
    let min_prob = (probs[0] / sum) * min_p;
    let mut filtered_sum = 0_f32;
    let mut filtered = 0_usize;
    for (i, &row) in probs.iter().enumerate().take(n) {
        let p = row / sum;
        if i > 0 && p < min_prob {
            break;
        }
        filtered_sum += row;
        filtered += 1;
        if filtered_sum / sum >= top_p {
            break;
        }
    }
    if filtered == 0 {
        return ids[0];
    }
    let mut r = rng_f32(rng) * filtered_sum;
    for i in 0..filtered {
        r -= probs[i];
        if r <= 0.0 {
            return ids[i];
        }
    }
    ids[filtered - 1]
}

/// The token sampled from one row of logits.
///
/// Stage order is the C sampler's and is observable: `temperature <= 0` returns
/// [`argmax`] without clamping anything or advancing `rng`; the clamps that
/// follow are `top_p` out of `(0, 1]` to `1`, a negative `min_p` to `0`, and
/// `top_k` to the smaller of [`TOP_K_LIMIT`] and the row length. A stochastic
/// stage consumes exactly one draw, including the stages that fall back to the
/// row maximum, so a caller that keeps one [`Seed`] per session can replay a
/// generation by seed alone.
///
/// The full-vocabulary stage needs a row-length scratch buffer. It is allocated
/// here rather than passed in, because the signature every caller already agrees
/// on has no place for it; a `top_k` sample never allocates one.
#[must_use]
pub fn sample_logits(logits: &[f32], p: &SamplingParams, rng: &mut Seed) -> TokenId {
    let temperature = p.temperature;
    if temperature <= 0.0 {
        return argmax(logits);
    }
    let mut top_p = p.top_p;
    if top_p <= 0.0 || top_p > 1.0 {
        top_p = 1.0;
    }
    let mut min_p = p.min_p;
    if min_p < 0.0 {
        min_p = 0.0;
    }
    let top_k = p.top_k.map_or(0, |k| {
        usize::try_from(k.get())
            .unwrap_or(usize::MAX)
            .min(TOP_K_LIMIT)
    });
    let top_k = top_k.min(logits.len());
    let id = if top_k == 0 {
        let stage = Stage::new(logits, temperature, top_p, min_p);
        full_vocab(&stage, rng)
    } else {
        sample_top_k(logits, temperature, top_k, top_p, min_p, rng)
    };
    TokenId::new(u32::try_from(id).unwrap_or(u32::MAX))
}

/// Advances the sampler stream and returns the scrambled output (xorshift64\*).
pub fn rng_next(rng: &mut Seed) -> u64 {
    rng.next_u64()
}

/// The next draw in `[0, 1)`, from the 24 high bits of [`rng_next`].
pub fn rng_f32(rng: &mut Seed) -> f32 {
    rng.next_f32()
}
#[cfg(test)]
mod tests {
    use super::{Stage, candidates, reject_boundary, sample_logits};
    use shiso_core::{SamplingParams, Seed, TokenId};

    #[test]
    fn the_rejection_boundary_is_strictly_below_the_floor() {
        let (boundary, usable) = reject_boundary(0.1);
        assert!(usable, "a finite floor in range yields a boundary");
        assert!(
            boundary.exp() < 0.1,
            "the boundary is only safe while its own exp lands below the floor"
        );

        for unusable in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let (_, usable) = reject_boundary(unusable);
            assert!(!usable, "{unusable} cannot bound a floor test");
        }
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "the floor drops no mass here, so the two sums must be the same bits; a tolerance would accept a perturbed sum, which names a different token"
    )]
    fn a_floor_narrows_the_candidates_without_moving_the_sum() {
        // p / sum is about 0.952, 0.047 and 0.00012 here, so a 0.2 floor leaves
        // one survivor while the sum stays the whole row's: the draw is scaled by
        // that sum, and a different sum names a different token.
        let logits = [3.0_f32, 0.0, -6.0];

        let floored = Stage::new(&logits, 1.0, 1.0, 0.2);
        let (kept, kept_sum) = candidates(&floored).expect("the row carries mass");
        let ids: Vec<usize> = kept.iter().map(|entry| entry.id).collect();
        assert_eq!(ids, vec![0], "only the top entry clears a 0.2 floor");

        let open = Stage::new(&logits, 1.0, 1.0, 0.0);
        let (all, open_sum) = candidates(&open).expect("the row carries mass");
        assert_eq!(all.len(), 3, "with no floor every finite entry survives");
        assert_eq!(
            kept_sum, open_sum,
            "the floor must not move the sum the draw is scaled by"
        );
    }

    #[test]
    fn a_floored_row_never_draws_the_entry_below_the_floor() {
        let logits = [3.0_f32, 0.0, -6.0];
        let params = SamplingParams {
            min_p: 0.2,
            ..SamplingParams::neutral()
        };
        for state in [1_u64, 42, 99, 12_345, u64::MAX] {
            let picked = sample_logits(&logits, &params, &mut Seed::new(state));
            assert_eq!(picked, TokenId::new(0), "state {state} left the floor");
        }
    }
}

/// The token drawn from one row of logits by a rule that depends only on the
/// row, the key and the token's position in the sequence.
///
/// Over the candidates the request allows - its top-k, cut to its top-p
/// nucleus and its min-p floor - the token is the argmax of
/// `logit / temperature + G(key, position, id)`, with `G` a Gumbel variate from
/// a hash of the three. That is an exact draw from the softmax over those
/// candidates, and it is the same draw whichever pass computed the row: a
/// one-token step, or row `i` of a speculative verify. So a drafted token is
/// kept exactly when it is the token this rule picks at its position, and a
/// sampled sequence decoded speculatively is the one decoded a token at a time.
///
/// `temperature <= 0` is the argmax, as for [`sample_logits`]. Nothing is
/// consumed from a stream: the key is fixed for the sequence.
#[must_use]
pub fn keyed_sample(logits: &[f32], p: &SamplingParams, key: u64, position: u32) -> TokenId {
    let temperature = p.temperature;
    if temperature <= 0.0 || logits.is_empty() {
        return argmax(logits);
    }
    let top_p = if p.top_p <= 0.0 || p.top_p > 1.0 {
        1.0
    } else {
        p.top_p
    };
    let min_p = p.min_p.max(0.0);
    let top_k = p.top_k.map_or(0, |k| {
        usize::try_from(k.get())
            .unwrap_or(usize::MAX)
            .min(TOP_K_LIMIT)
    });
    let top_k = top_k.min(logits.len());
    let noise = |id: usize| -> f32 {
        let u = keyed_uniform(key, position, u32::try_from(id).unwrap_or(u32::MAX));
        -(-u.ln()).ln()
    };
    let inv_t = 1.0 / temperature;
    // Pure temperature sampling: the Gumbel argmax over the whole row.
    if top_k == 0 && top_p >= 1.0 && min_p == 0.0 {
        let mut best = (f32::NEG_INFINITY, 0usize);
        for (id, &l) in logits.iter().enumerate() {
            let score = l * inv_t + noise(id);
            if score > best.0 {
                best = (score, id);
            }
        }
        return TokenId::new(u32::try_from(best.1).unwrap_or(u32::MAX));
    }
    // The candidates, highest first and ties to the lower id: the top-k, or
    // every token within reach of the nucleus. A token more than 40
    // temperature-scaled units below the maximum carries under e^-40 of its
    // mass, which no nucleus below 1 and no min-p floor keeps.
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut cands: Vec<(f32, u32)> = if top_k > 0 {
        let mut all: Vec<(f32, u32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (l, u32::try_from(i).unwrap_or(u32::MAX)))
            .collect();
        let k = top_k.min(all.len());
        all.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        all.truncate(k);
        all
    } else {
        logits
            .iter()
            .enumerate()
            .filter(|&(_, &l)| (l - max) * inv_t > -40.0)
            .map(|(i, &l)| (l, u32::try_from(i).unwrap_or(u32::MAX)))
            .collect()
    };
    cands.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let top = cands.first().map_or(max, |c| c.0);
    let weights: Vec<f64> = cands
        .iter()
        .map(|&(l, _)| f64::from((l - top) * inv_t).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let mut keep = cands.len();
    if top_p < 1.0 {
        let mut cum = 0.0;
        for (i, w) in weights.iter().enumerate() {
            cum += w / total;
            if cum >= f64::from(top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    if min_p > 0.0 {
        let floor = f64::from(min_p) * weights.first().copied().unwrap_or(1.0);
        keep = keep.min(weights.iter().take_while(|&&w| w >= floor).count().max(1));
    }
    let mut best = (f32::NEG_INFINITY, cands.first().map_or(0, |c| c.1));
    for &(l, id) in cands.iter().take(keep.max(1)) {
        let score = l * inv_t + noise(id as usize);
        if score > best.0 {
            best = (score, id);
        }
    }
    TokenId::new(best.1)
}

/// A uniform in `(0, 1)` from `(key, position, id)`: three rounds of the
/// splitmix64 finalizer, 24 bits, offset half a step off both ends.
pub(crate) fn keyed_uniform(key: u64, position: u32, id: u32) -> f32 {
    const fn mix(mut x: u64) -> u64 {
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }
    let mut x = mix(key.wrapping_add(0x9E37_79B9_7F4A_7C15));
    x = mix(x ^ u64::from(position).wrapping_mul(0xD1B5_4A32_D192_ED03));
    x = mix(x ^ u64::from(id));
    #[expect(
        clippy::cast_precision_loss,
        reason = "a 24-bit integer is exact in f32"
    )]
    let bits = (x >> 40) as f32;
    (bits + 0.5) * (1.0 / 16_777_216.0)
}

#[cfg(test)]
#[expect(clippy::cast_precision_loss, reason = "small fixture values")]
mod keyed_tests {
    use super::keyed_sample;
    use shiso_core::{SamplingParams, TokenId};

    fn params(temperature: f32, top_k: u32, top_p: f32) -> SamplingParams {
        let mut p = SamplingParams::neutral();
        p.temperature = temperature;
        p.top_k = std::num::NonZeroU32::new(top_k);
        p.top_p = top_p;
        p
    }

    #[test]
    fn the_draw_depends_on_the_row_and_the_position_only() {
        let row: Vec<f32> = (0..64).map(|i| ((i * 7) % 13) as f32 * 0.3).collect();
        let p = params(1.0, 20, 0.95);
        let a = keyed_sample(&row, &p, 42, 100);
        assert_eq!(
            a,
            keyed_sample(&row, &p, 42, 100),
            "same inputs, same token"
        );
        let spread: std::collections::HashSet<TokenId> = (0..200)
            .map(|pos| keyed_sample(&row, &p, 42, pos))
            .collect();
        assert!(
            spread.len() > 3,
            "positions draw different tokens: {spread:?}"
        );
    }

    #[test]
    fn the_draws_follow_the_distribution() {
        // Two tokens at logits ln 3 and 0: probabilities 0.75 and 0.25.
        let row = vec![3f32.ln(), 0.0];
        let p = params(1.0, 0, 1.0);
        let n = 20_000u32;
        let firsts = (0..n)
            .filter(|&pos| keyed_sample(&row, &p, 7, pos) == TokenId::new(0))
            .count();
        #[expect(clippy::cast_precision_loss, reason = "a small count")]
        let share = firsts as f64 / f64::from(n);
        assert!((share - 0.75).abs() < 0.015, "share {share}");
    }

    #[test]
    fn the_nucleus_and_top_k_are_respected() {
        let row = vec![5.0, 4.9, 0.0, -1.0, -2.0];
        for pos in 0..500 {
            let t = keyed_sample(&row, &params(1.0, 2, 1.0), 3, pos).get();
            assert!(t < 2, "top-k 2 drew {t}");
            let t = keyed_sample(&row, &params(1.0, 0, 0.5), 3, pos).get();
            assert!(t < 2, "a 0.5 nucleus drew {t}");
        }
        assert_eq!(
            keyed_sample(&row, &params(0.0, 0, 1.0), 3, 0),
            TokenId::new(0)
        );
    }
}
