//! The distribution the speculative-decoding path samples from.
//!
//! [`crate::sample_logits`] answers "which token"; this module answers "with
//! what probability", because acceptance-rejection needs the target distribution
//! and the draft distribution as arrays rather than as one draw each. The filter
//! stages are the ones the sampler runs, in the same order and with the same
//! running sums, so a token drawn from `probs` by [`sample_probabilities`] is
//! distributed exactly as one drawn by `sample_logits` under the same knobs.
//!
//! The caller owns `probs`, which must be at least as long as the row: a draft
//! step builds two distributions, and a per-step allocation in the decode loop is
//! the first cost that shows up on a token budget. `false` means no distribution
//! could be built at all - an empty row, or a buffer shorter than the row - which
//! the caller handles the way it handles any other unusable draft: argmax.

use shiso_core::{SamplingParams, Seed, TokenId};

use crate::sampling::{
    Candidate, NEG_MAX, argmax_index, candidate_gt, cmp_desc, sift_down, sift_up,
};

/// Candidate ceiling of the bounded pass. The heap holds this many of the
/// strongest weights and only gives up when the row's mass provably sits inside
/// it, which is the common case for a trained model at a sane temperature.
const PROB_FAST_CAP: usize = 512;

/// The knobs, clamped exactly the way [`crate::sample_logits`] clamps them.
///
/// The clamps are repeated rather than shared because the two entry points take
/// different shapes of input (one a row and a draw, the other a row and a
/// buffer), and a distribution built under different clamps than the sampler
/// would have drawn from is the failure this module exists to avoid.
struct Knobs {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
}

impl Knobs {
    fn new(p: &SamplingParams, n_vocab: usize) -> Self {
        let mut top_p = p.top_p;
        if top_p <= 0.0 || top_p > 1.0 {
            top_p = 1.0;
        }
        let mut min_p = p.min_p;
        if min_p < 0.0 {
            min_p = 0.0;
        }
        let top_k = p
            .top_k
            .map_or(0, |k| usize::try_from(k.get()).unwrap_or(usize::MAX))
            .min(crate::sampling::TOP_K_LIMIT)
            .min(n_vocab);
        Self {
            temperature: p.temperature,
            top_k,
            top_p,
            min_p,
        }
    }
}

/// Fill `probs` with the distribution the knobs describe.
///
/// Entries the filters reject are left at exactly zero, so a caller can read the
/// support straight out of the buffer. The mass that survives is normalized by
/// one reciprocal multiply (`prob * (1 / sum)`), never by a per-element division,
/// because the rounding of that product is part of what both the draft and the
/// target distributions agree on.
///
/// Returns `false` only when no distribution exists: an empty row, or a buffer
/// shorter than the row. A row whose every entry is non-finite is not that case -
/// it gets the one-hot distribution at [`crate::argmax`], which is what the
/// sampler would have returned.
#[must_use]
pub fn build_probabilities(logits: &[f32], p: &SamplingParams, probs: &mut [f32]) -> bool {
    let n_vocab = logits.len();
    if n_vocab == 0 || probs.len() < n_vocab {
        return false;
    }
    probs[..n_vocab].fill(0.0);
    let knobs = Knobs::new(p, n_vocab);
    if knobs.temperature <= 0.0 {
        probs[argmax_index(logits)] = 1.0;
        return true;
    }

    let (finite, max_logit) = finite_scan(logits);

    if finite == 0 {
        probs[argmax_index(logits)] = 1.0;
        return true;
    }

    let heap_cap = if knobs.top_k > 0 {
        knobs.top_k
    } else {
        PROB_FAST_CAP
    };
    let HeapRun {
        mut heap,
        heap_sum,
        full_sum,
    } = top_heap(
        logits,
        max_logit,
        knobs.temperature,
        heap_cap,
        heap_cap.min(finite),
    );
    let heap_n = heap.len();

    if full_sum <= 0.0 || !full_sum.is_finite() || heap_n == 0 {
        probs[argmax_index(logits)] = 1.0;
        return true;
    }

    heap.sort_unstable_by(cmp_desc);
    let mut base_sum = if knobs.top_k > 0 { heap_sum } else { full_sum };
    let mut filtered_sum = 0_f32;
    let mut filtered = 0_usize;
    let mut stopped_by_min_p = false;
    let mut reached_top_p = false;
    let min_raw = heap[0].prob * knobs.min_p;
    for (i, cand) in heap.iter().enumerate() {
        if i > 0 && cand.prob < min_raw {
            stopped_by_min_p = true;
            break;
        }
        filtered_sum += cand.prob;
        filtered += 1;
        if filtered_sum / base_sum >= knobs.top_p {
            reached_top_p = true;
            break;
        }
    }

    // The heap answers the whole row when it holds the whole row, when the
    // nucleus was reached inside it, when `top_k` bounded the set to begin with,
    // or when the floor stopped the walk at an entry too weak to survive: in that
    // last case every entry the heap never saw is weaker still. Anything else and
    // the row is rebuilt in full, because a candidate outside the heap could
    // belong to the support.
    let heap_has_all = heap_n == finite;
    let min_p_proves_tail_filtered = stopped_by_min_p && heap[heap_n - 1].prob < min_raw;
    let fast_complete =
        knobs.top_k > 0 || heap_has_all || reached_top_p || min_p_proves_tail_filtered;

    let mut full: Vec<Candidate> = Vec::new();
    if !fast_complete {
        full.reserve(finite);
        for (idx, v) in logits.iter().copied().enumerate() {
            if !v.is_finite() {
                continue;
            }
            full.push(Candidate {
                id: idx,
                logit: v,
                prob: ((v - max_logit) / knobs.temperature).exp(),
            });
        }
        full.sort_unstable_by(cmp_desc);
        base_sum = full_sum;
        filtered_sum = 0.0;
        filtered = 0;
        let full_min_raw = full[0].prob * knobs.min_p;
        for (i, cand) in full.iter().enumerate() {
            if i > 0 && cand.prob < full_min_raw {
                break;
            }
            filtered_sum += cand.prob;
            filtered += 1;
            if filtered_sum / base_sum >= knobs.top_p {
                break;
            }
        }
    }

    let chosen = if fast_complete { &heap } else { &full };
    if filtered == 0 || filtered_sum <= 0.0 || !filtered_sum.is_finite() {
        probs[chosen[0].id] = 1.0;
    } else {
        let inv_sum = 1.0 / filtered_sum;
        for cand in chosen.iter().take(filtered) {
            probs[cand.id] = cand.prob * inv_sum;
        }
    }
    true
}

/// The part of a row that takes part: how many entries are finite, and the
/// largest of them.
///
/// The maximum is subtracted before the exponent, so the strongest entry is
/// exactly `exp(0) = 1` and nothing overflows on the way up.
fn finite_scan(logits: &[f32]) -> (usize, f32) {
    let mut finite = 0_usize;
    let mut max_logit = NEG_MAX;
    for &v in logits {
        if !v.is_finite() {
            continue;
        }
        finite += 1;
        if v > max_logit {
            max_logit = v;
        }
    }
    (finite, max_logit)
}

/// The strongest entries of a row, still in heap order, with both sums.
struct HeapRun {
    /// The retained candidates.
    heap: Vec<Candidate>,
    /// Mass of the retained entries only.
    heap_sum: f32,
    /// Mass of every finite entry in the row.
    full_sum: f32,
}

/// Keeps the `cap` strongest entries of `logits` in a min-heap of that size.
///
/// Both sums come out of the one pass because the caller cannot tell from the
/// heap alone whether the heap *is* the whole support: that is answered by
/// weighing the retained mass against the row's, and getting it wrong either
/// rebuilds a row for nothing or drops a candidate that belonged in the support.
/// Entries are accumulated in index order - re-association changes the low bits,
/// and the golden vectors pin them.
fn top_heap(
    logits: &[f32],
    max_logit: f32,
    temperature: f32,
    cap: usize,
    reserve: usize,
) -> HeapRun {
    let mut heap: Vec<Candidate> = Vec::with_capacity(reserve);
    let mut heap_sum = 0_f32;
    let mut full_sum = 0_f32;
    for (idx, v) in logits.iter().copied().enumerate() {
        if !v.is_finite() {
            continue;
        }
        let raw = ((v - max_logit) / temperature).exp();
        full_sum += raw;
        let cand = Candidate {
            id: idx,
            logit: v,
            prob: raw,
        };
        if heap.len() < cap {
            heap.push(cand);
            heap_sum += raw;
            let last = heap.len() - 1;
            sift_up(&mut heap, last);
        } else if candidate_gt(cand, heap[0]) {
            heap_sum -= heap[0].prob;
            heap[0] = cand;
            heap_sum += raw;
            let held = heap.len();
            sift_down(&mut heap, held, 0);
        }
    }
    HeapRun {
        heap,
        heap_sum,
        full_sum,
    }
}

/// Draw one token from a distribution built by [`build_probabilities`].
///
/// Exactly one draw is consumed whenever the distribution carries mass, and the
/// walk is in index order, so the same buffer and the same stream state name the
/// same token on any machine. A buffer with no usable mass - all zero, all
/// non-finite - returns the argmax entry without drawing, and an empty buffer
/// returns `None` because there is no token to name.
#[must_use]
pub fn sample_probabilities(probs: &[f32], rng: &mut Seed) -> Option<TokenId> {
    if probs.is_empty() {
        return None;
    }
    let mut sum = 0_f32;
    let mut best = 0_usize;
    let mut best_p = probs[0];
    for (idx, p) in probs.iter().copied().enumerate() {
        if p > 0.0 && p.is_finite() {
            sum += p;
        }
        if p > best_p {
            best_p = p;
            best = idx;
        }
    }
    let id = |idx: usize| TokenId::new(u32::try_from(idx).unwrap_or(u32::MAX));
    if sum <= 0.0 || !sum.is_finite() {
        return Some(id(best));
    }
    let mut r = rng.next_f32() * sum;
    for (idx, p) in probs.iter().copied().enumerate() {
        if p <= 0.0 || !p.is_finite() {
            continue;
        }
        r -= p;
        if r <= 0.0 {
            return Some(id(idx));
        }
    }
    Some(id(best))
}
