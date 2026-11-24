//! Host-side top-K and target-token log-probability extraction from a logits row.
//!
//! Used by the prompt-logprobs prefill path: the engine projects hidden states
//! through `lm_head` in bounded strips into the existing logits scratch buffer,
//! copies each strip to host, and reduces each row here into a compact
//! [`PromptPositionLogprobs`] record.

use std::cmp::Ordering;

use shiso_core::{PromptLogprobEntry, PromptPositionLogprobs, TokenId};

/// Extracts the top-`top_k` vocabulary tokens (by logit) plus `target` (if it
/// fell outside the top-`top_k`) from a full-vocabulary logits `row`.
///
/// Log-probabilities are computed in `f64` as `logit - logsumexp(row)` before
/// rounding to `f32`, matching the precision of the decision readout path.
/// Ties are broken deterministically in favor of the smaller token id so both
/// top-`top_k` membership and 1-based `rank` agree everywhere.
#[must_use]
pub fn extract_prompt_position_logprobs(
    row: &[f32],
    target: TokenId,
    top_k: u32,
) -> PromptPositionLogprobs {
    if row.is_empty() {
        return PromptPositionLogprobs {
            entries: Vec::new(),
        };
    }

    let max_logit = row
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |best, v| best.max(f64::from(v)));
    let logsumexp = if max_logit.is_finite() {
        let sum: f64 = row.iter().map(|&v| (f64::from(v) - max_logit).exp()).sum();
        max_logit + sum.ln()
    } else {
        max_logit
    };

    let target_idx = usize::try_from(target.get()).unwrap_or(usize::MAX);
    let target_logit = row.get(target_idx).copied().unwrap_or(f32::NEG_INFINITY);

    let k = usize::try_from(top_k).unwrap_or(usize::MAX).min(row.len());
    // Maintain a small descending-sorted buffer of at most `k` entries.
    // For typical `k` (<= 20), an insertion-sorted Vec with a threshold check
    // beats a BinaryHeap because almost every vocabulary entry is rejected by
    // a single float comparison against the current `k`-th smallest logit.
    let mut best: Vec<(f32, u32)> = Vec::with_capacity(k);
    let mut target_rank: u32 = 1;

    for (idx, &logit) in row.iter().enumerate() {
        let id = u32::try_from(idx).unwrap_or(u32::MAX);
        if id != target.get() && beats(logit, id, target_logit, target.get()) {
            target_rank = target_rank.saturating_add(1);
        }
        if k == 0 {
            continue;
        }
        if best.len() < k {
            insert_desc(&mut best, (logit, id));
        } else if let Some(&(worst_logit, worst_id)) = best.last()
            && beats(logit, id, worst_logit, worst_id)
        {
            best.pop();
            insert_desc(&mut best, (logit, id));
        }
    }

    let mut entries = Vec::with_capacity(k + 1);
    let mut target_included = false;
    for (pos, &(logit, id)) in best.iter().enumerate() {
        if id == target.get() {
            target_included = true;
        }
        entries.push(PromptLogprobEntry {
            token: TokenId::new(id),
            logprob: to_logprob(logit, logsumexp),
            rank: u32::try_from(pos + 1).unwrap_or(u32::MAX),
        });
    }

    if !target_included && target_idx < row.len() {
        entries.push(PromptLogprobEntry {
            token: target,
            logprob: to_logprob(target_logit, logsumexp),
            rank: target_rank,
        });
    }

    PromptPositionLogprobs { entries }
}

/// Whether `(a_logit, a_id)` ranks strictly ahead of `(b_logit, b_id)`.
#[inline]
fn beats(a_logit: f32, a_id: u32, b_logit: f32, b_id: u32) -> bool {
    match a_logit.total_cmp(&b_logit) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => a_id < b_id,
    }
}

/// Inserts `item` into a descending-sorted slice ordered by [`beats`].
#[inline]
fn insert_desc(best: &mut Vec<(f32, u32)>, item: (f32, u32)) {
    let pos = best
        .iter()
        .position(|&(l, id)| beats(item.0, item.1, l, id))
        .unwrap_or(best.len());
    best.insert(pos, item);
}

#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "log-probabilities are stored as f32 on the wire after f64 logsumexp"
)]
fn to_logprob(logit: f32, logsumexp: f64) -> f32 {
    (f64::from(logit) - logsumexp) as f32
}

#[cfg(test)]
mod tests {
    use super::extract_prompt_position_logprobs;
    use shiso_core::TokenId;

    #[test]
    fn top_k_includes_sorted_top_and_out_of_topk_target() {
        let row = [1.0_f32, 4.0, 2.0, 3.0, 0.5];
        // Sorted order: id=1 (4.0, rank 1), id=3 (3.0, rank 2), id=2 (2.0, rank 3),
        // id=0 (1.0, rank 4), id=4 (0.5, rank 5).
        let got = extract_prompt_position_logprobs(&row, TokenId::new(4), 2);
        assert_eq!(got.entries.len(), 3);
        assert_eq!(got.entries[0].token, TokenId::new(1));
        assert_eq!(got.entries[0].rank, 1);
        assert_eq!(got.entries[1].token, TokenId::new(3));
        assert_eq!(got.entries[1].rank, 2);
        assert_eq!(got.entries[2].token, TokenId::new(4));
        assert_eq!(got.entries[2].rank, 5);

        // Verify exact logsumexp normalization.
        let lse: f64 = row.iter().map(|&v| f64::from(v).exp()).sum::<f64>().ln();
        assert!((f64::from(got.entries[0].logprob) - (4.0 - lse)).abs() < 1e-6);
        assert!((f64::from(got.entries[2].logprob) - (0.5 - lse)).abs() < 1e-6);
    }

    #[test]
    fn target_inside_top_k_is_not_duplicated() {
        let row = [1.0_f32, 4.0, 2.0, 3.0];
        let got = extract_prompt_position_logprobs(&row, TokenId::new(3), 2);
        assert_eq!(got.entries.len(), 2);
        assert_eq!(got.entries[0].token, TokenId::new(1));
        assert_eq!(got.entries[1].token, TokenId::new(3));
        assert_eq!(got.entries[1].rank, 2);
    }

    #[test]
    fn zero_top_k_returns_only_target() {
        let row = [1.0_f32, 4.0, 2.0];
        let got = extract_prompt_position_logprobs(&row, TokenId::new(2), 0);
        assert_eq!(got.entries.len(), 1);
        assert_eq!(got.entries[0].token, TokenId::new(2));
        assert_eq!(got.entries[0].rank, 2);
    }

    #[test]
    fn ties_break_in_favor_of_smaller_token_id() {
        let row = [2.0_f32, 2.0, 2.0];
        let got = extract_prompt_position_logprobs(&row, TokenId::new(2), 2);
        assert_eq!(got.entries.len(), 3);
        assert_eq!(got.entries[0].token, TokenId::new(0));
        assert_eq!(got.entries[0].rank, 1);
        assert_eq!(got.entries[1].token, TokenId::new(1));
        assert_eq!(got.entries[1].rank, 2);
        assert_eq!(got.entries[2].token, TokenId::new(2));
        assert_eq!(got.entries[2].rank, 3);
    }
}
