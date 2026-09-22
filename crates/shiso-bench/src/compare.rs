//! The comparator: what "the same answer" means, and the two budgets it means
//! it at.
//!
//! # Why a token sequence is not enough
//!
//! Greedy decoding throws away almost everything a forward pass computed. Two
//! stacks can agree on every argmax for sixty-four steps and still disagree by
//! whole nats on the rest of the row, which is what a wrong activation function
//! or a transposed weight pack usually looks like: the winner survives because
//! it was winning by a mile, and the damage shows up in ranks two through sixty
//! four. So a fixture here records the *slice*, and the comparator grades four
//! things at once - the winner, three overlap depths, and the worst logit
//! disagreement across the top of the row.
//!
//! # Two budgets, neither of them adjustable
//!
//! [`FLOOR`] is the cross-build budget: deliberately loose, because it exists to
//! catch substantial drift rather than the last bits of a reduction that ran in
//! a different order. [`STRICT`] is the same-source budget, for a candidate that
//! should be reproducing a reference bit for bit.
//!
//! A number here is a claim about what the hardware and the arithmetic can do.
//! A test that fails is a bug to find, never a number to raise; there is exactly
//! one definition of each constant in this crate and moving one is a review
//! stop.

use crate::vecfile::TopEntry;

/// A top-k slice in rank order, highest logit first.
pub type TopSlice = Vec<TopEntry>;

/// The overlap depths every comparison reports, shallowest first.
///
/// Three depths rather than one because they fail for different reasons: the
/// top of the row moves when the winner changes, and the tail moves when a
/// systematic error shifts every score by a little.
pub const DEPTHS: [usize; 3] = [5, 20, 64];

/// How far down the row logits are compared value by value.
///
/// Twenty is deep enough that a uniform shift shows up and shallow enough that
/// the comparison is not dominated by entries whose logits are all within
/// rounding of each other.
pub const LOGIT_DEPTH: usize = 20;

/// What a comparison has to satisfy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Budget {
    /// The name that appears in a report line.
    pub name: &'static str,
    /// Minimum agreeing ids at each of [`DEPTHS`].
    pub overlap_min: [usize; 3],
    /// Largest tolerated `|Δlogit|` over the shared entries of the top
    /// [`LOGIT_DEPTH`].
    pub max_abs: f32,
}

/// The cross-build budget.
///
/// Calibrated to catch a backend that has substantially drifted, not to catch
/// floating-point noise: two builds of the same arithmetic on two devices
/// disagree in the last bits of every reduction, and a budget that failed on
/// that would be failing on the hardware rather than on the port. The winner is
/// still required to be exact, because a changed argmax changes the answer a
/// user reads.
pub const FLOOR: Budget = Budget {
    name: "floor",
    overlap_min: [4, 15, 40],
    max_abs: 8.0,
};

/// The same-source budget.
///
/// For a candidate that ran the same kernels in the same order on the same
/// device as the reference. There the only permitted difference is the
/// accumulated rounding of a run-to-run non-deterministic reduction, which is
/// orders of magnitude below this. Anything larger is a bug in the port.
pub const STRICT: Budget = Budget {
    name: "strict",
    overlap_min: [5, 20, 64],
    max_abs: 1e-3,
};

/// The independent-reference budget.
///
/// For a fixture captured from a *different* implementation of the same
/// checkpoint on the same device - upstream llama.cpp, in the one case this tree
/// has. That is neither of the two situations above. The two stacks do not share
/// a kernel, a reduction order or a dequantization routine, so [`STRICT`]'s
/// thousandth of a nat is not available; but they are running the same weights
/// through the same arithmetic on the same hardware, so [`FLOOR`]'s eight nats
/// would pass a port that had a whole operator wrong.
///
/// The numbers: the winner must agree, because a changed argmax changes the
/// answer a user reads. Sixty of the top sixty-four ids must be shared, which
/// permits the bottom of the recorded window to churn on near-ties and nothing
/// more. And no shared entry of the top [`LOGIT_DEPTH`] may move by more than
/// five hundredths of a nat, which is roughly a five percent change in relative
/// probability - visible to any comparison of distributions and far above what
/// two f32 reduction orders over a four-thousand-token prefill can accumulate.
///
/// What the replay measures, on the Gemma 4 E4B container against llama.cpp
/// `c6824a9` on an M1 Max: every one of the five cases agrees on all sixty-four
/// ids at all three depths, and the worst shared-entry gap runs from 0.003 nats
/// to 0.011. The ceiling is not lowered to that, for one reason worth writing
/// down: the same five cases without the model's opening token measured 0.041,
/// four times worse, so the gap is a function of what is being evaluated and not
/// a constant of the two builds. Five hundredths leaves room for a prompt that
/// exercises the arithmetic harder while still catching a wrong operator by two
/// orders of magnitude. It moves in one direction: down, and only on evidence
/// from more than one prompt shape. A failure here is this port disagreeing with
/// an independent implementation, which is the one finding this crate exists to
/// be able to make.
pub const REFERENCE: Budget = Budget {
    name: "reference",
    overlap_min: [5, 20, 60],
    max_abs: 0.05,
};

/// The cross-precision independent-reference budget.
///
/// For a fixture captured from an implementation that does not carry its
/// activations at the width this one does. [`REFERENCE`] assumes the two stacks
/// disagree only about reduction order; against `mlx-lm` that assumption is
/// false, because its forward pass is `bfloat16` end to end - eight mantissa
/// bits against this port's twenty-four - and the gap that opens over
/// forty-two layers is a property of the two number formats rather than of
/// either implementation's arithmetic.
///
/// The numbers: the winner must still agree exactly, because a changed argmax
/// changes the answer a user reads, and the top five must agree entirely. The
/// two deeper windows are allowed to churn more than [`REFERENCE`] permits,
/// because a hundredth-of-a-nat reordering near the bottom of a sixty-four-wide
/// window is what eight mantissa bits produce.
///
/// What the replay measures, on the Gemma 4 E4B mlx container against `mlx-lm`
/// 0.31.3 / mlx 0.32.2 on an M1 Max: all five cases agree on the top-1 and the
/// top-5, the top-20 agrees 18 to 20 of 20, the top-64 agrees 62 to 63 of 64,
/// and the worst shared-entry gap runs from 0.117 to 0.183 nats. A quarter of a
/// nat leaves a little room above the worst of those and is still thirty-two
/// times tighter than [`FLOOR`], which would pass a port with a whole operator
/// wrong.
///
/// This budget is NOT the one to reach for when a comparison is merely
/// inconvenient. It exists for one situation - the reference computes in a
/// narrower float than this port does - and the same container graded against
/// llama.cpp, which does not, is held to [`REFERENCE`] and measures 0.011.
pub const MLX_REFERENCE: Budget = Budget {
    name: "mlx-reference",
    overlap_min: [5, 18, 60],
    max_abs: 0.25,
};

/// Maximum absolute difference budget for Gemma 4 vision tower against MLX reference.
///
/// On an M1 Max, candidate projected rows agree with mlx-vlm within 0.1983.
/// Budget is set to 0.25 to provide calibrated headroom.
pub const GEMMA4_VISION_TOWER_MLX_BUDGET: f32 = 0.25;

/// Maximum absolute difference budget for Gemma 4 vision tower (mmproj GGUF) against MLX reference.
///
/// On an M1 Max, GGUF projected rows agree with mlx-vlm 4-bit projector within 0.4188.
/// Budget is set to 0.50 to provide calibrated headroom.
pub const GEMMA4_VISION_TOWER_GGUF_BUDGET: f32 = 0.50;

/// Maximum absolute difference budget for Qwen 3.5 vision tower against MLX reference.
///
/// On an M1 Max, candidate projected rows agree with mlx-vlm within 0.1137.
/// Budget is set to 0.20 to provide calibrated headroom.
pub const QWEN35_VISION_TOWER_BUDGET: f32 = 0.20;

/// How many of the top 64 ids may move when only the prefill chunking changed.
///
/// Measured, not chosen. Feeding one 1200-token prompt at widths 32, 64, 97,
/// 100, 128, 192, 256, 320 and 512 moves the top-1 never, the top-20 never, and
/// the top-64 by at most one id - a single near-tie at the very bottom of the
/// recorded window swapping with the id below it. Two is one more than was ever
/// observed, and a third would mean the ordering itself had shifted rather than
/// a tie having resolved the other way.
///
/// This is the *rank* half of the chunk-invariance claim. The other half is
/// exact and carries no tolerance at all: widths that take the same route
/// through the recurrence produce bit-identical rows, and that is asserted as
/// equality of bits.
pub const CHUNK_RANK_DRIFT: usize = 2;

/// One failed expectation, in words that name what to look at next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Failure {
    /// The winner changed.
    Top1 {
        /// What the reference selected.
        reference: u32,
        /// What this run selected.
        candidate: u32,
    },
    /// Too few ids agreed at one depth.
    Overlap {
        /// How deep the comparison went.
        depth: usize,
        /// How many agreed.
        got: usize,
        /// How many had to.
        want: usize,
    },
    /// A logit moved too far.
    MaxAbs {
        /// The worst disagreement, as a string so the variant stays comparable.
        got: String,
        /// The budget it exceeded.
        want: String,
        /// The token it happened on.
        token: u32,
    },
    /// The candidate row is not a slice of a real distribution.
    Degenerate {
        /// What is wrong with it.
        detail: String,
    },
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Top1 {
                reference,
                candidate,
            } => write!(
                f,
                "top1 moved: reference {reference}, candidate {candidate}"
            ),
            Self::Overlap { depth, got, want } => {
                write!(
                    f,
                    "top{depth} overlap {got}/{depth} below the required {want}"
                )
            }
            Self::MaxAbs { got, want, token } => {
                write!(f, "logit on token {token} moved by {got}, budget {want}")
            }
            Self::Degenerate { detail } => write!(f, "the candidate row is not usable: {detail}"),
        }
    }
}

/// What one case's comparison produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    /// The case this describes.
    pub id: String,
    /// The budget it was graded at.
    pub budget: &'static str,
    /// The reference's winner.
    pub reference_top1: u32,
    /// This run's winner.
    pub candidate_top1: u32,
    /// Agreeing ids at each of [`DEPTHS`].
    pub overlap: [usize; 3],
    /// Entries actually compared at each depth, which is the depth clamped to
    /// the shorter of the two slices.
    pub compared: [usize; 3],
    /// The worst `|Δlogit|` over the shared entries of the top [`LOGIT_DEPTH`].
    pub max_abs: f32,
    /// Everything that did not hold.
    pub failures: Vec<Failure>,
}

impl Verdict {
    /// Whether every expectation held.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }

    /// The one-line summary a run prints per case.
    #[must_use]
    pub fn report_line(&self) -> String {
        format!(
            "shiso-parity: case {} [{}] top1 ref={} cand={} top5_overlap={}/{} \
             top20_overlap={}/{} top64_overlap={}/{} top20_max_abs={:.6} {}",
            self.id,
            self.budget,
            self.reference_top1,
            self.candidate_top1,
            self.overlap[0],
            self.compared[0],
            self.overlap[1],
            self.compared[1],
            self.overlap[2],
            self.compared[2],
            self.max_abs,
            if self.passed() { "OK" } else { "FAIL" },
        )
    }
}

/// Grades `candidate` against `reference` at `budget`.
///
/// Both slices are expected in rank order, highest logit first; a slice that is
/// not is reported as degenerate rather than quietly graded, because an
/// unsorted slice means whatever produced it is broken and every metric below
/// would be measuring that instead of the model.
#[must_use]
pub fn compare(
    id: &str,
    reference: &[TopEntry],
    candidate: &[TopEntry],
    budget: &Budget,
) -> Verdict {
    let mut failures = Vec::new();
    for (side, slice) in [("reference", reference), ("candidate", candidate)] {
        if let Some(detail) = degeneracy(slice) {
            failures.push(Failure::Degenerate {
                detail: format!("{side}: {detail}"),
            });
        }
    }

    let reference_top1 = reference.first().map_or(u32::MAX, |e| e.token);
    let candidate_top1 = candidate.first().map_or(u32::MAX, |e| e.token);
    if reference_top1 != candidate_top1 {
        failures.push(Failure::Top1 {
            reference: reference_top1,
            candidate: candidate_top1,
        });
    }

    let mut overlap = [0_usize; 3];
    let mut compared = [0_usize; 3];
    for (slot, depth) in DEPTHS.iter().copied().enumerate() {
        let depth = depth.min(reference.len()).min(candidate.len());
        compared[slot] = depth;
        let head: Vec<u32> = candidate.iter().take(depth).map(|e| e.token).collect();
        overlap[slot] = reference
            .iter()
            .take(depth)
            .filter(|e| head.contains(&e.token))
            .count();
        let want = budget.overlap_min[slot].min(depth);
        if overlap[slot] < want {
            failures.push(Failure::Overlap {
                depth,
                got: overlap[slot],
                want,
            });
        }
    }

    let (max_abs, worst_token) = worst_logit_delta(reference, candidate);
    if max_abs > budget.max_abs {
        failures.push(Failure::MaxAbs {
            got: format!("{max_abs:.6}"),
            want: format!("{:.6}", budget.max_abs),
            token: worst_token,
        });
    }

    Verdict {
        id: id.to_owned(),
        budget: budget.name,
        reference_top1,
        candidate_top1,
        overlap,
        compared,
        max_abs,
        failures,
    }
}

/// The largest `|Δlogit|` over the entries the two slices share in the top
/// [`LOGIT_DEPTH`], and the token it happened on.
///
/// Shared entries only: a reference token that fell out of the candidate's
/// recorded window has no logit to compare against, and inventing one would
/// turn a rank failure into a magnitude failure. The overlap metrics are what
/// catch that case.
fn worst_logit_delta(reference: &[TopEntry], candidate: &[TopEntry]) -> (f32, u32) {
    let mut worst = 0.0_f32;
    let mut token = u32::MAX;
    for entry in reference.iter().take(LOGIT_DEPTH) {
        let Some(other) = candidate.iter().find(|c| c.token == entry.token) else {
            continue;
        };
        let delta = (entry.logit - other.logit).abs();
        if delta > worst {
            worst = delta;
            token = entry.token;
        }
    }
    (worst, token)
}

/// Why a slice cannot be graded, or `None` when it can.
///
/// Three ways a forward pass fails without crashing, all of which produce a
/// slice that looks superficially fine: a row of non-finite values, a row where
/// every entry carries the same score (a collapsed reduction), and a slice whose
/// ranks are out of order (a broken top-k).
fn degeneracy(slice: &[TopEntry]) -> Option<String> {
    if slice.is_empty() {
        return Some("no entries".to_owned());
    }
    if let Some(bad) = slice.iter().find(|e| !e.logit.is_finite()) {
        return Some(format!("token {} carries {}", bad.token, bad.logit));
    }
    if let Some(pair) = slice.windows(2).find(|w| w[0].logit < w[1].logit) {
        return Some(format!(
            "ranks are out of order: {} scores {} but is ranked above {} at {}",
            pair[0].token, pair[0].logit, pair[1].token, pair[1].logit
        ));
    }
    let first = slice.first().map_or(0.0, |e| e.logit);
    let last = slice.last().map_or(0.0, |e| e.logit);
    if slice.len() > 1 && (first - last).abs() == 0.0 {
        return Some("every entry carries the same score".to_owned());
    }
    let mut ids: Vec<u32> = slice.iter().map(|e| e.token).collect();
    ids.sort_unstable();
    let before = ids.len();
    ids.dedup();
    if ids.len() != before {
        return Some("the same token appears at two ranks".to_owned());
    }
    None
}

/// The `k` highest-scoring entries of a logits row, highest first.
///
/// Ties go to the lower token id, which is the rule the greedy sampler already
/// follows; a top-k that broke ties the other way would disagree with the very
/// first token of every decode it was compared against.
#[must_use]
pub fn top_k(logits: &[f32], k: usize) -> TopSlice {
    let mut entries: Vec<TopEntry> = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, logit)| logit.is_finite())
        .map(|(index, logit)| TopEntry {
            token: u32::try_from(index).unwrap_or(u32::MAX),
            logit,
        })
        .collect();
    entries.sort_by(|a, b| {
        b.logit
            .partial_cmp(&a.logit)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.token.cmp(&b.token))
    });
    entries.truncate(k);
    entries
}

/// Reads a device logits row, little-endian, into single-precision host values.
///
/// # Panics
///
/// Never: a short tail that is not four bytes wide is ignored by
/// `chunks_exact`, and the destination is only written where a chunk exists.
#[must_use]
pub fn decode_row(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}
