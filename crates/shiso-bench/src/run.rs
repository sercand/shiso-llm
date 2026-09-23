//! One fixture case, from its prompt file to its recorded slice.
//!
//! The three steps a case goes through - resolve the prompt relative to the
//! fixture, encode it the way the case declares, truncate it to the frontier -
//! live here rather than in the command-line binary, because the tests replay
//! cases too and a second copy of "which tokens does this case mean" is exactly
//! the kind of drift a parity harness exists to prevent.

use std::path::{Path, PathBuf};

use shiso_core::ids::TokenId;
use shiso_engine::sparse::tap::ResidualTap;

use crate::compare::{TopSlice, top_k};
use crate::replay::{Harness, ReplayError, ReplayResult, Vocabulary};
use crate::vecfile::{LocalCase, Mode};

/// Prefill chunk a replay feeds a prompt in.
///
/// One chunk is what the engine's scratch is sized for, so this is the largest
/// value that reaches the prefill kernels at full width; a case that wanted a
/// different one would be measuring the chunking rather than the model.
pub const REPLAY_CHUNK: usize = 512;

/// The tokens one case means, already truncated to its frontier.
///
/// # Errors
///
/// [`ReplayError::Io`] when the prompt file cannot be read,
/// [`ReplayError::Vocabulary`] when it cannot be encoded, and
/// [`ReplayError::ShortPrompt`] when it yields fewer tokens than the case
/// declares - a case that silently ran short would compare a different frontier
/// against the recorded one and call the difference a parity failure.
pub fn case_tokens(
    vocabulary: &Vocabulary,
    base_dir: &Path,
    case: &LocalCase,
) -> ReplayResult<Vec<TokenId>> {
    case_tokens_with_leading(vocabulary, base_dir, case, None)
}

/// The tokens one case means, with `leading` prepended before the frontier is
/// counted.
///
/// Some vocabularies do not add the model's own opening token and some engines
/// do. Gemma 4 is both at once: its `config.json` declares a `bos_token_id`,
/// llama.cpp's tokenizer prepends it, and the sidecar `tokenizer.json`'s
/// post-processor is `single: [$A]` with no special tokens, so the
/// `tokenizers` crate adds nothing at either `special` setting. A fixture
/// captured from that engine therefore records the id it prepended and a replay
/// puts it back, or the two sides are answering about different sequences.
///
/// It is a parameter rather than a mode on the grammar because it is a fact
/// about the *capture*, not about the prompt file: the same file replayed
/// against a family with no opening token passes `None` and reads exactly as
/// before.
///
/// The frontier is counted over the sequence *including* `leading`, which is how
/// the capturing engine counted it.
///
/// # Errors
///
/// [`ReplayError::Io`] when the prompt file cannot be read,
/// [`ReplayError::Vocabulary`] when it cannot be encoded, and
/// [`ReplayError::ShortPrompt`] when it yields fewer tokens than the case
/// declares - a case that silently ran short would compare a different frontier
/// against the recorded one and call the difference a parity failure.
pub fn case_tokens_with_leading(
    vocabulary: &Vocabulary,
    base_dir: &Path,
    case: &LocalCase,
    leading: Option<TokenId>,
) -> ReplayResult<Vec<TokenId>> {
    encode_case(vocabulary, base_dir, case, leading).map(|(tokens, _)| tokens)
}

/// The body of [`case_tokens_with_leading`], which also returns how many tokens
/// the whole prompt encoded to before the frontier cut it.
fn encode_case(
    vocabulary: &Vocabulary,
    base_dir: &Path,
    case: &LocalCase,
    leading: Option<TokenId>,
) -> ReplayResult<(Vec<TokenId>, usize)> {
    let path = prompt_path(base_dir, case);
    let text = std::fs::read_to_string(&path).map_err(|source| ReplayError::Io {
        path: path.clone(),
        source,
    })?;
    let special = matches!(case.mode, Mode::Rendered);
    let encoded = vocabulary.encode(&text, special)?;
    let mut tokens = Vec::with_capacity(encoded.len() + usize::from(leading.is_some()));
    tokens.extend(leading);
    tokens.extend(encoded);
    let frontier = usize::try_from(case.frontier).unwrap_or(usize::MAX);
    if tokens.len() < frontier {
        return Err(ReplayError::ShortPrompt {
            case: case.id.clone(),
            available: tokens.len(),
            frontier: case.frontier,
        });
    }
    let total = tokens.len();
    tokens.truncate(frontier);
    Ok((tokens, total))
}

/// Where a case's prompt file lives.
///
/// Relative to the fixture's own directory, so a fixture and its prompts move
/// together and a run from any working directory reads the same bytes.
#[must_use]
pub fn prompt_path(base_dir: &Path, case: &LocalCase) -> PathBuf {
    if case.prompt.is_absolute() {
        case.prompt.clone()
    } else {
        base_dir.join(&case.prompt)
    }
}

/// One digest over every prompt file the given cases name, in case order.
///
/// A fixture records this and a replay recomputes it, because the frontier a
/// case declares is the whole of its prompt: an edited prompt file either runs
/// short - which is caught - or runs long, and a longer file would be silently
/// truncated back to the recorded frontier at a *different* token boundary. The
/// digest is what turns that second case from a mystifying parity failure into
/// a refusal that names the file.
///
/// # Errors
///
/// [`ReplayError::Io`] when a prompt file cannot be read.
pub fn prompts_digest(base_dir: &Path, cases: &[&LocalCase]) -> ReplayResult<String> {
    let mut all = Vec::new();
    for case in cases {
        let path = prompt_path(base_dir, case);
        let bytes = std::fs::read(&path).map_err(|source| ReplayError::Io { path, source })?;
        all.extend_from_slice(case.id.as_bytes());
        all.extend_from_slice(&bytes);
    }
    Ok(crate::replay::fnv_digest(&all))
}

/// The `k` highest entries of a logits row, as log-probabilities over the
/// **whole** row.
///
/// Why this exists rather than a flag on [`top_k`]: a fixture captured from an
/// engine that only exposes probabilities records log-probabilities, and a
/// log-probability is a logit minus the row's log-sum-exp. Subtracting a
/// constant from every entry of a row is the one transformation that makes two
/// engines' rows comparable - it removes the single degree of freedom a softmax
/// discards and changes nothing else - but the constant has to come from the
/// full vocabulary, which is what the capturing engine used. Normalizing over
/// the recorded sixty-four instead would subtract a *different* constant on each
/// side, because the two sides' sixty-four are not the same sixty-four, and the
/// difference would be reported as a parity failure that is really an artifact
/// of the comparison.
///
/// The sum accumulates in double precision. The row is a quarter of a million
/// single-precision values and the exponentials near the top of it are order
/// one, so a single-precision accumulator would lose the tail of the
/// distribution into rounding and shift every value here by more than the
/// budget this is compared at.
#[must_use]
pub fn logprob_top_k(row: &[f32], k: usize) -> TopSlice {
    let max = row
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = row
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(|v| f64::from(v - max).exp())
        .sum();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the offset is compared against single-precision fixture values"
    )]
    let offset = max + (sum.ln() as f32);
    let mut slice = top_k(row, k);
    for entry in &mut slice {
        entry.logit -= offset;
    }
    slice
}

/// A digest of a token sequence, over its ids as little-endian 32-bit words.
///
/// A fixture captured from another implementation records this beside the
/// frontier count, and a replay recomputes it from its own vocabulary. The count
/// alone would let two tokenizers that disagree about *which* ids a prompt is
/// still be compared, and a silently different prefix is the failure mode most
/// easily mistaken for a wrong kernel.
#[must_use]
pub fn tokens_digest(tokens: &[TokenId]) -> String {
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for token in tokens {
        bytes.extend_from_slice(&token.get().to_le_bytes());
    }
    crate::replay::fnv_digest(&bytes)
}

/// Runs one case and returns the slice it produced.
///
/// # Errors
///
/// Whatever [`case_tokens`] or the forward pass reports.
pub fn run_case(
    harness: &mut Harness,
    vocabulary: &Vocabulary,
    base_dir: &Path,
    case: &LocalCase,
) -> ReplayResult<TopSlice> {
    let produced = run_case_row(harness, vocabulary, base_dir, case)?;
    Ok(top_k(
        &produced.row,
        usize::try_from(case.top_count).unwrap_or(0),
    ))
}

/// The whole logits row one case produced, and where in its prompt it sits.
#[derive(Debug, Clone)]
pub struct CaseRow {
    /// The raw logits after the case's frontier, one per vocabulary entry.
    pub row: Vec<f32>,
    /// Tokens fed before the row was read: the case's frontier.
    pub consumed: usize,
    /// Tokens the whole prompt file encodes to, before the frontier cut it.
    pub prompt_tokens: usize,
}

/// Runs one case and returns its full row rather than a slice of it.
///
/// What [`run_case`] slices, and what a row dump writes whole: the tail of the
/// distribution is where an exact KL finds a quantized build's lost mass, and
/// a top-k slice has already thrown it away.
///
/// # Errors
///
/// Whatever [`case_tokens`] or the forward pass reports.
pub fn run_case_row(
    harness: &mut Harness,
    vocabulary: &Vocabulary,
    base_dir: &Path,
    case: &LocalCase,
) -> ReplayResult<CaseRow> {
    let (tokens, prompt_tokens) = encode_case(vocabulary, base_dir, case, None)?;
    let row = harness.frontier_row(&tokens, REPLAY_CHUNK)?;
    Ok(CaseRow {
        row,
        consumed: tokens.len(),
        prompt_tokens,
    })
}

/// Runs one case with the residual tap armed on its final pass, and returns
/// its full row beside the tap.
///
/// The row is the one [`run_case_row`] would return: the tapped pass issues
/// the same launches, only without replaying them from a graph.
///
/// # Errors
///
/// Whatever [`case_tokens`] or [`Harness::frontier_layers`] reports.
pub fn run_case_layers(
    harness: &mut Harness,
    vocabulary: &Vocabulary,
    base_dir: &Path,
    case: &LocalCase,
) -> ReplayResult<(CaseRow, ResidualTap)> {
    let (tokens, prompt_tokens) = encode_case(vocabulary, base_dir, case, None)?;
    let (row, tap) = harness.frontier_layers(&tokens, REPLAY_CHUNK)?;
    Ok((
        CaseRow {
            row,
            consumed: tokens.len(),
            prompt_tokens,
        },
        tap,
    ))
}
