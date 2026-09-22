//! The golden-vector fixture format: one parser, one writer, one set of rules.
//!
//! Two grammars live here, both line-oriented and both deliberately trivial to
//! parse from any language. They exist as separate shapes because they answer
//! different questions:
//!
//! - [`Grammar::LocalLogit`] records a **full-vocabulary top-k slice of raw
//!   logits at one frontier**, by token id. This is what an engine that owns the
//!   checkpoint can emit, so it is the working grammar for every fixture in this
//!   tree.
//! - [`Grammar::OfficialLogprob`] records **selected tokens as UTF-8 bytes plus
//!   log-probabilities per decode step**. A hosted API exposes no token ids and
//!   no logits, so a fixture captured from one can only be this shape. Nothing
//!   in this tree replays one today; the parser carries it so that the day a
//!   hosted slice exists, the fixture does not have to be invented under time
//!   pressure.
//!
//! # Why the file is text and why the fields are in this order
//!
//! The line shapes here are the reference engine's, field for field, so a
//! fixture produced by either implementation is readable by the other's tooling
//! after a one-line header substitution. Reordering a field, widening a column
//! or "improving" a name would cost exactly that and buy nothing. See
//! `shiso/README.md` § Provenance for what the upstream is.
//!
//! # Extension is by comment line, and only by comment line
//!
//! Every parser of this format skips `#` lines, so the provenance a shiso
//! fixture carries - the label, the checkpoint's byte length, the vocabulary
//! digest - rides in header comments and cannot break a reader that predates
//! them. A new fact about a capture is a new header comment, never a new column.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// The magic first line of a fixture recording logits by token id.
pub const LOCAL_LOGIT_HEADER: &str = "# shiso-local-logit-vectors-v1";

/// The magic first line of a fixture recording token bytes and log-probabilities.
pub const OFFICIAL_LOGPROB_HEADER: &str = "# shiso-official-logprob-vectors-v1";

/// The magic first line of a fixture recording sampled tower activations.
pub const VISION_TOWER_HEADER: &str = "# shiso-vision-tower-vectors-v1";

/// Most entries any one top-k block may carry.
///
/// A ceiling rather than a preference: the readers on both sides of this format
/// size a fixed buffer from it, so a fixture that exceeded it would be read
/// differently by the two implementations instead of being rejected by both.
pub const MAX_TOP_ENTRIES: usize = 128;

/// Which grammar a fixture is written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grammar {
    /// Token ids and raw logits at one frontier.
    LocalLogit,
    /// Token bytes and log-probabilities per decode step.
    OfficialLogprob,
    /// Sampled activation points of a vision tower block.
    VisionTower,
}

impl Grammar {
    /// The magic line a fixture in this grammar opens with.
    #[must_use]
    pub const fn header(self) -> &'static str {
        match self {
            Self::LocalLogit => LOCAL_LOGIT_HEADER,
            Self::OfficialLogprob => OFFICIAL_LOGPROB_HEADER,
            Self::VisionTower => VISION_TOWER_HEADER,
        }
    }
}

/// How a case's prompt file becomes tokens.
///
/// The distinction is the `add_special_tokens` flag and nothing else, but it is
/// a field rather than a caller's choice because a fixture that did not record
/// it could be replayed two ways and only one of them would match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The file is plain text; the vocabulary's control tokens are not added.
    Text,
    /// The file already carries its control tokens; tokenize it as written.
    Rendered,
}

impl Mode {
    /// The word this mode is spelled with in a fixture.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Rendered => "rendered",
        }
    }
}

/// One `top <rank> <token-id> <logit>` row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopEntry {
    /// The token this rank names.
    pub token: u32,
    /// Its raw logit, before any normalization.
    pub logit: f32,
}

/// One case of a [`Grammar::LocalLogit`] fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalCase {
    /// The name a report and a `--case` filter use.
    pub id: String,
    /// How the prompt file is tokenized.
    pub mode: Mode,
    /// Session context the case is replayed at.
    pub ctx: u32,
    /// Prompt tokens fed before the logits row is read.
    pub frontier: u32,
    /// Prompt file, relative to the fixture's own directory.
    pub prompt: PathBuf,
    /// Entries the capture was asked for, which a complete case supplies.
    pub top_count: u32,
    /// The recorded slice, rank 0 first. Empty in a skeleton.
    pub tops: Vec<TopEntry>,
}

impl LocalCase {
    /// Whether this case is a capture request rather than a recorded answer.
    #[must_use]
    pub fn is_skeleton(&self) -> bool {
        self.tops.is_empty()
    }
}

/// One `top <token-hex> <logprob>` row of a step.
#[derive(Debug, Clone, PartialEq)]
pub struct OfficialTop {
    /// The token's UTF-8 bytes, as the hosted answer spelled them.
    pub token: Vec<u8>,
    /// Its log-probability.
    pub logprob: f32,
}

/// One decode step of a [`Grammar::OfficialLogprob`] case.
#[derive(Debug, Clone, PartialEq)]
pub struct OfficialStep {
    /// Step index, counted from zero.
    pub index: u32,
    /// Bytes of the token the reference selected.
    pub selected: Vec<u8>,
    /// The slice the reference reported at this step.
    pub tops: Vec<OfficialTop>,
}

/// One case of a [`Grammar::OfficialLogprob`] fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct OfficialCase {
    /// The name a report and a `--case` filter use.
    pub id: String,
    /// Session context the case is replayed at.
    pub ctx: u32,
    /// Steps the capture recorded.
    pub steps: Vec<OfficialStep>,
    /// Prompt file, relative to the fixture's own directory.
    pub prompt: PathBuf,
}

/// One sampled activation point in a vision tower block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TowerSample {
    /// Sample index in the sequence.
    pub index: u32,
    /// Flat element index in the block.
    pub pos: usize,
    /// Sampled float value.
    pub value: f32,
}

/// One case of a vision tower fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct TowerCase {
    /// Identifier of the case.
    pub id: String,
    /// Path to the input image.
    pub image: PathBuf,
    /// Target sample count.
    pub sample_count: usize,
    /// Recorded samples.
    pub samples: Vec<TowerSample>,
}

/// The cases of a fixture, in the grammar they were written in.
#[derive(Debug, Clone, PartialEq)]
pub enum Body {
    /// Token ids and raw logits.
    Local(Vec<LocalCase>),
    /// Token bytes and log-probabilities.
    Official(Vec<OfficialCase>),
    /// Sampled vision tower activations.
    Tower(Vec<TowerCase>),
}

/// A parsed fixture: its provenance comments and its cases.
#[derive(Debug, Clone, PartialEq)]
pub struct VecFile {
    /// Header comments of the form `# key value`, in sorted order.
    ///
    /// Sorted rather than as-written so that a writer's output is a function of
    /// its input alone, which is what makes a round-trip byte-comparable.
    pub headers: BTreeMap<String, String>,
    /// Free-form comment lines, kept so a hand-written note survives a rewrite.
    pub notes: Vec<String>,
    /// The cases.
    pub body: Body,
}

/// What went wrong reading a fixture, and exactly where.
#[derive(Debug, thiserror::Error)]
pub enum VecError {
    /// The file could not be read at all.
    #[error("reading {path}: {source}")]
    Io {
        /// The file that could not be read.
        path: PathBuf,
        /// The operating system's reason.
        #[source]
        source: std::io::Error,
    },
    /// The first line was not one of the two known magic lines.
    #[error("{path}: first line is not a known fixture header, found {found:?}")]
    UnknownGrammar {
        /// The file.
        path: PathBuf,
        /// What the first line actually said.
        found: String,
    },
    /// A line did not have the shape its keyword requires.
    #[error("{path}:{line}: malformed {keyword} line: {detail}")]
    Malformed {
        /// The file.
        path: PathBuf,
        /// One-based line number.
        line: usize,
        /// The keyword that opened the line.
        keyword: &'static str,
        /// What is wrong with it.
        detail: String,
    },
    /// A line was well formed but says something the format forbids.
    #[error("{path}:{line}: {detail}")]
    Invalid {
        /// The file.
        path: PathBuf,
        /// One-based line number.
        line: usize,
        /// The rule that was broken.
        detail: String,
    },
}

/// A parse or write result.
pub type VecResult<T> = Result<T, VecError>;

/// Reads a fixture from disk.
///
/// # Errors
///
/// [`VecError::Io`] when the file cannot be read, and any parse variant when its
/// contents do not satisfy the grammar.
pub fn read(path: &Path) -> VecResult<VecFile> {
    let text = std::fs::read_to_string(path).map_err(|source| VecError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse(path, &text)
}

/// Parses a fixture already in memory, reporting `path` in every error.
///
/// # Errors
///
/// Any parse variant of [`VecError`] when the text does not satisfy the grammar.
pub fn parse(path: &Path, text: &str) -> VecResult<VecFile> {
    let mut lines = text.lines().enumerate();
    let (_, first) = lines.next().ok_or_else(|| VecError::UnknownGrammar {
        path: path.to_path_buf(),
        found: String::new(),
    })?;
    let grammar = match first.trim_end() {
        LOCAL_LOGIT_HEADER => Grammar::LocalLogit,
        OFFICIAL_LOGPROB_HEADER => Grammar::OfficialLogprob,
        VISION_TOWER_HEADER => Grammar::VisionTower,
        other => {
            return Err(VecError::UnknownGrammar {
                path: path.to_path_buf(),
                found: other.to_owned(),
            });
        }
    };

    let mut headers = BTreeMap::new();
    let mut notes = Vec::new();
    let mut local: Vec<LocalCase> = Vec::new();
    let mut official: Vec<OfficialCase> = Vec::new();
    let mut tower: Vec<TowerCase> = Vec::new();

    for (zero_based, raw) in lines {
        let line = zero_based + 1;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(comment) = trimmed.strip_prefix('#') {
            record_comment(comment.trim(), &mut headers, &mut notes);
            continue;
        }
        let mut fields = trimmed.split_whitespace();
        let keyword = fields.next().unwrap_or_default();
        let at = Cursor {
            path,
            line,
            keyword,
        };
        match grammar {
            Grammar::LocalLogit => consume_local(at, fields, &mut local)?,
            Grammar::OfficialLogprob => consume_official(at, fields, &mut official)?,
            Grammar::VisionTower => consume_tower(at, fields, &mut tower)?,
        }
    }

    let body = match grammar {
        Grammar::LocalLogit => Body::Local(local),
        Grammar::OfficialLogprob => Body::Official(official),
        Grammar::VisionTower => Body::Tower(tower),
    };
    let file = VecFile {
        headers,
        notes,
        body,
    };
    validate(path, &file)?;
    Ok(file)
}

/// Where a line is, for an error that has to name it.
#[derive(Debug, Clone, Copy)]
struct Cursor<'a> {
    path: &'a Path,
    line: usize,
    keyword: &'a str,
}

impl Cursor<'_> {
    /// An error about a rule this line breaks.
    fn invalid(self, detail: String) -> VecError {
        VecError::Invalid {
            path: self.path.to_path_buf(),
            line: self.line,
            detail,
        }
    }
}

/// Folds one line of a token-id fixture into the cases read so far.
fn consume_local<'a>(
    at: Cursor<'_>,
    fields: impl Iterator<Item = &'a str>,
    cases: &mut Vec<LocalCase>,
) -> VecResult<()> {
    match at.keyword {
        "case" => cases.push(parse_local_case(at.path, at.line, fields)?),
        "top" => {
            let case = cases
                .last_mut()
                .ok_or_else(|| at.invalid("a top line before any case line".to_owned()))?;
            if case.tops.len() >= MAX_TOP_ENTRIES {
                return Err(at.invalid(format!(
                    "more than {MAX_TOP_ENTRIES} entries in one top-k block"
                )));
            }
            let entry = parse_local_top(at.path, at.line, fields, case.tops.len())?;
            case.tops.push(entry);
        }
        "end" => {}
        other => {
            return Err(at.invalid(format!("unknown keyword {other:?} for this grammar")));
        }
    }
    Ok(())
}

/// Folds one line of a token-byte fixture into the cases read so far.
fn consume_official<'a>(
    at: Cursor<'_>,
    fields: impl Iterator<Item = &'a str>,
    cases: &mut Vec<OfficialCase>,
) -> VecResult<()> {
    match at.keyword {
        "case" => cases.push(parse_official_case(at.path, at.line, fields)?),
        "step" => {
            let case = cases
                .last_mut()
                .ok_or_else(|| at.invalid("a step line before any case line".to_owned()))?;
            case.steps
                .push(parse_official_step(at.path, at.line, fields)?);
        }
        "top" => {
            let step = cases
                .last_mut()
                .and_then(|c| c.steps.last_mut())
                .ok_or_else(|| at.invalid("a top line before any step line".to_owned()))?;
            if step.tops.len() >= MAX_TOP_ENTRIES {
                return Err(at.invalid(format!(
                    "more than {MAX_TOP_ENTRIES} entries in one top-k block"
                )));
            }
            step.tops
                .push(parse_official_top(at.path, at.line, fields)?);
        }
        "end" => {}
        other => {
            return Err(at.invalid(format!("unknown keyword {other:?} for this grammar")));
        }
    }
    Ok(())
}

/// Folds one line of a vision tower fixture into the cases read so far.
fn consume_tower<'a>(
    at: Cursor<'_>,
    mut fields: impl Iterator<Item = &'a str>,
    cases: &mut Vec<TowerCase>,
) -> VecResult<()> {
    match at.keyword {
        "case" => {
            let id = fields
                .next()
                .ok_or_else(|| at.invalid("case line missing id".into()))?
                .to_owned();
            let image = PathBuf::from(
                fields
                    .next()
                    .ok_or_else(|| at.invalid("case line missing image path".into()))?,
            );
            let sample_count = fields
                .next()
                .ok_or_else(|| at.invalid("case line missing sample count".into()))?
                .parse::<usize>()
                .map_err(|_| at.invalid("invalid sample count integer".into()))?;
            cases.push(TowerCase {
                id,
                image,
                sample_count,
                samples: Vec::with_capacity(sample_count),
            });
        }
        "sample" => {
            let case = cases
                .last_mut()
                .ok_or_else(|| at.invalid("a sample line before any case line".into()))?;
            let index = fields
                .next()
                .ok_or_else(|| at.invalid("sample line missing index".into()))?
                .parse::<u32>()
                .map_err(|_| at.invalid("invalid sample index integer".into()))?;
            let pos = fields
                .next()
                .ok_or_else(|| at.invalid("sample line missing position".into()))?
                .parse::<usize>()
                .map_err(|_| at.invalid("invalid sample pos integer".into()))?;
            let value = fields
                .next()
                .ok_or_else(|| at.invalid("sample line missing value".into()))?
                .parse::<f32>()
                .map_err(|_| at.invalid("invalid sample float value".into()))?;
            case.samples.push(TowerSample { index, pos, value });
        }
        "end" => {}
        other => {
            return Err(at.invalid(format!(
                "unknown keyword {other:?} for vision tower grammar"
            )));
        }
    }
    Ok(())
}

/// Splits a comment into a `key value` header or keeps it as a free note.
fn record_comment(comment: &str, headers: &mut BTreeMap<String, String>, notes: &mut Vec<String>) {
    let mut parts = comment.splitn(2, char::is_whitespace);
    let key = parts.next().unwrap_or_default();
    let value = parts.next().unwrap_or_default().trim();
    let is_header = !key.is_empty()
        && !value.is_empty()
        && key
            .chars()
            .all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit());
    if is_header {
        headers.insert(key.to_owned(), value.to_owned());
    } else if !comment.is_empty() {
        notes.push(comment.to_owned());
    }
}

/// `case <id> <mode> <ctx> <frontier> <prompt-file> <top-count>`
fn parse_local_case<'a>(
    path: &Path,
    line: usize,
    fields: impl Iterator<Item = &'a str>,
) -> VecResult<LocalCase> {
    let fields: Vec<&str> = fields.collect();
    let [id, mode, ctx, frontier, prompt, top_count] = fields.as_slice() else {
        return Err(VecError::Malformed {
            path: path.to_path_buf(),
            line,
            keyword: "case",
            detail: format!("expected 6 fields, found {}", fields.len()),
        });
    };
    let mode = match *mode {
        "text" => Mode::Text,
        "rendered" => Mode::Rendered,
        other => {
            return Err(VecError::Malformed {
                path: path.to_path_buf(),
                line,
                keyword: "case",
                detail: format!("unknown mode {other:?}"),
            });
        }
    };
    Ok(LocalCase {
        id: (*id).to_owned(),
        mode,
        ctx: number(path, line, "case", "ctx", ctx)?,
        frontier: number(path, line, "case", "frontier", frontier)?,
        prompt: PathBuf::from(*prompt),
        top_count: number(path, line, "case", "top-count", top_count)?,
        tops: Vec::new(),
    })
}

/// `top <rank> <token-id> <logit>`, with the rank checked against its position.
fn parse_local_top<'a>(
    path: &Path,
    line: usize,
    fields: impl Iterator<Item = &'a str>,
    expected_rank: usize,
) -> VecResult<TopEntry> {
    let fields: Vec<&str> = fields.collect();
    let [rank, token, logit] = fields.as_slice() else {
        return Err(VecError::Malformed {
            path: path.to_path_buf(),
            line,
            keyword: "top",
            detail: format!("expected 3 fields, found {}", fields.len()),
        });
    };
    let rank: usize = number::<u32>(path, line, "top", "rank", rank)?
        .try_into()
        .unwrap_or(usize::MAX);
    if rank != expected_rank {
        return Err(VecError::Invalid {
            path: path.to_path_buf(),
            line,
            detail: format!(
                "ranks must be sequential from 0: expected {expected_rank}, got {rank}"
            ),
        });
    }
    Ok(TopEntry {
        token: number(path, line, "top", "token-id", token)?,
        logit: float(path, line, "top", "logit", logit)?,
    })
}

/// `case <id> <ctx> <steps> <prompt-file>`
fn parse_official_case<'a>(
    path: &Path,
    line: usize,
    fields: impl Iterator<Item = &'a str>,
) -> VecResult<OfficialCase> {
    let fields: Vec<&str> = fields.collect();
    let [id, ctx, _steps, prompt] = fields.as_slice() else {
        return Err(VecError::Malformed {
            path: path.to_path_buf(),
            line,
            keyword: "case",
            detail: format!("expected 4 fields, found {}", fields.len()),
        });
    };
    Ok(OfficialCase {
        id: (*id).to_owned(),
        ctx: number(path, line, "case", "ctx", ctx)?,
        steps: Vec::new(),
        prompt: PathBuf::from(*prompt),
    })
}

/// `step <index> <selected-hex> <top-count>`
fn parse_official_step<'a>(
    path: &Path,
    line: usize,
    fields: impl Iterator<Item = &'a str>,
) -> VecResult<OfficialStep> {
    let fields: Vec<&str> = fields.collect();
    let [index, selected, _count] = fields.as_slice() else {
        return Err(VecError::Malformed {
            path: path.to_path_buf(),
            line,
            keyword: "step",
            detail: format!("expected 3 fields, found {}", fields.len()),
        });
    };
    Ok(OfficialStep {
        index: number(path, line, "step", "index", index)?,
        selected: hex_bytes(path, line, "step", selected)?,
        tops: Vec::new(),
    })
}

/// `top <token-hex> <logprob>`
fn parse_official_top<'a>(
    path: &Path,
    line: usize,
    fields: impl Iterator<Item = &'a str>,
) -> VecResult<OfficialTop> {
    let fields: Vec<&str> = fields.collect();
    let [token, logprob] = fields.as_slice() else {
        return Err(VecError::Malformed {
            path: path.to_path_buf(),
            line,
            keyword: "top",
            detail: format!("expected 2 fields, found {}", fields.len()),
        });
    };
    Ok(OfficialTop {
        token: hex_bytes(path, line, "top", token)?,
        logprob: float(path, line, "top", "logprob", logprob)?,
    })
}

/// Parses an unsigned field, naming the column that was wrong.
fn number<T: std::str::FromStr>(
    path: &Path,
    line: usize,
    keyword: &'static str,
    column: &'static str,
    text: &str,
) -> VecResult<T> {
    text.parse().map_err(|_| VecError::Malformed {
        path: path.to_path_buf(),
        line,
        keyword,
        detail: format!("{column} is not a number: {text:?}"),
    })
}

/// Parses a float field, refusing the non-finite spellings outright.
///
/// A fixture carrying `NaN` or `inf` would compare equal to nothing and unequal
/// to nothing, which is the one way a golden file can fail open.
fn float(
    path: &Path,
    line: usize,
    keyword: &'static str,
    column: &'static str,
    text: &str,
) -> VecResult<f32> {
    let value: f32 = number(path, line, keyword, column, text)?;
    if !value.is_finite() {
        return Err(VecError::Invalid {
            path: path.to_path_buf(),
            line,
            detail: format!("{column} is not finite: {text:?}"),
        });
    }
    Ok(value)
}

/// Decodes a lowercase hex byte string, which is how a token's bytes are spelled.
fn hex_bytes(path: &Path, line: usize, keyword: &'static str, text: &str) -> VecResult<Vec<u8>> {
    if text.is_empty() || !text.len().is_multiple_of(2) {
        return Err(VecError::Malformed {
            path: path.to_path_buf(),
            line,
            keyword,
            detail: format!("token bytes must be an even-length hex string: {text:?}"),
        });
    }
    let mut out = Vec::with_capacity(text.len() / 2);
    let bytes = text.as_bytes();
    for pair in bytes.as_chunks::<2>().0 {
        let hi = nibble(pair[0]);
        let lo = nibble(pair[1]);
        match (hi, lo) {
            (Some(hi), Some(lo)) => out.push(hi << 4 | lo),
            _ => {
                return Err(VecError::Malformed {
                    path: path.to_path_buf(),
                    line,
                    keyword,
                    detail: format!("token bytes are not hex: {text:?}"),
                });
            }
        }
    }
    Ok(out)
}

/// One hex digit's value, or `None` when the byte is not one.
const fn nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Encodes bytes as the lowercase hex a fixture spells them with.
#[must_use]
pub fn hex_string(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// The whole-file rules, applied after every line has been read.
///
/// These are checked here rather than line by line because two of them relate a
/// case's header to the block that follows it, and a reader that enforced them
/// early would have to guess where a block ends.
fn validate(path: &Path, file: &VecFile) -> VecResult<()> {
    match &file.body {
        Body::Local(cases) => {
            for case in cases {
                let where_ = |detail: String| VecError::Invalid {
                    path: path.to_path_buf(),
                    line: 0,
                    detail: format!("case {}: {detail}", case.id),
                };
                if case.frontier == 0 {
                    return Err(where_("frontier must be at least one token".to_owned()));
                }
                if case.ctx <= case.frontier {
                    return Err(where_(format!(
                        "ctx must exceed frontier, got ctx {} frontier {}",
                        case.ctx, case.frontier
                    )));
                }
                if case.top_count == 0 || case.top_count as usize > MAX_TOP_ENTRIES {
                    return Err(where_(format!(
                        "top-count must be between 1 and {MAX_TOP_ENTRIES}, got {}",
                        case.top_count
                    )));
                }
                if !case.tops.is_empty() && case.tops.len() != case.top_count as usize {
                    return Err(where_(format!(
                        "declares {} entries but carries {}",
                        case.top_count,
                        case.tops.len()
                    )));
                }
            }
        }
        Body::Official(cases) => {
            for case in cases {
                for (position, step) in case.steps.iter().enumerate() {
                    if step.index as usize != position {
                        return Err(VecError::Invalid {
                            path: path.to_path_buf(),
                            line: 0,
                            detail: format!(
                                "case {}: step indices must be sequential from 0, found {} at \
                                 position {position}",
                                case.id, step.index
                            ),
                        });
                    }
                }
            }
        }
        Body::Tower(cases) => {
            for case in cases {
                if case.sample_count == 0 {
                    return Err(VecError::Invalid {
                        path: path.to_path_buf(),
                        line: 0,
                        detail: format!("case {}: sample count must be at least one", case.id),
                    });
                }
                if !case.samples.is_empty() && case.samples.len() != case.sample_count {
                    return Err(VecError::Invalid {
                        path: path.to_path_buf(),
                        line: 0,
                        detail: format!(
                            "case {}: declares {} samples but carries {}",
                            case.id,
                            case.sample_count,
                            case.samples.len()
                        ),
                    });
                }
            }
        }
    }
    Ok(())
}

/// Renders a fixture back to text, byte-stable for a given value.
#[must_use]
pub fn write(file: &VecFile) -> String {
    let mut out = String::new();
    let grammar = match &file.body {
        Body::Local(_) => Grammar::LocalLogit,
        Body::Official(_) => Grammar::OfficialLogprob,
        Body::Tower(_) => Grammar::VisionTower,
    };
    let _ = writeln!(out, "{}", grammar.header());
    for (key, value) in &file.headers {
        let _ = writeln!(out, "# {key} {value}");
    }
    for note in &file.notes {
        let _ = writeln!(out, "# {note}");
    }
    match &file.body {
        Body::Local(cases) => {
            for case in cases {
                let _ = writeln!(
                    out,
                    "case {} {} {} {} {} {}",
                    case.id,
                    case.mode.as_str(),
                    case.ctx,
                    case.frontier,
                    case.prompt.display(),
                    case.top_count
                );
                for (rank, entry) in case.tops.iter().enumerate() {
                    let _ = writeln!(out, "top {rank} {} {}", entry.token, entry.logit);
                }
            }
        }
        Body::Official(cases) => {
            for case in cases {
                let _ = writeln!(
                    out,
                    "case {} {} {} {}",
                    case.id,
                    case.ctx,
                    case.steps.len(),
                    case.prompt.display()
                );
                for step in &case.steps {
                    let _ = writeln!(
                        out,
                        "step {} {} {}",
                        step.index,
                        hex_string(&step.selected),
                        step.tops.len()
                    );
                    for top in &step.tops {
                        let _ = writeln!(out, "top {} {}", hex_string(&top.token), top.logprob);
                    }
                }
                let _ = writeln!(out, "end");
            }
        }
        Body::Tower(cases) => {
            for case in cases {
                let _ = writeln!(
                    out,
                    "case {} {} {}",
                    case.id,
                    case.image.display(),
                    case.sample_count
                );
                for sample in &case.samples {
                    let _ = writeln!(
                        out,
                        "sample {} {} {:.8e}",
                        sample.index, sample.pos, sample.value
                    );
                }
            }
        }
    }
    out
}
