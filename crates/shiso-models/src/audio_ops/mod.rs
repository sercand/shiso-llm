//! Launches of the `audio` kernel set, for every audio family to share.
//!
//! The audio encoders are built from the same handful of operations - norms,
//! convolutions, a windowed attention, a recurrent cell - applied in different
//! orders with different constants. Those operations live here, each behind a
//! descriptor struct that names every dimension, and each checks its buffers
//! against those dimensions before launching: a view that is too short is an
//! error naming the field, never a kernel reading past its end.
//!
//! Nothing here knows which model it serves. A family decides the order, the
//! constants and the weights; this module only turns a descriptor into a
//! launch.

mod args;
mod batch;
#[cfg(test)]
mod batch_tests;
#[cfg(test)]
mod decode_tests;
mod fused;
mod linear;
pub mod load;
mod runs;
#[cfg(test)]
mod span_tests;
pub mod store;
#[cfg(test)]
pub(crate) mod testing;
mod workspace;

use std::sync::Arc;

use shiso_core::{
    Backend, CoreError, CoreResult, DeviceBuffer, DeviceView, Extent3, KernelArg, KernelRef,
};

pub use batch::{
    BatchPlanes, CausalAttnSegDesc, DecodeAttnBatchDesc, DecodeStepBatchDesc, EmbedTokenBatchDesc,
    QkRopeAppendBatchDesc, QkRopeAppendSegDesc, SEQ_WORDS, SeqSpan, SeqTable,
};
pub use fused::{AddRmsNormDesc, ArgmaxWideDesc, QkRopeAppendDesc, argmax_wide_parts};
pub use linear::{GemmType, Linear, LinearWeight, step_silu, step3};
use runs::WeightView;
pub use workspace::Workspace;

use args::{
    ArgmaxArgs, AxpyArgs, BiasRowsArgs, CacheAttnArgs, Conv2dArgs, DecodeStepArgs, DwConv1dArgs,
    EmbedRowArgs, GeluMulArgs, GemmArgs, GluArgs, GreedyStepArgs, GreedyWindowArgs, KeepArgs,
    KvAppendArgs, LstmArgs, NormArgs, RopeArgs, RopePartialArgs, SiluMulArgs, SpanAttnArgs,
    TransposeArgs, UnaryArgs, UnfoldArgs, WindowAttnArgs,
};

/// The kernel module every launch here names.
pub const MODULE: &str = "audio";

/// `u32` words of greedy search's device state: frame, symbols on it, done,
/// last token, emitted, and padding.
pub const GREEDY_STATE_WORDS: usize = 8;
/// `u32` words of a causal decoder's device state; the words are the
/// `DECODE_*` indices below, and the rest is padding.
pub const DECODE_STATE_WORDS: usize = 8;
/// Decoder state word: cache rows filled, which is the position of the token
/// the next pass feeds. On the device only the step kernel moves it.
///
/// A prefill hands over by a host write: its rows are appended from a state
/// at position 0, and the host then writes `n`, the prefill's row count, into
/// this word with `copy_h2d` before it queues the step over the prefill's
/// argmax (which advances by 0). No kernel adds a prefill's length, so a
/// state whose position was not written starts decoding over row 0.
pub const DECODE_POS: usize = 0;
/// Decoder state word: tokens emitted.
pub const DECODE_COUNT: usize = 1;
/// Decoder state word: nonzero once the decoder has stopped.
pub const DECODE_DONE: usize = 2;
/// Decoder state word: the last token emitted, the one the next pass feeds.
pub const DECODE_TOKEN: usize = 3;
/// Decoder state word: whether the last step emitted.
pub const DECODE_EMIT: usize = 4;
/// Decoder state word: why the decoder stopped, [`DECODE_STOP_EOS`] or
/// [`DECODE_STOP_CAP`]; 0 while it runs.
pub const DECODE_STOP: usize = 5;
/// [`DECODE_STOP`]'s value after an end id.
pub const DECODE_STOP_EOS: u32 = 1;
/// [`DECODE_STOP`]'s value after the token cap.
pub const DECODE_STOP_CAP: u32 = 2;
/// What the step kernel writes to a slot whose step emitted nothing.
pub const DECODE_NO_TOKEN: u32 = u32::MAX;
/// Splits of decode attention: with eight kv heads, 256 simdgroups over the
/// cached rows. On an M1 Max (24-core GPU), 16 query heads over 8 kv heads
/// of 128: 16 splits took 1.6x as long over 4096 rows, and 64 or more
/// gained under 10% there (none with the fast exponential the kernel now
/// takes) while the merge's share grew at short lengths.
pub const DECODE_ATTN_SPLITS: usize = 32;

/// Floats of scratch decode attention needs for `heads` query heads of
/// `head_dim` cut into `splits`: an accumulator and a (max, sum) pair per
/// (head, split).
#[must_use]
pub const fn decode_attention_partials(heads: usize, head_dim: usize, splits: usize) -> usize {
    heads * splits * (head_dim + 2)
}
/// Threads per group of the elementwise, norm and layout kernels.
const LINEAR_BLOCK: u32 = 256;
/// Threadgroup bytes of a norm's block reduction.
const NORM_SMEM: u32 = 132;
/// Output tile of the GEMM, in both axes.
const GEMM_TILE: usize = 32;
/// Output columns one GEMM threadgroup produces.
const GEMM_TILE_N: usize = 64;
/// Output rows one GEMM threadgroup produces.
const GEMM_TILE_M: usize = 32;
/// Output rows one threadgroup of the 64-row GEMM produces; it takes every
/// product of more than [`GEMM_TILE_M`] rows whose depth is a whole number of
/// its k slices.
const GEMM64_TILE_M: usize = 64;
/// k one 64-row GEMM slice stages.
const GEMM64_TILE_K: usize = 16;
/// Queries one attention simdgroup takes at once.
const ATTN_QUERY_BLOCK: usize = 8;

/// Whether a product of `rows` rows on the audio GEMM sums each row as it
/// would among any other rows: past [`GEMM_TILE_M`] rows every weight
/// takes the 64-row tile or the unsplit 32-row tile, neither of which
/// splits k, so a row's sum depends on nothing but that row. Fewer rows
/// pick the matrix-vector kernel or a k split by the count, so rows
/// stacked with others would be summed apart from how they are summed
/// alone.
#[must_use]
pub(crate) const fn gemm_rows_sum_alone(rows: usize) -> bool {
    rows > GEMM_TILE_M
}

/// Distance rows one relative-score simdgroup takes at once.
const REL_ROWS: usize = 4;

/// Threadgroup bytes the attention stages: the query block.
const fn attention_smem(head_dim: usize) -> usize {
    ATTN_QUERY_BLOCK * head_dim * 4
}
/// Rows at or below which the matrix-vector kernel runs instead of the tiled
/// GEMM.
const GEMV_MAX_ROWS: usize = 4;
/// Threadgroups a GEMM launch aims for before it splits k.
const GEMM_TARGET_GROUPS: usize = 128;
/// Output tiles below which a GEMM splits k at all.
const GEMM_SPLIT_BELOW: usize = 64;
/// Output columns one matrix-vector threadgroup produces: four simdgroups of
/// four columns.
const GEMV_TG_COLS: usize = 16;
/// Rows one matrix-vector threadgroup produces.
const GEMV_TG_ROWS: usize = 8;
/// Input rows one threadgroup of the multi-row matrix-vector-by-runs kernels
/// takes: a decoder batch's rows share each weight run it reads.
const MATVEC_ROWS: usize = 4;

/// The matrix-vector-by-runs entry and grid for `rows` rows of `cols`
/// columns: one row on the one-row kernel, more on the multi-row one, which
/// reads each weight run once for up to [`MATVEC_ROWS`] rows and sums each
/// row exactly as the one-row kernel does - so a row's value never depends on
/// how many rows share its launch.
fn matvec_launch(
    weight: &WeightView<'_>,
    rows: usize,
    cols: usize,
    silu: bool,
) -> CoreResult<(&'static str, Extent3)> {
    let (entry, row_groups) = match (rows > 1, silu) {
        (false, false) => (weight.matvec_entry(), 1),
        (false, true) => (weight.matvec_silu_entry(), 1),
        (true, false) => (weight.matvec_rows_entry(), rows.div_ceil(MATVEC_ROWS)),
        (true, true) => (weight.matvec_silu_rows_entry(), rows.div_ceil(MATVEC_ROWS)),
    };
    Ok((
        entry,
        Extent3::new(
            dim(cols.div_ceil(MATVEC_COLS).max(1), "matvec.grid")?,
            dim(row_groups, "matvec.grid")?,
            1,
        ),
    ))
}

/// Output columns one matrix-vector-by-runs threadgroup produces.
const MATVEC_COLS: usize = 4;
/// Threads of one matrix-vector-by-runs threadgroup: two simdgroups sharing
/// each column's k range.
const MATVEC_BLOCK: u32 = 64;
/// Threads of one GEMM tile, and of one matrix-vector threadgroup: four
/// simdgroups.
const GEMM_BLOCK: u32 = 128;
/// Threadgroup bytes of one GEMM tile: two staged operands and the output.
const GEMM_SMEM: u32 = 12_288;
/// Threads of one 64-row GEMM tile: four simdgroups.
const GEMM64_BLOCK: u32 = 128;
/// Threadgroup bytes of one 64-row GEMM tile: 16 k of the activations and of
/// the weights, each staged row padded by four floats (the Metal layout; the
/// CUDA twin stages the same slices unpadded, in 8 KiB).
const GEMM64_SMEM: u32 = 9_472;
/// Largest head dimension the attention holds in a lane's registers.
const MAX_HEAD_DIM: usize = 128;
/// Head columns one span-attention simdgroup takes: its queries and its
/// slice of the output are eight fragments each, 16 floats a lane apiece.
const SPAN_SLICE: usize = 64;
/// Keys one span-attention block scores at once.
const SPAN_KEYS: usize = 16;
/// Widest head the span attention takes.
const SPAN_MAX_HEAD_DIM: usize = 512;
/// Span-attention simdgroups a threadgroup runs: row groups of eight query
/// items times the head's column slices, the row groups capped at four (32
/// items, the most the kernel's bounds table holds). Measured on an M1 Max
/// against 4 and 16, which were no faster at 256 wide and up to 40% slower at
/// 512 (see the kernel's notes).
const SPAN_SIMDGROUPS: usize = 8;
/// Most row groups of eight query items one threadgroup takes.
const SPAN_MAX_ROW_GROUPS: usize = 4;
/// Query heads of a kv head's group one decode-attention simdgroup scores
/// against each cached row it reads; a larger group takes more simdgroups.
const DECODE_GROUP_TILE: usize = 2;
/// Splits one decode-attention threadgroup runs, a simdgroup each.
const DECODE_SPLITS_PER_TG: usize = 4;
/// Most splits decode attention's merge is written for.
const DECODE_MAX_SPLITS: usize = 256;
/// Threads of the decode-attention merge, per head.
const DECODE_MERGE_THREADS: u32 = 256;
/// Largest depthwise conv1d the kernels are written for.
const MAX_TAPS: usize = 16;

/// An elementwise function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    /// `max(x, 0)`.
    Relu,
    /// `x * sigmoid(x)`.
    Silu,
    /// `1 / (1 + e^-x)`.
    Sigmoid,
    /// The exact GELU, `0.5 x (1 + erf(x / sqrt 2))`.
    GeluErf,
    /// `tanh(x)`.
    Tanh,
    /// `clamp(x, min, max)`.
    Clamp {
        /// Lower bound.
        min: f32,
        /// Upper bound.
        max: f32,
    },
    /// `x * s`.
    Scale(f32),
}

/// `out = f(input)` over `count` values; `out` may be `input`.
#[derive(Debug, Clone, Copy)]
pub struct UnaryDesc<'a> {
    /// The function.
    pub op: UnaryOp,
    /// Destination.
    pub out: DeviceView<'a>,
    /// Source.
    pub input: DeviceView<'a>,
    /// Values.
    pub count: usize,
}

/// `y += alpha * x` over `count` values.
#[derive(Debug, Clone, Copy)]
pub struct AxpyDesc<'a> {
    /// Accumulator.
    pub y: DeviceView<'a>,
    /// Addend.
    pub x: DeviceView<'a>,
    /// Scale on the addend.
    pub alpha: f32,
    /// Values.
    pub count: usize,
}

/// `y[r][c] += bias[c]`.
#[derive(Debug, Clone, Copy)]
pub struct BiasRowsDesc<'a> {
    /// Rows to add to, `[rows][cols]`.
    pub y: DeviceView<'a>,
    /// One row, `[cols]`.
    pub bias: DeviceView<'a>,
    /// Rows.
    pub rows: usize,
    /// Columns.
    pub cols: usize,
}

/// `out[r]` = the first index of row `r`'s largest value.
#[derive(Debug, Clone, Copy)]
pub struct ArgmaxDesc<'a> {
    /// One `u32` per row, `[rows]`.
    pub out: DeviceView<'a>,
    /// Rows to search, `[rows][cols]`.
    pub input: DeviceView<'a>,
    /// Rows.
    pub rows: usize,
    /// Columns.
    pub cols: usize,
}

/// `out[r][c] = input[r][c] * sigmoid(input[r][C + c])`.
#[derive(Debug, Clone, Copy)]
pub struct GluDesc<'a> {
    /// Destination, `[rows][channels]`.
    pub out: DeviceView<'a>,
    /// Source, `[rows][2 * channels]`.
    pub input: DeviceView<'a>,
    /// Rows.
    pub rows: usize,
    /// Output channels.
    pub channels: usize,
}

/// `out[i] = silu(gate[i]) * up[i]` over `count` values; `out` may be
/// `gate`.
#[derive(Debug, Clone, Copy)]
pub struct SiluMulDesc<'a> {
    /// Destination.
    pub out: DeviceView<'a>,
    /// The gate projection's output.
    pub gate: DeviceView<'a>,
    /// The value projection's output.
    pub up: DeviceView<'a>,
    /// Values.
    pub count: usize,
}

/// `out[r][c] = gelu_tanh(gate[r][c]) * up[r * up_stride + up_offset + c]`
/// over `rows` rows of `cols`, with the tanh form of GELU (`0.5 x (1 +
/// tanh(sqrt(2/pi) (x + 0.044715 x^3)))`); `out` may be `gate`.
#[derive(Debug, Clone, Copy)]
pub struct GeluMulDesc<'a> {
    /// Destination, `[rows][cols]`.
    pub out: DeviceView<'a>,
    /// The gate projection's output, `[rows][cols]`.
    pub gate: DeviceView<'a>,
    /// The value projection's output, row `r` at `r * up_stride + up_offset`.
    pub up: DeviceView<'a>,
    /// Rows.
    pub rows: usize,
    /// Values a row.
    pub cols: usize,
    /// Floats between the starts of two rows of `up`: `cols` for a plain
    /// gated GELU, more when `up` is one column block of a wider matrix.
    pub up_stride: usize,
    /// First column of the block `up` is read from, within each row.
    pub up_offset: usize,
}

/// A row normalization; `out` may be `input`.
#[derive(Debug, Clone, Copy)]
pub struct NormDesc<'a> {
    /// Destination, `[rows][dim]`.
    pub out: DeviceView<'a>,
    /// Source, `[rows][dim]`.
    pub input: DeviceView<'a>,
    /// Per-dimension scale; required for a layer norm.
    pub weight: Option<DeviceView<'a>>,
    /// Per-dimension offset; layer norm only.
    pub bias: Option<DeviceView<'a>>,
    /// Rows.
    pub rows: usize,
    /// Row width.
    pub dim: usize,
    /// Added to the variance or mean square.
    pub eps: f32,
}

/// The shape of a stride-2 3x3 conv2d over channels-last `[t][f][c]`, applied
/// to each of `batch` stacked planes on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2dGeometry {
    /// Input time rows.
    pub t_in: usize,
    /// Input frequency columns.
    pub f_in: usize,
    /// Input channels.
    pub c_in: usize,
    /// Output time rows.
    pub t_out: usize,
    /// Output frequency columns.
    pub f_out: usize,
    /// Output channels.
    pub c_out: usize,
    /// Zero rows ahead of the input in time.
    pub pad_t: usize,
    /// Zero columns ahead of the input in frequency.
    pub pad_f: usize,
    /// One input channel per output channel.
    pub depthwise: bool,
    /// Input rows at or past this read as zero.
    pub in_valid: usize,
    /// Output rows at or past this are written as zero.
    pub out_valid: usize,
    /// Independent planes, stacked: each is convolved as its own image, so
    /// its first rows read the padding and never the previous plane's last.
    /// The valid-row cuts apply within every plane.
    pub batch: usize,
}

/// One stride-2 3x3 conv2d.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dDesc<'a> {
    /// Destination, `[batch][t_out][f_out][c_out]`.
    pub out: DeviceView<'a>,
    /// Source, `[batch][t_in][f_in][c_in]`.
    pub input: DeviceView<'a>,
    /// `[c_out][c_in or 1][3][3]`.
    pub weight: DeviceView<'a>,
    /// `[c_out]`, when the conv has one.
    pub bias: Option<DeviceView<'a>>,
    /// Shape.
    pub geometry: Conv2dGeometry,
    /// Apply `max(0, .)` to the result.
    pub relu: bool,
}

/// The patches of a full stride-2 3x3 conv2d, `[batch][t_out][f_out][9 *
/// c_in]` with tap `kt * 3 + kf` major and the channel minor, so a GEMM
/// against the kernel laid out `[c_out][3][3][c_in]` is the conv. Output rows
/// past `out_valid` are zero patches, so the GEMM writes its bias there
/// rather than the zeros the direct conv does.
#[derive(Debug, Clone, Copy)]
pub struct Im2colDesc<'a> {
    /// Destination, `[batch][t_out][f_out][9 * c_in]`.
    pub out: DeviceView<'a>,
    /// Source, `[batch][t_in][f_in][c_in]`.
    pub input: DeviceView<'a>,
    /// Shape; `c_out` is unread, and a depthwise conv is refused.
    pub geometry: Conv2dGeometry,
}

/// A depthwise conv1d over rows the caller already left-padded.
#[derive(Debug, Clone, Copy)]
pub struct DwConv1dDesc<'a> {
    /// Destination, `[rows][channels]`.
    pub out: DeviceView<'a>,
    /// Source, `[rows + taps - 1][channels]`.
    pub input: DeviceView<'a>,
    /// `[channels][taps]`.
    pub weight: DeviceView<'a>,
    /// Output rows.
    pub rows: usize,
    /// Channels.
    pub channels: usize,
    /// Kernel width.
    pub taps: usize,
}

/// `out[t][j*C + c] = input[t + j - pad][c]`, zero outside the rows.
#[derive(Debug, Clone, Copy)]
pub struct UnfoldDesc<'a> {
    /// Destination, `[rows][taps * channels]`.
    pub out: DeviceView<'a>,
    /// Source, `[rows][channels]`.
    pub input: DeviceView<'a>,
    /// Rows.
    pub rows: usize,
    /// Channels.
    pub channels: usize,
    /// Neighbours per row.
    pub taps: usize,
    /// How far back the first neighbour sits.
    pub pad: usize,
}

/// `out[b][c][r] = input[b][r][c]`.
#[derive(Debug, Clone, Copy)]
pub struct TransposeDesc<'a> {
    /// Destination.
    pub out: DeviceView<'a>,
    /// Source.
    pub input: DeviceView<'a>,
    /// Independent planes.
    pub batch: usize,
    /// Rows per source plane.
    pub rows: usize,
    /// Columns per source plane.
    pub cols: usize,
}

/// Which keys a query may see, in absolute positions.
///
/// Key `ka` is visible to query `qa` when `ka < k_end` and
/// `0 <= qa / chunk - ka / chunk <= left_chunks`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnWindow {
    /// Absolute position of query row 0.
    pub q_abs0: usize,
    /// Absolute position of key row 0.
    pub k_abs0: usize,
    /// Keys at or past this absolute position are padding.
    pub k_end: usize,
    /// Chunk length of the visibility rule.
    pub chunk: usize,
    /// Past chunks a query may see besides its own.
    pub left_chunks: usize,
}

/// A precomputed relative-position key table, `[rows][heads * head_dim]`,
/// row `r` holding distance `d_min + r`.
#[derive(Debug, Clone, Copy)]
pub struct RelPos<'a> {
    /// The table.
    pub table: DeviceView<'a>,
    /// Distance of row 0.
    pub d_min: i32,
    /// Rows.
    pub rows: usize,
}

/// Attention over a bounded window with an optional relative-position term.
#[derive(Debug, Clone, Copy)]
pub struct WindowAttnDesc<'a> {
    /// Destination, `[n_q][heads * head_dim]`.
    pub out: DeviceView<'a>,
    /// Queries.
    pub q: DeviceView<'a>,
    /// Keys, `[n_k][heads * head_dim]`.
    pub k: DeviceView<'a>,
    /// Values, as the keys.
    pub v: DeviceView<'a>,
    /// Query rows.
    pub n_q: usize,
    /// Key rows.
    pub n_k: usize,
    /// Heads.
    pub heads: usize,
    /// Width of one head.
    pub head_dim: usize,
    /// Visibility.
    pub window: AttnWindow,
    /// Content and position biases added to the query, `[heads * head_dim]` each.
    pub bias_uv: Option<(DeviceView<'a>, DeviceView<'a>)>,
    /// Per-dimension query scale, `[head_dim]`.
    pub q_dim_scale: Option<DeviceView<'a>>,
    /// Relative-position keys.
    pub rel: Option<RelPos<'a>>,
    /// Scale on the whole logit.
    pub scale: f32,
    /// Extra scale on the content term only.
    pub ac_scale: f32,
    /// `cap * tanh(logit / cap)` when set.
    pub softcap: Option<f32>,
}

/// Half-split rotary embedding, in place.
#[derive(Debug, Clone, Copy)]
pub struct RopeDesc<'a> {
    /// `[rows][heads * head_dim]`.
    pub x: DeviceView<'a>,
    /// Rows.
    pub rows: usize,
    /// Heads.
    pub heads: usize,
    /// Width of one head, even.
    pub head_dim: usize,
    /// Position of row 0.
    pub pos0: usize,
    /// Frequency base.
    pub theta: f32,
}

/// Half-split rotary embedding over the first `rot_dims` of each head only, in
/// place.
#[derive(Debug, Clone, Copy)]
pub struct RopePartialDesc<'a> {
    /// `[rows][heads * head_dim]`.
    pub x: DeviceView<'a>,
    /// Rows.
    pub rows: usize,
    /// Heads.
    pub heads: usize,
    /// Width of one head.
    pub head_dim: usize,
    /// Leading dimensions of a head that turn, even and at most `head_dim`.
    pub rot_dims: usize,
    /// Position of row 0.
    pub pos0: usize,
    /// Frequency base.
    pub theta: f32,
}

/// Half-split rotary embedding at a position read per row, in place: rows of
/// several packed sequences each count from their own start.
#[derive(Debug, Clone, Copy)]
pub struct RopePosDesc<'a> {
    /// `[rows][heads * head_dim]`.
    pub x: DeviceView<'a>,
    /// Each row's position, u32 `[rows]`.
    pub positions: DeviceView<'a>,
    /// Rows.
    pub rows: usize,
    /// Heads.
    pub heads: usize,
    /// Width of one head, even.
    pub head_dim: usize,
    /// Frequency base.
    pub theta: f32,
}

/// One step of the recurrent cell under greedy search's emit flag.
#[derive(Debug, Clone, Copy)]
pub struct LstmDesc<'a> {
    /// The new hidden state, always: the next layer's input, `[hidden]`.
    pub x_out: DeviceView<'a>,
    /// Hidden state, replaced by the new one only when the step emitted.
    pub h: DeviceView<'a>,
    /// Cell state, likewise.
    pub c: DeviceView<'a>,
    /// Gates in (i, f, g, o) order with biases folded in, `[4 * hidden]`.
    pub gates: DeviceView<'a>,
    /// The search state whose emit flag decides.
    pub state: DeviceView<'a>,
    /// Hidden width.
    pub hidden: usize,
}

/// The joint's input rows for one greedy step.
#[derive(Debug, Clone, Copy)]
pub struct GreedyWindowDesc<'a> {
    /// `[rows][cols]`.
    pub out: DeviceView<'a>,
    /// Projected encoder frames not yet searched past, `[avail][cols]`.
    pub pending: DeviceView<'a>,
    /// The prediction network's projection, `[cols]`.
    pub dec: DeviceView<'a>,
    /// The search state; its frame is the window's first row.
    pub state: DeviceView<'a>,
    /// Window rows.
    pub rows: usize,
    /// Joint width.
    pub cols: usize,
    /// Pending rows.
    pub avail: usize,
}

/// One decision of greedy search.
#[derive(Debug, Clone, Copy)]
pub struct GreedyStepDesc<'a> {
    /// The search state, read and advanced.
    pub state: DeviceView<'a>,
    /// `(token, frame)` pairs of `u32`, one per step of a batch.
    pub out: DeviceView<'a>,
    /// The window's argmax, `[rows]` of `u32`.
    pub ids: DeviceView<'a>,
    /// Window rows.
    pub rows: usize,
    /// Pending rows.
    pub avail: usize,
    /// The blank token.
    pub blank: u32,
    /// Symbols one frame may emit.
    pub max_symbols: usize,
    /// This step's pair in `out`.
    pub slot: usize,
}

/// The embedding row of the search state's last token.
#[derive(Debug, Clone, Copy)]
pub struct EmbedRowDesc<'a> {
    /// `[width]`.
    pub out: DeviceView<'a>,
    /// `[vocab][width]`.
    pub table: DeviceView<'a>,
    /// The search state.
    pub state: DeviceView<'a>,
    /// Row width.
    pub width: usize,
    /// Rows in the table.
    pub vocab: usize,
}

/// `dst = src` when the search state's last step emitted.
#[derive(Debug, Clone, Copy)]
pub struct KeepDesc<'a> {
    /// `[count]`.
    pub dst: DeviceView<'a>,
    /// `[count]`.
    pub src: DeviceView<'a>,
    /// The search state.
    pub state: DeviceView<'a>,
    /// Elements.
    pub count: usize,
}

/// One layer's key and value cache planes, each `[kv_heads][cap][head_dim]`
/// f32: the layout the cache kernels read, so a layer is two views.
#[derive(Debug, Clone, Copy)]
pub struct CachePlanes<'a> {
    /// Keys.
    pub k: DeviceView<'a>,
    /// Values.
    pub v: DeviceView<'a>,
    /// Key and value heads.
    pub kv_heads: usize,
    /// Rows each head's plane holds.
    pub cap: usize,
    /// Width of one head.
    pub head_dim: usize,
}

/// Causal grouped-query attention of many queries over a cache: query `i`
/// sits at `pos0 + i` and sees cache rows `0..=pos0 + i`.
#[derive(Debug, Clone, Copy)]
pub struct CausalAttnDesc<'a> {
    /// Destination, `[n_q][heads * head_dim]`.
    pub out: DeviceView<'a>,
    /// Queries, as the destination.
    pub q: DeviceView<'a>,
    /// The layer's cache, already holding rows `0..pos0 + n_q`.
    pub cache: CachePlanes<'a>,
    /// Query rows.
    pub n_q: usize,
    /// Query heads, a multiple of the cache's kv heads.
    pub heads: usize,
    /// Absolute position of query row 0.
    pub pos0: usize,
    /// Scale on the logit.
    pub scale: f32,
}

/// Grouped-query attention where query `i` sees keys `j` with
/// `max(start_i, i - left) <= j <= min(end_i - 1, i + right)`, `[start_i,
/// end_i)` being the rows of the sequence row `i` belongs to (no cache).
#[derive(Debug, Clone, Copy)]
pub struct SpanAttnDesc<'a> {
    /// Destination, `[n][heads * head_dim]`.
    pub out: DeviceView<'a>,
    /// Queries, as the destination.
    pub q: DeviceView<'a>,
    /// Keys, row-major `[n][kv_heads * head_dim]` (not cache planes).
    pub k: DeviceView<'a>,
    /// Values, as the keys.
    pub v: DeviceView<'a>,
    /// Several sequences packed in one pass: per row, the u32 pair
    /// `[start, end)` of its sequence's rows, `[n][2]`. A row outside its
    /// own range, or an empty range, sees no key and comes out zero. `None`
    /// is one sequence, `[0, n)`.
    pub segments: Option<DeviceView<'a>>,
    /// Rows: queries, keys and values alike.
    pub n: usize,
    /// Query heads, a multiple of `kv_heads`.
    pub heads: usize,
    /// Key and value heads; query head `h` reads kv head
    /// `h / (heads / kv_heads)`.
    pub kv_heads: usize,
    /// Width of one head, a multiple of 32 up to 512.
    pub head_dim: usize,
    /// Keys visible to the left of each query; `n` or more is unbounded.
    pub left: usize,
    /// Keys visible to the right of each query: 0 is causal, `n` or more is
    /// fully bidirectional.
    pub right: usize,
    /// Scale on the logit.
    pub scale: f32,
}

/// The one query of a decode step over a cache, at the decoder state's
/// position.
#[derive(Debug, Clone, Copy)]
pub struct DecodeAttnDesc<'a> {
    /// Destination, `[heads * head_dim]`.
    pub out: DeviceView<'a>,
    /// Scratch for the splits' partial results, at least
    /// [`decode_attention_partials`] floats; the caller keeps one for the
    /// whole decode rather than allocating per step.
    pub partials: DeviceView<'a>,
    /// Runs the cached rows are cut into, each on its own simdgroup per kv
    /// head. The length is only known on the device, so this is fixed per
    /// launch; [`DECODE_ATTN_SPLITS`] is the measured default.
    pub splits: usize,
    /// The query, as the destination.
    pub q: DeviceView<'a>,
    /// The layer's cache, holding rows `0..=state[DECODE_POS]`.
    pub cache: CachePlanes<'a>,
    /// Query heads, a multiple of the cache's kv heads.
    pub heads: usize,
    /// The decoder state, whose position sets the rows seen.
    pub state: DeviceView<'a>,
    /// Scale on the logit.
    pub scale: f32,
}

/// New key and value rows into a layer's cache at the decoder state's
/// position.
#[derive(Debug, Clone, Copy)]
pub struct KvAppendDesc<'a> {
    /// The layer's cache.
    pub cache: CachePlanes<'a>,
    /// Keys, `[rows][kv_heads * head_dim]`.
    pub k: DeviceView<'a>,
    /// Values, as the keys.
    pub v: DeviceView<'a>,
    /// The decoder state; its position is the first row written.
    pub state: DeviceView<'a>,
    /// Rows.
    pub rows: usize,
}

/// Half-split rotary embedding at the decoder state's position, in place.
#[derive(Debug, Clone, Copy)]
pub struct RopeStateDesc<'a> {
    /// `[rows][heads * head_dim]`.
    pub x: DeviceView<'a>,
    /// Rows.
    pub rows: usize,
    /// Heads.
    pub heads: usize,
    /// Width of one head, even.
    pub head_dim: usize,
    /// Added to the state's position for row 0.
    pub offset: usize,
    /// Frequency base.
    pub theta: f32,
    /// The decoder state.
    pub state: DeviceView<'a>,
}

/// One decision of a causal decoder.
#[derive(Debug, Clone, Copy)]
pub struct DecodeStepDesc<'a> {
    /// The decoder state, read and advanced.
    pub state: DeviceView<'a>,
    /// One `u32` per step of a batch.
    pub out: DeviceView<'a>,
    /// The head's argmax, one `u32`.
    pub ids: DeviceView<'a>,
    /// The ids that end the text; either stops the decoder and is not
    /// emitted.
    pub eos: [u32; 2],
    /// Tokens the decoder may emit in all; the last is kept.
    pub max_tokens: usize,
    /// Cache rows the pass before this step appended: 0 for the step over the
    /// prefill's argmax, 1 after a decode pass.
    pub advance: usize,
    /// This step's slot in `out`.
    pub slot: usize,
}

/// A `[vocab][width]` embedding table as it is stored.
#[derive(Debug, Clone, Copy)]
pub enum EmbedTable<'a> {
    /// A dense or `Q8_0` table, read as the audio GEMM reads that weight type.
    Dense {
        /// Storage.
        dtype: GemmType,
        /// The rows.
        rows: DeviceView<'a>,
    },
    /// An 8-bit affine table: per `group` elements a bf16 scale and bias.
    Affine8 {
        /// The bytes, four to a little-endian `u32`, `[vocab][width]`.
        words: DeviceView<'a>,
        /// bf16, `[vocab][width / group]`.
        scales: DeviceView<'a>,
        /// bf16, as the scales.
        biases: DeviceView<'a>,
        /// Elements per group.
        group: usize,
    },
}

/// The f32 embedding row of the decoder state's last token.
#[derive(Debug, Clone, Copy)]
pub struct EmbedTokenDesc<'a> {
    /// `[width]`.
    pub out: DeviceView<'a>,
    /// The table.
    pub table: EmbedTable<'a>,
    /// The decoder state.
    pub state: DeviceView<'a>,
    /// Row width.
    pub width: usize,
    /// Rows in the table.
    pub vocab: usize,
}

/// `n` as `f64`; exact for every count a model here holds.
#[expect(
    clippy::cast_precision_loss,
    reason = "counts stay below 2^52, where f64 is exact"
)]
pub(crate) fn f64_of(n: usize) -> f64 {
    n as f64
}

/// A dimension as the `u32` a kernel takes, refused if it does not fit.
fn dim(value: usize, key: &'static str) -> CoreResult<u32> {
    u32::try_from(value).map_err(|_| CoreError::InvalidValue {
        key,
        value: value.to_string(),
    })
}

/// Refuses a view shorter than `floats` f32 values.
fn need(view: DeviceView<'_>, floats: usize, field: &'static str) -> CoreResult<()> {
    let want = u64::try_from(floats).unwrap_or(u64::MAX).saturating_mul(4);
    if view.len() < want {
        return Err(CoreError::ShapeMismatch {
            field,
            expected: want,
            got: view.len(),
        });
    }
    Ok(())
}

/// Refuses a value outside what a kernel is written for.
fn limit(ok: bool, key: &'static str, value: usize) -> CoreResult<()> {
    if ok {
        Ok(())
    } else {
        Err(CoreError::InvalidValue {
            key,
            value: value.to_string(),
        })
    }
}

/// Refuses a view shorter than `bytes`.
fn need_bytes(view: DeviceView<'_>, bytes: usize, field: &'static str) -> CoreResult<()> {
    let want = u64::try_from(bytes).unwrap_or(u64::MAX);
    if view.len() < want {
        return Err(CoreError::ShapeMismatch {
            field,
            expected: want,
            got: view.len(),
        });
    }
    Ok(())
}

/// Refuses cache planes shorter than their geometry, or a geometry no cache
/// kernel is written for.
fn need_cache(c: &CachePlanes<'_>, heads: usize, what: &'static str) -> CoreResult<()> {
    limit(
        (1..=MAX_HEAD_DIM).contains(&c.head_dim),
        "cache.head_dim",
        c.head_dim,
    )?;
    limit(
        c.kv_heads > 0 && heads.is_multiple_of(c.kv_heads),
        what,
        heads,
    )?;
    let plane = c.kv_heads * c.cap * c.head_dim;
    need(c.k, plane, "cache.k")?;
    need(c.v, plane, "cache.v")
}

/// Refuses a view a four-wide read cannot start at.
fn aligned16(view: DeviceView<'_>, field: &'static str) -> CoreResult<()> {
    if view.offset().is_multiple_of(16) {
        Ok(())
    } else {
        Err(CoreError::InvalidValue {
            key: field,
            value: format!("offset {} is not 16-byte aligned", view.offset()),
        })
    }
}

fn linear_grid(count: usize) -> CoreResult<Extent3> {
    Ok(Extent3::linear(dim(
        count.div_ceil(LINEAR_BLOCK as usize).max(1),
        "grid",
    )?))
}

/// A GEMM's bindings: the arguments, the destination, the input, the weight,
/// the bias (the input when there is none), then an affine weight's scales
/// and biases.
fn gemm_bind<'a>(
    args: &'a GemmArgs,
    target: DeviceView<'a>,
    input: DeviceView<'a>,
    weight: &WeightView<'a>,
    bias: Option<DeviceView<'a>>,
) -> Vec<KernelArg<'a>> {
    let mut b = vec![
        KernelArg::Bytes(bytemuck::bytes_of(args)),
        KernelArg::Buffer(target),
        KernelArg::Buffer(input),
        KernelArg::Buffer(weight.main()),
        KernelArg::Buffer(bias.unwrap_or(input)),
    ];
    b.extend(weight.aux());
    b
}

/// Launches the audio kernel set on one backend.
#[derive(Debug, Clone)]
pub struct Ops {
    backend: Arc<dyn Backend>,
}

impl Ops {
    /// Binds the ops to a backend.
    #[must_use]
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend }
    }

    /// The backend every launch goes to.
    #[must_use]
    pub fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    /// A zero-filled buffer of `floats` f32 values.
    ///
    /// # Errors
    ///
    /// Allocation or fill failures from the backend.
    pub fn scratch(&self, floats: usize) -> CoreResult<DeviceBuffer> {
        let bytes = u64::try_from(floats.max(1))
            .unwrap_or(u64::MAX)
            .saturating_mul(4);
        let buffer = shiso_core::alloc(&self.backend, bytes)?;
        self.backend.fill_zero(buffer.view())?;
        Ok(buffer)
    }

    /// A buffer of `floats` f32 values whose contents are undefined, for
    /// one every element of which a kernel writes before any is read:
    /// [`Self::scratch`]'s fill costs a pass over the memory, and on a fresh
    /// buffer that pass is also its first touch.
    ///
    /// # Errors
    ///
    /// Allocation failures from the backend.
    pub fn scratch_uninit(&self, floats: usize) -> CoreResult<DeviceBuffer> {
        let bytes = u64::try_from(floats.max(1))
            .unwrap_or(u64::MAX)
            .saturating_mul(4);
        shiso_core::alloc(&self.backend, bytes)
    }

    fn launch(
        &self,
        entry: &'static str,
        grid: Extent3,
        block: u32,
        smem: u32,
        args: &[KernelArg<'_>],
    ) -> CoreResult<()> {
        shiso_engine::launch::run(
            &*self.backend,
            KernelRef::new(MODULE, entry),
            grid,
            Extent3::linear(block),
            smem,
            args,
        )
    }

    /// `out = f(input)`.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn unary(&self, d: &UnaryDesc<'_>) -> CoreResult<()> {
        need(d.out, d.count, "unary.out")?;
        need(d.input, d.count, "unary.input")?;
        let (op, a, b) = match d.op {
            UnaryOp::Relu => (0, 0.0, 0.0),
            UnaryOp::Silu => (1, 0.0, 0.0),
            UnaryOp::Sigmoid => (2, 0.0, 0.0),
            UnaryOp::GeluErf => (3, 0.0, 0.0),
            UnaryOp::Tanh => (4, 0.0, 0.0),
            UnaryOp::Clamp { min, max } => (5, min, max),
            UnaryOp::Scale(s) => (6, s, 0.0),
        };
        let args = UnaryArgs {
            count: dim(d.count, "unary.count")?,
            op,
            a,
            b,
        };
        self.launch(
            "kernel_audio_unary_f32",
            linear_grid(d.count)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.input),
            ],
        )
    }

    /// `y += alpha * x`.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn axpy(&self, d: &AxpyDesc<'_>) -> CoreResult<()> {
        need(d.y, d.count, "axpy.y")?;
        need(d.x, d.count, "axpy.x")?;
        let args = AxpyArgs {
            count: dim(d.count, "axpy.count")?,
            alpha: d.alpha,
        };
        self.launch(
            "kernel_audio_axpy_f32",
            linear_grid(d.count)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.y),
                KernelArg::Buffer(d.x),
            ],
        )
    }

    /// `y[r][c] += bias[c]`.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn bias_rows(&self, d: &BiasRowsDesc<'_>) -> CoreResult<()> {
        let count = d.rows * d.cols;
        need(d.y, count, "bias_rows.y")?;
        need(d.bias, d.cols, "bias_rows.bias")?;
        let args = BiasRowsArgs {
            rows: dim(d.rows, "bias_rows.rows")?,
            cols: dim(d.cols, "bias_rows.cols")?,
        };
        self.launch(
            "kernel_audio_bias_rows_f32",
            linear_grid(count)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.y),
                KernelArg::Buffer(d.bias),
            ],
        )
    }

    /// Per row, the first index of the largest value: a tie keeps the lowest
    /// index and a NaN never wins, as a strictly-greater scan does. Greedy
    /// search downloads these ids instead of whole rows of logits.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn argmax_rows(&self, d: &ArgmaxDesc<'_>) -> CoreResult<()> {
        need(d.out, d.rows, "argmax.out")?;
        need(d.input, d.rows * d.cols, "argmax.input")?;
        let args = ArgmaxArgs {
            rows: dim(d.rows, "argmax.rows")?,
            cols: dim(d.cols, "argmax.cols")?,
        };
        self.launch(
            "kernel_audio_argmax_rows_f32",
            Extent3::new(dim(d.rows.max(1), "argmax.grid")?, 1, 1),
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.input),
            ],
        )
    }

    /// The gated linear unit.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn glu(&self, d: &GluDesc<'_>) -> CoreResult<()> {
        let count = d.rows * d.channels;
        need(d.out, count, "glu.out")?;
        need(d.input, 2 * count, "glu.input")?;
        let args = GluArgs {
            rows: dim(d.rows, "glu.rows")?,
            channels: dim(d.channels, "glu.channels")?,
        };
        self.launch(
            "kernel_audio_glu_f32",
            linear_grid(count)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.input),
            ],
        )
    }

    /// A SiLU-gated MLP's product of its gate and value projections.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn silu_mul(&self, d: &SiluMulDesc<'_>) -> CoreResult<()> {
        need(d.out, d.count, "silu_mul.out")?;
        need(d.gate, d.count, "silu_mul.gate")?;
        need(d.up, d.count, "silu_mul.up")?;
        let args = SiluMulArgs {
            count: dim(d.count, "silu_mul.count")?,
        };
        self.launch(
            "kernel_audio_silu_mul_f32",
            linear_grid(d.count)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.gate),
                KernelArg::Buffer(d.up),
            ],
        )
    }

    /// A gated-GELU MLP's product of its gate and value projections, the value read
    /// from a strided column block.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for a column block that does not fit in
    /// `up_stride`; [`CoreError::ShapeMismatch`] for a short view; or the
    /// launch's error.
    pub fn gelu_tanh_mul(&self, d: &GeluMulDesc<'_>) -> CoreResult<()> {
        // A block running past its row would read the next row's columns, a
        // plausible wrong value rather than a fault.
        limit(
            d.up_offset + d.cols <= d.up_stride,
            "gelu_tanh_mul.up_stride",
            d.up_stride,
        )?;
        let count = d.rows * d.cols;
        need(d.out, count, "gelu_tanh_mul.out")?;
        need(d.gate, count, "gelu_tanh_mul.gate")?;
        if d.rows > 0 {
            need(
                d.up,
                (d.rows - 1) * d.up_stride + d.up_offset + d.cols,
                "gelu_tanh_mul.up",
            )?;
        }
        let args = GeluMulArgs {
            rows: dim(d.rows, "gelu_tanh_mul.rows")?,
            cols: dim(d.cols, "gelu_tanh_mul.cols")?,
            up_stride: dim(d.up_stride, "gelu_tanh_mul.up_stride")?,
            up_offset: dim(d.up_offset, "gelu_tanh_mul.up_offset")?,
        };
        // The kernel's flat index is a u32.
        dim(count, "gelu_tanh_mul.count")?;
        self.launch(
            "kernel_audio_gelu_tanh_mul_f32",
            linear_grid(count)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.gate),
                KernelArg::Buffer(d.up),
            ],
        )
    }

    fn norm(&self, d: &NormDesc<'_>, entry: &'static str, layer: bool) -> CoreResult<()> {
        // A zero-width row has nothing to normalise and no first value for the
        // layer norm's pivot; the kernel would read past an empty view.
        limit(d.dim > 0, "norm.dim", d.dim)?;
        let count = d.rows * d.dim;
        need(d.out, count, "norm.out")?;
        need(d.input, count, "norm.input")?;
        if layer && d.weight.is_none() {
            return Err(CoreError::ShapeMismatch {
                field: "layer_norm.weight",
                expected: u64::try_from(d.dim).unwrap_or(u64::MAX) * 4,
                got: 0,
            });
        }
        if let Some(w) = d.weight {
            need(w, d.dim, "norm.weight")?;
        }
        if let Some(b) = d.bias {
            need(b, d.dim, "norm.bias")?;
        }
        let mut flags = 0u32;
        if d.weight.is_some() {
            flags |= 1;
        }
        if d.bias.is_some() {
            flags |= 2;
        }
        let args = NormArgs {
            rows: dim(d.rows, "norm.rows")?,
            dim: dim(d.dim, "norm.dim")?,
            eps: d.eps,
            flags,
        };
        let bytes = bytemuck::bytes_of(&args);
        let weight = d.weight.unwrap_or(d.input);
        let mut kargs = vec![
            KernelArg::Bytes(bytes),
            KernelArg::Buffer(d.out),
            KernelArg::Buffer(d.input),
            KernelArg::Buffer(weight),
        ];
        if layer {
            kargs.push(KernelArg::Buffer(d.bias.unwrap_or(d.input)));
        }
        self.launch(
            entry,
            Extent3::linear(dim(d.rows.max(1), "norm.rows")?),
            LINEAR_BLOCK,
            NORM_SMEM,
            &kargs,
        )
    }

    /// Layer norm with an f32 weight and an optional bias.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a short view or a missing weight, or
    /// the launch's error.
    pub fn layer_norm(&self, d: &NormDesc<'_>) -> CoreResult<()> {
        self.norm(d, "kernel_audio_layer_norm_f32", true)
    }

    /// RMS norm with an optional f32 weight.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn rms_norm(&self, d: &NormDesc<'_>) -> CoreResult<()> {
        self.norm(d, "kernel_audio_rms_norm_f32", false)
    }

    /// `y = x * W^T (+ b)` on the audio GEMM, for weights the engine's
    /// projection does not take.
    pub(crate) fn gemm(
        &self,
        dtype: GemmType,
        out: DeviceView<'_>,
        input: DeviceView<'_>,
        weight: DeviceView<'_>,
        bias: Option<DeviceView<'_>>,
        shape: (usize, usize, usize),
    ) -> CoreResult<()> {
        self.gemm_of(WeightView::Dense(dtype, weight), out, input, bias, shape)
    }

    /// [`Self::gemm`] over any weight the audio set reads. An affine weight
    /// has no matrix-vector kernel of the old order, so a few rows of one
    /// take the 32-row tile.
    pub(crate) fn gemm_of(
        &self,
        weight: WeightView<'_>,
        out: DeviceView<'_>,
        input: DeviceView<'_>,
        bias: Option<DeviceView<'_>>,
        shape: (usize, usize, usize),
    ) -> CoreResult<()> {
        // A few rows go to the matrix-vector kernel, which spreads the
        // columns over enough threadgroups to stream the weights at the
        // memory's rate; more rows go to the tiled GEMM.
        let skinny = shape.0 <= GEMV_MAX_ROWS && matches!(weight, WeightView::Dense(..));
        let tall = shape.0 > GEMM_TILE_M;
        // The 64-row tile takes whole slices of an aligned k and never
        // splits; any other tall product takes the 32-row tile, also
        // unsplit, whose every element is summed exactly as the 64-row
        // tile's would be.
        let wide =
            tall && shape.2.is_multiple_of(GEMM64_TILE_K) && input.offset().is_multiple_of(16);
        let (entry32, entry64) = weight.tiled_entries();
        let entry = match weight {
            WeightView::Dense(dtype, _) if skinny => dtype.gemv_entry(),
            _ if wide => entry64,
            _ => entry32,
        };
        let (tile_m, block, smem) = if wide {
            (GEMM64_TILE_M, GEMM64_BLOCK, GEMM64_SMEM)
        } else {
            (GEMM_TILE_M, GEMM_BLOCK, GEMM_SMEM)
        };
        let (rows, cols, depth) = shape;
        need(out, rows * cols, "gemm.y")?;
        need(input, rows * depth, "gemm.x")?;
        if let Some(b) = bias {
            need(b, cols, "gemm.bias")?;
        }
        // Split k when the output tiles alone are too few to keep the device
        // busy: a few rows against a thousand columns is sixteen threadgroups,
        // and each would stream its whole slab of weights alone. The 64-row
        // tile never splits: at a few hundred rows it already fills the
        // device (splitting it measured a wash on M1 Max), and summing the
        // slices moved the diarizer's stacked projection from bit-equal with
        // the reference to 3e-6 of it.
        let tiles = cols.div_ceil(GEMM_TILE_N).max(1) * rows.div_ceil(tile_m).max(1);
        let k_tiles = depth.div_ceil(GEMM_TILE).max(1);
        let wanted = if skinny || tall || tiles >= GEMM_SPLIT_BELOW {
            1
        } else {
            GEMM_TARGET_GROUPS.div_ceil(tiles).clamp(1, k_tiles)
        };
        let k_chunk = k_tiles.div_ceil(wanted) * GEMM_TILE;
        let splits = depth.div_ceil(k_chunk).max(1);
        let args = GemmArgs {
            m: dim(rows, "gemm.m")?,
            n: dim(cols, "gemm.n")?,
            k: dim(depth, "gemm.k")?,
            flags: u32::from(bias.is_some()),
            k_chunk: dim(k_chunk, "gemm.k_chunk")?,
            splits: dim(splits, "gemm.splits")?,
        };
        if skinny {
            let grid = Extent3::new(
                dim(cols.div_ceil(GEMV_TG_COLS).max(1), "gemm.grid")?,
                dim(rows.div_ceil(GEMV_TG_ROWS).max(1), "gemm.grid")?,
                1,
            );
            return self.launch(
                entry,
                grid,
                GEMM_BLOCK,
                0,
                &gemm_bind(&args, out, input, &weight, bias),
            );
        }
        let grid = Extent3::new(
            dim(cols.div_ceil(GEMM_TILE_N).max(1), "gemm.grid")?,
            dim(rows.div_ceil(tile_m).max(1), "gemm.grid")?,
            dim(splits, "gemm.grid")?,
        );
        // Every in-bounds element of every plane is written, so the planes
        // need no clearing.
        let parts = if splits > 1 {
            Some(shiso_core::alloc(
                &self.backend,
                u64::try_from(splits * rows * cols * 4).unwrap_or(u64::MAX),
            )?)
        } else {
            None
        };
        let target = parts.as_ref().map_or(out, DeviceBuffer::view);
        self.launch(
            entry,
            grid,
            block,
            smem,
            &gemm_bind(&args, target, input, &weight, bias),
        )?;
        let Some(parts) = parts else {
            return Ok(());
        };
        self.launch(
            "kernel_audio_split_sum_f32",
            linear_grid(rows * cols)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out),
                KernelArg::Buffer(parts.view()),
                KernelArg::Buffer(bias.unwrap_or(input)),
            ],
        )
    }

    /// `y = x * W^T (+ b)` for a decoder step's few rows: each lane dots
    /// runs of eight consecutive weights, which streams a row-major weight at
    /// about twice the rate [`Self::gemm`]'s matrix-vector kernel does. The
    /// summation order is not that kernel's, so a model takes one or the
    /// other for a given product, never a mix. More than one row takes the
    /// multi-row kernel, which reads each weight run once for up to four
    /// rows and gives each row the one-row kernel's value, so a decoder's
    /// batch of rows costs about one row's weight traffic.
    ///
    /// Returns `false`, launching nothing, when `k` is not a whole number of
    /// runs or a view is not 16-byte aligned, so the caller takes
    /// [`Self::gemm`] instead.
    pub(crate) fn matvec(
        &self,
        weight: WeightView<'_>,
        out: DeviceView<'_>,
        input: DeviceView<'_>,
        bias: Option<DeviceView<'_>>,
        shape: (usize, usize, usize),
    ) -> CoreResult<bool> {
        let (rows, cols, depth) = shape;
        if !weight.runs_fit(depth) || !input.offset().is_multiple_of(16) {
            return Ok(false);
        }
        need(out, rows * cols, "matvec.y")?;
        need(input, rows * depth, "matvec.x")?;
        if let Some(b) = bias {
            need(b, cols, "matvec.bias")?;
        }
        let args = GemmArgs {
            m: dim(rows, "matvec.m")?,
            n: dim(cols, "matvec.n")?,
            k: dim(depth, "matvec.k")?,
            flags: u32::from(bias.is_some()),
            k_chunk: dim(depth, "matvec.k")?,
            splits: 1,
        };
        let (entry, grid) = matvec_launch(&weight, rows, cols, false)?;
        let mut bind = vec![
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out),
            KernelArg::Buffer(input),
            KernelArg::Buffer(weight.main()),
            KernelArg::Buffer(bias.unwrap_or(input)),
        ];
        bind.extend(weight.aux());
        self.launch(entry, grid, MATVEC_BLOCK, 0, &bind)?;
        Ok(true)
    }

    /// One stride-2 3x3 conv2d.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn conv2d_s2(&self, d: &Conv2dDesc<'_>) -> CoreResult<()> {
        let g = &d.geometry;
        let total = g.batch * g.t_out * g.f_out * g.c_out;
        let per = if g.depthwise { 1 } else { g.c_in };
        limit(
            !g.depthwise || g.c_in == g.c_out,
            "conv2d.depthwise.channels",
            g.c_in,
        )?;
        need(d.out, total, "conv2d.out")?;
        need(d.input, g.batch * g.t_in * g.f_in * g.c_in, "conv2d.input")?;
        need(d.weight, g.c_out * per * 9, "conv2d.weight")?;
        if let Some(b) = d.bias {
            need(b, g.c_out, "conv2d.bias")?;
        }
        let args = Conv2dArgs {
            t_in: dim(g.t_in, "conv2d.t_in")?,
            f_in: dim(g.f_in, "conv2d.f_in")?,
            c_in: dim(g.c_in, "conv2d.c_in")?,
            t_out: dim(g.t_out, "conv2d.t_out")?,
            f_out: dim(g.f_out, "conv2d.f_out")?,
            c_out: dim(g.c_out, "conv2d.c_out")?,
            pad_t: dim(g.pad_t, "conv2d.pad_t")?,
            pad_f: dim(g.pad_f, "conv2d.pad_f")?,
            depthwise: u32::from(g.depthwise),
            has_bias: u32::from(d.bias.is_some()),
            relu: u32::from(d.relu),
            in_valid: dim(g.in_valid, "conv2d.in_valid")?,
            out_valid: dim(g.out_valid, "conv2d.out_valid")?,
            batch: dim(g.batch, "conv2d.batch")?,
        };
        self.launch(
            "kernel_audio_conv2d_s2_f32",
            linear_grid(total)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.input),
                KernelArg::Buffer(d.weight),
                KernelArg::Buffer(d.bias.unwrap_or(d.weight)),
            ],
        )
    }

    /// The patches of a stride-2 3x3 conv2d, for the conv as a GEMM.
    ///
    /// The direct conv takes each output's whole receptive field in one
    /// thread, which is fine for one or a few input channels; with hundreds
    /// it reads each weight once per output position and runs at a small
    /// fraction of the device. The patches cost one copy and let the tiled
    /// GEMM do the arithmetic.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for a depthwise geometry or a patch matrix
    /// past `u32` elements, [`CoreError::ShapeMismatch`] for a short view, or
    /// the launch's error.
    pub fn im2col_s2(&self, d: &Im2colDesc<'_>) -> CoreResult<()> {
        let g = &d.geometry;
        limit(!g.depthwise, "im2col.depthwise", 1)?;
        let total = g.batch * g.t_out * g.f_out * 9 * g.c_in;
        dim(total, "im2col.total")?;
        need(d.out, total, "im2col.out")?;
        need(d.input, g.batch * g.t_in * g.f_in * g.c_in, "im2col.input")?;
        let args = Conv2dArgs {
            t_in: dim(g.t_in, "conv2d.t_in")?,
            f_in: dim(g.f_in, "conv2d.f_in")?,
            c_in: dim(g.c_in, "conv2d.c_in")?,
            t_out: dim(g.t_out, "conv2d.t_out")?,
            f_out: dim(g.f_out, "conv2d.f_out")?,
            c_out: dim(g.c_out, "conv2d.c_out")?,
            pad_t: dim(g.pad_t, "conv2d.pad_t")?,
            pad_f: dim(g.pad_f, "conv2d.pad_f")?,
            depthwise: 0,
            has_bias: 0,
            relu: 0,
            in_valid: dim(g.in_valid, "conv2d.in_valid")?,
            out_valid: dim(g.out_valid, "conv2d.out_valid")?,
            batch: dim(g.batch, "conv2d.batch")?,
        };
        self.launch(
            "kernel_audio_im2col_s2_f32",
            linear_grid(total)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.input),
            ],
        )
    }

    /// A depthwise conv1d over pre-padded rows.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] past 16 taps,
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn dwconv1d(&self, d: &DwConv1dDesc<'_>) -> CoreResult<()> {
        limit((1..=MAX_TAPS).contains(&d.taps), "dwconv1d.taps", d.taps)?;
        need(d.out, d.rows * d.channels, "dwconv1d.out")?;
        need(
            d.input,
            (d.rows + d.taps - 1) * d.channels,
            "dwconv1d.input",
        )?;
        need(d.weight, d.channels * d.taps, "dwconv1d.weight")?;
        let args = DwConv1dArgs {
            rows: dim(d.rows, "dwconv1d.rows")?,
            channels: dim(d.channels, "dwconv1d.channels")?,
            taps: dim(d.taps, "dwconv1d.taps")?,
        };
        self.launch(
            "kernel_audio_dwconv1d_f32",
            linear_grid(d.rows * d.channels)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.input),
                KernelArg::Buffer(d.weight),
            ],
        )
    }

    /// Stacks each row with its neighbours.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn unfold(&self, d: &UnfoldDesc<'_>) -> CoreResult<()> {
        let total = d.rows * d.taps * d.channels;
        need(d.out, total, "unfold.out")?;
        need(d.input, d.rows * d.channels, "unfold.input")?;
        let args = UnfoldArgs {
            rows: dim(d.rows, "unfold.rows")?,
            channels: dim(d.channels, "unfold.channels")?,
            taps: dim(d.taps, "unfold.taps")?,
            pad: dim(d.pad, "unfold.pad")?,
        };
        self.launch(
            "kernel_audio_unfold_f32",
            linear_grid(total)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.input),
            ],
        )
    }

    /// Swaps the inner two axes of each plane.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn transpose(&self, d: &TransposeDesc<'_>) -> CoreResult<()> {
        let total = d.batch * d.rows * d.cols;
        need(d.out, total, "transpose.out")?;
        need(d.input, total, "transpose.input")?;
        let args = TransposeArgs {
            batch: dim(d.batch, "transpose.batch")?,
            rows: dim(d.rows, "transpose.rows")?,
            cols: dim(d.cols, "transpose.cols")?,
        };
        self.launch(
            "kernel_audio_transpose_f32",
            linear_grid(total)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.input),
            ],
        )
    }

    /// Windowed attention.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for a head wider than 128 or a
    /// zero chunk, [`CoreError::ShapeMismatch`] for a short view, or the
    /// launch's error.
    pub fn window_attention(&self, d: &WindowAttnDesc<'_>) -> CoreResult<()> {
        limit(
            (1..=MAX_HEAD_DIM).contains(&d.head_dim),
            "window_attention.head_dim",
            d.head_dim,
        )?;
        limit(d.window.chunk > 0, "window_attention.chunk", d.window.chunk)?;
        let width = d.heads * d.head_dim;
        need(d.out, d.n_q * width, "window_attention.out")?;
        need(d.q, d.n_q * width, "window_attention.q")?;
        need(d.k, d.n_k * width, "window_attention.k")?;
        need(d.v, d.n_k * width, "window_attention.v")?;
        if let Some((u, v)) = d.bias_uv {
            need(u, width, "window_attention.bias_u")?;
            need(v, width, "window_attention.bias_v")?;
        }
        if let Some(s) = d.q_dim_scale {
            need(s, d.head_dim, "window_attention.q_dim_scale")?;
        }
        if let Some(r) = d.rel {
            need(r.table, r.rows * width, "window_attention.rel")?;
        }
        let args = WindowAttnArgs {
            n_q: dim(d.n_q, "window_attention.n_q")?,
            n_k: dim(d.n_k, "window_attention.n_k")?,
            heads: dim(d.heads, "window_attention.heads")?,
            head_dim: dim(d.head_dim, "window_attention.head_dim")?,
            q_abs0: dim(d.window.q_abs0, "window_attention.q_abs0")?,
            k_abs0: dim(d.window.k_abs0, "window_attention.k_abs0")?,
            k_end: dim(d.window.k_end, "window_attention.k_end")?,
            chunk: dim(d.window.chunk, "window_attention.chunk")?,
            left_chunks: dim(d.window.left_chunks, "window_attention.left_chunks")?,
            d_min: d.rel.map_or(0, |r| r.d_min),
            n_d: dim(d.rel.map_or(0, |r| r.rows), "window_attention.rel.rows")?,
            scale: d.scale,
            ac_scale: d.ac_scale,
            softcap: d.softcap.unwrap_or(0.0),
            has_bias_uv: u32::from(d.bias_uv.is_some()),
            has_q_dim_scale: u32::from(d.q_dim_scale.is_some()),
        };
        let (bu, bv) = d.bias_uv.unwrap_or((d.q, d.q));
        let qds = d.q_dim_scale.unwrap_or(d.q);
        // The relative term depends on the query and the distance only, so it
        // is scored once per (query, head, distance) up front instead of once
        // per (query, key) inside the attention.
        let scores = match d.rel {
            Some(r) => {
                let scores = shiso_core::alloc(
                    &self.backend,
                    u64::try_from(d.n_q.max(1) * d.heads * r.rows * 4).unwrap_or(u64::MAX),
                )?;
                self.launch(
                    "kernel_audio_rel_scores_f32",
                    Extent3::new(
                        dim(r.rows.div_ceil(REL_ROWS).max(1), "rel_scores.grid")?,
                        dim(d.heads.max(1), "rel_scores.grid")?,
                        dim(d.n_q.div_ceil(ATTN_QUERY_BLOCK).max(1), "rel_scores.grid")?,
                    ),
                    32,
                    0,
                    &[
                        KernelArg::Bytes(bytemuck::bytes_of(&args)),
                        KernelArg::Buffer(scores.view()),
                        KernelArg::Buffer(d.q),
                        KernelArg::Buffer(bv),
                        KernelArg::Buffer(qds),
                        KernelArg::Buffer(r.table),
                    ],
                )?;
                Some(scores)
            }
            None => None,
        };
        self.launch(
            "kernel_audio_window_attn_f32",
            Extent3::new(
                dim(d.heads.max(1), "window_attention.heads")?,
                dim(
                    d.n_q.div_ceil(ATTN_QUERY_BLOCK).max(1),
                    "window_attention.n_q",
                )?,
                1,
            ),
            32,
            dim(attention_smem(d.head_dim), "window_attention.smem")?,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.q),
                KernelArg::Buffer(d.k),
                KernelArg::Buffer(d.v),
                KernelArg::Buffer(bu),
                KernelArg::Buffer(qds),
                KernelArg::Buffer(scores.as_ref().map_or(d.q, DeviceBuffer::view)),
            ],
        )
    }

    /// Half-split rotary embedding.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for an odd head,
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn rope_half(&self, d: &RopeDesc<'_>) -> CoreResult<()> {
        limit(
            d.head_dim.is_multiple_of(2),
            "rope_half.head_dim",
            d.head_dim,
        )?;
        let total = d.rows * d.heads * d.head_dim;
        need(d.x, total, "rope_half.x")?;
        let args = RopeArgs {
            rows: dim(d.rows, "rope_half.rows")?,
            heads: dim(d.heads, "rope_half.heads")?,
            head_dim: dim(d.head_dim, "rope_half.head_dim")?,
            pos0: dim(d.pos0, "rope_half.pos0")?,
            theta: d.theta,
        };
        self.launch(
            "kernel_audio_rope_half_f32",
            linear_grid(total / 2)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.x),
            ],
        )
    }

    /// [`Ops::rope_half`] over the first `rot_dims` of each head only: pairs
    /// `(m, m + rot_dims/2)` turn at `theta^(-2m/rot_dims)`.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for an odd or over-wide `rot_dims`,
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn rope_partial(&self, d: &RopePartialDesc<'_>) -> CoreResult<()> {
        limit(
            d.rot_dims.is_multiple_of(2) && d.rot_dims > 0 && d.rot_dims <= d.head_dim,
            "rope_partial.rot_dims",
            d.rot_dims,
        )?;
        let total = d.rows * d.heads * d.head_dim;
        need(d.x, total, "rope_partial.x")?;
        let args = RopePartialArgs {
            rows: dim(d.rows, "rope_partial.rows")?,
            heads: dim(d.heads, "rope_partial.heads")?,
            head_dim: dim(d.head_dim, "rope_partial.head_dim")?,
            pos0: dim(d.pos0, "rope_partial.pos0")?,
            theta: d.theta,
            rot: dim(d.rot_dims, "rope_partial.rot_dims")?,
        };
        self.launch(
            "kernel_audio_rope_partial_f32",
            linear_grid(d.rows * d.heads * d.rot_dims / 2)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.x),
            ],
        )
    }

    /// Half-split rotary embedding with each row's position read from
    /// `positions`: [`Ops::rope_half`]'s rotation, row for row, when
    /// `positions[r] == pos0 + r`.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for an odd head,
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn rope_half_rows(&self, d: &RopePosDesc<'_>) -> CoreResult<()> {
        limit(
            d.head_dim.is_multiple_of(2),
            "rope_half_rows.head_dim",
            d.head_dim,
        )?;
        let total = d.rows * d.heads * d.head_dim;
        need(d.x, total, "rope_half_rows.x")?;
        need(d.positions, d.rows, "rope_half_rows.positions")?;
        let args = RopeArgs {
            rows: dim(d.rows, "rope_half_rows.rows")?,
            heads: dim(d.heads, "rope_half_rows.heads")?,
            head_dim: dim(d.head_dim, "rope_half_rows.head_dim")?,
            pos0: 0,
            theta: d.theta,
        };
        self.launch(
            "kernel_audio_rope_half_pos_f32",
            linear_grid(total / 2)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.x),
                KernelArg::Buffer(d.positions),
            ],
        )
    }

    /// One LSTM cell step under greedy search's emit flag: the new hidden
    /// state always goes to `x_out`, and replaces `h` and `c` only when the
    /// search state's last step emitted, so a step that found only blanks
    /// leaves the prediction network as it was.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn lstm_cell(&self, d: &LstmDesc<'_>) -> CoreResult<()> {
        need(d.x_out, d.hidden, "lstm.x_out")?;
        need(d.h, d.hidden, "lstm.h")?;
        need(d.c, d.hidden, "lstm.c")?;
        need(d.gates, 4 * d.hidden, "lstm.gates")?;
        need(d.state, GREEDY_STATE_WORDS, "lstm.state")?;
        let args = LstmArgs {
            hidden: dim(d.hidden, "lstm.hidden")?,
        };
        self.launch(
            "kernel_audio_lstm_cell_gated_f32",
            linear_grid(d.hidden)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.x_out),
                KernelArg::Buffer(d.h),
                KernelArg::Buffer(d.c),
                KernelArg::Buffer(d.gates),
                KernelArg::Buffer(d.state),
            ],
        )
    }

    /// The joint's input rows for one greedy step: the pending frames from
    /// the search state's frame on, plus the prediction, rectified; zero past
    /// the last pending row.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn greedy_window(&self, d: &GreedyWindowDesc<'_>) -> CoreResult<()> {
        need(d.out, d.rows * d.cols, "greedy_window.out")?;
        need(d.pending, d.avail * d.cols, "greedy_window.pending")?;
        need(d.dec, d.cols, "greedy_window.dec")?;
        need(d.state, GREEDY_STATE_WORDS, "greedy_window.state")?;
        let args = GreedyWindowArgs {
            rows: dim(d.rows, "greedy_window.rows")?,
            cols: dim(d.cols, "greedy_window.cols")?,
            avail: dim(d.avail, "greedy_window.avail")?,
        };
        self.launch(
            "kernel_audio_greedy_window_f32",
            linear_grid(d.rows * d.cols)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.pending),
                KernelArg::Buffer(d.dec),
                KernelArg::Buffer(d.state),
            ],
        )
    }

    /// One decision of greedy search over the window's argmax, on the
    /// device, so a run of steps needs no round trip between them.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn greedy_step(&self, d: &GreedyStepDesc<'_>) -> CoreResult<()> {
        need(d.state, GREEDY_STATE_WORDS, "greedy_step.state")?;
        need(d.out, 2 * (d.slot + 1), "greedy_step.out")?;
        need(d.ids, d.rows, "greedy_step.ids")?;
        let args = GreedyStepArgs {
            rows: dim(d.rows, "greedy_step.rows")?,
            avail: dim(d.avail, "greedy_step.avail")?,
            blank: d.blank,
            max_symbols: dim(d.max_symbols, "greedy_step.max_symbols")?,
            slot: dim(d.slot, "greedy_step.slot")?,
        };
        self.launch(
            "kernel_audio_greedy_step_u32",
            Extent3::new(1, 1, 1),
            1,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.state),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.ids),
            ],
        )
    }

    /// The embedding row of the search state's last token.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn embed_row(&self, d: &EmbedRowDesc<'_>) -> CoreResult<()> {
        need(d.out, d.width, "embed_row.out")?;
        need(d.table, d.vocab * d.width, "embed_row.table")?;
        need(d.state, GREEDY_STATE_WORDS, "embed_row.state")?;
        let args = EmbedRowArgs {
            width: dim(d.width, "embed_row.width")?,
            vocab: dim(d.vocab, "embed_row.vocab")?,
        };
        self.launch(
            "kernel_audio_embed_row_f32",
            linear_grid(d.width)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.table),
                KernelArg::Buffer(d.state),
            ],
        )
    }

    /// `dst = src` when the search state's last step emitted.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn keep(&self, d: &KeepDesc<'_>) -> CoreResult<()> {
        need(d.dst, d.count, "keep.dst")?;
        need(d.src, d.count, "keep.src")?;
        need(d.state, GREEDY_STATE_WORDS, "keep.state")?;
        let args = KeepArgs {
            count: dim(d.count, "keep.count")?,
        };
        self.launch(
            "kernel_audio_keep_f32",
            linear_grid(d.count)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.dst),
                KernelArg::Buffer(d.src),
                KernelArg::Buffer(d.state),
            ],
        )
    }

    /// Causal grouped-query attention of `n_q` queries over a layer's cache,
    /// which must already hold every row they see.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for a head wider than 128, query heads
    /// that are not a multiple of the kv heads, or queries past the cache;
    /// [`CoreError::ShapeMismatch`] for a short view; or the launch's error.
    pub fn causal_attention(&self, d: &CausalAttnDesc<'_>) -> CoreResult<()> {
        need_cache(&d.cache, d.heads, "causal_attention.heads")?;
        limit(
            d.pos0 + d.n_q <= d.cache.cap,
            "causal_attention.pos0",
            d.pos0 + d.n_q,
        )?;
        let hd = d.cache.head_dim;
        let width = d.heads * hd;
        need(d.out, d.n_q * width, "causal_attention.out")?;
        need(d.q, d.n_q * width, "causal_attention.q")?;
        let args = CacheAttnArgs {
            n_q: dim(d.n_q, "causal_attention.n_q")?,
            heads: dim(d.heads, "causal_attention.heads")?,
            kv_heads: dim(d.cache.kv_heads, "causal_attention.kv_heads")?,
            head_dim: dim(hd, "causal_attention.head_dim")?,
            pos0: dim(d.pos0, "causal_attention.pos0")?,
            cap: dim(d.cache.cap, "causal_attention.cap")?,
            scale: d.scale,
            splits: 0,
        };
        self.launch(
            "kernel_audio_causal_attn_f32",
            Extent3::new(
                dim(d.heads.max(1), "causal_attention.heads")?,
                dim(
                    d.n_q.div_ceil(ATTN_QUERY_BLOCK).max(1),
                    "causal_attention.n_q",
                )?,
                1,
            ),
            32,
            dim(attention_smem(hd), "causal_attention.smem")?,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.q),
                KernelArg::Buffer(d.cache.k),
                KernelArg::Buffer(d.cache.v),
            ],
        )
    }

    /// Grouped-query attention of `n` rows over themselves, each query
    /// seeing the keys of its own segment within `left` before and `right`
    /// after it, on the matrix units: a threadgroup per (kv head, up to 32
    /// query items of its group), a simdgroup per (8 items, 64 head
    /// columns).
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for a head width that is not a multiple of
    /// 32 up to 512, query heads that are not a multiple of the kv heads, or
    /// a view not 16-byte aligned; [`CoreError::ShapeMismatch`] for a short
    /// view; or the launch's error.
    pub fn span_attention(&self, d: &SpanAttnDesc<'_>) -> CoreResult<()> {
        let hd = d.head_dim;
        limit(
            hd > 0 && hd.is_multiple_of(32) && hd <= SPAN_MAX_HEAD_DIM,
            "span_attention.head_dim",
            hd,
        )?;
        limit(
            d.kv_heads > 0 && d.heads.is_multiple_of(d.kv_heads),
            "span_attention.heads",
            d.heads,
        )?;
        need(d.out, d.n * d.heads * hd, "span_attention.out")?;
        need(d.q, d.n * d.heads * hd, "span_attention.q")?;
        need(d.k, d.n * d.kv_heads * hd, "span_attention.k")?;
        need(d.v, d.n * d.kv_heads * hd, "span_attention.v")?;
        // Rows are read two floats at a time.
        for (view, field) in [
            (d.out, "span_attention.out"),
            (d.q, "span_attention.q"),
            (d.k, "span_attention.k"),
            (d.v, "span_attention.v"),
        ] {
            aligned16(view, field)?;
        }
        if let Some(seg) = d.segments {
            need(seg, 2 * d.n, "span_attention.segments")?;
        }
        let slices = hd.div_ceil(SPAN_SLICE);
        let row_groups = (SPAN_SIMDGROUPS / slices).clamp(1, SPAN_MAX_ROW_GROUPS);
        let items = d.n * (d.heads / d.kv_heads);
        // Clamped to `n` (still unbounded there), so the kernel's window
        // bounds `i + right + 1` cannot wrap a u32; nor can the item count.
        let args = SpanAttnArgs {
            n: dim(d.n, "span_attention.n")?,
            heads: dim(d.heads, "span_attention.heads")?,
            kv_heads: dim(d.kv_heads, "span_attention.kv_heads")?,
            head_dim: dim(hd, "span_attention.head_dim")?,
            left: dim(d.left.min(d.n), "span_attention.left")?,
            right: dim(d.right.min(d.n), "span_attention.right")?,
            scale: d.scale,
            row_groups: dim(row_groups, "span_attention.row_groups")?,
            segmented: u32::from(d.segments.is_some()),
        };
        dim(2 * d.n + 1, "span_attention.n")?;
        dim(items, "span_attention.n")?;
        let tile = 8 * row_groups;
        self.launch(
            "kernel_audio_span_attn_f32",
            Extent3::new(
                dim(d.kv_heads, "span_attention.kv_heads")?,
                dim(items.div_ceil(tile).max(1), "span_attention.n")?,
                1,
            ),
            dim(row_groups * slices * 32, "span_attention.threads")?,
            dim(
                row_groups * slices * 8 * SPAN_KEYS * 4,
                "span_attention.smem",
            )?,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.q),
                KernelArg::Buffer(d.k),
                KernelArg::Buffer(d.v),
                // Unread without segments; any live view binds the slot.
                KernelArg::Buffer(d.segments.unwrap_or(d.q)),
            ],
        )
    }

    /// The one query of a decode step over cache rows
    /// `0..=state[DECODE_POS]`, the length read on the device so a batch of
    /// steps needs no host value: the rows are cut into `splits` runs scored
    /// side by side, then merged in split order.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for a head that is not a multiple of four
    /// up to 128, query heads that are not a multiple of the kv heads, splits
    /// outside `1..=256`, or a view not 16-byte aligned;
    /// [`CoreError::ShapeMismatch`] for a short view; or the launch's error.
    pub fn decode_attention(&self, d: &DecodeAttnDesc<'_>) -> CoreResult<()> {
        need_cache(&d.cache, d.heads, "decode_attention.heads")?;
        let hd = d.cache.head_dim;
        limit(hd.is_multiple_of(4), "decode_attention.head_dim", hd)?;
        limit(
            (1..=DECODE_MAX_SPLITS).contains(&d.splits),
            "decode_attention.splits",
            d.splits,
        )?;
        let width = d.heads * hd;
        need(d.out, width, "decode_attention.out")?;
        need(d.q, width, "decode_attention.q")?;
        need(
            d.partials,
            decode_attention_partials(d.heads, hd, d.splits),
            "decode_attention.partials",
        )?;
        need(d.state, DECODE_STATE_WORDS, "decode_attention.state")?;
        for (view, field) in [
            (d.out, "decode_attention.out"),
            (d.partials, "decode_attention.partials"),
            (d.q, "decode_attention.q"),
            (d.cache.k, "decode_attention.cache.k"),
            (d.cache.v, "decode_attention.cache.v"),
        ] {
            aligned16(view, field)?;
        }
        let args = CacheAttnArgs {
            n_q: 1,
            heads: dim(d.heads, "decode_attention.heads")?,
            kv_heads: dim(d.cache.kv_heads, "decode_attention.kv_heads")?,
            head_dim: dim(hd, "decode_attention.head_dim")?,
            pos0: 0,
            cap: dim(d.cache.cap, "decode_attention.cap")?,
            scale: d.scale,
            splits: dim(d.splits, "decode_attention.splits")?,
        };
        self.launch(
            "kernel_audio_decode_attn_f32",
            Extent3::new(
                args.kv_heads,
                dim(
                    d.splits.div_ceil(DECODE_SPLITS_PER_TG),
                    "decode_attention.splits",
                )?,
                dim(
                    (d.heads / d.cache.kv_heads).div_ceil(DECODE_GROUP_TILE),
                    "decode_attention.group",
                )?,
            ),
            dim(32 * DECODE_SPLITS_PER_TG, "decode_attention.block")?,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.partials),
                KernelArg::Buffer(d.q),
                KernelArg::Buffer(d.cache.k),
                KernelArg::Buffer(d.cache.v),
                KernelArg::Buffer(d.state),
            ],
        )?;
        self.launch(
            "kernel_audio_decode_attn_merge_f32",
            Extent3::linear(args.heads),
            DECODE_MERGE_THREADS,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.partials),
            ],
        )
    }

    /// `rows` new key and value rows into a layer's cache from the decoder
    /// state's position on. The position is not moved: every layer of a pass
    /// appends at the same one, and the step kernel moves it after. A row
    /// that would land at or past the capacity is dropped on the device, so
    /// the caller grows the cache before it queues rows that reach it.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn kv_append(&self, d: &KvAppendDesc<'_>) -> CoreResult<()> {
        need_cache(&d.cache, d.cache.kv_heads, "kv_append.kv_heads")?;
        let count = d.rows * d.cache.kv_heads * d.cache.head_dim;
        need(d.k, count, "kv_append.k")?;
        need(d.v, count, "kv_append.v")?;
        need(d.state, DECODE_STATE_WORDS, "kv_append.state")?;
        let args = KvAppendArgs {
            rows: dim(d.rows, "kv_append.rows")?,
            kv_heads: dim(d.cache.kv_heads, "kv_append.kv_heads")?,
            head_dim: dim(d.cache.head_dim, "kv_append.head_dim")?,
            cap: dim(d.cache.cap, "kv_append.cap")?,
        };
        self.launch(
            "kernel_audio_kv_append_f32",
            linear_grid(count)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.cache.k),
                KernelArg::Buffer(d.cache.v),
                KernelArg::Buffer(d.k),
                KernelArg::Buffer(d.v),
                KernelArg::Buffer(d.state),
            ],
        )
    }

    /// Half-split rotary embedding at the decoder state's position plus
    /// `offset`, read on the device.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for an odd head,
    /// [`CoreError::ShapeMismatch`] for a short view, or the launch's error.
    pub fn rope_half_at_state(&self, d: &RopeStateDesc<'_>) -> CoreResult<()> {
        limit(
            d.head_dim.is_multiple_of(2),
            "rope_half_at_state.head_dim",
            d.head_dim,
        )?;
        let total = d.rows * d.heads * d.head_dim;
        need(d.x, total, "rope_half_at_state.x")?;
        need(d.state, DECODE_STATE_WORDS, "rope_half_at_state.state")?;
        let args = RopeArgs {
            rows: dim(d.rows, "rope_half_at_state.rows")?,
            heads: dim(d.heads, "rope_half_at_state.heads")?,
            head_dim: dim(d.head_dim, "rope_half_at_state.head_dim")?,
            pos0: dim(d.offset, "rope_half_at_state.offset")?,
            theta: d.theta,
        };
        self.launch(
            "kernel_audio_rope_half_state_f32",
            linear_grid(total / 2)?,
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.x),
                KernelArg::Buffer(d.state),
            ],
        )
    }

    /// One decision of a causal decoder over the head's argmax, on the
    /// device: the position moves by `advance`, an end id stops the decoder
    /// unemitted, any other id is emitted to `out[slot]` and becomes the last
    /// token, and the `max_tokens`-th stops it. A step on a stopped decoder
    /// writes [`DECODE_NO_TOKEN`], lowers the emit flag and changes nothing
    /// else, so a batch can run past the stop.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for a zero token cap or an advance past
    /// one row, [`CoreError::ShapeMismatch`] for a short view, or the
    /// launch's error.
    pub fn decode_step(&self, d: &DecodeStepDesc<'_>) -> CoreResult<()> {
        limit(d.max_tokens > 0, "decode_step.max_tokens", d.max_tokens)?;
        limit(d.advance <= 1, "decode_step.advance", d.advance)?;
        need(d.state, DECODE_STATE_WORDS, "decode_step.state")?;
        need(d.out, d.slot + 1, "decode_step.out")?;
        need(d.ids, 1, "decode_step.ids")?;
        let args = DecodeStepArgs {
            eos0: d.eos[0],
            eos1: d.eos[1],
            max_tokens: dim(d.max_tokens, "decode_step.max_tokens")?,
            advance: dim(d.advance, "decode_step.advance")?,
            slot: dim(d.slot, "decode_step.slot")?,
        };
        self.launch(
            "kernel_audio_decode_step_u32",
            Extent3::new(1, 1, 1),
            1,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.state),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.ids),
            ],
        )
    }

    /// The f32 embedding row of the decoder state's last token, dequantised
    /// from the table's own storage; zeros for a token past the table.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for a width that is not whole blocks or
    /// groups of the table's storage, [`CoreError::ShapeMismatch`] for a
    /// short view, or the launch's error.
    pub fn embed_token(&self, d: &EmbedTokenDesc<'_>) -> CoreResult<()> {
        need(d.out, d.width, "embed_token.out")?;
        need(d.state, DECODE_STATE_WORDS, "embed_token.state")?;
        self.embed_rows(d.out, d.table, d.state, (d.width, d.vocab, 1), false)
    }
}
