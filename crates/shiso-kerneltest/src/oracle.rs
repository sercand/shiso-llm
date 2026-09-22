//! Hand-written CPU oracles for the GPU tests.
//!
//! Shared by two backends since the dense cases moved here: the same oracle
//! grades the Metal kernel and its CUDA twin, which is what makes the twin a
//! twin rather than a second implementation graded against itself. Nothing here
//! names a device, and nothing here may start to.
//!
//! Every function here is a direct transcription of the arithmetic in one
//! `metal/generic/*.metal` kernel, written as a plain Rust `f32` loop with no
//! dependencies. They are deliberately naive: a kernel test passes when the
//! device agrees with the simplest possible reading of the same formula, so a
//! disagreement points at the kernel or at its launch geometry rather than at a
//! second, equally subtle implementation.
#![expect(
    clippy::cast_precision_loss,
    reason = "the oracles convert small fixture lengths and 24-bit mantissas, both of which are exact in f32"
)]
//!
//! The two `q4_K` unpackers are *not* interchangeable. `kernel_get_rows_q4_K_f32`
//! and the `mul_mv_ext` family were written against different nibble layouts
//! (`get_rows.metal` walks groups linearly, `mul_mv_ext.metal` walks the
//! 16-value chunks the matvec needs), so each has its own packer and dequantizer
//! here and each test uses the pair that matches its kernel.

/// Smallest relative tolerance the quantized paths are held to.
pub const TOL_QUANT: f32 = 1e-3;
/// Relative tolerance the `f32` paths are held to.
pub const TOL_F32: f32 = 1e-5;

/// Asserts `got` agrees with `expected` within `tol`, scaled by the magnitude of
/// the expected value so a large row is not failed by ordinary rounding.
#[macro_export]
macro_rules! assert_close {
    ($got:expr, $expected:expr, $tol:expr) => {{
        let got: f32 = $got;
        let expected: f32 = $expected;
        let bound = $tol * expected.abs().max(1.0);
        assert!(
            (got - expected).abs() <= bound,
            "got {got} differs from {expected} by {} > {bound}",
            (got - expected).abs(),
        );
    }};
    ($got:expr, $expected:expr, $tol:expr, $($arg:tt)+) => {{
        let got: f32 = $got;
        let expected: f32 = $expected;
        let bound = $tol * expected.abs().max(1.0);
        assert!(
            (got - expected).abs() <= bound,
            "got {got} differs from {expected} by {} > {bound}: {}",
            (got - expected).abs(),
            format_args!($($arg)+),
        );
    }};
}

/// Deterministic `f32` fixture data: a small LCG mapped to `[-1, 1)`.
///
/// Fixed rather than random so a failure reproduces byte for byte.
#[must_use]
pub fn signal(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 0x9E37_79B9;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
            unit * 2.0 - 1.0
        })
        .collect()
}

/// `f16` bits to `f32`, exactly as Metal's `(float)` cast of a `half` does.
///
/// Written out so the fixtures can carry `u16` scale fields without pulling in
/// an fp16 crate.
#[must_use]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1F);
    let frac = u32::from(bits & 0x3FF);
    let word = match exp {
        0 if frac == 0 => sign,
        0 => {
            // Subnormal: shift the leading 1 into the implicit slot and walk the
            // exponent field down by the same amount. 113 is the field of the
            // largest subnormal (`2^-14`), and at most ten shifts happen here, so
            // the field stays well inside `u32`.
            let mut field = 113u32;
            let mut frac = frac;
            while frac & 0x400 == 0 {
                frac <<= 1;
                field -= 1;
            }
            sign | (field << 23) | ((frac & 0x3FF) << 13)
        }
        0x1F => sign | (0xFF << 23) | (frac << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(word)
}

/// RMS norm of one row, matching `kernel_rms_norm_fuse_impl`.
///
/// `weight` is the learned gain: `None` is the `F == 1` path (no weight at all),
/// `Some` is the `F == 2` path. The kernel sums in a strided, per-thread order
/// and then reduces across simdgroups, which is why the caller compares with a
/// tolerance rather than bit equality.
#[must_use]
pub fn rms_norm(x: &[f32], weight: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let sumf: f32 = x.iter().map(|v| v * v).sum();
    let scale = 1.0 / (sumf / x.len() as f32 + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| match weight {
            None => v * scale,
            Some(w) => v * scale * w[i],
        })
        .collect()
}

/// Row softmax matching `kernel_soft_max_4` with no mask and no attention sink.
#[must_use]
pub fn softmax_row(x: &[f32], scale: f32) -> Vec<f32> {
    let max = x.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v * scale));
    let exp: Vec<f32> = x.iter().map(|v| (v * scale - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|v| v / sum).collect()
}

/// Gathers whole `f32` rows by id, matching `kernel_get_rows_f<float, float>`.
///
/// # Panics
///
/// When an id is negative, or names a row past the end of `table`. Both are
/// fixture errors rather than kernel ones, and both are worth a panic here: a
/// gather oracle that silently produced a shorter answer would fail the case it
/// is grading and say nothing about why.
#[must_use]
pub fn gather(table: &[f32], row_len: usize, ids: &[i32]) -> Vec<f32> {
    ids.iter()
        .flat_map(|&id| {
            let start = usize::try_from(id).expect("fixture id is negative") * row_len;
            table[start..start + row_len].to_vec()
        })
        .collect()
}

/// The 6-bit scale/min pair unpack shared by the gather helper in
/// `metal/generic/get_rows.metal` and the matvec helper in
/// `metal/generic/mul_mv_ext.metal`, which are identical.
#[must_use]
pub fn scale_min_k4(j: usize, k: usize, scales: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (scales[j + k] & 63, scales[j + 4 + k] & 63)
    } else {
        (
            (scales[j + 4 + k] & 0x0F) | ((scales[j - 4 + k] & 0xC0) >> 2),
            (scales[j + 4 + k] >> 4) | ((scales[j + k] & 0xC0) >> 2),
        )
    }
}

/// Element `idx` of a `q4_K` block as `kernel_get_rows_q4_K_f32` reads it: group
/// `idx / 32` selects the nibble half, and the group's scale/min apply whole.
#[must_use]
pub fn dequant_q4_k_gathered(block: &Q4K, idx: usize) -> f32 {
    let group = idx / 32;
    let lane = idx % 32;
    let (sc, smin) = scale_min_k4(group, 0, &block.scales);
    let nibble = u16::from(block.qs[(group / 2) * 32 + lane] >> ((group % 2) * 4)) & 0x0F;
    f16_to_f32(block.d) * f32::from(sc) * f32::from(nibble)
        - f16_to_f32(block.dmin) * f32::from(smin)
}

/// Element `e` of a `q4_K` block as `dequantize_dense_q4_K_t4` reads it.
///
/// The matvec walks 16-value chunks, so chunk `e / 4` picks the byte range and
/// the low two bits of that chunk index pick the nibble *and* whether the block
/// scale is used whole or divided by 16.
#[must_use]
pub fn dequant_q4_k_dense(block: &Q4K, e: usize) -> f32 {
    let chunk = e / 4;
    let il = chunk / 4;
    let i = (chunk % 4) * 4 + e % 4;
    let whole = il % 4 < 2;
    let mask: u16 = if whole { 0x0F } else { 0xF0 };
    let nibble = u16::from(block.qs[(il / 4) * 32 + 16 * (il % 2) + i]) & mask;
    let (sc, smin) = scale_min_k4((il / 4) * 2, (il % 4) / 2, &block.scales);
    let d = f16_to_f32(block.d);
    let dl = if whole { d } else { d * (1.0 / 16.0) } * f32::from(sc);
    let ml = f16_to_f32(block.dmin) * f32::from(smin);
    dl * f32::from(nibble) - ml
}

/// The 144-byte `q4_K` block, as a value the packers can fill.
///
/// `d` and `dmin` stay as `f16` bit patterns because that is how the loader's
/// `BlockQ4K` stores them and how the kernel casts them.
#[derive(Debug, Clone, Copy)]
pub struct Q4K {
    /// The block scale, as the `f16` bit pattern the loader stores.
    pub d: u16,
    /// The block minimum, as the `f16` bit pattern the loader stores.
    pub dmin: u16,
    /// The twelve packed 6-bit scale/min pairs, read through
    /// [`scale_min_k4`].
    pub scales: [u8; 12],
    /// The 256 four-bit weights. Which nibble holds which element depends on
    /// the reader, which is why there are two packers above.
    pub qs: [u8; 128],
}

/// Packs 256 nibble values into the `get_rows` layout (inverse of
/// [`dequant_q4_k_gathered`]).
#[must_use]
pub fn pack_q4_k_gathered(values: &[u8; 256]) -> [u8; 128] {
    let mut qs = [0u8; 128];
    for (idx, &v) in values.iter().enumerate() {
        let group = idx / 32;
        let lane = idx % 32;
        let slot = &mut qs[(group / 2) * 32 + lane];
        if group % 2 == 0 {
            *slot |= v & 0x0F;
        } else {
            *slot |= (v & 0x0F) << 4;
        }
    }
    qs
}

/// Packs 256 nibble values into the `mul_mv_ext` layout (inverse of
/// [`dequant_q4_k_dense`]).
#[must_use]
pub fn pack_q4_k_dense(values: &[u8; 256]) -> [u8; 128] {
    let mut qs = [0u8; 128];
    for (e, &v) in values.iter().enumerate() {
        let chunk = e / 4;
        let lane = e % 4;
        let il = chunk / 4;
        let i = (chunk % 4) * 4 + lane;
        let slot = &mut qs[(il / 4) * 32 + 16 * (il % 2) + i];
        if il % 4 < 2 {
            *slot |= v & 0x0F;
        } else {
            *slot |= (v & 0x0F) << 4;
        }
    }
    qs
}

/// Plain dot product, the oracle for the matvec once the weights are dequantized.
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
