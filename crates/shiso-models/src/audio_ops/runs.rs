//! A weight as the audio set's own products read it: a dense or `Q8_0`
//! matrix of the audio GEMM's types, or an MLX 8-bit affine one, which the
//! engine's projection would otherwise take.
//!
//! The affine weight's bytes are four to a little-endian `u32` along each
//! row, and every 64 of a row have a bf16 scale and bias; the kernels form
//! each value as the fused `q * scale + bias` the embedding-row fetch forms,
//! in f32. The engine's affine projection stages a prefill's rows at some
//! 0.55 TFLOPS on M1 Max, a fifth of what the tiled GEMM reaches, and its
//! one-row product streams at under 200 GB/s.

use shiso_core::{DeviceView, KernelArg};

use super::GemmType;

/// The elements of one affine scale-and-bias group.
pub(crate) const AFFINE_GROUP: usize = 64;

/// An MLX 8-bit affine weight, groups of [`AFFINE_GROUP`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct Affine8<'a> {
    /// The bytes, `[out][in / 4]` little-endian `u32`.
    pub(crate) q: DeviceView<'a>,
    /// bf16, `[out][in / 64]`.
    pub(crate) scales: DeviceView<'a>,
    /// bf16, as the scales.
    pub(crate) biases: DeviceView<'a>,
}

/// A weight the audio set's own products read.
#[derive(Debug, Clone, Copy)]
pub(crate) enum WeightView<'a> {
    /// Row-major `[out][in]` in the audio GEMM's storage.
    Dense(GemmType, DeviceView<'a>),
    /// 8-bit affine.
    Affine8(Affine8<'a>),
}

impl<'a> WeightView<'a> {
    /// Whether two weights take the same kernel.
    pub(crate) fn same_kind(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Dense(a, _), Self::Dense(b, _)) => a == b,
            (Self::Affine8(_), Self::Affine8(_)) => true,
            _ => false,
        }
    }

    /// Whether the matrix-vector-by-runs kernels take a product of depth `k`
    /// over this weight: whole runs (and whole groups), reads on their
    /// natural alignment.
    pub(crate) fn runs_fit(&self, k: usize) -> bool {
        match self {
            Self::Dense(_, w) => k.is_multiple_of(8) && w.offset().is_multiple_of(16),
            Self::Affine8(a) => k.is_multiple_of(AFFINE_GROUP) && a.q.offset().is_multiple_of(8),
        }
    }

    /// The bytes, then (affine only) the scales and biases, in the order a
    /// kernel lists them after its bias.
    pub(crate) fn main(&self) -> DeviceView<'a> {
        match self {
            Self::Dense(_, w) => *w,
            Self::Affine8(a) => a.q,
        }
    }

    /// The affine scales and biases, none for a dense weight.
    pub(crate) fn aux(&self) -> Vec<KernelArg<'a>> {
        match self {
            Self::Dense(..) => Vec::new(),
            Self::Affine8(a) => vec![KernelArg::Buffer(a.scales), KernelArg::Buffer(a.biases)],
        }
    }

    /// The one-product matrix-vector-by-runs kernel.
    pub(crate) fn matvec_entry(&self) -> &'static str {
        match self {
            Self::Dense(t, _) => t.matvec_entry(),
            Self::Affine8(_) => "kernel_audio_matvec_affine8w",
        }
    }

    /// The three-product kernel.
    pub(crate) fn matvec3_entry(&self) -> &'static str {
        match self {
            Self::Dense(t, _) => t.matvec3_entry(),
            Self::Affine8(_) => "kernel_audio_matvec3_affine8w",
        }
    }

    /// The gated two-product kernel.
    pub(crate) fn matvec_silu_entry(&self) -> &'static str {
        match self {
            Self::Dense(t, _) => t.matvec_silu_entry(),
            Self::Affine8(_) => "kernel_audio_matvec_silu_affine8w",
        }
    }

    /// The 32-row and 64-row tiled GEMMs.
    pub(crate) fn tiled_entries(&self) -> (&'static str, &'static str) {
        match self {
            Self::Dense(t, _) => (t.entry(), t.entry64()),
            Self::Affine8(_) => ("kernel_audio_gemm_affine8w", "kernel_audio_gemm64_affine8w"),
        }
    }
}
