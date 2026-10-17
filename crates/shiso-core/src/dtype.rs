//! Tensor element formats, keyed by their on-disk id.
//!
//! The discriminants are the ids stored in tensor-info records, so the reader
//! converts an integer to a `DType` once and never touches the raw value again.
//! The last four entries are not upstream GGUF types: they are the ids this
//! project's checkpoint converter writes for the trellis and n-gram codecs,
//! deliberately numbered above every GGUF id so a future upstream type cannot
//! collide with them.

use std::num::NonZeroU32;

/// Tensor element format. Discriminants are the on-disk type ids.
#[expect(non_camel_case_types, reason = "variants are GGUF on-disk format ids")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[non_exhaustive]
pub enum DType {
    /// IEEE 754 binary32.
    F32 = 0,
    /// IEEE 754 binary16.
    F16 = 1,
    /// 4-bit block quant, one shared scale per 32 elements.
    Q4_0 = 2,
    /// 5-bit block quant, 32 elements per block.
    Q5_0 = 6,
    /// 5-bit block quant with min, 32 elements per block.
    Q5_1 = 7,
    /// 8-bit block quant, 32 elements per block.
    Q8_0 = 8,
    /// K-quant 2-bit, 256 elements per block.
    Q2_K = 10,
    /// K-quant 3-bit, 256 elements per block.
    Q3_K = 11,
    /// K-quant 4-bit, 256 elements per block.
    Q4_K = 12,
    /// K-quant 5-bit, 256 elements per block.
    Q5_K = 13,
    /// K-quant 6-bit, 256 elements per block.
    Q6_K = 14,
    /// K-quant 8-bit, 256 elements per block.
    Q8_K = 15,
    /// 2-bit i-quant, extra-extra-small, 256 elements per block.
    IQ2_XXS = 16,
    /// 2-bit i-quant, extra-small, 256 elements per block: the wider sibling
    /// of [`DType::IQ2_XXS`].
    IQ2_XS = 17,
    /// 3-bit i-quant, extra-extra-small, 256 elements per block.
    IQ3_XXS = 18,
    /// 1-bit i-quant, small, 256 elements per block.
    IQ1_S = 19,
    /// 4-bit non-linear quant, 32 elements per block.
    IQ4_NL = 20,
    /// 3-bit i-quant, small, 256 elements per block.
    IQ3_S = 21,
    /// 2-bit i-quant, small, 256 elements per block.
    IQ2_S = 22,
    /// 4-bit i-quant, extra-small, 256 elements per block.
    IQ4_XS = 23,
    /// 32-bit signed integer tensor (vocabulary and index maps).
    I32 = 26,
    /// 1-bit i-quant, medium, 256 elements per block.
    IQ1_M = 29,
    /// Brain floating point, binary16 width exponent.
    BF16 = 30,
    /// Microscaling 4-bit with one shared exponent per 32 elements.
    MXFP4 = 39,
    /// Trellis 4-bit codec, one 16x16 tile per block.
    Exl3K4 = 68,
    /// Trellis 5-bit codec, one 16x16 tile per block.
    Exl3K5 = 69,
    /// Trellis 6-bit codec, one 16x16 tile per block.
    Exl3K6 = 70,
    /// N-gram row codec: 160 weights per row with an fp16 scale word.
    Exl3Ngram6 = 72,
    /// MLX 4-bit affine block quant, group size 64: 32 bytes weights + 2 bytes scale + 2 bytes bias.
    MlxAffine4Gs64 = 200,
    /// MLX 8-bit affine block quant, group size 64: 64 bytes weights + 2 bytes scale + 2 bytes bias.
    MlxAffine8Gs64 = 201,
    /// MLX 4-bit affine block quant, group size 32: 16 bytes weights + 2 bytes
    /// scale + 2 bytes bias.
    ///
    /// The row format of a streamed n-gram table and nothing else: no
    /// projection tier reads it, which is why [`DType::is_affine`] - the
    /// question the projection ladder asks - answers no for it.
    MlxAffine4Gs32 = 210,
    /// FP8 E4M3 weights with one brain-float scale per 128 x 128 block of the
    /// matrix, in an array beside the codes (a checkpoint's
    /// `weight_scale_inv`).
    ///
    /// One byte a weight, like [`DType::Fp8E4M3`]; what differs is where the
    /// scale lives, which is why this is a type of its own: a kernel that took
    /// one for the other would scale every block by the first block's scale.
    Fp8E4M3Block128 = 211,
    /// FP8 E4M3 weights under one `f32` per tensor.
    ///
    /// The scale is NOT in the block - there is no block, one byte is one
    /// weight - so it travels beside the buffer rather than inside it. That is
    /// what [`block_bytes`] answering 1 means here: a row of this format is
    /// exactly `in_dim` bytes and nothing else, and every scale the decode
    /// needs is somewhere a caller has to have carried.
    ///
    /// [`block_bytes`]: DType::block_bytes
    Fp8E4M3 = 202,
    /// NVFP4: four-bit codes at group size 16, under an `e4m3` byte per group
    /// and one `f32` per tensor.
    ///
    /// [`block_bytes`] is **8**, which is the packed weights of a group and
    /// not the group's whole storage: the `e4m3` scale array and the per-tensor
    /// `f32` are separate arrays in the file and separate buffers on the
    /// device. The number is chosen so that `row_bytes` answers the length of
    /// the buffer it is asked about - the MLX affine types chose the other
    /// convention, fold their scales into `block_bytes`, and consequently
    /// cannot use `row_bytes` for an import at all.
    ///
    /// [`block_bytes`]: DType::block_bytes
    Nvfp4Gs16 = 203,
    /// Half precision whose values are brain-float values, repacked by the
    /// engine at load into twelve bits a weight: a byte of sign and the seven
    /// mantissa bits a brain float has, and a four-bit exponent code against a
    /// per-row base, with the few values that fit neither - the subnormals and
    /// those far below their row's largest - kept exactly in a per-row list
    /// beside the planes. Lossless: every weight decodes to its stored half.
    ///
    /// Never read from a file. [`block_bytes`] is **3** over two elements -
    /// the two planes - and, as for [`DType::Nvfp4Gs16`], the rest of the
    /// payload is not in the block.
    ///
    /// [`block_bytes`]: DType::block_bytes
    F16Pack12 = 204,
    /// Ternary weights in 2-bit slots, 128 elements per block: an `f16` scale
    /// first, then 32 bytes of codes, four to a byte from the low bits up.
    ///
    /// A code `c` decodes to `scale * (c - 1)`. The format has room for
    /// `c = 3` (twice the scale) and a reader must honour it even though a
    /// ternary export never writes one.
    Pq2_0 = 142,
    /// Ternary weights packed five trits to a byte, 128 elements per block: 24
    /// bytes of five-trit codes, 2 bytes of four-trit codes, then an `f16`
    /// scale.
    ///
    /// A trit `t` decodes to `scale * (t - 1)`. The element order inside a
    /// block is not the byte order - see `shiso_modelfile::ternary` for the
    /// three stages the bytes are read in.
    Ptq1_0 = 143,
    /// MLX 2-bit affine block quant, group size 128: 32 bytes of words + 2
    /// bytes scale + 2 bytes bias, by the same convention as the other MLX
    /// affine types.
    ///
    /// Unlike the 4- and 8-bit types, whose scales and biases are brain
    /// floats, this one's are IEEE `f16`: the only export of it is a lossless
    /// re-encoding of a [`DType::Pq2_0`] file, whose scales are `f16`, and a
    /// brain float would round three of their mantissa bits away.
    MlxAffine2Gs128 = 212,
}

impl DType {
    /// Maps an on-disk id. Unknown ids are `None`, which the reader turns into
    /// a typed error naming the id; nothing silently defaults to `F32`.
    #[must_use]
    pub const fn from_gguf_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            15 => Some(Self::Q8_K),
            16 => Some(Self::IQ2_XXS),
            17 => Some(Self::IQ2_XS),
            18 => Some(Self::IQ3_XXS),
            19 => Some(Self::IQ1_S),
            20 => Some(Self::IQ4_NL),
            21 => Some(Self::IQ3_S),
            22 => Some(Self::IQ2_S),
            23 => Some(Self::IQ4_XS),
            26 => Some(Self::I32),
            29 => Some(Self::IQ1_M),
            30 => Some(Self::BF16),
            39 => Some(Self::MXFP4),
            68 => Some(Self::Exl3K4),
            69 => Some(Self::Exl3K5),
            70 => Some(Self::Exl3K6),
            72 => Some(Self::Exl3Ngram6),
            142 => Some(Self::Pq2_0),
            143 => Some(Self::Ptq1_0),
            _ => None,
        }
    }

    /// The on-disk id, for writing tensors back out.
    #[must_use]
    pub const fn gguf_id(self) -> u32 {
        self as u32
    }
}

impl DType {
    /// Lowercase format name, the spelling used in tensor metadata dumps and
    /// error messages.
    #[must_use]
    pub const fn type_name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q4_0 => "q4_0",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q2_K => "q2_k",
            Self::Q3_K => "q3_k",
            Self::Q4_K => "q4_k",
            Self::Q5_K => "q5_k",
            Self::Q6_K => "q6_k",
            Self::Q8_K => "q8_k",
            Self::IQ2_XXS => "iq2_xxs",
            Self::IQ2_XS => "iq2_xs",
            Self::IQ3_XXS => "iq3_xxs",
            Self::IQ1_S => "iq1_s",
            Self::IQ4_NL => "iq4_nl",
            Self::IQ3_S => "iq3_s",
            Self::IQ2_S => "iq2_s",
            Self::IQ4_XS => "iq4_xs",
            Self::I32 => "i32",
            Self::IQ1_M => "iq1_m",
            Self::BF16 => "bf16",
            Self::MXFP4 => "mxfp4",
            Self::Exl3K4 => "exl3_k4",
            Self::Exl3K5 => "exl3_k5",
            Self::Exl3K6 => "exl3_k6",
            Self::Exl3Ngram6 => "exl3_ngram6",
            Self::MlxAffine4Gs64 => "mlx_affine_4_gs64",
            Self::MlxAffine8Gs64 => "mlx_affine_8_gs64",
            Self::MlxAffine4Gs32 => "mlx_affine_4_gs32",
            Self::Fp8E4M3Block128 => "fp8_e4m3_block128",
            Self::Fp8E4M3 => "fp8_e4m3",
            Self::Nvfp4Gs16 => "nvfp4_gs16",
            Self::F16Pack12 => "f16_pack12",
            Self::Pq2_0 => "pq2_0",
            Self::Ptq1_0 => "ptq1_0",
            Self::MlxAffine2Gs128 => "mlx_affine_2_gs128",
        }
    }
}

impl DType {
    /// Elements covered by one block: one for the unquantized formats, so a
    /// caller can compute `elems * n_blocks` for any format uniformly.
    #[must_use]
    pub const fn block_elems(self) -> NonZeroU32 {
        let elems = match self {
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::I32
            | Self::Fp8E4M3
            | Self::Fp8E4M3Block128 => 1,
            Self::Nvfp4Gs16 => 16,
            Self::F16Pack12 => 2,
            Self::Q4_0
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::IQ4_NL
            | Self::MXFP4
            | Self::MlxAffine4Gs32 => 32,
            Self::MlxAffine4Gs64 | Self::MlxAffine8Gs64 => 64,
            Self::Exl3Ngram6 => 160,
            Self::Pq2_0 | Self::Ptq1_0 | Self::MlxAffine2Gs128 => 128,
            Self::Q2_K
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::Q8_K
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ3_XXS
            | Self::IQ1_S
            | Self::IQ3_S
            | Self::IQ2_S
            | Self::IQ4_XS
            | Self::IQ1_M
            | Self::Exl3K4
            | Self::Exl3K5
            | Self::Exl3K6 => 256,
        };
        expect_nonzero(elems)
    }
}

impl DType {
    /// Bytes occupied by one block. Note that the 32-element microscaling
    /// format is 17 bytes, not the 18 a same-width block quant uses: the
    /// shared exponent is one byte, not two.
    #[must_use]
    pub const fn block_bytes(self) -> NonZeroU32 {
        let bytes = match self {
            Self::Fp8E4M3 | Self::Fp8E4M3Block128 => 1,
            Self::F16 | Self::BF16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::Q4_0 | Self::IQ4_NL => 18,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 | Self::Pq2_0 => 34,
            Self::MXFP4 => 17,
            Self::IQ2_XXS => 66,
            Self::IQ2_XS => 74,
            Self::IQ3_XXS => 98,
            Self::IQ1_S => 50,
            Self::IQ3_S | Self::Q3_K => 110,
            Self::IQ2_S => 82,
            Self::IQ4_XS => 136,
            Self::IQ1_M => 56,
            Self::Q2_K => 84,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
            Self::Exl3K4 => 128,
            Self::Exl3K5 => 160,
            Self::Exl3K6 => 192,
            Self::Exl3Ngram6 => 122,
            Self::MlxAffine4Gs64 | Self::MlxAffine2Gs128 => 36,
            Self::MlxAffine8Gs64 => 68,
            Self::MlxAffine4Gs32 => 20,
            // The packed weights of one group, without the `e4m3` scale byte
            // that covers it: both live in arrays of their own. See the
            // variant's own note for why this is the useful convention.
            Self::Nvfp4Gs16 => 8,
            // The two planes of two weights; see the variant.
            Self::F16Pack12 => 3,
            Self::Ptq1_0 => 28,
        };
        expect_nonzero(bytes)
    }
}

impl DType {
    /// Whether the format packs more than one element per block.
    #[must_use]
    pub const fn is_quantized(self) -> bool {
        self.block_elems().get() > 1
    }

    /// Whether this is one of the MLX affine packed formats.
    #[must_use]
    pub const fn is_affine(self) -> bool {
        matches!(
            self,
            Self::MlxAffine4Gs64 | Self::MlxAffine8Gs64 | Self::MlxAffine2Gs128
        )
    }

    /// Whether this is one of the two ternary block formats, whose one `f16`
    /// scale per 128 weights lives inside the block and whose bias is always
    /// minus that scale.
    #[must_use]
    pub const fn is_ternary(self) -> bool {
        matches!(self, Self::Pq2_0 | Self::Ptq1_0)
    }

    /// Whether this is one of the two formats `modelopt` writes, whose scales
    /// live in arrays beside the weights rather than inside a block.
    ///
    /// Not the same question [`DType::is_affine`] asks. An affine weight's
    /// scales are also separate arrays, but the engine's affine path reads two
    /// of them and the group size is in the type; these have one array or none,
    /// plus a per-tensor `f32` that is not an array at all.
    #[must_use]
    pub const fn is_modelopt(self) -> bool {
        matches!(self, Self::Fp8E4M3 | Self::Nvfp4Gs16)
    }

    /// Number of bits per quantized element, when affine.
    #[must_use]
    pub const fn affine_bits(self) -> Option<u32> {
        match self {
            Self::MlxAffine4Gs64 => Some(4),
            Self::MlxAffine8Gs64 => Some(8),
            Self::MlxAffine2Gs128 => Some(2),
            _ => None,
        }
    }
}
/// Turns a literal block dimension into a `NonZeroU32` at compile time. Every
/// match arm above is a nonzero literal, so the panic branch documents an
/// invariant the table already holds instead of adding a runtime check.
const fn expect_nonzero(value: u32) -> NonZeroU32 {
    match NonZeroU32::new(value) {
        Some(v) => v,
        None => panic!("block dimensions must be non-zero"),
    }
}
