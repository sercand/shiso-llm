//! Weights stored in a rotated basis, and the rotation their inputs pass
//! through.
//!
//! A checkpoint can fold an orthogonal transform `R` into a matrix's input axis
//! offline - store `W R^T` rather than `W` - because a rotated basis spreads an
//! outlier across a block and makes the matrix far easier to quantize. The
//! product is unchanged only if the activation is rotated the same way first:
//! `(W R^T)(R x) = W x`. This module is that second half. `R` here is the
//! signed blockwise Walsh-Hadamard transform `x -> H(P(x) * signs)`, applied
//! to each consecutive `block` inputs, with `H` normalized so it is its own
//! inverse and `P` an optional permutation of value heads.
//!
//! An embedding table can be stored rotated too, and then each gathered row
//! has to be taken back: `y = signs * H(row)`. That is
//! [`RotationDirection::Inverse`], and it runs in place on the gathered rows.
//! The same table read as a tied output head is `E R^T`, and its logits are
//! right only over a rotated input, `(E R^T)(R h) = E h`: so an inverse
//! rotation that carries a scratch also runs the forward transform, and the
//! projection runs it for every rotated weight whatever its direction.
//!
//! A forward rotation writes into a [`RotationScratch`] one model shares
//! between all its rotated weights. That is sound because projections are
//! launched by one thread and run in queue order: by the time the next
//! rotation overwrites the scratch, the matmul that read it has been
//! enqueued ahead of it. Nothing outside [`crate::projection::projection`]
//! reads the scratch.

use std::sync::Arc;

use shiso_core::backend::{Backend, KernelArg, KernelRef};
use shiso_core::{CoreError, CoreResult, DeviceBuffer, DeviceView, Extent3, alloc};

use crate::launch::{F32_BYTES, TERNARY, run};

/// Threads one rotation threadgroup runs.
const ROTATE_THREADS: u32 = 256;

/// The rotation's kernel.
const ROTATE_ENTRY: &str = "kernel_rotate_wht_f32";

/// Which way a rotation runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationDirection {
    /// `H(P(x) * signs)`: a projection's input, before a weight the rotation
    /// was folded into.
    Forward,
    /// `signs * H(x)`: a row gathered from a rotated embedding table, back to
    /// the residual stream's basis.
    Inverse,
}

/// Value heads that arrive in one order and were rotated in another.
///
/// The activation holds `key_heads * group` heads of `head_dim` in tiled order
/// (head `m * key_heads + k`) and the weight's rotation was folded in over
/// grouped order (head `k * group + m`). A folded matrix's columns cannot be
/// permuted after the fact, because the rotation mixes a whole block of them,
/// so the permutation is applied to the activation as the rotation reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadOrder {
    /// Elements in one head.
    pub head_dim: u32,
    /// Distinct key heads; the tiled order's inner stride.
    pub key_heads: u32,
    /// Value heads per key head.
    pub group: u32,
}

/// Mirrors `ternary_args_rotate`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RotateArgs {
    n_tok: u32,
    width: u32,
    block: u32,
    inverse: u32,
    nk: u32,
    rep: u32,
    hd: u32,
    pad: u32,
}

/// The device buffer every forward rotation of one model writes into.
#[derive(Debug)]
pub struct RotationScratch {
    buffer: DeviceBuffer,
    floats: u64,
}

impl RotationScratch {
    /// Room for `max_tokens` rows of the widest rotated input.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for a zero dimension, and whatever the
    /// allocation reports.
    pub fn new(
        backend: &Arc<dyn Backend>,
        max_in_dim: u32,
        max_tokens: u32,
    ) -> CoreResult<Arc<Self>> {
        if max_in_dim == 0 || max_tokens == 0 {
            return Err(CoreError::InvalidValue {
                key: "rotation scratch",
                value: format!("{max_in_dim} wide x {max_tokens} rows"),
            });
        }
        let floats = u64::from(max_in_dim) * u64::from(max_tokens);
        let buffer = alloc(backend, floats * F32_BYTES)?;
        Ok(Arc::new(Self { buffer, floats }))
    }
}

/// What [`InputRotation::new`] is built from.
#[derive(Debug, Clone)]
pub struct RotationDesc<'a> {
    /// Inputs one transform covers; a power of two from 512 to 4096.
    pub block: u32,
    /// Which way it runs.
    pub direction: RotationDirection,
    /// One `+1` or `-1` per input; its length is the rotated width.
    pub signs: &'a [f32],
    /// A head permutation to read the input through (forward only).
    pub heads: Option<HeadOrder>,
    /// Where the forward transform stages its output. A forward rotation
    /// needs one; an inverse one runs its own direction in place and needs one
    /// only when its table is also read as a projection (a tied output head).
    pub scratch: Option<Arc<RotationScratch>>,
}

/// The rotation one weight expects its input (or, inverse, its gathered rows)
/// to pass through.
#[derive(Debug)]
pub struct InputRotation {
    block: u32,
    width: u32,
    direction: RotationDirection,
    heads: Option<HeadOrder>,
    signs: DeviceBuffer,
    /// The signs as bits, one per input (set for `-1`), kept on the host so
    /// two rotations can be told equal without reading the device.
    sign_bits: Box<[u64]>,
    scratch: Option<Arc<RotationScratch>>,
}

impl InputRotation {
    /// Checks `desc` and uploads its signs.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for a block that is not a power of two in
    /// 512..=4096 or does not divide the width, a sign that is not exactly
    /// `+1` or `-1`, a forward rotation with no scratch or an inverse one with
    /// a head permutation, or a permutation whose heads do not tile the width;
    /// and whatever the upload reports.
    pub fn new(backend: &Arc<dyn Backend>, desc: RotationDesc<'_>) -> CoreResult<Self> {
        let invalid = |value: String| CoreError::InvalidValue {
            key: "input rotation",
            value,
        };
        let width = u32::try_from(desc.signs.len()).map_err(|_| invalid("width".into()))?;
        if !desc.block.is_power_of_two() || !(512..=4096).contains(&desc.block) {
            return Err(invalid(format!(
                "block {} is not a power of two in 512..=4096",
                desc.block
            )));
        }
        if width == 0 || !width.is_multiple_of(desc.block) {
            return Err(invalid(format!(
                "width {width} is not whole blocks of {}",
                desc.block
            )));
        }
        if let Some(bad) = desc
            .signs
            .iter()
            .find(|s| s.abs().to_bits() != 1.0_f32.to_bits())
        {
            return Err(invalid(format!("sign {bad} is neither +1 nor -1")));
        }
        match (desc.direction, &desc.scratch, desc.heads) {
            (RotationDirection::Forward, None, _) => {
                return Err(invalid("a forward rotation needs a scratch".into()));
            }
            (RotationDirection::Inverse, _, Some(_)) => {
                return Err(invalid(
                    "an in-place inverse rotation cannot permute heads".into(),
                ));
            }
            _ => {}
        }
        if let Some(h) = desc.heads {
            let heads = u64::from(h.head_dim) * u64::from(h.key_heads) * u64::from(h.group);
            if h.head_dim == 0 || h.key_heads == 0 || h.group == 0 || heads != u64::from(width) {
                return Err(invalid(format!("heads {h:?} do not tile width {width}")));
            }
        }
        let mut sign_bits = vec![0_u64; desc.signs.len().div_ceil(64)].into_boxed_slice();
        for (at, sign) in desc.signs.iter().enumerate() {
            if sign.is_sign_negative()
                && let Some(word) = sign_bits.get_mut(at / 64)
            {
                *word |= 1 << (at % 64);
            }
        }
        let bytes: Vec<u8> = desc.signs.iter().flat_map(|s| s.to_le_bytes()).collect();
        let signs = alloc(backend, u64::from(width) * F32_BYTES)?;
        backend.copy_h2d(signs.view(), &bytes)?;
        Ok(Self {
            block: desc.block,
            width,
            direction: desc.direction,
            heads: desc.heads,
            signs,
            sign_bits,
            scratch: desc.scratch,
        })
    }

    /// Which way this rotation runs.
    #[must_use]
    pub const fn direction(&self) -> RotationDirection {
        self.direction
    }

    /// The width it rotates.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Whether `other` runs the same forward transform, so that an input
    /// rotated for one is the input the other reads.
    #[must_use]
    pub fn same_forward(&self, other: &Self) -> bool {
        self.block == other.block
            && self.width == other.width
            && self.heads == other.heads
            && self.sign_bits == other.sign_bits
    }

    /// Launches the transform of `n_tok` rows from `src` into `dst`, the
    /// inverse direction when `inverse`.
    fn launch(
        &self,
        backend: &dyn Backend,
        n_tok: u32,
        inverse: bool,
        src: DeviceView<'_>,
        dst: DeviceView<'_>,
    ) -> CoreResult<()> {
        let heads = self.heads.unwrap_or(HeadOrder {
            head_dim: 1,
            key_heads: 1,
            group: 1,
        });
        let args = RotateArgs {
            n_tok,
            width: self.width,
            block: self.block,
            inverse: u32::from(inverse),
            nk: heads.key_heads,
            rep: heads.group,
            hd: heads.head_dim,
            pad: 0,
        };
        run(
            backend,
            KernelRef::new(TERNARY, ROTATE_ENTRY),
            Extent3::new(self.width / self.block, n_tok, 1),
            Extent3::linear(ROTATE_THREADS),
            self.block * 4,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(src),
                KernelArg::Buffer(self.signs.view()),
                KernelArg::Buffer(dst),
            ],
        )
    }

    /// Rotates `n_tok` rows of `src` into the scratch and returns the view a
    /// projection then reads.
    ///
    /// This is the forward transform whatever the direction: an inverse
    /// rotation's table read as a projection is a tied output head, and that
    /// reads a forward-rotated input (see the module docs).
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for a rotation built without a scratch,
    /// [`CoreError::OutOfBounds`] when `src` is short or the rows do not fit
    /// the scratch, and whatever the launch reports.
    pub(crate) fn forward<'s>(
        &'s self,
        backend: &dyn Backend,
        n_tok: u32,
        src: DeviceView<'_>,
    ) -> CoreResult<DeviceView<'s>> {
        let Some(scratch) = &self.scratch else {
            return Err(CoreError::InvalidValue {
                key: "input rotation",
                value: "a rotated table read as a projection needs a scratch to stage \
                        its rotated input in, and this one was built without"
                    .into(),
            });
        };
        let floats = u64::from(self.width) * u64::from(n_tok);
        if floats > scratch.floats {
            return Err(CoreError::OutOfBounds {
                offset: 0,
                len: floats * F32_BYTES,
                buffer_bytes: scratch.floats * F32_BYTES,
            });
        }
        if src.len() < floats * F32_BYTES {
            return Err(CoreError::OutOfBounds {
                offset: 0,
                len: floats * F32_BYTES,
                buffer_bytes: src.len(),
            });
        }
        let dst = scratch.buffer.view();
        self.launch(backend, n_tok, false, src, dst)?;
        Ok(scratch.buffer.view())
    }

    /// Takes `n_tok` gathered rows back to the residual basis, in place.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for a forward rotation,
    /// [`CoreError::OutOfBounds`] when `rows` is short, and whatever the
    /// launch reports.
    pub fn inverse_in_place(
        &self,
        backend: &dyn Backend,
        n_tok: u32,
        rows: DeviceView<'_>,
    ) -> CoreResult<()> {
        if self.direction != RotationDirection::Inverse {
            return Err(CoreError::InvalidValue {
                key: "input rotation",
                value: "a forward rotation does not run in place".into(),
            });
        }
        let bytes = u64::from(self.width) * u64::from(n_tok) * F32_BYTES;
        if rows.len() < bytes {
            return Err(CoreError::OutOfBounds {
                offset: 0,
                len: bytes,
                buffer_bytes: rows.len(),
            });
        }
        self.launch(backend, n_tok, true, rows, rows)
    }
}
