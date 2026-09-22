//! The block launches this family owns.
//!
//! The engine holds the weights, derives the geometry, allocates the scratch and
//! runs the residual stream; what it does not hold is which kernels a block
//! dispatches over all of that. Those are a fact about the checkpoint family, so
//! they live here and reach the forward pass through
//! [`shiso_engine::blocks::BlockKernels`].
//!
//! [`Qwen35Kernels`] is the whole of the public surface. Behind it are the two
//! block files - [`attn`] and [`gdn`], the launches exactly as the engine used
//! to issue them - and [`gdn_tune`], this family's table of compiled recurrent
//! shapes and the measurement that picks one for the device in front of it.

mod attn;
mod gdn;
mod gdn_tune;

use shiso_core::backend::Backend;
use shiso_core::error::CoreResult;

use shiso_engine::attn::{Attn, AttnCache, AttnGeometry, AttnScratch};
use shiso_engine::blocks::{AttnIo, BlockKernels, GdnIo};
use shiso_engine::gdn::{Gdn, GdnGeometry, GdnScratch};

/// The qwen35 family's two block launchers, with the device fact they carry.
///
/// The pair of recurrent kernel shapes is measured once, when the stack is
/// built, and read at every recurrent launch. It sits on this value rather than
/// in the engine's geometry because it is a fact about the machine and not about
/// the architecture: two processes on two boxes run the same widths through
/// different entry points and, bit for bit, the same answer comes out.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35Kernels {
    /// The measured shapes of the two recurrent entry points.
    gdn_shape: gdn_tune::GdnShape,
}

impl Qwen35Kernels {
    /// Measures the device and fixes the recurrent kernel shapes.
    ///
    /// Called once per stack, before it is announced resident: a device that
    /// will run no recurrent shape at all cannot run this model and should say
    /// so at load rather than at the first prompt.
    ///
    /// # Errors
    ///
    /// [`shiso_core::CoreError::Unsupported`] when the device accepts no shape
    /// of either recurrent kernel.
    pub fn choose(backend: &dyn Backend, geometry: &GdnGeometry) -> CoreResult<Self> {
        let gdn_shape = gdn_tune::GdnShape::choose(backend, geometry)?;
        tracing::info!(
            recurrent = gdn_shape.recurrent.entry(),
            recurrent_lanes = gdn_shape.recurrent.lanes(),
            chunk = gdn_shape.chunk.entry(),
            chunk_lanes = gdn_shape.chunk.lanes(),
            // Not "the fastest shapes": a pinned run reaches this line too,
            // and the line a reader greps for should say what is dispatched
            // rather than how it was arrived at. The tuner logs that itself.
            "GDN shapes this stack dispatches",
        );
        Ok(Self { gdn_shape })
    }
}

impl BlockKernels for Qwen35Kernels {
    fn attention(
        &self,
        backend: &dyn Backend,
        attn: &Attn,
        geometry: &AttnGeometry,
        scratch: &AttnScratch,
        io: AttnIo<'_>,
        cache: &AttnCache<'_>,
        n_tok: u32,
    ) -> CoreResult<()> {
        // `io.bound` is not read: what a qwen35 query may see is its own
        // position and everything before it, which these kernels derive from
        // the linear position they already take.
        attn::run(
            backend,
            attn,
            geometry,
            scratch,
            io.input,
            io.pos,
            cache,
            io.block_out,
            n_tok,
        )
    }

    fn recurrent(
        &self,
        backend: &dyn Backend,
        gdn: &Gdn,
        geometry: &GdnGeometry,
        scratch: &GdnScratch,
        io: GdnIo<'_>,
        n_tok: u32,
    ) -> CoreResult<()> {
        gdn::run(backend, gdn, geometry, scratch, self.gdn_shape, io, n_tok)
    }
}
