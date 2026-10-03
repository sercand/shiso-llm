//! Expanding a trellis weight to half precision once, outside any pass.
//!
//! For a consumer that has no trellis tier of its own - a vision tower whose
//! encoder multiplies half-precision weights - but receives its weights in
//! the trellis format. The expansion is the kernel the prompt path already
//! runs on every wide pass, run once at load; what comes back are the
//! original-basis weights, so the consumer multiplies exactly what that path
//! would have.

use std::sync::Arc;

use shiso_core::backend::{Backend, KernelArg, LaunchArgs};
use shiso_core::buffer::alloc;
use shiso_core::error::{CoreError, CoreResult};

use crate::sparse::exl3;
use crate::sparse::kernels::{self, Entry, EntryTable, MOD_EXL3, TemplateArgs};
use crate::sparse::matmul::trellis_bits;
use crate::sparse::weights::{Weight, WeightPart};

/// Expands one trellis weight to half precision on the device and returns
/// it, contraction-major: `in_dim` rows of `out_dim` halves.
///
/// `entries` is the build's captured instantiation table
/// ([`crate::sparse::capture::capture`]), taken once by the caller rather than
/// per weight.
///
/// # Errors
///
/// [`CoreError::Unsupported`] for a weight that is not a single trellis
/// matrix or a build without the expansion kernel, whatever
/// [`exl3::check_shape`] refuses, and whatever the device reports.
pub fn trellis_to_half(
    backend: &Arc<dyn Backend>,
    entries: &EntryTable,
    weight: &Weight,
) -> CoreResult<Vec<u8>> {
    let bits = trellis_bits(weight.dtype).ok_or(CoreError::Unsupported {
        what: "an expansion of a weight that is not in a trellis format",
        backend: "engine",
    })?;
    if weight.slots != 1 {
        return Err(CoreError::Unsupported {
            what: "an expansion of a stacked bank rather than one matrix",
            backend: "engine",
        });
    }
    let (k, n) = (weight.in_dim, weight.out_dim);
    exl3::check_shape(k, n, bits)?;
    if !k.is_multiple_of(exl3::RECONSTRUCT_TILE_K) {
        return Err(CoreError::ShapeMismatch {
            field: "expanded trellis contraction",
            expected: u64::from(exl3::RECONSTRUCT_TILE_K),
            got: u64::from(k),
        });
    }
    let kernel = entries.resolve(Entry::template(
        MOD_EXL3,
        kernels::EXL3_RECONSTRUCT,
        TemplateArgs::TrellisBits(bits),
    ))?;
    let bytes = u64::from(k) * u64::from(n) * 2;
    let out = alloc(backend, bytes)?;
    let launch = kernels::trellis_expand(k, n);
    // `g_unpacked, g_packed, suh, svh, packed_blocks_n, packed_n_offset`
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(weight.part(WeightPart::Tiles)?),
        KernelArg::Buffer(weight.part(WeightPart::InputScales)?),
        KernelArg::Buffer(weight.part(WeightPart::OutputScales)?),
        KernelArg::I32(i32::try_from(n / 16).unwrap_or(i32::MAX)),
        KernelArg::I32(0),
    ];
    backend.run(
        kernel,
        &LaunchArgs {
            grid: launch.grid,
            block: launch.block,
            smem_bytes: launch.smem,
            args: &args,
        },
    )?;
    backend.sync()?;
    let mut host = vec![0u8; usize::try_from(bytes).unwrap_or(0)];
    backend.copy_d2h(out.view(), &mut host)?;
    Ok(host)
}
