//! Runtime MSL compilation.
//!
//! The kernel library is compiled at startup rather than shipped as a prebuilt
//! archive, so the sources in `metal/generic/` are the only kernel artifact and
//! a stale binary can never be what actually ran.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLCompileOptions, MTLDevice, MTLLibrary};
use parking_lot::Mutex;
use shiso_core::{CoreError, CoreResult};

use crate::device::MetalError;

/// Directory that overrides the embedded kernel sources, one file per entry of
/// [`SOURCES`]. Diagnostic only: a normal run never reads a kernel from disk.
pub(crate) const SOURCE_DIR_ENV: &str = "SHISO_METAL_SOURCE_DIR";

/// The kernel library, in compile order.
///
/// Order matters only for `#define` visibility: `00_common.metal` is the prelude
/// every later file assumes. The sources are embedded so a launch never depends
/// on the working directory, which is what makes the gate reproducible.
pub(crate) const SOURCES: &[(&str, &str)] = &[
    (
        "00_common.metal",
        include_str!("../metal/generic/00_common.metal"),
    ),
    // The i-quant codebook tables, before every file that reads them:
    // `get_rows.metal` decodes IQ1_M rows through `IQ1S_GRID` and
    // `mul_mv_ext.metal` decodes all nine formats through the grids, and the
    // library is one translation unit, so the declarations have to precede
    // both.
    (
        "iq_tables.metal",
        include_str!("../metal/generic/iq_tables.metal"),
    ),
    ("norm.metal", include_str!("../metal/generic/norm.metal")),
    (
        "softmax.metal",
        include_str!("../metal/generic/softmax.metal"),
    ),
    ("unary.metal", include_str!("../metal/generic/unary.metal")),
    ("bin.metal", include_str!("../metal/generic/bin.metal")),
    ("glu.metal", include_str!("../metal/generic/glu.metal")),
    (
        "concat.metal",
        include_str!("../metal/generic/concat.metal"),
    ),
    ("cpy.metal", include_str!("../metal/generic/cpy.metal")),
    (
        "get_rows.metal",
        include_str!("../metal/generic/get_rows.metal"),
    ),
    (
        "set_rows.metal",
        include_str!("../metal/generic/set_rows.metal"),
    ),
    (
        "sum_rows.metal",
        include_str!("../metal/generic/sum_rows.metal"),
    ),
    (
        "repeat.metal",
        include_str!("../metal/generic/repeat.metal"),
    ),
    (
        "argsort.metal",
        include_str!("../metal/generic/argsort.metal"),
    ),
    (
        "dsv4_rope.metal",
        include_str!("../metal/generic/dsv4_rope.metal"),
    ),
    (
        "mul_mv_q8_0.metal",
        include_str!("../metal/generic/mul_mv_q8_0.metal"),
    ),
    (
        "mul_mv_ext.metal",
        include_str!("../metal/generic/mul_mv_ext.metal"),
    ),
    (
        "mul_mv_dense.metal",
        include_str!("../metal/generic/mul_mv_dense.metal"),
    ),
    // --- qwen35 kernel set (T5); appended, nothing above this line changed ---
    ("q35_common.h", include_str!("../metal/qwen35/q35_common.h")),
    (
        "qwen35/gdn_conv.metal",
        include_str!("../metal/qwen35/gdn_conv.metal"),
    ),
    (
        "qwen35/gdn_recurrent.metal",
        include_str!("../metal/qwen35/gdn_recurrent.metal"),
    ),
    (
        "qwen35/gdn_recurrent_hybrid.metal",
        include_str!("../metal/qwen35/gdn_recurrent_hybrid.metal"),
    ),
    (
        "qwen35/gdn_chunk.metal",
        include_str!("../metal/qwen35/gdn_chunk.metal"),
    ),
    // After gdn_chunk.metal, not before: the hybrid is one more instantiation
    // of that file's template, so the template has to have been parsed.
    (
        "qwen35/gdn_chunk_hybrid.metal",
        include_str!("../metal/qwen35/gdn_chunk_hybrid.metal"),
    ),
    (
        "rmsnorm_rope.metal",
        include_str!("../metal/qwen35/rmsnorm_rope.metal"),
    ),
    (
        "gated_attn.metal",
        include_str!("../metal/qwen35/gated_attn.metal"),
    ),
    (
        "gated_attn_prefill.metal",
        include_str!("../metal/qwen35/gated_attn_prefill.metal"),
    ),
    (
        "gemv_ext_q5k_q6k.metal",
        include_str!("../metal/qwen35/gemv_ext_q5k_q6k.metal"),
    ),
    (
        "gemm_q4k.metal",
        include_str!("../metal/qwen35/gemm_q4k.metal"),
    ),
    // --- vision tower kernel set (T6c); appended, nothing above this line
    // changed. The header comes first for the same reason the qwen35 one does:
    // the files after it use its argument blocks and its reductions.
    ("vit_common.h", include_str!("../metal/vision/vit_common.h")),
    (
        "vision/vit_gemm.metal",
        include_str!("../metal/vision/vit_gemm.metal"),
    ),
    (
        "vision/vit_norm_gelu.metal",
        include_str!("../metal/vision/vit_norm_gelu.metal"),
    ),
    (
        "vision/vit_attn.metal",
        include_str!("../metal/vision/vit_attn.metal"),
    ),
    (
        "vision/vit_merge.metal",
        include_str!("../metal/vision/vit_merge.metal"),
    ),
    (
        "vision/vit_splice.metal",
        include_str!("../metal/vision/vit_splice.metal"),
    ),
    // --- dense kernel set (Gemma 4 phase 1); appended, nothing above this line
    // changed. The header comes first for the same reason the qwen35 and vision
    // ones do: the files after it use its argument blocks, its activation and
    // its reductions.
    (
        "dense_common.h",
        include_str!("../metal/dense/dense_common.h"),
    ),
    (
        "dense/dense_misc.metal",
        include_str!("../metal/dense/dense_misc.metal"),
    ),
    (
        "dense/dense_norm_rope.metal",
        include_str!("../metal/dense/dense_norm_rope.metal"),
    ),
    (
        "dense/dense_attn.metal",
        include_str!("../metal/dense/dense_attn.metal"),
    ),
    // The per-layer embedding pair reuses `dequantize_dense_q6_K` and the block
    // struct it reads out of `qwen35/gemv_ext_q5k_q6k.metal`, which is why it
    // is appended after that file rather than declaring either again.
    (
        "dense/dense_ple.metal",
        include_str!("../metal/dense/dense_ple.metal"),
    ),
    // The Gemma 4 vision tower's dense kernels: the attention-plane norm and
    // two-axis rotation, the pooler and the clamp. Appended after the rest of
    // the dense set because they read `dense_common.h`'s argument blocks and
    // reuse its reduction helpers and its neox rotation.
    (
        "dense/dense_vision.metal",
        include_str!("../metal/dense/dense_vision.metal"),
    ),
    // The MLX packed affine matrix-vector (qmv) and matrix-matrix (qmm) kernels.
    (
        "dense/dense_affine.metal",
        include_str!("../metal/dense/dense_affine.metal"),
    ),
    (
        "dense/dense_affine_rows.metal",
        include_str!("../metal/dense/dense_affine_rows.metal"),
    ),
    // The row-exact affine matmul: one arithmetic for every row count, so a
    // verify window's rows equal the serial steps they check.
    (
        "dense/dense_exact_affine.metal",
        include_str!("../metal/dense/dense_exact_affine.metal"),
    ),
    // The block drafter's two kernels of its own: the strided row copy that
    // gathers the target's tapped hidden states, and the grouped dynamic
    // causal convolution.
    (
        "dense/dense_block_draft.metal",
        include_str!("../metal/dense/dense_block_draft.metal"),
    ),
    // The state-space block (convolution, scan, gated norm) and the routed
    // feed-forward over row-major affine stacks with a biased sigmoid router.
    (
        "dense/dense_ssm.metal",
        include_str!("../metal/dense/dense_ssm.metal"),
    ),
    (
        "dense/dense_moe.metal",
        include_str!("../metal/dense/dense_moe.metal"),
    ),
    // The gated short convolution block's one sequential kernel.
    (
        "dense/dense_short_conv.metal",
        include_str!("../metal/dense/dense_short_conv.metal"),
    ),
    // --- qwen35moe kernel set; appended, nothing above this line changed. The
    // header comes first, for the same reason the qwen35, vision and dense
    // headers do: the files after it use its argument block and its affine
    // tile. It is also where the MetalPerformancePrimitives tensor headers
    // enter the translation unit.
    (
        "q35moe_common.h",
        include_str!("../metal/qwen35moe/q35moe_common.h"),
    ),
    (
        "qwen35moe/moe_experts.metal",
        include_str!("../metal/qwen35moe/moe_experts.metal"),
    ),
    (
        "qwen35moe/moe_route.metal",
        include_str!("../metal/qwen35moe/moe_route.metal"),
    ),
    (
        "qwen35moe/moe_misc.metal",
        include_str!("../metal/qwen35moe/moe_misc.metal"),
    ),
    // --- qwen4exp (q4e) kernel set; appended, nothing above this line
    // changed. The header comes first for the same reason every other set's
    // does. The router, the sort, the activation, the combine and the shared
    // add are already above, in the qwen35moe set, under the entry names the
    // sparse stack launches.
    ("q4e_common.h", include_str!("../metal/q4e/q4e_common.h")),
    ("q4e/hc.metal", include_str!("../metal/q4e/hc.metal")),
    ("q4e/gdn.metal", include_str!("../metal/q4e/gdn.metal")),
    ("q4e/ple.metal", include_str!("../metal/q4e/ple.metal")),
    (
        "q4e/matvec.metal",
        include_str!("../metal/q4e/matvec.metal"),
    ),
    ("q4e/qsa.metal", include_str!("../metal/q4e/qsa.metal")),
    (
        "q4e/qsa_sparse.metal",
        include_str!("../metal/q4e/qsa_sparse.metal"),
    ),
    ("q4e/idx.metal", include_str!("../metal/q4e/idx.metal")),
    ("q4e/misc.metal", include_str!("../metal/q4e/misc.metal")),
    ("q4e/embed.metal", include_str!("../metal/q4e/embed.metal")),
    (
        "q4e/affine.metal",
        include_str!("../metal/q4e/affine.metal"),
    ),
    ("q4e/fp8.metal", include_str!("../metal/q4e/fp8.metal")),
    // This family's own fused decode row of a recurrent layer; after
    // `gdn.metal`, whose helpers and constants it uses.
    ("q4e/exact.metal", include_str!("../metal/q4e/exact.metal")),
    // --- audio encoder kernel set; appended, nothing above this line changed.
    (
        "audio_common.h",
        include_str!("../metal/audio/audio_common.h"),
    ),
    (
        "audio/audio_elementwise.metal",
        include_str!("../metal/audio/audio_elementwise.metal"),
    ),
    (
        "audio/audio_norm.metal",
        include_str!("../metal/audio/audio_norm.metal"),
    ),
    (
        "audio/audio_gemm.metal",
        include_str!("../metal/audio/audio_gemm.metal"),
    ),
    (
        "audio/audio_conv.metal",
        include_str!("../metal/audio/audio_conv.metal"),
    ),
    (
        "audio/audio_attn.metal",
        include_str!("../metal/audio/audio_attn.metal"),
    ),
    (
        "audio/audio_lstm.metal",
        include_str!("../metal/audio/audio_lstm.metal"),
    ),
    // --- logits stages shared by every stack; appended, nothing above this
    // line changed.
    (
        "sample/suppress.metal",
        include_str!("../metal/sample/suppress.metal"),
    ),
    // --- diffusion DiT kernel set; appended, nothing above this line changed.
    // The header comes first for the same reason every other set's does: the
    // file after it uses its argument blocks, its decoders and its reduction.
    (
        "diffusion_common.h",
        include_str!("../metal/diffusion/diffusion_common.h"),
    ),
    (
        "diffusion.metal",
        include_str!("../metal/diffusion/diffusion.metal"),
    ),
    // --- ternary kernel set (rotated ternary checkpoints); appended, nothing
    // above this line changed. The header first, as for every other set.
    (
        "ternary_common.h",
        include_str!("../metal/ternary/ternary_common.h"),
    ),
    (
        "ternary/rotate.metal",
        include_str!("../metal/ternary/rotate.metal"),
    ),
    (
        "ternary/ternary_mv.metal",
        include_str!("../metal/ternary/ternary_mv.metal"),
    ),
    (
        "ternary/ternary_mm.metal",
        include_str!("../metal/ternary/ternary_mm.metal"),
    ),
];

/// Concatenates every source into the one translation unit the device compiles.
///
/// Each file is preceded by an `// appended <name>` marker so a compiler
/// diagnostic's line number maps back to the file it came from.
///
/// # Errors
///
/// [`CoreError::Io`] when `SHISO_METAL_SOURCE_DIR` is set and one of the files
/// it names cannot be read.
pub(crate) fn full_source() -> CoreResult<String> {
    let override_dir = std::env::var_os(SOURCE_DIR_ENV);
    let mut source = String::from("\n");
    for (name, embedded) in SOURCES {
        let body = match &override_dir {
            Some(dir) => {
                let path = std::path::Path::new(dir).join(name);
                std::fs::read_to_string(&path).map_err(|err| {
                    tracing::error!(
                        path = %path.display(),
                        error = %err,
                        "Metal source is unreadable",
                    );
                    CoreError::Io(err)
                })?
            }
            None => (*embedded).to_string(),
        };
        source.push_str("// appended ");
        source.push_str(name);
        source.push('\n');
        source.push_str(&body);
        source.push('\n');
    }
    Ok(source)
}

/// Compiles [`full_source`] into a device library.
///
/// # Errors
///
/// [`CoreError::Backend`] carrying the compiler's own diagnostic as the error
/// source, so `{:#}` reaches it. The full diagnostic is also logged, because a
/// truncated `CoreError` display is useless for fixing a kernel.
fn compile_source(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
) -> CoreResult<Retained<ProtocolObject<dyn MTLLibrary>>> {
    // Defaults, and the q35 sources depend on that: `MTLCompileOptions` leaves
    // math mode Fast, and the qwen35 kernels are written against it.
    //
    // Four reduction helpers in `q35_common.h` and the body of
    // `gdn_recurrent.metal` opt OUT with `#pragma METAL fp math_mode(safe)`,
    // because the GDN recurrent kernel's lane-count variants are only
    // bit-identical if the optimizer may not reassociate their additions.
    // MSL has no push/pop for that pragma, so each safe region ends by
    // spelling `math_mode(fast)` -- it restores THIS DEFAULT, not whatever
    // was in force before.
    //
    // The consequence: setting a non-default math mode here is not enough and
    // is not safe on its own. `full_source` concatenates the files in a fixed
    // order (see `SOURCES`), so those `fast` lines would silently force fast
    // math back on for every source after them -- gdn_chunk, rmsnorm_rope,
    // the gated_attn pair, the gemv/gemm set and all of vision/. Change the
    // pragmas and this call together, or not at all.
    let options = MTLCompileOptions::new();
    let compiled =
        device.newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options));
    match compiled {
        Ok(library) => {
            tracing::debug!(
                bytes = source.len(),
                files = SOURCES.len(),
                "compiled Metal kernel library",
            );
            Ok(library)
        }
        Err(err) => {
            let diagnostic = err.localizedDescription().to_string();
            tracing::error!(
                bytes = source.len(),
                diagnostic = %diagnostic,
                "Metal kernel library failed to compile",
            );
            Err(CoreError::Backend {
                backend: "metal",
                op: "newLibraryWithSource",
                source: Box::new(MetalError::from(diagnostic)),
            })
        }
    }
}

/// One compiled library, with the device and the exact source it came from.
struct CachedLibrary {
    device_id: u64,
    source: String,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

/// Libraries this process has compiled. Compiling the whole source takes about a
/// minute on an M1 Max, and a test binary that opens many backends paid it once
/// per backend, so a second open of the same source on the same device reuses
/// the first. The lock is held across the compile on purpose: callers racing for
/// the same library wait for it instead of each compiling their own.
static COMPILED: Mutex<Vec<CachedLibrary>> = Mutex::new(Vec::new());

/// The kernel library for `device`, compiled once per process per distinct
/// source (`SHISO_METAL_SOURCE_DIR` can change the source between calls).
///
/// # Errors
///
/// As [`full_source`] and the compiler; a failure is not cached.
pub(crate) fn compile_library(
    device: &ProtocolObject<dyn MTLDevice>,
) -> CoreResult<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let source = full_source()?;
    let device_id = device.registryID();
    let mut compiled = COMPILED.lock();
    if let Some(hit) = compiled
        .iter()
        .find(|c| c.device_id == device_id && c.source == source)
    {
        return Ok(hit.library.clone());
    }
    let library = compile_source(device, &source)?;
    compiled.push(CachedLibrary {
        device_id,
        source,
        library: library.clone(),
    });
    Ok(library)
}
