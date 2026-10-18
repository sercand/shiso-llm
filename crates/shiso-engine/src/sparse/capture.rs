//! Reading a build's own symbol table back into the table a template dispatch
//! resolves through.
//!
//! Most kernels in this set carry C linkage and are named by a constant. Ten are
//! C++ templates, and the names their compiler mangles from the argument list
//! exist in exactly one place: the module that compiler produced. This module is
//! the one that reads them.
//!
//! # Why this decodes rather than encodes
//!
//! The obvious shape - write the mangling down and look the name up - is the one
//! that cannot be made safe. A mangled name that is subtly wrong fails at lookup,
//! which is fine; a mangled name that is subtly wrong *and happens to name a
//! different instantiation of the same template* does not fail at all. It
//! launches, and the answer is wrong in a way nothing reports.
//!
//! So nothing here spells a mangled name. Every symbol the module exposes is
//! taken apart into the template it instantiates and the arguments it was
//! instantiated with, and that pair is compared with the pair the caller asked
//! for. A decoder that misreads a symbol produces a pair that matches nothing,
//! which is a typed refusal naming the missing instantiation.
//!
//! Two symbols that decode to the same pair are dropped rather than picked
//! between, for the same reason: the caller's key does not distinguish them, so
//! either answer would be a guess.
//!
//! # What it understands
//!
//! The subset of the Itanium C++ ABI these entry points use, and no more:
//! `_Z`, a length-prefixed name, and a template argument list of integer
//! literals, boolean literals and types. Everything after the argument list is
//! the function's own signature and is not read - two instantiations of one
//! template cannot differ there without differing in their arguments first.

use shiso_core::page::KvElem;
use shiso_core::{Backend, CoreError, CoreResult, Extent3};

use crate::sparse::exl3::{self as exl3, GemmShape};
use crate::sparse::kernels::{
    self, Elem, Entry, EntryTable, Instantiation, MOD_EMBED, MOD_EXACT, MOD_EXL3, MOD_GDN,
    MOD_GDN_FUSED, MOD_GDN_TC, MOD_HC, MOD_IDX, MOD_IDXRING, MOD_KV8, MOD_MATVEC, MOD_MISC,
    MOD_MOE, MOD_PLE, MOD_QSA, MOD_QSA_SPARSE, TemplateArgs,
};

/// Every module the sparse stack launches out of.
///
/// Walked whole rather than narrowed to the ones that hold templates today: a
/// module with none contributes nothing, and a template added to one of them
/// later is captured without anybody remembering to extend a list.
const MODULES: &[&str] = &[
    MOD_HC,
    MOD_MOE,
    MOD_GDN,
    MOD_GDN_TC,
    MOD_GDN_FUSED,
    MOD_PLE,
    MOD_MATVEC,
    MOD_QSA,
    MOD_QSA_SPARSE,
    MOD_IDX,
    MOD_IDXRING,
    MOD_MISC,
    MOD_EMBED,
    MOD_EXACT,
    MOD_KV8,
    MOD_EXL3,
];

/// Captures the template entry names this build's modules expose.
///
/// Empty when the backend compiled no kernels, which leaves every template
/// dispatch the typed refusal it already was.
///
/// # Leaking
///
/// The table is `&'static` because it travels in the options a stack is built
/// from and is copied at every dispatch; the names come from the device at
/// runtime, so the storage is leaked once here. A process loads its model once
/// and the table is a few hundred short strings, so the cost is a fixed few
/// kilobytes - paid instead of a process-wide `OnceLock`, which a second
/// backend in the same process would silently share.
#[must_use]
pub fn capture(backend: &dyn Backend) -> EntryTable {
    let mut rows: Vec<(Instantiation, &'static str)> = Vec::new();
    let mut dropped: Vec<Instantiation> = Vec::new();
    for module in MODULES.iter().copied() {
        for symbol in backend.module_entries(module) {
            let Some((template, args)) = decode(&symbol) else {
                continue;
            };
            let Some(inst) = classify(module, template, &args) else {
                continue;
            };
            if rows.iter().any(|(have, _)| *have == inst) {
                // Two symbols, one key. Which one the caller meant is not
                // recoverable from the key, so neither is offered.
                tracing::error!(
                    module,
                    template = inst.template,
                    "two entry points decode to one instantiation key; both are dropped"
                );
                dropped.push(inst);
                continue;
            }
            rows.push((inst, leak(symbol)));
        }
    }
    rows.retain(|(inst, _)| !dropped.contains(inst));
    tracing::debug!(
        captured = rows.len(),
        "template entry names read off the build"
    );
    EntryTable::new(Box::leak(rows.into_boxed_slice()))
}

/// One template argument, as the symbol spells it.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Arg {
    /// An integer literal: `Li4E`.
    Int(u32),
    /// A boolean literal: `Lb1E`.
    Bool(bool),
    /// A type, by the name the ABI gives it.
    Type(&'static str),
}

/// Splits a mangled entry point into the template it instantiates and the
/// arguments it carries, or `None` for anything this subset does not cover.
///
/// A mangled name with no argument list decodes to its name and no arguments:
/// a C++-linkage entry point that is not a template still cannot be named by a
/// constant, and [`classify`] decides which of those this build launches.
fn decode(symbol: &str) -> Option<(&str, Vec<Arg>)> {
    let rest = symbol.strip_prefix("_Z")?;
    let (name, rest) = length_prefixed(rest)?;
    let Some(mut rest) = rest.strip_prefix('I') else {
        return Some((name, Vec::new()));
    };
    let mut args = Vec::new();
    loop {
        if let Some(tail) = rest.strip_prefix('E') {
            // The argument list closes; the function signature follows and says
            // nothing two instantiations of one template could differ by.
            let _ = tail;
            return (!args.is_empty()).then_some((name, args));
        }
        let (arg, tail) = argument(rest)?;
        args.push(arg);
        rest = tail;
    }
}

/// Reads one `<length><identifier>` pair.
fn length_prefixed(text: &str) -> Option<(&str, &str)> {
    let digits = text.len() - text.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    if digits == 0 {
        return None;
    }
    let (len, rest) = text.split_at(digits);
    let len: usize = len.parse().ok()?;
    (rest.len() >= len).then(|| rest.split_at(len))
}

/// Reads one template argument.
fn argument(text: &str) -> Option<(Arg, &str)> {
    if let Some(rest) = text.strip_prefix("Li") {
        let end = rest.find('E')?;
        let value: u32 = rest.get(..end)?.parse().ok()?;
        return Some((Arg::Int(value), rest.get(end + 1..)?));
    }
    if let Some(rest) = text.strip_prefix("Lb") {
        let end = rest.find('E')?;
        let value = match rest.get(..end)? {
            "0" => false,
            "1" => true,
            _ => return None,
        };
        return Some((Arg::Bool(value), rest.get(end + 1..)?));
    }
    // The builtin types these entry points take. Named rather than accepted by
    // class, because an unrecognised builtin has to stop the decode: consuming
    // one character and carrying on would shift every later argument.
    for (code, name) in [
        ("f", "float"),
        ("d", "double"),
        ("i", "int"),
        ("j", "unsigned"),
    ] {
        if let Some(rest) = text.strip_prefix(code) {
            return Some((Arg::Type(name), rest));
        }
    }
    let (name, rest) = length_prefixed(text)?;
    Some((Arg::Type(intern_type(name)?), rest))
}

/// The class names these instantiations take, as `'static` strings.
///
/// A closed list rather than a leak: the element type of a kernel argument is
/// part of this crate's kernel table, and a checkpoint that produced some other
/// one would be describing a kernel set this build does not have.
fn intern_type(name: &str) -> Option<&'static str> {
    match name {
        "__half" => Some("__half"),
        "__nv_bfloat16" => Some("__nv_bfloat16"),
        _ => None,
    }
}

/// The element type a type name stands for.
fn elem(name: &str) -> Option<Elem> {
    match name {
        "float" => Some(Elem::F32),
        "__half" => Some(Elem::F16),
        _ => None,
    }
}

/// Reads one argument as an integer.
fn int(args: &[Arg], at: usize) -> Option<u32> {
    match args.get(at)? {
        Arg::Int(value) => Some(*value),
        Arg::Bool(_) | Arg::Type(_) => None,
    }
}

/// Reads one argument as an element type.
fn kind(args: &[Arg], at: usize) -> Option<Elem> {
    match args.get(at)? {
        Arg::Type(name) => elem(name),
        Arg::Int(_) | Arg::Bool(_) => None,
    }
}

/// Turns a decoded symbol into the instantiation key the stack dispatches by,
/// or `None` for a template this build's kernel table does not name.
///
/// The template name is matched against the constants rather than kept from the
/// symbol, so the `'static` name in the key is this crate's own and a symbol
/// carrying some other template is skipped rather than interned.
#[expect(
    clippy::too_many_lines,
    reason = "one arm per template the kernel set instantiates: the table is \
              the point, and splitting it would hide which name maps to which key"
)]
fn classify(module: &'static str, template: &str, args: &[Arg]) -> Option<Instantiation> {
    let of =
        |name: &'static str, template_args| Some(Instantiation::new(module, name, template_args));
    match template {
        "q4e_hc_norm_kernel" => of("q4e_hc_norm_kernel", TemplateArgs::Elem(kind(args, 0)?)),
        "q4e_hc_collapse_kernel" => {
            of("q4e_hc_collapse_kernel", TemplateArgs::Elem(kind(args, 0)?))
        }
        "q4e_hc_combine_norm_kernel" => of(
            "q4e_hc_combine_norm_kernel",
            TemplateArgs::Elem(kind(args, 0)?),
        ),
        "q4e_matvec_combine_t_kernel" => of(
            "q4e_matvec_combine_t_kernel",
            TemplateArgs::Elem(kind(args, 0)?),
        ),
        "q4e_matmul_bf16_rows_kernel" => of(
            "q4e_matmul_bf16_rows_kernel",
            TemplateArgs::Rows(int(args, 0)?),
        ),
        "q4e_matmul_q8_0_rows_narrow_kernel" => of(
            "q4e_matmul_q8_0_rows_narrow_kernel",
            TemplateArgs::Rows(int(args, 0)?),
        ),
        "q4e_matmul_q8_0_rows_warp_kernel" => of(
            "q4e_matmul_q8_0_rows_warp_kernel",
            TemplateArgs::Rows(int(args, 0)?),
        ),
        "q4e_matmul_p12_rows_kernel" => of(
            "q4e_matmul_p12_rows_kernel",
            TemplateArgs::Rows(int(args, 0)?),
        ),
        "q4e_p12_expand_kernel" => of("q4e_p12_expand_kernel", TemplateArgs::Elem(kind(args, 0)?)),
        "q4e_matmul_f16_rows_kernel" => of(
            "q4e_matmul_f16_rows_kernel",
            TemplateArgs::RowsIo {
                rows: int(args, 0)?,
                input: kind(args, 1)?,
                output: kind(args, 2)?,
            },
        ),
        "q4e_moe_down_kernel" => of(
            "q4e_moe_down_kernel",
            TemplateArgs::StoredType(int(args, 0)?),
        ),
        "q4e_moe_down_grouped_kernel" => of(
            "q4e_moe_down_grouped_kernel",
            TemplateArgs::StoredTypeBlocks {
                stored: int(args, 0)?,
                blocks: int(args, 1)?,
            },
        ),
        // `<bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES,
        //  FRAG_STAGES>`. The shape number is not an argument: it is recovered
        //  from the tile pair, which is what the shape table is keyed by.
        kernels::EXL3_GEMM => of(kernels::EXL3_GEMM, trellis(args)?),
        kernels::EXL3_MGEMM => of(kernels::EXL3_MGEMM, trellis(args)?),
        // `<bits, MOE_TILESIZE_N, cb>` and `<bits, cb>`: the output tile is
        // fixed in both, so the codec width is the whole key.
        kernels::EXL3_MOE => of(kernels::EXL3_MOE, TemplateArgs::TrellisBits(int(args, 0)?)),
        kernels::EXL3_RECONSTRUCT => of(
            kernels::EXL3_RECONSTRUCT,
            TemplateArgs::TrellisBits(int(args, 0)?),
        ),
        kernels::EXL3_MOE_WS_PREP if args.is_empty() => {
            of(kernels::EXL3_MOE_WS_PREP, TemplateArgs::Untemplated)
        }
        // `<bits, cb, STAGES>`: the codebook and the stage count are fixed by
        // the one instantiation per width the kernel file builds, so the width
        // is the key.
        kernels::EXL3_MOE_WS_GATEUP => of(
            kernels::EXL3_MOE_WS_GATEUP,
            TemplateArgs::TrellisBits(int(args, 0)?),
        ),
        kernels::EXL3_MOE_WS_DOWN => of(
            kernels::EXL3_MOE_WS_DOWN,
            TemplateArgs::TrellisBits(int(args, 0)?),
        ),
        kernels::EXL3_MOE_WS_DOWN_H => of(
            kernels::EXL3_MOE_WS_DOWN_H,
            TemplateArgs::TrellisBits(int(args, 0)?),
        ),
        // `<bits, cb, STAGES, WARPS, MT>`: the tile count is the only other
        // argument that varies between instances.
        kernels::EXL3_DGEMV => of(
            kernels::EXL3_DGEMV,
            TemplateArgs::TrellisTiles {
                bits: int(args, 0)?,
                tiles: int(args, 4)?,
            },
        ),
        kernels::EXL3_DGEMV_HAD if args.is_empty() => {
            of(kernels::EXL3_DGEMV_HAD, TemplateArgs::Untemplated)
        }
        // `<bits, cb, STAGES, WARPS>`, one instantiation per width.
        kernels::EXL3_MOE_WSD_GATEUP => of(
            kernels::EXL3_MOE_WSD_GATEUP,
            TemplateArgs::TrellisBits(int(args, 0)?),
        ),
        kernels::EXL3_MOE_WSD_DOWN => of(
            kernels::EXL3_MOE_WSD_DOWN,
            TemplateArgs::TrellisBits(int(args, 0)?),
        ),
        kernels::EXL3_MOE_WSD_PREP if args.is_empty() => {
            of(kernels::EXL3_MOE_WSD_PREP, TemplateArgs::Untemplated)
        }
        kernels::EXL3_MOE_WSD_ACT if args.is_empty() => {
            of(kernels::EXL3_MOE_WSD_ACT, TemplateArgs::Untemplated)
        }
        kernels::EXL3_MOE_WSD_COMBINE if args.is_empty() => {
            of(kernels::EXL3_MOE_WSD_COMBINE, TemplateArgs::Untemplated)
        }
        kernels::EXL3_MOE_WS_COMBINE if args.is_empty() => {
            of(kernels::EXL3_MOE_WS_COMBINE, TemplateArgs::Untemplated)
        }
        kernels::EXL3_MOE_WS_INVERT if args.is_empty() => {
            of(kernels::EXL3_MOE_WS_INVERT, TemplateArgs::Untemplated)
        }
        kernels::EXL3_MOE_WS_PREP_TOK if args.is_empty() => {
            of(kernels::EXL3_MOE_WS_PREP_TOK, TemplateArgs::Untemplated)
        }
        _ => None,
    }
}

/// The `<bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES,
/// FRAG_STAGES>` a cooperative projection carries, as the key the stack
/// dispatches by.
fn trellis(args: &[Arg]) -> Option<TemplateArgs> {
    let shape = GemmShape::from_tiles(int(args, 4)?, int(args, 5)?)?;
    Some(TemplateArgs::Trellis {
        bits: int(args, 0)?,
        shape: shape.get(),
    })
}

/// Which of the stack's optional kernel paths this device can launch at all.
///
/// Each path has an always-correct fallback the planner takes without it: the
/// dense attention and the scalar scorer for the tensor-core pair, the dense
/// attention for the tiled one, the sequential recurrence for the chunked
/// one. A backend whose kernel set leaves a path out - one whose device lacks
/// the matrix units, or the shared memory those layouts were written for -
/// answers here by not resolving its entry points, which is what
/// [`Backend::max_block_threads`] reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each flag is an independent probed fact about one kernel family; \
              no two are states of one thing"
)]
pub struct OptionalPaths {
    /// The tensor-core attention and indexer scorer.
    pub tensor_core: bool,
    /// The tiled attention.
    pub tiled_attention: bool,
    /// The chunked recurrence and its parallel convolution.
    pub chunked_recurrence: bool,
    /// The fused prompt kernels the chunked recurrence takes in place of its
    /// chain: on top of `chunked_recurrence`, never instead of it, because
    /// the fused path still launches that module's window rewrite.
    pub fused_recurrence: bool,
    /// The kernel a prompt pass's brain-float projections can take.
    pub prompt_matmul: crate::sparse::matmul::PromptMatmul,
    /// The block-scaled low-precision prompt kernels
    /// ([`kernels::LOWP_ENTRIES`]), which only a kernel set built for a
    /// family-specific architecture carries and only that family's devices
    /// load.
    pub lowp_prefill: bool,
    /// The one-launch routed selection a decode-sized pass takes.
    pub route_sort: crate::sparse::matmul::RouteSort,
}

/// Asks the device which optional paths it has kernels for.
#[must_use]
pub fn optional_paths(backend: &dyn Backend) -> OptionalPaths {
    use shiso_core::backend::KernelRef;
    let has = |module: &'static str, entry: &'static str| {
        backend
            .max_block_threads(KernelRef::new(module, entry))
            .is_some()
    };
    let paths = OptionalPaths {
        // The exact entry, because it is the one both tensor-core paths launch.
        tensor_core: has(MOD_EXACT, "q4e_qsa_attention_tc_exact_kernel")
            && has(MOD_IDX, "q4e_idx_score_tc_kernel"),
        tiled_attention: has(MOD_QSA, "q4e_qsa_attention_tiled_kernel"),
        chunked_recurrence: has(MOD_GDN_TC, "q4e_gdn_tc_prep_kernel")
            && has(MOD_GDN_TC, "q4e_gdn_tc_scan_kernel")
            && has(MOD_GDN_TC, "q4e_gdn_conv_par_kernel")
            && has(MOD_GDN_TC, "q4e_gdn_conv_tail_kernel"),
        fused_recurrence: has(MOD_GDN_FUSED, "q4e_gdf_prep_kernel")
            && has(MOD_GDN_FUSED, "q4e_gdf_scan_kernel"),
        prompt_matmul: if [
            "q4e_bf16_gemm_kernel",
            "q4e_bf16_gemm_split_kernel",
            "q4e_bf16_gemv_kernel",
            "q4e_bf16_rows_kernel",
            "q4e_bf16_rows_wide_kernel",
        ]
        .iter()
        .all(|entry| has(kernels::MOD_BF16, entry))
        {
            crate::sparse::matmul::PromptMatmul::Tensor
        } else if has(MOD_MATVEC, "q4e_bf16_gemm_kernel") {
            crate::sparse::matmul::PromptMatmul::Matrix
        } else {
            crate::sparse::matmul::PromptMatmul::Rows
        },
        lowp_prefill: kernels::LOWP_ENTRIES.iter().all(|&(m, e)| has(m, e)),
        route_sort: if has(MOD_MOE, "q4e_moe_route_sort_tiny_kernel")
            && has(kernels::MOD_FFN, "q4e_ffn_route_kernel")
            && has(kernels::MOD_FFN, "q4e_ffn_route_rows_kernel")
        {
            crate::sparse::matmul::RouteSort::Fused
        } else if has(MOD_MOE, "q4e_moe_route_sort_tiny_kernel") {
            crate::sparse::matmul::RouteSort::Tiny
        } else {
            crate::sparse::matmul::RouteSort::Small
        },
    };
    tracing::info!(?paths, "optional sparse kernel paths this device has");
    paths
}

/// Blocks of the trellis kernels this device can hold resident at once.
///
/// The number that bounds every cooperative grid, and the one device fact the
/// host rules in [`crate::sparse::exl3`] close over. It is the multiprocessor
/// count times the *smallest* occupancy any of these kernels achieves at the
/// shared-memory budget they are all launched with - smallest because one grid
/// size is used for all of them, and a grid sized for the roomiest kernel would
/// deadlock the tightest one: these kernels hand partial sums down a lock chain
/// and every block has to be running before any of them can finish.
///
/// Zero when the device has no cooperative launch, when this build captured no
/// trellis entry names, or when the device will not answer for any of them.
/// Zero is the same conservative answer the option defaults to - a single-block
/// grid, which is correct and slow - rather than a plausible-looking guess.
#[must_use]
pub fn co_resident_blocks(backend: &dyn Backend, entries: EntryTable) -> u32 {
    let Some(coop) = backend.as_coop() else {
        return 0;
    };
    let mut floor: Option<u32> = None;
    let mut ask = |entry: Entry, threads: u32| {
        let Ok(kernel) = entries.resolve(entry) else {
            return;
        };
        match coop.max_co_resident_blocks(kernel, Extent3::linear(threads), kernels::EXL3_SMEM_MAX)
        {
            Ok(blocks) => floor = Some(floor.map_or(blocks, |held| held.min(blocks))),
            Err(err) => tracing::warn!(
                entry = entry.name(),
                error = %err,
                "the device would not report an occupancy for a trellis kernel",
            ),
        }
    };
    for bits in exl3::MIN_BITS..=exl3::MAX_BITS {
        for shape in GemmShape::all() {
            let args = TemplateArgs::Trellis {
                bits,
                shape: shape.get(),
            };
            ask(
                Entry::template(MOD_EXL3, kernels::EXL3_GEMM, args),
                shape.block_dim(),
            );
            ask(
                Entry::template(MOD_EXL3, kernels::EXL3_MGEMM, args),
                shape.block_dim(),
            );
        }
        ask(
            Entry::template(MOD_EXL3, kernels::EXL3_MOE, TemplateArgs::TrellisBits(bits)),
            kernels::EXL3_MOE_BLOCK_DIM,
        );
    }
    let blocks = floor.unwrap_or(0);
    tracing::debug!(blocks, "cooperative grid ceiling measured on the device");
    blocks
}

/// Keeps one captured name for the life of the process.
fn leak(name: String) -> &'static str {
    Box::leak(name.into_boxed_str())
}

/// Refuses a cache element this device has no kernels for.
///
/// Asked once, before any weight is loaded, rather than left to the first launch: a stack whose store
/// or one of its six attention readers cannot resolve would fail partway into
/// the first prompt - or, for a reader only some shapes reach, hours later on
/// the first prompt long enough to take it. Every entry the element needs is
/// probed, so a kernel set that ships some of them is refused as loudly as one
/// that ships none (Metal's, or a build whose kernels were not compiled).
///
/// # Errors
///
/// [`CoreError::Unsupported`] naming the element when any of its entries is
/// missing.
pub fn check_kv_elem(backend: &dyn Backend, kv: KvElem) -> CoreResult<()> {
    let Some(entries) = kernels::kv_entries(kv) else {
        return Ok(());
    };
    let missing = entries.iter().find(|entry| {
        backend
            .max_block_threads(shiso_core::backend::KernelRef::new(entry.0, entry.1))
            .is_none()
    });
    if let Some((module, entry)) = missing {
        tracing::error!(
            kv_elem = %kv,
            module,
            entry,
            backend = backend.backend_name(),
            "the key/value cache element needs a kernel this device does not have"
        );
        return Err(CoreError::Unsupported {
            what: "an fp8 key/value cache on a device whose kernel set has no fp8 cache kernels \
                   (only the CUDA build compiled with its kernels has them; use \
                   --kv-cache-dtype f16)",
            backend: "engine",
        });
    }
    Ok(())
}

/// Refuses [`PrefillActivations::Lowp`] on a device whose kernel set lacks
/// any of the block-scaled prompt kernels, naming the first one missing.
///
/// Asked before the weights are read, for the reason [`check_kv_elem`] is:
/// a server that silently served the weight-only kernels under this option
/// would report speeds and numerics nobody asked for.
///
/// [`PrefillActivations::Lowp`]: crate::sparse::matmul::PrefillActivations::Lowp
///
/// # Errors
///
/// [`CoreError::Unsupported`] naming what is missing and what to do instead.
pub fn check_prefill_activations(
    backend: &dyn Backend,
    choice: crate::sparse::matmul::PrefillActivations,
) -> CoreResult<()> {
    if choice == crate::sparse::matmul::PrefillActivations::A16 {
        return Ok(());
    }
    let missing = kernels::LOWP_ENTRIES.iter().find(|entry| {
        backend
            .max_block_threads(shiso_core::backend::KernelRef::new(entry.0, entry.1))
            .is_none()
    });
    if let Some((module, entry)) = missing {
        tracing::error!(
            module,
            entry,
            backend = backend.backend_name(),
            "the low-precision prompt activations need a kernel this device does not have"
        );
        return Err(CoreError::Unsupported {
            what: "low-precision prompt activations on a device or kernel set without the \
                   block-scaled kernels (they need a CUDA build with SHISO_CUDA_COMPILE=1 and a \
                   GB10-class device, sm_120a/sm_121a; use --prefill-activations a16)",
            backend: "engine",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Arg, classify, decode};
    use crate::sparse::kernels::{Elem, MOD_EXL3, MOD_HC, MOD_MATVEC, TemplateArgs};

    #[test]
    fn a_one_type_instantiation_decodes_to_its_type() {
        let (name, args) = decode("_Z18q4e_hc_norm_kernelIfEvPT_PKfS3_jjf").expect("it decodes");
        assert_eq!(name, "q4e_hc_norm_kernel");
        assert_eq!(args, vec![Arg::Type("float")]);
    }

    #[test]
    fn a_class_argument_is_read_by_its_length_prefix() {
        let (name, args) =
            decode("_Z22q4e_hc_collapse_kernelI6__halfEvPfPKT_S4_jj").expect("it decodes");
        assert_eq!(name, "q4e_hc_collapse_kernel");
        assert_eq!(args, vec![Arg::Type("__half")]);
    }

    #[test]
    fn the_row_projection_carries_a_count_and_two_types() {
        let inst = classify(
            MOD_MATVEC,
            "q4e_matmul_f16_rows_kernel",
            &[Arg::Int(7), Arg::Type("__half"), Arg::Type("float")],
        )
        .expect("it classifies");
        assert_eq!(
            inst.args,
            TemplateArgs::RowsIo {
                rows: 7,
                input: Elem::F16,
                output: Elem::F32,
            }
        );
    }

    #[test]
    fn a_trellis_shape_is_recovered_from_its_tile_pair_and_not_its_position() {
        let symbol = "_Z16exl3_gemm_kernelILi5ELb1ELi2ELi16ELi32ELi256ELi4ELi3EEvPKfPKtPviiiPiPK6__halfPS6_S8_";
        let (name, args) = decode(symbol).expect("it decodes");
        let inst = classify(MOD_EXL3, name, &args).expect("it classifies");
        // Tiles 32x256 are shape three; nothing in the symbol says "three".
        assert_eq!(inst.args, TemplateArgs::Trellis { bits: 5, shape: 3 });
    }

    #[test]
    fn an_untemplated_mangled_entry_is_captured_by_name() {
        let (name, args) =
            decode("_Z16exl3_moe_ws_prepPKfP6__halfS2_PKS1_S4_PKiS6_iiii").expect("it decodes");
        assert_eq!(name, "exl3_moe_ws_prep");
        let inst = classify(MOD_EXL3, name, &args).expect("it classifies");
        assert_eq!(inst.args, TemplateArgs::Untemplated);
    }

    #[test]
    fn a_plain_c_linkage_name_is_not_a_template() {
        assert!(decode("q4e_hc_init_kernel").is_none());
    }

    #[test]
    fn a_template_this_build_does_not_name_is_skipped_rather_than_interned() {
        assert!(classify(MOD_HC, "some_other_kernel", &[Arg::Int(1)]).is_none());
    }

    #[test]
    fn a_truncated_argument_list_decodes_to_nothing() {
        assert!(decode("_Z18q4e_hc_norm_kernelIf").is_none());
    }
}
