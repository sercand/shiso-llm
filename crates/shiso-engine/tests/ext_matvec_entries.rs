//! Every extended-matvec name a launch-specializing backend is handed is
//! instantiated by that backend's sources.
//!
//! [`shiso_engine::launch::ext_name`] is a name table, and the import check
//! consults the table rather than a compiled library. A row that names a kernel
//! no source instantiates passes the import, and the first prefill chunk of that
//! width fails in the driver - the f16 row did exactly that for any chunk of two
//! to five tokens, a prompt length away from every test that happened to pass.
//!
//! Metal instantiates this family through `[[host_name("...")]]`, so that is
//! the spelling searched for. The by-name backend has its own table
//! ([`shiso_engine::launch::ext_name_for`]) and its own symbols: its CUDA
//! source instantiates each one through the `SHISO_MV_EXT_KQ` macro, and the
//! second test below reconstructs the symbol names from those macro
//! invocations, because the compiled module this machine cannot build is the
//! only other place the names exist.

use std::path::Path;

use shiso_core::DType;
use shiso_engine::launch::{ext_name, ext_name_for};

/// The formats the table answers for.
const DTYPES: [DType; 6] = [
    DType::Q4_K,
    DType::Q4_0,
    DType::Q8_0,
    DType::Q5_K,
    DType::Q6_K,
    DType::F16,
];

/// Every `.metal` source under `dir`, concatenated.
fn metal_sources(dir: &str) -> String {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../shiso-metal/metal")
        .join(dir);
    let mut text = String::new();
    for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let path = entry.expect("the directory lists").path();
        if path.extension().and_then(|e| e.to_str()) == Some("metal") {
            text.push_str(&std::fs::read_to_string(&path).expect("the source reads"));
        }
    }
    assert!(!text.is_empty(), "{} holds no Metal source", dir.display());
    text
}

#[test]
fn every_extended_matvec_name_is_instantiated_by_a_metal_source() {
    let sources = metal_sources("generic") + &metal_sources("qwen35");
    let mut named = 0usize;
    for dtype in DTYPES {
        for rows in 1..=5 {
            let Some(name) = ext_name(dtype, rows) else {
                continue;
            };
            assert!(
                sources.contains(&format!("[[host_name(\"{name}\")]]")),
                "{name} is what ext_name returns for {dtype:?} at {rows} rows, and no Metal \
                 source instantiates it; a chunk of that width would fail in the driver"
            );
            named += 1;
        }
    }
    assert!(named >= 20, "the table named only {named} entries");
}

/// The nine mixed sub-4-bit formats, spelled the way the CUDA generic module
/// instantiates them.
const SALUKI_TAGS: [(DType, &str); 9] = [
    (DType::Q2_K, "q2_K"),
    (DType::IQ2_XXS, "iq2_xxs"),
    (DType::IQ2_XS, "iq2_xs"),
    (DType::IQ2_S, "iq2_s"),
    (DType::IQ3_XXS, "iq3_xxs"),
    (DType::IQ3_S, "iq3_s"),
    (DType::IQ1_S, "iq1_s"),
    (DType::IQ1_M, "iq1_m"),
    (DType::IQ4_XS, "iq4_xs"),
];

/// Every by-name i-quant name the engine will hand a CUDA device is one the
/// CUDA generic source instantiates.
///
/// The check is over the source text, not a compiled module: `SHISO_MV_EXT_KQ`
/// pastes `{tag}`, `{r1ptg}` and `{nxpsg}` into the symbol name, so the macro
/// invocation is the most compact place the full name exists before nvcc runs,
/// and this machine has no nvcc. A missing row here is a prefill chunk that
/// dies in `cuModuleGetFunction` on the device.
#[test]
fn every_by_name_i_quant_name_is_instantiated_by_the_cuda_generic_source() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/cuda/generic/generic.cu");
    let source =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut named = 0usize;
    for (dtype, tag) in SALUKI_TAGS {
        for r1ptg in 1..=5u32 {
            for nxpsg in [4u32, 8, 16] {
                let want = format!("kernel_mul_mv_ext_{tag}_f32_r1_{r1ptg}_nx{nxpsg}");
                assert_eq!(
                    ext_name_for(dtype, r1ptg, nxpsg, true),
                    Some(want.as_str()),
                    "the by-name table and the CUDA macro spelling disagree for {dtype:?} \
                     at r1ptg {r1ptg}, nxpsg {nxpsg}"
                );
                assert!(
                    source.contains(&format!("SHISO_MV_EXT_KQ({tag}, {r1ptg}, {nxpsg})")),
                    "{want} is what ext_name_for returns for {dtype:?} at r1ptg {r1ptg} and \
                     nxpsg {nxpsg}, and the CUDA generic source does not instantiate it"
                );
                named += 1;
            }
        }
    }
    assert_eq!(named, 135, "the table named only {named} entries");
}
