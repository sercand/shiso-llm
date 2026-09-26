//! Every extended-matvec name a launch-specializing backend is handed is
//! instantiated by the Metal sources.
//!
//! [`shiso_engine::launch::ext_name`] is a name table, and the import check
//! consults the table rather than a compiled library. A row that names a kernel
//! no source instantiates passes the import, and the first prefill chunk of that
//! width fails in the driver - the f16 row did exactly that for any chunk of two
//! to five tokens, a prompt length away from every test that happened to pass.
//!
//! Metal instantiates this family through `[[host_name("...")]]`, so that is
//! the spelling searched for. The by-name backend has its own table
//! ([`shiso_engine::launch::ext_name_for`]) and its own symbols.

use std::path::Path;

use shiso_core::DType;
use shiso_engine::launch::ext_name;

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
