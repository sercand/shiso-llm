//! Throwaway: dump mmproj metadata keys, tensor inventory and clamp values.
#![expect(
    clippy::print_stdout,
    clippy::collapsible_if,
    clippy::default_trait_access,
    reason = "throwaway inspection binary"
)]
use shiso_modelfile::GgufFile;

fn main() {
    let path = std::env::args().nth(1).expect("path");
    let f = GgufFile::open(&path).expect("open");
    if let Some(which) = std::env::args().nth(2) {
        if which == "clamps" {
            let mut seen = Vec::new();
            for t in f.tensors() {
                let n = t.name.as_ref();
                if n.starts_with("v.")
                    && (n.ends_with(".input_max")
                        || n.ends_with(".input_min")
                        || n.ends_with(".output_max")
                        || n.ends_with(".output_min"))
                {
                    let v = f.get(n).expect("view");
                    let bytes = v.data;
                    let val = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    seen.push((n.to_owned(), val));
                }
            }
            let mut uniq: std::collections::BTreeMap<String, usize> = Default::default();
            for (_, v) in &seen {
                *uniq.entry(format!("{v}")).or_default() += 1;
            }
            println!("total: {}", seen.len());
            for (k, c) in &uniq {
                println!("{k} x{c}");
            }
            for (n, v) in &seen {
                if *v < 1e30 && *v > -1e30 {
                    println!("NONDEFAULT {n} = {v}");
                }
            }
            return;
        }
    }
    println!("--- metadata ---");
    for (k, v) in f.metadata().entries() {
        println!("{k} = {v:?}");
    }
    println!("--- tensors ({}) ---", f.tensors().len());
    for t in f.tensors() {
        println!("{} {:?} {:?}", t.name, t.dtype, t.shape.dims());
    }
}
