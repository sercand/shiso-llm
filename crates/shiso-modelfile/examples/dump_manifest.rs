//! Prints one line per tensor so a directory can be diffed against a
//! checked-in fixture.
//!
//! Format, tab separated, in directory order:
//!
//! ```text
//! name \t ndim \t d0[,d1,...] \t format id \t relative payload offset
//! ```
//!
//! The offset is relative to the payload base rather than absolute, so the
//! output stays stable when a metadata table changes size and the payload moves
//! as a block. That is the property a regression fixture wants to pin: which
//! tensor is where relative to its neighbours.
//!
//! ```text
//! cargo run -p shiso-modelfile --example dump_manifest -- <model.gguf>
//! ```

use shiso_modelfile::{GgufError, GgufFile, TensorDesc};

/// Prints the manifest of the model named by the first argument.
#[expect(clippy::print_stdout, reason = "this example's output is its product")]
fn main() -> Result<(), GgufError> {
    let path = std::env::args().nth(1).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: dump_manifest <model.gguf>",
        )
    })?;

    let file = GgufFile::open(&path)?;
    let base = file.tensor_data_pos();
    for desc in file.tensors() {
        println!("{}", manifest_line(desc, base));
    }
    Ok(())
}

/// Renders one tensor in manifest form.
fn manifest_line(desc: &TensorDesc, base: u64) -> String {
    let dims = desc
        .shape
        .dims()
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let rel = desc.abs_offset.saturating_sub(base);
    format!(
        "{}\t{}\t{}\t{}\t{rel}",
        desc.name,
        desc.shape.ndim(),
        dims,
        desc.dtype.gguf_id(),
    )
}
