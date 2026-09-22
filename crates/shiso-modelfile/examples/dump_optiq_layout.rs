//! Prints the layout fixture for a sharded checkpoint directory.
//!
//! The output is the file `test-vectors/loader/qwen35-optiq-layout.json` is made of:
//! the raw tensor table's size and format mix, a digest of that table, the
//! quantization config verbatim, and the shape of the translated tensor stream.
//! Regenerating it is how a fixture stops being a number somebody typed.
//!
//! The digest covers one sorted row per tensor, `name|dtype|d0,d1,...`, newline
//! separated with a trailing newline - the tensor table and nothing else, so it
//! is stable against a re-export that moves payloads around without changing
//! what they are.
//!
//! ```text
//! cargo run -p shiso-modelfile --example dump_optiq_layout -- <checkpoint-dir>
//! ```

use std::collections::BTreeMap;
use std::io;
use std::path::Path;
use std::process::Command;

use shiso_core::DType;
use shiso_modelfile::{GgufError, SafetensorsFile};

/// Where this checkpoint was published, recorded so the fixture says what it
/// is a fixture of.
const SOURCE: &str = "mlx-community/Qwen3.5-4B-OptiQ-4bit";
/// Shard the language model lives in.
const SHARD: &str = "model.safetensors";
/// Config naming the architecture and the quantization parameters.
const CONFIG: &str = "config.json";
/// Bytes of little-endian length in front of the header JSON.
const HEADER_LEN_BYTES: usize = 8;
/// Bits in one packed storage word.
const WORD_BITS: u64 = 32;

/// Prints the fixture for the directory named by the first argument.
#[expect(clippy::print_stdout, reason = "this example's output is its product")]
fn main() -> Result<(), GgufError> {
    let dir = std::env::args().nth(1).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: dump_optiq_layout <checkpoint-dir>",
        )
    })?;
    let dir = Path::new(&dir);

    let shard = std::fs::read(dir.join(SHARD))?;
    let (table, header_bytes) = header_table(&shard)?;
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join(CONFIG))?)
            .map_err(|err| io::Error::other(format!("config does not parse: {err}")))?;

    let mut dtype_counts: BTreeMap<&str, u64> = BTreeMap::new();
    for entry in table.values() {
        *dtype_counts.entry(entry.dtype.as_str()).or_default() += 1;
    }

    let file = SafetensorsFile::open(dir)?;
    let mut total = 0_u64;
    let mut f16 = 0_u64;
    let mut f32 = 0_u64;
    for desc in file.tensors() {
        total += 1;
        match desc.dtype {
            DType::F16 => f16 += 1,
            DType::F32 => f32 += 1,
            _ => {}
        }
    }

    let fixture = serde_json::json!({
        "source": SOURCE,
        "file": SHARD,
        "bytes": shard.len(),
        "header_bytes": header_bytes,
        "header_sha256": sha256(&digest_rows(&table))?,
        "n_tensors": table.len(),
        "dtype_counts": dtype_counts,
        "bits_histogram": bits_histogram(&table, &config["quantization_config"]),
        "gguf_tensors": { "total": total, "f16": f16, "f32": f32 },
        "quantization_config": config["quantization_config"].clone(),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&fixture)
            .map_err(|err| io::Error::other(format!("the fixture does not render: {err}")))?
    );
    Ok(())
}

/// One raw header entry.
#[derive(Debug)]
struct RawEntry {
    /// Storage format as the header spells it.
    dtype: String,
    /// Dimensions, slowest-varying axis first.
    shape: Vec<u64>,
}

/// Reads the shard's tensor table and the length of its header.
fn header_table(shard: &[u8]) -> Result<(BTreeMap<String, RawEntry>, u64), GgufError> {
    let len_bytes: [u8; HEADER_LEN_BYTES] = shard
        .get(..HEADER_LEN_BYTES)
        .and_then(|head| head.try_into().ok())
        .ok_or_else(|| io::Error::other("the shard is shorter than its length prefix"))?;
    let header_bytes = u64::from_le_bytes(len_bytes);
    let end = HEADER_LEN_BYTES
        + usize::try_from(header_bytes)
            .map_err(|_| io::Error::other("the header does not fit in memory"))?;
    let json = shard
        .get(HEADER_LEN_BYTES..end)
        .ok_or_else(|| io::Error::other("the header runs past the end of the shard"))?;
    let raw: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(json)
        .map_err(|err| io::Error::other(format!("the header does not parse: {err}")))?;

    let mut table = BTreeMap::new();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let dtype = value
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string();
        let shape = value
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .map(|dims| dims.iter().filter_map(serde_json::Value::as_u64).collect())
            .unwrap_or_default();
        table.insert(name, RawEntry { dtype, shape });
    }
    Ok((table, header_bytes))
}

/// Renders the digest's input: one sorted row per tensor.
fn digest_rows(table: &BTreeMap<String, RawEntry>) -> String {
    use std::fmt::Write as _;

    let mut text = String::new();
    for (name, entry) in table {
        let dims: Vec<String> = entry.shape.iter().map(u64::to_string).collect();
        writeln!(text, "{name}|{}|{}", entry.dtype, dims.join(","))
            .expect("writing into a String cannot fail");
    }
    text
}

/// Counts the packed tensors by the bit width their shapes imply.
fn bits_histogram(
    table: &BTreeMap<String, RawEntry>,
    quant: &serde_json::Value,
) -> BTreeMap<String, u64> {
    let default_group = quant["group_size"].as_u64().unwrap_or(0);
    let mut histogram: BTreeMap<String, u64> = BTreeMap::new();
    for (name, entry) in table {
        let Some(base) = name
            .strip_suffix(".weight")
            .filter(|_| entry.dtype == "U32")
        else {
            continue;
        };
        let Some(scales) = table.get(&format!("{base}.scales")) else {
            continue;
        };
        let group = quant
            .get(base)
            .and_then(|entry| entry.get("group_size"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(default_group);
        let packed_last = entry.shape.last().copied().unwrap_or(0);
        let scales_last = scales.shape.last().copied().unwrap_or(0);
        let divisor = scales_last * group;
        if divisor == 0 {
            continue;
        }
        *histogram
            .entry((packed_last * WORD_BITS / divisor).to_string())
            .or_default() += 1;
    }
    histogram
}

/// Hex `sha256` of `text`, computed by the system tool.
///
/// The digest is fixture bookkeeping, not a runtime feature, so it does not
/// earn a hash implementation in the crate's dependency tree.
fn sha256(text: &str) -> Result<String, GgufError> {
    let path = std::env::temp_dir().join("shiso-optiq-layout-rows.txt");
    std::fs::write(&path, text)?;
    let out = Command::new("shasum")
        .args(["-a", "256"])
        .arg(&path)
        .output()?;
    drop(std::fs::remove_file(&path));
    if !out.status.success() {
        return Err(GgufError::Io(io::Error::other("hashing the header failed")));
    }
    String::from_utf8(out.stdout)
        .ok()
        .and_then(|line| line.split_whitespace().next().map(str::to_string))
        .ok_or_else(|| GgufError::Io(io::Error::other("the hashing tool printed nothing usable")))
}
