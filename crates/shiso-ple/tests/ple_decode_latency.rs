//! How long a decode step's gather takes against the shipped NVFP4 n-gram
//! table: sixteen uncached rows, the batch one token of the NVIDIA export asks
//! for, read through the thread pool the sharded table is served by.
//!
//! Needs the checkpoint (`SHISO_TEST_MODEL_QWEN38_FP4`, the directory holding
//! `model.safetensors.index.json`), so it is ignored and says on stderr why it
//! skipped.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use shiso_core::ids::TokenId;
use shiso_ple::{
    PleConfig, PleIo, PleParams, PleParamsDesc, PleRowType, PleSegment, PleShards, PleStream,
};

/// Shards and row width of the shipped table.
const SHARDS: usize = 128;
const HEAD_DIM: u32 = 160;
const PREFIX: &str = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";

/// `(file, absolute offset, rows)` of one tensor, from its file's header.
fn locate(dir: &Path, map: &HashMap<String, String>, name: &str) -> (Arc<File>, u64, u64) {
    let rel = map
        .get(name)
        .unwrap_or_else(|| panic!("{name} is in the index"));
    let path = dir.join(rel);
    let mut file = File::open(&path).expect("the shard opens");
    let mut len = [0u8; 8];
    file.read_exact(&mut len).expect("a header length");
    let len = u64::from_le_bytes(len);
    let mut header = vec![0u8; usize::try_from(len).expect("a header fits")];
    file.read_exact(&mut header).expect("the header");
    let header: serde_json::Value = serde_json::from_slice(&header).expect("a JSON header");
    let entry = &header[name];
    let begin = entry["data_offsets"][0].as_u64().expect("an offset");
    let rows = entry["shape"][0].as_u64().expect("a shape");
    (Arc::new(file), 8 + len + begin, rows)
}

fn open(dir: &Path, cfg: &PleConfig) -> PleStream {
    let index: serde_json::Value = serde_json::from_reader(
        File::open(dir.join("model.safetensors.index.json")).expect("the index opens"),
    )
    .expect("a JSON index");
    let map: HashMap<String, String> = index["weight_map"]
        .as_object()
        .expect("a weight map")
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().expect("a file name").to_owned()))
        .collect();
    let mut shards = Vec::with_capacity(SHARDS);
    let mut rows_per_shard = 0;
    for s in 0..SHARDS {
        let mut segments = Vec::new();
        for (part, width) in [("weight", HEAD_DIM / 2), ("weight_scale", HEAD_DIM / 16)] {
            let (file, offset, rows) = locate(dir, &map, &format!("{PREFIX}.shard_{s}.{part}"));
            rows_per_shard = rows;
            segments.push(PleSegment {
                file,
                offset,
                row_bytes: width,
            });
        }
        shards.push(segments);
    }
    let params = PleParams::new(PleParamsDesc {
        row_type: PleRowType::Nvfp4 {
            global_bits: 1.0f32.to_bits(),
        },
        ngram_size: 3,
        heads_per_ngram: 1,
        head_dim: HEAD_DIM,
        eos_token: TokenId::new(7),
        multipliers: vec![3, 5, 7],
        head_offsets: vec![0, 11],
        head_vocab: vec![11, 13],
        head_bias: None,
    })
    .expect("valid parameters");
    PleStream::open_sharded(
        PleShards {
            rows_per_shard,
            shards,
        },
        rows_per_shard * SHARDS as u64,
        params,
        0,
        cfg,
    )
    .expect("the table opens")
}

#[test]
#[ignore = "reads the shipped checkpoint"]
fn a_decode_gather_of_sixteen_cold_rows() {
    let Ok(dir) = std::env::var("SHISO_TEST_MODEL_QWEN38_FP4") else {
        tracing::warn!("SHISO_TEST_MODEL_QWEN38_FP4 is unset; skipping");
        return;
    };
    let dir = Path::new(&dir);
    // A fresh seed each run, so the rows are not the page cache's from the
    // last; `SHISO_TEST_PLE_SEED` pins one, to time rows the cache holds.
    let mut state = std::env::var("SHISO_TEST_PLE_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0x9E37_79B9_7F4A_7C15, |d| u64::from(d.subsec_nanos()))
        })
        | 1;
    for workers in [64u32, 1] {
        let stream = open(
            dir,
            &PleConfig {
                io: PleIo::Pool,
                qd: 256,
                workers,
            },
        );
        let n_rows = stream.n_rows();
        let mut out = vec![0u8; 16 * 90];
        let mut times = Vec::new();
        for _ in 0..300 {
            let rows: Vec<u64> = (0..16)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state % n_rows
                })
                .collect();
            let t = Instant::now();
            stream.fetch(&rows, &mut out).expect("the rows read");
            times.push(t.elapsed().as_secs_f64() * 1e6);
        }
        times.sort_by(f64::total_cmp);
        let report = format!(
            "workers {workers}: median {:.0} us, p90 {:.0} us, p99 {:.0} us",
            times[150], times[270], times[297]
        );
        std::io::Write::write_all(&mut std::io::stderr(), format!("{report}\n").as_bytes())
            .expect("stderr");
    }
}
