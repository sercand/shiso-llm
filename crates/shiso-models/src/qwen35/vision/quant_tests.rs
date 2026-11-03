//! The mixed-precision export's tower served as stored, against the same
//! tower with its quantized linears decoded to brain floats on the host - the
//! way the loader read it before the native projections existed, kept here
//! only as the reference a test compares against.
//!
//! Device-gated and checkpoint-gated: `SHISO_TEST_MODEL_QWEN4EXP_NVFP4MX` names
//! the export directory. Only the tower is loaded, never the language model.
//!
//! `SHISO_CUDA_COMPILE=1 SHISO_TEST_MODEL_QWEN4EXP_NVFP4MX=<dir> cargo test
//! -p shiso-models --release --lib quant_tests -- --ignored --nocapture`

use std::sync::Arc;
use std::time::Instant;

use shiso_core::{Backend, DType};
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_vision::{ImageLimits, VisionItem, VisionSource, preprocess_image};

use super::{MODELOPT_TOWER_PREFIX, TowerTensor, VisionWeights, load_vision_weights, upload};
use crate::qwen35::vision::encoder::VisionEncoder;
use crate::qwen35::vision::manifest::TowerManifest;
use shiso_modelfile::modelopt::ModeloptFile;

/// The picture the comparison encodes: a real image, so the activations are a
/// real tower's.
const IMAGE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test-vectors/prompts/gemma4-vision-quadrants.png"
);

/// Encodes per timing sample, after one warm-up.
const RUNS: usize = 5;

/// Side of the tiled grid the picture-sized encode runs: 4096 patch rows.
const LARGE_SIDE: usize = 64;

/// Whether a tensor is stored in one of the formats served as stored.
fn quantized(dtype: DType) -> bool {
    matches!(dtype, DType::Mxfp8E4M3G32 | DType::Nvfp4Gs16)
}

/// The tower with every quantized linear `pick` selects replaced by its host
/// decode, rounded to brain floats: MXFP8 exactly, NVFP4 once from its `f32`
/// product, the way the loader read them before.
fn decoded(
    backend: &Arc<dyn Backend>,
    dir: &std::path::Path,
    native: VisionWeights,
    pick: fn(DType) -> bool,
) -> VisionWeights {
    let file = ModeloptFile::open(dir, &[MODELOPT_TOWER_PREFIX]).expect("the export opens");
    let manifest = TowerManifest::build(&native.prefix, native.dims);
    let mut specs = manifest.iter().cloned().collect::<Vec<_>>().into_iter();
    let tensors = native
        .tensors
        .try_map::<TowerTensor, std::convert::Infallible>(&mut |tensor: TowerTensor| {
            let spec = specs.next().expect("one spec a tensor");
            assert_eq!(spec.name(), tensor.name());
            if !quantized(tensor.dtype()) || !pick(tensor.dtype()) {
                return Ok(tensor);
            }
            let entry = file
                .get(tensor.name())
                .expect("the tensor is in the export");
            let mut values = Vec::new();
            if let Some(parts) = file.mxfp8_parts(entry).expect("parts") {
                shiso_modelfile::mxfp8::dequant_mxfp8(
                    tensor.name(),
                    parts.layout,
                    parts.codes,
                    parts.scales,
                    &mut values,
                )
                .expect("decodes");
            } else if let Some(parts) = file.nvfp4_parts(entry).expect("parts") {
                shiso_modelfile::nvfp4::dequant_nvfp4(
                    parts.layout,
                    parts.weights,
                    parts.scales,
                    &mut values,
                )
                .expect("decodes");
            }
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
                .collect();
            Ok(upload(backend, &spec, DType::BF16, &bytes).expect("uploads"))
        })
        .expect("infallible");
    VisionWeights {
        dims: native.dims,
        prefix: native.prefix,
        tensors,
    }
}

/// One encode's output, and the median wall time of [`RUNS`] encodes.
fn encode(
    backend: &Arc<dyn Backend>,
    weights: &VisionWeights,
    item: &VisionItem,
) -> (Vec<f32>, f64) {
    let encoder = VisionEncoder::new();
    let out = encoder.encode(backend, weights, item).expect("encodes");
    let mut times = Vec::new();
    for _ in 0..RUNS {
        let start = Instant::now();
        drop(encoder.encode(backend, weights, item).expect("encodes"));
        backend.sync().expect("drains");
        times.push(start.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(f64::total_cmp);
    let mut bytes = vec![0u8; usize::try_from(out.len()).expect("fits")];
    backend
        .copy_d2h(out.view(), &mut bytes)
        .expect("reads back");
    let values = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    (values, times[RUNS / 2])
}

/// `small`'s patches tiled onto a `side x side` grid: a picture-sized encode
/// whose activations are still a real image's.
fn tiled(small: &VisionItem, side: usize) -> VisionItem {
    let (_, gh, gw) = small.grid();
    let row = small.row_len();
    let src = small.as_flat();
    let mut pixels = Vec::with_capacity(side * side * row);
    for y in 0..side {
        for x in 0..side {
            let at = ((y % gh) * gw + (x % gw)) * row;
            pixels.extend_from_slice(&src[at..at + row]);
        }
    }
    VisionItem::new(pixels.into_boxed_slice(), row, 1, side, side).expect("a whole grid")
}

/// Relative L2 distance, largest absolute difference and largest reference
/// magnitude between two encoder outputs.
fn distance(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    assert_eq!(got.len(), want.len());
    let (mut diff2, mut ref2, mut max_abs, mut max_ref) = (0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        assert!(g.is_finite(), "a non-finite output");
        let d = f64::from(g) - f64::from(w);
        diff2 += d * d;
        ref2 += f64::from(w) * f64::from(w);
        max_abs = max_abs.max(d.abs());
        max_ref = max_ref.max(f64::from(w).abs());
    }
    ((diff2 / ref2).sqrt(), max_abs, max_ref)
}

#[test]
#[ignore = "requires a CUDA device and the mixed-precision export"]
#[expect(
    clippy::print_stderr,
    reason = "the measured error and times are the point of an opt-in device test"
)]
fn the_native_quantized_tower_matches_the_load_time_decode() {
    let Some(dir) = std::env::var_os("SHISO_TEST_MODEL_QWEN4EXP_NVFP4MX")
        .map(std::path::PathBuf::from)
        .filter(|dir| dir.is_dir())
    else {
        eprintln!("note: SHISO_TEST_MODEL_QWEN4EXP_NVFP4MX names no export directory; skipped");
        return;
    };
    let backend: Arc<dyn Backend> = match CudaBackend::new(0, CudaOptions::default()) {
        Ok(cuda) => cuda,
        Err(err) => {
            eprintln!("note: no CUDA device, skipped: {err}");
            return;
        }
    };
    let source = VisionSource::ModeloptDir(dir.clone());
    let load = || load_vision_weights(&backend, &source).expect("the tower loads as stored");
    let native = load();
    let count = native
        .tensors()
        .iter()
        .filter(|t| quantized(t.dtype()))
        .count();
    assert!(count > 0, "the export's tower carries quantized linears");
    let spec = shiso_vision::PreprocessSpec::qwen35(native.dims().geometry());
    let png = std::fs::read(IMAGE).expect("the test image is checked in");
    let small = preprocess_image(&png, &spec, ImageLimits::default()).expect("preprocesses");
    let large = tiled(&small, LARGE_SIDE);
    let items = [small, large];

    let native_bytes = native.device_bytes();
    let got: Vec<_> = items
        .iter()
        .map(|item| encode(&backend, &native, item))
        .collect();
    drop(native);
    // Only the MXFP8 linears decoded: those decode exactly, so what remains
    // is the native kernel's summation order against the BLAS pair's.
    let mx = decoded(&backend, &dir, load(), |d| d == DType::Mxfp8E4M3G32);
    assert!(
        mx.tensors()
            .iter()
            .all(|t| t.dtype() != DType::Mxfp8E4M3G32),
        "every MXFP8 linear of the reference is decoded"
    );
    let mx_out: Vec<_> = items
        .iter()
        .map(|item| encode(&backend, &mx, item))
        .collect();
    drop(mx);
    let all = decoded(&backend, &dir, load(), |_| true);
    let decoded_bytes = all.device_bytes();
    let all_out: Vec<_> = items
        .iter()
        .map(|item| encode(&backend, &all, item))
        .collect();

    eprintln!(
        "{count} quantized linears; tower {} MB as stored vs {} MB decoded",
        native_bytes >> 20,
        decoded_bytes >> 20
    );
    for (i, item) in items.iter().enumerate() {
        let (mx_l2, mx_worst, _) = distance(&got[i].0, &mx_out[i].0);
        let (l2, max_abs, max_ref) = distance(&got[i].0, &all_out[i].0);
        eprintln!(
            "grid {:?}: vs all decoded rel L2 {l2:.3e}, max |diff| {max_abs:.3e} of max |ref| \
             {max_ref:.3e}; vs MXFP8 decoded rel L2 {mx_l2:.3e}, max |diff| {mx_worst:.3e}; \
             encode {:.1} ms native vs {:.1} ms decoded",
            item.grid(),
            got[i].1,
            all_out[i].1,
        );
        // The MXFP8 weights are the same values on both sides; only the
        // order the f32 sums are taken in differs.
        assert!(
            mx_l2 < 1e-3,
            "rel L2 {mx_l2} with only the MXFP8 linears decoded"
        );
        // Against the full decode the NVFP4 weights differ too: the decode
        // rounded each `e2m1 * e4m3 * global` to a brain float (2^-9
        // relative), where the kernel multiplies the exact `e2m1 * e4m3` and
        // applies the tensor scale to the sum. A misread code, scale or tile
        // moves the output by tens of percent.
        assert!(
            l2 < 1e-2,
            "rel L2 {l2} between the native and decoded towers"
        );
    }
}
