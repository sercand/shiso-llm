//! A picture through the Qwen 3.5 mixture-of-experts stack, on this device.
//!
//! The last step of `docs/superpowers/plans/2026-09-20-qwen35moe-cuda.md`, and
//! the first family in this tree where both halves existed before the text
//! did: Phase 8 Part B brought the tower up on CUDA and Part C landed the
//! media seam, so what was left here was the container arm and the
//! `MediaModel` impl.
//!
//! `SHISO_CUDA_COMPILE=1 SHISO_TEST_MODEL_QWEN35MOE=<dir> cargo test -p
//! shiso-cuda --test q35moe_vision -- --ignored --nocapture`
//!
//! # What this asserts, and why it is two pictures
//!
//! That the pass completes is not the question - `q35moe_load.rs` already
//! answers it, and a splice that wrote nothing at all would still leave a
//! prompt of placeholder embeddings and a fluent answer. The question is
//! whether the PIXELS reach the logits, and the only thing that answers it is
//! two different pictures through the same prompt: identical dimensions, so an
//! identical placeholder run and an identical token sequence, and different
//! content. If the two answer the same thing, the tower's rows are not being
//! spliced, or are being spliced somewhere the attention does not read.
//!
//! The pictures are drawn here rather than read from a fixture for the same
//! reason `crates/shiso-server/tests/media_e2e.rs` draws its own: a flat field
//! is the one picture whose patches are all identical and therefore the one
//! least likely to move an answer, and a reader of this file should be able to
//! see what the model was shown.
#![cfg(target_os = "linux")]

use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Arc;

use image::{Rgb, RgbImage};
use shiso_core::Backend;
use shiso_core::media::{MediaModel, PadTokens, VisionPrefill};
use shiso_core::model::Model;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_engine::EngineOptions;
use shiso_models::qwen35::Qwen35Tower;
use shiso_models::qwen35moe::{Qwen35MoeConfig, Qwen35MoeModel};
use shiso_vision::{ImageLimits, VisionTower};

/// The independent reference the tower is graded against.
///
/// Captured by `scripts/qwen35moe_vision_reference.py`, which is
/// `transformers` 5.16.1's own `Qwen3_5MoeVisionModel` run over **the patch
/// rows this engine fed its own tower**, read back out of the dump the case
/// below writes. The input is read rather than re-derived so that
/// preprocessing is not a variable: a reference that preprocessed the same PNG
/// itself would be grading two resizes as well as two towers.
///
/// The script permutes those rows into the reference's own conventions first,
/// and both permutations are real differences neither tree is wrong about:
/// this one emits a patch row channel-FASTEST and walks the patch grid in
/// raster order, and `transformers` emits channel-slowest and walks it
/// merge-block-major.
#[derive(Debug, serde::Deserialize)]
struct TowerReference {
    /// Merged rows the picture encodes to.
    rows: usize,
    /// Width of one of them, which is the language model's residual width.
    out_hidden: u32,
    /// Sampled values of the projected block, as `(index, value)`.
    samples: Vec<(usize, f32)>,
}

/// What one projected value may diverge from the reference by.
///
/// Measured at **4.9e-5** over the whole block, which is 0.0027% of its
/// largest value; the budget is 1e-3, which leaves room for a different
/// reduction order without leaving room for a tower that is wrong.
const TOWER_BUDGET: f32 = 1e-3;

/// Edge of the pictures, in pixels.
///
/// A whole number of merge blocks at this tower's geometry - patch 16, merge
/// 2, so 32 pixels a block - which keeps the resize a no-op and the
/// placeholder run exactly the grid the encoder produces.
const EDGE: u32 = 224;

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// Prints a measurement.
#[expect(clippy::print_stdout, reason = "the measurement is this case's output")]
fn report(what: &str) {
    println!("{what}");
}

/// The checkpoint, or `None` when it is not on this machine.
fn snapshot() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_QWEN35MOE").ok()?);
    path.join("model.safetensors.index.json")
        .is_file()
        .then_some(path)
}

/// Opens device 0, or says on stderr why the case did not run.
#[expect(
    clippy::print_stderr,
    reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn open() -> Option<Arc<dyn Backend>> {
    match CudaBackend::new(0, CudaOptions::default()) {
        Ok(cuda) => Some(cuda),
        Err(err) => {
            eprintln!("note: no CUDA device on this machine, skipping: {err}");
            None
        }
    }
}

/// Four quadrants in four saturated colours.
///
/// Structured rather than flat: this one differs patch by patch in both axes,
/// which is what makes the tower's two-dimensional rotation observable at all.
fn quadrants() -> RgbImage {
    RgbImage::from_fn(EDGE, EDGE, |x, y| {
        let half = EDGE / 2;
        match (x < half, y < half) {
            (true, true) => Rgb([220, 30, 30]),
            (false, true) => Rgb([30, 220, 30]),
            (true, false) => Rgb([30, 30, 220]),
            (false, false) => Rgb([240, 230, 40]),
        }
    })
}

/// A fine black-and-white checkerboard, at the SAME pixel dimensions.
///
/// Identical dimensions is the whole point: the two render to the same tokens,
/// so a difference in the answer can only have come through the pixels.
fn checkerboard() -> RgbImage {
    RgbImage::from_fn(EDGE, EDGE, |x, y| {
        if (x / 16 + y / 16).is_multiple_of(2) {
            Rgb([250, 250, 250])
        } else {
            Rgb([5, 5, 5])
        }
    })
}

/// One picture as PNG bytes.
fn png(picture: &RgbImage) -> Vec<u8> {
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(picture.clone())
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .expect("the picture encodes");
    bytes
}

/// Everything about the prompt that does not change between pictures.
///
/// A struct rather than five more arguments, because they travel together and
/// a caller that transposed the two token ids would render a prompt this
/// family was never trained on and get a fluent answer.
struct Prompt<'a> {
    /// The template around the placeholder run; `u32::MAX` is the run.
    tokens: &'a [u32],
    /// The placeholder id the run is written with.
    pad_id: u32,
    /// The pair the vocabulary declares.
    pads: PadTokens,
    /// The language model's residual width, which the merger projects to.
    hidden: u32,
}

/// Compares the tower's projected block against the reference's record of it.
///
/// **This is the grading**, and it is what the two-picture case below cannot
/// do: a tower loaded from the wrong bytes, or with a weight transposed, still
/// answers two pictures differently. Only a reference says the rows are right.
#[expect(clippy::print_stdout, reason = "the measurement is this case's output")]
fn grade_tower(backend: &Arc<dyn Backend>, tower: &Qwen35Tower) {
    let reference: TowerReference = serde_json::from_str(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../test-vectors/vision/qwen35moe-vision-tower.json"),
        )
        .expect("the reference rows"),
    )
    .expect("the reference parses");

    let item = shiso_vision::preprocess_image(
        &png(&quadrants()),
        tower.preprocess(),
        ImageLimits::default(),
    )
    .expect("the picture preprocesses");
    let stages = tower
        .encode_stages(backend, &item)
        .expect("the tower encodes it");
    let mut host = vec![0u8; usize::try_from(stages.projected.len()).expect("fits")];
    backend
        .copy_d2h(stages.projected.view(), &mut host)
        .expect("the projected rows read back");
    let projected: Vec<f32> = host
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect();

    assert_eq!(
        projected.len(),
        reference.rows * reference.out_hidden as usize,
        "the projected block is not the reference's shape"
    );
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (at, want) in &reference.samples {
        let got = projected[*at];
        if (got - want).abs() > worst {
            worst = (got - want).abs();
            worst_at = *at;
        }
    }
    println!(
        "vision tower: {} samples over {} rows, worst |delta| {worst:e} at {worst_at}, \
         budget {TOWER_BUDGET:e}",
        reference.samples.len(),
        reference.rows,
    );
    assert!(
        worst <= TOWER_BUDGET,
        "value {worst_at} was {} against the reference's {}",
        projected[worst_at],
        reference
            .samples
            .iter()
            .find(|(at, _)| *at == worst_at)
            .map_or(f32::NAN, |(_, v)| *v),
    );
}

/// The last row's logits for one picture through one prompt.
fn answer(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen35MoeModel,
    tower: &Qwen35Tower,
    picture: &RgbImage,
    prompt: &Prompt<'_>,
) -> (Vec<f32>, u32) {
    let item =
        shiso_vision::preprocess_image(&png(picture), tower.preprocess(), ImageLimits::default())
            .expect("the picture preprocesses");
    let encoded = tower
        .encode(backend, std::slice::from_ref(&item))
        .expect("the tower encodes it");
    let rows: u32 = encoded
        .grids
        .iter()
        .map(shiso_core::VisionGrid::pad_count)
        .sum();

    // The prompt is built around the run the encoder actually produced, so the
    // placeholder count and the row count agree by construction - which is the
    // mismatch `VisionPrefill::new` exists to refuse and the one a caller maps
    // to a bad request.
    let mut tokens: Vec<shiso_core::TokenId> = Vec::new();
    for id in prompt.tokens {
        if *id == u32::MAX {
            tokens.extend(std::iter::repeat_n(
                shiso_core::TokenId::new(prompt.pad_id),
                rows as usize,
            ));
        } else {
            tokens.push(shiso_core::TokenId::new(*id));
        }
    }

    let media = VisionPrefill::new(
        encoded.rows.view(),
        &encoded.grids,
        prompt.pads,
        prompt.hidden,
    )
    .expect("the block matches the run");
    let session = model.open_session().expect("a session opens");
    let logits = model
        .prefill_media(session, &tokens, &media)
        .expect("the picture prefills");
    let view = logits.row(0).expect("one logits row");
    let mut host = vec![0u8; usize::try_from(view.len()).expect("one row fits")];
    backend
        .copy_d2h(view, &mut host)
        .expect("the row reads back");
    let row: Vec<f32> = host
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect();
    model.close_session(session).expect("the session closes");
    (row, rows)
}

#[test]
#[ignore = "requires a CUDA device and the 23.4 GB checkpoint"]
fn two_pictures_through_one_prompt_answer_differently() {
    let Some(dir) = snapshot() else {
        skip("SHISO_TEST_MODEL_QWEN35MOE names no checkpoint directory");
        return;
    };
    let Some(backend) = open() else { return };

    // The tower rides in this checkpoint's OWN shard set rather than in a
    // shard of its own, which is the container arm this step added; the tower
    // itself is the dense family's, unchanged.
    let tower = Qwen35Tower::open(&backend, &dir).expect("the tower loads");
    grade_tower(&backend, &tower);
    let (depth, hidden, out_hidden) = tower.geometry();
    report(&format!(
        "tower: {depth} blocks of width {hidden} projecting to {out_hidden}, \
         {} MiB resident",
        tower.device_bytes() / (1024 * 1024)
    ));

    let config = Qwen35MoeConfig::from_dir(&dir).expect("the two config documents");
    assert_eq!(
        out_hidden,
        config.shape().hidden,
        "the tower's merger projects to the language model's residual width"
    );
    let mut model = Qwen35MoeModel::load(&backend, &dir, &config, EngineOptions::default())
        .expect("the checkpoint loads onto the device");

    // The four media ids this checkpoint's config declares.
    let (image_pad, video_pad) = (248_056u32, 248_057u32);
    let (vision_start, vision_end) = (248_053u32, 248_054u32);
    let pads = PadTokens::new(
        Some(shiso_core::TokenId::new(image_pad)),
        Some(shiso_core::TokenId::new(video_pad)),
    );

    // `<|vision_start|> <run> <|vision_end|>` then a question. `u32::MAX` is
    // the marker the builder expands into the placeholder run.
    let template = [vision_start, u32::MAX, vision_end, 3838, 374, 419, 30];
    let prompt = Prompt {
        tokens: &template,
        pad_id: image_pad,
        pads,
        hidden: config.shape().hidden,
    };

    let (quads, rows) = answer(&backend, &mut model, &tower, &quadrants(), &prompt);
    let (check, rows2) = answer(&backend, &mut model, &tower, &checkerboard(), &prompt);
    assert_eq!(rows, rows2, "the two pictures render to different runs");
    report(&format!(
        "each picture is {rows} placeholder rows; the prompt is {} tokens",
        rows as usize + template.len() - 1
    ));

    for (name, row) in [("quadrants", &quads), ("checkerboard", &check)] {
        let finite = row.iter().filter(|v| v.is_finite()).count();
        assert_eq!(
            finite,
            row.len(),
            "{name}: {finite} of {} finite",
            row.len()
        );
    }

    // THE ASSERTION. Same prompt, same token sequence, same placeholder run -
    // so anything that differs came through the pixels. A splice that never
    // ran, or that wrote where the attention does not read, gives two
    // identical rows here and a fluent answer to both.
    let worst = quads
        .iter()
        .zip(&check)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i)
    };
    report(&format!(
        "quadrants argmax {}, checkerboard argmax {}, worst |delta| {worst:.4} of a logit",
        argmax(&quads),
        argmax(&check),
    ));
    assert!(
        worst > 0.1,
        "the two pictures produced the same logits to {worst}: the tower's rows \
         are not reaching the residual stream"
    );

    model.teardown().expect("the device memory is released");
}

/// Dumps the tower's input and output for the reference to grade, under
/// `SHISO_VISION_DUMP=<dir>`.
///
/// **The input is dumped, not re-derived**, and that is the point: the
/// reference then runs over the patch rows this engine actually fed its tower,
/// so preprocessing is not a variable in the comparison. A reference that
/// preprocessed the same PNG itself would be grading two resizes as well as
/// two towers, and a disagreement would name neither.
///
/// Off unless the variable is set. `scripts/qwen35moe_vision_reference.py`
/// reads what it writes.
#[test]
#[ignore = "requires a CUDA device, the checkpoint, and SHISO_VISION_DUMP"]
fn the_tower_s_input_and_output_are_dumped_for_the_reference() {
    let Ok(out) = std::env::var("SHISO_VISION_DUMP") else {
        skip("SHISO_VISION_DUMP names no directory");
        return;
    };
    let Some(dir) = snapshot() else {
        skip("SHISO_TEST_MODEL_QWEN35MOE names no checkpoint directory");
        return;
    };
    let Some(backend) = open() else { return };
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).expect("the dump directory");

    let tower = Qwen35Tower::open(&backend, &dir).expect("the tower loads");
    let item = shiso_vision::preprocess_image(
        &png(&quadrants()),
        tower.preprocess(),
        ImageLimits::default(),
    )
    .expect("the picture preprocesses");
    let stages = tower
        .encode_stages(&backend, &item)
        .expect("the tower encodes");

    let bytes: Vec<u8> = item
        .as_flat()
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(out.join("pixel_values.bin"), &bytes).expect("the patch rows");
    let mut projected = vec![0u8; usize::try_from(stages.projected.len()).expect("fits")];
    backend
        .copy_d2h(stages.projected.view(), &mut projected)
        .expect("the projected rows read back");
    std::fs::write(out.join("projected.bin"), &projected).expect("the projected rows");
    let (t, h, w) = item.grid();
    std::fs::write(
        out.join("manifest.json"),
        format!(
            "{{\"rows\": {}, \"row_len\": {}, \"grid\": [{t}, {h}, {w}], \"projected\": {}}}\n",
            item.rows(),
            item.row_len(),
            projected.len() / 4,
        ),
    )
    .expect("the manifest");
    report(&format!(
        "dumped {} patch rows of {} and {} projected floats to {}",
        item.rows(),
        item.row_len(),
        projected.len() / 4,
        out.display()
    ));
}
