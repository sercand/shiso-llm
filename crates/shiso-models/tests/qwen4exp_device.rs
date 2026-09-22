//! The runbook for the box that has a checkpoint and a device.
//!
//! Nothing here runs on the machine this was written on: there is no checkpoint
//! of this family on it and no device to load one onto. Every test is therefore
//! `#[ignore]` and compiled only where the device exists, and each one is
//! written as an instruction rather than as a claim - the claim is what the
//! Linux box gets to make.
//!
//! # What these can settle and what they cannot
//!
//! They can settle everything up to the first launch: that a real container's
//! metadata agrees with this crate's shape table key for key, that every tensor
//! the binding asks for is present at the shape it asks for, that the weights
//! reach the device, and that a chunk plans - every projection finding a tier,
//! every launch a grid. They cannot settle a number, because the pass cannot be
//! issued yet; parity against a reference run is the job of whoever closes the
//! two seams named in this crate's model documentation.
//!
//! Run them with the checkpoint named:
//!
//! ```text
//! SHISO_TEST_MODEL=/path/to/checkpoint.gguf \
//!   cargo test -p shiso-models --release --test qwen4exp_device -- --ignored --nocapture
//! ```

#![cfg(target_os = "linux")]

use std::path::PathBuf;
use std::sync::Arc;

use shiso_core::{Backend, Model};
use shiso_engine::sparse::stack::SparseOptions;
use shiso_modelfile::GgufFile;
use shiso_models::qwen4exp::{Qwen4ExpConfig, Qwen4ExpModel, bind_qwen4exp};

/// The checkpoint under test, or `None` when this box has none.
fn model_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL").ok()?);
    path.is_file().then_some(path)
}

#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

#[test]
#[ignore = "needs a checkpoint of this family, which no development machine carries"]
fn a_real_checkpoint_s_metadata_agrees_with_this_crate_s_shape_table() {
    let Some(path) = model_path() else {
        skip("no checkpoint named by SHISO_TEST_MODEL");
        return;
    };
    let file = GgufFile::open(&path).expect("the checkpoint opens");
    let config = Qwen4ExpConfig::from_metadata(file.metadata())
        .expect("every dimension in the container agrees with the table");
    assert_eq!(config.shape(), &shiso_models::qwen4exp::shape());
}

#[test]
#[ignore = "needs a checkpoint of this family, which no development machine carries"]
fn every_tensor_the_pass_reads_is_in_a_real_checkpoint() {
    let Some(path) = model_path() else {
        skip("no checkpoint named by SHISO_TEST_MODEL");
        return;
    };
    let file = GgufFile::open_shared(&path).expect("the checkpoint opens");
    let config = Qwen4ExpConfig::from_metadata(file.metadata()).expect("the metadata agrees");
    let binding = bind_qwen4exp(&file, &config).expect("every role resolves");
    assert_eq!(
        binding.layers.len(),
        shiso_models::qwen4exp::N_LAYERS as usize
    );
    let table = binding
        .ngram_table
        .as_ref()
        .expect("the streamed n-gram table is part of a complete checkpoint");
    // The trellis row codec finishes every weight with a per-head bias the
    // decode kernel takes as a pointer, so a checkpoint that stores its table
    // that way has to carry one. This is the first thing a box with a real
    // checkpoint can settle about the streaming tier, and the load refuses
    // without it.
    if table.dtype == shiso_core::DType::Exl3Ngram6 {
        assert!(
            binding.ngram_bias.is_some(),
            "a trellis-coded n-gram table carries its per-head bias"
        );
    }
}

#[test]
#[ignore = "needs a device and a checkpoint of this family"]
fn a_real_checkpoint_loads_onto_the_device_and_a_chunk_plans() {
    let Some(path) = model_path() else {
        skip("no checkpoint named by SHISO_TEST_MODEL");
        return;
    };
    let file = GgufFile::open(&path).expect("the checkpoint opens");
    let config = Qwen4ExpConfig::from_metadata(file.metadata()).expect("the metadata agrees");
    drop(file);

    // The device is whichever backend this box was built with. A backend crate
    // is not named here on purpose: this crate depends on none of them, and the
    // one line below is the only thing a box with a different device has to
    // change.
    let Some(backend) = device() else {
        skip("no device on this box");
        return;
    };
    let mut model = Qwen4ExpModel::load(&backend, &path, &config, SparseOptions::default())
        .expect("the checkpoint loads");
    let session = model.open_session().expect("a session opens");
    let tokens: Vec<_> = (0..8).map(shiso_core::TokenId::new).collect();
    let pass = model
        .plan_chunk(session, &tokens, 1)
        .expect("an eight-token chunk plans");
    assert_eq!(
        pass.islands.len(),
        2 * shiso_models::qwen4exp::N_LAYERS as usize,
        "every layer is captured as two islands"
    );
    pass.plan.validate().expect("no launch has an empty grid");
    model.teardown().expect("the model tears down");
}

/// The device this box runs on.
///
/// `None` here rather than a dependency on a backend crate: adding one would
/// make this crate build-depend on a device API it names nowhere else, and the
/// whole point of the `Backend` seam is that it does not.
fn device() -> Option<Arc<dyn Backend>> {
    None
}
