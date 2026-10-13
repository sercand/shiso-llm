//! `--profile-dir` through the engine thread: one report per request, named
//! after the id the request was given, and none left open between requests.

mod common;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt as _;
use shiso_core::{Backend, SamplingParams, Seed, TokenId};
use shiso_ir::StreamEvent;
use shiso_server::profile::ProfileSink;
use shiso_server::scheduler::{GenerateRequest, SchedulerConfig};

use common::FakeBackend;

fn request(max_tokens: u32) -> GenerateRequest {
    GenerateRequest::new(
        Arc::from(vec![TokenId::new(1)]),
        SamplingParams {
            temperature: 0.0,
            ..SamplingParams::neutral()
        },
        Seed::new(7),
        max_tokens,
    )
}

fn profiled_cfg(dir: &Path) -> SchedulerConfig {
    let device: Arc<dyn Backend> = FakeBackend::profiling();
    SchedulerConfig {
        max_num_seqs: 1,
        profile: Some(ProfileSink::new(dir, device).expect("the fake profiles")),
        ..SchedulerConfig::new()
    }
}

/// The id a turn was given, from its first event.
async fn run_turn(engine: &shiso_server::scheduler::EngineHandle) -> String {
    let mut stream = engine.generate(request(3)).await.expect("accepted");
    let mut id = None;
    while let Some(event) = stream.next().await {
        if let StreamEvent::Start { id: given, .. } = event {
            id = Some(given);
        }
    }
    id.expect("a turn opens with its id")
}

/// The report files in `dir`, waiting briefly for the engine thread to write
/// them: it settles a region once nothing is resident, just after the stream
/// ends.
async fn reports(dir: &Path, want: usize) -> Vec<String> {
    for _ in 0..500 {
        let names: Vec<String> = std::fs::read_dir(dir)
            .expect("the profile dir exists")
            .map(|entry| {
                entry
                    .expect("readable")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        if names.len() >= want {
            return names;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    panic!("fewer than {want} reports appeared in {}", dir.display());
}

#[tokio::test]
async fn each_request_writes_one_report_named_after_its_id() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let (engine, _join, _audit) =
        common::scheduled_engine(vec![10, 11, 12, 13, 14, 15], profiled_cfg(dir.path()), None);

    let first = run_turn(&engine).await;
    let second = run_turn(&engine).await;

    let mut names = reports(dir.path(), 2).await;
    names.sort();
    assert_eq!(names.len(), 2, "{names:?}");
    for id in [&first, &second] {
        assert!(
            names
                .iter()
                .any(|name| name.starts_with("engine-") && name.ends_with(&format!("-{id}.json"))),
            "no report for {id}: {names:?}"
        );
    }
    let body: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join(&names[0])).expect("readable"))
            .expect("a report is JSON");
    assert_eq!(body["kind"], "engine");
    assert_eq!(body["report"]["schema"], 1);
}

#[tokio::test]
async fn a_request_whose_caller_left_does_not_absorb_the_next_report() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let (engine, _join, _audit) =
        common::scheduled_engine(vec![10, 11, 12, 13, 14, 15], profiled_cfg(dir.path()), None);

    drop(engine.generate(request(3)).await.expect("accepted"));
    let id = run_turn(&engine).await;

    let names = reports(dir.path(), 1).await;
    assert!(
        names
            .iter()
            .any(|name| name.ends_with(&format!("-{id}.json"))),
        "the served request has its own report: {names:?}"
    );
}
