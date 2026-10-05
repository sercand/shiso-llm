//! The step lists of the passes a standard request runs, as one text.
//!
//! Scoring work may teach the stack new passes; it must not move a step of
//! the ones chat already runs. This renders those passes on the synthetic
//! fixture - a solo prompt chunk at sequential and chunked widths, the
//! step-role chunk, decode batches of one to the widest, and a speculative
//! verify - one step per line in its `Debug` form, which names the kernel or
//! multiply and carries its grid, shared memory and operand list. The test
//! that includes this compares the text against a snapshot recorded on the
//! commit the scoring work started from (8662c67a), in
//! `tests/data/qwen4exp_standard_plans.snap`.
//!
//! Only API that commit already had is used here, so the same file records
//! there and checks here.

use std::fmt::Write as _;

use shiso_core::Model;
use shiso_core::ids::SessionId;
use shiso_engine::sparse::bind::Args;
use shiso_engine::sparse::kernels::Launch;
use shiso_engine::sparse::plan::Step;
use shiso_engine::sparse::stack::Pass;

use crate::fixture::{loaded, loaded_with, speculative_options, tokens};

/// Appends one pass: a header, its islands, then every step.
fn render(out: &mut String, title: &str, pass: &Pass) {
    let _ = writeln!(
        out,
        "== {title}: rows {} capturable {} selected {} steps {}",
        pass.rows,
        pass.capturable,
        pass.selected,
        pass.plan.steps().len()
    );
    for span in &pass.islands {
        let _ = writeln!(
            out,
            "island {}+{} {:?}",
            span.start, span.len, span.key
        );
    }
    for step in pass.plan.steps() {
        let _ = writeln!(out, "{}", line(step));
    }
}

/// A launch's grid, block and shared memory.
fn launch(launch: &Launch) -> String {
    let Launch { grid, block, smem } = launch;
    format!(
        "grid {}x{}x{} block {}x{}x{} smem {smem}",
        grid.x, grid.y, grid.z, block.x, block.y, block.z
    )
}

/// One step: everything its `Debug` form carries, with the argument list cut
/// to the operands the step passes rather than the fixed-width array that
/// holds them.
fn line(step: &Step) -> String {
    let ops = |args: &Args| format!("{:?}", args.operands().expect("the list fits a launch"));
    match step {
        Step::Kernel {
            entry,
            launch: at,
            args,
        } => format!("kernel {entry:?} {} {}", launch(at), ops(args)),
        Step::Trellis {
            op,
            entry,
            shape,
            bits,
            blocks,
            groups,
            block_dim,
            smem,
            staging,
            args,
        } => format!(
            "trellis {op:?} {entry:?} {shape:?} bits {bits} blocks {blocks} groups {groups} \
             block {block_dim} smem {smem} {staging:?} {}",
            ops(args)
        ),
        Step::Staged {
            entry,
            launch: at,
            staging,
            args,
        } => format!("staged {entry:?} {} {staging:?} {}", launch(at), ops(args)),
        Step::Expand {
            entry,
            k,
            n,
            bits,
            launch: at,
            staging,
            args,
        } => format!(
            "expand {entry:?} {k}x{n} bits {bits} {} {staging:?} {}",
            launch(at),
            ops(args)
        ),
        other => format!("{other:?}"),
    }
}

/// Opens `rows` sessions with one cache row claimed each, as a decode batch's
/// rows have.
fn open_rows(model: &mut shiso_models::qwen4exp::Qwen4ExpModel, rows: u32) -> Vec<SessionId> {
    (0..rows)
        .map(|_| {
            let session = model.open_session().expect("a session opens");
            model
                .stack_mut()
                .reserve(session, 1)
                .expect("the cache has room for a row");
            session
        })
        .collect()
}

/// Every snapshotted pass, rendered.
pub(crate) fn snapshot() -> String {
    let mut out = String::new();

    // A solo prompt read: below the chunked kernel's threshold the recurrence
    // is the sequential one, from it the chunked one. Both a chunk that ends
    // the prompt (one logits row) and one that does not.
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    for (rows, logits) in [(1, 1), (4, 0), (4, 1), (40, 1), (127, 1), (128, 1), (256, 0)] {
        let pass = model
            .stack()
            .plan_prompt(session, &tokens(0, rows), logits)
            .expect("a prompt chunk plans");
        render(
            &mut out,
            &format!("prompt {rows} rows, {logits} logits"),
            &pass,
        );
    }
    for (rows, logits) in [(1, 1), (4, 1)] {
        let pass = model
            .plan_chunk(session, &tokens(0, rows), logits)
            .expect("a step chunk plans");
        render(&mut out, &format!("step {rows} rows, {logits} logits"), &pass);
    }
    // Past the indexer's budget, where a solo read's rows attend over a
    // selection.
    model
        .prefill_chunk(session, &tokens(0, 128))
        .expect("the opening prefills");
    let pass = model
        .stack()
        .plan_prompt(session, &tokens(128, 64), 1)
        .expect("a later prompt chunk plans");
    render(&mut out, "prompt 64 rows after 128", &pass);
    model.close_session(session).expect("closes");

    // Decode batches, one row up to the widest.
    let (_dir, _fake, mut model) = loaded();
    let sessions = open_rows(&mut model, 8);
    for width in [1_usize, 2, 3, 8] {
        let pass = model
            .stack()
            .plan_batch(&sessions[..width])
            .expect("a decode batch plans");
        render(&mut out, &format!("decode batch of {width}"), &pass);
    }

    // A speculative build: its decode batch and its segmented verify.
    let (_dir, _fake, mut model) = loaded_with(speculative_options(3));
    let a = model.open_session().expect("a session opens");
    model
        .stack_mut()
        .advance(a, &tokens(0, 9), false, None)
        .expect("the path grows");
    let b = model.open_session().expect("a session opens");
    model
        .stack_mut()
        .advance(b, &tokens(7, 21), false, None)
        .expect("the path grows");
    let pass = model.stack().plan_batch(&[a, b]).expect("plans");
    render(&mut out, "speculative decode batch of 2", &pass);
    let pass = model
        .stack()
        .plan_segments(&[a, b], &[4, 4])
        .expect("plans");
    render(&mut out, "verify of 2 runs of 4", &pass);
    let pass = model
        .stack()
        .plan_prompt(a, &tokens(9, 40), 1)
        .expect("plans");
    render(&mut out, "speculative prompt 40 rows", &pass);

    out
}
