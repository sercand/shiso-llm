//! A state-space block's per-row trail: the state a verify pass leaves after
//! each of its rows, so a rejected draft can be wound back.
//!
//! GPU case: `cargo test -p shiso-engine --test ssm_trail -- --ignored`.
#![cfg(target_os = "macos")]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "fixture widths are small"
)]

use std::sync::Arc;

use shiso_core::{Backend, DType, DeviceBuffer, alloc};
use shiso_engine::blocks::{GdnIo, GdnState, GdnStates, StateTrail};
use shiso_engine::projection::Weight;
use shiso_engine::shape::SsmShape;
use shiso_engine::ssm::{self, Ssm, SsmGeometry, SsmScratch};

const HIDDEN: usize = 32;
const ROWS: u32 = 4;

fn signal(len: usize, seed: u32, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            ((((i as u32).wrapping_mul(2_654_435_761) ^ seed) % 1000) as f32 / 500.0 - 1.0) * scale
        })
        .collect()
}

fn upload(backend: &Arc<dyn Backend>, v: &[f32]) -> DeviceBuffer {
    let buf = alloc(backend, (v.len() * 4) as u64).expect("alloc");
    backend
        .copy_h2d(buf.view(), bytemuck::cast_slice(v))
        .expect("h2d");
    buf
}

fn read(backend: &Arc<dyn Backend>, v: shiso_core::DeviceView<'_>) -> Vec<u32> {
    let mut raw = vec![0u8; v.len() as usize];
    backend.copy_d2h(v, &mut raw).expect("d2h");
    raw.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
#[ignore = "requires a Metal device"]
#[expect(
    clippy::print_stderr,
    reason = "a machine with no Metal device says why it skipped"
)]
fn ssm_trail_rows_equal_sequential_states() {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping");
        return;
    }
    let backend: Arc<dyn Backend> =
        shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");
    let shape = SsmShape {
        heads: 4,
        head_dim: 8,
        state: 32,
        groups: 2,
        conv_kernel: 4,
    };
    let geometry = SsmGeometry::new(shape, 1e-5).expect("geometry");
    let w = |seed: u32, inp: usize, out: usize| {
        Weight::from_buffer(
            DType::F32,
            inp as u32,
            out as u32,
            upload(&backend, &signal(inp * out, seed, 0.2)),
        )
    };
    let channels = shape.conv_channels() as usize;
    let heads = shape.heads as usize;
    let ssm = Ssm {
        in_proj: w(1, HIDDEN, shape.in_width() as usize),
        conv_weight: upload(&backend, &signal(channels * 4, 2, 0.3)),
        conv_bias: upload(&backend, &signal(channels, 3, 0.1)),
        dt_bias: upload(&backend, &signal(heads, 4, 0.5)),
        a_log: upload(&backend, &signal(heads, 5, 0.5)),
        d: upload(&backend, &signal(heads, 6, 1.0)),
        norm: upload(&backend, &signal(shape.inner() as usize, 7, 1.0)),
        out_proj: w(8, shape.inner() as usize, HIDDEN),
    };
    let scratch = SsmScratch::new(&backend, &geometry, ROWS).expect("scratch");
    let conv_floats = (shape.conv_kernel as usize - 1) * channels;
    let state_floats = heads * shape.head_dim as usize * shape.state as usize;
    let input = upload(&backend, &signal(ROWS as usize * HIDDEN, 9, 1.0));
    let out = alloc(&backend, u64::from(ROWS) * HIDDEN as u64 * 4).expect("alloc");
    let fresh = |n: usize| upload(&backend, &vec![0.0; n]);

    // One verify pass of four rows, recording a trail.
    let (conv_a, state_a) = (fresh(conv_floats), fresh(state_floats));
    let (trail_conv, trail_state) = (fresh(4 * conv_floats), fresh(4 * state_floats));
    ssm::run(
        backend.as_ref(),
        &ssm,
        &geometry,
        &scratch,
        GdnIo {
            input: input.view(),
            states: GdnStates::Chunk(GdnState {
                conv: conv_a.view(),
                state: state_a.view(),
            }),
            block_out: out.view(),
            trail: Some(StateTrail {
                conv: trail_conv.view(),
                state: trail_state.view(),
                rows: ROWS,
            }),
        },
        ROWS,
    )
    .expect("a recording pass");

    // The same rows one at a time; the state after each is a trail record.
    let (conv_b, state_b) = (fresh(conv_floats), fresh(state_floats));
    let row_bytes = HIDDEN as u64 * 4;
    for r in 0..u64::from(ROWS) {
        ssm::run(
            backend.as_ref(),
            &ssm,
            &geometry,
            &scratch,
            GdnIo {
                input: input.slice(r * row_bytes, row_bytes).expect("row"),
                states: GdnStates::Chunk(GdnState {
                    conv: conv_b.view(),
                    state: state_b.view(),
                }),
                block_out: out.slice(r * row_bytes, row_bytes).expect("row"),
                trail: None,
            },
            1,
        )
        .expect("a one-row pass");
        backend.sync().expect("sync");
        let (cb, sb) = ((conv_floats * 4) as u64, (state_floats * 4) as u64);
        assert_eq!(
            read(&backend, conv_b.view()),
            read(&backend, trail_conv.slice(r * cb, cb).expect("record")),
            "convolution record {r}"
        );
        assert_eq!(
            read(&backend, state_b.view()),
            read(&backend, trail_state.slice(r * sb, sb).expect("record")),
            "state record {r}"
        );
    }
}
