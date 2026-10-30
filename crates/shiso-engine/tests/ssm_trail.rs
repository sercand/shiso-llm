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

use shiso_core::{Backend, CoreError, CoreResult, DType, DeviceBuffer, alloc};
use shiso_engine::blocks::{GdnIo, GdnState, GdnStates, StateTrail};
use shiso_engine::projection::Weight;
use shiso_engine::shape::SsmShape;
use shiso_engine::ssm::{self, Ssm, SsmGeometry, SsmScratch};

const HIDDEN: usize = 32;
const ROWS: u32 = 4;
/// Rows the scratch and the input hold: one past a trail of [`ROWS`], the
/// widest pass such a trail is sized for.
const CAPACITY: u32 = ROWS + 1;

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

/// One block, its scratch and `CAPACITY` rows of input, on the Metal device.
struct Fixture {
    backend: Arc<dyn Backend>,
    geometry: SsmGeometry,
    ssm: Ssm,
    scratch: SsmScratch,
    conv_floats: usize,
    state_floats: usize,
    input: DeviceBuffer,
    out: DeviceBuffer,
}

impl Fixture {
    /// `None` on a machine with no Metal device, which says so on stderr.
    #[expect(
        clippy::print_stderr,
        reason = "a machine with no Metal device says why it skipped"
    )]
    fn open() -> Option<Self> {
        if !shiso_metal::metal_available() {
            eprintln!("note: no Metal device on this machine, skipping");
            return None;
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
        let scratch = SsmScratch::new(&backend, &geometry, CAPACITY).expect("scratch");
        let input = upload(&backend, &signal(CAPACITY as usize * HIDDEN, 9, 1.0));
        let out = alloc(&backend, u64::from(CAPACITY) * HIDDEN as u64 * 4).expect("alloc");
        Some(Self {
            conv_floats: (shape.conv_kernel as usize - 1) * channels,
            state_floats: heads * shape.head_dim as usize * shape.state as usize,
            backend,
            geometry,
            ssm,
            scratch,
            input,
            out,
        })
    }

    fn zeros(&self, n: usize) -> DeviceBuffer {
        upload(&self.backend, &vec![0.0; n])
    }

    /// A fresh, zeroed convolution window and state.
    fn state(&self) -> (DeviceBuffer, DeviceBuffer) {
        (self.zeros(self.conv_floats), self.zeros(self.state_floats))
    }

    /// A zeroed trail of `rows` records.
    fn trail(&self, rows: u32) -> (DeviceBuffer, DeviceBuffer) {
        (
            self.zeros(rows as usize * self.conv_floats),
            self.zeros(rows as usize * self.state_floats),
        )
    }

    fn run(
        &self,
        states: GdnStates<'_>,
        trail: Option<StateTrail<'_>>,
        n_tok: u32,
    ) -> CoreResult<()> {
        ssm::run(
            self.backend.as_ref(),
            &self.ssm,
            &self.geometry,
            &self.scratch,
            GdnIo {
                input: self.input.view(),
                states,
                block_out: self.out.view(),
                trail,
            },
            n_tok,
        )
    }

    /// Runs the first `rows` input rows one at a time into a fresh state and
    /// checks the state after each of the first `records` against the trail.
    fn assert_trail(
        &self,
        trail: (&DeviceBuffer, &DeviceBuffer),
        rows: u32,
        records: u32,
    ) -> (DeviceBuffer, DeviceBuffer) {
        let (conv_b, state_b) = self.state();
        let row_bytes = HIDDEN as u64 * 4;
        let (cb, sb) = (
            (self.conv_floats * 4) as u64,
            (self.state_floats * 4) as u64,
        );
        for r in 0..u64::from(rows) {
            ssm::run(
                self.backend.as_ref(),
                &self.ssm,
                &self.geometry,
                &self.scratch,
                GdnIo {
                    input: self.input.slice(r * row_bytes, row_bytes).expect("row"),
                    states: GdnStates::Chunk(GdnState {
                        conv: conv_b.view(),
                        state: state_b.view(),
                    }),
                    block_out: self.out.slice(r * row_bytes, row_bytes).expect("row"),
                    trail: None,
                },
                1,
            )
            .expect("a one-row pass");
            self.backend.sync().expect("sync");
            if r >= u64::from(records) {
                continue;
            }
            assert_eq!(
                read(&self.backend, conv_b.view()),
                read(&self.backend, trail.0.slice(r * cb, cb).expect("record")),
                "convolution record {r}"
            );
            assert_eq!(
                read(&self.backend, state_b.view()),
                read(&self.backend, trail.1.slice(r * sb, sb).expect("record")),
                "state record {r}"
            );
        }
        (conv_b, state_b)
    }
}

#[test]
#[ignore = "requires a Metal device"]
fn ssm_trail_rows_equal_sequential_states() {
    let Some(fx) = Fixture::open() else { return };
    // One verify pass of four rows, recording a trail.
    let (conv_a, state_a) = fx.state();
    let (trail_conv, trail_state) = fx.trail(ROWS);
    fx.run(
        GdnStates::Chunk(GdnState {
            conv: conv_a.view(),
            state: state_a.view(),
        }),
        Some(StateTrail {
            conv: trail_conv.view(),
            state: trail_state.view(),
            rows: ROWS,
        }),
        ROWS,
    )
    .expect("a recording pass");
    // The same rows one at a time; the state after each is a trail record.
    fx.assert_trail((&trail_conv, &trail_state), ROWS, ROWS);
}

#[test]
#[ignore = "requires a Metal device"]
fn ssm_trail_one_row_past_its_records_is_the_widest_pass() {
    let Some(fx) = Fixture::open() else { return };
    // `rows + 1` rows: the last is the one a verify pass either rejects or
    // keeps, so it is never recorded, and every earlier row is.
    let (conv_a, state_a) = fx.state();
    let (trail_conv, trail_state) = fx.trail(ROWS);
    fx.run(
        GdnStates::Chunk(GdnState {
            conv: conv_a.view(),
            state: state_a.view(),
        }),
        Some(StateTrail {
            conv: trail_conv.view(),
            state: trail_state.view(),
            rows: ROWS,
        }),
        CAPACITY,
    )
    .expect("a pass one row wider than its trail");
    fx.backend.sync().expect("sync");
    let (conv_b, state_b) = fx.assert_trail((&trail_conv, &trail_state), CAPACITY, ROWS);
    // The unrecorded last row still advances the live state.
    assert_eq!(
        read(&fx.backend, conv_a.view()),
        read(&fx.backend, conv_b.view())
    );
    assert_eq!(
        read(&fx.backend, state_a.view()),
        read(&fx.backend, state_b.view())
    );
}

#[test]
#[ignore = "requires a Metal device"]
fn ssm_trail_refuses_a_pass_wider_than_it_records() {
    let Some(fx) = Fixture::open() else { return };
    let (conv_a, state_a) = fx.state();
    let (trail_conv, trail_state) = fx.trail(ROWS - 1);
    let err = fx
        .run(
            GdnStates::Chunk(GdnState {
                conv: conv_a.view(),
                state: state_a.view(),
            }),
            Some(StateTrail {
                conv: trail_conv.view(),
                state: trail_state.view(),
                rows: ROWS - 1,
            }),
            CAPACITY,
        )
        .expect_err("five rows cannot be recorded in a trail of three");
    assert!(matches!(err, CoreError::OutOfBounds { .. }), "{err:?}");
    // Refused before any launch: the state is still the zeros it started as.
    fx.backend.sync().expect("sync");
    assert!(read(&fx.backend, state_a.view()).iter().all(|&w| w == 0));
    assert!(read(&fx.backend, conv_a.view()).iter().all(|&w| w == 0));
}

#[test]
#[ignore = "requires a Metal device"]
fn ssm_trail_refuses_a_batch_of_sessions() {
    let Some(fx) = Fixture::open() else { return };
    let sessions: Vec<_> = (0..ROWS).map(|_| fx.state()).collect();
    let states: Vec<GdnState<'_>> = sessions
        .iter()
        .map(|(conv, state)| GdnState {
            conv: conv.view(),
            state: state.view(),
        })
        .collect();
    let (trail_conv, trail_state) = fx.trail(ROWS);
    let err = fx
        .run(
            GdnStates::PerRow(&states),
            Some(StateTrail {
                conv: trail_conv.view(),
                state: trail_state.view(),
                rows: ROWS,
            }),
            ROWS,
        )
        .expect_err("a batch's rows are not one sequence to wind back along");
    assert!(matches!(err, CoreError::Unsupported { .. }), "{err:?}");
}
