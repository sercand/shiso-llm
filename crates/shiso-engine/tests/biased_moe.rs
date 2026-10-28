//! The biased routed feed-forward over `IQ4_NL` expert stacks, the whole block
//! at once - router, sort, both expert stages, the shared expert and the
//! combine - against the same arithmetic on the host in f64.
//!
//! GPU case: `cargo test -p shiso-engine --test biased_moe -- --ignored`.
#![cfg(target_os = "macos")]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "fixture widths and ids are small"
)]

use std::sync::Arc;

use shiso_core::{Backend, DType, DeviceBuffer, alloc};
use shiso_engine::ffn::Ffn;
use shiso_engine::moe::{MoeIo, MoeScratch};
use shiso_engine::moe_biased::{
    self, BiasedMoe, ExpertStack, RouterGuess, RowIq4NlStack, StackStore,
};
use shiso_engine::projection::Weight;
use shiso_engine::shape::MoeShape;
use shiso_engine::sparse::weights::WeightPart;
use shiso_engine::sparse::{ExpertSource, ExpertStream};
use shiso_modelfile::quant::BlockIq4Nl;
use shiso_modelfile::quant_tables::IQ4NL_VALUES;

const HIDDEN: usize = 64;
const WIDTH: usize = 64;
const SHARED: usize = 32;
const EXPERTS: usize = 8;
const USED: usize = 3;
const TOKENS: usize = 4;
const SCALE: f32 = 2.5;

fn signal(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i as u32).wrapping_mul(2_654_435_761) ^ seed) % 1000) as f32 / 500.0 - 1.0)
        .collect()
}

fn upload(backend: &Arc<dyn Backend>, bytes: &[u8]) -> DeviceBuffer {
    let buf = alloc(backend, bytes.len() as u64).expect("alloc");
    backend.copy_h2d(buf.view(), bytes).expect("h2d");
    buf
}

fn f32_weight(backend: &Arc<dyn Backend>, w: &[f32], in_dim: usize, out_dim: usize) -> Weight {
    Weight::from_buffer(
        DType::F32,
        in_dim as u32,
        out_dim as u32,
        upload(backend, bytemuck::cast_slice(w)),
    )
}

/// `[experts][out_dim][in_dim / 32]` blocks, and the same rows dequantized.
fn iq4nl_stack(in_dim: usize, out_dim: usize, seed: u32) -> (Vec<BlockIq4Nl>, Vec<f64>) {
    let n = EXPERTS * out_dim * in_dim / 32;
    let scales = signal(n, seed);
    let blocks: Vec<BlockIq4Nl> = (0..n)
        .map(|b| {
            let mut qs = [0u8; 16];
            for (j, q) in qs.iter_mut().enumerate() {
                *q = (((b * 16 + j) as u32 ^ seed).wrapping_mul(2_654_435_761) >> 13) as u8;
            }
            BlockIq4Nl {
                d: half::f16::from_f32(scales[b] * 0.004).to_bits(),
                qs,
            }
        })
        .collect();
    let dense = blocks
        .iter()
        .flat_map(|b| {
            let d = f64::from(half::f16::from_bits(b.d).to_f32());
            let mut v = [0.0f64; 32];
            for j in 0..16 {
                v[j] = d * f64::from(IQ4NL_VALUES[usize::from(b.qs[j] & 0xf)]);
                v[j + 16] = d * f64::from(IQ4NL_VALUES[usize::from(b.qs[j] >> 4)]);
            }
            v
        })
        .collect();
    (blocks, dense)
}

fn matvec(w: &[f64], x: &[f64], in_dim: usize, out_dim: usize) -> Vec<f64> {
    (0..out_dim)
        .map(|r| (0..in_dim).map(|k| w[r * in_dim + k] * x[k]).sum())
        .collect()
}

fn relu2(v: Vec<f64>) -> Vec<f64> {
    v.into_iter().map(|a| a.max(0.0) * a.max(0.0)).collect()
}

#[test]
#[ignore = "requires a Metal device"]
#[expect(
    clippy::print_stderr,
    reason = "a machine with no Metal device says why it skipped"
)]
fn biased_moe_iq4nl_matches_host_reference() {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping");
        return;
    }
    let backend: Arc<dyn Backend> =
        shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");
    let router = signal(EXPERTS * HIDDEN, 3);
    let bias: Vec<f32> = (0..EXPERTS).map(|e| e as f32 * 0.05).collect();
    let shared_up: Vec<f32> = signal(SHARED * HIDDEN, 5).iter().map(|v| v * 0.1).collect();
    let shared_down: Vec<f32> = signal(HIDDEN * SHARED, 7).iter().map(|v| v * 0.1).collect();
    let (up_blocks, up_dense) = iq4nl_stack(HIDDEN, WIDTH, 11);
    let (down_blocks, down_dense) = iq4nl_stack(WIDTH, HIDDEN, 13);
    let x = signal(TOKENS * HIDDEN, 17);

    let moe = BiasedMoe {
        router: f32_weight(&backend, &router, HIDDEN, EXPERTS),
        bias: upload(&backend, bytemuck::cast_slice(&bias)),
        scale: SCALE,
        renormalize: true,
        up: ExpertStack::Iq4Nl(RowIq4NlStack {
            store: StackStore::Resident(upload(&backend, bytemuck::cast_slice(&up_blocks))),
            in_dim: HIDDEN as u32,
            out_dim: WIDTH as u32,
        }),
        down: ExpertStack::Iq4Nl(RowIq4NlStack {
            store: StackStore::Resident(upload(&backend, bytemuck::cast_slice(&down_blocks))),
            in_dim: WIDTH as u32,
            out_dim: HIDDEN as u32,
        }),
        guess: None,
    };
    moe_biased::check_stack(&moe.up).expect("up stack");
    moe_biased::check_stack(&moe.down).expect("down stack");
    let shared = Ffn {
        gate: None,
        up: f32_weight(&backend, &shared_up, HIDDEN, SHARED),
        down: f32_weight(&backend, &shared_down, SHARED, HIDDEN),
    };
    let shape = MoeShape::new(EXPERTS as u32, USED as u32, 1, WIDTH as u32);
    let scratch = MoeScratch::new(&backend, shape, HIDDEN as u32, TOKENS as u32).expect("scratch");
    let input = upload(&backend, bytemuck::cast_slice(&x));
    let out = alloc(&backend, (TOKENS * HIDDEN * 4) as u64).expect("alloc");
    moe_biased::run(
        backend.as_ref(),
        &moe,
        &shared,
        shape,
        &scratch,
        MoeIo {
            input: input.view(),
            block_out: out.view(),
        },
        TOKENS as u32,
    )
    .expect("run");
    backend.sync().expect("sync");
    let mut raw = vec![0u8; TOKENS * HIDDEN * 4];
    backend.copy_d2h(out.view(), &mut raw).expect("d2h");
    let got: &[f32] = bytemuck::cast_slice(&raw);

    let widen = |v: &[f32]| v.iter().map(|a| f64::from(*a)).collect::<Vec<f64>>();
    let (router, shared_up, shared_down) = (widen(&router), widen(&shared_up), widen(&shared_down));
    let mut peak = 0.0f64;
    let mut worst = 0.0f64;
    for t in 0..TOKENS {
        let xt = widen(&x[t * HIDDEN..(t + 1) * HIDDEN]);
        let mut want = matvec(
            &shared_down,
            &relu2(matvec(&shared_up, &xt, HIDDEN, SHARED)),
            SHARED,
            HIDDEN,
        );
        let score: Vec<f64> = matvec(&router, &xt, HIDDEN, EXPERTS)
            .into_iter()
            .map(|l| 1.0 / (1.0 + (-l).exp()))
            .collect();
        let mut order: Vec<usize> = (0..EXPERTS).collect();
        order.sort_by(|&a, &b| {
            let (ca, cb) = (score[a] + f64::from(bias[a]), score[b] + f64::from(bias[b]));
            cb.total_cmp(&ca).then(a.cmp(&b))
        });
        let chosen = &order[..USED];
        let total: f64 = chosen.iter().map(|&e| score[e]).sum();
        for &e in chosen {
            let w = score[e] / total * f64::from(SCALE);
            let up = &up_dense[e * WIDTH * HIDDEN..(e + 1) * WIDTH * HIDDEN];
            let down = &down_dense[e * HIDDEN * WIDTH..(e + 1) * HIDDEN * WIDTH];
            let y = matvec(down, &relu2(matvec(up, &xt, HIDDEN, WIDTH)), WIDTH, HIDDEN);
            for (o, v) in want.iter_mut().zip(y) {
                *o += w * v;
            }
        }
        for (h, w) in want.iter().enumerate() {
            peak = peak.max(w.abs());
            worst = worst.max((f64::from(got[t * HIDDEN + h]) - w).abs());
        }
    }
    assert!(
        worst <= 1e-3 * peak,
        "worst error {worst} against a peak of {peak}"
    );
}

/// Routed layers, experts and slots of the streamed case: the pool holds
/// little more than one layer, so later layers evict earlier ones.
const S_LAYERS: usize = 3;
const S_EXPERTS: usize = 32;
const S_SLOTS: u32 = 40;

/// Every layer's up and down banks, served from host memory.
#[derive(Debug)]
struct HostBanks {
    /// `[layer][matrix]` bytes, experts contiguous.
    banks: Vec<[Vec<u8>; 2]>,
    expert_bytes: [u64; 2],
}

impl ExpertSource for HostBanks {
    fn parts(&self) -> &'static [WeightPart] {
        &[WeightPart::Whole]
    }

    fn matrices(&self) -> usize {
        2
    }

    fn layout(&self) -> Vec<u64> {
        self.expert_bytes.to_vec()
    }

    fn read(
        &self,
        layer: u32,
        expert: u32,
        array: usize,
        dst: &mut [u8],
    ) -> shiso_core::CoreResult<()> {
        let per = self.expert_bytes[array] as usize;
        let at = expert as usize * per;
        dst.copy_from_slice(&self.banks[layer as usize][array][at..at + per]);
        Ok(())
    }
}

fn blocks(n: usize, seed: u32) -> Vec<u8> {
    let scales = signal(n, seed);
    let blocks: Vec<BlockIq4Nl> = (0..n)
        .map(|b| {
            let mut qs = [0u8; 16];
            for (j, q) in qs.iter_mut().enumerate() {
                *q = (((b * 16 + j) as u32 ^ seed).wrapping_mul(2_654_435_761) >> 13) as u8;
            }
            BlockIq4Nl {
                d: half::f16::from_f32(scales[b] * 0.004).to_bits(),
                qs,
            }
        })
        .collect();
    bytemuck::cast_slice(&blocks).to_vec()
}

#[test]
#[ignore = "requires a Metal device"]
#[expect(
    clippy::print_stderr,
    reason = "a machine with no Metal device says why it skipped"
)]
#[expect(clippy::too_many_lines, reason = "one fixture run two ways")]
fn streamed_biased_moe_equals_resident() {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping");
        return;
    }
    let backend: Arc<dyn Backend> =
        shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");
    let up_blocks = S_EXPERTS * WIDTH * HIDDEN / 32;
    let down_blocks = S_EXPERTS * HIDDEN * WIDTH / 32;
    let banks: Vec<[Vec<u8>; 2]> = (0..S_LAYERS as u32)
        .map(|l| [blocks(up_blocks, 100 + l), blocks(down_blocks, 200 + l)])
        .collect();
    let per = [
        (WIDTH * HIDDEN / 32 * 18) as u64,
        (HIDDEN * WIDTH / 32 * 18) as u64,
    ];
    let stream = Arc::new(
        ExpertStream::open(
            &backend,
            Arc::new(HostBanks {
                banks: banks.clone(),
                expert_bytes: per,
            }),
            S_LAYERS as u32,
            S_EXPERTS as u32,
            S_SLOTS,
            S_EXPERTS as u32,
        )
        .expect("stream"),
    );
    let shape = MoeShape::new(S_EXPERTS as u32, USED as u32, 1, WIDTH as u32);
    let layer = |l: usize, streamed: bool| {
        let store = |matrix: usize| {
            if streamed {
                StackStore::Streamed {
                    stream: Arc::clone(&stream),
                    matrix,
                    layer: l as u32,
                }
            } else {
                StackStore::Resident(upload(&backend, &banks[l][matrix]))
            }
        };
        let router = signal(S_EXPERTS * HIDDEN, 300 + l as u32);
        let bias: Vec<f32> = (0..S_EXPERTS).map(|e| (e % 5) as f32 * 0.01).collect();
        // A streamed layer guesses the next one's experts; the guess must not
        // change a result.
        let guess = (streamed && l + 1 < S_LAYERS).then(|| RouterGuess {
            layer: l as u32 + 1,
            weights: signal(S_EXPERTS * HIDDEN, 301 + l as u32).into(),
            bias: bias.clone().into(),
        });
        let moe = BiasedMoe {
            router: f32_weight(&backend, &router, HIDDEN, S_EXPERTS),
            bias: upload(&backend, bytemuck::cast_slice(&bias)),
            scale: SCALE,
            renormalize: true,
            up: ExpertStack::Iq4Nl(RowIq4NlStack {
                store: store(0),
                in_dim: HIDDEN as u32,
                out_dim: WIDTH as u32,
            }),
            down: ExpertStack::Iq4Nl(RowIq4NlStack {
                store: store(1),
                in_dim: WIDTH as u32,
                out_dim: HIDDEN as u32,
            }),
            guess,
        };
        moe_biased::check_stack(&moe.up).expect("up stack");
        moe_biased::check_stack(&moe.down).expect("down stack");
        let shared = Ffn {
            gate: None,
            up: f32_weight(
                &backend,
                &signal(SHARED * HIDDEN, 400 + l as u32),
                HIDDEN,
                SHARED,
            ),
            down: f32_weight(
                &backend,
                &signal(HIDDEN * SHARED, 500 + l as u32),
                SHARED,
                HIDDEN,
            ),
        };
        (moe, shared)
    };
    let resident: Vec<_> = (0..S_LAYERS).map(|l| layer(l, false)).collect();
    let streamed: Vec<_> = (0..S_LAYERS).map(|l| layer(l, true)).collect();
    let scratch = MoeScratch::new(&backend, shape, HIDDEN as u32, 64).expect("scratch");
    // Four one-row steps, then a 64-row chunk, through every layer in turn.
    let passes: Vec<(usize, u32)> = (0..4)
        .map(|i| (1usize, 900 + i))
        .chain([(64, 999)])
        .collect();
    let run = |layers: &[(BiasedMoe, Ffn)], rows: usize, seed: u32| -> Vec<Vec<u8>> {
        let mut outs = Vec::new();
        for (l, (moe, shared)) in layers.iter().enumerate() {
            let x = signal(rows * HIDDEN, seed + l as u32);
            let input = upload(&backend, bytemuck::cast_slice(&x));
            let out = alloc(&backend, (rows * HIDDEN * 4) as u64).expect("alloc");
            moe_biased::run(
                backend.as_ref(),
                moe,
                shared,
                shape,
                &scratch,
                MoeIo {
                    input: input.view(),
                    block_out: out.view(),
                },
                rows as u32,
            )
            .expect("run");
            backend.sync().expect("sync");
            let mut raw = vec![0u8; rows * HIDDEN * 4];
            backend.copy_d2h(out.view(), &mut raw).expect("d2h");
            outs.push(raw);
        }
        outs
    };
    for (i, &(rows, seed)) in passes.iter().enumerate() {
        let want = run(&resident, rows, seed);
        let got = run(&streamed, rows, seed);
        assert_eq!(want, got, "pass {i} ({rows} rows) differs when streamed");
        if i == 1 {
            assert!(stream.stats().hits > 0, "a repeated step hits the pool");
        }
    }
    assert!(
        stream.stats().prefetch_reads > 0,
        "the decode steps guessed ahead"
    );
}
