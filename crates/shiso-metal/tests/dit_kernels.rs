//! Metal's run of the diffusion kernel pack.
//!
//! Every entry point the diffusion driver (`shiso-models/src/diffusion/ops.rs`)
//! launches is exercised here against expected values computed independently
//! in this file - the host twins are never called, so a kernel and its twin
//! cannot agree on the same mistake. The launch geometry, argument structs
//! and buffer binding order mirror the driver's marshalling exactly: these
//! tests are the kernel side of the Task-3 contract table.
//!
//! The cases run on any machine with a Metal device (this crate's default
//! feature); without one they say so on stderr rather than pass silently.
#![cfg(all(feature = "metal", target_os = "macos"))]
#![expect(
    clippy::cast_possible_truncation,
    reason = "test dimensions are single- to double-digit element counts converted to u32 kernel arguments"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "indices converted for expected-value math are small test extents, exact in f32"
)]

use std::sync::Arc;

use bytemuck::{Pod, Zeroable, bytes_of};
use half::bf16;
use shiso_core::{
    Backend, DeviceBuffer, DeviceView, Extent3, KernelArg, KernelRef, LaunchArgs, alloc,
};
use shiso_metal::MetalBackend;

/// The module every diffusion entry lives under.
const MODULE: &str = "diffusion";
/// Threads per elementwise threadgroup, as the driver launches it.
const BLOCK: u32 = 64;
/// Tile edge of the GEMM kernels, as the driver launches it.
const GEMM_TILE: u32 = 16;
/// Output tile edge of the simdgroup GEMM and convolution kernels.
const G64_TILE: u32 = 64;
/// Threads per simdgroup-tile threadgroup.
const G64_THREADS: u32 = 128;
/// Query rows per fused-attention threadgroup.
const FA_QROWS: u32 = 32;
/// Threads per fused-attention threadgroup.
const FA_THREADS: u32 = 128;

/// Every entry point the driver launches, per the Task-3 contract table.
const ENTRIES: &[&str] = &[
    "kernel_dit_gemm_bf16",
    "kernel_dit_gemm_fp8",
    "kernel_dit_gemm_f32",
    "kernel_dit_silu_f32",
    "kernel_dit_swiglu_f32",
    "kernel_dit_swiglu2_f32",
    "kernel_dit_modulate_f32",
    "kernel_dit_add_gated_f32",
    "kernel_dit_add_f32",
    "kernel_dit_axpy_f32",
    "kernel_dit_layer_norm_f32",
    "kernel_dit_rmsnorm_f32",
    "kernel_dit_qk_norm_f32",
    "kernel_dit_concat_rows_f32",
    "kernel_dit_sdpa_f32",
    "kernel_dit_gqa_f32",
    "kernel_dit_rope_4d",
    "kernel_dit_rope_half",
    "kernel_dit_conv2d_f32",
    "kernel_dit_group_norm_stats_f32",
    "kernel_dit_group_norm_apply_f32",
    "kernel_dit_softmax_rows_f32",
    "kernel_dit_upsample_nearest_f32",
    "kernel_dit_transpose_f32",
    "kernel_dit_embed_bf16_f32",
    "kernel_dit_gemm64_f32",
    "kernel_dit_gemm64_bf16",
    "kernel_dit_gemm64_fp8",
    "kernel_dit_conv2d_gemm_f32",
    "kernel_dit_flash_sdpa_f32",
    "kernel_dit_gemm64_fused_f32",
    "kernel_dit_gemm64_fused_bf16",
    "kernel_dit_gemm64_fused_fp8",
];

// --- argument blocks: byte-for-byte the driver's `#[repr(C)]` Pods ---------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GemmArgs {
    m: u32,
    n: u32,
    k: u32,
    has_bias: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GemmFusedArgs {
    m: u32,
    n: u32,
    cols_a: u32,
    inner: u32,
    gu_stride: u32,
    gu_offset: u32,
    has_bias: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UnaryArgs {
    count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SwigluArgs {
    rows: u32,
    inner: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ModulateArgs {
    count: u32,
    period: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AddGatedArgs {
    count: u32,
    period: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AxpyArgs {
    count: u32,
    dt: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NormArgs {
    rows: u32,
    dim: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QkNormArgs {
    tokens: u32,
    heads: u32,
    head_dim: u32,
    row_stride: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ConcatArgs {
    rows: u32,
    cols_a: u32,
    cols_b: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SdpaArgs {
    txt_tokens: u32,
    img_tokens: u32,
    cache_tokens: u32,
    heads: u32,
    head_dim: u32,
    txt_stride: u32,
    img_stride: u32,
    cache_stride: u32,
    q_offset: u32,
    k_offset: u32,
    v_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GqaArgs {
    tokens: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Rope4dArgs {
    tokens: u32,
    heads: u32,
    head_dim: u32,
    freqs_per_axis: u32,
    row_stride: u32,
    q_offset: u32,
    k_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RopeHalfArgs {
    tokens: u32,
    heads: u32,
    head_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Conv2dArgs {
    batch: u32,
    in_c: u32,
    in_h: u32,
    in_w: u32,
    out_c: u32,
    out_h: u32,
    out_w: u32,
    k_h: u32,
    k_w: u32,
    stride: u32,
    pad: u32,
    has_bias: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GroupNormArgs {
    batch: u32,
    num_groups: u32,
    channels_per_group: u32,
    spatial: u32,
    eps: f32,
    chunks: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UpsampleArgs {
    batch: u32,
    channels: u32,
    h: u32,
    w: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EmbedArgs {
    tokens: u32,
    hidden: u32,
}

// --- harness ---------------------------------------------------------------

/// Opens the device, or says on stderr why the cases did not run.
#[expect(
    clippy::print_stderr,
    reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn open() -> Option<Arc<dyn Backend>> {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the diffusion GPU cases");
        return None;
    }
    let metal = MetalBackend::new().expect("the embedded kernel library compiles");
    Some(metal.clone())
}

fn grid_for(total: u32, tile: u32) -> u32 {
    total.div_ceil(tile).max(1)
}

fn upload_f32(b: &Arc<dyn Backend>, values: &[f32]) -> DeviceBuffer {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buffer = alloc(b, bytes.len() as u64).expect("alloc");
    b.copy_h2d(buffer.view(), &bytes).expect("upload f32");
    buffer
}

fn upload_u16(b: &Arc<dyn Backend>, values: &[u16]) -> DeviceBuffer {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buffer = alloc(b, bytes.len() as u64).expect("alloc");
    b.copy_h2d(buffer.view(), &bytes).expect("upload u16");
    buffer
}

fn upload_u8(b: &Arc<dyn Backend>, values: &[u8]) -> DeviceBuffer {
    let buffer = alloc(b, values.len() as u64).expect("alloc");
    b.copy_h2d(buffer.view(), values).expect("upload u8");
    buffer
}

fn upload_u32(b: &Arc<dyn Backend>, values: &[u32]) -> DeviceBuffer {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buffer = alloc(b, bytes.len() as u64).expect("alloc");
    b.copy_h2d(buffer.view(), &bytes).expect("upload u32");
    buffer
}

fn read_f32(b: &Arc<dyn Backend>, view: DeviceView<'_>, count: usize) -> Vec<f32> {
    b.sync().expect("sync before read");
    let mut bytes = vec![0u8; count * 4];
    b.copy_d2h(view, &mut bytes).expect("download");
    bytes
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

fn launch(
    b: &Arc<dyn Backend>,
    entry: &'static str,
    grid: Extent3,
    block: Extent3,
    args: &[KernelArg<'_>],
) {
    b.run(
        KernelRef::new(MODULE, entry),
        &LaunchArgs::new(grid, block, args),
    )
    .unwrap_or_else(|err| panic!("{entry} failed to launch: {err}"));
}

/// A zero-filled 4-byte buffer: the driver's stand-in for an absent operand.
fn dummy(b: &Arc<dyn Backend>) -> DeviceBuffer {
    let buffer = alloc(b, 4).expect("alloc dummy");
    b.fill_zero(buffer.view()).expect("zero dummy");
    buffer
}

/// Deterministic distinct values, so a wrong offset lands on different
/// numbers instead of a coincidental match.
fn vals(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let step = (i * 37 + seed * 11) % 101;
            f32::from(u16::try_from(step).unwrap_or(u16::MAX)) * 0.125 - 6.25
        })
        .collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn assert_close(got: &[f32], expected: &[f32], what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: length");
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert!(
            (g - e).abs() <= 1e-4 * e.abs().max(1.0),
            "{what}: element {i} got {g}, expected {e}"
        );
    }
}

/// The fused attention stages q, k, v and the probabilities as half and
/// accumulates in f32: about three decimal digits on these cases' values,
/// still well inside what [`assert_differs`] tells apart.
fn assert_close_half(got: &[f32], expected: &[f32], what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: length");
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert!(
            (g - e).abs() <= 2e-3 * e.abs().max(0.05),
            "{what}: element {i} got {g}, expected {e}"
        );
    }
}

/// The two candidate readings must differ, or the case proves nothing.
fn assert_differs(a: &[f32], b: &[f32], what: &str) {
    let discriminates = a
        .iter()
        .zip(b)
        .any(|(&x, &y)| (x - y).abs() > 1e-3 * y.abs().max(1.0));
    assert!(
        discriminates,
        "{what}: data does not distinguish the readings"
    );
}

// --- entry resolution and the driver's probe -------------------------------

#[test]
#[expect(
    clippy::print_stderr,
    reason = "a machine with no Metal device skips rather than fails, and the reason belongs on stderr"
)]
fn every_dit_entry_point_resolves_as_a_pipeline() {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the pipeline check");
        return;
    }
    let backend = MetalBackend::new().expect("the embedded kernel library compiles");
    for &entry in ENTRIES {
        backend
            .prepare(KernelRef::new(MODULE, entry))
            .unwrap_or_else(|err| panic!("{entry} does not resolve: {err}"));
    }
}

/// The driver decides kernel-vs-host by probing `module_entries` for a name
/// containing `kernel_dit_gemm_bf16` (`ops.rs::Ops::new`). Metal's table
/// lists CUDA-style mangled names only, so the registration must carry one
/// or the whole kernel pack is dead code behind the host fallback.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "a machine with no Metal device skips rather than fails, and the reason belongs on stderr"
)]
fn the_driver_probe_finds_the_diffusion_kernel_set() {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the probe check");
        return;
    }
    let backend = MetalBackend::new().expect("the embedded kernel library compiles");
    let entries = backend.module_entries(MODULE);
    assert!(
        entries
            .iter()
            .any(|name| name.contains("kernel_dit_gemm_bf16")),
        "the driver's probe would fall back to the host path: {entries:?}"
    );
}

// --- gemm -------------------------------------------------------------------

/// `out[m,n] = in[m,k] @ weight[n,k]^T * scale + bias[n]`, weight bf16.
fn expect_gemm(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += input[i * k + kk] * weight[j * k + kk];
            }
            out[i * n + j] = acc * scale + bias[j];
        }
    }
    out
}

/// One GEMM launch with the driver's geometry. `has_bias` is the driver's
/// presence flag, set from whether a real bias was bound rather than from
/// the values.
#[derive(Clone, Copy)]
struct GemmRun<'a> {
    entry: &'static str,
    weight: DeviceView<'a>,
    scale: f32,
    bias: DeviceView<'a>,
    has_bias: bool,
    input: &'a [f32],
    shape: (usize, usize, usize),
}

/// Launches one GEMM and reads the result back. `shape` is `(m, n, k)`.
fn run_gemm(b: &Arc<dyn Backend>, run: GemmRun<'_>) -> Vec<f32> {
    let (m, n, k) = run.shape;
    let in_buf = upload_f32(b, run.input);
    let out_buf = alloc(b, (m * n * 4) as u64).expect("alloc out");
    let args = GemmArgs {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        has_bias: u32::from(run.has_bias),
    };
    launch(
        b,
        run.entry,
        Extent3::new(
            grid_for(n as u32, GEMM_TILE),
            grid_for(m as u32, GEMM_TILE),
            1,
        ),
        Extent3::new(GEMM_TILE, GEMM_TILE, 1),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(in_buf.view()),
            KernelArg::Buffer(run.weight),
            KernelArg::F32(run.scale),
            KernelArg::Buffer(run.bias),
        ],
    );
    read_f32(b, out_buf.view(), m * n)
}

#[test]
fn gemm_bf16_is_the_transposed_product_scaled_and_biased() {
    let Some(b) = open() else { return };
    let (m, n, k) = (2usize, 5usize, 20usize); // k crosses the 16-wide tile
    let input = vals(m * k, 1);
    // Exact bf16 values (multiples of 1/4 in a small range), so the expected
    // product is the product of what the kernel actually reads.
    let weight: Vec<f32> = (0..n * k).map(|i| ((i % 13) as f32) * 0.25 - 1.5).collect();
    let bias = vals(n, 5);
    let scale = 0.5f32;
    let wbuf = upload_u16(
        &b,
        &weight
            .iter()
            .map(|&v| u16::from_le_bytes(bf16::from_f32(v).to_le_bytes()))
            .collect::<Vec<u16>>(),
    );
    let bias_buf = upload_f32(&b, &bias);
    let got = run_gemm(
        &b,
        GemmRun {
            entry: "kernel_dit_gemm_bf16",
            weight: wbuf.view(),
            scale,
            bias: bias_buf.view(),
            has_bias: true,
            input: &input,
            shape: (m, n, k),
        },
    );
    let expected = expect_gemm(&input, &weight, &bias, scale, m, n, k);
    assert_close(&got, &expected, "gemm_bf16");
}

#[test]
fn gemm_bf16_treats_the_zero_bias_stand_in_as_no_bias() {
    let Some(b) = open() else { return };
    let (m, n, k) = (2usize, 3usize, 4usize);
    let input = vals(m * k, 2);
    let weight: Vec<f32> = (0..n * k).map(|i| ((i % 7) as f32) * 0.5 - 1.0).collect();
    let wbuf = upload_u16(
        &b,
        &weight
            .iter()
            .map(|&v| u16::from_le_bytes(bf16::from_f32(v).to_le_bytes()))
            .collect::<Vec<u16>>(),
    );
    // The driver's stand-in for an absent bias is a dedicated 4-byte zero
    // buffer with the presence flag cleared; the kernel must trust the
    // flag, so the case binds the driver's real shape.
    let stand_in = dummy(&b);
    let got = run_gemm(
        &b,
        GemmRun {
            entry: "kernel_dit_gemm_bf16",
            weight: wbuf.view(),
            scale: 1.0,
            bias: stand_in.view(),
            has_bias: false,
            input: &input,
            shape: (m, n, k),
        },
    );
    let expected = expect_gemm(&input, &weight, &vec![0.0f32; n], 1.0, m, n, k);
    assert_close(&got, &expected, "gemm_bf16 no-bias stand-in");
}

/// Regression lock from the zero-sentinel era: a real bias whose first
/// word is -0.0 (bit pattern 0x80000000) must be applied. Under the
/// presence flag this is trivially correct - which is exactly what the
/// flag was for. It catches a sentinel relaxed to a value comparison
/// (`bias[0] == 0.0f`, which reads -0.0 as absent); a bit-exact sentinel
/// would pass it, and the +0.0 case is what
/// [`gemm_bf16_applies_a_real_bias_whose_first_word_is_positive_zero`]
/// pins.
#[test]
fn gemm_bf16_applies_a_real_bias_whose_first_word_is_negative_zero() {
    let Some(b) = open() else { return };
    let (m, n, k) = (2usize, 3usize, 4usize);
    let input = vals(m * k, 6);
    let weight: Vec<f32> = (0..n * k).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
    let wbuf = upload_u16(
        &b,
        &weight
            .iter()
            .map(|&v| u16::from_le_bytes(bf16::from_f32(v).to_le_bytes()))
            .collect::<Vec<u16>>(),
    );
    let bias = vec![-0.0f32, 0.5, -0.25];
    let bias_buf = upload_f32(&b, &bias);
    let got = run_gemm(
        &b,
        GemmRun {
            entry: "kernel_dit_gemm_bf16",
            weight: wbuf.view(),
            scale: 1.0,
            bias: bias_buf.view(),
            has_bias: true,
            input: &input,
            shape: (m, n, k),
        },
    );
    assert_differs(
        &got,
        &expect_gemm(&input, &weight, &vec![0.0f32; n], 1.0, m, n, k),
        "gemm_bf16 negative-zero-first bias",
    );
    let expected = expect_gemm(&input, &weight, &bias, 1.0, m, n, k);
    assert_close(&got, &expected, "gemm_bf16 negative-zero-first bias");
}

/// The case the zero sentinel silently got wrong: a real bias whose first
/// element is exactly +0.0, bound with the presence flag set, must be
/// applied in full. The flag, not the values, decides.
#[test]
fn gemm_bf16_applies_a_real_bias_whose_first_word_is_positive_zero() {
    let Some(b) = open() else { return };
    let (m, n, k) = (2usize, 3usize, 4usize);
    let input = vals(m * k, 7);
    let weight: Vec<f32> = (0..n * k).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
    let wbuf = upload_u16(
        &b,
        &weight
            .iter()
            .map(|&v| u16::from_le_bytes(bf16::from_f32(v).to_le_bytes()))
            .collect::<Vec<u16>>(),
    );
    let bias = vec![0.0f32, 0.5, -0.25];
    let bias_buf = upload_f32(&b, &bias);
    let got = run_gemm(
        &b,
        GemmRun {
            entry: "kernel_dit_gemm_bf16",
            weight: wbuf.view(),
            scale: 1.0,
            bias: bias_buf.view(),
            has_bias: true,
            input: &input,
            shape: (m, n, k),
        },
    );
    assert_differs(
        &got,
        &expect_gemm(&input, &weight, &vec![0.0f32; n], 1.0, m, n, k),
        "gemm_bf16 positive-zero-first bias",
    );
    let expected = expect_gemm(&input, &weight, &bias, 1.0, m, n, k);
    assert_close(&got, &expected, "gemm_bf16 positive-zero-first bias");
}

/// The flag wins over the binding: a real, nonzero bias bound with the
/// presence flag cleared must not be read - the output is unbiased. The
/// stand-in case (dummy bound, flag cleared) is pinned by
/// `gemm_bf16_treats_the_zero_bias_stand_in_as_no_bias`.
#[test]
fn gemm_bf16_ignores_the_bias_binding_when_the_flag_is_clear() {
    let Some(b) = open() else { return };
    let (m, n, k) = (2usize, 3usize, 4usize);
    let input = vals(m * k, 8);
    let weight: Vec<f32> = (0..n * k).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
    let wbuf = upload_u16(
        &b,
        &weight
            .iter()
            .map(|&v| u16::from_le_bytes(bf16::from_f32(v).to_le_bytes()))
            .collect::<Vec<u16>>(),
    );
    let bias = vals(n, 9);
    let bias_buf = upload_f32(&b, &bias);
    let got = run_gemm(
        &b,
        GemmRun {
            entry: "kernel_dit_gemm_bf16",
            weight: wbuf.view(),
            scale: 1.0,
            bias: bias_buf.view(),
            has_bias: false,
            input: &input,
            shape: (m, n, k),
        },
    );
    let unbiased = expect_gemm(&input, &weight, &vec![0.0f32; n], 1.0, m, n, k);
    assert_differs(
        &unbiased,
        &expect_gemm(&input, &weight, &bias, 1.0, m, n, k),
        "gemm_bf16 flag-clear bias: readings",
    );
    assert_close(&got, &unbiased, "gemm_bf16 flag-clear bias");
}

#[test]
fn gemm_fp8_decodes_e4m3_bytes_and_scales_the_accumulator() {
    let Some(b) = open() else { return };
    let (m, n, k) = (2usize, 3usize, 4usize);
    // (byte, value) pairs, every byte exactly representable in E4M3
    // (sign, exponent bias 7, three mantissa bits): 0x38=1.0, 0xB8=-1.0,
    // 0x30=0.5, 0xB0=-0.5, 0x40=2.0, 0x3C=1.5, 0x28=0.25, 0xA8=-0.25.
    let table: [(u8, f32); 8] = [
        (0x38, 1.0),
        (0xB8, -1.0),
        (0x30, 0.5),
        (0xB0, -0.5),
        (0x40, 2.0),
        (0x3C, 1.5),
        (0x28, 0.25),
        (0xA8, -0.25),
    ];
    let bytes: Vec<u8> = (0..n * k).map(|i| table[i % 8].0).collect();
    let weight: Vec<f32> = (0..n * k).map(|i| table[i % 8].1).collect();
    let input = vals(m * k, 3);
    let bias = vals(n, 6);
    let scale = 0.25f32;
    let wbuf = upload_u8(&b, &bytes);
    let bias_buf = upload_f32(&b, &bias);
    let got = run_gemm(
        &b,
        GemmRun {
            entry: "kernel_dit_gemm_fp8",
            weight: wbuf.view(),
            scale,
            bias: bias_buf.view(),
            has_bias: true,
            input: &input,
            shape: (m, n, k),
        },
    );
    let expected = expect_gemm(&input, &weight, &bias, scale, m, n, k);
    // A weight read as e5m2 or as raw bytes differs by far more than the
    // tolerance on at least one element.
    let raw: Vec<f32> = bytes.iter().map(|&v| f32::from(v)).collect();
    assert_differs(
        &expected,
        &expect_gemm(&input, &raw, &bias, scale, m, n, k),
        "gemm_fp8: decoded vs raw weight",
    );
    assert_close(&got, &expected, "gemm_fp8");
}

#[test]
fn gemm_f32_is_the_transposed_product() {
    let Some(b) = open() else { return };
    let (m, n, k) = (3usize, 4usize, 5usize);
    let input = vals(m * k, 4);
    let weight = vals(n * k, 8);
    let bias = vals(n, 9);
    let wbuf = upload_f32(&b, &weight);
    let bias_buf = upload_f32(&b, &bias);
    let got = run_gemm(
        &b,
        GemmRun {
            entry: "kernel_dit_gemm_f32",
            weight: wbuf.view(),
            scale: 1.0,
            bias: bias_buf.view(),
            has_bias: true,
            input: &input,
            shape: (m, n, k),
        },
    );
    let expected = expect_gemm(&input, &weight, &bias, 1.0, m, n, k);
    assert_close(&got, &expected, "gemm_f32");
}

// --- elementwise ------------------------------------------------------------

#[test]
fn silu_matches_the_sigmoid_product() {
    let Some(b) = open() else { return };
    let x = vals(200, 10);
    let in_buf = upload_f32(&b, &x);
    let out_buf = alloc(&b, (x.len() * 4) as u64).expect("alloc");
    let args = UnaryArgs {
        count: x.len() as u32,
    };
    launch(
        &b,
        "kernel_dit_silu_f32",
        Extent3::linear(grid_for(x.len() as u32, BLOCK)),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(in_buf.view()),
        ],
    );
    let expected: Vec<f32> = x.iter().map(|&v| silu(v)).collect();
    let got = read_f32(&b, out_buf.view(), x.len());
    assert_close(&got, &expected, "silu");
}

#[test]
fn swiglu_gates_the_first_half_of_strided_rows() {
    let Some(b) = open() else { return };
    let (rows, inner, stride) = (2usize, 3usize, 8usize); // stride > 2*inner: the halves are not adjacent
    let x = vals(rows * stride, 11);
    let in_buf = upload_f32(&b, &x);
    let out_buf = alloc(&b, (rows * inner * 4) as u64).expect("alloc");
    let args = SwigluArgs {
        rows: rows as u32,
        inner: inner as u32,
        stride: stride as u32,
    };
    launch(
        &b,
        "kernel_dit_swiglu_f32",
        Extent3::new(grid_for(inner as u32, BLOCK), rows as u32, 1),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(in_buf.view()),
        ],
    );
    let mut expected = vec![0f32; rows * inner];
    for r in 0..rows {
        for i in 0..inner {
            let g = x[r * stride + i];
            let v = x[r * stride + inner + i];
            expected[r * inner + i] = silu(g) * v;
        }
    }
    // The opposite gate order (value gates the silu) must differ.
    let mut flipped = vec![0f32; rows * inner];
    for r in 0..rows {
        for i in 0..inner {
            let g = x[r * stride + inner + i];
            let v = x[r * stride + i];
            flipped[r * inner + i] = silu(g) * v;
        }
    }
    assert_differs(&expected, &flipped, "swiglu: gate-first vs gate-second");
    let got = read_f32(&b, out_buf.view(), rows * inner);
    assert_close(&got, &expected, "swiglu");
}

#[test]
fn swiglu_reads_a_view_that_starts_mid_row() {
    let Some(b) = open() else { return };
    // The single block's swiglu view starts at column 3h of a fused row: the
    // kernel must index from the view base, not the allocation base.
    let (rows, inner, stride) = (2usize, 3usize, 10usize);
    let mut full = vec![99.0f32; 4 + rows * stride];
    let body = vals(rows * stride, 12);
    full[4..4 + rows * stride].copy_from_slice(&body);
    let buf = upload_f32(&b, &full);
    let view = buf
        .view()
        .slice(16, (rows * stride * 4) as u64)
        .expect("slice");
    let out_buf = alloc(&b, (rows * inner * 4) as u64).expect("alloc");
    let args = SwigluArgs {
        rows: rows as u32,
        inner: inner as u32,
        stride: stride as u32,
    };
    launch(
        &b,
        "kernel_dit_swiglu_f32",
        Extent3::new(grid_for(inner as u32, BLOCK), rows as u32, 1),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(view),
        ],
    );
    let mut expected = vec![0f32; rows * inner];
    for r in 0..rows {
        for i in 0..inner {
            expected[r * inner + i] = silu(body[r * stride + i]) * body[r * stride + inner + i];
        }
    }
    let got = read_f32(&b, out_buf.view(), rows * inner);
    assert_close(&got, &expected, "swiglu mid-row view");
}

#[test]
fn swiglu2_is_silu_of_a_times_b() {
    let Some(b) = open() else { return };
    let a = vals(100, 13);
    let up = vals(100, 14);
    let a_buf = upload_f32(&b, &a);
    let up_buf = upload_f32(&b, &up);
    let out_buf = alloc(&b, (a.len() * 4) as u64).expect("alloc");
    let args = UnaryArgs {
        count: a.len() as u32,
    };
    launch(
        &b,
        "kernel_dit_swiglu2_f32",
        Extent3::linear(grid_for(a.len() as u32, BLOCK)),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(a_buf.view()),
            KernelArg::Buffer(up_buf.view()),
        ],
    );
    let expected: Vec<f32> = a.iter().zip(&up).map(|(&g, &v)| silu(g) * v).collect();
    let got = read_f32(&b, out_buf.view(), a.len());
    assert_close(&got, &expected, "swiglu2");
}

#[test]
fn modulate_broadcasts_scale_and_shift_over_the_period() {
    let Some(b) = open() else { return };
    let (count, period) = (7usize, 3usize);
    let x = vals(count, 15);
    let scale = vals(period, 16);
    let shift = vals(period, 17);
    let x_buf = upload_f32(&b, &x);
    let s_buf = upload_f32(&b, &scale);
    let h_buf = upload_f32(&b, &shift);
    let out_buf = alloc(&b, (count * 4) as u64).expect("alloc");
    let args = ModulateArgs {
        count: count as u32,
        period: period as u32,
    };
    launch(
        &b,
        "kernel_dit_modulate_f32",
        Extent3::linear(grid_for(count as u32, BLOCK)),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(x_buf.view()),
            KernelArg::Buffer(s_buf.view()),
            KernelArg::Buffer(h_buf.view()),
        ],
    );
    let expected: Vec<f32> = (0..count)
        .map(|i| x[i] * (1.0 + scale[i % period]) + shift[i % period])
        .collect();
    let got = read_f32(&b, out_buf.view(), count);
    assert_close(&got, &expected, "modulate");
}

#[test]
fn add_gated_is_a_plus_the_broadcast_gate_times_b() {
    let Some(b) = open() else { return };
    let (count, period) = (6usize, 4usize);
    let a = vals(count, 18);
    let gate = vals(period, 19);
    let c = vals(count, 20);
    let a_buf = upload_f32(&b, &a);
    let g_buf = upload_f32(&b, &gate);
    let c_buf = upload_f32(&b, &c);
    let out_buf = alloc(&b, (count * 4) as u64).expect("alloc");
    let args = AddGatedArgs {
        count: count as u32,
        period: period as u32,
    };
    launch(
        &b,
        "kernel_dit_add_gated_f32",
        Extent3::linear(grid_for(count as u32, BLOCK)),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(a_buf.view()),
            KernelArg::Buffer(g_buf.view()),
            KernelArg::Buffer(c_buf.view()),
        ],
    );
    let expected: Vec<f32> = (0..count).map(|i| a[i] + gate[i % period] * c[i]).collect();
    let got = read_f32(&b, out_buf.view(), count);
    assert_close(&got, &expected, "add_gated");
}

#[test]
fn add_is_the_elementwise_sum() {
    let Some(b) = open() else { return };
    let a = vals(130, 21);
    let c = vals(130, 22);
    let a_buf = upload_f32(&b, &a);
    let c_buf = upload_f32(&b, &c);
    let out_buf = alloc(&b, (a.len() * 4) as u64).expect("alloc");
    let args = UnaryArgs {
        count: a.len() as u32,
    };
    launch(
        &b,
        "kernel_dit_add_f32",
        Extent3::linear(grid_for(a.len() as u32, BLOCK)),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(a_buf.view()),
            KernelArg::Buffer(c_buf.view()),
        ],
    );
    let expected: Vec<f32> = a.iter().zip(&c).map(|(&x, &y)| x + y).collect();
    let got = read_f32(&b, out_buf.view(), a.len());
    assert_close(&got, &expected, "add");
}

#[test]
fn axpy_adds_dt_times_v() {
    let Some(b) = open() else { return };
    let x = vals(70, 23);
    let v = vals(70, 24);
    let dt = -0.75f32;
    let x_buf = upload_f32(&b, &x);
    let v_buf = upload_f32(&b, &v);
    let out_buf = alloc(&b, (x.len() * 4) as u64).expect("alloc");
    let args = AxpyArgs {
        count: x.len() as u32,
        dt,
    };
    launch(
        &b,
        "kernel_dit_axpy_f32",
        Extent3::linear(grid_for(x.len() as u32, BLOCK)),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(x_buf.view()),
            KernelArg::Buffer(v_buf.view()),
        ],
    );
    let expected: Vec<f32> = x.iter().zip(&v).map(|(&a, &c)| a + dt * c).collect();
    let got = read_f32(&b, out_buf.view(), x.len());
    assert_close(&got, &expected, "axpy");
}

// --- norms -------------------------------------------------------------------

#[test]
fn layer_norm_subtracts_the_mean_and_divides_by_the_deviation() {
    let Some(b) = open() else { return };
    let (rows, dim, eps) = (2usize, 5usize, 1e-6f32);
    let x = vals(rows * dim, 25);
    let in_buf = upload_f32(&b, &x);
    let out_buf = alloc(&b, (x.len() * 4) as u64).expect("alloc");
    let args = NormArgs {
        rows: rows as u32,
        dim: dim as u32,
        eps,
    };
    launch(
        &b,
        "kernel_dit_layer_norm_f32",
        Extent3::linear(rows as u32),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(in_buf.view()),
        ],
    );
    let mut expected = vec![0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for (i, &v) in row.iter().enumerate() {
            expected[r * dim + i] = (v - mean) * inv;
        }
    }
    // Without the mean subtraction this is RMSNorm, and it must differ.
    let mut rms = vec![0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for (i, &v) in row.iter().enumerate() {
            rms[r * dim + i] = v * inv;
        }
    }
    assert_differs(&expected, &rms, "layer_norm: mean-subtracting vs rms");
    let got = read_f32(&b, out_buf.view(), x.len());
    assert_close(&got, &expected, "layer_norm");
}

#[test]
fn rmsnorm_scales_the_normalized_row_by_the_weight() {
    let Some(b) = open() else { return };
    let (rows, dim, eps) = (2usize, 4usize, 1e-6f32);
    let x = vals(rows * dim, 26);
    let w = vals(dim, 27);
    let in_buf = upload_f32(&b, &x);
    let w_buf = upload_f32(&b, &w);
    let out_buf = alloc(&b, (x.len() * 4) as u64).expect("alloc");
    let args = NormArgs {
        rows: rows as u32,
        dim: dim as u32,
        eps,
    };
    launch(
        &b,
        "kernel_dit_rmsnorm_f32",
        Extent3::linear(rows as u32),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(in_buf.view()),
            KernelArg::Buffer(w_buf.view()),
        ],
    );
    let mut expected = vec![0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for (i, &v) in row.iter().enumerate() {
            expected[r * dim + i] = v * inv * w[i];
        }
    }
    let got = read_f32(&b, out_buf.view(), x.len());
    assert_close(&got, &expected, "rmsnorm");
}

#[test]
fn qk_norm_normalizes_the_q_and_k_sections_and_leaves_v() {
    let Some(b) = open() else { return };
    let (tokens, heads, head_dim, eps) = (2usize, 2usize, 4usize, 1e-6f32);
    let hidden = heads * head_dim;
    // A fused row is [q | k | v] plus two pad elements, so a kernel that
    // strides by 3*hidden instead of row_stride lands on the wrong values.
    let row_stride = 3 * hidden + 2;
    let x = vals(tokens * row_stride, 28);
    let q_scale = vals(head_dim, 29);
    let k_scale = vals(head_dim, 30);
    let qkv_buf = upload_f32(&b, &x);
    let qs_buf = upload_f32(&b, &q_scale);
    let ks_buf = upload_f32(&b, &k_scale);
    let args = QkNormArgs {
        tokens: tokens as u32,
        heads: heads as u32,
        head_dim: head_dim as u32,
        row_stride: row_stride as u32,
        eps,
    };
    launch(
        &b,
        "kernel_dit_qk_norm_f32",
        Extent3::new(tokens as u32, heads as u32, 1),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(qkv_buf.view()),
            KernelArg::Buffer(qs_buf.view()),
            KernelArg::Buffer(ks_buf.view()),
        ],
    );
    let rms_scale = |row: &[f32], scale: &[f32]| -> Vec<f32> {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        row.iter().zip(scale).map(|(&v, &s)| v * inv * s).collect()
    };
    let mut expected = x.clone();
    for t in 0..tokens {
        for h in 0..heads {
            let q0 = t * row_stride + h * head_dim;
            let q = rms_scale(&x[q0..q0 + head_dim], &q_scale);
            expected[q0..q0 + head_dim].copy_from_slice(&q);
            let k0 = q0 + hidden;
            let k = rms_scale(&x[k0..k0 + head_dim], &k_scale);
            expected[k0..k0 + head_dim].copy_from_slice(&k);
        }
    }
    let got = read_f32(&b, qkv_buf.view(), tokens * row_stride);
    assert_close(&got, &expected, "qk_norm");
    // The v sections and the row pad are untouched.
    for t in 0..tokens {
        let v0 = t * row_stride + 2 * hidden;
        assert_close(
            &got[v0..v0 + head_dim + 2],
            &x[v0..v0 + head_dim + 2],
            "qk_norm: v section and pad untouched",
        );
    }
}

#[test]
fn concat_rows_joins_the_two_row_blocks() {
    let Some(b) = open() else { return };
    let (rows, cols_a, cols_b) = (2usize, 3usize, 2usize);
    let a = vals(rows * cols_a, 31);
    let c = vals(rows * cols_b, 32);
    let a_buf = upload_f32(&b, &a);
    let c_buf = upload_f32(&b, &c);
    let width = cols_a + cols_b;
    let out_buf = alloc(&b, (rows * width * 4) as u64).expect("alloc");
    let args = ConcatArgs {
        rows: rows as u32,
        cols_a: cols_a as u32,
        cols_b: cols_b as u32,
    };
    launch(
        &b,
        "kernel_dit_concat_rows_f32",
        Extent3::linear(rows as u32),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(a_buf.view()),
            KernelArg::Buffer(c_buf.view()),
        ],
    );
    let mut expected = vec![0f32; rows * width];
    for r in 0..rows {
        expected[r * width..r * width + cols_a].copy_from_slice(&a[r * cols_a..(r + 1) * cols_a]);
        expected[r * width + cols_a..(r + 1) * width]
            .copy_from_slice(&c[r * cols_b..(r + 1) * cols_b]);
    }
    let got = read_f32(&b, out_buf.view(), rows * width);
    assert_close(&got, &expected, "concat_rows");
}

// --- attention ----------------------------------------------------------------

/// One joint-attention case: strided fused or packed rows per stream, the
/// same section offsets for every operand, and cached rows at their own
/// stride. `queries[t]` scores every key row and mixes every value row.
struct Joint<'a> {
    q_txt: &'a [f32],
    k_txt: &'a [f32],
    v_txt: &'a [f32],
    q_img: &'a [f32],
    k_img: &'a [f32],
    v_img: &'a [f32],
    cache_k: &'a [f32],
    cache_v: &'a [f32],
    txt: usize,
    img: usize,
    cache: usize,
    heads: usize,
    hd: usize,
    txt_stride: usize,
    img_stride: usize,
    cache_stride: usize,
    q_off: usize,
    k_off: usize,
    v_off: usize,
}

/// The naive joint attention for query row `t` (txt rows first) and head `h`,
/// key order [txt, img, cache], scale 1/sqrt(hd), max-subtracted softmax.
fn expect_joint(c: &Joint<'_>, t: usize, h: usize) -> Vec<f32> {
    let scale = 1.0 / (c.hd as f32).sqrt();
    let (qsrc, qrow) = if t < c.txt {
        (c.q_txt, t * c.txt_stride + c.q_off + h * c.hd)
    } else {
        (c.q_img, (t - c.txt) * c.img_stride + c.q_off + h * c.hd)
    };
    let mut scores = Vec::with_capacity(c.txt + c.img + c.cache);
    for s in 0..(c.txt + c.img) {
        let (ksrc, krow) = if s < c.txt {
            (c.k_txt, s * c.txt_stride + c.k_off + h * c.hd)
        } else {
            (c.k_img, (s - c.txt) * c.img_stride + c.k_off + h * c.hd)
        };
        let dot: f32 = (0..c.hd).map(|d| qsrc[qrow + d] * ksrc[krow + d]).sum();
        scores.push(dot * scale);
    }
    for s in 0..c.cache {
        let krow = s * c.cache_stride + c.k_off + h * c.hd;
        let dot: f32 = (0..c.hd)
            .map(|d| qsrc[qrow + d] * c.cache_k[krow + d])
            .sum();
        scores.push(dot * scale);
    }
    let max = scores.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
    let mut sum = 0f32;
    for v in &mut scores {
        *v = (*v - max).exp();
        sum += *v;
    }
    let mut acc = vec![0f32; c.hd];
    for (s, &p) in scores.iter().enumerate() {
        let (vsrc, vrow) = if s < c.txt {
            (c.v_txt, s * c.txt_stride + c.v_off + h * c.hd)
        } else if s < c.txt + c.img {
            (c.v_img, (s - c.txt) * c.img_stride + c.v_off + h * c.hd)
        } else {
            (
                c.cache_v,
                (s - c.txt - c.img) * c.cache_stride + c.v_off + h * c.hd,
            )
        };
        for d in 0..c.hd {
            acc[d] += p * vsrc[vrow + d];
        }
    }
    acc.iter().map(|v| v / sum).collect()
}

fn run_sdpa(b: &Arc<dyn Backend>, c: &Joint<'_>) -> (Vec<f32>, Vec<f32>) {
    let hidden = c.heads * c.hd;
    // An empty stream binds the driver's stand-in, exactly as `Ops::sdpa`
    // substitutes the dummy for an absent img stream or cache.
    let stream = |v: &[f32]| -> DeviceBuffer {
        if v.is_empty() {
            dummy(b)
        } else {
            upload_f32(b, v)
        }
    };
    let q_txt_buf = stream(c.q_txt);
    let k_txt_buf = stream(c.k_txt);
    let v_txt_buf = stream(c.v_txt);
    let q_img_buf = stream(c.q_img);
    let k_img_buf = stream(c.k_img);
    let v_img_buf = stream(c.v_img);
    let ckey_buf = stream(c.cache_k);
    let cval_buf = stream(c.cache_v);
    let out_txt_buf = alloc(b, (c.txt * hidden * 4) as u64).expect("alloc");
    let out_img_buf = if c.img == 0 {
        dummy(b)
    } else {
        alloc(b, (c.img * hidden * 4) as u64).expect("alloc")
    };
    let args = SdpaArgs {
        txt_tokens: c.txt as u32,
        img_tokens: c.img as u32,
        cache_tokens: c.cache as u32,
        heads: c.heads as u32,
        head_dim: c.hd as u32,
        txt_stride: c.txt_stride as u32,
        img_stride: c.img_stride as u32,
        cache_stride: c.cache_stride as u32,
        q_offset: c.q_off as u32,
        k_offset: c.k_off as u32,
        v_offset: c.v_off as u32,
    };
    launch(
        b,
        "kernel_dit_sdpa_f32",
        Extent3::new((c.txt + c.img) as u32, c.heads as u32, 1),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_txt_buf.view()),
            KernelArg::Buffer(out_img_buf.view()),
            KernelArg::Buffer(q_txt_buf.view()),
            KernelArg::Buffer(k_txt_buf.view()),
            KernelArg::Buffer(v_txt_buf.view()),
            KernelArg::Buffer(q_img_buf.view()),
            KernelArg::Buffer(k_img_buf.view()),
            KernelArg::Buffer(v_img_buf.view()),
            KernelArg::Buffer(ckey_buf.view()),
            KernelArg::Buffer(cval_buf.view()),
        ],
    );
    let out_txt = read_f32(b, out_txt_buf.view(), c.txt * hidden);
    let out_img = if c.img == 0 {
        Vec::new()
    } else {
        read_f32(b, out_img_buf.view(), c.img * hidden)
    };
    (out_txt, out_img)
}

#[test]
fn sdpa_reads_fused_rows_from_their_own_sections() {
    let Some(b) = open() else { return };
    let (heads, hd) = (2usize, 8usize);
    let fused = 3 * heads * hd;
    let (txt, img) = (2usize, 1usize);
    let hidden = heads * hd;
    let txt_rows = vals(txt * fused, 0);
    let img_rows = vals(img * fused, 7);
    let empty: Vec<f32> = Vec::new();
    let case = Joint {
        q_txt: &txt_rows,
        k_txt: &txt_rows,
        v_txt: &txt_rows,
        q_img: &img_rows,
        k_img: &img_rows,
        v_img: &img_rows,
        cache_k: &empty,
        cache_v: &empty,
        txt,
        img,
        cache: 0,
        heads,
        hd,
        txt_stride: fused,
        img_stride: fused,
        cache_stride: fused,
        q_off: 0,
        k_off: hidden,
        v_off: 2 * hidden,
    };
    let (got_txt, got_img) = run_sdpa(&b, &case);
    // Packed row order: row = [head0 | head1].
    let mut expected = Vec::new();
    for t in 0..(txt + img) {
        for h in 0..heads {
            expected.extend(expect_joint(&case, t, h));
        }
    }
    // The old bug: k and v read from the q section.
    let buggy = Joint {
        k_off: 0,
        v_off: 0,
        ..case
    };
    let mut buggy_expected = Vec::new();
    for t in 0..(txt + img) {
        for h in 0..heads {
            buggy_expected.extend(expect_joint(&buggy, t, h));
        }
    }
    assert_differs(
        &expected,
        &buggy_expected,
        "sdpa: true sections vs q section",
    );
    assert_close(&got_txt, &expected[..txt * hidden], "sdpa fused: txt rows");
    assert_close(&got_img, &expected[txt * hidden..], "sdpa fused: img rows");
}

#[test]
fn sdpa_cached_rows_use_the_cache_stride_and_offsets() {
    let Some(b) = open() else { return };
    let (heads, hd) = (2usize, 8usize);
    let fused = 3 * heads * hd;
    let hidden = heads * hd;
    let txt_rows = vals(2 * fused, 0);
    let cache_row = vals(fused, 13);
    let empty: Vec<f32> = Vec::new();
    let case = Joint {
        q_txt: &txt_rows,
        k_txt: &txt_rows,
        v_txt: &txt_rows,
        q_img: &empty,
        k_img: &empty,
        v_img: &empty,
        cache_k: &cache_row,
        cache_v: &cache_row,
        txt: 2,
        img: 0,
        cache: 1,
        heads,
        hd,
        txt_stride: fused,
        img_stride: fused,
        cache_stride: fused,
        q_off: 0,
        k_off: hidden,
        v_off: 2 * hidden,
    };
    let (got_txt, _) = run_sdpa(&b, &case);
    let mut expected = Vec::new();
    for t in 0..2 {
        for h in 0..heads {
            expected.extend(expect_joint(&case, t, h));
        }
    }
    // The buggy reading: the cache row's k and v taken from its q section.
    // Rebuilt here rather than through `Joint` because only the cache operand
    // changes offset, and the joint rows keep their own.
    let mut buggy_expected = Vec::new();
    for t in 0..2 {
        for h in 0..heads {
            let scale = 1.0 / (hd as f32).sqrt();
            let qrow = t * fused + h * hd;
            let mut scores = Vec::new();
            for s in 0..2 {
                let krow = s * fused + hidden + h * hd;
                let dot: f32 = (0..hd)
                    .map(|d| txt_rows[qrow + d] * txt_rows[krow + d])
                    .sum();
                scores.push(dot * scale);
            }
            let dot: f32 = (0..hd)
                .map(|d| txt_rows[qrow + d] * cache_row[h * hd + d])
                .sum();
            scores.push(dot * scale);
            let max = scores.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
            let mut sum = 0f32;
            for v in &mut scores {
                *v = (*v - max).exp();
                sum += *v;
            }
            let mut acc = vec![0f32; hd];
            for (s, &p) in scores.iter().enumerate() {
                if s < 2 {
                    let vrow = s * fused + 2 * hidden + h * hd;
                    for d in 0..hd {
                        acc[d] += p * txt_rows[vrow + d];
                    }
                } else {
                    for d in 0..hd {
                        acc[d] += p * cache_row[h * hd + d];
                    }
                }
            }
            buggy_expected.extend(acc.iter().map(|v| v / sum));
        }
    }
    assert_differs(
        &expected,
        &buggy_expected,
        "sdpa cache: own sections vs q section",
    );
    assert_close(&got_txt, &expected, "sdpa cache: txt rows");
}

#[test]
fn sdpa_packed_rows_with_zero_offsets_match_packed_attention() {
    let Some(b) = open() else { return };
    // The VAE's self-attention shape: separate packed q/k/v tensors, stride
    // = heads * head_dim, all offsets zero.
    let (heads, hd) = (2usize, 4usize);
    let hidden = heads * hd;
    let (txt, img) = (1usize, 2usize);
    let q_txt = vals(txt * hidden, 40);
    let k_txt = vals(txt * hidden, 41);
    let v_txt = vals(txt * hidden, 42);
    let q_img = vals(img * hidden, 43);
    let k_img = vals(img * hidden, 44);
    let v_img = vals(img * hidden, 45);
    let empty: Vec<f32> = Vec::new();
    let case = Joint {
        q_txt: &q_txt,
        k_txt: &k_txt,
        v_txt: &v_txt,
        q_img: &q_img,
        k_img: &k_img,
        v_img: &v_img,
        cache_k: &empty,
        cache_v: &empty,
        txt,
        img,
        cache: 0,
        heads,
        hd,
        txt_stride: hidden,
        img_stride: hidden,
        cache_stride: hidden,
        q_off: 0,
        k_off: 0,
        v_off: 0,
    };
    let (got_txt, got_img) = run_sdpa(&b, &case);
    let mut expected = Vec::new();
    for t in 0..(txt + img) {
        for h in 0..heads {
            expected.extend(expect_joint(&case, t, h));
        }
    }
    assert_close(&got_txt, &expected[..txt * hidden], "sdpa packed: txt row");
    assert_close(&got_img, &expected[txt * hidden..], "sdpa packed: img rows");
}

#[test]
fn gqa_is_causal_grouped_attention() {
    let Some(b) = open() else { return };
    let (tokens, q_heads, kv_heads, hd) = (3usize, 2usize, 1usize, 4usize);
    let q = vals(tokens * q_heads * hd, 46);
    let k = vals(tokens * kv_heads * hd, 47);
    let v = vals(tokens * kv_heads * hd, 48);
    let q_buf = upload_f32(&b, &q);
    let k_buf = upload_f32(&b, &k);
    let v_buf = upload_f32(&b, &v);
    let out_buf = alloc(&b, (q.len() * 4) as u64).expect("alloc");
    let args = GqaArgs {
        tokens: tokens as u32,
        q_heads: q_heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: hd as u32,
    };
    launch(
        &b,
        "kernel_dit_gqa_f32",
        Extent3::new(tokens as u32, q_heads as u32, 1),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(q_buf.view()),
            KernelArg::Buffer(k_buf.view()),
            KernelArg::Buffer(v_buf.view()),
        ],
    );
    let scale = 1.0 / (hd as f32).sqrt();
    let group = q_heads / kv_heads;
    let mut expected = vec![0f32; q.len()];
    for t in 0..tokens {
        for h in 0..q_heads {
            let kvh = h / group;
            let qrow = (t * q_heads + h) * hd;
            let mut scores: Vec<f32> = (0..=t)
                .map(|s| {
                    let krow = (s * kv_heads + kvh) * hd;
                    let dot: f32 = (0..hd).map(|d| q[qrow + d] * k[krow + d]).sum();
                    dot * scale
                })
                .collect();
            let max = scores.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
            let mut sum = 0f32;
            for x in &mut scores {
                *x = (*x - max).exp();
                sum += *x;
            }
            for (s, &p) in scores.iter().enumerate() {
                let vrow = (s * kv_heads + kvh) * hd;
                for d in 0..hd {
                    expected[qrow + d] += p * v[vrow + d];
                }
            }
            for d in 0..hd {
                expected[qrow + d] /= sum;
            }
        }
    }
    // A non-causal reading (every key visible) must differ on the first row.
    let mut noncausal = vec![0f32; q.len()];
    for h in 0..q_heads {
        let kvh = h / group;
        let qrow = h * hd; // token 0
        let mut scores: Vec<f32> = (0..tokens)
            .map(|s| {
                let krow = (s * kv_heads + kvh) * hd;
                let dot: f32 = (0..hd).map(|d| q[qrow + d] * k[krow + d]).sum();
                dot * scale
            })
            .collect();
        let max = scores.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
        let mut sum = 0f32;
        for x in &mut scores {
            *x = (*x - max).exp();
            sum += *x;
        }
        for (s, &p) in scores.iter().enumerate() {
            let vrow = (s * kv_heads + kvh) * hd;
            for d in 0..hd {
                noncausal[qrow + d] += p * v[vrow + d];
            }
        }
        for d in 0..hd {
            noncausal[qrow + d] /= sum;
        }
    }
    assert_differs(
        &expected[..hd],
        &noncausal[..hd],
        "gqa: causal vs non-causal first row",
    );
    let got = read_f32(&b, out_buf.view(), q.len());
    assert_close(&got, &expected, "gqa");
}

// --- rope ----------------------------------------------------------------------

#[test]
fn rope_4d_rotates_each_fused_section_exactly_once() {
    let Some(b) = open() else { return };
    let (tokens, heads, hd, fpa) = (3usize, 2usize, 8usize, 1usize);
    let fused = 3 * heads * hd;
    let hidden = heads * hd;
    let x = vals(tokens * fused, 0);
    let positions = vec![
        0.5, 1.5, -2.0, 3.25, 1.0, -0.5, 2.0, 0.25, -1.75, 3.0, 0.5, 1.25,
    ];
    let freqs = vec![1.0, 0.5, 0.25, 2.0];
    let fused_buf = upload_f32(&b, &x);
    let pos_buf = upload_f32(&b, &positions);
    let freqs_buf = upload_f32(&b, &freqs);
    let args = Rope4dArgs {
        tokens: tokens as u32,
        heads: heads as u32,
        head_dim: hd as u32,
        freqs_per_axis: fpa as u32,
        row_stride: fused as u32,
        q_offset: 0,
        k_offset: hidden as u32,
    };
    launch(
        &b,
        "kernel_dit_rope_4d",
        Extent3::new(tokens as u32, heads as u32, 1),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(fused_buf.view()),
            KernelArg::Buffer(fused_buf.view()),
            KernelArg::Buffer(pos_buf.view()),
            KernelArg::Buffer(freqs_buf.view()),
        ],
    );
    let rotate = |src: &[f32], sections: &[usize]| -> Vec<f32> {
        let mut y = src.to_vec();
        for &section in sections {
            for t in 0..tokens {
                for h in 0..heads {
                    let row = t * fused + section + h * hd;
                    for a in 0..4 {
                        for j in 0..fpa {
                            let angle = positions[t * 4 + a] * freqs[a * fpa + j];
                            let (cos, sin) = (angle.cos(), angle.sin());
                            let i0 = row + a * (hd / 4) + j * 2;
                            let (x0, x1) = (y[i0], y[i0 + 1]);
                            y[i0] = x0 * cos - x1 * sin;
                            y[i0 + 1] = x0 * sin + x1 * cos;
                        }
                    }
                }
            }
        }
        y
    };
    let expected = rotate(&x, &[0, hidden]);
    // The old bug: both passes rotated the q section and neither the k.
    let double_q = rotate(&x, &[0, 0]);
    assert_differs(&expected, &double_q, "rope: q twice vs each once");
    let got = read_f32(&b, fused_buf.view(), tokens * fused);
    assert_close(&got, &expected, "rope_4d fused");
    for t in 0..tokens {
        let v0 = t * fused + 2 * hidden;
        assert_close(
            &got[v0..v0 + hidden],
            &x[v0..v0 + hidden],
            "rope_4d: v section untouched",
        );
    }
}

#[test]
fn rope_4d_rotates_packed_operands_with_several_frequencies() {
    let Some(b) = open() else { return };
    // head_dim 16: axis_dim 4, two interleaved pairs per axis.
    let (tokens, heads, hd, fpa) = (2usize, 2usize, 16usize, 2usize);
    let q = vals(tokens * heads * hd, 50);
    let k = vals(tokens * heads * hd, 51);
    let positions = vec![1.5, -0.5, 2.0, 0.25, -1.25, 3.0, 0.5, -2.5];
    let freqs = vec![1.0, 0.25, 0.5, 0.125, 2.0, 0.75, 0.125, 1.5];
    let q_buf = upload_f32(&b, &q);
    let k_buf = upload_f32(&b, &k);
    let pos_buf = upload_f32(&b, &positions);
    let freqs_buf = upload_f32(&b, &freqs);
    let args = Rope4dArgs {
        tokens: tokens as u32,
        heads: heads as u32,
        head_dim: hd as u32,
        freqs_per_axis: fpa as u32,
        row_stride: (heads * hd) as u32,
        q_offset: 0,
        k_offset: 0,
    };
    launch(
        &b,
        "kernel_dit_rope_4d",
        Extent3::new(tokens as u32, heads as u32, 1),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(q_buf.view()),
            KernelArg::Buffer(k_buf.view()),
            KernelArg::Buffer(pos_buf.view()),
            KernelArg::Buffer(freqs_buf.view()),
        ],
    );
    let rotate = |src: &[f32]| -> Vec<f32> {
        let mut y = src.to_vec();
        for t in 0..tokens {
            for h in 0..heads {
                let row = (t * heads + h) * hd;
                for a in 0..4 {
                    for j in 0..fpa {
                        let angle = positions[t * 4 + a] * freqs[a * fpa + j];
                        let (cos, sin) = (angle.cos(), angle.sin());
                        let i0 = row + a * (hd / 4) + j * 2;
                        let (x0, x1) = (y[i0], y[i0 + 1]);
                        y[i0] = x0 * cos - x1 * sin;
                        y[i0 + 1] = x0 * sin + x1 * cos;
                    }
                }
            }
        }
        y
    };
    let got_q = read_f32(&b, q_buf.view(), q.len());
    let got_k = read_f32(&b, k_buf.view(), k.len());
    assert_close(&got_q, &rotate(&q), "rope_4d packed q");
    assert_close(&got_k, &rotate(&k), "rope_4d packed k");
}

#[test]
fn rope_half_rotates_pairs_across_the_halves() {
    let Some(b) = open() else { return };
    let (tokens, heads, hd) = (3usize, 2usize, 4usize);
    let x = vals(tokens * heads * hd, 52);
    let freqs = vec![0.5, 2.0];
    let x_buf = upload_f32(&b, &x);
    let f_buf = upload_f32(&b, &freqs);
    let args = RopeHalfArgs {
        tokens: tokens as u32,
        heads: heads as u32,
        head_dim: hd as u32,
    };
    launch(
        &b,
        "kernel_dit_rope_half",
        Extent3::new(tokens as u32, heads as u32, 1),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(x_buf.view()),
            KernelArg::Buffer(f_buf.view()),
        ],
    );
    let mut expected = x.clone();
    for t in 0..tokens {
        for h in 0..heads {
            let row = (t * heads + h) * hd;
            for i in 0..hd / 2 {
                let angle = (t as f32) * freqs[i];
                let (cos, sin) = (angle.cos(), angle.sin());
                let x0 = x[row + i];
                let x1 = x[row + i + hd / 2];
                expected[row + i] = x0 * cos - x1 * sin;
                expected[row + i + hd / 2] = x1 * cos + x0 * sin;
            }
        }
    }
    let got = read_f32(&b, x_buf.view(), x.len());
    assert_close(&got, &expected, "rope_half");
}

// --- vae shapes ------------------------------------------------------------------

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one case runs the driver's conv entry twice, padded and strided, each against its own naive reference"
)]
fn conv2d_matches_the_padded_and_the_strided_sum() {
    // Declared before the statements so the case and the two passes below read as one table.
    struct Conv2dCase<'a> {
        input: &'a [f32],
        weight: &'a [f32],
        bias: &'a [f32],
        has_bias: bool,
        batch: usize,
        in_c: usize,
        in_h: usize,
        in_w: usize,
        out_c: usize,
        out_h: usize,
        out_w: usize,
        k_h: usize,
        k_w: usize,
        stride: usize,
        pad: usize,
    }
    let Some(b) = open() else { return };
    let reference = |c: &Conv2dCase<'_>| -> Vec<f32> {
        let mut out = vec![0f32; c.batch * c.out_c * c.out_h * c.out_w];
        for n in 0..c.batch {
            for oc in 0..c.out_c {
                for oh in 0..c.out_h {
                    for ow in 0..c.out_w {
                        let mut acc = if c.has_bias { c.bias[oc] } else { 0.0 };
                        for ic in 0..c.in_c {
                            for kh in 0..c.k_h {
                                for kw in 0..c.k_w {
                                    let ih = oh * c.stride + kh;
                                    let iw = ow * c.stride + kw;
                                    if ih < c.pad
                                        || iw < c.pad
                                        || ih - c.pad >= c.in_h
                                        || iw - c.pad >= c.in_w
                                    {
                                        continue;
                                    }
                                    let xi = ((n * c.in_c + ic) * c.in_h + (ih - c.pad)) * c.in_w
                                        + (iw - c.pad);
                                    let wi = ((oc * c.in_c + ic) * c.k_h + kh) * c.k_w + kw;
                                    acc += c.input[xi] * c.weight[wi];
                                }
                            }
                        }
                        out[((n * c.out_c + oc) * c.out_h + oh) * c.out_w + ow] = acc;
                    }
                }
            }
        }
        out
    };
    let run = |case: &Conv2dCase<'_>, b: &Arc<dyn Backend>| -> Vec<f32> {
        let in_buf = upload_f32(b, case.input);
        let w_buf = upload_f32(b, case.weight);
        // The driver's shape: a real bias bound with the flag set, the
        // 4-byte zero stand-in bound with it clear.
        let bias_buf = if case.has_bias {
            upload_f32(b, case.bias)
        } else {
            dummy(b)
        };
        let out_buf = alloc(
            b,
            (case.batch * case.out_c * case.out_h * case.out_w * 4) as u64,
        )
        .expect("alloc");
        let args = Conv2dArgs {
            batch: case.batch as u32,
            in_c: case.in_c as u32,
            in_h: case.in_h as u32,
            in_w: case.in_w as u32,
            out_c: case.out_c as u32,
            out_h: case.out_h as u32,
            out_w: case.out_w as u32,
            k_h: case.k_h as u32,
            k_w: case.k_w as u32,
            stride: case.stride as u32,
            pad: case.pad as u32,
            has_bias: u32::from(case.has_bias),
        };
        launch(
            b,
            "kernel_dit_conv2d_f32",
            Extent3::new(
                case.out_w as u32,
                case.out_c as u32,
                (case.batch * case.out_h) as u32,
            ),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(out_buf.view()),
                KernelArg::Buffer(in_buf.view()),
                KernelArg::Buffer(w_buf.view()),
                KernelArg::Buffer(bias_buf.view()),
            ],
        );
        read_f32(
            b,
            out_buf.view(),
            case.batch * case.out_c * case.out_h * case.out_w,
        )
    };
    // 3x3, pad 1, stride 1: the padded border must read as zero.
    let case1 = Conv2dCase {
        input: &vals(2 * 3 * 3, 53),
        weight: &vals(2 * 2 * 3 * 3, 54),
        bias: &vals(2, 55),
        has_bias: true,
        batch: 1,
        in_c: 2,
        in_h: 3,
        in_w: 3,
        out_c: 2,
        out_h: 3,
        out_w: 3,
        k_h: 3,
        k_w: 3,
        stride: 1,
        pad: 1,
    };
    let got = run(&case1, &b);
    assert_close(&got, &reference(&case1), "conv2d pad 1 stride 1");
    // 3x3, pad 0, stride 2: the strided sampling must skip inputs.
    let case2 = Conv2dCase {
        input: &vals(2 * 4 * 4, 56),
        weight: &vals(3 * 2 * 3 * 3, 57),
        bias: &vals(3, 58),
        has_bias: true,
        batch: 1,
        in_c: 2,
        in_h: 4,
        in_w: 4,
        out_c: 3,
        out_h: 2,
        out_w: 2,
        k_h: 3,
        k_w: 3,
        stride: 2,
        pad: 0,
    };
    let got = run(&case2, &b);
    assert_close(&got, &reference(&case2), "conv2d pad 0 stride 2");
}

/// Runs conv2d with a fixed 1x1-kernel shape (`in_c` 1, `out_c` 2, 2x2
/// spatial) for a given bias binding and presence flag, returning the
/// output with the fixed input and weight so the caller can compute
/// `out[oc][p] = in[p] * w[oc] + bias[oc]` (or without the bias term).
fn conv2d_bias_case(
    b: &Arc<dyn Backend>,
    bias: DeviceView<'_>,
    has_bias: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let input = vals(4, 61);
    let weight = vec![0.75f32, -1.25];
    let in_buf = upload_f32(b, &input);
    let w_buf = upload_f32(b, &weight);
    let out_buf = alloc(b, 8 * 4).expect("alloc");
    let args = Conv2dArgs {
        batch: 1,
        in_c: 1,
        in_h: 2,
        in_w: 2,
        out_c: 2,
        out_h: 2,
        out_w: 2,
        k_h: 1,
        k_w: 1,
        stride: 1,
        pad: 0,
        has_bias: u32::from(has_bias),
    };
    launch(
        b,
        "kernel_dit_conv2d_f32",
        Extent3::new(2, 2, 2),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(in_buf.view()),
            KernelArg::Buffer(w_buf.view()),
            KernelArg::Buffer(bias),
        ],
    );
    (read_f32(b, out_buf.view(), 8), input, weight)
}

fn expect_conv2d_bias(input: &[f32], weight: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    (0..2)
        .flat_map(|oc| {
            input
                .iter()
                .map(move |&x| x * weight[oc] + bias.map_or(0.0, |b| b[oc]))
        })
        .collect()
}

/// The case the zero sentinel silently got wrong, on the conv path: a
/// real bias whose first element is exactly +0.0, bound with the presence
/// flag set, must be applied in full.
#[test]
fn conv2d_applies_a_bias_whose_first_word_is_positive_zero() {
    let Some(b) = open() else { return };
    let bias = vec![0.0f32, 0.5];
    let bias_buf = upload_f32(&b, &bias);
    let (got, input, weight) = conv2d_bias_case(&b, bias_buf.view(), true);
    assert_differs(
        &got,
        &expect_conv2d_bias(&input, &weight, None),
        "conv2d positive-zero-first bias",
    );
    assert_close(
        &got,
        &expect_conv2d_bias(&input, &weight, Some(&bias)),
        "conv2d positive-zero-first bias",
    );
}

/// The flag wins over the binding on the conv path too, in both shapes the
/// driver can present: the 4-byte zero stand-in bound with the flag clear
/// (the binding `Ops` actually makes for an absent bias), and a real,
/// nonzero bias bound with the flag clear. The stand-in case alone cannot
/// tell "the flag was honoured" from "the flag was ignored and the bound
/// zeros were added"; the real-bias case can, exactly as the GEMM path's
/// `gemm_bf16_ignores_the_bias_binding_when_the_flag_is_clear` does.
#[test]
fn conv2d_ignores_bias_when_the_flag_is_clear() {
    let Some(b) = open() else { return };
    let stand_in = dummy(&b);
    let (got, input, weight) = conv2d_bias_case(&b, stand_in.view(), false);
    assert_differs(
        &got,
        &expect_conv2d_bias(&input, &weight, Some(&[0.25, -0.5])),
        "conv2d flag-clear bias",
    );
    assert_close(
        &got,
        &expect_conv2d_bias(&input, &weight, None),
        "conv2d flag-clear bias",
    );
    // The discriminating shape: a real, nonzero bias bound with the flag
    // clear must not be read.
    let bias = vec![0.25f32, -0.5];
    let bias_buf = upload_f32(&b, &bias);
    let (got, input, weight) = conv2d_bias_case(&b, bias_buf.view(), false);
    assert_differs(
        &got,
        &expect_conv2d_bias(&input, &weight, Some(&bias)),
        "conv2d flag-clear real bias",
    );
    assert_close(
        &got,
        &expect_conv2d_bias(&input, &weight, None),
        "conv2d flag-clear real bias",
    );
}

#[test]
fn group_norm_normalizes_each_group_and_affines_by_channel() {
    let Some(b) = open() else { return };
    let (batch, groups, cpg, spatial, eps) = (2usize, 2usize, 2usize, 37usize, 1e-5f32);
    let channels = groups * cpg;
    let per_group = cpg * spatial;
    let x = vals(batch * groups * per_group, 59);
    // Affine parameters are per channel: `channels` elements, not one per
    // group element. Distinct values so a group-wise index lands elsewhere.
    let w = (0..channels)
        .map(|i| 1.0 + (i as f32) * 0.5)
        .collect::<Vec<f32>>();
    let bias = (0..channels)
        .map(|i| -0.25 * (i as f32))
        .collect::<Vec<f32>>();
    let mut expected = vec![0f32; x.len()];
    for g in 0..(batch * groups) {
        let base = g * per_group;
        let mean: f32 = x[base..base + per_group].iter().sum::<f32>() / per_group as f32;
        let var: f32 = x[base..base + per_group]
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f32>()
            / per_group as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..per_group {
            let ch = (g % groups) * cpg + i / spatial;
            expected[base + i] = (x[base + i] - mean) * inv * w[ch] + bias[ch];
        }
    }
    let in_buf = upload_f32(&b, &x);
    let w_buf = upload_f32(&b, &w);
    let bias_buf = upload_f32(&b, &bias);
    // One slice (the whole group), ragged slices across a channel boundary,
    // and more slices than the group can fill (25 slices of 3 over 74
    // elements leave the last one empty).
    for chunks in [1usize, 3, 25] {
        let out_buf = alloc(&b, (x.len() * 4) as u64).expect("alloc");
        let partials = alloc(&b, (batch * groups * chunks * 2 * 4) as u64).expect("alloc");
        let args = GroupNormArgs {
            batch: batch as u32,
            num_groups: groups as u32,
            channels_per_group: cpg as u32,
            spatial: spatial as u32,
            eps,
            chunks: chunks as u32,
        };
        let grid = Extent3::new((batch * groups) as u32, chunks as u32, 1);
        launch(
            &b,
            "kernel_dit_group_norm_stats_f32",
            grid,
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(partials.view()),
                KernelArg::Buffer(in_buf.view()),
            ],
        );
        launch(
            &b,
            "kernel_dit_group_norm_apply_f32",
            grid,
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(out_buf.view()),
                KernelArg::Buffer(in_buf.view()),
                KernelArg::Buffer(w_buf.view()),
                KernelArg::Buffer(bias_buf.view()),
                KernelArg::Buffer(partials.view()),
            ],
        );
        let got = read_f32(&b, out_buf.view(), x.len());
        assert_close(&got, &expected, "group_norm");
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SoftmaxArgs {
    rows: u32,
    cols: u32,
    scale: f32,
}

#[test]
fn softmax_rows_normalizes_each_scaled_row_in_place() {
    let Some(b) = open() else { return };
    // 70 columns: more than one pass of the 64-thread block, ragged.
    let (rows, cols, scale) = (3usize, 70usize, 0.3f32);
    let x: Vec<f32> = vals(rows * cols, 71).iter().map(|v| v * 20.0).collect();
    let buf = upload_f32(&b, &x);
    let args = SoftmaxArgs {
        rows: rows as u32,
        cols: cols as u32,
        scale,
    };
    launch(
        &b,
        "kernel_dit_softmax_rows_f32",
        Extent3::linear(rows as u32),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(buf.view()),
        ],
    );
    let mut expected = Vec::with_capacity(x.len());
    for row in x.chunks_exact(cols) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = row.iter().map(|v| ((v - max) * scale).exp()).collect();
        let sum: f32 = e.iter().sum();
        expected.extend(e.iter().map(|v| v / sum));
    }
    let got = read_f32(&b, buf.view(), x.len());
    assert_close(&got, &expected, "softmax_rows");
}

#[test]
#[expect(
    clippy::many_single_char_names,
    reason = "the four-axis grid walk reads clearer with h/w/i/j than with long names"
)]
fn upsample_nearest_doubles_the_field() {
    let Some(b) = open() else { return };
    let (batch, channels, h, w) = (1usize, 2usize, 2usize, 3usize);
    let x = vals(batch * channels * h * w, 60);
    let in_buf = upload_f32(&b, &x);
    let out_buf = alloc(&b, (batch * channels * 2 * h * 2 * w * 4) as u64).expect("alloc");
    let args = UpsampleArgs {
        batch: batch as u32,
        channels: channels as u32,
        h: h as u32,
        w: w as u32,
    };
    launch(
        &b,
        "kernel_dit_upsample_nearest_f32",
        Extent3::new(w as u32, h as u32, (batch * channels) as u32),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(in_buf.view()),
        ],
    );
    let mut expected = vec![0f32; batch * channels * 2 * h * 2 * w];
    for n in 0..batch {
        for c in 0..channels {
            for i in 0..h {
                for j in 0..w {
                    let v = x[((n * channels + c) * h + i) * w + j];
                    for di in 0..2 {
                        for dj in 0..2 {
                            expected
                                [((n * channels + c) * 2 * h + 2 * i + di) * 2 * w + 2 * j + dj] =
                                v;
                        }
                    }
                }
            }
        }
    }
    let got = read_f32(&b, out_buf.view(), expected.len());
    assert_close(&got, &expected, "upsample_nearest");
}

#[test]
fn transpose_swaps_the_two_axes() {
    let Some(b) = open() else { return };
    let (rows, cols) = (2usize, 3usize);
    let x = vals(rows * cols, 61);
    let in_buf = upload_f32(&b, &x);
    let out_buf = alloc(&b, (x.len() * 4) as u64).expect("alloc");
    let args = NormArgs {
        rows: rows as u32,
        dim: cols as u32,
        eps: 0.0,
    };
    launch(
        &b,
        "kernel_dit_transpose_f32",
        Extent3::linear(grid_for((rows * cols) as u32, BLOCK)),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(in_buf.view()),
        ],
    );
    let mut expected = vec![0f32; x.len()];
    for r in 0..rows {
        for c in 0..cols {
            expected[c * rows + r] = x[r * cols + c];
        }
    }
    let got = read_f32(&b, out_buf.view(), x.len());
    assert_close(&got, &expected, "transpose");
}

#[test]
fn embed_gathers_bf16_table_rows_as_f32() {
    let Some(b) = open() else { return };
    let (tokens, hidden, vocab) = (3usize, 3usize, 4usize);
    let table: Vec<f32> = (0..vocab * hidden)
        .map(|i| ((i % 9) as f32) * 0.25 - 1.0)
        .collect();
    let ids = vec![2u32, 0, 3];
    let table_buf = upload_u16(
        &b,
        &table
            .iter()
            .map(|&v| u16::from_le_bytes(bf16::from_f32(v).to_le_bytes()))
            .collect::<Vec<u16>>(),
    );
    let ids_buf = upload_u32(&b, &ids);
    let out_buf = alloc(&b, (tokens * hidden * 4) as u64).expect("alloc");
    let args = EmbedArgs {
        tokens: tokens as u32,
        hidden: hidden as u32,
    };
    launch(
        &b,
        "kernel_dit_embed_bf16_f32",
        Extent3::new(tokens as u32, grid_for(hidden as u32, BLOCK), 1),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(table_buf.view()),
            KernelArg::Buffer(ids_buf.view()),
        ],
    );
    let mut expected = Vec::with_capacity(tokens * hidden);
    for &id in &ids {
        for d in 0..hidden {
            expected.push(bf16::from_f32(table[id as usize * hidden + d]).to_f32());
        }
    }
    let got = read_f32(&b, out_buf.view(), tokens * hidden);
    assert_close(&got, &expected, "embed_bf16");
}

// --- simdgroup tiles: gemm64, implicit-gemm conv, fused attention -----------

/// One simdgroup-tile GEMM launch with the driver's geometry; `shape` is
/// `(m, n, k)` with `k % 16 == 0`.
fn run_gemm64(b: &Arc<dyn Backend>, run: GemmRun<'_>) -> Vec<f32> {
    let (m, n, k) = run.shape;
    let in_buf = upload_f32(b, run.input);
    let out_buf = alloc(b, (m * n * 4) as u64).expect("alloc out");
    let args = GemmArgs {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        has_bias: u32::from(run.has_bias),
    };
    launch(
        b,
        run.entry,
        Extent3::new(
            grid_for(n as u32, G64_TILE),
            grid_for(m as u32, G64_TILE),
            1,
        ),
        Extent3::linear(G64_THREADS),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(in_buf.view()),
            KernelArg::Buffer(run.weight),
            KernelArg::F32(run.scale),
            KernelArg::Buffer(run.bias),
        ],
    );
    read_f32(b, out_buf.view(), m * n)
}

/// Shapes that leave partial 64x64 tiles in both axes and cross several
/// 16-wide k slices, so the edge zero-fill and the slice prefetch both run.
const G64_SHAPE: (usize, usize, usize) = (70, 67, 48);

#[test]
fn gemm64_f32_matches_the_transposed_product_on_ragged_tiles() {
    let Some(b) = open() else { return };
    let (m, n, k) = G64_SHAPE;
    let input = vals(m * k, 61);
    let weight = vals(n * k, 62);
    let bias = vals(n, 63);
    let w_buf = upload_f32(&b, &weight);
    let bias_buf = upload_f32(&b, &bias);
    let got = run_gemm64(
        &b,
        GemmRun {
            entry: "kernel_dit_gemm64_f32",
            weight: w_buf.view(),
            scale: 1.0,
            bias: bias_buf.view(),
            has_bias: true,
            input: &input,
            shape: (m, n, k),
        },
    );
    let expected = expect_gemm(&input, &weight, &bias, 1.0, m, n, k);
    assert_close(&got, &expected, "gemm64_f32");
}

#[test]
fn gemm64_bf16_widens_the_weight_and_skips_an_unflagged_bias() {
    let Some(b) = open() else { return };
    let (m, n, k) = G64_SHAPE;
    let input = vals(m * k, 64);
    // Multiples of 1/8 in a small range are exact in bf16.
    let weight: Vec<f32> = vals(n * k, 65);
    let bits: Vec<u16> = weight
        .iter()
        .map(|&v| bf16::from_f32(v).to_bits())
        .collect();
    let w_buf = upload_u16(&b, &bits);
    let stand_in = dummy(&b);
    let zero = vec![0f32; n];
    let got = run_gemm64(
        &b,
        GemmRun {
            entry: "kernel_dit_gemm64_bf16",
            weight: w_buf.view(),
            scale: 0.5,
            bias: stand_in.view(),
            has_bias: false,
            input: &input,
            shape: (m, n, k),
        },
    );
    let expected = expect_gemm(&input, &weight, &zero, 0.5, m, n, k);
    assert_close(&got, &expected, "gemm64_bf16");
}

#[test]
fn gemm64_fp8_decodes_every_e4m3_byte_and_scales_the_accumulator() {
    let Some(b) = open() else { return };
    let (m, n, k) = G64_SHAPE;
    let input: Vec<f32> = vals(m * k, 66).iter().map(|v| v * 0.01).collect();
    // Every byte value except the two NaN patterns, so the branch-free
    // decode is checked against the reference decode across the whole code.
    let bytes: Vec<u8> = (0..n * k)
        .map(|i| {
            let byte = u8::try_from((i * 7) % 256).unwrap_or(0);
            if byte & 0x7F == 0x7F { 0x11 } else { byte }
        })
        .collect();
    let weight: Vec<f32> = bytes.iter().map(|&v| e4m3(v)).collect();
    let bias = vals(n, 67);
    let w_buf = upload_u8(&b, &bytes);
    let bias_buf = upload_f32(&b, &bias);
    let got = run_gemm64(
        &b,
        GemmRun {
            entry: "kernel_dit_gemm64_fp8",
            weight: w_buf.view(),
            scale: 0.25,
            bias: bias_buf.view(),
            has_bias: true,
            input: &input,
            shape: (m, n, k),
        },
    );
    let expected = expect_gemm(&input, &weight, &bias, 0.25, m, n, k);
    assert_close(&got, &expected, "gemm64_fp8");
}

/// The reference E4M3 decode: 1 sign, 4 exponent (bias 7), 3 mantissa.
fn e4m3(byte: u8) -> f32 {
    let e = i32::from((byte >> 3) & 0xF);
    let m = f32::from(byte & 7);
    let v = if e == 0 {
        m * 2f32.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    if byte & 0x80 == 0 { v } else { -v }
}

/// One NCHW convolution case for the implicit-GEMM entry.
struct ConvCase {
    batch: usize,
    in_c: usize,
    in_h: usize,
    in_w: usize,
    out_c: usize,
    k: usize,
    stride: usize,
    pad: usize,
}

impl ConvCase {
    fn out_h(&self) -> usize {
        (self.in_h + 2 * self.pad - self.k) / self.stride + 1
    }
    fn out_w(&self) -> usize {
        (self.in_w + 2 * self.pad - self.k) / self.stride + 1
    }
}

fn expect_conv(c: &ConvCase, input: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let (out_rows, out_cols) = (c.out_h(), c.out_w());
    let mut out = vec![0f32; c.batch * c.out_c * out_rows * out_cols];
    for n in 0..c.batch {
        for oc in 0..c.out_c {
            for oh in 0..out_rows {
                for ow in 0..out_cols {
                    let mut acc = 0f32;
                    for ic in 0..c.in_c {
                        for kh in 0..c.k {
                            for kw in 0..c.k {
                                let ih = oh * c.stride + kh;
                                let iw = ow * c.stride + kw;
                                if ih < c.pad
                                    || iw < c.pad
                                    || ih - c.pad >= c.in_h
                                    || iw - c.pad >= c.in_w
                                {
                                    continue;
                                }
                                acc += input[((n * c.in_c + ic) * c.in_h + ih - c.pad) * c.in_w
                                    + iw
                                    - c.pad]
                                    * weight[((oc * c.in_c + ic) * c.k + kh) * c.k + kw];
                            }
                        }
                    }
                    out[((n * c.out_c + oc) * out_rows + oh) * out_cols + ow] = acc + bias[oc];
                }
            }
        }
    }
    out
}

fn run_conv_gemm(b: &Arc<dyn Backend>, c: &ConvCase, seed: usize) -> (Vec<f32>, Vec<f32>) {
    let input: Vec<f32> = vals(c.batch * c.in_c * c.in_h * c.in_w, seed)
        .iter()
        .map(|v| v * 0.1)
        .collect();
    let weight: Vec<f32> = vals(c.out_c * c.in_c * c.k * c.k, seed + 1)
        .iter()
        .map(|v| v * 0.1)
        .collect();
    let bias = vals(c.out_c, seed + 2);
    let (oh, ow) = (c.out_h(), c.out_w());
    let in_buf = upload_f32(b, &input);
    let w_buf = upload_f32(b, &weight);
    let bias_buf = upload_f32(b, &bias);
    let out_buf = alloc(b, (c.batch * c.out_c * oh * ow * 4) as u64).expect("alloc");
    let args = Conv2dArgs {
        batch: c.batch as u32,
        in_c: c.in_c as u32,
        in_h: c.in_h as u32,
        in_w: c.in_w as u32,
        out_c: c.out_c as u32,
        out_h: oh as u32,
        out_w: ow as u32,
        k_h: c.k as u32,
        k_w: c.k as u32,
        stride: c.stride as u32,
        pad: c.pad as u32,
        has_bias: 1,
    };
    launch(
        b,
        "kernel_dit_conv2d_gemm_f32",
        Extent3::new(
            grid_for(c.out_c as u32, G64_TILE),
            grid_for((oh * ow) as u32, G64_TILE),
            c.batch as u32,
        ),
        Extent3::linear(G64_THREADS),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(in_buf.view()),
            KernelArg::Buffer(w_buf.view()),
            KernelArg::Buffer(bias_buf.view()),
        ],
    );
    let got = read_f32(b, out_buf.view(), c.batch * c.out_c * oh * ow);
    (got, expect_conv(c, &input, &weight, &bias))
}

#[test]
fn conv2d_gemm_matches_the_padded_3x3_sum_across_batches() {
    let Some(b) = open() else { return };
    // 16 input channels x 9 taps = 144, a multiple of the k slice; 70 output
    // channels and 121 pixels leave partial tiles in both axes.
    let case = ConvCase {
        batch: 2,
        in_c: 16,
        in_h: 11,
        in_w: 11,
        out_c: 70,
        k: 3,
        stride: 1,
        pad: 1,
    };
    let (got, expected) = run_conv_gemm(&b, &case, 71);
    assert_close(&got, &expected, "conv2d_gemm 3x3 pad 1");
}

#[test]
fn conv2d_gemm_matches_the_strided_and_the_pointwise_sum() {
    let Some(b) = open() else { return };
    let strided = ConvCase {
        batch: 1,
        in_c: 16,
        in_h: 9,
        in_w: 7,
        out_c: 5,
        k: 3,
        stride: 2,
        pad: 0,
    };
    let (got, expected) = run_conv_gemm(&b, &strided, 74);
    assert_close(&got, &expected, "conv2d_gemm 3x3 stride 2");
    let pointwise = ConvCase {
        batch: 1,
        in_c: 32,
        in_h: 5,
        in_w: 13,
        out_c: 9,
        k: 1,
        stride: 1,
        pad: 0,
    };
    let (got, expected) = run_conv_gemm(&b, &pointwise, 77);
    assert_close(&got, &expected, "conv2d_gemm 1x1");
}

/// Launches the fused attention with the driver's geometry.
fn run_flash(b: &Arc<dyn Backend>, c: &Joint<'_>) -> (Vec<f32>, Vec<f32>) {
    let hidden = c.heads * c.hd;
    let stream = |v: &[f32]| -> DeviceBuffer {
        if v.is_empty() {
            dummy(b)
        } else {
            upload_f32(b, v)
        }
    };
    let bufs: Vec<DeviceBuffer> = [
        c.q_txt, c.k_txt, c.v_txt, c.q_img, c.k_img, c.v_img, c.cache_k, c.cache_v,
    ]
    .iter()
    .map(|v| stream(v))
    .collect();
    let out_txt_buf = alloc(b, (c.txt * hidden * 4) as u64).expect("alloc");
    let out_img_buf = alloc(b, (c.img * hidden * 4) as u64).expect("alloc");
    let args = SdpaArgs {
        txt_tokens: c.txt as u32,
        img_tokens: c.img as u32,
        cache_tokens: c.cache as u32,
        heads: c.heads as u32,
        head_dim: c.hd as u32,
        txt_stride: c.txt_stride as u32,
        img_stride: c.img_stride as u32,
        cache_stride: c.cache_stride as u32,
        q_offset: c.q_off as u32,
        k_offset: c.k_off as u32,
        v_offset: c.v_off as u32,
    };
    let mut kargs = vec![
        KernelArg::Bytes(bytes_of(&args)),
        KernelArg::Buffer(out_txt_buf.view()),
        KernelArg::Buffer(out_img_buf.view()),
    ];
    kargs.extend(bufs.iter().map(|buf| KernelArg::Buffer(buf.view())));
    launch(
        b,
        "kernel_dit_flash_sdpa_f32",
        Extent3::new(
            grid_for((c.txt + c.img) as u32, FA_QROWS),
            c.heads as u32,
            1,
        ),
        Extent3::linear(FA_THREADS),
        &kargs,
    );
    (
        read_f32(b, out_txt_buf.view(), c.txt * hidden),
        read_f32(b, out_img_buf.view(), c.img * hidden),
    )
}

fn expect_all(c: &Joint<'_>) -> Vec<f32> {
    let mut expected = Vec::new();
    for t in 0..(c.txt + c.img) {
        for h in 0..c.heads {
            expected.extend(expect_joint(c, t, h));
        }
    }
    expected
}

#[test]
fn flash_sdpa_matches_joint_attention_over_fused_rows_and_a_cache() {
    let Some(b) = open() else { return };
    // The transformer's head width; 5 txt + 40 img rows straddle two query
    // tiles, and 45 + 19 keys leave a partial 16-key block at the end.
    let (heads, hd) = (2usize, 128usize);
    let hidden = heads * hd;
    let fused = 3 * hidden;
    let (txt, img, cache) = (5usize, 40usize, 19usize);
    let small =
        |n: usize, seed: usize| -> Vec<f32> { vals(n, seed).iter().map(|v| v * 0.05).collect() };
    let txt_rows = small(txt * fused, 81);
    let img_rows = small(img * fused, 82);
    let cache_rows = small(cache * fused, 83);
    let case = Joint {
        q_txt: &txt_rows,
        k_txt: &txt_rows,
        v_txt: &txt_rows,
        q_img: &img_rows,
        k_img: &img_rows,
        v_img: &img_rows,
        cache_k: &cache_rows,
        cache_v: &cache_rows,
        txt,
        img,
        cache,
        heads,
        hd,
        txt_stride: fused,
        img_stride: fused,
        cache_stride: fused,
        q_off: 0,
        k_off: hidden,
        v_off: 2 * hidden,
    };
    let expected = expect_all(&case);
    let (got_txt, got_img) = run_flash(&b, &case);
    assert_close_half(&got_txt, &expected[..txt * hidden], "flash sdpa: txt rows");
    assert_close_half(&got_img, &expected[txt * hidden..], "flash sdpa: img rows");
}

#[test]
fn flash_sdpa_matches_the_per_row_kernel_on_packed_narrow_heads() {
    let Some(b) = open() else { return };
    // A 64-wide head (fewer fragments than the maximum) over packed rows.
    let (heads, hd) = (3usize, 64usize);
    let hidden = heads * hd;
    let (txt, img) = (3usize, 30usize);
    let small =
        |n: usize, seed: usize| -> Vec<f32> { vals(n, seed).iter().map(|v| v * 0.1).collect() };
    let (qt, kt, vt) = (
        small(txt * hidden, 91),
        small(txt * hidden, 92),
        small(txt * hidden, 93),
    );
    let (qi, ki, vi) = (
        small(img * hidden, 94),
        small(img * hidden, 95),
        small(img * hidden, 96),
    );
    let empty: Vec<f32> = Vec::new();
    let case = Joint {
        q_txt: &qt,
        k_txt: &kt,
        v_txt: &vt,
        q_img: &qi,
        k_img: &ki,
        v_img: &vi,
        cache_k: &empty,
        cache_v: &empty,
        txt,
        img,
        cache: 0,
        heads,
        hd,
        txt_stride: hidden,
        img_stride: hidden,
        cache_stride: hidden,
        q_off: 0,
        k_off: 0,
        v_off: 0,
    };
    let (flash_txt, flash_img) = run_flash(&b, &case);
    let (row_txt, row_img) = run_sdpa(&b, &case);
    assert_close_half(&flash_txt, &row_txt, "flash vs per-row: txt rows");
    assert_close_half(&flash_img, &row_img, "flash vs per-row: img rows");
    let expected = expect_all(&case);
    assert_close_half(
        &flash_img,
        &expected[txt * hidden..],
        "flash sdpa packed: img rows",
    );
}

/// Runs the fused attention over one packed-fused stream (no img rows, no
/// cache), as the single blocks call it.
fn run_flash_joint(b: &Arc<dyn Backend>, c: &Joint<'_>) -> Vec<f32> {
    let hidden = c.heads * c.hd;
    let qkv = upload_f32(b, c.q_txt);
    let stand_in = dummy(b);
    let out = alloc(b, (c.txt * hidden * 4) as u64).expect("alloc");
    let args = SdpaArgs {
        txt_tokens: c.txt as u32,
        img_tokens: 0,
        cache_tokens: 0,
        heads: c.heads as u32,
        head_dim: c.hd as u32,
        txt_stride: c.txt_stride as u32,
        img_stride: c.img_stride as u32,
        cache_stride: c.cache_stride as u32,
        q_offset: c.q_off as u32,
        k_offset: c.k_off as u32,
        v_offset: c.v_off as u32,
    };
    launch(
        b,
        "kernel_dit_flash_sdpa_f32",
        Extent3::new(grid_for(c.txt as u32, FA_QROWS), c.heads as u32, 1),
        Extent3::linear(FA_THREADS),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(qkv.view()),
            KernelArg::Buffer(qkv.view()),
            KernelArg::Buffer(qkv.view()),
            KernelArg::Buffer(stand_in.view()),
            KernelArg::Buffer(stand_in.view()),
            KernelArg::Buffer(stand_in.view()),
            KernelArg::Buffer(stand_in.view()),
            KernelArg::Buffer(stand_in.view()),
        ],
    );
    read_f32(b, out.view(), c.txt * hidden)
}

fn assert_bit_identical(a: &[f32], b: &[f32], what: &str) {
    let differing = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    assert_eq!(
        differing, 0,
        "{what}: {differing} values differ between runs"
    );
}

/// A fixed seed must give the same image on every run, so the simdgroup
/// kernels must give the same bits on every launch. Run at the transformer's
/// own sizes (1046 joint rows, 24 heads of 128): a race between lane stores
/// and fragment loads in the fused attention showed up there - one row in
/// eight occasionally kept a stale rescale - and not on the small cases.
#[test]
fn flash_sdpa_and_gemm64_are_bit_identical_across_runs_at_model_sizes() {
    let Some(b) = open() else { return };
    let (heads, hd) = (24usize, 128usize);
    let hidden = heads * hd;
    let fused = 3 * hidden;
    let rows = 1046usize;
    let data: Vec<f32> = vals(rows * fused, 5).iter().map(|v| v * 0.03).collect();
    let empty: Vec<f32> = Vec::new();
    let case = Joint {
        q_txt: &data,
        k_txt: &data,
        v_txt: &data,
        q_img: &empty,
        k_img: &empty,
        v_img: &empty,
        cache_k: &empty,
        cache_v: &empty,
        txt: rows,
        img: 0,
        cache: 0,
        heads,
        hd,
        txt_stride: fused,
        img_stride: fused,
        cache_stride: fused,
        q_off: 0,
        k_off: hidden,
        v_off: 2 * hidden,
    };
    let first = run_flash_joint(&b, &case);
    for _ in 0..5 {
        assert_bit_identical(&first, &run_flash_joint(&b, &case), "flash sdpa");
    }

    let (m, n, k) = (1046usize, 3072usize, 3072usize);
    let input: Vec<f32> = vals(m * k, 9).iter().map(|v| v * 0.01).collect();
    let weight: Vec<f32> = vals(n * k, 10).iter().map(|v| v * 0.01).collect();
    let w_buf = upload_f32(&b, &weight);
    let stand_in = dummy(&b);
    let run = || {
        run_gemm64(
            &b,
            GemmRun {
                entry: "kernel_dit_gemm64_f32",
                weight: w_buf.view(),
                scale: 1.0,
                bias: stand_in.view(),
                has_bias: false,
                input: &input,
                shape: (m, n, k),
            },
        )
    };
    let first = run();
    for _ in 0..5 {
        assert_bit_identical(&first, &run(), "gemm64");
    }
}

/// `residual + gate * (x . w^T * scale + bias)` with `x = [a | silu(g) * u]`.
struct FusedCase<'a> {
    residual: &'a [f32],
    a: &'a [f32],
    gu: &'a [f32],
    weight: &'a [f32],
    bias: &'a [f32],
    gate: &'a [f32],
    scale: f32,
    m: usize,
    n: usize,
    cols_a: usize,
    inner: usize,
    gu_stride: usize,
    gu_offset: usize,
}

fn expect_fused(c: &FusedCase<'_>) -> Vec<f32> {
    let k = c.cols_a + c.inner;
    let mut out = c.residual.to_vec();
    for r in 0..c.m {
        let mut x = Vec::with_capacity(k);
        x.extend_from_slice(&c.a[r * c.cols_a..(r + 1) * c.cols_a]);
        let row = &c.gu[r * c.gu_stride + c.gu_offset..];
        x.extend((0..c.inner).map(|i| silu(row[i]) * row[c.inner + i]));
        for col in 0..c.n {
            let dot: f32 = (0..k).map(|kk| x[kk] * c.weight[col * k + kk]).sum();
            out[r * c.n + col] += c.gate[col] * (dot * c.scale + c.bias[col]);
        }
    }
    out
}

fn run_fused(
    b: &Arc<dyn Backend>,
    c: &FusedCase<'_>,
    entry: &'static str,
    weight: DeviceView<'_>,
) -> Vec<f32> {
    let res_buf = upload_f32(b, c.residual);
    let a_buf = if c.cols_a == 0 {
        dummy(b)
    } else {
        upload_f32(b, c.a)
    };
    let gu_buf = if c.inner == 0 {
        dummy(b)
    } else {
        upload_f32(b, c.gu)
    };
    let bias_buf = upload_f32(b, c.bias);
    let gate_buf = upload_f32(b, c.gate);
    let args = GemmFusedArgs {
        m: c.m as u32,
        n: c.n as u32,
        cols_a: c.cols_a as u32,
        inner: c.inner as u32,
        gu_stride: c.gu_stride as u32,
        gu_offset: c.gu_offset as u32,
        has_bias: 1,
    };
    launch(
        b,
        entry,
        Extent3::new(
            grid_for(c.n as u32, G64_TILE),
            grid_for(c.m as u32, G64_TILE),
            1,
        ),
        Extent3::linear(G64_THREADS),
        &[
            KernelArg::Bytes(bytes_of(&args)),
            KernelArg::Buffer(res_buf.view()),
            KernelArg::Buffer(a_buf.view()),
            KernelArg::Buffer(gu_buf.view()),
            KernelArg::Buffer(weight),
            KernelArg::F32(c.scale),
            KernelArg::Buffer(bias_buf.view()),
            KernelArg::Buffer(gate_buf.view()),
        ],
    );
    read_f32(b, res_buf.view(), c.m * c.n)
}

#[test]
#[expect(
    clippy::many_single_char_names,
    reason = "m, n, k and a are the GEMM's own names for its extents and operand"
)]
fn gemm64_fused_reads_dense_then_swiglu_columns_and_adds_the_gated_residual() {
    let Some(b) = open() else { return };
    // The single block's shape in miniature: 16 dense columns, then a 32-wide
    // swiglu whose gate starts 8 columns into a 72-wide fused row.
    let (m, n, cols_a, inner) = (70usize, 67usize, 16usize, 32usize);
    let (gu_stride, gu_offset) = (2 * inner + 8, 8usize);
    let k = cols_a + inner;
    let small = |count: usize, seed: usize| -> Vec<f32> {
        vals(count, seed).iter().map(|v| v * 0.1).collect()
    };
    let (residual, a, gu) = (
        small(m * n, 101),
        small(m * cols_a, 102),
        small(m * gu_stride, 103),
    );
    let (weight, bias, gate) = (small(n * k, 104), small(n, 105), small(n, 106));
    let case = FusedCase {
        residual: &residual,
        a: &a,
        gu: &gu,
        weight: &weight,
        bias: &bias,
        gate: &gate,
        scale: 1.0,
        m,
        n,
        cols_a,
        inner,
        gu_stride,
        gu_offset,
    };
    let w_buf = upload_f32(&b, &weight);
    let got = run_fused(&b, &case, "kernel_dit_gemm64_fused_f32", w_buf.view());
    let expected = expect_fused(&case);
    // The same rows with gate and up swapped must give a different answer,
    // or the case cannot tell gate-first from gate-second.
    let mut swapped_gu = gu.clone();
    for r in 0..m {
        for i in 0..inner {
            swapped_gu.swap(
                r * gu_stride + gu_offset + i,
                r * gu_stride + gu_offset + inner + i,
            );
        }
    }
    let swapped = expect_fused(&FusedCase {
        gu: &swapped_gu,
        ..case
    });
    assert_differs(
        &expected,
        &swapped,
        "gemm64_fused: gate-first vs gate-second",
    );
    assert_close(&got, &expected, "gemm64_fused f32");
}

#[test]
fn gemm64_fused_takes_a_swiglu_only_input_over_fp8_weights() {
    let Some(b) = open() else { return };
    // The double block's MLP down projection: no dense part.
    let (m, n, inner) = (40usize, 70usize, 48usize);
    let gu_stride = 2 * inner;
    let small = |count: usize, seed: usize| -> Vec<f32> {
        vals(count, seed).iter().map(|v| v * 0.1).collect()
    };
    let (residual, gu) = (small(m * n, 111), small(m * gu_stride, 112));
    let (bias, gate) = (small(n, 113), small(n, 114));
    let bytes: Vec<u8> = (0..n * inner)
        .map(|i| u8::try_from((i * 13) % 120).unwrap_or(0))
        .collect();
    let weight: Vec<f32> = bytes.iter().map(|&v| e4m3(v)).collect();
    let empty: Vec<f32> = Vec::new();
    let case = FusedCase {
        residual: &residual,
        a: &empty,
        gu: &gu,
        weight: &weight,
        bias: &bias,
        gate: &gate,
        scale: 0.5,
        m,
        n,
        cols_a: 0,
        inner,
        gu_stride,
        gu_offset: 0,
    };
    let w_buf = upload_u8(&b, &bytes);
    let got = run_fused(&b, &case, "kernel_dit_gemm64_fused_fp8", w_buf.view());
    assert_close(&got, &expect_fused(&case), "gemm64_fused fp8 swiglu-only");
}

#[test]
fn gemm64_fused_takes_a_dense_only_input_over_bf16_weights() {
    let Some(b) = open() else { return };
    // The double block's attention output projection: no swiglu part.
    let (m, n, cols_a) = (33usize, 65usize, 64usize);
    let small = |count: usize, seed: usize| -> Vec<f32> {
        vals(count, seed).iter().map(|v| v * 0.1).collect()
    };
    let (residual, a) = (small(m * n, 121), small(m * cols_a, 122));
    let (bias, gate) = (small(n, 123), small(n, 124));
    // Multiples of 1/80 are not all exact in bf16, so the expected product
    // uses the widened values the kernel reads.
    let weight_bits: Vec<u16> = small(n * cols_a, 125)
        .iter()
        .map(|&v| bf16::from_f32(v).to_bits())
        .collect();
    let weight: Vec<f32> = weight_bits
        .iter()
        .map(|&v| bf16::from_bits(v).to_f32())
        .collect();
    let empty: Vec<f32> = Vec::new();
    let case = FusedCase {
        residual: &residual,
        a: &a,
        gu: &empty,
        weight: &weight,
        bias: &bias,
        gate: &gate,
        scale: 1.0,
        m,
        n,
        cols_a,
        inner: 0,
        gu_stride: 0,
        gu_offset: 0,
    };
    let w_buf = upload_u16(&b, &weight_bits);
    let got = run_fused(&b, &case, "kernel_dit_gemm64_fused_bf16", w_buf.view());
    assert_close(&got, &expect_fused(&case), "gemm64_fused bf16 dense-only");
}
