//! GPU kernel cases: the compiled kernel set against hand-written CPU oracles.
//!
//! Every case launches a real kernel with the geometry the engine will use and
//! compares the whole output buffer, so a wrong stride, a wrong binding index or
//! a wrong threadgroup count fails here rather than as a bad token.
//!
//! The cases are `#[ignore]`-gated so the default gate never needs a GPU; run
//! them with `cargo test -p shiso-metal -- --ignored`. The oracle self-checks at
//! the bottom are pure CPU and stay in the default gate.
// The whole file is Metal-only: without the feature, or on a target where the
// Objective-C stack is absent, `MetalBackend` does not exist and this target is
// empty rather than failing to compile.
#![cfg(all(feature = "metal", target_os = "macos"))]

//! The one `unsafe` here is `Mmap::map`, which is unsafe because the file behind
//! a mapping must not change underneath it: the mapping is read-only and the file
//! is held open by the same value, so it cannot change.
#![expect(unsafe_code, reason = "memmap2's mapping constructor is unsafe")]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "kernel argument structs take i32/u32/u64 dimensions while the fixtures are written in usize, and every fixture dimension here is a small literal"
)]
use shiso_kerneltest::assert_close;

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use shiso_core::{
    Backend, DeviceBuffer, Extent3, FnConst, FnConstValue, HostBytes, KernelArg, KernelRef,
    LaunchArgs, alloc, import_host,
};
use shiso_metal::MetalBackend;
use shiso_modelfile::quant::BlockQ4K;

use shiso_kerneltest::oracle::{
    Q4K, TOL_F32, TOL_QUANT, dequant_q4_k_dense, dequant_q4_k_gathered, dot, f16_to_f32, gather,
    pack_q4_k_dense, pack_q4_k_gathered, rms_norm, scale_min_k4, signal, softmax_row,
};

/// `f16` bits of `1.0`: the block scale the quantized fixtures use.
const ONE_F16: u16 = 0x3C00;
/// `f16` bits of `0.25`.
const QUARTER_F16: u16 = 0x3400;
/// Every group scale unpacks to 1 and every group minimum to 0.
const UNIT_SCALES: [u8; 12] = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1];
/// Scale bytes that exercise both halves of the 6-bit unpack: groups 0..4 read
/// the low bytes, groups 4..8 read the packed high nibbles.
const MIXED_SCALES: [u8; 12] = [61, 17, 42, 63, 5, 30, 11, 48, 9, 55, 22, 37];

/// One open device, held both concretely (for the batching counter) and as the
/// trait object an engine would carry.
struct Gpu {
    metal: Arc<MetalBackend>,
    device: Arc<dyn Backend>,
}

impl Gpu {
    #[expect(
        clippy::print_stderr,
        reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
    )]
    fn open() -> Option<Self> {
        if !shiso_metal::metal_available() {
            eprintln!("note: no Metal device on this machine, skipping the GPU cases");
            return None;
        }
        let metal = MetalBackend::new().expect("the embedded kernel library compiles");
        let device: Arc<dyn Backend> = metal.clone();
        Some(Self { metal, device })
    }

    /// Allocates and fills a device buffer.
    fn upload(&self, bytes: &[u8]) -> DeviceBuffer {
        let buffer = alloc(&self.device, bytes.len() as u64).expect("device allocation");
        self.device
            .copy_h2d(buffer.view(), bytes)
            .expect("host-to-device copy");
        buffer
    }

    /// Reads a whole device buffer back.
    fn read(&self, buffer: &DeviceBuffer) -> Vec<u8> {
        let mut out = vec![0u8; usize::try_from(buffer.len()).expect("buffer length fits usize")];
        self.device
            .copy_d2h(buffer.view(), &mut out)
            .expect("device-to-host copy");
        out
    }

    /// Reads a whole device buffer as `f32` values.
    fn read_f32(&self, buffer: &DeviceBuffer) -> Vec<f32> {
        f32_values(&self.read(buffer))
    }
}

/// `f32` values from little-endian bytes, without assuming any alignment.
fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_bits(u32::from_le_bytes([word[0], word[1], word[2], word[3]])))
        .collect()
}

/// `f32` values as little-endian bytes.
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `i32` values as little-endian bytes.
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `f16` bit patterns as little-endian bytes.
fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Launches one kernel, reporting the geometry when it fails.
fn run(gpu: &Gpu, kernel: KernelRef, launch: &LaunchArgs<'_>) {
    gpu.device.run(kernel, launch).unwrap_or_else(|err| {
        panic!(
            "{} grid {:?} block {:?} smem {}: {err}",
            kernel.entry, launch.grid, launch.block, launch.smem_bytes,
        )
    });
}

/// Drains the queue and fails with the operation named.
fn drain(gpu: &Gpu) {
    gpu.device.sync().expect("the queue drains");
}

// --- kernel argument layouts ------------------------------------------------
//
// Each struct mirrors one argument-block declaration in `metal/generic/`,
// field for field, so the bytes the kernel reads come from a struct definition
// rather than a hand-packed array. The `size_of` assertions pin the total, which
// is what a field-order or width mistake would change.
//
// Every instance is built with `..Zeroable::zeroed()` so the padding the C
// layout leaves between fields holds zeros instead of uninitialised bytes when
// `bytes_of` reads the whole struct.

/// The norm argument block (`metal/generic/norm.metal`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct NormArgs {
    ne00: i32,
    ne00_t: i32,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    eps: f32,
    nef1: [i32; 3],
    nef2: [i32; 3],
    nef3: [i32; 3],
    nbf1: [u64; 3],
    nbf2: [u64; 3],
    nbf3: [u64; 3],
}
const _: () = assert!(size_of::<NormArgs>() == 144);

/// The soft-max argument block (`metal/generic/softmax.metal`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SoftMaxArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    /// The C layout leaves four bytes here; spelled out so `Pod` can hold.
    _pad0: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne11: i32,
    ne12: i32,
    ne13: i32,
    /// The C layout leaves four bytes here; spelled out so `Pod` can hold.
    _pad1: i32,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    scale: f32,
    max_bias: f32,
    m0: f32,
    m1: f32,
    n_head_log2: i32,
    /// Trailing padding to the struct's eight-byte alignment.
    _pad2: i32,
}
const _: () = assert!(size_of::<SoftMaxArgs>() == 128);

/// The gather argument block (`metal/generic/get_rows.metal`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GetRowsArgs {
    ne00t: i32,
    ne00: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    /// The C layout leaves four bytes here; spelled out so `Pod` can hold.
    _pad0: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
}
const _: () = assert!(size_of::<GetRowsArgs>() == 88);

/// The quantized gather argument block (`metal/generic/get_rows.metal`),
/// which the `q4_K` gather also takes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GetRowsQuantArgs {
    n_embd: i32,
    n_vocab: i32,
    n_tokens: i32,
    /// The C layout leaves four bytes here; spelled out so `Pod` can hold.
    _pad0: i32,
    src_row_bytes: u64,
    dst_row_bytes: u64,
    token_stride: u64,
}
const _: () = assert!(size_of::<GetRowsQuantArgs>() == 40);

/// The extended-matvec argument block (`metal/generic/mul_mv_ext.metal`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MulMvExtArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    /// The C layout leaves four bytes here; spelled out so `Pod` can hold.
    _pad0: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    /// The C layout leaves four bytes here; spelled out so `Pod` can hold.
    _pad1: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
    /// Trailing padding to the struct's eight-byte alignment.
    _pad2: [i8; 4],
}
const _: () = assert!(size_of::<MulMvExtArgs>() == 112);

/// The `q4_K` matvec specialization: two simdgroups, 16 threads per row.
///
/// `r0ptg = (32 / nxpsg) * nsg = 4` rows per threadgroup, which is what turns an
/// `out_dim` of 32 into a grid of 8.
const MUL_MV_Q4_K: [FnConst; 2] = [
    FnConst {
        index: 600,
        value: FnConstValue::I16(2),
    },
    FnConst {
        index: 601,
        value: FnConstValue::I16(16),
    },
];
/// Builds a loader `BlockQ4K` from 256 nibble values.
///
/// The block comes from `shiso-modelfile`'s own layout rather than a hand-packed byte
/// array, so a change to the loader's idea of `q4_K` shows up here.
fn q4_k_block(
    values: &[u8; 256],
    pack: fn(&[u8; 256]) -> [u8; 128],
    d: u16,
    dmin: u16,
    scales: [u8; 12],
) -> BlockQ4K {
    BlockQ4K {
        d,
        dmin,
        scales,
        qs: pack(values),
    }
}

/// The same block as an oracle value, so the fixture and the oracle cannot drift
/// apart into two different layouts by accident.
fn q4_k_oracle(block: &BlockQ4K) -> Q4K {
    Q4K {
        d: block.d,
        dmin: block.dmin,
        scales: block.scales,
        qs: block.qs,
    }
}

/// Nibble values for one fixture row: a cycling pattern that hits every value
/// 0..15 and differs per row.
fn nibbles(row: usize) -> [u8; 256] {
    let mut values = [0u8; 256];
    for (e, slot) in values.iter_mut().enumerate() {
        *slot = ((e + row * 7) % 16) as u8;
    }
    values
}

/// A file mapping that can be handed to `import_host`.
#[derive(Debug)]
struct Mapping {
    /// Read but never *used*: it keeps the mapped file open for exactly as long
    /// as the mapping is alive.
    #[expect(
        dead_code,
        reason = "keeps the mapped file open for the mapping's lifetime"
    )]
    file: std::fs::File,
    bytes: Mmap,
}

impl AsRef<[u8]> for Mapping {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

// --- host-visible entry points --------------------------------------------
//
// The gate below resolves every entry point the kernel sources declare, not only
// the handful the numeric cases dispatch: reaching a kernel for the first time in
// the middle of a request must not be the first time anything resolves it.
//
// Several kernels are only complete once specialized, and Metal refuses to say so
// politely: building a pipeline from a function with an unset required constant
// trips an internal assertion that aborts the process. So each entry carries the
// constant table an engine would pass, taken from the bases in
// `metal/generic/00_common.metal`.

/// Function-constant bases, mirroring `00_common.metal`.
const FC_MUL_MV: u16 = 600;
const FC_UNARY: u16 = 1200;
const FC_BIN: u16 = 1300;
const FC_SUM_ROWS: u16 = 1400;

/// The row-blocking pair every `mul_mv` kernel reads.
const MUL_MV_SHAPES: [FnConst; 2] = [
    FnConst {
        index: FC_MUL_MV,
        value: FnConstValue::I16(1),
    },
    FnConst {
        index: FC_MUL_MV + 1,
        value: FnConstValue::I16(1),
    },
];

/// The op selector and count toggle the `unary` kernels read.
const UNARY_OPS: [FnConst; 2] = [
    FnConst {
        index: FC_UNARY,
        value: FnConstValue::I16(0),
    },
    FnConst {
        index: FC_UNARY + 1,
        value: FnConstValue::Bool(false),
    },
];

/// The op, scalar and two broadcast toggles the `bin` kernel reads.
const BIN_OPS: [FnConst; 4] = [
    FnConst {
        index: FC_BIN,
        value: FnConstValue::I16(0),
    },
    FnConst {
        index: FC_BIN + 1,
        value: FnConstValue::I16(0),
    },
    FnConst {
        index: FC_BIN + 2,
        value: FnConstValue::Bool(false),
    },
    FnConst {
        index: FC_BIN + 3,
        value: FnConstValue::Bool(false),
    },
];

/// The op selector the `sum_rows` kernel reads.
const SUM_ROWS_OPS: [FnConst; 1] = [FnConst {
    index: FC_SUM_ROWS,
    value: FnConstValue::I16(0),
}];

/// Every entry point the kernel sources make visible to the host, each with the
/// specialization it needs.
const HOST_KERNELS: &[KernelRef] = &[
    KernelRef::new("generic", "kernel_add2_f32"),
    KernelRef::new("generic", "kernel_add3_f32"),
    KernelRef::new("generic", "kernel_add_rms_norm_mul_f32_4"),
    KernelRef::new("generic", "kernel_argsort_f32_i32_desc"),
    KernelRef::new("generic", "kernel_argsort_merge_f32_i32_desc"),
    KernelRef::specialized("generic", "kernel_bin_fuse_f32_f32_f32", &BIN_OPS),
    KernelRef::new("generic", "kernel_concat"),
    KernelRef::new("generic", "kernel_cpy_contig_f16_f16_bits_4"),
    KernelRef::new("generic", "kernel_cpy_contig_f16_f32_4"),
    KernelRef::new("generic", "kernel_cpy_contig_f32_f16_4"),
    KernelRef::new("generic", "kernel_cpy_f16_f16"),
    KernelRef::new("generic", "kernel_cpy_f16_f32"),
    KernelRef::new("generic", "kernel_cpy_f32_f16"),
    KernelRef::new("generic", "kernel_cpy_f32_f32"),
    KernelRef::new("generic", "kernel_dsv4_rope_tail_f32"),
    KernelRef::new("generic", "kernel_dsv4_rope_tail_f32_inplace_pair"),
    KernelRef::new("generic", "kernel_dsv4_rope_tail_f32_inplace_pair_affine"),
    KernelRef::new("generic", "kernel_fill_zero_u8"),
    KernelRef::new("generic", "kernel_get_rows_f16"),
    KernelRef::new("generic", "kernel_get_rows_f16_f32"),
    KernelRef::new("generic", "kernel_get_rows_f32"),
    KernelRef::new("generic", "kernel_get_rows_i32"),
    KernelRef::new("generic", "kernel_get_rows_iq1_m_f32"),
    KernelRef::new("generic", "kernel_get_rows_q4_0_f32"),
    KernelRef::new("generic", "kernel_get_rows_q4_K_f32"),
    KernelRef::new("generic", "kernel_get_rows_q8_0_f32"),
    KernelRef::new("generic", "kernel_iq_table_copy"),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq1_m_f32_r1_1",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq1_m_f32_r1_2",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq1_m_f32_r1_3",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq1_m_f32_r1_4",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq1_m_f32_r1_5",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq1_s_f32_r1_1",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq1_s_f32_r1_2",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq1_s_f32_r1_3",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq1_s_f32_r1_4",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq1_s_f32_r1_5",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq2_s_f32_r1_1",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq2_s_f32_r1_2",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq2_s_f32_r1_3",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq2_s_f32_r1_4",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq2_s_f32_r1_5",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq2_xs_f32_r1_1",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq2_xs_f32_r1_2",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq2_xs_f32_r1_3",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq2_xs_f32_r1_4",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq2_xs_f32_r1_5",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_1",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_2",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_3",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_4",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_5",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq3_s_f32_r1_1",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq3_s_f32_r1_2",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq3_s_f32_r1_3",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq3_s_f32_r1_4",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq3_s_f32_r1_5",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_1",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_2",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_3",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_4",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_5",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq4_xs_f32_r1_1",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq4_xs_f32_r1_2",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq4_xs_f32_r1_3",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq4_xs_f32_r1_4",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized(
        "generic",
        "kernel_mul_mv_ext_iq4_xs_f32_r1_5",
        &MUL_MV_SHAPES,
    ),
    KernelRef::specialized("generic", "kernel_mul_mv_ext_q2_K_f32_r1_1", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_ext_q2_K_f32_r1_2", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_ext_q2_K_f32_r1_3", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_ext_q2_K_f32_r1_4", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_ext_q2_K_f32_r1_5", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_ext_q4_0_f32_r1_1", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_ext_q4_0_f32_r1_2", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_ext_q4_0_f32_r1_3", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_ext_q4_0_f32_r1_4", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_ext_q4_0_f32_r1_5", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_ext_q4_K_f32_r1_1", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_ext_q4_K_f32_r1_2", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_ext_q4_K_f32_r1_3", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_ext_q4_K_f32_r1_4", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_ext_q4_K_f32_r1_5", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_ext_q8_0_f32_r1_2", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_ext_q8_0_f32_r1_3", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_ext_q8_0_f32_r1_4", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_ext_q8_0_f32_r1_5", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_f16_f32", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_f16_f32_4", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_f16_f32_short", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_f32_f32", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_f32_f32_4", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_f32_f32_short", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_q8_0_f32", &MUL_MV_SHAPES),
    KernelRef::specialized("generic", "kernel_mul_mv_q8_0_f32_pair", &MUL_MV_SHAPES),
    KernelRef::new("generic", "kernel_repeat_f32"),
    KernelRef::new("generic", "kernel_rms_norm_f32_4"),
    KernelRef::new("generic", "kernel_rms_norm_mul_f32_4"),
    KernelRef::new("generic", "kernel_rms_norm_scale_f32_4"),
    KernelRef::new("generic", "kernel_set_rows_f32_i32"),
    KernelRef::new("generic", "kernel_soft_max_f32"),
    KernelRef::new("generic", "kernel_soft_max_f32_4"),
    KernelRef::specialized("generic", "kernel_sum_rows_f32_f32", &SUM_ROWS_OPS),
    KernelRef::new("generic", "kernel_swiglu_f32"),
    KernelRef::new("generic", "kernel_swiglu_flat_f32"),
    KernelRef::new("generic", "kernel_touch_u8_stride"),
    KernelRef::specialized("generic", "kernel_unary_f16_f16", &UNARY_OPS),
    KernelRef::specialized("generic", "kernel_unary_f32_f32", &UNARY_OPS),
    KernelRef::specialized("generic", "kernel_unary_f32_f32_4", &UNARY_OPS),
];

/// The one specialization the gate resolves beyond the plain list, with the
/// row-blocking values the numeric `q4_K` case actually dispatches.
///
/// The module is registry metadata rather than a lookup dimension, so `"dense"`
/// resolving the function `"generic"` declares is part of what this covers.
const SPECIALIZED_MUL_MV: KernelRef =
    KernelRef::specialized("dense", "kernel_mul_mv_ext_q4_K_f32_r1_1", &MUL_MV_Q4_K);

/// The kernel sources as `(file name, text)`, sorted by name.
///
/// The backend compiles the embedded copy of exactly these files, so reading the
/// crate directory reads the same bytes the device compiled.
fn kernel_sources() -> Vec<(String, String)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("metal/generic");
    let mut files = match std::fs::read_dir(&dir) {
        Ok(entries) => entries
            .map(|entry| entry.expect("a readable kernel source").path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "metal"))
            .collect::<Vec<_>>(),
        Err(err) => panic!("{} is unreadable: {err}", dir.display()),
    };
    files.sort();
    assert!(
        !files.is_empty(),
        "no kernel source under {}",
        dir.display()
    );
    files
        .into_iter()
        .map(|path| {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("{} is unreadable: {err}", path.display()));
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            (name, text)
        })
        .collect()
}

/// The entry points the sources declare: every `host_name` attribute, plus every
/// `kernel void` that no template parameter list introduces.
fn scanned_host_names() -> Vec<String> {
    const ATTR: &str = "host_name(\"";
    let mut names = Vec::new();
    for (file, text) in kernel_sources() {
        let lines: Vec<&str> = text.lines().collect();
        for (index, line) in lines.iter().enumerate() {
            let mut rest = *line;
            while let Some(at) = rest.find(ATTR) {
                let tail = &rest[at + ATTR.len()..];
                let end = tail
                    .find('"')
                    .unwrap_or_else(|| panic!("{file}: unterminated host_name in {line:?}"));
                names.push(tail[..end].to_owned());
                rest = &tail[end..];
            }
            let Some(after) = line.trim_start().strip_prefix("kernel void ") else {
                continue;
            };
            let name: String = after
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if name.is_empty() {
                continue;
            }
            // A templated definition reaches the host through its `host_name`
            // instantiation, so it is not an entry point under its own name.
            let previous = lines[..index]
                .iter()
                .rev()
                .map(|previous| previous.trim())
                .find(|previous| !previous.is_empty())
                .unwrap_or("");
            if previous.starts_with("template") && previous.ends_with('>') {
                continue;
            }
            names.push(name);
        }
    }
    names.sort();
    names.dedup();
    names
}

/// The list above and the sources have to agree in both directions.
///
/// A kernel added to a `.metal` file but not listed here would never be resolved
/// by the GPU gate, which is the exact gap this closes; a name listed here that
/// the sources no longer declare would keep the gate green while naming nothing.
#[test]
fn enumerated_kernels_match_the_sources() {
    let mut listed: Vec<&str> = HOST_KERNELS.iter().map(|kernel| kernel.entry).collect();
    listed.sort_unstable();
    listed.dedup();
    let scanned = scanned_host_names();
    let scanned: Vec<&str> = scanned.iter().map(String::as_str).collect();
    assert_eq!(
        listed, scanned,
        "HOST_KERNELS has drifted from metal/generic/*.metal",
    );
    assert_eq!(
        listed.len(),
        HOST_KERNELS.len(),
        "HOST_KERNELS lists an entry point twice",
    );
}

#[test]
#[ignore = "requires Metal GPU"]
fn kernel_library_compiles_and_unknown_entry_is_named() {
    let Some(gpu) = Gpu::open() else { return };
    assert_eq!(gpu.device.backend_name(), "metal");
    assert_eq!(gpu.metal.live_buffers(), 0, "a fresh backend holds nothing");
    assert_eq!(
        gpu.metal.commits(),
        0,
        "a fresh backend has submitted nothing"
    );

    // No graph, BLAS or cooperative-launch surface in this backend.
    assert!(gpu.device.as_graph().is_none());
    assert!(gpu.device.as_blas().is_none());
    assert!(gpu.device.as_coop().is_none());

    let table = gpu.upload(&f32_bytes(&[7.0]));
    let ids = gpu.upload(&i32_bytes(&[0]));
    let out = gpu.upload(&[0u8; 4]);
    let args = GetRowsArgs {
        ne00t: 1,
        ne00: 1,
        nb01: 4,
        nb02: 4,
        nb03: 4,
        ne10: 1,
        nb10: 4,
        nb11: 4,
        nb12: 4,
        nb1: 4,
        nb2: 4,
        nb3: 4,
        ..Zeroable::zeroed()
    };
    let params = [
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(table.view()),
        KernelArg::Buffer(ids.view()),
        KernelArg::Buffer(out.view()),
    ];
    let launch = LaunchArgs::new(Extent3::linear(1), Extent3::linear(1), &params);

    run(
        &gpu,
        KernelRef::new("generic", "kernel_get_rows_f32"),
        &launch,
    );
    drain(&gpu);
    assert_close!(gpu.read_f32(&out)[0], 7.0, TOL_F32);

    // Every declared entry point has to build a pipeline, including the ones no
    // numeric case dispatches. Resolving is not launching: nothing is submitted,
    // so this leaves the queue as it found it.
    for kernel in HOST_KERNELS {
        gpu.metal
            .prepare(*kernel)
            .unwrap_or_else(|err| panic!("no pipeline for {}: {err}", kernel.entry));
    }
    gpu.metal
        .prepare(SPECIALIZED_MUL_MV)
        .expect("the specialized q4_K matvec pipeline resolves");
    assert_eq!(
        gpu.metal.pipelines(),
        HOST_KERNELS.len() + 1,
        "each prepared kernel is cached once, under its own specialization",
    );

    // Nothing parses a kernel name, so an entry point the sources do not declare
    // has to fail by that name rather than resolve to something nearby.
    let missing = KernelRef::new("generic", "kernel_no_such_entry");
    let err = gpu
        .device
        .run(missing, &launch)
        .expect_err("an undeclared entry point must not resolve");
    // `CoreError::Backend`'s own `Display` names the operation and carries the
    // detail as its `source`, which is where the entry point has to appear.
    let message = err.to_string();
    let detail = std::error::Error::source(&err)
        .map(ToString::to_string)
        .unwrap_or_default();
    assert!(
        message.contains("newFunctionWithName"),
        "the error should name the operation, got {message}",
    );
    assert!(
        detail.contains(missing.entry),
        "the driver error should name the entry point, got {detail}",
    );
    // The failed lookup must not have left a half-open command buffer behind.
    drain(&gpu);
}

/// Every allocation has to leave the backend's bookkeeping when the engine drops
/// it. A table entry that outlives its buffer is a slow leak in exactly the place
/// a long-running engine allocates and frees weights, and it is invisible until
/// memory has already grown.
#[test]
#[ignore = "requires Metal GPU"]
fn buffers_leave_the_table_when_dropped() {
    const BUFFERS: usize = 64;

    let Some(gpu) = Gpu::open() else { return };
    let kept = gpu.upload(&f32_bytes(&[1.0, 2.0, 3.0]));
    {
        let mut batch = Vec::with_capacity(BUFFERS);
        for _ in 0..BUFFERS {
            batch.push(gpu.upload(&f32_bytes(&[1.0, 2.0, 3.0])));
        }
        assert_eq!(
            gpu.metal.live_buffers(),
            BUFFERS + 1,
            "the batch and the surviving buffer are all live",
        );
        drain(&gpu);
    }
    assert_eq!(
        gpu.metal.live_buffers(),
        1,
        "the scope held {BUFFERS} buffers and dropped every one of them",
    );

    drop(kept);
    assert_eq!(
        gpu.metal.live_buffers(),
        0,
        "the last buffer leaving empties the table"
    );
}

#[test]
#[ignore = "requires Metal GPU"]
fn rms_norm_rows() {
    const ROWS: usize = 3;
    const N: usize = 128;
    const EPS: f32 = 1e-5;
    let Some(gpu) = Gpu::open() else { return };

    let x = signal(ROWS * N, 0x1234);
    let weight = signal(N, 0x5678);
    let src = gpu.upload(&f32_bytes(&x));
    let w = gpu.upload(&f32_bytes(&weight));
    let dst = gpu.upload(&vec![0u8; ROWS * N * 4]);

    let args = NormArgs {
        ne00: N as i32,
        // The kernel walks `float4`, so the trip count is a quarter of the row.
        ne00_t: (N / 4) as i32,
        nb1: (N * 4) as u64,
        nb2: (ROWS * N * 4) as u64,
        nb3: (ROWS * N * 4) as u64,
        eps: EPS,
        // Every weight modulo bound is 1, so one weight row serves all rows.
        nef1: [1; 3],
        nef2: [1; 3],
        nef3: [1; 3],
        nbf1: [(N * 4) as u64; 3],
        nbf2: [(ROWS * N * 4) as u64, (N * 4) as u64, (N * 4) as u64],
        nbf3: [(ROWS * N * 4) as u64, (N * 4) as u64, (N * 4) as u64],
    };
    // Bindings: args, src, weight, second weight (aliased), dst.
    let params = [
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(src.view()),
        KernelArg::Buffer(w.view()),
        KernelArg::Buffer(w.view()),
        KernelArg::Buffer(dst.view()),
    ];
    let launch = LaunchArgs {
        grid: Extent3::new(ROWS as u32, 1, 1),
        block: Extent3::linear(32),
        smem_bytes: (32 * 4) as u32,
        args: &params,
    };

    run(
        &gpu,
        KernelRef::new("generic", "kernel_rms_norm_f32_4"),
        &launch,
    );
    drain(&gpu);
    let got = gpu.read_f32(&dst);
    for row in 0..ROWS {
        let expected = rms_norm(&x[row * N..(row + 1) * N], None, EPS);
        for col in 0..N {
            assert_close!(
                got[row * N + col],
                expected[col],
                TOL_F32 * 10.0,
                "plain row {row} col {col}"
            );
        }
    }

    run(
        &gpu,
        KernelRef::new("generic", "kernel_rms_norm_mul_f32_4"),
        &launch,
    );
    drain(&gpu);
    let got = gpu.read_f32(&dst);
    for row in 0..ROWS {
        let expected = rms_norm(&x[row * N..(row + 1) * N], Some(&weight), EPS);
        for col in 0..N {
            assert_close!(
                got[row * N + col],
                expected[col],
                TOL_F32 * 10.0,
                "weighted row {row} col {col}"
            );
        }
    }
}

#[test]
#[ignore = "requires Metal GPU"]
fn soft_max_rows() {
    const ROWS: usize = 2;
    const W: usize = 64;
    let Some(gpu) = Gpu::open() else { return };

    let scores = signal(ROWS * W, 0xABCD);
    let src = gpu.upload(&f32_bytes(&scores));
    let dst = gpu.upload(&vec![0u8; ROWS * W * 4]);

    let args = SoftMaxArgs {
        ne00: W as i32,
        ne01: ROWS as i32,
        ne02: 1,
        nb01: (W * 4) as u64,
        nb02: (ROWS * W * 4) as u64,
        nb03: (ROWS * W * 4) as u64,
        ne11: 1,
        ne12: 1,
        ne13: 1,
        nb11: (W * 4) as u64,
        nb12: (ROWS * W * 4) as u64,
        nb13: (ROWS * W * 4) as u64,
        nb1: (W * 4) as u64,
        nb2: (ROWS * W * 4) as u64,
        nb3: (ROWS * W * 4) as u64,
        scale: 1.0,
        max_bias: 0.0,
        m0: 1.0,
        m1: 1.0,
        n_head_log2: 1,
        ..Zeroable::zeroed()
    };
    // The mask and the sink buffer alias the scores: the kernel turns both into
    // null pointers when they are the same address, which is the no-mask,
    // no-sink path.
    let params = [
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(src.view()),
        KernelArg::Buffer(src.view()),
        KernelArg::Buffer(src.view()),
        KernelArg::Buffer(dst.view()),
    ];
    let launch = LaunchArgs {
        grid: Extent3::new(ROWS as u32, 1, 1),
        block: Extent3::linear(32),
        smem_bytes: (32 * 4) as u32,
        args: &params,
    };

    run(
        &gpu,
        KernelRef::new("generic", "kernel_soft_max_f32_4"),
        &launch,
    );
    drain(&gpu);
    let got = gpu.read_f32(&dst);
    for row in 0..ROWS {
        let expected = softmax_row(&scores[row * W..(row + 1) * W], 1.0);
        for col in 0..W {
            assert_close!(
                got[row * W + col],
                expected[col],
                TOL_F32 * 10.0,
                "row {row} col {col}"
            );
        }
    }
}
#[test]
#[ignore = "requires Metal GPU"]
fn get_rows_gathers_f32_table_rows() {
    const ROWS: usize = 8;
    const N: usize = 128;
    let Some(gpu) = Gpu::open() else { return };
    let ids = [3i32, 0, 7];

    let table = signal(ROWS * N, 0x0F0F);
    let src = gpu.upload(&f32_bytes(&table));
    let index = gpu.upload(&i32_bytes(&ids));
    let dst = gpu.upload(&vec![0u8; ids.len() * N * 4]);

    let args = GetRowsArgs {
        // One element per thread: the kernel copies a row and breaks.
        ne00t: N as i32,
        ne00: N as i32,
        nb01: (N * 4) as u64,
        nb02: (ROWS * N * 4) as u64,
        nb03: (ROWS * N * 4) as u64,
        ne10: ids.len() as i32,
        nb10: 4,
        nb11: (ids.len() * 4) as u64,
        nb12: (ids.len() * 4) as u64,
        nb1: (N * 4) as u64,
        nb2: (N * 4) as u64,
        nb3: (ids.len() * N * 4) as u64,
        ..Zeroable::zeroed()
    };
    let params = [
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(src.view()),
        KernelArg::Buffer(index.view()),
        KernelArg::Buffer(dst.view()),
    ];
    let launch = LaunchArgs::new(
        Extent3::new((ids.len() * N) as u32, 1, 1),
        Extent3::linear(1),
        &params,
    );

    run(
        &gpu,
        KernelRef::new("generic", "kernel_get_rows_f32"),
        &launch,
    );
    drain(&gpu);
    let got = gpu.read_f32(&dst);
    let expected = gather(&table, N, &ids);
    for (i, (got, want)) in got.iter().zip(&expected).enumerate() {
        assert_close!(*got, *want, TOL_F32, "element {i}");
    }
}

/// Half-precision bit patterns the unpacked gather's fixture is built from.
///
/// Chosen rather than generated. A table of pseudo-random halves would sit in
/// one exponent range and call that coverage; these are the cases a widening
/// cast gets wrong one at a time - both zeros, both signs, a power of two, a
/// full mantissa, the largest finite half, and the subnormal boundary, which is
/// the only place the exponent field has to be rebuilt rather than shifted.
const F16_SAMPLES: [u16; 12] = [
    0x0000, // +0
    0x8000, // -0
    0x3C00, // 1
    0xBC00, // -1
    0x3800, // 1/2
    0x4900, // 10
    0xC900, // -10
    0x3555, // 0.333251953125, a mantissa with every other bit set
    0x0001, // the smallest subnormal, 2^-24
    0x03FF, // the largest subnormal
    0x0400, // the smallest normal, 2^-14
    0x7BFF, // 65504, the largest finite half
];

/// The half at row `row`, element `e` of the fixture table.
///
/// The stride through the sample table is coprime with its length, so no two
/// rows are the same sequence: a gather that read a neighbouring row would
/// otherwise have a fair chance of agreeing with the oracle anyway.
fn f16_table_bits(row: usize, e: usize) -> u16 {
    F16_SAMPLES[(row * 7 + e) % F16_SAMPLES.len()]
}

/// The unpacked gather, against a CPU oracle, at a width that is not a whole
/// number of spans.
///
/// This kernel is on the embedding path of every checkpoint whose vocabulary
/// table arrives unpacked, and it is the one gather with no codec - so what can
/// be wrong with it is not arithmetic but addressing, and the fixture is built
/// to make each kind of wrong visible:
///
/// - **the row.** Every row is a different sequence, so reading a neighbour is
///   not a near miss that a tolerance forgives.
/// - **the span.** 300 elements is two threadgroups' worth and a bit, so the
///   second span is partial and the `n_embd` bound is load-bearing. A kernel
///   that trusted the span width would run 212 elements off the end of the row.
/// - **the id.** Two of the six ids are out of range, one negative and one one
///   past the vocabulary. Their output rows are prefilled with a sentinel and
///   asserted *unchanged*, so the early return is checked as a property of
///   memory rather than of the value that happened to land there.
///
/// The comparison is bit-exact rather than within a tolerance, because every
/// half is exactly representable as a float: there is no rounding here to
/// allow for, and a tolerance would hide a flushed subnormal or a dropped sign
/// on zero.
#[test]
#[ignore = "requires Metal GPU"]
fn get_rows_widens_f16_table_rows() {
    const VOCAB: usize = 8;
    /// Deliberately not a multiple of the 256-element span.
    const N: usize = 300;
    const SPANS: u32 = N.div_ceil(256) as u32;
    /// A value no fixture half converts to, so an untouched row is recognisable.
    const SENTINEL: f32 = -98765.25;
    let Some(gpu) = Gpu::open() else { return };
    let ids = [6i32, 0, 7, 3, -1, VOCAB as i32];

    let bits: Vec<u16> = (0..VOCAB)
        .flat_map(|row| (0..N).map(move |e| f16_table_bits(row, e)))
        .collect();
    let src = gpu.upload(&u16_bytes(&bits));
    let index = gpu.upload(&i32_bytes(&ids));
    let dst = gpu.upload(&f32_bytes(&vec![SENTINEL; ids.len() * N]));

    let args = GetRowsQuantArgs {
        n_embd: N as i32,
        n_vocab: VOCAB as i32,
        n_tokens: ids.len() as i32,
        src_row_bytes: (N * 2) as u64,
        dst_row_bytes: (N * 4) as u64,
        token_stride: 4,
        ..Zeroable::zeroed()
    };
    let params = [
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(src.view()),
        KernelArg::Buffer(index.view()),
        KernelArg::Buffer(dst.view()),
    ];
    // The engine's own gather grid: one threadgroup per 256-element span of a
    // row, one threadgroup row per token.
    let launch = LaunchArgs::new(
        Extent3::new(SPANS, ids.len() as u32, 1),
        Extent3::linear(256),
        &params,
    );

    run(
        &gpu,
        KernelRef::new("generic", "kernel_get_rows_f16_f32"),
        &launch,
    );
    drain(&gpu);
    let got = gpu.read_f32(&dst);
    assert_eq!(got.len(), ids.len() * N);

    for (t, &id) in ids.iter().enumerate() {
        let row = usize::try_from(id).ok().filter(|&row| row < VOCAB);
        for e in 0..N {
            let want = row.map_or(SENTINEL, |row| f16_to_f32(f16_table_bits(row, e)));
            assert_eq!(
                got[t * N + e].to_bits(),
                want.to_bits(),
                "token {t} id {id} element {e}: got {} want {want}",
                got[t * N + e]
            );
        }
    }
}

#[test]
#[ignore = "requires Metal GPU"]
fn get_rows_dequantizes_q4_k_table_rows() {
    const VOCAB: usize = 8;
    const N: usize = 256;
    const ROW_BYTES: usize = N / 2 + 16;
    let Some(gpu) = Gpu::open() else { return };
    let ids = [6i32, 1, 4];

    let blocks: Vec<BlockQ4K> = (0..VOCAB)
        .map(|row| {
            q4_k_block(
                &nibbles(row),
                pack_q4_k_gathered,
                ONE_F16,
                QUARTER_F16,
                MIXED_SCALES,
            )
        })
        .collect();
    let table: Vec<u8> = blocks
        .iter()
        .flat_map(bytemuck::bytes_of)
        .copied()
        .collect();
    assert_eq!(table.len(), VOCAB * ROW_BYTES);

    let src = gpu.upload(&table);
    let index = gpu.upload(&i32_bytes(&ids));
    let dst = gpu.upload(&vec![0u8; ids.len() * N * 4]);

    let args = GetRowsQuantArgs {
        n_embd: N as i32,
        n_vocab: VOCAB as i32,
        n_tokens: ids.len() as i32,
        src_row_bytes: ROW_BYTES as u64,
        dst_row_bytes: (N * 4) as u64,
        token_stride: 4,
        ..Zeroable::zeroed()
    };
    let params = [
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(src.view()),
        KernelArg::Buffer(index.view()),
        KernelArg::Buffer(dst.view()),
    ];
    // One threadgroup per 256-element block, one threadgroup row per token.
    let launch = LaunchArgs::new(
        Extent3::new(1, ids.len() as u32, 1),
        Extent3::linear(N as u32),
        &params,
    );

    run(
        &gpu,
        KernelRef::new("generic", "kernel_get_rows_q4_K_f32"),
        &launch,
    );
    drain(&gpu);
    let got = gpu.read_f32(&dst);
    for (t, &id) in ids.iter().enumerate() {
        let row = q4_k_oracle(&blocks[usize::try_from(id).expect("fixture id")]);
        for e in 0..N {
            assert_close!(
                got[t * N + e],
                dequant_q4_k_gathered(&row, e),
                TOL_QUANT,
                "token {t} id {id} element {e}"
            );
        }
    }
}

#[test]
#[ignore = "requires Metal GPU"]
fn mul_mv_ext_q4_k_matvec() {
    const IN: usize = 256;
    const OUT: usize = 32;
    const ROW_BYTES: usize = IN / 2 + 16;
    let Some(gpu) = Gpu::open() else { return };

    let blocks: Vec<BlockQ4K> = (0..OUT)
        .map(|row| {
            q4_k_block(
                &nibbles(row),
                pack_q4_k_dense,
                ONE_F16,
                QUARTER_F16,
                MIXED_SCALES,
            )
        })
        .collect();
    let weights: Vec<u8> = blocks
        .iter()
        .flat_map(bytemuck::bytes_of)
        .copied()
        .collect();

    let x = signal(IN, 0x4242);
    let weights = gpu.upload(&weights);
    let src = gpu.upload(&f32_bytes(&x));
    let dst = gpu.upload(&[0u8; OUT * 4]);

    let args = MulMvExtArgs {
        ne00: IN as i32,
        ne01: OUT as i32,
        ne02: 1,
        nb00: ROW_BYTES as u64,
        nb01: ROW_BYTES as u64,
        nb02: ROW_BYTES as u64 * OUT as u64,
        nb03: ROW_BYTES as u64 * OUT as u64,
        ne10: IN as i32,
        ne11: 1,
        ne12: 1,
        nb10: 4,
        nb11: (IN * 4) as u64,
        nb12: (IN * 4) as u64,
        nb13: (IN * 4) as u64,
        ne0: OUT as i32,
        ne1: 1,
        r2: 1,
        r3: 1,
        ..Zeroable::zeroed()
    };
    let params = [
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(weights.view()),
        KernelArg::Buffer(src.view()),
        KernelArg::Buffer(dst.view()),
    ];
    // `r0ptg = (32 / nxpsg) * nsg = 4` output rows per threadgroup, one token per
    // threadgroup column, and no threadgroup memory on this path.
    let launch = LaunchArgs::new(
        Extent3::new((OUT / 4) as u32, 1, 1),
        Extent3::new(32, 2, 1),
        &params,
    );
    let kernel = KernelRef::specialized("generic", "kernel_mul_mv_ext_q4_K_f32_r1_1", &MUL_MV_Q4_K);

    run(&gpu, kernel, &launch);
    drain(&gpu);
    let got = gpu.read_f32(&dst);
    for row in 0..OUT {
        let block = q4_k_oracle(&blocks[row]);
        let dequantized: Vec<f32> = (0..IN).map(|e| dequant_q4_k_dense(&block, e)).collect();
        assert_close!(
            got[row],
            dot(&dequantized, &x),
            TOL_QUANT,
            "output row {row}"
        );
    }
}

/// Two hundred dispatches must cost one command buffer, not two hundred.
///
/// This is the batching invariant the T6 baselines depend on: `run` appends to
/// the open encoder and never commits, so the counter has to stay flat until
/// `sync`. A regression to one commit per dispatch fails here instead of showing
/// up as unexplained overhead in a later baseline.
#[test]
#[ignore = "requires Metal GPU"]
fn dispatches_batch_into_few_command_buffers() {
    const DISPATCHES: usize = 200;
    let Some(gpu) = Gpu::open() else { return };

    let table = gpu.upload(&f32_bytes(&[10.0, 20.0, 30.0, 40.0]));
    let index = gpu.upload(&i32_bytes(&[0, 1, 2, 3]));
    let dst = gpu.upload(&vec![0u8; DISPATCHES * 4]);

    let args = GetRowsArgs {
        ne00t: 1,
        ne00: 1,
        nb01: 4,
        nb02: 16,
        nb03: 16,
        ne10: 1,
        nb10: 4,
        nb11: 4,
        nb12: 4,
        nb1: 4,
        nb2: 4,
        nb3: 4,
        ..Zeroable::zeroed()
    };
    let kernel = KernelRef::new("generic", "kernel_get_rows_f32");
    let before = gpu.metal.commits();

    for i in 0..DISPATCHES {
        // Each dispatch reads a different id and writes a different slot, both
        // through a view offset, so nothing here can be folded away.
        let id_view = index
            .slice(((i % 4) * 4) as u64, 4)
            .expect("id offset in range");
        let out_view = dst
            .slice((i * 4) as u64, 4)
            .expect("output offset in range");
        let params = [
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(table.view()),
            KernelArg::Buffer(id_view),
            KernelArg::Buffer(out_view),
        ];
        let launch = LaunchArgs::new(Extent3::linear(1), Extent3::linear(1), &params);
        run(&gpu, kernel, &launch);
    }

    assert_eq!(
        gpu.metal.commits(),
        before,
        "run() must never commit: a per-dispatch regression shows up here first",
    );
    drain(&gpu);
    let committed = gpu.metal.commits() - before;
    assert!(
        (1..5).contains(&committed),
        "{DISPATCHES} dispatches committed {committed} command buffers, expected 1..5",
    );

    let got = gpu.read_f32(&dst);
    for (i, &value) in got.iter().enumerate() {
        let expected = (f32::from((i % 4) as u16) + 1.0) * 10.0;
        assert_close!(value, expected, TOL_F32, "slot {i}");
    }
    assert_eq!(gpu.metal.live_buffers(), 3, "the three buffers are live");
}

/// The zero-copy wrap has to own what it points at.
///
/// The caller drops its own `Arc` before the read, so if `import_host` did not
/// keep the mapping alive inside the returned buffer this would read unmapped
/// pages instead of agreeing with the file.
#[test]
#[ignore = "requires Metal GPU"]
fn import_host_keeps_the_mapping_alive() {
    const LEN: usize = 16 * 1024;
    // Deliberately not a multiple of any plausible page size, so the wrap has to
    // start on a page boundary below the range and carry leading slack.
    const OFFSET: usize = 4096 + 3;
    const SPAN: usize = 1000;
    let Some(gpu) = Gpu::open() else { return };

    let pattern: Vec<u8> = (0..LEN).map(|i| (i as u32 * 31 + 7) as u8).collect();
    let temp = TempFile::new(format!(
        "shiso-metal-import-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos()),
    ));
    std::fs::write(temp.path(), &pattern).expect("pattern written");
    let file = std::fs::File::open(temp.path()).expect("mapped file handle");
    // SAFETY: the file is opened read-only, is not written to again, and lives in
    // `Mapping` beside the mapping for as long as the mapping is alive, which is
    // the precondition `Mmap::map` asks for.
    let bytes = unsafe { Mmap::map(&file) }.expect("shared file mapping");
    let host: Arc<dyn HostBytes> = Arc::new(Mapping { file, bytes });

    let range = OFFSET..OFFSET + SPAN;
    let buffer = import_host(&gpu.device, Arc::clone(&host), range)
        .expect("page-misaligned mid-mapping import");
    assert_eq!(usize::try_from(buffer.len()).expect("length"), SPAN);
    assert!(buffer.host().is_some(), "the buffer owns the mapping");

    // The last caller-side handle goes away here. Everything after this point
    // reads through what the backend kept.
    drop(host);

    let mut got = vec![0u8; SPAN];
    gpu.device
        .copy_d2h(buffer.view(), &mut got)
        .expect("read the imported range");
    assert_eq!(got, pattern[OFFSET..OFFSET + SPAN]);
}

/// A named file under the temporary directory, removed whatever path the test
/// takes, panic included.
struct TempFile {
    path: std::path::PathBuf,
}

impl TempFile {
    /// A fresh name in `std::env::temp_dir()`; nothing is created yet.
    fn new(name: String) -> Self {
        Self {
            path: std::env::temp_dir().join(name),
        }
    }

    fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

// --- oracle self-checks (no GPU needed) -------------------------------------

/// The fixtures carry `f16` bit patterns, so the conversion they rely on is
/// checked against values that are known independently of any kernel.
#[test]
#[expect(
    clippy::float_cmp,
    reason = "each constant is exactly representable in f32, so equality is the assertion"
)]
fn f16_bits_convert_as_the_kernel_casts_them() {
    assert_eq!(f16_to_f32(0x0000), 0.0);
    assert_eq!(f16_to_f32(ONE_F16), 1.0);
    assert_eq!(f16_to_f32(QUARTER_F16), 0.25);
    assert_eq!(f16_to_f32(0x3800), 0.5);
    assert_eq!(f16_to_f32(0x6000), 512.0);
    assert_eq!(f16_to_f32(0x7C00).to_bits(), f32::INFINITY.to_bits());
    assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
    assert!(f16_to_f32(0x7E00).is_nan());
}

/// The packers have to be the inverse of the kernels' unpack arithmetic, or a
/// GPU case could pass by both sides being wrong in the same way.
#[test]
#[expect(
    clippy::float_cmp,
    reason = "unit scales make both sides exact integers, so equality is the assertion"
)]
fn q4_k_packers_match_the_kernel_unpack() {
    for row in [0usize, 1, 5] {
        let values = nibbles(row);
        let scales = if row % 2 == 0 {
            UNIT_SCALES
        } else {
            MIXED_SCALES
        };

        let gathered = q4_k_oracle(&q4_k_block(&values, pack_q4_k_gathered, ONE_F16, 0, scales));
        for (e, value) in values.iter().enumerate() {
            let (scale, _min) = scale_min_k4(e / 32, 0, &scales);
            assert_eq!(
                dequant_q4_k_gathered(&gathered, e),
                f32::from(scale) * f32::from(*value),
                "gathered layout, element {e}, row {row}",
            );
        }

        let dense = q4_k_oracle(&q4_k_block(
            &values,
            pack_q4_k_dense,
            ONE_F16,
            0,
            UNIT_SCALES,
        ));
        // The matvec never shifts the high nibble down: it masks with `0xF0` and
        // divides the block scale by 16 instead, so the two cancel and the value
        // is still `nibble * d`. Both halves have to land on the same number, or
        // the matvec would be off by a factor of 16 on half its chunks.
        for (e, value) in values.iter().enumerate() {
            assert_eq!(
                dequant_q4_k_dense(&dense, e),
                f32::from(*value),
                "dense layout, element {e}, row {row}",
            );
        }
    }
}
