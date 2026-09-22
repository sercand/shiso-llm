//! What only a real device can answer.
//!
//! Every test here is `#[ignore]` and gated to the one target that has driver
//! bindings, so the default gate on a developer machine neither runs nor
//! pretends to have run them. On the machine that does have a device they are
//! run explicitly:
//!
//! ```text
//! cargo test -p shiso-cuda -- --ignored
//! ```
//!
//! Each one is deliberately free of kernel entry-point names. Names and launch
//! geometry belong to the crate that owns the kernels and the forward pass, not
//! to the backend, so what is checked here is the machinery: that a context
//! opens, that memory round-trips, that the module set this binary embeds
//! actually loads, that a capture replays to the same bytes an eager run
//! produced, and that the occupancy query answers. Anything that needs a
//! specific kernel is the kernel crate's test to write, against this same
//! backend.

#![cfg(target_os = "linux")]

use shiso_core::{Backend, GemmDType, GemmDesc, GemmOp};
use shiso_cuda::{CudaBackend, CudaOptions};

/// Opens a backend, or skips when the host has bindings but no device.
fn open() -> Option<std::sync::Arc<CudaBackend>> {
    if !shiso_cuda::cuda_available() {
        return None;
    }
    Some(CudaBackend::new(0, CudaOptions::default()).expect("opening the first device"))
}

#[test]
#[ignore = "needs a CUDA device"]
fn a_backend_opens_and_loads_every_module_this_binary_embeds() {
    let Some(backend) = open() else {
        return;
    };
    assert_eq!(
        backend.loaded_modules(),
        shiso_kernels::ptx_modules().len(),
        "every module the kernel crate embeds has to have loaded, or the registry silently \
         dropped one",
    );
    assert_eq!(backend.live_buffers(), 0);
    assert_eq!(backend.captured_graphs(), 0);
}

#[test]
#[ignore = "needs a CUDA device"]
fn host_bytes_survive_a_round_trip_through_device_memory() {
    let Some(backend) = open() else {
        return;
    };
    let handle: std::sync::Arc<dyn Backend> = backend.clone();
    let buffer = shiso_core::alloc(&handle, 4096).expect("allocation");
    assert_eq!(backend.live_buffers(), 1);

    let written: Vec<u8> = (0..4096_u32)
        .map(|i| u8::try_from(i % 251).expect("a value below 251 is a byte"))
        .collect();
    backend
        .copy_h2d(buffer.view(), &written)
        .expect("host to device");
    let mut read = vec![0_u8; 4096];
    backend
        .copy_d2h(buffer.view(), &mut read)
        .expect("device to host");
    assert_eq!(read, written);

    drop(buffer);
    assert_eq!(backend.live_buffers(), 0, "dropping a buffer frees it");
}

#[test]
#[ignore = "needs a CUDA device"]
fn a_device_to_device_copy_moves_exactly_the_range_it_was_given() {
    let Some(backend) = open() else {
        return;
    };
    let handle: std::sync::Arc<dyn Backend> = backend.clone();
    let src = shiso_core::alloc(&handle, 256).expect("allocation");
    let dst = shiso_core::alloc(&handle, 256).expect("allocation");
    let pattern: Vec<u8> = (0..256_u32)
        .map(|i| u8::try_from(i % 256).expect("a value below 256 is a byte"))
        .collect();
    backend.copy_h2d(src.view(), &pattern).expect("fill");
    backend.copy_h2d(dst.view(), &[0_u8; 256]).expect("clear");

    let from = src.slice(64, 128).expect("sub-range");
    let into = dst.slice(0, 128).expect("sub-range");
    backend.copy_d2d(into, from).expect("device to device");
    backend.sync().expect("drain");

    let mut read = vec![0_u8; 256];
    backend.copy_d2h(dst.view(), &mut read).expect("read back");
    assert_eq!(&read[..128], &pattern[64..192]);
    assert!(
        read[128..].iter().all(|&b| b == 0),
        "the copy must not touch a byte past the range it was given",
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn a_view_that_runs_past_its_allocation_is_refused_before_the_device_sees_it() {
    let Some(backend) = open() else {
        return;
    };
    let handle: std::sync::Arc<dyn Backend> = backend.clone();
    let buffer = shiso_core::alloc(&handle, 64).expect("allocation");
    let mut too_long = vec![0_u8; 128];
    assert!(backend.copy_h2d(buffer.view(), &too_long).is_err());
    assert!(backend.copy_d2h(buffer.view(), &mut too_long).is_err());
}

#[test]
#[ignore = "needs a CUDA device"]
fn a_captured_copy_replays_to_the_same_bytes_the_eager_one_produced() {
    let Some(backend) = open() else {
        return;
    };
    let handle: std::sync::Arc<dyn Backend> = backend.clone();
    let graphs = backend
        .as_graph()
        .expect("graphs are on in the default options");

    let src = shiso_core::alloc(&handle, 512).expect("allocation");
    let dst = shiso_core::alloc(&handle, 512).expect("allocation");
    let pattern: Vec<u8> = (0..512_u32)
        .map(|i| u8::try_from(i % 97).expect("a value below 97 is a byte"))
        .collect();
    backend.copy_h2d(src.view(), &pattern).expect("fill");

    let key = shiso_core::GraphKey::new(1);
    assert!(
        !graphs.replay(key).expect("replay of an empty slot"),
        "an unfilled slot reports that it is unfilled rather than failing",
    );

    {
        let capture = graphs.capture(key).expect("capture");
        backend
            .copy_d2d(dst.view(), src.view())
            .expect("a device-to-device copy is capturable");
        capture.finish().expect("install");
    }
    assert_eq!(backend.captured_graphs(), 1);

    backend.copy_h2d(dst.view(), &[0_u8; 512]).expect("clear");
    assert!(graphs.replay(key).expect("replay"));
    backend.sync().expect("drain");

    let mut read = vec![0_u8; 512];
    backend.copy_d2h(dst.view(), &mut read).expect("read back");
    assert_eq!(read, pattern, "the replay has to do what the capture did");

    graphs.invalidate_all();
    assert_eq!(backend.captured_graphs(), 0);
    assert!(!graphs.replay(key).expect("replay after invalidation"));
}

#[test]
#[ignore = "needs a CUDA device"]
fn an_abandoned_capture_leaves_the_stream_usable() {
    let Some(backend) = open() else {
        return;
    };
    let handle: std::sync::Arc<dyn Backend> = backend.clone();
    let graphs = backend.as_graph().expect("graphs are on");
    let buffer = shiso_core::alloc(&handle, 64).expect("allocation");

    {
        // Dropped without finishing, which is what an early `?` inside a
        // captured region looks like.
        let _capture = graphs
            .capture(shiso_core::GraphKey::new(2))
            .expect("capture");
    }
    assert_eq!(backend.captured_graphs(), 0);
    backend
        .copy_h2d(buffer.view(), &[1_u8; 64])
        .expect("the work stream still works after an abandoned capture");
    backend.sync().expect("drain");
}

#[test]
#[ignore = "needs a CUDA device"]
fn allocation_is_refused_while_a_capture_is_running() {
    let Some(backend) = open() else {
        return;
    };
    let handle: std::sync::Arc<dyn Backend> = backend.clone();
    let graphs = backend.as_graph().expect("graphs are on");
    let _capture = graphs
        .capture(shiso_core::GraphKey::new(3))
        .expect("capture");
    assert!(
        shiso_core::alloc(&handle, 64).is_err(),
        "a graph bakes in the addresses it was captured with, so a fresh allocation inside one \
         would be named by a graph that outlives it",
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn a_second_ordinal_in_one_process_is_refused() {
    if !shiso_cuda::cuda_available() {
        return;
    }
    let _first = CudaBackend::new(0, CudaOptions::default()).expect("the first device");
    assert!(
        CudaBackend::new(1, CudaOptions::default()).is_err(),
        "two contexts on two devices would make one backend's buffers unreachable from the \
         other's launches",
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn turning_graphs_off_removes_the_capability_rather_than_failing_later() {
    if !shiso_cuda::cuda_available() {
        return;
    }
    let backend = CudaBackend::new(0, CudaOptions::default().with_decode_graphs(false))
        .expect("opening the first device");
    assert!(backend.as_graph().is_none());
    assert!(backend.as_blas().is_some());
    assert!(backend.as_coop().is_some());
}

#[test]
#[ignore = "needs a CUDA device"]
fn the_accumulation_mode_can_be_changed_after_the_backend_is_open() {
    let Some(backend) = open() else {
        return;
    };
    backend
        .set_tf32(false)
        .expect("full-precision accumulation");
    backend
        .set_tf32(true)
        .expect("reduced-precision accumulation");
}

// Left for the crate that owns the kernels and the forward pass, because each
// needs an entry-point name and a launch geometry that this crate deliberately
// does not know:
//
//   * resolving every entry point the kernel name table lists, so a rename in a
//     kernel fails at load rather than at the first token;
//   * a trivial launch whose output is compared against a host computation;
//   * a launch above the default shared-memory ceiling, proving the opt-in
//     happens once and that the second launch of the same entry point skips it
//     (`entry_cache_hits` is what makes the second half observable);
//   * `max_co_resident_blocks` on a cooperative entry point, and a cooperative
//     launch at exactly that grid.

#[test]
#[ignore = "needs a CUDA device"]
fn the_first_cooperative_reservation_returns_and_hands_back_a_zeroed_slab() {
    // Two things this settles, and neither can be settled anywhere else.
    //
    // The first is that it returns at all. The reservation takes the backend's
    // state lock and the allocation beneath it takes the same one, which is not
    // reentrant - so a reservation that held the lock across the allocation
    // would hang here rather than fail, and nothing on a machine without a
    // device ever reaches it.
    //
    // The second is the contents. The lock chain the cooperative kernels hand
    // partial sums down sits at the head of this slab and every one of them
    // *leaves* it zeroed rather than initializing it, so a slab that arrives
    // holding the last tenant's bytes is a grid waiting on a barrier nobody
    // will release. Reading it back is the only way to see that.
    let Some(backend) = open() else {
        return;
    };
    let coop = backend
        .as_coop()
        .expect("a CUDA device launches cooperatively");
    let bytes = 1 << 20;
    coop.reserve_scratch(bytes).expect("the first reservation");
    let slab = coop.scratch().expect("a reservation leaves a slab");
    assert!(slab.len() >= bytes);

    let mut read = vec![0xA5_u8; usize::try_from(bytes).expect("a mebibyte fits")];
    backend
        .copy_d2h(slab.slice(0, bytes).expect("the slab holds it"), &mut read)
        .expect("the slab reads back");
    assert!(
        read.iter().all(|byte| *byte == 0),
        "a fresh cooperative slab was handed over unzeroed"
    );

    // Asking again for what it already has is an answer, not a second
    // allocation, and growing it still returns.
    let live = backend.live_buffers();
    coop.reserve_scratch(bytes).expect("the same size again");
    assert_eq!(backend.live_buffers(), live, "the slab was reallocated");
    coop.reserve_scratch(bytes * 2).expect("growing the slab");
}

#[test]
#[ignore = "needs a CUDA device"]
fn a_half_precision_multiply_writes_single_precision_into_the_destination() {
    // The shape the sparse stack's wide half-precision projection builds:
    // `C = A^T B` with A the weight stored one output row at a time, B the
    // chunk's activations, and C a single-precision destination. The operand
    // type and the destination type are different, and a library call that used
    // the operand type for both succeeds and writes halves into a float buffer -
    // so the check is on the numbers, not on the status.
    /// Contraction width of the multiply under test.
    const K: u32 = 64;
    /// Output width: the rows of the weight.
    const M: u32 = 8;
    /// Rows of the chunk.
    const N: u32 = 3;

    let Some(backend) = open() else { return };
    let handle: std::sync::Arc<dyn Backend> = backend.clone();
    let Some(blas) = handle.as_blas() else {
        panic!("this backend offers no matrix tier")
    };
    let ramp = |count: u32, period: u32, mid: f32, scale: f32| -> Vec<f32> {
        (0..count)
            .map(|i| (f32::from(u16::try_from(i % period).unwrap_or(0)) - mid) / scale)
            .collect()
    };
    let weight = ramp(M * K, 7, 3.0, 8.0);
    let acts = ramp(N * K, 5, 2.0, 4.0);
    let halves = |v: &[f32]| -> Vec<u8> {
        v.iter()
            .flat_map(|x| half::f16::from_f32(*x).to_le_bytes())
            .collect()
    };
    let a = shiso_core::alloc(&handle, u64::from(M * K * 2)).expect("weight");
    let b = shiso_core::alloc(&handle, u64::from(N * K * 2)).expect("activations");
    let c = shiso_core::alloc(&handle, u64::from(N * M * 4)).expect("destination");
    handle.copy_h2d(a.view(), &halves(&weight)).expect("upload");
    handle.copy_h2d(b.view(), &halves(&acts)).expect("upload");

    blas.gemm(&GemmDesc {
        op_a: GemmOp::Transposed,
        op_b: GemmOp::AsStored,
        m: M,
        n: N,
        k: K,
        a: a.view(),
        lda: K,
        b: b.view(),
        ldb: K,
        c: c.view(),
        ldc: M,
        alpha: 1.0,
        beta: 0.0,
        dtype: GemmDType::F16,
    })
    .expect("the multiply runs");
    handle.sync().expect("the stream drains");

    let mut bytes = vec![0u8; (N * M * 4) as usize];
    handle.copy_d2h(c.view(), &mut bytes).expect("read back");
    let got: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect();
    let at = |row: u32, col: u32, stride: u32| (row * stride + col) as usize;
    for row in 0..N {
        for out in 0..M {
            let want: f32 = (0..K)
                .map(|i| {
                    f32::from(half::f16::from_f32(acts[at(row, i, K)]))
                        * f32::from(half::f16::from_f32(weight[at(out, i, K)]))
                })
                .sum();
            let seen = got[at(row, out, M)];
            assert!(
                (seen - want).abs() < 1e-2,
                "row {row} column {out}: {seen} against {want}",
            );
        }
    }
}

#[test]
#[ignore = "needs a CUDA device"]
fn a_packed_pointer_array_holds_the_address_of_each_range_and_nothing_else() {
    // The one thing a caller cannot check for itself: a `DeviceView` is three
    // numbers and none of them is an address, so whether the slots a batched
    // kernel dereferences hold the right addresses is answerable only here.
    //
    // It is checked the only way it can be without a kernel: the packed array
    // is read back and each slot compared against the addresses of ranges whose
    // *relative* offsets are known. Two allocations have unrelated addresses,
    // but a slice of one is exactly its offset past the whole of it, and a
    // packer that wrote the first source into every slot, or wrote handles
    // instead of addresses, breaks that relation.
    let Some(backend) = open() else {
        return;
    };
    let handle: std::sync::Arc<dyn Backend> = backend.clone();
    let pack = shiso_core::alloc(&handle, 4 * 8).expect("allocation");
    let first = shiso_core::alloc(&handle, 4096).expect("allocation");
    let second = shiso_core::alloc(&handle, 4096).expect("allocation");
    let pointers = handle
        .as_pointers()
        .expect("a CUDA device hands out the addresses of its own allocations");
    let srcs = [
        first.view(),
        first.slice(256, 16).expect("inside the allocation"),
        second.view(),
        second.slice(1024, 16).expect("inside the allocation"),
    ];
    pointers
        .pack_pointers(pack.view(), &srcs)
        .expect("four ranges fit four slots");

    let mut raw = [0_u8; 4 * 8];
    handle
        .copy_d2h(pack.view(), &mut raw)
        .expect("the pack reads back");
    let held: Vec<u64> = raw
        .as_chunks::<8>()
        .0
        .iter()
        .map(|slot| u64::from_ne_bytes(*slot))
        .collect();
    assert_eq!(
        held[1] - held[0],
        256,
        "a slice is its offset past the whole"
    );
    assert_eq!(held[3] - held[2], 1024);
    assert_ne!(held[0], held[2], "two allocations share an address");
    assert!(held.iter().all(|address| *address != 0));

    // A destination too short for the slots is a refusal rather than a partial
    // write: an array of pointers with a hole in it is a kernel dereferencing
    // whatever was there before.
    let narrow = shiso_core::alloc(&handle, 8).expect("allocation");
    pointers
        .pack_pointers(narrow.view(), &srcs)
        .expect_err("one slot does not hold four pointers");
}
