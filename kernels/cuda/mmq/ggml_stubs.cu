// SPDX-License-Identifier: MIT
// Implementations of the ggml-API stubs declared in ggml_stubs.h plus
// the bodies of ggml_backend_cuda_context / ggml_cuda_info /
// new_pool_for_device that the vendored common.cuh declares without
// defining.
//
// Phase 0: pool is plain cudaMallocAsync / cudaFreeAsync. Phase 4 swaps
// in shiso's existing cuda_tmp_alloc slab allocator.

#include <execinfo.h>
#include "common.cuh"   // pulls in ggml_stubs.h via redirect headers

#if defined(GGML_USE_HIP)
#include "vendors/hip.h"
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>

// ----------------------------------------------------------------------------
// Device info singleton.
//
// Common.cuh declares `const ggml_cuda_device_info & ggml_cuda_info();` -
// we provide the body. The struct layout (common.cuh:1091) is:
//   { int device_count; cuda_device_info devices[GGML_CUDA_MAX_DEVICES];
//     std::array<float, GGML_CUDA_MAX_DEVICES> default_tensor_split; }
// where cuda_device_info has { cc, nsm, smpb, smpbo, integrated, vmm,
// vmm_granularity, total_vram, warp_size, supports_cooperative_launch }.
// ----------------------------------------------------------------------------

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info;
    static std::once_flag once;
    std::call_once(once, []{
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess) {
            fprintf(stderr, "ggml_cuda_info: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
            count = 0;
        }
        if (count > GGML_CUDA_MAX_DEVICES) count = GGML_CUDA_MAX_DEVICES;
        info.device_count = count;
        for (int i = 0; i < count; i++) {
            cudaDeviceProp p;
            CUDA_CHECK(cudaGetDeviceProperties(&p, i));
#if defined(GGML_USE_HIP)
            unsigned gfx_id = 0;
            if (sscanf(p.gcnArchName, "gfx%x", &gfx_id) == 1) {
                info.devices[i].cc = GGML_CUDA_CC_OFFSET_AMD + (int)gfx_id;
            } else {
                info.devices[i].cc = GGML_CUDA_CC_OFFSET_AMD + p.major * 0x100 + p.minor * 0x10;
            }
#else
            info.devices[i].cc                          = p.major * 100 + p.minor * 10;
#endif
            info.devices[i].nsm                         = p.multiProcessorCount;
            info.devices[i].smpb                        = p.sharedMemPerBlock;
#if defined(GGML_USE_HIP)
            info.devices[i].smpbo                       = p.sharedMemPerBlock;
#else
            info.devices[i].smpbo                       = p.sharedMemPerBlockOptin;
#endif
            info.devices[i].integrated                  = p.integrated != 0;
            info.devices[i].vmm                         = false;
            info.devices[i].vmm_granularity             = 0;
            info.devices[i].total_vram                  = p.totalGlobalMem;
            info.devices[i].warp_size                   = p.warpSize;
            info.devices[i].supports_cooperative_launch = p.cooperativeLaunch != 0;
        }
    });
    return info;
}

// ggml_cuda_get_device / ggml_cuda_set_device are declared (not defined) in
// common.cuh. We provide thin wrappers.

int ggml_cuda_get_device() {
    int dev = 0;
    (void)cudaGetDevice(&dev);
    return dev;
}

void ggml_cuda_set_device(int device) {
    int cur = -1;
    (void)cudaGetDevice(&cur);
    if (cur != device) {
        CUDA_CHECK(cudaSetDevice(device));
    }
}

int64_t ggml_time_us() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();
}

// ----------------------------------------------------------------------------
// ggml_cuda_error: invoked by the CUDA_CHECK / CUBLAS_CHECK macros defined
// in common.cuh on the error path. Marked [[noreturn]] in the declaration
// (common.cuh:155) - abort() satisfies that contract.
// ----------------------------------------------------------------------------

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "CUDA error: %s\n  call: %s\n  in: %s at %s:%d\n", msg, stmt, func, file, line);
    fflush(stderr);
    abort();
}

// ----------------------------------------------------------------------------
// Concrete pool wrapping cudaMallocAsync / cudaFreeAsync.
// ----------------------------------------------------------------------------

namespace {

/* Thread-local stream that shs_naive_pool uses for cudaMallocAsync /
 * cudaFreeAsync.  Defaults to cudaStreamPerThread (preserves prior
 * behaviour).  Step 8 / CUDA Graphs sets this to the capture stream
 * before allocating, so the alloc node lives on the captured stream
 * and capture is not invalidated.  Set via shs_pool_set_stream() from
 * mmq.cu wrappers; an explicit stream=0 means the legacy default
 * stream so pool ops stay ordered with legacy-stream kernels. */
static thread_local cudaStream_t t_shs_pool_stream = cudaStreamPerThread;

} // anonymous namespace

extern "C" void shs_pool_set_stream(cudaStream_t stream) {
    t_shs_pool_stream = stream;
}

extern "C" cudaStream_t shs_pool_get_stream(void) {
    return t_shs_pool_stream;
}

namespace {

struct shs_naive_pool : public ggml_cuda_pool {
    int device;

    explicit shs_naive_pool(int device) : device(device) {}

    void * alloc(size_t size, size_t * actual_size) override {
        ggml_cuda_set_device(device);
        void * ptr = nullptr;
        /* SHISO_POOL_TRACE=1: who allocates from the pool, and on which stream.
         * Used to find allocations that land inside a graph capture. */
        static const bool trace = getenv("SHISO_POOL_TRACE") != nullptr;
        if (trace) {
            cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
            (void)cudaStreamIsCapturing(t_shs_pool_stream, &st);
            fprintf(stderr, "shiso pool alloc %zu bytes stream=%p capturing=%d\n",
                    size, (void *)t_shs_pool_stream, (int)st);
            void *bt[8];
            const int n = backtrace(bt, 8);
            backtrace_symbols_fd(bt, n, 2);
        }
        CUDA_CHECK(cudaMallocAsync(&ptr, size, t_shs_pool_stream));
        if (actual_size) *actual_size = size;
        return ptr;
    }

    void free(void * ptr, size_t /*size*/) override {
        if (!ptr) return;
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFreeAsync(ptr, t_shs_pool_stream));
    }
};

} // anonymous namespace

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device, int /*stream_no*/) {
    return std::unique_ptr<ggml_cuda_pool>(new shs_naive_pool(device));
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    if (copy_event) {
        (void)cudaEventDestroy(copy_event);
        copy_event = nullptr;
    }
    // streams[][], cublas_handles[], and pools[][] are owned-by-value
    // (cudaStream_t and cublasHandle_t are opaque handles - destroying the
    // context "should" tear them down, but in our shim shiso manages streams
    // externally and we leave them alone. The pools auto-destruct via
    // unique_ptr.).
}
