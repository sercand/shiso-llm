# shiso on NVIDIA CUDA (Linux, arm64 or amd64).
#
# Two stages. The builder is the CUDA `devel` image because the kernel tree is
# compiled to PTX by `shiso-kernels`' build script at build time (gated behind
# SHISO_CUDA_COMPILE=1), and because cuBLAS has to be linked. The runtime is
# distroless `cc` because the binary is glibc-dynamic and cuBLAS is a C++
# library. Only the two binaries and the cuBLAS shared objects cross the stage
# boundary: no nvcc, no headers, no shell, no model weights.
#
# Build on the target architecture (arm64 for the DGX Spark):
#   container build -t shiso-cuda .
#   container build --build-arg SHISO_CUDA_ARCH=sm_121a -t shiso-cuda .
#
# The kernel manifests default to compute_75 PTX, which the driver JITs on any
# newer device. SHISO_CUDA_ARCH replaces that list wholesale when a fixed
# architecture is wanted.

ARG CUDA_VERSION=13.0.3
ARG RUST_VERSION=1.100.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS builder

ARG RUST_VERSION
# Empty by default: the kernel manifests' own architecture list is used. A
# value here (e.g. sm_121a) replaces that list wholesale.
ARG SHISO_CUDA_ARCH=

# cudarc's build script reads the toolkit version from the build system, so
# CUDA_HOME and nvcc must be visible. Both already are in the CUDA image; they
# are restated so a base change fails the kernel compile instead of silently
# producing an empty registry.
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/usr/local/cargo/bin:${PATH} \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    DEBIAN_FRONTEND=noninteractive

# curl for rustup; ca-certificates for the fetch. gcc/g++/make (present in the
# devel image) cover the C and C++ build scripts: ring, onig_sys, esaxx-rs and
# the bundled SQLite.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# rust-toolchain.toml pins the same channel. Installing it up front keeps a
# missing channel a clear failure here rather than a surprise inside the build.
RUN curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o /tmp/rustup.sh \
 && sh /tmp/rustup.sh -y --profile minimal --no-modify-path --default-toolchain "${RUST_VERSION}" \
 && rm /tmp/rustup.sh \
 && rustc --version

WORKDIR /src
COPY . .

# `--locked` because Cargo.lock is committed. `shiso-server/cuda` keeps the
# default `cpu` and `decode-ffmpeg` features, matching a bare
# `--features cuda --features decode-native` build; `decode-native` adds the
# pure-Rust media decoders. Both binaries are built in one invocation so the
# registry shares the dependency graph.
#
# The target directory is a cache mount, so it is invisible to later layers:
# the binaries are copied out inside the same RUN.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    set -eux; \
    export SHISO_CUDA_COMPILE=1; \
    if [ -n "${SHISO_CUDA_ARCH}" ]; then export SHISO_CUDA_ARCH; fi; \
    cargo build --release --locked \
        -p shiso-server -p shiso-registry \
        --features shiso-server/cuda,shiso-server/decode-native; \
    mkdir -p /out/lib; \
    cp target/release/shiso target/release/shiso-registry /out/; \
    strip /out/shiso /out/shiso-registry; \
    cuda_lib=$(dirname "$(find -L /usr/local/cuda -name 'libcublasLt.so.13*' -print -quit)"); \
    for pattern in libcublas.so.13 libcublasLt.so.13 libnvJitLink.so.13; do \
        for file in "$cuda_lib/$pattern"*; do \
            if [ -e "$file" ]; then cp -a "$file" /out/lib/; fi; \
        done; \
    done; \
    ls -l /out/lib; \
    ldd /out/lib/libcublasLt.so.13* || true; \
    if LD_LIBRARY_PATH=/out/lib ldd /out/lib/libcublasLt.so.13* \
        | grep 'not found' | grep -v 'libcuda\.so\.1'; then \
        echo 'cuBLAS has an unresolved dependency' >&2; exit 1; \
    fi

# `cc` rather than `base`: cuBLAS needs libstdc++/libgomp. glibc here is 2.41,
# newer than the 2.39 the Ubuntu 24.04 builder links against, which is the
# direction that is safe. The tag's user is 65532.
FROM gcr.io/distroless/cc-debian13:nonroot

# The cuBLAS pair and its toolkit-local dependencies live here. There is no
# ldconfig in this image, so the loader is pointed at the directory directly.
ENV LD_LIBRARY_PATH=/usr/local/lib \
    HOME=/home/nonroot \
    SHISO_HOME=/home/nonroot/.shiso \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

COPY --from=builder /out/shiso /out/shiso-registry /usr/local/bin/
COPY --from=builder /out/lib/ /usr/local/lib/

# No default subcommand: `serve` is an operator decision (model, bind address,
# device), so the image only supplies the entry point. Kubernetes passes
# `args: ["serve", ...]`, and `--help` keeps a bare `run` informative.
ENTRYPOINT ["/usr/local/bin/shiso"]
CMD ["--help"]

EXPOSE 8090 8095
