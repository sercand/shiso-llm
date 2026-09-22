# Provenance

Where the code in this repository came from. This file and the Acknowledgements
section of `README.md` are the only places upstream projects are named (rule 15);
`scripts/api_gate.sh` checks that every workspace member has a row here.

Some of the CUDA kernels came from [antirez/ds4](https://github.com/antirez/ds4)
(MIT): the q4e kernels, and the adapter layers ds4 wrote around the vendored
llama.cpp and exllamav3 kernels. That is why the `VENDOR.md` files still use
the name "ds4" for the host project.

| Path | Upstream | Relationship |
|---|---|---|
| `kernels/cuda/q4e` | [antirez/ds4](https://github.com/antirez/ds4), MIT | ported from ds4 |
| `kernels/cuda/mmq` | [llama.cpp](https://github.com/ggml-org/llama.cpp) `ggml-cuda`, MIT; ds4 adapter | vendored, pinned commit, reached through ds4's shim (`ggml_stubs`, `mmq.{h,cu}`); see `kernels/cuda/mmq/VENDOR.md` |
| `kernels/cuda/exl3` | [exllamav3](https://github.com/turboderp-org/exllamav3), MIT; ds4 modifications | vendored and modified, pinned commit; see `kernels/cuda/exl3/VENDOR.md` |
| `crates/shiso-core` | - | original to this repository |
| `crates/shiso-modelfile` | llama.cpp (GGUF format, `ggml-quants` block layouts); MLX (affine quantization); NVIDIA modelopt (FP8/NVFP4 exports) | file-format readers written against those formats |
| `crates/shiso-kvstore` | Marconi (hybrid-model prefix caching) | eviction policy follows the paper's utility scoring |
| `crates/shiso-engine` | - | original to this repository |
| `crates/shiso-models` | Hugging Face `transformers`, MLX, llama.cpp; the Splash package format | per-family configs and tensor names follow the reference exports |
| `crates/shiso-ir` | - | original; shaped after the OpenAI Responses and Anthropic Messages APIs |
| `crates/shiso-parsers` | - | original; tag formats are the model families' own |
| `crates/shiso-vision` | Hugging Face `transformers`, mlx-vlm | preprocessing matches the reference processors |
| `crates/shiso-ple` | - | original to this repository |
| `crates/shiso-metal` | llama.cpp Metal kernels; Splash Metal runtime kernels | several MSL kernels ported from those sources |
| `crates/shiso-cuda` | - | original to this repository |
| `crates/shiso-kernels` | ds4, llama.cpp, exllamav3 | build and embed the kernel trees above |
| `crates/shiso-kerneltest` | Splash Metal runtime kernels | some CPU oracles restate that arithmetic |
| `crates/shiso-server` | - | original; wire formats follow the OpenAI and Anthropic APIs |
| `crates/shiso-bench` | llama.cpp, `mlx-lm`, `transformers` | independent reference fixtures are captured from these |
