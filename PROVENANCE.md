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
| `kernels/cuda/audio` | - | original; the CUDA twin of `crates/shiso-metal/metal/audio`, nothing copied |
| `crates/shiso-core` | - | original to this repository |
| `crates/shiso-modelfile` | llama.cpp (GGUF format, `ggml-quants` block layouts); MLX (affine quantization); NVIDIA modelopt (FP8/NVFP4 exports) | file-format readers written against those formats |
| `crates/shiso-kvstore` | Marconi (hybrid-model prefix caching) | eviction policy follows the paper's utility scoring |
| `crates/shiso-engine` | - | original to this repository |
| `crates/shiso-models` | Hugging Face `transformers`, MLX, llama.cpp; the Splash package format | per-family configs and tensor names follow the reference exports |
| `crates/shiso-ir` | - | original; shaped after the OpenAI Responses and Anthropic Messages APIs |
| `crates/shiso-parsers` | - | original; tag formats are the model families' own |
| `crates/shiso-vision` | Hugging Face `transformers`, mlx-vlm | preprocessing matches the reference processors |
| `crates/shiso-audio` | Hugging Face `transformers` feature extractors, librosa (Slaney mel filterbank), NVIDIA NeMo (log-mel frontend conventions) | original; frontends are written to match those references numerically, nothing is vendored |
| `crates/shiso-models/src/gemma4/audio` | Hugging Face `transformers` (`Gemma4AudioModel`), Apache-2.0; llama.cpp mmproj tensor names (`a.*`), MIT | original; the graph is written to match the reference numerically and reads both exports' names, nothing is vendored |
| `crates/shiso-models/src/qwen3_asr` | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), Apache-2.0; [netease-youdao/Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2), NetEase Model Use License Agreement (weights), Apache-2.0 (code); the `qwen-asr` 0.0.6 reference package (`Qwen3ASRForConditionalGeneration`, its processor and inference utilities), Apache-2.0; llama.cpp mmproj and `qwen3vl` GGUF tensor names, MIT; MLX export layout | original; configs, tensor names and constants follow the reference exports and package, and the graph is written to match the reference numerically; nothing is vendored |
| `crates/shiso-models/src/nemotron_diar` | Hugging Face `transformers` (`Nemotron3DiarizationForAudioFrameClassification`), Apache-2.0 | the speaker-cache update and compression (`cache.rs`) is a port of the reference's offline-mode algorithm; the encoder and head are written to match the reference numerically; nothing is vendored |
| `crates/shiso-models/src/nemotron_h`, `crates/shiso-engine/src/ssm.rs`, `crates/shiso-engine/src/moe_biased.rs`, `crates/shiso-metal/metal/dense/dense_ssm.metal`, `crates/shiso-metal/metal/dense/dense_moe.metal` | [mlx-lm](https://github.com/ml-explore/mlx-lm) 0.31.3 (`models/nemotron_h.py`, `models/ssm.py`), MIT; the NVIDIA Nemotron-H config and MLX export layout | original; the state-space block (convolution, softplus step, diagonal recurrence, gated grouped norm), the biased sigmoid router and the squared-rectifier experts are written to match the reference model numerically and graded against it (`crates/shiso-bench/tests/parity_nemotron_h.rs`); nothing is vendored |
| `crates/shiso-ple` | - | original to this repository |
| `crates/shiso-metal` | llama.cpp Metal kernels; Splash Metal runtime kernels | several MSL kernels ported from those sources |
| `crates/shiso-metal/metal/dense/dense_exact_affine.metal` | [TensorFold](https://github.com/ashhart/TensorFold) 0.3.4 (`bb4b4a3`), `kernels/qwen/dense/v1/simd_qmm.py`, MIT | ported: the row-exact affine matmul's arithmetic (one FMA chain per group over unshifted values against pre-scaled inputs, shape-fixed chunking, the 8x8 simdgroup MMA whose rows equal a one-row chain) and the one-row kernel's staging; adapted to f32 activations and outputs and extended to 8-bit weights |
| `crates/shiso-engine/src/block_draft.rs`, `crates/shiso-models/src/qwen35/block_draft.rs`, `crates/shiso-metal/metal/dense/dense_block_draft.metal` | [z-lab/dflash](https://github.com/z-lab/dflash) `model_mlx.py` (the DFlash2 block drafter reference), MIT | written for this repository to the reference's graph: context from the target's tapped layers, grouped dynamic causal convolutions around each block, non-causal block attention over a windowed context, and the greedy low-rank candidate selector; nothing is vendored |
| `crates/shiso-engine/src/quantize.rs` | [MLX](https://github.com/ml-explore/mlx) `affine_quantize`, MIT | written for this repository to the same quantization rule, so a weight quantized at load matches the MLX exports' own packing |
| `crates/shiso-metal/metal/q4e/affine.metal` | [MLX](https://github.com/ml-explore/mlx) Metal quantized kernels (`qmv_fast`, `qmm_t`), MIT | written for this repository; the one-row projections' pre-scaled four-bit dot and the prompt projections' staged-tile matrix multiply follow the technique of those kernels |
| `crates/shiso-cuda` | - | original to this repository |
| `crates/shiso-kernels` | ds4, llama.cpp, exllamav3 | build and embed the kernel trees above |
| `crates/shiso-kerneltest` | Splash Metal runtime kernels | some CPU oracles restate that arithmetic |
| `crates/shiso-server` | - | original; wire formats follow the OpenAI and Anthropic APIs |
| `crates/shiso-router` | [sercand/llm-router](https://github.com/sercand/llm-router), MIT | full copy of llm-router |
| `crates/shiso-bench` | llama.cpp, `mlx-lm`, `transformers` | independent reference fixtures are captured from these |
| `scripts/reference/` | Hugging Face `transformers`, Apache-2.0 | reference tooling. `render_chat_fixtures.py` calls the reference chat-template implementation from `transformers` at run time; nothing is vendored |
| `scripts/reference/block_draft_reference.py` | mlx-lm 0.31.3 and MLX 0.31.2, MIT; the z-lab/dflash reference drafter (`model_mlx.py`, passed at run time), MIT | original; imports and runs both references to capture a block drafter's first block and greedy acceptance; nothing is vendored |
| `scripts/reference/mlx_logit_vectors.py` | [mlx-lm](https://github.com/ml-explore/mlx-lm) 0.31.3 and MLX 0.31.2, MIT; `tokenizers` | original; imports and runs the reference's own model loader and forward pass to capture independent logit fixtures; nothing is vendored |
| `scripts/reference/qwen4exp_fp8_ref.py` | Hugging Face `transformers` 5.16.1 (`modeling_qwen4_exp`), Apache-2.0; `torch`, `safetensors`, `tokenizers` | original; imports and runs the `transformers` modules as the independent reference and replaces only its expert and n-gram weight storage |
| `scripts/reference/audio_reference.py` | Hugging Face `transformers` 5.18.0.dev0, Apache-2.0; `torch`, `librosa`, `soundfile`; ffmpeg (clip conversion) | original; imports and runs the reference feature extractors and the reference models (the streaming ASR model, the speaker diarizer and the Gemma 4 audio tower); nothing is vendored |
