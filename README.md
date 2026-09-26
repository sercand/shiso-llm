# shiso

**Fast LLM inference on the hardware in front of you.**

shiso is an inference engine and OpenAI/Anthropic-compatible server for
hybrid-attention and mixture-of-experts models. It runs natively on Apple
Silicon through Metal and on NVIDIA GPUs through CUDA, as one Rust binary with
no Python runtime.

## Why shiso

- **Built for hybrid models.** Recurrent/attention hybrids like Qwen 3.5 keep a
  recurrent state that ordinary prefix caches throw away. shiso checkpoints that
  state alongside the paged KV cache, so a shared prefix is served from cache
  instead of being prefilled again.
- **Prefix caching that survives restarts.** A radix-tree KV cache with
  cost-aware eviction, plus an optional disk tier (`--kvstore-dir`) that stores
  content-addressed pages and checkpoints behind the request path. A prompt or
  a conversation seen by an earlier process skips prefill.
- **Efficient serving.** Continuous batching, chunked prefill that never stalls
  a stream already decoding, decode-graph replay, and speculative decoding
  with MTP draft heads.
- **Quantized weights, read in place.** GGUF (Q4_K_M and friends) is
  memory-mapped with no conversion step. MLX 4-bit, FP8, NVFP4 and EXL3
  checkpoints are supported as well.
- **Drop-in APIs.** OpenAI Chat Completions, Completions and Responses, and
  Anthropic Messages, with streaming, reasoning output and tool calling.
- **Multimodal.** Image input, plus video input through ffmpeg (opt-in).

## Supported models

| Family | Example checkpoints | Formats |
|---|---|---|
| Qwen 3.5 (dense hybrid) | Qwen3.5-4B | GGUF, MLX |
| Qwen 3.6 MoE | Qwen3.6-35B-A3B | GGUF, NVFP4, Splash |
| Gemma 4 | Gemma 4 E4B | GGUF, MLX |
| Qwen4Exp (MoE) | - | EXL3 |

Vision towers are supported for Qwen 3.5, Qwen 3.6 MoE and Gemma 4.

Speech models (for the audio endpoints): the Nemotron 3.5 streaming ASR
(HF, MLX and GGUF exports) and Nemotron-3-Diarization (HF, MLX, and bf16 and
q8_0 GGUF exports).

Gemma 4 also takes audio input: `input_audio` parts on Chat Completions and
Responses, up to 30 seconds a clip. The audio tower loads from the same
`--vision` path, either the MLX export directory or the mmproj GGUF. WAV is
decoded natively; other containers need `--audio-allow-ffmpeg`.

## Supported hardware

- **Apple Silicon** Macs (Metal)
- **NVIDIA** GPUs on Linux (CUDA). Development happens on a DGX Spark (GB10).

## Getting started

### Build

You need Rust; the toolchain is pinned in `rust-toolchain.toml` and `rustup`
installs it on first use.

```sh
git clone <this repository> shiso && cd shiso

# macOS
cargo build --release -p shiso-server --features metal --bin shiso

# Linux + NVIDIA (requires the CUDA toolkit)
SHISO_CUDA_COMPILE=1 cargo build --release -p shiso-server --features cuda --bin shiso
```

### Serve a model

Point shiso at a checkpoint and the `config.json` of its Hugging Face release.
The tokenizer and chat template are picked up from the same directory.

```sh
./target/release/shiso serve \
    --model ~/models/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf \
    --config ~/models/Qwen3.5-4B/config.json \
    --ctx 8192
```

The server listens on `http://127.0.0.1:8090`.

### Send a request

```sh
curl http://127.0.0.1:8090/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "Qwen3.5-4B-Q4_K_M",
        "messages": [{"role": "user", "content": "Write a haiku about caching."}],
        "stream": true
    }'
```

Any OpenAI or Anthropic client works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="unused")
reply = client.chat.completions.create(
    model="Qwen3.5-4B-Q4_K_M",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(reply.choices[0].message.content)
```

### Transcribe audio

`--model` is optional when a speech recognizer is given; the server then
serves transcription alone.

```sh
./target/release/shiso serve \
    --asr-model ~/models/nemotron-3.5-asr-streaming-0.6b/mlx --asr-id nemotron-asr \
    --diarization-model ~/models/Nemotron-3-Diarization/mlx

curl http://127.0.0.1:8090/v1/audio/transcriptions \
    -F model=nemotron-asr -F file=@speech.wav
curl http://127.0.0.1:8090/v1/audio/transcriptions \
    -F model=nemotron-asr -F response_format=diarized_json -F file=@meeting.wav
```

`response_format` is one of `json`, `text`, `verbose_json` (with
`timestamp_granularities[]` of `segment` and/or `word`), `srt`, `vtt` and
`diarized_json`; `stream=true` streams `transcript.text.delta` events (or, for
`diarized_json`, `transcript.text.segment` events) ending in
`transcript.text.done`.

## Configuration

Run `shiso serve --help` for every option. Each flag also reads a `SHISO_*`
environment variable. The ones you are most likely to want:

| Flag | What it does |
|---|---|
| `--ctx` | Tokens one session may hold, prompt included (default 4096) |
| `--max-num-seqs` | Sequences resident at once (default 4) |
| `--kv-cache-tokens` | Tokens the shared KV cache holds across all sessions (default 131072) |
| `--gpu-memory-utilization` | Size the KV cache to fill this share of device memory instead, in (0, 1] |
| `--bind`, `--port` | Listen address (default `127.0.0.1:8090`) |
| `--model-name` | The name clients use in the `model` field |
| `--auth-tokens` | Comma-separated bearer tokens required on inference routes |
| `--kvstore-dir` | Persist the prefix cache to disk |
| `--kvstore-budget-mb` | Disk budget for that tier (default: twice the KV cache's bytes) |
| `--kvstore-min-tokens` | Shortest prefix the disk tier stores or reads (default 512) |
| `--spec-k` | Speculative decoding depth (0 = off) |
| `--vision` | Vision tower weights; enables image input |
| `--vision-allow-remote-images` | Allow fetching images from `http(s)` URLs |
| `--video-allow-ffmpeg` | Allow video input by spawning ffmpeg |
| `--audio-allow-ffmpeg` | Allow audio containers other than WAV by spawning ffmpeg |
| `--audio-max-seconds` | Longest audio clip a request may carry (default 7200) |
| `--audio-max-upload-mb` | Largest transcription upload, in MiB (default 512); other routes keep the 32 MiB body limit |
| `--asr-model`, `--asr-id` | Speech recognizer for `/v1/audio/transcriptions`, and the id it answers to |
| `--diarization-model` | Speaker diarizer, for `response_format=diarized_json` |
| `--asr-lookahead` | Encoder lookahead frames, one the checkpoint was trained for (default 13) |
| `--audio-max-streams` | Transcriptions advanced at once; the rest queue (default 4) |

Logging is controlled by `SHISO_LOG` (for example `SHISO_LOG=info,shiso_engine=debug`).

## API endpoints

| Endpoint | Compatible with |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/completions` | OpenAI Completions |
| `POST /v1/responses` | OpenAI Responses |
| `POST /v1/messages` | Anthropic Messages |
| `POST /v1/audio/transcriptions` | OpenAI Audio Transcriptions (also at `/audio/transcriptions`) |
| `GET /v1/models` | OpenAI Models |
| `GET /health`, `GET /metrics`, `GET /cache` | health, metrics, cache report |

## Contributing

`bash scripts/check.sh` runs the full test and lint suite; it has to pass
before a change lands. Read [`AGENTS.md`](AGENTS.md) for the architecture and
the rules the checks enforce.

## Acknowledgements

shiso builds on the work of other open-source projects. Some of the CUDA kernels
came from [antirez/ds4](https://github.com/antirez/ds4), which in turn vendors
kernels from [llama.cpp](https://github.com/ggml-org/llama.cpp) and
[exllamav3](https://github.com/turboderp-org/exllamav3). Reference outputs used
to check correctness come from llama.cpp, [MLX](https://github.com/ml-explore/mlx)
and Hugging Face [transformers](https://github.com/huggingface/transformers).
The prefix cache's eviction policy follows the Marconi paper on prefix caching
for hybrid models. [`PROVENANCE.md`](PROVENANCE.md) lists exactly what came
from where.

## License

shiso is released under the [MIT License](LICENSE). Vendored code keeps its
original license; see `PROVENANCE.md`.
