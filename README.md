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
- **Multimodal.** Image input, plus video input decoded by the OS (VideoToolbox, NVDEC), by pure-Rust decoders in a `decode-native` build, or by ffmpeg (opt-in).

## Supported models

| Family | Example checkpoints | Formats |
|---|---|---|
| Qwen 3.5 (dense hybrid) | Qwen3.5-4B | GGUF, MLX |
| Qwen 3.6 MoE | Qwen3.6-35B-A3B | GGUF, NVFP4, Splash |
| Gemma 4 | Gemma 4 E4B | GGUF, MLX |
| Qwen4Exp (MoE) | - | EXL3 |
| Nemotron-H (hybrid state-space MoE) | Nemotron 3.5 Lightning 30B-A3B | MLX 4-bit |
| EmbeddingGemma 2 (embeddings) | embeddinggemma-2 | GGUF (BF16, Q8_0) |

Vision towers are supported for Qwen 3.5, Qwen 3.6 MoE and Gemma 4.

Speech models (for the audio endpoints): the Nemotron 3.5 streaming ASR
(HF, MLX and GGUF exports), Qwen3-ASR-1.7B and its R2T2 fine-tune (HF, MLX
8-bit, and a text GGUF with its mmproj), and Nemotron-3-Diarization (HF, MLX,
and bf16 and q8_0 GGUF exports).

Gemma 4 also takes audio input: `input_audio` parts on Chat Completions and
Responses, up to 30 seconds a clip. The audio tower loads from the same
`--vision` path, either the MLX export directory or the mmproj GGUF. WAV is
decoded natively; other containers are decoded by a `decode-native` build, or by ffmpeg when `--ffmpeg-path` is set.

EmbeddingGemma 2 serves `/v1/embeddings` instead of the chat routes: start
the server with `--model` naming its GGUF and `--vision` naming its mmproj,
and text, images, video and audio - alone or interleaved in one input - land
in one 768-d space. `dimensions` cuts a vector to 512, 256 or 128 (the widths
the model was trained for), and `input_type` selects the checkpoint's task
prompt (`search_query`, `search_document`, or any prompt name it ships, such
as `Classification`). The tokenizer is read from `tokenizer.json` beside
`--config`, or from `--tokenizer`. Video needs `--video-allow-decode` for
containers other than GIF; media is sent inline (data URIs or base64).

```sh
shiso serve embeddinggemma-2 --video-allow-decode
curl localhost:8090/v1/embeddings -H 'Content-Type: application/json' -d '{
  "input": [{"content": [
    {"type": "text", "text": "A scenic boardwalk through a green meadow"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
  ]}],
  "input_type": "search_document", "dimensions": 256
}'
```

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

Qwen3-ASR and R2T2 load the same way from an HF or MLX export directory, or
from a text GGUF. The GGUF export splits the audio tower into an
`mmproj-*.gguf`: the one beside the text file whose quantisation matches is
taken, and `--asr-mmproj` names it when there are none or several to choose
from. These models transcribe in chunks of at most 35 s, cut at the quietest
point near every 30 s, or at the speaker turns when the job is diarized, so
each diarized segment is one turn. They give no word times of their own, so a
word's time is interpolated inside its chunk: times are chunk-accurate, not
word-accurate.

A clip of at least 5 minutes (`--asr-offline-min-secs`) sent without
`stream=true` is planned whole instead and decoded `--asr-batch` chunks at a
time (default 8), one weight read per step serving the whole batch, with the
same tokens as one chunk at a time. With a diarizer loaded, that plan follows
the speaker turns even for plain `json`, and a silence of 3 s or more where
no speaker is active is not transcribed at all: skipped audio yields no text,
so speech the diarizer misses there is lost. `--asr-skip-silence=false`
transcribes it. The plan batches its longest spans first, and a job's results
arrive in plan order as the batches covering them complete, so a streaming
job running beside an offline one waits at most one batch — a few seconds —
each turn. Streaming requests and shorter clips keep the chunk-by-chunk
path. An export with a decoder weight outside the audio kernels (an
engine-dispatched K-quant, for example) has no batched path and always goes
chunk by chunk, as the load log says.

```sh
./target/release/shiso serve \
    --asr-model ~/models/Qwen3-ASR-1.7B/gguf/Qwen3-ASR-1.7B-Q8_0.gguf --asr-id qwen3 \
    --diarization-model ~/models/Nemotron-3-Diarization/mlx
```

`response_format` is one of `json`, `text`, `verbose_json` (with
`timestamp_granularities[]` of `segment` and/or `word`), `srt`, `vtt` and
`diarized_json`; `stream=true` streams `transcript.text.delta` events (or, for
`diarized_json`, `transcript.text.segment` events) ending in
`transcript.text.done`.

A request to the Qwen3-ASR family can also carry `keywords[]` (repeated) and
`prompt`; they are written into its system turn as `prompt` followed by a
`Keywords: a, b, c` line, so the model is more likely to spell those words
the way you give them. A span whose text reads back like the keyword list -
most of its words are keywords, or it lists three or more of them in a row -
is decoded again without that context; the trade-off is that a span that
genuinely speaks such a list keeps the no-context wording instead. Keywords
and prompt together are capped by `--asr-max-context-tokens` (default 256, at
most the recognizer's own 512) once tokenized, and by 16 KiB of raw text
before that; an empty keyword is refused. Nemotron, or any recognizer without
context support, refuses either with a 400 whose `param` names the field
that was sent (`keywords`, or `prompt` when there were no keywords). With the
OpenAI SDK, `extra_body={"keywords": [...]}` sends `keywords[]`.

```sh
curl http://127.0.0.1:8090/v1/audio/transcriptions \
    -F model=qwen3 -F file=@meeting.m4a \
    -F 'keywords[]=Marigold' -F 'keywords[]=Lighthouse'
```

## Library use

`shiso-runtime` is the server's engine without the HTTP layer or the
command-line parser: scheduler, model loading, tokenizer, media, and typed
cores for chat, images and decisions. `Runtime` is one handle over them. It
takes and returns the `shiso-ir` types, so the caller never builds a wire
format. A model directory with a language model, an image model or both is
served; a call for a capability the model does not have returns a typed
error.

```rust,ignore
use shiso_ir::decision::{
    Choice, ChoiceValue, Input, Question, QuestionKind, ReadoutState, Request,
};
use shiso_runtime::{Runtime, RuntimeOptions, Source};

// Opens the model, pulling it first when it is not installed, and warms it.
let options = RuntimeOptions { pull_if_missing: true, ..RuntimeOptions::default() };
let runtime = Runtime::open(
    Source::Registry("smollm2-135m-decision"),
    options,
    &mut |done, total| {
        tracing::info!(done, total, "pulling");
        true // return false to cancel the pull
    },
)?;

let choice = |v: &str| Choice { value: ChoiceValue::Str(v.into()), description: None };
let response = runtime.decide_blocking(Request {
    input: Input::Text("cwd: /repo\ntask: unknown\ncmd: ls -la".into()),
    readout_state: ReadoutState::Input,
    questions: vec![Question {
        name: "decision".into(),
        instructions: "Is this command safe to run without asking?".into(),
        kind: QuestionKind::Choice(vec![choice("approve"), choice("ask"), choice("deny")]),
        probability_answer: false,
    }],
    temperature: None,
    preserve_thinking: None,
    token_ids: false,
})?;
tracing::info!(?response.answers[0].probabilities);

// A chat model answers a turn the same way, from any async executor.
let request = shiso_ir::Request::new("model", vec![shiso_ir::Message::user("hello")]);
let reply = runtime.generate(&request).await?;
```

A model with a decision readout or head answers every question from one
prompt; any other chat model scores each question's labels from its
next-token row, as `/v1/decisions` does.

`Source::Dir` opens a local installed variant or checkpoint and never touches
the network; `Source::Registry` reaches it only with `pull_if_missing`.
`generate_stream`, `images`, `edit_image` and their streaming forms are the
other entry points.

- **Warm-up on open.** `open` blocks while the weights load, then runs one
  throwaway pass (one decision, or one generated token, plus one small image
  when there is an image model) so the first real call does not pay for
  kernel compilation. A failed warm-up is `OpenError::WarmUp`.
- **Async and blocking.** The async methods run on the caller's executor,
  whichever it is: the runtime keeps a private one-worker tokio runtime for
  timers and the work itself runs on the engine thread. The `*_blocking`
  methods park the calling thread on a channel and work from any thread;
  on an async worker they block that worker, so prefer the async form there.
  The only refused caller is the runtime's own worker thread, which would wait
  for itself: it gets an error, not a hang or panic.
- **Sharing and dropping.** Methods take `&self`; wrap the `Runtime` in an
  `Arc` to share it. Dropping it joins the engine thread after the step in
  flight, so do not drop it on an async worker you cannot stall.

Build with `--features metal` or `--features cuda`. The runtime has no HTTP
server or command-line parser in its dependencies unless you turn on its
`clap` feature.

## Configuration

Run `shiso serve --help` for every option. Each flag also reads a `SHISO_*`
environment variable. The ones you are most likely to want:

| Flag | What it does |
|---|---|
| `--ctx` | Tokens one session may hold, prompt included (default 4096) |
| `--max-num-seqs` | Sequences resident at once (default 4) |
| `--kv-cache-tokens` | Tokens the shared KV cache holds across all sessions (default 131072) |
| `--gpu-memory-utilization` | Size the KV cache to fill this share of device memory instead, in (0, 1] |
| `--kv-cache-dtype` | `f16` (default) or `fp8`: store attention keys and values as FP8 E4M3, about 43% fewer cache bytes per token (sparse models on CUDA only) |
| `--prefill-activations` | `a16` (default) or `lowp`: quantize prompt activations to NVFP4 (routed experts) and MXFP8 (dense projections) and run those products on the block-scaled tensor-core instructions; faster prefill at a measured accuracy cost; decode and verify unchanged (sparse models on a CUDA build for `sm_121a` only) |
| `--bind`, `--port` | Listen address (default `127.0.0.1:8090`) |
| `--model-name` | The name clients use in the `model` field |
| `--auth-tokens` | Comma-separated bearer tokens required on inference routes |
| `--kvstore-dir` | Persist the prefix cache to disk |
| `--kvstore-budget-mb` | Disk budget for that tier (default: twice the KV cache's bytes) |
| `--kvstore-min-tokens` | Shortest prefix the disk tier stores or reads (default 512) |
| `--spec-k` | Speculative decoding depth (0 = off) |
| `--vision` | Vision tower weights; enables image input |
| `--vision-allow-remote-images` | Allow fetching images from `http(s)` URLs |
| `--video-allow-decode` | Allow video input other than GIF: the OS decoder, then the native one, then ffmpeg when `--ffmpeg-path` is set |
| `--ffmpeg-path` | The ffmpeg binary, for video and audio nothing else decodes. No default: without it ffmpeg is never run |
| `--audio-max-seconds` | Longest audio clip a request may carry (default 7200) |
| `--audio-max-upload-mb` | Largest transcription upload, in MiB (default 512); other routes keep the 32 MiB body limit |
| `--asr-model`, `--asr-id` | Speech recognizer for `/v1/audio/transcriptions`, and the id it answers to |
| `--asr-mmproj` | The audio-tower GGUF of a recognizer whose GGUF export splits it out, when it cannot be found beside the text file |
| `--diarization-model` | Speaker diarizer, for `response_format=diarized_json` |
| `--asr-lookahead` | Encoder lookahead frames, one the checkpoint was trained for (default 13; ignored by a recognizer without lookahead) |
| `--audio-max-streams` | Transcriptions advanced at once; the rest queue (default 4) |
| `--asr-batch` | Chunks an offline transcription decodes at once (default 8; 0 is refused) |
| `--asr-skip-silence` | Leave out silences of 3 s or more that the diarizer finds in an offline transcription; they yield no text (default true; `--asr-skip-silence=false` turns it off) |
| `--asr-offline-min-secs` | Shortest non-streaming clip transcribed offline, in seconds (default 300; 0 sends every one) |
| `--asr-max-context-tokens` | Most tokens a transcription's `keywords` and `prompt` may take once tokenized (default 256; at most the recognizer's own cap, 512 for Qwen3-ASR) |
| `--image-model` | Image generation checkpoint (file or directory); serves the image generation routes |
| `--image-model-id` | The name the image generator answers to on the wire (default: the name the loaded checkpoint reports) |
| `--image-assets-dir` | Directory holding shared image assets (`text_encoder`, `vae` (FLUX.3 Klein: `ae.safetensors`), `tokenizer`; for Qwen-Image-2.1 also `model_index.json`, `scheduler/` and `transformer/config.json`, which carry the schedule) when they are not collocated with `--image-model` |
| `--image-text-encoder` | Text encoder weights path when it is split from `--image-model`; overrides where the locator would look |
| `--image-vae` | VAE weights path when it is split from `--image-model`; overrides where the locator would look |
| `--image-tokenizer` | Tokenizer path when it is split from `--image-model`; overrides where the locator would look |
| `--image-max-upload-mb` | Largest image edit request body in MiB (default 64); only the edit routes take it |
| `--image-max-queue` | Image requests that may wait in the queue beyond the one generating (default 16); the next is refused with a 503 a retry can fix |
| `--image-default-steps` | Denoising steps a request that names none gets (default: the loaded model's own: 4 for FLUX.2-klein, 16 for FLUX.3 Klein; for Qwen-Image-2.1 the length of `model_index.json`'s `sample_sigmas`, fixed, when it ships them (8 for Turbo), else 40) |
| `--image-default-size` | The image size a request that names none gets, as `WxH` (default: the loaded model's own: 1024x1024 on a 16-pixel grid for FLUX.2-klein and FLUX.3 Klein (FLUX.3 Klein sides run 64 to 4096), 512x512 on a 32-pixel grid for Qwen-Image-2.1, where sizes must be exact multiples of 32); a size off the model's grid is refused at boot |
| `--image-residency` | Which image pipeline components stay loaded across jobs: `auto` (default) keeps all of them when they fit the device's working set beside a job and otherwise swaps, `resident` refuses a job that does not fit, `swap` lets the text encoder and the transformer take turns (FLUX.3 Klein and Qwen-Image-2.1; FLUX.2-klein refuses it) |
| `--image-reserve-gb` | GiB of the device working set the image pipeline leaves to the rest of the host when it plans what stays loaded (default 1.5) |
| `--image-draft-steps` | Leading denoising steps of every image job run at half the requested resolution, then handed over to the full size (Qwen-Image-2.1 only; default 0). A half-size step costs about a quarter of a full one; at 1024x1024 on an M4, 4 of a 6-step distilled checkpoint's steps take a job from 150 s to 77 s at a small cost in fine detail |
| `--image-final-reuse` | `START:END`: transformer blocks the last step skips, replaying the residual the step before recorded over them (Qwen-Image-2.1 only; the last two steps must run at full size). Faster, visibly softer |
| `--embedding-max-queue` | Embedding requests that may wait in the queue beyond the one running (default 64), when `--model` is an embedding checkpoint |

Logging is controlled by `SHISO_LOG` (for example `SHISO_LOG=info,shiso_engine=debug`).

## API endpoints

| Endpoint | Compatible with |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/completions` | OpenAI Completions |
| `POST /v1/responses` | OpenAI Responses |
| `POST /v1/messages` | Anthropic Messages |
| `POST /v1/audio/transcriptions` | OpenAI Audio Transcriptions (also at `/audio/transcriptions`) |
| `POST /v1/images/generations` | OpenAI Images (also at `/v1/images` and `/images`; serves FLUX.2-klein, the distilled FLUX.3 Klein and Qwen-Image-2.1 (Turbo, base, and fine-tunes in the same format; the schedule is the assets' `model_index.json` `sample_sigmas`, shifted by `scheduler/scheduler_config.json`); `b64_json` only) |
| `POST /v1/images/edits` | OpenAI Image edits (also at `/images/edits`; FLUX.3 Klein only; multipart `image`/`image[]` (up to 10) or JSON `images` with `data:` URLs, plus `prompt`, `size`, `seed`, `steps`, `output_format`, `output_compression`; `n` must be 1, no `mask`; `b64_json` only; body limit `--image-max-upload-mb`) |
| `POST /v1/embeddings` | OpenAI and OpenRouter Embeddings (also at `/embeddings`; text, token-id and multimodal `content` inputs, `dimensions`, `encoding_format` `float` or `base64`, `input_type`) |
| `GET /v1/models` | OpenAI Models |
| `GET /health`, `GET /metrics`, `GET /cache` | health, metrics, cache report |

## Contributing

`bash scripts/check.sh` runs the test and lint suite over the crates a change
touches and everything that depends on them (`--all` for the whole
workspace); it has to pass before a change lands. Read [`AGENTS.md`](AGENTS.md) for the architecture and
the rules the checks enforce.

## Acknowledgements

shiso builds on the work of other open-source projects. Some of the CUDA kernels
came from [antirez/ds4](https://github.com/antirez/ds4), which in turn vendors
kernels from [llama.cpp](https://github.com/ggml-org/llama.cpp) and
[exllamav3](https://github.com/turboderp-org/exllamav3). Reference outputs used
to check correctness come from llama.cpp, [MLX](https://github.com/ml-explore/mlx)
and Hugging Face [transformers](https://github.com/huggingface/transformers);
the speech recognizers' from their reference packages, `qwen-asr` among them.
The prefix cache's eviction policy follows the Marconi paper on prefix caching
for hybrid models. [`PROVENANCE.md`](PROVENANCE.md) lists exactly what came
from where.

## License

shiso is released under the [MIT License](LICENSE). Vendored code keeps its
original license; see `PROVENANCE.md`.
