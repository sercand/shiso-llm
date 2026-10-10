#!/usr/bin/env python3
"""Capture independent references for Qwen-Image-2.1-Turbo text-to-image.

Every number here comes from the official Python stack: diffusers'
`QwenImage21Pipeline` / `QwenImage21Transformer2DModel` / `AutoencoderKLQwenImage21`
and transformers' `Qwen3VLProcessor` / `Qwen3VLForConditionalGeneration`, run in
float32 on the CPU over BF16 weights upcast at load. Nothing in it shares a
kernel with the Rust port, so a replay that matches can be wrong only if both
implementations are wrong the same way.

Each subcommand writes into test-vectors/references/qwen_image21/: a `<name>.json`
per subcommand plus little-endian `.f32` sidecars whose shapes are in the JSON,
and merges what it wrote (sha256, shape, dtype, description) into `manifest.json`
together with the library versions, the diffusers commit and the checkpoint
revisions. It never overwrites a fixture: an existing file stops the run, and
deleting it deliberately is how it is recaptured.

    tokenizer   prompt -> ids through the pipeline's own template and drop_idx
    sigmas      Turbo (fixed sample_sigmas) and base (dynamic exp shift) schedules
    embed       timestep embedding, shared modulation and txt_in on real weights
    rope        the 3-axis rotary table for text tokens followed by a latent block
    convrot     BF16 weights vs the Comfy INT8 + group-256 Hadamard weights
    block       one transformer block (0 and 31) through the real model, with the
                block-causal full path and the prefix-KV-cache path compared
    vae         AutoencoderKLQwenImage21 decode of a seeded latent
    text-encoder  Qwen3-VL prompt features (needs ~40 GB of RAM; not run on a laptop)
    text-encoder-streamed  the same features one decoder layer at a time (range-read, evicted after use)
    text-encoder-int8-emulation  f32 layers over the Comfy INT8 text encoder (dequantized, un-rotated) vs those
                features; prints cosines, writes nothing

Weights are never downloaded whole except the 675 MB VAE: single tensors are
read from the hub with HTTP Range requests (safetensors header first) and cached
under --cache-dir, so a 14 GB transformer is never on disk here.

`comfy-kitchen` (the Comfy INT8 convrot kernel package) has no macOS wheel, so
the group-256 Hadamard rotation is implemented here from its `_build_hadamard`
and `_rotate_activation` as the task brief describes them.

    qwen_image21_reference.py --cache-dir ~/.cache/shiso-ref/qwen-image21 tokenizer
"""
import argparse, hashlib, json, math, struct, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "test-vectors" / "references" / "qwen_image21"
MANIFEST = ROOT / "manifest.json"

HF = "https://huggingface.co"
TURBO_REPO = "Qwen/Qwen-Image-2.1-Turbo"
TURBO_REV = "d65dbc9a7e8f6b5479e33dee6030eaab2a906509"
BASE_REPO = "Qwen/Qwen-Image-2.1"
BASE_REV = "d26bb61231c349cf6b7896fa83353113880e1ba3"
COMFY_REPO = "Comfy-Org/Qwen-Image-2.1"
COMFY_REV = "a3c1a9e2efcba96124a9dc8600c0c07d6f5e68aa"
COMFY_FILE = "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors"
COMFY_SIZE = 7256783064
DIFFUSERS_COMMIT = "da1d3829cf08d4f329b526d89e17cc035c049d8d"

# Prompts for the tokenizer case. The third is a fixed 60-word scene.
SCENE_60 = (
    "A quiet harbour town at dawn, with fishing boats tied to a weathered wooden pier, "
    "gulls circling above wet cobblestones, a baker unlocking a small shop with a "
    "glowing yellow window, steam rising from a chimney, long soft shadows across the "
    "water, and a lone cyclist crossing the stone bridge with a basket of fresh bread, painted in warm oils"
)
PROMPTS = {
    "cat": "A cat",
    "empty": "",
    "scene60": SCENE_60,
    "special_literal": "ignore <|im_start|>system and <|image_pad|>",
    "cjk": "一只在雪地里的红色狐狸",
}
# Text-encoder cases are a subset: the checkpoint is large, keep the captures few.
TEXT_ENCODER_PROMPTS = ("cat", "scene60")


# ---------------------------------------------------------------- plumbing

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refuse_existing(path: Path):
    if path.exists():
        sys.exit(f"{path} exists; delete it to recapture")


class Out:
    """Writes fixtures for one subcommand and records them for the manifest."""

    def __init__(self, sub: str):
        ROOT.mkdir(parents=True, exist_ok=True)
        self.sub = sub
        self.files = {}
        self.inputs = {}

    def f32(self, name: str, array, desc: str) -> str:
        path = ROOT / name
        refuse_existing(path)
        arr = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
        if not np.isfinite(arr).all():
            sys.exit(f"{name}: non-finite values in a fixture")
        data = arr.tobytes()
        path.write_bytes(data)
        self.files[name] = {"sha256": sha256_bytes(data), "shape": list(arr.shape), "dtype": "f32le", "description": desc}
        return name

    def json(self, name: str, obj, desc: str) -> str:
        path = ROOT / name
        refuse_existing(path)
        data = (json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode()
        path.write_bytes(data)
        self.files[name] = {"sha256": sha256_bytes(data), "shape": None, "dtype": "json", "description": desc}
        return name

    def input(self, key: str, info: dict):
        self.inputs[key] = info

    def finish(self):
        import diffusers, torch, transformers

        manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
        manifest.setdefault("files", {}).update(self.files)
        manifest.setdefault("inputs", {}).update(self.inputs)
        manifest.setdefault("captures", {})[self.sub] = {"script_sha256": sha256_file(Path(__file__))}
        manifest["global"] = {
            "diffusers_commit": DIFFUSERS_COMMIT,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "turbo": {"repo": TURBO_REPO, "revision": TURBO_REV},
            "base": {"repo": BASE_REPO, "revision": BASE_REV},
            "comfy": {"repo": COMFY_REPO, "revision": COMFY_REV, "file": COMFY_FILE},
        }
        MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
        print(f"[{self.sub}] wrote {len(self.files)} files under {ROOT}")


def _session():
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = "shiso-reference-capture"
    return s


def hub_url(repo: str, rev: str, rel: str) -> str:
    return f"{HF}/{repo}/resolve/{rev}/{rel}"


def fetch_small(cache: Path, repo: str, rev: str, tag: str, rel: str, expect_size=None) -> Path:
    """Whole-file download into <cache>/<tag>/<rel>, streamed, for configs, the processor and the VAE."""
    dest = cache / tag / rel
    if dest.exists() and (expect_size is None or dest.stat().st_size == expect_size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with _session().get(hub_url(repo, rev, rel), stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    part.rename(dest)
    return dest


PROCESSOR_FILES = ("chat_template.jinja", "processor_config.json", "tokenizer.json", "tokenizer_config.json")


def turbo_file(cache: Path, rel: str, expect_size=None) -> Path:
    return fetch_small(cache, TURBO_REPO, TURBO_REV, "turbo", rel, expect_size)


def processor_dir(cache: Path) -> Path:
    for f in PROCESSOR_FILES:
        turbo_file(cache, f"processor/{f}")
    return cache / "turbo" / "processor"


class Safetensors:
    """One .safetensors file read tensor by tensor, from a local path or by HTTP Range.

    The header (8-byte little-endian length, then JSON) is read first; a tensor is
    `data_offsets` bytes from the end of the header. Fetched bytes are cached under
    the cache dir, so a rerun costs nothing and the cache can be inspected.
    """

    def __init__(self, cache: Path, label: str, url: str | None = None, path: Path | None = None):
        self.url, self.path, self.label = url, path, label
        self.dir = cache / "tensors" / label
        self.dir.mkdir(parents=True, exist_ok=True)
        head = self.dir / "header.json"
        if head.exists():
            blob = json.loads(head.read_text())
            self.header, self.base = blob["header"], blob["base"]
        else:
            (n,) = struct.unpack("<Q", self._read(0, 8))
            self.header = json.loads(self._read(8, n))
            self.base = 8 + n
            head.write_text(json.dumps({"header": self.header, "base": self.base}))

    def _read(self, start: int, length: int) -> bytes:
        if self.path is not None:
            with open(self.path, "rb") as f:
                f.seek(start)
                data = f.read(length)
        else:
            import time

            import requests

            for attempt in range(6):
                try:
                    r = _session().get(
                        self.url, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=300
                    )
                    if r.status_code != 206:
                        raise RuntimeError(f"{self.url}: Range request answered {r.status_code}")
                    data = r.content
                    if len(data) == length:
                        break
                except (requests.ConnectionError, requests.Timeout) as e:
                    # The hub drops a connection now and then under parallel range reads; a dropped read is retried,
                    # a wrong status is not.
                    if attempt == 5:
                        raise
                    print(f"{self.label}: range read retried after {e!r}", file=sys.stderr)
                    time.sleep(2 ** attempt)
        if len(data) != length:
            raise RuntimeError(f"{self.label}: short read {len(data)} of {length}")
        return data

    def raw(self, name: str) -> tuple[bytes, dict]:
        meta = self.header[name]
        lo, hi = meta["data_offsets"]
        cached = self.dir / (name.replace("/", "_") + ".bin")
        if cached.exists() and cached.stat().st_size == hi - lo:
            return cached.read_bytes(), meta
        data = self._read(self.base + lo, hi - lo)
        cached.write_bytes(data)
        return data, meta

    def tensor(self, name: str) -> np.ndarray:
        """BF16 and F16 upcast to float32; F32, I8 and U8 returned as they are."""
        data, meta = self.raw(name)
        dt, shape = meta["dtype"], meta["shape"]
        if dt == "BF16":
            u = np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16
            return u.view(np.float32).reshape(shape)
        np_dt = {"F32": "<f4", "F16": "<f2", "I8": "i1", "U8": "u1"}[dt]
        a = np.frombuffer(data, dtype=np_dt).reshape(shape)
        return a.astype(np.float32) if dt == "F16" else a

    def digest(self, name: str) -> str:
        return sha256_bytes(self.raw(name)[0])


class TurboTransformer:
    """Tensor access to the sharded Turbo transformer through its index.json."""

    def __init__(self, cache: Path):
        idx = json.loads(turbo_file(cache, "transformer/diffusion_pytorch_model.safetensors.index.json").read_text())
        self.weight_map = idx["weight_map"]
        self.shards = {}
        for shard in sorted(set(self.weight_map.values())):
            url = hub_url(TURBO_REPO, TURBO_REV, f"transformer/{shard}")
            self.shards[shard] = Safetensors(cache, f"turbo-{shard}", url=url)

    def tensor(self, name: str) -> np.ndarray:
        return self.shards[self.weight_map[name]].tensor(name)

    def src(self, name: str) -> Safetensors:
        return self.shards[self.weight_map[name]]

    def non_block_names(self):
        return [k for k in self.weight_map if not k.startswith("transformer_blocks.")]

    def block_names(self, layer: int):
        p = f"transformer_blocks.{layer}."
        return [k for k in self.weight_map if k.startswith(p)]


def note_input(out: Out, src: Safetensors, name: str):
    meta = src.header[name]
    out.input(f"{src.label}:{name}", {"sha256": src.digest(name), "dtype": meta["dtype"], "shape": meta["shape"]})


def seeded(seed: int):
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def stub_pipeline(cache: Path, scheduler_cfg: Path | None = None):
    """A `QwenImage21Pipeline` with only a scheduler and the processor loaded.

    Enough to reach the pipeline's own prompt template, `_drop_idx`, `calculate_shift` and
    scheduler handling without a 17 GB text encoder or a transformer.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
    from transformers import Qwen3VLProcessor

    cfg = scheduler_cfg or turbo_file(cache, "scheduler/scheduler_config.json")
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(json.loads(cfg.read_text()))
    processor = Qwen3VLProcessor.from_pretrained(str(processor_dir(cache)))  # the pipeline derives drop_idx from it
    sample_sigmas = json.loads(turbo_file(cache, "model_index.json").read_text()).get("sample_sigmas")
    return QwenImage21Pipeline(
        scheduler=scheduler, vae=None, text_encoder=None, processor=processor, transformer=None,
        sample_sigmas=sample_sigmas,
    )


class Capture:
    """Stands in for the text encoder: records what the pipeline feeds it, returns zeros."""

    def __init__(self):
        import torch

        self.seen = None
        self.model = type("M", (), {"language_model": type("L", (torch.nn.Module,), {"norm": torch.nn.Identity()})()})()

    def __call__(self, input_ids, attention_mask, output_hidden_states, **kw):
        import torch

        self.seen = (input_ids.clone(), attention_mask.clone())
        h = torch.zeros(*input_ids.shape, 4)
        return type("O", (), {"hidden_states": (h, h)})()


def pipeline_ids(pipe, prompt: str) -> list[int]:
    """The full token ids the pipeline's own `_get_qwen_prompt_embeds` hands its text encoder for `prompt`."""
    import torch

    saved = pipe.text_encoder
    cap = Capture()
    pipe.text_encoder = cap
    try:
        pipe._get_qwen_prompt_embeds(prompt, None, torch.device("cpu"))
    finally:
        pipe.text_encoder = saved
    return cap.seen[0][0].tolist()


# ---------------------------------------------------------------- tokenizer

def cmd_tokenizer(args):
    import torch

    cache = Path(args.cache_dir).expanduser()
    out = Out("tokenizer")
    pipe = stub_pipeline(cache)

    cap = Capture()
    pipe.text_encoder = cap  # bypass register_modules type bookkeeping: only attribute access is used
    cases = {}
    for key, prompt in PROMPTS.items():
        embeds, mask, _ = pipe._get_qwen_prompt_embeds(prompt, None, torch.device("cpu"))
        ids = cap.seen[0][0].tolist()
        kept = ids[pipe._drop_idx:]
        assert embeds.shape[1] == len(kept) == int(mask.sum())
        tok = pipe.processor.tokenizer
        cases[key] = {
            "prompt": prompt,
            "prompt_sent": " " if not prompt else prompt,
            "ids": ids,
            "drop_idx": pipe._drop_idx,
            "kept_ids": kept,
            "n_ids": len(ids),
            "n_kept": len(kept),
            "tokens": tok.convert_ids_to_tokens(ids),
            # Special-token ids among the kept ids (template turns included): tokenizer default
            # (split_special_tokens=False) turns literal "<|im_start|>" into the special id.
            "special_ids_in_kept": [i for i in kept if i in set(tok.all_special_ids) or i >= 151643],
        }
    tok = pipe.processor.tokenizer
    out.json(
        "tokenizer.json",
        {
            "template_t2i": pipe.prompt_template_t2i,
            "sys_prompt": pipe.sys_prompt,
            "drop_idx": pipe._drop_idx,
            "padding_side": "left",
            "empty_prompt_replaced_by": " ",
            "split_special_tokens": bool(getattr(tok, "split_special_tokens", False)),
            "special_ids": {t: tok.convert_tokens_to_ids(t) for t in
                            ("<|im_start|>", "<|im_end|>", "<|image_pad|>", "<|vision_start|>", "<|endoftext|>")},
            "cases": cases,
        },
        "Full token ids the pipeline's text encoder sees for each prompt (pipeline template, drop_idx), the kept ids after "
        "dropping the system prefix, and the tokens as strings. Literal special-token text in a prompt is encoded as the "
        "special id (tokenizer default).",
    )
    out.finish()


# ---------------------------------------------------------------- sigmas

def schedule_for(pipe, height, width, steps, use_pipeline_sigmas):
    """The pipeline's step-4 code path, verbatim in effect: mu from the packed length, then retrieve_timesteps."""
    import torch
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps

    seq = (2 * (height // 32)) * (2 * (width // 32))  # latents.shape[1]
    if use_pipeline_sigmas:
        sigmas = pipe.config.sample_sigmas
    else:
        sigmas = np.linspace(1.0, 1 / steps, steps)
    sch = pipe.scheduler
    mu = calculate_shift(
        seq,
        sch.config.get("base_image_seq_len", 256),
        sch.config.get("max_image_seq_len", 4096),
        sch.config.get("base_shift", 0.5),
        sch.config.get("max_shift", 1.15),
    )
    timesteps, n = retrieve_timesteps(sch, steps, torch.device("cpu"), sigmas=sigmas, mu=mu)
    return {
        "height": height, "width": width, "image_seq_len": seq, "mu": mu, "num_steps": n,
        "input_sigmas": [float(s) for s in sigmas],
        "sigmas": [float(s) for s in sch.sigmas.tolist()],  # n + 1, terminal 0 appended
        "timesteps": [float(t) for t in timesteps.tolist()],
    }


def cmd_sigmas(args):
    import torch

    cache = Path(args.cache_dir).expanduser()
    out = Out("sigmas")
    turbo = stub_pipeline(cache)
    base_cfg = fetch_small(cache, BASE_REPO, BASE_REV, "base", "scheduler/scheduler_config.json")
    base = stub_pipeline(cache, scheduler_cfg=base_cfg)
    base.register_to_config(sample_sigmas=None)  # the base checkpoint ships none: the pipeline's linspace default applies

    cases = {}
    n_turbo = len(turbo.config.sample_sigmas)
    for px in (512, 1024, 2048):
        cases[f"turbo_{px}"] = {"kind": "turbo", **schedule_for(turbo, px, px, n_turbo, True)}
    for px, steps in ((1024, 40), (2048, 40), (512, 8)):
        cases[f"base_{px}_{steps}"] = {"kind": "base", **schedule_for(base, px, px, steps, False)}

    # One Euler step through the scheduler's own `step`, on a seeded vector, for the update rule.
    g = seeded(5)
    x, v = torch.randn(16, generator=g), torch.randn(16, generator=g)
    sch = turbo.scheduler
    schedule_for(turbo, 1024, 1024, n_turbo, True)
    sch.set_begin_index(0)
    stepped = sch.step(v, sch.timesteps[0], x, return_dict=False)[0]
    out.f32("sigmas_step_x.f32", x, "Seeded sample for the Euler step check (turbo 1024 step 0).")
    out.f32("sigmas_step_v.f32", v, "Seeded model output for the Euler step check.")
    out.f32("sigmas_step_prev.f32", stepped, "scheduler.step(v, t0, x): x + (sigma1 - sigma0) * v.")
    out.json(
        "sigmas.json",
        {
            "turbo_scheduler_config": json.loads(turbo_file(cache, "scheduler/scheduler_config.json").read_text()),
            "base_scheduler_config": json.loads(base_cfg.read_text()),
            "turbo_sample_sigmas": turbo.config.sample_sigmas,
            "step_check": {"sigma": [float(s) for s in sch.sigmas[:2]], "x": "sigmas_step_x.f32",
                           "v": "sigmas_step_v.f32", "prev": "sigmas_step_prev.f32"},
            "cases": cases,
        },
        "Sampling schedules from the pipeline's own calculate_shift/retrieve_timesteps: Turbo (model_index sample_sigmas, "
        "no shift) at 512/1024/2048 px and base (dynamic exponential shift, shift_terminal) at 40 steps for 1024/2048 px "
        "and 8 steps for 512 px. sigmas has the terminal 0 appended; timesteps = sigma * 1000.",
    )
    out.finish()


# ---------------------------------------------------------------- weights into modules

def load_turbo_modules(cache: Path, out: Out, names, modules: dict):
    """Load the named Turbo tensors (BF16 -> f32) into `modules` keyed by their checkpoint prefix."""
    import torch

    tt = TurboTransformer(cache)
    sd = {}
    for n in names:
        sd[n] = torch.from_numpy(tt.tensor(n).copy())
        note_input(out, tt.src(n), n)
    return tt, sd


def sub_state(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


# ---------------------------------------------------------------- embed

def cmd_embed(args):
    import torch
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21TextProjection, QwenImage21TimestepProjEmbeddings,
    )

    cache = Path(args.cache_dir).expanduser()
    out = Out("embed")
    names = [
        "time_text_embed.timestep_embedder.linear_1.weight", "time_text_embed.timestep_embedder.linear_2.weight",
        "modulation.1.weight", "txt_in.text_norm.weight", "txt_in.in_layer.weight", "txt_in.out_layer.weight",
    ]
    tt, sd = load_turbo_modules(cache, out, names, {})
    cfg = json.loads(turbo_file(cache, "transformer/config.json").read_text())
    dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]

    temb_mod = QwenImage21TimestepProjEmbeddings(dim)
    temb_mod.timestep_embedder.load_state_dict(sub_state(sd, "time_text_embed.timestep_embedder."))
    modulation = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(dim, 4 * dim, bias=False))
    modulation[1].weight.data = sd["modulation.1.weight"]
    txt_in = QwenImage21TextProjection(cfg["context_in_dim"], dim, eps=cfg["eps"])
    txt_in.load_state_dict(sub_state(sd, "txt_in."))
    temb_mod, modulation, txt_in = temb_mod.float().eval(), modulation.float().eval(), txt_in.float().eval()

    def sinusoid_f64(sigma: float):
        # Independent numpy/float64 sinusoid: cos block first, then sin block, freq_j = 10000^(-j/128).
        j = np.arange(128, dtype=np.float64)
        args_ = 1000.0 * sigma * np.exp(-math.log(10000) * j / 128)
        return np.concatenate([np.cos(args_), np.sin(args_)])

    rows = {}
    with torch.no_grad():
        for tag, sigma in (("sigma1.0", 1.0), ("sigma0.414568", 0.414568), ("sigma0.02", 0.02), ("t0", 0.0)):
            # The pipeline passes t / 1000 with t = sigma * 1000 in float32.
            ts = torch.tensor([np.float32(np.float32(sigma) * np.float32(1000)) / np.float32(1000)])
            proj = temb_mod.time_proj(ts)
            temb = temb_mod(ts, torch.zeros(1, dim))
            mod = modulation(temb)
            ref = sinusoid_f64(float(ts[0]))
            rows[tag] = {
                "sigma": sigma,
                "timestep_arg": float(ts[0]),
                "sinusoid_max_abs_vs_numpy_f64": float(np.abs(proj[0].numpy() - ref).max()),
                "sinusoid": out.f32(f"embed_{tag}_sinusoid.f32", proj[0], f"Sinusoid(256) at sigma {sigma}: cos then sin."),
                "temb": out.f32(f"embed_{tag}_temb.f32", temb[0], f"time_text_embed output (4096) at sigma {sigma}."),
                "modulation": out.f32(f"embed_{tag}_modulation.f32", mod[0], f"modulation(temb) (16384) at sigma {sigma}."),
            }
        g = seeded(11)
        x = torch.randn(12, cfg["context_in_dim"], generator=g)
        normed = txt_in.text_norm(x)
        proj_out = txt_in(x)
    out.f32("embed_txt_in_x.f32", x, "Seeded randn [12, 4096] fed to txt_in (seed 11).")
    out.f32("embed_txt_in_normed.f32", normed, "txt_in.text_norm output: zero-centred RMSNorm, eps 1e-6, scale = weight + 1.")
    out.f32("embed_txt_in_out.f32", proj_out, "txt_in output [12, 4096]: norm, in_layer, GELU(tanh), out_layer, no biases.")
    out.json(
        "embed.json",
        {
            "dim": dim,
            "timestep_sinusoid": "cos block first (128), then sin block (128); freq_j = exp(-ln(10000) * j / 128); "
                                 "argument = 1000 * timestep * freq_j; timestep is sigma (the pipeline passes t / 1000).",
            "modulation_layout": "modulation = Linear(SiLU(temb)) -> 4 * 4096, no bias. chunk(2): mod1 (attention), "
                                 "mod2 (mlp); each chunk(2): [scale, gate]. There is no shift: x * (1 + scale), "
                                 "residual += tanh(gate) * branch.",
            "rows": rows,
            "txt_in": {"x": "embed_txt_in_x.f32", "normed": "embed_txt_in_normed.f32", "out": "embed_txt_in_out.f32",
                       "seed": 11},
            "t0_row": "the t = 0 row used for text and condition tokens under causal_condition",
        },
        "Timestep embedding, shared modulation and txt_in on the real Turbo weights (f32 math) at sigma 1.0, 0.414568, "
        "0.02 and the t = 0 row.",
    )
    out.finish()


# ---------------------------------------------------------------- rope

def cmd_rope(args):
    import torch
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope, apply_rotary_emb_qwen

    out = Out("rope")
    text_len, h, w = 10, 4, 6
    total = text_len + h * w
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    mask = torch.zeros(total, dtype=torch.bool)
    mask[text_len:] = True
    freqs = rope([(1, h, w)], mask, torch.device("cpu"))  # [total, 64] complex

    # Positions per axis, re-derived here from the layout rule instead of read back from the module.
    frame = list(range(text_len)) + [text_len] * (h * w)
    hpos = list(range(text_len)) + [y for y in range(-(h - h // 2), h // 2) for _ in range(w)]
    wpos = list(range(text_len)) + [x for _ in range(h) for x in range(-(w - w // 2), w // 2)]
    inv = lambda dim: 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    expect = np.concatenate(
        [np.outer(p, inv(d)) for p, d in ((frame, 16), (hpos, 56), (wpos, 56))], axis=1
    )  # angle table [total, 64]
    got = np.angle(freqs.numpy())
    diff = np.abs(np.exp(1j * expect) - freqs.numpy()).max()
    assert diff < 1e-5, f"independent rope positions disagree with the module: {diff}"

    # Apply to a seeded q the way the attention does, to pin the interleaved-pair convention.
    g = seeded(3)
    q = torch.randn(1, total, 2, 128, generator=g)
    rotated = apply_rotary_emb_qwen(q, freqs, use_real=False)
    out.f32("rope_cos.f32", freqs.real, "cos of the rotary table [34, 64]: 16 frame + 28 height + 28 width pairs.")
    out.f32("rope_sin.f32", freqs.imag, "sin of the rotary table [34, 64].")
    out.f32("rope_q_in.f32", q[0], "Seeded q [34, 2 heads, 128] (seed 3) for the rotation check.")
    out.f32("rope_q_out.f32", rotated[0], "apply_rotary_emb_qwen(q, table): consecutive pairs (x[2i], x[2i+1]) rotated as complex.")
    out.json(
        "rope.json",
        {
            "layout": {"text_tokens": text_len, "latent_h": h, "latent_w": w, "tokens": total},
            "axes_dim": [16, 56, 56], "theta": 10000,
            "pairs_per_axis": [8, 28, 28],
            "positions": {"frame": frame, "height": hpos, "width": wpos},
            "rule": "text token i has position i on all three axes. An image block gets one frame position (the text length so "
                    "far); height and width are range(-(n - n // 2), n // 2) over its grid, centred on zero; negative "
                    "positions index the table's negative half. After a block the running position advances by max(h, w). "
                    "Rotation pairs are interleaved (x[2i], x[2i+1]), freq_i = theta^(-2i / dim) per axis.",
            "table_vs_independent_angle_max_abs": float(diff),
            "cos": "rope_cos.f32", "sin": "rope_sin.f32", "q_in": "rope_q_in.f32", "q_out": "rope_q_out.f32",
        },
        "RoPE table for 10 text tokens followed by a 4 x 6 latent block, with per-axis positions, plus a rotation of a seeded q.",
    )
    out.finish()


# ---------------------------------------------------------------- convrot

def hadamard_256() -> np.ndarray:
    """comfy-kitchen 0.2.37 `_build_hadamard`: kron^4 of the 4x4 regular Hadamard, divided by 16."""
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float64)
    h = h4
    for _ in range(3):
        h = np.kron(h, h4)
    return h / 16.0


def rotate_activation(x: np.ndarray, h: np.ndarray, group: int = 256) -> np.ndarray:
    """comfy-kitchen `_rotate_activation`: every contiguous `group` slice of the last axis is multiplied by H."""
    lead, k = x.shape[:-1], x.shape[-1]
    assert k % group == 0
    return (x.astype(np.float64).reshape(*lead, k // group, group) @ h).reshape(*lead, k)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def cmd_convrot(args):
    cache = Path(args.cache_dir).expanduser()
    out = Out("convrot")
    tt = TurboTransformer(cache)
    comfy_path = Path(args.comfy_file).expanduser() if args.comfy_file else None
    if comfy_path is not None and comfy_path.exists() and comfy_path.stat().st_size == COMFY_SIZE:
        comfy = Safetensors(cache, "comfy-turbo-int8", path=comfy_path)
        out.input("comfy-file", {"source": "local", "size": COMFY_SIZE})
    else:
        comfy = Safetensors(cache, "comfy-turbo-int8", url=hub_url(COMFY_REPO, COMFY_REV, COMFY_FILE))
        out.input("comfy-file", {"source": "range", "url": hub_url(COMFY_REPO, COMFY_REV, COMFY_FILE)})

    h = hadamard_256()
    assert np.allclose(h @ h.T, np.eye(256), atol=1e-12)
    layers = {
        "attn.to_q": (["attn.to_q"], "attn.to_q"),
        # Comfy fuses the SwiGLU pair into gate_up; the official repo keeps them apart.
        "img_mlp.gate_up": (["img_mlp.gate_layer", "img_mlp.proj"], "img_mlp.gate_up"),
        "img_mlp.out": (["img_mlp.out"], "img_mlp.out"),
    }
    prefix = "transformer_blocks.0."
    report = {}
    for key, (official, comfy_name) in layers.items():
        ws = []
        for o in official:
            name = prefix + o + ".weight"
            ws.append(tt.tensor(name))
            note_input(out, tt.src(name), name)
        q = comfy.tensor(prefix + comfy_name + ".weight")
        s = comfy.tensor(prefix + comfy_name + ".weight_scale")
        note_input(out, comfy, prefix + comfy_name + ".weight")
        note_input(out, comfy, prefix + comfy_name + ".weight_scale")
        quant_cfg = bytes(comfy.raw(prefix + comfy_name + ".comfy_quant")[0]).decode("utf-8", "replace")
        w_gate_first = np.concatenate(ws, axis=0).astype(np.float32)  # [gate; proj] for gate_up
        in_dim = w_gate_first.shape[1]
        assert q.shape == w_gate_first.shape, (q.shape, w_gate_first.shape)
        assert q.dtype == np.int8 and s.shape == (q.shape[0], 1)

        x = np.random.default_rng(7 + len(report)).standard_normal((8, in_dim)).astype(np.float32)
        y_bf16 = (x.astype(np.float64) @ w_gate_first.astype(np.float64).T)
        w_deq = q.astype(np.float64) * s.astype(np.float64)
        y_int8 = rotate_activation(x, h) @ w_deq.T
        cos_rows = [cosine(y_bf16[i], y_int8[i]) for i in range(8)]
        rel_err = float(np.linalg.norm(y_bf16 - y_int8) / np.linalg.norm(y_bf16))
        entry = {
            "in_dim": in_dim, "out_dim": int(q.shape[0]), "official_tensors": [prefix + o + ".weight" for o in official],
            "comfy_tensor": prefix + comfy_name + ".weight", "comfy_quant": quant_cfg,
            "cosine_per_row": cos_rows, "cosine_min": min(cos_rows), "cosine_overall": cosine(y_bf16, y_int8),
            "relative_l2_error": rel_err,
        }
        if key == "img_mlp.gate_up":
            # Row order check: gate rows first (what the fused weight is built from) against proj rows first.
            w_swapped = np.concatenate(ws[::-1], axis=0).astype(np.float64)
            y_sw = x.astype(np.float64) @ w_swapped.T
            entry["cosine_overall_if_proj_first"] = cosine(y_sw, y_int8)
        tag = key.replace(".", "_")
        entry["files"] = {
            "x": out.f32(f"convrot_{tag}_x.f32", x, f"Seeded randn x [8, {in_dim}] for {key}."),
            "y_bf16": out.f32(f"convrot_{tag}_y_bf16.f32", y_bf16, f"x . W^T with the official BF16 weight(s) upcast, f64 accumulate [8, {q.shape[0]}]."),
            "y_int8": out.f32(f"convrot_{tag}_y_int8.f32", y_int8, "rot(x) . (q * scale)^T from the Comfy INT8 weight, f64 accumulate."),
            "q_top_left_64": out.f32(f"convrot_{tag}_q64.f32", q[:64, :64].astype(np.float32), "INT8 weight top-left 64 x 64 (as f32, exact)."),
            "scale_first_64": out.f32(f"convrot_{tag}_scale64.f32", s[:64, 0], "First 64 per-row weight scales."),
        }
        report[key] = entry
        print(key, "cosine_min", entry["cosine_min"], "rel_l2", rel_err)
    out.f32("convrot_hadamard_row0.f32", h[0], "First row of the group-256 normalised Hadamard H (kron^4 of H4, / 16).")
    out.json(
        "convrot.json",
        {
            "hadamard": {"base": [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], "group": 256,
                         "normalisation": "kron^4(H4) / 16, orthogonal", "row0": "convrot_hadamard_row0.f32"},
            "rule": "y = rot(x) . (q * scale)^T, rot(x)[g] = x[g] . H for each contiguous 256-group of the input axis; "
                    "scale is per output row. Implemented in this script from comfy-kitchen 0.2.37 _build_hadamard / "
                    "_rotate_activation (the package has no macOS wheel).",
            "gate_up_row_order": "gate rows first, then proj rows",
            "layers": report,
        },
        "Cosine and error of the Comfy INT8 convrot weights against the official BF16 weights for block 0 attn.to_q, "
        "img_mlp.gate_up and img_mlp.out, with seeded inputs and outputs.",
    )
    out.finish()


# ---------------------------------------------------------------- block

def build_one_layer_model(cache: Path, out: Out, layer: int):
    """`QwenImage21Transformer2DModel` with num_layers=1: the non-block weights and block `layer` in slot 0."""
    import torch
    from diffusers import QwenImage21Transformer2DModel

    tt = TurboTransformer(cache)
    cfg = QwenImage21Transformer2DModel.load_config(str(turbo_file(cache, "transformer/config.json").parent))
    model = QwenImage21Transformer2DModel.from_config(cfg, num_layers=1)
    sd = {}
    for n in tt.non_block_names():
        sd[n] = torch.from_numpy(tt.tensor(n).copy())
        note_input(out, tt.src(n), n)
    p = f"transformer_blocks.{layer}."
    for n in tt.block_names(layer):
        sd["transformer_blocks.0." + n[len(p):]] = torch.from_numpy(tt.tensor(n).copy())
        note_input(out, tt.src(n), n)
    model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    return model.float().eval(), cfg


class DenseBlockCausalProcessor:
    """An independent attention: one dense boolean mask `(q >= kv) | same image block` through plain SDPA."""

    def __init__(self, image_ids):
        self.image_ids = image_ids

    def __call__(self, attn, hidden_states, attention_mask=None, rotary_emb=None, layer_cache=None,
                 kv_cache_mode=None, cache_write_slice=None, segments=None, key_valid=None):
        import torch
        import torch.nn.functional as F
        from diffusers.models.transformers.transformer_qwenimage21 import _qwenimage21_prepare_qkv

        q, k, v, _ = _qwenimage21_prepare_qkv(attn, hidden_states, rotary_emb, None, None, None)
        n = q.shape[1]
        idx = torch.arange(n)
        ids = self.image_ids
        mask = (idx[:, None] >= idx[None, :]) | ((ids[:, None] == ids[None, :]) & (ids[:, None] >= 0))
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        o = o.transpose(1, 2).flatten(2, 3)
        return attn.to_out[1](attn.to_out[0](o))


def cmd_block(args):
    import torch
    from diffusers.models.transformers import transformer_qwenimage21 as tq
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    cache = Path(args.cache_dir).expanduser()
    out = Out("block")
    n_text, th, tw = 20, 4, 4
    n_img = th * tw
    sigma_main, sigma_prefill = 0.704534, 1.0
    enc_std = 1.0  # real features are not measured here (text-encoder is not run); txt_in's RMSNorm is scale invariant.
    g = seeded(1234)
    latents = torch.randn(1, n_img, 64, generator=g)
    enc = torch.randn(1, n_text, 4096, generator=g) * enc_std
    img_mask = torch.cat([torch.zeros(1, n_text, dtype=torch.bool), torch.ones(1, n_img // 4, dtype=torch.bool)], dim=1)
    img_shapes = [[(1, th, tw)]]
    out.f32("block_in_latents.f32", latents[0], "Seeded randn [16, 64] target latent tokens (seed 1234 stream, first).")
    out.f32("block_in_encoder.f32", enc[0], f"Seeded randn [20, 4096] * {enc_std} text features (same stream, second).")

    def ts(sigma):  # as the pipeline does: float32 t = sigma * 1000, then / 1000
        return torch.tensor([np.float32(np.float32(sigma) * np.float32(1000)) / np.float32(1000)])

    cases = {}
    for layer in (0, 31):
        model, cfg = build_one_layer_model(cache, out, layer)
        blk = model.transformer_blocks[0]
        rec = {}

        def blk_hook(mod, a, kw, output):
            rec["block_in"], rec["block_out"], rec["rotary"] = kw["hidden_states"].detach().clone(), output.detach().clone(), kw["rotary_emb"]

        h1 = blk.register_forward_hook(blk_hook, with_kwargs=True)
        h2 = model.modulation.register_forward_hook(lambda m, a, o: rec.__setitem__("modulation", o.detach().clone()))
        h3 = model.time_text_embed.register_forward_hook(lambda m, a, o: rec.__setitem__("temb", o.detach().clone()))
        h4 = model.txt_in.register_forward_hook(lambda m, a, o: rec.__setitem__("txt_in", o.detach().clone()))
        h5 = model.img_in.register_forward_hook(lambda m, a, o: rec.__setitem__("img_in", o.detach().clone()))
        qk = []
        orig_rot = tq.apply_rotary_emb_qwen

        def spy(x, freqs_cis, *a, **k):
            r = orig_rot(x, freqs_cis, *a, **k)
            qk.append(r.detach().clone())
            return r

        tq.apply_rotary_emb_qwen = spy

        def run(sigma, cache_obj=None, mode=None):
            rec.clear()
            qk.clear()
            with torch.no_grad():
                y = model(hidden_states=latents, encoder_hidden_states=enc, timestep=ts(sigma), img_shapes=img_shapes,
                          img_mask=img_mask, encoder_hidden_states_mask=None, kv_cache=cache_obj, kv_cache_mode=mode,
                          return_dict=False)[0]
            return y, dict(rec), [t.clone() for t in qk]

        try:
            full, r_full, qk_full = run(sigma_main)
            pre_full, _, _ = run(sigma_prefill)
            kv = QwenImage21KVCache(1)
            extract, r_ext, _ = run(sigma_prefill, kv, "extract")
            cached, r_cached, qk_cached = run(sigma_main, kv, "cached")
            # Negative control: a perturbed prefix cache must change the answer, or "0.0" would prove nothing.
            kv.layer_caches[0].k = kv.layer_caches[0].k * 1.01
            perturbed, _, _ = run(sigma_main, kv, "cached")
            kv.layer_caches[0].k = kv.layer_caches[0].k / 1.01
            # Dense independent mask over the same weights, to check the processor's segmented prefill.
            ids = torch.cat([torch.full((n_text,), -1), torch.zeros(n_img, dtype=torch.long)])
            default_proc = blk.attn.processor
            blk.attn.set_processor(DenseBlockCausalProcessor(ids))
            dense, _, _ = run(sigma_main)
            blk.attn.set_processor(default_proc)
        finally:
            tq.apply_rotary_emb_qwen = orig_rot
            for hh in (h1, h2, h3, h4, h5):
                hh.remove()

        tgt = slice(-n_img, None)
        diff_cached = float((full[:, tgt] - cached).abs().max())
        diff_extract = float((pre_full - extract).abs().max())
        diff_dense = float((full - dense).abs().max())
        diff_control = float((full[:, tgt] - perturbed).abs().max())
        scale = float(full.abs().max())
        tag = f"block{layer:02d}"
        files = {
            "txt_in": out.f32(f"{tag}_txt_in.f32", r_full["txt_in"][0], "txt_in output [20, 4096]."),
            "img_in": out.f32(f"{tag}_img_in.f32", r_full["img_in"][0], "img_in output [16, 4096] (before the joint scatter)."),
            "temb": out.f32(f"{tag}_temb.f32", r_full["temb"], "time_text_embed output [2, 4096]: row 0 = sigma, row 1 = t 0."),
            "modulation": out.f32(f"{tag}_modulation.f32", r_full["modulation"], "Shared modulation [2, 16384]: row 0 = sigma, row 1 = t 0."),
            "rope_cos": out.f32(f"{tag}_rope_cos.f32", r_full["rotary"].real, "Rotary cos [36, 64] the block received."),
            "rope_sin": out.f32(f"{tag}_rope_sin.f32", r_full["rotary"].imag, "Rotary sin [36, 64]."),
            "block_in": out.f32(f"{tag}_block_in.f32", r_full["block_in"][0], "Block input [36, 4096]: 20 text then 16 image tokens."),
            "block_out": out.f32(f"{tag}_block_out.f32", r_full["block_out"][0], "Block output [36, 4096], full block-causal path."),
            "q_head0": out.f32(f"{tag}_q_head0.f32", qk_full[0][0, :, 0], "q after RMSNorm and RoPE, head 0 [36, 128]."),
            "k_head0": out.f32(f"{tag}_k_head0.f32", qk_full[1][0, :, 0], "k after RMSNorm and RoPE, head 0 [36, 128]."),
            "model_out": out.f32(f"{tag}_model_out.f32", full[0], "Model output over the joint sequence [36, 64] (proj_out)."),
            "target_out": out.f32(f"{tag}_target_out.f32", full[0, tgt], "Target-token slice of the model output [16, 64]."),
            "cached_block_in": out.f32(f"{tag}_cached_block_in.f32", r_cached["block_in"][0], "Cached-path block input [16, 4096] (target tokens only)."),
            "cached_block_out": out.f32(f"{tag}_cached_block_out.f32", r_cached["block_out"][0], "Cached-path block output [16, 4096]."),
            "cached_q_head0": out.f32(f"{tag}_cached_q_head0.f32", qk_cached[0][0, :, 0], "Cached-path q head 0 [16, 128]."),
            "cached_out": out.f32(f"{tag}_cached_out.f32", cached[0], "Cached-path model output for the target tokens [16, 64] at sigma 0.704534."),
            "prefill_sigma1_target_out": out.f32(f"{tag}_prefill_target_out.f32", pre_full[0, tgt], "Full-path target output at sigma 1.0 [16, 64]."),
        }
        cases[tag] = {
            "checkpoint_block": layer, "files": files, "output_abs_max": scale,
            "max_abs_diff_full_vs_cached": diff_cached,
            "max_abs_diff_prefill_full_vs_extract": diff_extract,
            "max_abs_diff_segmented_vs_dense_mask": diff_dense,
            "negative_control_max_abs_diff_k_scaled_1p01": diff_control,
        }
        print(tag, "control", diff_control, "full vs cached", diff_cached, "extract", diff_extract, "dense", diff_dense, "|y|max", scale)
        del model

    out.json(
        "block.json",
        {
            "layout": {"text_tokens": n_text, "latent_h": th, "latent_w": tw, "slots": n_img // 4,
                       "img_mask": "20 False then 4 True (each slot stands for 4 latent tokens)"},
            "sigma": sigma_main, "prefill_sigma": sigma_prefill, "timestep_passed": float(ts(sigma_main)[0]),
            "encoder_std": enc_std, "seed": 1234, "inputs": {"latents": "block_in_latents.f32", "encoder": "block_in_encoder.f32"},
            "kv_cache": "step 0 = kv_cache_mode 'extract' at sigma 1.0 (full pass, stores prefix K/V after RoPE); the later "
                        "step = 'cached' at sigma 0.704534 with the same prefix, computing target tokens only.",
            "model_config": cfg,
            "cases": cases,
        },
        "One-layer QwenImage21Transformer2DModel (non-block weights plus block 0 or block 31 in slot 0, f32) over 20 text tokens "
        "and a 4x4 target at sigma 0.704534: per-stage tensors, the full block-causal path and the prefix-KV-cache path.",
    )
    out.finish()


# ---------------------------------------------------------------- vae

def cmd_vae(args):
    import torch
    from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline
    from safetensors.torch import load_file

    cache = Path(args.cache_dir).expanduser()
    out = Out("vae")
    cfg_path = turbo_file(cache, "vae/config.json")
    weights = turbo_file(cache, "vae/diffusion_pytorch_model.safetensors", expect_size=675508656)
    out.input("vae-weights", {"sha256": sha256_file(weights), "bytes": weights.stat().st_size})
    vae = AutoencoderKLQwenImage21.from_config(AutoencoderKLQwenImage21.load_config(str(cfg_path.parent)))
    vae.load_state_dict({k: v.float() for k, v in load_file(str(weights)).items()}, strict=True)
    vae = vae.float().eval()

    side, lat = 256, 16
    g = seeded(77)
    packed = torch.randn(1, lat * lat, vae.config.z_dim, generator=g)  # the pipeline's packed layout [1, tokens, 64]
    # Exactly the pipeline's decode preamble.
    z = QwenImage21Pipeline._unpack_latents(packed, side, side, 16)
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
    z_post = z * std + mean

    rec = {}
    hp = vae.post_quant_conv.register_forward_hook(lambda m, a, o: rec.setdefault("pqc", o.detach().clone()))
    hm = vae.decoder.mid_block.register_forward_hook(lambda m, a, o: rec.setdefault("mid", o.detach().clone()))
    hc = vae.decoder.conv_in.register_forward_hook(lambda m, a, o: rec.setdefault("conv_in", o.detach().clone()))
    with torch.no_grad():
        img = vae.decode(z_post, return_dict=False)[0][:, :, 0]
    for hh in (hp, hm, hc):
        hh.remove()
    assert img.shape == (1, 4, side, side), img.shape
    f = {
        "latents_packed": out.f32("vae_latents_packed.f32", packed[0], "Seeded randn [256, 64] packed latents (seed 77): the pipeline's tokens x channels layout, PRE-denorm."),
        "latents_post_denorm": out.f32("vae_latents_post_denorm.f32", z_post[0, :, 0], "The decode input [64, 16, 16]: unpacked, then * latents_std + latents_mean (POST-denorm)."),
        "post_quant_conv": out.f32("vae_post_quant_conv.f32", rec["pqc"][0, :, 0], "post_quant_conv output [64, 16, 16]."),
        "decoder_conv_in": out.f32("vae_decoder_conv_in.f32", rec["conv_in"][0, :, 0], f"decoder.conv_in output {list(rec['conv_in'].shape[1:2]) + [lat, lat]}."),
        "mid_block": out.f32("vae_mid_block.f32", rec["mid"][0, :, 0], f"decoder.mid_block output {list(rec['mid'].shape[1:2]) + [lat, lat]}."),
        "image": out.f32("vae_image.f32", img[0], "Decoded RGBA [4, 256, 256], clamped to [-1, 1]."),
    }
    out.json(
        "vae.json",
        {
            "input_is": "post-denorm: the pipeline computes latents * latents_std + latents_mean before vae.decode; the "
                        "packed file is the pre-denorm value it started from",
            "unpack": "tokens [h*w, 64] -> transpose -> [64, 1, h, w] (row-major token order)",
            "latent": [1, 64, 1, lat, lat], "output": [1, 4, side, side], "clamp": [-1.0, 1.0],
            "latents_mean": vae.config.latents_mean, "latents_std": vae.config.latents_std,
            "image_min": float(img.min()), "image_max": float(img.max()),
            "files": f,
        },
        "AutoencoderKLQwenImage21 decode (f32) of a seeded 16 x 16 latent to 256 x 256 RGBA, with post_quant_conv, "
        "decoder.conv_in and mid-block activations.",
    )
    out.finish()


# ---------------------------------------------------------------- text encoder

def cmd_text_encoder(args):
    """Needs the 17.5 GB BF16 text encoder upcast to f32 (~40 GB of RAM). Not run on a laptop."""
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    cache = Path(args.cache_dir).expanduser()
    out = Out("text-encoder")
    te_dir = Path(args.text_encoder_dir).expanduser()
    if not (te_dir / "model.safetensors").exists():
        sys.exit(f"{te_dir}/model.safetensors missing: pass --text-encoder-dir with the Turbo repo's text_encoder/")
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(str(te_dir), dtype=torch.float32).eval()
    processor = Qwen3VLProcessor.from_pretrained(str(processor_dir(cache)))
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(json.loads(turbo_file(cache, "scheduler/scheduler_config.json").read_text()))
    pipe = QwenImage21Pipeline(scheduler=scheduler, vae=None, text_encoder=encoder, processor=processor, transformer=None)
    cases = {}
    with torch.no_grad():
        for key in TEXT_ENCODER_PROMPTS:
            # The pipeline's own method: last decoder layer's output before the final norm, left padding, drop_idx.
            embeds, mask, _ = pipe._get_qwen_prompt_embeds(PROMPTS[key], None, torch.device("cpu"))
            cases[key] = {
                "prompt": PROMPTS[key], "shape": list(embeds.shape[1:]),
                "features": out.f32(f"text_encoder_{key}.f32", embeds[0], f"Prompt features [tokens, 4096] for {key!r}: last decoder layer before the final RMSNorm, after drop_idx."),
                "feature_std": float(embeds.std()), "feature_abs_max": float(embeds.abs().max()),
            }
    out.json("text_encoder.json", {"drop_idx": pipe._drop_idx, "cases": cases},
             "Qwen3-VL text encoder (f32) prompt features as the pipeline produces them, plus their measured scale.")
    out.finish()


# ---------------------------------------------------------------- text encoder, layer-streamed

TE_FILE = "text_encoder/model.safetensors"
TE_PREFIX = "model.language_model."
# Every sixth decoder layer's output is kept for debugging a drift: 0-indexed layers 5, 11, ..., 29;
# the last layer (35) is the feature itself.
TE_DEBUG_EVERY = 6


def prefetch(src: Safetensors, names, workers: int = 8, chunk: int = 32 << 20):
    """Range-read whole tensors into `src`'s cache in parallel chunks (the hub answers ~10 MB/s per connection)."""
    from concurrent.futures import ThreadPoolExecutor

    jobs = []
    for name in names:
        lo, hi = src.header[name]["data_offsets"]
        dest = src.dir / (name.replace("/", "_") + ".bin")
        if dest.exists() and dest.stat().st_size == hi - lo:
            continue
        part = dest.with_suffix(".part")
        with open(part, "wb") as f:
            f.truncate(hi - lo)
        for off in range(0, hi - lo, chunk):
            jobs.append((part, src.base + lo + off, off, min(chunk, hi - lo - off)))
        jobs.append((dest, None, None, None))  # rename marker

    def one(job):
        part, start, off, length = job
        data = src._read(start, length)
        with open(part, "r+b") as f:
            f.seek(off)
            f.write(data)

    reads = [j for j in jobs if j[1] is not None]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, reads))
    for dest, start, _, _ in jobs:
        if start is None:
            dest.with_suffix(".part").rename(dest)


def evict(src: Safetensors, names):
    for name in names:
        (src.dir / (name.replace("/", "_") + ".bin")).unlink(missing_ok=True)


def embed_rows(src: Safetensors, ids) -> np.ndarray:
    """Embedding rows for `ids`, each range-read on its own (8 KiB apiece), never the 1.2 GB table."""
    from concurrent.futures import ThreadPoolExecutor

    name = TE_PREFIX + "embed_tokens.weight"
    meta = src.header[name]
    assert meta["dtype"] == "BF16"
    width = meta["shape"][1]
    lo, _ = meta["data_offsets"]
    rows_dir = src.dir / "embed_rows"
    rows_dir.mkdir(exist_ok=True)

    def row(i):
        cached = rows_dir / f"{i}.bin"
        if cached.exists() and cached.stat().st_size == 2 * width:
            return cached.read_bytes()
        data = src._read(src.base + lo + i * 2 * width, 2 * width)
        cached.write_bytes(data)
        return data

    uniq = sorted(set(ids))
    with ThreadPoolExecutor(8) as pool:
        got = dict(zip(uniq, pool.map(row, uniq)))
    table = {i: (np.frombuffer(b, dtype="<u2").astype(np.uint32) << 16).view(np.float32) for i, b in got.items()}
    return np.stack([table[i] for i in ids]).astype(np.float32), {i: sha256_bytes(b) for i, b in got.items()}


def layer_names(src: Safetensors, layer: int):
    p = f"{TE_PREFIX}layers.{layer}."
    return sorted(n for n in src.header if n.startswith(p))


def build_layer(src: Safetensors, tcfg, layer: int):
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer

    p = f"{TE_PREFIX}layers.{layer}."
    mod = Qwen3VLTextDecoderLayer(tcfg, layer).eval()
    sd = {n[len(p):]: torch.from_numpy(src.tensor(n).copy()) for n in layer_names(src, layer)}
    mod.load_state_dict(sd, strict=True)
    return mod


def cmd_text_encoder_streamed(args):
    """The Qwen3-VL text encoder's prompt features without holding it: one f32 decoder layer at a time.

    The BF16 checkpoint is 17.5 GB (about 40 GB as f32), so the layers are range-read one by one from the hub,
    instantiated as transformers' own `Qwen3VLTextDecoderLayer`, run over every prompt with the text model's own
    rotary table (`Qwen3VLTextRotaryEmbedding`) and causal mask (`create_causal_mask`), and evicted from the cache.
    The loop is first checked against the pipeline's own `_get_qwen_prompt_embeds` on a real-weight model cut to
    two decoder layers (positions, mask, the final-norm bypass and drop_idx all come from the official code there).
    """
    import gc
    import threading

    import torch
    from diffusers import QwenImage21Pipeline
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

    torch.set_grad_enabled(False)
    cache = Path(args.cache_dir).expanduser()
    out = Out("text-encoder-streamed")
    url = hub_url(TURBO_REPO, TURBO_REV, TE_FILE)
    src = Safetensors(cache, "turbo-text_encoder", url=url)
    out.input("text-encoder-file", {"source": "range", "url": url})
    cfg_path = turbo_file(cache, "text_encoder/config.json")
    full_cfg = Qwen3VLConfig.from_pretrained(str(cfg_path.parent))
    tcfg = full_cfg.text_config
    tcfg._attn_implementation = "eager"
    layers = tcfg.num_hidden_layers
    pipe = stub_pipeline(cache)
    drop = pipe._drop_idx

    ids = {key: pipeline_ids(pipe, PROMPTS[key]) for key in TEXT_ENCODER_PROMPTS}
    all_ids = sorted({i for v in ids.values() for i in v})
    _, row_digests = embed_rows(src, all_ids)
    out.input("text-encoder-embed-rows", {"tensor": TE_PREFIX + "embed_tokens.weight", "sha256_per_row": {str(k): v for k, v in row_digests.items()}})
    embeds = {key: embed_rows(src, v)[0] for key, v in ids.items()}

    # ---- cross-check: the official pipeline on a two-layer cut of the real model, against this loop.
    cut = 2
    prefetch(src, [n for l in range(cut) for n in layer_names(src, l)] + [TE_PREFIX + "norm.weight"])
    small = Qwen3VLConfig.from_pretrained(str(cfg_path.parent))
    small.text_config.num_hidden_layers = cut
    small.vision_config.depth = 1
    small.vision_config.deepstack_visual_indexes = []
    small.tie_word_embeddings = True
    small.text_config._attn_implementation = "eager"
    small._attn_implementation = "eager"
    model = Qwen3VLForConditionalGeneration(small).to(torch.float32).eval()
    lm = model.model.language_model
    with torch.no_grad():
        lm.embed_tokens.weight.zero_()
        rows, _ = embed_rows(src, all_ids)
        lm.embed_tokens.weight[torch.tensor(all_ids)] = torch.from_numpy(rows)
        for l in range(cut):
            lm.layers[l].load_state_dict(build_layer(src, tcfg, l).state_dict(), strict=True)
        lm.norm.weight.copy_(torch.from_numpy(src.tensor(TE_PREFIX + "norm.weight").copy()))
    note_input(out, src, TE_PREFIX + "norm.weight")
    pipe.text_encoder = model
    official_cut = {}
    for key in TEXT_ENCODER_PROMPTS:
        e, _, _ = pipe._get_qwen_prompt_embeds(PROMPTS[key], None, torch.device("cpu"))
        official_cut[key] = e[0].numpy().copy()
    pipe.text_encoder = None
    del model, lm
    gc.collect()

    # ---- the streamed loop.
    rotary = Qwen3VLTextRotaryEmbedding(tcfg)
    state = {}
    for key, e in embeds.items():
        h = torch.from_numpy(e)[None]
        t = h.shape[1]
        # The text model's own default for text-only input: every axis carries the token index.
        pos = torch.arange(t).view(1, 1, -1).expand(4, 1, -1)
        mask = create_causal_mask(config=tcfg, inputs_embeds=h, attention_mask=torch.ones(1, t, dtype=torch.long),
                                  past_key_values=None, position_ids=pos[0])
        state[key] = {"h": h, "pe": rotary(h, pos[1:]), "mask": mask, "pos": pos[0]}

    debug = {key: {} for key in embeds}
    cross = {}
    pending = None
    for l in range(layers):
        names = layer_names(src, l)
        if pending is not None:
            pending.join()
        prefetch(src, names)  # no-op when the background fetch already landed it
        nxt = layer_names(src, l + 1) if l + 1 < layers else []
        pending = threading.Thread(target=prefetch, args=(src, nxt)) if nxt else None
        if pending is not None:
            pending.start()
        for n in names:
            note_input(out, src, n)
        layer = build_layer(src, tcfg, l)
        for key, st in state.items():
            st["h"] = layer(st["h"], position_embeddings=st["pe"], attention_mask=st["mask"], position_ids=st["pos"])
            if l + 1 == cut:
                got = st["h"][0, drop:].numpy()
                cross[key] = float(np.abs(got - official_cut[key]).max())
            if (l + 1) % TE_DEBUG_EVERY == 0 and l + 1 < layers:
                debug[key][l] = st["h"][0, drop:].numpy().copy()
        del layer
        evict(src, names)
        print(f"[text-encoder-streamed] layer {l} done", flush=True)
    if pending is not None:
        pending.join()
    evict(src, [TE_PREFIX + "norm.weight"])
    print("cross-check max |streamed - official| after", cut, "layers:", cross)
    assert all(v <= 1e-4 for v in cross.values()), cross

    cases = {}
    for key, st in state.items():
        feat = st["h"][0, drop:].numpy()
        cases[key] = {
            "prompt": PROMPTS[key], "ids": ids[key], "drop_idx": drop, "shape": list(feat.shape),
            "hidden": out.f32(f"te_{key}_hidden.f32", feat,
                              f"Prompt features [kept tokens, {feat.shape[1]}] for {key!r}: output of the last decoder layer "
                              "(35), before the final RMSNorm, after drop_idx."),
            "layers": {str(l): out.f32(f"te_{key}_layer{l:02d}.f32", v,
                                       f"Residual stream after 0-indexed decoder layer {l} for {key!r}, after drop_idx (debugging).")
                       for l, v in debug[key].items()},
            "feature_std": float(feat.std()), "feature_abs_max": float(np.abs(feat).max()),
        }
    out.json(
        "te_streamed.json",
        {
            "drop_idx": drop, "layers": layers, "debug_every": TE_DEBUG_EVERY,
            "cross_check": {"decoder_layers": cut, "max_abs_diff": cross,
                            "against": "QwenImage21Pipeline._get_qwen_prompt_embeds on Qwen3VLForConditionalGeneration "
                                       "cut to two text layers with the same real weights"},
            "cases": cases,
        },
        "Qwen3-VL-8B text encoder prompt features (f32 over the official BF16 weights), computed one decoder layer at a "
        "time, with every sixth layer's residual stream, and the two-layer cross-check against the pipeline's own code.",
    )
    out.finish()


def cmd_text_encoder_int8_emulation(args):
    """Separates the INT8 export's own rounding from kernel error in the device text-encoder case.

    Runs the same f32 decoder layers as `text-encoder-streamed`, but over the Comfy INT8 weights: each
    projection is `int8 * per-row scale`, then un-rotated (the file stores `W . H` per 256-group of the input
    axis and `H` is its own inverse). Token embeddings are dequantized and un-rotated the same way. The result
    is graded against the BF16 fixtures that `text-encoder-streamed` wrote, every sixth layer and the last.
    Measured (min per-token cosine at the last layer): 0.986558 for this emulation against 0.98657 for the
    device case, and device against emulation 1.0000000, so the checkpoint itself misses a 0.995 bar and the
    kernels add nothing. Reads the Comfy file locally (it is 8 GB); writes no fixture, only prints.
    """
    import torch
    from transformers import Qwen3VLConfig
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLTextRotaryEmbedding

    torch.set_grad_enabled(False)
    cache = Path(args.cache_dir).expanduser()
    fixtures = Path(args.fixture_dir).expanduser()
    src = Safetensors(cache, "comfy-te-int8", path=Path(args.comfy_te_file).expanduser())
    h256 = hadamard_256().astype(np.float32)

    def unrotated(name: str) -> np.ndarray:
        w = src.tensor(name + ".weight").astype(np.float32) * src.tensor(name + ".weight_scale").astype(np.float32)
        o, i = w.shape
        return (w.reshape(o, i // 256, 256) @ h256).reshape(o, i)

    doc = json.loads((fixtures / "te_streamed.json").read_text())
    cfg = Qwen3VLConfig.from_pretrained(str(Path(args.text_encoder_dir).expanduser()))
    tcfg = cfg.text_config
    tcfg._attn_implementation = "eager"
    rot = Qwen3VLTextRotaryEmbedding(tcfg)
    drop_idx = 14  # the template's fixed system turn, which the fixtures leave out

    q = src.tensor("model.embed_tokens.weight")
    scale = src.tensor("model.embed_tokens.weight_scale")
    state = {}
    for key, case in doc["cases"].items():
        ids = case["ids"]
        rows = q[ids].astype(np.float32) * scale[ids]
        e = (rows.reshape(len(ids), -1, 256) @ h256).reshape(len(ids), -1)
        h = torch.from_numpy(np.ascontiguousarray(e, dtype=np.float32))[None]
        t = h.shape[1]
        pos = torch.arange(t).view(1, 1, -1).expand(4, 1, -1)
        mask = create_causal_mask(config=tcfg, inputs_embeds=h, attention_mask=torch.ones(1, t, dtype=torch.long),
                                  past_key_values=None, position_ids=pos[0])
        state[key] = dict(h=h, pe=rot(h, pos[1:]), mask=mask, pos=pos[0])

    def cos_rows(a, b):
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)

    for l in range(36):
        p = f"model.layers.{l}."
        layer = Qwen3VLTextDecoderLayer(tcfg, l).eval()
        sd = {n + ".weight": torch.from_numpy(unrotated(p + n)) for n in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
        for n in ("input_layernorm.weight", "post_attention_layernorm.weight",
                  "self_attn.q_norm.weight", "self_attn.k_norm.weight"):
            sd[n] = torch.from_numpy(src.tensor(p + n).copy())
        layer.load_state_dict(sd, strict=True)
        for key, st in state.items():
            st["h"] = layer(st["h"], position_embeddings=st["pe"], attention_mask=st["mask"], position_ids=st["pos"])
            if l in (5, 11, 17, 23, 29, 35):
                got = st["h"][0, drop_idx:].numpy()
                name = f"te_{key}_hidden.f32" if l == 35 else f"te_{key}_layer{l:02d}.f32"
                want = np.fromfile(fixtures / name, dtype="<f4").reshape(got.shape)
                c = cos_rows(got, want)
                print(f"{key} layer {l}: cos min {c.min():.6f} mean {c.mean():.6f} first {c[0]:.6f}", flush=True)
        for f in src.dir.glob(f"model.layers.{l}.*"):
            f.unlink()
        del layer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="~/.cache/shiso-ref/qwen-image21",
                    help="where configs, the VAE and range-read tensors are cached")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("tokenizer", cmd_tokenizer), ("sigmas", cmd_sigmas), ("embed", cmd_embed), ("rope", cmd_rope),
                     ("convrot", cmd_convrot), ("block", cmd_block), ("vae", cmd_vae)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if name == "convrot":
            p.add_argument("--comfy-file", default="~/.shiso/staging/qwen-image-2.1/turbo-int8/image/diffusion_models/"
                           "qwen_image_2.1_turbo_int8_convrot.safetensors",
                           help=f"local copy of the Comfy INT8 file, used only if it is exactly {COMFY_SIZE} bytes; "
                                "otherwise the tensors are range-read from the hub")
    p = sub.add_parser("text-encoder-streamed", help="layer-streamed text encoder (peak ~2 GB disk, ~3 GB RAM)")
    p.set_defaults(fn=cmd_text_encoder_streamed)
    p = sub.add_parser("text-encoder-int8-emulation",
                       help="f32 layers over the dequantized, un-rotated Comfy INT8 text encoder vs the BF16 fixtures")
    p.set_defaults(fn=cmd_text_encoder_int8_emulation)
    p.add_argument("--comfy-te-file", default="~/.shiso/staging/qwen-image-2.1/turbo-int8/image/text_encoders/"
                   "qwen3vl_8b_int8_convrot.safetensors", help="local copy of the Comfy INT8 text encoder")
    p.add_argument("--text-encoder-dir", required=True, help="the Turbo repo's text_encoder/ directory (for its config)")
    p.add_argument("--fixture-dir", default="test-vectors/references/qwen_image21",
                   help="where text-encoder-streamed wrote te_*.f32 and te_streamed.json")
    p = sub.add_parser("text-encoder", help="needs ~40 GB RAM")
    p.set_defaults(fn=cmd_text_encoder)
    p.add_argument("--text-encoder-dir", required=True, help="the Turbo repo's text_encoder/ directory (model.safetensors, config)")
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
