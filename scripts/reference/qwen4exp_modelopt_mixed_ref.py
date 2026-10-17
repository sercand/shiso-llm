#!/usr/bin/env python3
"""Independent reference for NVIDIA's mixed-precision qwen4exp export (FP4-Local).

Runs the Python library's own `modeling_qwen4_exp` modules over the
`modelopt` MIXED_PRECISION checkpoint (Qwen3.8-Flash-Next-FP4-Local),
teacher-forced, with no KV cache, and records what it answered as a
`shiso-local-logit-vectors-v1` fixture of top-64 log-probabilities, in the
shape of `test-vectors/references/qwen38-flash-fp4local-vllm-golden.vec`.

Numerics (W-dequant, A16). Every quantized matrix is dequantized to BF16 once,
on the fly, and multiplied with BF16 activations by the library's own modules,
which is exactly how the library runs a BF16 checkpoint:

- MXFP8 projections (attention, indexer, recurrent and shared-expert; `e4m3`
  weight plus one `ue8m0` byte per 32 inputs): `bf16(e4m3 * 2^(s-127))`, exact.
- NVFP4 routed experts with CSF-compressed group scales: the CSF stream is
  expanded to one `e4m3` byte per 16 inputs, then
  `bf16((e2m1 * e4m3) * weight_scale_2)` (the product order the reference
  server uses: the exact e2m1*e4m3 product first, then the f32 global).
- The NVFP4 n-gram table: rows gathered straight from the 128 shards (row r in
  shard r // rows_per_shard, as the library concatenates them), dequantized the
  same way with the table's one `weight_scale_2`. The hashing that picks the
  rows is the library's module, not ours.
- BF16 tensors (head, embeddings, routers, stream mixes, n-gram projections,
  norms stored as BF16) are used bit for bit. The 100 norms stored as F32
  (stream-mix and n-gram norms) are kept F32 by default: the library's norm
  computes `x_f32 * (1 + w.float())`, so an F32 parameter is used exactly
  (`--norm-dtype bf16` rounds them as a bf16 `from_pretrained` would).

The library's own attention, indexer, recurrence (pure-torch paths: the fused
kernel packages are hidden before import, as in `qwen4exp_fp8_ref.py`) and
sparse-MoE loop run unmodified. Only two modules are swapped: each layer's
routed experts stream one expert at a time from the shards (the same eager loop
as the library's), and the n-gram embedding is the shard gather above.

The logits at the recorded frontiers are fp32 from the BF16 final hidden state
and the BF16 head; the log-probabilities subtract the full-row log-sum-exp the
same way `shiso-bench`'s `logprob_top_k` does.

Run inside the reference container (transformers 5.19), see `--help`:

    selfcheck  CPU: CSF vs golden fixtures, MXFP8/NVFP4/n-gram decode checks,
               and the full loading plan against the checkpoint (meta device)
    smoke      CPU: a forward through the first N layers (default 4)
    capture    GPU: the logprob fixture
    greedy     GPU: 64 greedy ids after the first case's prompt, as JSON

Helpers shared with the FP8 reference are imported from `qwen4exp_fp8_ref.py`
by path, so this runs under `python3 -I`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _load_template():
    path = Path(__file__).resolve().parent / "qwen4exp_fp8_ref.py"
    spec = importlib.util.spec_from_file_location("qwen4exp_fp8_ref", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen4exp_fp8_ref"] = mod
    # Importing it also hides the fused kernel packages and pins the cuBLAS
    # workspace, both of which must happen before the library is imported.
    spec.loader.exec_module(mod)
    return mod


T = _load_template()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

log = T.log
mem_available = T.mem_available
REPO = Path(__file__).resolve().parents[2]
LM = "model.language_model"
TOP_COUNT = 64
LOCAL_LOGIT_HEADER = T.LOCAL_LOGIT_HEADER
GIB = 2**30
ABORT_MEM_AVAILABLE = 8 * GIB

ITEMSIZE = {"F8_E4M3": 1, "BF16": 2, "F32": 4, "U8": 1, "U32": 4, "I64": 8}
TORCH_DTYPE = {
    "F8_E4M3": torch.float8_e4m3fn,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "U8": torch.uint8,
    "U32": torch.int32,  # reinterpreted; decode_csf masks it back to 32 bits
    "I64": torch.int64,
}
E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
E2M1 = torch.tensor(E2M1_VALUES, dtype=torch.float32)
E2M1_NP = np.array(E2M1_VALUES, dtype=np.float32)
# ue8m0: 2^(s-127), 255 is NaN. Every finite entry is a power of two an f32
# holds exactly (2^-127 as a subnormal).
UE8M0 = torch.tensor([math.ldexp(1.0, s - 127) for s in range(255)] + [float("nan")],
                     dtype=torch.float32)


# --------------------------------------------------------------------------
# Resource guards
# --------------------------------------------------------------------------


class MemWatchdog(threading.Thread):
    """Kills the process the moment MemAvailable drops under the floor.

    On this machine the GPU allocates from the same memory as the host, so a
    runaway allocation starves everything else on the box. A thread rather than
    a check in the loop, so it fires even while the main thread waits on the
    device.
    """

    def __init__(self, floor: int):
        super().__init__(daemon=True)
        self.floor = floor
        self.min_seen = mem_available()

    def run(self) -> None:
        while True:
            avail = mem_available()
            self.min_seen = min(self.min_seen, avail)
            if avail < self.floor:
                log(f"ABORT: MemAvailable {avail / GIB:.1f} GiB < {self.floor / GIB:.0f} GiB")
                os._exit(3)
            time.sleep(0.2)


WATCHDOG: MemWatchdog | None = None


def guard(min_start_gib: float, gpu: bool) -> None:
    """Refuses to start when the box cannot take the run, then arms the watchdog."""
    global WATCHDOG
    if gpu:
        busy = []
        if shutil.which("nvidia-smi"):
            r = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                 "--format=csv,noheader"], capture_output=True, text=True)
            apps = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
            if r.returncode != 0:
                log(f"guard: nvidia-smi failed ({r.returncode}); cannot see other GPU users")
            busy += [f"GPU process {a}" for a in apps]
        else:
            log("guard: no nvidia-smi here; cannot see other GPU users")
        if shutil.which("pgrep"):
            for name in ("shiso", "ds4", "ds4-server"):
                r = subprocess.run(["pgrep", "-x", name], capture_output=True, text=True)
                if r.stdout.strip():
                    busy.append(f"{name} (pid {' '.join(r.stdout.split())})")
            r = subprocess.run(["pgrep", "-f", r"vllm serve|vllm\.entrypoints"],
                               capture_output=True, text=True)
            pids = [p for p in r.stdout.split() if int(p) != os.getpid()]
            if pids:
                busy.append(f"vllm (pid {' '.join(pids)})")
        if busy:
            sys.exit(f"refusing: the GPU is in use: {', '.join(busy)}")
    avail = mem_available()
    if avail < min_start_gib * GIB:
        sys.exit(f"refusing: MemAvailable {avail / GIB:.1f} GiB < {min_start_gib:.0f} GiB")
    WATCHDOG = MemWatchdog(ABORT_MEM_AVAILABLE)
    WATCHDOG.start()
    log(f"guard: MemAvailable {avail / GIB:.1f} GiB; aborting below "
        f"{ABORT_MEM_AVAILABLE / GIB:.0f} GiB")


# --------------------------------------------------------------------------
# Checkpoint access (the shards are untrusted: every offset is checked)
# --------------------------------------------------------------------------


class Checkpoint:
    def __init__(self, root: Path):
        self.root = root.resolve()
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        wm = index.get("weight_map")
        if not isinstance(wm, dict):
            raise ValueError("index has no weight_map")
        self.files: dict[str, Path] = {}
        resolved: dict[str, Path] = {}
        for name, fname in wm.items():
            if not isinstance(fname, str):
                raise ValueError(f"{name}: shard name is not a string")
            if fname not in resolved:
                p = (self.root / fname).resolve()
                if not p.is_relative_to(self.root) or not p.is_file():
                    raise ValueError(f"{name}: shard {fname!r} is not a file inside the checkpoint")
                resolved[fname] = p
            self.files[name] = resolved[fname]
        self._headers: dict[Path, tuple[int, dict, int]] = {}
        self._fds: dict[Path, int] = {}
        self._lock = threading.Lock()
        self.bytes_read = 0

    def header(self, path: Path) -> tuple[int, dict, int]:
        with self._lock:
            if path not in self._headers:
                size = path.stat().st_size
                with open(path, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    if n > 512 * 2**20 or 8 + n > size:
                        raise ValueError(f"{path.name}: header length {n} is implausible")
                    hdr = json.loads(f.read(n))
                self._headers[path] = (8 + n, hdr, size)
            return self._headers[path]

    def has(self, name: str) -> bool:
        return name in self.files

    def info(self, name: str) -> tuple[Path, int, int, str, list[int]]:
        path = self.files[name]
        base, hdr, size = self.header(path)
        if name not in hdr:
            raise KeyError(f"{name} is not in {path.name}")
        i = hdr[name]
        dtype, shape = i["dtype"], [int(d) for d in i["shape"]]
        start, end = (int(x) for x in i["data_offsets"])
        if dtype not in ITEMSIZE:
            raise TypeError(f"{name}: unsupported dtype {dtype}")
        if not (0 <= start <= end and base + end <= size):
            raise ValueError(f"{name}: data offsets outside {path.name}")
        if end - start != math.prod(shape) * ITEMSIZE[dtype]:
            raise ValueError(f"{name}: {end - start} bytes for {dtype} {shape}")
        return path, base + start, end - start, dtype, shape

    def dtype_shape(self, name: str) -> tuple[str, list[int]]:
        _, _, _, dtype, shape = self.info(name)
        return dtype, shape

    def _fd(self, path: Path) -> int:
        with self._lock:
            if path not in self._fds:
                self._fds[path] = os.open(path, os.O_RDONLY)
            return self._fds[path]

    def read(self, name: str, want: str | None = None) -> torch.Tensor:
        """The tensor's bytes read with pread into owned host memory, as torch."""
        path, off, nbytes, dtype, shape = self.info(name)
        if want is not None and dtype != want:
            raise TypeError(f"{name}: {dtype}, want {want}")
        buf = bytearray(nbytes)
        view = memoryview(buf)
        fd = self._fd(path)
        got = 0
        while got < nbytes:
            n = os.preadv(fd, [view[got:]], off + got)
            if n <= 0:
                raise IOError(f"short read of {name}")
            got += n
        self.bytes_read += nbytes
        if nbytes == 0:
            return torch.empty(shape, dtype=TORCH_DTYPE[dtype])
        return torch.frombuffer(buf, dtype=torch.uint8).view(TORCH_DTYPE[dtype]).reshape(shape)

    def memmap(self, name: str, want: str, row_bytes: int) -> np.memmap:
        path, off, nbytes, dtype, shape = self.info(name)
        if dtype != want or len(shape) != 2 or shape[1] * ITEMSIZE[dtype] != row_bytes:
            raise TypeError(f"{name}: {dtype} {shape}, want {want} rows of {row_bytes} bytes")
        return np.memmap(path, dtype=np.uint8, mode="r", offset=off, shape=(shape[0], row_bytes))


# --------------------------------------------------------------------------
# Decoders (torch, any device)
# --------------------------------------------------------------------------


def as_u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype != torch.uint8 else t


def e4m3_f32(t: torch.Tensor, what: str) -> torch.Tensor:
    b = as_u8(t)
    if bool(((b & 0x7F) == 0x7F).any()):
        raise ValueError(f"{what}: NaN e4m3 code")
    return b.view(torch.float8_e4m3fn).float()


def decode_csf(fixed: torch.Tensor, exceptions: torch.Tensor | None, rows: int, cols: int) -> torch.Tensor:
    """CSF (codec 0) -> one e4m3 byte per 16-input group, [rows, cols] uint8.

    `fixed` is [rows/16, 16*(1+cols/2)]: per 16-row record, 16 base bytes (one
    per row), then the 4-bit codes of those 16 rows row-major, two per byte with
    the even column in the low nibble; byte = base + code. Each exception word
    then overrides one byte: position = w & 0xFFFFFF (= row*cols + col), value =
    w >> 24. Positions must be strictly increasing.
    """
    if rows % 16 or cols % 2:
        raise ValueError(f"CSF needs rows % 16 == 0 and even cols, got {rows}x{cols}")
    want = (rows // 16, 16 * (1 + cols // 2))
    if tuple(fixed.shape) != want or fixed.dtype != torch.uint8:
        raise ValueError(f"CSF fixed is {fixed.dtype} {tuple(fixed.shape)}, want uint8 {want}")
    bases = fixed[:, :16].reshape(rows).to(torch.int32)
    codes = fixed[:, 16:].reshape(rows, cols // 2)
    nib = torch.stack([codes & 0xF, codes >> 4], dim=-1).reshape(rows, cols).to(torch.int32)
    out = nib + bases[:, None]
    if int(out.max()) > 255:
        raise ValueError("CSF base + code overflows a byte")
    out = out.to(torch.uint8)
    if exceptions is not None and exceptions.numel():
        w = exceptions.reshape(-1).to(torch.int64) & 0xFFFFFFFF
        pos = w & 0xFFFFFF
        if int(pos.max()) >= rows * cols:
            raise ValueError("CSF exception position outside the matrix")
        if pos.numel() > 1 and not bool((pos[1:] > pos[:-1]).all()):
            raise ValueError("CSF exception positions are not strictly increasing")
        out.view(-1)[pos] = (w >> 24).to(torch.uint8)
    return out


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., n/2] -> f32 [..., n]; the low nibble is the earlier element."""
    lut = E2M1.to(packed.device)
    lo = lut[(packed & 0xF).long()]
    hi = lut[(packed >> 4).long()]
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def dequant_nvfp4(packed: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor,
                  what: str) -> torch.Tensor:
    """bf16((e2m1 * e4m3[group of 16]) * global), [out, in]."""
    if packed.dtype != torch.uint8 or packed.dim() != 2:
        raise TypeError(f"{what}: packed weight is {packed.dtype} {tuple(packed.shape)}")
    out, half = packed.shape
    n = half * 2
    if tuple(scales.shape) != (out, n // 16):
        raise ValueError(f"{what}: scales {tuple(scales.shape)} for weight [{out}, {n}]")
    if global_scale.numel() != 1 or global_scale.dtype != torch.float32:
        raise TypeError(f"{what}: global scale is {global_scale.dtype} {tuple(global_scale.shape)}")
    g = global_scale.reshape(())
    if not bool(torch.isfinite(g)) or float(g) <= 0:
        raise ValueError(f"{what}: global scale {float(g)}")
    q = unpack_e2m1(packed).view(out, n // 16, 16)
    s = e4m3_f32(scales, what + " scales")
    w = (q * s[..., None]) * g
    return w.reshape(out, n).to(torch.bfloat16)


def dequant_mxfp8(weight: torch.Tensor, scale: torch.Tensor, what: str) -> torch.Tensor:
    """bf16(e4m3 * 2^(s-127)), one ue8m0 byte per 32 inputs, [out, in]."""
    if weight.dim() != 2:
        raise TypeError(f"{what}: weight {tuple(weight.shape)}")
    out, k = weight.shape
    if k % 32 or tuple(scale.shape) != (out, k // 32) or scale.dtype != torch.uint8:
        raise ValueError(f"{what}: scale {scale.dtype} {tuple(scale.shape)} for weight [{out}, {k}]")
    if bool((scale == 255).any()):
        raise ValueError(f"{what}: NaN ue8m0 scale")
    q = e4m3_f32(weight, what)
    s = UE8M0.to(weight.device)[scale.long()]
    return (q.view(out, k // 32, 32) * s[..., None]).reshape(out, k).to(torch.bfloat16)


# NumPy re-derivations for the self-check (no torch float8, no shared code).

def np_nvfp4_bf16_bits(packed: np.ndarray, scales: np.ndarray, g: float) -> np.ndarray:
    table = T.shiso_e4m3_table()
    out, half = packed.shape
    q = np.empty((out, half * 2), np.float32)
    q[:, 0::2] = E2M1_NP[packed & 15]
    q[:, 1::2] = E2M1_NP[packed >> 4]
    s = table[scales]
    w = (q.reshape(out, -1, 16) * s[:, :, None]).astype(np.float32) * np.float32(g)
    return T.bf16_round_np(w.reshape(out, half * 2).astype(np.float32))


def bf16_bits(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def bf16_bits_to_f32(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32)


# --------------------------------------------------------------------------
# Streamed NVFP4 experts and the NVFP4 n-gram table
# --------------------------------------------------------------------------


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def expert_prefix(layer: int, expert: int, proj: str) -> str:
    return f"{LM}.layers.{layer}.mlp.experts.{expert}.{proj}"


class Nvfp4ExpertStore:
    """Reads one expert's NVFP4 tensors from the shards and dequantizes them."""

    def __init__(self, ck: Checkpoint, device: torch.device, hidden: int, inter: int):
        self.ck = ck
        self.device = device
        self.hidden = hidden
        self.inter = inter
        self.pool = ThreadPoolExecutor(max_workers=6)
        self.loaded = 0
        self.read_seconds = 0.0
        self.wait_seconds = 0.0
        self.dequant_seconds = 0.0

    def read_cpu(self, layer: int, expert: int) -> dict:
        t0 = time.perf_counter()
        out = {}
        for proj in PROJECTIONS:
            p = expert_prefix(layer, expert, proj)
            exc_name = p + ".weight_scale.nvfp4_csf_exceptions"
            out[proj] = (
                self.ck.read(p + ".weight", "U8"),
                self.ck.read(p + ".weight_scale.nvfp4_csf_fixed", "U8"),
                self.ck.read(exc_name, "U32") if self.ck.has(exc_name) else None,
                self.ck.read(p + ".weight_scale_2", "F32"),
            )
        self.read_seconds += time.perf_counter() - t0
        return out

    def dequant(self, parts, what: str) -> torch.Tensor:
        w, fixed, exc, g = (None if t is None else t.to(self.device) for t in parts)
        rows, half = w.shape
        scales = decode_csf(fixed, exc, rows, half * 2 // 16)
        return dequant_nvfp4(w, scales, g, what)

    def to_device(self, cpu: dict, layer: int, expert: int) -> tuple[torch.Tensor, torch.Tensor]:
        mats = {proj: self.dequant(cpu[proj], expert_prefix(layer, expert, proj)) for proj in PROJECTIONS}
        for proj, shape in (("gate_proj", (self.inter, self.hidden)), ("up_proj", (self.inter, self.hidden)),
                            ("down_proj", (self.hidden, self.inter))):
            if tuple(mats[proj].shape) != shape:
                raise ValueError(f"layer {layer} expert {expert} {proj}: {tuple(mats[proj].shape)}, want {shape}")
        self.loaded += 1
        # The library holds the pair fused as gate_up_proj = cat(gate, up) along
        # the output dimension and splits the product with chunk(2).
        return torch.cat([mats["gate_proj"], mats["up_proj"]], dim=0), mats["down_proj"]


class StreamedExperts(nn.Module):
    """Drop-in for the library's experts module: the same eager loop, weights from the shards."""

    PREFETCH = 24

    def __init__(self, store: Nvfp4ExpertStore, layer: int, num_experts: int, act):
        super().__init__()
        self.store = store
        self.layer = layer
        self.num_experts = num_experts
        self.act_fn = act
        self.hit_log: list[int] = []

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # Line for line the library's loop (one_hot over num_experts + 1, the
        # extra class skipped); only where gate_up_proj[e] and down_proj[e]
        # come from differs.
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        hits = [int(e) for e in expert_hit.flatten().tolist() if int(e) != self.num_experts]
        self.hit_log.append(len(hits))
        pending: deque = deque()
        nxt = 0

        def refill():
            nonlocal nxt
            while nxt < len(hits) and len(pending) < self.PREFETCH:
                pending.append(self.store.pool.submit(self.store.read_cpu, self.layer, hits[nxt]))
                nxt += 1

        refill()
        for expert_idx in hits:
            tw = time.perf_counter()
            cpu = pending.popleft().result()
            self.store.wait_seconds += time.perf_counter() - tw
            refill()
            td = time.perf_counter()
            gate_up_proj, down_proj = self.store.to_device(cpu, self.layer, expert_idx)
            self.store.dequant_seconds += time.perf_counter() - td
            del cpu
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, gate_up_proj).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, down_proj)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
            del gate_up_proj, down_proj
        return final_hidden_states


class Nvfp4ShardedEmbedding(nn.Module):
    """The n-gram table as the library's embedding would index it, read from the NVFP4 shards."""

    def __init__(self, ck: Checkpoint, prefix: str, parts: int, dim: int, num_embeddings: int):
        super().__init__()
        self.register_buffer("weight", torch.empty(0), persistent=False)
        if dim % 16:
            raise ValueError(f"n-gram row width {dim} is not a multiple of 16")
        self.dim = dim
        self.codes, self.scales = [], []
        rows_per = None
        for i in range(parts):
            w = ck.memmap(f"{prefix}.shard_{i}.weight", "U8", dim // 2)
            s = ck.memmap(f"{prefix}.shard_{i}.weight_scale", "F8_E4M3", dim // 16)
            if w.shape[0] != s.shape[0]:
                raise ValueError(f"n-gram shard {i}: {w.shape[0]} code rows, {s.shape[0]} scale rows")
            if rows_per is None:
                rows_per = w.shape[0]
            elif w.shape[0] != rows_per:
                raise ValueError(f"n-gram shard {i} has {w.shape[0]} rows, shard 0 has {rows_per}")
            self.codes.append(w)
            self.scales.append(s)
        self.rows_per = rows_per
        if rows_per * parts != num_embeddings:
            raise ValueError(f"n-gram shards hold {rows_per * parts} rows, the library's table "
                             f"has {num_embeddings}")
        self.global_scale = ck.read(f"{prefix}.weight_scale_2", "F32").reshape(1)
        self.rows_gathered = 0

    def gather(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if flat.size and (flat.min() < 0 or flat.max() >= self.rows_per * len(self.codes)):
            raise IndexError("n-gram id outside the table")
        shard, off = flat // self.rows_per, flat % self.rows_per
        codes = np.empty((flat.size, self.dim // 2), np.uint8)
        scales = np.empty((flat.size, self.dim // 16), np.uint8)
        for s in np.unique(shard):
            sel = shard == s
            codes[sel] = self.codes[int(s)][off[sel]]
            scales[sel] = self.scales[int(s)][off[sel]]
        return codes, scales

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).cpu().numpy().astype(np.int64)
        codes, scales = self.gather(flat)
        self.rows_gathered += flat.size
        dev = self.weight.device
        t = dequant_nvfp4(torch.from_numpy(codes).to(dev), torch.from_numpy(scales).to(dev),
                          self.global_scale.to(dev), "n-gram rows")
        return t.view(*ids.shape, self.dim)


# --------------------------------------------------------------------------
# Model assembly
# --------------------------------------------------------------------------


def ckpt_name(name: str) -> str:
    return LM + "." + name[len("model."):] if name.startswith("model.") else name


def build_meta_model(ckpt: Path, layers: int | None):
    transformers, mq, Qwen4ExpConfig = T.hf_modules()
    cfg = Qwen4ExpConfig.from_pretrained(str(ckpt))
    tc = cfg.text_config
    tc._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = mq.Qwen4ExpForCausalLM(tc)
    if layers is not None:
        if not 1 <= layers <= tc.num_hidden_layers:
            raise ValueError(f"--layers {layers} outside 1..{tc.num_hidden_layers}")
        model.model.layers = model.model.layers[:layers]
        tc.num_hidden_layers = layers  # the library's loop runs layers[:num_hidden_layers]
    model = model.to(torch.bfloat16)
    model.eval()
    versions = f"transformers {transformers.__version__}, torch {torch.__version__}"
    return mq, tc, model, versions


def dense_plan(model: nn.Module, ck: Checkpoint, n_layers: int) -> tuple[list, dict]:
    """How each library tensor is filled from the checkpoint, and proof nothing is left over."""
    plan = []
    consumed = set()
    kinds: dict[str, int] = {}
    for name, dst in model.state_dict(keep_vars=True).items():
        ckn = ckpt_name(name)
        if not ck.has(ckn):
            raise KeyError(f"{name} ({ckn}) is not in the checkpoint")
        dtype, shape = ck.dtype_shape(ckn)
        want = tuple(dst.shape)
        if dtype == "F8_E4M3":
            if not ckn.endswith(".weight"):
                raise TypeError(f"{ckn}: e4m3 tensor that is not a weight")
            sname = ckn + "_scale"
            sdt, sshape = ck.dtype_shape(sname)
            if sdt != "U8" or len(shape) != 2 or sshape != [shape[0], shape[1] // 32]:
                raise TypeError(f"{ckn}: MXFP8 scale {sdt} {sshape} for weight {shape}")
            kind = "mxfp8"
            consumed.add(sname)
            ok = dst.dtype == torch.bfloat16 and tuple(shape) == want
        elif dtype == "BF16":
            kind = "bf16"
            ok = dst.dtype == torch.bfloat16 and tuple(shape) == want
        elif dtype == "F32":
            kind = "f32" if dst.dtype == torch.float32 else "f32->bf16"
            ok = dst.dtype in (torch.float32, torch.bfloat16) and tuple(shape) == want
        elif dtype == "I64":
            kind = "i64"
            ok = dst.dtype == torch.int64 and tuple(shape) == want
        else:
            ok, kind = False, dtype
        if not ok:
            raise TypeError(f"{name}: checkpoint {dtype} {shape}, model {dst.dtype} {want}")
        consumed.add(ckn)
        kinds[kind] = kinds.get(kind, 0) + 1
        plan.append((name, ckn, kind))
    # Every language-model tensor must be used by this plan, the streamed
    # experts or the n-gram gather; the activation scales are deliberately
    # unused (A16).
    leftovers = []
    for name in ck.files:
        if name.startswith(("mtp.", "model.visual.")) or name in consumed:
            continue
        parts = name.split(".")
        if name.startswith(LM + ".layers."):
            layer = int(parts[3])
            if layer >= n_layers:
                continue
            if ".mlp.experts." in name and parts[-1] in (
                    "weight", "nvfp4_csf_fixed", "nvfp4_csf_exceptions", "weight_scale_2", "input_scale"):
                continue
            if ".ple.ple_embedding.ngram_embedding." in name:
                continue
        leftovers.append(name)
    if leftovers:
        raise ValueError(f"{len(leftovers)} checkpoint tensors unused by the plan, e.g. {leftovers[:8]}")
    return plan, kinds


class Reference:
    """The library's model with the swapped modules, loaded and ready.

    With `offload`, only the embedding, the head and the final mixer stay
    resident; each decoder layer's dense tensors are materialized in a
    pre-forward hook and dropped after it (the library's own forward still runs
    unchanged). That fits the whole model in a few GB on the CPU.
    """

    def __init__(self, ckpt: Path, device: str, layers: int | None = None, norm_dtype: str = "f32",
                 offload: bool = False):
        self.device = torch.device(device)
        self.offload = offload
        # warn_only: an op without a deterministic kernel logs instead of
        # failing a run that needed the GPU to itself.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        t0 = time.perf_counter()
        self.ck = Checkpoint(ckpt)
        mq, tc, model, self.versions = build_meta_model(ckpt, layers)
        self.mq, self.config = mq, tc
        n_layers = len(model.model.layers)

        # The F32 norms keep an F32 parameter unless asked otherwise.
        self.f32_norms = 0
        for name, mod in model.named_modules():
            if isinstance(mod, mq.Qwen4ExpTextRMSNorm):
                ckn = ckpt_name(name + ".weight")
                dtype, shape = self.ck.dtype_shape(ckn)
                if dtype == "F32" and norm_dtype == "f32":
                    mod.weight = nn.Parameter(torch.empty(shape, dtype=torch.float32, device="meta"),
                                              requires_grad=False)
                    self.f32_norms += 1

        self.store = Nvfp4ExpertStore(self.ck, self.device, tc.hidden_size, tc.moe_intermediate_size)
        self.experts = []
        self.ple = []
        for i, layer in enumerate(model.model.layers):
            ex = StreamedExperts(self.store, i, tc.num_experts, layer.mlp.experts.act_fn)
            layer.mlp.experts = ex
            self.experts.append(ex)
            if layer.ple is not None:
                emb = layer.ple.ple_embedding
                old = emb.ngram_embedding
                emb.ngram_embedding = Nvfp4ShardedEmbedding(
                    self.ck, f"{LM}.layers.{i}.ple.ple_embedding.ngram_embedding",
                    tc.split_ngram_parts, old.embedding_dim, old.num_embeddings)
                self.ple.append((i, emb))
        self.plan, self.kinds = dense_plan(model, self.ck, n_layers)
        self.model = model
        if offload:
            for part in (model.lm_head, model.model.embed_tokens, model.model.hyper_connection_mixer,
                         model.model.rotary_emb):
                part.to_empty(device=self.device)
            self._load_dense(lambda n: not n.startswith("model.layers."))
            for i, layer in enumerate(model.model.layers):
                layer.register_forward_pre_hook(self._materialize)
                layer.register_forward_hook(self._release)
        else:
            model.to_empty(device=self.device)
            self._load_dense()
        self._restore_derived()
        if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
            raise RuntimeError("the head is tied to the embedding; this checkpoint has both")
        self.load_seconds = time.perf_counter() - t0
        alloc = (f", allocated {torch.cuda.memory_allocated() / GIB:.2f} GiB"
                 if self.device.type == "cuda" else "")
        log(f"reference: {n_layers} layers, dense tensors {self.kinds}, F32 norms kept {self.f32_norms}, "
            f"PLE layers {[i for i, _ in self.ple]}, loaded in {self.load_seconds:.1f}s{alloc} "
            f"({self.versions})")

    def _materialize(self, layer, args):
        i = layer.layer_idx_ref
        layer.to_empty(device=self.device)
        self._load_dense(lambda n: n.startswith(f"model.layers.{i}."))
        for j, emb in self.ple:
            if j == i:
                self._check_ngram(j, emb)

    def _release(self, layer, args, out):
        layer.to_empty(device="meta")

    def _load_dense(self, keep=lambda name: True) -> None:
        state = self.model.state_dict(keep_vars=True)
        with torch.no_grad():
            for name, ckn, kind in self.plan:
                if not keep(name):
                    continue
                dst = state[name]
                if kind == "mxfp8":
                    w = self.ck.read(ckn, "F8_E4M3").to(self.device)
                    s = self.ck.read(ckn + "_scale", "U8").to(self.device)
                    dst.copy_(dequant_mxfp8(w, s, ckn))
                    del w, s
                else:
                    t = self.ck.read(ckn).to(self.device)
                    if kind == "f32->bf16":
                        t = t.to(torch.bfloat16)
                    if t.dtype != dst.dtype:
                        raise TypeError(f"{name}: {t.dtype} into {dst.dtype}")
                    dst.copy_(t)

    def _restore_derived(self) -> None:
        mq, tc = self.mq, self.config
        # Non-persistent: recomputed exactly as the library's constructor does.
        rot = self.model.model.rotary_emb
        inv, scaling = rot.compute_default_rope_parameters(tc, self.device)
        rot.inv_freq = inv
        rot.original_inv_freq = inv.clone()
        rot.attention_scaling = scaling
        for i, layer in enumerate(self.model.model.layers):
            layer.layer_idx_ref = i
        if not self.offload:
            for i, emb in self.ple:
                self._check_ngram(i, emb)

    def _check_ngram(self, i: int, emb) -> None:
        # The n-gram buffers came from the checkpoint; they must equal what the
        # library derives from the config, or the hashing is not the library's.
        want = self.mq._build_layer_multipliers(emb.unigram_vocab_size, emb.ngram_size,
                                                emb.ple_layer_index, emb.seed)
        for got, ref, what in (
            (emb.layer_multipliers, want, "layer_multipliers"),
            (emb.ngram_heads_vocab_sizes, torch.tensor(emb.head_vocab_sizes), "vocab sizes"),
            (emb.ngram_heads_offsets, torch.tensor(emb.head_offsets), "offsets"),
        ):
            if not torch.equal(got.cpu(), ref.to(torch.long)):
                raise ValueError(f"layer {i} n-gram {what}: checkpoint and library disagree")

    @torch.no_grad()
    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """fp32 logits of one bf16 hidden row against the bf16 head.

        On the GPU the head is widened once; on the CPU it is widened in
        slices of the vocabulary, so the f32 copy never exists whole.
        """
        head = self.model.lm_head.weight
        if self.device.type == "cuda":
            if getattr(self, "_head32", None) is None:
                self._head32 = head.float()
            return F.linear(h.float(), self._head32)
        return torch.cat([F.linear(h.float(), head[s: s + 16384].float())
                          for s in range(0, head.shape[0], 16384)])

    def sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def hidden(self, ids: list[int], timings: list | None = None) -> torch.Tensor:
        hooks = []
        if timings is not None:
            state = {}

            def pre(mod, args, kwargs):
                self.sync()
                state["t"] = time.perf_counter()

            def post(mod, args, kwargs, out):
                self.sync()
                timings.append((mod._ref_idx, time.perf_counter() - state["t"]))

            for i, layer in enumerate(self.model.model.layers):
                layer._ref_idx = i
                hooks.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
                hooks.append(layer.register_forward_hook(post, with_kwargs=True))
        try:
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            out = self.model.model(input_ids=input_ids, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        self.sync()
        return out.last_hidden_state[0]

    @torch.no_grad()
    def forward(self, ids: list[int], positions: list[int]) -> dict:
        """One teacher-forced pass. Returns fp32 logit rows at `positions`."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        timings: list = []
        t0 = time.perf_counter()
        hidden = self.hidden(ids, timings)
        body = time.perf_counter() - t0
        head = self.model.lm_head.weight
        rows = {p: self.logits(hidden[p]).cpu().numpy() for p in positions}
        agree = 0
        for s in range(0, len(ids) - 1, 512):
            e = min(s + 512, len(ids) - 1)
            am = F.linear(hidden[s:e], head).argmax(-1).cpu().tolist()
            agree += sum(int(a == b) for a, b in zip(am, ids[s + 1: e + 1]))
        return {
            "rows": rows,
            "seconds": body,
            "timings": timings,
            "agreement": agree / max(1, len(ids) - 1),
            "peak_bytes": torch.cuda.max_memory_allocated() if self.device.type == "cuda" else 0,
        }


# --------------------------------------------------------------------------
# Fixture I/O
# --------------------------------------------------------------------------


def read_cases(path: Path) -> list:
    """The case lines of a skeleton or of a recorded fixture (its top lines are ignored)."""
    lines = path.read_text().splitlines()
    if not lines or lines[0].strip() != LOCAL_LOGIT_HEADER:
        sys.exit(f"{path}: not a {LOCAL_LOGIT_HEADER} file")
    cases = []
    for n, line in enumerate(lines[1:], start=2):
        f = line.split()
        if not f or f[0].startswith("#") or f[0] == "top":
            continue
        if f[0] != "case" or len(f) != 7 or f[2] not in ("text", "rendered"):
            sys.exit(f"{path}:{n}: malformed line")
        c = T.Case(*f[1:])
        if c.ctx <= c.frontier or c.frontier < 1:
            sys.exit(f"{path}:{n}: ctx must exceed frontier")
        c.top_count = TOP_COUNT if c.top_count > TOP_COUNT else c.top_count
        cases.append(c)
    if len({c.id for c in cases}) != len(cases):
        sys.exit(f"{path}: duplicate case ids")
    return cases


def logprob_top(row: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Top-k as log-probabilities over the whole row, the arithmetic of `logprob_top_k`."""
    row = row.astype(np.float32)
    finite = row[np.isfinite(row)]
    mx = np.float32(finite.max())
    total = np.exp((finite - mx).astype(np.float64)).sum()
    offset = np.float32(mx + np.float32(np.log(total)))
    return [(t, float(np.float32(v) - offset)) for t, v in T.top_k(row, k)]


def encode(tokenizer, text: str, mode: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=(mode == "rendered")).ids)


def cmd_capture(args) -> None:
    skeleton = Path(args.skeleton).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        sys.exit(f"{out} already exists; a recorded answer is replaced only deliberately")
    if out.parent != skeleton.parent:
        sys.exit("the fixture must sit beside its skeleton: prompt paths are relative to it")
    rows_dir = Path(args.rows_dir).resolve() if args.rows_dir else None
    if rows_dir and (rows_dir / "index.json").exists():
        sys.exit(f"{rows_dir / 'index.json'} already exists")
    cases = read_cases(skeleton)
    if args.cases:
        want = args.cases.split(",")
        cases = [c for c in cases if c.id in want]
        if len(cases) != len(want):
            sys.exit(f"--cases names ids the skeleton does not have: {want}")
    base = skeleton.parent

    from tokenizers import Tokenizer

    tok_path = Path(args.tokenizer or args.checkpoint) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocabulary_digest = T.fnv_digest(tok_path.read_bytes())
    prompt_bytes = b""
    tokens: dict[str, list[int]] = {}
    for c in cases:
        data = (base / c.prompt).read_bytes()
        prompt_bytes += c.id.encode() + data
        ids = encode(tokenizer, data.decode("utf-8"), c.mode)
        if len(ids) < c.frontier:
            sys.exit(f"case {c.id}: prompt has {len(ids)} tokens, frontier {c.frontier}")
        tokens[c.id] = ids[: c.frontier]
    prompts_digest = T.fnv_digest(prompt_bytes)

    guard(min_start(args), gpu=args.device == "cuda")
    ref = Reference(Path(args.checkpoint), args.device, norm_dtype=args.norm_dtype,
                    offload=args.device == "cpu")
    groups: dict[tuple[str, str], list] = {}
    for c in cases:
        groups.setdefault((c.prompt, c.mode), []).append(c)
    results: dict[str, np.ndarray] = {}
    passes = []
    for (prompt, mode), group in groups.items():
        longest = max(group, key=lambda c: c.frontier)
        ids = tokens[longest.id]
        for c in group:
            if tokens[c.id] != ids[: c.frontier]:
                sys.exit(f"case {c.id}: not a prefix of {longest.id}")
        positions = sorted({c.frontier - 1 for c in group})
        log(f"pass: {prompt} ({mode}), {len(ids)} tokens, positions {positions}")
        r = ref.forward(ids, positions)
        log(f"pass: {r['seconds']:.1f}s, peak {r['peak_bytes'] / GIB:.2f} GiB, next-token agreement "
            f"{r['agreement']:.3f}, expert reads {ref.store.read_seconds:.1f}s, waits "
            f"{ref.store.wait_seconds:.1f}s, dequant {ref.store.dequant_seconds:.1f}s (cumulative), "
            f"MemAvailable low {WATCHDOG.min_seen / GIB:.1f} GiB")
        for idx, dt in r["timings"]:
            log(f"  layer {idx:2d}: {dt * 1000:8.1f} ms  experts {ref.experts[idx].hit_log[-1]}")
        passes.append({"prompt": prompt, "tokens": len(ids), "seconds": round(r["seconds"], 1),
                       "agreement": r["agreement"]})
        for c in group:
            results[c.id] = r["rows"][c.frontier - 1]

    ckpt = Path(args.checkpoint)
    headers = {
        "captured_from": "reference",
        "reference": f"modeling_qwen4_exp ({ref.versions}) teacher-forced no-cache sdpa "
                     f"W-dequant bf16 A16",
        "values": "logprob",
        "model_file": args.model_file,
        "vocabulary_digest": vocabulary_digest,
        "prompts_digest": prompts_digest,
        "label": args.label,
        "checkpoint": f"config {T.fnv_digest((ckpt / 'config.json').read_bytes())} index "
                      f"{T.fnv_digest((ckpt / 'model.safetensors.index.json').read_bytes())}",
        "precision": "MXFP8 projections, NVFP4-CSF routed experts and the NVFP4 n-gram table "
                     f"dequantized to bf16; {ref.f32_norms} F32 norms kept "
                     f"{'f32' if args.norm_dtype == 'f32' else 'rounded to bf16'}; bf16 activations; "
                     "torch recurrent paths; fp32 logits from the bf16 final hidden state",
    }
    lines = [LOCAL_LOGIT_HEADER]
    lines += [f"# {k} {v}" for k, v in headers.items()]
    lines += [f"# tokens_digest_{c.id} {T.tokens_digest(tokens[c.id])}" for c in cases]
    lines += [
        "# WHAT THIS FILE IS: the answers the Python library's own modules gave over the "
        "FP4-Local export, every quantized weight dequantized to bf16 and every activation "
        "kept bf16. It is a third implementation beside the reference server and shiso.",
    ]
    for c in cases:
        row = results[c.id]
        if not np.isfinite(row).all():
            sys.exit(f"case {c.id}: non-finite logits")
        tops = logprob_top(row, c.top_count)
        lines.append(f"case {c.id} {c.mode} {c.ctx} {c.frontier} {c.prompt} {len(tops)}")
        lines += [f"top {r} {t} {T.fmt_f32(v)}" for r, (t, v) in enumerate(tops)]
    text = "\n".join(lines) + "\n"
    out.write_text(text)
    log(f"wrote {out} ({len(text)} bytes)")
    if rows_dir:
        rows_dir.mkdir(parents=True, exist_ok=True)
        index = {}
        for c in cases:
            np.save(rows_dir / f"{c.id}.f32.npy", results[c.id].astype("<f4"), allow_pickle=False)
            index[c.id] = {"prompt_file": T.prompt_key(base / c.prompt), "frontier": c.frontier - 1,
                           "vec_frontier": c.frontier, "mode": c.mode,
                           "tokens_digest": T.tokens_digest(tokens[c.id]), "fixture": out.name}
        (rows_dir / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
        log(f"wrote {len(cases)} full fp32 logit rows under {rows_dir}")
    log("summary " + json.dumps({"passes": passes, "experts_loaded": ref.store.loaded,
                                 "bytes_read": ref.ck.bytes_read, "load_seconds": round(ref.load_seconds, 1),
                                 "mem_available_low_gib": round(WATCHDOG.min_seen / GIB, 1)}))


# --------------------------------------------------------------------------
# Greedy continuation (cached decode through the library)
# --------------------------------------------------------------------------


def cmd_greedy(args) -> None:
    from tokenizers import Tokenizer
    from transformers.cache_utils import DynamicCache

    out = Path(args.out).resolve()
    if out.exists():
        sys.exit(f"{out} already exists")
    tokenizer = Tokenizer.from_file(str(Path(args.tokenizer or args.checkpoint) / "tokenizer.json"))
    if args.prompt_file:
        prompt_path, mode = Path(args.prompt_file), "text"
    else:
        skeleton = Path(args.skeleton).resolve()
        first = read_cases(skeleton)[0]
        prompt_path, mode = skeleton.parent / first.prompt, first.mode
    text = prompt_path.read_text()
    ids = encode(tokenizer, text, mode)
    guard(min_start(args), gpu=args.device == "cuda")
    ref = Reference(Path(args.checkpoint), args.device, norm_dtype=args.norm_dtype,
                    offload=args.device == "cpu")
    t0 = time.perf_counter()
    gen: list[int] = []
    with torch.no_grad():
        cache = DynamicCache(config=ref.config)
        nxt_ids = ids
        for _ in range(args.tokens):
            o = ref.model.model(input_ids=torch.tensor([nxt_ids], device=ref.device),
                                past_key_values=cache, use_cache=True)
            # The argmax of the fp32 logits, the same rows the capture records.
            logits = ref.logits(o.last_hidden_state[0, -1])
            nxt = int(logits.argmax())
            gen.append(nxt)
            nxt_ids = [nxt]
    log(f"{prompt_path.name}: {len(ids)} prompt tokens, {len(gen)} greedy ids in "
        f"{time.perf_counter() - t0:.1f}s")
    record = {
        "prompt": text,
        "prompt_tokens": ids,
        "text": tokenizer.decode(gen, skip_special_tokens=False),
        "tokens": gen,
        "reference": f"modeling_qwen4_exp ({ref.versions}) W-dequant bf16 A16, fp32 argmax, "
                     f"{args.label}",
    }
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    log(f"wrote {out}: {record['text']!r}")


# --------------------------------------------------------------------------
# CPU checks
# --------------------------------------------------------------------------


def cmd_selfcheck(args) -> None:
    ok = True

    def verdict(cond: bool, msg: str) -> None:
        nonlocal ok
        ok &= bool(cond)
        log(f"{'ok  ' if cond else 'FAIL'} {msg}")

    guard(args.min_start_gib, gpu=False)
    ck = Checkpoint(Path(args.checkpoint))

    # 1. e4m3: shiso's decode (transcribed) against torch's float8_e4m3fn.
    table = T.shiso_e4m3_table()
    torch_table = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
    verdict(np.array_equal(T.f32_bits(table), T.f32_bits(torch_table)),
            "e4m3: all 256 codes bit-equal to torch float8_e4m3fn")

    # 2. CSF against the fixtures decoded by the reference server's own code.
    gdir = Path(args.csf_golden_dir)
    for proj in ("gate_proj", "down_proj"):
        p = expert_prefix(5, 137, proj)
        fixed = ck.read(p + ".weight_scale.nvfp4_csf_fixed", "U8")
        exc = ck.read(p + ".weight_scale.nvfp4_csf_exceptions", "U32")
        g_fixed = np.load(gdir / f"golden_l5_e137_{proj}_csf_fixed.npy", allow_pickle=False)
        g_exc = np.load(gdir / f"golden_l5_e137_{proj}_csf_exceptions.npy", allow_pickle=False)
        g_scales = np.load(gdir / f"golden_l5_e137_{proj}_scales_e4m3.npy", allow_pickle=False)
        same_in = (np.array_equal(fixed.numpy(), g_fixed)
                   and np.array_equal(exc.numpy().view(np.uint32), g_exc.astype(np.uint32)))
        verdict(same_in, f"CSF l5 e137 {proj}: checkpoint fixed {list(fixed.shape)} + {exc.numel()} "
                         "exceptions byte-equal to the fixture inputs")
        rows, half = ck.dtype_shape(p + ".weight")[1]
        mine = decode_csf(fixed, exc, rows, half * 2 // 16).numpy()
        bad = int((mine != g_scales).sum())
        verdict(bad == 0 and mine.shape == g_scales.shape,
                f"CSF l5 e137 {proj}: decode {list(mine.shape)} byte-exact to the golden scales "
                f"({mine.size - bad}/{mine.size})")

    # 3. NVFP4 experts: torch vs NumPy bit-equal on one expert; magnitude
    #    relation max|W| ~ 6 * 448 * weight_scale_2 on several.
    store = Nvfp4ExpertStore(ck, torch.device("cpu"), 2560, 640)
    cpu = store.read_cpu(5, 137)
    for proj in PROJECTIONS:
        w, fixed, exc, g = cpu[proj]
        rows, half = w.shape
        scales = decode_csf(fixed, exc, rows, half * 2 // 16)
        got = bf16_bits(store.dequant(cpu[proj], proj))
        want = np_nvfp4_bf16_bits(w.numpy(), scales.numpy(), float(g))
        verdict(np.array_equal(got, want), f"NVFP4 l5 e137 {proj}: torch dequant bit-equal to NumPy "
                                           f"({got.size} values)")
    # The export ties gate and up to one weight_scale_2 (they are one fused
    # matrix to the server), so the relation holds for the pair, not for each.
    worst = (math.inf, -math.inf)
    tied = True
    for layer in (0, 1, 5, 23, 30, 47):
        for expert in (0, 137, 300, 511):
            cpu = store.read_cpu(layer, expert)
            amax, gs = {}, {}
            for proj in PROJECTIONS:
                wq = store.dequant(cpu[proj], proj).float()
                gs[proj] = float(cpu[proj][3])
                amax[proj] = float(wq.abs().max())
                if layer in (0, 47) and expert == 0:
                    log(f"     layer {layer:2d} expert {expert:3d} {proj}: max|W| {amax[proj]:.4g}, "
                        f"weight_scale_2 {gs[proj]:.4g}, ratio {amax[proj] / (6 * 448 * gs[proj]):.4f}, "
                        f"std {float(wq.std()):.4g}")
            tied &= gs["gate_proj"] == gs["up_proj"]
            for ratio in (max(amax["gate_proj"], amax["up_proj"]) / (6 * 448 * gs["gate_proj"]),
                          amax["down_proj"] / (6 * 448 * gs["down_proj"])):
                worst = (min(worst[0], ratio), max(worst[1], ratio))
    verdict(tied, "NVFP4: gate and up share one weight_scale_2 in every sampled expert")
    verdict(0.85 <= worst[0] and worst[1] <= 1.0 + 1 / 256,
            f"NVFP4 magnitude: max|W| / (6*448*weight_scale_2) in [{worst[0]:.4f}, {worst[1]:.4f}] "
            "over 24 experts (gate+up pair, down; want ~1)")

    # 4. MXFP8: torch vs NumPy bit-equal, and the bf16 cast exact.
    for name in (f"{LM}.layers.0.linear_attn.in_proj_qkv.weight", f"{LM}.layers.3.self_attn.o_proj.weight",
                 f"{LM}.layers.47.mlp.shared_expert.down_proj.weight"):
        w = ck.read(name, "F8_E4M3")
        s = ck.read(name + "_scale", "U8")
        got = bf16_bits(dequant_mxfp8(w, s, name))
        out_, k = w.shape
        prod = (table[w.view(torch.uint8).numpy()].reshape(out_, k // 32, 32)
                * np.ldexp(np.float32(1.0), s.numpy().astype(np.int32) - 127)[:, :, None].astype(np.float32))
        prod = prod.reshape(out_, k).astype(np.float32)
        want = T.bf16_round_np(prod)
        inexact = int((bf16_bits_to_f32(want) != prod).sum())
        verdict(np.array_equal(got, want) and inexact == 0,
                f"MXFP8 {name.split('layers.')[1]} {list(w.shape)}: torch bit-equal to NumPy, "
                f"{inexact} values inexact in bf16, scale bytes [{int(s.min())}, {int(s.max())}], "
                f"max|W| {float(np.abs(prod).max()):.4g}")

    # 5. The n-gram table: shard geometry against the library, buffers against
    #    the library's derivation, row decode torch vs NumPy, magnitudes.
    mq, tc, model, versions = build_meta_model(Path(args.checkpoint), None)
    log(f"     library: {versions}; layer types {sorted(set(tc.layer_types))}; "
        f"ple_layer_ids {tc.ple_layer_ids} (one-indexed)")
    ple_layers = [i for i, layer in enumerate(model.model.layers) if layer.ple is not None]
    verdict(ple_layers == [1], f"PLE sits on decoder layer(s) {ple_layers} (0-based); checkpoint has layers.1.ple")
    emb_lib = model.model.layers[ple_layers[0]].ple.ple_embedding
    prefix = f"{LM}.layers.{ple_layers[0]}.ple.ple_embedding"
    emb = Nvfp4ShardedEmbedding(ck, prefix + ".ngram_embedding", tc.split_ngram_parts,
                                emb_lib.ngram_embedding.embedding_dim, emb_lib.ngram_embedding.num_embeddings)
    verdict(True, f"n-gram table: {tc.split_ngram_parts} shards x {emb.rows_per} rows = library's "
                  f"{emb_lib.ngram_embedding.num_embeddings} x {emb.dim} (total vocab {emb_lib.total_vocab_size})")
    want_mult = mq._build_layer_multipliers(emb_lib.unigram_vocab_size, emb_lib.ngram_size,
                                            emb_lib.ple_layer_index, emb_lib.seed)
    same = (torch.equal(ck.read(prefix + ".layer_multipliers", "I64"), want_mult)
            and torch.equal(ck.read(prefix + ".ngram_heads_vocab_sizes", "I64"),
                            torch.tensor(emb_lib.head_vocab_sizes))
            and torch.equal(ck.read(prefix + ".ngram_heads_offsets", "I64"), torch.tensor(emb_lib.head_offsets)))
    verdict(same, "n-gram multipliers, head vocab sizes and offsets equal the library's derivation")
    rng = np.random.default_rng(0)
    total = emb.rows_per * len(emb.codes)
    sample = np.concatenate([[0, 1, emb.rows_per - 1, emb.rows_per, total - 1],
                             rng.integers(0, total, 4096)]).astype(np.int64)
    rows_t = emb(torch.from_numpy(sample))
    codes, scales = emb.gather(sample)
    want = np_nvfp4_bf16_bits(codes, scales, float(emb.global_scale[0]))
    g = float(emb.global_scale[0])
    ratio = float(rows_t.float().abs().max()) / (6 * 448 * g)
    verdict(np.array_equal(bf16_bits(rows_t), want) and ratio <= 1.0 + 1 / 256,
            f"n-gram rows: {sample.size} rows torch bit-equal to NumPy; global {g:.4g}, max|row| / "
            f"(6*448*global) {ratio:.4f}, row rms {float(rows_t.float().pow(2).mean().sqrt()):.4g}")
    log(f"     n-gram row 0 first 8: {rows_t[0, :8].float().tolist()}")

    # 6. The whole loading plan on the meta device.
    norm_f32 = sum(1 for n, m in model.named_modules() if isinstance(m, mq.Qwen4ExpTextRMSNorm)
                   and ck.dtype_shape(ckpt_name(n + ".weight"))[0] == "F32")
    for i, layer in enumerate(model.model.layers):
        layer.mlp.experts = nn.Module()
        if layer.ple is not None:
            layer.ple.ple_embedding.ngram_embedding = nn.Module()
    plan, kinds = dense_plan(model, ck, len(model.model.layers))
    verdict(True, f"loading plan: {len(plan)} library tensors {kinds} (F32 norms {norm_f32}), "
                  "every language-model tensor accounted for")
    if not ok:
        sys.exit("selfcheck FAILED")
    log(f"selfcheck ok (MemAvailable low {WATCHDOG.min_seen / GIB:.1f} GiB)")


def cmd_smoke(args) -> None:
    """A forward through the first N layers on the CPU: the loader, the experts and the n-gram path."""
    from tokenizers import Tokenizer

    guard(args.min_start_gib, gpu=False)
    tokenizer = Tokenizer.from_file(str(Path(args.tokenizer or args.checkpoint) / "tokenizer.json"))
    if args.prompt_file:
        text, mode = Path(args.prompt_file).read_text(), "text"
    else:
        skeleton = Path(args.skeleton).resolve()
        first = read_cases(skeleton)[0]
        text, mode = (skeleton.parent / first.prompt).read_text(), first.mode
    ids = encode(tokenizer, text, mode)
    torch.set_num_threads(args.threads)
    ref = Reference(Path(args.checkpoint), "cpu", layers=args.layers, norm_dtype=args.norm_dtype,
                    offload=args.offload)
    timings: list = []
    t0 = time.perf_counter()
    hidden = ref.hidden(ids, timings)
    finite = bool(torch.isfinite(hidden).all())
    logits = ref.logits(hidden[-1])
    top = torch.topk(logits, 5)
    log(f"smoke: {len(ids)} tokens through {args.layers} layers in {time.perf_counter() - t0:.1f}s; "
        f"final hidden finite {finite}, rms {float(hidden.float().pow(2).mean().sqrt()):.4g}; "
        f"experts per layer {[e.hit_log[-1] for e in ref.experts]}; n-gram rows gathered "
        f"{sum(e.ngram_embedding.rows_gathered for _, e in ref.ple)}")
    for idx, dt in timings:
        log(f"  layer {idx:2d} {ref.model.model.layers[idx].layer_type:17s}: {dt * 1000:8.1f} ms")
    log(f"smoke: truncated-model top-5 at the last position (meaningless as an answer): "
        f"{list(zip(top.indices.tolist(), [round(v, 3) for v in top.values.tolist()]))}")
    log(f"smoke: MemAvailable low {WATCHDOG.min_seen / GIB:.1f} GiB")
    if not finite:
        sys.exit("smoke FAILED: non-finite hidden state")


def min_start(args) -> float:
    if args.min_start_gib is not None:
        return args.min_start_gib
    return 40 if args.device == "cuda" else 12


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True, help="the FP4-Local export directory")
    p.add_argument("--tokenizer", help="directory holding tokenizer.json (default: the checkpoint)")
    p.add_argument("--norm-dtype", choices=("f32", "bf16"), default="f32",
                   help="keep the F32-stored norms F32 (default) or round them to bf16")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selfcheck")
    s.add_argument("--csf-golden-dir", default=str(REPO / "test-vectors/references/qwen4exp-csf"))
    s.add_argument("--min-start-gib", type=float, default=6)
    m = sub.add_parser("smoke")
    m.add_argument("--layers", type=int, default=4)
    m.add_argument("--threads", type=int, default=8)
    m.add_argument("--offload", action="store_true", help="materialize one decoder layer at a time")
    m.add_argument("--skeleton", default=str(REPO / "test-vectors/references/qwen38-27b-mlx-f32-golden.vec"))
    m.add_argument("--prompt-file")
    m.add_argument("--min-start-gib", type=float, default=12)
    c = sub.add_parser("capture")
    c.add_argument("--skeleton", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--label", required=True)
    c.add_argument("--model-file", default="qwen3.8-flash")
    c.add_argument("--rows-dir", help="also write each case's full fp32 logit row here")
    c.add_argument("--cases", help="comma-separated case ids to run (default: all)")
    c.add_argument("--device", default="cuda", choices=("cuda", "cpu"),
                   help="cpu streams the decoder layers one at a time (a few GB)")
    c.add_argument("--threads", type=int, default=16)
    c.add_argument("--min-start-gib", type=float)
    g = sub.add_parser("greedy")
    g.add_argument("--skeleton", help="use this fixture's first case's prompt")
    g.add_argument("--prompt-file")
    g.add_argument("--tokens", type=int, default=64)
    g.add_argument("--out", required=True)
    g.add_argument("--label", default="")
    g.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    g.add_argument("--threads", type=int, default=16)
    g.add_argument("--min-start-gib", type=float)
    args = p.parse_args()
    if args.cmd == "greedy" and not (args.skeleton or args.prompt_file):
        p.error("greedy needs --skeleton or --prompt-file")
    if getattr(args, "threads", None):
        torch.set_num_threads(args.threads)
    {"selfcheck": cmd_selfcheck, "smoke": cmd_smoke, "capture": cmd_capture,
     "greedy": cmd_greedy}[args.cmd](args)


if __name__ == "__main__":
    main()
