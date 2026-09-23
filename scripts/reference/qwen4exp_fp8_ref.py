#!/usr/bin/env python3
"""Independent FP8 reference for the qwen4exp (Qwen3.8-Flash-Next) family.

Runs the independent implementation's own modules (the `modeling_qwen4_exp`
file of the Python library installed in the reference venv) over the FP8
checkpoint, teacher-forced, with no KV cache, and records what it answered:

- a `shiso-local-logit-vectors-v1` fixture (`.vec`) that `shiso-bench parity`
  replays, one case per chosen frontier, top-128 raw logits;
- full fp16 logit rows at the chosen frontiers (`<rows>/<case>.f16.npy` plus
  `<rows>/index.json`), for exact KL;
- optionally, per-layer residual rows at the chosen frontiers.

Numerics (W8A16). The checkpoint stores its routed experts and its n-gram table
as e4m3 bytes. Every other tensor is BF16 and is used bit for bit. The experts
are dequantized block by block (`bf16(f32(e4m3) * f32(scale[i//128, j//128]))`,
the same arithmetic the library's own dequantizing loader uses) and multiplied
with BF16 activations. The library's FP8 integration is deliberately not used:
with `activation_scheme: dynamic` it quantizes the activations too (W8A8), which
is further from the BF16 original that the served builds were quantized from.
The n-gram rows are gathered straight from the 128 e4m3 shards and multiplied by
the table's one scalar `weight_scale`; the hashing that picks the rows is the
library's own module, not ours.

Only the routed experts of one layer are ever resident, and each expert is freed
as soon as its tokens are done. The logits at the recorded frontiers are
computed in fp32 from the BF16 final hidden state and the BF16 head, because a
BF16 logit is rounded to 1/8 of a nat at magnitude 20, which is coarser than the
budget the replay grades at.

The fused recurrent-scan kernels are refused on purpose: the optional kernel
package is hidden before the library is imported, so the gated delta rule and
the short convolutions take the library's pure-torch paths. That keeps the
reference free of any tuned kernel and makes it deterministic.

Run with the reference venv, under the GPU lock, e.g.

    ~/work/qwen-3.8-flash/gpu_lock.sh ~/work/exllamav3/.venv/bin/python \\
        scripts/reference/qwen4exp_fp8_ref.py capture \\
        --skeleton test-vectors/references/q4e-fp8-short.skeleton.vec \\
        --out test-vectors/references/q4e-fp8-short.vec \\
        --rows-dir ~/work/qwen-3.8-flash/reference-rows/fp8 --label "..."

Subcommands: `selfcheck` (the B3 codec checks), `capture` (the fixture),
`greedy` (a short cached greedy continuation, read by a person), `probe` (time
the first few layers at long lengths), `check-attention` (the query-blocked
attention against the library's own, bit for bit; see `ChunkedAttention`).
"""

from __future__ import annotations

import argparse
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

# Deterministic cuBLAS needs its workspace pinned before CUDA initializes.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Hide the fused recurrent kernels so the library resolves its torch fallbacks
# (see the module note). Must precede the library import.
sys.modules["fla"] = None  # type: ignore[assignment]
sys.modules["causal_conv1d"] = None  # type: ignore[assignment]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

DEFAULT_CKPT = Path(
    "/home/otsimo/work/sercand/models/Qwen3.8-Flash-Next-FP8/Qwen3.8-Flash-Next-FP8"
)
REPO = Path(__file__).resolve().parents[2]
VOCAB = 248320
HIDDEN = 2560
TOP_COUNT = 128
MIN_MEM_AVAILABLE = 60 * 2**30
MIN_FREE_DISK = 40 * 2**30
ROW_BYTES = VOCAB * 2 + 128  # one .npy row: fp16 payload plus its v1 header
LOCAL_LOGIT_HEADER = "# shiso-local-logit-vectors-v1"
RAW_DTYPES = {"F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Resource guards
# --------------------------------------------------------------------------


def mem_available() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("no MemAvailable in /proc/meminfo")


def guard(dump_bytes: int = 0, budget: int | None = None) -> None:
    """Refuses to start when the box cannot take a run (see the spec's rules)."""
    busy = []
    for name in ("shiso", "ds4", "ds4-server"):
        r = subprocess.run(["pgrep", "-x", name], capture_output=True, text=True)
        if r.stdout.strip():
            busy.append(f"{name} (pid {' '.join(r.stdout.split())})")
    r = subprocess.run(
        ["pgrep", "-f", r"vllm serve|vllm\.entrypoints"], capture_output=True, text=True
    )
    pids = [p for p in r.stdout.split() if int(p) != os.getpid()]
    if pids:
        busy.append(f"vllm (pid {' '.join(pids)})")
    if busy:
        sys.exit(f"refusing: a server holds the GPU: {', '.join(busy)}")
    avail = mem_available()
    if avail < MIN_MEM_AVAILABLE:
        sys.exit(f"refusing: MemAvailable {avail / 2**30:.1f} GiB < 60 GiB")
    free = shutil.disk_usage("/").free
    if free < MIN_FREE_DISK:
        sys.exit(f"refusing: free disk on / {free / 2**30:.1f} GiB < 40 GiB")
    if budget is not None and dump_bytes > budget:
        sys.exit(
            f"refusing: this run would write {dump_bytes} bytes, over the budget of "
            f"{budget} (raise --max-dump-bytes deliberately)"
        )
    if free - dump_bytes < MIN_FREE_DISK:
        sys.exit(
            f"refusing: writing {dump_bytes} bytes would leave less than 40 GiB free on /"
        )
    log(
        f"guard: MemAvailable {avail / 2**30:.1f} GiB, free disk {free / 2**30:.1f} GiB, "
        f"dump {dump_bytes} bytes"
    )


# --------------------------------------------------------------------------
# Digests: the same FNV-1a the replay recomputes (crates/shiso-bench/src/replay.rs)
# --------------------------------------------------------------------------


def fnv_digest(data: bytes) -> str:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def tokens_digest(ids: list[int]) -> str:
    return fnv_digest(b"".join(struct.pack("<I", t) for t in ids))


# --------------------------------------------------------------------------
# e4m3, transcribed from crates/shiso-modelfile/src/fp8.rs::e4m3 (branch for branch)
# --------------------------------------------------------------------------


def shiso_e4m3(byte: int) -> float:
    sign = 1.0 if byte & 0x80 == 0 else -1.0
    exponent = (byte >> 3) & 0x0F
    mantissa = float(byte & 0x07)
    if exponent == 0x0F:
        if byte & 0x07 == 0x07:
            return float("nan")
        return sign * (256.0 * (1.0 + mantissa / 8.0))
    if exponent == 0:
        return sign * (mantissa / 8.0) * 2.0**-6
    return sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def shiso_e4m3_table() -> np.ndarray:
    return np.array([shiso_e4m3(b) for b in range(256)], dtype=np.float32)


def f32_bits(a: np.ndarray) -> np.ndarray:
    bits = a.astype(np.float32).view(np.uint32).copy()
    bits[np.isnan(a)] = 0x7FC00000  # one canonical NaN; both sides have exactly two
    return bits


def bf16_round_np(a: np.ndarray) -> np.ndarray:
    """f32 -> bf16 bit patterns, round to nearest even (finite inputs)."""
    bits = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    bias = ((bits >> 16) & 1) + 0x7FFF
    return ((bits + bias) >> 16).astype(np.uint16)


# --------------------------------------------------------------------------
# Raw safetensors access (header parse; used by the n-gram gather and the checks)
# --------------------------------------------------------------------------


class RawTensors:
    def __init__(self, ckpt: Path):
        self.ckpt = ckpt
        self.weight_map: dict[str, str] = json.loads(
            (ckpt / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self._headers: dict[str, tuple[int, dict]] = {}

    def header(self, fname: str) -> tuple[int, dict]:
        if fname not in self._headers:
            with open(self.ckpt / fname, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._headers[fname] = (8 + n, json.loads(f.read(n)))
        return self._headers[fname]

    def memmap(self, name: str) -> tuple[np.memmap, str, list[int]]:
        fname = self.weight_map[name]
        base, hdr = self.header(fname)
        info = hdr[name]
        start, end = info["data_offsets"]
        mm = np.memmap(self.ckpt / fname, dtype=np.uint8, mode="r", offset=base + start,
                       shape=(end - start,))
        return mm, info["dtype"], info["shape"]


# --------------------------------------------------------------------------
# Block dequantization
# --------------------------------------------------------------------------


def dequant_block(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """bf16(f32(e4m3 w) * f32(s[i//bn, j//bk])), block size from the scale grid."""
    n, k = w.shape
    sn, sk = s.shape
    if n % sn or k % sk:
        raise ValueError(f"weight {tuple(w.shape)} not divisible by scale grid {tuple(s.shape)}")
    bn, bk = n // sn, k // sk
    q = w.to(torch.float32).view(sn, bn, sk, bk)
    return (q * s.to(torch.float32)[:, None, :, None]).to(torch.bfloat16).view(n, k)


class ExpertStore:
    """Reads one expert's e4m3 tensors from the mapped shards and dequantizes them."""

    def __init__(self, ckpt: Path, raw: "RawTensors", device: torch.device):
        self.ckpt = ckpt
        self.raw = raw
        self.weight_map = raw.weight_map
        self._fds: dict[str, int] = {}
        self._fd_lock = threading.Lock()
        self.device = device
        self._local = threading.local()
        self.pool = ThreadPoolExecutor(max_workers=6)
        self.read_seconds = 0.0
        self.bytes_read = 0
        self.loaded = 0
        self.wait_seconds = 0.0
        self.upload_seconds = 0.0

    def _handle(self, fname: str):
        from safetensors import safe_open

        cache = getattr(self._local, "cache", None)
        if cache is None:
            cache = self._local.cache = {}
        if fname not in cache:
            cache[fname] = safe_open(str(self.ckpt / fname), framework="pt", device="cpu")
        return cache[fname]

    @staticmethod
    def names(layer: int, expert: int):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight", "weight_scale_inv"):
                yield (f"{proj}.{suffix}",
                       f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.{suffix}")

    def _fd(self, fname: str) -> int:
        with self._fd_lock:
            if fname not in self._fds:
                self._fds[fname] = os.open(self.ckpt / fname, os.O_RDONLY)
            return self._fds[fname]

    def read_cpu(self, layer: int, expert: int) -> dict[str, torch.Tensor]:
        """One expert's six tensors, read with pread into owned host memory.

        Not through the library's mapped tensors: those fault the file in one
        page at a time on first touch, which measured ~150 MB/s here (34 ms per
        expert), while a whole-tensor pread from a worker thread runs at NVMe
        speed and releases the interpreter lock while it waits.
        """
        t0 = time.perf_counter()
        out = {}
        for key, name in self.names(layer, expert):
            fname = self.weight_map[name]
            base, hdr = self.raw.header(fname)
            info = hdr[name]
            start, end = info["data_offsets"]
            fd = self._fd(fname)
            buf = bytearray(end - start)
            view = memoryview(buf)
            got = 0
            while got < len(buf):
                n = os.preadv(fd, [view[got:]], base + start + got)
                if n <= 0:
                    raise IOError(f"short read of {name} in {fname}")
                got += n
            t = torch.frombuffer(buf, dtype=torch.uint8).view(RAW_DTYPES[info["dtype"]])
            out[key] = t.reshape(info["shape"])
            self.bytes_read += len(buf)
        self.read_seconds += time.perf_counter() - t0
        return out

    def read_cpu_safetensors(self, layer: int, expert: int) -> dict[str, torch.Tensor]:
        """The same tensors through the library's own reader, for the self-check."""
        return {key: self._handle(self.weight_map[name]).get_tensor(name).clone()
                for key, name in self.names(layer, expert)}

    def to_device(self, cpu: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        dev = {k: v.to(self.device) for k, v in cpu.items()}
        if dev["gate_proj.weight"].dtype != torch.float8_e4m3fn:
            raise TypeError(f"expert weight is {dev['gate_proj.weight'].dtype}, want e4m3")
        gate = dequant_block(dev["gate_proj.weight"], dev["gate_proj.weight_scale_inv"])
        up = dequant_block(dev["up_proj.weight"], dev["up_proj.weight_scale_inv"])
        down = dequant_block(dev["down_proj.weight"], dev["down_proj.weight_scale_inv"])
        self.loaded += 1
        # The library stores the pair fused as gate_up_proj = cat(gate, up) along
        # the output dimension and splits the product with chunk(2); keep that shape.
        return torch.cat([gate, up], dim=0), down


class StreamedExperts(nn.Module):
    """Drop-in for the library's experts module: same loop, weights from the shards."""

    PREFETCH = 24

    def __init__(self, store: ExpertStore, layer: int, num_experts: int, act):
        super().__init__()
        self.store = store
        self.layer = layer
        self.num_experts = num_experts
        self.act_fn = act
        self.hit_log: list[int] = []

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # The body mirrors the library's eager expert loop line for line; only
        # where gate_up_proj[e] and down_proj[e] come from differs.
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        hits = [int(e) for e in expert_hit.flatten().tolist()]
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
            tu = time.perf_counter()
            gate_up_proj, down_proj = self.store.to_device(cpu)
            self.store.upload_seconds += time.perf_counter() - tu
            del cpu
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, gate_up_proj).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, down_proj)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )
            del gate_up_proj, down_proj
        return final_hidden_states


class Fp8ShardedEmbedding(nn.Module):
    """The n-gram table as the library's module would index it, read from the e4m3 shards.

    Row r lives in shard r // rows_per_shard at row r % rows_per_shard: the
    library concatenates the shards along dim 0 when it loads them.
    """

    def __init__(self, raw: RawTensors, prefix: str, parts: int, dim: int):
        super().__init__()
        self.register_buffer("weight", torch.empty(0), persistent=False)
        self.shards = []
        rows_per = None
        for i in range(parts):
            mm, dtype, shape = raw.memmap(f"{prefix}.shard_{i}.weight")
            if dtype != "F8_E4M3" or shape[1] != dim:
                raise TypeError(f"n-gram shard {i}: {dtype} {shape}")
            if rows_per is None:
                rows_per = shape[0]
            elif shape[0] != rows_per:
                raise ValueError(f"n-gram shard {i} has {shape[0]} rows, shard 0 has {rows_per}")
            self.shards.append(mm.reshape(shape[0], dim))
        self.rows_per = rows_per
        self.dim = dim
        mm, dtype, shape = raw.memmap(f"{prefix}.weight_scale")
        if dtype != "BF16" or shape != [1]:
            raise TypeError(f"n-gram weight_scale: {dtype} {shape}")
        bits = np.frombuffer(bytes(mm), dtype=np.uint16).astype(np.uint32) << 16
        self.scale = float(bits.view(np.float32)[0])
        self.rows_gathered = 0

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).cpu().numpy().astype(np.int64)
        if flat.min() < 0 or flat.max() >= self.rows_per * len(self.shards):
            raise IndexError("n-gram id outside the table")
        shard = flat // self.rows_per
        off = flat % self.rows_per
        out = np.empty((flat.size, self.dim), dtype=np.uint8)
        for s in np.unique(shard):
            sel = shard == s
            out[sel] = self.shards[int(s)][off[sel]]
        self.rows_gathered += flat.size
        dev = self.weight.device
        t = torch.from_numpy(out).to(dev).view(torch.float8_e4m3fn).to(torch.float32)
        t = (t * torch.tensor(self.scale, dtype=torch.float32, device=dev)).to(torch.bfloat16)
        return t.view(*ids.shape, self.dim)


# --------------------------------------------------------------------------
# Long-context attention: the library's arithmetic, without a [T, T] anything
# --------------------------------------------------------------------------


class CausalSentinel:
    """Stands in for the full-attention mask the library would build.

    The library's model builds one `[1, 1, T, T]` bool causal mask per pass,
    and each full-attention layer's indexer builds a second one of the same
    size (plus a `[T, T+1]` scatter target) before the two are combined and
    handed to sdpa. At 32k tokens each of those is a GiB, and sdpa's bool-to-
    bias conversion doubles it. `ChunkedAttention` rebuilds the same mask one
    block of queries at a time, so the model is handed this marker instead,
    which also proves no library code consumed the mask on the way.
    """

    def __init__(self, length: int):
        self.length = length


class ChunkedAttention:
    """The library's full-attention forward and QSA indexer, one query block at a time.

    Every tensor that feeds the arithmetic is computed by the library's own
    modules in the library's own order. What changes is only the iteration:

    - The indexer's per-query loop re-pools every complete key block for every
      query. With a plain causal mask (one sequence, no padding, no cache) the
      visible tokens of query i are exactly 0..i, so its blocks are the first
      (i+1)//r blocks of the same fixed partition, and the pooled, normed and
      rotated block keys are one tensor shared by all queries. It is computed
      once, with the same ops the loop ran (index_select, fp32 mean, bf16
      cast, k_layernorm, rope at each block's first token).
    - The scores of a block of queries are one matmul instead of one per query,
      then the same transpose, relu, sum over heads and 1/sqrt(d). Only the
      queries with more visible blocks than the budget are scored at all (the
      rest keep every block, as the library's topk(min(budget, n)) does), and
      their top-k is still one 1-D call per query on the same slice the library
      passes, because tied scores break by slice shape (see selection_block).
    - The mask (causal AND selected) exists for one query block at a time, and
      sdpa runs over that block of queries against all keys.

    `check_attention` holds this to the library's original, bit for bit, on
    real hidden states; see `cmd_check_attention`.

    sdpa is pinned to the memory-efficient kernel. That kernel takes an
    arbitrary mask and never forms the score matrix; the math kernel would form
    24 x T x T fp32 scores (96 GiB at 32k), and the flash kernel takes no mask
    at all. If the efficient kernel cannot run these inputs torch raises rather
    than silently falling back, which is the point.
    """

    # Only the CPU unit test of the selection logic overrides this (CPU has no
    # efficient kernel); every capture runs with it.
    SDPA_BACKEND = "EFFICIENT_ATTENTION"

    def __init__(self, mq, attn: nn.Module, query_block: int, check: bool = False):
        self.mq = mq
        self.attn = attn
        self.query_block = query_block
        self.check = check
        self.check_log: list[dict] = []
        self.library_forward = attn.forward
        self.tied_rows = 0
        self.over_budget_rows = 0

    def install(self) -> None:
        self.attn.forward = self.forward

    # -- the indexer, vectorized over queries ------------------------------

    def indexer_inputs(self, hidden_states, position_embeddings):
        """q, raw keys and the rotated block keys, exactly as the library's loop forms them."""
        mq = self.mq
        idx = self.attn.indexer
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, -1, idx.index_head_dim)
        full_cos, full_sin = position_embeddings
        current_cos, current_sin = full_cos[:, -seq_length:, :], full_sin[:, -seq_length:, :]
        qk = idx.index_qk_proj(hidden_states)
        q, token_k = torch.split(
            qk,
            [idx.index_n_heads * idx.index_head_dim, idx.index_kv_heads * idx.index_head_dim],
            dim=-1,
        )
        q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
        q = idx.q_layernorm(q)
        q = mq.apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)

        r = idx.compress_ratio
        nb = seq_length // r
        if nb == 0:
            return q, None, nb
        block_token_indices = torch.arange(nb * r, device=hidden_states.device).view(nb, r)
        key_groups = raw_keys[0].index_select(0, block_token_indices.flatten())
        key_groups = key_groups.view(*block_token_indices.shape, idx.index_head_dim)
        pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)
        pooled_keys = idx.k_layernorm(pooled_keys)
        group_starts = block_token_indices[:, 0]
        block_key_states = mq.apply_rotary_pos_emb(
            pooled_keys.unsqueeze(1),
            cos=full_cos[0].index_select(0, group_starts),
            sin=full_sin[0].index_select(0, group_starts),
        ).squeeze(1)
        return q, block_key_states, nb

    def selection_block(self, q, block_key_states, nb: int, s: int, e: int, kv_length: int):
        """The indexer's selection for queries s..e-1 as a [e-s, kv_length] bool mask.

        Row i-s is True exactly at the tokens the library's loop puts in
        `selected_token_indices[0, i]`: the chosen complete blocks plus the tail
        of incomplete-block tokens. It is not yet ANDed with causality (the
        library does that when it combines the two masks), though every
        selected token is visible anyway.
        """
        idx = self.attn.indexer
        r = idx.compress_ratio
        dev = q.device
        qpos = torch.arange(s, e, device=dev)
        nb_i = (qpos + 1) // r  # complete blocks query i can see
        sel_blocks = None
        if nb > 0:
            blk = torch.arange(nb, device=dev)
            visible_blocks = blk[None, :] < nb_i[:, None]
            # A query with no more visible blocks than the budget keeps them all
            # (the library's topk(min(block_topk, n)) is then a permutation).
            sel_blocks = visible_blocks.clone()
            over = nb_i > idx.block_topk
            if bool(over.any()):
                rows = torch.nonzero(over).flatten()
                qq = q[0, s:e].index_select(0, rows).float()  # [n, H, D]
                scores = torch.matmul(qq, block_key_states.float().transpose(-1, -2)).transpose(-1, -2)
                scores = torch.relu(scores).sum(dim=-1) / math.sqrt(idx.index_head_dim)  # [n, nb]
                # The top-k runs per query on that query's own contiguous 1-D
                # vector of visible-block scores, exactly the call the library
                # makes. Relu sums tie exactly (at 0.0 above all), and which
                # tied block survives the cut is the top-k kernel's choice,
                # which torch makes differently for a padded [n, nb] batch
                # (measured: the sets differ on tied rows). So only the scores
                # are batched; the selection is not.
                chosen = torch.zeros((rows.numel(), nb), dtype=torch.bool, device=dev)
                nb_rows = nb_i.index_select(0, rows).tolist()
                for j, n in enumerate(nb_rows):
                    top = scores[j, :n].topk(idx.block_topk, dim=0).indices
                    chosen[j].index_fill_(0, top, True)
                sel_blocks[rows] = chosen
                # How many rows had a tie straddling the cut: there the
                # selection is the kernel's arbitrary pick among equals, and a
                # second implementation may legitimately pick others.
                padded = scores.masked_fill(~visible_blocks.index_select(0, rows), float("-inf"))
                kk = padded.topk(idx.block_topk + 1, dim=-1).values
                self.tied_rows += int((kk[:, -2] == kk[:, -1]).sum())
                self.over_budget_rows += rows.numel()
        tok = torch.arange(kv_length, device=dev)
        mask = torch.zeros((e - s, kv_length), dtype=torch.bool, device=dev)
        if sel_blocks is not None:
            mask[:, : nb * r] = sel_blocks.repeat_interleave(r, dim=1)
        tail = (tok[None, :] >= (nb_i * r)[:, None]) & (tok[None, :] <= qpos[:, None])
        return mask | tail

    # -- the attention forward ---------------------------------------------

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                **kwargs):
        mq = self.mq
        attn = self.attn
        if past_key_values is not None:
            # A cached decode (the `greedy` subcommand) is short; it keeps the
            # library's own path, mask and all.
            if isinstance(attention_mask, CausalSentinel):
                raise RuntimeError("a cached pass needs the library's own mask")
            return self.library_forward(hidden_states, position_embeddings, attention_mask,
                                        past_key_values, **kwargs)
        if hidden_states.shape[0] != 1:
            raise RuntimeError("the chunked attention runs one sequence at a time")
        seq_length = hidden_states.shape[1]
        if not isinstance(attention_mask, CausalSentinel) or attention_mask.length != seq_length:
            raise RuntimeError("the chunked attention expects the causal sentinel for this pass")

        full_position_embeddings = position_embeddings
        q_idx, block_key_states, nb = self.indexer_inputs(hidden_states, position_embeddings)

        # From here to the sdpa call: the library's forward, line for line.
        position_embeddings = tuple(x[:, -seq_length:, :] for x in position_embeddings)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        query_states, gate = torch.chunk(
            attn.q_proj(hidden_states).view(*input_shape, -1, attn.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = mq.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # The library's sdpa wrapper repeats the kv heads itself whenever a mask
        # is passed (enable_gqa would force the math kernel); do the same.
        key_states = mq.repeat_kv(key_states, attn.num_key_value_groups)
        value_states = mq.repeat_kv(value_states, attn.num_key_value_groups)

        from torch.nn.attention import SDPBackend, sdpa_kernel

        kv_length = key_states.shape[2]
        tok = torch.arange(kv_length, device=hidden_states.device)
        outs = []
        for s in range(0, seq_length, self.query_block):
            e = min(seq_length, s + self.query_block)
            causal = tok[None, :] <= torch.arange(s, e, device=tok.device)[:, None]
            mask = (causal & self.selection_block(q_idx, block_key_states, nb, s, e, kv_length))
            with sdpa_kernel([getattr(SDPBackend, self.SDPA_BACKEND)]):
                o = F.scaled_dot_product_attention(
                    query_states[:, :, s:e],
                    key_states,
                    value_states,
                    attn_mask=mask[None, None],
                    dropout_p=0.0,
                    scale=attn.scaling,
                    is_causal=False,
                )
            outs.append(o.transpose(1, 2))
        attn_output = torch.cat(outs, dim=1).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = attn.o_proj(attn_output)

        if self.check:
            self.compare_with_library(hidden_states, full_position_embeddings, q_idx, block_key_states,
                                      nb, attn_output, **kwargs)
        return attn_output, None

    # -- the equality check against the library's own path -----------------

    def compare_with_library(self, hidden_states, position_embeddings, q_idx, block_key_states, nb,
                             attn_output, **kwargs):
        """Runs the library's indexer and attention on the same input and records exact equality."""
        seq_length = hidden_states.shape[1]
        dev = hidden_states.device
        causal = torch.ones((seq_length, seq_length), dtype=torch.bool, device=dev).tril()[None, None]
        lib_sel = self.attn.indexer(hidden_states, position_embeddings, causal, None)  # [1,1,T,T]
        ours = torch.cat([
            self.selection_block(q_idx, block_key_states, nb, s, min(seq_length, s + self.query_block),
                                 seq_length)
            for s in range(0, seq_length, self.query_block)
        ])[None, None]
        # The library's own forward: builds its combined mask and calls its sdpa
        # wrapper (whatever kernel torch picks for it, as in the committed runs).
        lib_out, _ = self.library_forward(hidden_states, position_embeddings, attention_mask=causal,
                                          past_key_values=None, **kwargs)
        sel_equal = bool(torch.equal(lib_sel, ours))
        out_equal = bool(torch.equal(lib_out, attn_output))
        entry = {
            "layer": self.attn.layer_idx,
            "tokens": seq_length,
            "selection_equal": sel_equal,
            "selection_mismatches": int((lib_sel != ours).sum()),
            "output_equal": out_equal,
            "output_max_abs_diff": float((lib_out.float() - attn_output.float()).abs().max()),
            "rows_over_budget": max(0, seq_length - (self.attn.indexer.block_topk + 1)
                                    * self.attn.indexer.compress_ratio + 1),
        }
        self.check_log.append(entry)
        log(f"check layer {entry['layer']:2d} T={seq_length}: selection "
            f"{'EQUAL' if sel_equal else 'DIFFERENT'} ({entry['selection_mismatches']} cells), "
            f"output {'EQUAL' if out_equal else 'DIFFERENT'} (max |d| {entry['output_max_abs_diff']:.3e})")
        del lib_sel, ours, lib_out, causal


# --------------------------------------------------------------------------
# Model assembly
# --------------------------------------------------------------------------


def hf_modules():
    import transformers
    from transformers.models.qwen4_exp import modeling_qwen4_exp as mq
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig

    # The fallbacks must have been taken; a kernel implementation would have
    # replaced these module-level functions at import.
    for fn in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn"):
        wrapped = getattr(mq, fn)
        closure = [c.cell_contents for c in (wrapped.__closure__ or []) if callable(c.cell_contents)]
        mods = {getattr(c, "__module__", "") for c in closure}
        if any(m.startswith("fla") or m.startswith("causal_conv1d") for m in mods):
            raise RuntimeError(f"{fn} resolved to a fused kernel ({mods}); the torch path is required")
    return transformers, mq, Qwen4ExpConfig


class Reference:
    def __init__(self, ckpt: Path, device: str = "cuda", query_block: int = 1024,
                 check_attention: bool = False):
        transformers, mq, Qwen4ExpConfig = hf_modules()
        self.mq = mq
        self.versions = f"transformers {transformers.__version__}, torch {torch.__version__}"
        self.ckpt = ckpt
        self.device = torch.device(device)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        cfg = Qwen4ExpConfig.from_pretrained(str(ckpt))
        tc = cfg.text_config
        tc._attn_implementation = "sdpa"
        self.config = tc
        self.raw = RawTensors(ckpt)
        t0 = time.perf_counter()
        with torch.device("meta"):
            model = mq.Qwen4ExpForCausalLM(tc)
        model = model.to(torch.bfloat16)
        model.eval()

        self.store = ExpertStore(ckpt, self.raw, self.device)
        self.experts = []
        for i, layer in enumerate(model.model.layers):
            ex = StreamedExperts(self.store, i, tc.num_experts, layer.mlp.experts.act_fn)
            layer.mlp.experts = ex
            self.experts.append(ex)
            if layer.ple is not None:
                emb = layer.ple.ple_embedding
                prefix = f"model.language_model.layers.{i}.ple.ple_embedding.ngram_embedding"
                emb.ngram_embedding = Fp8ShardedEmbedding(
                    self.raw, prefix, tc.split_ngram_parts, emb.ngram_embedding.embedding_dim
                )
        model.to_empty(device=self.device)
        self.model = model
        if query_block < 1:
            raise ValueError(f"query block {query_block} must be positive")
        self.attention = []
        # Every softmax-attention layer (the library calls their type
        # "qwen_sparse_attention"; the recurrent ones have no self_attn).
        for layer in model.model.layers:
            if hasattr(layer, "self_attn"):
                ca = ChunkedAttention(mq, layer.self_attn, query_block, check=check_attention)
                ca.install()
                self.attention.append(ca)
        n_softmax = sum(t != "linear_attention" for t in tc.layer_types)
        if len(self.attention) != n_softmax:
            raise RuntimeError(f"installed the chunked attention on {len(self.attention)} layers, "
                               f"the config has {n_softmax} softmax-attention layers")
        self._load_dense()
        self._restore_derived()
        self.load_seconds = time.perf_counter() - t0
        log(
            f"reference: dense weights on {self.device} in {self.load_seconds:.1f}s, "
            f"allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB ({self.versions})"
        )

    def _ckpt_name(self, name: str) -> str:
        if name.startswith("model."):
            return "model.language_model." + name[len("model."):]
        return name

    def _load_dense(self) -> None:
        from safetensors import safe_open

        state = self.model.state_dict(keep_vars=True)
        by_file: dict[str, list[str]] = {}
        for name in state:
            ck = self._ckpt_name(name)
            if ck not in self.raw.weight_map:
                raise KeyError(f"{name} ({ck}) is not in the checkpoint")
            by_file.setdefault(self.raw.weight_map[ck], []).append(name)
        n = 0
        for fname, names in sorted(by_file.items()):
            with safe_open(str(self.ckpt / fname), framework="pt", device="cpu") as f:
                for name in names:
                    t = f.get_tensor(self._ckpt_name(name))
                    dst = state[name]
                    if t.dtype != dst.dtype or tuple(t.shape) != tuple(dst.shape):
                        raise TypeError(
                            f"{name}: checkpoint {t.dtype} {tuple(t.shape)}, "
                            f"model {dst.dtype} {tuple(dst.shape)}"
                        )
                    with torch.no_grad():
                        dst.copy_(t.to(self.device))
                    n += 1
        self.dense_tensors = n

    def _restore_derived(self) -> None:
        mq = self.mq
        tc = self.config
        # Non-persistent: recompute exactly as the library's constructor does, in fp32.
        rot = self.model.model.rotary_emb
        inv, scaling = rot.compute_default_rope_parameters(tc, self.device)
        rot.inv_freq = inv
        rot.original_inv_freq = inv.clone()
        rot.attention_scaling = scaling
        # The n-gram buffers came from the checkpoint; they must equal what the
        # library derives from the config, or the hashing is not the library's.
        for layer in self.model.model.layers:
            if layer.ple is None:
                continue
            emb = layer.ple.ple_embedding
            want = mq._build_layer_multipliers(
                emb.unigram_vocab_size, emb.ngram_size, emb.ple_layer_index, emb.seed
            )
            for got, ref, what in (
                (emb.layer_multipliers, want, "layer_multipliers"),
                (emb.ngram_heads_vocab_sizes, torch.tensor(emb.head_vocab_sizes), "vocab sizes"),
                (emb.ngram_heads_offsets, torch.tensor(emb.head_offsets), "offsets"),
            ):
                if not torch.equal(got.cpu(), ref.to(torch.long)):
                    raise ValueError(f"n-gram {what}: checkpoint and library disagree")

    def _layer_dump_hooks(self, positions: list[int], layer_rows: dict) -> list:
        """Hooks that copy the residual streams at `positions` at every layer boundary.

        The names and the points are those of the engine's residual tap
        (`scripts/reference/LAYER_DUMP.md`, `crates/shiso-bench/src/layers.rs`):

        - `embed`: the input of layer 0, the embedding repeated across the streams;
        - `L{nn}.mixer`: the input of layer nn's feed-forward hyper-connection, which
          is the residual after the token mixer's combine (and after the n-gram add,
          on the layer that carries it, as in the engine);
        - `L{nn}`: layer nn's output.

        Each value is `[hc_count, hidden]` f32, stream-major, which is how the
        library's flat `hc_count * hidden` residual unflattens.
        """
        tc = self.config
        layers = list(self.model.model.layers[: tc.num_hidden_layers])
        width = max(2, len(str(max(1, len(layers) - 1))))
        streams = tc.hc_count

        def grab(name: str, x: torch.Tensor) -> None:
            for p in positions:
                layer_rows[p][name] = x[0, p].float().reshape(streams, -1).cpu().numpy()

        def first(args, kwargs, key):
            return args[0] if args else kwargs[key]

        hooks = [layers[0].register_forward_pre_hook(
            lambda mod, args, kwargs: grab("embed", first(args, kwargs, "hidden_states")),
            with_kwargs=True)]
        for i, layer in enumerate(layers):
            tag = f"L{i:0{width}d}"
            hooks.append(layer.mlp_hyper_connection.register_forward_pre_hook(
                lambda mod, args, kwargs, tag=tag: grab(
                    f"{tag}.mixer", first(args, kwargs, "hyper_input")),
                with_kwargs=True))
            hooks.append(layer.register_forward_hook(
                lambda mod, args, out, tag=tag: grab(tag, out)))
        return hooks

    @torch.no_grad()
    def forward(self, ids: list[int], positions: list[int], dump_layers: bool = False,
                per_layer_timing: bool = True) -> dict:
        """One teacher-forced pass. Returns fp32 logit rows at `positions`."""
        model = self.model
        timings: list[tuple[int, float, float, int]] = []
        hooks = []
        layer_rows: dict[int, dict[str, np.ndarray]] = {p: {} for p in positions} if dump_layers else {}
        state = {}

        if per_layer_timing:
            def pre(mod, args, kwargs, idx=None):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                state["t"] = time.perf_counter()
                state["read"] = self.store.read_seconds

            def post(mod, args, kwargs, out, idx=None):
                torch.cuda.synchronize()
                timings.append((
                    mod._ref_idx,
                    time.perf_counter() - state["t"],
                    self.store.read_seconds - state["read"],
                    torch.cuda.max_memory_allocated(),
                ))

            for i, layer in enumerate(model.model.layers):
                layer._ref_idx = i
                hooks.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
                hooks.append(layer.register_forward_hook(post, with_kwargs=True))
        if dump_layers:
            hooks += self._layer_dump_hooks(positions, layer_rows)

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        t0 = time.perf_counter()
        try:
            # The dict skips the library's [T, T] causal mask; the chunked
            # attention rebuilds it per query block. None is what the library's
            # recurrent-mask helper returns for one unpadded sequence.
            masks = {"full_attention": CausalSentinel(len(ids)), "linear_attention": None}
            out = model.model(input_ids=input_ids, attention_mask=masks, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        hidden = out.last_hidden_state[0]  # [T, HIDDEN] bf16
        torch.cuda.synchronize()
        body = time.perf_counter() - t0

        head = model.lm_head.weight
        head32 = head.float()
        rows = {}
        for p in positions:
            rows[p] = F.linear(hidden[p].float(), head32).cpu().numpy()
        # A teacher-forced sanity number: how often the argmax is the next prompt token.
        agree = 0
        for s in range(0, len(ids) - 1, 512):
            e = min(s + 512, len(ids) - 1)
            am = F.linear(hidden[s:e], head).argmax(-1).cpu().tolist()
            agree += sum(int(a == b) for a, b in zip(am, ids[s + 1 : e + 1]))
        del head32
        result = {
            "rows": rows,
            "seconds": body,
            "timings": timings,
            "next_token_agreement": agree / max(1, len(ids) - 1),
            "peak_bytes": torch.cuda.max_memory_allocated(),
        }
        if dump_layers:
            result["layer_rows"] = layer_rows
        return result


# --------------------------------------------------------------------------
# Fixture I/O
# --------------------------------------------------------------------------


class Case:
    def __init__(self, cid, mode, ctx, frontier, prompt, top_count):
        self.id = cid
        self.mode = mode
        self.ctx = int(ctx)
        self.frontier = int(frontier)
        self.prompt = prompt
        self.top_count = int(top_count)


def read_skeleton(path: Path) -> list[Case]:
    lines = path.read_text().splitlines()
    if not lines or lines[0].strip() != LOCAL_LOGIT_HEADER:
        sys.exit(f"{path}: not a {LOCAL_LOGIT_HEADER} file")
    cases = []
    for n, line in enumerate(lines[1:], start=2):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if f[0] == "case":
            if len(f) != 7 or f[2] not in ("text", "rendered"):
                sys.exit(f"{path}:{n}: malformed case line")
            c = Case(*f[1:])
            if c.ctx <= c.frontier or c.frontier < 1 or not 1 <= c.top_count <= 128:
                sys.exit(f"{path}:{n}: ctx must exceed frontier, top-count in 1..128")
            cases.append(c)
        elif f[0] == "top":
            sys.exit(f"{path}:{n}: a skeleton carries no recorded entries")
        else:
            sys.exit(f"{path}:{n}: unknown keyword {f[0]!r}")
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        sys.exit(f"{path}: duplicate case ids")
    return cases


def fmt_f32(x: float) -> str:
    # The shortest decimal that reads back as the same f32, positional, like the
    # Rust writer's Display.
    return np.format_float_positional(np.float32(x), unique=True, trim="-")


def top_k(row: np.ndarray, k: int) -> list[tuple[int, float]]:
    # Highest logit first; equal logits by ascending id so the order is total.
    order = np.lexsort((np.arange(row.size), -row.astype(np.float64)))[:k]
    return [(int(i), float(row[i])) for i in order]


def prompt_key(path: Path) -> str:
    path = path.resolve()
    root = (REPO / "test-vectors").resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def cmd_capture(args) -> None:
    skeleton = Path(args.skeleton).resolve()
    out = Path(args.out).resolve()
    rows_dir = Path(args.rows_dir).expanduser().resolve() if args.rows_dir else None
    if out.exists():
        sys.exit(f"{out} already exists; a recorded answer is replaced only deliberately")
    cases = read_skeleton(skeleton)
    base = skeleton.parent
    if out.parent != base:
        sys.exit("the fixture must sit beside its skeleton: prompt paths are relative to it")

    from tokenizers import Tokenizer

    tok_path = Path(args.tokenizer).expanduser() / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocabulary_digest = fnv_digest(tok_path.read_bytes())
    prompt_bytes = b""
    tokens: dict[str, list[int]] = {}
    full_len: dict[str, int] = {}
    for c in cases:
        data = (base / c.prompt).read_bytes()
        prompt_bytes += c.id.encode() + data
        enc = tokenizer.encode(data.decode("utf-8"), add_special_tokens=(c.mode == "rendered"))
        ids = list(enc.ids)
        if len(ids) < c.frontier:
            sys.exit(f"case {c.id}: prompt has {len(ids)} tokens, frontier {c.frontier}")
        full_len[c.id] = len(ids)
        tokens[c.id] = ids[: c.frontier]
    prompts_digest = fnv_digest(prompt_bytes)

    index_path = rows_dir / "index.json" if rows_dir else None
    index = {}
    if rows_dir:
        if index_path.exists():
            index = json.loads(index_path.read_text())
        for c in cases:
            if (rows_dir / f"{c.id}.f16.npy").exists() or c.id in index:
                sys.exit(f"row for case {c.id} already exists in {rows_dir}")
    layer_dir = Path(args.layer_dump_dir).expanduser().resolve() if args.layer_dump_dir else None
    if layer_dir:
        # The engine's layout (LAYER_DUMP.md): refused up front, before the load.
        if (layer_dir / "index.json").exists():
            sys.exit(f"{layer_dir / 'index.json'} already exists; a layer dump is replaced only deliberately")
        for c in cases:
            if (layer_dir / c.id).exists():
                sys.exit(f"layers for case {c.id} already exist in {layer_dir}")
    dump = len(cases) * TOP_COUNT * 40 + 8192
    if rows_dir:
        dump += len(cases) * ROW_BYTES + 4096 * len(cases)
    if layer_dir:
        # embed, then a mixer and a layer file per layer, each [4, 2560] f32.
        n_layers = 48
        dump += len(cases) * ((1 + 2 * n_layers) * (4 * HIDDEN * 4 + 128) + 4096)
    guard(dump, args.max_dump_bytes)

    ref = Reference(Path(args.checkpoint), query_block=args.query_block)
    # One pass per distinct token prefix: cases on the same prompt and mode
    # share the pass that reaches their furthest frontier.
    groups: dict[tuple[str, str], list[Case]] = {}
    for c in cases:
        groups.setdefault((c.prompt, c.mode), []).append(c)
    results: dict[str, np.ndarray] = {}
    layer_rows: dict[str, list[np.ndarray]] = {}
    passes = []
    for (prompt, mode), group in groups.items():
        longest = max(group, key=lambda c: c.frontier)
        ids = tokens[longest.id]
        for c in group:
            if tokens[c.id] != ids[: c.frontier]:
                sys.exit(f"case {c.id}: not a prefix of {longest.id}")
        positions = sorted({c.frontier - 1 for c in group})
        log(f"pass: {prompt} ({mode}), {len(ids)} tokens, positions {positions}")
        tied0 = sum(ca.tied_rows for ca in ref.attention)
        over0 = sum(ca.over_budget_rows for ca in ref.attention)
        r = ref.forward(ids, positions, dump_layers=layer_dir is not None)
        log(f"pass: indexer rows over budget {sum(ca.over_budget_rows for ca in ref.attention) - over0}, "
            f"with a tie straddling the top-k cut {sum(ca.tied_rows for ca in ref.attention) - tied0}")
        tot_read = sum(t[2] for t in r["timings"])
        log(
            f"pass: {r['seconds']:.1f}s (expert reads {tot_read:.1f}s wall in workers), "
            f"peak {r['peak_bytes'] / 2**30:.2f} GiB, next-token agreement "
            f"{r['next_token_agreement']:.3f}"
        )
        for idx, dt, rd, pk in r["timings"]:
            log(f"  layer {idx:2d}: {dt * 1000:8.1f} ms  peak {pk / 2**30:6.2f} GiB  "
                f"experts {ref.experts[idx].hit_log[-1]}")
        passes.append({"prompt": prompt, "tokens": len(ids), "seconds": r["seconds"],
                       "agreement": r["next_token_agreement"]})
        for c in group:
            results[c.id] = r["rows"][c.frontier - 1]
            if layer_dir:
                layer_rows[c.id] = r["layer_rows"][c.frontier - 1]

    # The fixture.
    headers = {
        "captured_from": "reference",
        "checkpoint": f"{Path(args.checkpoint).name} config {fnv_digest((Path(args.checkpoint) / 'config.json').read_bytes())}",
        "label": args.label,
        "model_file": Path(args.checkpoint).name,
        "precision": "W8A16: e4m3 128x128-block experts and the e4m3 n-gram table dequantized "
                     "to bf16 (f32 product); bf16 activations; sdpa; torch recurrent paths; "
                     "fp32 logits from the bf16 final hidden state",
        "prompts_digest": prompts_digest,
        "reference": f"modeling_qwen4_exp ({ref.versions}), teacher-forced, no cache",
        "values": "logit",
        "vocabulary_digest": vocabulary_digest,
    }
    for c in cases:
        headers[f"tokens_digest_{c.id}"] = tokens_digest(tokens[c.id])
    notes = [
        "WHAT THIS FILE IS: the answers an independent implementation of this family gave "
        "over the FP8 checkpoint on this machine. Unlike a fixture this port captured of "
        "itself, it can show that this port is wrong rather than only that it changed.",
        "IT IS NOT THE BF16 ORIGINAL: the routed experts and the n-gram table are e4m3 in "
        "this checkpoint, so the reference is W8A16. Every other tensor is the original BF16.",
    ]
    lines = [LOCAL_LOGIT_HEADER]
    lines += [f"# {k} {v}" for k, v in sorted(headers.items())]
    lines += [f"# {n}" for n in notes]
    for c in cases:
        row = results[c.id]
        if not np.isfinite(row).all():
            sys.exit(f"case {c.id}: non-finite logits")
        tops = top_k(row, c.top_count)
        lines.append(f"case {c.id} {c.mode} {c.ctx} {c.frontier} {c.prompt} {len(tops)}")
        lines += [f"top {r} {t} {fmt_f32(v)}" for r, (t, v) in enumerate(tops)]
    text = "\n".join(lines) + "\n"
    out.write_text(text)
    log(f"wrote {out} ({len(text)} bytes)")

    if rows_dir:
        rows_dir.mkdir(parents=True, exist_ok=True)
        for c in cases:
            np.save(rows_dir / f"{c.id}.f16.npy", results[c.id].astype("<f2"), allow_pickle=False)
            index[c.id] = {
                # Shared with the comparison tool: the prompt path relative to
                # test-vectors/, the 0-based position of the row's token, and
                # how many tokens the WHOLE prompt file encodes to.
                "prompt_file": prompt_key(base / c.prompt),
                "frontier": c.frontier - 1,
                "prompt_tokens": full_len[c.id],
                "vec_frontier": c.frontier,
                "mode": c.mode,
                "tokens_digest": tokens_digest(tokens[c.id]),
                "fixture": out.name,
            }
        index_path.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
        log(f"wrote {len(cases)} rows and {index_path}")
    if layer_dir:
        # <dir>/<case>/{embed,L{nn}.mixer,L{nn}}.f32.npy, each [4, 2560] <f4, plus
        # index.json in the row dump's schema (frontier is the 0-based position).
        layer_dir.mkdir(parents=True, exist_ok=True)
        layer_index = {}
        for c in cases:
            case_dir = layer_dir / c.id
            case_dir.mkdir()
            for name, v in layer_rows[c.id].items():
                arr = np.ascontiguousarray(v, dtype="<f4")
                with open(case_dir / f"{name}.f32.npy", "xb") as f:
                    np.save(f, arr, allow_pickle=False)
                rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
                log(f"{c.id} {name}: shape={list(arr.shape)} rms={rms:.6e} "
                    f"absmax={float(np.abs(arr).max()):.6e} "
                    f"non_finite={int((~np.isfinite(arr)).sum())}")
            layer_index[c.id] = {
                "prompt_file": prompt_key(base / c.prompt),
                "frontier": c.frontier - 1,
                "prompt_tokens": full_len[c.id],
                "vec_frontier": c.frontier,
                "mode": c.mode,
                "tokens_digest": tokens_digest(tokens[c.id]),
                "files": len(layer_rows[c.id]),
            }
        with open(layer_dir / "index.json", "x") as f:
            f.write(json.dumps(layer_index, indent=1, sort_keys=True) + "\n")
        log(f"wrote layers for {len(cases)} cases under {layer_dir}")
    summary = {"passes": passes, "experts_loaded": ref.store.loaded,
               "expert_bytes_read": ref.store.bytes_read, "load_seconds": ref.load_seconds}
    log("summary " + json.dumps(summary))


# --------------------------------------------------------------------------
# The chunked attention against the library's own path
# --------------------------------------------------------------------------


def topk_tie_check(device: torch.device) -> bool:
    """Batched top-k over -inf-padded rows picks the same set as the library's 1-D calls.

    The indexer's scores are relu sums, so exact ties (most often at 0.0) are
    real, and which tied block survives the cut is the top-k kernel's choice.
    The library calls it once per query on a 1-D vector of that query's
    visible blocks; the chunked path calls it on a padded [queries, blocks]
    tensor. Kernel selection in torch depends on slice size and slice count,
    so this is checked on the shapes a 32k pass produces, with heavy ties.
    """
    k = 512
    ok = True
    gen = torch.Generator(device="cpu").manual_seed(0)
    for nb_max in (513, 600, 2048, 4096, 8192):
        rows = 64
        nb_i = torch.linspace(k + 1, nb_max, rows).round().long()
        # Scores on a coarse grid, a third of them exactly zero: many ties.
        vals = torch.randint(0, 8, (rows, nb_max), generator=gen).float() * 0.25
        vals[torch.rand((rows, nb_max), generator=gen) < 0.33] = 0.0
        vals = vals.to(device)
        padded = vals.clone()
        for r in range(rows):
            padded[r, int(nb_i[r]):] = float("-inf")
        batched = padded.topk(k, dim=-1).indices
        bad = 0
        for r in range(rows):
            one = vals[r, : int(nb_i[r])].topk(min(k, int(nb_i[r])), dim=0).indices
            a = torch.zeros(nb_max, dtype=torch.bool, device=device)
            b = torch.zeros(nb_max, dtype=torch.bool, device=device)
            a[one] = True
            b[batched[r]] = True
            bad += int(not torch.equal(a, b))
        log(f"topk ties, blocks up to {nb_max}: {rows - bad}/{rows} rows select the same set")
        ok &= bad == 0
    return ok


def cmd_check_attention(args) -> None:
    """Holds the chunked attention to the library's original, bit for bit, on real inputs.

    Every full-attention layer of each pass runs both paths on the same hidden
    states and compares the indexer's selection mask and the attention output
    with torch.equal. The pass itself continues with the chunked output.
    """
    from tokenizers import Tokenizer

    guard()
    tokenizer = Tokenizer.from_file(str(Path(args.tokenizer).expanduser() / "tokenizer.json"))
    ids_all = list(tokenizer.encode(Path(args.prompt_file).read_text(),
                                    add_special_tokens=False).ids)
    lengths = [int(x) for x in args.lengths.split(",")]
    if max(lengths) > len(ids_all):
        sys.exit(f"{args.prompt_file} encodes to {len(ids_all)} tokens, fewer than {max(lengths)}")
    ok = topk_tie_check(torch.device("cuda"))
    ref = Reference(Path(args.checkpoint), query_block=args.query_block, check_attention=True)
    for n in lengths:
        r = ref.forward(ids_all[:n], [n - 1], per_layer_timing=False)
        log(f"pass T={n}: {r['seconds']:.1f}s, peak {r['peak_bytes'] / 2**30:.2f} GiB")
    entries = [e for ca in ref.attention for e in ca.check_log]
    bad = [e for e in entries if not (e["selection_equal"] and e["output_equal"])]
    log(f"check-attention: {len(entries) - len(bad)}/{len(entries)} layer passes equal "
        f"(lengths {lengths}, query block {args.query_block})")
    if bad or not ok:
        sys.exit("check-attention FAILED")
    log("check-attention ok")


def cmd_probe(args) -> None:
    """Times the first few decoder layers at long lengths, to size a full pass before running one."""
    from tokenizers import Tokenizer

    guard()
    tokenizer = Tokenizer.from_file(str(Path(args.tokenizer).expanduser() / "tokenizer.json"))
    ids_all = list(tokenizer.encode(Path(args.prompt_file).read_text(),
                                    add_special_tokens=False).ids)
    lengths = [int(x) for x in args.lengths.split(",")]
    if max(lengths) > len(ids_all):
        sys.exit(f"{args.prompt_file} encodes to {len(ids_all)} tokens, fewer than {max(lengths)}")
    ref = Reference(Path(args.checkpoint), query_block=args.query_block)
    cfg = ref.model.model.config
    full = cfg.num_hidden_layers
    cfg.num_hidden_layers = args.layers  # the library's loop runs layers[:num_hidden_layers]
    try:
        for n in lengths:
            r = ref.forward(ids_all[:n], [n - 1])
            log(f"probe T={n}: {args.layers} layers in {r['seconds']:.1f}s, "
                f"peak {r['peak_bytes'] / 2**30:.2f} GiB")
            for idx, dt, rd, pk in r["timings"]:
                kind = ref.model.model.layers[idx].layer_type
                log(f"  layer {idx:2d} {kind:17s}: {dt:7.1f} s  peak {pk / 2**30:6.2f} GiB  "
                    f"experts {ref.experts[idx].hit_log[-1]}  expert reads {rd:.1f} s")
    finally:
        cfg.num_hidden_layers = full


# --------------------------------------------------------------------------
# B3 self-checks
# --------------------------------------------------------------------------


def cmd_selfcheck(args) -> None:
    ok = True
    # 1. e4m3: shiso's decode (transcribed) against torch's float8_e4m3fn, all 256 codes.
    table = shiso_e4m3_table()
    torch_table = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
    a, b = f32_bits(table), f32_bits(torch_table)
    mism = np.nonzero(a != b)[0]
    log(f"e4m3: {256 - mism.size}/256 codes bit-equal to torch float8_e4m3fn; "
        f"table digest {fnv_digest(a.astype('<u4').tobytes())}")
    if mism.size:
        ok = False
        log(f"e4m3 mismatches at {mism.tolist()}")

    # 2. Block dequant of one expert matrix on the device against NumPy from raw bytes.
    guard()
    raw = RawTensors(Path(args.checkpoint))
    store = ExpertStore(Path(args.checkpoint), raw, torch.device("cuda"))
    layer, expert = args.layer, args.expert
    cpu = store.read_cpu(layer, expert)
    lib = store.read_cpu_safetensors(layer, expert)
    for key in cpu:
        same = cpu[key].dtype == lib[key].dtype and torch.equal(
            cpu[key].view(torch.uint8), lib[key].view(torch.uint8))
        log(f"pread vs library reader, {key}: {'identical' if same else 'DIFFERENT'}")
        ok &= same
    for proj in ("gate_proj", "up_proj", "down_proj"):
        name = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"
        wmm, wdt, wshape = raw.memmap(name + ".weight")
        smm, sdt, sshape = raw.memmap(name + ".weight_scale_inv")
        assert wdt == "F8_E4M3" and sdt == "BF16", (wdt, sdt)
        w = table[np.asarray(wmm)].reshape(wshape)
        s = (np.frombuffer(bytes(smm), dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        s = s.reshape(sshape)
        bn, bk = wshape[0] // sshape[0], wshape[1] // sshape[1]
        prod = (w.reshape(sshape[0], bn, sshape[1], bk) * s[:, None, :, None]).reshape(wshape)
        want = bf16_round_np(prod.astype(np.float32))
        got = dequant_block(cpu[f"{proj}.weight"].cuda(), cpu[f"{proj}.weight_scale_inv"].cuda())
        got = got.view(torch.int16).cpu().numpy().view(np.uint16)
        bad = int((got != want).sum())
        log(f"dequant layer {layer} expert {expert} {proj} {wshape} scale {sshape}: "
            f"{got.size - bad}/{got.size} bf16 bit-equal to NumPy")
        ok &= bad == 0
    if not ok:
        sys.exit("selfcheck FAILED")
    log("selfcheck ok")


# --------------------------------------------------------------------------
# Greedy continuation (read by a person; cached decode through the library)
# --------------------------------------------------------------------------


def cmd_greedy(args) -> None:
    from tokenizers import Tokenizer

    guard()
    tokenizer = Tokenizer.from_file(str(Path(args.tokenizer).expanduser() / "tokenizer.json"))
    ref = Reference(Path(args.checkpoint))
    for pf in args.prompt_file:
        text = Path(pf).read_text()
        ids = list(tokenizer.encode(text, add_special_tokens=False).ids)
        t0 = time.perf_counter()
        from transformers.cache_utils import DynamicCache

        with torch.no_grad():
            cache = DynamicCache(config=ref.config)
            out = ref.model(input_ids=torch.tensor([ids], device=ref.device), past_key_values=cache,
                            use_cache=True, logits_to_keep=1)
            gen = []
            for _ in range(args.tokens):
                nxt = int(out.logits[0, -1].float().argmax())
                gen.append(nxt)
                out = ref.model(input_ids=torch.tensor([[nxt]], device=ref.device),
                                past_key_values=cache, use_cache=True, logits_to_keep=1)
        log(f"{pf}: {len(ids)} prompt tokens, {time.perf_counter() - t0:.1f}s")
        log(f"  greedy ids {gen}")
        log(f"  continuation {tokenizer.decode(gen)!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p.add_argument("--tokenizer", default=str(DEFAULT_CKPT),
                   help="directory holding tokenizer.json (its digest goes in the fixture)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selfcheck")
    s.add_argument("--layer", type=int, default=0)
    s.add_argument("--expert", type=int, default=0)
    c = sub.add_parser("capture")
    c.add_argument("--skeleton", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--label", required=True)
    c.add_argument("--rows-dir")
    c.add_argument("--layer-dump-dir",
                   help="also write each case's residual streams at every layer boundary here, "
                        "in the engine's layout: <case>/{embed,L{nn}.mixer,L{nn}}.f32.npy "
                        "(see scripts/reference/LAYER_DUMP.md)")
    c.add_argument("--max-dump-bytes", type=int, default=64 * 2**20)
    c.add_argument("--query-block", type=int, default=1024,
                   help="queries per attention block; the result does not depend on it")
    pr = sub.add_parser("probe")
    pr.add_argument("--prompt-file", required=True)
    pr.add_argument("--lengths", default="8192,16384,32768")
    pr.add_argument("--layers", type=int, default=4,
                    help="run only the first N decoder layers (the timing probe)")
    pr.add_argument("--query-block", type=int, default=1024)
    k = sub.add_parser("check-attention")
    k.add_argument("--prompt-file", required=True)
    k.add_argument("--lengths", default="7,1000,2052,2376",
                   help="comma-separated prefix lengths, one pass each")
    k.add_argument("--query-block", type=int, default=1024)
    g = sub.add_parser("greedy")
    g.add_argument("--tokens", type=int, default=24)
    g.add_argument("prompt_file", nargs="+")
    args = p.parse_args()
    {"selfcheck": cmd_selfcheck, "capture": cmd_capture, "greedy": cmd_greedy,
     "check-attention": cmd_check_attention, "probe": cmd_probe}[args.cmd](args)


if __name__ == "__main__":
    main()
