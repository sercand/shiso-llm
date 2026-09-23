#!/usr/bin/env python3
"""Build a qwen4exp draft head whose dense projections are BF16.

The shipped draft head GGUF stores every 2-D weight as a trellis (EXL3) tensor.
This script copies it and swaps each quantized NON-expert matrix - attention
q/k/v/o, the indexer's q/k projections, the shared expert, and the two MTP
fc projections - for the BF16 original from the FP8 checkpoint, whose `mtp.*`
dense tensors are unquantized BF16. The routed expert stacks stay trellis, and
every other tensor (norms, routers, hyper-connection mixes) and every metadata
key is copied byte for byte. The question it answers is whether draft-side
quantization error is what holds MTP acceptance down.

Mapping (from the converter that produced the draft GGUF): no permutation is
applied to any of these roles. A trellis tensor's GGUF dims are [in, out]; the
HF weight is [out, in] row-major, which is exactly GGUF dims [in, out] for a
plain type, so BF16 bytes are copied verbatim. The one split is the indexer's
fused `index_qk_proj` [640, 2560]: rows [0, 512) are q (4 heads x 128) and rows
[512, 640) are k (1 head x 128), the order the converter slices it in.

Stdlib + numpy only; safetensors are read by parsing their JSON header.

  python make_hybrid_mtp.py --list <draft.gguf>
  python make_hybrid_mtp.py <draft.gguf> <fp8 checkpoint dir> <out.gguf>
"""
import argparse
import json
import os
import shutil
import struct
import sys

GGUF_MAGIC = b"GGUF"
G_F32, G_F16, G_BF16 = 0, 1, 30
EXL3 = {68: "exl3_k4", 69: "exl3_k5", 70: "exl3_k6"}
TYPE_NAME = {G_F32: "f32", G_F16: "f16", G_BF16: "bf16", **EXL3}

# GGUF value types: size of the fixed-width ones.
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
T_STR, T_ARR = 8, 9

# GGUF tensor -> (HF tensor, row slice or None).
REPLACE = {
    "blk.0.attn_q.weight": ("mtp.layers.0.self_attn.q_proj.weight", None),
    "blk.0.attn_k.weight": ("mtp.layers.0.self_attn.k_proj.weight", None),
    "blk.0.attn_v.weight": ("mtp.layers.0.self_attn.v_proj.weight", None),
    "blk.0.attn_output.weight": ("mtp.layers.0.self_attn.o_proj.weight", None),
    "blk.0.indexer.q_proj.weight": ("mtp.layers.0.self_attn.indexer.index_qk_proj.weight", (0, 512)),
    "blk.0.indexer.k_proj.weight": ("mtp.layers.0.self_attn.indexer.index_qk_proj.weight", (512, 640)),
    "blk.0.ffn_gate_shexp.weight": ("mtp.layers.0.mlp.shared_expert.gate_proj.weight", None),
    "blk.0.ffn_up_shexp.weight": ("mtp.layers.0.mlp.shared_expert.up_proj.weight", None),
    "blk.0.ffn_down_shexp.weight": ("mtp.layers.0.mlp.shared_expert.down_proj.weight", None),
    "mtp.fc_embedding.weight": ("mtp.fc_embedding.weight", None),
    "mtp.fc_hidden.weight": ("mtp.fc_hidden.weight", None),
}


def fail(msg):
    sys.stderr.write(f"error: {msg}\n")
    raise SystemExit(1)


class Reader:
    def __init__(self, f):
        self.f = f

    def take(self, n):
        b = self.f.read(n)
        if len(b) != n:
            fail("truncated GGUF header")
        return b

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf-8")

    def skip_value(self, vtype):
        """Returns the value for scalars and strings, skips arrays."""
        if vtype == T_STR:
            return self.string()
        if vtype == T_ARR:
            etype, n = self.u32(), self.u64()
            if etype == T_STR:
                for _ in range(n):
                    self.string()
            else:
                self.take(FIXED[etype] * n)
            return None
        raw = self.take(FIXED[vtype])
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
               10: "<Q", 11: "<q", 12: "<d"}[vtype]
        return struct.unpack(fmt, raw)[0]


def read_gguf(path):
    """Header of a GGUF: (kv bytes, kvs, tensors, data start, alignment)."""
    with open(path, "rb") as f:
        r = Reader(f)
        if r.take(4) != GGUF_MAGIC:
            fail(f"{path} is not a GGUF")
        version = r.u32()
        if version != 3:
            fail(f"GGUF version {version}, expected 3")
        n_tensors, n_kv = r.u64(), r.u64()
        kv_start = f.tell()
        kvs = {}
        for _ in range(n_kv):
            key = r.string()
            kvs[key] = r.skip_value(r.u32())
        kv_bytes_end = f.tell()
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.u64() for _ in range(r.u32())]
            tensors.append({"name": name, "dims": dims, "type": r.u32(), "offset": r.u64()})
        info_end = f.tell()
        f.seek(kv_start)
        kv_raw = f.read(kv_bytes_end - kv_start)
    align = int(kvs.get("general.alignment", 32))
    data_start = (info_end + align - 1) // align * align
    size = os.path.getsize(path)
    ordered = sorted(tensors, key=lambda t: t["offset"])
    for i, t in enumerate(ordered):
        end = ordered[i + 1]["offset"] if i + 1 < len(ordered) else size - data_start
        t["span"] = end - t["offset"]
    return {"version": version, "kv_raw": kv_raw, "n_kv": n_kv, "kvs": kvs,
            "tensors": tensors, "data_start": data_start, "align": align}


def elements(dims):
    n = 1
    for d in dims:
        n *= d
    return n


class SafetensorsDir:
    """Tensor lookup across a sharded safetensors checkpoint, header-only parse."""

    def __init__(self, d):
        self.dir = d
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.headers = {}

    def info(self, name):
        if name not in self.map:
            fail(f"{name} is not in the checkpoint")
        shard = self.map[name]
        if shard not in self.headers:
            with open(os.path.join(self.dir, shard), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.headers[shard] = (8 + n, json.loads(f.read(n)))
        base, header = self.headers[shard]
        meta = header[name]
        lo, hi = meta["data_offsets"]
        return shard, meta["dtype"], meta["shape"], base + lo, hi - lo

    def rows(self, name, row_slice):
        """Raw little-endian bytes of a BF16 2-D tensor, optionally a row range."""
        shard, dtype, shape, off, nbytes = self.info(name)
        if dtype != "BF16" or len(shape) != 2:
            fail(f"{name} is {dtype} {shape}, expected a 2-D BF16 tensor")
        if name.replace(".weight", ".weight_scale_inv") in self.map:
            fail(f"{name} carries a block scale: it is quantized, not BF16")
        rows, cols = shape
        lo, hi = (0, rows) if row_slice is None else row_slice
        if not 0 <= lo < hi <= rows:
            fail(f"row slice {row_slice} outside {name} {shape}")
        with open(os.path.join(self.dir, shard), "rb") as f:
            f.seek(off + lo * cols * 2)
            data = f.read((hi - lo) * cols * 2)
        return [hi - lo, cols], data


def list_gguf(path):
    g = read_gguf(path)
    print(f"{path}: {len(g['tensors'])} tensors, {g['n_kv']} kv, data at {g['data_start']}")
    for k, v in g["kvs"].items():
        if v is not None:
            print(f"  kv {k} = {v}")
    for t in g["tensors"]:
        print(f"  {t['name']:40s} {TYPE_NAME.get(t['type'], t['type']):8s} {t['dims']} "
              f"off={t['offset']} bytes={t['span']}")


def build(src, ckpt_dir, out):
    if os.path.exists(out):
        fail(f"{out} exists; refusing to overwrite")
    g = read_gguf(src)
    ck = SafetensorsDir(ckpt_dir)
    names = {t["name"] for t in g["tensors"]}
    missing = set(REPLACE) - names
    if missing:
        fail(f"draft GGUF lacks {sorted(missing)}")

    # Every quantized non-expert matrix must be in REPLACE, so none is left
    # trellis by an oversight in the table.
    for t in g["tensors"]:
        if t["type"] in EXL3 and "_exps" not in t["name"] and t["name"] not in REPLACE:
            fail(f"{t['name']} is trellis and not an expert, but has no BF16 source")

    plan = []
    for t in g["tensors"]:
        if t["name"] in REPLACE:
            hf, sl = REPLACE[t["name"]]
            shape = ck.info(hf)[2]
            rows = shape[0] if sl is None else sl[1] - sl[0]
            dims = [shape[1], rows]
            if dims != t["dims"]:
                fail(f"{t['name']} dims {t['dims']} but {hf}{'' if sl is None else sl} gives {dims}")
            plan.append({**t, "type": G_BF16, "nbytes": elements(dims) * 2, "hf": (hf, sl)})
        else:
            plan.append({**t, "nbytes": t["span"], "hf": None})

    # Offsets in the source's order, each aligned; untouched tensors keep their
    # exact span (payload plus the padding the source wrote after it).
    align = g["align"]
    order = sorted(plan, key=lambda t: t["offset"])
    cursor = 0
    for t in order:
        t["new_offset"] = cursor
        cursor += (t["nbytes"] + align - 1) // align * align if t["hf"] else t["nbytes"]

    need = cursor + g["data_start"] + (64 << 20)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(out))).free
    if free - need < 40 << 30:
        fail(f"writing {need / 1e9:.2f} GB would leave under 40 GB free ({free / 1e9:.1f} GB now)")

    header = bytearray(GGUF_MAGIC)
    header += struct.pack("<IQQ", g["version"], len(plan), g["n_kv"])
    header += g["kv_raw"]
    for t in plan:  # directory in the source's order
        b = t["name"].encode("utf-8")
        header += struct.pack("<Q", len(b)) + b + struct.pack("<I", len(t["dims"]))
        header += b"".join(struct.pack("<Q", d) for d in t["dims"])
        header += struct.pack("<IQ", t["type"], t["new_offset"])
    pad = (-len(header)) % align
    header += b"\0" * pad

    tmp = out + ".partial"
    with open(src, "rb") as fin, open(tmp, "xb") as fout:
        fout.write(header)
        for t in order:
            if fout.tell() != len(header) + t["new_offset"]:
                fail(f"layout drift at {t['name']}")
            if t["hf"]:
                _, data = ck.rows(*t["hf"])
                if len(data) != t["nbytes"]:
                    fail(f"{t['name']}: read {len(data)} bytes, expected {t['nbytes']}")
                fout.write(data)
                fout.write(b"\0" * ((-len(data)) % align))
                print(f"  bf16  {t['name']:36s} <- {t['hf'][0]}{'' if t['hf'][1] is None else list(t['hf'][1])}")
            else:
                fin.seek(g["data_start"] + t["offset"])
                left = t["nbytes"]
                while left:
                    chunk = fin.read(min(left, 64 << 20))
                    if not chunk:
                        fail(f"source ended inside {t['name']}")
                    fout.write(chunk)
                    left -= len(chunk)
    os.rename(tmp, out)
    print(f"wrote {out}: {os.path.getsize(out)} bytes")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="print the GGUF directory and exit")
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    if a.list:
        for p in a.paths:
            list_gguf(p)
        return
    if len(a.paths) != 3:
        fail("usage: make_hybrid_mtp.py <draft.gguf> <fp8 checkpoint dir> <out.gguf>")
    build(*a.paths)


if __name__ == "__main__":
    main()
