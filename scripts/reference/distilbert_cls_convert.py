#!/usr/bin/env python3
"""Recover the command classifier's weights from its int8 ONNX file, and grade them.

Convert mode dequantizes every weight to float32 (`q = (int - zero_point) *
scale`, per-output-channel for the linear layers, per-tensor for the embedding
tables), maps the graph's tensors to the standard names, and writes
`model.safetensors` (F16 for matrices and embedding tables, F32 for LayerNorm
parameters and biases), `config.json` (architecture plus the `calibration`
object the host applies outside the model) and a copy of `tokenizer.json`.

Verify mode runs the converted weights through a float32 NumPy forward pass
over the corpus ids stored in the ONNX Runtime goldens, compares against those
goldens, writes `goldens_numpy.json` (same rows, NumPy logits) and exits
non-zero if the M0 criterion fails. The original graph quantizes activations to
int8 at run time and this forward does not, so the comparison is loose; the
tight parity is graded later against the device implementation. The calibration
constants are the full values from the metadata; the upstream runtime rounds
them (1.3366 / -2.5434), which flips no decision on the corpus.

Usage:
    uv run --with onnx --with numpy --with safetensors \
        scripts/reference/distilbert_cls_convert.py \
        --onnx model.onnx --tokenizer tokenizer.json \
        --meta classifier_meta.json --out-dir OUT
    uv run --with numpy --with safetensors --with scipy \
        scripts/reference/distilbert_cls_convert.py --verify \
        --out-dir OUT --goldens OUT/goldens.json
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

LAYERS, DIM, HEADS, HIDDEN, VOCAB, POSITIONS, LABELS = 6, 768, 12, 3072, 30522, 512, 3
LN_EPS = 1e-12
# One copy of the calibration, shared with the goldens script, so the decision
# comparison in verify mode cannot drift from the reference's decision logic.
# The script's own directory goes first so this works from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from distilbert_cls_reference import ENERGY_P95, SAFE_THRESHOLD, TEMPERATURE, decide  # noqa: E402

# (module path within a layer, expected `[out, in]` shape) for one encoder layer's linears.
LAYER_LINEARS = [
    ("attention.q_lin", (DIM, DIM)),
    ("attention.k_lin", (DIM, DIM)),
    ("attention.v_lin", (DIM, DIM)),
    ("attention.out_lin", (DIM, DIM)),
    ("ffn.lin1", (HIDDEN, DIM)),
    ("ffn.lin2", (DIM, HIDDEN)),
]
LAYER_NORMS = ["sa_layer_norm", "output_layer_norm"]


def expected_tensors():
    """Every tensor name -> (shape, stored dtype); linear weights are `[out, in]`."""
    t = {
        "distilbert.embeddings.word_embeddings.weight": ((VOCAB, DIM), "f16"),
        "distilbert.embeddings.position_embeddings.weight": ((POSITIONS, DIM), "f16"),
        "distilbert.embeddings.LayerNorm.weight": ((DIM,), "f32"),
        "distilbert.embeddings.LayerNorm.bias": ((DIM,), "f32"),
        "pre_classifier.weight": ((DIM, DIM), "f16"),
        "pre_classifier.bias": ((DIM,), "f32"),
        "classifier.weight": ((LABELS, DIM), "f16"),
        "classifier.bias": ((LABELS,), "f32"),
    }
    for i in range(LAYERS):
        p = f"distilbert.transformer.layer.{i}."
        for name, shape in LAYER_LINEARS:
            t[p + name + ".weight"] = (shape, "f16")
            t[p + name + ".bias"] = ((shape[0],), "f32")
        for name in LAYER_NORMS:
            t[p + name + ".weight"] = ((DIM,), "f32")
            t[p + name + ".bias"] = ((DIM,), "f32")
    return t


def extract(onnx_path):
    """Dequantized float32 tensors keyed by their standard names."""
    import onnx
    from onnx import numpy_helper

    g = onnx.load(onnx_path).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    out = {}

    # Linear weights: each MatMulInteger node is named `/<module path>/<op>_quant`
    # and takes `<w>_quantized` stored as [in, out] with a per-output-channel
    # `<w>_scale` / `<w>_zero_point`. Walk the nodes, never guess from order.
    for n in g.node:
        if n.op_type != "MatMulInteger":
            continue
        base = n.input[1].removesuffix("_quantized")
        if base + "_quantized" not in init:
            continue  # activation x activation product (attention scores/context)
        module = ".".join(n.name.strip("/").split("/")[:-1])
        q, scale, zp = init[base + "_quantized"], init[base + "_scale"], init[base + "_zero_point"]
        assert q.ndim == 2 and scale.shape == zp.shape == (q.shape[1],), (n.name, q.shape, scale.shape)
        w = (q.astype(np.float32) - zp.astype(np.float32)[None, :]) * scale[None, :]
        out[module + ".weight"] = np.ascontiguousarray(w.T)

    # Embedding tables: per-tensor uint8 tables that a Gather reads before the
    # gathered rows are dequantized with the table's scale and zero point.
    for n in g.node:
        if n.op_type != "Gather" or not n.input[0].endswith("_quantized") or n.input[0] not in init:
            continue
        base = n.input[0].removesuffix("_quantized")
        q, scale, zp = init[base + "_quantized"], init[base + "_scale"], init[base + "_zero_point"]
        assert scale.shape == () and zp.shape == (), n.name
        out[base.removesuffix("_weight").removesuffix(".weight") + ".weight"] = (
            q.astype(np.float32) - np.float32(zp)
        ) * np.float32(scale)

    # Biases and LayerNorm parameters are plain float initializers under the final names.
    for name, (_, dtype) in expected_tensors().items():
        if dtype == "f32":
            out[name] = init[name].astype(np.float32)

    # LayerNormalization epsilon must be the one the forward pass assumes.
    for n in g.node:
        if n.op_type == "LayerNormalization":
            eps = next((a.f for a in n.attribute if a.name == "epsilon"), 1e-5)
            assert abs(eps - LN_EPS) < 1e-13, (n.name, eps)
    return out


def check(tensors):
    """Assert exactly the expected tensor set, shapes and finite values."""
    want = expected_tensors()
    missing = sorted(set(want) - set(tensors))
    extra = sorted(set(tensors) - set(want))
    assert not missing and not extra, f"missing {missing}, unexpected {extra}"
    for name, (shape, _) in want.items():
        t = tensors[name]
        assert t.shape == shape, f"{name}: {t.shape} != {shape}"
        assert np.isfinite(t).all(), f"{name}: non-finite"


def convert(a):
    from safetensors.numpy import save_file

    tensors = extract(a.onnx)
    check(tensors)
    want = expected_tensors()
    stored = {
        k: v.astype(np.float16 if want[k][1] == "f16" else np.float32) for k, v in tensors.items()
    }
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(stored, out / "model.safetensors", metadata={"format": "pt"})
    with open(a.meta, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["max_length"] == 128 and meta["labels"] == ["safe", "needs-approval", "dangerous"]
    config = {
        "model_type": "distilbert",
        "architectures": ["DistilBertForSequenceClassification"],
        "n_layers": LAYERS,
        "n_heads": HEADS,
        "dim": DIM,
        "hidden_dim": HIDDEN,
        "vocab_size": VOCAB,
        "max_position_embeddings": POSITIONS,
        "num_labels": LABELS,
        "id2label": meta["id2label"],
        "label2id": meta["label2id"],
        "layer_norm_eps": LN_EPS,
        "calibration": {
            "temperature": meta["temperature"],
            "energy_p95": meta["energy_thresholds"]["energy_p95"],
            "safe_threshold": SAFE_THRESHOLD,
            "max_length": meta["max_length"],
        },
    }
    assert config["calibration"]["temperature"] == TEMPERATURE
    assert config["calibration"]["energy_p95"] == ENERGY_P95
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    shutil.copyfile(a.tokenizer, out / "tokenizer.json")
    print(f"wrote {len(stored)} tensors to {out}")


# ---------------------------------------------------------------- verification


def layer_norm(x, w, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + np.float32(LN_EPS)) * w + b


def forward(w, ids, erf):
    """float32 logits `[3]` for one unpadded id sequence (post-LN encoder, CLS head)."""
    n = len(ids)
    h = w["distilbert.embeddings.word_embeddings.weight"][ids] + w[
        "distilbert.embeddings.position_embeddings.weight"
    ][:n]
    h = layer_norm(h, w["distilbert.embeddings.LayerNorm.weight"], w["distilbert.embeddings.LayerNorm.bias"])
    hd = DIM // HEADS
    for i in range(LAYERS):
        p = f"distilbert.transformer.layer.{i}."

        def lin(x, name):
            return x @ w[p + name + ".weight"].T + w[p + name + ".bias"]

        def split(x):
            return x.reshape(n, HEADS, hd).transpose(1, 0, 2)

        q, k, v = (split(lin(h, f"attention.{x}_lin")) for x in "qkv")
        s = q @ k.transpose(0, 2, 1) / np.float32(np.sqrt(hd))
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        a = lin((s @ v).transpose(1, 0, 2).reshape(n, DIM), "attention.out_lin")
        h = layer_norm(h + a, w[p + "sa_layer_norm.weight"], w[p + "sa_layer_norm.bias"])
        f = lin(h, "ffn.lin1")
        f = f * np.float32(0.5) * (1 + erf(f / np.float32(np.sqrt(2))).astype(np.float32))
        h = layer_norm(h + lin(f, "ffn.lin2"), w[p + "output_layer_norm.weight"], w[p + "output_layer_norm.bias"])
    c = np.maximum(h[0] @ w["pre_classifier.weight"].T + w["pre_classifier.bias"], 0)
    return c @ w["classifier.weight"].T + w["classifier.bias"]


def verify(a):
    from safetensors.numpy import load_file
    from scipy.special import erf

    out = Path(a.out_dir)
    w = {k: v.astype(np.float32) for k, v in load_file(out / "model.safetensors").items()}
    check(w)
    with open(a.goldens, encoding="utf-8") as f:
        gold = json.load(f)
    rows, max_dl = [], 0.0
    arg_ok = dec_ok = 0
    false_approve, dec_pairs, arg_bad, dec_bad = [], {}, [], []
    for g in gold:
        lg = [float(x) for x in forward(w, g["ids"], erf)]
        d = decide(lg)
        rows.append({**g, "logits": lg, "decision": d})
        max_dl = max(max_dl, max(abs(x - y) for x, y in zip(lg, g["logits"])))
        if np.argmax(lg) == np.argmax(g["logits"]):
            arg_ok += 1
        else:
            arg_bad.append((g, lg))
        if d == g["decision"]:
            dec_ok += 1
        else:
            dec_bad.append((g, lg, d))
        dec_pairs[(g["decision"], d)] = dec_pairs.get((g["decision"], d), 0) + 1
        if d == "approve" and g["decision"] != "approve" and g["label"] in ("needs-approval", "dangerous"):
            false_approve.append(g)
    n = len(gold)
    with open(a.numpy_out or out / "goldens_numpy.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"rows {n}")
    print(f"(a) argmax equal   {arg_ok}/{n} = {arg_ok / n:.4f}  (need >= 0.97)")
    print(f"(b) decision equal {dec_ok}/{n} = {dec_ok / n:.4f}  (need >= 0.97)")
    print(f"(c) false approvals {len(false_approve)}  (need 0)")
    print(f"(d) max |dlogit| {max_dl:.4f}")
    print("(onnx decision, numpy decision) counts:", {f"{k[0]}->{k[1]}": v for k, v in sorted(dec_pairs.items())})
    for g, lg in arg_bad:
        print("  ARGMAX MISMATCH", g["label"], repr(g["text"][:60]), "onnx", g["logits"], "numpy", lg)
    for g, lg, d in dec_bad:
        print("  DECISION MISMATCH", g["label"], repr(g["text"][:60]), g["decision"], "->", d, "numpy", lg)
    for g in false_approve:
        print("  FALSE APPROVE", g["label"], repr(g["text"][:80]), g["logits"])
    ok = arg_ok / n >= 0.97 and dec_ok / n >= 0.97 and not false_approve
    print("M0 criterion:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx")
    ap.add_argument("--tokenizer")
    ap.add_argument("--meta")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--goldens")
    ap.add_argument("--numpy-out")
    a = ap.parse_args()
    if a.verify:
        if not a.goldens:
            ap.error("--verify needs --goldens")
        sys.exit(verify(a))
    if not (a.onnx and a.tokenizer and a.meta):
        ap.error("convert needs --onnx, --tokenizer and --meta")
    convert(a)


if __name__ == "__main__":
    main()
