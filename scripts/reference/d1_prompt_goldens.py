"""Golden d1 prompts, ids and readout groups from the release's own prompt.py.

Usage: python -I d1_prompt_goldens.py <d1 snapshot dir> <out.json>
Single-question cases are tokenized whole (as runner.run does); multi-question
cases as trunk + branches (as runner._request does).
"""
import importlib.util, json, sys, types

snap, out = sys.argv[1], sys.argv[2]
pkg = types.ModuleType("d1"); pkg.__path__ = [snap]; sys.modules["d1"] = pkg
spec = importlib.util.spec_from_file_location("d1.prompt", f"{snap}/prompt.py")
P = importlib.util.module_from_spec(spec); sys.modules["d1.prompt"] = P; spec.loader.exec_module(P)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(snap)
bos = tok.bos_token

def enc(t):
    return tok.encode(t, add_special_tokens=False)

def readout(q):
    return P.readout_ids(tok, q)

CASES = [
    ("noul_plain", "I was charged twice this month, please refund one of them.",
     [{"type": "noul", "instructions": "Is the customer asking for a refund?"}]),
    ("noul_criteria", "The door is open.",
     [{"type": "noul", "instructions": "Is it safe?", "criteria": {"true": "Nothing is open", "false": None}}]),
    ("choice_desc", "I was charged twice this month, please refund one of them.",
     [{"type": "choice", "instructions": "Which team should handle this?",
       "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                    "fraud": "Suspected unauthorised use"}}]),
    ("choice_bare", "hello", [{"type": "choice", "instructions": "Pick one",
                               "criteria": {"red_wine": None, "blue": "", "x": "the x"}}]),
    ("choice_letters", "hello", [{"type": "choice", "instructions": "Which?",
                                  "criteria": {"B": "bee", "A": "ay", "q": "queue"}}]),
    ("choice_wide", "w", [{"type": "choice", "instructions": "Wide",
                           "criteria": {f"opt{i}": f"option {i}" for i in range(30)}}]),
    ("score", "Down for an hour.", [{"type": "score", "instructions": "How urgent is this?",
                                     "criteria": ["Can wait", "Today", "Blocking the customer now"]}]),
    ("json_state", {"user": "Zoë", "items": [1, 2.5, True, None], "nested": {"b": 1, "a": "x"}},
     [{"type": "noul", "instructions": "Is there a user?"}]),
    ("no_state", None, [{"type": "noul", "instructions": "Is this a cat?"}]),
    ("multi", "I was charged twice this month, please refund one of them.",
     [{"type": "noul", "instructions": "Is the customer asking for a refund?"},
      {"type": "choice", "instructions": "Which team should handle this?",
       "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults"}},
      {"type": "score", "instructions": "How urgent is this?", "criteria": ["Can wait", "Today"]}]),
]

cases = []
for name, state, qs in CASES:
    qs_ = [P.as_question(q) for q in qs]
    entry = {"name": name, "state": state, "questions": qs}
    prefix = P.prefix_text(tok, state, bos)
    suffixes = [P.suffix_text(tok, q) for q in qs_]
    entry["prefix_text"] = prefix
    entry["suffix_texts"] = suffixes
    if len(qs_) == 1:
        entry["text"] = prefix + suffixes[0]
        entry["ids"] = enc(entry["text"])
    else:
        entry["trunk"] = enc(prefix)
        entry["branches"] = [enc(s) for s in suffixes]
    entry["groups"] = [readout(q) for q in qs_]
    cases.append(entry)

# The image markup the processor writes for one small (single-tile) picture.
try:
    from transformers import AutoProcessor
    from PIL import Image
    proc = AutoProcessor.from_pretrained(snap)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "\x00"}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
    head = "<|im_start|>user\n"
    markup = text[text.index(head) + len(head):text.index("\x00")]
    pics = [Image.new("RGB", (256, 256), (200, 30, 30)), Image.new("RGB", (320, 192), (30, 30, 200))]
    inputs = proc(text=[markup + "Q"], images=[pics], return_tensors="pt", add_special_tokens=False)
    ids = inputs["input_ids"][0].tolist()
    image_tokens = {"<|image_start|>": tok.convert_tokens_to_ids("<|image_start|>"),
                    "<image>": tok.convert_tokens_to_ids("<image>"),
                    "<|image_end|>": tok.convert_tokens_to_ids("<|image_end|>")}
    runs, cur = [], None
    for t in ids:
        if t == image_tokens["<image>"]:
            cur = (cur or 0) + 1
        elif cur:
            runs.append(cur); cur = None
    extra = {"markup": markup, "ids": ids, "image_tokens": image_tokens, "runs": runs}
except Exception as e:  # recorded, not fatal
    extra = {"error": repr(e)}

json.dump({"cases": cases, "images": extra, "bos": bos}, open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", len(cases), "cases; images:", {k: (v if k != "ids" else len(v)) for k, v in extra.items()})
