#!/usr/bin/env python3
"""
Reproduce the "read-back hallucination" failure mode of a local LLM.

Symptom being hunted: after the model writes a file with edit_file and then
reads it back with read_file, it starts claiming the returned text differs
from what it wrote ("some occurrences of 'Times New Roman' have become
'Times New Roman'...") and spirals in its reasoning trying to compare two
identical strings.

The script runs a minimal agent loop against an OpenAI-compatible endpoint
with exactly two tools -- read_file and edit_file -- over a small in-memory
email workspace, and flags runs where the spiral shows up.

Stdlib only. No third-party dependencies.

    python3 repro_readback.py                 # one run, default model
    python3 repro_readback.py --runs 5        # five independent runs
    LLM_MODEL=qwen/qwen3.8-flash \
    LLM_BASE_URL=https://openrouter.ai/api \
    LLM_API_KEY=sk-or-... python3 repro_readback.py   # control run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "qwen3.8-flash")
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY", "")

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the email template editor in the Otsimo journey console. You work
on one email revision through a small file workspace, and your edits to
those files are the email, there is no submit step. Whatever the files
say when your turn ends is what gets saved.

## The workspace

Four text files, always present:

- email.html -- the HTML part. This is the email.
- text.txt -- the plain-text alternative. Sending one alongside the HTML
  materially improves deliverability, so it is not optional. Keep it
  saying what the HTML says.
- meta.json -- subject and preheader, each a map keyed by language code.
  The key "*" means one string for every language.
- data.json -- a "data" map of merge variables shared by every language,
  and a "localizedData" map whose per-language entries override them.

You have exactly two tools: read_file and edit_file. There is no shell, no
network, no other tool. Read before you write, and never edit a file you
have not read in this session -- unless the user tells you a file is empty,
in which case you may write it directly with clear=true.

meta.json and data.json must stay valid JSON: edit_file refuses an edit
that would break either and tells you why. Match the two-space indentation
already in the file.

## Merge variables

Subjects and bodies are Go templates, so a variable reads
{{ .FirstName }} and every name you use must exist as a key in
data.json. Add it there, with a sensible placeholder value, in the same
turn you first use it. A name with no key renders as literal braces --
in the preview, and in the sent mail.

## Images

Images are referenced as asset:hero.png, never as a URL. Only reference
names that exist in the workspace.

## What you cannot change

The sender name, the from address, the reply-to and the ESP categories
have no file here. A human edits them in the fields sheet. If one of them
is wrong, say so; do not try to write it somewhere.

## HTML that survives an inbox

Mail clients are not browsers. Lay out with tables, put every style in a
style attribute, keep the body near 600px wide, and end the font stack in
a web-safe family, because a webfont will not load. No flexbox, no grid,
no external stylesheet, no script. Every image gets an alt attribute and
an explicit width. The preheader goes at the top of the body as hidden
text.

## This app

The brief below is this app's own metadata. Its colours, logo, legal
text, URLs and voice are facts -- use them instead of inventing your own.
If it is missing something you need, ask rather than fill the gap.

{
  "context": "Crayon Club is a digital coloring and drawing subscription app for children ages 2-8 and their families. It offers creative play without physical clean-up, with 300+ coloring pages, blank-page drawing, digital crayons, paints, stamps, stickers, magical brushes, and new content every month. The app is always ad-free and playable offline. Keep messaging focused on creative play, imagination, self-expression, and family-friendly use. Do not imply medical, therapeutic, educational, or guaranteed developmental outcomes.",
  "languages": ["en", "fr", "de", "ja", "pt", "ru", "zh", "es"],
  "product": {
    "displayName": "Crayon Club",
    "tagline": "The magic of arts & crafts at your fingertips",
    "audience": "Families and parents of children ages 2-8",
    "websiteUrl": "https://crayonclubapp.com"
  },
  "brand": {
    "primaryColor": "#0000EE",
    "accentColor": "#073767",
    "backgroundColor": "#FFFFFF",
    "textColor": "#54514F",
    "fontStack": "Times New Roman, serif",
    "layoutNotes": "Use a playful, colorful house style with a clean white background, blue primary accents, dark blue supporting accents, warm gray text, and generous space for children's artwork and colorful illustrations."
  },
  "sender": { "fromName": "Crayon Club", "fromEmail": "hello@otsimo.xyz" },
  "legal": {
    "companyName": "Sago Sago Toys Inc., doing business as Piknik",
    "postalAddress": "171 John Street, 5th Floor, Toronto, Ontario, M5T 1X3, Canada",
    "privacyUrl": "https://sagomini.com/privacy-policy/",
    "termsUrl": "https://sagomini.com/terms-of-use/",
    "supportEmail": "support@playpiknik.com"
  },
  "links": { "webBaseUrl": "https://crayonclubapp.com" },
  "voice": {
    "tone": ["playful", "imaginative", "encouraging", "family-friendly"],
    "avoid": ["medical or therapeutic claims", "guaranteed developmental outcomes", "pressure or fear-based language", "adult-oriented language"]
  }
}

## Finishing

There is no submit tool. Your file edits are the result and are saved
when the turn ends. Close by saying what you changed, file by file, and
name what you left for a human: a colour you guessed, a language you
could not write, a sender field you cannot reach.
"""

USER_PROMPT = """lets build a simple welcome email.

we need to put to links
- https://crayonclubapp.com/#branch-a
- https://crayonclubapp.com/#branch-b

We are building a testing email for each link we go to different section on the flow. Don't need to add image asset and think mobile first on emails.

email, text data and meta files are empty. You don't need to check them. Only English is enough. Create stacked 2 buttons with generic titles.

The plan is already approved -- do not ask any questions, just write the four files now."""

# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the email workspace and return its full contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path, e.g. email.html",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Edit a file in the email workspace. With clear=true the whole file is "
                "replaced by new_text. Otherwise the single occurrence of old_text is "
                "replaced by new_text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative path."},
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace. Ignored when clear is true.",
                    },
                    "new_text": {"type": "string", "description": "Replacement text."},
                    "clear": {
                        "type": "boolean",
                        "description": "Replace the entire file contents with new_text.",
                    },
                },
                "required": ["path", "new_text"],
            },
        },
    },
]

WORKSPACE_SEED = {
    "email.html": "",
    "text.txt": "",
    "meta.json": '{\n  "subject": {},\n  "preheader": {}\n}\n',
    "data.json": '{\n  "data": {},\n  "localizedData": {}\n}\n',
}


class ToolError(Exception):
    pass


class Workspace:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        for name, content in WORKSPACE_SEED.items():
            (self.root / name).write_text(content, encoding="utf-8")

    def _resolve(self, path: str) -> Path:
        p = (self.root / str(path).lstrip("/")).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ToolError(f"path escapes the workspace: {path}")
        return p

    def read_file(self, path: str) -> str:
        p = self._resolve(path)
        if not p.exists():
            raise ToolError(f"no such file: {path}")
        return p.read_text(encoding="utf-8")

    def edit_file(self, path: str, new_text: str, old_text: str = "", clear: bool = False) -> str:
        p = self._resolve(path)
        if not p.exists():
            raise ToolError(f"no such file: {path}")
        current = p.read_text(encoding="utf-8")

        if clear or (not old_text and not current.strip()):
            updated = new_text
        else:
            if old_text not in current:
                raise ToolError(f"old_text not found in {path}")
            if current.count(old_text) > 1:
                raise ToolError(
                    f"old_text occurs {current.count(old_text)} times in {path}; make it unique"
                )
            updated = current.replace(old_text, new_text, 1)

        if p.suffix == ".json":
            try:
                json.loads(updated)
            except json.JSONDecodeError as exc:
                raise ToolError(f"edit refused: {path} would not be valid JSON ({exc})") from exc

        p.write_text(updated, encoding="utf-8")
        return f"wrote /home/user/{path} ({len(updated.encode('utf-8'))} bytes)"

    def dispatch(self, name: str, args: dict) -> str:
        if name == "read_file":
            return self.read_file(args["path"])
        if name == "edit_file":
            return self.edit_file(
                path=args["path"],
                new_text=args.get("new_text", ""),
                old_text=args.get("old_text", "") or "",
                clear=bool(args.get("clear", False)),
            )
        raise ToolError(f"unknown tool: {name}")


# --------------------------------------------------------------------------
# Reasoning budget
# --------------------------------------------------------------------------

REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")


def reasoning_fields(args) -> dict:
    """Ask the target model to think harder.

    Every server spells this differently and this endpoint accepted all three
    spellings without complaint, so "both" sends the two portable ones. If a
    stricter backend rejects them the caller retries without.
    """
    effort = (args.reasoning_effort or "").lower()
    if args.reasoning_style == "off" or effort in ("", "off", "none"):
        return {}

    fields: dict = {}
    if args.reasoning_style in ("both", "openai"):
        fields["reasoning_effort"] = effort
    if args.reasoning_style in ("both", "openrouter"):
        block: dict = {"effort": effort}
        if args.reasoning_max_tokens:
            block["max_tokens"] = args.reasoning_max_tokens
        fields["reasoning"] = block
    if args.reasoning_style == "qwen":
        kwargs: dict = {"enable_thinking": True}
        if args.reasoning_max_tokens:
            kwargs["thinking_budget"] = args.reasoning_max_tokens
        fields["chat_template_kwargs"] = kwargs
    return fields


def strip_reasoning_fields(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in ("reasoning_effort", "reasoning", "chat_template_kwargs")}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def chat_completion(
    base_url: str,
    api_key: str,
    payload: dict,
    timeout: int,
    retries: int = 3,
) -> dict:
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        if not re.search(r"/v\d+$", url):
            url += "/v1"
        url += "/chat/completions"

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:2000]
            last_err = RuntimeError(f"HTTP {exc.code} from {url}: {detail}")
            if exc.code < 500 and exc.code != 429:
                raise last_err
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = RuntimeError(f"request to {url} failed: {exc}")
        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))
    raise last_err  # type: ignore[misc]


# --------------------------------------------------------------------------
# Failure detection
# --------------------------------------------------------------------------

# Weighted: generic hedges score low, read-back-specific doubt scores high.
DOUBT_PATTERNS = {
    r"\bno wait\b": 3,
    r"\bwait\s*[\u2014\u2013-]": 2,
    r"\bhmm\b": 1,
    r"\bactually,? (the|looking|did)\b": 1,
    r"\bI can'?t tell\b": 3,
    r"\bI keep going back and forth\b": 4,
    r"\blet me look carefully\b": 2,
    r"\blet me (re-?read|read it again|double-?check|verify)\b": 2,
    r"\bhave become\b": 3,
    r"\bthat'?s (not right|wrong)\b": 2,
    r"\banomal(y|ies)\b": 2,
    r"\bthe read-?back\b": 2,
    r"\bdid I (actually )?(write|type)\b": 3,
    r"\b(I )?can'?t remember\b": 2,
    r"\bwith a space inserted\b": 3,
}
DOUBT_RE = [(re.compile(pat, re.IGNORECASE), w) for pat, w in DOUBT_PATTERNS.items()]

SHINGLE = 8


def max_shingle_repeat(text: str) -> int:
    """How many times the most-repeated 8-word window recurs in one reasoning
    block. A model comparing an identical string against itself over and over
    restates the same phrase, so this climbs fast during the spiral."""
    words = re.findall(r"\S+", text.lower())
    if len(words) < SHINGLE * 2:
        return 0
    counts: dict = {}
    best = 0
    for i in range(len(words) - SHINGLE + 1):
        key = tuple(words[i : i + SHINGLE])
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > best:
            best = counts[key]
    return best


@dataclass
class RunResult:
    index: int
    model: str
    steps: int = 0
    tool_calls: list = field(default_factory=list)
    reasoning_chars: int = 0
    reasoning_tokens: int = 0
    completion_tokens: int = 0
    reasoning_rejected: bool = False
    doubt_hits: dict = field(default_factory=dict)
    doubt_score: int = 0
    post_read_doubt: int = 0
    max_shingle: int = 0
    repeated_reads: dict = field(default_factory=dict)
    duplicate_tool_calls: int = 0
    finished: bool = False
    degenerate: bool = False
    error: str | None = None
    files: dict = field(default_factory=dict)
    transcript: list = field(default_factory=list)

    @property
    def spiral(self) -> bool:
        """The read-back hallucination looks like doubt language appearing only
        after a read_file, the same phrase restated over and over, or the model
        re-reading / re-issuing the same call to settle a question about text it
        has already been shown."""
        return (
            self.post_read_doubt >= 5
            or self.max_shingle >= 4
            or max(self.repeated_reads.values(), default=0) >= 3
            or self.duplicate_tool_calls >= 2
        )


def scan_doubt(text: str, into: dict) -> int:
    score = 0
    for rx, weight in DOUBT_RE:
        n = len(rx.findall(text))
        if n:
            into[rx.pattern] = into.get(rx.pattern, 0) + n
            score += n * weight
    return score


def extract_reasoning(message: dict) -> str:
    for key in ("reasoning_content", "reasoning", "thinking"):
        val = message.get(key)
        if isinstance(val, str) and val:
            return val
        if isinstance(val, list):
            parts = [p.get("text", "") for p in val if isinstance(p, dict)]
            if any(parts):
                return "\n".join(parts)
    return ""


# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------


def run_once(args, index: int) -> RunResult:
    ws_root = Path(args.workspace).expanduser().resolve() / f"run-{index:02d}"
    if ws_root.exists():
        shutil.rmtree(ws_root)
    ws = Workspace(ws_root)

    system_prompt = Path(args.system_file).read_text(encoding="utf-8") if args.system_file else SYSTEM_PROMPT
    user_prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else USER_PROMPT

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    result = RunResult(index=index, model=args.model)
    seen_calls: dict[str, int] = {}
    has_read = False

    for step in range(args.max_steps):
        payload = {
            "model": args.model,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": args.temperature,
        }
        if args.max_tokens:
            payload["max_tokens"] = args.max_tokens
        if args.seed is not None:
            payload["seed"] = args.seed
        if args.top_p is not None:
            payload["top_p"] = args.top_p
        if args.top_k is not None:
            payload["top_k"] = args.top_k
        if not result.reasoning_rejected:
            payload.update(reasoning_fields(args))

        try:
            resp = chat_completion(args.base_url, args.api_key, payload, args.timeout)
        except Exception as exc:  # noqa: BLE001 - surfaced in the report
            msg = str(exc)
            retryable = "HTTP 400" in msg or "HTTP 422" in msg
            if retryable and not result.reasoning_rejected and reasoning_fields(args):
                # A stricter backend refused the reasoning field. Drop it for the
                # rest of the run rather than losing the run entirely.
                result.reasoning_rejected = True
                print(f"  [warn] backend rejected the reasoning budget, continuing without it: {msg[:160]}", flush=True)
                try:
                    resp = chat_completion(args.base_url, args.api_key, strip_reasoning_fields(payload), args.timeout)
                except Exception as exc2:  # noqa: BLE001
                    result.error = str(exc2)
                    break
            else:
                result.error = msg
                break

        result.steps = step + 1
        choice = (resp.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        reasoning = extract_reasoning(message)
        tool_calls = message.get("tool_calls") or []

        usage = resp.get("usage") or {}
        result.completion_tokens += usage.get("completion_tokens") or 0
        result.reasoning_tokens += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        result.reasoning_chars += len(reasoning)
        score = scan_doubt(reasoning + "\n" + content, result.doubt_hits)
        result.doubt_score += score
        if has_read:
            result.post_read_doubt += score
        result.max_shingle = max(result.max_shingle, max_shingle_repeat(reasoning))
        result.transcript.append(
            {
                "step": step + 1,
                "reasoning": reasoning,
                "content": content,
                "tool_calls": [
                    {"name": tc.get("function", {}).get("name"), "arguments": tc.get("function", {}).get("arguments")}
                    for tc in tool_calls
                ],
                "usage": resp.get("usage"),
                "finish_reason": choice.get("finish_reason"),
            }
        )

        log_step(args, step + 1, reasoning, content, tool_calls)

        # Incremental persistence: this box's external killer SIGTERMs
        # long-running measurement processes (qwen4exp-perf.md rule 3, twice
        # confirmed 2026-08-31), and an end-of-run-only write then loses the
        # whole run. Dump the run-in-progress after every step.
        if args.out:
            partial = Path(args.out).expanduser().with_suffix(f".run{index:02d}.partial.json")
            partial.write_text(
                json.dumps(
                    {
                        "index": result.index,
                        "steps": result.steps,
                        "spiral": result.spiral,
                        "degenerate": result.degenerate,
                        "doubt_score": result.doubt_score,
                        "post_read_doubt": result.post_read_doubt,
                        "max_shingle_repeat": result.max_shingle,
                        "repeated_reads": result.repeated_reads,
                        "duplicate_tool_calls": result.duplicate_tool_calls,
                        "reasoning_chars": result.reasoning_chars,
                        "transcript": result.transcript,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        # Echo the reasoning trace back too: this family's template replays it
        # inside <think>, so dropping it makes the next prompt diverge from the
        # generated tokens and the server can reuse only the previous prompt.
        assistant_msg = {"role": "assistant", "content": content}
        if reasoning:
            assistant_msg["reasoning_content"] = reasoning
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            result.finished = True
            if len((reasoning + content).strip()) < 10:
                # The backend sometimes emits one or two tokens and stops. That is a
                # broken route, not a clean run -- do not let it read as a pass.
                result.degenerate = True
                usage = resp.get("usage") or {}
                result.error = (
                    f"degenerate completion: {usage.get('completion_tokens')} completion tokens, "
                    f"finish_reason={choice.get('finish_reason')}, text={(reasoning + content)!r}"
                )
            break

        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments") or "{}"
            try:
                parsed = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                if isinstance(parsed, str):  # some servers double-encode
                    parsed = json.loads(parsed)
            except json.JSONDecodeError as exc:
                out = f"error: could not parse arguments as JSON ({exc})"
                parsed = {}
            else:
                try:
                    out = ws.dispatch(name, parsed)
                except ToolError as exc:
                    out = f"error: {exc}"
                except Exception as exc:  # noqa: BLE001
                    out = f"error: {type(exc).__name__}: {exc}"

            path = parsed.get("path", "?") if isinstance(parsed, dict) else "?"
            result.tool_calls.append({"name": name, "path": path})
            if name == "read_file":
                result.repeated_reads[path] = result.repeated_reads.get(path, 0) + 1
                has_read = True

            fingerprint = f"{name}:{raw_args}"
            seen_calls[fingerprint] = seen_calls.get(fingerprint, 0) + 1
            if seen_calls[fingerprint] > 1:
                result.duplicate_tool_calls += 1

            if args.verbose:
                preview = out if len(out) < 400 else out[:400] + f"... [{len(out)} chars]"
                print(f"      -> {preview}", flush=True)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{step}"),
                    "content": out,
                }
            )

    result.files = {p.name: p.read_text(encoding="utf-8") for p in sorted(ws_root.glob("*"))}
    return result


def log_step(args, step: int, reasoning: str, content: str, tool_calls: list) -> None:
    print(f"\n--- step {step} ---", flush=True)
    if reasoning:
        shown = reasoning if args.verbose else reasoning[: args.preview]
        suffix = "" if args.verbose or len(reasoning) <= args.preview else f"\n    ... [{len(reasoning)} chars total]"
        print(f"  [reasoning] {shown}{suffix}", flush=True)
    if content:
        shown = content if args.verbose else content[: args.preview]
        suffix = "" if args.verbose or len(content) <= args.preview else f"\n    ... [{len(content)} chars total]"
        print(f"  [say] {shown}{suffix}", flush=True)
    for tc in tool_calls:
        fn = tc.get("function") or {}
        raw = fn.get("arguments") or ""
        try:
            path = json.loads(raw).get("path", "?")
        except Exception:  # noqa: BLE001
            path = "?"
        print(f"  [tool] {fn.get('name')}({path})  args={len(raw)} chars", flush=True)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def report(results: list[RunResult], args) -> None:
    print("\n" + "=" * 72)
    budget = reasoning_fields(args) or {"reasoning": "off"}
    print(f"model={args.model}  base_url={args.base_url}  temperature={args.temperature}")
    print(f"reasoning budget: {json.dumps(budget)}  (style={args.reasoning_style})")
    print("=" * 72)
    for r in results:
        if r.degenerate:
            verdict = "EMPTY"
        elif r.spiral:
            verdict = "SPIRAL"
        elif r.error:
            verdict = "error"
        else:
            verdict = "clean"
        reads = ", ".join(f"{k}x{v}" for k, v in sorted(r.repeated_reads.items())) or "-"
        print(
            f"run {r.index:>2}  {verdict:<6}  steps={r.steps:<3} tools={len(r.tool_calls):<3} "
            f"reasoning={r.reasoning_chars:<7}c/{r.reasoning_tokens:<6}t doubt={r.doubt_score}(post-read {r.post_read_doubt}) "
            f"repeat={r.max_shingle} dup_calls={r.duplicate_tool_calls} reads[{reads}]"
        )
        if r.reasoning_rejected:
            print("          note: backend rejected the reasoning budget; ran without it")
        if r.error:
            print(f"          error: {r.error}")
        if r.doubt_hits:
            top = sorted(r.doubt_hits.items(), key=lambda kv: -kv[1])[:5]
            print("          doubt: " + ", ".join(f"{p}={n}" for p, n in top))
        written = [n for n, c in r.files.items() if c.strip() and c != WORKSPACE_SEED.get(n, "")]
        print(f"          wrote: {', '.join(written) or 'nothing'}")

    spirals = sum(1 for r in results if r.spiral and not r.degenerate)
    empties = sum(1 for r in results if r.degenerate)
    errors = sum(1 for r in results if r.error and not r.degenerate)
    print("-" * 72)
    print(
        f"{spirals}/{len(results)} runs hit the read-back spiral heuristic; "
        f"{empties} returned a degenerate empty completion; {errors} errored."
    )
    if empties:
        print("NOTE: an EMPTY verdict means the route answered with ~no tokens -- the model")
        print("      never got far enough to exhibit anything. Retry when the backend is healthy.")

    if args.out:
        out = Path(args.out).expanduser()
        out.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "base_url": args.base_url,
                    "temperature": args.temperature,
                    "seed": args.seed,
                    "reasoning": reasoning_fields(args),
                    "reasoning_style": args.reasoning_style,
                    "runs": [
                        {
                            "index": r.index,
                            "steps": r.steps,
                            "error": r.error,
                            "finished": r.finished,
                            "degenerate": r.degenerate,
                            "spiral": r.spiral,
                            "doubt_hits": r.doubt_hits,
                            "doubt_score": r.doubt_score,
                            "post_read_doubt": r.post_read_doubt,
                            "max_shingle_repeat": r.max_shingle,
                            "repeated_reads": r.repeated_reads,
                            "duplicate_tool_calls": r.duplicate_tool_calls,
                            "reasoning_chars": r.reasoning_chars,
                            "reasoning_tokens": r.reasoning_tokens,
                            "completion_tokens": r.completion_tokens,
                            "reasoning_rejected": r.reasoning_rejected,
                            "tool_calls": r.tool_calls,
                            "transcript": r.transcript,
                            "files": r.files,
                        }
                        for r in results
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"transcripts written to {out}")


# --------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("LLM_MODEL", DEFAULT_MODEL))
    p.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", DEFAULT_API_KEY))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", "0")))
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=_env_int("LLM_SEED"))
    p.add_argument("--max-tokens", type=int, default=_env_int("LLM_MAX_TOKENS"))
    p.add_argument(
        "--reasoning-effort",
        default=os.environ.get("LLM_REASONING_EFFORT", "xhigh"),
        help="reasoning budget for the target model: %s, or 'off' (default: xhigh)" % "/".join(REASONING_EFFORTS),
    )
    p.add_argument(
        "--reasoning-max-tokens",
        type=int,
        default=_env_int("LLM_REASONING_MAX_TOKENS"),
        help="explicit thinking-token budget, sent alongside the effort level",
    )
    p.add_argument(
        "--reasoning-style",
        default=os.environ.get("LLM_REASONING_STYLE", "both"),
        choices=["both", "openai", "openrouter", "qwen", "off"],
        help="which spelling to send: reasoning_effort, reasoning{effort}, chat_template_kwargs, both, or none",
    )
    p.add_argument("--runs", type=int, default=1, help="independent runs, fresh workspace each")
    p.add_argument("--max-steps", type=int, default=16, help="max assistant turns per run")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--workspace", default="./_workspaces")
    p.add_argument("--out", default="./repro-transcript.json", help="JSON transcript path ('' to skip)")
    p.add_argument("--system-file", default=None, help="override the system prompt")
    p.add_argument("--prompt-file", default=None, help="override the user prompt")
    p.add_argument("--preview", type=int, default=700, help="chars of reasoning/content to print")
    p.add_argument("-v", "--verbose", action="store_true", help="print full reasoning and tool output")
    args = p.parse_args()

    if not args.api_key:
        print("no API key: pass --api-key or set LLM_API_KEY", file=sys.stderr)
        return 2
    if not args.base_url:
        print("no endpoint: pass --base-url or set LLM_BASE_URL", file=sys.stderr)
        return 2

    results = []
    for i in range(1, args.runs + 1):
        print(f"\n########## run {i}/{args.runs} :: {args.model} ##########", flush=True)
        results.append(run_once(args, i))

    report(results, args)
    return 1 if any(r.error for r in results) else 0


def _env_int(name: str):
    val = os.environ.get(name)
    return int(val) if val else None


if __name__ == "__main__":
    sys.exit(main())
