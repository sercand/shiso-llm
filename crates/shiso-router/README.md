# shiso-router

An HTTP router that proxies LLM API requests to third-party backend services. Clients talk to the router with the provider wire formats they already use, so official SDKs work unchanged — just point `base_url` at the router and use a router-issued API token.

## Supported endpoints

| Endpoint | API |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/responses` | OpenAI Responses |
| `POST /v1/embeddings` | OpenAI Embeddings |
| `POST /v1/messages` | Anthropic Messages |

The router picks a backend based on the `model` field in the request body, swaps the client's router token for the backend's real API key, optionally renames the model, and streams the response through untouched.

## Features

- **Model-based routing** — each inbound model name maps to one or more backends.
- **Model aliases** — a route can answer to several inbound names. Aliases resolve to the model name first, so logs, costs, key allowlists and the name sent upstream all use the canonical one, and aliased traffic balances as a single stream.
- **Load balancing** — smooth weighted round-robin across a model's backends.
- **Failover** — on connection failure or upstream 5xx, the next backend is tried. Safe by construction: nothing is sent to the client until an upstream has answered. 4xx responses are relayed immediately (no retry).
- **Model renaming** — a route target can rewrite the model name before forwarding (e.g. inbound `gpt-4o` becomes `llama-3.3-70b` on a local vLLM backend).
- **Raw streaming** — response bodies (including SSE) are piped through as raw bytes with no re-serialization or buffering. Request bodies are forwarded byte-for-byte unless a rename or usage injection applies.
- **Key auth & controls** — clients authenticate with router-issued keys (`Authorization: Bearer` or `x-api-key`, both accepted), stored in SQLite as SHA-256 hashes. Keys can be disabled/re-enabled and restricted to a model allowlist.
- **Request logs** — every proxied request is recorded (model, backend, status, latency, token usage incl. cached tokens, cost, finish reason, errors) without slowing the streaming path: the response is observed by a tee and rows are written by an async batching task.
- **Usage limits** — per-key credit (USD) and token-count limits, each with a `never`/`daily`/`weekly`/`monthly` UTC reset window. Token limits make free internal backends (vLLM) boundable too. Exhausted keys get `429 insufficient_quota`.
- **Model pricing** — per-model (or per-target) USD-per-million-token rates, including discounted cache-read and Anthropic cache-write rates, drive logged costs and credit limits.
- **Device-aware scheduling** — backends that share one accelerator can be grouped so that at most one of them generates at a time, with per-backend concurrency limits and generated-token caps. See [Device-aware scheduling](#device-aware-scheduling).
- **QoS deadlines & load shedding** — a route can declare how long a request may be *predicted* to queue on a busy device; over that, it is routed to a remote fallback target instead of waiting.
- **Live catalog** — exclusive groups, backends and model routes live in the database and are created/edited/deleted through the admin API or UI; changes take effect on the next request, no restart.
- **Admin API + UI** — a JSON admin API (keys, groups, backends, models, logs, usage aggregates) plus a built-in single-page admin UI at `/admin/ui`, optionally on a separate listener.

## Quick start

```bash
cp config.example.toml config.toml   # adjust listen address, db path, and admin token
export LLM_ROUTER_ADMIN_TOKEN=$(openssl rand -hex 32)
export OPENAI_API_KEY=sk-...

# Run via the shiso CLI:
cargo run --release -p shiso-server -- router --config config.toml
# Or run the standalone shiso-router binary:
cargo run --release -p shiso-router -- --config config.toml
```

Create a client token:

```bash
curl -s -X POST localhost:8080/admin/tokens \
  -H "Authorization: Bearer $LLM_ROUTER_ADMIN_TOKEN" \
  -H "content-type: application/json" \
  -d '{"name":"dev"}'
# => {"id":"...","name":"dev","created_at":"...","token":"sk-..."}
# The token is shown exactly once — only its hash is stored.
```

Make a (streaming) request:

```bash
curl -N -X POST localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -H "content-type: application/json" \
  -d '{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

Or with the official SDKs:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-...")

from anthropic import Anthropic
client = Anthropic(base_url="http://localhost:8080", api_key="sk-...")
```

## Configuration

See [`config.example.toml`](config.example.toml) for an annotated example.

The configuration file defines network and storage settings:

```toml
# Address to listen on.
listen = "127.0.0.1:8080"

# Optional: serve the admin API + UI on a separate address. When set, the
# /admin routes (and the UI) are served ONLY here and are absent from the
# main listener. When omitted, admin shares `listen`.
# admin_listen = "127.0.0.1:9090"

# SQLite database file for API tokens, routing catalog, and request logs.
sqlite_path = "shiso-router.db"

# Maximum accepted request body size in bytes (default 1 GiB).
max_body_bytes = 1073741824

# Admin token protecting /admin/* endpoints and the UI.
# Provide exactly one of admin_token / admin_token_env.
admin_token_env = "LLM_ROUTER_ADMIN_TOKEN"
# admin_token = "not-recommended-in-files"
```

Command-line options override settings in the config file:

```bash
# Via the shiso CLI:
shiso router --config config.toml
shiso router --listen 0.0.0.0:8080 \
             --sqlite-path /var/lib/shiso-router.db \
             --max-body-bytes 5242880 \
             --admin-token-env LLM_ROUTER_ADMIN_TOKEN

# Or via the standalone shiso-router binary:
shiso-router --config config.toml
```

Run `shiso router --help` (or `shiso-router --help`) for the full list of options.

Backends, models, and exclusive groups are not set in the configuration file: they live in the SQLite database and are managed dynamically through the admin API and Web UI (`/admin/ui`, `/admin/backends`, `/admin/models`, `/admin/groups`). Changes take effect on the next request without a restart. Secrets can stay in the environment: a backend created via the admin UI may reference `api_key_env` instead of storing the plaintext key in the database.

Several inference servers can be resident on one accelerator at the same time without
contending for *memory* while still contending badly for *compute*. Two large models that both
use speculative decoding are the clearest case: speculation only pays off when there is idle
compute to spend on draft/verify, so if both generate at once they wreck each other's
throughput. Within one model, concurrent requests are fine — the engine's own continuous
batching handles those.

An **exclusive group** names one such device. At most one `grouped` member of a group holds the
group's *lease* — and so may generate — at a time; the rest queue, or shed to a remote fallback
if they would have to wait past their deadline.

Three mechanisms, deliberately separate:

| | what it does |
|---|---|
| **Exclusive group** | at most one member generating at a time, with timed handovers between them |
| **Concurrency slot** | a per-backend cap on in-flight requests (`max_concurrency`), applied to every backend, grouped or not |
| **QoS deadline** | a per-request budget for *predicted* queue time; over it, the request is shed to a fallback instead of enqueued |

Each backend has exactly one **scheduling class**:

- `unrestricted` (default) — not on a contended device. Remote providers, a Mac mini, anything
  with the accelerator to itself.
- `exempt` — on a device that has a group, but never waits for the lease. For work cheap enough
  to interleave harmlessly, like a small embedding model. Behaves identically to `unrestricted`
  at runtime; the two stay distinct so the admin UI can say *why* a backend on a contended
  device is unscheduled.
- `grouped` — full gang scheduling. Requires both `exclusive_group` and `max_tokens_cap`.

```toml
[[exclusive_groups]]
id = "dgx"
min_hold_ms = 5000       # anti-thrash floor
max_hold_ms = 30000      # anti-starvation ceiling
drain_timeout_ms = 45000 # watchdog

[[backends]]
name = "moe-local"
kind = "openai"
base_url = "http://dgx:8000"
scheduling_class = "grouped"
exclusive_group = "dgx"
max_concurrency = 4      # the engine batches this many fine
max_tokens_cap = 1024    # required for grouped — see below
tok_s_estimate = 90.0    # seed for the throughput estimator

[[models]]
name = "qwen-moe"
  [models.qos]
  max_wait_ms = 2000              # interactive deadline
  background_max_wait_ms = 60000  # x-router-qos: background
  [[models.targets]]
  backend = "moe-local"
  [[models.targets]]
  backend = "openrouter"
  model = "qwen/qwen3-30b-a3b"
  fallback_only = true            # reached only on a shed
```

### `max_tokens_cap` is the tuning knob

`max_tokens_cap` clamps the generated-token limit on every request to a grouped backend
(`max_completion_tokens`/`max_tokens` on chat, `max_output_tokens` on responses, `max_tokens` on
messages), **injecting** it when the client sent no limit at all. The client sees the effect as
an ordinary `finish_reason: "length"`. A fallback to a remote provider is never clamped — it
sends the client's own value.

This is not a cost control. It is what makes the whole scheme work: the cap bounds how long a
request can occupy the device, which bounds how long the device takes to drain when the lease
moves, which bounds every *other* member's queue wait. Without it, drain time is unbounded and
every wait estimate built on it is meaningless — which is why a `grouped` backend without one is
rejected outright.

Set it **asymmetrically**, in inverse proportion to each model's speed. On the example above the
dense model runs ~3.5× slower per token than the MoE model, so it gets 512 against the MoE's
1024 — giving the two a comparable worst-case drain, and therefore a comparable worst-case
impact on each other.

### Tuning

- **High switch rate *and* high shed rate** → raise `min_hold_ms`. The device is spending its
  time handing over instead of generating.
- **One backend's queue wait far above the other's** → raise the fairness pressure with a lower
  `max_hold_ms`, or lower the *other* backend's `max_tokens_cap` so its epochs are shorter.
- **Shed rate near 100% for one model** → its deadline is below the physical drain floor. Lower
  the cap, raise the deadline, or accept that this model lives remotely.

**Shedder vs. scheduler.** `min_hold_ms = 5000` against an interactive `max_wait_ms = 2000` means
a request arriving while the *other* model holds the device essentially always sheds: the
predicted wait includes the remaining hold, which exceeds the deadline by construction. At those
settings the system is a **load shedder** — "first model to grab the device serves locally,
everyone else goes to OpenRouter" — not a time-slicing scheduler. That is a coherent design, but
choose it deliberately. `min_hold_ms ≈ 1000` with `max_wait_ms ≈ 4000` gives actual interleaving.

### Deadlines and shedding

A request's deadline is the route's `qos.max_wait_ms` by default. Clients can select or narrow
it per request:

| Header | Effect |
|---|---|
| `x-router-qos: background` | use `qos.background_max_wait_ms` instead — batch work is happy to wait a minute for local hardware |
| `x-router-max-wait-ms: <int>` | name a shorter deadline; clamped to the class deadline, so a client can only make itself shed *sooner*, never hold a slot longer |

Neither header is in the outbound allowlist, so neither reaches a backend. With no deadline
configured and none requested, a request waits indefinitely — exactly the pre-scheduling
behavior.

Before a request is enqueued, the router estimates its wait:

```
predicted_wait(b) = time_to_reach_epoch(b) + queue_ahead_delay(b)
```

— the in-flight work on the device now, plus the remainder of the current minimum hold, plus
`min_hold_ms` and one batch for every epoch queued ahead, plus the batches of `b`'s own waiters
that must clear first. Request durations come from a per-backend ring of recently observed
tokens/second, read at a **low quantile** rather than the mean (with speculative decoding the
acceptance rate is prompt-dependent, so the mean systematically under-predicts and admits
requests that then miss their deadline), seeded from `tok_s_estimate` until enough requests have
been seen.

Over the deadline, the request is **shed**: the router tries the next non-fallback target that
is not on a device it has already shed from (shedding `moe-local → dense-local` would be
pointless — they contend for the same lease), then the route's `fallback_only` targets, weighted
round-robin among themselves. With no fallback at all, `on_shed_without_fallback` decides:
`wait` (the default — queue anyway) or `error` (`503 overloaded_error`). Silently dropping the
deadline is worse than either.

Shed requests are logged and billed exactly like any other, priced from the fallback target's
own pricing — the point of a remote fallback is that it costs real money, so it has to show up
in cost reporting. The shed reason lands in the log row's `error` field, the same channel
failover reasons already use, so it appears in the existing logs UI.

Live group state (epoch starts, drains, force-flips) goes to `tracing`; a stuck drain logs at
ERROR with the wedged backend, elapsed time and in-flight count before force-switching.

### Running several engines on one device

`gpu_memory_utilization` in vLLM is a fraction of **total** device memory, not of free memory.
With three engines on one accelerator each must set it explicitly so the sum stays under ~0.95 —
roughly `0.36 / 0.34 / 0.03` for a 42 GB MoE model, a 40 GB dense model and a 2 GB embedder on
128 GB of unified memory, leaving headroom. Otherwise whichever engine starts second tries to
claim memory the first already holds and fails. SGLang's `--mem-fraction-static` behaves the
same way.

The router deliberately does **not** load, unload, or sleep/wake models: all of them stay
resident, because memory is not the constraint here — compute is.

## Admin UI

Open `/admin/ui` (on the admin listener if `admin_listen` is set, otherwise the main one) and sign in with the admin token. It covers key management (create with limits/allowlist, edit, disable, delete), per-key usage stat cards, a 30-day token chart split by input/output, backend, model and exclusive-group editors (including scheduling class, concurrency, token caps and QoS deadlines), and a filterable request log with a detail drawer. The page is a single embedded file that talks to the JSON API below — a custom UI (e.g. Next.js) can use the same API.

## Admin API

All endpoints require `Authorization: Bearer <admin token>`. When `admin_listen` is configured they are served only on that listener.

| Method | Path | Description |
|---|---|---|
| `POST` | `/admin/tokens` | Create a key. Body: `{"name", "credit_limit"?, "credit_limit_reset"?, "token_limit"?, "token_limit_reset"?, "allowed_models"?}`. Returns the plaintext token once. |
| `GET` | `/admin/tokens` | List keys with limit settings and current-window usage — never secrets. |
| `GET` | `/admin/tokens/{id}` | Key detail with today / this-week / this-month / all-time usage aggregates. |
| `PATCH` | `/admin/tokens/{id}` | Update any of name, `disabled`, limits, resets, `allowed_models` (`null` clears). Takes effect immediately. |
| `DELETE` | `/admin/tokens/{id}` | Delete a key (request logs are retained). Takes effect immediately. |
| `GET` | `/admin/logs` | Request logs; filters `token_id`, `model`, `status`, `since`/`until` (RFC3339); keyset pagination via `limit` + `cursor`/`next_cursor`. |
| `GET` | `/admin/logs/{id}` | One log row. |
| `GET` | `/admin/usage` | Aggregates (`requests`, tokens, `cost`) with `group_by=day\|model\|token` and the same filters. |
| `GET` | `/admin/models` | Model routes with their pricing and targets. |
| `POST` | `/admin/models` | Create a model route. Body: `{"name", "aliases"?, "pricing"?, "targets":[{"backend", "model"?, "weight"?, "pricing"?}]}`. |
| `GET` | `/admin/models/{name}` | One model route. |
| `PUT` | `/admin/models/{name}` | Replace a model's aliases, pricing and target list wholesale. Names are not renameable (aliases are). |
| `DELETE` | `/admin/models/{name}` | Delete a model route; requests for that name then 404. |
| `GET` | `/admin/backends` | Backends with `has_api_key`/`api_key_env` — stored secrets are never returned. |
| `POST` | `/admin/backends` | Create a backend. Body: `{"name", "kinds":["openai"\|"anthropic"], "base_url", "api_key"?, "api_key_env"?, "anthropic_version"?, "inject_usage"?}`. |
| `GET` | `/admin/backends/{name}` | One backend. |
| `PATCH` | `/admin/backends/{name}` | Update any field; absent = unchanged, `null` = clear. Setting one key source clears the other. |
| `DELETE` | `/admin/backends/{name}` | Delete a backend; `409` while a model still targets it. |
| `GET` | `/admin/groups` | Exclusive groups with their hold/drain timings. |
| `POST` | `/admin/groups` | Create a group. Body: `{"id", "min_hold_ms", "max_hold_ms", "drain_timeout_ms"}` (`min ≤ max ≤ drain`, all > 0). |
| `GET` | `/admin/groups/{id}` | One group. |
| `PUT` | `/admin/groups/{id}` | Retune a group's timings in place; running leases and queues are preserved. |
| `DELETE` | `/admin/groups/{id}` | Delete a group; `409` while a backend still names it. |

Backends additionally accept `scheduling_class`, `exclusive_group_id`, `max_concurrency`,
`max_tokens_cap` and `tok_s_estimate`; models accept `qos_max_wait_ms`,
`qos_background_max_wait_ms` and `on_shed_without_fallback`, and their targets accept
`fallback_only`. See [Device-aware scheduling](#device-aware-scheduling).

Catalog writes are validated (unknown backend in a target, zero weight, both key sources set, unset `api_key_env`, …) and are applied to the live routing table before the response returns; the previous table stays in use if the rebuild fails.

Reset intervals are `never` (default), `daily`, `weekly` (Monday 00:00 UTC), or `monthly` — all windows are UTC. There is also an unauthenticated `GET /healthz` on each listener, and `GET /admin/ui` (the page itself carries no data).

## Behavior details

- **Header handling** — outbound headers are built from an allowlist (`content-type`, `accept`, `anthropic-version`, `anthropic-beta`); client credentials and hop-by-hop headers are never forwarded. Upstream response headers are relayed minus hop-by-hop headers.
- **Timeouts** — 10 s connect timeout, 120 s read (idle-stall) timeout, no total timeout so long-running SSE streams are never cut off.
- **Errors** — rendered as provider-style JSON (`{"error":{"message","type"}}`): 400 invalid body, 401 bad token, 403 disabled key / model not allowed, 404 unknown model, 429 limit exhausted, 503 QoS deadline missed with no fallback (`on_shed_without_fallback = "error"`), 502 all backends unreachable. If every backend returns 5xx, the last upstream 5xx response is relayed verbatim.
- **Usage capture** — streaming OpenAI-style chat requests get `stream_options.include_usage: true` injected so usage appears in the stream (the extra final chunk passes through to the client, as OpenRouter does; set `inject_usage = false` on backends that reject `stream_options`). Anthropic and Responses streams already carry usage. Token counts are normalized so `prompt_tokens` always includes cached tokens; costs bill cache reads/writes at their configured rates.
- **Embeddings** — `/v1/embeddings` routes like any other model name, so a batch job can be renamed onto a cheaper provider and stays inside the same key limits. Responses report only `prompt_tokens`, billed at `input_per_mtok` (set `output_per_mtok = 0.0`). Bodies are still streamed straight through: a multi-megabyte batch response is never buffered, and usage is recovered from the ends of the body.
- **Limit semantics** — checks happen before dispatch against in-memory window counters (warmed from the log table at startup); usage accrues when the response finishes streaming, so concurrent in-flight requests can overshoot a limit by at most their own usage. Requests with no extractable usage accrue nothing but are still logged (with a note in `error`).

## Development

```bash
cargo test -p shiso-router      # unit + integration tests (mock backends, no network)
cargo clippy -p shiso-router --all-targets
```

## Not yet implemented

Rate limiting (requests/minute), TLS termination (run behind a reverse proxy), hot config reload, per-backend health checks / circuit breaking, retrying 429s on alternate backends, log retention/pruning.

On the scheduling side, deliberately out of scope: model load/unload or vLLM sleep/wake (every
model stays resident), preempting a running generation for scheduling reasons (only the drain
watchdog interrupts anything, and only as failure recovery), aborting straggler upstream
requests on a drain timeout, cross-device or global placement (a group is device-local), and a
dedicated scheduler metrics endpoint — live group state is visible through `tracing`, and
per-request outcomes through `request_logs` and `/admin/usage`.
