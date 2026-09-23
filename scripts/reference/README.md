# Reference tooling

Scripts that produce or consume *independent* evidence about this engine: renders
and logits from the reference implementation, and behavioural probes run against
several servers. None of it is run by `scripts/check.sh`.

- `repro_readback.py`: an agent loop with `read_file`/`edit_file` over an email
  workspace. It flags runs where the model claims text it just wrote came back
  different. Stdlib only. Endpoint and key come from `LLM_BASE_URL`/`LLM_API_KEY`
  or flags; nothing is embedded.
- `render_chat_fixtures.py`: renders `test-vectors/references/chat/qwen4exp/*.request.json`
  with the reference chat template. Needs `SHISO_REF_PYTHON` (default
  `~/work/exllamav3/.venv/bin/python`).

GB10 rules: one model server at a time; check `free -g` and `df -h /` first; stop
servers with `pkill -x shiso`.
