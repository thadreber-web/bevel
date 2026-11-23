# bevel

[![CI](https://github.com/thadreber-web/bevel/actions/workflows/ci.yml/badge.svg)](https://github.com/thadreber-web/bevel/actions/workflows/ci.yml)

A terminal coding agent for model servers you already run.

bevel talks to **OpenAI-compatible endpoints only** — [vLLM](https://docs.vllm.ai/) and [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`. No API keys, no per-token billing, no code leaving your network.

> **Status: under construction.** The scaffold, capability probing, and `bevel doctor` work today, as do the provider layer, the tool layer, the agent loop, the policy engine, checkpoints, the TUI, and session persistence — run `bevel` with no subcommand to open it, and `bevel resume` to reopen a session you closed. The loop is covered by tests run against a real vLLM endpoint, and the TUI by Textual's Pilot. Memory and sandboxing arrive in m09–m12 — see [Build order](#build-order). `bevel` refuses to print a friendly lie about anything that isn't built yet: unimplemented subcommands exit non-zero and name the milestone that delivers them, and the TUI prints the seams the loop does not implement instead of letting you assume they exist.

## Why this exists

Most coding agents are thin wrappers over a hosted API. They are impressive demos and poor tools: they stop working when the API changes price or policy, they cannot be pointed at the model you actually want, and their sandboxing is usually a `subprocess.run` with a string comparison in front of it.

bevel is built around three commitments:

**Your hardware is the backend.** A 70B on a DGX Spark, a quantised 14B on a laptop, a speculative-decoding `llama-server` — all first-class. Endpoints are profiles in config, and each profile records what that server actually supports (`/tokenize`, `/embeddings`, parallel tool calls, streaming) rather than assuming the intersection of everyone's servers.

**Enforcement you can verify.** Sandboxing is not a permission check dressed up as security. bevel has three tiers, and the UI reports the level it *verified*, not the level you asked for:

| Tier | Linux | Windows |
|---|---|---|
| **T0** policy jail | always on — path jail, command allowlist, egress policy | same |
| **T1** in-process kernel | Landlock + seccomp + rlimits via `ctypes` | AppContainer + restricted token + low IL + Job Object |
| **T2** OS container | `bwrap` | Windows Sandbox |

T1 needs **zero installs** — pure `ctypes`, no compiled wheels, so it works on a machine you cannot apt-get on. T2 is opt-in (`--untrusted-repo`, m12) and deliberately not the default: a 10–30 s cold start per command is unusable for an agent that runs `pytest` forty times a session, and a disposable container is the wrong shape for a tool whose job is writing real edits to real files.

If a self-test cannot *prove* a tier denies what it claims to deny, bevel drops to the tier it can prove and says so in the UI.

**Three runtime dependencies.** `textual`, `httpx`, `pygments`. All pure Python. `uv tool install bevel` must work on Windows and ARM64 with no build step, so the MCP client, the memory store, and the sandbox bindings are hand-rolled on the standard library rather than pulled from PyPI.

## Install

```powershell
uv tool install bevel
```

Or from a clone:

```powershell
git clone https://github.com/thadreber-web/bevel
cd bevel
uv sync
uv run bevel doctor
```

Requires Python 3.11+.

## Getting started

`bevel doctor` is the honest starting point — it reports what your machine can genuinely enforce, which is worth reading before you trust anything else this project tells you:

Real output from a Windows 11 machine (the Linux columns are the same idea with Landlock and seccomp in place of AppContainer):

```
bevel doctor
============
  platform        Windows 10.0.26200 (AMD64)
  python          3.11.16
  in container    no
  best tier       1 (0=policy-only, 1=kernel, 2=os-container; verified at runtime by self-test)

  sandbox capabilities
    [no ] landlock          not Linux
    [no ] seccomp           not Linux
    [yes] appcontainer      userenv+kernelbase appcontainer exports present
    [yes] job_object        CreateJobObjectW present
    [no ] windows_sandbox   binaries staged but WindowsSandboxService is not installed; enable the Containers-DisposableClientVM feature
    [no ] bwrap             not Linux

  Static probing only. bevel reports the enforcement level it *verified*,
  which can be lower than 'best tier' above.
```

Note the last two lines and the `windows_sandbox` row. `WindowsSandbox.exe` is staged on capable SKUs whether or not the feature is enabled, so bevel checks for the service rather than the executable and refuses to advertise tier 2 it cannot reach.

Then point it at a server. bevel refuses to invent an endpoint, so `bevel` exits with `no endpoint configured` until you give it one. Copy the example and fill in two values:

```powershell
New-Item -ItemType Directory -Force $HOME\.config\bevel
Copy-Item bevel.toml.example $HOME\.config\bevel\config.toml
notepad $HOME\.config\bevel\config.toml
```

Config is layered — defaults → `~/.config/bevel/config.toml` → `./bevel.toml` → `$env:BEVEL_CONFIG` → environment → flags. Only `base_url` and `model` are required:

```toml
# ~/.config/bevel/config.toml
[profile.home]
base_url = "http://127.0.0.1:8000/v1"
model    = "Qwen2.5-Coder-32B-Instruct"
```

`base_url` includes the `/v1` suffix; `model` must match what the server reports, exactly:

```powershell
curl.exe http://127.0.0.1:8000/v1/models
```

A local server usually needs no auth — leave `api_key_env` out and bevel sends no `Authorization` header. If it does need one, `api_key_env = "VLLM_API_KEY"` names the variable to read at request time, so the secret stays out of the file. `bevel.toml.example` documents every key, the environment-only alternative, and the trap where `BEVEL_BASE_URL` silently configures a profile you aren't using.

Verify bevel parsed it before launching the TUI:

```powershell
uv run python -c "from bevel.config.loader import load; c = load(); print(c.sources); print(c.profiles[c.default_profile])"
uv run bevel
```

vLLM needs tool calling enabled, or the model will narrate tool calls instead of emitting them:

```bash
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 32768
```

For llama.cpp, the chat template is what does the tool-call formatting — a template without tool support produces plausible-looking prose instead of calls:

```bash
llama-server -hf ggml-org/Qwen2.5-Coder-32B-Instruct-GGUF \
  --jinja --port 8080 --parallel 4 --cont-batching
```

Per-provider notes, including the parallel-tool-call and grammar (`--grammar-file` / `--grammar`) requirements, are a m14 deliverable. Until they exist, the flags above are the ones that actually matter and `bevel doctor` is the source of truth for what your endpoint supports.

Then run it:

```powershell
uv run bevel
```

That opens the TUI over the current directory. `f2` cycles plan → default → auto mode, `f1` opens the command palette, `Ctrl+C` stops a run (and quits when nothing is running), and focusing a tool card that carries a diff then pressing `enter` opens it full-screen. `bevel undo` from a second terminal rewinds the working tree to the last checkpoint.

Close it and the conversation is not gone: `bevel sessions` lists what is stored, and `bevel resume` reopens the most recent session for this directory (or `bevel resume <id>` for a specific one) with the transcript read back from disk and the model continuing with that history.

## What it does

- **Agent loop** — orient → plan → act → observe, under explicit iteration, tool-call, and token budgets, with cancellation wired end-to-end: `cancel()` abandons the in-flight request *and* the tool tasks, which for the shell tool means the whole process tree. `core/agent.py` is covered by tests run against a real endpoint, and [docs/agent-loop.md](docs/agent-loop.md) explains the decisions.
- **Tools** — read, grep, glob, write, edit (real diff application), shell (timeouts, process-tree kill). Each declares its JSON Schema and risk tier; `core/tools/registry.py` is the single source of truth for the offered list, and [docs/tool-layer.md](docs/tool-layer.md) explains the decisions.
- **Tiered approvals** — `READ` free, `WRITE_WORKSPACE` ask, `WRITE_OUTSIDE` always ask, `EXEC_SAFE` free once allowlisted, `EXEC_UNSAFE` ask, `EXEC_DESTRUCTIVE` always ask with typed confirmation, `NETWORK_EGRESS` ask. Rules are learnable and every decision is logged with the rule that produced it. Checkpoints run even under `--yolo`.
- **TUI** — `bevel` with no subcommand. A streaming transcript (markdown answers, collapsed italic reasoning), one card per tool call coloured by its outcome, a full-screen diff viewer, plan/default/auto modes, an approval dialog, and `Ctrl+C` that genuinely stops the run. Tested with Textual's Pilot against `MockProvider`.
- **Sessions** — every turn, tool call (name, arguments, outcome, observation) and usage row is written through to SQLite as the loop goes, one commit per write, so a turn that completed survives the process. `bevel sessions` lists recent sessions with their workspace, times, turn count and token totals; `bevel resume [id]` reopens one — the transcript you see is read back from the file, and the model continues with that history. Resume trims an unusable tail (a crash between asking for calls and getting results) and says how much it dropped. The reopened transcript is verified against a real endpoint by a test that asks the model to recall a fact from the closed session.
- **Checkpoints** — the working tree, uncommitted work-in-progress included, is committed to a `bevel/checkpoint/<session>` ref before the agent's first write, and `bevel undo` rewinds it from either the CLI or the TUI.
- **Context management** — a project map on the model's first request, and two levels of compaction: L1 truncates stale tool observations to head+tail plus a pointer naming the call to re-run and the session holding the full text, L2 summarises the middle of the transcript and keeps the last N turns byte-identical. Both are triggered by the server's own `prompt_tokens`, never a chars/4 guess. The L3 handoff is not built — it restarts the context from recalled memory (m09a) and todos (m09b), and the loop says so.
- **Memory** — what bevel learned in an earlier session, in `~/.local/share/bevel/memories.db`, off until you turn `[memory]` on. Retrieval is two-stage: FTS5 `MATCH` + `bm25` narrows to ~100 candidates in single-digit milliseconds, and cosine re-ranks only those — a full-corpus cosine scan in pure Python is 5.4 s at 50 000 rows, which is why it is not what happens. Memories are typed (`fact`/`feedback`/`decision`/`reference`) and ranked importance-dominant, so a decision from six months ago beats yesterday's trivia; recency is at most a 2% tiebreaker. A contradicting memory is **superseded, never deleted** — the old row keeps `superseded_by`, stops being recalled, and stays readable with `bevel memories show <id>`. Vectors are opt-in, stored pre-normalised, and tagged with the model that produced them; recall refuses to compare vectors from different models rather than averaging incompatible spaces, and when `/embeddings` is absent it stays lexical and says so on the result. Recalled memories are injected as their own message after the system prompt, budgeted in tokens via `/tokenize`, and name what they dropped rather than silently truncating. The model reaches it with `remember` (`WRITE_OUTSIDE`, so always-ask) and `recall` (`READ`); you reach it with `bevel memories list|show|supersede`. Porter stemming is a known limit: it unifies `index`/`indexes`/`indexed` but not `broke`/`broken`/`break`.
- **MCP** — hand-rolled JSON-RPC over stdio and streamable-HTTP, implementing the subset that matters.
- **Evals** — TOML task specs against a fixture repo with a seeded bug, asserting that tests pass, that the diff contains what it should, and that it stays under token and cost ceilings.

## Testing

```powershell
uv run pytest                # unit + TUI
uv run ruff check .
uv run mypy                  # --strict over src/bevel
```

The TUI is tested with Textual's Pilot, which drives the real widget tree: submits a task, watches tool cards render, approves a dialog via keypress, asserts the diff widget shows the right hunks. Integration tests need a live endpoint and are deselected by default:

```powershell
uv run pytest -m integration
```

## Layout

```
src/bevel/
  cli.py            argparse entry; TUI is the default action
  _capability.py    read-only sandbox probes; never enforces, never mutates
  config/           layered configuration
  providers/        everything that speaks HTTP; base.py is the only seam
  core/agent.py     the loop
  core/tools/       one file per tool; declares JSON Schema + risk tier
  core/context/     budgeting, compaction, project map
  core/policy/      tiered engine, rules, approvals, decision log
  core/sandbox/     T0/T1/T2 backends + self-test
  core/checkpoint/  session snapshot commit on a side ref, restore API
  core/session/     SQLite: turns, tool calls, token + cost accounting
  core/memory/      FTS5 + optional vector recall
  mcp/  evals/  tui/  render/
tests/{unit,integration,tui,evals}/
```

Probing, enforcement, and verification are three separate roles. `_capability.py` only reads. `core/sandbox/` backends only enforce. `selftest.py` only verifies. Keeping them apart is what makes the reported enforcement level trustworthy.

## Build order

| | Milestone |
|---|---|
| ✅ | m01 scaffold, `bevel doctor` |
| ✅ | m02 providers (SSE, `/tokenize`, `MockProvider`, tool-call repair) |
| ✅ | m03 tools |
| ✅ | m04 agent loop |
| ✅ | m05 policy + approvals |
| ✅ | m06 checkpoints |
| ✅ | m07 TUI |
| 🔶 | m08 context management + sessions — project map ✅, L1/L2 compaction ✅, session persistence + resume ✅; the L3 handoff remains (it lands after m09a + m09b) |
| ✅ | m09a memory store + recall |
| ⬜ | m09b todo tool |
| ⬜ | m10 MCP client |
| ⬜ | m11 git + PRs |
| ⬜ | m12 sandbox backends |
| ⬜ | m13 evals |
| ⬜ | m14 docs + polish |

## Deliberately not included

Subagents · codebase indexing / embedding search · web fetch or search · auto-fix-CI · voice input · a VS Code extension · native Anthropic or Google SDKs.

"OpenAI-compatible only" is a boundary, not an oversight: vLLM and llama.cpp are the target, and `providers/base.py` is where any future backend would attach.

## Contributing

Read [`AGENTS.md`](AGENTS.md) first — it carries the working rules, including why the runtime dependency list is capped at three and what bevel does instead of each tempting package.

## License

MIT
