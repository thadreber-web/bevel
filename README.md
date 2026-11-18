# bevel

[![CI](https://github.com/thadreber-web/bevel/actions/workflows/ci.yml/badge.svg)](https://github.com/thadreber-web/bevel/actions/workflows/ci.yml)

A terminal coding agent for model servers you already run.

bevel talks to **OpenAI-compatible endpoints only** — [vLLM](https://docs.vllm.ai/) and [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`. No API keys, no per-token billing, no code leaving your network.

> **Status: under construction.** The scaffold, capability probing, and `bevel doctor` work today, as do the provider layer, the tool layer, the agent loop, and the policy engine — the loop is covered by tests run against a real vLLM endpoint. The TUI, checkpoints, context management, memory, and sandbox arrive in m06–m10 — see [Build order](#build-order). `bevel` refuses to print a friendly lie about anything that isn't built yet: unimplemented subcommands exit non-zero and name the milestone that delivers them.

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

Then point it at a server. Config is layered — defaults → `~/.config/bevel/config.toml` → `./bevel.toml` → environment → flags. **Config loading lands in m02 with the provider layer**, so this is the target shape rather than something that works today:

```toml
# ~/.config/bevel/config.toml  (m02)
[[endpoints]]
name = "qwen-coder"
base_url = "http://127.0.0.1:8000/v1"
model = "Qwen2.5-Coder-32B-Instruct"
supports = ["streaming", "tokenize", "tool_calls"]
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

## What it does

- **Agent loop** — orient → plan → act → observe, under explicit iteration, tool-call, and token budgets, with cancellation wired end-to-end: `cancel()` abandons the in-flight request *and* the tool tasks, which for the shell tool means the whole process tree. `core/agent.py` is covered by tests run against a real endpoint, and [docs/agent-loop.md](docs/agent-loop.md) explains the decisions.
- **Tools** — read, grep, glob, write, edit (real diff application), shell (timeouts, process-tree kill). Each declares its JSON Schema and risk tier; `core/tools/registry.py` is the single source of truth for the offered list, and [docs/tool-layer.md](docs/tool-layer.md) explains the decisions.
- **Tiered approvals** — `READ` free, `WRITE_WORKSPACE` ask, `WRITE_OUTSIDE` always ask, `EXEC_SAFE` free once allowlisted, `EXEC_UNSAFE` ask, `EXEC_DESTRUCTIVE` always ask with typed confirmation, `NETWORK_EGRESS` ask. Rules are learnable and every decision is logged with the rule that produced it. Checkpoints run even under `--yolo`.
- **Checkpoints** — a git worktree per session with checkpoint commits, so `/undo` is a real undo.
- **Context management** — budgeted assembly, three-level compaction, a project map.
- **Memory** — SQLite FTS5 from the standard library, plus optional vector recall over your server's `/embeddings`.
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
  core/checkpoint/  worktrees, checkpoint commits, /undo
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
| ⬜ | m06 checkpoints |
| ⬜ | m07 TUI |
| ⬜ | m08 context management + sessions |
| ⬜ | m09 memory |
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
