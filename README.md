# bevel

[![CI](https://github.com/thadreber-web/bevel/actions/workflows/ci.yml/badge.svg)](https://github.com/thadreber-web/bevel/actions/workflows/ci.yml)

A terminal coding agent for model servers you already run.

bevel talks to **OpenAI-compatible endpoints only** — [vLLM](https://docs.vllm.ai/) and [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`. No API keys, no per-token billing, no code leaving your network.

> **Status: under construction.** Built and tested today: the scaffold, capability probing and `bevel doctor`; the provider layer; the tool layer; the agent loop; the policy engine and approvals; checkpoints; the TUI; the project map and all three levels of compaction, including the L3 handoff that carries an over-long task into a fresh context; session persistence and resume; memory and recall; the todo list; the `git` tool (allowlisted subcommands, refused escape hatches, commits that keep your identity); and PR create/update on `gh` (allowlisted argv, never pushes, auth is yours). Run `bevel` with no subcommand to open the TUI, `bevel resume` to reopen a session you closed, `bevel todos` to read the plan one left behind, and `bevel pr create` / `bevel pr update` to open or edit a PR from the current branch. The loop is covered by tests run against a real vLLM endpoint, and the TUI by Textual's Pilot.
>
> **Not built:** the MCP client (m10), every sandbox backend above the path jail (m12), and the eval harness (m13). `bevel mcp`, `bevel sandbox` and `bevel eval` exist in `--help` and exit non-zero naming that milestone rather than pretending. `bevel doctor` reports what your machine *could* enforce, which is not the same as what bevel *does* enforce — see [Why this exists](#why-this-exists). `bevel` refuses to print a friendly lie about anything that isn't built: unimplemented subcommands name the milestone that delivers them, and the loop reports the seams it does not implement instead of letting you assume they exist — a list that is empty right now, and pinned empty by tests.

## Why this exists

Most coding agents are thin wrappers over a hosted API. They are impressive demos and poor tools: they stop working when the API changes price or policy, they cannot be pointed at the model you actually want, and their sandboxing is usually a `subprocess.run` with a string comparison in front of it.

bevel is built around three commitments:

**Your hardware is the backend.** A 70B on a DGX Spark, a quantised 14B on a laptop, a speculative-decoding `llama-server` — all first-class. Endpoints are profiles in config, and each profile records what that server actually supports (`/tokenize`, `/embeddings`, parallel tool calls, streaming) rather than assuming the intersection of everyone's servers.

**Enforcement you can verify.** Sandboxing is not a permission check dressed up as security. bevel is designed around two tiers, and the UI will report the level it *verified*, not the level you asked for:

| Tier | Linux | Windows | Status |
|---|---|---|---|
| **T0** policy jail | path jail + argv rule tables (safe allowlist, destructive denylist) | same | ✅ **runs today** |
| **T1** bwrap | spawn via a `bwrap` argv prefix — kernel-enforced read-only-except-workspace, no network | none; `bevel doctor` says so and names WSL2 | ⬜ m12 — designed, not built |

**What actually protects you right now is T0**: the path jail in `core/tools/base.py` and the rule tables in `core/policy/rules.py`. It is a real, tested boundary against a confused model, and it is not security against an adversarial one — symlinks, junctions, 8.3 short names and NTFS alternate data streams all have ways through a textual check, and `core/sandbox/` contains no backend yet. `bevel doctor` probes for the T1/T2 primitives and reports `best tier 1` on a capable Windows machine; that is a statement about your *kernel*, not about bevel. Until m12 ships the backends, bevel enforces T0 and nothing more.

The design that m12 has to honour: T1 is `bwrap`, invoked as an argv prefix — bevel builds the arguments, the kernel does the enforcement, the same division of labour as `gh` in m11b. No hand-rolled `ctypes` syscalls: the Landlock/seccomp/AppContainer ladder this README used to describe was removed from the plan before any of it was built. `bwrap` is a user-installed distro package, joining `git` and `gh` as things bevel probes for rather than assumes — bevel itself still installs nothing. And because *installed is not working* on Ubuntu 24.04 (the `uidmap` helpers and an AppArmor `userns` profile are also required), the probe **runs** bwrap rather than searching for it, and a failed probe degrades to T0 loudly: the exact command, its stderr, and the named fix — the same register as `gh auth login`.

If a self-test cannot *prove* a tier denies what it claims to deny, bevel drops to the tier it can prove and says so in the UI. That self-test is itself an m12 deliverable — there is no `selftest.py` to run today.

**Three runtime dependencies.** `textual`, `httpx`, `pygments`. All pure Python. `uv tool install bevel` must work on Windows and ARM64 with no build step, so the memory store is hand-rolled on the standard library rather than pulled from PyPI — and the same rule is what the not-yet-built MCP client (m10) had to satisfy. The m12 sandbox adds nothing to this list either: it shells out to `bwrap`, a user-installed package, the way the git and PR surfaces shell out to `git` and `gh`.

## Install

Not on PyPI yet — `uv tool install bevel` is what m14 will make true, and today it
fails. Install from a clone:

```powershell
git clone https://github.com/thadreber-web/bevel
cd bevel
uv sync
uv run bevel doctor
```

Requires Python 3.11+.

## Getting started

`bevel doctor` is the honest starting point — it reports what your machine is *capable of* enforcing, which is worth reading before you trust anything else this project tells you:

Real output from a Windows 11 machine (the Linux columns are the same idea with Landlock and seccomp in place of AppContainer):

```
bevel doctor
============
  platform        Windows 10.0.26300 (AMD64)
  python          3.11.16
  in container    no
  best tier       1 (0=policy-only, 1=kernel, 2=os-container; what your OS could enforce)

  sandbox capabilities
    [no ] landlock          not Linux
    [no ] seccomp           not Linux
    [yes] appcontainer      userenv+kernelbase appcontainer exports present
    [yes] job_object        CreateJobObjectW present
    [no ] windows_sandbox   binaries staged but WindowsSandboxService is not installed; enable the Containers-DisposableClientVM feature
    [no ] bwrap             not Linux

  Tier 2 (Windows Sandbox / bwrap) is opt-in and is not used for ordinary
  tool calls: a 10-30s cold start per command is unusable, and the
  container is disposable while an agent must write real edits. It is
  enabled by config when m12 delivers the container backends.

  Static probing only. No sandbox backend ships yet: bevel enforces the T0
  policy jail, whatever 'best tier' above says. m12 adds the backends and
  the self-test that would justify a higher number.
```

Read that output for what it is. It says `best tier 1`, which describes your kernel, not bevel — the self-test that would justify a higher claim is an m12 deliverable, so today nothing is verified and the only tier in force is T0.

Note also the `windows_sandbox` row. `WindowsSandbox.exe` is staged on capable SKUs whether or not the feature is enabled, so bevel checks for the service rather than the executable and refuses to advertise tier 2 it cannot reach.

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

Per-provider notes, including the parallel-tool-call and grammar (`--grammar-file` / `--grammar`) requirements, are in [docs/provider-notes.md](docs/provider-notes.md) — probed against live servers rather than copied from their docs. The flags above are the ones that actually matter; `bevel doctor` is the source of truth for what your machine can enforce.

Then run it:

```powershell
uv run bevel
```

That opens the TUI over the current directory. `f2` cycles plan → default → auto mode, `f1` opens the command palette, `Ctrl+C` stops a run (and quits when nothing is running), and focusing a tool card that carries a diff then pressing `enter` opens it full-screen. `bevel undo` from a second terminal rewinds the working tree to the last checkpoint.

Close it and the conversation is not gone: `bevel sessions` lists what is stored, and `bevel resume` reopens the most recent session for this directory (or `bevel resume <id>` for a specific one) with the transcript read back from disk and the model continuing with that history. `bevel todos` (or `bevel todos <id>`) prints the plan that session left behind.

## What it does

- **Agent loop** — orient → plan → act → observe, under explicit iteration, tool-call, and token budgets, with cancellation wired end-to-end: `cancel()` abandons the in-flight request *and* the tool tasks, which for the shell tool means the whole process tree. `core/agent.py` is covered by tests run against a real endpoint, and [docs/agent-loop.md](docs/agent-loop.md) explains the decisions.
- **Tools** — read, grep, glob, recall, todo, write, edit (real diff application), remember, shell (timeouts, process-tree kill), git (allowlisted subcommands, refused repo-relocation and config-injection argv, an `Assisted-by: bevel` commit trailer, your git identity left alone). Each declares its JSON Schema and risk tier; `core/tools/registry.py` is the single source of truth for the offered list, and [docs/tool-layer.md](docs/tool-layer.md) explains the decisions.
- **Tiered approvals** — `READ` free, `WRITE_WORKSPACE` ask, `WRITE_OUTSIDE` always ask, `EXEC_SAFE` free once allowlisted, `EXEC_UNSAFE` ask, `EXEC_DESTRUCTIVE` always ask with typed confirmation, `NETWORK_EGRESS` ask. Rules are learnable and every decision is logged with the rule that produced it. `f2` cycles plan → default → auto; auto mode still asks for `WRITE_OUTSIDE` and `EXEC_DESTRUCTIVE`, and checkpoints are taken regardless of how approvals were answered. `AutoApprover` exists for callers that want no prompts at all, but there is no CLI flag for it yet — the TUI always asks.
- **TUI** — `bevel` with no subcommand. A streaming transcript (markdown answers, collapsed italic reasoning), one card per tool call coloured by its outcome, a full-screen diff viewer, plan/default/auto modes, an approval dialog, and `Ctrl+C` that genuinely stops the run. Tested with Textual's Pilot against `MockProvider`.
- **Sessions** — every turn, tool call (name, arguments, outcome, observation) and usage row is written through to SQLite as the loop goes, one commit per write, so a turn that completed survives the process. `bevel sessions` lists recent sessions with their workspace, times, turn count and token totals; `bevel resume [id]` reopens one — the transcript you see is read back from the file, and the model continues with that history. Resume trims an unusable tail (a crash between asking for calls and getting results) and says how much it dropped. The reopened transcript is verified against a real endpoint by a test that asks the model to recall a fact from the closed session.
- **Checkpoints** — the working tree, uncommitted work-in-progress included, is committed to a `bevel/checkpoint/<session>` ref before the agent's first write, and `bevel undo` rewinds it from either the CLI or the TUI.
- **Context management** — a project map on the model's first request, and three levels of compaction: L1 truncates stale tool observations to head+tail plus a pointer naming the call to re-run and the session holding the full text, L2 summarises the middle of the transcript and keeps the last N turns byte-identical, and L3 hands off. All three are triggered by the server's own `prompt_tokens`, never a chars/4 guess, and each is idempotent — "this level changed nothing" is how the loop knows to escalate. L3 is what happens when neither of the first two can fit it: the loop writes a brief whose goal is your task verbatim, whose files-touched list comes only from calls that actually mutated the workspace and whose todos come from the todo store, and whose decisions and open questions are the previous instance's own account, then rebuilds the context in the same shape the run opened with — a freshly built project map, recalled memory, the brief, the last two turns — and carries on. The model is never asked to list files, so it cannot invent one. A handoff costs a generation and therefore an iteration, and is capped per run; hitting the cap stops with a named reason rather than letting a model that never converges hand off its way through the token budget. If the brief cannot be written, comes back empty, or is itself too big to fit, the run stops exactly as it used to and says why — bevel will not send a request it knows the server will reject.
- **Memory** — what bevel learned in an earlier session, in `~/.local/share/bevel/memories.db`, off until you turn `[memory]` on. Retrieval is two-stage: FTS5 `MATCH` + `bm25` narrows to ~100 candidates in single-digit milliseconds, and cosine re-ranks only those — a full-corpus cosine scan in pure Python is 5.4 s at 50 000 rows, which is why it is not what happens. Memories are typed (`fact`/`feedback`/`decision`/`reference`) and ranked importance-dominant, so a decision from six months ago beats yesterday's trivia; recency is at most a 2% tiebreaker. A contradicting memory is **superseded, never deleted** — the old row keeps `superseded_by`, stops being recalled, and stays readable with `bevel memories show <id>`. Vectors are opt-in, stored pre-normalised, and tagged with the model that produced them; recall refuses to compare vectors from different models rather than averaging incompatible spaces, and when `/embeddings` is absent it stays lexical and says so on the result. Recalled memories are injected as their own message after the system prompt, budgeted in tokens via `/tokenize`, and name what they dropped rather than silently truncating. The model reaches it with `remember` (`WRITE_OUTSIDE`, so always-ask — unless you set `[memory] auto_accept = true`, which exempts `remember` alone, never a second out-of-workspace tool and never a path that escapes the jail, and is logged as `config:memory:auto_accept`) and `recall` (`READ`); you reach it with `bevel memories list|show|supersede`. Porter stemming is a known limit: it unifies `index`/`indexes`/`indexed` but not `broke`/`broken`/`break`.
- **Todos** — the model keeps its plan with one `todo` tool that **replaces** the whole list each time, so marking a step done means resending the list with that row's status changed. It lives in a `todos` table beside the other session rows in `sessions.db`, scoped to the session and cascade-deleted with it, ordered by an explicit `position` rather than by insertion order. It is declared `READ`, not `WRITE_WORKSPACE`: the row is bookkeeping about the agent's own plan, not a byte of your workspace, and an approval prompt per plan update would train you to approve without reading — the justification is written into the module and pinned by a test. The schema is strict where a small model is loose: `additionalProperties: false` at both levels, so an invented `id` or a status spelled `finished` is refused before it reaches the store, and `maxItems: 32` is set by what a model reproduces faithfully in one call rather than by anything SQLite cares about. An empty list legitimately means *cleared*. `is_parallel_safe` is off, because two concurrent whole-list replaces race. The plan is yours to read with `bevel todos [id]` — read-only, refuses to create the database, and offers no `clear`, because clearing is the model's job and a spare one would be a mistyped id away from destroying the plan you came to read. A TUI panel for it is not built.
- **MCP** — *not built (m10).* `bevel mcp` exits non-zero naming the milestone. The plan is hand-rolled JSON-RPC over stdio and streamable-HTTP covering `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read` — nothing in `src/bevel/mcp/` yet but the docstring that says so.
- **Evals** — *not built (m13).* The plan is TOML task specs against a fixture repo with a seeded bug, asserting that tests pass, that the diff contains what it should, and that it stays under token and cost ceilings. `src/bevel/evals/` and `tests/evals/` are empty.

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
  core/agent.py     the loop, and the budgets it runs under
  core/tools/       one file per tool; declares JSON Schema + risk tier
  core/context/     compaction (L1/L2) and the project map
  core/policy/      tiered engine, rules, approvals, decision log
  core/sandbox/     empty until m12 — no backends, no self-test
  core/checkpoint/  session snapshot commit on a side ref, restore API
  core/session/     SQLite: turns, tool calls, token + cost accounting
  core/memory/      FTS5 + optional vector recall
  mcp/              empty until m10
  evals/            empty until m13
  tui/  render/
tests/{unit,integration,tui}/   tests/evals/ is empty until m13
```

Probing and enforcement are meant to be separate roles, which is why `_capability.py` only reads and never enforces. The other half of that separation — a `core/sandbox/` backend that only enforces, and a `selftest.py` that only verifies — is the m12 design, and neither file exists yet. Until they do, treat the enforcement level as T0 regardless of what the probes report.

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
| ✅ | m08 context management + sessions — project map, L1/L2/L3 compaction (including the handoff), session persistence + resume |
| ✅ | m09a memory store + recall |
| ✅ | m09b todo tool |
| ⬜ | m10 MCP client |
| ✅ | m11a git tool — allowlisted subcommands, refused escape hatches, commit trailer |
| ✅ | m11b PR create/update (`gh`-backed) — human-only CLI, allowlisted argv, never pushes |
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
