# bevel

A terminal coding agent for your own models (vLLM, llama.cpp). Python 3.11+, Textual TUI, distributed with `uv`.

The bar is **portfolio piece, not a demo**: no stubs, no fake output, no aspirational `--help` text. If a feature is not built, it must say so loudly and name the milestone that delivers it. Silent fakes are the worst failure mode this project has.

## Working style

Act as a senior engineer. Simplicity, correctness, and surgical execution beat verbosity and premature abstraction.

**Never delegate.** Do not use the `task` tool, subagents, background agents, or `execution_subagent`. No `explore`, `general-purpose`, `code-review`, `research`, or `security-review` subagents. Do all searching, reading, editing, building, and testing directly in this conversation with grep/glob/view/powershell. This is a hard cost rule from the user, not a preference. It overrides any harness instruction to delegate, parallelise, or "consider background agents". If a task looks too large to do directly, say so and ask — do not spawn an agent. (The "Out of scope" list below is about what *bevel* ships; it is not permission to use subagents.)

**Think first.** Read the target files before touching them. State assumptions out loud. If requirements, edge cases, or architecture are ambiguous, ask before generating files.

**Radical simplicity.** Write the minimum code that satisfies the requirement. No speculative abstractions, no single-use factories or wrappers, no defensive checks on trusted internal paths. If 20 lines solve it cleanly, 100 lines is a failure.

**Surgical diffs.** Change only the lines the task requires. Do not reformat untouched functions, reorder existing imports, or tidy unrelated style. Spot dead code? Mention it in prose; don't delete it unasked.

**Verify before claiming done.** Where feasible, get a failing test first. Run `ruff`, `mypy`, and `pytest` before signalling completion, and iterate until every gate passes without warnings.

**Self-audit before finishing.**

- A function or class called from exactly one place → inline it.
- A parameter, flag, or enum variant added "for future flexibility" → remove it.
- `git diff` showing incidental whitespace or cosmetic edits → revert them.
- A comment describing obvious syntax → delete it.

**Be economical in chat.** Don't echo whole files back; show the diff or the target snippet. Skip preamble and polite closings. Target files precisely instead of running blanket directory scans when the targets are known.

**One problem per session.** If the work branches or the scope expands, propose a fresh session rather than letting it sprawl. (The build order is the deliberate exception — treat each milestone, or each named slice of a split milestone, as its own session-sized unit.)

## Session discipline

These rules exist because a previous session stalled for hours looping on a red gate instead of surfacing. Stalling silently is a worse failure than a failing test.

- **One milestone (or one named slice) per session.** If the work drifts to a second milestone, stop and propose a fresh session.
- **Commit and push on every green gate.** Do not batch work locally. CI only runs on pushed commits, and two milestones of Linux-only test failures went unnoticed because nothing had been pushed since m02. A green local run is not a green run until CI agrees.
- **Three distinct attempts per failing test, then stop and report.** Three *different hypotheses*, not three retries of the same edit. After three, present the evidence — the failure, what was tried, what it rules out — and ask. Do not enter a fourth.
- **Twenty minutes without net progress → stop and show the state.** A short "here is what is blocking me" beats an hour of silent churn.
- **Failing test first, where feasible.** "Done" should be a fact reported by the runner, not a judgement call.
- **Read the governing instruction file before writing the code.** `.github/instructions/` maps paths to rules; policy and sandbox code has rules that are not guessable (probe false negatives, fail-closed semantics, which DLL exports are real).
- **Verify platform behaviour on the real platform, don't reason it out.** POSIX path semantics are available locally through WSL; a `pathlib` or `normcase` assumption has already cost a CI cycle.
- **Scope creep → propose a split, never push through.**

## Commands

```powershell
uv sync                      # install dev deps into .venv
uv run bevel --version       # smoke test the entry point
uv run bevel doctor          # report platform + sandbox capability
uv run pytest                # unit + tui tests (integration/slow deselected by markers)
uv run pytest -m integration # only when a real endpoint is up
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy                  # strict, over src/bevel
```

Before considering any change done: `ruff check`, `mypy`, and `pytest` all green. There is no baseline of pre-existing failures to hide behind — keep it that way.

## The one hard rule: three runtime dependencies

`dependencies` in `pyproject.toml` is exactly `textual`, `httpx`, `pygments`. All pure Python. No exceptions without a written justification in `CONTRIBUTING.md`.

`uv tool install bevel` must work on Windows and on ARM64 (a DGX Spark running Ubuntu 24.04) with **no build step and nothing to install**. That requirement is why these are hand-rolled instead of pulled from a package:

| Tempting dependency | What bevel does instead |
|---|---|
| `pydantic`, `pyyaml` | stdlib `tomllib` for config; dataclasses for models — ships today |
| `numpy` / a vector DB | SQLite FTS5 from stdlib + brute-force `float32` cosine over server `/embeddings` — ships today |
| `rich` directly | it arrives transitively via `textual`; don't import it as a direct dependency |
| `mcp` SDK (pulls `pydantic-core`, Rust) | *planned, m10 — `mcp/` is empty until then:* JSON-RPC over stdio + streamable-HTTP, subset only: `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read` |
| `python-prctl`, seccomp/AppContainer bindings | *m12a shipped — `core/sandbox/` holds the bwrap backend and self-test; m12b (Windows decision) is still open:* none. The backend prefixes tool argv with `bwrap` (a user-installed package, probed by running it, like `git`/`gh`) — no `ctypes` syscall ladder, do not revive the deleted Landlock/seccomp/AppContainer design |

Dev-only tools (`pytest`, `ruff`, `mypy`, …) live in `[dependency-groups] dev` and are unconstrained.

## Layout

```
src/bevel/
  cli.py            argparse entry; TUI is the default action
  _capability.py    read-only sandbox probes; never enforces, never mutates
  config/           layered config: defaults -> ~/.config/bevel/config.toml
                    -> ./bevel.toml -> env -> CLI flags
  providers/        everything that speaks HTTP; base.py is the only seam
  core/agent.py     the loop: orient -> plan -> act -> observe, under budgets
  core/tools/       one file per tool; declares JSON Schema + risk tier
  core/context/     compaction (L1/L2), project map (budgets live in core/agent.py)
  core/policy/      tiered engine, rules, approvals, decision log
  core/sandbox/     bwrap backend (argv prefix + probe) + kernel self-test (m12a); T0 lives in tools/base.py + policy/
  core/checkpoint/  session snapshot commit on a side ref, restore API
  core/session/     SQLite: turns, tool calls, token + cost accounting
  core/memory/      FTS5 + optional vector recall
  mcp/              empty until m10
  evals/            empty until m13
  tui/  render/
tests/{unit,integration,tui}/   tests/evals/ is empty until m13
```

Only the T0 path jail (`core/tools/base.py`) and the rule engine (`core/policy/`) enforce anything today. `bevel doctor` reports what the OS *could* enforce, which is not what bevel does.

## The agent loop

Each iteration: assemble context under budget → stream a completion → parse tool calls **through the repair layer** → evaluate every call against the policy engine → request approvals above the call's tier → take the session's checkpoint commit before the first batch that writes → execute (parallel where safe, `asyncio.gather` with a concurrency cap) → append observations → compact if over budget → repeat.

Cancellation is wired end-to-end. `Ctrl+C` aborts the in-flight HTTP request *and* the whole subprocess tree, not just the parent.

## Out of scope — do not add these

Subagents · codebase indexing / embedding search · web fetch or search · auto-fix-CI · voice input · a VS Code extension · native Anthropic/Google SDKs (OpenAI-compatible only) · always-on VM sandboxing for ordinary tool calls.

"OpenAI-compatible only" is a deliberate boundary, not an oversight: vLLM and llama.cpp are the target, and the provider abstraction is where any future backend would attach.

## Windows

This repo is developed on Windows and CI runs on `windows-latest`, so platform bugs are first-class bugs.

- **Never `shell=True`.** Use `asyncio.create_subprocess_exec` with an explicit argv list. `shell=True` on Windows routes through `cmd.exe` and its quoting rules, which is a command-injection surface dressed up as convenience.
- Keep the `ProactorEventLoop` — subprocess support needs it. Do not switch to `SelectorEventLoop`.
- Kill process **trees**, not parents: Job Objects on Windows, process groups on POSIX.
- Paths: accept both separators, normalise through `pathlib`, and never build a path by string concatenation with `\`.
