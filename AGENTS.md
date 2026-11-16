# bevel

A terminal coding agent for your own models (vLLM, llama.cpp). Python 3.11+, Textual TUI, distributed with `uv`.

The bar is **portfolio piece, not a demo**: no stubs, no fake output, no aspirational `--help` text. If a feature is not built, it must say so loudly and name the milestone that delivers it. Silent fakes are the worst failure mode this project has.

## Working style

Act as a senior engineer. Simplicity, correctness, and surgical execution beat verbosity and premature abstraction.

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

**One problem per session.** If the work branches or the scope expands, propose a fresh session rather than letting it sprawl. (The 14-milestone build order is the deliberate exception — treat each milestone as its own session-sized unit.)

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
| `mcp` SDK (pulls `pydantic-core`, Rust) | JSON-RPC over stdio + streamable-HTTP in `mcp/`, subset only: `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read` |
| `pydantic`, `pyyaml` | stdlib `tomllib` for config; dataclasses for models |
| `numpy` / a vector DB | SQLite FTS5 from stdlib + brute-force `float32` cosine over server `/embeddings` |
| `python-prctl`, seccomp/AppContainer bindings | `ctypes` against `libc` / `kernel32` / `kernelbase` / `userenv` / `advapi32` |
| `rich` directly | it arrives transitively via `textual`; don't import it as a direct dependency |

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
  core/context/     budgeting, 3-level compaction, project map
  core/policy/      tiered engine, rules, approvals, decision log
  core/sandbox/     T0/T1/T2 backends + self-test
  core/checkpoint/  git worktree per session, checkpoint commits, /undo
  core/session/     SQLite: turns, tool calls, token + cost accounting
  core/memory/      FTS5 + optional vector recall
  mcp/  evals/  tui/  render/
tests/{unit,integration,tui,evals}/
```

## The agent loop

Each iteration: assemble context under budget → stream a completion → parse tool calls **through the repair layer** → evaluate every call against the policy engine → request approvals above the call's tier → execute (parallel where safe, `asyncio.gather` with a concurrency cap) → append observations → checkpoint if anything mutated → compact if over budget → repeat.

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
