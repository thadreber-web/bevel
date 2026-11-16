---
applyTo: "src/**/*.py"
---

# Python style for bevel

## Typing

`mypy --strict` runs over `src/bevel` in CI. Treat a typing shortcut as a bug, not a tradeoff.

- Every function annotated, including `-> None`.
- `from __future__ import annotations` at the top of every module.
- No `Any` unless it is the boundary of genuinely untyped input (a model's JSON payload, an MCP response) — and then validate immediately and narrow to a dataclass.
- `cast()` needs a comment saying why the checker can't see it. Prefer a guard.
- Return `None` implicitly rather than `return None` where ruff agrees.

## Data over classes-of-one

Prefer frozen dataclasses for values crossing module boundaries (`Capability`, `ToolCall`, `Decision`). Mutability is for the agent loop's transcript and nothing else unless there's a reason.

## Errors

- Raise specific exceptions from a small `bevel`-local hierarchy; never bare `Exception` and never `raise Exception(...)`.
- Error messages must name the thing that failed and what would fix it. "invalid response" is a bug report nobody can act on; "endpoint 'llama-70b' returned tool_calls.arguments as a string, expected object" is actionable.
- Never swallow an exception to keep going without recording it. A degraded run is a *reported* degraded run — same principle as the sandbox badge.

## Async

- No blocking calls on the event loop. No `time.sleep`, no `requests`, no synchronous `subprocess`, no file I/O in hot paths.
- Every awaited network or subprocess call has a timeout. A hung local server must not wedge the TUI.
- `asyncio.gather` for parallel-safe tools only — writes, shell, and anything with side effects stay serial. Enforce with an explicit `Tool.is_parallel_safe`.
- Cancellation must be safe: clean up subprocesses and close streams in `finally`/`except asyncio.CancelledError`, then re-raise.

## Imports

- Absolute imports within `bevel`; relative only for sibling imports inside one subpackage.
- Keep platform-specific imports (`winreg`, `ctypes.wintypes`) inside the function that needs them with a `# noqa: PLC0415` and a comment on why. Importing them at module scope breaks Linux.
- No import cycles between `core/` subpackages — `core/policy` must not import `core/sandbox`, and so on. Compose at the loop level.

## Naming and comments

- `snake_case` everywhere; abbreviations spelled out (`config`, `id`, `env`, `stdin`/`stdout` are fine, `prjMgr` is not).
- Comments explain *why*, especially where a constraint is invisible (a kernel ABI quirk, a Windows quoting trap, a llama.cpp deviation). Do not narrate what the code plainly does.
- `# type: ignore` always carries an error code and a reason.

## Anti-bloat

- A helper called from one place is not a helper. Inline it.
- No parameters, flags, kwargs, or enum variants added "for future flexibility". Add them when a caller exists.
- No wrapper types, protocols, or base classes with a single implementation. `providers/base.py` earns its protocol because it has two implementations (`openai_compat`, `mock`) and the loop depends only on it — that's the bar.
- No defensive validation on values that came from our own code. Validate at boundaries (model output, config files, MCP responses, user input), then trust.
- Don't reformat or re-order code the task didn't require you to touch. If `ruff` demands a fix in an untouched region, make that fix and nothing else around it.
- No comment that restates the syntax. Comments are for invisible constraints: kernel ABI quirks, Windows quoting traps, llama.cpp deviations from the OpenAI spec.

## What must not ship

`TODO`, `FIXME`, `NotImplementedError`, stubbed tools, `print()` in library code (use the session log; `ruff`'s `T20` enforces this in `src/`), or a `--help` string describing a flag that does nothing.
