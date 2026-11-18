# Tool layer (m03)

Everything the model can actually *do* lives in `src/bevel/core/tools/`. This page
records the decisions that are not obvious from the code, and the two things the
layer deliberately does **not** do yet.

The entry point is `tools/registry.py` — the single source of truth for the offered
tool list, as `core/tools/types.py` has always claimed. `tools()` returns the
instances in offer order, `get(name)` looks one up, `specs()` returns the
wire-facing `ToolSpec` tuple. Adding a tool means adding one element to `_TOOLS`;
nothing else needs to know it exists.

## The contract

`base.py` defines it. A tool is a class with a `spec` (`ToolSpec`: name,
description, JSON Schema), a `risk_tier`, an `is_parallel_safe` flag, and
`async def execute(args, ctx) -> ToolResult`. `call_tool()` is the seam the
harness calls: it validates arguments against the schema with
`core/tools/schema.py` and converts `ToolError` into a failed result. **Tools may
assume their arguments are valid** — `execute()` indexes required keys directly and
will raise `KeyError` if called bare. Always go through `call_tool`.

`ToolContext` carries the workspace root, the platform capabilities from m01, and a
logger. Every path argument is normalised through it (`resolve`,
`require_within_workspace`, `display`, `relative`) — no tool builds a path by
concatenation, and no tool reports a path the user can't paste back.

The `Tool` protocol declares `spec` / `risk_tier` / `is_parallel_safe` as
read-only properties rather than attributes. That is deliberate: mypy rejects a
`Final` class attribute against a mutable protocol attribute, and the tools are all
`Final` on purpose.

## Risk tiers are declarations the policy engine acts on

Tiers are declared honestly, and `core/policy/` enforces them: the engine decides
allow/ask/deny from the tier plus the argv, and a tool may not self-declare
`EXEC_SAFE` or `EXEC_DESTRUCTIVE` — those come only from a rule match. The table:

| tier | meaning |
| --- | --- |
| `READ` | no side effects; safe to run concurrently |
| `WRITE_WORKSPACE` | mutates files inside the workspace root |
| `EXEC_UNSAFE` | runs arbitrary code |

`read`, `grep`, `glob` are `READ` and `is_parallel_safe = True`. `write` and `edit`
are `WRITE_WORKSPACE` and **not** parallel-safe — two concurrent writes to the same
file lose one of them. `shell` is `EXEC_UNSAFE` and not parallel-safe.

## The path jail is not security

`require_within_workspace` rejects `..` escapes and absolute paths outside the root,
and it is tested. It exists to stop a confused model from tripping over itself, not
to stop an adversarial one. Symlinks, junctions, short (8.3) paths, and NTFS
alternate data streams all have ways through a purely textual check, and m03 does
not attempt any of them. Real containment is **m12 (sandbox)**. Do not treat the
jail as a boundary until then.

## What each tool does

**`read`** — line-numbered output, `start_line`/`lines` windowed, defaults to 200
lines, hard-capped at 2000 lines / 128 KB. Long lines are truncated at 2000
characters. Binary files are refused by sniffing the first 8 KB rather than
returning mojibake.

**`grep`** — regex over the file tree, `pathlib`-walked (there is no ripgrep
binary; the three-dependency rule rules out a compiled one). Skips `SKIP_DIRS`
from `_fs.py`, skips files over 4 MB, truncates matched lines at 500 characters,
returns at most 200 matches (cap 2000). Invalid patterns come back as a tool
failure naming the pattern error, not a traceback.

**`glob`** — `fnmatch`-style patterns matched against `/`-separated relative paths,
so `src/**/*.py` and `*.py` both behave the way the model expects. 500 results by
default, 5000 cap. "No matches" is a successful result, not a failure.

**`write`** — creates or replaces a whole file, creating parent directories on the
way. An existing file is never clobbered unless `overwrite: true`; the result is a
unified diff of the change (or "no change"), so the transcript shows what happened
even for a whole-file write.

**`edit`** — real diff application, not whole-file overwrite. `old_str` must match
exactly one location; zero matches and multiple matches are both failures that say
so. The result reports the changed line range so the TUI can highlight it.

Both writers are atomic: a sibling `name.tmp-<pid>` is written and `os.replace`d
into place, with a `finally` that removes the temp file if anything raised. Files
are opened with `newline=""` so nothing is ever translated implicitly.

**Line endings.** `edit` normalises the whole file to LF internally, applies the
replacement, then restores the original style. Style is detected as CRLF only when
*every* newline is CRLF (`count(b"\r\n") == count(b"\n")` and non-zero) — a
mixed-ending file is treated as LF and left mixed rather than silently converted.
The trailing newline is preserved in both directions. This is what makes the tool
usable on a Windows dev machine without producing whole-file diffs.

**`shell`** — see below.

## `shell`: argv, not strings

`command` is an **array**, never a string, and execution goes through
`asyncio.create_subprocess_exec` with no `shell=True` anywhere. A model that wants
a shell asks for `["cmd", "/c", "..."]` or `["sh", "-c", "..."]` explicitly and
that intent is visible in the transcript and to the policy engine. There is no
path where an unquoted metacharacter is interpreted by accident.

Timeouts default to 30 s, are clamped to 600 s. Output is capped at 32 KB by
default (1 MB ceiling) and the cap is enforced *while reading*, not after buffering
a 2 GB `find /`. stdout and stderr are drained by separate tasks into separate
lists, so each stream keeps its own order and neither can deadlock the other.

## Killing the whole tree

This is the part that matters on Windows, and the reason `ProactorEventLoop` must
stay: `asyncio` subprocess pipes need it.

A timeout has to kill the child *and everything the child spawned*. `Process.kill()`
kills the parent only, which leaves `cmd.exe`'s children, `pip`'s compiler, or a
`python -m http.server` holding the port.

- **Windows**: the child is assigned to a Job Object created with
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. Closing the handle kills every process in
  the job, so `kill()` *is* `CloseHandle`. This works for grandchildren and needs
  no second process. If job assignment fails at spawn time, the tool falls back to
  `taskkill /T /F /PID`, resolved through `%SystemRoot%\System32` via `pathlib`
  rather than trusted to `PATH`.
- **POSIX**: the child is spawned with `start_new_session=True` and killed with
  `os.killpg` — `SIGKILL` where available, otherwise `SIGTERM`.

After a kill the pipes get a short grace period (2 s) to reach EOF, so output the
process wrote *before* being killed is still reported rather than lost.

Two caveats worth stating plainly. The Job Object path is what runs on the dev
machine and in the `windows-latest` CI lane; the `taskkill` fallback and the POSIX
`killpg` branch are exercised by the Linux CI lane, not locally. And the tree-kill
test binds a probe port — if something on the machine is already listening there,
that test fails for an unrelated reason.

## Cross-platform type checking

CI typechecks on both platforms, so a `sys.platform` branch that only compiles on
one has to survive the *other* lane's `--strict --warn-unreachable` run. Two idioms
from `_capability.py` carry over:

- `_CURRENT_PLATFORM: Final[str] = sys.platform`. mypy special-cases literal
  comparisons against `sys.platform`, but not against a plain local name; binding
  it through a `Final` keeps both lanes honest.
- Anything POSIX-only (`os.killpg`, `signal.SIGKILL`) or missing from typeshed
  (`asyncio.subprocess.Process.close`) is reached through `getattr`. Writing them
  directly fails the Windows lane, and `# type: ignore` is worse — an unused
  ignore is itself an error on the lane where the symbol *does* exist.
