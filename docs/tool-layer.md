# Tool layer (m03)

Everything the model can actually *do* lives in `src/bevel/core/tools/`. This page
records the decisions that are not obvious from the code, and what the layer
deliberately does **not** do: contain anything beyond the path jail.

The entry point is `tools/registry.py` — the single source of truth for the offered
tool list, as `core/tools/types.py` has always claimed. `tools()` returns the
instances in offer order, `get(name)` looks one up, `specs()` returns the
wire-facing `ToolSpec` list. Adding a tool means adding one element to `_TOOLS`;
nothing else needs to know it exists.

## The contract

`base.py` defines it. A tool is a class with a `spec` (`ToolSpec`: name,
description, JSON Schema), a `risk_tier`, an `is_parallel_safe` flag, and
`async def execute(args, ctx) -> ToolResult`. `call_tool()` is the seam the
harness calls: it validates arguments against the schema with
`core/tools/schema.py` and converts `ToolError` into a failed result. **Tools may
assume their arguments are valid** — `execute()` indexes required keys directly and
will raise `KeyError` if called bare. Always go through `call_tool`.

`ToolContext` carries the workspace root plus two optional stores — `memory`
(m09a) and `todos` (m09b) — both `None` when unconfigured, and both optional and
last so `ToolContext(workspace)` still works. It does **not** carry the platform
capabilities from m01, a logger, or a sandbox handle; a tool that needs one of
those does not have it, and `mcp/`, `evals/` and `core/sandbox/` are still empty
docstring-only packages. Every path argument is normalised through the context
(`resolve`, `require_within_workspace`, `display`, `relative`) — no tool builds a
path by concatenation, and no tool reports a path the user can't paste back.

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
| `WRITE_OUTSIDE` | writes outside the root; always asks |
| `EXEC_SAFE` | runs a command the allowlist covers; free once allowlisted |
| `EXEC_UNSAFE` | runs arbitrary code |
| `EXEC_DESTRUCTIVE` | always asks, with typed confirmation |
| `NETWORK_EGRESS` | leaves the machine; asks |

`read`, `grep`, `glob`, `recall` and `todo` are `READ`. `write` and `edit`
are `WRITE_WORKSPACE` and **not** parallel-safe — two concurrent writes to the
same file lose one of them. `remember` is `WRITE_OUTSIDE` because the memory
database lives in `~/.local/share/bevel/`, outside every workspace. `shell` and
`git` are `EXEC_UNSAFE` and not parallel-safe — two git calls racing on the
index is a corrupt index.

`todo` being `READ` looks like a bug — it writes to SQLite — and the reason is in
`tools/todo.py`. Nothing it does touches a workspace file, the rows are scoped to
its own session and discarded with it, and declaring it `WRITE_WORKSPACE` would
put an approval prompt in front of every progress update, which trains the user to
approve without reading. `recall` is `READ` and parallel-safe; `remember` is not.

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

**`recall`** — `query` in, ranked memories out. Two-stage: FTS5 `MATCH` + `bm25`
narrows to ~100 candidates, and cosine re-ranks only those when a vector model is
configured. Read-only and parallel-safe, so the model can ask before assuming it
knows a project convention. When memory is not configured the tool fails loudly
rather than returning an empty list — "nothing matched" and "memory is off" are
different answers. See [agent-loop.md](agent-loop.md) for the ranking rules.

**`todo`** — the model's plan for the session. One call replaces the whole list;
there are no ids, no `action` enum and no merge semantics, because an
item-addressable interface asks a small model to remember an identifier it
invented three turns ago. `content` is capped at 200 characters and the list at 32
items — the ceiling is what a small model reproduces faithfully in one call, not
anything SQLite cares about. `additionalProperties: false` at both levels, so an
invented `id` or a status spelled `finished` is refused before it reaches the
store. Not parallel-safe: two concurrent whole-list replaces race. An empty list
legitimately means *cleared*.

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

**`remember`** — store one durable thing, permanently. `content` plus a `type` of
`fact` / `feedback` / `decision` / `reference`, and the row is stamped with the
repository scope it was learned in. Writes outside the workspace, so it always
asks — unless `[memory] auto_accept = true`, which exempts `remember` alone. A
contradicting memory is superseded, never deleted.

**`git`** — see below.

**`shell`** — see below.

## `git`: an allowlist, because git is an interpreter

`git` (m11a) exists for one reason `shell` cannot serve: it is **narrower than
"any program"**. The cwd is pinned to the workspace (there is no `cwd`
parameter), the subcommands are an allowlist, and the argv that escapes the
workspace or executes arbitrary code is refused before anything spawns. If
those three constraints were removed, the tool would be `shell` with extra
steps and AGENTS.md's anti-bloat rule would delete it.

The tier is `EXEC_UNSAFE` and the argv is exposed as `command` — that is not
incidental. `PolicyEngine._draft` computes an argv *only* for tiers in
`_EXEC_TIERS`, reading `arguments["command"]`. Declaring `EXEC_UNSAFE` with a
`command` property is what buys, for free: the built-in destructive rules
(`git push --force`, `git reset --hard`, `git clean -f`, `git checkout .`,
`git restore .`, `git branch -D`, `git gc --prune`), the built-in free rules
(`git status`/`diff`/`log`/`show`/`blame`/`branch`/`rev-parse`/`ls-files`/
`remote`), and an ask for everything else. A `git` tool that declared `READ`
would skip the argv matching entirely and hand the model `git reset --hard`
for free.

**The refusals are an allowlist expressed as refusals, never a denylist of
known-bad tokens.** Refused anywhere in the argv, each with its own test:

| token | why |
| --- | --- |
| `-C`, `--git-dir`, `--work-tree`, `--namespace` | relocate the repo — defeats the path jail, the only thing confining the model today |
| `-c` (global position), `--exec-path` | config injection: `core.pager`/`editor`/`fsmonitor`/`sshCommand`/`gitProxy`/`askPass`/`credential.helper` all execute programs |
| `ext::` URLs, `--upload-pack`, `--receive-pack` | remote-code surfaces — git runs a named program to move bytes |
| `--output`, `--ext-diff`, `--textconv` | file writes and external filter programs |
| `--no-verify` | skips the user's hooks — the user's guardrails are the user's |
| `-F`, `--file` | commit message from a file; `-m` is the only message channel |
| any other option before the subcommand | a global we have not reasoned about; refused rather than guessed |

`git config` is not on the subcommand allowlist at all, so
`git config --global/--system/--file` is refused as a subcommand. The one
legitimate `-c` is `git switch -c <name>` (create branch), allowed only in
argument position after the subcommand.

**The four decisions**, each written in the module and pinned by a test:

- **Trailer** — `Assisted-by: bevel` is appended as the final paragraph of
  every commit this tool makes, and only when it is not already present, so a
  retried commit does not stack them. It is appended, never substituted: a
  message the user dictated survives byte-for-byte above it.
- **Identity** — unlike the checkpoint module, which forges
  `GIT_AUTHOR_*`/`GIT_COMMITTER_*` so a snapshot never depends on the machine,
  this tool leaves identity entirely to the user's git config. A commit in the
  user's own history authored by `bevel <bevel@localhost>` would be a lie about
  who wrote it. The cost is pre-checked: a repo with no `user.name` comes back
  as a `ToolResult.failure` naming the exact `git config` fix, not an opaque
  non-zero exit.
- **Branch hygiene** — the tool reports the branch a commit landed on (git's
  own output names it) and refuses `switch -c` / `checkout -b` / `branch
  <name>` when that name already exists. It does *not* validate names beyond
  what git rejects, does not auto-create branches, and does not offer
  force-push — `push` is not on the allowlist at all in m11a, and PR operations
  are m11b.
- **Shared runner** — the tool spawns nothing itself. It calls
  `bevel.core._git.run_git`, the single git spawner extracted from the two
  private ones that existed before (`checkpoint/snapshot.py::_git`,
  `memory/store.py::_origin_url`). The extraction was behaviour-preserving,
  with the checkpoint and memory suites as the net, and landed as its own
  commit.

**Checkpoint interaction.** A mutating subcommand (`add`, `commit`, `checkout`,
`switch`, `restore`, `reset`, `stash`, `mv`, `rm`, `tag`) triggers the same
session checkpoint the first `write` does — the loop asks the tool via
`mutates(arguments)` before scheduling, so a commit that later fails still cost
a checkpoint, which is the conservative direction. A read (`status`, `log`,
`diff`, …) does not. Mutating calls are also refused from a subdirectory of the
repo, because they act on the whole repository regardless of cwd.


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
