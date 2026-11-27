"""The PR surface: ``gh`` behind an allowlist, for humans at the CLI.

Does bevel need PR operations when the model can already run ``gh`` through
``shell``? Yes for the human, and only in this shape. ``shell`` runs any
program with any argv; this module runs one program (``gh``), with two groups
(``pr``, ``auth status``), an allowlist of options per subcommand, and every
other token refused **before a child is spawned** — the same refusal style as
:mod:`bevel.core.tools.git`, for the same reason: an allowlist of what is
permitted, never a denylist of known-bad tokens, because a denylist is one
un-listed spelling away from a bypass.

There is no second path to GitHub here. No httpx call, no hand-rolled REST.
``gh`` is the plan's chosen backend, and a second way to reach GitHub is drift
— the one place this module touches the network is spawning ``gh`` and letting
it do its own job.

What the tests can and cannot prove
-----------------------------------
The unit oracle (``tests/fixtures/gh.py``) is a real program spawned as a real
subprocess that speaks enough of the gh surface to be an oracle: it parses the
argv it was given, emits gh-shaped JSON, exits with chosen codes, records what
it saw, and refuses non-compliant argv rather than echoing it. That proves
argv construction, JSON parsing, exit-code mapping, the timeout and tree kill,
the refusals, and the auth probe. What it **cannot** prove is that GitHub's
API accepts what gh sends — no fixture can, because the fixture is not
GitHub. That half belongs to ``tests/integration/test_live_pr.py``, gated on
``BEVEL_GH_INTEGRATION`` the way the live-endpoint tests gate on
``BEVEL_BASE_URL``. Stated plainly here rather than left for a reader to
discover, the same way m10e said resumability is not implemented and why.

The allowlist, and what is refused
---------------------------------
Permitted: ``gh pr create`` (``--title``, ``--body``, ``--base``, ``--head``,
``--fill``), ``gh pr edit <n>`` (``--title``, ``--body``, ``--ready``), and
``gh auth status``. ``--json url,number,title,state`` is appended by this
module — never accepted from a caller — because the success report is built
from gh's own JSON instead of scraping its human-facing prose.

Refused, each with its own test naming the token and why:

- ``gh api`` — raw curl to GitHub. The arbitrary-escape hatch; if it were
  allowed, every refusal below would be decorative.
- ``--web`` — opens a browser. bevel is a terminal agent and nothing may block
  on a GUI.
- ``--repo`` — retargets which repository the PR lands in. This is the gh
  analogue of ``git -C``: it defeats the same confinement, and the workspace
  is the only repository bevel speaks about.
- ``--dry-run`` — a dry run inside bevel would be reported to the user as a
  real PR. The decision is CLI-only in the strictest sense: bevel never builds
  the token at all, so a human who wants a dry run runs ``gh`` directly.
- ``merge`` / ``close`` / ``checkout`` / ``comment`` / ``review`` / ``ready`` —
  merging and closing are human decisions; checkout is not a PR operation
  bevel performs; comments and reviews are not part of create/update; the
  ready/draft transition is covered by ``edit --ready`` and the subcommand
  form is refused so there is exactly one spelling.
- any other option, group, or positional shape — refused rather than guessed.

The five decisions, written down
--------------------------------
**Push.** ``gh pr create`` needs the head branch on the remote. bevel never
pushes. It asks git — through :func:`bevel.core._git.run_git`, the one git
spawner; a second private spawner here would be a defect — whether the branch
exists on ``origin`` and whether the local HEAD matches it, and on either
mismatch refuses with the exact ``git push`` to run. The reason is the same
one that kept ``push`` off the git tool's allowlist in m11a: a push is a
network write the policy engine cannot classify, and a tool that pushes on the
model's behalf turns "propose a PR" into "publish to the internet" without a
single decision point. Force-push stays out of scope entirely; m11a said so
and this module does not walk it back in through the side door.

**Auth.** Auth is the user's, never bevel's. ``gh`` reads ``GH_TOKEN`` /
``GITHUB_TOKEN`` from the environment; bevel probes (``gh auth status``) and
on failure returns a :class:`ToolResult.failure` naming the fix
(``gh auth login``), the way the git tool names the ``user.name`` fix. No code
path here reads, prints, or logs a token value — presence only, the rule
bevel already follows for API keys. The child gets the parent environment plus
``GH_PROMPT_DISABLED=1`` and nothing else: no model-controlled env reaches it,
and a gh that wants to prompt dies with a named error instead of hanging the
loop.

**Tool or not.** PR is a human-only CLI operation. PLAN's diagram lists
``pr.py`` under ``core/tools``, and this is the decision the diagram could not
make: a model-facing ``pr`` tool has no honest risk tier. ``NETWORK_EGRESS``
has no producer and the registry pins the declared set; ``EXEC_UNSAFE``
without a ``command`` argument is denied by the engine as unclassifiable, and
with one it would re-derive the whole argv allowlist a second time;
``WRITE_OUTSIDE`` is a filesystem label on a network write. It would always
ask and usually refuse — the push check means the loop cannot complete the
flow without a human anyway, because bevel does not push. So the four-pin-file
cost buys nothing, and the tool does not ship. If that ever changes, the tier
question above has to be answered first, in this file.

**Trailer.** m11a appends ``Assisted-by: bevel`` to commits. A bevel-authored
PR carries the same provenance the same way: through the commit. ``create``
uses ``gh pr create --fill``, so the body is composed by gh from the commit
messages, which already carry the trailer. bevel authors no PR body of its
own, never appends a line to one it was handed, and never sends ``--body``
when the user dictated one — a user-dictated body survives byte-for-byte.
Provenance is metadata added below the author's words, never a rewrite of
them; that was the commit-trailer rule and it is this one too.

**Update means edit.** ``bevel pr update <n> <title>`` edits title and body
via ``gh pr edit``. It does not comment: a comment is a new artifact on the
PR, not an update of one, and "update" that silently posts is the kind of
surprise this project refuses.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from bevel.core._git import run_git
from bevel.core._process import TreeKiller, close_process, uses_new_session
from bevel.core.tools.base import ToolResult

#: The program this module may spawn, by name. The plan's chosen backend; the
#: tests inject the fixture by passing a different ``gh`` argv.
GH: Final[tuple[str, ...]] = ("gh",)

#: How long one gh call may take before its tree is killed. Longer than git's
#: 30 s because gh is making real HTTPS requests to GitHub.
DEFAULT_GH_TIMEOUT_SECONDS: Final[float] = 60.0
_DRAIN_GRACE_SECONDS: Final[float] = 2.0

#: The fields bevel asks gh to print, so the report is built from gh's own
#: JSON. Appended by this module; never accepted from a caller.
JSON_FIELDS: Final[str] = "url,number,title,state"

_CREATE_OPTIONS: Final[frozenset[str]] = frozenset(
    {"--title", "--body", "--base", "--head", "--fill", "--json"}
)
_EDIT_OPTIONS: Final[frozenset[str]] = frozenset({"--title", "--body", "--ready", "--json"})

#: PR subcommands off the allowlist, each with the reason it is refused.
_REFUSED_PR_SUBCOMMANDS: Final[dict[str, str]] = {
    "merge": "merging a PR is a human decision, not an update",
    "close": "closing a PR is a human decision, not an update",
    "checkout": "checking out a PR branch is not a PR operation bevel performs",
    "comment": "a comment is a new artifact on the PR, not an update of one",
    "review": "reviews are not part of create/update",
    "ready": "the ready/draft transition is 'edit --ready'; there is one spelling",
}

#: Options refused wherever they appear, each a way for argv to leave the
#: terminal, leave the workspace, or lie about what happened.
_REFUSED_OPTIONS: Final[dict[str, str]] = {
    "--web": "it opens a browser, and bevel is a terminal agent — nothing may block on a GUI",
    "--repo": "it retargets which repository the PR lands in — the gh analogue of 'git -C'",
    "--dry-run": "a dry run inside bevel would be reported to the user as a real PR",
    "--recover": "recovering a failed PR is not a create/update operation",
}


@dataclass(frozen=True)
class GhResult:
    """One gh call's exit code and output.

    Mirrors :class:`bevel.core._git.GitResult`, including the shell
    conventions where gh never ran: ``127`` when the child could not be
    started at all (the usual cause: gh is not installed), ``124`` when it was
    killed for running too long.
    """

    code: int
    out: str
    err: str


def validate_gh_argv(argv: Sequence[str]) -> str:  # noqa: PLR0911
    """``""`` if this argv may run; otherwise the refusal, naming the token and why.

    Runs entirely in-process — nothing here spawns, so a refusal costs no
    child. The argv is the *logical* one, beginning with the literal ``"gh"``;
    the program name is replaced only at spawn time. One return per distinct
    refusal, each naming its own reason — the same shape as
    :func:`bevel.core.tools.git._validate`, for the same reason: collapsing the
    refusals would make the failures vaguer.
    """
    args = list(argv)
    if not args:
        return "gh: no argv given"
    program = args[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    if program.removesuffix(".exe") != "gh":
        return (
            f"gh: the first element must be 'gh', not {args[0]!r}. This module runs "
            "gh only; use the shell tool for any other program."
        )
    group = args[1] if len(args) > 1 else None
    if group == "auth":
        if args[2:] != ["status"]:
            return (
                f"gh: only 'auth status' is allowed, not {args[2:]!r}. bevel probes "
                "auth; it never logs in, logs out, or touches tokens."
            )
        return ""
    if group != "pr":
        return (
            f"gh: group {group!r} is not allowed. bevel speaks 'pr create', 'pr edit', "
            "and 'auth status' only — 'gh api' in particular is raw curl to GitHub and "
            "is refused as the arbitrary-escape hatch it is."
        )
    subcommand = args[2] if len(args) > 2 else None
    if subcommand not in ("create", "edit"):
        why = _REFUSED_PR_SUBCOMMANDS.get(
            subcommand or "", "only 'create' and 'edit' are PR operations bevel performs"
        )
        return f"gh: refusing pr subcommand {subcommand!r} — {why}."
    rest = args[3:]
    if subcommand == "edit":
        if not rest or not rest[0].isdigit():
            return (
                f"gh: pr edit needs the PR number as a digits-only first argument, not "
                f"{rest[:1]!r}."
            )
        rest = rest[1:]
    return _validate_options(rest, _CREATE_OPTIONS if subcommand == "create" else _EDIT_OPTIONS)


def _validate_options(rest: list[str], allowed: frozenset[str]) -> str:
    """Every option in ``rest`` against the subcommand's allowlist."""
    for arg in rest:
        if arg.startswith("-"):
            if arg in _REFUSED_OPTIONS:
                return f"gh: refusing {arg!r} — {_REFUSED_OPTIONS[arg]}."
            if arg not in allowed:
                return (
                    f"gh: refusing option {arg!r} — it is not on the allowlist "
                    f"({', '.join(sorted(allowed))})."
                )
    return ""


async def run_gh(
    args: Sequence[str],
    *,
    gh: Sequence[str] | None = None,
    cwd: Path | str | None = None,
    timeout_seconds: float = DEFAULT_GH_TIMEOUT_SECONDS,
) -> GhResult:
    """Run one gh command. Never raises; read ``code``.

    The same contract as :func:`bevel.core._git.run_git`, because the same
    three properties are the point: never raises, always bounded (a timeout
    kills the whole process tree through :class:`~bevel.core._process.TreeKiller`),
    and never inherits a caller's whims — the child gets the parent environment
    plus ``GH_PROMPT_DISABLED=1`` and nothing else. A gh that wants to prompt
    dies with a named error instead of hanging the loop.
    """
    program = list(gh) if gh is not None else list(GH)
    argv = [*program, *args]
    env = {**os.environ, "GH_PROMPT_DISABLED": "1"}
    proc: asyncio.subprocess.Process | None = None
    try:
        kwargs: dict[str, Any] = {
            "stdin": asyncio.subprocess.DEVNULL,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "env": env,
        }
        if cwd is not None:
            kwargs["cwd"] = str(cwd)
        if uses_new_session():
            kwargs["start_new_session"] = True
        proc = await asyncio.create_subprocess_exec(*argv, **kwargs)
    except (OSError, ValueError, RuntimeError) as exc:
        return GhResult(127, "", f"cannot run gh: {exc} (is gh installed?)")

    killer = TreeKiller(proc.pid)
    try:
        try:
            raw_out, raw_err = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        except TimeoutError:
            killer.kill()
            with contextlib.suppress(TimeoutError, OSError):
                await asyncio.wait_for(proc.communicate(), timeout=_DRAIN_GRACE_SECONDS)
            return GhResult(124, "", f"gh {' '.join(args)} timed out after {timeout_seconds:g}s")
        return GhResult(
            proc.returncode if proc.returncode is not None else 124,
            raw_out.decode("utf-8", "replace"),
            raw_err.decode("utf-8", "replace"),
        )
    except asyncio.CancelledError:
        killer.kill()
        raise
    finally:
        killer.close()
        close_process(proc)


async def auth_status(workspace: Path | str, *, gh: Sequence[str] | None = None) -> GhResult:
    """Ask gh whether the user is authenticated. The probe, nothing more.

    bevel never logs in. A non-zero code is the user's to fix, and
    :func:`_auth_failure` is the sentence that names the fix.
    """
    return await run_gh(["auth", "status"], gh=gh, cwd=workspace)


async def create_pr(
    workspace: Path | str,
    *,
    base: str | None = None,
    head: str | None = None,
    title: str | None = None,
    body: str | None = None,
    gh: Sequence[str] | None = None,
    timeout_seconds: float = DEFAULT_GH_TIMEOUT_SECONDS,
) -> ToolResult:
    """Create a pull request for the current branch.

    The order is the contract: local git questions first (branch, pushed,
    base), then the argv gate, then the auth probe, then gh. Every refusal
    happens before any child is spawned — a dangerous token does not even
    cost the auth probe — and the push refusal means a user-dictated flow
    always has a human decision point: bevel proposes, the user pushes, bevel
    creates.
    """
    branch = await _head_branch(workspace)
    if branch is None:
        return ToolResult.failure(
            "pr: HEAD is detached (or the repository has no commits), so there is no "
            "head branch to open a PR from. Switch to a branch first "
            "(['git', 'switch', '-c', '<name>'])."
        )
    head_name = head or branch
    unpushed = await _refuse_if_unpushed(workspace, head_name)
    if unpushed is not None:
        return ToolResult.failure(unpushed)
    same = await _refuse_if_same_branch(workspace, head_name, base)
    if same is not None:
        return ToolResult.failure(same)

    argv = ["gh", "pr", "create", "--head", head_name]
    if base is not None:
        argv += ["--base", base]
    if title is not None:
        argv += ["--title", title]
        if body is not None:
            argv += ["--body", body]
    else:
        argv.append("--fill")
    argv += ["--json", JSON_FIELDS]
    refusal = validate_gh_argv(argv)
    if refusal:
        return ToolResult.failure(refusal)

    auth = await auth_status(workspace, gh=gh)
    if auth.code != 0:
        return ToolResult.failure(_auth_failure(auth))

    result = await run_gh(argv[1:], gh=gh, cwd=workspace, timeout_seconds=timeout_seconds)
    return _report(result, "created")


async def update_pr(
    workspace: Path | str,
    number: int | str,
    *,
    title: str | None = None,
    body: str | None = None,
    ready: bool = False,
    gh: Sequence[str] | None = None,
    timeout_seconds: float = DEFAULT_GH_TIMEOUT_SECONDS,
) -> ToolResult:
    """Edit an existing pull request: title, body, ready. Never comment.

    No push check: updating an existing PR needs no local branch state — the
    PR already exists on GitHub, which is exactly what makes this the safe
    half of the surface.
    """
    if not str(number).isdigit():
        return ToolResult.failure(f"pr: the PR number must be digits only, not {str(number)!r}.")
    if title is None and body is None and not ready:
        return ToolResult.failure(
            "pr update: nothing to change — pass a title, a body, or --ready."
        )

    argv = ["gh", "pr", "edit", str(number)]
    if title is not None:
        argv += ["--title", title]
    if body is not None:
        argv += ["--body", body]
    if ready:
        argv.append("--ready")
    argv += ["--json", JSON_FIELDS]
    refusal = validate_gh_argv(argv)
    if refusal:
        return ToolResult.failure(refusal)

    auth = await auth_status(workspace, gh=gh)
    if auth.code != 0:
        return ToolResult.failure(_auth_failure(auth))

    result = await run_gh(argv[1:], gh=gh, cwd=workspace, timeout_seconds=timeout_seconds)
    return _report(result, "updated")


# ------------------------------------------------------------------ local checks


async def _head_branch(workspace: Path | str) -> str | None:
    """The current branch, or ``None`` when HEAD is detached or unborn.

    ``symbolic-ref`` is git's own answer to "is HEAD a branch", the same probe
    the git tool uses for its detached-HEAD refusal.
    """
    result = await run_git(workspace, ["symbolic-ref", "--quiet", "--short", "HEAD"])
    name = result.out.strip()
    return name if result.code == 0 and name else None


async def _refuse_if_unpushed(workspace: Path | str, branch: str) -> str | None:
    """Refuse when the head branch is not on ``origin`` exactly as it is locally.

    The push decision, enforced: bevel never pushes. Two git questions, both
    through the shared runner — does the branch exist on origin, and does the
    local HEAD match it — and each mismatch names the exact push to run.
    """
    local = await run_git(workspace, ["rev-parse", "--verify", "--quiet", "HEAD"])
    if local.code != 0 or not local.out.strip():
        return (
            f"pr: {workspace} has no commits on {branch!r} yet, so there is nothing "
            "to open a PR from. Commit first."
        )
    remote = await run_git(
        workspace, ["rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{branch}"]
    )
    if remote.code != 0 or not remote.out.strip():
        return (
            f"pr: branch {branch!r} has never been pushed to origin, and gh can only "
            f"open a PR for a branch that is on the remote. bevel does not push — run "
            f"'git push -u origin {branch}', then 'bevel pr create' again."
        )
    if remote.out.strip() != local.out.strip():
        return (
            f"pr: {branch!r} has commits that are not on origin, and the PR would be "
            f"created without them. bevel does not push — run 'git push origin {branch}', "
            "then 'bevel pr create' again."
        )
    return None


async def _refuse_if_same_branch(workspace: Path | str, head: str, base: str | None) -> str | None:
    """Refuse a PR whose head is its base — gh would reject it, after a round trip.

    The base is gh's to know (it asks GitHub); bevel only consults
    ``origin/HEAD``, the ref a real clone has, and only to catch the obvious
    mistake. Unresolvable means no opinion, not a refusal.
    """
    if base is None:
        resolved = await run_git(workspace, ["symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"])
        if resolved.code != 0:
            return None
        base = resolved.out.strip().removeprefix("refs/remotes/origin/")
    if base and base == head:
        return (
            f"pr: {head!r} is the base branch — a PR from a branch to itself is not a "
            "thing. Switch to the branch you want merged, or name a different base."
        )
    return None


# ----------------------------------------------------------------------- reports


def _auth_failure(auth: GhResult) -> str:
    """Name the fix, in gh's own words plus the command that resolves it.

    The git tool's missing-``user.name`` shape: never an opaque exit code.
    Token values are never echoed — this function has no way to read them,
    which is the point.
    """
    first = auth.err.strip().splitlines()[0] if auth.err.strip() else "gh did not say why"
    return (
        f"pr: gh is not authenticated ({first}). Auth is yours, not bevel's — run "
        "'gh auth login', or set GH_TOKEN / GITHUB_TOKEN in your environment."
    )


def _report(result: GhResult, verb: str) -> ToolResult:
    """Turn a gh call into a ToolResult, built from gh's own JSON.

    Unparsable JSON is a named failure, never a fake success: a gh upgrade
    that reshapes its output must not read as "PR created".
    """
    if result.code != 0:
        first = result.err.strip().splitlines()[0] if result.err.strip() else "gh gave no reason"
        return ToolResult.failure(f"gh exited {result.code}: {first}")
    try:
        payload = json.loads(result.out)
    except ValueError:
        return ToolResult.failure(
            f"pr: could not parse gh's JSON response ({result.out.strip()[:120]!r}); "
            "the PR state is unknown — check it before retrying."
        )
    if not isinstance(payload, dict):
        return ToolResult.failure(
            f"pr: gh's JSON response was not an object ({result.out.strip()[:120]!r}); "
            "the PR state is unknown — check it before retrying."
        )
    number = payload.get("number", "?")
    title = str(payload.get("title", ""))
    url = str(payload.get("url", ""))
    line = f"PR #{number} {verb}: {title}" if title else f"PR #{number} {verb}"
    return ToolResult(ok=True, content=f"{line}\n{url}" if url else line)


__all__ = [
    "DEFAULT_GH_TIMEOUT_SECONDS",
    "GH",
    "GhResult",
    "auth_status",
    "create_pr",
    "update_pr",
    "validate_gh_argv",
]
