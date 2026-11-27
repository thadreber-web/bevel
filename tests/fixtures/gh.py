"""A real ``gh``, and the oracle every m11b slice tests against.

This is not a test module and must never import pytest. It is a plain script
runnable as ``[sys.executable, "tests/fixtures/gh.py", "pr", "create", ...]``
-- the arguments *after* the program name, which is exactly what
``bevel.core.pr`` spawns it with once a test injects the fixture as the ``gh``
program (the spawn replaces the program name, so the fixture sees the rest).
Like ``tests/fixtures/mcp_server.py`` for m10, the point of a real child is
that argv construction, JSON parsing, exit-code mapping, and the
timeout/tree-kill path are proven against a process that actually holds a pipe
-- a mock of the subprocess layer would prove only that the code agrees with
its own guesses.

If bevel ever forgets to strip the program name, the fixture's first argument
is ``"gh"`` rather than a group, and the group dispatch refuses it with exit
4. The refusal is the catch.

What this fixture can prove: that bevel builds the argv it claims to, parses
gh-shaped JSON, maps exit codes, kills the tree on timeout, refuses dangerous
argv before spawning, and sets ``GH_PROMPT_DISABLED`` on the child. What it
cannot prove: that GitHub's API accepts what gh sends. That question belongs
to the integration-marked tests behind ``BEVEL_GH_INTEGRATION``, the same way
the live-endpoint tests are gated on ``BEVEL_BASE_URL``.

The first argument must be the literal ``"gh"`` -- the fixture is standing in
for the ``gh`` executable, and a caller that forgot to say which program it
meant is a caller whose argv construction is broken. The rest of the argv is
validated against the same allowlist ``bevel.core.pr`` enforces, and
non-compliant argv is *refused* (exit 4, gh-shaped stderr), not echoed. An
echo would be satisfied by a client that leaks ``--web`` only when convenient;
a refusal will not be.

Every knob is an environment variable, not a flag, because bevel builds the
argv itself -- a test cannot smuggle fixture flags through an allowlist that
exists to reject them. The environment is otherwise inherited untouched, which
is also what makes the ``GH_PROMPT_DISABLED`` assertion honest: the value is
read back out of the child's own environment, not out of the parent that set it.

Environment
    ``BEVEL_GH_FIXTURE_SEEN``
        Path to a file. Every invocation appends one JSON line recording the
        argv it received and the value of ``GH_PROMPT_DISABLED`` as this
        process actually saw it. Token values are never recorded -- presence
        only, the same rule bevel follows.

    ``BEVEL_GH_FIXTURE_MODE``
        ``unauthenticated``
            ``auth status`` exits 1 with gh-shaped stderr ("gh: not
            authenticated"). Everything else still works, so a client that
            skips the probe and calls ``pr create`` directly would sail past
            the auth failure -- and bevel's tests pin that it does not.
        ``no-branch``
            ``pr create`` exits 1 with gh-shaped stderr ("no such branch").
        ``garbage``
            ``pr create`` exits 0 and prints a non-JSON body, which is how a
            gh upgrade that reshapes its output catches a client that assumed.
        ``hang``
            ``pr create`` goes silent forever. The timeout must kill the tree.
        ``grandchild``
            ``pr create`` spawns a heartbeat grandchild, then hangs. Only a
            tree kill takes the grandchild with it, and only a grandchild
            proves the kill is a tree kill.

Exit codes
    ``0`` success (gh-shaped JSON on stdout for ``pr create``/``pr edit``).
    ``1`` a gh-shaped operational failure for the chosen mode.
    ``4`` non-compliant argv -- the fixture refused, and bevel should never
    have built it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

_CREATE_OPTIONS = {"--title", "--body", "--base", "--head", "--fill", "--json"}
_EDIT_OPTIONS = {"--title", "--body", "--ready", "--json"}
_PR_SUBCOMMANDS = {"create", "edit"}
#: Subcommands a model would reach for that bevel must never send. Named here
#: so the refusal message is as specific as bevel's own.
_REFUSED_PR_SUBCOMMANDS = {
    "merge": "merging is a human decision",
    "close": "closing a PR is a human decision",
    "checkout": "checking out a PR branch is not a PR operation bevel performs",
    "comment": "comments are not part of create/update",
    "review": "reviews are not part of create/update",
    "ready": "ready/draft transitions are not part of create/update",
}
_REFUSED_OPTIONS = {
    "--web": "it opens a browser, and bevel is a terminal agent",
    "--repo": "it retargets which repository the PR lands in",
    "--dry-run": "a dry run from inside bevel would be reported as a real one",
    "--recover": "recovering a failed PR is not a create/update operation",
}

#: The heartbeat grandchild. Runs as ``[sys.executable, "-c", code, path]`` and
#: appends ``<pid> <tick>`` until killed, so the test can read the pid and then
#: prove the tree kill reached it.
_HEARTBEAT_CODE = (
    "import os, sys, time\n"
    "with open(sys.argv[1], 'a', encoding='utf-8') as handle:\n"
    "    for tick in range(1000):\n"
    "        handle.write(f'{os.getpid()} {tick}\\n')\n"
    "        handle.flush()\n"
    "        time.sleep(0.05)\n"
)


def _record() -> None:
    """Append one JSON line describing this invocation, before any refusal."""
    path = os.environ.get("BEVEL_GH_FIXTURE_SEEN")
    if not path:
        return
    row = {
        "argv": sys.argv[1:],
        "gh_prompt_disabled": os.environ.get("GH_PROMPT_DISABLED"),
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _refuse(reason: str) -> int:
    sys.stderr.write(f"gh fixture: refusing — {reason}\n")
    return 4


def _err(message: str) -> int:
    sys.stderr.write(f"gh: {message}\n")
    return 1


def _emit(payload: dict[str, object]) -> int:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()
    return 0


def _value_after(argv: list[str], flag: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _pr(argv: list[str], mode: str) -> int:
    """``pr create`` / ``pr edit``. ``argv`` begins with the subcommand."""
    if not argv or argv[0] not in _PR_SUBCOMMANDS:
        name = argv[0] if argv else "(missing)"
        why = _REFUSED_PR_SUBCOMMANDS.get(name, "only create and edit are allowed")
        return _refuse(f"pr subcommand {name!r} — {why}")
    subcommand = argv[0]
    rest = argv[1:]

    for arg in rest:
        if arg in _REFUSED_PR_SUBCOMMANDS or (arg.startswith("-") and arg in _REFUSED_OPTIONS):
            why = _REFUSED_PR_SUBCOMMANDS.get(arg) or _REFUSED_OPTIONS[arg]
            return _refuse(f"{arg!r} — {why}")
        if arg.startswith("-") and arg not in (
            _CREATE_OPTIONS if subcommand == "create" else _EDIT_OPTIONS
        ):
            return _refuse(f"option {arg!r} is not on the allowlist")

    body = rest[1:] if subcommand == "edit" else rest
    if subcommand == "edit":
        if not rest or not rest[0].isdigit():
            return _refuse("pr edit needs a numeric PR number as its first argument")

    if mode == "no-branch" and subcommand == "create":
        return _err("could not determine current branch: no such branch")
    if mode == "garbage" and subcommand == "create":
        sys.stdout.write("this is not json, it is a table\n")
        sys.stdout.flush()
        return 0
    if mode in ("hang", "grandchild") and subcommand == "create":
        if mode == "grandchild":
            heartbeat = os.environ.get("BEVEL_GH_FIXTURE_HEARTBEAT")
            if heartbeat:
                subprocess.Popen(  # noqa: S603 - fixed argv, the fixture's own script
                    [sys.executable, "-c", _HEARTBEAT_CODE, heartbeat],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        time.sleep(3600)
        return 1

    number = int(rest[0]) if subcommand == "edit" else 42
    title = _value_after(body, "--title") or ("fill" if "--fill" in body else "untitled")
    return _emit(
        {
            "number": number,
            "title": title,
            "state": "OPEN",
            "url": f"https://github.com/bevel-fixture/repo/pull/{number}",
        }
    )


def main() -> int:
    argv = sys.argv[1:]
    _record()
    mode = os.environ.get("BEVEL_GH_FIXTURE_MODE", "")

    group = argv[0] if argv else None
    if group == "auth":
        if argv[1:] != ["status"]:
            return _refuse("only 'auth status' is allowed")
        if mode == "unauthenticated":
            return _err(
                "not authenticated — run 'gh auth login', or set GH_TOKEN / GITHUB_TOKEN"
            )
        return _emit({"logged_in": True})
    if group == "pr":
        return _pr(argv[1:], mode)
    return _refuse(f"group {group!r} is not allowed — bevel speaks 'pr' and 'auth status' only")


if __name__ == "__main__":
    raise SystemExit(main())
