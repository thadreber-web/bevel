"""The L3 handoff against a real endpoint.

``tests/unit/test_agent_handoff.py`` proves the ladder reaches L3 and that a
scripted run continues. It cannot prove the thing that would actually embarrass
this milestone: that a *real* model, over a *real* transcript, is handed a brief
it did not write and still finishes the task. A mock hands back a transcript
someone authored by hand; only a server can reject the rebuilt one with a 400.

The window is forced small so the server's own ``prompt_tokens`` exceed it, and
the workspace is a handful of small files so the loop has to make several turns
to get through them. Files are kept under L1's truncation threshold on purpose:
if L1 could cut the observations it would, and the run would never reach the
level under test.

Two fixture choices are load-bearing and easy to get wrong, so they are pinned
here rather than left to the reader:

*   ``tools`` is ``[ReadTool(), WriteTool()]`` rather than the default set. Tool
    schemas are counted in the prompt, and the full default set costs ~2400
    tokens of floor on its own -- more than ``WINDOW``, so every run would die
    as "a window nothing can fit" and L3 would never be reached. Read *and*
    write are passed (not read alone) so that the second test's claim, "no
    files were written", is a provenance claim rather than a tautology.
*   ``max_tokens`` is 1024, not the 512 the other live tests use. The brief is
    written by the same reasoning model that produced the transcript, and it
    thinks before it writes: at 512 the whole ceiling is spent on reasoning
    tokens and the brief comes back empty, which the loop honestly reports as a
    declined handoff. 1024 is where its account stops being truncated.

The task names every file and asks for one read per turn. A pattern like
``partNN.txt`` gets a prose answer with no tool calls at all, and a model that
calls several tools per turn collapses the transcript below the point where
there is anything for L3 to fold.

What is asserted is mechanical. The deterministic sections of the brief are
matched exactly, because the loop wrote them. The model's own account is only
ever checked to be *present under its heading*, never for wording -- a
rephrasing must not redden this test, and a handoff that dropped it must.

Cost: a handful of real generations per attempt, marked ``slow``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Final, TypeVar

import pytest

from bevel.core.agent import (
    Agent,
    AgentEvent,
    AgentEventType,
    AgentResult,
    Budgets,
    StopReason,
)
from bevel.core.tools.read import ReadTool
from bevel.core.tools.write import WriteTool
from bevel.providers.base import GenOptions, ModelProvider

# The endpoint fixture is module-scoped (one probe for the whole module), so these
# tests must share its event loop scope rather than the project default of "function".
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.asyncio(loop_scope="module"),
]

#: The word the model can only produce by reading every file. It appears in no
#: prompt, so an answer containing it proves work survived the handoff.
SECRET: Final[str] = "cobalt-lantern-8123"

#: How many part files the workspace holds. The provenance test has the model
#: read all of them; the headline test reads a prefix of them.
FILE_COUNT: Final[int] = 10

#: How many files the headline test names. Enough turns that the transcript
#: outgrows ``WINDOW``, few enough that the run still finishes inside the
#: per-handoff cap -- which is the point: the task has to survive the handoff.
HEADLINE_READS: Final[int] = 4

#: How many lines each file has. Each read is ~340 tokens, so the transcript
#: crosses ``WINDOW`` on the fourth turn and the handoff has something to fold.
LINES_PER_FILE: Final[int] = 14

#: Wide enough that the rebuilt context (system prompt + project map + brief + the
#: two kept turns) fits with room to spare -- it comes back at ~960 tokens -- and
#: narrow enough that the fourth read does not fit. The floor of a request with
#: this tool set is ~790 tokens; four turns reach ~1800, which is where the
#: handoff earns its keep.
WINDOW: Final[int] = 1800

#: Terse generations, but not too terse -- see the module docstring for why the
#: ceiling has to clear the model's reasoning tokens.
GEN: Final[GenOptions] = GenOptions(temperature=0.0, max_tokens=1024)

#: The tools under test. See the module docstring: the default set's schemas alone
#: would put the floor above ``WINDOW``.
TOOLS: Final = (ReadTool, WriteTool)

#: A handoff is model-volunteered behaviour: it needs enough turns to overflow, and
#: a model that answers in prose instead of calling tools never gets there. Re-asking
#: is the same bargain ``test_live_agent.py`` strikes -- fair only because every
#: assertion made afterwards is hard.
VOLUNTEER_ATTEMPTS: Final[int] = 3


def _task(count: int) -> str:
    """Name the files explicitly and demand one call per turn.

    Both parts matter: a pattern like ``partNN.txt`` gets a prose answer with no
    tool calls at all, and a model that batches several reads into one turn
    collapses the transcript below the point where L3 has anything to fold.
    """
    names = ", ".join(f"part{i:02d}.txt" for i in range(count))
    return (
        f"Read these files one at a time, making exactly one read call per turn: {names}. "
        "Then read ledger.txt and tell me its verification word."
    )


#: The brief's goal section must be the task verbatim, so the tests name the
#: strings rather than rebuilding them.
HEADLINE_TASK: Final[str] = _task(HEADLINE_READS)
PROVENANCE_TASK: Final[str] = _task(FILE_COUNT)

#: The headings the loop writes itself. Exact on purpose: these are not the model's.
BRIEF_HEADINGS: Final[tuple[str, ...]] = (
    "[handoff brief]",
    "## goal",
    "## decisions and open questions",
    "## files touched",
    "## todos",
)

_T = TypeVar("_T")


def _requires_tools(endpoint: ModelProvider) -> None:
    if not endpoint.capabilities.supports_tools:
        pytest.skip(f"{endpoint.model!r} does not support tool calling (probed; see bevel doctor)")


@contextmanager
def _tiny_window(endpoint: ModelProvider, tokens: int) -> Iterator[ModelProvider]:
    """Pretend the endpoint's window is ``tokens`` wide, then put it back.

    Nothing counts tokens locally: the loop still only ever reads the number the
    server sent back, which is the whole point of the trigger.
    """
    original = endpoint.capabilities
    endpoint.capabilities = replace(original, max_model_len=tokens)
    try:
        yield endpoint
    finally:
        endpoint.capabilities = original


def _events(result: AgentResult, type_: AgentEventType) -> list[AgentEvent]:
    return [event for event in result.events if event.type is type_]


def _starter(
    endpoint: ModelProvider, workspace: Path, task: str
) -> Callable[[], Awaitable[AgentResult]]:
    """Build a fresh agent for one attempt at ``task``.

    A new ``Agent`` per attempt, because a run carries its own message list and
    handoff counter; re-using one would let attempt two inherit attempt one's.
    """

    async def attempt() -> AgentResult:
        agent = Agent(
            endpoint,
            workspace,
            tools=[tool() for tool in TOOLS],
            budgets=Budgets(max_iterations=24, max_tool_calls=60),
            options=GEN,
        )
        return await agent.run(task)

    return attempt


def _handoff_events(result: AgentResult) -> list[AgentEvent]:
    """Only the handoffs that happened, not the ones that declined or found nothing."""
    return [e for e in _events(result, AgentEventType.COMPACTION) if "handoff replaced" in e.text]


def _context_was_measured(result: AgentResult) -> bool:
    return any(
        event.context is not None and event.context.status.value != "unknown"
        for event in _events(result, AgentEventType.COMPACTION)
    )


async def _until_handoff(
    start: Callable[[], Awaitable[AgentResult]],
    attempts: int = VOLUNTEER_ATTEMPTS,
) -> AgentResult:
    """Run until a handoff actually happened.

    A run that stops for another reason first (the model answered in prose, or
    burned its turn budget without ever overflowing) is retried. A run that stops
    with ``context_overflow`` is returned immediately: that is the regression, and
    re-asking would only hide it behind a luckier sample.
    """
    last: AgentResult | None = None
    for _ in range(attempts):
        last = await start()
        if _handoff_events(last) or last.stop_reason is StopReason.CONTEXT_OVERFLOW:
            return last
    assert last is not None
    pytest.fail(
        f"after {attempts} attempts the run never overflowed its {WINDOW}-token window, "
        f"so there was no handoff to test; it stopped with {last.stop_reason.value} after "
        f"{last.iterations} iteration(s) and {last.usage.total_tokens} tokens"
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A throwaway repo of small files, each too short for L1 to cut.

    It is a repo because the loop takes a checkpoint commit before its first write,
    and a run that never gets that far is not the run this file is about.
    """
    for index in range(FILE_COUNT):
        body = "\n".join(
            f"item {index}.{line} :: pad pad pad pad pad pad pad pad pad pad"
            for line in range(LINES_PER_FILE)
        )
        (tmp_path / f"part{index:02d}.txt").write_text(
            f"Part {index} of the ledger. The word for this part is word{index}.\n{body}\n",
            encoding="utf-8",
        )
    (tmp_path / "ledger.txt").write_text(
        f"Sum every part. The verification word is {SECRET}.\n", encoding="utf-8"
    )
    for argv in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=bevel",
            "-c",
            "user.email=bevel@localhost",
            "commit",
            "-q",
            "-m",
            "seed",
        ],
    ):
        subprocess.run(
            ["git", *argv], cwd=tmp_path, check=True, capture_output=True, text=True, shell=False
        )
    return tmp_path


# ------------------------------------------------------------------ the real thing


async def test_an_over_long_run_hands_off_and_continues(
    endpoint: ModelProvider, workspace: Path
) -> None:
    """The milestone's claim, against a server: overflow becomes a continuation.

    Before m08d this scenario ended with ``StopReason.CONTEXT_OVERFLOW`` and an
    error naming the unbuilt handoff. Now the loop must build a brief, rebuild the
    context from the opening shape, and keep going.
    """
    _requires_tools(endpoint)

    with _tiny_window(endpoint, WINDOW) as small:
        result = await _until_handoff(_starter(small, workspace, HEADLINE_TASK))

    if not _context_was_measured(result):
        pytest.skip(
            f"{endpoint.model!r} returned no usage for any request, so the loop honestly "
            "knows nothing about its context size and cannot trigger a handoff"
        )

    # The headline: the run did not die of overflow.
    assert result.stop_reason is not StopReason.CONTEXT_OVERFLOW, result.error
    assert result.error is None or "budget" in result.error.lower(), result.error

    handoffs = _handoff_events(result)
    assert handoffs, [event.text for event in _events(result, AgentEventType.COMPACTION)]
    assert len(handoffs) <= 2, "the per-run cap was not honoured"

    # It continued: at least one more iteration ran *after* the handoff was taken.
    # Without this, a loop that handed off and then stopped would pass every other
    # assertion here while doing exactly what the old CONTEXT_OVERFLOW did.
    assert result.iterations > handoffs[0].iteration, (
        f"the handoff landed at iteration {handoffs[0].iteration} but the run ended at "
        f"{result.iterations}: nothing continued after it"
    )

    # The task actually survived. Everything above proves the loop kept going; this
    # proves the continuation was the *same task*. The word is in no prompt and in
    # no file but ledger.txt, which sits at the end of the task, so producing it
    # means the rebuilt context carried the goal, the reads already done, and the
    # will to finish. A handoff that continued and lost the thread would pass every
    # other assertion in this test.
    assert SECRET in result.final_text, (
        f"the run continued past the handoff but never produced the verification word; "
        f"it stopped with {result.stop_reason} saying: {result.final_text!r}"
    )

    # The rebuilt context: the opening shape, not an invented second one.
    assert result.messages[0].role == "system"
    # A later handoff folds the earlier brief into its own, so the transcript carries
    # the *latest* brief rather than one per handoff. What must hold is that a brief
    # is on the wire at all, and that every brief still present is complete.
    briefs = [m for m in result.messages if m.content.startswith("[handoff brief]")]
    assert briefs, "the run handed off but no brief survived in the transcript"
    assert len(briefs) <= len(handoffs), (
        f"{len(briefs)} brief(s) on the wire after only {len(handoffs)} handoff(s)"
    )
    for brief in briefs:
        for heading in BRIEF_HEADINGS:
            assert heading in brief.content, f"{heading!r} missing from the brief:\n{brief}"
        # The goal is the task, verbatim -- the loop wrote that section, so it is exact.
        assert HEADLINE_TASK in brief.content, brief.content
        # The model's account is embedded under a heading that names it as the
        # previous instance's own. Its wording is nobody's business; its presence is.
        account = brief.content.split("## decisions and open questions", 1)[1]
        assert account.split("## files touched", 1)[0].strip(), (
            f"the model's section of the brief is empty:\n{brief.content}"
        )
    # The brief is not messages[1] -- the recall block owns that slot when memory is
    # on -- but it must sit above the kept tail, i.e. before any assistant message.
    first_assistant = next(i for i, m in enumerate(result.messages) if m.role == "assistant")
    assert result.messages.index(briefs[0]) < first_assistant

    # Nothing orphaned across the cut: the rebuilt transcript is one the server accepts.
    asked = [call.id for m in result.messages for call in m.tool_calls]
    answered = [m.tool_call_id for m in result.messages if m.role == "tool"]
    assert sorted(asked) == sorted(answered), (
        f"{len(asked)} calls but {len(answered)} observations after the handoff"
    )


async def test_the_brief_names_only_files_the_loop_actually_wrote(
    endpoint: ModelProvider, workspace: Path
) -> None:
    """Provenance against a real model: a read must not become a "file touched".

    The task makes the model read several files and write none. The brief's files
    section is assembled from executions that mutated the workspace, so on this
    run it must say no files were written -- even though the transcript is full of
    file names the model could have volunteered. That is the difference between a
    list the loop knows and a list the model imagined. The write tool is offered
    on this run, so "nothing was written" is the model's choice, not the fixture's.
    """
    _requires_tools(endpoint)

    with _tiny_window(endpoint, WINDOW) as small:
        result = await _until_handoff(_starter(small, workspace, PROVENANCE_TASK))

    if not _context_was_measured(result):
        pytest.skip(f"{endpoint.model!r} reported no usage, so no handoff could trigger")
    handoffs = _handoff_events(result)
    if not handoffs:
        pytest.skip("the run finished inside the window, so there was no brief to inspect")

    brief = next(m.content for m in result.messages if m.content.startswith("[handoff brief]"))
    files = brief.split("## files touched", 1)[1].split("## todos", 1)[0]
    assert "no files were written" in files, (
        f"nothing on this run mutated the workspace, so the brief invented a file: {files!r}"
    )
    assert "part00.txt" not in files, "a file that was only read is listed as touched"
    assert "ledger.txt" not in files, "a file that was only read is listed as touched"
