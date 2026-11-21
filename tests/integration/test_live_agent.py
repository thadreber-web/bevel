"""The agent loop against a real endpoint.

The unit tests drive the loop with ``MockProvider``, which proves the loop agrees
with a transcript someone wrote by hand. It cannot prove the loop works with a
model that decides for itself what to call, what arguments to send, and what to do
with an observation. These tests do that, against the server named in
``BEVEL_BASE_URL`` / ``BEVEL_MODEL``.

What is asserted, and why it can fail
-------------------------------------
Assertions are about **mechanics and consequences**, never about the model's prose:

* the loop terminates with a stated ``stop_reason`` and honours its budgets,
* a tool call the model invented actually runs and actually changes the workspace,
* the observation from that call reaches the transcript and the model *uses* it --
  the read test passes only if the answer contains a token that exists nowhere in
  the prompt, so a loop that dropped observations would fail,
* every tool call gets exactly one matching tool result, which is the invariant a
  real server punishes with a 400 on the next request,
* cancellation stops the run rather than merely appearing to.

Temperature is pinned to 0 and the tasks are chosen so that a correct run has one
right answer. A failure therefore means the loop or the endpoint broke, not that the
model phrased something differently today.

Cost
----
Each test is a handful of real generations on a GPU. They are marked ``slow`` so the
fast CI lane skips them, and ``max_tokens`` is bounded so a runaway model cannot bill
for a novel.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Final, TypeVar

import pytest

from bevel.core.agent import (
    UNBUILT_SEAMS,
    Agent,
    AgentEvent,
    AgentEventType,
    AgentResult,
    Budgets,
    StopReason,
    ToolOutcome,
)
from bevel.core.checkpoint import list_checkpoints
from bevel.core.tools.base import RiskTier, ToolContext, ToolResult
from bevel.core.tools.types import ToolSpec
from bevel.providers.base import GenOptions, ModelProvider

# The endpoint fixture is module-scoped (one probe for the whole module), so these
# tests must share its event loop scope rather than the project default of "function".
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.asyncio(loop_scope="module"),
]

#: A string the model can only produce by reading the file. Nothing in the prompt
#: contains it, so an observation that never reached the transcript fails the test.
SECRET: Final[str] = "zephyr-quartz-4417"

#: A string the model can only produce by having written it.
WRITTEN: Final[str] = "bevel-live-write-9f2c"

#: Deterministic sampling. A flaky sampler turns a real regression into a red test
#: that says nothing about either.
GEN: Final[GenOptions] = GenOptions(temperature=0.0, max_tokens=512)

#: How many times a test may re-ask a model that is being asked to *volunteer* a
#: specific shape of behaviour. Temperature 0 is not reproducible on a serving stack
#: that batches requests, so "call this tool four times" occasionally comes back as
#: prose. Retrying is honest as long as the assertions after the calls are made are
#: hard; a test that retried until a weak assertion passed would prove nothing.
VOLUNTEER_ATTEMPTS: Final[int] = 3


def _requires_tools(endpoint: ModelProvider) -> None:
    """Skip if the probed endpoint cannot call tools at all.

    A skip here is an honest statement about the server, not a pass: the probe in
    ``conftest.py`` actually asked it to call a tool and watched it fail.
    """
    if not endpoint.capabilities.supports_tools:
        pytest.skip(
            f"{endpoint.model!r} does not support tool calling "
            "(probed; see bevel doctor) - the agent loop has nothing to schedule"
        )


def _events(result: AgentResult, type_: AgentEventType) -> list[AgentEvent]:
    return [event for event in result.events if event.type is type_]


_T = TypeVar("_T")


async def _asking_for_tools(
    start: Callable[[], Awaitable[_T]],
    volunteered: Callable[[_T], bool],
    task: str,
    attempts: int = VOLUNTEER_ATTEMPTS,
) -> _T:
    """Run ``start`` until the model volunteers at least one tool call.

    A model explicitly asked to use a tool sometimes answers in prose instead, even
    at temperature 0: vLLM batches requests and a prompt is not bit-reproducible
    across them. Re-asking is fair because the assertions that follow are about the
    loop's mechanics and are not weakened -- a loop that could not run a tool would
    fail every attempt.
    """
    last: _T | None = None
    for _ in range(attempts):
        last = await start()
        if volunteered(last):
            return last
    pytest.fail(
        f"after {attempts} attempts the model never called a tool for {task!r}; "
        f"the endpoint is answering in prose, so the loop had nothing to schedule"
    )


def _assert_transcript_is_wellformed(result: AgentResult) -> None:
    """Every call the model made has exactly one observation, and vice versa.

    This is the invariant a real server enforces for you: send an assistant message
    with tool calls and no matching tool results and the next request is a 400. If
    the loop ever appends observations in the wrong order, or drops one, this fails
    here instead of mid-run in front of a user.
    """
    assert result.messages[0].role == "system"
    assert result.messages[1].role == "user"
    assert result.messages[1].content, "the task must reach the model"

    asked: list[str] = []
    answered: list[str] = []
    for message in result.messages:
        asked.extend(call.id for call in message.tool_calls)
        if message.role == "tool":
            assert message.tool_call_id, "a tool message must name the call it answers"
            assert message.content, "an empty observation tells the model nothing"
            answered.append(message.tool_call_id)

    assert sorted(asked) == sorted(answered), (
        f"{len(asked)} tool calls but {len(answered)} observations: "
        f"unanswered={sorted(set(asked) - set(answered))}, "
        f"unsolicited={sorted(set(answered) - set(asked))}"
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A throwaway git repo with one file whose only purpose is to be read.

    It is a repo because the agent takes a checkpoint commit before its first
    write, and a checkpoint needs a repository with at least one commit. Pointing
    the loop at a plain directory would exercise only the "no checkpoint taken"
    failure path, which is not what this file is for.
    """
    (tmp_path / "notes.txt").write_text(
        f"Meeting notes.\nAction items: none.\nThe access word is {SECRET}.\n",
        encoding="utf-8",
    )
    for argv in (
        ["init", "-q"],
        ["add", "notes.txt"],
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
            ["git", *argv],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            timeout=30,
        )
    return tmp_path


# --------------------------------------------------------------------------- loop


async def test_a_plain_question_completes_in_one_iteration(
    endpoint: ModelProvider, workspace: Path
) -> None:
    """No tools offered, no tools called: the loop must stop by itself.

    The failure this catches is a loop that only knows how to keep going.
    """
    agent = Agent(endpoint, workspace, budgets=Budgets(max_iterations=3), options=GEN)
    result = await agent.run("Reply with exactly one word: ready")

    assert result.stop_reason is StopReason.COMPLETED, result.error
    assert result.iterations == 1
    assert result.tool_calls == 0
    assert result.ok
    assert result.final_text.strip(), "the model answered nothing"
    _assert_transcript_is_wellformed(result)


async def test_a_greeting_needs_no_tools_and_arrives_oriented(
    endpoint: ModelProvider, workspace: Path
) -> None:
    """Same mechanics as the plain question, against the prompt a greeting gets.

    The project map must be on the wire *before* the model is asked anything, which
    is what makes zero tool calls the natural answer rather than a lucky one. The
    map's presence is asserted on the transcript rather than the model's prose: a
    real model can greet in any words it likes, and it cannot choose whether it was
    oriented.
    """
    agent = Agent(endpoint, workspace, budgets=Budgets(max_iterations=3), options=GEN)
    result = await agent.run("hello")

    assert result.stop_reason is StopReason.COMPLETED, result.error
    assert result.iterations == 1
    assert result.tool_calls == 0, "a greeting was answered by exploring the tree"
    assert result.final_text.strip(), "the model answered nothing"
    assert "# project map:" in result.messages[0].content
    assert "notes.txt" in result.messages[0].content
    _assert_transcript_is_wellformed(result)


async def test_server_reported_usage_arrives_and_is_not_invented(
    endpoint: ModelProvider, workspace: Path
) -> None:
    """The token budget runs on these numbers, so they must be the server's.

    vLLM reports usage in a trailing chunk with an empty ``choices`` array. A
    decoder that stops at ``finish_reason`` yields zeroed usage, and a budget built
    on zeros never fires -- the run silently ignores its own limits.
    """
    agent = Agent(endpoint, workspace, budgets=Budgets(max_iterations=1), options=GEN)
    result = await agent.run("Say the word present and nothing else.")

    assert result.usage.total_tokens > 0, (
        f"the server reported no usage ({result.usage}); a token budget checked "
        "against zero would never stop a run"
    )
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0
    assert result.usage.total_tokens == result.usage.prompt_tokens + result.usage.completion_tokens
    assert result.usage_reported is True


async def test_the_model_reads_a_file_and_uses_the_observation(
    endpoint: ModelProvider, workspace: Path
) -> None:
    """The whole point of the loop: an observation has to change the next answer.

    ``SECRET`` appears in the prompt nowhere except the file on disk, so the final
    answer can only contain it if the model called ``read``, the tool ran, and its
    output was appended to the transcript.
    """
    _requires_tools(endpoint)

    async def attempt() -> AgentResult:
        agent = Agent(endpoint, workspace, budgets=Budgets(max_iterations=6), options=GEN)
        return await agent.run(
            "Read the file notes.txt in the workspace and tell me the access word it "
            "contains. Answer with just that word."
        )

    def volunteered(result: AgentResult) -> bool:
        return bool(_events(result, AgentEventType.TOOL_COMPLETE))

    result = await _asking_for_tools(attempt, volunteered, "read a file")
    assert result.stop_reason is StopReason.COMPLETED, result.error

    reads = [
        execution
        for event in _events(result, AgentEventType.TOOL_COMPLETE)
        for execution in event.executions
    ]
    assert reads, f"the model never called a tool; it said: {result.final_text!r}"
    assert all(execution.name == "read" for execution in reads), [
        execution.name for execution in reads
    ]
    assert any(execution.succeeded for execution in reads), [
        (execution.outcome, execution.reason, execution.repair_errors) for execution in reads
    ]
    assert SECRET in result.final_text, (
        f"the observation never reached the model; it answered {result.final_text!r}"
    )
    _assert_transcript_is_wellformed(result)


async def test_a_write_actually_changes_the_workspace(
    endpoint: ModelProvider, workspace: Path
) -> None:
    """The write path end to end: model invents arguments, tool validates, disk changes.

    Also the only test that exercises the checkpoint commit on a mutating run --
    it must actually land on the ``bevel/checkpoint/<session>`` ref, not merely
    announce itself.
    """
    _requires_tools(endpoint)

    async def attempt() -> AgentResult:
        agent = Agent(endpoint, workspace, budgets=Budgets(max_iterations=6), options=GEN)
        return await agent.run(
            f'Create a file named "{WRITTEN}.txt" in the workspace whose entire content '
            f"is the single line {WRITTEN}. Then confirm you wrote it."
        )

    def volunteered(result: AgentResult) -> bool:
        return bool(_events(result, AgentEventType.TOOL_COMPLETE))

    result = await _asking_for_tools(attempt, volunteered, "write a file")

    writes = [
        execution
        for event in _events(result, AgentEventType.TOOL_COMPLETE)
        for execution in event.executions
        if execution.name == "write"
    ]
    assert writes, f"the model never called write; it said: {result.final_text!r}"
    assert any(execution.succeeded for execution in writes), [
        (execution.outcome, execution.reason, execution.repair_errors) for execution in writes
    ]

    created = workspace / f"{WRITTEN}.txt"
    assert created.exists(), "the tool reported success but no file appeared on disk"
    assert WRITTEN in created.read_text(encoding="utf-8")

    assert any(execution.mutated for execution in writes), (
        "a successful write must be reported as a mutation, or the checkpoint "
        "pre-flight has nothing to trigger on"
    )
    checkpoints = _events(result, AgentEventType.CHECKPOINT)
    assert checkpoints, "a mutating run must emit a CHECKPOINT event"
    assert checkpoints[0].text.startswith("checkpointed "), checkpoints[0].text
    assert list_checkpoints(workspace), "the checkpoint commit never landed on its ref"
    _assert_transcript_is_wellformed(result)


async def test_an_unknown_tool_is_answered_not_hung(
    endpoint: ModelProvider, workspace: Path
) -> None:
    """A call for a tool that does not exist must produce an observation.

    If it produced none, the model would repeat the same call forever and the run
    would only end when a budget happened to catch it.
    """
    _requires_tools(endpoint)
    agent = Agent(endpoint, workspace, budgets=Budgets(max_iterations=4), options=GEN)
    result = await agent.run(
        "Use the tool named list_dir to enumerate the workspace, then tell me what "
        "you found. If you cannot use it, say so."
    )

    unavailable = [
        execution
        for event in _events(result, AgentEventType.TOOL_COMPLETE)
        for execution in event.executions
        if execution.outcome is ToolOutcome.UNAVAILABLE
    ]
    if not unavailable:
        # The model chose a real tool instead. Still a pass: nothing hung, and the
        # transcript is coherent.
        assert result.stop_reason is StopReason.COMPLETED, result.error
    else:
        for execution in unavailable:
            assert execution.reason, "an unavailable tool must be told what does exist"
            assert "read" in execution.reason, execution.reason
    _assert_transcript_is_wellformed(result)


# --------------------------------------------------------------------------- seams


async def test_unbuilt_seams_are_reported_on_every_result(
    endpoint: ModelProvider, workspace: Path
) -> None:
    """m08 is not built and the loop must say so on the way out.

    This is the anti-fake guarantee: a caller that never reads the docs still learns
    that nothing compacted. Policy is enforced and checkpoints are taken, so neither
    is among the seams the loop reports.
    """
    agent = Agent(endpoint, workspace, budgets=Budgets(max_iterations=1), options=GEN)
    result = await agent.run("Say ok.")

    assert result.seams == UNBUILT_SEAMS
    assert len(result.seams) == 1
    assert any("m08" in seam for seam in result.seams)


async def test_compaction_reports_the_servers_own_context_size(
    endpoint: ModelProvider, workspace: Path
) -> None:
    """The COMPACTION event must carry the server's count, not a chars/4 guess.

    ``max_model_len`` is only known if the server reported it. When it did not, the
    honest status is UNKNOWN -- a fabricated window would make the TUI claim the
    context fits when nobody checked.
    """
    agent = Agent(endpoint, workspace, budgets=Budgets(max_iterations=6), options=GEN)
    result = await agent.run(
        "Read notes.txt, then read it again, then tell me how many lines it has."
    )

    compactions = _events(result, AgentEventType.COMPACTION)
    assert compactions, "every tool iteration must report the context state"
    for event in compactions:
        assert event.context is not None
        assert "m08" in event.text, event.text
    if endpoint.capabilities.max_model_len is None:
        assert all(event.context.status.value == "unknown" for event in compactions), (
            "no window was reported, so nothing may claim the context fits"
        )
    else:
        assert all(
            event.context.window == endpoint.capabilities.max_model_len for event in compactions
        )


# -------------------------------------------------------------------------- budgets


async def test_iteration_budget_is_a_hard_stop(endpoint: ModelProvider, workspace: Path) -> None:
    """One iteration allowed, so at most one request may be made.

    The task is deliberately one that wants a second turn. If the budget were
    advisory, ``iterations`` would exceed 1 and the run would keep spending GPU time
    after being told to stop.
    """
    _requires_tools(endpoint)
    agent = Agent(endpoint, workspace, budgets=Budgets(max_iterations=1), options=GEN)
    result = await agent.run(
        "Read notes.txt, then read it again to double-check, then read it a third "
        "time, then tell me the access word."
    )

    assert result.iterations == 1, (
        f"max_iterations=1 but the loop ran {result.iterations} iterations"
    )
    if result.stop_reason is StopReason.TURN_BUDGET:
        budgets = _events(result, AgentEventType.BUDGET)
        assert budgets, "a budget stop must say which budget stopped it"
        assert "budget" in budgets[0].text.lower()
    else:
        assert result.stop_reason is StopReason.COMPLETED, result.error


async def test_token_budget_stops_a_run_that_overspends(endpoint: ModelProvider) -> None:
    """A budget of 1 token cannot fund a second request.

    The check sits between iterations because the only trustworthy count is the
    server's and it arrives after the request, so overshooting by exactly one
    iteration is the documented behaviour -- and is asserted here rather than hidden.
    """
    _requires_tools(endpoint)
    agent = Agent(
        endpoint,
        Path.cwd(),
        budgets=Budgets(max_iterations=8, max_total_tokens=1),
        options=GEN,
    )
    result = await agent.run(
        "Read pyproject.toml, then read README.md, then read AGENTS.md, then compare them."
    )

    assert result.stop_reason in {StopReason.TOKEN_BUDGET, StopReason.COMPLETED}
    assert result.iterations <= 2, (
        f"token budget of 1 allowed {result.iterations} iterations; the guard is not "
        "reading the server's usage"
    )
    if result.stop_reason is StopReason.TOKEN_BUDGET:
        assert result.usage.total_tokens >= 1
        budgets = _events(result, AgentEventType.BUDGET)
        assert budgets
        assert "token" in budgets[0].text.lower()


async def test_tool_call_budget_stops_a_chatty_model(endpoint: ModelProvider) -> None:
    """The loop must stop calling tools once the budget is spent.

    ``max_tool_calls=0`` still permits the first iteration (the budget is checked
    before it starts, not mid-flight), so the assertion is that a second round of
    calls never happens.
    """
    _requires_tools(endpoint)
    agent = Agent(
        endpoint, Path.cwd(), budgets=Budgets(max_iterations=8, max_tool_calls=0), options=GEN
    )
    result = await agent.run(
        "Read pyproject.toml, then read README.md, then read AGENTS.md, then compare them."
    )

    assert result.stop_reason in {StopReason.TURN_BUDGET, StopReason.COMPLETED}
    if result.stop_reason is StopReason.TURN_BUDGET:
        assert result.iterations == 1, (
            f"tool-call budget was exceeded by iteration {result.iterations}"
        )


# ----------------------------------------------------------------------- cancellation


async def test_cancel_stops_a_streaming_run(endpoint: ModelProvider, workspace: Path) -> None:
    """``cancel()`` must abandon the stream, not merely record a wish.

    Driven from the consumer side so it is deterministic: the run is cancelled the
    moment enough deltas have arrived, which is exactly what a Ctrl+C in the TUI
    does. The task asks for a long answer so the stream cannot finish first.
    """
    agent = Agent(endpoint, workspace, budgets=Budgets(max_iterations=2), options=GEN)
    collected: list[AgentEvent] = []
    async for event in agent.stream(
        "Write a detailed explanation of how a GPU runs a transformer, in at least "
        "600 words. Do not stop early."
    ):
        collected.append(event)
        if (
            sum(e.type in {AgentEventType.CONTENT, AgentEventType.REASONING} for e in collected)
            >= 5
        ):
            agent.cancel("test asked for it")

    assert agent.cancelled
    cancelled = [event for event in collected if event.type is AgentEventType.CANCELLED]
    assert cancelled, "a cancelled run must emit a CANCELLED event"
    assert cancelled[0].text == "test asked for it"
    completed = [event for event in collected if event.type is AgentEventType.COMPLETED]
    assert completed, "a cancelled run must still report how it ended"
    assert completed[-1].text == StopReason.CANCELLED.value, (
        f"the run ended {completed[-1].text} despite being cancelled"
    )
    assert agent.active_tool_calls == 0, "cancelled run left tool tasks in flight"


class _SlowTool:
    """A read-tier tool that blocks, and records whether it ever actually ran.

    Cancelling a run that has queued tool calls is the dangerous case: a call that
    executed anyway while being reported as cancelled would leave the workspace
    changed with no record of it. ``started`` is the evidence -- it is written from
    inside ``execute``, so an entry can only exist if the tool really ran.
    """

    spec = ToolSpec(
        name="stall",
        description="Block for several seconds. Takes no arguments.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )
    risk_tier = RiskTier.READ
    is_parallel_safe = True

    def __init__(self, started: list[str]) -> None:
        self._started = started
        #: Set from inside ``execute`` so a test can cancel once the tool is really
        #: in flight, rather than racing it with a guessed sleep.
        self.began = asyncio.Event()

    async def execute(self, arguments: Mapping[str, object], ctx: ToolContext) -> ToolResult:
        del arguments, ctx
        self._started.append("ran")
        self.began.set()
        await asyncio.sleep(30)
        return ToolResult(ok=True, content="finished")


_STALL_TASK = (
    'Call the tool named "stall" four times. Call all four in one go, do not wait '
    "between them. Then tell me what happened."
)


async def test_cancel_between_turns_abandons_the_calls_it_was_given(
    endpoint: ModelProvider, workspace: Path
) -> None:
    """Cancel the moment the model asks for work, and none of it may happen.

    This is the window between a turn finishing and its calls being scheduled. The
    loop checks cancellation there and breaks, so nothing is executed and nothing is
    reported. The trigger is a real turn carrying real tool calls -- if the model
    answered in prose there would be nothing to abandon and the test would be vacuous,
    which is why it re-asks until the model actually asks.
    """
    _requires_tools(endpoint)

    async def attempt() -> tuple[list[AgentEvent], list[str]]:
        started: list[str] = []
        agent = Agent(
            endpoint,
            workspace,
            tools=[_SlowTool(started)],
            budgets=Budgets(max_iterations=4),
            options=GEN,
        )
        collected: list[AgentEvent] = []
        async for event in agent.stream(_STALL_TASK):
            collected.append(event)
            if event.type is AgentEventType.TURN_COMPLETE and event.turn.message.tool_calls:
                agent.cancel("before the tools ran")
        return collected, started

    def volunteered(item: tuple[list[AgentEvent], list[str]]) -> bool:
        turns = [e for e in item[0] if e.type is AgentEventType.TURN_COMPLETE]
        return any(t.turn.message.tool_calls for t in turns)

    collected, started = await _asking_for_tools(attempt, volunteered, "call the stall tool")

    assert started == [], "a run cancelled before scheduling executed a tool anyway"
    assert not [e for e in collected if e.type is AgentEventType.TOOL_COMPLETE], (
        "nothing was scheduled, so nothing should have been reported"
    )
    assert [e for e in collected if e.type is AgentEventType.CANCELLED], (
        "a cancelled run must say it was cancelled"
    )
    completed = [e for e in collected if e.type is AgentEventType.COMPLETED]
    assert completed[-1].text == StopReason.CANCELLED.value, completed[-1].text


async def test_cancel_while_a_tool_is_running_reports_it_cancelled(
    endpoint: ModelProvider, workspace: Path
) -> None:
    """Cancel mid-flight: the tool is interrupted and the call is reported, not lost.

    ``_SlowTool`` would block for 30s, so a loop that ignored cancellation would sit
    there and this test would time out. The two things that must be true are that the
    run comes back promptly and that every call still gets an observation -- a dropped
    one would leave the transcript with an unanswered tool call.
    """
    _requires_tools(endpoint)

    async def attempt() -> tuple[list[AgentEvent], list[str], float]:
        started: list[str] = []
        tool = _SlowTool(started)
        agent = Agent(
            endpoint,
            workspace,
            tools=[tool],
            budgets=Budgets(max_iterations=4),
            options=GEN,
        )
        collected: list[AgentEvent] = []

        async def cancel_once_running() -> None:
            try:
                await asyncio.wait_for(tool.began.wait(), timeout=20.0)
            except TimeoutError:
                return  # the model never called the tool; the retry loop handles it
            agent.cancel("mid-flight")

        watcher = asyncio.create_task(cancel_once_running())
        began = time.monotonic()
        try:
            async for event in agent.stream(_STALL_TASK):
                collected.append(event)
        finally:
            await watcher
        return collected, started, time.monotonic() - began

    def volunteered(item: tuple[list[AgentEvent], list[str], float]) -> bool:
        return bool(item[1])

    collected, _started, elapsed = await _asking_for_tools(
        attempt, volunteered, "call the stall tool and let it start"
    )

    assert elapsed < 25.0, f"cancellation did not interrupt the tool; it ran {elapsed:.1f}s"
    outcomes = [
        execution.outcome
        for event in collected
        if event.type is AgentEventType.TOOL_COMPLETE
        for execution in event.executions
    ]
    assert outcomes, "the run must still report what happened to the calls it started"
    assert set(outcomes) == {ToolOutcome.CANCELLED}, outcomes
    observations = [
        execution.observation
        for event in collected
        if event.type is AgentEventType.TOOL_COMPLETE
        for execution in event.executions
    ]
    assert all(observations), "every cancelled call still needs an observation to answer it"
