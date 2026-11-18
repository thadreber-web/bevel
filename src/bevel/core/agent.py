"""The agent loop: orient → plan → act → observe, under budgets.

One iteration is:

1. assemble the context that goes on the wire,
2. stream a completion,
3. parse its tool calls **through the repair layer**,
4. evaluate each call against policy,
5. execute — parallel where the tool says it is safe, capped, serial otherwise,
6. append an observation for every call,
7. checkpoint if anything mutated,
8. compact if the context is over budget,
9. repeat, until the model stops asking for tools or a budget runs out.

Steps 1, 7 and 8 are **seams**, not features. They are named in
:data:`UNBUILT_SEAMS`, reported on every :class:`AgentResult`, and logged once per
run. Nothing here pretends to take a checkpoint or compact a transcript. Policy
(step 4) is enforced by :class:`~bevel.core.policy.PolicyEngine`.

Cancellation
------------
Two paths, deliberately different:

* :meth:`Agent.cancel` is the graceful one. It sets a flag and cancels the in-flight
  tool tasks — for the shell tool that means the whole process tree, not just the
  parent — and makes the stream loop abandon the provider's generator, whose
  ``finally`` closes the HTTP response. :meth:`run` then returns normally with
  ``stop_reason`` CANCELLED.
* Cancelling the surrounding ``asyncio.Task`` is the forceful one. ``CancelledError``
  is never swallowed here, so it propagates through the provider generator (closing
  the response) and through ``asyncio.gather`` (cancelling every child task).

Budgets
-------
The only trustworthy token count is the server's, and it arrives *after* the request
(see ``docs/provider-notes.md``: there is no ``/tokenize`` to predict with). The
token budget is therefore checked **between** iterations, so a run can overshoot by
exactly one iteration. That is stated rather than hidden.

There is no cost budget: nothing in this repo knows what a token costs, so nothing
here prints a dollar figure.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from bevel.config.schema import PolicyConfig
from bevel.core.messages import Message, ToolCall
from bevel.core.policy import (
    Approver,
    Decision,
    PolicyEngine,
    PolicyEvaluator,
    PolicyRequest,
    PolicyVerdict,
    allow_all_policy,
)
from bevel.core.tools.base import RiskTier, Tool, ToolContext, ToolResult, call_tool
from bevel.core.tools.registry import tools as builtin_tools
from bevel.core.tools.types import ToolSpec
from bevel.providers.base import (
    AssistantTurn,
    ContentDelta,
    GenOptions,
    ModelProvider,
    ProviderError,
    ReasoningDelta,
    StreamEvent,
    Usage,
    assemble,
)
from bevel.providers.repair import ToolCallRepairer

logger = logging.getLogger("bevel.agent")

__all__ = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_SYSTEM_PROMPT",
    "UNBUILT_SEAMS",
    "Agent",
    "AgentEvent",
    "AgentEventType",
    "AgentResult",
    "Budgets",
    "CheckpointHook",
    "CheckpointRequest",
    "CompactionStatus",
    "ContextState",
    "Decision",
    "PolicyEvaluator",
    "PolicyRequest",
    "PolicyVerdict",
    "StopReason",
    "ToolExecution",
    "ToolOutcome",
    "allow_all_policy",
    "no_checkpoint",
]

#: How many parallel-safe tool calls may be in flight at once.
DEFAULT_CONCURRENCY: Final[int] = 4

_SEAM_CHECKPOINT: Final[str] = (
    "m06 (per-session worktree, checkpoint commits, /undo) is not built: writes land "
    "directly in the working tree and there is nothing to roll back to"
)
_SEAM_CONTEXT: Final[str] = (
    "m08 (context budgeting, L1/L2/L3 compaction, session persistence) is not built: "
    "the transcript is never summarised, truncated, or saved — it only grows"
)

#: What this loop does *not* do, in the words the loop itself reports.
UNBUILT_SEAMS: Final[tuple[str, ...]] = (_SEAM_CHECKPOINT, _SEAM_CONTEXT)

DEFAULT_SYSTEM_PROMPT: Final[str] = (
    "You are bevel, a coding agent working in a local workspace.\n"
    "\n"
    "Find things out instead of guessing: read before you edit, grep to locate, glob to "
    "enumerate. Make the smallest change that fixes the problem, and never claim a change "
    'works unless you have run something that proves it. "Not built yet" is a correct '
    "answer; invented output is not.\n"
    "\n"
    "Limits of this build: there is no checkpointing (milestone m06), so a write cannot "
    "be rolled back. Policy is enforced: reads and workspace writes proceed, and anything "
    "the policy engine decides needs your approval is refused unless an approver is "
    "installed. Say what you need instead of assuming you have it."
)


class ToolOutcome(StrEnum):
    """Why a tool call ended the way it did."""

    OK = "ok"
    FAILED = "failed"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    REPAIR_FAILED = "repair_failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ToolExecution:
    """One tool call and what happened to it.

    Carries the repair fixes too: a silently repaired call and a first-time-right call
    are not the same event, and the transcript should be able to tell them apart.
    """

    call: ToolCall
    tool: Tool | None
    outcome: ToolOutcome
    result: ToolResult | None = None
    fixes: tuple[str, ...] = ()
    repair_errors: tuple[str, ...] = ()
    reason: str = ""

    @property
    def name(self) -> str:
        return self.call.name

    @property
    def succeeded(self) -> bool:
        return self.outcome is ToolOutcome.OK and self.result is not None and self.result.ok

    @property
    def mutated(self) -> bool:
        """True when this call is expected to have changed the working tree."""
        return (
            self.succeeded
            and self.tool is not None
            and self.tool.risk_tier is RiskTier.WRITE_WORKSPACE
        )

    @property
    def repaired(self) -> bool:
        return bool(self.fixes)

    @property
    def observation(self) -> str:
        """The text appended to the transcript as this call's observation."""
        if self.outcome is ToolOutcome.OK and self.result is not None:
            return self.result.content
        if self.outcome is ToolOutcome.FAILED and self.result is not None:
            return f"Tool {self.call.name!r} failed: {self.result.error}\nNothing further happened."
        if self.outcome is ToolOutcome.UNAVAILABLE:
            return (
                f"Tool {self.call.name!r} does not exist. Tools you can use: {self.reason}. "
                "Nothing was executed."
            )
        if self.outcome is ToolOutcome.REJECTED:
            return f"Tool {self.call.name!r} was not executed: {self.reason}"
        if self.outcome is ToolOutcome.REPAIR_FAILED:
            lines = "\n".join(f"- {error}" for error in self.repair_errors)
            return (
                f"Tool call {self.call.name!r} was rejected before execution: its arguments "
                f"could not be made to match the tool's schema.\n{lines}\n"
                "Nothing was executed. Correct the arguments and try again."
            )
        return (
            f"Tool {self.call.name!r} was cancelled before it finished. It may have partially "
            "run; the result is unknown."
        )


@dataclass(frozen=True)
class CheckpointRequest:
    """What a checkpoint hook (m06) would be handed after a mutating iteration."""

    iteration: int
    mutated: tuple[str, ...]


CheckpointHook = Callable[[CheckpointRequest], Awaitable[None]]


async def no_checkpoint(request: CheckpointRequest) -> None:
    """The m06 placeholder: does nothing at all.

    Named rather than omitted so that the absence of checkpointing is a decision
    visible in the code, the event stream, and the result.
    """


class CompactionStatus(StrEnum):
    WITHIN_BUDGET = "within_budget"
    OVER_BUDGET = "over_budget"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ContextState:
    """How full the context is, according to the only number we trust.

    ``prompt_tokens`` is the server's own count from the last completed request. When
    the server did not report it, or advertises no window, the status is UNKNOWN — the
    loop refuses to invent a window size it was never told.
    """

    status: CompactionStatus
    prompt_tokens: int | None = None
    window: int | None = None
    over_by: int | None = None


class AgentEventType(StrEnum):
    STARTED = "started"
    REASONING = "reasoning"
    CONTENT = "content"
    TURN_COMPLETE = "turn_complete"
    TOOL_COMPLETE = "tool_complete"
    CHECKPOINT = "checkpoint"
    COMPACTION = "compaction"
    BUDGET = "budget"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


@dataclass(frozen=True)
class AgentEvent:
    """Something the loop did, for a UI (m07) or a log to render.

    Which fields are populated depends on ``type``: ``text`` for the delta, budget and
    seam events, ``turn``/``usage`` for TURN_COMPLETE, ``executions`` for
    TOOL_COMPLETE, ``context`` for COMPACTION.
    """

    type: AgentEventType
    iteration: int
    text: str = ""
    turn: AssistantTurn | None = None
    usage: Usage | None = None
    executions: tuple[ToolExecution, ...] = ()
    context: ContextState | None = None


class StopReason(StrEnum):
    COMPLETED = "completed"
    TURN_BUDGET = "turn_budget"
    TOKEN_BUDGET = "token_budget"  # noqa: S105
    CANCELLED = "cancelled"
    PROVIDER_ERROR = "provider_error"


@dataclass(frozen=True)
class Budgets:
    """Hard stops for one run.

    ``max_total_tokens`` counts **cumulative** tokens across the run — every request's
    ``prompt_tokens`` again, since that is what the server actually processed each
    time. It is not a context-window limit; that is
    :attr:`~bevel.providers.base.Capabilities.max_model_len`.
    """

    max_iterations: int = 24
    max_tool_calls: int = 200
    max_total_tokens: int | None = None


@dataclass(frozen=True)
class AgentResult:
    """How a run ended, and everything the loop put on the wire."""

    messages: tuple[Message, ...]
    stop_reason: StopReason
    iterations: int
    tool_calls: int
    usage: Usage
    events: tuple[AgentEvent, ...]
    seams: tuple[str, ...] = UNBUILT_SEAMS
    usage_reported: bool = True
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.stop_reason is StopReason.COMPLETED

    @property
    def final_text(self) -> str:
        """The last assistant message's content, or empty if there was none."""
        for message in reversed(self.messages):
            if message.role == "assistant":
                return message.content
        return ""


@dataclass
class _Pending:
    """A tool call on its way from the stream to the scheduler.

    ``outcome`` is set by :meth:`Agent._prepare` for calls that never reach
    ``execute`` (unknown tool, failed repair, denial, cancellation-before-start), so
    the scheduler can skip them and the caller can still record an observation.
    """

    call: ToolCall
    tool: Tool | None
    arguments: dict[str, Any] | None = None
    fixes: tuple[str, ...] = ()
    repair_errors: tuple[str, ...] = ()
    reason: str = ""
    outcome: ToolOutcome | None = None

    @property
    def parallel_safe(self) -> bool:
        return self.tool is not None and self.tool.is_parallel_safe

    def as_execution(self) -> ToolExecution:
        return ToolExecution(
            call=self.call,
            tool=self.tool,
            outcome=self.outcome or ToolOutcome.CANCELLED,
            result=None,
            fixes=self.fixes,
            repair_errors=self.repair_errors,
            reason=self.reason,
        )


@dataclass
class _RunState:
    """What :meth:`Agent.run` needs back after the event generator is exhausted."""

    messages: list[Message] = field(default_factory=list)
    events: list[AgentEvent] = field(default_factory=list)
    iterations: int = 0
    tool_calls: int = 0
    stop: StopReason = StopReason.COMPLETED
    error: str | None = None
    usage: Usage = field(default_factory=Usage)
    usage_reported: bool = True


class _Totals:
    """Server-reported usage for one run: the running total, and the last context size.

    Two different numbers, deliberately kept apart. ``cumulative`` is what the run
    cost — every request's prompt tokens again, because that is what the server
    reprocessed each time. ``context_tokens`` is how full the *last* request was,
    which is the only thing comparable to ``max_model_len``.
    """

    def __init__(self) -> None:
        self.cumulative = Usage()
        self.context_tokens: int | None = None
        self.reported = True

    def add(self, usage: Usage) -> None:
        if not (usage.total_tokens or usage.prompt_tokens or usage.completion_tokens):
            self.reported = False
        self.cumulative = Usage(
            prompt_tokens=self.cumulative.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self.cumulative.completion_tokens + usage.completion_tokens,
            reasoning_tokens=self.cumulative.reasoning_tokens + usage.reasoning_tokens,
            total_tokens=self.cumulative.total_tokens + usage.total_tokens,
        )
        if usage.prompt_tokens:
            self.context_tokens = usage.prompt_tokens

    def over_budget(self, limit: int | None) -> bool:
        return limit is not None and self.cumulative.total_tokens >= limit

    def context_state(self, window: int | None) -> ContextState:
        prompt = self.context_tokens
        if prompt is None or window is None:
            return ContextState(CompactionStatus.UNKNOWN, prompt_tokens=prompt, window=window)
        if prompt > window:
            return ContextState(
                CompactionStatus.OVER_BUDGET,
                prompt_tokens=prompt,
                window=window,
                over_by=prompt - window,
            )
        return ContextState(CompactionStatus.WITHIN_BUDGET, prompt_tokens=prompt, window=window)


async def _replay(events: Sequence[StreamEvent]) -> AsyncIterator[StreamEvent]:
    """Re-feed collected stream events through :func:`assemble`.

    The loop has to *see* each event to forward it as it arrives, which means it
    cannot also hand the same generator to ``assemble``. Replaying the collected
    list keeps ``assemble`` the single fold rule instead of a second copy of it.
    """
    for event in events:
        yield event


class Agent:
    """Drives a model to completion over a workspace.

    The loop owns the transcript; the caller owns the provider and the workspace.
    Tool definitions come from the registry and nowhere else — pass ``tools`` to
    narrow them, never to become a second source of truth.
    """

    def __init__(
        self,
        provider: ModelProvider,
        workspace: Path | str,
        *,
        system: str = DEFAULT_SYSTEM_PROMPT,
        tools: Sequence[Tool] | None = None,
        budgets: Budgets = Budgets(),
        concurrency: int = DEFAULT_CONCURRENCY,
        options: GenOptions | None = None,
        policy: PolicyEvaluator | None = None,
        policy_config: PolicyConfig | None = None,
        approver: Approver | None = None,
        on_checkpoint: CheckpointHook | None = None,
        repairer: ToolCallRepairer | None = None,
    ) -> None:
        if budgets.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if budgets.max_tool_calls < 0:
            raise ValueError("max_tool_calls cannot be negative")
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.provider = provider
        self.ctx = ToolContext(Path(workspace))
        self.system = system
        self.options = options
        self.budgets = budgets
        self.tools: tuple[Tool, ...] = tuple(tools) if tools is not None else tuple(builtin_tools())
        self._by_name: dict[str, Tool] = {tool.spec.name: tool for tool in self.tools}
        self._repairer = repairer or ToolCallRepairer(provider)
        self._policy: PolicyEvaluator = policy or PolicyEngine(
            self.ctx,
            config=policy_config,
            approver=approver,
        )
        self._on_checkpoint: CheckpointHook = on_checkpoint or no_checkpoint
        self._semaphore = asyncio.Semaphore(concurrency)
        self._tool_tasks: set[asyncio.Task[ToolExecution]] = set()
        self._cancel_event = asyncio.Event()
        self._cancel_reason = ""
        self._running = False

    # ------------------------------------------------------------------- public

    async def run(self, task: str) -> AgentResult:
        """Run the loop to completion and return the transcript."""
        state = _RunState()
        async for event in self._iterate(task, state):
            state.events.append(event)
        return AgentResult(
            messages=tuple(state.messages),
            stop_reason=state.stop,
            iterations=state.iterations,
            tool_calls=state.tool_calls,
            usage=state.usage,
            events=tuple(state.events),
            usage_reported=state.usage_reported,
            error=state.error,
        )

    def stream(self, task: str) -> AsyncIterator[AgentEvent]:
        """Run the loop, yielding each event as it happens.

        For the TUI (m07) and for tests. Stopping iteration early abandons the
        generator, which cancels nothing on its own — call :meth:`cancel` too.
        """
        return self._iterate(task, _RunState())

    def cancel(self, reason: str = "cancelled by user") -> None:
        """Stop the run at the next opportunity.

        Safe to call from another task or from a UI keybinding. In-flight tool tasks
        are cancelled — for the shell tool that means the whole process tree, not just
        the parent — and the provider stream is abandoned, which closes the HTTP
        response.
        """
        if self._cancel_event.is_set():
            return
        self._cancel_reason = reason
        self._cancel_event.set()
        for task in tuple(self._tool_tasks):
            task.cancel()

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def active_tool_calls(self) -> int:
        """Tool calls currently in flight as tasks."""
        return len(self._tool_tasks)

    # ------------------------------------------------------------------ private

    async def _iterate(self, task: str, state: _RunState) -> AsyncIterator[AgentEvent]:
        if self._running:
            raise RuntimeError("this Agent is already running")
        self._running = True
        self._cancel_reason = ""
        self._cancel_event.clear()
        state.messages.extend((Message.system(self.system), Message.user(task)))
        totals = _Totals()
        logger.warning("agent loop starting with unbuilt seams: %s", "; ".join(UNBUILT_SEAMS))
        yield AgentEvent(AgentEventType.STARTED, iteration=0, text=task)
        try:
            while True:
                stop = self._budget_stop(state, totals)
                if stop is not None:
                    state.stop = stop[0]
                    yield AgentEvent(
                        AgentEventType.BUDGET, iteration=state.iterations, text=stop[1]
                    )
                    break
                state.iterations += 1

                collected: list[StreamEvent] = []
                async for event in self._forward(state.messages, state.iterations, collected):
                    yield event
                turn = await assemble(_replay(collected))
                totals.add(turn.usage or Usage())
                state.messages.append(turn.message)
                yield AgentEvent(
                    AgentEventType.TURN_COMPLETE,
                    iteration=state.iterations,
                    turn=turn,
                    usage=turn.usage,
                    text=turn.message.content,
                )
                if self._cancel_event.is_set():
                    break
                if not turn.message.tool_calls:
                    state.stop = StopReason.COMPLETED
                    break

                async for event in self._run_tools(turn, state, totals):
                    yield event
                if self._cancel_event.is_set():
                    break
        except ProviderError as exc:
            state.stop = StopReason.PROVIDER_ERROR
            state.error = f"{type(exc).__name__}: {exc}"
            logger.warning("agent loop stopped on a provider error: %s", state.error)
        finally:
            for tool_task in tuple(self._tool_tasks):
                tool_task.cancel()
            self._tool_tasks.clear()
            self._running = False

        if self._cancel_event.is_set():
            state.stop = StopReason.CANCELLED
            yield AgentEvent(
                AgentEventType.CANCELLED, iteration=state.iterations, text=self._cancel_reason
            )
        state.usage = totals.cumulative
        state.usage_reported = totals.reported
        yield AgentEvent(
            AgentEventType.COMPLETED, iteration=state.iterations, text=state.stop.value
        )

    def _budget_stop(self, state: _RunState, totals: _Totals) -> tuple[StopReason, str] | None:
        """Why the next iteration must not start, or ``None`` to continue.

        Checked at the top of the loop rather than mid-request, so a run can overshoot
        the token budget by exactly one iteration — the server only reports usage
        after the fact, and there is nothing to predict it with.
        """
        if self._cancel_event.is_set():
            return StopReason.CANCELLED, self._cancel_reason
        if state.iterations >= self.budgets.max_iterations:
            return (
                StopReason.TURN_BUDGET,
                f"turn budget reached after {state.iterations} iterations",
            )
        if totals.over_budget(self.budgets.max_total_tokens):
            return (
                StopReason.TOKEN_BUDGET,
                f"token budget reached at {totals.cumulative.total_tokens} tokens",
            )
        if state.tool_calls > self.budgets.max_tool_calls:
            return (
                StopReason.TURN_BUDGET,
                f"tool-call budget reached after {state.tool_calls} calls "
                f"(limit {self.budgets.max_tool_calls})",
            )
        return None

    async def _forward(
        self, messages: Sequence[Message], iteration: int, collected: list[StreamEvent]
    ) -> AsyncIterator[AgentEvent]:
        """Forward deltas as they arrive, collecting the raw events on the way.

        The stream is abandoned the moment cancellation is requested, and closed
        explicitly in ``finally`` so the provider closes the HTTP response rather than
        leaving it for the garbage collector.
        """
        stream = self.provider.stream(list(messages), self._specs(), self.options)
        # The protocol only promises AsyncIterator; the generators the real providers
        # return do have aclose(), and closing one closes its HTTP response.
        close = getattr(stream, "aclose", None)
        try:
            async for event in stream:
                if self._cancel_event.is_set():
                    break
                collected.append(event)
                if isinstance(event, ReasoningDelta):
                    yield AgentEvent(AgentEventType.REASONING, iteration=iteration, text=event.text)
                elif isinstance(event, ContentDelta):
                    yield AgentEvent(AgentEventType.CONTENT, iteration=iteration, text=event.text)
        finally:
            if close is not None:
                await close()

    async def _run_tools(
        self, turn: AssistantTurn, state: _RunState, totals: _Totals
    ) -> AsyncIterator[AgentEvent]:
        """Prepare, schedule and record the observations for one turn's tool calls.

        Every call gets an observation, including the ones that never ran — the model
        has to be told why, otherwise it repeats the same call forever.
        """
        pending = await self._prepare(turn.message.tool_calls, state.messages)
        executions: list[ToolExecution] = []
        for item in pending:
            if item.outcome is None:
                continue
            execution = item.as_execution()
            executions.append(execution)
            yield AgentEvent(
                AgentEventType.TOOL_COMPLETE,
                iteration=state.iterations,
                executions=(execution,),
            )
            state.messages.append(Message.tool_result(execution.call.id, execution.observation))

        schedulable = [item for item in pending if item.outcome is None]
        async for batch in self._execute(schedulable):
            executions.extend(batch)
            for execution in batch:
                yield AgentEvent(
                    AgentEventType.TOOL_COMPLETE,
                    iteration=state.iterations,
                    executions=(execution,),
                )
                state.messages.append(Message.tool_result(execution.call.id, execution.observation))
        state.tool_calls += len(pending)

        mutated = tuple(sorted({execution.name for execution in executions if execution.mutated}))
        if mutated:
            await self._on_checkpoint(CheckpointRequest(state.iterations, mutated))
            yield AgentEvent(
                AgentEventType.CHECKPOINT,
                iteration=state.iterations,
                text=f"no checkpoint taken ({_SEAM_CHECKPOINT}): {', '.join(mutated)}",
            )

        context = totals.context_state(self.provider.capabilities.max_model_len)
        yield AgentEvent(
            AgentEventType.COMPACTION,
            iteration=state.iterations,
            context=context,
            text=(
                f"{context.status.value}: {context.prompt_tokens} prompt tokens against "
                f"a window of {context.window}; nothing compacted ({_SEAM_CONTEXT})"
            ),
        )

    async def _prepare(
        self, calls: Sequence[ToolCall], history: Sequence[Message]
    ) -> list[_Pending]:
        """Repair every streamed call, then put each one through the policy seam.

        Nothing here executes. Calls that cannot run (unknown tool, failed repair,
        denial) come back with an ``outcome`` already set, so the caller records an
        observation for them without scheduling anything. ``history`` is passed to the
        repairer so a re-ask is answerable rather than a fresh blind guess.
        """
        pending: list[_Pending] = []
        for call in calls:
            if self._cancel_event.is_set():
                pending.append(_Pending(call, None, outcome=ToolOutcome.CANCELLED))
                continue
            tool = self._by_name.get(call.name)
            if tool is None:
                pending.append(
                    _Pending(
                        call,
                        None,
                        outcome=ToolOutcome.UNAVAILABLE,
                        reason=", ".join(sorted(self._by_name)),
                    )
                )
                continue

            outcome = await self._repairer.repair(call, tool.spec, history=history)
            if not outcome.ok or outcome.arguments is None:
                pending.append(
                    _Pending(
                        call,
                        tool,
                        outcome=ToolOutcome.REPAIR_FAILED,
                        fixes=outcome.fixes,
                        repair_errors=outcome.errors,
                    )
                )
                continue
            if outcome.repaired:
                logger.info(
                    "repaired tool call %s (%s): %s", call.id, call.name, ", ".join(outcome.fixes)
                )

            verdict = self._policy(PolicyRequest(call, tool, call.name, outcome.arguments))
            if verdict.decision is Decision.DENY:
                pending.append(
                    _Pending(
                        call,
                        tool,
                        outcome=ToolOutcome.REJECTED,
                        fixes=outcome.fixes,
                        reason=verdict.reason or "denied by policy",
                    )
                )
                continue
            pending.append(_Pending(call, tool, arguments=outcome.arguments, fixes=outcome.fixes))
        return pending

    async def _execute(self, pending: Sequence[_Pending]) -> AsyncIterator[list[ToolExecution]]:
        """Run the schedulable calls: serial when unsafe, capped gather when safe.

        A batch is a maximal run of consecutive parallel-safe calls. Everything else
        runs alone, in transcript order, so ``write``/``edit``/``shell`` never
        interleave with each other or with a parallel batch.
        """
        index = 0
        while index < len(pending):
            if self._cancel_event.is_set():
                for item in pending[index:]:
                    item.outcome = ToolOutcome.CANCELLED
                    yield [item.as_execution()]
                return
            if not pending[index].parallel_safe:
                yield [await self._run_serial(pending[index])]
                index += 1
                continue
            end = index
            while end < len(pending) and pending[end].parallel_safe:
                end += 1
            yield await self._run_batch(pending[index:end])
            index = end

    async def _run_serial(self, item: _Pending) -> ToolExecution:
        task = self._start(self._run_one(item), item.call.name)
        try:
            return await task
        except asyncio.CancelledError:
            # A hard task-cancel does not cancel an awaited child, so cancel and reap
            # it here: the tool's own cancellation is what kills the process tree.
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if not self._cancel_event.is_set():
                raise
            item.outcome = ToolOutcome.CANCELLED
            return item.as_execution()

    async def _run_batch(self, batch: Sequence[_Pending]) -> list[ToolExecution]:
        tasks = [self._start(self._guarded(item), item.call.name) for item in batch]
        try:
            results: Sequence[ToolExecution | BaseException] = await asyncio.gather(
                *tasks, return_exceptions=True
            )
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if not self._cancel_event.is_set():
                raise
            results = [asyncio.CancelledError()] * len(tasks)
        return [self._record(item, result) for item, result in zip(batch, results, strict=True)]

    def _record(self, item: _Pending, result: ToolExecution | BaseException) -> ToolExecution:
        if isinstance(result, ToolExecution):
            return result
        if isinstance(result, asyncio.CancelledError):
            item.outcome = ToolOutcome.CANCELLED
            return item.as_execution()
        # call_tool converts the errors a tool is expected to produce; anything else is
        # a bug, and the model still has to be told the call is over.
        logger.error("tool %s crashed", item.call.name, exc_info=result)
        return ToolExecution(
            call=item.call,
            tool=item.tool,
            outcome=ToolOutcome.FAILED,
            result=ToolResult(
                ok=False,
                content=f"Tool {item.call.name!r} crashed.",
                error=f"{type(result).__name__}: {result}",
            ),
            fixes=item.fixes,
        )

    def _start(self, coro: Awaitable[ToolExecution], name: str) -> asyncio.Task[ToolExecution]:
        task = asyncio.ensure_future(coro)
        task.set_name(f"bevel-tool:{name}")
        self._tool_tasks.add(task)
        task.add_done_callback(self._tool_tasks.discard)
        return task

    async def _guarded(self, item: _Pending) -> ToolExecution:
        async with self._semaphore:
            return await self._run_one(item)

    async def _run_one(self, item: _Pending) -> ToolExecution:
        tool, arguments = item.tool, item.arguments
        if tool is None or arguments is None:  # _prepare guarantees both, defensive anyway
            raise RuntimeError("scheduled tool call is missing its tool or arguments")
        result = await call_tool(tool, arguments, self.ctx)
        return ToolExecution(
            call=item.call,
            tool=tool,
            outcome=ToolOutcome.OK if result.ok else ToolOutcome.FAILED,
            result=result,
            fixes=item.fixes,
        )

    def _specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self.tools]
