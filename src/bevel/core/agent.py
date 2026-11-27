"""The agent loop: orient → plan → act → observe, under budgets.

One iteration is:

1. assemble the context that goes on the wire,
2. stream a completion,
3. parse its tool calls **through the repair layer**,
4. evaluate each call against policy,
5. execute — parallel where the tool says it is safe, capped, serial otherwise,
6. append an observation for every call,
7. checkpoint the workspace before the first batch that writes,
8. compact if the context is over budget,
9. repeat, until the model stops asking for tools or a budget runs out.

Step 7 takes the session's checkpoint commit before the first batch that writes (see
:mod:`bevel.core.checkpoint`). Step 1 now orients: the project map from
:mod:`bevel.core.context.project_map` goes on the wire before the first completion, so
the model knows the layout without spending a tool call discovering it. Step 8 now
compacts: when the server's own ``prompt_tokens`` come back over
``Capabilities.max_model_len``, L1 truncates stale tool observations, L2 summarises
the middle of the transcript, and when neither can fit it L3 hands off — it writes a
brief, rebuilds the context from the same shape the run opened with, and carries on
(see :mod:`bevel.core.context.compact`). When a :class:`~bevel.core.session.store.SessionStore`
is attached, every message, tool call and usage row is written through as it completes,
and a run can continue a session that was closed earlier (see
:mod:`bevel.core.session.resume`). What the loop does *not* do is named in
:data:`UNBUILT_SEAMS`, reported on every :class:`AgentResult`, and logged once per run;
that list is empty, so every step above really happens. Policy (step 4) is enforced by
:class:`~bevel.core.policy.PolicyEngine`.

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
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from bevel.config.schema import PolicyConfig
from bevel.core.checkpoint import snapshot
from bevel.core.context.compact import compact_l1, compact_l2, compact_l3, handoff_cut
from bevel.core.context.project_map import build_project_map
from bevel.core.memory.recall import memory_block
from bevel.core.memory.store import MemoryStore
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
from bevel.core.session.resume import ResumedSession
from bevel.core.session.store import SessionStore, TodoStore
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
]

#: How many parallel-safe tool calls may be in flight at once.
DEFAULT_CONCURRENCY: Final[int] = 4

#: How many L3 handoffs one run may perform. Each one costs a generation and an
#: iteration, so an uncapped L3 lets a model that never converges hand off its
#: way through the whole token budget while going nowhere. Two is enough for a
#: long task to be carried twice, and small enough that a run that needs a third
#: is reported as out of room rather than quietly spending more.
MAX_HANDOFFS: Final[int] = 2

#: What this loop does *not* do, in the words the loop itself reports. Empty means
#: every step described above actually happens. An entry here is a disclaimer the
#: model is told about, so it must be removed the moment the thing is built *and*
#: tested -- and never added back to cover an unverified claim.
UNBUILT_SEAMS: Final[tuple[str, ...]] = ()

DEFAULT_SYSTEM_PROMPT: Final[str] = (
    "You are bevel, a coding agent working in a local workspace.\n"
    "\n"
    "Answer conversational messages directly: a greeting, or a question about bevel "
    "itself, gets a reply with zero tool calls. A project map of the workspace is "
    "already in your context -- the tree, the manifests, the README head, the entry "
    "points. That map is your orientation; do not re-explore the tree to get it. Reach "
    "for a tool only when the map cannot answer the question in front of you.\n"
    "\n"
    "When you are working, find things out instead of guessing: read before you edit, "
    "grep to locate, glob to enumerate. Make the smallest change that fixes the "
    "problem, and never claim a change works unless you have run something that proves "
    'it. "Not built yet" is a correct answer; invented output is not.\n'
    "\n"
    "Limits of this build: the workspace is snapshotted once before the first write, so "
    "a run can be undone, but there is no per-step rollback -- an undo rewinds the whole "
    "run. Policy is enforced: reads and workspace writes proceed, and anything "
    "the policy engine decides needs your approval is refused unless an approver is "
    "installed. Say what you need instead of assuming you have it."
)

#: Appended to the system prompt when a memory store is attached.
#:
#: The model discovers ``remember`` from the tool schema on its own -- it called
#: it unprompted against a real endpoint -- so this is not about teaching it the
#: tool exists. It is about telling it to *use* it, which the bare schema does
#: not, and about staying silent when there is no store: the registry offers
#: ``remember`` and ``recall`` either way, so guidance that ignored the flag
#: would have the model spend calls on tools that can only fail.
MEMORY_SYSTEM: Final[str] = (
    "Long-term memory is on for this run. `recall` retrieves what bevel learned "
    "in earlier sessions about this project and about you; `remember` saves "
    "something worth keeping. Recall what is relevant before you start working, "
    "and save a durable fact, preference, or decision as soon as you learn one "
    "-- memory only helps a later session if you use it now. Correcting an "
    "earlier memory means saving the new one and superseding the old. "
    "`remember` writes outside the workspace, so it may pause for your approval."
)

#: Appended to the system prompt when a todo store is attached.
#:
#: Same reasoning as :data:`MEMORY_SYSTEM`: the schema tells the model the tool
#: exists, not that it should use it, and the registry offers ``todo`` whether or
#: not a store is attached -- guidance that ignored the flag would have the
#: model spending calls on a tool that can only fail. The wording is weighted
#: toward *replacing* the list, because that is the one thing a model gets wrong
#: in the direction that is invisible: it appends, and nobody notices the list
#: has grown tasks it abandoned three turns ago.
TODO_SYSTEM: Final[str] = (
    "Task tracking is on for this run. Use `todo` to keep a plan for any task "
    "with more than a couple of steps. Send the complete list every time -- it "
    "replaces what is stored, it does not append -- so marking a task done means "
    "sending the same list again with that task's status set to 'done'. Keep "
    "exactly one task 'in_progress' while you work, and mark it 'done' only once "
    "it has actually been verified. Send an empty list when the work is finished."
)

#: What L2's summariser is told to do. Kept out of the loop so the one place that
#: asks for a summary is the one place that says what a summary is for.
SUMMARY_SYSTEM: Final[str] = (
    "You are compacting an agent transcript. Summarise the conversation you are given "
    "so that the agent can continue from your summary alone. State as facts: the task "
    "asked for; what each tool call was for and what it established; every file, path, "
    "symbol, command and number the agent has actually seen; what is settled; and what "
    "is still unresolved. Be dense and literal. Do not add advice, do not speculate "
    "about anything not in the transcript, and do not answer the task yourself."
)

#: What the L3 handoff summariser is told to do. Deliberately narrower than
#: SUMMARY_SYSTEM: the goal, the files touched and the todos are assembled
#: deterministically by the loop, so the model is asked only for the two sections
#: it can honestly supply, and is explicitly forbidden from naming files.
BRIEF_SYSTEM: Final[str] = (
    "You are writing the handoff brief for a coding agent whose conversation has run "
    "out of context window. Your notes are handed to a fresh instance of the same "
    "agent that has seen none of this transcript, so anything you leave out is lost.\n"
    "Write two sections, in this order, with these exact headings:\n"
    "## decisions\n"
    "The choices made during the work that a newcomer would otherwise repeat or "
    "contradict: approaches taken and rejected, and why.\n"
    "## open questions\n"
    "What is still unresolved, including anything the work was in the middle of when "
    "it ran out of room.\n"
    "Only use what is in the transcript you are given. Never name a file, a command, "
    "or a result that does not appear in it - the files the work touched are listed "
    "elsewhere and are not yours to add to. Be terse; no preamble. Stay under about "
    "eighty words: this text is prepended to a context that is already too long."
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
    TOOL_COMPLETE, ``context`` for COMPACTION, ``error`` for a COMPLETED event whose
    stop reason was ``provider_error``.

    ``error`` exists because a stop reason alone is not actionable: "provider error"
    tells the user nothing, whereas the exception text distinguishes a certificate
    they must trust from a model name the server does not have.
    """

    type: AgentEventType
    iteration: int
    text: str = ""
    turn: AssistantTurn | None = None
    usage: Usage | None = None
    executions: tuple[ToolExecution, ...] = ()
    context: ContextState | None = None
    error: str | None = None


class StopReason(StrEnum):
    COMPLETED = "completed"
    TURN_BUDGET = "turn_budget"
    TOKEN_BUDGET = "token_budget"  # noqa: S105
    CANCELLED = "cancelled"
    PROVIDER_ERROR = "provider_error"
    CONTEXT_OVERFLOW = "context_overflow"


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
    session_id: str | None = None

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
    handoffs: int = 0
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

    def add(self, usage: Usage, counts_as_context: bool = True) -> None:
        """Fold one request's server-reported usage into the run.

        ``counts_as_context`` is False for a request that was not the next thing
        the model would have seen — L2's summariser call. Its prompt tokens are
        the size of the middle being summarised, not of the next request, so
        counting them as the context size would report a window reading the loop
        never sent. They still count against what the run cost.
        """
        if counts_as_context and not (
            usage.total_tokens or usage.prompt_tokens or usage.completion_tokens
        ):
            self.reported = False
        self.cumulative = Usage(
            prompt_tokens=self.cumulative.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self.cumulative.completion_tokens + usage.completion_tokens,
            reasoning_tokens=self.cumulative.reasoning_tokens + usage.reasoning_tokens,
            total_tokens=self.cumulative.total_tokens + usage.total_tokens,
        )
        if usage.prompt_tokens and counts_as_context:
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
        repairer: ToolCallRepairer | None = None,
        summariser: ModelProvider | None = None,
        store: SessionStore | None = None,
        resume: ResumedSession | None = None,
        memory: MemoryStore | None = None,
        todos: TodoStore | None = None,
    ) -> None:
        if budgets.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if budgets.max_tool_calls < 0:
            raise ValueError("max_tool_calls cannot be negative")
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.provider = provider
        self._memory = memory
        if memory is not None and memory.config.vectors and memory.embedder is None:
            # The store outlives the run, but only the run knows what endpoint is
            # talking to it. Recall refuses to compare vectors across models, so
            # tagging them with *this* provider is what makes them comparable.
            memory.embedder = provider
        self._todos = todos
        self.ctx = ToolContext(Path(workspace), memory, todos)
        self.system = system
        self.options = options
        self.budgets = budgets
        self.tools: tuple[Tool, ...] = tuple(tools) if tools is not None else tuple(builtin_tools())
        self._by_name: dict[str, Tool] = {tool.spec.name: tool for tool in self.tools}
        self._repairer = repairer or ToolCallRepairer(provider)
        self._summariser = summariser
        self._policy: PolicyEvaluator = policy or PolicyEngine(
            self.ctx,
            config=policy_config,
            approver=approver,
        )
        self._checkpoint_session = f"{self.ctx.workspace.name}-{datetime.now(UTC):%Y%m%dT%H%M%S}"
        self._checkpointed = False
        self._semaphore = asyncio.Semaphore(concurrency)
        self._tool_tasks: set[asyncio.Task[ToolExecution]] = set()
        self._cancel_event = asyncio.Event()
        self._cancel_reason = ""
        self._running = False
        self._store = store
        self._session_id = resume.session.id if resume is not None else None
        self._resumed = resume
        self._turn_row: int | None = None
        # The task, verbatim, for the L3 handoff brief's goal. Set by _seed().
        self._task = ""
        # Relative paths the run actually changed, in the order it changed them.
        # Assembled from ToolExecution.mutated, never from the model's account of
        # what it did — a model is not allowed to name a file it did not touch.
        self._mutated_paths: list[str] = []

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
            session_id=self._session_id,
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

    @property
    def session_id(self) -> str | None:
        """The session this run is writing to, or ``None`` with no store attached."""
        return self._session_id

    # ------------------------------------------------------------------ private

    async def _iterate(self, task: str, state: _RunState) -> AsyncIterator[AgentEvent]:
        if self._running:
            raise RuntimeError("this Agent is already running")
        self._running = True
        self._cancel_reason = ""
        self._cancel_event.clear()
        await self._seed(task, state)
        totals = _Totals()
        logger.info("unbuilt seams: %s", "; ".join(UNBUILT_SEAMS) or "none")
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
                self._persist(turn.message, state.iterations)
                self._persist_usage(turn.usage, state.iterations)
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
                # Either the user stopped the run or compaction could not fit the
                # transcript; both mean there is no point asking the model again.
                if self._cancel_event.is_set() or state.stop is StopReason.CONTEXT_OVERFLOW:
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
            AgentEventType.COMPLETED,
            iteration=state.iterations,
            text=state.stop.value,
            error=state.error,
        )

    async def _opening(self, recall_query: str) -> list[Message]:
        """The opening of a context: system prompt with a fresh map, then recall.

        Shared by :meth:`_seed` and the L3 handoff so that a run resumed after a
        handoff starts from the *same* shape as one that never needed one. The
        project map is rebuilt rather than reused, so neither a resumed session
        nor a handed-off one inherits a stale picture of the tree.

        Recalled memory goes at **index 1**, in its own message. It cannot be
        merged into the system prompt and cannot be ``messages[0]``, because
        :mod:`bevel.core.context.compact` never evicts ``messages[0]`` and a
        stale memory block has to be evictable.

        ``recall_query`` is what to search memory with. At the start of a run it
        is the task; mid-run the task is stale, so the handoff passes what the
        agent was last working on instead.
        """
        # The walk is bounded but real disk I/O, and it happens while the TUI is
        # live, so it goes off the event loop.
        project_map = await asyncio.to_thread(build_project_map, self.ctx.workspace)
        system = self.system
        if self._memory is not None:
            system = f"{system}\n\n{MEMORY_SYSTEM}"
        if self._todos is not None:
            system = f"{system}\n\n{TODO_SYSTEM}"
        messages = [Message.system(f"{system}\n\n{project_map}")]
        block = await self._recall_block(recall_query)
        if block is not None:
            messages.append(block)
        return messages

    async def _seed(self, task: str, state: _RunState) -> None:
        """Build the message list the first request is made of.

        The opening is :meth:`_opening`; the rest is the resumed conversation (if
        any) and the task itself.
        """
        self._task = task
        state.messages.extend(await self._opening(task))
        if self._resumed is not None:
            state.messages.extend(self._resumed.messages)
        task_message = Message.user(task)
        state.messages.append(task_message)
        self._open_session()
        self._persist(task_message)

    async def _recall_block(self, task: str) -> Message | None:
        """The recalled-memory message, or ``None`` when there is nothing to inject.

        Runs off the loop for the same reason the project map does. A memory
        failure is never allowed to fail the run: the user asked for their task,
        not for their memory database, so it is logged and the run continues
        without the block.
        """
        if self._memory is None:
            return None
        try:
            block = await memory_block(
                self._memory, self.provider, query=task, workspace=self.ctx.workspace
            )
        except Exception:
            # Memory is an enhancement, not a dependency: a broken store must
            # not cost the user the task they actually asked for.
            logger.exception("recall for the opening context failed; continuing without it")
            return None
        return block.message

    # ------------------------------------------------------------------ handoff

    def _record_mutation(self, execution: ToolExecution) -> None:
        """Remember the path a successful write touched, for the handoff brief.

        The list of files a run touched is assembled from what the loop actually
        executed, never from what the model says it edited: a brief naming a file
        nobody wrote is worse than one naming none. A call whose arguments cannot
        be read contributes nothing rather than a guess.
        """
        if not execution.mutated:
            return
        try:
            path = json.loads(execution.call.arguments).get("path")
        except (ValueError, AttributeError):
            return
        if isinstance(path, str) and path and path not in self._mutated_paths:
            self._mutated_paths.append(path)

    def _file_lines(self) -> str:
        if not self._mutated_paths:
            return "no files were written"
        return "\n".join(f"- {path}" for path in self._mutated_paths)

    def _todo_lines(self) -> str:
        """The todo list as the brief states it, or an honest account of its absence.

        Read from the store rather than from the transcript, so the brief reports
        the list as it stands now -- including the tasks the model marked done in
        turns the handoff is about to drop.
        """
        if self._todos is None:
            return "no todo list was kept"
        try:
            todos = self._todos.list_todos()
        except Exception:
            # Same rule as recall: an enhancement must not cost the user the task.
            logger.exception("the todo list could not be read for the handoff brief")
            return "the todo list could not be read"
        if not todos:
            return "no todo list was kept"
        return "\n".join(f"- [{todo.status}] {todo.content}" for todo in todos)

    def _recall_query(self, state: _RunState) -> str:
        """What to search memory with when handing off.

        The opening searched the task. Mid-run the task alone is stale -- the
        agent is dozens of steps past it -- so the current work is searched
        alongside it, and the task stays in the query because the goal did not
        change just because the context did.
        """
        for message in reversed(state.messages):
            if message.role == "assistant" and message.content.strip():
                return f"{self._task} {message.content}"
        return self._task

    def _handoff_scaffold(self, dropped: Sequence[Message]) -> str:
        """What the model is shown when it is asked for its account of the work.

        The deterministic sections go first so the model is writing into a
        structure rather than inventing one, and so the file list it is handed is
        the list it is not allowed to extend.
        """
        transcript = "\n\n".join(
            f"{message.role}: {message.content}" for message in dropped if message.content
        )
        return "\n\n".join(
            [
                f"## goal\n\n{self._task}",
                f"## files touched\n\n{self._file_lines()}",
                f"## todos\n\n{self._todo_lines()}",
                f"## transcript being dropped\n\n{transcript}",
            ]
        )

    def _brief_text(self, account: str) -> str:
        """The finished brief: three sections the loop knows are true, one the model wrote."""
        return "\n\n".join(
            [
                f"## goal\n\n{self._task}",
                "## decisions and open questions (the previous instance's own account)\n\n"
                f"{account}",
                f"## files touched\n\n{self._file_lines()}",
                f"## todos\n\n{self._todo_lines()}",
            ]
        )

    async def _write_brief(self, scaffold: str, totals: _Totals) -> str:
        """Ask the model for the two sections only it can supply.

        Billed to the run, but not as the context reading: this prompt is the
        material being dropped, not the next thing the model will see -- exactly
        the case ``_Totals.add``'s ``counts_as_context`` exists for.
        """
        writer = self._summariser or self.provider
        turn = await writer.complete(
            [Message.system(BRIEF_SYSTEM), Message.user(scaffold)], None, self.options
        )
        totals.add(turn.usage or Usage(), counts_as_context=False)
        return turn.message.content.strip()

    async def _handoff(
        self, state: _RunState, totals: _Totals, context: ContextState, against: str
    ) -> tuple[AgentEvent, StopReason | None]:
        """L3: carry the task into a fresh context instead of ending the run.

        The transcript is rebuilt from the opening shape ``_seed()`` produces --
        system prompt with a *fresh* project map, the recall block at index 1 --
        followed by the brief and the last turns verbatim. Everything the brief
        accounts for is dropped, including any L2 summary, which it supersedes.

        Every way this can fail is a reported ``CONTEXT_OVERFLOW``, not a silent
        one. A brief that cannot be written, that comes back empty, or that is
        itself too big to fit leaves the run exactly where L1 and L2 left it,
        because the alternative is sending a request the server is known to
        reject. The per-run cap exists because a model that never converges can
        otherwise hand off its way through the whole budget while going nowhere.
        """

        def declined(reason: str, error: str) -> tuple[AgentEvent, StopReason]:
            state.error = error
            logger.warning("handoff declined: %s", error)
            return (
                AgentEvent(
                    AgentEventType.COMPACTION,
                    iteration=state.iterations,
                    context=context,
                    text=f"l1: nothing left to cut; l2: no progress; {reason}; {against}",
                ),
                StopReason.CONTEXT_OVERFLOW,
            )

        if state.handoffs >= MAX_HANDOFFS:
            return declined(
                f"l3: per-run handoff cap of {MAX_HANDOFFS} reached",
                f"transcript will not fit: {against}, and the per-run handoff cap of "
                f"{MAX_HANDOFFS} is already spent.",
            )
        dropped = handoff_cut(state.messages)
        if not dropped:
            return declined(
                "l3: no turns left to fold into a brief",
                f"transcript will not fit: {against}, and there were no turns left to hand off.",
            )
        try:
            account = await self._write_brief(self._handoff_scaffold(dropped), totals)
        except Exception as exc:
            return declined(
                "l3: the brief request failed",
                f"transcript will not fit: {against}, and the handoff brief could not be "
                f"written: {type(exc).__name__}: {exc}",
            )
        if not account:
            return declined(
                "l3: the brief came back empty",
                f"transcript will not fit: {against}, and the handoff brief was empty.",
            )
        head = await self._opening(self._recall_query(state))
        l3 = compact_l3(state.messages, head=head, brief=self._brief_text(account))
        if not l3.changed:
            return declined(
                f"l3: refused, {l3.reason}",
                f"the handoff brief would not fit: {l3.reason}; {against}.",
            )
        text = (
            f"l1: nothing left to cut; l2: no progress; l3: handoff replaced "
            f"{l3.turns_replaced} turn(s), removed {l3.chars_removed} chars; {against}"
        )
        event = AgentEvent(
            AgentEventType.COMPACTION, iteration=state.iterations, context=context, text=text
        )
        state.messages[:] = l3.messages
        state.handoffs += 1
        # The brief cost a generation, so the handoff costs an iteration: work the
        # budget cannot see is work the budget cannot stop.
        state.iterations += 1
        logger.info("handed off: %s", text)
        return (event, None)

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

    def _mutates_filesystem(self, item: _Pending) -> bool:
        """Whether a scheduled call is expected to change the working tree.

        Two producers. A ``WRITE_WORKSPACE`` tool (write, edit) always counts.
        The ``git`` tool counts only for a mutating subcommand — a ``commit``
        rewrites the tree a checkpoint ref describes, so it must trigger the same
        snapshot the first write does, but a ``status`` must not. The git tool
        answers that itself via ``mutates()``; a tool without that method is
        judged on its tier alone.
        """
        tool = item.tool
        if tool is None:
            return False
        if tool.risk_tier is RiskTier.WRITE_WORKSPACE:
            return True
        mutates = getattr(tool, "mutates", None)
        if callable(mutates) and item.arguments is not None:
            return bool(mutates(item.arguments))
        return False

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
            self._persist_execution(execution)
            self._record_mutation(execution)
            yield AgentEvent(
                AgentEventType.TOOL_COMPLETE,
                iteration=state.iterations,
                executions=(execution,),
            )
            observation = Message.tool_result(execution.call.id, execution.observation)
            state.messages.append(observation)
            self._persist(observation, state.iterations)

        schedulable = [item for item in pending if item.outcome is None]
        if not self._checkpointed and any(self._mutates_filesystem(item) for item in schedulable):
            self._checkpointed = True
            result = await snapshot(self.ctx.workspace, self._checkpoint_session)
            if not result.ok:
                logger.warning("checkpoint failed: %s", result.error)
            yield AgentEvent(
                AgentEventType.CHECKPOINT,
                iteration=state.iterations,
                text=result.summary if result.ok else str(result.error),
            )

        async for batch in self._execute(schedulable):
            executions.extend(batch)
            for execution in batch:
                self._persist_execution(execution)
                self._record_mutation(execution)
                yield AgentEvent(
                    AgentEventType.TOOL_COMPLETE,
                    iteration=state.iterations,
                    executions=(execution,),
                )
                observation = Message.tool_result(execution.call.id, execution.observation)
                state.messages.append(observation)
                self._persist(observation, state.iterations)
        state.tool_calls += len(pending)

        event, stop = await self._compact(state, totals)
        if stop is not None:
            state.stop = stop
        yield event

    # ------------------------------------------------------------- persistence

    def _open_session(self) -> None:
        """Create the session row this run writes through, unless resuming one."""
        if self._store is not None and self._session_id is None:
            self._session_id = self._store.create_session(
                workspace=self.ctx.workspace,
                provider=self.provider.name,
                model=self.provider.model,
            ).id
        if self._todos is not None:
            # The tool object is built before a session id exists, so the store
            # learns it here -- the same way MemoryStore learns its embedder.
            # Assigned on the resume path and the store-less path too, so the
            # todo tool's idea of the session is always the loop's.
            self._todos.session_id = self._session_id

    def _persist(self, message: Message, iteration: int = 1) -> None:
        """Append one message. Commits immediately: a completed message is recoverable.

        An assistant message that asked for tools owns the execution rows that
        follow it, so its row id is remembered for them to attach to.
        """
        if self._store is None or self._session_id is None:
            return
        row = self._store.append_turn(self._session_id, message, iteration=iteration)
        if message.tool_calls:
            self._turn_row = row

    def _persist_execution(self, execution: ToolExecution) -> None:
        """Record the call and its outcome beside the assistant turn that asked for it."""
        if self._store is None or self._session_id is None or self._turn_row is None:
            return
        self._store.append_tool_call(self._session_id, self._turn_row, execution)

    def _persist_usage(self, usage: Usage | None, iteration: int) -> None:
        if self._store is None or self._session_id is None or usage is None:
            return
        if not (usage.prompt_tokens or usage.completion_tokens or usage.total_tokens):
            return  # the server reported nothing; there is no row to write
        self._store.record_usage(
            self._session_id,
            iteration=iteration,
            prompt=usage.prompt_tokens,
            completion=usage.completion_tokens,
            reasoning=usage.reasoning_tokens,
            total=usage.total_tokens,
        )

    async def _compact(
        self, state: _RunState, totals: _Totals
    ) -> tuple[AgentEvent, StopReason | None]:
        """Step 8: fit the transcript back into the window, cheapest level first.

        The trigger is the server's own ``prompt_tokens`` from the last request
        against ``Capabilities.max_model_len``. When either is unknown the loop does
        nothing and says so — it refuses to invent a window size, so it must not
        start guessing one here either.

        L1 is free and deterministic, so it goes first. L2 costs a generation and is
        only tried once L1 has nothing left to give. A level that changed nothing is
        exhausted, and when both are exhausted the run stops with CONTEXT_OVERFLOW
        rather than sending a request the server is known to reject: the recovery
        from here is the L3 handoff, and L3 is not built.

        One escalation per over-budget measurement. A level that did real work lets
        the next request go out and the server re-measure it, because there is no
        way to count the new transcript locally.
        """
        window = self.provider.capabilities.max_model_len
        context = totals.context_state(window)
        size = (
            "no server prompt_tokens"
            if context.prompt_tokens is None
            else f"{context.prompt_tokens} prompt tokens"
        )
        against = f"{size} against a window of {window}"
        if context.status is CompactionStatus.UNKNOWN:
            why = (
                "the server reported no context window"
                if window is None
                else "the server reported no prompt_tokens"
            )
            return (
                AgentEvent(
                    AgentEventType.COMPACTION,
                    iteration=state.iterations,
                    context=context,
                    text=f"unknown: {why}; nothing compacted",
                ),
                None,
            )
        if context.status is CompactionStatus.WITHIN_BUDGET:
            return (
                AgentEvent(
                    AgentEventType.COMPACTION,
                    iteration=state.iterations,
                    context=context,
                    text=f"within_budget: {against}; nothing to compact",
                ),
                None,
            )

        l1 = compact_l1(state.messages, session_id=self._session_id)
        if l1.changed:
            state.messages[:] = l1.messages
            text = (
                f"l1: cut {l1.observations_cut} observation(s) to head+tail, removed "
                f"{l1.chars_removed} chars; {against}"
            )
        else:
            summariser = self._summariser or self.provider
            model = summariser.model

            async def summarise(middle: Sequence[Message]) -> str:
                turn = await summariser.complete(
                    [Message.system(SUMMARY_SYSTEM), *middle], None, self.options
                )
                # The summariser's prompt is the size of the middle, not of the next
                # request, so it counts against the run but not as the context size.
                totals.add(turn.usage or Usage(), counts_as_context=False)
                return turn.message.content

            l2 = await compact_l2(state.messages, summarise)
            if l2.changed:
                state.messages[:] = l2.messages
                head = "l1: nothing left to cut; " if l1.chars_removed == 0 else ""
                text = (
                    f"{head}l2: summarised {l2.turns_replaced} turn(s) into {len(l2.summary)} "
                    f"chars, removed {l2.chars_removed} chars (summarised by {model}); {against}"
                )
                logger.info("compacted: %s", text)
                return (
                    AgentEvent(
                        AgentEventType.COMPACTION,
                        iteration=state.iterations,
                        context=context,
                        text=text,
                    ),
                    None,
                )
            # L3: neither level can fit the transcript, so hand off instead of stopping.
            return await self._handoff(state, totals, context, against)
        logger.info("compacted: %s", text)
        return (
            AgentEvent(
                AgentEventType.COMPACTION,
                iteration=state.iterations,
                context=context,
                text=text,
            ),
            None,
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

            verdict = await self._policy(PolicyRequest(call, tool, call.name, outcome.arguments))
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
