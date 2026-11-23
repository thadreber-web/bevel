"""The bevel TUI: a transcript, an input line, and the loop underneath.

The app is deliberately thin. It owns no agent logic -- it subscribes to
:meth:`~bevel.core.agent.Agent.stream`, renders what arrives, and calls
:meth:`~bevel.core.agent.Agent.cancel` when the user asks it to stop. Everything
about *what the model should do* stays in the loop.

Three things are non-negotiable here, because each is a way a TUI can lie:

* **Ctrl+C really stops the run.** Abandoning the event generator cancels
  nothing, so the binding calls ``cancel()`` -- and dismisses an open approval
  prompt first, which is the one place the loop waits on the UI rather than the
  reverse.
* **A run that hit a limit says so.** Budget stops, provider errors and
  cancellations each render as their own coloured outcome, never as a blank
  pause. The loop reports why it stopped in the ``COMPLETED`` event's text; that
  word is the UI's only clue, so it is translated and coloured rather than
  printed raw.
* **What is not built is announced.** The loop reports the seams it does not
  implement; the app prints them on mount instead of letting the user assume
  context management exists.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Final

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import SimpleCommand
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Footer, Header, Input, Static

from bevel.config.loader import capability_cache_path, load, memory_db_path, session_db_path
from bevel.config.schema import ConfigError, PolicyConfig, PolicyMode
from bevel.core.agent import (
    UNBUILT_SEAMS,
    Agent,
    AgentEvent,
    AgentEventType,
    ToolExecution,
    ToolOutcome,
)
from bevel.core.checkpoint import list_checkpoints, restore
from bevel.core.memory.store import MemoryStore
from bevel.core.messages import ToolCall
from bevel.core.policy.approvals import Approval, ApprovalRequest, Approver, Decision
from bevel.core.session.resume import ResumedSession, rebuild_transcript
from bevel.core.session.store import SessionStore, StoredToolCall, TodoStore
from bevel.core.tools.base import ToolResult
from bevel.providers.base import ModelProvider
from bevel.providers.registry import Registry
from bevel.tui.widgets.tool_cards import ToolCard
from bevel.tui.widgets.transcript import ContentBlock, ReasoningBlock

__all__ = ["BevelApp", "RunMode", "run_tui"]


class RunMode(StrEnum):
    """How much the app lets a run do without asking.

    +-------------+-----------------------------------------------------------+
    | ``plan``    | reads run; anything else is refused and says why            |
    +-------------+-----------------------------------------------------------+
    | ``default`` | workspace writes and commands both ask                      |
    +-------------+-----------------------------------------------------------+
    | ``auto``    | workspace writes are free; commands and escapes still ask   |
    +-------------+-----------------------------------------------------------+

    No mode approves a command automatically. ``auto`` relaxes the policy
    *mode*, never the approver, so a destructive command still costs a keypress.
    """

    PLAN = "plan"
    DEFAULT = "default"
    AUTO = "auto"


_MODE_HINTS: Final[dict[RunMode, str]] = {
    RunMode.PLAN: "read-only; writes and commands are refused",
    RunMode.DEFAULT: "asks for writes and commands",
    RunMode.AUTO: "workspace writes are free; commands still ask",
}

#: The loop's own stop-reason values, translated. Printing the raw value would be
#: the UI reciting its internals at the user.
_STOP_LINES: Final[dict[str, tuple[str, str]]] = {
    "completed": ("done", "#3fb950"),
    "cancelled": ("cancelled", "#8b949e"),
    "turn_budget": ("stopped: iteration or tool-call budget", "#e3b341"),
    "token_budget": ("stopped: token budget", "#e3b341"),
    "provider_error": ("provider error", "#f85149"),
    "context_overflow": ("stopped: context would not fit", "#f85149"),
}


class BevelApp(App[None]):
    """The interactive front end for one workspace."""

    ENABLE_COMMAND_PALETTE = False

    DEFAULT_CSS = """
    VerticalScroll#transcript {
        height: 1fr;
        padding: 0 1;
    }
    Input#task {
        dock: bottom;
    }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("ctrl+c", "cancel", "Stop", priority=True),
        Binding("f2", "cycle_mode", "Mode", priority=True),
        Binding("f1", "commands", "Commands", priority=True),
    ]

    def __init__(
        self,
        *,
        provider: ModelProvider,
        workspace: Path | str | None = None,
        policy: PolicyConfig | None = None,
        mode: RunMode = RunMode.DEFAULT,
        store: SessionStore | None = None,
        resume: ResumedSession | None = None,
        memory: MemoryStore | None = None,
        todos: TodoStore | None = None,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.workspace = Path(workspace) if workspace is not None else Path.cwd()
        self.policy = policy if policy is not None else PolicyConfig()
        self.mode = mode
        self.store = store
        self.resume = resume
        self.memory = memory
        self.todos = todos
        self._agent: Agent | None = None
        self._content: ContentBlock | None = None
        self._reasoning: ReasoningBlock | None = None
        self._content_chars = 0
        self._tokens = 0

    # ------------------------------------------------------------------- layout

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield VerticalScroll(id="transcript")
        yield Input(placeholder="ask bevel to do something; enter to run", id="task")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "bevel"
        self._refresh_title()
        if self.resume is not None:
            self._show_resumed()
        for seam in UNBUILT_SEAMS:
            self._write(Text.assemble(("not built  ", "bold #bc8cff"), seam))
        self.query_one("#task", Input).focus()

    def _show_resumed(self) -> None:
        """Render the transcript that was read back from the store.

        The lines below come out of ``sessions.db``, not out of anything this
        process remembers -- which is the whole point of resume.
        """
        resumed = self.resume
        assert resumed is not None
        head = Text.assemble(
            ("resumed  ", "bold #3fb950"),
            f"{resumed.session.id} - {resumed.session.workspace}"
            f" ({resumed.turns} stored message(s), {resumed.session.total_tokens:,} tokens)",
        )
        self._write(head)
        if resumed.dropped:
            self._write(
                Text.assemble(
                    ("dropped  ", "bold #e3b341"),
                    f"{resumed.dropped} trailing message(s) left the last run incomplete"
                    " (a tool call with no result); they are still in the store",
                )
            )
        for turn in resumed.stored:
            if turn.role == "user":
                self._write(Text.assemble(("you  ", "bold #58a6ff"), turn.content))
                continue
            if turn.role == "assistant":
                if turn.reasoning:
                    reasoning = ReasoningBlock()
                    self._mount(reasoning)
                    reasoning.append(turn.reasoning)
                if turn.content:
                    content = ContentBlock()
                    self._mount(content)
                    content.append(turn.content)
                for call in turn.tool_calls:
                    self._mount(ToolCard(_stored_execution(call)))
        self._write(Text("the model has this history; anything you ask continues from it", "dim"))

    def _refresh_title(self) -> None:
        self.sub_title = f"{self.provider.name}/{self.provider.model} - {self.mode} mode"

    # ------------------------------------------------------------------ actions

    def action_cancel(self) -> None:
        """Stop the run, or quit when there is nothing to stop.

        Two meanings for one key, because Ctrl+C means "stop doing the thing you
        are doing" -- and when nothing is running, that thing is bevel itself. An
        open approval prompt is dismissed with a refusal first: the loop is parked
        inside :meth:`Approver.approve` waiting for that screen, so cancelling
        without closing it would deadlock the very run being stopped.
        """
        screen = self.screen
        if isinstance(screen, ApprovalScreen):
            screen.dismiss(Approval(Decision.DENY, note="the run was cancelled while asking"))
        if self._agent is not None:
            self._agent.cancel()
            self._write(Text("stopping...", style="dim"))
        else:
            self.exit()

    def action_cycle_mode(self) -> None:
        modes = list(RunMode)
        self._set_mode(modes[(modes.index(self.mode) + 1) % len(modes)])

    async def action_commands(self) -> None:
        await self.search_commands(
            [
                SimpleCommand(
                    f"mode {mode.value}",
                    lambda mode=mode: self._set_mode(mode),
                    _MODE_HINTS[mode],
                )
                for mode in RunMode
            ]
            + [
                SimpleCommand("undo", self.undo_last_run, "rewind the workspace to a checkpoint"),
                SimpleCommand("quit", self.exit, "leave bevel"),
            ],
            placeholder="bevel commands",
        )

    def _set_mode(self, mode: RunMode) -> None:
        if self._agent is not None:
            self._write(Text("a run is going -- Ctrl+C to stop it before changing mode", "yellow"))
            return
        self.mode = mode
        self._refresh_title()
        self._write(Text.assemble(("mode  ", "bold"), f"{mode} - {_MODE_HINTS[mode]}"))

    # --------------------------------------------------------------- submission

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "task":
            return
        task = event.value.strip()
        event.input.value = ""
        if not task:
            return
        if self._agent is not None:
            self._write(Text("a run is already going -- Ctrl+C to stop it first", "yellow"))
            return
        self._write(Text.assemble(("you  ", "bold #58a6ff"), task))
        self._tokens = 0
        self.run_worker(self._run(task), exclusive=True)

    async def _run(self, task: str) -> None:
        agent = Agent(
            self.provider,
            self.workspace,
            policy_config=replace(self.policy, mode=self._policy_mode()),
            approver=self._approver(),
            store=self.store,
            resume=self.resume,
            memory=self.memory,
            todos=self.todos,
        )
        self._agent = agent
        try:
            async for event in agent.stream(task):
                self._handle(event)
        finally:
            self._agent = None
            self._end_blocks()
        self._continue_from_store(agent)

    def _continue_from_store(self, agent: Agent) -> None:
        """Point the next task at what this run just wrote, not at what it remembers.

        Rebuilding from the store after every run keeps one source of truth for
        the conversation, so a task submitted after a crash continues from the
        turns that actually made it to disk.
        """
        if self.store is None or agent.session_id is None:
            return
        self.resume = rebuild_transcript(self.store, agent.session_id)

    def _policy_mode(self) -> PolicyMode:
        return PolicyMode.AUTO if self.mode is RunMode.AUTO else PolicyMode.ASK

    def _approver(self) -> Approver:
        return PlanApprover() if self.mode is RunMode.PLAN else TUIApprover(self)

    # ------------------------------------------------------------- event render

    def _handle(self, event: AgentEvent) -> None:
        if event.type is AgentEventType.REASONING:
            self._append_reasoning(event.text)
        elif event.type is AgentEventType.CONTENT:
            self._append_content(event.text)
        elif event.type is AgentEventType.TURN_COMPLETE:
            self._end_blocks()
            if event.usage is not None:
                self._tokens += event.usage.total_tokens
            if self._content_chars == 0:
                self._write(Text("(the model returned no content)", style="dim italic"))
            self._content_chars = 0
        elif event.type is AgentEventType.TOOL_COMPLETE:
            self._end_blocks()
            for execution in event.executions:
                self._mount(ToolCard(execution))
        elif event.type in (AgentEventType.CHECKPOINT, AgentEventType.COMPACTION):
            self._end_blocks()
            label = "checkpoint" if event.type is AgentEventType.CHECKPOINT else "context"
            self._write(Text.assemble((f"{label}  ", "bold #58a6ff"), event.text or "(no detail)"))
        elif event.type is AgentEventType.BUDGET:
            self._end_blocks()
            self._write(Text.assemble(("budget  ", "bold #e3b341"), event.text))
        elif event.type is AgentEventType.CANCELLED:
            self._end_blocks()
            self._write(Text.assemble(("cancelled  ", "bold #8b949e"), event.text))
        elif event.type is AgentEventType.COMPLETED:
            self._write(_completion_line(event, self._tokens))

    def _write(self, content: Text) -> None:
        self._mount(Static(content))

    def _mount(self, widget: Widget) -> None:
        self.query_one("#transcript", VerticalScroll).mount(widget)

    def _append_content(self, text: str) -> None:
        self._reasoning = None
        if self._content is None:
            self._content = ContentBlock()
            self._mount(self._content)
        self._content.append(text)
        self._content_chars += len(text)

    def _append_reasoning(self, text: str) -> None:
        self._content = None
        if self._reasoning is None:
            self._reasoning = ReasoningBlock()
            self._mount(self._reasoning)
        self._reasoning.append(text)

    def _end_blocks(self) -> None:
        self._content = None
        self._reasoning = None

    # --------------------------------------------------------------------- undo

    async def undo_last_run(self) -> None:
        """Rewind the workspace to the most recent checkpoint."""
        if self._agent is not None:
            self._write(Text("undo: Ctrl+C to stop the run first", "yellow"))
            return
        checkpoints = await list_checkpoints(self.workspace)
        if not checkpoints:
            self._write(Text("undo: this workspace has no checkpoints", "yellow"))
            return
        session, commit = checkpoints[0]
        result = await restore(self.workspace, session)
        if not result.ok:
            self._write(Text.assemble(("undo failed  ", "bold #f85149"), result.error or ""))
            return
        detail = f"restored {session} ({commit[:8]})"
        if result.removed:
            detail += f"; removed {len(result.removed)} untracked file(s)"
        if result.untracked:
            detail += f"; left {len(result.untracked)} untracked file(s) in place"
        self._write(Text.assemble(("undo  ", "bold #3fb950"), detail))


def _stored_execution(call: StoredToolCall) -> ToolExecution:
    """A stored call, dressed as an execution so the card can render it.

    Resume only ever hands back groups in which every call has a result, so the
    stored outcome is always one of ``ToolOutcome``'s values.
    """
    return ToolExecution(
        call=ToolCall(id=call.call_id, name=call.name, arguments=call.arguments),
        tool=None,
        outcome=ToolOutcome(call.outcome),
        result=ToolResult(ok=call.outcome == ToolOutcome.OK.value, content=call.observation),
        fixes=call.fixes,
        reason=call.reason,
    )


def _completion_line(event: AgentEvent, tokens: int) -> Text:
    """The last line of a run: why it stopped, and what it cost.

    A provider error also prints the exception text. The stop word alone is not
    actionable -- "provider error" cannot tell a certificate you must trust from
    a model name the server does not recognise, and the user has no other way to
    find out short of reading the log.
    """
    word, colour = _STOP_LINES.get(event.text, (event.text, "#e3b341"))
    line = Text.assemble((word, f"bold {colour}"))
    if event.error:
        line.append(f"  {event.error}", style="#f85149")
    if tokens:
        line.append(f"  {tokens:,} tokens", style="dim")
    return line


class PlanApprover:
    """Plan mode: the engine asks, and the answer is always no, with a reason.

    It refuses rather than filtering, because by the time the engine asks, it has
    already decided the call needs a human. Saying no is that human's answer.
    """

    async def approve(self, request: ApprovalRequest) -> Approval:
        return Approval(
            Decision.DENY,
            note=(
                f"plan mode is read-only, so {request.name!r} was not run. Press f2 to switch "
                "to default or auto mode if you meant to let it through."
            ),
        )


class TUIApprover:
    """Asks a human in a modal and hands the answer back to the engine.

    Awaiting the screen from inside the loop's worker is what keeps the UI alive
    while the question is open: the loop waits, the app keeps rendering, and the
    answer is whatever the modal dismissed with. A screen closed without an
    answer is a refusal, not an approval.
    """

    def __init__(self, app: BevelApp) -> None:
        self._app = app

    async def approve(self, request: ApprovalRequest) -> Approval:
        answer: Approval | None = await self._app.push_screen_wait(ApprovalScreen(request))
        if answer is None:
            return Approval(Decision.DENY, note="the approval prompt closed without an answer")
        return answer


class ApprovalScreen(ModalScreen[Approval | None]):
    """One question: may this run?"""

    DEFAULT_CSS = """
    ApprovalScreen {
        align: center middle;
    }
    ApprovalScreen > VerticalScroll#approval {
        width: 80%;
        max-height: 80%;
        padding: 0 1;
        border: round #e3b341;
        background: #0d1117;
    }
    """

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="approval"):
            yield Static(_question(self.request))
            if self.request.needs_typed_confirmation:
                yield Input(placeholder=f"type {self.request.confirmation} to approve", id="typed")

    def on_mount(self) -> None:
        self.query_one("#approval", VerticalScroll).border_title = f"approval: {self.request.name}"
        if self.request.needs_typed_confirmation:
            self.query_one("#typed", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(Approval(Decision.ALLOW, typed=event.value))

    def key_escape(self) -> None:
        self._refuse("refused at the keyboard")

    def key_n(self) -> None:
        self._refuse("refused at the keyboard")

    def key_y(self) -> None:
        if not self.request.needs_typed_confirmation:
            self.dismiss(Approval(Decision.ALLOW, typed=self.request.confirmation))

    def key_a(self) -> None:
        """Allow this command and teach the engine to allow it unasked."""
        if self.request.argv and not self.request.needs_typed_confirmation:
            self.dismiss(Approval(Decision.ALLOW, typed=self.request.confirmation, remember=True))

    def _refuse(self, note: str) -> None:
        self.dismiss(Approval(Decision.DENY, note=note))


def _question(request: ApprovalRequest) -> Text:
    body = Text()
    body.append(f"Allow {request.name} ", style="bold")
    body.append(f"({request.tier.value})?\n", style="dim")
    body.append(f"{request.reason}\n")
    if request.argv:
        body.append(f"  command  {' '.join(request.argv)}\n")
    if request.paths:
        body.append(f"  paths    {', '.join(request.paths)}\n")
    if request.needs_typed_confirmation:
        body.append(f"type {request.confirmation} to approve, n to refuse", style="#e3b341")
    else:
        keys = "y to allow, n to refuse"
        if request.argv:
            keys += ", a to allow and always remember this command"
        body.append(keys, style="dim")
    return body


def run_tui(workspace: Path | str | None = None, resume: str | None = None) -> int:
    """Load config and run the TUI. Returns a process exit code.

    Config is loaded here rather than in ``cli`` so that the one thing that can
    stop bevel starting -- an unusable or missing endpoint -- is reported in the
    terminal that asked for bevel, before an alternate screen swallows it.

    ``resume`` names a session to reopen: an id, or a prefix of one. A name that
    matches nothing is an error here rather than a TUI that opens empty and
    silently starts a new history.
    """
    try:
        config = load()
    except ConfigError as exc:
        sys.stderr.write(f"bevel: {exc}\n")
        return 2
    registry = Registry(
        config.profiles,
        default_profile=config.default_profile,
        cache_path=capability_cache_path(),
    )
    store = SessionStore(session_db_path())
    # Constructing a MemoryStore opens nothing, so the gate is purely whether
    # the feature is on. When it is off the app gets None, the remember/recall
    # tools refuse with "not configured for this session", and no block is
    # injected -- which is the difference between "memory is off" and "memory
    # is broken and nobody noticed".
    memory = MemoryStore(memory_db_path(), config=config.memory) if config.memory.enabled else None
    # Same reasoning as the memory store: constructing it opens nothing, so this
    # is cheap whenever a session store exists. It shares sessions.db, and the
    # agent assigns its session id once one is open.
    todos = TodoStore(store.db_path)
    resumed: ResumedSession | None = None
    if resume is not None:
        try:
            info = store.resolve(resume)
        except ValueError as exc:
            store.close()
            todos.close()
            sys.stderr.write(f"bevel resume: {exc}\n")
            return 2
        resumed = rebuild_transcript(store, info.id)
    app = BevelApp(
        provider=registry.provider(),
        workspace=workspace,
        policy=config.policy,
        store=store,
        resume=resumed,
        memory=memory,
        todos=todos,
    )
    return asyncio.run(_close_after(app, registry))


async def _close_after(app: BevelApp, registry: Registry) -> int:
    try:
        await app.run_async()
    finally:
        await registry.aclose()
        if app.store is not None:
            app.store.close()
        if app.memory is not None:
            app.memory.close()
        if app.todos is not None:
            app.todos.close()
    return 0
