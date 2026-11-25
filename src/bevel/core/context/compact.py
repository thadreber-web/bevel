"""L1, L2 and L3 compaction: fit an over-long transcript back into the window.

The trigger is always the server's own ``prompt_tokens`` against
``Capabilities.max_model_len`` — never a chars/4 guess (``docs/provider-notes.md``:
the target endpoint has no ``/tokenize``, so there is nothing to predict with).
These three functions therefore decide nothing about budgets. They take a transcript
the loop has already measured as too big and hand back a smaller one, plus the
truth of what they did.

**L1** cuts stale tool observations to head + tail. Pure string work: no model
call, no I/O, no new dependency, deterministic — the same transcript compacts to
the same bytes every time. The dropped text still has a persisted copy in the
session store, so the replacement names the call that produced it, points at the
stored session, and tells the model to re-run it.

**L2** replaces the middle of the transcript with one summary and leaves the last
N turns byte-identical. It costs a generation, which is why the loop tries L1
first and escalates only when L1 has nothing left to give.

**L3** is the handoff: it replaces everything before the kept tail — including
any L2 summary, which it supersedes — with a head the loop supplies. L3 is the
one level that is not self-contained, and deliberately so. The head is the
opening shape from ``Agent._seed()``: a system prompt carrying a freshly built
project map, a recall block, and the brief. The project map lives on disk, the
recall block in the memory database and the brief in the todo store, so the
level that composes them cannot be the pure one. ``compact_l3`` is given the
finished head and does the only thing that needs doing here: cut the transcript
down to head + kept tail, and refuse if that is not actually smaller.

All three levels are **idempotent**: L1 skips text it has already cut, L2 refuses
to summarise its own summary, and L3 refuses to hand off a transcript that is
already a handoff with nothing new to fold in. "This level changed nothing" is
therefore a reliable signal that the level is exhausted, and the loop uses
exactly that to escalate — and to stop when no level can help.

The transcript's shape is an invariant, not a nicety: an assistant message that
asked for tools must stay next to every tool result answering it, or the next
request is a 400 from the server. Both levels cut only at turn-group boundaries
and never touch ``messages[0]``, the system message that carries the project map.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Final

from ..messages import Message

__all__ = [
    "L1Result",
    "L2Result",
    "L3Result",
    "Summarise",
    "compact_l1",
    "compact_l2",
    "compact_l3",
    "handoff_cut",
]

# An observation this short is already cheaper to keep than to describe.
MIN_OBSERVATION_CHARS: Final = 1_200
HEAD_CHARS: Final = 400
TAIL_CHARS: Final = 400

# L2 keeps the last N assistant turns verbatim. Anything older is fair game.
KEEP_RECENT_TURNS: Final = 4

# L3 keeps fewer. Each rung of the ladder has to be able to do what the rung above
# it could not: were L3 to keep L2's four turns, the two levels would run out at
# exactly the same transcript and L3 would be unreachable by escalation. Keeping
# two means the transcript L2 gives up on at four turns is the one L3 begins from.
HANDOFF_KEEP_RECENT_TURNS: Final = 2

_TRUNCATED: Final = "[observation truncated by L1 compaction]"
_SUMMARY_PREFIX: Final = "[summary of earlier turns]"
_BRIEF_PREFIX: Final = "[handoff brief]"

# L2 asks for a summary through this; the loop supplies it from a ModelProvider.
Summarise = Callable[[list[Message]], Awaitable[str]]


@dataclass(frozen=True)
class L1Result:
    """What L1 did. ``chars_removed`` is 0 when there was nothing stale to cut."""

    messages: list[Message]
    chars_removed: int
    observations_cut: int

    @property
    def changed(self) -> bool:
        return self.chars_removed > 0


@dataclass(frozen=True)
class L2Result:
    """What L2 did. ``chars_removed`` is 0 when the middle was already minimal."""

    messages: list[Message]
    chars_removed: int
    turns_replaced: int
    summary: str

    @property
    def changed(self) -> bool:
        return self.chars_removed > 0


def compact_l1(messages: list[Message], *, session_id: str | None = None) -> L1Result:
    """Cut stale tool observations to head + tail plus a pointer to the saved text.

    Only observations over ``MIN_OBSERVATION_CHARS`` are touched, and only once:
    text already carrying the truncation marker is left alone so a second pass
    reports "changed nothing" instead of shaving the same message forever.

    ``session_id`` is the session whose store holds the full text of every
    observation. When it is given, the replacement points the model at that
    saved copy; when it is not (no store attached), the replacement can only
    name the call to re-run, and says the text was not saved.
    """
    origins = _tool_call_origins(messages)
    out: list[Message] = []
    removed = 0
    cut = 0
    for message in messages:
        if message.role != "tool" or len(message.content) <= MIN_OBSERVATION_CHARS:
            out.append(message)
            continue
        if _TRUNCATED in message.content:
            out.append(message)
            continue
        replacement = _truncate_observation(
            message, origins.get(message.tool_call_id or ""), session_id
        )
        removed += len(message.content) - len(replacement)
        cut += 1
        out.append(
            Message(
                role=message.role,
                content=replacement,
                tool_call_id=message.tool_call_id,
            )
        )
    return L1Result(messages=out, chars_removed=removed, observations_cut=cut)


def _tool_call_origins(messages: list[Message]) -> dict[str, str]:
    """Map each tool call id to the ``name(args)`` that produced it."""
    origins: dict[str, str] = {}
    for message in messages:
        for call in message.tool_calls:
            origins[call.id] = f"{call.name}({call.arguments})"
    return origins


def _truncate_observation(message: Message, origin: str | None, session_id: str | None) -> str:
    content = message.content
    head = content[:HEAD_CHARS]
    tail = content[-TAIL_CHARS:]
    omitted = len(content) - len(head) - len(tail)
    if origin is None:
        pointer = f"{_TRUNCATED} {omitted} characters omitted; the originating call is unknown."
    elif session_id is not None:
        pointer = (
            f"{_TRUNCATED} {omitted} characters omitted; the full text is saved in session "
            f"{session_id} under tool call {message.tool_call_id}, so it can be read back "
            f"with `bevel resume {session_id}`. Re-run {origin} to see it again now."
        )
    else:
        pointer = (
            f"{_TRUNCATED} {omitted} characters omitted and the full text was not saved. "
            f"Re-run {origin} to see it again."
        )
    return f"{head}\n\n{pointer}\n\n{tail}"


async def compact_l2(messages: list[Message], summarise: Summarise) -> L2Result:
    """Summarise the middle of the transcript, keeping the last N turns verbatim.

    ``summarise`` is given the messages being replaced and returns the summary
    text. The kept tail is the *same* ``Message`` objects, byte-identical. If the
    summary does not actually shrink the transcript, the transcript is returned
    unchanged with ``chars_removed == 0`` — a level that cannot help must say so.
    """
    head, middle, tail = _split_for_summary(messages)
    if not middle:
        # Nothing to compress: either the transcript is already short, or the middle
        # is only a previous summary whose turns are now kept verbatim. Either way a
        # generation would be spent for no reduction.
        return L2Result(messages=messages, chars_removed=0, turns_replaced=0, summary="")
    summary = (await summarise(middle)).strip()
    if not summary:
        return L2Result(messages=messages, chars_removed=0, turns_replaced=0, summary="")
    summary_message = Message(role="user", content=f"{_SUMMARY_PREFIX}\n{summary}")
    out = [*head, summary_message, *tail]
    before = sum(len(m.content) for m in messages)
    after = sum(len(m.content) for m in out)
    if after >= before:
        return L2Result(messages=messages, chars_removed=0, turns_replaced=0, summary=summary)
    return L2Result(
        messages=out,
        chars_removed=before - after,
        turns_replaced=_count_turns(middle),
        summary=summary,
    )


def _split_for_summary(
    messages: list[Message],
) -> tuple[list[Message], list[Message], list[Message]]:
    """Split into (head, middle to replace, last N turns kept verbatim).

    The head is everything before the first assistant turn — the system message
    carrying the project map, and the user message carrying the task. Both are the
    transcript's anchors: a small local model's paraphrase of either is exactly the
    thing that goes wrong, so neither is ever offered to the summariser.

    Cuts land on turn-group boundaries: a group is one assistant message and the
    tool results that answer it. Keeping a group whole is what stops the result
    from being a request the server rejects.
    """
    start = next(
        (i for i, message in enumerate(messages) if message.role == "assistant"),
        len(messages),
    )
    head, body = messages[:start], messages[start:]
    starts = _turn_group_starts(body)
    if len(starts) <= KEEP_RECENT_TURNS:
        # Fewer turns than we keep: the level is exhausted. After one pass the
        # transcript holds exactly KEEP_RECENT_TURNS assistant turns, so this is also
        # what stops a second pass from re-summarising the first pass's summary.
        return messages, [], []
    return head, body[: starts[-KEEP_RECENT_TURNS]], body[starts[-KEEP_RECENT_TURNS] :]


def _turn_group_starts(body: list[Message]) -> list[int]:
    """Indices in ``body`` where a new assistant turn begins."""
    return [i for i, message in enumerate(body) if message.role == "assistant"]


def _count_turns(middle: list[Message]) -> int:
    return sum(1 for message in middle if message.role == "assistant")


def _handoff_split(
    messages: list[Message],
) -> tuple[list[Message], list[Message], list[Message]]:
    """Split into (body, region to hand off, last N turns kept verbatim).

    The tail is ``HANDOFF_KEEP_RECENT_TURNS`` groups, fewer than L2 keeps, and the
    region is everything above it: the task message, any L2 summary, and the turns
    older than the tail. A non-empty region always holds at least one assistant
    turn, because the cut is the start of the second-to-last group.
    """
    body = messages[1:] if messages and messages[0].role == "system" else list(messages)
    starts = _turn_group_starts(body)
    if len(starts) <= HANDOFF_KEEP_RECENT_TURNS:
        return body, [], body
    cut = starts[-HANDOFF_KEEP_RECENT_TURNS]
    return body, body[:cut], body[cut:]


def handoff_cut(messages: list[Message]) -> list[Message]:
    """The messages L3 would replace, or empty when there is nothing to hand off.

    The loop asks this *before* writing the brief, because the brief has to
    account for the turns it is about to drop. Empty is the same verdict
    ``compact_l3`` reaches, so a transcript that is already a handoff and has not
    grown since never buys a generation to find that out.
    """
    _, dropped, _ = _handoff_split(messages)
    return dropped


@dataclass(frozen=True)
class L3Result:
    """What L3 did. ``chars_removed`` is 0 when there was nothing to hand off.

    ``reason`` says why nothing was done, in the words the loop puts in front of
    the user. A level that refused and a level that found no work are the same
    outcome to the loop and different ones to whoever is reading the log.
    """

    messages: list[Message]
    chars_removed: int
    turns_replaced: int
    summaries_removed: int
    brief: str
    reason: str

    @property
    def changed(self) -> bool:
        return self.chars_removed > 0


def compact_l3(messages: list[Message], *, head: Sequence[Message], brief: str) -> L3Result:
    """Replace everything before the kept tail with ``head`` plus the handoff brief.

    ``head`` is the opening shape ``Agent._seed()`` builds — the system prompt
    with a freshly built project map, then the recall block — and ``brief`` is
    the structured summary that stands in for the turns being dropped. Both are
    the loop's to build, because both need things this module must not reach for:
    the filesystem, the memory database, the todo store, and a generation. The
    loop checks :func:`handoff_cut` first, so a transcript with nothing to fold
    in never buys the brief's request.

    Unlike L2, L3 drops the earlier summary rather than keeping it. The summary
    covered the turns this brief now covers, so keeping both is paying twice for
    one account — and it is exactly why L3 can fit a transcript L2 could not.
    The kept tail is the same ``Message`` objects, byte-identical, cut at a
    turn-group boundary so no tool call is separated from its result.

    If head + brief + tail is not smaller than the transcript the server already
    measured as too big, the transcript comes back untouched with a reason. The
    loop must not send a request it knows will be rejected, so a brief that is
    itself the problem has to be reported rather than shipped.
    """
    body = messages[1:] if messages and messages[0].role == "system" else messages
    starts = _turn_group_starts(body)
    if len(starts) <= HANDOFF_KEEP_RECENT_TURNS:
        already = any(_BRIEF_PREFIX in message.content for message in messages)
        return L3Result(
            messages,
            0,
            0,
            0,
            brief,
            "the transcript is already a handoff with nothing new to fold in"
            if already
            else "there are no turns to hand off",
        )

    cut = starts[-HANDOFF_KEEP_RECENT_TURNS]
    dropped, tail = body[:cut], body[cut:]
    brief_message = Message(role="user", content=f"{_BRIEF_PREFIX}\n{brief}")
    out = [*head, brief_message, *tail]
    before = sum(len(message.content) for message in messages)
    after = sum(len(message.content) for message in out)
    if after >= before:
        return L3Result(
            messages,
            0,
            0,
            0,
            brief,
            f"the rebuilt transcript is {after} chars against the {before} chars that "
            "already did not fit",
        )
    summaries_removed = sum(1 for message in dropped if message.content.startswith(_SUMMARY_PREFIX))
    return L3Result(
        messages=out,
        chars_removed=before - after,
        # The superseded summary is not a turn the model took, so it is not counted
        # as one that was folded into the brief.
        turns_replaced=_count_turns(dropped) - summaries_removed,
        summaries_removed=summaries_removed,
        brief=brief,
        reason="",
    )
