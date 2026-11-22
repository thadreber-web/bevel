"""L1 and L2 compaction: fit an over-long transcript back into the window.

The trigger is always the server's own ``prompt_tokens`` against
``Capabilities.max_model_len`` — never a chars/4 guess (``docs/provider-notes.md``:
the target endpoint has no ``/tokenize``, so there is nothing to predict with).
These two functions therefore decide nothing about budgets. They take a transcript
the loop has already measured as too big and hand back a smaller one, plus the
truth of what they did.

**L1** cuts stale tool observations to head + tail. Pure string work: no model
call, no I/O, no new dependency, deterministic — the same transcript compacts to
the same bytes every time. The dropped text is not stored anywhere (session
persistence is a later slice of m08), so the replacement names the call that
produced it and tells the model to re-run it.

**L2** replaces the middle of the transcript with one summary and leaves the last
N turns byte-identical. It costs a generation, which is why the loop tries L1
first and escalates only when L1 has nothing left to give.

Both levels are **idempotent**: L1 skips text it has already cut, and L2 refuses to
summarise its own summary. "This level changed nothing" is therefore a reliable
signal that the level is exhausted, and the loop uses exactly that to escalate —
and to stop when neither level can help, which is the L3 handoff's job, and L3 is
not built.

The transcript's shape is an invariant, not a nicety: an assistant message that
asked for tools must stay next to every tool result answering it, or the next
request is a 400 from the server. Both levels cut only at turn-group boundaries
and never touch ``messages[0]``, the system message that carries the project map.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final

from ..messages import Message

__all__ = ["L1Result", "L2Result", "Summarise", "compact_l1", "compact_l2"]

# An observation this short is already cheaper to keep than to describe.
MIN_OBSERVATION_CHARS: Final = 1_200
HEAD_CHARS: Final = 400
TAIL_CHARS: Final = 400

# L2 keeps the last N assistant turns verbatim. Anything older is fair game.
KEEP_RECENT_TURNS: Final = 4

_TRUNCATED: Final = "[observation truncated by L1 compaction]"
_SUMMARY_PREFIX: Final = "[summary of earlier turns]"

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


def compact_l1(messages: list[Message]) -> L1Result:
    """Cut stale tool observations to head + tail plus a pointer to re-run the call.

    Only observations over ``MIN_OBSERVATION_CHARS`` are touched, and only once:
    text already carrying the truncation marker is left alone so a second pass
    reports "changed nothing" instead of shaving the same message forever.
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
        replacement = _truncate_observation(message, origins.get(message.tool_call_id or ""))
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


def _truncate_observation(message: Message, origin: str | None) -> str:
    content = message.content
    head = content[:HEAD_CHARS]
    tail = content[-TAIL_CHARS:]
    omitted = len(content) - len(head) - len(tail)
    if origin is None:
        pointer = f"{_TRUNCATED} {omitted} characters omitted; the originating call is unknown."
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
