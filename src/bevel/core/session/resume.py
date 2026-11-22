"""Reopening a session: rebuilding the transcript the loop continues from.

A stored transcript is a record of what happened, which is not automatically a
requestable conversation. A server rejects a transcript with 400 if an
assistant message named tool calls that have no results beside them, and a
crash between "the model asked" and "the results came back" leaves exactly
that. So resume trims the tail until the history it hands the loop is one the
endpoint will accept, and reports how much it discarded rather than hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass

from bevel.core.messages import Message, ToolCall
from bevel.core.session.store import SessionInfo, SessionStore, StoredTurn


@dataclass(frozen=True)
class ResumedSession:
    """What a closed session comes back as.

    ``stored`` is every row that survived, with its tool calls attached -- what
    the UI renders. ``messages`` is the same history in the shape the loop
    sends. They are built from one read of the store so the two can never
    disagree about what the session contained.

    ``dropped`` is how many stored rows were unusable at the tail. It is
    reported rather than swallowed because a session that lost its last turn is
    a thing the user should know happened.
    """

    session: SessionInfo
    messages: list[Message]
    stored: tuple[StoredTurn, ...]
    dropped: int

    @property
    def turns(self) -> int:
        return len(self.stored)


def _message(turn: StoredTurn) -> Message:
    return Message(
        role=turn.role,  # type: ignore[arg-type]
        content=turn.content,
        reasoning=turn.reasoning,
        tool_calls=tuple(
            ToolCall(id=call.call_id, name=call.name, arguments=call.arguments)
            for call in turn.tool_calls
        ),
        tool_call_id=turn.tool_call_id,
    )


def _groups(turns: list[StoredTurn]) -> list[list[StoredTurn]]:
    """Partition into requestable units: a message, or a call plus its results.

    A tool result belongs to the assistant message that asked for it, so the
    two travel as one unit and are kept or dropped together.
    """
    groups: list[list[StoredTurn]] = []
    for turn in turns:
        if turn.role == "tool" and groups and groups[-1][0].tool_calls:
            groups[-1].append(turn)
        else:
            groups.append([turn])
    return groups


def _complete(group: list[StoredTurn]) -> bool:
    """Does every call the assistant named have exactly one result?"""
    if group[0].role == "tool":
        return False  # a result whose call is gone cannot be sent
    asked = group[0].tool_calls
    if not asked:
        return len(group) == 1
    answered = sorted(turn.tool_call_id or "" for turn in group[1:])
    return answered == sorted(call.call_id for call in asked)


def _unusable(group: list[StoredTurn]) -> int:
    """How many stored rows the dropped group held."""
    return len(group)


def _kept(group: list[StoredTurn]) -> list[StoredTurn]:
    """The group's rows in the order they should be shown and sent.

    Parallel tools finish out of order; the wire format wants results back in
    the sequence the model listed, which is also the sequence it reads them in.
    """
    asked = group[0].tool_calls
    if not asked:
        return list(group)
    by_id = {turn.tool_call_id: turn for turn in group[1:]}
    return [group[0]] + [by_id[call.call_id] for call in asked]


def rebuild_transcript(store: SessionStore, session_id: str) -> ResumedSession | None:
    """Rebuild a session's messages, or ``None`` if there is no such session."""
    session = store.get(session_id)
    if session is None:
        return None
    turns = list(store.turns(session_id))

    groups = _groups(turns)
    dropped = 0
    while groups and not _complete(groups[-1]):
        dropped += _unusable(groups.pop())

    messages: list[Message] = []
    stored: list[StoredTurn] = []
    for group in groups:
        kept = _kept(group)
        stored.extend(kept)
        messages.extend(_message(turn) for turn in kept)
    return ResumedSession(session=session, messages=messages, stored=tuple(stored), dropped=dropped)


__all__ = ["ResumedSession", "rebuild_transcript"]
