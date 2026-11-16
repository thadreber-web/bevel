"""Bounded, observable repair of malformed tool calls.

A local model will sometimes emit ``{"path": "src/app.py",}``, or wrap the
arguments in a prose sentence, or truncate the JSON at ``max_tokens``. Each of
those is a syntax problem, and fixing syntax is legitimate. Inventing a value
the model never wrote is not: the line between the two is the whole design of
this module.

The rules, in order:

1. Try the arguments exactly as sent.
2. Apply only *lossless* normalisations (strip fences, trim whitespace, drop a
   trailing comma). Each one applied is recorded.
3. Validate against the tool's schema. If it passes, done.
4. Re-ask the model, at most ``max_attempts`` times, quoting the specific
   violations back to it.
5. Give up and return a :class:`RepairOutcome` that says so.

Nothing here ever fills in a missing property, coerces a string to a number, or
guesses a path. If the model did not say it, bevel does not say it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from bevel.core.messages import Message, ToolCall
from bevel.core.tools.schema import SchemaError, validate
from bevel.core.tools.types import ToolSpec
from bevel.providers.base import (
    GenOptions,
    ModelProvider,
    ProviderError,
    TransportError,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 2
"""Re-asks per tool call. Two is enough to catch a transient formatting slip and
small enough that a model which cannot produce valid JSON fails fast instead of
burning the context window on the same mistake."""


@dataclass(frozen=True)
class RepairOutcome:
    """The result of one repair attempt, including what was changed.

    ``fixes`` names every normalisation applied and every re-ask performed, so
    the TUI can show "repaired: stripped code fence, re-asked once" rather than
    silently presenting a call the model never literally sent.
    """

    call: ToolCall | None
    arguments: dict[str, Any] | None
    fixes: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    attempts: int = 0

    @property
    def ok(self) -> bool:
        return self.arguments is not None

    @property
    def repaired(self) -> bool:
        return self.ok and bool(self.fixes)


_FENCE_OPEN = "```"
_TRAILING_COMMA = ",}"
_TRAILING_COMMA_ARRAY = ",]"


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    """Parse ``text`` as a JSON object, tolerating common wrappers.

    Returns ``(arguments, fixes)``. Only transformations that cannot change the
    data are applied, and each is named in ``fixes``. If nothing works the
    result is ``(None, fixes)`` -- the caller then decides whether to re-ask.
    """
    fixes: list[str] = []
    raw = text

    stripped = raw.strip()
    if stripped != raw:
        fixes.append("trimmed whitespace")
        raw = stripped

    if raw.startswith(_FENCE_OPEN) and raw.endswith(_FENCE_OPEN):
        body = raw[len(_FENCE_OPEN) : -len(_FENCE_OPEN)]
        newline = body.find("\n")
        if newline != -1 and body[:newline].strip() in {"", "json", "JSON"}:
            body = body[newline + 1 :]
        raw = body.strip()
        fixes.append("stripped code fence")

    parsed = _try_object(raw)
    if parsed is not None:
        return parsed, tuple(fixes)

    sliced = _slice_to_braces(raw)
    if sliced is not None and sliced != raw:
        parsed = _try_object(sliced)
        if parsed is not None:
            fixes.append("extracted outermost object")
            return parsed, tuple(fixes)
        relaxed = _drop_trailing_commas(sliced)
        if relaxed != sliced:
            parsed = _try_object(relaxed)
            if parsed is not None:
                fixes.append("extracted outermost object")
                fixes.append("dropped trailing commas")
                return parsed, tuple(fixes)

    relaxed = _drop_trailing_commas(raw)
    if relaxed != raw:
        parsed = _try_object(relaxed)
        if parsed is not None:
            fixes.append("dropped trailing commas")
            return parsed, tuple(fixes)

    return None, tuple(fixes)


def _try_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _slice_to_braces(text: str) -> str | None:
    """Return the substring from the first ``{`` to the last ``}``.

    Lets a model say "sure, here you go: {...}" without that prose breaking the
    parse. The braces themselves are kept, so nothing is added to the payload.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    return text[start : end + 1]


def _drop_trailing_commas(text: str) -> str:
    return text.replace(_TRAILING_COMMA, "}").replace(_TRAILING_COMMA_ARRAY, "]")


def format_violations(errors: Sequence[SchemaError]) -> str:
    """Render violations for the re-ask prompt.

    Quoting the validator's own words is what makes the second attempt succeed;
    a generic "your JSON was invalid" gives the model nothing to correct.
    """
    return "; ".join(str(error) for error in errors)


class ToolCallRepairer:
    """Repairs tool calls against their declared schema.

    Stateless between calls, so one instance can serve the whole agent loop.
    """

    def __init__(
        self,
        provider: ModelProvider,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        if max_attempts < 0:
            raise ValueError("max_attempts must be non-negative")
        self._provider = provider
        self._max_attempts = max_attempts

    async def repair(
        self,
        call: ToolCall,
        spec: ToolSpec,
        history: Sequence[Message] = (),
    ) -> RepairOutcome:
        """Return valid arguments for ``call``, or an outcome explaining why not.

        ``history`` is included in re-asks so the model can see the file it was
        reading or the command it just ran; without it, a repair request is
        unanswerable and the second attempt is wasted.
        """
        fixes: list[str] = []
        parsed, applied = extract_json_object(call.arguments)
        fixes.extend(applied)

        if parsed is not None:
            violations = validate(parsed, spec.parameters)
            if not violations:
                return RepairOutcome(
                    call=_with_arguments(call, parsed),
                    arguments=parsed,
                    fixes=tuple(fixes),
                    attempts=0,
                )
        else:
            violations = [SchemaError("$", "arguments were not a JSON object")]

        attempts = 0
        last_errors = tuple(str(v) for v in violations)

        while attempts < self._max_attempts:
            attempts += 1
            fixes.append(f"re-asked attempt {attempts}")
            reask = await self._reask(spec, call, violations, history)
            if reask is None:
                # The model could not be asked at all; another attempt would
                # fail the same way, so stop rather than burn the budget.
                break
            candidate, applied = reask
            fixes.extend(applied)
            if candidate is None:
                # The model answered with something that is not an object. Keep
                # the violations we already have and add this one -- the
                # original errors are the actionable ones, and dropping them to
                # say only "that wasn't JSON" would leave the user (and the
                # next re-ask) with nothing to correct.
                violations = [
                    *violations,
                    SchemaError("$", "the corrected arguments were not a JSON object"),
                ]
            else:
                violations = validate(candidate, spec.parameters)
            last_errors = tuple(str(v) for v in violations)
            if not violations and candidate is not None:
                return RepairOutcome(
                    call=_with_arguments(call, candidate),
                    arguments=candidate,
                    fixes=tuple(fixes),
                    attempts=attempts,
                )
        logger.debug(
            "repair failed for tool %s after %d attempt(s): %s",
            spec.name,
            attempts,
            "; ".join(last_errors),
        )
        return RepairOutcome(
            call=None,
            arguments=None,
            fixes=tuple(fixes),
            errors=last_errors,
            attempts=attempts,
        )

    async def _reask(
        self,
        spec: ToolSpec,
        call: ToolCall,
        violations: Sequence[SchemaError],
        history: Sequence[Message],
    ) -> tuple[dict[str, Any] | None, tuple[str, ...]] | None:
        """Ask the model to reissue the call.

        Returns ``None`` when the model could not be asked at all -- a transport
        or API failure. That is different from asking successfully and getting
        back something unusable, which is ``(None, fixes)``: the first is worth
        giving up on immediately, the second deserves the remaining attempts.
        """
        prompt = _reask_prompt(spec, call, violations)
        messages = [*history, Message.user(prompt)]
        options = None
        if self._provider.capabilities.supports_structured_output:
            options = GenOptions(json_mode=True, temperature=0.0)

        try:
            turn = await self._provider.complete(messages, tools=None, options=options)
        except TransportError as exc:
            logger.debug("repair re-ask aborted, transport failure: %s", exc)
            return None
        except ProviderError as exc:
            logger.debug("repair re-ask failed: %s", exc)
            return None

        text = turn.message.content
        if not text:
            return (None, ())
        return extract_json_object(text)


def _with_arguments(call: ToolCall, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(id=call.id, name=call.name, arguments=json.dumps(arguments))


def _reask_prompt(spec: ToolSpec, call: ToolCall, violations: Sequence[SchemaError]) -> str:
    """Build the correction prompt.

    Asks for arguments only -- no prose, no explanation -- because every extra
    token the model writes is another token that can be wrong.
    """
    return (
        f"Your previous tool call to `{spec.name}` had arguments that failed validation.\n"
        f"Arguments sent: {call.arguments}\n"
        f"Validation errors: {format_violations(violations)}\n\n"
        f"Expected argument schema:\n{json.dumps(spec.parameters, indent=2)}\n\n"
        "Respond with ONLY a corrected JSON object for the arguments. "
        "No prose, no code fence, no commentary. "
        "Do not invent values that were not present in your original call."
    )


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "RepairOutcome",
    "ToolCallRepairer",
    "extract_json_object",
    "format_violations",
]
