"""OpenAI-compatible chat provider over httpx.

Targets the subset of the OpenAI chat-completions API that vLLM and llama.cpp
server actually implement. Deliberately does *not* depend on the ``openai``
SDK: the divergences that matter here (``delta.reasoning``, the empty-``choices``
usage chunk, per-``index`` argument fragments, unproxied routes) all live
outside the SDK's documented surface, and httpx is already a declared
dependency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from bevel.core.messages import Message
from bevel.core.tools.types import ToolSpec
from bevel.providers._wire import (
    ChatStreamDecoder,
    iter_sse_data,
    parse_completion_response,
    parse_error_body,
    parse_tool_calls_from_message,
)
from bevel.providers.base import (
    APIError,
    AssistantTurn,
    Capabilities,
    GenOptions,
    Health,
    MalformedResponseError,
    StreamEvent,
    TokenCount,
    TransportError,
    assemble,
)

logger = logging.getLogger(__name__)

USER_AGENT = "bevel"
"""Sent on every request so server-side logs can tell bevel apart from the
other clients pointed at the same box."""


def _serialize_message(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role}
    if message.role == "tool":
        payload["content"] = message.content
        payload["tool_call_id"] = message.tool_call_id or ""
        return payload

    if message.content or not message.tool_calls:
        payload["content"] = message.content
    else:
        # Some servers reject a null content alongside tool_calls; an empty
        # string is accepted everywhere.
        payload["content"] = ""

    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in message.tool_calls
        ]
    return payload


def build_request(
    model: str,
    messages: Sequence[Message],
    tools: Sequence[ToolSpec] | None,
    options: GenOptions | None,
    *,
    stream: bool,
) -> dict[str, Any]:
    """Build a chat-completions body.

    Only non-``None`` options are sent: ``temperature: null`` is not the same as
    omitting the field on every server, and "use the server's own default" is
    the behaviour we want when bevel has no opinion.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": [_serialize_message(m) for m in messages],
        "stream": stream,
    }
    if tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
    if options is not None:
        if options.temperature is not None:
            body["temperature"] = options.temperature
        if options.top_p is not None:
            body["top_p"] = options.top_p
        if options.max_tokens is not None:
            body["max_tokens"] = options.max_tokens
        if options.json_mode:
            body["response_format"] = {"type": "json_object"}
    if stream:
        # Without this the stream ends with no usage at all, which would leave
        # the budget guard blind on the path that matters most.
        body["stream_options"] = {"include_usage": True}
    return body


class OpenAICompatProvider:
    """One endpoint profile.

    ``capabilities`` starts pessimistic and is replaced by the registry once a
    probe has run. Constructing this object performs no I/O -- ``health()`` and
    the first ``stream()`` do.
    """

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        connect_timeout: float = 10.0,
        verify_tls: bool = True,
        extra_headers: dict[str, str] | None = None,
        capabilities: Capabilities | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.verify_tls = verify_tls
        self.capabilities = capabilities or Capabilities()
        self._api_key = api_key
        self._closed = False

        headers = {"User-Agent": USER_AGENT}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)

        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            headers=headers,
            verify=verify_tls,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
        )
        if not self._owns_client:
            # An injected client would otherwise be used exactly as given,
            # silently dropping the API key and custom headers -- every request
            # would go out unauthenticated. Applying them to the client rather
            # than to each call site means no request path can forget them.
            self._client.headers.update(headers)

    async def aclose(self) -> None:
        if self._closed or not self._owns_client:
            return
        self._closed = True
        await self._client.aclose()

    async def __aenter__(self) -> OpenAICompatProvider:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # -- requests ---------------------------------------------------------

    async def _raise_for_response(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        message, code = parse_error_body(response.status_code, response.text)
        raise APIError(response.status_code, message, code=code)

    async def _post_json(
        self, path: str, body: dict[str, Any], accept: str = "application/json"
    ) -> httpx.Response:
        url = f"{self.base_url}{path}"
        try:
            response = await self._client.post(url, json=body, headers={"Accept": accept})
        except httpx.TimeoutException as exc:
            raise TransportError(f"timed out calling {url}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise TransportError(f"could not reach {url}: {exc}") from exc
        await self._raise_for_response(response)
        return response

    # -- ModelProvider ----------------------------------------------------

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        options: GenOptions | None = None,
    ) -> AsyncIterator[StreamEvent]:
        return self._stream(messages, tools, options)

    async def _stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None,
        options: GenOptions | None,
    ) -> AsyncIterator[StreamEvent]:
        body = build_request(self.model, messages, tools, options, stream=True)
        url = f"{self.base_url}/chat/completions"
        headers = {"Accept": "text/event-stream"}
        decoder = ChatStreamDecoder()

        try:
            async with self._client.stream("POST", url, json=body, headers=headers) as response:
                if not response.is_success:
                    await response.aread()
                    await self._raise_for_response(response)
                    return

                async for line in response.aiter_lines():
                    for payload in iter_sse_data([line]):
                        for event in decoder.feed(payload):
                            yield event
                    if decoder.finished:
                        # Only true once [DONE] arrives or a bounded number of
                        # payloads have passed since the finish reason -- the
                        # usage chunk comes after it. Breaking on the finish
                        # reason itself would discard the token count, and
                        # waiting forever would hang on a gateway that keeps
                        # the connection open.
                        break
        except httpx.TimeoutException as exc:
            raise TransportError(f"timed out streaming from {url}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise TransportError(f"could not reach {url}: {exc}") from exc

        end = decoder.finalize()
        if end.finish_reason == "abort":
            logger.debug("stream from %s ended without a finish reason", url)
        yield end

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        options: GenOptions | None = None,
    ) -> AssistantTurn:
        return await assemble(self.stream(messages, tools, options))

    async def complete_non_streaming(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        options: GenOptions | None = None,
    ) -> AssistantTurn:
        """One-shot request without SSE.

        Separate from :meth:`complete` because it exercises a genuinely
        different server-side path; having both lets us compare them when a
        stream misbehaves.
        """
        body = build_request(self.model, messages, tools, options, stream=False)
        response = await self._post_json("/chat/completions", body)
        text = response.text
        content, reasoning, finish, usage, model = parse_completion_response(text)
        return AssistantTurn(
            message=Message(
                "assistant",
                content=content,
                reasoning=reasoning,
                tool_calls=parse_tool_calls_from_message(text),
            ),
            finish_reason=finish,
            usage=usage,
            model=model,
        )

    async def count_tokens(self, messages: Sequence[Message]) -> TokenCount:
        """Exact count via ``/tokenize``, or ``TokenCount(None)``.

        There is no fallback and no estimate. A gateway that does not proxy the
        route answers 404, which is indistinguishable from "build without the
        route", and both mean bevel cannot predict cost.
        """
        if not self.capabilities.supports_token_api:
            return TokenCount(None, source="not advertised")
        body = {"model": self.model, "messages": [_serialize_message(m) for m in messages]}
        try:
            response = await self._post_json("/tokenize", body)
        except APIError as exc:
            if exc.status == 404:
                return TokenCount(None, source="no /tokenize route")
            raise
        except TransportError:
            return TokenCount(None, source="tokenize unreachable")

        payload = _json_or_raise(response, "/tokenize")
        count = payload.get("count") if isinstance(payload, dict) else None
        if not isinstance(count, int):
            raise MalformedResponseError(
                "/tokenize response had no integer count", body=response.text
            )
        return TokenCount(count, source="server /tokenize")

    async def health(self) -> Health:
        """Liveness via ``GET /models``.

        ``/health`` is not used because gateways commonly do not proxy it, so a
        404 there says nothing about whether the model is reachable. ``/models``
        is required by the API shape, costs no tokens, and reports the context
        window the budget guard needs.
        """
        started = time.perf_counter()
        url = f"{self.base_url}/models"
        try:
            response = await self._client.get(url, headers={"Accept": "application/json"})
        except httpx.TimeoutException as exc:
            return Health(False, f"timed out calling {url}: {exc}")
        except httpx.HTTPError as exc:
            return Health(False, f"could not reach {url}: {exc}")

        latency_ms = (time.perf_counter() - started) * 1000.0
        if not response.is_success:
            message, code = parse_error_body(response.status_code, response.text)
            detail = f"{url} answered {response.status_code}: {message}"
            if code:
                detail = f"{detail} (code {code})"
            return Health(False, detail, latency_ms)

        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError):
            return Health(False, f"{url} returned a non-JSON body", latency_ms)

        models = _model_ids(payload)
        if not models:
            return Health(False, f"{url} listed no models", latency_ms)
        return Health(True, f"{url} OK", latency_ms, models=models)

    # -- probing ----------------------------------------------------------

    async def probe(self) -> Capabilities:
        """Discover what this endpoint supports, by asking it.

        Every flag is earned. The cost is a handful of cheap requests, run once
        per profile and then cached by the registry. A probe that fails reports
        the feature as absent rather than raising, so an unreachable route never
        prevents bevel from starting.
        """
        caps = Capabilities()
        health = await self.health()
        if not health.ok:
            self.capabilities = caps
            return caps

        token_api, embeddings, json_mode, reasoning, max_model_len = await asyncio.gather(
            self._route_exists("/tokenize", {"model": self.model, "prompt": "ping"}),
            self._route_exists("/embeddings", {"model": self.model, "input": "ping"}),
            self._probe_json_mode(),
            self._probe_reasoning(),
            self._probe_max_model_len(),
        )
        supports_tools, parallel_calls = await self._probe_tools()

        caps = Capabilities(
            supports_tools=supports_tools,
            supports_parallel_tool_calls=parallel_calls,
            supports_structured_output=json_mode,
            supports_token_api=token_api,
            supports_embeddings=embeddings,
            max_model_len=max_model_len,
            reasoning=reasoning,
        )
        self.capabilities = caps
        return caps

    async def _probe_max_model_len(self) -> int | None:
        try:
            response = await self._client.get(
                f"{self.base_url}/models", headers={"Accept": "application/json"}
            )
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        entries = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            return None
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("id") != self.model:
                continue
            value = entry.get("max_model_len")
            if isinstance(value, int):
                return value
        return None

    async def _route_exists(self, path: str, body: dict[str, Any]) -> bool:
        """True only if the route is proxied *and* answers usefully.

        A gateway returns 404 for anything it does not forward, and FastAPI
        answers unrecognised ``/v1`` routes with ``{"detail": "Not Found"}``.
        Both mean "unavailable" here, and neither is reported as a failure.
        """
        try:
            response = await self._post_json(path, body)
        except APIError as exc:
            return exc.status not in (404, 405)
        except TransportError:
            return False
        return response.is_success

    async def _probe_json_mode(self) -> bool:
        turn = await self._ask(
            [Message.user('Reply with exactly this JSON object and nothing else: {"ok": true}')],
            GenOptions(json_mode=True, max_tokens=64, temperature=0.0),
        )
        if turn is None or not turn.message.content:
            return False
        try:
            json.loads(turn.message.content)
        except json.JSONDecodeError:
            return False
        return True

    async def _probe_tools(self) -> tuple[bool, bool]:
        spec = ToolSpec(
            name="echo",
            description="Echo the given text back.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
        turn = await self._ask(
            [
                Message.user(
                    "Call the echo tool twice in parallel: once with text 'alpha' "
                    "and once with text 'beta'. Do not answer in prose."
                )
            ],
            GenOptions(max_tokens=256, temperature=0.0),
            tools=[spec],
        )
        if turn is None:
            return (False, False)
        named = [c for c in turn.message.tool_calls if c.name]
        return (bool(named), len(named) > 1)

    async def _probe_reasoning(self) -> bool:
        turn = await self._ask(
            [Message.user("What is 17 times 23? Think it through, then give the number.")],
            GenOptions(max_tokens=128, temperature=0.0),
        )
        return turn is not None and bool(turn.message.reasoning)

    async def _ask(
        self,
        messages: Sequence[Message],
        options: GenOptions,
        tools: Sequence[ToolSpec] | None = None,
    ) -> AssistantTurn | None:
        """Run one request for probing purposes; swallow failures.

        A probe that cannot run reports "not supported", never "supported" and
        never an exception -- the registry must still produce a usable profile.
        """
        try:
            return await self.complete(messages, tools, options)
        except (APIError, TransportError, MalformedResponseError) as exc:
            logger.debug("probe request failed for profile %s: %s", self.name, exc)
            return None

    def __repr__(self) -> str:
        return (
            f"OpenAICompatProvider(name={self.name!r}, model={self.model!r}, "
            f"base_url={self.base_url!r})"
        )


def _json_or_raise(response: httpx.Response, what: str) -> Any:
    try:
        return response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise MalformedResponseError(f"{what} returned non-JSON", body=response.text) from exc


def _model_ids(payload: Any) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return ()
    data = payload.get("data")
    if not isinstance(data, list):
        return ()
    return tuple(
        entry["id"]
        for entry in data
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    )


__all__ = [
    "OpenAICompatProvider",
    "build_request",
]
