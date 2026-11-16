# Live endpoint ground truth

Findings from probing a real vLLM server (2026-11-15). These are the facts the provider
layer's capability sniffing, token accounting, and streaming decoder are written against.
If you're adding a backend, probe your own server the same way and add a section here —
assumptions about OpenAI-compatible servers are wrong often enough that sniffing is
load-bearing, not defensive padding.

Base: `https://<your-vllm-host>/v1` · auth `Authorization: Bearer $VLLM_API_KEY`
Self-signed cert (curl needs `-k`; bevel needs an explicit config opt-out with a loud
warning, never a silently-disabled verification).
A **gateway sits in front** of this deployment: non-`/v1` paths return plain `Not found`,
`/v1/*` returns FastAPI `{"detail":"Not Found"}`. Do not treat the 404 shapes as vLLM-native.

## Present

- `GET /v1/models` -> models with `max_model_len: 131072`, plus a short alias per model.
- `POST /v1/chat/completions` — non-stream 200 in ~0.55s, stream works.

## Absent (all 404 — capability sniffing must discover this, not assume it)

`/tokenize` · `/detokenize` · `/health` · `/metrics` · `/v1/embeddings` · `/v1/version`
· `/v1/models/<id>`

## Response shape surprises

1. **Reasoning model.** Non-stream: `message.content` is `null` while `message.reasoning`
   holds the chain of thought; `finish_reason: "length"` when cut off mid-thought.
   Stream: `delta.reasoning` arrives in its own chunks, `delta.content` starts later.
   `usage.completion_tokens_details.reasoning_tokens` is populated.
   -> reasoning is a first-class channel, never silently dropped, never merged into content.
2. **Usage chunk has `choices: []`.** With `stream_options: {"include_usage": true}` the
   final data chunk carries `usage` and an empty `choices` array. Indexing `choices[0]`
   unconditionally is a crash.
3. `finish_reason` observed: `stop`, `length`, `tool_calls`.
4. Extra fields to tolerate: `routed_experts`, `token_ids`, `stop_reason`, `service_tier`,
   `refusal`, `annotations`, `audio`, `function_call`, `prompt_token_ids`, `prompt_text`,
   `kv_transfer_params`, `ec_transfer_params`, `metrics`, `logprobs`.

## Tool calling (verified working)

- `finish_reason: "tool_calls"`, `message.tool_calls[]` with `id`, `type: "function"`,
  `function.name`, `function.arguments` as a **JSON string**.
- **Parallel tool calls work** — asked for two, got two in one response.
- Streamed fragments: first chunk for an index carries `id`/`type`/`function.name`;
  later chunks carry only `{"index":N,"function":{"arguments":"<slice>"}}`.
  `arguments` must be concatenated per `index`, not per chunk.
  Last content chunk has `delta: {}` with `finish_reason: "tool_calls"`.

## Consequences for the provider layer

- No `/tokenize` anywhere -> `count_tokens` reports **unknown**, never a chars/4 guess.
  Exact counts *are* available retrospectively from `usage.prompt_tokens`, which is
  server-reported and therefore honest; budget checks use that, so enforcement is
  reactive-and-exact rather than predictive-and-fake.
- TLS verification cannot be silently disabled. Needs an explicit config opt-out with a
  loud warning, otherwise the portfolio piece ships a MITM hole.
- Capability sniffing is load-bearing: `/health` is missing, so liveness = a cheap
  `GET /v1/models` with a timeout.
