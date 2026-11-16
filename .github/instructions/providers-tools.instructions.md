---
applyTo: "src/bevel/providers/**/*.py,src/bevel/core/tools/**/*.py,src/bevel/core/agent.py"
---

# Providers, tools, and the loop

Everything here assumes the model is a **local open-weights model on someone's own hardware**, and that it will misbehave. Design for that; code that only works when the model cooperates is the demo version of this project.

## Providers

- `providers/base.py` is the only seam. The agent loop must not learn that HTTP exists. Adding a backend should be ~100 lines against the `ModelProvider` protocol.
- OpenAI-compatible only: vLLM and llama.cpp. Do not add native Anthropic/Google SDKs.
- Endpoint profiles carry base URL, model id, context window, tool-call format quirks, and capability flags (parallel tool calls / grammars / embeddings). Capabilities are **sniffed and cached**, not assumed from the server's self-report.
- Use `/tokenize` for exact token counts when the server offers it; fall back to a documented estimate and label it as an estimate. Never present an estimate as a count.
- `/health` probes must be cheap and must have a timeout. A dead endpoint is reported, not hung on.
- Streaming is SSE from `/chat/completions`. Handle partial chunks, mid-stream errors, and a stream that ends without a finish reason.

## Tool-call repair (`providers/repair.py`)

This is the highest-credibility module in the repo. llama.cpp models routinely emit malformed `tool_calls`. Handle at minimum:

- trailing commas · single-quoted strings · JSON wrapped in markdown fences · truncated JSON · `arguments` arriving as a JSON **string** instead of an object · a `name` in the wrong field · multiple calls concatenated into one blob

Approach, in order: tolerant extraction → schema-guided validation against the tool's JSON Schema → a re-ask loop that shows the model its own malformed output and the specific violation → request a **GBNF grammar** from llama.cpp's `/completion` to force valid JSON where the endpoint supports grammars.

- The re-ask loop is bounded. After N failures, fail the tool call with a clear message; never spin.
- Repair is observable: log what was wrong and what fixed it, so per-endpoint quality is measurable and evals can regress on it.
- Never guess a value the model didn't provide. Repairing syntax is fine; inventing an argument is fabrication.

## Tools

- One file per tool. Each declares `name`, a JSON Schema for its arguments, a risk tier, `is_parallel_safe`, and `async execute(ctx)`.
- Tools receive a context object; they do not reach for globals, and they never read config from the environment directly.
- Every tool validates its own arguments against its schema *before* executing, and returns a structured result (ok / error + content) rather than raising into the loop.
- Shell: `create_subprocess_exec` with explicit argv, **never `shell=True`**, always a timeout, always a process-tree kill on cancel or timeout.
- Edits apply a real diff and must fail loudly if context doesn't match. A silently misapplied edit is worse than a refused one.
- Anything that mutates the filesystem is preceded by a checkpoint.

## The loop

- Bounded by a token budget and a max-iteration ceiling. Both are surfaced in the TUI, and hitting either is a reported outcome, not a silent stop.
- Parallelise only `is_parallel_safe` tools. Writes, shell, and git stay serial.
- Cancellation propagates to the in-flight request and the subprocess tree, and leaves the transcript in a consistent state.
- Compaction triggers are logged and visible. A run that compacted must be distinguishable from one that didn't.
