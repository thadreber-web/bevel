# The agent loop

`core/agent.py` runs *orient → plan → act → observe* under explicit budgets, and is
the piece every later milestone sits on. Policy (`core/policy/`) now decides every
tool call, and checkpoints (`core/checkpoint/`) snapshot the workspace before the
first write. Context management (`core/context/`) still attaches at the one seam
named below. The loop stays agnostic of all three: it asks a `PolicyEvaluator` and
records the verdict, without knowing what an engine is.

## Shape

An `Agent` is built around a `Provider` and a `ToolRegistry` and is **single-use** —
`run()` or `stream()` once, then discard. `run()` drains `stream()` and returns an
`AgentResult`: final text, iteration count, every `ToolExecution`, token usage, and
the `StopReason`.

Each iteration assembles the conversation, streams one completion, executes the tool
calls it asked for, and appends one `role="tool"` message per call. The loop ends when
the model answers without asking for tools, or when a budget, a cancellation, or a
provider failure ends it.

`AgentEvent` is the one stream that carries everything the TUI (`tui/app.py`) needs: text
deltas, tool activity, compaction, budget warnings, unbuilt seams, and the terminal
event. Consumers must assume **any** event can carry **no** payload — a delta with no
text, a turn with no usage. The live tests assert that.

## Two cancellation windows

`cancel()` is not one thing, and the difference matters because it is the difference
between a tool that never ran and a tool that ran and was abandoned:

- **Between turns** — after a completion is assembled, `_iterate` checks the flag
  *before* scheduling anything. The tool calls the model asked for are never started,
  so no `TOOL_COMPLETE` event is emitted and the transcript has no tool messages for
  them. The result is `StopReason.CANCELLED`.
- **Mid-flight** — with calls in progress, the tasks are cancelled and each becomes a
  `ToolOutcome.CANCELLED` execution, so the caller can report exactly what was
  abandoned. Cancelling the *task* rather than calling `cancel()` is a hard cancel and
  propagates `CancelledError` instead.

A test that cancels at the first window and asserts the second window's outcome can
never pass. Both are covered separately in
`tests/integration/test_live_agent.py`.

## Budgets

`Budgets` caps iterations, tool calls, and total tokens. Iteration and tool-call
budgets are enforced before the work starts. The token budget is checked *between*
iterations, so a run can overshoot by one completion — documented rather than
papered over, and asserted in the tests.

## Policy

Every repaired tool call goes through `self._policy`, a `PolicyEvaluator` callable.
The default is the real `PolicyEngine` from `core/policy/`: it classifies the call
against its declared `RiskTier`, matches the argv against the built-in safe and
destructive rule tables and the learned allowlist in `policy.db`, consults an
`Approver` for anything that needs a human, and logs the decision together with the
rule that produced it. `allow_all_policy` remains exported as a caller's explicit
opt-out, not the default.

## Unbuilt seams

`UNBUILT_SEAMS` is the list of capabilities the loop would use but does not have yet.
Each is announced as an `AgentEvent` on every run and recorded in the result, and the
loop never fakes the behaviour it stands in for:

| Seam | Arrives with |
|---|---|
| Context management and compaction | m08 |

A `CHECKPOINT` event is emitted once per run, before the first batch that writes to the
workspace. Its text is the truth of what happened: either the session's checkpoint
commit, or `no checkpoint taken: <reason>` when the workspace is not a git repository,
git is missing, or the commit failed.

## Testing against the real endpoint

`tests/integration/test_live_agent.py` runs the loop against a real vLLM server. It
asserts **mechanics and consequences, never prose** — that a write changed the file on
disk, that every call the model asked for got exactly one tool message back, that
usage came from the server rather than being estimated.

Two things learned the hard way, worth knowing before writing a test here:

- **Temperature 0 is not reproducible** under continuous batching. The same prompt can
  answer with tool calls one moment and nothing the next. Tests that need the model to
  *volunteer* a behaviour re-ask up to `VOLUNTEER_ATTEMPTS` times; what is asserted
  after the call is never weakened to accommodate the model.
- **The model is a reasoning model.** It streams a `reasoning` channel alongside
  `content`, and the loop must not treat reasoning text as the answer.

Run it with the endpoint configured — the suite skips with a reason rather than
passing vacuously when it is not:

```powershell
$env:BEVEL_BASE_URL = "https://your-host/v1"
$env:BEVEL_MODEL = "the-served-model"
$env:VLLM_API_KEY = "..."
uv run pytest -m integration
```
