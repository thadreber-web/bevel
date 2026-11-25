# The agent loop

`core/agent.py` runs *orient → plan → act → observe* under explicit budgets, and is
the piece every later milestone sits on. Policy (`core/policy/`) now decides every
tool call, checkpoints (`core/checkpoint/`) snapshot the workspace before the
first write, the orient step (`core/context/project_map.py`) puts the repo on the
model's first request, three levels of compaction (`core/context/compact.py`) fit an
over-long transcript back into the window — L1 truncates, L2 summarises, and L3 hands
off and continues — and the session store
(`core/session/`) writes every turn, tool call and usage row through to SQLite as
the loop goes. The loop stays
agnostic of all of them: it asks a `PolicyEvaluator` and records the verdict, without
knowing what an engine is.

## Shape

An `Agent` is built around a `Provider` and a `ToolRegistry` and is **single-use** —
`run()` or `stream()` once, then discard. `run()` drains `stream()` and returns an
`AgentResult`: final text, iteration count, every `ToolExecution`, token usage, the
`StopReason`, and the `session_id` it wrote to (or `None` with no store attached).

Each iteration assembles the conversation, streams one completion, executes the tool
calls it asked for, and appends one `role="tool"` message per call. The loop ends when
the model answers without asking for tools, or when a budget, a cancellation, a
provider failure, or a transcript that will not fit the window ends it.

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

## Orient

Before the first completion, the loop builds a project map
(`core/context/project_map.py`) and puts it in front of the model. It is one bounded
directory walk: the tree to a capped depth, the heads of the manifests and the README,
and the declared entry points. Every section is capped and every cut says it was cut
and names the tool to use instead — a map that grows with the repo would eat the
context it exists to save.

The map is built once per run, off the event loop, and never refreshed. It honours the
workspace's own `.gitignore` for names, and prunes the same junk directories the `glob`
tool prunes, so the two never disagree about what exists. Pruning is a convenience, not
a jail: anything the map hides is one `glob` call away. A workspace that cannot be read
is reported *inside* the map rather than raising — a run with no orientation still has
to answer a greeting.

The system prompt tells the model the map is its orientation and that a greeting or a
question about bevel is answered with zero tool calls. `tests/unit/test_agent_orient.py`
and the TUI test of the same shape assert both halves: the map reached the wire, and no
tool card appeared.

## Compaction

Step 8 runs after each batch of tool calls. Its trigger is the server's own
`prompt_tokens` from the last request against `Capabilities.max_model_len` — never a
chars/4 estimate, because the target endpoint has no `/tokenize` to predict with (see
`docs/provider-notes.md`). When either number is unknown the step does nothing and
says so in the `COMPACTION` event: the loop refuses to invent a window size, so it
does not guess one here either.

`core/context/compact.py` holds the three levels:

- **L1** cuts tool observations over `MIN_OBSERVATION_CHARS` to head + tail plus a
  pointer. Pure string work — no model call, no I/O, no new dependency, and the same
  transcript always compacts to the same bytes. The text it drops has a persisted
  copy: the observation row in the session store still holds what the model was
  originally shown, so the replacement points at the stored session (`saved in
  session <id>`) as well as naming the originating call and its arguments.
- **L2** replaces the middle of the transcript with one summary and leaves the last
  `KEEP_RECENT_TURNS` turns byte-identical. It costs a generation, which is why the
  loop tries L1 first and escalates only when L1 has nothing left to give. The
  summariser is `Agent(summariser=...)` if one was passed, otherwise the loop's own
  provider — there is no second endpoint in the config, so there is no second endpoint
  here. The `COMPACTION` text names the model that summarised.
- **L3** is the handoff. It replaces everything before a shorter kept tail — including
  any L2 summary, which it supersedes — with a head the loop rebuilds in the shape the
  run opened with: a system prompt carrying a *freshly built* project map, the recall
  block at index 1, then the brief, then the last `HANDOFF_KEEP_RECENT_TURNS` turns
  verbatim. It costs a generation and therefore an iteration.

L3 keeps a strictly smaller tail than L2 on purpose. If both kept four turns they
would run out at exactly the same transcript, L2 would always be exhausted the moment
L3 was reachable, and escalation could never arrive at a level that helps.

The brief is assembled from three sources, and which is which is the point. The goal
is the task string verbatim, the files touched come from executions that actually
mutated the workspace (a call whose arguments will not parse contributes no path — a
false negative, never an invention), and the todos come from `TodoStore`. Only
decisions and open questions are the model's own words, embedded verbatim under a
heading that names them as the previous instance's account. The model is never asked
to list files, so it cannot invent one.

The brief is not persisted as a turn. `rebuild_transcript` replays every stored row,
so a stored brief would be replayed alongside all the turns it replaced and a resumed
run would carry two. It is announced as a `COMPACTION` event instead.

Escalation is cheapest-first and one pass per over-budget measurement, so compaction
is not a per-iteration tax. A level that changed nothing is exhausted — all three
levels are idempotent, which is what makes "changed nothing" a reliable signal. When
all three are exhausted the run stops with `StopReason.CONTEXT_OVERFLOW` rather than
sending a request the server is known to reject; the error names the token count, the
window, and the reason the handoff declined. `MAX_HANDOFFS` caps how many a single run
may perform, because a model that never converges can otherwise hand off its way
through the whole token budget while going nowhere; hitting the cap is that same
reported overflow, not a silent stop.

L1 and L2 cut only at turn-group boundaries and never touch `messages[0]`, the system
message that carries the project map. L3 replaces it — with a freshly built one, from
the same code path the first request used, so a handed-off run is never less oriented
than a fresh one. A run that silently went unoriented after compaction would be worse
than one that ran out of context, so the tests assert the map is still there after
each level.

## Unbuilt seams

`UNBUILT_SEAMS` is the list of capabilities the loop would use but does not have yet.
Each is announced on every run and recorded in the result, and the loop never fakes
the behaviour it stands in for. **It is currently empty** — every step described in
this document happens.

The list is the mechanism, not its contents. An entry is a disclaimer the model is
told about and the TUI prints, so a stale entry is a lie about a built feature and a
missing one is a lie about an unbuilt one. Tests in `tests/unit/test_agent_seams.py`,
`tests/unit/test_agent_compact.py` and `tests/tui/test_app.py` pin that it is empty,
which is what makes quietly walking a feature back impossible.

L1, L2 and L3 are **not** in this list: they run, and their `COMPACTION` events carry
the numbers that prove it. Neither is session persistence: every turn, tool call and
usage row a run produces is written through to `sessions.db` as it goes, and
`core/session/resume.py` rebuilds a closed session from those rows.

## Sessions

`core/session/store.py` is the loop's write-through target: one `sessions` row, one
`turns` row per message, one `tool_calls` row per execution, one `usage` row per
iteration the server reported numbers for. Stdlib `sqlite3` only, and every write
commits, so a turn that completed is in the file before the loop moves on — closing
the TUI, or losing it to a crash, costs at most the turn in flight. The connection
opens on first write, so importing bevel or constructing an agent never creates a
database file.

The loop persists, it does not read. `core/session/resume.py` is the reader:
`rebuild_transcript` turns rows back into a requestable transcript and reports
`dropped`, the number of stored rows it had to discard from the tail. It has to — a
crash between "the model asked for these calls" and "the results came back" leaves an
assistant message naming calls with no results beside them, which a real server
answers with a 400. Resume trims whole call-groups rather than sending a transcript
the endpoint would reject, and says how much it trimmed rather than hiding it.

The `system` message is deliberately **not** stored. A resumed run rebuilds it, so
the project map is the current tree rather than the one from when the session was
opened; the conversation itself is what comes back from the file.

The same file also holds the session's plan, in a `todos` table — see
`core/tools/todo.py` and the Todos section of the README. `TodoStore` is a sibling of
`SessionStore` rather than a method on it, and takes its `session_id` as a mutable
attribute, because the tool object is constructed before a session row exists; the
loop assigns it in `_open_session()` on every path, so the tool's idea of the session
is always the loop's. With no session id there is nothing to write, and the tool says
so instead of storing an orphan row.

`bevel sessions` lists what is stored and `bevel resume [id]` reopens it (with no id,
the most recent session for the current directory). The database lives at
`~/.local/share/bevel/sessions.db` — one documented path on every platform, including
Windows, and shared across workspaces so that `bevel sessions` can list them all.

A `CHECKPOINT` event is emitted once per run, before the first batch that writes to the
workspace. Its text is the truth of what happened: either the session's checkpoint
commit, or `no checkpoint taken: <reason>` when the workspace is not a git repository,
git is missing, or the commit failed.

## Testing against the real endpoint

`tests/integration/test_live_agent.py` runs the loop against a real vLLM server. It
asserts **mechanics and consequences, never prose** — that a write changed the file on
disk, that every call the model asked for got exactly one tool message back, that
usage came from the server rather than being estimated.

`tests/integration/test_live_resume.py` does the same for persistence: it runs a
session, closes the store, reopens the file with a *second* `SessionStore` (a fresh
connection, so nothing can be coming from memory), and asks the model to recall a
random token only the first session contained. An answer that reproduces the token
can only have come out of the rebuilt transcript.

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
