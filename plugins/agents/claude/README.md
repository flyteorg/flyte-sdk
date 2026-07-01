# flyteplugins-agents-claude

Run [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)
agents on Flyte. You keep writing Claude agents; Flyte is the runtime underneath.

```bash
pip install flyteplugins-agents-claude
```

```python
import flyte
from flyteplugins.agents.claude import function_tool, run_agent

env = flyte.TaskEnvironment(
    "claude-agent",
    secrets=[flyte.Secret(key="anthropic_api_key", as_env_var="ANTHROPIC_API_KEY")],
)

@function_tool
@env.task(cache="auto", retries=3)
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 22°C."

@env.task(report=True, retries=3)
async def city_agent(question: str) -> str:
    return await run_agent(question, tools=[get_weather], model="claude-sonnet-4-5")
```

## How it maps to Flyte

- Tools are in-process MCP tools (the SDK's `@tool`/`SdkMcpTool`): `function_tool`
  wraps an `@env.task` so that when Claude calls it, the task runs as a durable
  Flyte child action (its own container/resources, retries, caching). The input
  schema is derived via the Flyte type engine.
- The loop runs in the Claude Code runtime. `run_agent` runs that loop inside
  your `@env.task`, streams the messages, and renders the timeline (assistant
  turns, tool calls, cost) into the task report.

## Durability

Two layers, both real:

- Tool calls are durable Flyte child actions (own container/resources, retries,
  caching) — always, regardless of `durable`.
- The conversation survives a crash. With `durable=True`, `run_agent` wires the
  Claude SDK's own session mirror + resume onto a `flyte.Checkpoint`: a
  deterministic `session_id` (derived from the task's action, so it's stable across
  retries) is pinned on the first attempt, and on a retry the prior attempt's
  transcript is restored from the checkpoint and the run resumes instead of
  restarting.

We delegate to the SDK's resume because the model loop runs in the Claude Code
runtime (a subprocess Flyte doesn't intercept), so a model turn can't be a
`flyte.trace` leaf the way it is for other client-side SDKs. Session
resume is the coarser-grained equivalent — whole-session, not per-turn — and it
no-ops cleanly when there's no checkpoint context (e.g. local runs).

## Observability

`run_agent` renders a timeline into the task report (`report=True`): the assistant
turns from the streamed messages, plus each tool's outcome — `PostToolUse` /
`PostToolUseFailure` hooks record the result or error the message stream doesn't
surface. If you pass your own `ClaudeAgentOptions(hooks=...)`, ours are merged in,
not substituted.

## Runtime

The `claude-agent-sdk` wheel bundles the native `claude` CLI (a per-platform
binary, ~250 MB — including the `manylinux` wheel), so `pip install
flyteplugins-agents-claude` is all the runtime image needs.

## Memory

Pass `memory_key` (a user/thread id) for cross-run memory — the agent resumes the
same conversation across separate runs:

```python
await run_agent(message, model="claude-sonnet-4-5", memory_key="user-alice")
```

The transcript is persisted to a durable, keyed `MemoryStore` and resumed via the
SDK's session-mirror on the next run with the same key — which also covers
crash-resume, so it supersedes the per-run `durable` checkpoint.

## Examples

See [`examples/`](examples/):

- [`claude_durable_agent.py`](examples/claude_durable_agent.py) — a single durable
  agent: tools as Flyte tasks, tool outcomes + assistant turns in the report.
- [`claude_crash_resume.py`](examples/claude_crash_resume.py) — crash & resume:
  the task crashes on its first attempt; on retry the conversation resumes from the
  `flyte.Checkpoint`-backed session and completed tool calls are cache hits. Run on a
  backend to see resume.
- [`claude_multi_agent.py`](examples/claude_multi_agent.py) — multi-agent
  orchestration: a planner agent decomposes a topic, researcher agents fan out in
  parallel, an editor agent synthesizes — each agent its own durable action.
- [`claude_hitl.py`](examples/claude_hitl.py) — human-in-the-loop: a sensitive
  `issue_refund` tool pauses on a Flyte condition (`flyte.new_condition`) for a human
  to approve before it runs — a durable gate the agent SDK has no equivalent for.
- [`claude_memory.py`](examples/claude_memory.py) — cross-run memory: two
  separate runs share a `memory_key`; the agent learns a fact in run 1 and recalls
  it in run 2.
- [`claude_handoffs.py`](examples/claude_handoffs.py) — native subagent delegation: a
  triage prompt delegates to a billing or technical-support subagent, the whole run
  durable on Flyte.

## Conformance

This adapter passes the shared `flyteplugins.agents.core.testing.assert_adapter_conforms`
check — the same one every adapter runs — so it follows the common format
(`function_tool` + `run_agent`, tool tasks wired to the resolver) despite a very
different underlying SDK shape.
