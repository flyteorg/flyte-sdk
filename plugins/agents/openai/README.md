# flyteplugins-agents-openai

Run [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) agents on
Flyte. You keep writing agents in the SDK's own idioms; Flyte is the durable
orchestration runtime underneath — replay, automatic retries / self-healing,
per-tool containerized execution (CPU/GPU, caching), and observability.

```bash
pip install flyteplugins-agents-openai
```

```python
import flyte
from flyteplugins.agents.openai import tool, run_agent

env = flyte.TaskEnvironment(
    "openai-agent",
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
)

# A tool that is also a durable, cached Flyte task.
@tool
@env.task(cache="auto", retries=3)
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 22°C."

# The durable parent. retries=3 -> self-healing; report=True -> agent timeline.
@env.task(report=True, retries=3)
async def city_agent(question: str) -> str:
    return await run_agent.aio(
        question,
        tools=[get_weather],
        instructions="You are a concise assistant. Use the tools to answer.",
        model="gpt-4.1",
    )
```

## How it maps to Flyte

- The SDK owns the loop — we don't reimplement it. `run_agent` drives the
  OpenAI Agents `Runner`; the agent run is your `@env.task` (the durable
  parent) and the SDK runs its agent loop inside it.
- Tools as durable child actions. `tool` wraps an `@env.task` so that
  when the agent calls a tool, the task runs as a durable Flyte child action
  (its own container/resources, retries, caching) — not inline in the agent
  process. The SDK derives the tool's JSON schema, name and description from the
  task signature, so strict tool-calling works unchanged.
- Durable, replayable model turns. Each model turn is recorded via
  `flyte.trace` by tracing the seam below the loop — a `FlyteModelProvider`
  set on `RunConfig.model_provider`. If the task crashes and Flyte retries it,
  completed turns and tool calls replay from their recorded outputs instead of
  re-calling (and re-billing) the model.
- Self-healing. `retries=...` on the agent task plus per-turn / per-tool replay
  means transient failures recover automatically without redoing completed work.
- Observability. The OpenAI Agents trace (turns, tool calls, handoffs, token
  usage) renders into the task report (`report=True`); `install_flyte_tracing()`
  replaces the OpenAI exporter (`exclusive=True`) so nothing is uploaded
  externally.

The API key is read from the environment. Wire it as a Flyte secret.

### Bring your own agent

Already wrote an `agents.Agent` with handoffs and guardrails? Pass it through:

```python
from agents import Agent

triage = Agent(name="triage", handoffs=[...], input_guardrails=[...])

@env.task(report=True, retries=3)
async def run(goal: str) -> str:
    return await run_agent.aio(goal, agent=triage)
```

### Power-user building blocks

`run_agent` wires three independently usable pieces; reach for them directly when
driving `Runner.run` yourself:

- `tool` — turn a Flyte task into an OpenAI Agents tool.
- `FlyteModelProvider` — set on `RunConfig.model_provider` to make model turns
  durable (the seam below the loop).
- `install_flyte_tracing()` / `FlyteTracingProcessor` — render the trace into the
  report (`exclusive=True` by default replaces the OpenAI exporter so nothing is
  uploaded externally).

## Memory

Pass `memory_key` (a user/thread id) for cross-run memory — the agent continues
the same conversation across separate runs, workers and restarts:

```python
await run_agent.aio(message, model="gpt-4.1", memory_key="user-alice")
```

It backs the OpenAI Agents SDK `Session` with a durable, keyed `MemoryStore` (object
storage), so unlike the SDK's default local-SQLite session, it persists on a
distributed backend. The same store also holds path-addressed facts for long-term
`remember` / `recall` memory.

## Examples

See [`examples/`](examples/):

- [`openai_durable_agent.py`](examples/openai_durable_agent.py) — a single durable
  agent: tools as Flyte tasks, traced model turns, agent timeline in the report.
- [`openai_multi_agent.py`](examples/openai_multi_agent.py) — multi-agent
  orchestration: a planner agent decomposes a topic, researcher agents fan out
  in parallel, an editor agent synthesizes — each agent its own durable action.
- [`openai_handoffs.py`](examples/openai_handoffs.py) — handoffs + HITL: a
  triage agent hands off to billing / technical specialists inside one
  `Runner.run`; durability spans the handoff (a mid-chain crash replays both
  agents' turns), a sensitive `issue_refund` tool is gated on a human-approval
  form, and a diagnostic tool runs in a higher-CPU environment.
- [`openai_crash_resume.py`](examples/openai_crash_resume.py) — crash &
  resume: the task crashes on its first attempt after doing real work; on retry
  the completed model turns replay from their `flyte.trace` records and the tool
  calls are cache hits, so it finishes without re-calling the model. Run on a
  backend to see the replay.
- [`openai_memory.py`](examples/openai_memory.py) — cross-run memory: two
  separate runs share a `memory_key`; the agent learns a fact in run 1 and recalls
  it in run 2.

## Notes

- Streamed runs (`Runner.run_streamed`) are not memoized per-turn in this
  version; tool calls remain durable regardless.

## Conformance

This adapter passes the shared `flyteplugins.agents.core.testing.assert_adapter_conforms`
check, so it follows the common format (`tool` + `run_agent`, tool tasks wired
to the resolver), shared with the Claude and Mistral adapters.
