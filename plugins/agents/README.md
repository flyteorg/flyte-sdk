# Flyte Agents Plugin (`flyteplugins-agents`)

Run agents from the agent SDK of your choice on Flyte. You keep writing agents in
your framework's own idioms; Flyte becomes the **durable orchestration runtime**
underneath — durability and replay, automatic retries / self-healing, per-tool
containerized execution (CPU/GPU, caching), and observability.

Each framework is a separate, lazily-imported adapter behind its own optional
extra, so installing one never pulls the others.

| Adapter                      | Extra                         | Underlying SDK                      |
| ---------------------------- | ----------------------------- | ----------------------------------- |
| `flyteplugins.agents.openai` | `flyteplugins-agents[openai]` | OpenAI Agents SDK (`openai-agents`) |

More framework adapters (Claude Agent SDK, Mistral Agents, Strands, LangGraph)
follow the same contract and land under the same package.

## Install

```bash
pip install flyteplugins-agents[openai]
```

## The model

- The **agent run** is a Flyte `@env.task` — the durable parent;
- Each **model turn** is a `flyte.trace` — a memoized, replayable leaf;
- Each **tool** is a Flyte task, invoked as a durable **child action**.

## OpenAI Agents SDK

```python
import flyte
from flyteplugins.agents.openai import function_tool, run_agent

env = flyte.TaskEnvironment(
    "openai-agent",
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
)

# A tool that is also a durable, cached Flyte task.
@function_tool
@env.task(cache="auto", retries=3)
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 22°C."

# The durable parent. retries=3 -> self-healing; report=True -> agent timeline.
@env.task(report=True, retries=3)
async def city_agent(question: str) -> str:
    return await run_agent(
        question,
        tools=[get_weather],
        instructions="You are a concise assistant. Use the tools to answer.",
        model="gpt-4.1",
    )
```

What you get:

- **Tools as durable actions**: when the agent calls a tool, the matching Flyte
  task runs as a child action with its own resources, retries and caching, not
  inline in the agent process.
- **Durable, replayable model turns**: each model turn is recorded via
  `flyte.trace`. If the task crashes and Flyte retries it, completed turns and
  tool calls are replayed from their recorded outputs instead of re-calling (and
  re-billing) the model.
- **Self-healing**: `retries=...` on the agent task plus per-turn / per-tool
  replay means transient failures recover automatically without redoing work.
- **Observability**: the OpenAI Agents trace (turns, tool calls, handoffs, token
  usage) is rendered into the task report (`report=True`), alongside the tool
  tasks that already appear as Flyte actions.

### Bring your own agent

Already wrote an `agents.Agent` with handoffs and guardrails? Pass it through:

```python
from agents import Agent

triage = Agent(name="triage", handoffs=[...], input_guardrails=[...])

@env.task(report=True, retries=3)
async def run(goal: str) -> str:
    return await run_agent(goal, agent=triage)
```

### Power-user building blocks

`run_agent` wires three independently usable pieces; reach for them directly when
driving `Runner.run` yourself:

- `function_tool` — turn a Flyte task into an OpenAI Agents tool.
- `FlyteModelProvider` — set on `RunConfig.model_provider` to make turns durable.
- `install_flyte_tracing()` / `FlyteTracingProcessor` — render the trace into the
  report (`exclusive=True` by default replaces the OpenAI exporter so nothing is
  uploaded externally).

## Examples

See [`examples/`](examples/):

- [`openai_durable_agent.py`](examples/openai_durable_agent.py) — a single durable
  agent: tools as Flyte tasks, traced model turns, agent timeline in the report.
- [`openai_multi_agent.py`](examples/openai_multi_agent.py) — **multi-agent
  orchestration**: a planner agent decomposes a topic, researcher agents fan out
  in parallel, an editor agent synthesizes — each agent its own durable action.
- [`openai_crash_resume.py`](examples/openai_crash_resume.py) — **crash &
  resume**: the task crashes on its first attempt after doing real work; on retry
  the completed model turns replay from their `flyte.trace` records and the tool
  calls are cache hits, so it finishes without re-calling the model. Run on a
  backend to see the replay.

## Notes

- Call `run_agent` (or `Runner.run` with these pieces) **from inside an**
  `@env.task` — that task is the durable parent. Outside a task context the
  durability/observability layers are transparent no-ops, so the same code runs
  locally unchanged.
- Streamed runs (`Runner.run_streamed`) are not memoized per-turn in this
  version; tool calls remain durable regardless.
