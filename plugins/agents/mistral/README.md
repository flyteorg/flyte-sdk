# flyteplugins-agents-mistral

Run [Mistral Agents](https://docs.mistral.ai/agents/agents_introduction/) (the
Conversations API, `mistralai` 2.x) on Flyte. You keep writing Mistral agents;
Flyte is the runtime underneath.

```bash
pip install flyteplugins-agents-mistral
```

```python
import flyte
from flyteplugins.agents.mistral import function_tool, run_agent

env = flyte.TaskEnvironment(
    "mistral-agent",
    secrets=[flyte.Secret(key="mistral_api_key", as_env_var="MISTRAL_API_KEY")],
)

@function_tool
@env.task(cache="auto", retries=3)
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 22°C."

@env.task(report=True, retries=3)
async def city_agent(question: str) -> str:
    return await run_agent(question, tools=[get_weather], model="mistral-large-latest")
```

## How it maps to Flyte

- **The SDK owns the loop — we don't reimplement it.** Mistral's own runner
  (`conversations.run_async` + `RunContext`) drives the agent loop and executes
  the tools; we just register Flyte-task-backed tools with it. `function_tool`
  produces the Python function the runner calls, and its body dispatches to
  `task.aio()`, so each tool call is a **durable Flyte child action**.
- **Both per-turn and per-tool durability.** The runner makes each model turn by
  calling `start_async`/`append_async` (in-process HTTP), so with `durable=True`
  we wrap those two methods — the **seam below the loop** — and record each turn
  via `flyte.trace` (the `ConversationResponse` round-trips through pydantic JSON).
  On a crash/retry, completed turns replay and completed tool calls are cache
  hits — all while the SDK still owns the loop. (Same idea as swapping OpenAI's
  `ModelProvider`: trace the model-call seam, not the loop.)
- **Observability**: the turns, tool calls and final answer render into the task
  report.

The API key is read from the environment (never an argument), so it can't leak
into task inputs — wire it as a Flyte secret.

## Memory

Pass `memory_key` (a user/thread id) for **cross-run memory** — the agent continues
the same conversation across separate runs:

```python
await run_agent(message, model="mistral-large-latest", memory_key="user-alice")
```

Mistral keeps the transcript server-side, so Flyte durably persists the thread's
`conversation_id` (in a keyed `MemoryStore`) and continues that conversation when the
key recurs.

## Examples

See [`examples/`](examples/):

- [`mistral_durable_agent.py`](examples/mistral_durable_agent.py) — a single durable
  agent: tools as Flyte tasks, per-turn traced conversation, agent timeline in the
  report.
- [`mistral_crash_resume.py`](examples/mistral_crash_resume.py) — **crash & resume**:
  the task crashes on its first attempt after doing real work; on retry the completed
  conversation turns replay from their `flyte.trace` records and the tool calls are
  cache hits. Run on a backend to see the replay.
- [`mistral_multi_agent.py`](examples/mistral_multi_agent.py) — **multi-agent
  orchestration**: a planner agent decomposes a topic, researcher agents fan out in
  parallel, an editor agent synthesizes — each agent its own durable action.
- [`mistral_agent_id.py`](examples/mistral_agent_id.py) — drive a **pre-created
  server-side agent** by `agent_id` (instead of an inline model) while its tool calls
  still run as durable Flyte actions.
- [`mistral_memory.py`](examples/mistral_memory.py) — **cross-run memory**: two
  separate runs share a `memory_key`; the agent learns a fact in run 1 and recalls
  it in run 2.

## Conformance

This adapter passes the shared `flyteplugins.agents.core.testing.assert_adapter_conforms`
check — the same one every adapter runs — so it follows the common format despite
a server-side, conversation-based SDK shape very different from OpenAI's or
Claude's.
