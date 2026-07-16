# Flyte agent-SDK adapters

Run agents written with the agent SDK of your choice on Flyte. You keep writing
agents in each framework's own idioms; Flyte is the durable orchestration runtime
underneath — replay, automatic retries / self-healing, per-tool containerized
execution (CPU/GPU, caching), cross-run memory, and observability.

Each SDK is a separate, co-located package on a shared core:

| Adapter | Package | Underlying SDK |
| --- | --- | --- |
| [`flyteplugins.agents.openai`](openai/) | `flyteplugins-agents-openai` | [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) |
| [`flyteplugins.agents.claude`](claude/) | `flyteplugins-agents-claude` | [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) |
| [`flyteplugins.agents.mistral`](mistral/) | `flyteplugins-agents-mistral` | [Mistral Agents](https://docs.mistral.ai/agents/agents_introduction/) (`mistralai` 2.x) |
| [`flyteplugins.agents.google`](google/) | `flyteplugins-agents-google` | [Google ADK](https://github.com/google/adk-python) |
| [`flyteplugins.agents.deepagents`](deepagents/) | `flyteplugins-agents-deepagents` | [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) (LangChain's agent harness) |
| [`flyteplugins.agents.langchain`](langchain/) | `flyteplugins-agents-langchain` | [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents) (`create_agent`, 1.x) |
| [`flyteplugins.agents.langgraph`](langgraph/) | `flyteplugins-agents-langgraph` | [LangGraph](https://langchain-ai.github.io/langgraph/) (bring your own `StateGraph`) |
| [`flyteplugins.agents.crewai`](crewai/) | `flyteplugins-agents-crewai` | [CrewAI](https://docs.crewai.com/) |
| [`flyteplugins.agents.pydantic_ai`](pydantic_ai/) | `flyteplugins-agents-pydantic-ai` | [Pydantic AI](https://ai.pydantic.dev/) (2.x) |
| [`flyteplugins.agents.hermes`](hermes/) | `flyteplugins-agents-hermes` | [Hermes Agent](https://pypi.org/project/hermes-agent/) (Nous Research; Python ≥3.11) |

```bash
pip install flyteplugins-agents-openai   # or -claude / -mistral / -google / -deepagents / -langchain / -langgraph / -crewai / -pydantic-ai
```

Each adapter has its own README (linked above) with the SDK-specific details.

## The shared model

Every adapter follows the same division of labor, so the call shape is identical
across SDKs:

- The agent run is a Flyte `@env.task` — the durable parent (`retries=` =
  self-healing, `report=True` = the agent timeline in the report).
- Each tool is a Flyte task, invoked as a durable child action (its own
  container/resources, retries, caching) — via `tool` on an `@env.task`.
- Each model turn is recorded for replay by tracing the seam below the SDK's
  loop (`durable=True`), so a crashed/retried run replays completed turns instead of
  re-calling (and re-billing) the model. (Where the SDK runs its loop in a subprocess
  — Claude — durability is the SDK's own session-resume instead.)
- Cross-run memory via `memory_key` — the conversation continues across separate
  runs and workers, backed by a durable keyed store.

```python
import flyte
from flyteplugins.agents.openai import tool, run_agent  # same shape for every adapter

env = flyte.TaskEnvironment(
    "agent",
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
)

# A tool that is also a durable, cached Flyte task.
@tool
@env.task(cache="auto", retries=3)
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 22°C."

# The durable parent: the SDK's agent loop runs here, on Flyte.
@env.task(report=True, retries=3)
async def city_agent(question: str) -> str:
    return await run_agent.aio(question, tools=[get_weather], model="gpt-4.1", memory_key="user-1")
```

`run_agent` is syncified: call it synchronously as `run_agent(...)` from sync
code, or as `await run_agent.aio(...)` from async code (as above).

## Architecture

- `flyteplugins-agents-core` holds the SDK-agnostic contract every adapter builds
  on: `tool` / `ToolTaskResolver` (tools as durable actions),
  `durable_step` (the model-turn replay primitive), `resolve_memory` (cross-run
  memory over a keyed store), and `ReportTimeline` / `flush_report` (report rendering).
- Every adapter passes the same
  `flyteplugins.agents.core.testing.assert_adapter_conforms` check, so `tool`
  + `run_agent` (with `tools` / `model` / `instructions` / `durable` /
  `observability` / `memory_key`) present a uniform surface across SDKs — CI-enforced.

## Notes

- Call `run_agent` from inside an `@env.task` — that task is the durable parent.
  Outside a task context the durability / observability / memory layers are transparent
  no-ops, so the same code runs locally unchanged.
