# flyteplugins-agents-google

Run [Google ADK](https://github.com/google/adk-python) (Agent Development Kit)
agents on Flyte. You keep writing ADK agents; Flyte is the runtime underneath.

```bash
pip install flyteplugins-agents-google
```

```python
import flyte
from flyteplugins.agents.google import function_tool, run_agent

env = flyte.TaskEnvironment(
    "google-agent",
    secrets=[flyte.Secret(key="google_api_key", as_env_var="GOOGLE_API_KEY")],
)

@function_tool
@env.task(cache="auto", retries=3)
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 22°C."

@env.task(report=True, retries=3)
async def city_agent(question: str) -> str:
    return await run_agent(question, tools=[get_weather], model="gemini-2.0-flash")
```

## How it maps to Flyte

- **The SDK owns the loop — we don't reimplement it.** ADK's `Runner` drives the
  agent loop (model + tools, yielding `Event`s); `run_agent` builds an `LlmAgent`,
  runs `Runner.run_async` inside your `@env.task`, and returns the final answer.
- **Tools as durable child actions.** `function_tool` wraps an `@env.task` as the
  Python function ADK calls; its body dispatches to `task.aio()`, so each tool call
  runs as a **durable Flyte child action**. ADK derives the tool declaration from the
  task signature.
- **Durable, replayable model turns.** With `durable=True`, the agent's model is
  wrapped (`FlyteLlm`) so each turn through `BaseLlm.generate_content_async` — the
  **seam below the loop** — is recorded via `flyte.trace`. On a crash/retry, completed
  turns replay from their recorded `LlmResponse` and tools are cache hits. (Same idea
  as swapping OpenAI's `ModelProvider`: trace the model-call seam, not the loop.)
- **Observability**: the turns and tool calls render into the task report.

The API key is read from the environment (e.g. `GOOGLE_API_KEY` for Gemini, or your
Vertex AI config), so it can't leak into task inputs — wire it as a Flyte secret.

## Memory

Pass `memory_key` (a user/thread id) for **cross-run memory** — the agent continues
the same conversation across separate runs:

```python
await run_agent(message, model="gemini-2.0-flash", memory_key="user-alice")
```

ADK keeps the conversation as a list of `Event`s on the session; we persist those to a
durable, keyed `MemoryStore` and restore them into a fresh session on the next run
with the same key.

## Examples

See [`examples/`](examples/):

- [`google_durable_agent.py`](examples/google_durable_agent.py) — a single durable
  agent: tools as Flyte tasks, traced model turns, agent timeline in the report.
- [`google_multi_agent.py`](examples/google_multi_agent.py) — **multi-agent
  orchestration**: a planner agent decomposes a topic, researcher agents fan out in
  parallel, an editor agent synthesizes — each agent its own durable action.
- [`google_crash_resume.py`](examples/google_crash_resume.py) — **crash & resume**: the
  task crashes on its first attempt; on retry the completed model turns replay from
  their `flyte.trace` records and the tool calls are cache hits. Run on a backend.
- [`google_memory.py`](examples/google_memory.py) — **cross-run memory**: two separate
  runs share a `memory_key`; the agent learns a fact in run 1 and recalls it in run 2.

## Conformance

This adapter passes the shared `flyteplugins.agents.core.testing.assert_adapter_conforms`
check — the same one every adapter runs — so it follows the common format
(`function_tool` + `run_agent`, tool tasks wired to the resolver), shared with the
OpenAI, Claude and Mistral adapters.
