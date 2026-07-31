# flyteplugins-agents-deepagents

Run [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) —
LangChain's agent harness with built-in planning, a virtual filesystem, and
subagents — durably on Flyte.

```bash
pip install flyteplugins-agents-deepagents
```

You keep writing Deep Agents code; Flyte is the durable runtime underneath:

- **Tools are Flyte tasks.** Stack `@tool` on `@env.task` and each tool call the
  agent (or any of its subagents) makes runs as a durable child action — its own
  container/resources, retries, and caching.
- **Model turns are replayable.** On the builder path (or by wrapping your own
  model in `DurableChatModel`), every model turn is recorded via `flyte.trace`,
  so a crashed/retried run replays completed turns instead of re-calling (and
  re-billing) the model.
- **Memory spans runs.** `run_agent(..., memory_key=...)` persists the
  conversation *and* the agent's virtual filesystem to a durable keyed store, so
  a later run with the same key picks up both.

```python
import flyte
from flyteplugins.agents.deepagents import run_agent, tool

env = flyte.TaskEnvironment("deep-agent")

@tool
@env.task(cache="auto", retries=3)
async def search_web(query: str) -> str:
    """Search the web for a query."""
    ...

@env.task(report=True, retries=3)
async def research_agent(question: str) -> str:
    return await run_agent(
        question,
        tools=[search_web],
        instructions="You are an expert researcher.",
        model="anthropic:claude-sonnet-4-6",
        subagents=[{
            "name": "critic",
            "description": "Critiques draft answers.",
            "system_prompt": "You are a ruthless critic.",
        }],
    )
```

To bring your own agent, build it with `create_deep_agent` (attaching
`@tool`-wrapped tasks natively) and pass it as `run_agent(agent=...)`; wrap the
model in `DurableChatModel(inner=...)` to keep durable model turns on that path.
See [examples/](examples/) for the full patterns.
