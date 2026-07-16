"""Build your own Deep Agent — durable tools on Flyte.

This example shows the "bring your own agent" path: you define a Deep Agent
using the framework's native SDK (``create_deep_agent`` with tools, subagents,
planning, and the virtual filesystem), wrap your Flyte tasks as durable tools
with ``@tool``, and pass the compiled agent to ``run_agent(agent=...)``.

- ``@tool`` stacked on ``@env.task`` makes each function both a durable, cached
  Flyte task and a LangChain ``BaseTool`` — attach it to the main agent or to a
  subagent, and every call runs as a durable child action (own container/GPU,
  retries, caching).
- ``DurableChatModel(inner=...)`` wraps the chat model so each model turn is
  recorded via ``flyte.trace`` — a crashed/retried run replays completed turns
  instead of re-calling (and re-billing) the model. Because you own the agent
  build here, you wrap the model yourself.
- ``run_agent(agent=...)`` drives the agent loop durably on Flyte.

Run:  flyte run deepagents_custom_agent.py city_agent_task --city "San Francisco"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import flyte

from flyteplugins.agents.deepagents import DurableChatModel, run_agent, tool

env = flyte.TaskEnvironment(
    "deepagents-custom-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="anthropic_api_key", as_env_var="ANTHROPIC_API_KEY")],
    image=flyte.Image.from_debian_base(name="deepagents-custom-agent").with_local_v2_plugins(
        ["flyteplugins-agents-core", "flyteplugins-agents-deepagents"]
    ),
)


# ── Step 1: Define your durable Flyte tasks and decorate them as tools ────────
# Stacking @tool on @env.task makes each one both a durable, cached Flyte task
# and a LangChain tool the deep agent (and its subagents) can call.


@tool
@env.task(cache="auto", retries=3)
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 22°C."


@tool
@env.task(cache="auto", retries=3)
async def get_population(city: str) -> int:
    """Get the population of a city."""
    return {"San Francisco": 808988, "Paris": 2102650, "Tokyo": 13929286}.get(city, 1_000_000)


# ── Step 2: Define your own Deep Agent ────────────────────────────────────────
# You control the agent definition — system prompt, model, tools, subagents,
# planning, filesystem. Flyte provides the durable runtime: the tools are
# attached natively via ``tools=[...]`` / a subagent's tool list, and the model
# is wrapped in ``DurableChatModel`` for replayable turns.


def _build_city_agent():
    """Build a Deep Agent with durable Flyte tools and a fact-checking subagent."""
    from deepagents import create_deep_agent
    from langchain.chat_models import init_chat_model

    model = DurableChatModel(inner=init_chat_model("anthropic:claude-sonnet-4-6"))

    return create_deep_agent(
        model=model,
        tools=[get_weather, get_population],
        system_prompt=(
            "You are a helpful city-facts assistant. Use the available tools to answer "
            "questions about cities. Be concise and accurate."
        ),
        # Subagents get their own context window; their tool calls are just as
        # durable, because the tools are the same Flyte tasks.
        subagents=[
            {
                "name": "fact-checker",
                "description": "Double-checks city facts before they are reported.",
                "system_prompt": "You verify city facts using the available tools. Be strict.",
                "tools": [get_weather, get_population],
            }
        ],
    )


# Build the agent once (module scope, reused across runs).
city_agent = _build_city_agent()


# ── Step 3: Run your agent durably inside a Flyte task ────────────────────────
# The agent definition is fully yours; run_agent drives the loop durably.


@env.task(report=True, retries=3)
async def city_agent_task(city: str) -> str:
    """Run the custom-built deep agent durably on Flyte."""
    return await run_agent.aio(
        f"What's the weather and population of {city}?",
        agent=city_agent,
    )


# ── Alternative: let run_agent build the agent from tools ─────────────────────
# If you don't need to hand-craft the agent, pass the durable tools (plus any
# Deep-Agents options like subagents=) to run_agent and it builds one for you —
# including the durable model wrapping.


@env.task(report=True, retries=3)
async def quick_city_agent(city: str) -> str:
    """Build and run in one call using run_agent's builder."""
    return await run_agent.aio(
        f"What's the weather and population of {city}?",
        tools=[get_weather, get_population],
        instructions="You are a concise city-facts assistant. Use the tools to answer.",
        model="anthropic:claude-sonnet-4-6",
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(city_agent_task, city="San Francisco")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")
