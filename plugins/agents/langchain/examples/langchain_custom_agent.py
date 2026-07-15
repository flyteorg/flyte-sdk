"""Build your own LangChain agent — durable tools on Flyte.

This example shows the "bring your own agent" path: you define a LangChain
``AgentExecutor`` (or ``create_tool_calling_agent``) using the framework's
native SDK, wrap your Flyte tasks as durable tools with ``tool()``, and pass
the agent to ``run_agent(agent=...)``.

- ``tool()`` wraps Flyte ``@env.task`` functions as durable LangChain tools
  that execute as child actions (own container/GPU, retries, caching).
- ``run_agent(agent=...)`` drives the agent loop durably on Flyte.
- The agent definition is fully yours — custom prompts, models, memory, etc.

Run:  flyte run langchain_custom_agent.py city_agent --city "Paris"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.agents.langchain import run_agent, tool

env = flyte.TaskEnvironment(
    "langchain-custom-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="langchain-custom-agent")
        .clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent / "dist",
                package_name="flyteplugins-agents-core",
                pre=True,
            ),
        )
        .clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent / "dist",
                package_name="flyteplugins-agents-langchain",
                pre=True,
            ),
        )
    ),
)


# ── Step 1: Define your durable Flyte tasks ──────────────────────────────────
# These are normal Flyte tasks — they can be cached, retried, sized independently.


@env.task(cache="auto", retries=3)
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 22°C."


@env.task(cache="auto", retries=3)
async def get_population(city: str) -> int:
    """Get the population of a city."""
    return {"San Francisco": 808988, "Paris": 2102650, "Tokyo": 13929286}.get(city, 1_000_000)


# ── Step 2: Define your own LangChain agent ──────────────────────────────────
# You control the agent definition — prompt, model, tools, memory, etc.
# Flyte provides the durable runtime via ``tool()`` and ``run_agent(agent=...)``.


def _build_city_agent():
    """Build a LangChain AgentExecutor with durable tools."""
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_openai import ChatOpenAI

    # Wrap Flyte tasks as durable LangChain tools
    weather_tool = tool(get_weather, name="get_weather", description="Get the current weather for a city.")
    population_tool = tool(get_population, name="get_population", description="Get the population of a city.")

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful city-facts assistant. Use the available tools to answer "
                "questions about cities. Be concise and accurate.",
            ),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # Create the tool-calling agent
    agent_creator = create_tool_calling_agent(
        ChatOpenAI(model="gpt-4o"),
        [weather_tool, population_tool],
        prompt,
    )

    # Wrap in AgentExecutor
    return AgentExecutor(
        agent=agent_creator,
        tools=[weather_tool, population_tool],
        name="city-facts-agent",
        handle_parsing_errors=True,
        max_iterations=10,
    )


# Build the agent once (in practice, cache this in module scope)
city_agent = _build_city_agent()


# ── Step 3: Run your agent durably inside a Flyte task ───────────────────────
# The agent definition is fully yours; run_agent drives the loop durably.


@env.task(report=True, retries=3)
async def city_agent_task(city: str) -> str:
    """Run the custom-built agent durably on Flyte."""
    result = await run_agent(
        f"What's the weather in {city}?",
        agent=city_agent,
    )
    return result


# ── Alternative: Pass tools directly to run_agent ────────────────────────────
# If you don't need to separate agent definition from execution:


@env.task(report=True, retries=3)
async def quick_city_agent(city: str) -> str:
    """Build and run in one call using run_agent's builder."""
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_openai import ChatOpenAI

    weather_tool = tool(get_weather, name="get_weather", description="Get the current weather for a city.")
    population_tool = tool(get_population, name="get_population", description="Get the population of a city.")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful city-facts assistant."),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent_creator = create_tool_calling_agent(ChatOpenAI(model="gpt-4o"), [weather_tool, population_tool], prompt)
    agent = AgentExecutor(
        agent=agent_creator,
        tools=[weather_tool, population_tool],
        name="quick-agent",
        handle_parsing_errors=True,
    )

    return await run_agent(
        f"What's the weather in {city}?",
        agent=agent,
    )


# ── Alternative: Pre-build with a custom model ───────────────────────────────
# You can inject your own LangChain model and customize the agent further.


@env.task(report=True, retries=3)
async def custom_city_agent(city: str) -> str:
    """Customize the agent with a specific model before running."""
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_openai import ChatOpenAI

    weather_tool = tool(get_weather, name="get_weather", description="Get the current weather for a city.")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a weather expert. Always include temperature and conditions.",
            ),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent_creator = create_tool_calling_agent(ChatOpenAI(model="gpt-4o-mini"), [weather_tool], prompt)
    agent = AgentExecutor(
        agent=agent_creator,
        tools=[weather_tool],
        name="custom-model-agent",
        handle_parsing_errors=True,
    )

    # You can access and modify the underlying agent here if needed
    # For example, add memory, callbacks, or custom chains

    result = await run_agent(
        f"What's the weather in {city}?",
        agent=agent,
    )
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(city_agent_task, city="Paris")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")
