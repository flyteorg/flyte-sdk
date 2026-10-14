"""Basic Agent example — calculator + weather lookup.

A minimal end-to-end agent that demonstrates the core surface area:

- declare a few tools (plain async functions),
- spin up a :class:`flyte.ai.agents.Agent`,
- call ``agent.run(message)`` synchronously and print the response.

Inside ``async def`` (Flyte tasks, web handlers, etc.) use ``await agent.run.aio(...)``.

Run locally::

    uv pip install litellm
    export ANTHROPIC_API_KEY=sk-...
    uv run python examples/agents/flyte_agent/basic_agent.py "What's 17 * 23 plus the temperature in NYC?"
"""

from __future__ import annotations

import sys

from flyte.ai.agents import Agent


async def add(x: float, y: float) -> float:
    """Add two numbers and return their sum."""
    print(f"Adding {x} and {y}")
    return x + y


async def multiply(x: float, y: float) -> float:
    """Multiply two numbers and return their product."""
    print(f"Multiplying {x} and {y}")
    return x * y


async def get_weather(city: str) -> dict[str, str | float]:
    """Return a synthetic weather snapshot for *city*.

    In a real agent, replace this stub with a call to a weather API (and
    promote it to a ``@env.task`` for durable, retryable execution).
    """
    print(f"Getting weather for {city}")
    fake: dict[str, dict[str, str | float]] = {
        "new york": {"temperature_f": 68.4, "conditions": "partly cloudy"},
        "san francisco": {"temperature_f": 61.0, "conditions": "foggy"},
        "tokyo": {"temperature_f": 74.2, "conditions": "sunny"},
    }
    return fake.get(city.lower(), {"temperature_f": 70.0, "conditions": "clear"})


agent = Agent(
    name="basic-helper",
    instructions=(
        "You are a friendly assistant. Use the available tools to look up "
        "weather and compute math. Reply with a single sentence summary."
    ),
    model="claude-haiku-4-5",
    tools=[add, multiply, get_weather],
    max_turns=6,
)


def main(message: str) -> None:
    result = agent.run(message)
    if result.error:
        print(f"[error] {result.error}")
        sys.exit(1)
    print(result.summary)


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "What's 17 * 23 plus the temperature in NYC?"
    main(prompt)
