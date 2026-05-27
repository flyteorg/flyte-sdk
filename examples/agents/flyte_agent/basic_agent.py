"""Basic Agent example — calculator + weather lookup.

A minimal end-to-end agent that demonstrates the core surface area:

- declare a few tools (plain async functions),
- spin up a :class:`flyte.ai.agents.Agent`,
- call ``agent.run(message)`` and print the response.

Run locally::

    export ANTHROPIC_API_KEY=sk-...
    uv run python examples/agents/flyte_agent/basic_agent.py "What's 17 * 23 plus the temperature in NYC?"
"""

from __future__ import annotations

import asyncio
import sys

from flyte.ai.agents import Agent


async def add(x: float, y: float) -> float:
    """Add two numbers and return their sum."""
    return x + y


async def multiply(x: float, y: float) -> float:
    """Multiply two numbers and return their product."""
    return x * y


async def get_weather(city: str) -> dict[str, str | float]:
    """Return a synthetic weather snapshot for *city*.

    In a real agent, replace this stub with a call to a weather API (and
    promote it to a ``@env.task`` for durable, retryable execution).
    """
    fake = {
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


async def main(message: str) -> None:
    result = await agent.run(message)
    if result.error:
        print(f"[error] {result.error}")
        sys.exit(1)
    print(result.summary)


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "What's 17 * 23 plus the temperature in NYC?"
    asyncio.run(main(prompt))
