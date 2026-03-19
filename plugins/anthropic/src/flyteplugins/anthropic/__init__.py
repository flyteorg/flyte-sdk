"""Anthropic Claude plugin for Flyte.

This plugin provides integration between Flyte tasks and Anthropic's Claude API,
enabling you to use Flyte tasks as tools for Claude agents. Tool calls run with
full Flyte observability, retries, and caching.

Key features:

- Use any Flyte task as a Claude tool via `function_tool`
- Full agent loop with automatic tool dispatch via `run_agent`
- Configurable agent via `Agent` (model, system prompt, tools, iteration limits)

Basic usage example:

```python
import flyte
from flyteplugins.anthropic import Agent, function_tool, run_agent

env = flyte.TaskEnvironment(
    name="agent_env",
    image=flyte.Image.from_debian_base(name="agent").with_pip_packages(
        "flyteplugins-anthropic"
    ),
)

@env.task
async def get_weather(city: str) -> str:
    '''Get the current weather for a city.'''
    return f"Weather in {city}: sunny, 22°C"

weather_tool = function_tool(get_weather)

@env.task
async def run_weather_agent(question: str) -> str:
    return await run_agent(
        prompt=question,
        tools=[weather_tool],
        model="claude-sonnet-4-20250514",
    )
```
"""

from .agents import Agent, function_tool, run_agent

__all__ = ["Agent", "function_tool", "run_agent"]
