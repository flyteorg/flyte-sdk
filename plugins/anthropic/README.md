# Flyte Anthropic Plugin

This plugin provides integration between Flyte and Anthropic's Claude API, enabling you to use Flyte tasks as tools for Claude agents.

## Installation

```bash
pip install flyteplugins-anthropic
```

## Usage

```python
import flyte
from flyteplugins.anthropic import function_tool, Agent, run_agent

env = flyte.TaskEnvironment(
    "claude-agent",
    secrets=[flyte.Secret(key="anthropic_api_key", as_env_var="ANTHROPIC_API_KEY")],
)

@env.task
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 72F"

@env.task
async def agent_task(prompt: str) -> str:
    tools = [function_tool(get_weather)]
    return await run_agent(
        prompt=prompt,
        tools=tools,
        model="claude-sonnet-4-20250514",
    )

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(agent_task, prompt="What's the weather in San Francisco?")
    print(run.result)
```

## Features

- Convert Flyte tasks to Claude tool definitions automatically
- Support for both sync and async tasks
- Automatic type conversion from Python type hints to JSON schema
- Integration with Flyte's task environment for secrets and resources
