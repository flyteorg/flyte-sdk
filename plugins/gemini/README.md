# Flyte Gemini Plugin

This plugin provides integration between Flyte and Google's Gemini API, enabling you to use Flyte tasks as tools for Gemini agents.

## Installation

```bash
pip install flyteplugins-gemini
```

## Usage

```python
import flyte
from flyteplugins.gemini import function_tool, Agent, run_agent

env = flyte.TaskEnvironment(
    "gemini-agent",
    secrets=[flyte.Secret(key="google_api_key", as_env_var="GOOGLE_API_KEY")],
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
        model="gemini-2.5-flash",
    )

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(agent_task, prompt="What's the weather in San Francisco?")
    print(run.result)
```

## Features

- Convert Flyte tasks to Gemini tool definitions automatically
- Support for both sync and async tasks
- Automatic type conversion from Python type hints to JSON schema
- Integration with Flyte's task environment for secrets and resources
