"""OpenAI Agents with Flyte, basic tool example.

Usage:

Create secret:

```
flyte secret create OPENAI_API_KEY <value>
uv run tools.py
```
"""

# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0",
#    "flyteplugins-openai>=2.0.0",
#    "openai-agents>=0.11.1",
# ]
# ///

import random

from agents import Agent, Runner
from flyteplugins.openai.agents import function_tool

import flyte

env = flyte.TaskEnvironment(
    name="openai_agents_tools",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    image=flyte.Image.from_uv_script(__file__, name="openai_agents_tools"),
    secrets=flyte.Secret("OPENAI_API_KEY", as_env_var="OPENAI_API_KEY"),
)


@function_tool
@env.task
async def get_weather(city: str) -> dict:
    """Get random weather data."""
    low = random.randint(-5, 25)
    high = low + random.randint(3, 10)
    conditions = random.choice(
        [
            "Sunny",
            "Partly cloudy",
            "Overcast",
            "Light rain",
            "Heavy rain",
            "Thunderstorms",
            "Snowy",
            "Foggy",
            "Windy",
            "Sunny with wind",
            "Clear skies",
            "Hail",
        ]
    )
    return {"city": city, "temperature_range": f"{low}-{high}C", "conditions": conditions}


agent = Agent(
    name="Hello world",
    instructions="You are a helpful agent.",
    tools=[get_weather],
)


@env.task
async def main() -> str:
    result = await Runner.run(agent, input="What's the weather in Tokyo?")
    print(result.final_output)
    return result.final_output


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
    run.wait()
