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
#    "flyte>=2.0.0b1",
#    "openai-agents==0.2.4",
#    "pydantic>=2.10.6",
# ]
# ///

import flyte
from pydantic import BaseModel

from agents import Agent, Runner, function_tool


class Weather(BaseModel):
    city: str
    temperature_range: str
    conditions: str


env = flyte.TaskEnvironment(
    name="openai_agents_tools", resources=flyte.Resources(cpu=1, memory="250Mi"),
    image=flyte.Image.from_uv_script(__file__, name="openai_agents_tools"),
    secrets=flyte.Secret("OPENAI_API_KEY", as_env_var="OPENAI_API_KEY"),
)


@function_tool
@flyte.trace
async def get_weather(city: str) -> Weather:
    print("[debug] get_weather tool called")
    return Weather(city=city, temperature_range="14-20C", conditions="Sunny with wind.")


agent = Agent(
    name="Hello world",
    instructions="You are a helpful agent.",
    tools=[get_weather],
)

@env.task
async def main() -> str:
    result = await Runner.run(agent, input="What's the weather in Tokyo?")
    print(result.final_output)
    return str(result.final_output)


if __name__ == "__main__":
    flyte.init_from_config("../../../config.yaml")
    run = flyte.run(main)
    print(run.url)
    run.wait(run)
