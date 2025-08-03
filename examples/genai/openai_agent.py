# /// script
# requires-python = "==3.13"
# dependencies = [
#     "flyte==2.0.0b0",
#     "openai-agents==0.2.4",
# ]
# ///

import asyncio
from typing import Optional

from agents import Agent, Runner, function_tool

import flyte

agent_env = flyte.TaskEnvironment(
    "openai-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    image=flyte.Image.from_uv_script(__file__, name="openai-agent", pre=True),
)


@agent_env.task
async def get_bread() -> str:
    await asyncio.sleep(1)
    return "bread"


@agent_env.task
async def get_peanut_butter() -> str:
    await asyncio.sleep(1)
    return "peanut butter"


@agent_env.task
async def get_jelly() -> str:
    await asyncio.sleep(1)
    return "jelly"


@agent_env.task
async def spread_peanut_butter(bread: str, peanut_butter: str) -> str:
    await asyncio.sleep(1)
    return f"{bread} with {peanut_butter}"


@agent_env.task
async def spread_jelly(bread: str, jelly: str) -> str:
    await asyncio.sleep(1)
    return f"{bread} with {jelly}"


@agent_env.task
async def assemble_sandwich(
    pb_bread: Optional[str] = None, j_bread: Optional[str] = None
) -> str:
    await asyncio.sleep(1)
    return f"{pb_bread} and {j_bread} combined"


@agent_env.task
async def eat_sandwich(sandwich: str) -> str:
    await asyncio.sleep(1)
    return f"Ate: {sandwich} 😋"


@function_tool
async def get_bread_tool() -> str:
    """Get bread for the sandwich."""
    return await get_bread()


@function_tool
async def get_peanut_butter_tool() -> str:
    """Get peanut butter for the sandwich."""
    return await get_peanut_butter()


@function_tool
async def get_jelly_tool() -> str:
    """Get jelly for the sandwich."""
    return await get_jelly()


@function_tool
async def spread_peanut_butter_tool(bread: str, peanut_butter: str) -> str:
    """Spread peanut butter on bread."""
    return await spread_peanut_butter(bread, peanut_butter)


@function_tool
async def spread_jelly_tool(bread: str, jelly: str) -> str:
    """Spread jelly on bread."""
    return await spread_jelly(bread, jelly)


@function_tool
async def assemble_sandwich_tool(
    pb_bread: Optional[str] = None, j_bread: Optional[str] = None
) -> str:
    """Assemble a sandwich from any combination of components (e.g., peanut butter bread, jelly bread)."""
    return await assemble_sandwich(pb_bread, j_bread)


@function_tool
async def eat_sandwich_tool(sandwich: str) -> str:
    """Eat the completed sandwich."""
    return await eat_sandwich(sandwich)


@agent_env.task
async def agent(goals: list[str]) -> list[str]:
    async def run_agent(goal: str, index: int) -> str:
        with flyte.group(f"sandwich-maker-{index}"):
            result = await Runner.run(
                Agent(
                    name="sandwich_maker",
                    instructions="You are a sandwich-making assistant.",
                    tools=[
                        get_bread_tool,
                        get_peanut_butter_tool,
                        get_jelly_tool,
                        spread_peanut_butter_tool,
                        spread_jelly_tool,
                        assemble_sandwich_tool,
                        eat_sandwich_tool,
                    ],
                ),
                input=goal,
            )
            return result.final_output

    tasks = [run_agent(goal, idx) for idx, goal in enumerate(goals, start=1)]
    results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    flyte.init_from_config("../config.yaml")
    run = flyte.run(
        agent,
        [
            "Make a plain peanut butter sandwich.",
            "Make a peanut butter and jelly sandwich.",
            "Make a jelly sandwich with no peanut butter.",
            "You have one slice with peanut butter and one with jelly. Just assemble them into a sandwich.",
            "Make one peanut butter sandwich and one jelly sandwich.",
        ],
    )
    print(run.url)
