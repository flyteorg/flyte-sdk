# /// script
# requires-python = "==3.13"
# dependencies = [
#     "flyte>=2.0.0",
#     "flyteplugins-openai>=2.0.0",
#     "openai-agents>=0.11.1",
# ]
# ///

import asyncio
from typing import Optional

from agents import Agent, Runner
from flyteplugins.openai.agents import function_tool

import flyte

agent_env = flyte.TaskEnvironment(
    "openai-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="niels_openai_api_key", as_env_var="OPENAI_API_KEY")],
    image=flyte.Image.from_uv_script(__file__, name="openai-agent"),
)


@function_tool
@agent_env.task
async def get_bread() -> str:
    await asyncio.sleep(1)
    return "bread"


@function_tool
@agent_env.task
async def get_peanut_butter() -> str:
    await asyncio.sleep(1)
    return "peanut butter"


@function_tool
@agent_env.task
async def get_jelly() -> str:
    await asyncio.sleep(1)
    return "jelly"


@function_tool
@agent_env.task
async def spread_peanut_butter(bread: str, peanut_butter: str) -> str:
    await asyncio.sleep(1)
    return f"{bread} with {peanut_butter}"


@function_tool
@agent_env.task
async def spread_jelly(bread: str, jelly: str) -> str:
    await asyncio.sleep(1)
    return f"{bread} with {jelly}"


@function_tool
@agent_env.task
async def assemble_sandwich(pb_bread: Optional[str] = None, j_bread: Optional[str] = None) -> str:
    await asyncio.sleep(1)
    return f"{pb_bread} and {j_bread} combined"


@function_tool
@agent_env.task
async def eat_sandwich(sandwich: str) -> str:
    await asyncio.sleep(1)
    return f"Ate: {sandwich} 😋"


@agent_env.task
async def agent(goals: list[str]) -> list[str]:
    async def run_agent(goal: str, index: int) -> str:
        with flyte.group(f"sandwich-maker-{index}"):
            result = await Runner.run(
                Agent(
                    name="sandwich_maker",
                    instructions="You are a sandwich-making assistant.",
                    tools=[
                        get_bread,
                        get_peanut_butter,
                        get_jelly,
                        spread_peanut_butter,
                        spread_jelly,
                        assemble_sandwich,
                        eat_sandwich,
                    ],
                ),
                input=goal,
            )
            return result.final_output

    tasks = [run_agent(goal, idx) for idx, goal in enumerate(goals, start=1)]
    results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    flyte.init_from_config()
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
