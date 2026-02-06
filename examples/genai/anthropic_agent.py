# /// script
# requires-python = "==3.13"
# dependencies = [
#     "flyte==2.0.0b0",
#     "flyteplugins-anthropic",
# ]
# ///

"""Example of using Anthropic Claude as an agent with Flyte tasks as tools.

This example demonstrates how to create a Claude agent that can use Flyte tasks
as tools to accomplish complex tasks. The agent orchestrates multiple tool calls
to make a sandwich.
"""

import asyncio
from typing import Optional

from flyteplugins.anthropic import function_tool, run_agent

import flyte

agent_env = flyte.TaskEnvironment(
    "anthropic-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="anthropic_api_key", as_env_var="ANTHROPIC_API_KEY")],
    image=flyte.Image.from_uv_script(__file__, name="anthropic-agent", pre=True),
)


@agent_env.task
async def get_bread() -> str:
    """Get bread for making a sandwich."""
    await asyncio.sleep(0.5)
    return "sourdough bread"


@agent_env.task
async def get_peanut_butter() -> str:
    """Get peanut butter for the sandwich."""
    await asyncio.sleep(0.5)
    return "creamy peanut butter"


@agent_env.task
async def get_jelly() -> str:
    """Get jelly for the sandwich."""
    await asyncio.sleep(0.5)
    return "grape jelly"


@agent_env.task
async def spread_ingredient(bread: str, ingredient: str) -> str:
    """Spread an ingredient on bread."""
    await asyncio.sleep(0.5)
    return f"{bread} with {ingredient}"


@agent_env.task
async def assemble_sandwich(
    top_slice: Optional[str] = None,
    bottom_slice: Optional[str] = None,
) -> str:
    """Assemble the sandwich from prepared slices."""
    await asyncio.sleep(0.5)
    if top_slice and bottom_slice:
        return f"Sandwich: {bottom_slice} + {top_slice}"
    elif top_slice:
        return f"Open sandwich: {top_slice}"
    elif bottom_slice:
        return f"Open sandwich: {bottom_slice}"
    return "Empty sandwich (no ingredients)"


@agent_env.task
async def eat_sandwich(sandwich: str) -> str:
    """Eat the completed sandwich."""
    await asyncio.sleep(0.5)
    return f"Delicious! Ate: {sandwich}"


@agent_env.task
async def sandwich_agent(goals: list[str]) -> list[str]:
    """Run the sandwich-making agent for multiple goals."""

    # Create tools from Flyte tasks
    tools = [
        function_tool(get_bread),
        function_tool(get_peanut_butter),
        function_tool(get_jelly),
        function_tool(spread_ingredient),
        function_tool(assemble_sandwich),
        function_tool(eat_sandwich),
    ]

    async def run_single_goal(goal: str, index: int) -> str:
        with flyte.group(f"sandwich-maker-{index}"):
            result = await run_agent(
                prompt=goal,
                tools=tools,
                system=(
                    "You are a sandwich-making assistant. Use the available tools to make"
                    " sandwiches according to user requests. Always get ingredients before"
                    " spreading them, and assemble the sandwich before eating it."
                ),
                model="claude-sonnet-4-20250514",
            )
            return result

    tasks = [run_single_goal(goal, idx) for idx, goal in enumerate(goals, start=1)]
    results = await asyncio.gather(*tasks)
    return list(results)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(
        sandwich_agent,
        goals=[
            "Make a peanut butter sandwich.",
            "Make a peanut butter and jelly sandwich.",
            "Make a jelly-only sandwich.",
        ],
    )
    print(f"View at: {run.url}")
    print(f"Results: {run.result}")
