import asyncio
import logging
from typing import Any, Callable, Dict, List, Union

import flyte

agent_env = flyte.TaskEnvironment(
    "agent",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    reusable=flyte.ReusePolicy(
        replicas=1,
        idle_ttl=60,
        concurrency=10,
        scaledown_ttl=60,
    ),
    image=flyte.Image.from_debian_base().with_pip_packages("unionai-reuse==0.1.5"),
)


# --- Dummy DummyAgent class with dependency-aware planning ---
@agent_env.task
async def get_plan(goal: str) -> List[Dict[str, Union[str, List[str]]]]:
    # Each step has a name, function ID, and dependencies
    return [
        {"id": "get_bread", "deps": []},
        {"id": "get_peanut_butter", "deps": []},
        {"id": "get_jelly", "deps": []},
        {"id": "spread_peanut_butter", "deps": ["get_bread", "get_peanut_butter"]},
        {"id": "spread_jelly", "deps": ["get_bread", "get_jelly"]},
        {"id": "assemble_sandwich", "deps": ["spread_peanut_butter", "spread_jelly"]},
        {"id": "eat", "deps": ["assemble_sandwich"]},
    ]


# --- Step function definitions ---
@agent_env.task
async def get_bread(context: Dict[str, str]) -> str:
    return "bread"


@agent_env.task
async def get_peanut_butter(context: Dict[str, str]) -> str:
    return "peanut butter"


@agent_env.task
async def get_jelly(context: Dict[str, str]) -> str:
    return "jelly"


@agent_env.task
async def spread_peanut_butter(context: Dict[str, str]) -> str:
    return f"{context['get_bread']} with {context['get_peanut_butter']}"


@agent_env.task
async def spread_jelly(context: Dict[str, str]) -> str:
    return f"{context['get_bread']} with {context['get_jelly']}"


@agent_env.task
async def assemble_sandwich(context: Dict[str, str]) -> str:
    return f"{context['spread_peanut_butter']} and {context['spread_jelly']} combined"


@agent_env.task
async def eat(context: Dict[str, Any]) -> str:
    return f"Ate: {context['assemble_sandwich']} 😋"


# --- Registry of step functions ---
STEP_FUNCTIONS: Dict[str, Any | Callable[[Dict[str, Any]], Any]] = {
    "get_bread": get_bread,
    "get_peanut_butter": get_peanut_butter,
    "get_jelly": get_jelly,
    "spread_peanut_butter": spread_peanut_butter,
    "spread_jelly": spread_jelly,
    "assemble_sandwich": assemble_sandwich,
    "eat": eat,
}


# --- Executor that respects dependencies ---
@agent_env.task
async def execute_plan(plan: List[Dict[str, Union[str, List[str]]]]) -> Dict[str, str]:
    step_funcs = STEP_FUNCTIONS
    results = {}
    remaining = {step["id"]: step for step in plan}

    i = 0
    while remaining:
        with flyte.group(f"step-set-{i}"):
            print(f"{results}")
            # Find all steps that are ready to run (no unmet dependencies)
            ready = [step_id for step_id, step in remaining.items() if all(dep in results for dep in step["deps"])]

            # Run all ready steps concurrently
            tasks = {step_id: asyncio.create_task(step_funcs[step_id](results)) for step_id in ready}

            for step_id, task in tasks.items():
                result = await task
                print(f"✅ {step_id}: {result}")
                results[step_id] = result
                del remaining[step_id]
            i = i + 1

    return results


# --- Main async driver ---
@agent_env.task
async def main(goal: str) -> Dict[str, str]:
    plan = await get_plan(goal)
    print(f"📋 Plan with dependencies:\n{plan}\n")
    results = await execute_plan(plan)
    print("\n🏁 Final Result:")
    return results


if __name__ == "__main__":
    # asyncio.run(main("Make a peanut butter and jelly sandwich"))
    import flyte.git
    flyte.init_from_config(flyte.git.config_from_root())
    r = flyte.run(main, goal="Make a peanut butter and jelly sandwich")
    print(r.url)
