# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b1",
#    "unionai-reuse>=0.1.3",
# ]
# ///

import asyncio
from typing import Dict, List

import flyte

coordinator_env = flyte.TaskEnvironment(
    "coordinator_env",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    reusable=flyte.ReusePolicy(replicas=2, idle_ttl=300),
    image=flyte.Image.from_uv_script(__file__, name="agent_simulation_image"),
)

coordinator_decision_env = coordinator_env.clone_with(
    name="coordinator_decision_env",
    reusable=flyte.ReusePolicy(replicas=4, idle_ttl=300),
)

research_assistant_env = coordinator_env.clone_with(
    name="research_assistant_env",
    reusable=flyte.ReusePolicy(replicas=8, idle_ttl=300),
)

tool_env = coordinator_env.clone_with(name="tool_env", reusable=flyte.ReusePolicy(replicas=8, idle_ttl=300))


# Mock tools that research agents can use
@tool_env.task
async def search_web(query: str) -> str:
    # await asyncio.sleep(1.0)  # Simulate API call
    return f"Web results for: {query}"


@tool_env.task
async def extract_entities(text: str) -> List[str]:
    # await asyncio.sleep(1.0)
    return [f"Entity from: {text}"]


@tool_env.task
async def analyze_text(text: str) -> Dict[str, str]:
    # await asyncio.sleep(1.0)
    return {"analysis": f"Analysis of: {text}", "sentiment": "positive"}


@tool_env.task
async def summarize_text(text: str) -> Dict[str, str]:
    # await asyncio.sleep(1.0)
    return {"summary": f"Summary of: {text}"}


@tool_env.task
async def finalize_answer(text: str) -> Dict[str, str]:
    # await asyncio.sleep(1.0)
    return {"answer": f"Answer of: {text}"}


# Research agent that runs a sequence of tools
@research_assistant_env.task
async def research_assistant(prompt: str, tool_sequence: List[str]) -> Dict[str, str]:
    results = {}
    current_input = prompt

    tool_map = {
        "search": search_web,
        "analyze": analyze_text,
        "extract": extract_entities,
        "summarize": summarize_text,
        "finalize": finalize_answer,
    }

    for tool_name in tool_sequence:
        tool_fn = tool_map[tool_name]
        result = await tool_fn(current_input)
        results[tool_name] = str(result)
        # current_input = str(result)

    return results


@coordinator_decision_env.task
async def resource_coordinator_decision(
    prompt: str,
    num_agents: int,
) -> List[Dict[str, str]]:
    tool_sequence = ["search", "extract", "analyze", "summarize", "finalize"] * 10

    tasks = []
    for _ in range(num_agents):
        tasks.append(research_assistant(prompt, tool_sequence))

    task_results = await asyncio.gather(*tasks)
    return task_results


# Research coordinator that spawns multiple agents
@coordinator_env.task
async def research_coordinator(
    prompt: str,
    num_rounds: int = 4,
    num_agents: int = 5,
) -> List[List[Dict[str, str]]]:
    # Do multiple rounds of research
    results = []
    for _ in range(num_rounds):
        # Select tool sequence for this agent
        results.append(await resource_coordinator_decision(prompt, num_agents))

    # Gather results from all agents
    return results


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")

    prompt = "What are the latest developments in AI?"
    runs = []
    for i in range(10):
        runs.append(
            flyte.run(
                research_coordinator,
                prompt=prompt,
            )
        )
    print([run.url for run in runs])
