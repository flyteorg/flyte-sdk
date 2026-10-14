"""Run the three interleaving examples across every task type in one go.

    flyte run examples/typesafe_ai/run_all.py run_everything
    flyte run examples/typesafe_ai/run_all.py run_everything --tasks '["code_review"]'

Each stage runs through the same ``typesafe-ai`` environment; the benchmark
(``benchmark.py``) is run separately because it is much larger.
"""

import asyncio
import json
import pathlib

import durable_agent as da
import guardrail_agent as ga
import tool_agent as ta
from _config import BENCHMARK_TASKS
from _runtime import env
from tasks import get_task

import flyte


@env.task
async def run_for_task(task: str, num_cases: int = 4) -> dict:
    """Guard -> fan-out tool agent -> durable loop, for one task type."""
    spec = get_task(task)
    guard, tools = await asyncio.gather(
        ga.run_guard(task=task, num_cases=num_cases),
        ta.plan_and_execute(task=task, num_cases=num_cases),
    )
    durable = await da.durable_agent(task=task, case_id=spec.cases[0].id)
    return {"guard": guard, "tool_agent": tools, "durable": json.loads(durable)["actions"]}


@env.task
async def run_everything(tasks: list[str] | None = None, num_cases: int = 4) -> str:
    """Chain the examples across every task type and return a status summary."""
    task_keys = tasks or BENCHMARK_TASKS
    reports = await asyncio.gather(*[run_for_task(task=t, num_cases=num_cases) for t in task_keys])
    return json.dumps(dict(zip(task_keys, reports)), indent=2, default=str)


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(run_everything)
    print(run.name, run.url)
    run.wait()
