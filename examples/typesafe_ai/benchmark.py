"""The experiment: {with, without System 1} x {Qwen, Sonnet, Opus} x {task type},
with **every cell run several times**.

    tasks      = {customer support, code review, contract review}
    conditions = {with System 1 (Jev), without System 1}
    providers  = {Qwen 3.8 27B, Claude Sonnet, Claude Opus}
    units      = tasks x conditions x providers x cases x repeats

Each unit is an independent Flyte action, grouped by task type and named for its
arm (``evaluate_unit_jev`` / ``evaluate_unit_no_jev``), so the matrix fans out
across the cluster while the run graph stays readable — the "AI runtime executes with fan-out, really fast" step of the
architecture. Repeats are what make the numbers trustworthy: one sample per cell
tells you nothing about variance, and the interesting claim about System 1 is not
just that it is faster but that it is *stable* — the same input yields the same
typed decision run after run, where free-text classification drifts.

Run (uses the default `demo` org + TYPESAFE_API_KEY / DEMO_* secrets):

    flyte run examples/typesafe_ai/benchmark.py run_benchmark
    flyte run examples/typesafe_ai/benchmark.py run_benchmark \
        --tasks '["code_review"]' --num_cases 8 --repeats 5
"""

import asyncio
import pathlib

from _config import (
    BENCHMARK_CONDITIONS,
    BENCHMARK_TASKS,
    FANOUT_CONCURRENCY,
    NUM_EVAL_CASES,
    REPEATS_PER_CASE,
)
from _pipeline import UnitResult, evaluate_case
from _report import build_report, cell_summary
from _runtime import driver_env, env
from tasks import get_task

import flyte
import flyte.report


@env.task
async def evaluate_unit(task: str, case_id: str, with_system1: bool, provider: str, repeat: int) -> UnitResult:
    """Evaluate one (task x condition x case x repeat) run as a standalone action.

    ``repeat`` is part of the signature on purpose: it makes every repetition a
    distinct action, so repeats are never collapsed and each one shows up
    individually in the run graph.
    """
    return await evaluate_case(task, case_id, with_system1, provider, repeat)


@driver_env.task(report=True)
async def run_benchmark(
    tasks: list[str] | None = None,
    num_cases: int = NUM_EVAL_CASES,
    repeats: int = REPEATS_PER_CASE,
    concurrency: int = FANOUT_CONCURRENCY,
) -> str:
    """Fan out the whole matrix, aggregate over repeats, and render the report.

    The fan-out is organised so the run graph reads the way the experiment does:
    every unit for a task type sits inside a ``flyte.group`` named after that
    task, and each action is named for the arm it ran — ``evaluate_unit_jev`` or
    ``evaluate_unit_no_jev`` — so the two arms are distinguishable at a glance
    without opening a single action.
    """
    task_keys = tasks or BENCHMARK_TASKS
    sem = asyncio.Semaphore(concurrency)

    async def unit(task_key: str, case_id: str, with_s1: bool, provider: str, repeat: int):
        async with sem:
            named = evaluate_unit.override(short_name="evaluate_unit_jev" if with_s1 else "evaluate_unit_no_jev")
            return await named(task_key, case_id, with_s1, provider, repeat)

    async def run_task_group(task_key: str) -> list[UnitResult]:
        """Every unit for one task type, under a group named after that task.

        The ``with`` has to sit *inside* the coroutine that ``gather`` wraps.
        ``flyte.group`` sets a ``ContextVar``, and asyncio copies the context
        when it turns a coroutine into a Task — so one group per Task stays
        isolated even though all three run at once. Hoisting the ``with`` out to
        the caller would put every group in the same context and let them race.
        """
        with flyte.group(task_key):
            return list(
                await asyncio.gather(
                    *[
                        unit(task_key, case.id, with_s1, provider, repeat)
                        for case in get_task(task_key).cases[:num_cases]
                        for (with_s1, provider) in BENCHMARK_CONDITIONS
                        for repeat in range(repeats)
                    ]
                )
            )

    # Task groups run concurrently; `sem` still caps total units in flight, so
    # this widens the fan-out without adding load on the model gateways.
    grouped = await asyncio.gather(*[run_task_group(tk) for tk in task_keys])
    results: list[UnitResult] = [unit_result for group in grouped for unit_result in group]

    build_report(results, task_keys, repeats=repeats, num_cases=num_cases)
    await flyte.report.flush.aio()

    ok = sum(1 for r in results if not r.error)
    total_tokens = sum(r.total_tokens for r in results)
    header = (
        f"typesafe benchmark: {len(results)} runs ({ok} ok) — {len(task_keys)} tasks x "
        f"{len(BENCHMARK_CONDITIONS)} conditions x {num_cases} cases x {repeats} repeats; "
        f"{total_tokens:,} tokens total."
    )
    return header + "\n" + cell_summary(results, task_keys)


@driver_env.task
async def run_task_benchmark(
    task: str = "code_review",
    num_cases: int = NUM_EVAL_CASES,
    repeats: int = REPEATS_PER_CASE,
    concurrency: int = FANOUT_CONCURRENCY,
) -> str:
    """Benchmark a single task type (handy while iterating on a new one).

    The report is rendered by the ``run_benchmark`` sub-action this delegates to.
    """
    return await run_benchmark(tasks=[task], num_cases=num_cases, repeats=repeats, concurrency=concurrency)


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(run_benchmark)
    print(run.name)
    print(run.url)
    run.wait()
