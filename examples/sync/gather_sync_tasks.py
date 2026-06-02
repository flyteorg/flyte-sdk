"""Tiny repro: an async task that collects sync tasks and runs them
concurrently via ``asyncio.to_thread`` + ``asyncio.gather``.

Mirrors the downstream-task dispatch in a larger deployment workflow:
a builder assembles ``(key, fn, kwargs)`` specs (like ``build_*_task``),
and the async parent dispatches each sync task to its own thread so they
run concurrently even though they're synchronous.
"""

import asyncio
from typing import Any

import flyte
from flyte.errors import RuntimeUserError

env = flyte.TaskEnvironment(name="gather_sync_tasks")


@env.task
def run_worker(name: str, seconds: float) -> str:
    """A synchronous task that does some blocking work.  more"""
    print(f"Worker '{name}' starting ({seconds}s)...", flush=True)
    import time

    time.sleep(seconds)
    print(f"Worker '{name}' done.", flush=True)
    return f"{name} slept {seconds}s"


def build_worker_tasks(names: list[str]) -> list[tuple[str, Any, dict[str, Any]]]:
    """Collect one sync task spec per name (mirrors build_vkmpi_task)."""
    tasks = []
    for i, name in enumerate(names):
        tasks.append((name, run_worker, dict(name=name, seconds=1.0 + i)))
    return tasks


@env.task(retries=2)
async def main(names: list[str] = ["a", "b", "c"]) -> list[str]:
    tasks_to_run = build_worker_tasks(names)

    # Dispatch each sync task to its own thread for true concurrency.
    coroutines = [asyncio.to_thread(fn, **kw) for _key, fn, kw in tasks_to_run]
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    out = []
    for (key, _, _), result in zip(tasks_to_run, results):
        if isinstance(result, BaseException):
            print(f"Task '{key}' failed: {result}", flush=True)
            raise result
        print(f"Task '{key}' completed: {result}", flush=True)
        out.append(result)
    raise RuntimeUserError("intentional", "recover time")
    return out


@env.task(retries=2)
async def main_with_aio(names: list[str] = ["a", "b", "c"]) -> list[str]:
    tasks_to_run = build_worker_tasks(names)

    # Dispatch each sync task to its own thread for true concurrency.
    coroutines = [fn.aio(**kw) for _key, fn, kw in tasks_to_run]
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    out = []
    for (key, _, _), result in zip(tasks_to_run, results):
        if isinstance(result, BaseException):
            print(f"Task '{key}' failed: {result}", flush=True)
            raise result
        print(f"Task '{key}' completed: {result}", flush=True)
        out.append(result)
    raise RuntimeUserError("intentional", "recover time")
    return out


@env.task(retries=2)
async def main_with_ordering(names: list[str] = ["a", "b", "c"]) -> list[str]:
    tasks_to_run = build_worker_tasks(names)

    # Dispatch each sync task to its own thread for true concurrency.
    results = []
    for _key, fn, kw in tasks_to_run:
        print(f"Running {_key=}", flush=True)
        res = await fn.aio(**kw)
        results.append(res)

    out = []
    for (key, _, _), result in zip(tasks_to_run, results):
        if isinstance(result, BaseException):
            print(f"Task '{key}' failed: {result}", flush=True)
            raise result
        print(f"Task '{key}' completed: {result}", flush=True)
        out.append(result)
    raise RuntimeUserError("intentional", "recover time")
    return out


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
    run.wait()
