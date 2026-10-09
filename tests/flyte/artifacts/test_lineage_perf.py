"""Micro-benchmarks for the lineage authoring surface.

Thresholds are generous (10-100x the measured numbers on a laptop) so they only catch regressions in
complexity, not machine noise. Marked `perf` and deselected by default (pyproject `addopts`); run them with
`uv run pytest -m perf tests/flyte/artifacts/test_lineage_perf.py`.
"""

import asyncio
import time
from datetime import datetime

import pytest

import flyte
import flyte.artifacts as artifacts
from flyte._internal.runtime.convert import convert_from_native_to_outputs
from flyte.artifacts._lineage import check_conflicts, declared_output_metadata, extract_task_lineage, summarize
from flyte.io import DataFrame, File

pytestmark = pytest.mark.perf

OUT = artifacts.Artifact("perf_out", type=File, partitions={"date": artifacts.Daily})
UP = artifacts.Artifact("perf_up", type=DataFrame, partitions={"date": artifacts.Daily, "r": str})
env = flyte.TaskEnvironment(name="perf")


def _per_op(fn, n):
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) / n


def test_handle_construction_is_cheap():
    # Measured ~1.3 us. Source location uses sys._getframe, not inspect.stack().
    assert _per_op(lambda: artifacts.Artifact("h", type=File, partitions={"date": artifacts.Daily}), 5000) < 100e-6


def test_handle_construction_does_not_use_inspect_stack(monkeypatch):
    import inspect

    def boom(*a, **k):
        raise AssertionError("inspect.stack() must not be used")

    monkeypatch.setattr(inspect, "stack", boom)
    artifacts.Artifact("no_stack")


async def _f(x: list[DataFrame], date: datetime) -> File:
    raise NotImplementedError


def test_decorating_with_declarations():
    # Measured ~60-110 us, the same as an undeclared task (dominated by NativeInterface.from_callable).
    def deco():
        env.task(
            produces_artifacts=(OUT,), consumes_artifacts={"x": UP.all("r"), "date": OUT.get_partition_value("date")}
        )(_f)

    assert _per_op(deco, 200) < 5e-3


def test_bindings_for_fifty_parameters():
    params = ", ".join(f"p{i}: int = {i}" for i in range(48))
    ns: dict = {}
    exec(
        f"async def big(x: list[DataFrame], date: datetime, {params}) -> File: ...",
        {"DataFrame": DataFrame, "datetime": datetime, "File": File},
        ns,
    )
    t = env.task(
        produces_artifacts=(OUT,), consumes_artifacts={"x": UP.all("r"), "date": OUT.get_partition_value("date")}
    )(ns["big"])
    assert len(extract_task_lineage(t).bindings["parameters"]) == 50
    # Measured ~120 us.
    assert _per_op(lambda: extract_task_lineage(t), 100) < 10e-3


def _tasks(n_tasks):
    handles = [artifacts.Artifact(f"a{i}", type=File, partitions={"date": artifacts.Daily}) for i in range(2 * n_tasks)]
    e = flyte.TaskEnvironment(name=f"perf_{n_tasks}")
    out = []
    for i in range(n_tasks):

        async def g(x: File, date: datetime) -> File:
            raise NotImplementedError

        g.__name__ = g.__qualname__ = f"g{i}"
        o, inp = handles[2 * i], handles[2 * i + 1]
        out.append(
            e.task(produces_artifacts=(o,), consumes_artifacts={"x": inp, "date": o.get_partition_value("date")})(g)
        )
    return out


def test_deploy_validation_500_tasks_1000_handles():
    tasks = _tasks(500)
    start = time.perf_counter()
    s = summarize(tasks)
    elapsed = time.perf_counter() - start
    assert (s.tasks, s.handles, s.edges) == (500, 1000, 500)
    # Measured ~35 ms.
    assert elapsed < 3.0


def test_conflict_check_is_linear():
    def timed(n):
        hs = [artifacts.Artifact(f"c{i % (n // 2)}", type=File, partitions={"date": artifacts.Daily}) for i in range(n)]
        best = float("inf")
        for _ in range(5):
            start = time.process_time()  # CPU time of this process: robust to other xdist workers
            check_conflicts(hs)
            best = min(best, time.process_time() - start)
        return best

    small, large = timed(10000), timed(80000)
    # 8x the input: linear is ~8x (measured ~8.3x), quadratic would be ~64x.
    assert large / max(small, 1e-4) < 32


def test_runtime_publishing_overhead_per_output():
    async def fh(date: datetime) -> File:
        raise NotImplementedError

    async def fp(date: datetime) -> File:
        raise NotImplementedError

    declared = env.task(produces_artifacts=(OUT,), consumes_artifacts={"date": OUT.get_partition_value("date")})(fh)
    plain = env.task(fp)

    async def run(n=500):
        start = time.perf_counter()
        for _ in range(n):
            await convert_from_native_to_outputs(File(path="s3://b/x"), plain.native_interface, "x")
        base = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(n):
            md = declared_output_metadata(declared, {"date": datetime(2026, 9, 8)})
            await convert_from_native_to_outputs(
                File(path="s3://b/x"), declared.native_interface, "x", handle_declared=md
            )
        return base / n, (time.perf_counter() - start) / n

    base, with_handle = asyncio.run(run())
    # Measured ~11 us of overhead on ~30 us.
    assert with_handle - base < 2e-3
