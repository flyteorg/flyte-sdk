"""`flyte.materialize`: pull an artifact partition (or a range of them) through the emergent lineage graph."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from flyte.syncify import syncify

_PLUGIN_HINT = (
    "flyte.materialize requires flyteplugins-union (the lineage planner runs on Union). "
    "Install it with `pip install flyteplugins-union`."
)


@syncify
async def materialize(
    target: Any,
    *,
    inputs: Optional[Dict[Any, Any]] = None,
    concurrency: Optional[int] = None,
    plan_only: bool = False,
    rebuild: Optional[Sequence[Any]] = None,
    rebuild_all: bool = False,
    project: Optional[str] = None,
    domain: Optional[str] = None,
    queue: Optional[str] = None,
    source_check: bool = True,
    partitions: Optional[Mapping[str, Any]] = None,
    **partition_kwargs: Any,
) -> Any:
    """
    Build an artifact partition, or a range of them, by walking the lineage graph backwards from it.

    The graph is the one that emerges from deployed `produces_artifacts=` / `consumes_artifacts=`
    declarations; the planner supplies every parameter of every task in the walk from a partition binding,
    an override in `inputs`, or the parameter's default. Freshness is the task cache. Returns the `Run` (or,
    with `plan_only=True`, the plan without launching anything).

    ```python
    m = flyte.materialize(daily_report, date=datetime(2026, 9, 8))
    m.wait()

    flyte.materialize(daily_report, date=flyte.TimeRange("2026-08-01", "2026-08-31"), concurrency=50)
    flyte.materialize(events, date=datetime(2026, 9, 8), region=["us", "eu"])
    flyte.materialize(daily_report, date=datetime(2026, 9, 8), inputs={"ingest.clean.min_quality": 20})

    # A dimension whose name collides with a keyword of this function goes in `partitions=`:
    flyte.materialize(per_project_report, partitions={"project": "alpha", "date": datetime(2026, 9, 8)})

    # From async code:
    run = await flyte.materialize.aio(daily_report, date=datetime(2026, 9, 8))
    ```

    This is a thin delegator to `flyteplugins.union.factory.derived.materialize`.

    Args:
        target: The `flyte.artifacts.Artifact` handle (or artifact name) to build.
        inputs: Constant overrides, keyed by `"<task>.<param>"` or by task object to a dict of params.
        concurrency: Maximum number of task instances running at once.
        plan_only: Build the plan and probe the cache, launch nothing.
        rebuild: Tasks to rebuild regardless of the cache; everything downstream follows.
        rebuild_all: Rebuild the whole walk.
        project: Project to materialize in; defaults to the init configuration.
        domain: Domain to materialize in; defaults to the init configuration.
        queue: Queue to run the materialization's actions on.
        source_check: Before anything is compiled or launched, check that every source partition the plan
            reads is in the registry, and raise naming the missing ones. Pass False for sources that will
            land while the materialization runs (`flyte materialize --no-source-check`).
        partitions: Partition values of the target, by dimension. Use this for a dimension named like one of
            this function's keywords (`target`, `inputs`, `concurrency`, `plan_only`, `rebuild`,
            `rebuild_all`, `project`, `domain`, `queue`, `source_check`, `partitions`); keyword values win on
            overlap. When any such name is present the planner receives every partition value as one
            `partitions=` mapping.
        partition_kwargs: Partition values of the target as keywords: a value, a list of values, or a
            `flyte.TimeRange(start, end)`.

    Returns:
        The `Run` of the materialization, or the planner's `PlanResult` (the instance DAG and cache probe,
        nothing launched) when `plan_only=True`.

    Raises:
        flyte.errors.MaterializeError: when the plugin is not installed, or the plan cannot be built.
    """
    import asyncio

    from flyte.errors import MaterializeError

    try:
        from flyteplugins.union.factory.derived import materialize as _materialize  # type: ignore[import-not-found]
    except ModuleNotFoundError as e:
        if not (e.name or "").startswith("flyteplugins"):
            raise  # the plugin is installed but one of its own imports failed: surface that, not a hint
        raise MaterializeError(_PLUGIN_HINT) from e
    except ImportError as e:
        # `derived` (or `materialize` in it) is missing: an older flyteplugins-union without the planner.
        raise MaterializeError(f"{_PLUGIN_HINT} ({e})") from e

    kwargs: Dict[str, Any] = {
        "inputs": inputs,
        "concurrency": concurrency,
        "plan_only": plan_only,
        "rebuild": rebuild,
        "rebuild_all": rebuild_all,
        "project": project,
        "domain": domain,
    }
    if queue is not None:
        kwargs["queue"] = queue
    if not source_check:
        kwargs["source_check"] = False
    values = {**dict(partitions or {}), **partition_kwargs}
    if any(k in _reserved_names() for k in values):
        # A dimension named like a keyword cannot travel as **kwargs; hand the planner the whole mapping.
        kwargs["partitions"] = values
    else:
        kwargs.update(values)
    aio = getattr(_materialize, "aio", None)
    if callable(aio):
        return await aio(target, **kwargs)
    # A plain (or syncify'd) callable: run it off this event loop's thread so it may block.
    return await asyncio.to_thread(_materialize, target, **kwargs)


def _reserved_names() -> frozenset:
    """The keyword names of `materialize` itself, which a partition dimension cannot share as a keyword."""
    import inspect

    params = inspect.signature(materialize).parameters.values()
    return frozenset(p.name for p in params if p.kind is not inspect.Parameter.VAR_KEYWORD)
