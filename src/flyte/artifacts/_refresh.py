"""
Keeping an artifact fresh: `Artifact(..., refresh=...)` (the owner) and `handle.materialize_on(...)` (anyone).

Both compile to the same thing, the colocated form of a factory's `fc.on(event, target, lag=...)`: a trigger on
a small generated task whose body calls `flyte.materialize(target, <partition>)`. The task lives in a generated
`refresh-<target>` environment and is rebuilt at run time by `InternalTaskResolver` from the policy itself, so
nobody defines, names or imports it. Deploy records it as an inbound trigger on the target (`materialize_on` in
the task's `lineage.bindings`, plus a `lineage.consumes` edge), which the graph draws beside the target and
`flyte factory snapshot` turns into `fc.on(...)`.
"""

from __future__ import annotations

import base64
import inspect
import json
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ._handle import (
    GRANULARITY_MARKERS,
    Artifact,
    ArtifactRef,
    TimeRange,
    _caller_location,
    _Granularity,
    is_handle,
)

if TYPE_CHECKING:
    from flyte import TaskEnvironment

#: `flyte.materialize` delegates to the lineage planner in flyteplugins-union, so the refresh task's image needs it.
PLUGIN_PACKAGE = "flyteplugins-union"
_BUILDER = "flyte.artifacts._refresh.build_refresh_task"

# One generated environment per target (name, project, domain) in this process, so several policies on the same
# target share it and deploying it twice is idempotent.
_ENVS: Dict[Tuple[str, Optional[str], Optional[str]], "TaskEnvironment"] = {}


def _slug(name: str, sep: str) -> str:
    return re.sub(r"[^a-z0-9]+", sep, name.lower()).strip(sep) or "target"


class Refresh:
    """
    When to materialize an artifact: on a schedule, or on each new version of a source artifact.

    Pass it as `refresh=` when you declare the artifact (one policy or a list), or use `handle.materialize_on(...)`
    for an artifact you only read. A bare `flyte.Cron(...)`, `flyte.FixedRate(...)` or source handle works as
    `refresh=` too; `Refresh` adds `lag`, a `name` and source partition filters.

    ```python
    daily_report = artifacts.Artifact(
        "daily_report", type=File, partitions={"date": artifacts.Daily},
        refresh=artifacts.Refresh(flyte.Cron("0 2 * * *"), lag=artifacts.TimeRange(days=1)),  # yesterday, nightly
    )
    events = artifacts.Artifact(
        "events", type=DataFrame, partitions={"date": artifacts.Daily, "region": str},
        refresh=artifacts.Refresh(raw_events, region="us"),  # each new us raw file, as it lands
    )
    ```

    - On a schedule, each firing materializes the target partition at the trigger time minus `lag`, floored to
      the target's time granularity. The target needs a time dimension; its other dimensions are left to the
      planner (their expected values).
    - On a source, each new source version materializes the target partition with the same values (every
      target dimension must be a source dimension; a time value is floored to the target's granularity).

    Args:
        event: `flyte.Cron(...)`, `flyte.FixedRate(...)`, or a source `artifacts.Artifact` handle.
        lag: For a schedule, how far behind the trigger time the materialized partition is (a trailing
            `TimeRange`). Not allowed for a source event.
        name: Name of the generated task and its trigger (default: `<target>_on_schedule`, or
            `<target>_on_<source>`). A snapshot names the factory trigger after it.
        filter: For a source event, string partition values the new source version must carry (`region="us"`).
    """

    def __init__(self, event: Any, *, lag: Optional[TimeRange] = None, name: Optional[str] = None, **filter: str):
        self.event = event
        self.lag = lag
        self.name = name
        self.filter = dict(filter)
        self.src_path, self.src_line = _caller_location()

    def __repr__(self) -> str:
        extra = (f", lag={self.lag!r}" if self.lag else "") + (f", name={self.name!r}" if self.name else "")
        return f"Refresh({self.event!r}{extra})"

    def spec(self, target: Artifact) -> Dict[str, Any]:
        """
        Validate this policy against `target` and return its spec: everything needed to rebuild the task, and
        the `materialize_on` record deploy writes (`target`, `event`, `lag`).
        """
        from flyte._trigger import Cron, FixedRate

        what = f"{target.name} refresh"
        if not target.partitions:
            raise ValueError(f"{what}: {target.name} has no partitions, so there is no partition to choose")
        lag = self.lag
        if lag is not None and (not isinstance(lag, TimeRange) or lag.is_absolute):
            raise ValueError(f"{what}: lag must be a trailing TimeRange, e.g. TimeRange(days=1)")
        if self.name is not None and not self.name.isidentifier():
            raise ValueError(f"{what}: name {self.name!r} must be a Python identifier")
        spec: Dict[str, Any] = {
            "target": target.name,
            "project": target.project,
            "domain": target.domain,
            "dims": [
                [d, "time", t.registry] if isinstance(t, _Granularity) else [d, "int" if t is int else "str"]
                for d, t in target.partitions.items()
            ],
        }
        event = self.event
        if isinstance(event, (Cron, FixedRate)):
            if self.filter:
                raise ValueError(f"{what}: partition filters apply to a source event only")
            tdim = target.time_dim
            if tdim is None:
                raise ValueError(
                    f"{what}: a schedule picks a time partition, but {target.name} has no time dimension "
                    f"(declared: {', '.join(target.partitions)}). Refresh it on a source handle instead."
                )
            if isinstance(event, Cron):
                spec["event"] = {"kind": "cron", "cron": event.expression, "timezone": event.timezone}
            else:
                spec["event"] = {"kind": "fixed_rate", "interval_minutes": event.interval_minutes}
            spec["lag"] = {"days": lag.days, "hours": lag.hours} if lag is not None else None
            spec["name"] = self.name or f"{_slug(target.name, '_')}_on_schedule"
        elif is_handle(event):
            if lag is not None:
                raise ValueError(f"{what}: lag applies to a schedule, not a source event")
            missing = [d for d in target.partitions if d not in event.partitions]
            if missing:
                raise ValueError(
                    f"{what} on {event.name}: {target.name} has dimension(s) {', '.join(repr(d) for d in missing)} "
                    f"that {event.name} does not, so a new {event.name} version cannot choose the partition."
                )
            from flyte._trigger import OnArtifact

            OnArtifact(event, partitions=self.filter or None)  # the trigger's own checks, now rather than at deploy
            spec["event"] = {"kind": "source", "source": event.name, "filter": dict(self.filter)}
            spec["lag"] = None
            spec["name"] = self.name or f"{_slug(target.name, '_')}_on_{_slug(event.name, '_')}"
        else:
            raise TypeError(
                f"{what}: the event must be flyte.Cron, flyte.FixedRate or a source artifacts.Artifact handle, "
                f"got {type(event).__name__}"
            )
        return spec


def as_policies(refresh: Any) -> Tuple[Refresh, ...]:
    """`refresh=` normalized: a `Refresh`, a bare event, or a list of either."""
    if refresh is None:
        return ()
    items = list(refresh) if isinstance(refresh, (list, tuple)) else [refresh]
    return tuple(r if isinstance(r, Refresh) else Refresh(r) for r in items)


def record(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """The `materialize_on` record in `lineage.bindings` (the wire name the graph and snapshots read)."""
    return {"target": spec["target"], "event": spec["event"], "lag": spec["lag"]}


def _target_from_spec(spec: Mapping[str, Any]) -> ArtifactRef:
    dims: Dict[str, Any] = {}
    for d in spec["dims"]:
        if d[1] == "time":
            dims[d[0]] = next(g for g in GRANULARITY_MARKERS if g.registry == d[2])
        else:
            dims[d[0]] = int if d[1] == "int" else str
    return ArtifactRef(spec["target"], partitions=dims, project=spec.get("project"), domain=spec.get("domain"))


def _reuse_own_image(target: str) -> None:
    """
    Point the factory image at the image this task runs in. `flyte.materialize` deploys a derived factory, whose
    image is the factory image; this task already runs in it, so the deploy reuses it instead of building one
    from inside the cluster (which has no builder on a devbox, and needs none anywhere).
    """
    import os

    import flyte

    if os.environ.get("FLYTE_FACTORY_IMAGE"):
        return
    try:
        ctx = flyte.ctx()
        cache = ctx.compiled_image_cache if ctx is not None else None
    except Exception:
        cache = None
    uri = (cache.image_lookup if cache is not None else {}).get(f"refresh-{_slug(target, '-')}")
    if uri:
        os.environ["FLYTE_FACTORY_IMAGE"] = uri


def _body(spec: Mapping[str, Any], target: Artifact) -> Any:
    """The task function for `spec`: computes the partition from the trigger's inputs and materializes it."""
    import flyte

    scope = {"project": spec.get("project"), "domain": spec.get("domain")}
    event = spec["event"]
    if event["kind"] in ("cron", "fixed_rate"):
        tdim = target.time_dim
        assert tdim is not None
        gran = target.partitions[tdim]
        assert isinstance(gran, _Granularity)
        lag = spec.get("lag") or {}
        delta = timedelta(days=lag.get("days", 0), hours=lag.get("hours", 0))

        async def on_schedule(trigger_time: datetime) -> str:
            when = gran.floor(trigger_time - delta)
            _reuse_own_image(target.name)
            run = await flyte.materialize.aio(target, partitions={tdim: when}, **scope)
            return f"materializing {target.name}[{tdim}={gran.format(when)}]: {getattr(run, 'url', run)}"

        body: Any = on_schedule
    else:
        dims = dict(target.partitions)

        async def on_source(**values: Any) -> str:
            parts = {d: (t.floor(values[d]) if isinstance(t, _Granularity) else values[d]) for d, t in dims.items()}
            _reuse_own_image(target.name)
            run = await flyte.materialize.aio(target, partitions=parts, **scope)
            return f"materializing {target.name}{parts}: {getattr(run, 'url', run)}"

        params = [
            inspect.Parameter(
                d, inspect.Parameter.KEYWORD_ONLY, annotation=datetime if isinstance(t, _Granularity) else str
            )
            for d, t in dims.items()
        ]
        on_source.__signature__ = inspect.Signature(params, return_annotation=str)  # type: ignore[attr-defined]
        on_source.__annotations__ = {**{p.name: p.annotation for p in params}, "return": str}
        body = on_source
    body.__name__ = body.__qualname__ = spec["name"]
    body.__module__ = __name__
    when = (
        f"a new version of {event['source']}" + (f" with {event['filter']}" if event.get("filter") else "")
        if event["kind"] == "source"
        else (event.get("cron") or f"every {event.get('interval_minutes')} minutes")
    )
    body.__doc__ = f"Keep {target.name} fresh: materialize it on {when}."
    return body


def _encode(spec: Mapping[str, Any]) -> str:
    return base64.urlsafe_b64encode(json.dumps(spec, separators=(",", ":")).encode()).decode()


def _env(target: str, image: Any) -> "TaskEnvironment":
    import flyte

    return flyte.TaskEnvironment(
        name=f"refresh-{_slug(target, '-')}",
        image=image,
        resources=flyte.Resources(cpu="1", memory="1Gi"),
        description=f"Keeps {target} fresh (generated from its refresh policies)",
    )


def default_image() -> Any:
    """The refresh task's image: the factory image when flyteplugins-union is installed here (it honors
    `FLYTE_FACTORY_IMAGE` and `FLYTE_FACTORY_WHEEL`), else the default base plus flyteplugins-union from PyPI."""
    try:
        from flyteplugins.union.factory._internal.task import factory_image
    except ImportError:
        import flyte

        return flyte.Image.from_debian_base(name="refresh").with_pip_packages(PLUGIN_PACKAGE)
    return factory_image()


def build_refresh_task(spec: str, **_: Any) -> Any:
    """Resolver entry point: rebuild a refresh task from its encoded spec (no trigger: it already fired)."""
    from flyte._internal.resolvers.internal import InternalTaskResolver

    decoded = json.loads(base64.urlsafe_b64decode(spec.encode()))
    env = _env(decoded["target"], image=None)
    resolver = InternalTaskResolver(task_builder=_BUILDER, spec=spec)
    return env.task(task_resolver=resolver)(_body(decoded, _target_from_spec(decoded)))


def refresh_env(target: Artifact, policies: Sequence[Refresh], *, image: Any = None) -> "TaskEnvironment":
    """
    The generated `refresh-<target>` environment with one task (and its trigger) per policy.

    Repeated calls for the same target add to the same environment; a policy already in it is skipped, and
    two different policies with the same name fail.
    """
    from flyte._internal.resolvers.internal import InternalTaskResolver
    from flyte._trigger import OnArtifact, Trigger, TriggeredPartition, TriggerTime

    key = (target.name, target.project, target.domain)
    specs = [(p, p.spec(target)) for p in policies]
    env = _ENVS.get(key)
    if env is None:
        env = _ENVS[key] = _env(target.name, image if image is not None else default_image())
    for policy, spec in specs:
        existing = env.tasks.get(f"{env.name}.{spec['name']}")
        if existing is not None:
            func = getattr(existing, "func", None)
            prior = getattr(func, "__dict__", {}).get("_flyte_materialize_on", {}).get("record")
            if prior != record(spec):
                raise ValueError(
                    f"{target.name}: two refresh policies are named {spec['name']!r} ({prior} and {record(spec)}); "
                    "give one a name=."
                )
            continue
        body = _body(spec, target)
        event = policy.event
        source: Optional[Artifact] = event if is_handle(event) else None
        body._flyte_source = (policy.src_path, policy.src_line)
        body._flyte_materialize_on = {"target": target, "source": source, "record": record(spec)}
        if source is not None:
            automation: Any = OnArtifact(source, partitions=policy.filter or None)
            inputs: Dict[str, Any] = {d: TriggeredPartition(d) for d in target.partitions}
        else:
            automation, inputs = event, {"trigger_time": TriggerTime}
        trigger = Trigger(
            name=_slug(spec["name"], "-"),  # the task name with - for _, as a snapshot names it
            automation=automation,
            inputs=inputs,
            description=(body.__doc__ or "")[:255],
        )
        env.task(
            task_resolver=InternalTaskResolver(task_builder=_BUILDER, spec=_encode(spec)),
            triggers=(trigger,),
            labels={"lineage.consumes": target.name},
        )(body)
    return env


def _walk_envs(envs: Iterable[Any]) -> List[Any]:
    """`envs` and, transitively, their `depends_on` environments (each once), as a deploy plans them."""
    out: Dict[int, Any] = {}
    stack = list(envs)
    while stack:
        env = stack.pop(0)
        if id(env) in out:
            continue
        out[id(env)] = env
        stack.extend(getattr(env, "depends_on", None) or [])
    return list(out.values())


def refresh_envs(envs: Iterable[Any]) -> List["TaskEnvironment"]:
    """
    The refresh environments a deploy of `envs` brings along: one per artifact that a task in `envs` (or an
    environment they depend on) produces and that was declared with `refresh=`. Only a producing deploy registers
    an owner's policy; importing the handle elsewhere does not.
    """
    from flyte import TaskEnvironment

    out: Dict[int, TaskEnvironment] = {}
    all_envs = _walk_envs(envs)
    seen = {id(e) for e in all_envs}
    for env in all_envs:
        if not isinstance(env, TaskEnvironment):
            continue
        for task in env.tasks.values():
            produced = getattr(task, "produces_artifacts", None)
            for h in produced if isinstance(produced, tuple) else ():
                if is_handle(h) and h.refresh:
                    renv = refresh_env(h, h.refresh)
                    if id(renv) not in seen:
                        out[id(renv)] = renv
    return list(out.values())


def _describe_automation(automation: Any) -> str:
    from flyte._trigger import Cron, FixedRate, OnArtifact

    if isinstance(automation, Cron):
        tz = "" if automation.timezone == "UTC" else f" {automation.timezone}"
        return f"cron {automation.expression}{tz}"
    if isinstance(automation, FixedRate):
        return f"every {automation.interval_minutes}m"
    if isinstance(automation, OnArtifact):
        return f"on new {getattr(automation, 'name', '') or 'artifact'}"
    return str(automation)


def describe_refresh_envs(renvs: Iterable[Any]) -> List[str]:
    """
    One line per trigger a deploy adds through refresh environments, for the deploy summary:
    `+ refresh-daily-report: keep_report_fresh (cron 0 2 * * *)`.
    """
    lines: List[str] = []
    for env in renvs:
        for task in getattr(env, "tasks", {}).values():
            short = task.name.split(".", 1)[-1] if task.name.startswith(f"{env.name}.") else task.name
            for t in getattr(task, "triggers", None) or ():
                lines.append(f"+ {env.name}: {short} ({_describe_automation(t.automation)})")
    return lines
