"""Artifact-lineage graph collection.

Three kinds of node make up an artifact's lineage: **artifacts**, **runs**, and
**apps**. Two relations wire them together: ``produces`` (run/action -> artifact)
and ``consumes`` (artifact -> run/app).

The ``produces`` side is always knowable from the backend: every artifact
records its own provenance (:attr:`flyte.remote.Artifact.pb2.spec.source`, a
``task_action`` pointing at the run/action that created it), and
``Artifact.listall(source_run=...)`` answers "what did this run produce".

The ``consumes`` side is knowable from the backend *only* when the artifact
was bound as a typed run input — the platform stamps ``core.Literal.artifact_id``
on any literal that came from (or round-tripped through) an ``Artifact``, and
that survives as a run's input literal. That covers both manual
``flyte.run(task, model=artifact)`` calls and artifact-triggered runs (an
``OnArtifact`` trigger binding ``flyte.TriggeredArtifact`` to an input writes
the same stamp). See :func:`flyte.types._string_literals.artifact_annotation`.

Nothing else is knowable automatically — a task that resolves an artifact's
URI itself (rather than binding the ``Artifact``/``File``/``Dir`` as an input),
or an ``AppEnvironment`` that reads an artifact at startup, leaves no trace the
backend can query. For those cases this module falls back to a private label,
``__upstream_artifact__``, set to the artifact's ``tracker`` string
(``org/project/domain/name@version``). Runs pick it up via
``flyte.with_runcontext(labels={LABEL_UPSTREAM_ARTIFACT: artifact.tracker})``
(labels have been a first-class ``with_runcontext`` parameter all along); apps
pick it up via the ``labels=`` field on ``flyte.app.AppEnvironment``.

Lineage is not stored anywhere — it is re-derived on every scan from these two
signals, exactly as the reference ``RunLineageDashboard`` re-derives run
lineage from labels rather than a stored graph.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass, field
from typing import Iterable

logger = logging.getLogger(__name__)

# Private label key stamped on runs/apps that consume an artifact in a way the
# backend can't otherwise see. Value is the artifact's `tracker` string
# (org/project/domain/name@version).
LABEL_UPSTREAM_ARTIFACT = "__upstream_artifact__"

_TERMINAL_PHASES = {"succeeded", "failed", "aborted", "timed_out"}


@dataclass
class ArtifactNode:
    node_id: str
    name: str
    version: str
    tracker: str
    url: str
    description: str
    created_by: str
    produced_by_run: str = ""  # "" when externally published (no producing run)
    kind: str = "artifact"


@dataclass
class RunNode:
    node_id: str
    run_name: str
    task_name: str
    url: str
    phase: str
    kind: str = "run"


@dataclass
class AppNode:
    node_id: str
    app_name: str
    url: str
    endpoint: str
    kind: str = "app"


@dataclass
class Edge:
    source: str
    target: str
    relation: str  # "produces" | "consumes"


@dataclass
class LineageGraph:
    root: str  # node_id of the artifact the graph was built for
    nodes: dict = field(default_factory=dict)  # node_id -> asdict(node)
    edges: list = field(default_factory=list)  # list[asdict(Edge)]


def _artifact_node_id(tracker: str) -> str:
    return f"artifact:{tracker}"


def _run_node_id(run_name: str) -> str:
    return f"run:{run_name}"


def _app_node_id(app_name: str) -> str:
    return f"app:{app_name}"


def _artifact_to_node(artifact) -> ArtifactNode:
    src = artifact.pb2.spec.source
    produced_by_run = src.task_action.action.run.name if src.WhichOneof("source") == "task_action" else ""
    return ArtifactNode(
        node_id=_artifact_node_id(artifact.tracker),
        name=artifact.name,
        version=artifact.version,
        tracker=artifact.tracker,
        url=artifact.url,
        description=artifact.pb2.spec.info.description or "",
        created_by=artifact.created_by,
        produced_by_run=produced_by_run,
    )


def _run_to_node(run) -> RunNode:
    return RunNode(
        node_id=_run_node_id(run.name),
        run_name=run.name,
        task_name="",
        url=run.url,
        phase=run.phase.value,
    )


async def _app_to_node(app) -> AppNode:
    return AppNode(
        node_id=_app_node_id(app.name),
        app_name=app.name,
        url=app.url,
        endpoint=app.endpoint,
    )


def _run_task_name(run) -> str:
    return run.action.task_name or ""


async def _consumed_artifacts_of_run(run) -> list:
    """Every artifact this run consumed, via bound-input literals or the fallback label.

    Bound-input detection covers manual `flyte.run(task, model=artifact)` calls and
    `OnArtifact`-triggered runs alike (both stamp `Literal.artifact_id`). The label is
    the only signal for runs that resolved an artifact's value without binding it as a
    typed input.
    """
    from flyte.remote import Artifact

    trackers: set[str] = set()

    label_value = dict(run.pb2.labels).get(LABEL_UPSTREAM_ARTIFACT) if run.pb2.labels else None
    if label_value:
        trackers.add(label_value)

    try:
        literals = await run.input_literals.aio()
    except Exception:
        logger.debug("Could not fetch input literals for run %s", run.name, exc_info=True)
        literals = {}
    for lit in literals.values():
        if lit.HasField("artifact_id"):
            key = lit.artifact_id.key
            trackers.add(f"{key.org}/{key.project}/{key.domain}/{key.name}@{lit.artifact_id.version}")

    artifacts = []
    for tracker in trackers:
        try:
            _org, project, domain, rest = tracker.split("/", 3)
            name, version = rest.rsplit("@", 1)
            artifacts.append(await Artifact.get.aio(name=name, version=version, project=project, domain=domain))
        except Exception:
            logger.debug("Could not resolve consumed artifact %s", tracker, exc_info=True)
    return artifacts


async def _consumer_runs_of_artifact(artifact, watched_tasks: Iterable[str], scan_limit: int) -> list:
    """Runs that consumed `artifact`, found via the fallback label plus (optionally) a
    bound-input scan over `watched_tasks`'s recent runs."""
    from flyte.remote import Run

    by_name: dict[str, object] = {}

    async for run in Run.listall.aio(with_labels={LABEL_UPSTREAM_ARTIFACT: artifact.tracker}, limit=scan_limit):
        by_name[run.name] = run

    for task_name in watched_tasks:
        try:
            async for run in Run.listall.aio(task_name=task_name, limit=scan_limit, sort_by=("created_at", "desc")):
                if run.name in by_name:
                    continue
                consumed = await _consumed_artifacts_of_run(run)
                if any(a.tracker == artifact.tracker for a in consumed):
                    by_name[run.name] = run
        except Exception:
            logger.exception("Consumer scan failed for task %s", task_name)

    return list(by_name.values())


async def _consumer_apps_of_artifact(artifact, watched_apps: Iterable[str]) -> list:
    from flyte.remote import App

    apps = []
    for app_name in watched_apps:
        try:
            app = await App.get.aio(name=app_name)
        except Exception:
            logger.debug("Could not fetch app %s", app_name, exc_info=True)
            continue
        if dict(app.pb2.metadata.labels).get(LABEL_UPSTREAM_ARTIFACT) == artifact.tracker:
            apps.append(app)
    return apps


async def build_artifact_lineage(
    artifact,
    *,
    watched_tasks: Iterable[str] = (),
    watched_apps: Iterable[str] = (),
    scan_limit: int = 50,
    max_depth: int = 25,
) -> LineageGraph:
    """The full lineage of `artifact`: every ancestor back to the origin run/artifact,
    and every descendant run/artifact/app downstream of it.

    Walks two directions from the root artifact:

    - **upstream**: the artifact's producing run, that run's own consumed artifacts,
      their producing runs, and so on, until a run consumed nothing (the origin).
    - **downstream**: runs/apps that consumed the artifact, artifacts *those* runs in
      turn produced, their consumers, and so on.

    `watched_tasks`/`watched_apps` (task/app names) opt into the bound-input scan for
    consumer discovery beyond what the `__upstream_artifact__` label already finds —
    see the module docstring for why that scan can't be global.
    """
    from flyte.remote import Artifact, Run

    graph = LineageGraph(root=_artifact_node_id(artifact.tracker))
    seen_artifacts: set[str] = set()
    seen_runs: set[str] = set()

    def add_artifact(a) -> str:
        node = _artifact_to_node(a)
        graph.nodes.setdefault(node.node_id, asdict(node))
        return node.node_id

    def add_run(run) -> str:
        node = _run_to_node(run)
        node.task_name = _run_task_name(run)
        graph.nodes[node.node_id] = asdict(node)
        return node.node_id

    def add_edge(source: str, target: str, relation: str) -> None:
        edge = asdict(Edge(source=source, target=target, relation=relation))
        if edge not in graph.edges:
            graph.edges.append(edge)

    async def walk_upstream(a, depth: int) -> None:
        if a.tracker in seen_artifacts or depth > max_depth:
            return
        seen_artifacts.add(a.tracker)
        artifact_id = add_artifact(a)

        src = a.pb2.spec.source
        if src.WhichOneof("source") != "task_action":
            return  # externally published (or no source) — this is the origin
        run_name = src.task_action.action.run.name
        if not run_name or run_name in seen_runs:
            if run_name:
                add_edge(_run_node_id(run_name), artifact_id, "produces")
            return
        seen_runs.add(run_name)
        try:
            run = await Run.get.aio(name=run_name)
        except Exception:
            logger.debug("Could not fetch producing run %s", run_name, exc_info=True)
            return
        run_id = add_run(run)
        add_edge(run_id, artifact_id, "produces")

        consumed = await _consumed_artifacts_of_run(run)
        await asyncio.gather(*(_upstream_consumed(run_id, upstream, depth + 1) for upstream in consumed))

    async def _upstream_consumed(run_id: str, a, depth: int) -> None:
        artifact_id = add_artifact(a)
        add_edge(artifact_id, run_id, "consumes")
        await walk_upstream(a, depth)

    async def walk_downstream(a, depth: int) -> None:
        if depth > max_depth:
            return
        artifact_id = add_artifact(a)

        consumer_runs, consumer_apps = await asyncio.gather(
            _consumer_runs_of_artifact(a, watched_tasks, scan_limit),
            _consumer_apps_of_artifact(a, watched_apps),
        )

        for app in consumer_apps:
            app_node = await _app_to_node(app)
            graph.nodes.setdefault(app_node.node_id, asdict(app_node))
            add_edge(artifact_id, app_node.node_id, "consumes")

        for run in consumer_runs:
            run_id = add_run(run)
            add_edge(artifact_id, run_id, "consumes")
            if run.name in seen_runs:
                continue
            seen_runs.add(run.name)
            produced = []
            try:
                async for produced_artifact in Artifact.listall.aio(source_run=run.name):
                    produced.append(produced_artifact)
            except Exception:
                logger.exception("Could not list artifacts produced by run %s", run.name)
            await asyncio.gather(*(_downstream_produced(run_id, p, depth + 1) for p in produced))

    async def _downstream_produced(run_id: str, a, depth: int) -> None:
        artifact_id = add_artifact(a)
        add_edge(run_id, artifact_id, "produces")
        if a.tracker not in seen_artifacts:
            seen_artifacts.add(a.tracker)
            await walk_downstream(a, depth)

    await walk_upstream(artifact, 0)
    await walk_downstream(artifact, 0)

    return graph
