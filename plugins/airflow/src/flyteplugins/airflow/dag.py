"""
Monkey-patches airflow.DAG so that a standard Airflow DAG definition is transparently
converted into a runnable Flyte task, with no changes to the DAG code required.

Usage
-----
    from flyteplugins.airflow.task import AirflowContainerTask  # triggers patches
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    import flyte

    env = flyte.TaskEnvironment(name="hello_airflow", image=...)

    with DAG(dag_id="my_dag", flyte_env=env) as dag:
        t1 = BashOperator(task_id="step1", bash_command='echo step1')
        t2 = BashOperator(task_id="step2", bash_command='echo step2')
        t1 >> t2          # optional: explicit dependency

    if __name__ == "__main__":
        flyte.init_from_config()
        run = dag.run(mode="local")
        print(run.url)

Notes
-----
- ``flyte_env`` is an optional kwarg accepted by the patched DAG. If omitted a
  default ``TaskEnvironment(name=dag_id)`` is created.
- Operator dependency arrows (``>>``, ``<<``) update the execution order.
  If no explicit dependencies are declared the operators run in definition order.
- ``dag.run(**kwargs)`` is a convenience wrapper around
  ``flyte.with_runcontext(**kwargs).run(dag.flyte_task)``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional

import airflow.models.dag as _airflow_dag_module

if TYPE_CHECKING:
    from flyteplugins.airflow.task import AirflowContainerTask

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

#: Set when the code is inside a ``with DAG(...) as dag:`` block.
_current_flyte_dag: Optional["FlyteDAG"] = None


# ---------------------------------------------------------------------------
# FlyteDAG – collects operators and builds the Flyte workflow task
# ---------------------------------------------------------------------------

class FlyteDAG:
    """Collects Airflow operators during a DAG definition and converts them
    into a single Flyte task that runs them in dependency order."""

    def __init__(self, dag_id: str, env=None) -> None:
        self.dag_id = dag_id
        self.env = env
        # Ordered dict preserves insertion (creation) order as the default.
        self._tasks: Dict[str, "AirflowContainerTask"] = {}
        # task_id -> set of upstream task_ids
        self._upstream: Dict[str, Set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # Registration (called by _flyte_operator during DAG definition)
    # ------------------------------------------------------------------

    def add_task(self, task_id: str, task: "AirflowContainerTask") -> None:
        self._tasks[task_id] = task
        # Ensure a dependency entry exists even with no upstream tasks.
        _ = self._upstream[task_id]

    def set_dependency(self, upstream_id: str, downstream_id: str) -> None:
        """Record that *upstream_id* must run before *downstream_id*."""
        self._upstream[downstream_id].add(upstream_id)

    # ------------------------------------------------------------------
    # Flyte task construction
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Annotate each task with its downstream tasks and create a Flyte
        workflow task whose entry function calls only the root tasks.

        Each root task's execute() will trigger its downstream tasks in
        parallel via asyncio.gather, propagating the chain automatically.
        """
        import flyte

        env = self.env
        if env is None:
            env = flyte.TaskEnvironment(name=self.dag_id)

        # Build downstream map from the upstream map.
        downstream: Dict[str, List[str]] = defaultdict(list)
        for tid, upstreams in self._upstream.items():
            for up in upstreams:
                downstream[up].append(tid)

        # Annotate each AirflowContainerTask with its downstream tasks.
        for tid, task in self._tasks.items():
            task._downstream_flyte_tasks = [
                self._tasks[d] for d in downstream[tid] if d in self._tasks
            ]

        # Root tasks: those with no upstream dependencies.
        root_tasks = [
            self._tasks[tid]
            for tid, ups in self._upstream.items()
            if len(ups) == 0
        ]

        # Snapshot to avoid capturing mutable references in the closure.
        root_snapshot = list(root_tasks)

        def _dag_entry() -> None:
            for task in root_snapshot:
                task()  # _call_as_synchronous=True → submit_sync → blocks until done

        _dag_entry.__name__ = f"dag_{self.dag_id}"
        _dag_entry.__qualname__ = f"dag_{self.dag_id}"

        self.flyte_task = env.task(_dag_entry)


# ---------------------------------------------------------------------------
# DAG monkey-patch helpers
# ---------------------------------------------------------------------------

_original_dag_init = _airflow_dag_module.DAG.__init__
_original_dag_enter = _airflow_dag_module.DAG.__enter__
_original_dag_exit = _airflow_dag_module.DAG.__exit__


def _patched_dag_init(self, *args, **kwargs) -> None:  # type: ignore[override]
    # Pull out our custom kwarg before passing the rest to Airflow.
    flyte_env = kwargs.pop("flyte_env", None)
    _original_dag_init(self, *args, **kwargs)
    self._flyte_env = flyte_env


def _patched_dag_enter(self):  # type: ignore[override]
    global _current_flyte_dag
    _current_flyte_dag = FlyteDAG(dag_id=self.dag_id, env=getattr(self, "_flyte_env", None))
    return _original_dag_enter(self)


def _patched_dag_exit(self, exc_type, exc_val, exc_tb):  # type: ignore[override]
    global _current_flyte_dag
    try:
        if exc_type is None and _current_flyte_dag is not None:
            flyte_dag = _current_flyte_dag
            flyte_dag.build()
            # Attach the Flyte task and a convenience run() to the DAG object.
            self.flyte_task = flyte_dag.flyte_task
            self.run = _make_run(flyte_dag.flyte_task)
    finally:
        _current_flyte_dag = None

    return _original_dag_exit(self, exc_type, exc_val, exc_tb)


def _make_run(flyte_task):
    """Return a ``run(**kwargs)`` helper bound to *flyte_task*."""
    import flyte

    def run(**kwargs):
        return flyte.with_runcontext(**kwargs).run(flyte_task)

    return run


# ---------------------------------------------------------------------------
# Apply patches
# ---------------------------------------------------------------------------

_airflow_dag_module.DAG.__init__ = _patched_dag_init
_airflow_dag_module.DAG.__enter__ = _patched_dag_enter
_airflow_dag_module.DAG.__exit__ = _patched_dag_exit
