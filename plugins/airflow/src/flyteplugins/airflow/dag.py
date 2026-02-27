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
        t1 >> t2 # optional: explicit dependency

    if __name__ == "__main__":
        flyte.init_from_config()
        run = flyte.with_runcontext(mode="remote", log_level="10").run(dag)
        print(run.url)

Notes
-----
- ``flyte_env`` is an optional kwarg accepted by the patched DAG. If omitted a
  default ``TaskEnvironment`` is created using the dag_id as the name and a
  Debian-base image with ``flyteplugins-airflow`` and ``jsonpickle`` installed.
- Operator dependency arrows (``>>``, ``<<``) update the execution order.
  If no explicit dependencies are declared, the operators run in definition order.
"""

from __future__ import annotations

import inspect
import logging
import sys as _sys
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from flyte._task import TaskTemplate

import airflow.models.dag as _airflow_dag_module

if TYPE_CHECKING:
    import types

    from flyteplugins.airflow.task import AirflowPythonFunctionTask, AirflowShellTask

    AirflowTask = AirflowPythonFunctionTask | AirflowShellTask

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_CURRENT_FLYTE_DAG = "current_flyte_dag"

#: Mutable container for the active FlyteDAG (set inside a ``with DAG(...)`` block).
#: Using a dict avoids ``global`` statements in the patch functions.
_state: Dict[str, Optional["FlyteDAG"]] = {_CURRENT_FLYTE_DAG: None}


# ---------------------------------------------------------------------------
# FlyteDAG - collects operators and builds the Flyte workflow task
# ---------------------------------------------------------------------------


class FlyteDAG:
    """Collects Airflow operators during a DAG definition and converts them
    into a single Flyte task that runs them in dependency order."""

    def __init__(self, dag_id: str, env=None) -> None:
        self.dag_id = dag_id
        self.env = env
        # Ordered dict preserves insertion (creation) order as the default.
        self._tasks: Dict[str, "AirflowTask"] = {}
        # task_id -> set of upstream task_ids
        self._upstream: Dict[str, Set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # Registration (called by _flyte_operator during DAG definition)
    # ------------------------------------------------------------------

    def add_task(self, task_id: str, task: "AirflowTask") -> None:
        self._tasks[task_id] = task
        # Ensure a dependency entry exists even with no upstream tasks.
        _ = self._upstream[task_id]

    def set_dependency(self, upstream_id: str, downstream_id: str) -> None:
        """Record that *upstream_id* must run before *downstream_id*."""
        self._upstream[downstream_id].add(upstream_id)

    # ------------------------------------------------------------------
    # Flyte task construction
    # ------------------------------------------------------------------

    def _build_downstream_map(self) -> Dict[str, List[str]]:
        """Invert ``self._upstream`` to get downstream adjacency lists."""
        downstream: Dict[str, List[str]] = defaultdict(list)
        for tid, upstreams in self._upstream.items():
            for up in upstreams:
                downstream[up].append(tid)
        return downstream

    def _find_caller_module(self) -> Tuple[str, Optional["types.ModuleType"]]:
        """Walk the call stack to find the first frame outside this module.

        The Flyte task must be registered under the *user's* module so that
        ``DefaultTaskResolver`` can locate it via ``getattr(module, name)``
        on the remote worker (which re-imports the module and re-runs the
        DAG definition).
        """
        for fi in inspect.stack():
            mod = fi.frame.f_globals.get("__name__", "")
            if mod and mod != __name__:
                return mod, _sys.modules.get(mod)
        return __name__, None

    def _create_dag_entry(
        self,
        all_tasks: Dict[str, "AirflowTask"],
        downstream_map: Dict[str, List[str]],
        root_tasks: List["AirflowTask"],
    ):
        """Build the async entry function that orchestrates task execution."""
        # Snapshot to avoid capturing mutable references in the closure.
        root_snapshot = list(root_tasks)
        downstream_snapshot = dict(downstream_map)

        async def _dag_entry() -> None:
            import asyncio

            async def _run_chain(task):
                await task.aio()
                ds = downstream_snapshot.get(task.name, [])
                if ds:
                    await asyncio.gather(*[_run_chain(all_tasks[d]) for d in ds])

            await asyncio.gather(*[_run_chain(t) for t in root_snapshot])

        caller_module_name, caller_module = self._find_caller_module()
        _dag_entry.__name__ = f"dag_{self.dag_id}"
        _dag_entry.__qualname__ = f"dag_{self.dag_id}"
        _dag_entry.__module__ = caller_module_name
        return _dag_entry, caller_module

    def build(self) -> None:
        """Create a Flyte workflow task whose entry function runs all
        operator tasks in dependency order.

        The entry function captures the full dependency graph in its closure
        and orchestrates execution directly, starting from root tasks and
        chaining downstream tasks after each completes.  This ensures
        correct ordering in both local and remote execution (where sub-tasks
        are resolved independently and lose their in-memory references).
        """
        import flyte

        if self.env is None:
            self.env = flyte.TaskEnvironment(
                name=self.dag_id,
                image=flyte.Image.from_debian_base()
                .with_pip_packages("apache-airflow<3.0.0", "jsonpickle")
                .with_local_v2(),
            )

        downstream = self._build_downstream_map()

        # Root tasks: those with no upstream dependencies.
        root_tasks = [self._tasks[tid] for tid, ups in self._upstream.items() if len(ups) == 0]

        _dag_entry, caller_module = self._create_dag_entry(
            all_tasks=dict(self._tasks),
            downstream_map=downstream,
            root_tasks=root_tasks,
        )

        # Set image and register operator tasks with the DAG's TaskEnvironment.
        for _op_task in self._tasks.values():
            if _op_task.image is None:
                _op_task.image = self.env.image
            self.env.add_dependency(flyte.TaskEnvironment.from_task(_op_task.name, _op_task))

        self.flyte_task = self.env.task(_dag_entry)

        # Inject the task into the caller's module so DefaultTaskResolver
        # can find it via getattr(module, task_name) on both local and remote.
        if caller_module is not None:
            setattr(caller_module, _dag_entry.__name__, self.flyte_task)


# ---------------------------------------------------------------------------
# Proxy class — makes DAG instances pass isinstance(dag, TaskTemplate)
# ---------------------------------------------------------------------------


# All names defined on TaskTemplate (fields, methods, properties) that should
# be proxied to the underlying flyte_task rather than resolved on the DAG.
_TASK_TEMPLATE_NAMES = frozenset(name for name in dir(TaskTemplate) if not name.startswith("_")) | frozenset(
    TaskTemplate.__dataclass_fields__.keys()
)


class _FlyteDAG(_airflow_dag_module.DAG, TaskTemplate):
    """Makes an Airflow DAG pass ``isinstance(dag, TaskTemplate)`` by proxying
    TaskTemplate attribute access to the attached ``flyte_task``.
    """

    def __getattribute__(self, name):
        if name in _TASK_TEMPLATE_NAMES:
            ft = object.__getattribute__(self, "__dict__").get("flyte_task")
            if ft is not None:
                return getattr(ft, name)
        return super().__getattribute__(name)

_original_dag_init = _airflow_dag_module.DAG.__init__
_original_dag_enter = _airflow_dag_module.DAG.__enter__
_original_dag_exit = _airflow_dag_module.DAG.__exit__


def _patched_dag_init(self, *args, **kwargs) -> None:  # type: ignore[override]
    # Pull out our custom kwarg before passing the rest to Airflow.
    flyte_env = kwargs.pop("flyte_env", None)
    _original_dag_init(self, *args, **kwargs)
    self._flyte_env = flyte_env


def _patched_dag_enter(self):  # type: ignore[override]
    _state[_CURRENT_FLYTE_DAG] = FlyteDAG(dag_id=self.dag_id, env=getattr(self, "_flyte_env", None))
    return _original_dag_enter(self)


def _patched_dag_exit(self, exc_type, exc_val, exc_tb):  # type: ignore[override]
    try:
        if exc_type is None and _state[_CURRENT_FLYTE_DAG] is not None:
            flyte_dag = _state[_CURRENT_FLYTE_DAG]
            flyte_dag.build()
            # Attach the Flyte task and a convenience run() to the DAG object,
            # then swap __class__ so the DAG passes isinstance(dag, TaskTemplate).
            self.flyte_task = flyte_dag.flyte_task
            self.run = _make_run(flyte_dag.flyte_task)
            self.__class__ = _FlyteDAG
    finally:
        _state[_CURRENT_FLYTE_DAG] = None

    return _original_dag_exit(self, exc_type, exc_val, exc_tb)


def _make_run(flyte_task):
    """Return a ``run(**kwargs)`` helper bound to *flyte_task*."""
    import flyte

    def run(**kwargs):
        return flyte.with_runcontext(**kwargs).run(flyte_task)

    return run


_airflow_dag_module.DAG.__init__ = _patched_dag_init
_airflow_dag_module.DAG.__enter__ = _patched_dag_enter
_airflow_dag_module.DAG.__exit__ = _patched_dag_exit
