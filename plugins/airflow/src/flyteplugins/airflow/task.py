import importlib
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import flyte
import jsonpickle
from flyte import get_custom_context, logger
from flyte._context import internal_ctx
from flyte._internal.resolvers.common import Resolver
from flyte._module import extract_obj_module
from flyte._task import TaskTemplate
from flyte.extend import AsyncFunctionTaskTemplate
from flyte.models import NativeInterface, SerializationContext

import airflow.models as airflow_models
import airflow.sensors.base as airflow_sensors
import airflow.triggers.base as airflow_triggers
import airflow.utils.context as airflow_context
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Import dag module to apply DAG monkey-patches when this module is imported.
from flyteplugins.airflow import dag as _dag_module

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class AirflowTaskMetadata:
    """Stores the Airflow operator class location and constructor kwargs.

    For example, given::

        FileSensor(task_id="id", filepath="/tmp/1234")

    the fields would be:
        module: "airflow.sensors.filesystem"
        name: "FileSensor"
        parameters: {"task_id": "id", "filepath": "/tmp/1234"}
    """

    module: str
    name: str
    parameters: typing.Dict[str, Any]


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class AirflowPythonTaskResolver(Resolver):
    """Resolves an AirflowPythonFunctionTask on the remote worker.

    The resolver records the Airflow operator metadata and the wrapped Python
    callable so that the task can be reconstructed from loader args alone.
    """

    @property
    def import_path(self) -> str:
        return "flyteplugins.airflow.task.AirflowPythonTaskResolver"

    def load_task(self, loader_args: typing.List[str]) -> AsyncFunctionTaskTemplate:
        _, airflow_task_module, _, airflow_task_name, _, airflow_task_parameters, _, func_module, _, func_name = (
            loader_args
        )
        func_module = importlib.import_module(name=func_module)
        func_def = getattr(func_module, func_name)
        return AirflowPythonFunctionTask(
            name=airflow_task_name,
            airflow_task_metadata=AirflowTaskMetadata(
                module=airflow_task_module,
                name=airflow_task_name,
                parameters=jsonpickle.decode(airflow_task_parameters),
            ),
            func=func_def,
        )

    def loader_args(self, task: "AirflowPythonFunctionTask", root_dir: Path) -> List[str]:  # type:ignore
        entity_module_name, _ = extract_obj_module(task.func, root_dir)
        return [
            "airflow-task-module",
            task.airflow_task_metadata.module,
            "airflow-task-name",
            task.airflow_task_metadata.name,
            "airflow-task-parameters",
            jsonpickle.encode(task.airflow_task_metadata.parameters),
            "airflow-func-module",
            entity_module_name,
            "airflow-func-name",
            task.func.__name__,
        ]


# ---------------------------------------------------------------------------
# Shared task behaviour (mixin)
# ---------------------------------------------------------------------------


class _AirflowTaskMixin:
    """Shared behaviour for both raw-container and function Airflow tasks.

    Provides Airflow-style dependency arrows (``>>`` / ``<<``) and the
    ``ExecutorSafeguard`` workaround needed when tasks run on background threads.
    """

    def _init_airflow_mixin(self) -> None:
        self._call_as_synchronous = True

    # Airflow dependency-arrow support (>> / <<)
    # Records the dependency in the active FlyteDAG if one is being built.

    def __rshift__(self, other: "AirflowPythonFunctionTask") -> "AirflowPythonFunctionTask":
        """``self >> other`` — other runs after self."""
        if _dag_module._state[_dag_module._CURRENT_FLYTE_DAG] is not None:
            _dag_module._state[_dag_module._CURRENT_FLYTE_DAG].set_dependency(self.name, other.name)
        return other

    def __lshift__(self, other: "AirflowPythonFunctionTask") -> "AirflowPythonFunctionTask":
        """``self << other`` — self runs after other."""
        if _dag_module._state[_dag_module._CURRENT_FLYTE_DAG] is not None:
            _dag_module._state[_dag_module._CURRENT_FLYTE_DAG].set_dependency(other.name, self.name)
        return other

    @staticmethod
    def _patch_executor_safeguard() -> None:
        """Ensure ExecutorSafeguard's thread-local has a ``callers`` dict.

        ExecutorSafeguard stores a sentinel in a ``threading.local()`` dict
        that is initialised on the main thread at import time. Tasks may run
        on a background thread where the thread-local has no ``callers`` key.
        """
        from airflow.models.baseoperator import ExecutorSafeguard

        if not hasattr(ExecutorSafeguard._sentinel, "callers"):
            ExecutorSafeguard._sentinel.callers = {}


# ---------------------------------------------------------------------------
# Task classes
# ---------------------------------------------------------------------------


class AirflowShellTask(_AirflowTaskMixin, TaskTemplate):
    """Wraps an Airflow BashOperator as a Flyte raw-container task."""

    def __init__(
        self,
        name: str,
        airflow_task_metadata: AirflowTaskMetadata,
        command: str,
        **kwargs,
    ):
        super().__init__(
            name=name,
            interface=NativeInterface(inputs={}, outputs={}),
            **kwargs,
        )
        self._init_airflow_mixin()
        self._airflow_task_metadata = airflow_task_metadata
        self._command = command

    def container_args(self, sctx: SerializationContext) -> List[str]:
        return self._command.split()

    async def execute(self, **kwargs) -> Any:
        self._patch_executor_safeguard()
        logger.info("Executing Airflow bash operator")
        _get_airflow_instance(self._airflow_task_metadata).execute(context=airflow_context.Context())


class AirflowPythonFunctionTask(_AirflowTaskMixin, AsyncFunctionTaskTemplate):
    """Wraps an Airflow PythonOperator as a Flyte function task.

    The airflow task module, name, and parameters are stored in the task
    config.  Some Airflow operators are not deferrable (e.g.
    ``BeamRunJavaPipelineOperator``).  These tasks lack an async method to
    poll job status so they cannot use the Flyte connector — we run them in
    a container instead.
    """

    def __init__(
        self,
        name: str,
        airflow_task_metadata: AirflowTaskMetadata,
        func: Optional[callable],
        **kwargs,
    ):
        super().__init__(
            name=name,
            func=func,
            interface=NativeInterface(inputs={}, outputs={}),
            **kwargs,
        )
        self._init_airflow_mixin()
        self.resolver = AirflowPythonTaskResolver()
        self.airflow_task_metadata = airflow_task_metadata

    async def execute(self, **kwargs) -> Any:
        logger.info("Executing Airflow python task")
        self.airflow_task_metadata.parameters["python_callable"] = self.func
        _get_airflow_instance(self.airflow_task_metadata).execute(context=airflow_context.Context())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_airflow_instance(
    airflow_task_metadata: AirflowTaskMetadata,
) -> typing.Union[airflow_models.BaseOperator, airflow_sensors.BaseSensorOperator, airflow_triggers.BaseTrigger]:
    """Instantiate the original Airflow operator from its metadata."""
    # Set GET_ORIGINAL_TASK so that obj_def returns the real Airflow
    # operator instead of being intercepted by _flyte_operator.
    with flyte.custom_context(GET_ORIGINAL_TASK="True"):
        obj_module = importlib.import_module(name=airflow_task_metadata.module)
        obj_def = getattr(obj_module, airflow_task_metadata.name)
        return obj_def(**airflow_task_metadata.parameters)


# ---------------------------------------------------------------------------
# Operator intercept (monkey-patch)
# ---------------------------------------------------------------------------


def _flyte_operator(*args, **kwargs):
    """Intercept Airflow operator construction and return a Flyte task instead.

    Called via the monkey-patched ``BaseOperator.__new__``.  Depending on
    context this either registers the task with an active FlyteDAG, submits
    it as a sub-task during execution, or returns the task object for later
    serialization.
    """
    cls = args[0]
    if get_custom_context().get("GET_ORIGINAL_TASK", "False") == "True":
        logger.debug("Returning original Airflow task")
        return object.__new__(cls)

    container_image = kwargs.pop("container_image", None)
    task_id = kwargs.get("task_id", cls.__name__)
    airflow_task_metadata = AirflowTaskMetadata(module=cls.__module__, name=cls.__name__, parameters=kwargs)

    if cls == BashOperator:
        command = kwargs.get("bash_command", "")
        task = AirflowShellTask(name=task_id, airflow_task_metadata=airflow_task_metadata, command=command)
    elif cls == PythonOperator:
        func = kwargs.get("python_callable", None)
        kwargs.pop("python_callable", None)
        task = AirflowPythonFunctionTask(
            name=task_id, airflow_task_metadata=airflow_task_metadata, func=func, image=container_image
        )
    else:
        raise ValueError(f"Unsupported Airflow operator: {cls.__name__}")

    # Case 1: inside a ``with DAG(...) as dag:`` block — register with FlyteDAG.
    if _dag_module._state[_dag_module._CURRENT_FLYTE_DAG] is not None:
        _dag_module._state[_dag_module._CURRENT_FLYTE_DAG].add_task(task_id, task)
        return task

    # Case 2: inside a Flyte task execution — submit the operator as a sub-task.
    if internal_ctx().is_task_context():
        return task()

    # Case 3: outside any context (e.g. serialization / import scan).
    return task


# Monkey-patch: intercept Airflow operator construction.
airflow_models.BaseOperator.__new__ = _flyte_operator
