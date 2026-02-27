import importlib
import logging
import os
import threading
import typing
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, List

import airflow
from pathlib import Path
import airflow.models as airflow_models
import airflow.sensors.base as airflow_sensors
import jsonpickle
import airflow.triggers.base as airflow_triggers
import airflow.utils.context as airflow_context
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import flyte
from flyte import logger, get_custom_context
from flyte._context import internal_ctx, root_context_var
from flyte._internal.controllers import get_controller
from flyte._internal.controllers._local_controller import _TaskRunner
from flyte._internal.resolvers.common import Resolver
from flyte._module import extract_obj_module
from flyte._task import TaskTemplate
from flyte.extend import AsyncFunctionTaskTemplate, TaskPluginRegistry
from flyte.models import SerializationContext, NativeInterface

# Per-thread _TaskRunner instances used by _flyte_operator for sync blocking submission.
_airflow_runners: Dict[str, _TaskRunner] = {}

# Import dag module to apply DAG monkey-patches when this module is imported.
from flyteplugins.airflow import dag as _dag_module  # noqa: E402


@dataclass
class AirflowTaskMetadata(object):
    """
    This class is used to store the Airflow task configuration. It is serialized and stored in the Flyte task config.
    It can be trigger, hook, operator or sensor. For example:

    from airflow.sensors.filesystem import FileSensor
    sensor = FileSensor(task_id="id", filepath="/tmp/1234")

    In this case, the attributes of AirflowObj will be:
    module: airflow.sensors.filesystem
    name: FileSensor
    parameters: {"task_id": "id", "filepath": "/tmp/1234"}
    """

    module: str
    name: str
    parameters: typing.Dict[str, Any]


class AirflowTaskResolver(Resolver):
    """
    This class is used to resolve an Airflow task. It will load an airflow task in the container.
    """

    @property
    def import_path(self) -> str:
        return "flyteplugins.airflow.task.AirflowTaskResolver"

    def load_task(self, loader_args: typing.List[str]) -> AsyncFunctionTaskTemplate:
        """
        This method is used to load an Airflow task.
        """
        _, airflow_task_module, _, airflow_task_name, _, airflow_task_parameters, _, func_module, _, func_name = loader_args
        func_module = importlib.import_module(name=func_module)
        func_def = getattr(func_module, func_name)
        return AirflowFunctionTask(
            name=airflow_task_name,
            airflow_task_metadata=AirflowTaskMetadata(
                module=airflow_task_module,
                name=airflow_task_name,
                parameters=jsonpickle.decode(airflow_task_parameters)
            ),
            func=func_def,
        )

    def loader_args(self, task: "AirflowFunctionTask", root_dir: Path) -> List[str]:  # type:ignore
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
            task.func.__name__
        ]


class AirflowRawContainerTask(TaskTemplate):
    """
    Running Bash command in the container.
    """

    def __init__(
        self,
        name: str,
        airflow_task_metadata: AirflowTaskMetadata,
        command: str,
        # inputs: Optional[Dict[str, Type]] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            interface=NativeInterface(inputs={}, outputs={}),
            **kwargs,
        )
        self.resolver = AirflowTaskResolver()
        self._airflow_task_metadata = airflow_task_metadata
        self._command = command
        self._call_as_synchronous = True
        self._downstream_flyte_tasks: List["AirflowFunctionTask"] = []

    # ------------------------------------------------------------------
    # Airflow dependency-arrow support (>> / <<)
    # Records the dependency in the active FlyteDAG if one is being built.
    # ------------------------------------------------------------------

    def __rshift__(self, other: "AirflowFunctionTask") -> "AirflowFunctionTask":
        """``self >> other`` — other runs after self."""
        if _dag_module._current_flyte_dag is not None:
            _dag_module._current_flyte_dag.set_dependency(self.name, other.name)
        return other

    def __lshift__(self, other: "AirflowFunctionTask") -> "AirflowFunctionTask":
        """``self << other`` — self runs after other."""
        if _dag_module._current_flyte_dag is not None:
            _dag_module._current_flyte_dag.set_dependency(other.name, self.name)
        return other

    def container_args(self, sctx: SerializationContext) -> List[str]:
        return self._command.split()

    async def execute(self, **kwargs) -> Any:
        # ExecutorSafeguard stores a sentinel in a threading.local() dict. That
        # dict is initialised on the main thread at import time, but tasks may
        # run in a background thread where the thread-local has no 'callers' key.
        from airflow.models.baseoperator import ExecutorSafeguard
        if not hasattr(ExecutorSafeguard._sentinel, "callers"):
            ExecutorSafeguard._sentinel.callers = {}
        logger.info("Executing Airflow task")
        return _get_airflow_instance(self._airflow_task_metadata).execute(context=airflow_context.Context())


class AirflowFunctionTask(AsyncFunctionTaskTemplate):
    """
    This python container task is used to wrap an Airflow task. It is used to run an Airflow task in a container.
    The airflow task module, name and parameters are stored in the task config.

    Some of the Airflow operators are not deferrable, For example, BeamRunJavaPipelineOperator, BeamRunPythonPipelineOperator.
    These tasks don't have an async method to get the job status, so cannot be used in the Flyte connector. We run these tasks in a container.
    """

    def __init__(
        self,
        name: str,
        airflow_task_metadata: AirflowTaskMetadata,
        func: Optional[callable],
        # inputs: Optional[Dict[str, Type]] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            # plugin_config=plugin_config,
            func=func,
            interface=NativeInterface(inputs={}, outputs={}),
            **kwargs,
        )
        self.resolver = AirflowTaskResolver()
        self.airflow_task_metadata = airflow_task_metadata
        self._call_as_synchronous = True
        self._downstream_flyte_tasks: List["AirflowFunctionTask"] = []

    # ------------------------------------------------------------------
    # Airflow dependency-arrow support (>> / <<)
    # Records the dependency in the active FlyteDAG if one is being built.
    # ------------------------------------------------------------------

    def __rshift__(self, other: "AirflowFunctionTask") -> "AirflowFunctionTask":
        """``self >> other`` — other runs after self."""
        if _dag_module._current_flyte_dag is not None:
            _dag_module._current_flyte_dag.set_dependency(self.name, other.name)
        return other

    def __lshift__(self, other: "AirflowFunctionTask") -> "AirflowFunctionTask":
        """``self << other`` — self runs after other."""
        if _dag_module._current_flyte_dag is not None:
            _dag_module._current_flyte_dag.set_dependency(other.name, self.name)
        return other

    async def execute(self, **kwargs) -> Any:
        # ExecutorSafeguard stores a sentinel in a threading.local() dict. That
        # dict is initialised on the main thread at import time, but tasks may
        # run in a background thread where the thread-local has no 'callers' key.
        from airflow.models.baseoperator import ExecutorSafeguard
        if not hasattr(ExecutorSafeguard._sentinel, "callers"):
            ExecutorSafeguard._sentinel.callers = {}
        logger.info("Executing Airflow task")
        self.airflow_task_metadata.parameters["python_callable"] = self.func
        return _get_airflow_instance(self.airflow_task_metadata).execute(context=airflow_context.Context())


def _get_airflow_instance(
    airflow_task_metadata: AirflowTaskMetadata,
) -> typing.Union[airflow_models.BaseOperator, airflow_sensors.BaseSensorOperator, airflow_triggers.BaseTrigger]:
    # Set the GET_ORIGINAL_TASK attribute to True so that obj_def will return the original
    # airflow task instead of the Flyte task.
    with flyte.custom_context(GET_ORIGINAL_TASK="True"):
        obj_module = importlib.import_module(name=airflow_task_metadata.module)
        obj_def = getattr(obj_module, airflow_task_metadata.name)
        return obj_def(**airflow_task_metadata.parameters)


def _flyte_operator(*args, **kwargs):
    """
    This function is called by the Airflow operator to create a new task. We intercept this call and return a Flyte
    task instead.
    """
    cls = args[0]
    try:
        if get_custom_context().get("GET_ORIGINAL_TASK", "False") == "True":
            # Return an original task when running in the connector.
            print("Returning original Airflow task")
            return object.__new__(cls)
    except AssertionError:
        # This happens when the task is created in the dynamic workflow.
        # We don't need to return the original task in this case.
        logging.debug("failed to get the attribute GET_ORIGINAL_TASK from user space params")

    container_image = kwargs.pop("container_image", None)
    task_id = kwargs.get("task_id", cls.__name__)
    airflow_task_metadata = AirflowTaskMetadata(module=cls.__module__, name=cls.__name__, parameters=kwargs)

    if cls == BashOperator:
        command = kwargs.get("bash_command", "")
        task = AirflowRawContainerTask(name=task_id, airflow_task_metadata=airflow_task_metadata, command=command)
    elif cls == PythonOperator:
        func = kwargs.get("python_callable", None)
        kwargs.pop("python_callable", None)
        task = AirflowFunctionTask(name=task_id, airflow_task_metadata=airflow_task_metadata, func=func, image=container_image)
    else:
        raise ValueError(f"Unsupported Airflow operator: {cls.__name__}")

    # ── Case 1: inside a ``with DAG(...) as dag:`` block ────────────────────
    # Register the task with the active FlyteDAG collector so it can be wired
    # into the Flyte workflow when the DAG context exits.  Do NOT execute yet.
    if _dag_module._current_flyte_dag is not None:
        _dag_module._current_flyte_dag.add_task(task_id, task)
        return task

    # ── Case 2: inside a Flyte task execution ───────────────────────────────
    # The dag workflow function is executing; submit the operator as a sub-task.
    if internal_ctx().is_task_context():
        return task()

    # ── Case 3: outside any context (e.g. serialization / import scan) ──────
    return task


# Monkey patches the Airflow operator. Instead of creating an airflow task, it returns a Flyte task.
airflow_models.BaseOperator.__new__ = _flyte_operator
