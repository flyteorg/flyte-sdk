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

import flyte
from flyte import logger, get_custom_context
from flyte._context import internal_ctx, root_context_var
from flyte._internal.controllers import get_controller
from flyte._internal.controllers._local_controller import _TaskRunner
from flyte._internal.resolvers.common import Resolver
from flyte._task import TaskTemplate
from flyte.extend import AsyncFunctionTaskTemplate, TaskPluginRegistry
from flyte.models import SerializationContext, NativeInterface

# Per-thread _TaskRunner instances used by _flyte_operator for sync blocking submission.
_airflow_runners: Dict[str, _TaskRunner] = {}

# Import dag module to apply DAG monkey-patches when this module is imported.
from flyteplugins.airflow import dag as _dag_module  # noqa: E402


@dataclass
class AirflowObj(object):
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
        _, task_module, _, task_name, _, task_config = loader_args
        task_module = importlib.import_module(name=task_module)  # type: ignore
        task_def = getattr(task_module, task_name)
        return task_def(name=task_name, task_config=jsonpickle.decode(task_config))

    def loader_args(self, task: AsyncFunctionTaskTemplate, root_dir: Path) -> List[str]:  # type:ignore
        return [
            "task-module",
            task.__module__,
            "task-name",
            task.__class__.__name__,
            "task-config",
            jsonpickle.encode(task.plugin_config),
        ]


airflow_task_resolver = AirflowTaskResolver()


class AirflowContainerTask(TaskTemplate):
    """
    This python container task is used to wrap an Airflow task. It is used to run an Airflow task in a container.
    The airflow task module, name and parameters are stored in the task config.

    Some of the Airflow operators are not deferrable, For example, BeamRunJavaPipelineOperator, BeamRunPythonPipelineOperator.
    These tasks don't have an async method to get the job status, so cannot be used in the Flyte connector. We run these tasks in a container.
    """

    def __init__(
        self,
        name: str,
        plugin_config: AirflowObj,
        # inputs: Optional[Dict[str, Type]] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            # plugin_config=plugin_config,
            interface=NativeInterface(inputs={}, outputs={}),
            **kwargs,
        )
        self._task_resolver = airflow_task_resolver
        self._plugin_config = plugin_config
        self._call_as_synchronous = True
        self._downstream_flyte_tasks: List["AirflowContainerTask"] = []

    # ------------------------------------------------------------------
    # Airflow dependency-arrow support (>> / <<)
    # Records the dependency in the active FlyteDAG if one is being built.
    # ------------------------------------------------------------------

    def __rshift__(self, other: "AirflowContainerTask") -> "AirflowContainerTask":
        """``self >> other`` — other runs after self."""
        if _dag_module._current_flyte_dag is not None:
            _dag_module._current_flyte_dag.set_dependency(self.name, other.name)
        return other

    def __lshift__(self, other: "AirflowContainerTask") -> "AirflowContainerTask":
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
        _get_airflow_instance(self._plugin_config).execute(context=airflow_context.Context())
        # Trigger downstream tasks in parallel after this operator completes.
        if self._downstream_flyte_tasks:
            import asyncio
            await asyncio.gather(*[t.aio() for t in self._downstream_flyte_tasks])


def _get_airflow_instance(
    airflow_obj: AirflowObj,
) -> typing.Union[airflow_models.BaseOperator, airflow_sensors.BaseSensorOperator, airflow_triggers.BaseTrigger]:
    # Set the GET_ORIGINAL_TASK attribute to True so that obj_def will return the original
    # airflow task instead of the Flyte task.
    with flyte.custom_context(GET_ORIGINAL_TASK="True"):

        obj_module = importlib.import_module(name=airflow_obj.module)
        obj_def = getattr(obj_module, airflow_obj.name)
        if _is_deferrable(obj_def):
            try:
                return obj_def(**airflow_obj.parameters, deferrable=True)
            except airflow.exceptions.AirflowException as e:
                logger.debug(f"Failed to create operator {airflow_obj.name} with err: {e}.")
                logger.debug(f"Airflow operator {airflow_obj.name} does not support deferring.")

        return obj_def(**airflow_obj.parameters)


def _is_deferrable(cls: Type) -> bool:
    """
    This function is used to check if the Airflow operator is deferrable.
    If the operator is not deferrable, we run it in a container instead of the connector.
    """
    # Only Airflow operators are deferrable.
    if not issubclass(cls, airflow_models.BaseOperator):
        return False
    # Airflow sensors are not deferrable. The Sensor is a subclass of BaseOperator.
    if issubclass(cls, airflow_sensors.BaseSensorOperator):
        return False
    try:
        from airflow.providers.apache.beam.operators.beam import BeamBasePipelineOperator

        # Dataflow operators are not deferrable.
        if issubclass(cls, BeamBasePipelineOperator):
            return False
    except ImportError:
        logger.debug("Failed to import BeamBasePipelineOperator")
    return True


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
    config = AirflowObj(module=cls.__module__, name=cls.__name__, parameters=kwargs)

    task = AirflowContainerTask(name=task_id, plugin_config=config, image=container_image)

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