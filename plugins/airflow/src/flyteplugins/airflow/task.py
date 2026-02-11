import importlib
import logging
import typing
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

import airflow

import airflow.models as airflow_models
import airflow.sensors.base as airflow_sensors
import jsonpickle
from airflow.triggers.base as airflow_triggers
import airflow.utils.context as airflow_context

from flyte import logger
from flyte._internal.resolvers.common import Resolver
from flyte._task import TaskTemplate
from flyte.extend import AsyncFunctionTaskTemplate, TaskPluginRegistry


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

    def load_task(self, loader_args: typing.List[str]) -> TaskTemplate:
        """
        This method is used to load an Airflow task.
        """
        _, task_module, _, task_name, _, task_config = loader_args
        task_module = importlib.import_module(name=task_module)  # type: ignore
        task_def = getattr(task_module, task_name)
        return task_def(name=task_name, task_config=jsonpickle.decode(task_config))

    def loader_args(self, task: TaskTemplate, root_dir: Path) -> List[str]:  # type:ignore
        return [
            "task-module",
            task.__module__,
            "task-name",
            task.__class__.__name__,
            "task-config",
            jsonpickle.encode(task.task_config),
        ]

airflow_task_resolver = AirflowTaskResolver()


class AirflowContainerTask(AsyncFunctionTaskTemplate):
    """
    This python container task is used to wrap an Airflow task. It is used to run an Airflow task in a container.
    The airflow task module, name and parameters are stored in the task config.

    Some of the Airflow operators are not deferrable, For example, BeamRunJavaPipelineOperator, BeamRunPythonPipelineOperator.
    These tasks don't have an async method to get the job status, so cannot be used in the Flyte connector. We run these tasks in a container.
    """

    def __init__(
        self,
        name: str,
        task_config: AirflowObj,
        # inputs: Optional[Dict[str, Type]] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            plugin_config=task_config,
            # interface=Interface(inputs=inputs or {}),
            **kwargs,
        )
        self._task_resolver = airflow_task_resolver

    def execute(self, **kwargs) -> Any:
        logger.info("Executing Airflow task")
        _get_airflow_instance(self.plugin_config).execute(context=airflow_context.Context())


class AirflowTask(PythonTask[AirflowObj]):
    """
    This python task is used to wrap an Airflow task.
    It is used to run an Airflow task in Flyte connector.
    The airflow task module, name and parameters are stored in the task config.
    We run the Airflow task in the connector.
    """

    _TASK_TYPE = "airflow"

    def __init__(
            self,
            name: str,
            task_config: Optional[AirflowObj],
            inputs: Optional[Dict[str, Type]] = None,
            **kwargs,
    ):
        super().__init__(
            name=name,
            task_config=task_config,
            interface=Interface(inputs=inputs or {}),
            task_type=self._TASK_TYPE,
            **kwargs,
        )

    def get_custom(self, settings: SerializationSettings) -> Dict[str, Any]:
        # Use jsonpickle to serialize the Airflow task config since the return value should be json serializable.
        return {"task_config_pkl": jsonpickle.encode(self.task_config)}


def _get_airflow_instance(
        airflow_obj: AirflowObj,
) -> typing.Union[airflow_models.BaseOperator, airflow_sensors.BaseSensorOperator, airflow_triggers.BaseTrigger]:
    # Set the GET_ORIGINAL_TASK attribute to True so that obj_def will return the original
    # airflow task instead of the Flyte task.
    ctx = FlyteContextManager.current_context()
    ctx.user_space_params.builder().add_attr("GET_ORIGINAL_TASK", True).build()

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
        if FlyteContextManager.current_context().user_space_params.get_original_task:
            # Return an original task when running in the connector.
            return object.__new__(cls)
    except AssertionError:
        # This happens when the task is created in the dynamic workflow.
        # We don't need to return the original task in this case.
        logging.debug("failed to get the attribute GET_ORIGINAL_TASK from user space params")

    container_image = kwargs.pop("container_image", None)
    task_id = kwargs.get("task_id", cls.__name__)
    config = AirflowObj(module=cls.__module__, name=cls.__name__, parameters=kwargs)

    if not issubclass(cls, airflow_sensors.BaseSensorOperator) and not _is_deferrable(cls):
        # Dataflow operators are not deferrable, so we run them in a container.
        return AirflowContainerTask(name=task_id, task_config=config, container_image=container_image)()
    return AirflowTask(name=task_id, task_config=config)()


# Monkey patches the Airflow operator. Instead of creating an airflow task, it returns a Flyte task.
airflow_models.BaseOperator.__new__ = _flyte_operator

