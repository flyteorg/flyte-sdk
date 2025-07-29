import datetime
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import flytekit
from flytekit.core import launch_plan, workflow
from flytekit.core.base_task import T, TaskResolverMixin
from flytekit.core.python_function_task import PythonFunctionTask
from flytekit.core.task import FuncOut
from flytekit.deck import DeckField
from flytekit.extras.accelerators import BaseAccelerator

import flyte
from flyte import Image, Resources, TaskEnvironment
from flyte._doc import Documentation
from flyte._task import AsyncFunctionTaskTemplate, P, R


def task_shim(
    _task_function: Optional[Callable[P, FuncOut]] = None,
    task_config: Optional[T] = None,
    cache: Union[bool, flytekit.Cache] = False,
    retries: int = 0,
    interruptible: Optional[bool] = None,
    deprecated: str = "",
    timeout: Union[datetime.timedelta, int] = 0,
    container_image: Optional[Union[str, flytekit.ImageSpec]] = None,
    environment: Optional[Dict[str, str]] = None,
    requests: Optional[flytekit.Resources] = None,
    limits: Optional[flytekit.Resources] = None,
    secret_requests: Optional[List[flytekit.Secret]] = None,
    execution_mode: PythonFunctionTask.ExecutionBehavior = PythonFunctionTask.ExecutionBehavior.DEFAULT,
    node_dependency_hints: Optional[
        Iterable[
            Union[
                flytekit.PythonFunctionTask,
                launch_plan.LaunchPlan,
                workflow.WorkflowBase,
            ]
        ]
    ] = None,
    task_resolver: Optional[TaskResolverMixin] = None,
    docs: Optional[flytekit.Documentation] = None,
    disable_deck: Optional[bool] = None,
    enable_deck: Optional[bool] = None,
    deck_fields: Optional[Tuple[DeckField, ...]] = (
        DeckField.SOURCE_CODE,
        DeckField.DEPENDENCIES,
        DeckField.TIMELINE,
        DeckField.INPUT,
        DeckField.OUTPUT,
    ),
    pod_template: Optional[flytekit.PodTemplate] = None,
    pod_template_name: Optional[str] = None,
    accelerator: Optional[BaseAccelerator] = None,
    pickle_untyped: bool = False,
    shared_memory: Optional[Union[Literal[True], str]] = None,
    resources: Optional[Resources] = None,
    labels: Optional[dict[str, str]] = None,
    annotations: Optional[dict[str, str]] = None,
    **kwargs,
) -> Union[AsyncFunctionTaskTemplate, Callable[P, R]]:
    plugin_config = task_config
    pod_template = (
        flyte.PodTemplate(
            pod_spec=pod_template.pod_spec,
            primary_container_name=pod_template.primary_container_name,
            labels=pod_template.labels,
            annotations=pod_template.annotations,
        )
        if pod_template
        else None
    )

    if isinstance(container_image, flytekit.ImageSpec):
        image = Image.from_debian_base()
        if container_image.apt_packages:
            image = image.with_apt_packages(*container_image.apt_packages)
        pip_packages = ["flytekit"]
        if container_image.packages:
            pip_packages.extend(container_image.packages)
        image = image.with_pip_packages(*pip_packages)
    elif isinstance(container_image, str):
        image = Image.from_base(container_image).with_pip_packages("flyte")
    else:
        image = Image.from_debian_base().with_pip_packages("flytekit")

    docs = Documentation(description=docs.short_description) if docs else None

    env = TaskEnvironment(
        name="flytekit",
        resources=Resources(cpu=0.8, memory="800Mi"),
        image=image,
        cache="enabled" if cache else "disable",
        plugin_config=plugin_config,
    )
    return env.task(retries=retries, pod_template=pod_template_name or pod_template, docs=docs)


flytekit.task = task_shim
