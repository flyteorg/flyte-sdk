from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import flyte
from flyte import Resources
from flyte._internal.runtime.resources_serde import get_proto_resources
from flyte.extend import AsyncFunctionTaskTemplate, TaskPluginRegistry
from flyte.models import SerializationContext
from flyteidl.plugins.dask_pb2 import DaskJob, DaskScheduler, DaskWorkerGroup
from google.protobuf.json_format import MessageToDict


@dataclass
class Scheduler:
    """
    Configuration for the scheduler pod

    :param image: Custom image to use. If ``None``, will use the same image the task was registered with. Optional,
        defaults to None. The image must have ``dask[distributed]`` installed and should have the same Python
        environment as the rest of the cluster (job runner pod + worker pods).
    :param resources: Resources to request for the scheduler pod. Optional, defaults to None.
    """

    image: Optional[str] = None
    resources: Optional[Resources] = None


@dataclass
class WorkerGroup:
    """
    Configuration for a group of dask worker pods

    :param number_of_workers: Number of workers to use. Optional, defaults to 1.
    :param image: Custom image to use. If ``None``, will use the same image the task was registered with. Optional,
        defaults to None. The image must have ``dask[distributed]`` installed. The provided image should have the
        same Python environment as the job runner/driver as well as the scheduler.
    :param resources: Resources to request for the worker pods. Optional, defaults to None.
    """

    number_of_workers: Optional[int] = 1
    image: Optional[str] = None
    resources: Optional[Resources] = None


@dataclass
class Dask:
    """
    Configuration for the dask task

    :param scheduler: Configuration for the scheduler pod. Optional, defaults to ``Scheduler()``.
    :param workers: Configuration for the pods of the default worker group. Optional, defaults to ``WorkerGroup()``.
    """

    scheduler: Scheduler = field(default_factory=lambda: Scheduler())
    workers: WorkerGroup = field(default_factory=lambda: WorkerGroup())


@dataclass(kw_only=True)
class DaskTask(AsyncFunctionTaskTemplate):
    """
    Actual Plugin that transforms the local python code for execution within a spark context
    """

    plugin_config: Dask
    task_type: str = "dask"

    async def pre(self, *args, **kwargs) -> Dict[str, Any]:
        from distributed import Client
        from distributed.diagnostics.plugin import UploadDirectory

        if flyte.ctx().code_bundle:
            client = Client()
            client.register_plugin(UploadDirectory(flyte.ctx().code_bundle.destination))

        return {}

    def custom_config(self, sctx: SerializationContext) -> Dict[str, Any]:
        scheduler = self.plugin_config.scheduler
        wg = self.plugin_config.workers

        job = DaskJob(
            scheduler=DaskScheduler(image=scheduler.image, resources=get_proto_resources(scheduler.resources)),
            workers=DaskWorkerGroup(
                number_of_workers=wg.number_of_workers, image=wg.image, resources=get_proto_resources(wg.resources)
            ),
        )

        return MessageToDict(job)


TaskPluginRegistry.register(Dask, DaskTask)
