import os
import typing as ty
from dataclasses import asdict, dataclass
from functools import cache, wraps

from flyte import logger
from flyte.connectors import AsyncConnector, ConnectorRegistry, Resource, ResourceMeta
from flyte.connectors.utils import convert_to_flyte_phase
from flyteidl2.connector.connector_pb2 import TaskExecutionMetadata
from flyteidl2.core.execution_pb2 import TaskExecution, TaskLog
from flyteidl2.core.tasks_pb2 import TaskTemplate
from google.protobuf.json_format import MessageToDict

from databricks.sdk import WorkspaceClient, core
from databricks.sdk.service import compute, jobs

P = ty.ParamSpec("P")
R = ty.TypeVar("R")

DATABRICKS_API_ENDPOINT = "/api/2.1/jobs"
DEFAULT_DATABRICKS_INSTANCE_ENV_KEY = "FLYTE_DATABRICKS_INSTANCE"


def cache_with_sig(f: ty.Callable[P, R]) -> ty.Callable[P, R]:
    @wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return cache(f)(*args, **kwargs)

    return wrapper


@dataclass(kw_only=True)
class DatabricksSubmitTask(jobs.SubmitTask):
    run_name: ty.Optional[str] = None


@dataclass(kw_only=True)
class RunResource(ResourceMeta, jobs.Run):
    databricks_instance: str


@cache_with_sig
def _get_wc(host: str, token: str) -> WorkspaceClient:
    return WorkspaceClient(config=core.Config(host=host, token=token))


def _get_databricks_submit_task(task_template: TaskTemplate) -> DatabricksSubmitTask:
    custom = MessageToDict(task_template.custom)
    container = task_template.container
    envs = task_template.container.env
    databricks_job = custom.get("databricksConf")
    if databricks_job is None:
        raise ValueError("Missing Databricks job configuration in task template.")
    submit_task = DatabricksSubmitTask(**databricks_job)
    if submit_task.existing_cluster_id is None:
        new_cluster = submit_task.new_cluster
        if new_cluster is None:
            raise ValueError("Either existing_cluster_id or new_cluster must be specified")
        if not new_cluster.docker_image:
            new_cluster.docker_image = compute.DockerImage(url=container.image)
        if not new_cluster.spark_conf:
            new_cluster.spark_conf = custom.get("sparkConf", {})
        if not new_cluster.spark_env_vars:
            new_cluster.spark_env_vars = {env.key: env.value for env in envs}
        else:
            new_cluster.spark_env_vars.update({env.key: env.value for env in envs})
    # https://docs.databricks.com/api/workspace/jobs/submit
    submit_task.spark_python_task = jobs.SparkPythonTask(
        python_file="flyteplugins/databricks/entrypoint.py",
        parameters=list(container.args),
        source=jobs.Source.GIT,
    )
    # https://github.com/flyteorg/flytetools/blob/master/flyteplugins/databricks/entrypoint.py
    logger.debug("submit_task:", submit_task)
    return submit_task


class DatabricksConnector(AsyncConnector[RunResource]):
    name: str = "Databricks Connector"
    task_type_name: str = "databricks"
    metadata_type = RunResource

    async def create(
        self,
        task_template: TaskTemplate,
        output_prefix: str,
        inputs: ty.Optional[ty.Dict[str, ty.Any]] = None,
        task_execution_metadata: ty.Optional[TaskExecutionMetadata] = None,
        databricks_token: ty.Optional[str] = None,
        **kwargs: str,
    ) -> RunResource:
        submit_task = _get_databricks_submit_task(task_template)
        custom = MessageToDict(task_template.custom)
        databricks_instance = custom.get("databricksInstance", os.getenv(DEFAULT_DATABRICKS_INSTANCE_ENV_KEY))

        if not databricks_instance:
            raise ValueError(
                f"Missing databricks instance. Please set the value through the task config or"
                f" set the {DEFAULT_DATABRICKS_INSTANCE_ENV_KEY} environment variable in the connector."
            )

        wc = _get_wc(host=f"https://{databricks_instance}", token=databricks_token or "")
        run_waiter = wc.jobs.submit(
            run_name="",
            tasks=[submit_task],
            git_source=jobs.GitSource(
                git_url="https://github.com/flyteorg/flytetools",
                git_provider=jobs.GitProvider.GIT_HUB,
                git_commit="194364210c47c49ce66c419e8fb68d6f9c06fd7e",
            ),
        )
        return RunResource(**asdict(wc.jobs.get_run(run_id=run_waiter.run_id)), databricks_instance=databricks_instance)

    async def get(
        self, resource_meta: RunResource, databricks_token: ty.Optional[str] = None, **kwargs: str
    ) -> Resource:

        wc = _get_wc(host=f"https://{resource_meta.databricks_instance}", token=databricks_token or "")
        assert resource_meta.run_id is not None
        run = wc.jobs.get_run(resource_meta.run_id)
        cur_phase = TaskExecution.UNDEFINED
        message = ""
        status = run.status

        # The databricks job's state is determined by life_cycle_state and result_state.
        # https://docs.databricks.com/en/workflows/jobs/jobs-2.0-api.html#runresultstate
        if status and status.state:
            if termination_type := get_result_state(status):
                result_state = termination_type.value
                cur_phase = convert_to_flyte_phase(result_state)
            else:
                cur_phase = convert_to_flyte_phase(status.state.value)

            message = run.state.state_message if run.state and run.state.state_message else ""
            # ^ TODO: state field is deprecated. We'll need to figure out how to get the same data
            # once this gets removed from the API responses.

        log_links = [TaskLog(uri=run.run_page_url, name="Databricks Console")] if run.run_page_url else []

        return Resource(phase=cur_phase, message=message, log_links=log_links)

    async def delete(self, resource_meta: RunResource, databricks_token: ty.Optional[str] = None, **kwargs: ty.Any):
        wc = _get_wc(host=f"https://{resource_meta.databricks_instance}", token=databricks_token or "")
        assert resource_meta.run_id is not None
        wc.jobs.cancel_run(resource_meta.run_id)


def get_result_state(status: jobs.RunStatus) -> jobs.TerminationTypeType | None:
    if (
        status.state == jobs.RunLifecycleStateV2State.TERMINATED
        and status.termination_details
        and status.termination_details.type
    ):
        return status.termination_details.type
    return None


ConnectorRegistry.register(DatabricksConnector())
