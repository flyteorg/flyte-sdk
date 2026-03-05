import http
import json
import os
import typing
from dataclasses import dataclass
from typing import Optional

import aiohttp
from flyte import logger
from flyte.connectors import AsyncConnector, ConnectorRegistry, Resource, ResourceMeta
from flyte.connectors.utils import convert_to_flyte_phase
from flyteidl2.core.execution_pb2 import TaskExecution, TaskLog
from flyteidl2.core.tasks_pb2 import TaskTemplate
from google.protobuf.json_format import MessageToDict

DATABRICKS_API_ENDPOINT = "/api/2.1/jobs"
DEFAULT_DATABRICKS_INSTANCE_ENV_KEY = "FLYTE_DATABRICKS_INSTANCE"


@dataclass
class DatabricksJobMetadata(ResourceMeta):
    databricks_instance: str
    run_id: str


def _get_databricks_job_spec(task_template: TaskTemplate) -> dict:
    custom = MessageToDict(task_template.custom)
    container = task_template.container
    envs = task_template.container.env
    databricks_job = custom.get("databricksConf")
    if databricks_job is None:
        raise ValueError("Missing Databricks job configuration in task template.")
    if databricks_job.get("existing_cluster_id") is None:
        new_cluster = databricks_job.get("new_cluster")
        if new_cluster is None:
            raise ValueError("Either existing_cluster_id or new_cluster must be specified")
        if not new_cluster.get("docker_image"):
            new_cluster["docker_image"] = {"url": container.image}
        if not new_cluster.get("spark_conf"):
            new_cluster["spark_conf"] = custom.get("sparkConf", {})
        if not new_cluster.get("spark_env_vars"):
            new_cluster["spark_env_vars"] = {env.key: env.value for env in envs}
        else:
            new_cluster["spark_env_vars"].update({env.key: env.value for env in envs})
    # https://docs.databricks.com/api/workspace/jobs/submit
    databricks_job["spark_python_task"] = {
        "python_file": "flyteplugins/databricks/entrypoint.py",
        "parameters": list(container.args),
        "source": "GIT",
    }
    # https://github.com/flyteorg/flytetools/blob/master/flyteplugins/databricks/entrypoint.py
    databricks_job["git_source"] = {
        "git_url": "https://github.com/flyteorg/flytetools",
        "git_provider": "gitHub",
        "git_commit": "194364210c47c49ce66c419e8fb68d6f9c06fd7e",
    }

    logger.debug("databricks_job spec:", databricks_job)
    return databricks_job


class DatabricksConnector(AsyncConnector):
    name: str = "Databricks Connector"
    task_type_name: str = "databricks"
    metadata_type: type = DatabricksJobMetadata

    async def create(
        self,
        task_template: TaskTemplate,
        inputs: Optional[typing.Dict[str, typing.Any]] = None,
        databricks_token: Optional[str] = None,
        **kwargs,
    ) -> DatabricksJobMetadata:
        data = json.dumps(_get_databricks_job_spec(task_template))
        custom = MessageToDict(task_template.custom)
        databricks_instance = custom.get("databricksInstance", os.getenv(DEFAULT_DATABRICKS_INSTANCE_ENV_KEY))

        if not databricks_instance:
            raise ValueError(
                f"Missing databricks instance. Please set the value through the task config or"
                f" set the {DEFAULT_DATABRICKS_INSTANCE_ENV_KEY} environment variable in the connector."
            )

        databricks_url = f"https://{databricks_instance}{DATABRICKS_API_ENDPOINT}/runs/submit"

        async with aiohttp.ClientSession() as session:
            async with session.post(databricks_url, headers=get_header(databricks_token), data=data) as resp:
                response = await resp.json()
                if resp.status != http.HTTPStatus.OK:
                    raise RuntimeError(f"Failed to create databricks job with error: {response}")

        return DatabricksJobMetadata(databricks_instance=databricks_instance, run_id=str(response["run_id"]))

    async def get(
        self, resource_meta: DatabricksJobMetadata, databricks_token: Optional[str] = None, **kwargs
    ) -> Resource:
        databricks_instance = resource_meta.databricks_instance
        databricks_url = (
            f"https://{databricks_instance}{DATABRICKS_API_ENDPOINT}/runs/get?run_id={resource_meta.run_id}"
        )

        async with aiohttp.ClientSession() as session:
            async with session.get(databricks_url, headers=get_header(databricks_token)) as resp:
                if resp.status != http.HTTPStatus.OK:
                    raise RuntimeError(f"Failed to get databricks job {resource_meta.run_id} with error: {resp.reason}")
                response = await resp.json()

        cur_phase = TaskExecution.UNDEFINED
        message = ""
        state = response.get("state")

        # The databricks job's state is determined by life_cycle_state and result_state.
        # https://docs.databricks.com/en/workflows/jobs/jobs-2.0-api.html#runresultstate
        if state:
            life_cycle_state = state.get("life_cycle_state")
            if result_state_is_available(life_cycle_state):
                result_state = state.get("result_state")
                cur_phase = convert_to_flyte_phase(result_state)
            else:
                cur_phase = convert_to_flyte_phase(life_cycle_state)

            message = state.get("state_message")

        job_id = response.get("job_id")
        databricks_console_url = f"https://{databricks_instance}/#job/{job_id}/run/{resource_meta.run_id}"
        log_links = [TaskLog(uri=databricks_console_url, name="Databricks Console")]

        return Resource(phase=cur_phase, message=message, log_links=log_links)

    async def delete(self, resource_meta: DatabricksJobMetadata, databricks_token: Optional[str] = None, **kwargs):
        databricks_url = f"https://{resource_meta.databricks_instance}{DATABRICKS_API_ENDPOINT}/runs/cancel"
        data = json.dumps({"run_id": resource_meta.run_id})

        async with aiohttp.ClientSession() as session:
            async with session.post(databricks_url, headers=get_header(databricks_token), data=data) as resp:
                if resp.status != http.HTTPStatus.OK:
                    raise RuntimeError(
                        f"Failed to cancel databricks job {resource_meta.run_id} with error: {resp.reason}"
                    )
                await resp.json()


def get_header(token: str) -> typing.Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "content-type": "application/json"}


def result_state_is_available(life_cycle_state: str) -> bool:
    return life_cycle_state == "TERMINATED"


ConnectorRegistry.register(DatabricksConnector())
