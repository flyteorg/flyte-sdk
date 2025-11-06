import datetime
from dataclasses import dataclass
from typing import Any, Dict, Optional

from async_lru import alru_cache
from flyte import logger
from flyte.connectors import (
    AsyncConnector,
    ConnectorRegistry,
    Resource,
    ResourceMeta,
)
from flyte.connectors.utils import convert_to_flyte_phase
from flyte.io import DataFrame
from flyte.types import TypeEngine
from flyteidl2.core.execution_pb2 import TaskExecution, TaskLog
from flyteidl2.core.tasks_pb2 import TaskTemplate
from google.api_core.client_info import ClientInfo
from google.cloud import bigquery
from google.oauth2 import service_account
from google.protobuf import json_format

pythonTypeToBigQueryType: Dict[type, str] = {
    # https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#data_type_sizes
    list: "ARRAY",
    bool: "BOOL",
    bytes: "BYTES",
    datetime.datetime: "DATETIME",
    float: "FLOAT64",
    int: "INT64",
    str: "STRING",
}


@dataclass
class BigQueryMetadata(ResourceMeta):
    job_id: str
    project: str
    location: str
    user_agent: str


@alru_cache
async def _get_bigquery_client(
    project: str, location: str, user_agent: str, google_application_credentials: str
) -> bigquery.Client:
    if google_application_credentials is not None:
        credentials = service_account.Credentials.from_service_account_info(google_application_credentials)
    else:
        credentials = None
    cinfo = ClientInfo(user_agent=user_agent)
    return bigquery.Client(project=project, location=location, client_info=cinfo, credentials=credentials)


class BigQueryConnector(AsyncConnector):
    name = "Bigquery Connector"
    task_type_name = "bigquery_query_job_task"
    metadata_type = BigQueryMetadata

    async def create(
        self,
        task_template: TaskTemplate,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> BigQueryMetadata:
        job_config = None
        if inputs:
            python_interface_inputs = {
                name: TypeEngine.guess_python_type(lt.type)
                for name, lt in task_template.interface.inputs.variables.items()
            }
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(name, pythonTypeToBigQueryType[python_interface_inputs[name]], val)
                    for name, val in inputs.items()
                ]
            )

        custom = json_format.MessageToDict(task_template.custom) if task_template.custom else None

        domain = custom.get("Domain")
        sdk_version = task_template.metadata.runtime.version

        user_agent = f"Flyte/{sdk_version} (GPN:Union;{domain or ''})"
        project = custom["ProjectID"]
        location = custom["Location"]

        client = await _get_bigquery_client(
            project=project,
            location=location,
            user_agent=user_agent,
            google_application_credentials=custom.get("google_application_credentials"),
        )
        query_job = client.query(task_template.sql.statement, job_config=job_config)

        return BigQueryMetadata(job_id=str(query_job.job_id), location=location, project=project, user_agent=user_agent)

    async def get(
        self, resource_meta: BigQueryMetadata, google_application_credentials: Optional[str] = None, **kwargs
    ) -> Resource:
        client = await _get_bigquery_client(
            project=resource_meta.project,
            location=resource_meta.location,
            user_agent=resource_meta.user_agent,
            google_application_credentials=google_application_credentials,
        )
        log_link = TaskLog(
            uri=f"https://console.cloud.google.com/bigquery?project={resource_meta.project}&j=bq:{resource_meta.location}:{resource_meta.job_id}&page=queryresults",
            name="BigQuery Console",
        )

        job = client.get_job(resource_meta.job_id, resource_meta.project, resource_meta.location)
        if job.errors:
            logger.error("failed to run BigQuery job with error:", job.errors.__str__())
            return Resource(phase=TaskExecution.FAILED, message=job.errors.__str__(), log_links=[log_link])

        cur_phase = convert_to_flyte_phase(str(job.state))
        res = None

        if cur_phase == TaskExecution.SUCCEEDED:
            dst = job.destination
            if dst:
                output_location = f"bq://{dst.project}:{dst.dataset_id}.{dst.table_id}"
                res = {"results": DataFrame(uri=output_location)}

        return Resource(phase=cur_phase, message=str(job.state), log_links=[log_link], outputs=res)

    async def delete(
        self, resource_meta: BigQueryMetadata, google_application_credentials: Optional[str] = None, **kwargs
    ):
        client = await _get_bigquery_client(
            project=resource_meta.project,
            location=resource_meta.location,
            user_agent=resource_meta.user_agent,
            google_application_credentials=google_application_credentials,
        )
        client.cancel_job(resource_meta.job_id, resource_meta.project, resource_meta.location)


ConnectorRegistry.register(BigQueryConnector())
