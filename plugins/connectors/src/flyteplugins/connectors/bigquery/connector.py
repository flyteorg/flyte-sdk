import datetime
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
        cinfo = ClientInfo(user_agent=user_agent)

        project = custom["ProjectID"]
        location = custom["Location"]

        client = bigquery.Client(project=project, location=location, client_info=cinfo)
        query_job = client.query(task_template.sql.statement, job_config=job_config)

        return BigQueryMetadata(job_id=str(query_job.job_id), location=location, project=project)

    async def get(self, resource_meta: BigQueryMetadata, **kwargs) -> Resource:
        client = bigquery.Client()
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

    async def delete(self, resource_meta: BigQueryMetadata, **kwargs):
        client = bigquery.Client()
        client.cancel_job(resource_meta.job_id, resource_meta.project, resource_meta.location)


ConnectorRegistry.register(BigQueryConnector())
