import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from flyte._task import TaskTemplate
from flyte.io import DataFrame
from flyte.models import SerializationContext
from flyteidl.core import tasks_pb2
from google.cloud import bigquery
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct


@dataclass
class BigQueryConfig(object):
    """
    BigQueryConfig should be used to configure a BigQuery Task.
    """

    ProjectID: str
    Location: Optional[str] = None
    QueryJobConfig: Optional[bigquery.QueryJobConfig] = None


class BigQueryTask(TaskTemplate):
    _TASK_TYPE = "bigquery_query_job_task"

    def __init__(
        self,
        name: str,
        query_template: str,
        plugin_config: BigQueryConfig,
        inputs: Optional[Dict[str, Type]] = None,
        output_dataframe_type: Optional[Type[DataFrame]] = None,
        **kwargs,
    ):
        """
        To be used to query BigQuery Tables.

        :param name: The Name of this task, should be unique in the project
        :param query_template: The actual query to run. We use Flyte's Golang templating format for Query templating.
         Refer to the templating documentation
        :param plugin_config: BigQueryConfig object
        :param inputs: Name and type of inputs specified as an ordered dictionary
        :param output_dataframe_type: If some data is produced by this query, then you can specify the
         output dataframe type.
        """
        outputs = None
        if output_dataframe_type is not None:
            outputs = {
                "results": output_dataframe_type,
            }
        super().__init__(
            name=name,
            plugin_config=plugin_config,
            inputs=inputs,
            outputs=outputs,
            task_type=self._TASK_TYPE,
            **kwargs,
        )
        self.plugin_config = plugin_config
        self.query_template = re.sub(r"\s+", " ", query_template.replace("\n", " ").replace("\t", " ")).strip()

    def custom_config(self, sctx: SerializationContext) -> Optional[Dict[str, Any]]:
        config = {
            "Location": self.plugin_config.Location,
            "ProjectID": self.plugin_config.ProjectID,
            "Domain": sctx.domain,
        }
        if self.plugin_config.QueryJobConfig is not None:
            config.update(self.plugin_config.QueryJobConfig.to_api_repr()["query"])
        s = Struct()
        s.update(config)
        return json_format.MessageToDict(s)

    def sql(self, sctx: SerializationContext) -> Optional[str]:
        sql = tasks_pb2.Sql(statement=self.query_template, dialect=tasks_pb2.Sql.Dialect.ANSI)
        return sql
