from flyteplugins.bigquery import BigQueryConfig, BigQueryTask

import flyte
from flyte.io import DataFrame

bigquery_task = BigQueryTask(
    name="bigquery",
    inputs={"version": int},
    output_dataframe_type=DataFrame,
    plugin_config=BigQueryConfig(ProjectID="dogfood-gcp-dataplane"),
    query_template="SELECT * from dataset.flyte_table3;",
)

flyte.TaskEnvironment.from_task("bigquery_env", bigquery_task)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="local").run(bigquery_task, 123)
    print(run.url)
