from flyteplugins.connectors.bigquery.task import BigQueryConfig, BigQueryTask

import flyte
from flyte.io import DataFrame

bigquery_env = flyte.TaskEnvironment(
    name="bigquery_env",
)


bigquery_task = BigQueryTask(
    name="bigquery",
    inputs={"version": int},
    output_dataFrame_type=DataFrame,
    plugin_config=BigQueryConfig(ProjectID="flyte"),
    query_template="SELECT * from dataset.flyte;",
)

bigquery_env.add_task(bigquery_task)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote").run(bigquery_task, 123)
    print(run.url)
