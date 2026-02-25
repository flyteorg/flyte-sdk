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

bigquery_env = flyte.TaskEnvironment.from_task("bigquery_env", bigquery_task)

env = flyte.TaskEnvironment(
    name="bigquery_example_env",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-bigquery"),
    depends_on=[bigquery_env],
)


@env.task()
def main(version: int):
    bigquery_task(version=version)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote").run(main, 123)
    print(run.url)
