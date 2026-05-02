"""A task that runs on the custom batch job connector."""

from pathlib import Path

from my_connector.task import BatchJobConfig, BatchJobTask

import flyte

batch_job = BatchJobTask(
    name="my_batch_job",
    plugin_config=BatchJobConfig(timeout_seconds=60),
    inputs={"name": str},
    outputs={"result": str},
)

flyte.TaskEnvironment.from_task("batch-job-env", batch_job)

if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)
    result = flyte.with_runcontext(env_vars={"_U_USE_ACTIONS": "1"}).run(batch_job, name="hello")
    print(result.url)
