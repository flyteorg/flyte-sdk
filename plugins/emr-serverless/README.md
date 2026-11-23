# AWS EMR Serverless Plugin for Flyte

Submit existing Spark or Hive scripts to an AWS EMR Serverless application.
The plugin provides the `emr_serverless` task type and an async connector that
starts stopped applications without blocking task creation, submits jobs
idempotently, reports provider state, exposes an AWS Console link, and handles
cancellation.

## Installation

```bash
pip install flyteplugins-emr-serverless
```

Install the package in the `flyteconnector` image as well as the task-authoring
environment, then route the `emr_serverless` task type to the connector service.
The connector uses boto3's standard credential chain; grant only the EMR
Serverless operations and resources required by the jobs it runs.

## Usage

```python
import flyte
from flyteplugins.emr_serverless import EMRServerless

env = flyte.TaskEnvironment(
    name="spark-jobs",
    plugin_config=EMRServerless(
        application_id="00exampleapplication",
        execution_role_arn="arn:aws:iam::123456789012:role/emr-job-role",
        region="us-west-2",
        job_driver={
            "sparkSubmit": {
                "entryPoint": "s3://example-bucket/jobs/main.py",
                "entryPointArguments": ["10"],
                "sparkSubmitParameters": "--conf spark.executor.instances=2",
            }
        },
        tags={"application": "example"},
    ),
)


@env.task
async def count_records() -> None:
    """The connector submits the configured Spark script."""
```

The decorated function declares a Flyte task but does not run inside EMR
Serverless. This initial plugin supports existing Spark and Hive scripts and
therefore requires a `None` return type. Pythonic execution of the function
body is a separate follow-up.

## Lifecycle behavior

- `CREATED` and `STOPPED` applications receive `StartApplication`; job
  submission is deferred to connector polling instead of blocking `CreateTask`.
- `StartJobRun.clientToken` is preserved when provided and otherwise derived
  deterministically from Flyte execution metadata.
- Repeated polls are safe: AWS returns the same job for the same client token.
- Successful script jobs return an empty output map so no `outputs.pb` is
  expected.
- Aborting while the application is still starting submits no job. If the
  application becomes ready before deletion, the connector submits
  idempotently and immediately cancels the resulting job.
- Transient AWS throttling and service errors request a Flyte retry.

## AWS permissions

For an existing application, the connector needs:

```text
emr-serverless:GetApplication
emr-serverless:StartApplication
emr-serverless:StartJobRun
emr-serverless:GetJobRun
emr-serverless:CancelJobRun
```

The execution role passed to `StartJobRun` must be usable by EMR Serverless and
must have access to the script, data, logs, and outputs referenced by the job.
The connector does not read credentials from task configuration.

## Current scope

This first version intentionally does not create applications, update
application images, upload Python entrypoints, or execute Flyte Python function
bodies. Those capabilities require additional lifecycle and runtime contracts
and can be added independently without changing the script-mode payload.
