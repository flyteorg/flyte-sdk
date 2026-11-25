# AWS EMR Serverless plugin

Run Pythonic Spark tasks, existing Spark scripts, and Hive queries through the
`emr_serverless` task type. The connector manages application lookup, optional
creation, startup, image synchronization, job submission, status, provider
links, retries, and exact idempotent cancellation.

## Installation

```bash
pip install flyteplugins-emr-serverless
```

Install the package in the authoring environment and connector image. Pythonic
workers need the Flyte SDK in their EMR custom image; script Spark and Hive
workers do not.

## Usage

### Pythonic Spark

```python
import flyte
from flyteplugins.emr_serverless import EMRServerless

env = flyte.TaskEnvironment(
    name="python-spark",
    plugin_config=EMRServerless(
        application_id="00exampleapplication",
        execution_role_arn="arn:aws:iam::123456789012:role/emr-job-role",
        region="us-west-2",
        spark_submit_parameters="--conf spark.executor.instances=2",
    ),
)


@env.task
async def count_records() -> int:
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate().range(100).count()
```

The connector stages a content-addressed bootstrap to S3 and passes the task's
serialized Flyte 2 `a0` command unchanged. `a0` downloads its CodeBundle and
uses the standard Flyte input, `outputs.pb`, and `error.pb` protocol.

Set `FLYTE_EMR_ENTRYPOINT_S3_BUCKET` on the connector, configure an S3
monitoring log URI, or set `FLYTE_AWS_S3_BUCKET` to choose the staging bucket.

### Script Spark and Hive

```python
from flyteplugins.emr_serverless import (
    EMRServerless,
    EMRServerlessHiveJobDriver,
    EMRServerlessSparkJobDriver,
)

spark = flyte.TaskEnvironment(
    name="script-spark",
    plugin_config=EMRServerless(
        application_id="00exampleapplication",
        execution_role_arn="arn:aws:iam::123456789012:role/emr-job-role",
        spark_job_driver=EMRServerlessSparkJobDriver(
            entry_point="s3://example-bucket/jobs/main.py",
            entry_point_arguments=["10"],
        ),
    ),
)

hive = flyte.TaskEnvironment(
    name="hive-query",
    plugin_config=EMRServerless(
        application_id="hive-application-name",
        execution_role_arn="arn:aws:iam::123456789012:role/emr-job-role",
        application_type="HIVE",
        hive_job_driver=EMRServerlessHiveJobDriver(query="SELECT COUNT(*) FROM example"),
    ),
)
```

Script tasks must return `None`; they write results to durable storage.

## Application policy

`application_id` accepts an AWS ID or exact active application name. To create
an application, omit it and set `application_name`. Creation is denied unless
the connector has `FLYTE_EMR_ALLOW_CREATE_APPLICATION=true`. An optional
`FLYTE_EMR_APPLICATION_NAME_PREFIX` is prepended once.

The connector fails closed unless its operator configures exact comma-separated
allowlists for task-selected AWS resources:

- `FLYTE_EMR_ALLOWED_REGIONS`
- `FLYTE_EMR_ALLOWED_EXECUTION_ROLE_ARNS`
- `FLYTE_EMR_ALLOWED_APPLICATION_IDS` for approved shared applications
- `FLYTE_EMR_ALLOWED_APPLICATION_NAME_PREFIXES` for connector-created
  application namespaces

An existing application not listed by ID is accepted only when it carries the
connector-owned `Application=flyte` and `ManagedBy=flyte-connector` tags and its
name starts with an operator-approved prefix. Connector-created applications
always receive those tags, and task tags cannot override them. This
authorization check runs before application start, image synchronization, or
job submission.

Creation forwards release, type, capacities, network, runtime, scheduler,
automatic-stop, architecture, image, and tags. Existing applications can
synchronize the task or explicit image when `sync_image=True`.

## Worker image

The Pythonic image must derive from an EMR Serverless Spark base image and
contain Python 3.11 plus the same Flyte SDK/plugin version as the authoring
environment. The connector configures PySpark to use `/usr/bin/python3.11`,
because the EMR image's default `python3` may still point to Python 3.9:

```bash
docker build \
  --build-arg VERSION=<version> \
  -t <registry>/emr-serverless-flyte:<tag> \
  plugins/emr-serverless
```

Associate the resulting ECR image with the application, pass it as the task
image, or set `image_configuration={"imageUri": "..."}`. EMR Serverless custom
images must satisfy AWS ownership, region, architecture, and filesystem
requirements.

## Lifecycle behavior

- Startup is asynchronous; submission is deferred while an application starts.
- Job tokens are explicit or deterministically derived from Flyte execution
  metadata. Deferred polling and cancellation reuse the exact request.
- `CANCELLED` maps to Flyte `ABORTED`; transient provider errors request retry.
- Flyte platform/runtime environment and task environment values are injected
  into Spark drivers and executors unless `inject_flyte_env=False`.
- The AWS Console job page is attached as a provider log link.

## AWS permissions

The connector may need:

```text
emr-serverless:ListApplications
emr-serverless:GetApplication
emr-serverless:CreateApplication
emr-serverless:UpdateApplication
emr-serverless:StartApplication
emr-serverless:StartJobRun
emr-serverless:GetJobRun
emr-serverless:CancelJobRun
s3:GetObject
s3:PutObject
iam:PassRole
```

Narrow these permissions when lookup, creation, image synchronization, or
Pythonic staging is not enabled. Scope `iam:PassRole` to the same roles listed
in `FLYTE_EMR_ALLOWED_EXECUTION_ROLE_ARNS`. The execution role needs access to
task code, inputs, outputs, logs, and job data.

## Flytekit parity matrix

| Flytekit capability/export/test family | Flyte SDK implementation | Coverage |
|---|---|---|
| `EMRServerless` | Public config with every baseline field | `test_task.py` |
| `EMRServerlessSparkJobDriver` | Typed Spark driver | `test_task.py` |
| `EMRServerlessHiveJobDriver` | Typed Hive driver | `test_task.py` |
| `EMRServerlessTask` | `AsyncFunctionTaskTemplate` connector task | `test_task.py` |
| `EMRServerlessConnector` | Async create/get/delete lifecycle | `test_connector.py` |
| `EMRServerlessJobMetadata` | JSON `ResourceMeta` with deferred request | `test_connector.py` |
| Spark script mode | Explicit `sparkSubmit` driver | task/connector tests |
| Hive mode | Explicit `hive` driver | task/connector tests |
| Pythonic Spark | Flyte 2 `a0` bootstrap and typed I/O | connector/entrypoint tests |
| Application ID/name lookup | Exact active-name pagination | handler/connector tests |
| Gated create and name prefix | Connector environment policy | connector tests |
| Release/type/capacity/network | Forwarded to create | handler/connector tests |
| Runtime/scheduler/auto-stop/architecture | Forwarded and validated | task/handler tests |
| Async startup/deferred submit | Persisted idempotent request | handler/connector tests |
| Image resolution/synchronization | Explicit/task/default image rules | connector tests |
| Tags and retry policy | Managed + user tags; job retries | handler/connector tests |
| Context/environment injection | Flyte 2 platform, identity, and task env | connector tests |
| Provider logs | AWS Console external task log | connector tests |
| Cancellation/idempotency | Exact request/token; terminal-safe cancel | handler/connector tests |
| Entrypoint S3 staging | Content-addressed bootstrap object | connector tests |
| Worker image documentation | Build/runtime requirements above | README/Dockerfile |
| Task/config/driver family | Ported to Flyte SDK | `tests/test_task.py` |
| Handler family | Ported to Flyte SDK | `tests/test_boto_handler.py` |
| Connector family | Ported to Flyte SDK | `tests/test_connector.py` |
| Entrypoint family | Flyte 2 runtime semantics | `tests/test_entrypoint.py` |
