from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from flyte._task_plugins import TaskPluginRegistry
from flyte.connectors import AsyncConnectorExecutorMixin
from flyte.extend import AsyncFunctionTaskTemplate
from flyte.models import SerializationContext

TASK_TYPE = "emr_serverless"
EMR_SERVERLESS_BASE_IMAGE = "public.ecr.aws/emr-serverless/spark/emr-7.0.0:latest"
_VALID_ARCHITECTURES = ("X86_64", "ARM64")


@dataclass
class EMRServerlessSparkJobDriver:
    """Spark job driver configuration.

    Args:
        entry_point: S3 URI of the Python, JAR, or other Spark entrypoint.
        entry_point_arguments: Arguments passed to the entrypoint.
        spark_submit_parameters: Additional `spark-submit` parameters.
    """

    entry_point: str
    entry_point_arguments: Optional[List[str]] = None
    spark_submit_parameters: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert this driver to the AWS `sparkSubmit` payload."""
        result: Dict[str, Any] = {"entryPoint": self.entry_point}
        if self.entry_point_arguments:
            result["entryPointArguments"] = list(self.entry_point_arguments)
        if self.spark_submit_parameters:
            result["sparkSubmitParameters"] = self.spark_submit_parameters
        return result


@dataclass
class EMRServerlessHiveJobDriver:
    """Hive job driver configuration.

    Args:
        query: Inline Hive query or S3 URI of a query file.
        init_query_file: Optional S3 URI of an initialization query.
        parameters: Optional Hive parameters string.
    """

    query: str
    init_query_file: Optional[str] = None
    parameters: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert this driver to the AWS `hive` payload."""
        result: Dict[str, Any] = {"query": self.query}
        if self.init_query_file:
            result["initQueryFile"] = self.init_query_file
        if self.parameters:
            result["parameters"] = self.parameters
        return result


@dataclass
class EMRServerless:
    """Configuration for Spark and Hive jobs on EMR Serverless.

    An explicit Spark or Hive driver selects script mode. Omitting both selects
    Pythonic Spark mode, in which the task's Flyte 2 `a0` command runs in the
    application's worker image and retains typed Flyte inputs and outputs.

    Args:
        execution_role_arn: IAM role used by EMR Serverless for the job.
        application_id: Existing application ID or exact application name.
        release_label: EMR release used when creating an application.
        application_type: `SPARK` or `HIVE`.
        application_name: Name used for gated application creation.
        sync_image: Update an existing application's image when it differs.
        inject_flyte_env: Forward Flyte runtime context to Spark processes.
        spark_job_driver: Explicit Spark script driver.
        hive_job_driver: Explicit Hive driver.
        spark_submit_parameters: Additional Spark submit parameters.
        application_configuration: Job-level application configuration.
        runtime_configuration: Application runtime configuration.
        scheduler_configuration: Application scheduler configuration.
        auto_stop_config: Application automatic-stop configuration.
        architecture: `X86_64` or `ARM64`.
        retry_policy: EMR Serverless job retry policy.
        configuration_overrides: Job monitoring and configuration overrides.
        tags: Tags applied to applications and jobs.
        execution_timeout_minutes: Maximum job duration in minutes.
        initial_capacity: Initial application worker capacity.
        maximum_capacity: Maximum application capacity.
        network_configuration: Application VPC configuration.
        image_configuration: Explicit EMR worker image configuration.
        region: AWS region, or the boto3 default when omitted.
        client_token: Explicit job idempotency token. Normally Flyte derives a
            deterministic token from execution metadata.
    """

    execution_role_arn: str
    application_id: Optional[str] = None
    release_label: str = "emr-7.0.0"
    application_type: str = "SPARK"
    application_name: Optional[str] = None
    sync_image: bool = True
    inject_flyte_env: bool = True
    spark_job_driver: Optional[EMRServerlessSparkJobDriver] = None
    hive_job_driver: Optional[EMRServerlessHiveJobDriver] = None
    spark_submit_parameters: Optional[str] = None
    application_configuration: Optional[List[Dict[str, Any]]] = None
    runtime_configuration: Optional[List[Dict[str, Any]]] = None
    scheduler_configuration: Optional[Dict[str, Any]] = None
    auto_stop_config: Optional[Dict[str, Any]] = None
    architecture: Optional[str] = None
    retry_policy: Optional[Dict[str, Any]] = None
    configuration_overrides: Optional[Dict[str, Any]] = None
    tags: Optional[Dict[str, str]] = None
    execution_timeout_minutes: int = 60
    initial_capacity: Optional[Dict[str, Any]] = None
    maximum_capacity: Optional[Dict[str, Any]] = None
    network_configuration: Optional[Dict[str, Any]] = None
    image_configuration: Optional[Dict[str, Any]] = None
    region: Optional[str] = None
    client_token: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.execution_role_arn:
            raise ValueError("execution_role_arn is required")
        if self.application_type not in ("SPARK", "HIVE"):
            raise ValueError(f"application_type must be 'SPARK' or 'HIVE', got {self.application_type!r}")
        if self.application_type == "HIVE" and self.hive_job_driver is None:
            raise ValueError("hive_job_driver is required when application_type is 'HIVE'")
        if self.spark_job_driver and self.hive_job_driver:
            raise ValueError("Only one of spark_job_driver or hive_job_driver can be specified")
        if self.execution_timeout_minutes < 1:
            raise ValueError("execution_timeout_minutes must be at least 1")
        if self.architecture and self.architecture not in _VALID_ARCHITECTURES:
            raise ValueError(f"architecture must be one of {_VALID_ARCHITECTURES}, got {self.architecture!r}")

    @property
    def is_script_mode(self) -> bool:
        """Return whether this configuration supplies an explicit job driver."""
        return self.spark_job_driver is not None or self.hive_job_driver is not None

    def get_job_driver(self) -> Dict[str, Any]:
        """Return the configured AWS job-driver payload."""
        if self.spark_job_driver:
            return {"sparkSubmit": self.spark_job_driver.to_dict()}
        if self.hive_job_driver:
            return {"hive": self.hive_job_driver.to_dict()}
        raise ValueError("No job driver configured; use Pythonic Spark mode instead")

    def get_effective_configuration_overrides(self) -> Optional[Dict[str, Any]]:
        """Merge application configuration into job configuration overrides."""
        if not self.application_configuration and not self.configuration_overrides:
            return None
        result = dict(self.configuration_overrides or {})
        if self.application_configuration:
            result["applicationConfiguration"] = [
                *(result.get("applicationConfiguration") or []),
                *self.application_configuration,
            ]
        return result or None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this configuration into `TaskTemplate.custom`."""
        result: Dict[str, Any] = {
            "execution_role_arn": self.execution_role_arn,
            "release_label": self.release_label,
            "application_type": self.application_type,
            "execution_timeout_minutes": self.execution_timeout_minutes,
            "sync_image": self.sync_image,
            "inject_flyte_env": self.inject_flyte_env,
        }
        optional = (
            "application_id",
            "application_name",
            "spark_submit_parameters",
            "application_configuration",
            "runtime_configuration",
            "scheduler_configuration",
            "auto_stop_config",
            "architecture",
            "retry_policy",
            "configuration_overrides",
            "tags",
            "initial_capacity",
            "maximum_capacity",
            "network_configuration",
            "image_configuration",
            "region",
            "client_token",
        )
        for name in optional:
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.spark_job_driver:
            result["spark_job_driver"] = self.spark_job_driver.to_dict()
        if self.hive_job_driver:
            result["hive_job_driver"] = self.hive_job_driver.to_dict()
        return result

    # Kept for compatibility with the initial Flyte SDK draft.
    def to_custom_config(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Serialize this configuration for the connector."""
        return self.to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EMRServerless":
        """Deserialize a configuration from `TaskTemplate.custom`."""
        spark = data.get("spark_job_driver")
        hive = data.get("hive_job_driver")
        return cls(
            execution_role_arn=data["execution_role_arn"],
            application_id=data.get("application_id"),
            release_label=data.get("release_label", "emr-7.0.0"),
            application_type=data.get("application_type", "SPARK"),
            application_name=data.get("application_name"),
            sync_image=data.get("sync_image", True),
            inject_flyte_env=data.get("inject_flyte_env", True),
            spark_job_driver=(
                EMRServerlessSparkJobDriver(
                    entry_point=spark["entryPoint"],
                    entry_point_arguments=spark.get("entryPointArguments"),
                    spark_submit_parameters=spark.get("sparkSubmitParameters"),
                )
                if spark
                else None
            ),
            hive_job_driver=(
                EMRServerlessHiveJobDriver(
                    query=hive["query"],
                    init_query_file=hive.get("initQueryFile"),
                    parameters=hive.get("parameters"),
                )
                if hive
                else None
            ),
            spark_submit_parameters=data.get("spark_submit_parameters"),
            application_configuration=data.get("application_configuration"),
            runtime_configuration=data.get("runtime_configuration"),
            scheduler_configuration=data.get("scheduler_configuration"),
            auto_stop_config=data.get("auto_stop_config"),
            architecture=data.get("architecture"),
            retry_policy=data.get("retry_policy"),
            configuration_overrides=data.get("configuration_overrides"),
            tags=data.get("tags"),
            execution_timeout_minutes=int(data.get("execution_timeout_minutes", 60)),
            initial_capacity=data.get("initial_capacity"),
            maximum_capacity=data.get("maximum_capacity"),
            network_configuration=data.get("network_configuration"),
            image_configuration=data.get("image_configuration"),
            region=data.get("region"),
            client_token=data.get("client_token"),
        )


class EMRServerlessFunctionTask(AsyncConnectorExecutorMixin, AsyncFunctionTaskTemplate):
    """A Flyte function task executed through EMR Serverless."""

    plugin_config: EMRServerless

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.task_type = TASK_TYPE

    def custom_config(self, sctx: SerializationContext) -> Dict[str, Any]:
        """Return the serialized EMR Serverless configuration."""
        if self.plugin_config.is_script_mode and self.interface.outputs:
            raise ValueError(
                "EMR Serverless script tasks cannot declare Flyte outputs; "
                "write results to durable storage and return no values"
            )
        return self.plugin_config.to_dict()


# Public parity name; the longer name remains for compatibility with the draft.
EMRServerlessTask = EMRServerlessFunctionTask


TaskPluginRegistry.register(EMRServerless, EMRServerlessFunctionTask)
