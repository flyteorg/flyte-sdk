from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from flyteplugins.sagemaker.task import (
    SageMakerDeleteEndpointConfigTask,
    SageMakerDeleteEndpointTask,
    SageMakerDeleteModelTask,
    SageMakerEndpointConfigTask,
    SageMakerEndpointTask,
    SageMakerModelTask,
)


def _append_token(config: Dict[str, Any], key: str, token: str, name: str) -> None:
    config[key] = f"{config.get(key, name)}-{{{token}}}"


def _result(value: Any) -> tuple[dict[str, Any], str]:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    if isinstance(value, dict) and "result" in value:
        return value["result"], value.get("idempotence_token", "")
    raise TypeError(f"Unexpected Boto task result {value!r}")


@dataclass
class SageMakerDeployment:
    """Flyte 2 callable composition for model, endpoint config and endpoint creation."""

    model_task: SageMakerModelTask
    endpoint_config_task: SageMakerEndpointConfigTask
    endpoint_task: SageMakerEndpointTask
    region_at_runtime: bool = False

    async def __call__(self, **inputs: Any) -> list[dict[str, Any]]:
        model, model_token = _result(await self.model_task.aio(**inputs))
        endpoint_config_inputs = {**inputs, "idempotence_token": model_token}
        endpoint_config, endpoint_token = _result(await self.endpoint_config_task.aio(**endpoint_config_inputs))
        endpoint_inputs = {**inputs, "idempotence_token": endpoint_token}
        endpoint = await self.endpoint_task.aio(**endpoint_inputs)
        if isinstance(endpoint, tuple):
            endpoint = endpoint[0]
        return [model, endpoint_config, endpoint]


@dataclass
class SageMakerDeploymentDeletion:
    """Flyte 2 callable composition for endpoint deployment deletion."""

    endpoint_task: SageMakerDeleteEndpointTask
    endpoint_config_task: SageMakerDeleteEndpointConfigTask
    model_task: SageMakerDeleteModelTask

    async def __call__(
        self,
        endpoint_name: str,
        endpoint_config_name: str,
        model_name: str,
        **inputs: Any,
    ) -> None:
        await self.endpoint_task.aio(endpoint_name=endpoint_name, **inputs)
        await self.endpoint_config_task.aio(endpoint_config_name=endpoint_config_name, **inputs)
        await self.model_task.aio(model_name=model_name, **inputs)


def create_sagemaker_deployment(
    name: str,
    model_config: Dict[str, Any],
    endpoint_config_config: Dict[str, Any],
    endpoint_config: Dict[str, Any],
    images: Optional[Dict[str, Any]] = None,
    model_input_types: Optional[Dict[str, Type]] = None,
    endpoint_config_input_types: Optional[Dict[str, Type]] = None,
    endpoint_input_types: Optional[Dict[str, Type]] = None,
    region: Optional[str] = None,
    region_at_runtime: bool = False,
    idempotence_token: bool = True,
) -> SageMakerDeployment:
    """Build a Flyte 2 deployment composition with the public Flytekit semantics."""
    if not region and not region_at_runtime:
        raise ValueError("Region parameter is required.")
    model = copy.deepcopy(model_config)
    endpoint_cfg = copy.deepcopy(endpoint_config_config)
    endpoint = copy.deepcopy(endpoint_config)
    if idempotence_token:
        _append_token(model, "ModelName", "idempotence_token", name)
        _append_token(endpoint_cfg, "EndpointConfigName", "idempotence_token", name)
        if endpoint_cfg.get("ProductionVariants"):
            _append_token(endpoint_cfg["ProductionVariants"][0], "ModelName", "inputs.idempotence_token", name)
        _append_token(endpoint, "EndpointName", "idempotence_token", name)
        _append_token(endpoint, "EndpointConfigName", "inputs.idempotence_token", name)

    runtime = {"region": str} if region_at_runtime else {}
    model_inputs = {**(model_input_types or {}), **runtime}
    endpoint_cfg_inputs = {**(endpoint_config_input_types or {}), "idempotence_token": str, **runtime}
    endpoint_inputs = {**(endpoint_input_types or {}), "idempotence_token": str, **runtime}
    return SageMakerDeployment(
        model_task=SageMakerModelTask(
            name=f"sagemaker-model-{name}",
            config=model,
            region=region,
            images=images,
            inputs=model_inputs,
        ),
        endpoint_config_task=SageMakerEndpointConfigTask(
            name=f"sagemaker-endpoint-config-{name}",
            config=endpoint_cfg,
            region=region,
            inputs=endpoint_cfg_inputs,
        ),
        endpoint_task=SageMakerEndpointTask(
            name=f"sagemaker-endpoint-{name}",
            config=endpoint,
            region=region,
            inputs=endpoint_inputs,
        ),
        region_at_runtime=region_at_runtime,
    )


def delete_sagemaker_deployment(
    name: str,
    region: Optional[str] = None,
    region_at_runtime: bool = False,
) -> SageMakerDeploymentDeletion:
    """Build a Flyte 2 endpoint, endpoint-config and model deletion composition."""
    if not region and not region_at_runtime:
        raise ValueError("Region parameter is required.")
    runtime = {"region": str} if region_at_runtime else {}
    return SageMakerDeploymentDeletion(
        endpoint_task=SageMakerDeleteEndpointTask(
            name=f"sagemaker-delete-endpoint-{name}",
            config={"EndpointName": "{inputs.endpoint_name}"},
            region=region,
            inputs={"endpoint_name": str, **runtime},
        ),
        endpoint_config_task=SageMakerDeleteEndpointConfigTask(
            name=f"sagemaker-delete-endpoint-config-{name}",
            config={"EndpointConfigName": "{inputs.endpoint_config_name}"},
            region=region,
            inputs={"endpoint_config_name": str, **runtime},
        ),
        model_task=SageMakerDeleteModelTask(
            name=f"sagemaker-delete-model-{name}",
            config={"ModelName": "{inputs.model_name}"},
            region=region,
            inputs={"model_name": str, **runtime},
        ),
    )
