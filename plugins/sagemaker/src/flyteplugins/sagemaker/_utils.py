from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Mapping, Optional, Sequence

import xxhash
from botocore.response import StreamingBody
from flyteidl2.connector.connector_pb2 import TaskExecutionMetadata

_PLACEHOLDER = re.compile(r"\{(inputs|images)\.([^{}]+)\}")
_SAGEMAKER_NAME_LENGTHS = {
    "EndpointConfigName": 63,
    "EndpointName": 63,
    "HyperParameterTuningJobName": 32,
    "JobName": 64,
    "ModelName": 63,
    "ProcessingJobName": 63,
    "TrainingJobName": 63,
    "TransformJobName": 63,
}

TRITON_ACCOUNT_IDS = {
    "us-east-1": "785573368785",
    "us-east-2": "007439368137",
    "us-west-1": "710691900526",
    "us-west-2": "301217895009",
    "eu-west-1": "802834080501",
    "eu-west-2": "205493899709",
    "eu-west-3": "254080097072",
    "eu-north-1": "601324751636",
    "eu-south-1": "966458181534",
    "eu-central-1": "746233611703",
    "ap-east-1": "110948597952",
    "ap-south-1": "763008648453",
    "ap-northeast-1": "941853720454",
    "ap-northeast-2": "151534178276",
    "ap-southeast-1": "324986816169",
    "ap-southeast-2": "355873309152",
    "cn-northwest-1": "474822919863",
    "cn-north-1": "472730292857",
    "sa-east-1": "756306329178",
    "ca-central-1": "464438896020",
    "me-south-1": "836785723513",
    "af-south-1": "774647643957",
}


def normalize_integers(value: Any) -> Any:
    """Recursively turn protobuf-Struct whole-number floats back into integers."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [normalize_integers(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_integers(item) for key, item in value.items()}
    return value


def stable_value(value: Any) -> str:
    """Return a deterministic representation while preserving list order."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{stable_value(k)}: {stable_value(v)}" for k, v in sorted(value.items())) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(stable_value(item) for item in value) + "]"
    return str(value)


def json_safe(value: Any) -> Any:
    """Convert boto response values to Flyte's JSON-compatible native values."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, StreamingBody):
        return json_safe(value.read())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [json_safe(item) for item in value]
    return value


def render_config(
    config: Any,
    *,
    inputs: Optional[dict[str, Any]] = None,
    images: Optional[dict[str, str]] = None,
    region: Optional[str] = None,
    metadata: Optional[TaskExecutionMetadata] = None,
) -> tuple[Any, str]:
    """Render public Flytekit input, image and idempotence placeholders."""
    values = {"inputs": inputs or {}, "images": images or {}}

    def render(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: render(item) for key, item in value.items()}
        if isinstance(value, list):
            return [render(item) for item in value]
        if not isinstance(value, str):
            return value
        exact = _PLACEHOLDER.fullmatch(value)
        if exact:
            group, key = exact.groups()
            if key not in values[group]:
                raise ValueError(f"Missing value for {{{group}.{key}}}")
            return values[group][key]

        def replace(match: re.Match[str]) -> str:
            group, key = match.groups()
            if key not in values[group]:
                raise ValueError(f"Missing value for {{{group}.{key}}}")
            return str(values[group][key])

        return _PLACEHOLDER.sub(replace, value)

    rendered = normalize_integers(render(config))
    token = ""
    if "{idempotence_token}" in str(rendered):
        # Match the public Flytekit connector: hash the fully input/image-rendered
        # request, with deterministic dict order and significant list order.
        token = xxhash.xxh64(stable_value(rendered).encode()).hexdigest()

        def replace_token(value: Any) -> Any:
            if isinstance(value, str):
                return value.replace("{idempotence_token}", token)
            if isinstance(value, list):
                return [replace_token(item) for item in value]
            if isinstance(value, dict):
                return {key: replace_token(item) for key, item in value.items()}
            return value

        rendered = replace_token(rendered)

    def truncate_names(value: Any) -> Any:
        if isinstance(value, list):
            return [truncate_names(item) for item in value]
        if isinstance(value, dict):
            result = {key: truncate_names(item) for key, item in value.items()}
            for key, limit in _SAGEMAKER_NAME_LENGTHS.items():
                if key in result and isinstance(result[key], str):
                    result[key] = result[key][:limit]
            return result
        return value

    rendered = truncate_names(rendered)
    return rendered, token


def resolve_region(configured: Optional[str], inputs: Optional[dict[str, Any]]) -> str:
    region = (inputs or {}).get("region") or configured
    if not region:
        raise ValueError("Region parameter is required.")
    return str(region)


def resolve_images(images: Optional[dict[str, Any]], region: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, image in (images or {}).items():
        value = getattr(image, "uri", image)
        if not isinstance(value, str):
            raise TypeError(f"Image {name!r} must be a URI string or flyte.Image")
        if "sagemaker-tritonserver" in value:
            if region not in TRITON_ACCOUNT_IDS:
                raise ValueError(f"No SageMaker Triton image account is known for region {region!r}")
            base = "amazonaws.com.cn" if region.startswith("cn-") else "amazonaws.com"
            value = value.format(account_id=TRITON_ACCOUNT_IDS[region], region=region, base=base)
        result[name] = value
    return result


def triton_image_uri(version: str = "23.12") -> str:
    """Return a region-rendered SageMaker Triton image URI template."""
    return f"{{account_id}}.dkr.ecr.{{region}}.{{base}}/sagemaker-tritonserver:{version}-py3"
