# /// script
# requires-python = "==3.12"
# dependencies = [
#    "kubernetes",
# ]
# ///

from kubernetes.client import (
    V1Container,
    V1EnvVar,
    V1LocalObjectReference,
    V1PodSpec,
)

import flyte

pod_template = flyte.PodTemplate(
    primary_container_name="primary",
    labels={"lKeyA": "lValA"},
    annotations={"aKeyA": "aValA"},
    pod_spec=V1PodSpec(
        containers=[V1Container(name="primary", env=[V1EnvVar(name="hello", value="world")])],
        image_pull_secrets=[V1LocalObjectReference(name="regcred-test")],
    ),
)

env = flyte.TaskEnvironment(
    name="hello_world", pod_template=pod_template, image="ghcr.io/flyteorg/flyte:py3.12-v0.2.0b8"
)


@env.task
async def say_hello(data: str) -> str:
    return f"Hello {data}"


@env.task
async def say_hello_nested(data: str = "default string") -> str:
    return await say_hello(data=data)


if __name__ == "__main__":
    flyte.init("../../config.yaml")
    result = flyte.run(say_hello_nested, data="hello world")
    print(result)
