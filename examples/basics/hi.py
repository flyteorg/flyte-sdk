from kubernetes.client import (
    V1Container,
    V1PodSpec,
)

import flyte

pod_template = flyte.PodTemplate(
    primary_container_name="primary",
    labels={
        "kueue.x-k8s.io/queue-name": "flytesnacks-prod-queue",
    },
    pod_spec=V1PodSpec(
        containers=[V1Container(name="primary")],
    ),
)

env = flyte.TaskEnvironment(
    name="hello_world", pod_template=pod_template, image="chrismatteson/flyte:ce80b2d93da5278106047e5ec697a7d2"
)


@env.task
def fn(x: int) -> int:
    slope, intercept = 2, 5
    return slope * x + intercept


@env.task
def main(n: int) -> float:
    y_list = list(flyte.map(fn, range(n)))
    return sum(y_list) / len(y_list)
