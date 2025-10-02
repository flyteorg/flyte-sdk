import flyte
from flyte.models import SerializationContext

from flyteplugins.pytorch.task import (
    Elastic,
    RunPolicy,
    TorchFunctionTask,
)


def test_torch_post_init():
    t = Elastic(nnodes=2, nproc_per_node=1)

    task = TorchFunctionTask(
        name="n",
        interface=None,
        func=lambda: None,
        image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime",
        resources=flyte.Resources(cpu=1, memory="1Gi"),
        plugin_config=t,
    )

    assert task.max_nodes == 2
    assert task.task_type == "pytorch"


def test_custom_config():
    sctx = SerializationContext(
        version="123",
    )

    torch = Elastic(
        nnodes=2,
        nproc_per_node=2,
        run_policy=RunPolicy(
            clean_pod_policy="all",
            backoff_limit=4,
            ttl_seconds_after_finished=100,
            active_deadline_seconds=200,
        ),
    )
    task = TorchFunctionTask(
        name="n",
        interface=None,
        func=lambda: None,
        image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime",
        resources=flyte.Resources(cpu=1, memory="1Gi"),
        plugin_config=torch,
    )
    result = task.custom_config(sctx)

    expect = {
        "workerReplicas": {"replicas": 2},
        "runPolicy": {
            "cleanPodPolicy": "CLEANPOD_POLICY_ALL",
            "ttlSecondsAfterFinished": 100,
            "activeDeadlineSeconds": 200,
            "backoffLimit": 4,
        },
        "elasticConfig": {
            "rdzvBackend": "c10d",
            "minReplicas": 2,
            "maxReplicas": 2,
            "nprocPerNode": 2,
            "maxRestarts": 3,
        },
    }

    assert result is not None
    assert result["elasticConfig"] == expect["elasticConfig"]
