# from unittest.mock import MagicMock, patch

import flyte
from flyte.models import SerializationContext

from flyteplugins.pytorch.task import (
    CleanPodPolicy,
    MasterNodeConfig,
    RunPolicy,
    TorchFunctionTask,
    TorchJobConfig,
    WorkerNodeConfig,
)


def test_torch_post_init():
    t = TorchJobConfig()
    assert isinstance(t.rdzv_configs, dict)


def test_custom_config():
    sctx = SerializationContext(
        version="123",
    )

    torch = TorchJobConfig(
        nnodes=2,
        nproc_per_node=2,
        master_node_config=MasterNodeConfig(
            replicas=1,
        ),
        worker_node_config=WorkerNodeConfig(
            replicas=2,
        ),
        run_policy=RunPolicy(
            clean_pod_policy=CleanPodPolicy.ALL,
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
        "workerReplicas": {
            "replicas": 2,
            "resources": {"requests": [{"name": "CPU", "value": "1"}, {"name": "MEMORY", "value": "1Gi"}]},
            "image": "Image(base_image='pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime', dockerfile=None,"
            " registry=None, name=None, platform=('linux/amd64',), python_version=(3, 13),"
            " _identifier_override=None, _layers=(), _tag=None)",
            "restartPolicy": "RESTART_POLICY_ON_FAILURE",
        },
        "masterReplicas": {
            "replicas": 1,
            "image": "Image(base_image='pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime', dockerfile=None, registry=None,"
            " name=None, platform=('linux/amd64',), python_version=(3, 13),"
            " _identifier_override=None, _layers=(), _tag=None)",
            "resources": {"requests": [{"name": "CPU", "value": "1"}, {"name": "MEMORY", "value": "1Gi"}]},
            "restartPolicy": "RESTART_POLICY_ON_FAILURE",
        },
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

    assert result["workerReplicas"] == expect["workerReplicas"]
    assert result["masterReplicas"] == expect["masterReplicas"]
    assert result["elasticConfig"] == expect["elasticConfig"]
