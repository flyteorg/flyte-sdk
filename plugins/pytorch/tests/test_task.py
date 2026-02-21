import asyncio
import os

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


# --- Elastic nccl_heartbeat_timeout_sec tests ---


def test_elastic_nccl_heartbeat_default():
    e = Elastic(nnodes=2, nproc_per_node=1)
    assert e.nccl_heartbeat_timeout_sec == 300


def test_elastic_nccl_heartbeat_custom():
    e = Elastic(nnodes=2, nproc_per_node=1, nccl_heartbeat_timeout_sec=600)
    assert e.nccl_heartbeat_timeout_sec == 600


def test_elastic_nccl_heartbeat_none():
    e = Elastic(nnodes=2, nproc_per_node=1, nccl_heartbeat_timeout_sec=None)
    assert e.nccl_heartbeat_timeout_sec is None


# --- pre() env var tests ---


def _make_task(nproc_per_node=2, nccl_heartbeat_timeout_sec=300):
    cfg = Elastic(
        nnodes=2,
        nproc_per_node=nproc_per_node,
        nccl_heartbeat_timeout_sec=nccl_heartbeat_timeout_sec,
    )
    return TorchFunctionTask(
        name="t",
        interface=None,
        func=lambda: None,
        image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime",
        resources=flyte.Resources(cpu=1, memory="1Gi"),
        plugin_config=cfg,
    )


def test_pre_sets_nccl_heartbeat_env(monkeypatch):
    monkeypatch.delenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    task = _make_task(nproc_per_node=2, nccl_heartbeat_timeout_sec=300)
    asyncio.run(task.pre())
    assert os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] == "300"


def test_pre_sets_omp_num_threads(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", raising=False)
    task = _make_task(nproc_per_node=4)
    asyncio.run(task.pre())
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_pre_skips_omp_when_single_proc(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", raising=False)
    task = _make_task(nproc_per_node=1, nccl_heartbeat_timeout_sec=None)
    asyncio.run(task.pre())
    assert "OMP_NUM_THREADS" not in os.environ


def test_pre_does_not_override_existing_nccl_env(monkeypatch):
    monkeypatch.setenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "999")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    task = _make_task(nproc_per_node=2, nccl_heartbeat_timeout_sec=300)
    asyncio.run(task.pre())
    assert os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] == "999"


def test_pre_skips_nccl_when_none(monkeypatch):
    monkeypatch.delenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    task = _make_task(nproc_per_node=2, nccl_heartbeat_timeout_sec=None)
    asyncio.run(task.pre())
    assert "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC" not in os.environ
