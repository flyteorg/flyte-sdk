import asyncio
import os

import flyte
from cloudpickle import cloudpickle
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


def _make_task(
    nproc_per_node=2,
    nccl_heartbeat_timeout_sec=300,
    nccl_async_error_handling=False,
    nccl_collective_timeout_sec=None,
):
    cfg = Elastic(
        nnodes=2,
        nproc_per_node=nproc_per_node,
        nccl_heartbeat_timeout_sec=nccl_heartbeat_timeout_sec,
        nccl_async_error_handling=nccl_async_error_handling,
        nccl_collective_timeout_sec=nccl_collective_timeout_sec,
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


# --- nccl_async_error_handling tests ---


def test_elastic_nccl_async_error_handling_default():
    e = Elastic(nnodes=2, nproc_per_node=1)
    assert e.nccl_async_error_handling is False


def test_pre_sets_async_error_handling_when_enabled(monkeypatch):
    monkeypatch.delenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", raising=False)
    task = _make_task(nccl_async_error_handling=True)
    asyncio.run(task.pre())
    assert os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "1"


def test_pre_skips_async_error_handling_when_disabled(monkeypatch):
    monkeypatch.delenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", raising=False)
    task = _make_task(nccl_async_error_handling=False)
    asyncio.run(task.pre())
    assert "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in os.environ


def test_pre_does_not_override_existing_async_error_handling(monkeypatch):
    monkeypatch.setenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", raising=False)
    task = _make_task(nccl_async_error_handling=True)
    asyncio.run(task.pre())
    assert os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "0"


# --- nccl_collective_timeout_sec tests ---


def test_elastic_nccl_collective_timeout_default():
    e = Elastic(nnodes=2, nproc_per_node=1)
    assert e.nccl_collective_timeout_sec is None


def test_pre_sets_collective_timeout_env(monkeypatch):
    monkeypatch.delenv("FLYTE_NCCL_COLLECTIVE_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", raising=False)
    task = _make_task(nccl_collective_timeout_sec=60)
    asyncio.run(task.pre())
    assert os.environ["FLYTE_NCCL_COLLECTIVE_TIMEOUT_SEC"] == "60"


def test_pre_skips_collective_timeout_when_none(monkeypatch):
    monkeypatch.delenv("FLYTE_NCCL_COLLECTIVE_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", raising=False)
    task = _make_task(nccl_collective_timeout_sec=None)
    asyncio.run(task.pre())
    assert "FLYTE_NCCL_COLLECTIVE_TIMEOUT_SEC" not in os.environ


def test_pre_does_not_override_existing_collective_timeout(monkeypatch):
    monkeypatch.setenv("FLYTE_NCCL_COLLECTIVE_TIMEOUT_SEC", "120")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", raising=False)
    task = _make_task(nccl_collective_timeout_sec=60)
    asyncio.run(task.pre())
    assert os.environ["FLYTE_NCCL_COLLECTIVE_TIMEOUT_SEC"] == "120"


def test_launcher_entrypoint_overrides_nccl_default(monkeypatch):
    """Verify launcher_entrypoint patches both PyTorch modules when env var is set."""
    from datetime import timedelta
    from unittest.mock import MagicMock

    import torch.distributed.constants
    import torch.distributed.distributed_c10d

    from flyteplugins.pytorch.task import launcher_entrypoint

    orig_constants = torch.distributed.constants.default_pg_nccl_timeout
    orig_c10d = torch.distributed.distributed_c10d.default_pg_nccl_timeout
    monkeypatch.setenv("FLYTE_NCCL_COLLECTIVE_TIMEOUT_SEC", "45")

    # Minimal mock TaskContext so launcher_entrypoint doesn't do real init
    tctx = MagicMock()
    fn = cloudpickle.dumps(lambda **kw: kw.get("x"))
    try:
        launcher_entrypoint(tctx, fn, {"x": 1})
    except Exception:
        pass  # flyte.init will fail in test, that's fine

    assert torch.distributed.constants.default_pg_nccl_timeout == timedelta(seconds=45)
    assert torch.distributed.distributed_c10d.default_pg_nccl_timeout == timedelta(seconds=45)
    # restore
    torch.distributed.constants.default_pg_nccl_timeout = orig_constants
    torch.distributed.distributed_c10d.default_pg_nccl_timeout = orig_c10d
