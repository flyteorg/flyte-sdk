"""Tests for the `clustered` torchrun launcher.

The launcher (`flyte._bin.clustered`) runs as the pod PID 1, derives the torchrun rendezvous from
JobSet env vars, and execs `torchrun ... -- a0 <args>` — each worker is the standard `a0` runtime
entrypoint. The launcher does not run the task itself.
"""

import socket
from unittest.mock import patch

import pytest

from flyte._bin.clustered import _exec_torchrun_launcher, _master_addr, _wait_for_dns

BASE_ENV = {
    "JOBSET_NAME": "f-abc123",
    "POD_NAMESPACE": "my-project-development",
    "JOB_COMPLETION_INDEX": "2",
    "JOBSET_RESTART_ATTEMPT": "1",
    "NNODES": "4",
    "NPROC_PER_NODE": "8",
    "RDZV_BACKEND": "static",
    "MASTER_PORT": "29500",
}

EXPECTED_MASTER = "f-abc123-workers-0-0.f-abc123.my-project-development.svc.cluster.local"

WORKER_ARGV = ["a0", "--inputs", "s3://bucket/in"]


def test_master_addr_format():
    assert _master_addr("f-abc123", "my-project-development") == EXPECTED_MASTER


def test_happy_path(monkeypatch):
    """DNS resolves immediately — execvp called with correct torchrun args + worker argv."""
    for k, v in BASE_ENV.items():
        monkeypatch.setenv(k, v)

    with (
        patch("socket.getaddrinfo"),
        patch("shutil.which", return_value="/usr/bin/torchrun"),
        patch("os.execvp") as mock_exec,
    ):
        _exec_torchrun_launcher(WORKER_ARGV)

    mock_exec.assert_called_once_with(
        "torchrun",
        [
            "torchrun",
            "--nnodes=4",
            "--nproc-per-node=8",
            "--node-rank=2",
            "--rdzv-backend=static",
            "--rdzv-id=f-abc123-1",
            f"--rdzv-endpoint={EXPECTED_MASTER}:29500",
            "--no-python",
            "--",
            "a0",
            "--inputs",
            "s3://bucket/in",
        ],
    )


def test_master_port_default(monkeypatch):
    """MASTER_PORT not set — defaults to 29500."""
    for k, v in BASE_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("MASTER_PORT", raising=False)

    with (
        patch("socket.getaddrinfo"),
        patch("shutil.which", return_value="/usr/bin/torchrun"),
        patch("os.execvp") as mock_exec,
    ):
        _exec_torchrun_launcher(WORKER_ARGV)

    endpoint_arg = next(a for a in mock_exec.call_args[0][1] if a.startswith("--rdzv-endpoint="))
    assert endpoint_arg.endswith(":29500")


def test_rdzv_id_rotation(monkeypatch):
    """RDZV_ID = JOBSET_NAME-JOBSET_RESTART_ATTEMPT."""
    for k, v in BASE_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("JOBSET_RESTART_ATTEMPT", "3")

    with (
        patch("socket.getaddrinfo"),
        patch("shutil.which", return_value="/usr/bin/torchrun"),
        patch("os.execvp") as mock_exec,
    ):
        _exec_torchrun_launcher(WORKER_ARGV)

    rdzv_id_arg = next(a for a in mock_exec.call_args[0][1] if a.startswith("--rdzv-id="))
    assert rdzv_id_arg == "--rdzv-id=f-abc123-3"


def test_node_rank_from_completion_index(monkeypatch):
    """NODE_RANK == JOB_COMPLETION_INDEX."""
    for k, v in BASE_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("JOB_COMPLETION_INDEX", "3")

    with (
        patch("socket.getaddrinfo"),
        patch("shutil.which", return_value="/usr/bin/torchrun"),
        patch("os.execvp") as mock_exec,
    ):
        _exec_torchrun_launcher(WORKER_ARGV)

    node_rank_arg = next(a for a in mock_exec.call_args[0][1] if a.startswith("--node-rank="))
    assert node_rank_arg == "--node-rank=3"


def test_missing_required_env_var(monkeypatch):
    """Missing required env var exits with code 1."""
    for k, v in BASE_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("NNODES")

    with pytest.raises(SystemExit) as exc_info:
        _exec_torchrun_launcher(WORKER_ARGV)

    assert exc_info.value.code == 1


def test_missing_torchrun_exits(monkeypatch):
    """torchrun not installed — exits with code 1 and a helpful message."""
    for k, v in BASE_ENV.items():
        monkeypatch.setenv(k, v)

    with patch("socket.getaddrinfo"), patch("shutil.which", return_value=None):
        with pytest.raises(SystemExit) as exc_info:
            _exec_torchrun_launcher(WORKER_ARGV)

    assert exc_info.value.code == 1


def test_dns_timeout_exits():
    """DNS never resolves — exits with code 1 after timeout."""
    with patch("socket.getaddrinfo", side_effect=socket.gaierror("no such host")):
        with pytest.raises(SystemExit) as exc_info:
            _wait_for_dns(EXPECTED_MASTER, timeout=0.1, interval=0.05)

    assert exc_info.value.code == 1


def test_launcher_execs_torchrun_with_a0_worker(monkeypatch):
    """`clustered` is a pure launcher: it execs torchrun with `a0` as the worker command."""
    from click.testing import CliRunner

    from flyte._bin import clustered

    captured = {}

    def fake_launcher(worker_argv):
        captured["worker_argv"] = worker_argv

    monkeypatch.setattr(clustered, "_exec_torchrun_launcher", fake_launcher)

    result = CliRunner().invoke(
        clustered.main,
        ["--inputs", "i", "--outputs-path", "o", "--version", "v", "--run-base-dir", "b"],
    )

    assert result.exit_code == 0, result.output
    assert captured["worker_argv"][0] == "a0"
