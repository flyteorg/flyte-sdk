"""Tests for the `clustered` launcher.

The launcher (`flyte._bin.clustered`) runs as the pod PID 1, derives the process topology from
JobSet env vars, and execs the runtime selected by `--runtime` with `a0 <args>` as the worker
command — torchrun re-spawns `a0` per process, the jax runtime execs `a0` directly. Each worker is
the standard `a0` runtime entrypoint; the launcher does not run the task itself.
"""

import os
import socket
import sys
from unittest.mock import patch

import pytest

from flyte._bin.clustered import (
    _exec_jax_launcher,
    _exec_torchrun_launcher,
    _master_addr,
    _strip_option,
    _wait_for_dns,
)

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

# What the (unchanged) Go plugin injects that the jax runtime consumes: no RDZV_BACKEND, one process per pod.
JAX_ENV = {**{k: v for k, v in BASE_ENV.items() if k != "RDZV_BACKEND"}, "NPROC_PER_NODE": "1"}

EXPECTED_MASTER = "f-abc123-workers-0-0.f-abc123.my-project-development.svc.cluster.local"

WORKER_ARGV = ["a0", "--inputs", "s3://bucket/in"]

ACTION_ARGS = ["--inputs", "i", "--outputs-path", "o", "--version", "v", "--run-base-dir", "b"]

# Everything a launcher may export; captured at exec time and restored by patch.dict afterwards.
EXPORTED_VARS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "NODE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "JAX_COORDINATOR_ADDRESS",
    "FLYTE_CLUSTERED_WORKER",
)


@pytest.fixture(autouse=True)
def _restore_environ():
    """The launchers export topology (and the clustered-worker marker) straight into os.environ before
    exec; monkeypatch only restores what it set itself, so snapshot and restore the whole environment
    to keep those exports from leaking into later tests (a stray FLYTE_CLUSTERED_WORKER would disable
    the controller in unrelated `a0` tests)."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


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
    """`clustered` is a pure launcher: by default it execs torchrun with `a0` as the worker command."""
    from click.testing import CliRunner

    from flyte._bin import clustered

    captured = {}

    def fake_launcher(worker_argv):
        captured["worker_argv"] = worker_argv

    monkeypatch.setattr(clustered, "_exec_torchrun_launcher", fake_launcher)
    monkeypatch.setattr(clustered, "_exec_jax_launcher", lambda _argv: pytest.fail("jax launcher must not run"))

    result = CliRunner().invoke(clustered.main, ACTION_ARGS)

    assert result.exit_code == 0, result.output
    assert captured["worker_argv"][0] == "a0"


def test_torchrun_exports_clustered_worker_marker():
    """Every worker torchrun spawns inherits the launcher-neutral marker (torchrun copies os.environ)."""
    with (
        patch.dict(os.environ, BASE_ENV),
        patch("socket.getaddrinfo"),
        patch("shutil.which", return_value="/usr/bin/torchrun"),
        patch("os.execvp"),
    ):
        os.environ.pop("FLYTE_CLUSTERED_WORKER", None)
        _exec_torchrun_launcher(WORKER_ARGV)
        exported = {k: os.environ.get(k) for k in ("FLYTE_CLUSTERED_WORKER", "NODE_RANK", "MASTER_ADDR")}

    assert exported == {"FLYTE_CLUSTERED_WORKER": "1", "NODE_RANK": "2", "MASTER_ADDR": EXPECTED_MASTER}


def test_torchrun_requires_rdzv_backend():
    """RDZV_BACKEND is torchrun-only: absent (as for the jax runtime) the torchrun path exits 1."""
    with patch.dict(os.environ, JAX_ENV), patch("os.execvp") as mock_exec:
        os.environ.pop("RDZV_BACKEND", None)
        with pytest.raises(SystemExit) as exc_info:
            _exec_torchrun_launcher(WORKER_ARGV)

    assert exc_info.value.code == 1
    mock_exec.assert_not_called()


# ---------------------------------------------------------------------------
# jax runtime
# ---------------------------------------------------------------------------


def test_jax_happy_path():
    """DNS resolves — execs `a0` directly with the one-process-per-pod topology exported."""
    with (
        patch.dict(os.environ, JAX_ENV),
        patch("socket.getaddrinfo"),
        patch("os.execvp") as mock_exec,
    ):
        for k in EXPORTED_VARS:
            os.environ.pop(k, None)
        os.environ["MASTER_PORT"] = "29500"
        _exec_jax_launcher(WORKER_ARGV)
        exported = {k: os.environ.get(k) for k in EXPORTED_VARS}

    mock_exec.assert_called_once_with("a0", WORKER_ARGV)
    assert exported == {
        "RANK": "2",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "4",
        "LOCAL_WORLD_SIZE": "1",
        "NODE_RANK": "2",
        "MASTER_ADDR": EXPECTED_MASTER,
        "MASTER_PORT": "29500",
        "JAX_COORDINATOR_ADDRESS": f"{EXPECTED_MASTER}:29500",
        "FLYTE_CLUSTERED_WORKER": "1",
    }


def test_jax_master_port_default():
    """MASTER_PORT not injected — defaults to 29500 and is exported for flyte.ctx().master_port."""
    with patch.dict(os.environ, JAX_ENV), patch("socket.getaddrinfo"), patch("os.execvp"):
        os.environ.pop("MASTER_PORT", None)
        _exec_jax_launcher(WORKER_ARGV)
        exported = {k: os.environ.get(k) for k in ("MASTER_PORT", "JAX_COORDINATOR_ADDRESS")}

    assert exported == {"MASTER_PORT": "29500", "JAX_COORDINATOR_ADDRESS": f"{EXPECTED_MASTER}:29500"}


def test_jax_rejects_multi_proc():
    """One JAX process per pod: NPROC_PER_NODE != 1 exits 1 before DNS wait or exec."""
    with (
        patch.dict(os.environ, {**JAX_ENV, "NPROC_PER_NODE": "8"}),
        patch("socket.getaddrinfo") as mock_dns,
        patch("os.execvp") as mock_exec,
    ):
        with pytest.raises(SystemExit) as exc_info:
            _exec_jax_launcher(WORKER_ARGV)

    assert exc_info.value.code == 1
    mock_dns.assert_not_called()
    mock_exec.assert_not_called()


def test_jax_missing_required_env_var():
    with patch.dict(os.environ, JAX_ENV), patch("os.execvp") as mock_exec:
        os.environ.pop("NNODES", None)
        with pytest.raises(SystemExit) as exc_info:
            _exec_jax_launcher(WORKER_ARGV)

    assert exc_info.value.code == 1
    mock_exec.assert_not_called()


def test_jax_does_not_require_torchrun():
    """No launcher binary is involved: a torchrun-less image must not be rejected."""
    with (
        patch.dict(os.environ, JAX_ENV),
        patch("socket.getaddrinfo"),
        patch("shutil.which", return_value=None),
        patch("os.execvp") as mock_exec,
    ):
        _exec_jax_launcher(WORKER_ARGV)

    mock_exec.assert_called_once_with("a0", WORKER_ARGV)


# ---------------------------------------------------------------------------
# --runtime dispatch
# ---------------------------------------------------------------------------


def test_launcher_dispatches_jax(monkeypatch):
    """`--runtime=jax` (emitted by the SDK in the container args) selects the jax launcher and is
    stripped from the argv forwarded to `a0`."""
    from click.testing import CliRunner

    from flyte._bin import clustered

    captured = {}
    monkeypatch.setattr(clustered, "_exec_jax_launcher", lambda worker_argv: captured.setdefault("argv", worker_argv))
    monkeypatch.setattr(
        clustered, "_exec_torchrun_launcher", lambda _argv: pytest.fail("torchrun launcher must not run")
    )

    argv = ["--runtime=jax", *ACTION_ARGS]
    with patch.object(sys, "argv", ["clustered", *argv]):
        result = CliRunner().invoke(clustered.main, argv)

    assert result.exit_code == 0, result.output
    assert captured["argv"] == ["a0", *ACTION_ARGS]


def test_launcher_rejects_unknown_runtime(monkeypatch):
    from click.testing import CliRunner

    from flyte._bin import clustered

    monkeypatch.setattr(clustered, "_exec_torchrun_launcher", lambda _argv: pytest.fail("must not launch"))
    monkeypatch.setattr(clustered, "_exec_jax_launcher", lambda _argv: pytest.fail("must not launch"))

    result = CliRunner().invoke(clustered.main, ["--runtime=mpi", *ACTION_ARGS])

    assert result.exit_code != 0
    assert "--runtime" in result.output


def test_strip_option_handles_both_forms():
    assert _strip_option(["--runtime=jax", "--inputs", "i"], "--runtime") == ["--inputs", "i"]
    assert _strip_option(["--runtime", "jax", "--inputs", "i"], "--runtime") == ["--inputs", "i"]
    assert _strip_option(["--inputs", "i"], "--runtime") == ["--inputs", "i"]
    assert _strip_option(["--inputs", "--runtime=jax"], "--runtime") == ["--inputs"]
