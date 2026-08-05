import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner
from flyteidl2.core import execution_pb2

import flyte.errors
from flyte._bin.runtime import main


def test_a0_disables_controller_under_torchrun(monkeypatch):
    """`a0` workers spawned by torchrun (clustered tasks) run with no controller.

    The launcher execs `torchrun ... -- a0 <args>`; torchrun sets TORCHELASTIC_RUN_ID on every
    worker, which `a0` uses to gate the controller off (clustered tasks never enqueue subtasks).
    """
    import flyte._bin.runtime as runtime

    captured = {}

    def fake_run_action(ctx, *, controller_enabled, **params):
        captured["controller_enabled"] = controller_enabled

    monkeypatch.setattr(runtime, "_run_action", fake_run_action)
    base_args = ["--inputs", "i", "--outputs-path", "o", "--version", "v", "--run-base-dir", "b"]

    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "run-xyz")
    assert CliRunner().invoke(main, base_args).exit_code == 0
    assert captured["controller_enabled"] is False

    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    assert CliRunner().invoke(main, base_args).exit_code == 0
    assert captured["controller_enabled"] is True


def test_reexec_under_wrapper(monkeypatch):
    """`_F_EXEC_WRAPPER` re-execs the task under a wrapper (e.g. `nsys launch`), once.

    The wrapper string is tokenized and env-expanded, then prepended to the original argv. The
    _F_EXEC_WRAPPED guard and TORCHELASTIC_RUN_ID (clustered workers) both short-circuit it, and an
    unset variable is a no-op so the normal startup path is untouched.
    """
    import flyte._bin.runtime as runtime

    argv = ["/opt/venv/bin/a0", "--inputs", "x"]

    def call(env):
        for k in ("_F_EXEC_WRAPPER", "_F_EXEC_WRAPPED", "TORCHELASTIC_RUN_ID", "_F_EXEC_WRAPPER_CLUSTERED", "RANK"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        with patch.object(sys, "argv", list(argv)), patch.object(os, "execvp") as ex:
            runtime._maybe_reexec_under_wrapper()
        return ex.call_args

    # No wrapper -> no re-exec.
    assert call({}) is None

    # Wrapper set -> exec `nsys ... <expanded> <original argv>`; $ACTION_NAME is expanded.
    ca = call({"_F_EXEC_WRAPPER": "nsys launch --session-new=flyte-$ACTION_NAME", "ACTION_NAME": "a7"})
    assert ca.args == (
        "nsys",
        ["nsys", "launch", "--session-new=flyte-a7", "/opt/venv/bin/a0", "--inputs", "x"],
    )

    # Already wrapped -> no second re-exec.
    assert call({"_F_EXEC_WRAPPER": "nsys launch", "_F_EXEC_WRAPPED": "1"}) is None

    # Clustered/torchrun worker -> left unwrapped by default.
    assert call({"_F_EXEC_WRAPPER": "nsys launch", "TORCHELASTIC_RUN_ID": "run-xyz"}) is None

    # Clustered opt-in (_F_EXEC_WRAPPER_CLUSTERED): only the primary worker is wrapped. RANK 0 (or
    # unset, which defaults to primary) re-execs; any other rank is left unwrapped.
    w = {"_F_EXEC_WRAPPER": "nsys launch", "TORCHELASTIC_RUN_ID": "run-xyz", "_F_EXEC_WRAPPER_CLUSTERED": "1"}
    assert call(w) is not None
    assert call({**w, "RANK": "0"}) is not None
    assert call({**w, "RANK": "3"}) is None


def test_runtime_task_coroutine_exception():
    """Test that task_coroutine exceptions are properly handled and uploaded to outputs_path"""

    runner = CliRunner()

    with tempfile.TemporaryDirectory() as temp_dir:
        outputs_path = f"{temp_dir}/outputs"
        inputs_path = f"{temp_dir}/inputs"
        run_base_dir = f"{temp_dir}/run_base"

        # Create the directories
        os.makedirs(outputs_path, exist_ok=True)
        os.makedirs(inputs_path, exist_ok=True)
        os.makedirs(run_base_dir, exist_ok=True)

        # Set required environment variables
        env_vars = {
            "ACTION_NAME": "test_action",
            "RUN_NAME": "test_run",
            "FLYTE_INTERNAL_EXECUTION_PROJECT": "test_project",
            "FLYTE_INTERNAL_EXECUTION_DOMAIN": "test_domain",
            "_U_ORG_NAME": "test_org",
        }

        # Mock the task coroutine to raise RuntimeSystemError
        task_error = flyte.errors.RuntimeSystemError("TASK_FAILED", "Task execution failed")

        # Mock controller that never fails in watch_for_errors
        mock_controller = AsyncMock()
        mock_controller.watch_for_errors = AsyncMock()
        mock_controller.watch_for_errors.__await__ = lambda: (x for x in ())  # Never completes
        mock_controller.stop = AsyncMock()

        # Mock init_in_cluster to return controller kwargs
        def mock_init_in_cluster(*args, **kwargs):
            return {"endpoint": "test-endpoint", "insecure": True}

        with (
            patch("flyte._initialize.init_in_cluster", side_effect=mock_init_in_cluster),
            patch("flyte._internal.controllers.create_controller", return_value=mock_controller),
            patch(
                "flyte._internal.runtime.entrypoints.load_and_run_task", new_callable=AsyncMock, side_effect=task_error
            ),
            patch("faulthandler.register"),  # Mock faulthandler to avoid fileno issues in tests
            patch.dict(os.environ, env_vars, clear=False),
        ):
            # Run the CLI command
            result = runner.invoke(
                main,
                [
                    "--inputs",
                    inputs_path,
                    "--outputs-path",
                    outputs_path,
                    "--version",
                    "test_version",
                    "--run-base-dir",
                    run_base_dir,
                ],
            )

            # Verify the command failed (non-zero exit code)
            assert result.exit_code == 0

            # Verify that error files were created in outputs_path
            outputs_dir = Path(outputs_path)
            error_files = list(outputs_dir.glob("*.pb"))
            assert len(error_files) > 0, f"Expected error files in {outputs_path}, but found none"

            # Load the error file as protobuf and check contents
            error_file_path = error_files[0]
            with open(error_file_path, "rb") as f:
                error_document = execution_pb2.ErrorDocument()
                error_document.ParseFromString(f.read())

            # Verify error contents match our expected RuntimeSystemError
            assert error_document.error.code == "TASK_FAILED"
            assert error_document.error.message == "Task execution failed"
            assert error_document.error.kind == execution_pb2.ContainerError.RECOVERABLE


def test_runtime_controller_failure_exception():
    """Test that controller_failure exceptions are properly handled and uploaded to outputs_path"""

    runner = CliRunner()

    with tempfile.TemporaryDirectory() as temp_dir:
        outputs_path = f"{temp_dir}/outputs"
        inputs_path = f"{temp_dir}/inputs"
        run_base_dir = f"{temp_dir}/run_base"

        # Create the directories
        os.makedirs(outputs_path, exist_ok=True)
        os.makedirs(inputs_path, exist_ok=True)
        os.makedirs(run_base_dir, exist_ok=True)

        # Set required environment variables
        env_vars = {
            "ACTION_NAME": "test_action",
            "RUN_NAME": "test_run",
            "FLYTE_INTERNAL_EXECUTION_PROJECT": "test_project",
            "FLYTE_INTERNAL_EXECUTION_DOMAIN": "test_domain",
            "_U_ORG_NAME": "test_org",
        }

        # Mock controller that fails in watch_for_errors
        controller_error = flyte.errors.RuntimeSystemError("CONTROLLER_FAILED", "Controller failure detected")
        mock_controller = AsyncMock()
        mock_controller.watch_for_errors = AsyncMock(side_effect=controller_error)
        mock_controller.stop = AsyncMock()

        # Mock init_in_cluster to return controller kwargs
        def mock_init_in_cluster(*args, **kwargs):
            return {"endpoint": "test-endpoint", "insecure": True}

        async def _never_completes(*args, **kwargs):
            await asyncio.sleep(3600)

        with (
            patch("flyte._initialize.init_in_cluster", side_effect=mock_init_in_cluster),
            patch("flyte._internal.controllers.create_controller", return_value=mock_controller),
            patch("flyte._internal.runtime.entrypoints.load_and_run_task", return_value=_never_completes()),
            patch("faulthandler.register"),  # Mock faulthandler to avoid fileno issues in tests
            patch.dict(os.environ, env_vars, clear=False),
        ):
            # Run the CLI command
            result = runner.invoke(
                main,
                [
                    "--inputs",
                    inputs_path,
                    "--outputs-path",
                    outputs_path,
                    "--version",
                    "test_version",
                    "--run-base-dir",
                    run_base_dir,
                ],
            )

            # Verify the command failed (non-zero exit code)
            assert result.exit_code == 0

            # Verify that error files were created in outputs_path
            outputs_dir = Path(outputs_path)
            error_files = list(outputs_dir.glob("*.pb"))
            assert len(error_files) > 0, f"Expected error files in {outputs_path}, but found none"

            # Load the error file as protobuf and check contents
            error_file_path = error_files[0]
            with open(error_file_path, "rb") as f:
                error_document = execution_pb2.ErrorDocument()
                error_document.ParseFromString(f.read())

            # Verify error contents match our expected RuntimeSystemError
            assert error_document.error.code == "CONTROLLER_FAILED"
            assert error_document.error.message == "Controller failure detected"
            assert error_document.error.kind == execution_pb2.ContainerError.RECOVERABLE


def test_non_recoverable_error_sets_kind():
    """Test that NonRecoverableError uploads an error with ContainerError.NON_RECOVERABLE kind."""

    runner = CliRunner()

    with tempfile.TemporaryDirectory() as temp_dir:
        outputs_path = f"{temp_dir}/outputs"
        inputs_path = f"{temp_dir}/inputs"
        run_base_dir = f"{temp_dir}/run_base"

        os.makedirs(outputs_path, exist_ok=True)
        os.makedirs(inputs_path, exist_ok=True)
        os.makedirs(run_base_dir, exist_ok=True)

        env_vars = {
            "ACTION_NAME": "test_action",
            "RUN_NAME": "test_run",
            "FLYTE_INTERNAL_EXECUTION_PROJECT": "test_project",
            "FLYTE_INTERNAL_EXECUTION_DOMAIN": "test_domain",
            "_U_ORG_NAME": "test_org",
        }

        task_error = flyte.errors.NonRecoverableError("Input is invalid and will never succeed.")

        mock_controller = AsyncMock()
        mock_controller.watch_for_errors = AsyncMock()
        mock_controller.watch_for_errors.__await__ = lambda: (x for x in ())
        mock_controller.stop = AsyncMock()

        def mock_init_in_cluster(*args, **kwargs):
            return {"endpoint": "test-endpoint", "insecure": True}

        with (
            patch("flyte._initialize.init_in_cluster", side_effect=mock_init_in_cluster),
            patch("flyte._internal.controllers.create_controller", return_value=mock_controller),
            patch(
                "flyte._internal.runtime.entrypoints.load_and_run_task",
                new_callable=AsyncMock,
                side_effect=task_error,
            ),
            patch("faulthandler.register"),
            patch.dict(os.environ, env_vars, clear=False),
        ):
            result = runner.invoke(
                main,
                [
                    "--inputs",
                    inputs_path,
                    "--outputs-path",
                    outputs_path,
                    "--version",
                    "test_version",
                    "--run-base-dir",
                    run_base_dir,
                ],
            )

            assert result.exit_code == 0

            outputs_dir = Path(outputs_path)
            error_files = list(outputs_dir.glob("*.pb"))
            assert len(error_files) > 0, f"Expected error files in {outputs_path}, but found none"

            error_file_path = error_files[0]
            with open(error_file_path, "rb") as f:
                error_document = execution_pb2.ErrorDocument()
                error_document.ParseFromString(f.read())

            assert error_document.error.code == "NonRecoverableError"
            assert error_document.error.message == "Input is invalid and will never succeed."
            assert error_document.error.kind == execution_pb2.ContainerError.NON_RECOVERABLE
