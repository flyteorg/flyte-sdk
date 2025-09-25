import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner
from flyteidl.core import errors_pb2

import flyte.errors
from flyte._bin.runtime import main


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

        with (
            patch("flyte._initialize.init"),
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
            assert result.exit_code != 0

            # Verify that error files were created in outputs_path
            outputs_dir = Path(outputs_path)
            error_files = list(outputs_dir.glob("*.pb"))
            assert len(error_files) > 0, f"Expected error files in {outputs_path}, but found none"

            # Load the error file as protobuf and check contents
            error_file_path = error_files[0]
            with open(error_file_path, "rb") as f:
                error_document = errors_pb2.ErrorDocument()
                error_document.ParseFromString(f.read())

            # Verify error contents match our expected RuntimeSystemError
            assert error_document.error.code == "TASK_FAILED"
            assert error_document.error.message == "Task execution failed"
            assert error_document.error.kind == errors_pb2.ContainerError.RECOVERABLE


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

        # Mock task coroutine that never completes
        mock_task_coroutine = AsyncMock()
        mock_task_coroutine.__await__ = lambda: (x for x in ())  # Never completes

        with (
            patch("flyte._initialize.init"),
            patch("flyte._internal.controllers.create_controller", return_value=mock_controller),
            patch("flyte._internal.runtime.entrypoints.load_and_run_task", return_value=mock_task_coroutine),
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
            assert result.exit_code != 0

            # Verify that error files were created in outputs_path
            outputs_dir = Path(outputs_path)
            error_files = list(outputs_dir.glob("*.pb"))
            assert len(error_files) > 0, f"Expected error files in {outputs_path}, but found none"

            # Load the error file as protobuf and check contents
            error_file_path = error_files[0]
            with open(error_file_path, "rb") as f:
                error_document = errors_pb2.ErrorDocument()
                error_document.ParseFromString(f.read())

            # Verify error contents match our expected RuntimeSystemError
            assert error_document.error.code == "CONTROLLER_FAILED"
            assert error_document.error.message == "Controller failure detected"
            assert error_document.error.kind == errors_pb2.ContainerError.RECOVERABLE
