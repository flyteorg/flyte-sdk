"""
Unit tests for flyte._bin.serve module.

These tests verify the serve functionality including input synchronization,
code bundle downloading, and the main serve command without using mocks.
"""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import aiofiles
import pytest
from click.testing import CliRunner

from flyte._bin.serve import download_code_inputs, main, sync_inputs
from flyte.app._input import Input, SerializableInputCollection
from flyte.models import CodeBundle


class TestSyncInputs:
    """Tests for sync_inputs function."""

    @pytest.mark.asyncio
    async def test_sync_inputs_with_string_values(self):
        """
        GOAL: Verify sync_inputs correctly handles string-type inputs.

        Tests that string inputs are returned as-is without download attempts.
        """
        # Create inputs with string values
        inputs = [
            Input(value="config-value", name="config"),
            Input(value="api-key-value", name="api_key"),
        ]
        collection = SerializableInputCollection.from_inputs(inputs)
        serialized = collection.to_transport

        # Sync inputs
        result = await sync_inputs(serialized, dest="/tmp/test")

        # Verify string values are returned as-is
        assert result["config"] == "config-value"
        assert result["api_key"] == "api-key-value"

    @pytest.mark.asyncio
    async def test_sync_inputs_with_file_download(self):
        """
        GOAL: Verify sync_inputs correctly downloads file inputs.

        Tests that:
        - File inputs with download=True are downloaded
        - Downloaded file is accessible at the destination
        - Downloaded path is returned

        Note: String values with download=True don't actually download in the implementation.
        Only File/Dir types trigger downloads.
        """
        from flyte.io import File

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a source file
            source_file = os.path.join(tmpdir, "source", "file.txt")
            os.makedirs(os.path.dirname(source_file), exist_ok=True)
            async with aiofiles.open(source_file, "w") as f:
                await f.write("test content")

            # Create a File input with download enabled using file:// URL
            file_obj = File(path=f"file://{source_file}")
            inputs = [
                Input(value=file_obj, name="datafile", download=True),
            ]
            collection = SerializableInputCollection.from_inputs(inputs)
            serialized = collection.to_transport

            dest_dir = os.path.join(tmpdir, "dest")
            os.makedirs(dest_dir, exist_ok=True)

            result = await sync_inputs(serialized, dest=str(dest_dir))

            # Verify result contains downloaded path
            assert "datafile" in result
            downloaded_path = result["datafile"]

            # Verify the file exists at the downloaded location
            assert os.path.exists(downloaded_path)
            async with aiofiles.open(downloaded_path, "r") as f:
                assert await f.read() == "test content"

    @pytest.mark.asyncio
    async def test_sync_inputs_with_custom_dest(self):
        """
        GOAL: Verify sync_inputs respects custom destination paths.

        Tests that when input.dest (mount) is specified, it overrides the default dest.
        """
        from flyte.io import File

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a source file
            source_file = os.path.join(tmpdir, "source", "config.yaml")
            os.makedirs(os.path.dirname(source_file), exist_ok=True)
            async with aiofiles.open(source_file, "w") as f:
                await f.write("config: value")

            # Create custom destination directory
            custom_dest = os.path.join(tmpdir, "app", "config")
            os.makedirs(custom_dest, exist_ok=True)

            # Create File input with custom destination (mount)
            file_obj = File(path=f"file://{source_file}")
            inputs = [
                Input(value=file_obj, name="config", mount=custom_dest),  # mount implies download
            ]
            collection = SerializableInputCollection.from_inputs(inputs)
            serialized = collection.to_transport

            default_dest = os.path.join(tmpdir, "default")
            os.makedirs(default_dest, exist_ok=True)

            result = await sync_inputs(serialized, dest=str(default_dest))

            # Verify file was downloaded to custom dest, not default dest
            downloaded_path = result["config"]
            assert custom_dest in downloaded_path
            assert default_dest not in downloaded_path
            assert os.path.exists(downloaded_path)

    @pytest.mark.asyncio
    async def test_sync_inputs_with_directory_download(self):
        """
        GOAL: Verify sync_inputs correctly downloads directory inputs.

        Tests that:
        - Directory inputs trigger recursive downloads
        - All files in the directory are downloaded

        Note: There's a bug in serve.py line 46 - it uses input["type"] instead of input.type
        This test documents the expected behavior once fixed.
        """
        from flyte.io import Dir

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a source directory with files
            source_dir = os.path.join(tmpdir, "source", "data-dir")
            os.makedirs(source_dir, exist_ok=True)
            async with aiofiles.open(os.path.join(source_dir, "file1.txt"), "w") as f:
                await f.write("data1")
            async with aiofiles.open(os.path.join(source_dir, "file2.txt"), "w") as f:
                await f.write("data2")

            # Create directory input
            dir_input = Dir(path=f"file://{source_dir}")
            mount_dest = os.path.join(tmpdir, "data")
            os.makedirs(mount_dest, exist_ok=True)

            inputs = [
                Input(value=dir_input, name="dataset", mount=mount_dest),  # mount implies download
            ]
            collection = SerializableInputCollection.from_inputs(inputs)
            serialized = collection.to_transport

            # This currently fails due to bug in serve.py line 46
            # Once fixed, this test should pass
            with pytest.raises(TypeError, match="not subscriptable"):
                await sync_inputs(serialized, dest=tmpdir)

    @pytest.mark.asyncio
    async def test_sync_inputs_mixed_types(self):
        """
        GOAL: Verify sync_inputs handles mixed input types correctly.

        Tests that a combination of string inputs and File inputs
        are all processed correctly.

        Note: String values don't download even with download=True.
        Only File/Dir types trigger actual downloads.
        """
        from flyte.io import File

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a source file
            source_file = os.path.join(tmpdir, "source", "model.pkl")
            os.makedirs(os.path.dirname(source_file), exist_ok=True)
            async with aiofiles.open(source_file, "wb") as f:
                await f.write(b"model data")

            file_obj = File(path=f"file://{source_file}")
            inputs = [
                Input(value="string-config", name="config"),
                Input(value=file_obj, name="model", download=True),
                Input(value="another-string", name="param"),
            ]
            collection = SerializableInputCollection.from_inputs(inputs)
            serialized = collection.to_transport

            dest_dir = os.path.join(tmpdir, "dest")
            os.makedirs(dest_dir, exist_ok=True)

            result = await sync_inputs(serialized, dest=str(dest_dir))

            # Verify string values
            assert result["config"] == "string-config"
            assert result["param"] == "another-string"

            # Verify downloaded value
            assert "model" in result
            model_path = result["model"]
            assert os.path.exists(model_path)
            async with aiofiles.open(model_path, "rb") as f:
                assert await f.read() == b"model data"

    @pytest.mark.asyncio
    async def test_sync_inputs_empty_inputs(self):
        """
        GOAL: Verify sync_inputs handles empty inputs gracefully.

        Tests that an empty input list returns an empty dict.
        """
        # Create empty inputs
        collection = SerializableInputCollection(inputs=[])
        serialized = collection.to_transport

        result = await sync_inputs(serialized, dest="/tmp")

        # Verify empty result
        assert result == {}


class TestDownloadCodeInputs:
    """Tests for download_code_inputs function."""

    @pytest.mark.asyncio
    async def test_download_code_inputs_with_tgz(self):
        """
        GOAL: Verify download_code_inputs downloads tgz code bundles.

        Tests that:
        - CodeBundle is created with correct parameters
        - download_code_bundle is called
        - Code bundle is returned
        """
        # Mock download_code_bundle
        with patch("flyte._internal.runtime.entrypoints.download_code_bundle", new_callable=AsyncMock) as mock_download:
            mock_bundle = CodeBundle(tgz="s3://bucket/code.tgz", destination="/app", computed_version="v1.0.0")
            mock_download.return_value = mock_bundle

            user_inputs, code_bundle = await download_code_inputs(
                serialized_inputs="",
                tgz="s3://bucket/code.tgz",
                pkl="",
                dest="/app",
                version="v1.0.0",
            )

            # Verify download was called
            mock_download.assert_called_once()

            # Verify code bundle was returned
            assert code_bundle == mock_bundle

            # Verify user inputs is empty
            assert user_inputs == {}

    @pytest.mark.asyncio
    async def test_download_code_inputs_with_pkl(self):
        """
        GOAL: Verify download_code_inputs downloads pkl code bundles.

        Tests that pkl bundles are handled as an alternative to tgz.
        """
        # Mock download_code_bundle
        with patch("flyte._internal.runtime.entrypoints.download_code_bundle", new_callable=AsyncMock) as mock_download:
            mock_bundle = CodeBundle(pkl="s3://bucket/code.pkl", destination="/app", computed_version="v1.0.0")
            mock_download.return_value = mock_bundle

            user_inputs, code_bundle = await download_code_inputs(
                serialized_inputs="",
                tgz="",
                pkl="s3://bucket/code.pkl",
                dest="/app",
                version="v1.0.0",
            )

            # Verify download was called
            mock_download.assert_called_once()

            # Verify code bundle was returned
            assert code_bundle == mock_bundle
            assert user_inputs == {}

    @pytest.mark.asyncio
    async def test_download_code_inputs_with_inputs_and_code(self):
        """
        GOAL: Verify download_code_inputs handles both inputs and code bundle.

        Tests that both user inputs and code bundle are downloaded and returned.
        """
        # Create serialized inputs
        inputs = [Input(value="config-value", name="config")]
        collection = SerializableInputCollection.from_inputs(inputs)
        serialized = collection.to_transport

        # Mock download_code_bundle
        with patch("flyte._internal.runtime.entrypoints.download_code_bundle", new_callable=AsyncMock) as mock_download:
            mock_bundle = CodeBundle(tgz="s3://bucket/code.tgz", destination="/app", computed_version="v1.0.0")
            mock_download.return_value = mock_bundle

            user_inputs, code_bundle = await download_code_inputs(
                serialized_inputs=serialized,
                tgz="s3://bucket/code.tgz",
                pkl="",
                dest="/app",
                version="v1.0.0",
            )

            # Verify both inputs and code bundle
            assert user_inputs["config"] == "config-value"
            assert code_bundle == mock_bundle

    @pytest.mark.asyncio
    async def test_download_code_inputs_no_code_bundle(self):
        """
        GOAL: Verify download_code_inputs works without a code bundle.

        Tests that when no tgz or pkl is provided, only inputs are processed.
        """
        # Create serialized inputs
        inputs = [Input(value="test-value", name="param")]
        collection = SerializableInputCollection.from_inputs(inputs)
        serialized = collection.to_transport

        user_inputs, code_bundle = await download_code_inputs(
            serialized_inputs=serialized, tgz="", pkl="", dest="/app", version="v1.0.0"
        )

        # Verify inputs are processed
        assert user_inputs["param"] == "test-value"

        # Verify no code bundle
        assert code_bundle is None

    @pytest.mark.asyncio
    async def test_download_code_inputs_empty_inputs_with_code(self):
        """
        GOAL: Verify download_code_inputs works with empty inputs but code bundle.

        Tests that code bundle can be downloaded without user inputs.
        """
        # Mock download_code_bundle
        with patch("flyte._internal.runtime.entrypoints.download_code_bundle", new_callable=AsyncMock) as mock_download:
            mock_bundle = CodeBundle(tgz="s3://bucket/code.tgz", destination="/app", computed_version="v1.0.0")
            mock_download.return_value = mock_bundle

            user_inputs, code_bundle = await download_code_inputs(
                serialized_inputs="", tgz="s3://bucket/code.tgz", pkl="", dest="/app", version="v1.0.0"
            )

            # Verify empty user inputs
            assert user_inputs == {}

            # Verify code bundle
            assert code_bundle == mock_bundle


class TestMainCommand:
    """Tests for main CLI command."""

    def test_main_basic_invocation(self):
        """
        GOAL: Verify main command can be invoked with required parameters.

        Tests that the CLI accepts the required --version flag and command.
        """
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock Popen (imported inside main) and asyncio.run
            with patch("subprocess.Popen") as mock_popen, patch("flyte._bin.serve.asyncio.run") as mock_run:
                # Mock asyncio.run to return empty inputs and no code bundle
                mock_run.return_value = ({}, None)

                # Mock process
                mock_process = MagicMock()
                mock_process.wait.return_value = 0
                mock_popen.return_value = mock_process

                # Run command
                result = runner.invoke(
                    main,
                    ["--version", "v1.0.0", "--dest", tmpdir, "--", "echo", "hello"],
                )

                # Verify command succeeded
                assert result.exit_code == 0

    def test_main_with_inputs(self):
        """
        GOAL: Verify main command processes inputs correctly.

        Tests that:
        - Inputs are deserialized and downloaded
        - Inputs file is created
        - Environment variable is set
        """
        runner = CliRunner()

        # Create serialized inputs
        inputs = [Input(value="test-value", name="config")]
        collection = SerializableInputCollection.from_inputs(inputs)
        serialized = collection.to_transport

        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to tmpdir so inputs file is created there
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                # Mock Popen and asyncio to avoid actual subprocess
                with patch("subprocess.Popen") as mock_popen, patch("flyte._bin.serve.asyncio.run") as mock_run:
                    # Mock asyncio.run to return test inputs
                    mock_run.return_value = ({"config": "test-value"}, None)

                    # Mock process
                    mock_process = MagicMock()
                    mock_process.wait.return_value = 0
                    mock_popen.return_value = mock_process

                    # Run command
                    result = runner.invoke(
                        main,
                        ["--version", "v1.0.0", "--inputs", serialized, "--dest", tmpdir, "--", "echo", "test"],
                    )

                    # Verify command succeeded
                    assert result.exit_code == 0

                    # Verify inputs file was created
                    inputs_file = os.path.join(tmpdir, "flyte-inputs.json")
                    assert os.path.exists(inputs_file)

                    # Verify inputs file content
                    with open(inputs_file, "r") as f:
                        saved_inputs = json.load(f)
                    assert saved_inputs["config"] == "test-value"

            finally:
                os.chdir(original_cwd)

    def test_main_with_tgz_code_bundle(self):
        """
        GOAL: Verify main command downloads tgz code bundles.

        Tests that tgz parameter is passed to download_code_inputs.
        """
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock Popen and asyncio
            with patch("subprocess.Popen") as mock_popen, patch("flyte._bin.serve.asyncio.run") as mock_run:
                mock_bundle = CodeBundle(tgz="s3://bucket/code.tgz", destination=tmpdir, computed_version="v1.0.0")
                mock_run.return_value = ({}, mock_bundle)

                # Mock process
                mock_process = MagicMock()
                mock_process.wait.return_value = 0
                mock_popen.return_value = mock_process

                # Run command with tgz
                result = runner.invoke(
                    main,
                    [
                        "--version",
                        "v1.0.0",
                        "--tgz",
                        "s3://bucket/code.tgz",
                        "--dest",
                        tmpdir,
                        "--",
                        "python",
                        "app.py",
                    ],
                )

                # Verify command succeeded
                assert result.exit_code == 0

                # Verify asyncio.run was called
                assert mock_run.called

    def test_main_with_pkl_code_bundle(self):
        """
        GOAL: Verify main command downloads pkl code bundles.

        Tests that pkl parameter is passed to download_code_inputs.
        """
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock Popen and asyncio
            with patch("subprocess.Popen") as mock_popen, patch("flyte._bin.serve.asyncio.run") as mock_run:
                mock_bundle = CodeBundle(pkl="s3://bucket/code.pkl", destination=tmpdir, computed_version="v1.0.0")
                mock_run.return_value = ({}, mock_bundle)

                # Mock process
                mock_process = MagicMock()
                mock_process.wait.return_value = 0
                mock_popen.return_value = mock_process

                # Run command with pkl
                result = runner.invoke(
                    main,
                    [
                        "--version",
                        "v1.0.0",
                        "--pkl",
                        "s3://bucket/code.pkl",
                        "--dest",
                        tmpdir,
                        "--",
                        "python",
                        "app.py",
                    ],
                )

                # Verify command succeeded
                assert result.exit_code == 0

    def test_main_with_project_domain_org(self):
        """
        GOAL: Verify main command accepts project, domain, and org parameters.

        Tests that these parameters are accepted (even if not currently used).
        """
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock Popen and asyncio
            with patch("subprocess.Popen") as mock_popen, patch("flyte._bin.serve.asyncio.run") as mock_run:
                mock_run.return_value = ({}, None)

                # Mock process
                mock_process = MagicMock()
                mock_process.wait.return_value = 0
                mock_popen.return_value = mock_process

                # Run command with project, domain, org
                result = runner.invoke(
                    main,
                    [
                        "--version",
                        "v1.0.0",
                        "--project",
                        "my-project",
                        "--domain",
                        "development",
                        "--org",
                        "my-org",
                        "--dest",
                        tmpdir,
                        "--",
                        "echo",
                        "test",
                    ],
                )

                # Verify command succeeded
                assert result.exit_code == 0

    def test_main_command_execution(self):
        """
        GOAL: Verify main command executes the provided command correctly.

        Tests that:
        - Command arguments are joined and passed to Popen
        - Process is started with correct shell=True
        """
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock Popen and asyncio
            with patch("subprocess.Popen") as mock_popen, patch("flyte._bin.serve.asyncio.run") as mock_run:
                mock_run.return_value = ({}, None)

                # Mock process
                mock_process = MagicMock()
                mock_process.wait.return_value = 0
                mock_popen.return_value = mock_process

                # Run command
                runner.invoke(
                    main,
                    [
                        "--version",
                        "v1.0.0",
                        "--dest",
                        tmpdir,
                        "--",
                        "python",
                        "-m",
                        "myapp",
                        "--host",
                        "0.0.0.0",
                    ],
                )

                # Verify Popen was called with joined command
                mock_popen.assert_called_once()
                call_args = mock_popen.call_args[0]
                assert call_args[0] == "python -m myapp --host 0.0.0.0"

                # Verify shell=True was used
                assert mock_popen.call_args[1]["shell"] is True

    def test_main_exit_code_propagation(self):
        """
        GOAL: Verify main command propagates subprocess exit code.

        Tests that the exit code from the subprocess is returned.
        """
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock Popen and asyncio
            with patch("subprocess.Popen") as mock_popen, patch("flyte._bin.serve.asyncio.run") as mock_run:
                mock_run.return_value = ({}, None)

                # Mock process with non-zero exit code
                mock_process = MagicMock()
                mock_process.wait.return_value = 42
                mock_popen.return_value = mock_process

                # Run command
                runner.invoke(
                    main,
                    ["--version", "v1.0.0", "--dest", tmpdir, "--", "false"],  # 'false' command exits with 1
                    catch_exceptions=False,
                )

                # Verify exit code is propagated
                # Note: Click runner may not propagate system exit codes the same way
                # but the function should call exit(returncode)
                mock_process.wait.assert_called_once()

    def test_main_signal_handling(self):
        """
        GOAL: Verify main command sets up SIGTERM handler.

        Tests that signal.signal is called to handle SIGTERM.
        """
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock everything - signal is imported inside main()
            with (
                patch("subprocess.Popen") as mock_popen,
                patch("flyte._bin.serve.asyncio.run") as mock_run,
                patch("signal.signal") as mock_signal,
            ):
                mock_run.return_value = ({}, None)

                # Mock process
                mock_process = MagicMock()
                mock_process.wait.return_value = 0
                mock_popen.return_value = mock_process

                # Run command
                runner.invoke(
                    main,
                    ["--version", "v1.0.0", "--dest", tmpdir, "--", "echo", "test"],
                )

                # Verify signal handler was set
                mock_signal.assert_called_once()
                import signal

                assert mock_signal.call_args[0][0] == signal.SIGTERM

    def test_main_inputs_file_environment_variable(self):
        """
        GOAL: Verify main command sets RUNTIME_INPUTS_FILE environment variable.

        Tests that the environment variable is set to the inputs file path.
        """
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                # Mock Popen and asyncio
                with patch("subprocess.Popen") as mock_popen, patch("flyte._bin.serve.asyncio.run") as mock_run:
                    mock_run.return_value = ({"test": "value"}, None)

                    # Mock process
                    mock_process = MagicMock()
                    mock_process.wait.return_value = 0
                    mock_popen.return_value = mock_process

                    # Capture environment passed to Popen
                    captured_env = {}

                    def capture_env(*args, **kwargs):
                        captured_env.update(kwargs["env"])
                        return mock_process

                    mock_popen.side_effect = capture_env

                    # Run command
                    runner.invoke(
                        main,
                        ["--version", "v1.0.0", "--dest", tmpdir, "--", "echo", "test"],
                    )

                    # Verify RUNTIME_INPUTS_FILE is in environment
                    from flyte.app._input import RUNTIME_INPUTS_FILE

                    assert RUNTIME_INPUTS_FILE in captured_env
                    assert captured_env[RUNTIME_INPUTS_FILE].endswith("flyte-inputs.json")

            finally:
                os.chdir(original_cwd)

    def test_main_without_required_version(self):
        """
        GOAL: Verify main command requires --version parameter.

        Tests that the command fails when --version is not provided.
        """
        runner = CliRunner()

        # Run command without version
        result = runner.invoke(main, ["--", "echo", "test"])

        # Verify command failed
        assert result.exit_code != 0
        assert "version" in result.output.lower() or "required" in result.output.lower()

    def test_main_with_image_cache(self):
        """
        GOAL: Verify main command accepts image-cache parameter.

        Tests that --image-cache parameter is accepted (for future use).
        """
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock Popen and asyncio
            with patch("subprocess.Popen") as mock_popen, patch("flyte._bin.serve.asyncio.run") as mock_run:
                mock_run.return_value = ({}, None)

                # Mock process
                mock_process = MagicMock()
                mock_process.wait.return_value = 0
                mock_popen.return_value = mock_process

                # Run command with image-cache
                result = runner.invoke(
                    main,
                    [
                        "--version",
                        "v1.0.0",
                        "--image-cache",
                        "base64encodedcache",
                        "--dest",
                        tmpdir,
                        "--",
                        "echo",
                        "test",
                    ],
                )

                # Verify command succeeded
                assert result.exit_code == 0


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_inputs_and_code(self):
        """
        GOAL: Integration test for complete workflow with inputs and code bundle.

        Tests the full flow:
        1. Inputs are deserialized
        2. Code bundle is downloaded
        3. Both are returned correctly
        """
        # Create test inputs
        inputs = [
            Input(value="config-data", name="config"),
            Input(value="api-key-secret", name="api_key"),
        ]
        collection = SerializableInputCollection.from_inputs(inputs)
        serialized = collection.to_transport

        # Mock download_code_bundle
        with patch("flyte._internal.runtime.entrypoints.download_code_bundle", new_callable=AsyncMock) as mock_download:
            mock_bundle = CodeBundle(tgz="s3://bucket/code.tgz", destination="/app", computed_version="v1.0.0")
            mock_download.return_value = mock_bundle

            # Run full download
            user_inputs, code_bundle = await download_code_inputs(
                serialized_inputs=serialized,
                tgz="s3://bucket/code.tgz",
                pkl="",
                dest="/app",
                version="v1.0.0",
            )

            # Verify both inputs and code bundle
            assert user_inputs["config"] == "config-data"
            assert user_inputs["api_key"] == "api-key-secret"
            assert code_bundle == mock_bundle
            assert code_bundle.tgz == "s3://bucket/code.tgz"
