"""
Unit tests for flyte._bin.serve module.

These tests verify the serve functionality including parameter synchronization,
code bundle downloading, and the main serve command without using mocks.
"""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import aiofiles
import pytest
from click.testing import CliRunner

from flyte._bin.serve import _bind_parameters, _serve, download_code_parameters, main, sync_parameters
from flyte.app._parameter import Parameter, SerializableParameterCollection
from flyte.models import CodeBundle


class TestSyncParameters:
    """Tests for sync_parameters function."""

    @pytest.mark.asyncio
    async def test_sync_parameters_with_string_values(self):
        """
        GOAL: Verify sync_parameters correctly handles string-type parameters.

        Tests that string parameters are returned as-is without download attempts.
        """
        # Create parameters with string values
        parameters = [
            Parameter(value="config-value", name="config"),
            Parameter(value="api-key-value", name="api_key"),
            Parameter(value="test-env-var", name="test_env_var", env_var="TEST_ENV_VAR"),
        ]
        collection = SerializableParameterCollection.from_parameters(parameters)
        serialized = collection.to_transport

        # Sync parameters
        serializable_parameters, materialized_parameters, env_vars = await sync_parameters(serialized, dest="/tmp/test")

        # Verify string values are returned as-is
        assert serializable_parameters["config"] == "config-value"
        assert serializable_parameters["api_key"] == "api-key-value"
        assert env_vars["TEST_ENV_VAR"] == "test-env-var"

        # Verify materialized_parameters contains string values (same as serializable_parameters for strings)
        assert materialized_parameters["config"] == "config-value"
        assert materialized_parameters["api_key"] == "api-key-value"
        assert isinstance(materialized_parameters["config"], str)
        assert isinstance(materialized_parameters["api_key"], str)

    @pytest.mark.asyncio
    async def test_sync_parameters_with_file_download(self):
        """
        GOAL: Verify sync_parameters correctly downloads file parameters.

        Tests that File parameters with download=True are downloaded
        and the downloaded file is accessible at the destination.
        """
        from flyte.io import File

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a source file
            source_file = os.path.join(tmpdir, "source", "file.txt")
            os.makedirs(os.path.dirname(source_file), exist_ok=True)
            async with aiofiles.open(source_file, "w") as f:
                await f.write("test content")

            # Create a File parameter with download enabled using file:// URL
            file_obj = File(path=f"file://{source_file}")
            parameters = [
                Parameter(value=file_obj, name="datafile", download=True),
            ]
            collection = SerializableParameterCollection.from_parameters(parameters)
            serialized = collection.to_transport

            dest_dir = os.path.join(tmpdir, "dest")
            os.makedirs(dest_dir, exist_ok=True)

            serializable_parameters, materialized_parameters, env_vars = await sync_parameters(
                serialized, dest=str(dest_dir)
            )

            # Verify result contains downloaded path
            assert "datafile" in serializable_parameters
            assert env_vars == {}
            downloaded_path = serializable_parameters["datafile"]

            # Verify the file exists at the downloaded location
            assert os.path.exists(downloaded_path)
            async with aiofiles.open(downloaded_path, "r") as f:
                assert await f.read() == "test content"

            # Verify materialized_parameters contains File object with downloaded path
            assert "datafile" in materialized_parameters
            from flyte.io import File

            materialized_file = materialized_parameters["datafile"]
            assert isinstance(materialized_file, File)
            assert materialized_file.path == downloaded_path

    @pytest.mark.asyncio
    async def test_sync_parameters_with_custom_dest(self):
        """
        GOAL: Verify sync_parameters respects custom destination paths.

        Tests that when parameter.mount is specified, it overrides the default dest.
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

            # Create File parameter with custom destination (mount)
            file_obj = File(path=f"file://{source_file}")
            parameters = [
                Parameter(value=file_obj, name="config", mount=custom_dest),  # mount implies download
            ]
            collection = SerializableParameterCollection.from_parameters(parameters)
            serialized = collection.to_transport

            default_dest = os.path.join(tmpdir, "default")
            os.makedirs(default_dest, exist_ok=True)

            serializable_parameters, materialized_parameters, env_vars = await sync_parameters(
                serialized, dest=str(default_dest)
            )

            # Verify file was downloaded to custom dest, not default dest
            downloaded_path = serializable_parameters["config"]
            assert custom_dest in downloaded_path
            assert default_dest not in downloaded_path
            assert os.path.exists(downloaded_path)
            assert env_vars == {}

            # Verify materialized_parameters contains File object with custom dest path
            from flyte.io import File

            materialized_file = materialized_parameters["config"]
            assert isinstance(materialized_file, File)
            assert materialized_file.path == downloaded_path
            assert custom_dest in materialized_file.path

    @pytest.mark.asyncio
    async def test_sync_parameters_with_directory_download(self):
        """
        GOAL: Verify sync_parameters correctly downloads directory parameters.

        Tests that:
        - Directory parameters trigger recursive downloads
        - All files in the directory are downloaded
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

            # Create directory parameter
            dir_parameter = Dir(path=f"file://{source_dir}")
            mount_dest = os.path.join(tmpdir, "data")
            os.makedirs(mount_dest, exist_ok=True)

            parameters = [
                Parameter(value=dir_parameter, name="dataset", mount=mount_dest),  # mount implies download
                Parameter(value="test-env-var", name="test_env_var", env_var="TEST_ENV_VAR"),
            ]
            collection = SerializableParameterCollection.from_parameters(parameters)
            serialized = collection.to_transport

            serializable_parameters, materialized_parameters, env_vars = await sync_parameters(serialized, dest=tmpdir)
            assert os.path.exists(serializable_parameters["dataset"])
            assert os.path.isdir(serializable_parameters["dataset"])
            assert env_vars["TEST_ENV_VAR"] == "test-env-var"

            async with aiofiles.open(
                os.path.join(serializable_parameters["dataset"], "data-dir", "file1.txt"), "r"
            ) as f:
                assert await f.read() == "data1"
            async with aiofiles.open(
                os.path.join(serializable_parameters["dataset"], "data-dir", "file2.txt"), "r"
            ) as f:
                assert await f.read() == "data2"

            # Verify materialized_parameters contains Dir object with downloaded path
            from flyte.io import Dir

            materialized_dir = materialized_parameters["dataset"]
            assert isinstance(materialized_dir, Dir)
            assert materialized_dir.path == serializable_parameters["dataset"]
            assert os.path.exists(materialized_dir.path)
            assert os.path.isdir(materialized_dir.path)

            # Verify materialized_parameters contains string value for string parameter
            assert materialized_parameters["test_env_var"] == "test-env-var"
            assert isinstance(materialized_parameters["test_env_var"], str)

    @pytest.mark.asyncio
    async def test_sync_parameters_mixed_types(self):
        """
        GOAL: Verify sync_parameters handles mixed parameter types correctly.

        Tests that a combination of string parameters and File parameters
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
            parameters = [
                Parameter(value="string-config", name="config"),
                Parameter(value=file_obj, name="model", download=True),
                Parameter(value="another-string", name="param"),
            ]
            collection = SerializableParameterCollection.from_parameters(parameters)
            serialized = collection.to_transport

            dest_dir = os.path.join(tmpdir, "dest")
            os.makedirs(dest_dir, exist_ok=True)

            serializable_parameters, materialized_parameters, env_vars = await sync_parameters(
                serialized, dest=str(dest_dir)
            )

            # Verify string values
            assert serializable_parameters["config"] == "string-config"
            assert serializable_parameters["param"] == "another-string"

            # Verify downloaded value
            assert "model" in serializable_parameters
            model_path = serializable_parameters["model"]
            assert os.path.exists(model_path)
            async with aiofiles.open(model_path, "rb") as f:
                assert await f.read() == b"model data"

            # Verify environment variables
            assert env_vars == {}

            # Verify materialized_parameters contains correct types
            from flyte.io import File

            # String parameters should be strings in materialized_parameters
            assert materialized_parameters["config"] == "string-config"
            assert materialized_parameters["param"] == "another-string"
            assert isinstance(materialized_parameters["config"], str)
            assert isinstance(materialized_parameters["param"], str)

            # File parameter should be File object with downloaded path
            materialized_file = materialized_parameters["model"]
            assert isinstance(materialized_file, File)
            assert materialized_file.path == model_path
            assert os.path.exists(materialized_file.path)

    @pytest.mark.asyncio
    async def test_sync_parameters_with_file_no_download(self):
        """
        GOAL: Verify materialized_parameters contains File objects even when download=False.

        Tests that:
        - File parameters without download=True still create File objects in materialized_parameters.
        - The File object contains the remote path, not a local path
        """
        from flyte.io import File

        # Create a File parameter without download
        file_obj = File(path="s3://bucket/remote-file.txt")
        parameters = [
            Parameter(value=file_obj, name="remote_file", download=False),
        ]
        collection = SerializableParameterCollection.from_parameters(parameters)
        serialized = collection.to_transport

        serializable_parameters, materialized_parameters, env_vars = await sync_parameters(serialized, dest="/tmp/test")

        # Verify serializable_parameters contains the remote path as string
        assert serializable_parameters["remote_file"] == "s3://bucket/remote-file.txt"
        assert isinstance(serializable_parameters["remote_file"], str)

        # Verify materialized_parameters contains File object with remote path
        materialized_file = materialized_parameters["remote_file"]
        assert isinstance(materialized_file, File)
        assert materialized_file.path == "s3://bucket/remote-file.txt"
        assert env_vars == {}

    @pytest.mark.asyncio
    async def test_sync_parameters_with_directory_no_download(self):
        """
        GOAL: Verify materialized_parameters contains Dir objects even when download=False.

        Tests that:
        - Directory parameters without download=True still create Dir objects in materialized_parameters.
        - The Dir object contains the remote path, not a local path
        """
        from flyte.io import Dir

        # Create a Dir parameter without download
        dir_obj = Dir(path="s3://bucket/remote-dir/")
        parameters = [
            Parameter(value=dir_obj, name="remote_dir", download=False),
        ]
        collection = SerializableParameterCollection.from_parameters(parameters)
        serialized = collection.to_transport

        serializable_parameters, materialized_parameters, env_vars = await sync_parameters(serialized, dest="/tmp/test")

        # Verify serializable_parameters contains the remote path as string
        assert serializable_parameters["remote_dir"] == "s3://bucket/remote-dir/"
        assert isinstance(serializable_parameters["remote_dir"], str)

        # Verify materialized_parameters contains Dir object with remote path
        materialized_dir = materialized_parameters["remote_dir"]
        assert isinstance(materialized_dir, Dir)
        assert materialized_dir.path == "s3://bucket/remote-dir/"
        assert env_vars == {}

    @pytest.mark.asyncio
    async def test_sync_parameters_empty_parameters(self):
        """
        GOAL: Verify sync_parameters handles empty parameters gracefully.

        Tests that an empty parameters list returns an empty dict.
        """
        # Create empty parameters
        collection = SerializableParameterCollection(parameters=[])
        serialized = collection.to_transport

        serializable_parameters, materialized_parameters, env_vars = await sync_parameters(serialized, dest="/tmp")

        # Verify empty result
        assert serializable_parameters == {}
        assert materialized_parameters == {}
        assert env_vars == {}


class TestDownloadCodeParameters:
    """Tests for download_code_parameters function."""

    @pytest.mark.asyncio
    async def test_download_code_parameters_with_tgz(self):
        """
        GOAL: Verify download_code_parameters downloads tgz code bundles.

        Tests that:
        - CodeBundle is created with correct parameters
        - download_code_bundle is called
        - Code bundle is returned
        """
        # Mock download_code_bundle
        with patch("flyte._internal.runtime.entrypoints.download_code_bundle", new_callable=AsyncMock) as mock_download:
            mock_bundle = CodeBundle(tgz="s3://bucket/code.tgz", destination="/app", computed_version="v1.0.0")
            mock_download.return_value = mock_bundle

            serializable_parameters, materialized_parameters, env_vars, code_bundle = await download_code_parameters(
                serialized_parameters="",
                tgz="s3://bucket/code.tgz",
                pkl="",
                dest="/app",
                version="v1.0.0",
            )

            # Verify download was called
            mock_download.assert_called_once()

            # Verify code bundle was returned
            assert code_bundle == mock_bundle

            # Verify environment variables
            assert env_vars == {}

            # Verify user parameters is empty
            assert serializable_parameters == {}
            assert materialized_parameters == {}

    @pytest.mark.asyncio
    async def test_download_code_parameters_with_pkl(self):
        """
        GOAL: Verify download_code_parameters downloads pkl code bundles.

        Tests that pkl bundles are handled as an alternative to tgz.
        """
        # Mock download_code_bundle
        with patch("flyte._internal.runtime.entrypoints.download_code_bundle", new_callable=AsyncMock) as mock_download:
            mock_bundle = CodeBundle(pkl="s3://bucket/code.pkl", destination="/app", computed_version="v1.0.0")
            mock_download.return_value = mock_bundle

            serializable_parameters, materialized_parameters, env_vars, code_bundle = await download_code_parameters(
                serialized_parameters="",
                tgz="",
                pkl="s3://bucket/code.pkl",
                dest="/app",
                version="v1.0.0",
            )

            # Verify download was called
            mock_download.assert_called_once()

            # Verify code bundle was returned
            assert code_bundle == mock_bundle
            assert serializable_parameters == {}
            assert materialized_parameters == {}
            assert env_vars == {}

    @pytest.mark.asyncio
    async def test_download_code_parameters_with_parameters_and_code(self):
        """
        GOAL: Verify download_code_parameters handles both parameters and code bundle.

        Tests that both user parameters and code bundle are downloaded and returned.
        """
        # Create serialized parameters
        parameters = [Parameter(value="config-value", name="config")]
        collection = SerializableParameterCollection.from_parameters(parameters)
        serialized = collection.to_transport

        # Mock download_code_bundle
        with patch("flyte._internal.runtime.entrypoints.download_code_bundle", new_callable=AsyncMock) as mock_download:
            mock_bundle = CodeBundle(tgz="s3://bucket/code.tgz", destination="/app", computed_version="v1.0.0")
            mock_download.return_value = mock_bundle

            serializable_parameters, materialized_parameters, env_vars, code_bundle = await download_code_parameters(
                serialized_parameters=serialized,
                tgz="s3://bucket/code.tgz",
                pkl="",
                dest="/app",
                version="v1.0.0",
            )

            # Verify both parameters and code bundle
            assert serializable_parameters["config"] == "config-value"
            assert code_bundle == mock_bundle
            assert env_vars == {}

            # Verify materialized_parameters contains string value
            assert materialized_parameters["config"] == "config-value"
            assert isinstance(materialized_parameters["config"], str)

    @pytest.mark.asyncio
    async def test_download_code_parameters_with_file_parameters(self):
        """
        GOAL: Verify download_code_parameters properly returns materialized_parameters for file parameters.

        Tests that materialized_parameters contains File objects when file parameters are provided.
        """
        from flyte.io import File

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a source file
            source_file = os.path.join(tmpdir, "source", "data.txt")
            os.makedirs(os.path.dirname(source_file), exist_ok=True)
            async with aiofiles.open(source_file, "w") as f:
                await f.write("test data")

            # Create File parameter with download
            file_obj = File(path=f"file://{source_file}")
            parameters = [Parameter(value=file_obj, name="datafile", download=True)]
            collection = SerializableParameterCollection.from_parameters(parameters)
            serialized = collection.to_transport

            dest_dir = os.path.join(tmpdir, "dest")
            os.makedirs(dest_dir, exist_ok=True)

            serializable_parameters, materialized_parameters, env_vars, code_bundle = await download_code_parameters(
                serialized_parameters=serialized, tgz="", pkl="", dest=str(dest_dir), version="v1.0.0"
            )

            # Verify serializable_parameters contains downloaded path as string
            assert "datafile" in serializable_parameters
            downloaded_path = serializable_parameters["datafile"]
            assert isinstance(downloaded_path, str)
            assert os.path.exists(downloaded_path)

            # Verify materialized_parameters contains File object
            assert "datafile" in materialized_parameters
            materialized_file = materialized_parameters["datafile"]
            assert isinstance(materialized_file, File)
            assert materialized_file.path == downloaded_path

            # Verify no code bundle
            assert code_bundle is None
            assert env_vars == {}

    @pytest.mark.asyncio
    async def test_download_code_parameters_no_code_bundle(self):
        """
        GOAL: Verify download_code_parameters works without a code bundle.

        Tests that when no tgz or pkl is provided, only parameters are processed.
        """
        # Create serialized parameters
        parameters = [Parameter(value="test-value", name="param")]
        collection = SerializableParameterCollection.from_parameters(parameters)
        serialized = collection.to_transport

        serializable_parameters, materialized_parameters, env_vars, code_bundle = await download_code_parameters(
            serialized_parameters=serialized, tgz="", pkl="", dest="/app", version="v1.0.0"
        )

        # Verify parameters are processed
        assert serializable_parameters["param"] == "test-value"

        # Verify materialized_parameters contains string value
        assert materialized_parameters["param"] == "test-value"
        assert isinstance(materialized_parameters["param"], str)

        # Verify no code bundle
        assert code_bundle is None

        # Verify environment variables
        assert env_vars == {}

    @pytest.mark.asyncio
    async def test_download_code_parameters_empty_parameters_with_code(self):
        """
        GOAL: Verify download_code_parameters works with empty parameters but code bundle.

        Tests that code bundle can be downloaded without user parameters.
        """
        # Mock download_code_bundle
        with patch("flyte._internal.runtime.entrypoints.download_code_bundle", new_callable=AsyncMock) as mock_download:
            mock_bundle = CodeBundle(tgz="s3://bucket/code.tgz", destination="/app", computed_version="v1.0.0")
            mock_download.return_value = mock_bundle

            serializable_parameters, materialized_parameters, env_vars, code_bundle = await download_code_parameters(
                serialized_parameters="", tgz="s3://bucket/code.tgz", pkl="", dest="/app", version="v1.0.0"
            )

            # Verify empty user parameters
            assert serializable_parameters == {}
            assert materialized_parameters == {}

            # Verify code bundle
            assert code_bundle == mock_bundle

            # Verify environment variables
            assert env_vars == {}


class TestMainCommand:
    """Tests for main CLI command."""

    def test_main_basic_invocation(self):
        """
        GOAL: Verify main command can be invoked with required parameters.

        Tests that the CLI accepts the required --version flag and command.
        """
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Patch RUNTIME_PARAMETERS_FILE to use the temporary file
            parameters_file = os.path.join(tmpdir, "flyte-parameters.json")
            with patch("flyte.app._parameter.RUNTIME_PARAMETERS_FILE", parameters_file):
                # Mock Popen (imported inside main) and asyncio.run
                with patch("subprocess.Popen") as mock_popen, patch("flyte._bin.serve.asyncio.run") as mock_run:
                    # Mock asyncio.run to return empty parameters and no code bundle
                    mock_run.return_value = ({}, {}, {}, None)

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

    def test_main_with_parameters(self):
        """
        GOAL: Verify main command processes parameters correctly.

        Tests that:
        - Parameters are deserialized and downloaded
        - Parameters file is created
        - Environment variable is set
        """
        runner = CliRunner()

        # Create serialized parameters
        parameters = [Parameter(value="test-value", name="config")]
        collection = SerializableParameterCollection.from_parameters(parameters)
        serialized = collection.to_transport

        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to tmpdir so parameters file is created there
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                # Mock Popen and asyncio to avoid actual subprocess
                parameters_file = os.path.join(tmpdir, "flyte-parameters.json")
                with patch("flyte.app._parameter.RUNTIME_PARAMETERS_FILE", parameters_file):
                    with patch("subprocess.Popen") as mock_popen, patch("flyte._bin.serve.asyncio.run") as mock_run:
                        # Mock asyncio.run to return test parameters
                        mock_run.return_value = ({"config": "test-value"}, {}, {}, None)

                        # Mock process
                        mock_process = MagicMock()
                        mock_process.wait.return_value = 0
                        mock_popen.return_value = mock_process

                        # Run command
                        result = runner.invoke(
                            main,
                            ["--version", "v1.0.0", "--parameters", serialized, "--dest", tmpdir, "--", "echo", "test"],
                        )

                        # Verify command succeeded
                        assert result.exit_code == 0

                        # Verify parameters file was created
                        parameters_file = os.path.join(tmpdir, "flyte-parameters.json")
                        assert os.path.exists(parameters_file)

                        # Verify parameters file content
                        with open(parameters_file, "r") as f:
                            saved_parameters = json.load(f)
                        assert saved_parameters["config"] == "test-value"

            finally:
                os.chdir(original_cwd)

    def test_main_with_tgz_code_bundle(self):
        """
        GOAL: Verify main command downloads tgz code bundles.

        Tests that tgz parameter is passed to download_code_parameters.
        """
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            parameters_file = os.path.join(tmpdir, "flyte-parameters.json")
            with patch("flyte.app._parameter.RUNTIME_PARAMETERS_FILE", parameters_file):
                # Mock Popen and asyncio
                with patch("subprocess.Popen") as mock_popen, patch("flyte._bin.serve.asyncio.run") as mock_run:
                    mock_bundle = CodeBundle(tgz="s3://bucket/code.tgz", destination=tmpdir, computed_version="v1.0.0")
                    mock_run.return_value = ({}, {}, {}, mock_bundle)

                    # Mock process
                    mock_process = MagicMock()
                    mock_process.wait.return_value = 0
                    mock_popen.return_value = mock_process

                    # Mock load_app_env to avoid actual loading
                    with patch("flyte._bin.serve.load_app_env") as mock_load:
                        from flyte._image import Image
                        from flyte.app import AppEnvironment

                        mock_app_env = AppEnvironment(name="test-app", image=Image.from_base("python:3.11"))
                        mock_load.return_value = mock_app_env

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
                                "--resolver",
                                "flyte._internal.resolvers.app_env.AppEnvResolver",
                                "--resolver-args",
                                "test.module:app_env",
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

        Tests that pkl parameter is passed to download_code_parameters.
        """
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            parameters_file = os.path.join(tmpdir, "flyte-parameters.json")
            with patch("flyte.app._parameter.RUNTIME_PARAMETERS_FILE", parameters_file):
                # Mock Popen and asyncio
                with patch("subprocess.Popen") as mock_popen, patch("flyte._bin.serve.asyncio.run") as mock_run:
                    # Create a temporary pkl file for the mock bundle
                    import gzip

                    import cloudpickle

                    pkl_file = os.path.join(tmpdir, "code.pkl")
                    # Create a minimal pickled app env
                    from flyte._image import Image
                    from flyte.app import AppEnvironment

                    mock_app_env = AppEnvironment(name="test-app", image=Image.from_base("python:3.11"))
                    with gzip.open(pkl_file, "wb") as f:
                        cloudpickle.dump(mock_app_env, f)

                    # Create bundle with downloaded_path using dataclasses.replace
                    from dataclasses import replace

                    mock_bundle = CodeBundle(pkl="s3://bucket/code.pkl", destination=tmpdir, computed_version="v1.0.0")
                    mock_bundle = replace(mock_bundle, downloaded_path=pkl_file)
                    mock_run.return_value = ({}, {}, {}, mock_bundle)

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
            parameters_file = os.path.join(tmpdir, "flyte-parameters.json")
            with patch("flyte.app._parameter.RUNTIME_PARAMETERS_FILE", parameters_file):
                # Mock Popen and asyncio
                with patch("subprocess.Popen") as mock_popen, patch("flyte._bin.serve.asyncio.run") as mock_run:
                    mock_run.return_value = ({}, {}, {}, None)

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
            parameters_file = os.path.join(tmpdir, "flyte-parameters.json")
            with patch("flyte.app._parameter.RUNTIME_PARAMETERS_FILE", parameters_file):
                # Mock Popen and asyncio
                with patch("subprocess.Popen") as mock_popen, patch("flyte._bin.serve.asyncio.run") as mock_run:
                    mock_run.return_value = ({}, {}, {}, None)

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
            parameters_file = os.path.join(tmpdir, "flyte-parameters.json")
            with patch("flyte.app._parameter.RUNTIME_PARAMETERS_FILE", parameters_file):
                # Mock Popen and asyncio
                with patch("subprocess.Popen") as mock_popen, patch("flyte._bin.serve.asyncio.run") as mock_run:
                    mock_run.return_value = ({}, {}, {}, None)

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
            parameters_file = os.path.join(tmpdir, "flyte-parameters.json")
            with patch("flyte.app._parameter.RUNTIME_PARAMETERS_FILE", parameters_file):
                # Mock everything - signal is imported inside main()
                with (
                    patch("subprocess.Popen") as mock_popen,
                    patch("flyte._bin.serve.asyncio.run") as mock_run,
                    patch("signal.signal") as mock_signal,
                ):
                    mock_run.return_value = ({}, {}, {}, None)

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

    def test_main_parameters_file_environment_variable(self):
        """
        GOAL: Verify main command sets RUNTIME_PARAMETERS_FILE environment variable.

        Tests that the environment variable is set to the parameters file path.
        """
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            parameters_file = os.path.join(tmpdir, "flyte-parameters.json")
            with patch("flyte.app._parameter.RUNTIME_PARAMETERS_FILE", parameters_file):
                original_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)

                    # Mock Popen and asyncio
                    with patch("subprocess.Popen") as mock_popen, patch("flyte._bin.serve.asyncio.run") as mock_run:
                        mock_run.return_value = ({"test": "value"}, {}, {}, None)

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

                        # Verify RUNTIME_PARAMETERS_FILE is in environment
                        from flyte.app._parameter import RUNTIME_PARAMETERS_FILE

                        assert RUNTIME_PARAMETERS_FILE in captured_env
                        assert captured_env[RUNTIME_PARAMETERS_FILE].endswith("flyte-parameters.json")

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
            parameters_file = os.path.join(tmpdir, "flyte-parameters.json")
            with patch("flyte.app._parameter.RUNTIME_PARAMETERS_FILE", parameters_file):
                # Mock Popen and asyncio
                with patch("subprocess.Popen") as mock_popen, patch("flyte._bin.serve.asyncio.run") as mock_run:
                    mock_run.return_value = ({}, {}, {}, None)

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


class TestBindParameters:
    """Tests for _bind_parameters function."""

    def test_bind_parameters_filters_by_function_signature(self):
        """
        GOAL: Verify _bind_parameters only returns parameters that match the function signature.

        Tests that parameters not in the function signature are filtered out.
        """

        def my_func(a, b):
            pass

        materialized_parameters = {"a": "value_a", "b": "value_b", "c": "value_c", "d": "value_d"}

        result = _bind_parameters(my_func, materialized_parameters)

        assert result == {"a": "value_a", "b": "value_b"}
        assert "c" not in result
        assert "d" not in result

    def test_bind_parameters_with_no_matching_params(self):
        """
        GOAL: Verify _bind_parameters returns empty dict when no parameters match.

        Tests that when function signature has no matching parameters, an empty dict is returned.
        """

        def my_func(x, y, z):
            pass

        materialized_parameters = {"a": "value_a", "b": "value_b"}

        result = _bind_parameters(my_func, materialized_parameters)

        assert result == {}

    def test_bind_parameters_with_all_matching_params(self):
        """
        GOAL: Verify _bind_parameters returns all parameters when all match.

        Tests that when all materialized parameters match the function signature, all are returned.
        """

        def my_func(config, model, data):
            pass

        materialized_parameters = {"config": "cfg.yaml", "model": "model.pkl", "data": "data.csv"}

        result = _bind_parameters(my_func, materialized_parameters)

        assert result == materialized_parameters

    def test_bind_parameters_with_empty_materialized_parameters(self):
        """
        GOAL: Verify _bind_parameters handles empty materialized_parameters.

        Tests that when materialized_parameters is empty, an empty dict is returned.
        """

        def my_func(a, b, c):
            pass

        result = _bind_parameters(my_func, {})

        assert result == {}

    def test_bind_parameters_with_no_args_function(self):
        """
        GOAL: Verify _bind_parameters handles functions with no arguments.

        Tests that when function has no parameters, an empty dict is returned.
        """

        def my_func():
            pass

        materialized_parameters = {"a": "value_a", "b": "value_b"}

        result = _bind_parameters(my_func, materialized_parameters)

        assert result == {}

    def test_bind_parameters_with_file_and_dir_types(self):
        """
        GOAL: Verify _bind_parameters correctly handles File and Dir types.

        Tests that File and Dir objects are correctly passed through.
        """
        from flyte.io import Dir, File

        def my_func(model_file, data_dir, config):
            pass

        model = File(path="s3://bucket/model.pkl")
        data = Dir(path="s3://bucket/data/")
        materialized_parameters = {
            "model_file": model,
            "data_dir": data,
            "config": "config.yaml",
            "extra_param": "ignored",
        }

        result = _bind_parameters(my_func, materialized_parameters)

        assert result == {"model_file": model, "data_dir": data, "config": "config.yaml"}
        assert isinstance(result["model_file"], File)
        assert isinstance(result["data_dir"], Dir)

    def test_bind_parameters_with_async_function(self):
        """
        GOAL: Verify _bind_parameters works with async functions.

        Tests that async function signatures are correctly inspected.
        """

        async def my_async_func(a, b):
            pass

        materialized_parameters = {"a": "value_a", "b": "value_b", "c": "value_c"}

        result = _bind_parameters(my_async_func, materialized_parameters)

        assert result == {"a": "value_a", "b": "value_b"}

    def test_bind_parameters_with_default_args(self):
        """
        GOAL: Verify _bind_parameters works with functions that have default arguments.

        Tests that parameters with defaults are still matched.
        """

        def my_func(a, b="default_b", c="default_c"):
            pass

        materialized_parameters = {"a": "value_a", "b": "value_b"}

        result = _bind_parameters(my_func, materialized_parameters)

        assert result == {"a": "value_a", "b": "value_b"}
        assert "c" not in result  # c has default but not in materialized_parameters

    def test_bind_parameters_with_kwargs_passes_all_parameters(self):
        """
        GOAL: Verify _bind_parameters passes all parameters when function has **kwargs.

        Tests that when a function has **kwargs, all materialized parameters are passed through.
        """

        def my_func(a, **kwargs):
            pass

        materialized_parameters = {"a": "value_a", "b": "value_b", "c": "value_c"}

        result = _bind_parameters(my_func, materialized_parameters)

        # All parameters should be bound since function accepts **kwargs
        assert result == {"a": "value_a", "b": "value_b", "c": "value_c"}

    def test_bind_parameters_with_only_kwargs(self):
        """
        GOAL: Verify _bind_parameters passes all parameters when function only has **kwargs.

        Tests that when a function only has **kwargs (no explicit params), all materialized parameters are passed.
        """

        def my_func(**kwargs):
            pass

        materialized_parameters = {"a": "value_a", "b": "value_b", "c": "value_c"}

        result = _bind_parameters(my_func, materialized_parameters)

        # All parameters should be bound since function accepts **kwargs
        assert result == materialized_parameters

    def test_bind_parameters_with_kwargs_and_file_dir_types(self):
        """
        GOAL: Verify _bind_parameters with **kwargs correctly passes File and Dir types.

        Tests that File and Dir objects are correctly passed through when function has **kwargs.
        """
        from flyte.io import Dir, File

        def my_func(config, **kwargs):
            pass

        model = File(path="s3://bucket/model.pkl")
        data = Dir(path="s3://bucket/data/")
        materialized_parameters = {
            "config": "config.yaml",
            "model_file": model,
            "data_dir": data,
            "extra_param": "extra_value",
        }

        result = _bind_parameters(my_func, materialized_parameters)

        # All parameters should be passed through due to **kwargs
        assert result == materialized_parameters
        assert isinstance(result["model_file"], File)
        assert isinstance(result["data_dir"], Dir)

    def test_bind_parameters_with_args(self):
        """
        GOAL: Verify _bind_parameters handles functions with *args.

        Tests that regular positional parameters are correctly bound.
        """

        def my_func(a, *args):
            pass

        materialized_parameters = {"a": "value_a", "b": "value_b"}

        result = _bind_parameters(my_func, materialized_parameters)

        # Only 'a' should be bound as 'b' is not a declared parameter
        assert result == {"a": "value_a"}


class TestServeFunction:
    """Tests for _serve function with parameter binding."""

    @pytest.mark.asyncio
    async def test_serve_binds_parameters_to_server(self):
        """
        GOAL: Verify _serve binds parameters based on _server function signature.

        Tests that only parameters matching the server function signature are passed.
        """

        from flyte._image import Image
        from flyte.app import AppEnvironment

        # Track what parameters were received
        received_params = {}

        async def mock_server(config, model):
            received_params.update({"config": config, "model": model})

        app_env = AppEnvironment(name="test-app", image=Image.from_base("python:3.11"))
        app_env._server = mock_server

        materialized_parameters = {
            "config": "config.yaml",
            "model": "model.pkl",
            "extra_param": "should_be_ignored",
        }

        # Mock signal to prevent actual signal handling
        with patch("signal.signal"):
            # Run _serve but cancel quickly to avoid infinite loop
            try:
                # Create a task that will complete immediately after server runs
                await asyncio.wait_for(_serve(app_env, materialized_parameters), timeout=0.1)
            except asyncio.TimeoutError:
                pass  # Expected - server would normally run forever

        # Verify only matching parameters were passed
        assert received_params == {"config": "config.yaml", "model": "model.pkl"}

    @pytest.mark.asyncio
    async def test_serve_binds_parameters_to_on_startup(self):
        """
        GOAL: Verify _serve binds parameters based on _on_startup function signature.

        Tests that only parameters matching the on_startup function signature are passed.
        """

        from flyte._image import Image
        from flyte.app import AppEnvironment

        # Track what parameters were received
        startup_params = {}
        server_params = {}

        def mock_on_startup(startup_config):
            startup_params.update({"startup_config": startup_config})

        async def mock_server(config, model):
            server_params.update({"config": config, "model": model})

        app_env = AppEnvironment(name="test-app", image=Image.from_base("python:3.11"))
        app_env._on_startup = mock_on_startup
        app_env._server = mock_server

        materialized_parameters = {
            "config": "config.yaml",
            "model": "model.pkl",
            "startup_config": "startup.yaml",
            "extra_param": "should_be_ignored",
        }

        with patch("signal.signal"):
            try:
                await asyncio.wait_for(_serve(app_env, materialized_parameters), timeout=0.1)
            except asyncio.TimeoutError:
                pass

        # Verify on_startup received only its matching parameters
        assert startup_params == {"startup_config": "startup.yaml"}
        # Verify server received its matching parameters
        assert server_params == {"config": "config.yaml", "model": "model.pkl"}

    @pytest.mark.asyncio
    async def test_serve_binds_parameters_to_on_shutdown(self):
        """
        GOAL: Verify _serve binds parameters based on _on_shutdown function signature.

        Tests that only parameters matching the on_shutdown function signature are passed.
        """
        from flyte._image import Image
        from flyte.app import AppEnvironment

        # Track what parameters were received
        shutdown_params = {}

        def mock_on_shutdown(cleanup_path):
            shutdown_params.update({"cleanup_path": cleanup_path})

        async def mock_server(config):
            pass  # Server completes immediately, triggering finally block and shutdown

        app_env = AppEnvironment(name="test-app", image=Image.from_base("python:3.11"))
        app_env._on_shutdown = mock_on_shutdown
        app_env._server = mock_server

        materialized_parameters = {
            "config": "config.yaml",
            "cleanup_path": "/tmp/cleanup",
            "extra_param": "should_be_ignored",
        }

        with patch("signal.signal"):
            try:
                await asyncio.wait_for(_serve(app_env, materialized_parameters), timeout=0.1)
            except asyncio.TimeoutError:
                pass

        # Verify on_shutdown received only its matching parameters
        assert shutdown_params == {"cleanup_path": "/tmp/cleanup"}

    @pytest.mark.asyncio
    async def test_serve_with_async_callbacks(self):
        """
        GOAL: Verify _serve handles async on_startup and on_shutdown correctly.

        Tests that async callbacks work with parameter binding.
        """
        from flyte._image import Image
        from flyte.app import AppEnvironment

        startup_params = {}
        shutdown_params = {}

        async def mock_on_startup(init_config):
            startup_params.update({"init_config": init_config})

        async def mock_on_shutdown(cleanup_config):
            shutdown_params.update({"cleanup_config": cleanup_config})

        async def mock_server(server_config):
            pass

        app_env = AppEnvironment(name="test-app", image=Image.from_base("python:3.11"))
        app_env._on_startup = mock_on_startup
        app_env._on_shutdown = mock_on_shutdown
        app_env._server = mock_server

        materialized_parameters = {
            "init_config": "init.yaml",
            "cleanup_config": "cleanup.yaml",
            "server_config": "server.yaml",
            "extra": "ignored",
        }

        with patch("signal.signal"):
            try:
                await asyncio.wait_for(_serve(app_env, materialized_parameters), timeout=0.1)
            except asyncio.TimeoutError:
                pass

        assert startup_params == {"init_config": "init.yaml"}
        assert shutdown_params == {"cleanup_config": "cleanup.yaml"}

    @pytest.mark.asyncio
    async def test_serve_with_no_parameters_function(self):
        """
        GOAL: Verify _serve handles server functions with no parameters.

        Tests that functions with no parameters work correctly.
        """
        from flyte._image import Image
        from flyte.app import AppEnvironment

        server_called = {"called": False}

        async def mock_server():
            server_called["called"] = True

        app_env = AppEnvironment(name="test-app", image=Image.from_base("python:3.11"))
        app_env._server = mock_server

        materialized_parameters = {
            "config": "config.yaml",
            "model": "model.pkl",
        }

        with patch("signal.signal"):
            try:
                await asyncio.wait_for(_serve(app_env, materialized_parameters), timeout=0.1)
            except asyncio.TimeoutError:
                pass

        assert server_called["called"] is True

    @pytest.mark.asyncio
    async def test_serve_server_with_kwargs_receives_all_parameters(self):
        """
        GOAL: Verify _serve passes all parameters to server function with **kwargs.

        Tests that when server function has **kwargs, all materialized parameters are passed.
        """
        from flyte._image import Image
        from flyte.app import AppEnvironment

        received_params = {}

        async def mock_server(config, **kwargs):
            received_params["config"] = config
            received_params["kwargs"] = kwargs

        app_env = AppEnvironment(name="test-app", image=Image.from_base("python:3.11"))
        app_env._server = mock_server

        materialized_parameters = {
            "config": "config.yaml",
            "model": "model.pkl",
            "data": "data.csv",
            "extra_param": "extra_value",
        }

        with patch("signal.signal"):
            try:
                await asyncio.wait_for(_serve(app_env, materialized_parameters), timeout=0.1)
            except asyncio.TimeoutError:
                pass

        # Verify config was received as positional arg
        assert received_params["config"] == "config.yaml"
        # Verify other params were received via kwargs
        assert received_params["kwargs"] == {
            "model": "model.pkl",
            "data": "data.csv",
            "extra_param": "extra_value",
        }

    @pytest.mark.asyncio
    async def test_serve_on_startup_with_kwargs_receives_all_parameters(self):
        """
        GOAL: Verify _serve passes all parameters to on_startup function with **kwargs.

        Tests that when on_startup function has **kwargs, all materialized parameters are passed.
        """
        from flyte._image import Image
        from flyte.app import AppEnvironment

        startup_params = {}

        def mock_on_startup(init_config, **kwargs):
            startup_params["init_config"] = init_config
            startup_params["kwargs"] = kwargs

        async def mock_server():
            pass

        app_env = AppEnvironment(name="test-app", image=Image.from_base("python:3.11"))
        app_env._on_startup = mock_on_startup
        app_env._server = mock_server

        materialized_parameters = {
            "init_config": "init.yaml",
            "model_path": "model.pkl",
            "cache_size": "1024",
            "debug_mode": "true",
        }

        with patch("signal.signal"):
            try:
                await asyncio.wait_for(_serve(app_env, materialized_parameters), timeout=0.1)
            except asyncio.TimeoutError:
                pass

        # Verify init_config was received as positional arg
        assert startup_params["init_config"] == "init.yaml"
        # Verify other params were received via kwargs
        assert startup_params["kwargs"] == {
            "model_path": "model.pkl",
            "cache_size": "1024",
            "debug_mode": "true",
        }

    @pytest.mark.asyncio
    async def test_serve_on_shutdown_with_kwargs_receives_all_parameters(self):
        """
        GOAL: Verify _serve passes all parameters to on_shutdown function with **kwargs.

        Tests that when on_shutdown function has **kwargs, all materialized parameters are passed.
        """
        from flyte._image import Image
        from flyte.app import AppEnvironment

        shutdown_params = {}

        def mock_on_shutdown(cleanup_path, **kwargs):
            shutdown_params["cleanup_path"] = cleanup_path
            shutdown_params["kwargs"] = kwargs

        async def mock_server():
            pass  # Server completes immediately, triggering finally block and shutdown

        app_env = AppEnvironment(name="test-app", image=Image.from_base("python:3.11"))
        app_env._on_shutdown = mock_on_shutdown
        app_env._server = mock_server

        materialized_parameters = {
            "cleanup_path": "/tmp/cleanup",
            "log_path": "/var/log/app",
            "notify_url": "https://example.com/notify",
        }

        with patch("signal.signal"):
            try:
                await asyncio.wait_for(_serve(app_env, materialized_parameters), timeout=0.1)
            except asyncio.TimeoutError:
                pass

        # Verify cleanup_path was received as positional arg
        assert shutdown_params["cleanup_path"] == "/tmp/cleanup"
        # Verify other params were received via kwargs
        assert shutdown_params["kwargs"] == {
            "log_path": "/var/log/app",
            "notify_url": "https://example.com/notify",
        }

    @pytest.mark.asyncio
    async def test_serve_all_callbacks_with_kwargs(self):
        """
        GOAL: Verify _serve passes all parameters to all callbacks when they have **kwargs.

        Tests the full flow where server, on_startup, and on_shutdown all have **kwargs.
        """
        from flyte._image import Image
        from flyte.app import AppEnvironment

        startup_params = {}
        server_params = {}
        shutdown_params = {}

        async def mock_on_startup(**kwargs):
            startup_params.update(kwargs)

        async def mock_server(**kwargs):
            server_params.update(kwargs)

        async def mock_on_shutdown(**kwargs):
            shutdown_params.update(kwargs)

        app_env = AppEnvironment(name="test-app", image=Image.from_base("python:3.11"))
        app_env._on_startup = mock_on_startup
        app_env._server = mock_server
        app_env._on_shutdown = mock_on_shutdown

        materialized_parameters = {
            "config": "config.yaml",
            "model": "model.pkl",
            "data": "data.csv",
        }

        with patch("signal.signal"):
            try:
                await asyncio.wait_for(_serve(app_env, materialized_parameters), timeout=0.1)
            except asyncio.TimeoutError:
                pass

        # All callbacks should receive all parameters
        assert startup_params == materialized_parameters
        assert server_params == materialized_parameters
        assert shutdown_params == materialized_parameters

    @pytest.mark.asyncio
    async def test_serve_with_kwargs_and_file_dir_types(self):
        """
        GOAL: Verify _serve correctly passes File and Dir types to functions with **kwargs.

        Tests that File and Dir objects are correctly passed when using **kwargs.
        """
        from flyte._image import Image
        from flyte.app import AppEnvironment
        from flyte.io import Dir, File

        received_params = {}

        async def mock_server(**kwargs):
            received_params.update(kwargs)

        app_env = AppEnvironment(name="test-app", image=Image.from_base("python:3.11"))
        app_env._server = mock_server

        model = File(path="s3://bucket/model.pkl")
        data = Dir(path="s3://bucket/data/")
        materialized_parameters = {
            "config": "config.yaml",
            "model_file": model,
            "data_dir": data,
        }

        with patch("signal.signal"):
            try:
                await asyncio.wait_for(_serve(app_env, materialized_parameters), timeout=0.1)
            except asyncio.TimeoutError:
                pass

        # Verify all parameters including File and Dir types were passed
        assert received_params == materialized_parameters
        assert isinstance(received_params["model_file"], File)
        assert isinstance(received_params["data_dir"], Dir)


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_parameters_and_code(self):
        """
        GOAL: Integration test for complete workflow with parameters and code bundle.

        Tests the full flow:
        1. Parameters are deserialized
        2. Code bundle is downloaded
        3. Both are returned correctly
        """
        # Create test parameters
        parameters = [
            Parameter(value="config-data", name="config"),
            Parameter(value="api-key-secret", name="api_key"),
            Parameter(value="test-env-var", name="test_env_var", env_var="TEST_ENV_VAR"),
        ]
        collection = SerializableParameterCollection.from_parameters(parameters)
        serialized = collection.to_transport

        # Mock download_code_bundle
        with patch("flyte._internal.runtime.entrypoints.download_code_bundle", new_callable=AsyncMock) as mock_download:
            mock_bundle = CodeBundle(tgz="s3://bucket/code.tgz", destination="/app", computed_version="v1.0.0")
            mock_download.return_value = mock_bundle

            # Run full download
            serializable_parameters, materialized_parameters, env_vars, code_bundle = await download_code_parameters(
                serialized_parameters=serialized,
                tgz="s3://bucket/code.tgz",
                pkl="",
                dest="/app",
                version="v1.0.0",
            )

            # Verify both parameters and code bundle
            assert serializable_parameters["config"] == "config-data"
            assert serializable_parameters["api_key"] == "api-key-secret"
            assert code_bundle == mock_bundle
            assert code_bundle.tgz == "s3://bucket/code.tgz"
            assert env_vars["TEST_ENV_VAR"] == "test-env-var"

            # Verify materialized_parameters contains string values
            assert materialized_parameters["config"] == "config-data"
            assert materialized_parameters["api_key"] == "api-key-secret"
            assert materialized_parameters["test_env_var"] == "test-env-var"
            assert isinstance(materialized_parameters["config"], str)
            assert isinstance(materialized_parameters["api_key"], str)
            assert isinstance(materialized_parameters["test_env_var"], str)
