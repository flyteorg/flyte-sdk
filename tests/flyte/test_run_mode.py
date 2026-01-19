"""
Tests for the run mode context variable (_run_mode_var) and related functionality.

These tests verify that:
1. _run_mode_var context variable works correctly
2. _get_main_run_mode() returns the correct mode
3. with_runcontext() sets the run mode correctly
4. The run mode is properly used by offloaded types (File, Dir, DataFrame)
"""

import tempfile
from pathlib import Path

import pytest

import flyte
from flyte._run import _get_main_run_mode, _run_mode_var


class TestRunModeContextVariable:
    """Tests for the run mode context variable."""

    def test_run_mode_var_default_is_none(self):
        """Test that _run_mode_var defaults to None."""
        # Reset to default
        _run_mode_var.set(None)
        assert _run_mode_var.get() is None

    def test_run_mode_var_can_be_set_to_local(self):
        """Test that _run_mode_var can be set to 'local'."""
        _run_mode_var.set("local")
        assert _run_mode_var.get() == "local"
        # Reset
        _run_mode_var.set(None)

    def test_run_mode_var_can_be_set_to_remote(self):
        """Test that _run_mode_var can be set to 'remote'."""
        _run_mode_var.set("remote")
        assert _run_mode_var.get() == "remote"
        # Reset
        _run_mode_var.set(None)

    def test_run_mode_var_can_be_set_to_hybrid(self):
        """Test that _run_mode_var can be set to 'hybrid'."""
        _run_mode_var.set("hybrid")
        assert _run_mode_var.get() == "hybrid"
        # Reset
        _run_mode_var.set(None)


class TestGetMainRunMode:
    """Tests for _get_main_run_mode() function."""

    def test_get_main_run_mode_returns_none_by_default(self):
        """Test that _get_main_run_mode returns None when not set."""
        _run_mode_var.set(None)
        assert _get_main_run_mode() is None

    def test_get_main_run_mode_returns_local(self):
        """Test that _get_main_run_mode returns 'local' when set."""
        _run_mode_var.set("local")
        assert _get_main_run_mode() == "local"
        _run_mode_var.set(None)

    def test_get_main_run_mode_returns_remote(self):
        """Test that _get_main_run_mode returns 'remote' when set."""
        _run_mode_var.set("remote")
        assert _get_main_run_mode() == "remote"
        _run_mode_var.set(None)


class TestWithRunContextSetsMode:
    """Tests for with_runcontext() setting the run mode."""

    def test_with_runcontext_sets_local_mode(self):
        """Test that with_runcontext(mode='local') sets _run_mode_var to 'local'."""
        flyte.init()
        _run_mode_var.set(None)  # Reset first

        # with_runcontext should set the mode
        flyte.with_runcontext(mode="local")

        assert _get_main_run_mode() == "local"
        _run_mode_var.set(None)

    def test_with_runcontext_sets_remote_mode(self):
        """Test that with_runcontext(mode='remote') sets _run_mode_var to 'remote'."""
        flyte.init()
        _run_mode_var.set(None)  # Reset first

        # with_runcontext should set the mode
        flyte.with_runcontext(mode="remote")

        assert _get_main_run_mode() == "remote"
        _run_mode_var.set(None)

    def test_with_runcontext_sets_none_mode(self):
        """Test that with_runcontext(mode=None) sets _run_mode_var to None."""
        flyte.init()
        _run_mode_var.set("local")  # Set to something first

        # with_runcontext with mode=None should set to None
        flyte.with_runcontext(mode=None)

        assert _get_main_run_mode() is None


class TestRunModeWithFlyteRun:
    """Tests for run mode integration with flyte.run."""

    @pytest.fixture
    def sample_env(self):
        """Create a sample TaskEnvironment for testing."""
        return flyte.TaskEnvironment(name="test-run-mode")

    def test_flyte_run_local_mode(self, sample_env):
        """Test that flyte.run in local mode works correctly."""
        flyte.init()
        _run_mode_var.set(None)  # Reset

        @sample_env.task
        async def simple_task(x: int) -> int:
            return x * 2

        run = flyte.with_runcontext(mode="local").run(simple_task, x=5)
        run.wait()
        assert run.outputs()[0] == 10

    @pytest.mark.asyncio
    async def test_flyte_run_aio_local_mode(self, sample_env):
        """Test that flyte.run.aio in local mode works correctly."""
        await flyte.init.aio()
        _run_mode_var.set(None)  # Reset

        @sample_env.task
        async def simple_task(x: int) -> int:
            return x * 2

        run = await flyte.with_runcontext(mode="local").run.aio(simple_task, x=5)
        await run.wait.aio()
        outputs = await run.outputs.aio()
        assert outputs[0] == 10


class TestRunModeWithLocalFile:
    """Tests for run mode with local File inputs."""

    @pytest.fixture
    def sample_env(self):
        """Create a sample TaskEnvironment for testing."""
        return flyte.TaskEnvironment(name="test-run-mode-file")

    def test_file_from_local_in_local_mode(self, sample_env):
        """Test that File.from_local works in local mode without uploading."""
        from flyte.io import File

        flyte.init()

        @sample_env.task
        async def read_file(file: File) -> str:
            async with file.open("r") as f:
                return await f.read()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
            tmp.write("Hello, Flyte!")
            tmp.flush()
            temp_path = tmp.name

        file = File.from_local_sync(temp_path)

        # In local mode, the file should use the local path
        run = flyte.with_runcontext(mode="local").run(read_file, file=file)
        run.wait()
        assert run.outputs()[0] == "Hello, Flyte!"

        # Clean up
        Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_file_from_local_aio_in_local_mode(self, sample_env):
        """Test that File.from_local (async) works in local mode without uploading."""
        from flyte.io import File

        await flyte.init.aio()

        @sample_env.task
        async def read_file(file: File) -> str:
            async with file.open("r") as f:
                return await f.read()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
            tmp.write("Hello, Async Flyte!")
            tmp.flush()
            temp_path = tmp.name

        file = await File.from_local(temp_path)

        # In local mode, the file should use the local path
        run = await flyte.with_runcontext(mode="local").run.aio(read_file, file=file)
        await run.wait.aio()
        outputs = await run.outputs.aio()
        assert outputs[0] == "Hello, Async Flyte!"

        # Clean up
        Path(temp_path).unlink()


class TestRunModeWithLocalDir:
    """Tests for run mode with local Dir inputs."""

    @pytest.fixture
    def sample_env(self):
        """Create a sample TaskEnvironment for testing."""
        return flyte.TaskEnvironment(name="test-run-mode-dir")

    def test_dir_from_local_in_local_mode(self, sample_env):
        """Test that Dir.from_local works in local mode without uploading."""
        from flyte.io import Dir

        flyte.init()

        @sample_env.task
        async def list_files(dir: Dir) -> list[str]:
            files = []
            async for file in dir.walk(recursive=False):
                files.append(file.name)
            return sorted(files)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create some test files
            (Path(tmp_dir) / "file1.txt").write_text("content1")
            (Path(tmp_dir) / "file2.txt").write_text("content2")

            dir_obj = Dir.from_local_sync(tmp_dir)

            # In local mode, the dir should use the local path
            run = flyte.with_runcontext(mode="local").run(list_files, dir=dir_obj)
            run.wait()
            assert "file1.txt" in run.outputs()[0]
            assert "file2.txt" in run.outputs()[0]

    @pytest.mark.asyncio
    async def test_dir_from_local_aio_in_local_mode(self, sample_env):
        """Test that Dir.from_local (async) works in local mode without uploading."""
        from flyte.io import Dir

        await flyte.init.aio()

        @sample_env.task
        async def list_files(dir: Dir) -> list[str]:
            files = []
            async for file in dir.walk(recursive=False):
                files.append(file.name)
            return sorted(files)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create some test files
            (Path(tmp_dir) / "file_a.txt").write_text("content_a")
            (Path(tmp_dir) / "file_b.txt").write_text("content_b")

            dir_obj = await Dir.from_local(tmp_dir)

            # In local mode, the dir should use the local path
            run = await flyte.with_runcontext(mode="local").run.aio(list_files, dir=dir_obj)
            await run.wait.aio()
            outputs = await run.outputs.aio()
            assert "file_a.txt" in outputs[0]
            assert "file_b.txt" in outputs[0]


class TestRunModeWithLocalDataFrame:
    """Tests for run mode with local DataFrame inputs."""

    @pytest.fixture
    def sample_env(self):
        """Create a sample TaskEnvironment for testing."""
        return flyte.TaskEnvironment(name="test-run-mode-dataframe")

    def test_dataframe_from_local_in_local_mode(self, sample_env):
        """Test that DataFrame.from_local works in local mode without uploading."""
        pd = pytest.importorskip("pandas")
        from flyte.io import DataFrame

        flyte.init()

        @sample_env.task
        async def process_df(df: pd.DataFrame) -> int:
            return len(df)

        test_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        fdf = DataFrame.from_local_sync(test_df)

        # In local mode, the dataframe should work without remote upload
        run = flyte.with_runcontext(mode="local").run(process_df, df=fdf)
        run.wait()
        assert run.outputs()[0] == 3

    @pytest.mark.asyncio
    async def test_dataframe_from_local_aio_in_local_mode(self, sample_env):
        """Test that DataFrame.from_local (async) works in local mode without uploading."""
        pd = pytest.importorskip("pandas")
        from flyte.io import DataFrame

        await flyte.init.aio()

        @sample_env.task
        async def process_df(df: pd.DataFrame) -> int:
            return len(df)

        test_df = pd.DataFrame({"x": [10, 20, 30, 40], "y": [100, 200, 300, 400]})
        fdf = await DataFrame.from_local(test_df)

        # In local mode, the dataframe should work without remote upload
        run = await flyte.with_runcontext(mode="local").run.aio(process_df, df=fdf)
        await run.wait.aio()
        outputs = await run.outputs.aio()
        assert outputs[0] == 4
