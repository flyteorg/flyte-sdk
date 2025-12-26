import pathlib
import tempfile
from unittest.mock import Mock

import pytest

import flyte
from flyte._code_bundle._utils import list_all_files, ls_relative_files
from flyte._code_bundle.bundle import build_pkl_bundle
from flyte._internal.runtime.entrypoints import load_pkl_task
from flyte.extras import ContainerTask


def test_list_all_files():
    """Test list_all_files function with a simple directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test directory structure
        test_dir = pathlib.Path(tmpdir)

        # Create subdirectories
        src_dir = test_dir / "src"
        src_dir.mkdir()

        utils_dir = src_dir / "utils"
        utils_dir.mkdir()

        # Create test files
        (test_dir / "main.py").write_text("print('hello')")
        (test_dir / "README.md").write_text("# Test Project")
        (src_dir / "app.py").write_text("import os")
        (utils_dir / "helper.py").write_text("def helper(): pass")

        # Test without ignore_group
        files = list_all_files(test_dir, deref_symlinks=False)

        # Verify all files are found
        assert len(files) == 4

        # Convert to relative paths for easier comparison
        relative_files = [str(pathlib.Path(f).relative_to(test_dir)) for f in files]
        relative_files.sort()

        expected_files = ["README.md", "main.py", "src/app.py", "src/utils/helper.py"]

        assert relative_files == expected_files

        # Test with ignore_group (mock)
        mock_ignore_group = Mock()
        mock_ignore_group.is_ignored.return_value = False

        files_with_ignore = list_all_files(test_dir, deref_symlinks=False, ignore_group=mock_ignore_group)
        assert len(files_with_ignore) == 4


def test_ls_relative_files_with_files():
    """Test ls_relative_files function with individual files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = pathlib.Path(tmpdir)

        # Create test files
        (test_dir / "main.py").write_text("print('hello')")
        (test_dir / "utils.py").write_text("def helper(): pass")
        (test_dir / "config.yaml").write_text("key: value")

        # Test with relative file paths
        files, digest = ls_relative_files(["main.py", "utils.py"], test_dir)

        assert len(files) == 2
        assert str(test_dir / "main.py") in files
        assert str(test_dir / "utils.py") in files
        assert len(digest) > 0  # Should have a hash digest


def test_ls_relative_files_with_directory():
    """Test ls_relative_files function with a directory path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = pathlib.Path(tmpdir)

        # Create subdirectory with files
        utils_dir = test_dir / "utils"
        utils_dir.mkdir()
        (utils_dir / "helper.py").write_text("def helper(): pass")
        (utils_dir / "constants.py").write_text("VALUE = 42")

        # Create another file in root
        (test_dir / "main.py").write_text("print('hello')")

        # Test with directory path - should include all files in the directory
        files, digest = ls_relative_files(["utils"], test_dir)

        # Should find both files in the utils directory
        assert len(files) == 2
        # Convert Path objects to strings for comparison
        file_strs = [str(f) for f in files]
        assert any("helper.py" in f for f in file_strs)
        assert any("constants.py" in f for f in file_strs)
        assert len(digest) > 0


def test_ls_relative_files_with_nested_directory():
    """Test ls_relative_files function with nested directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = pathlib.Path(tmpdir)

        # Create nested directory structure
        src_dir = test_dir / "src"
        src_dir.mkdir()
        (src_dir / "app.py").write_text("import os")

        nested_dir = src_dir / "utils"
        nested_dir.mkdir()
        (nested_dir / "helper.py").write_text("def helper(): pass")

        # Test with parent directory - should recursively find all files
        files, _ = ls_relative_files(["src"], test_dir)

        # Should find both files in src and src/utils
        assert len(files) == 2
        file_strs = [str(f) for f in files]
        assert any("app.py" in f for f in file_strs)
        assert any("helper.py" in f for f in file_strs)


def test_ls_relative_files_with_glob_pattern():
    """Test ls_relative_files function with glob patterns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = pathlib.Path(tmpdir)

        # Create subdirectory with files
        utils_dir = test_dir / "utils"
        utils_dir.mkdir()
        (utils_dir / "helper.py").write_text("def helper(): pass")
        (utils_dir / "constants.py").write_text("VALUE = 42")
        (utils_dir / "README.md").write_text("# Utils")

        # Test with glob pattern - should match only .py files
        files, _ = ls_relative_files(["utils/*.py"], test_dir)

        # Should find only the .py files
        assert len(files) == 2
        file_strs = [str(f) for f in files]
        assert any("helper.py" in f for f in file_strs)
        assert any("constants.py" in f for f in file_strs)
        assert not any("README.md" in f for f in file_strs)


def test_ls_relative_files_with_mixed_inputs():
    """Test ls_relative_files function with a mix of files, directories, and globs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = pathlib.Path(tmpdir)

        # Create directory structure
        (test_dir / "main.py").write_text("print('hello')")

        utils_dir = test_dir / "utils"
        utils_dir.mkdir()
        (utils_dir / "helper.py").write_text("def helper(): pass")

        config_dir = test_dir / "config"
        config_dir.mkdir()
        (config_dir / "dev.yaml").write_text("env: dev")
        (config_dir / "prod.yaml").write_text("env: prod")
        (config_dir / "README.md").write_text("# Config")

        # Test with mix of file, directory, and glob
        files, _ = ls_relative_files(["main.py", "utils", "config/*.yaml"], test_dir)

        # Should find: main.py, utils/helper.py, config/dev.yaml, config/prod.yaml
        assert len(files) == 4
        file_strs = [str(f) for f in files]
        assert any("main.py" in f for f in file_strs)
        assert any("helper.py" in f for f in file_strs)
        assert any("dev.yaml" in f for f in file_strs)
        assert any("prod.yaml" in f for f in file_strs)
        assert not any("README.md" in f for f in file_strs)


def test_ls_relative_files_invalid_path():
    """Test ls_relative_files raises ValueError for invalid paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = pathlib.Path(tmpdir)

        # Create a valid file
        (test_dir / "main.py").write_text("print('hello')")

        # Test with non-existent path that doesn't match any glob
        with pytest.raises(ValueError, match="is not a valid file, directory, or glob pattern"):
            ls_relative_files(["nonexistent.py"], test_dir)


def test_ls_relative_files_empty_directory():
    """Test ls_relative_files with an empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = pathlib.Path(tmpdir)

        # Create empty subdirectory
        empty_dir = test_dir / "empty"
        empty_dir.mkdir()

        # Test with empty directory - should return empty list
        files, _ = ls_relative_files(["empty"], test_dir)

        # Empty directory has no files
        assert len(files) == 0


@pytest.mark.asyncio
async def test_from_task_sets_env():
    greeting_task = ContainerTask(
        name="echo_and_return_greeting",
        image=flyte.Image.from_base("alpine:3.18"),
        input_data_dir="/var/inputs",
        output_data_dir="/var/outputs",
        inputs={"name": str},
        outputs={"greeting": str},
        command=["/bin/sh", "-c", "echo 'Hello, my name is {{.inputs.name}}.' | tee -a /var/outputs/greeting"],
    )

    flyte.TaskEnvironment.from_task("container_env", greeting_task)

    assert greeting_task.parent_env_name == "container_env"

    with tempfile.TemporaryDirectory() as tmp_dir:
        pkled = await build_pkl_bundle(
            greeting_task, upload_to_controlplane=False, copy_bundle_to=pathlib.Path(tmp_dir)
        )
        object.__setattr__(pkled, "downloaded_path", pkled.pkl)
        tt = load_pkl_task(pkled)
        assert tt.parent_env_name == "container_env"
