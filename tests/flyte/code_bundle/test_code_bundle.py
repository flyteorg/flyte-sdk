import pathlib
import tempfile
from unittest.mock import Mock

import pytest

import flyte
from flyte._code_bundle._utils import list_all_files
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
