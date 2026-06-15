import os
import tempfile
from unittest import mock

import pytest

from flyte._internal.runtime.entrypoints import _list_user_files, load_and_run_task
from flyte.models import ActionID, RawDataPath


def test_list_user_files_excludes_venv():
    """Test that _list_user_files filters out venv directories."""
    with tempfile.TemporaryDirectory() as cwd:
        # Create user files
        os.makedirs(os.path.join(cwd, "mypackage"), exist_ok=True)
        open(os.path.join(cwd, "mypackage", "main.py"), "w").close()
        open(os.path.join(cwd, "mypackage", "__init__.py"), "w").close()

        # Create venv files that should be excluded
        os.makedirs(os.path.join(cwd, "venv", "lib", "python3.12", "site-packages"), exist_ok=True)
        open(os.path.join(cwd, "venv", "lib", "python3.12", "site-packages", "pkg.py"), "w").close()

        # Create .venv files that should be excluded
        os.makedirs(os.path.join(cwd, ".venv", "lib"), exist_ok=True)
        open(os.path.join(cwd, ".venv", "lib", "something.py"), "w").close()

        files = _list_user_files(cwd)

        assert os.path.join("mypackage", "main.py") in files
        assert os.path.join("mypackage", "__init__.py") in files

        # venv and .venv files should not appear
        for f in files:
            assert not f.startswith("venv" + os.sep), f"venv file should be excluded: {f}"
            assert not f.startswith(".venv" + os.sep), f".venv file should be excluded: {f}"


def test_list_user_files_excludes_uv_and_local():
    """Test that _list_user_files filters out .local and .uv directories."""
    with tempfile.TemporaryDirectory() as cwd:
        # Create user file
        open(os.path.join(cwd, "hello.py"), "w").close()

        # Create .local directory (uv installs Python here)
        os.makedirs(os.path.join(cwd, ".local", "share", "uv", "python"), exist_ok=True)
        open(os.path.join(cwd, ".local", "share", "uv", "python", "cpython.py"), "w").close()

        # Create .uv directory
        os.makedirs(os.path.join(cwd, ".uv", "cache"), exist_ok=True)
        open(os.path.join(cwd, ".uv", "cache", "cached.whl"), "w").close()

        files = _list_user_files(cwd)

        assert "hello.py" in files

        for f in files:
            assert not f.startswith(".local" + os.sep), f".local file should be excluded: {f}"
            assert not f.startswith(".uv" + os.sep), f".uv file should be excluded: {f}"


def test_list_user_files_excludes_pycache_and_git():
    """Test that _list_user_files filters out __pycache__ and .git directories."""
    with tempfile.TemporaryDirectory() as cwd:
        open(os.path.join(cwd, "app.py"), "w").close()

        os.makedirs(os.path.join(cwd, "__pycache__"), exist_ok=True)
        open(os.path.join(cwd, "__pycache__", "app.cpython-312.pyc"), "w").close()

        os.makedirs(os.path.join(cwd, ".git", "objects"), exist_ok=True)
        open(os.path.join(cwd, ".git", "objects", "abc123"), "w").close()

        files = _list_user_files(cwd)

        assert "app.py" in files

        for f in files:
            assert not f.startswith("__pycache__" + os.sep), f"__pycache__ file should be excluded: {f}"
            assert not f.startswith(".git" + os.sep), f".git file should be excluded: {f}"


def test_list_user_files_excludes_site_packages():
    """Test that _list_user_files filters out site-packages even when nested."""
    with tempfile.TemporaryDirectory() as cwd:
        open(os.path.join(cwd, "workflow.py"), "w").close()

        # site-packages nested inside a lib directory
        os.makedirs(os.path.join(cwd, "lib", "site-packages", "numpy"), exist_ok=True)
        open(os.path.join(cwd, "lib", "site-packages", "numpy", "__init__.py"), "w").close()

        files = _list_user_files(cwd)

        assert "workflow.py" in files

        for f in files:
            assert "site-packages" not in f, f"site-packages file should be excluded: {f}"


def test_list_user_files_only_lists_files_not_dirs():
    """Test that _list_user_files only returns files, not directory entries."""
    with tempfile.TemporaryDirectory() as cwd:
        os.makedirs(os.path.join(cwd, "mypackage"), exist_ok=True)
        open(os.path.join(cwd, "mypackage", "main.py"), "w").close()

        files = _list_user_files(cwd)

        # Should contain files only, not bare directory names
        assert os.path.join("mypackage", "main.py") in files
        assert "mypackage" not in files


def test_list_user_files_empty_directory():
    """Test that _list_user_files returns empty list for empty directory."""
    with tempfile.TemporaryDirectory() as cwd:
        files = _list_user_files(cwd)
        assert files == []


@pytest.mark.asyncio
async def test_load_failure_uploads_error_document():
    """A failure while loading the task (e.g. a missing module import) must write an error
    document so the real reason is surfaced in the UI, instead of an empty
    "terminated with exit code (1)" message. See entrypoints.load_and_run_task.
    """
    boom = ModuleNotFoundError("No module named 'emoji'")

    with (
        mock.patch(
            "flyte._internal.runtime.entrypoints._download_and_load_task",
            side_effect=boom,
        ),
        mock.patch(
            "flyte._internal.runtime.io.upload_error",
            new_callable=mock.AsyncMock,
        ) as mock_upload_error,
    ):
        with pytest.raises(ModuleNotFoundError):
            await load_and_run_task(
                action=ActionID(name="a0", run_name="r0"),
                raw_data_path=RawDataPath(path="/tmp/raw"),
                output_path="s3://bucket/outputs",
                run_base_dir="s3://bucket/run",
                version="v1",
                controller=None,
                resolver="some.resolver",
                resolver_args=["mod", "task"],
            )

    # The error document was uploaded to the task's output path before re-raising.
    mock_upload_error.assert_awaited_once()
    uploaded_err = mock_upload_error.call_args.args[0]
    assert "emoji" in uploaded_err.message
    assert uploaded_err.code == "ModuleNotFoundError"


@pytest.mark.asyncio
async def test_load_failure_without_output_path_still_raises():
    """If there is no output path to write to, the load failure must still propagate."""
    boom = ModuleNotFoundError("No module named 'emoji'")

    with (
        mock.patch(
            "flyte._internal.runtime.entrypoints._download_and_load_task",
            side_effect=boom,
        ),
        mock.patch(
            "flyte._internal.runtime.io.upload_error",
            new_callable=mock.AsyncMock,
        ) as mock_upload_error,
    ):
        with pytest.raises(ModuleNotFoundError):
            await load_and_run_task(
                action=ActionID(name="a0", run_name="r0"),
                raw_data_path=RawDataPath(path="/tmp/raw"),
                output_path="",
                run_base_dir="s3://bucket/run",
                version="v1",
                controller=None,
                resolver="some.resolver",
                resolver_args=["mod", "task"],
            )

    mock_upload_error.assert_not_awaited()
