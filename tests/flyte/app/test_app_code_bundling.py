"""
Unit tests for app deployment code bundling.

These tests verify that code bundles produced by `build_code_bundle_from_relative_paths`
are deterministic and hash-stable across repeated calls with the same inputs.

The per-app `include` handling that previously lived in `_deploy_app` has been
unified into the global bundling path — see tests/flyte/_code_bundle/ for
coverage of the new unified behavior.
"""

import pathlib
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flyte._code_bundle.bundle import build_code_bundle_from_relative_paths
from flyte._image import Image
from flyte.app import AppEnvironment
from flyte.app._deploy import _deploy_app
from flyte.models import CodeBundle, SerializationContext


@pytest.fixture
def temp_app_directory():
    """
    Create a temporary directory with sample files to include in the code bundle.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)

        (tmp_path / "app.py").write_text("print('Hello, World!')")
        (tmp_path / "utils.py").write_text("def helper(): pass")
        (tmp_path / "config.yaml").write_text("key: value")

        subdir = tmp_path / "submodule"
        subdir.mkdir()
        (subdir / "module.py").write_text("class MyClass: pass")

        yield tmp_path


@pytest.mark.asyncio
async def test_code_bundle_consistency_with_include_files(temp_app_directory):
    include_files = ("app.py", "utils.py", "config.yaml")

    bundle1 = await build_code_bundle_from_relative_paths(
        relative_paths=include_files,
        from_dir=temp_app_directory,
        dryrun=True,
    )
    build_code_bundle_from_relative_paths.cache_clear()
    bundle2 = await build_code_bundle_from_relative_paths(
        relative_paths=include_files,
        from_dir=temp_app_directory,
        dryrun=True,
    )

    assert bundle1.computed_version == bundle2.computed_version
    assert bundle1.files == bundle2.files


@pytest.mark.asyncio
async def test_code_bundle_consistency_with_subdirectory_files(temp_app_directory):
    include_files = ("app.py", "submodule/module.py")

    bundle1 = await build_code_bundle_from_relative_paths(
        relative_paths=include_files,
        from_dir=temp_app_directory,
        dryrun=True,
    )
    build_code_bundle_from_relative_paths.cache_clear()
    bundle2 = await build_code_bundle_from_relative_paths(
        relative_paths=include_files,
        from_dir=temp_app_directory,
        dryrun=True,
    )

    assert bundle1.computed_version == bundle2.computed_version
    assert bundle1.files == bundle2.files
    assert len(bundle1.files) >= 2


@pytest.mark.asyncio
async def test_code_bundle_different_files_produce_different_versions(temp_app_directory):
    bundle1 = await build_code_bundle_from_relative_paths(
        relative_paths=("app.py",),
        from_dir=temp_app_directory,
        dryrun=True,
    )
    build_code_bundle_from_relative_paths.cache_clear()
    bundle2 = await build_code_bundle_from_relative_paths(
        relative_paths=("app.py", "utils.py"),
        from_dir=temp_app_directory,
        dryrun=True,
    )

    assert bundle1.computed_version != bundle2.computed_version


@pytest.mark.asyncio
async def test_code_bundle_file_content_changes_version(temp_app_directory):
    bundle1 = await build_code_bundle_from_relative_paths(
        relative_paths=("app.py",),
        from_dir=temp_app_directory,
        dryrun=True,
    )
    build_code_bundle_from_relative_paths.cache_clear()
    (temp_app_directory / "app.py").write_text("print('Modified content!')")
    bundle2 = await build_code_bundle_from_relative_paths(
        relative_paths=("app.py",),
        from_dir=temp_app_directory,
        dryrun=True,
    )

    assert bundle1.computed_version != bundle2.computed_version


@pytest.mark.asyncio
async def test_code_bundle_empty_include_produces_empty_bundle(temp_app_directory):
    bundle = await build_code_bundle_from_relative_paths(
        relative_paths=(),
        from_dir=temp_app_directory,
        dryrun=True,
    )

    assert bundle.computed_version is not None
    assert bundle.files == []


def test_app_environment_with_include_sets_attribute():
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        include=["app.py", "utils.py", "config/"],
    )

    # `include` is normalized to a tuple so Environment stays hashable.
    assert app_env.include == ("app.py", "utils.py", "config/")


def test_app_environment_default_include_is_empty():
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
    )

    assert app_env.include == ()


@pytest.mark.asyncio
async def test_deploy_app_pkl_bundle_with_include_raises(temp_app_directory):
    """
    Pkl bundles cannot carry include files — the feature must surface a clear
    error rather than silently dropping the user's requested files.
    """

    def mock_server():
        pass

    app_env = AppEnvironment(
        name="test-app-pkl",
        image=Image.from_base("python:3.11"),
        include=["app.py", "utils.py"],
    )
    app_env._server = mock_server

    pkl_bundle = CodeBundle(
        computed_version="pkl123",
        pkl="s3://bucket/code.pkl",
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=pkl_bundle,
        root_dir=temp_app_directory,
    )

    with (
        patch("flyte.app._runtime.translate_app_env_to_idl") as mock_translate,
        patch("flyte.app._deploy.ensure_client"),
    ):
        mock_translate.aio = AsyncMock(return_value=MagicMock())

        with pytest.raises(ValueError, match="include is not supported with pkl"):
            await _deploy_app(app_env, ctx, dryrun=True)
