"""
Unit tests for app deployment code bundling.

These tests verify that code bundles are consistent across multiple deploy/serve calls
when AppEnvironment has include files specified.
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

        # Create sample files to include
        (tmp_path / "app.py").write_text("print('Hello, World!')")
        (tmp_path / "utils.py").write_text("def helper(): pass")
        (tmp_path / "config.yaml").write_text("key: value")

        # Create a subdirectory with files
        subdir = tmp_path / "submodule"
        subdir.mkdir()
        (subdir / "module.py").write_text("class MyClass: pass")

        yield tmp_path


@pytest.mark.asyncio
async def test_code_bundle_consistency_with_include_files(temp_app_directory):
    """
    GOAL: Verify that code bundles produced with include files are consistent
    across multiple calls.

    Tests that:
    - When an AppEnvironment has `include` specified, calling build_code_bundle_from_relative_paths
      multiple times with the same files produces the same computed_version and files list.
    - The code bundle's computed_version (hash) is deterministic for the same file contents.
    """
    include_files = ("app.py", "utils.py", "config.yaml")

    # Build code bundle first time (dryrun to avoid network calls)
    bundle1 = await build_code_bundle_from_relative_paths(
        relative_paths=include_files,
        from_dir=temp_app_directory,
        dryrun=True,
    )

    # Clear the alru_cache to ensure we're testing actual bundle creation
    build_code_bundle_from_relative_paths.cache_clear()

    # Build code bundle second time
    bundle2 = await build_code_bundle_from_relative_paths(
        relative_paths=include_files,
        from_dir=temp_app_directory,
        dryrun=True,
    )

    # Verify consistency
    assert bundle1.computed_version == bundle2.computed_version, (
        f"Code bundle computed_version should be consistent across calls. "
        f"Got {bundle1.computed_version} and {bundle2.computed_version}"
    )
    assert bundle1.files == bundle2.files, (
        f"Code bundle files should be consistent across calls. Got {bundle1.files} and {bundle2.files}"
    )


@pytest.mark.asyncio
async def test_code_bundle_consistency_with_subdirectory_files(temp_app_directory):
    """
    GOAL: Verify that code bundles including subdirectory files are consistent.

    Tests that files from subdirectories are properly included and the
    computed_version remains consistent across multiple calls.
    """
    include_files = ("app.py", "submodule/module.py")

    # Build code bundle first time
    bundle1 = await build_code_bundle_from_relative_paths(
        relative_paths=include_files,
        from_dir=temp_app_directory,
        dryrun=True,
    )

    # Clear cache
    build_code_bundle_from_relative_paths.cache_clear()

    # Build code bundle second time
    bundle2 = await build_code_bundle_from_relative_paths(
        relative_paths=include_files,
        from_dir=temp_app_directory,
        dryrun=True,
    )

    # Verify consistency
    assert bundle1.computed_version == bundle2.computed_version
    assert bundle1.files == bundle2.files

    # Verify that the files list contains expected files
    assert len(bundle1.files) >= 2


@pytest.mark.asyncio
async def test_code_bundle_different_files_produce_different_versions(temp_app_directory):
    """
    GOAL: Verify that different file sets produce different computed_versions.

    Tests that the hash is actually dependent on the file contents, not just
    producing the same hash every time.
    """
    include_files_v1 = ("app.py",)
    include_files_v2 = ("app.py", "utils.py")

    bundle1 = await build_code_bundle_from_relative_paths(
        relative_paths=include_files_v1,
        from_dir=temp_app_directory,
        dryrun=True,
    )

    build_code_bundle_from_relative_paths.cache_clear()

    bundle2 = await build_code_bundle_from_relative_paths(
        relative_paths=include_files_v2,
        from_dir=temp_app_directory,
        dryrun=True,
    )

    # Different file sets should produce different versions
    assert bundle1.computed_version != bundle2.computed_version, (
        "Different file sets should produce different computed_versions"
    )


@pytest.mark.asyncio
async def test_code_bundle_file_content_changes_version(temp_app_directory):
    """
    GOAL: Verify that changing file contents changes the computed_version.

    Tests that:
    - If file content changes between calls, the computed_version changes.
    - This ensures the hash is actually based on file contents.
    """
    include_files = ("app.py",)

    # Build code bundle with original content
    bundle1 = await build_code_bundle_from_relative_paths(
        relative_paths=include_files,
        from_dir=temp_app_directory,
        dryrun=True,
    )

    build_code_bundle_from_relative_paths.cache_clear()

    # Modify file content
    (temp_app_directory / "app.py").write_text("print('Modified content!')")

    # Build code bundle with modified content
    bundle2 = await build_code_bundle_from_relative_paths(
        relative_paths=include_files,
        from_dir=temp_app_directory,
        dryrun=True,
    )

    # Different file contents should produce different versions
    assert bundle1.computed_version != bundle2.computed_version, (
        "Modified file contents should produce different computed_versions"
    )


@pytest.mark.asyncio
async def test_code_bundle_preexisting_files_merge_with_include():
    """
    GOAL: Verify that preexisting code bundle files are merged with include files
    without duplication.

    Tests the logic in _deploy_app that merges preexisting code bundle files
    with the app's include list:
        files = (*_preexisting_code_bundle_files, *[f for f in app.include if f not in _preexisting_code_bundle_files])
    """
    preexisting_files = ["common.py", "shared_utils.py"]
    include_files = ["app.py", "common.py", "config.yaml"]  # common.py is in both lists

    # Simulate the merge logic from _deploy_app
    merged_files = (*preexisting_files, *[f for f in include_files if f not in preexisting_files])

    # Expected: common.py should only appear once
    assert list(merged_files) == ["common.py", "shared_utils.py", "app.py", "config.yaml"]
    assert merged_files.count("common.py") == 1


@pytest.mark.asyncio
async def test_code_bundle_empty_include_produces_empty_bundle(temp_app_directory):
    """
    GOAL: Verify that an empty include list produces a bundle with no files.
    """
    include_files = ()

    bundle = await build_code_bundle_from_relative_paths(
        relative_paths=include_files,
        from_dir=temp_app_directory,
        dryrun=True,
    )

    # Empty include should still create a valid bundle
    assert bundle.computed_version is not None
    assert bundle.files == []


def test_app_environment_with_include_sets_attribute():
    """
    GOAL: Verify that AppEnvironment correctly stores the include list.
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
        include=["app.py", "utils.py", "config/"],
    )

    assert app_env.include == ["app.py", "utils.py", "config/"]


def test_app_environment_default_include_is_empty():
    """
    GOAL: Verify that AppEnvironment defaults to an empty include list.
    """
    app_env = AppEnvironment(
        name="test-app",
        image=Image.from_base("python:3.11"),
    )

    assert app_env.include == []


# =============================================================================
# Tests for _deploy_app code bundling logic
# =============================================================================


@pytest.fixture
def mock_app_env_with_include(temp_app_directory):
    """
    Create an AppEnvironment with include files specified.
    """
    app_env = AppEnvironment(
        name="test-app-include",
        image=Image.from_base("python:3.11"),
        include=["app.py", "utils.py"],
    )
    # Set the _app_filename to simulate the app being defined in a file
    app_env._app_filename = str(temp_app_directory / "app.py")
    return app_env


@pytest.fixture
def mock_serialization_context(temp_app_directory):
    """
    Create a SerializationContext without a code bundle.
    """
    return SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=None,
        root_dir=temp_app_directory,
    )


@pytest.mark.asyncio
async def test_deploy_app_include_only_builds_code_bundle(temp_app_directory):
    """
    GOAL: Verify that when AppEnvironment has only `include` specified (no pre-existing
    code bundle), build_code_bundle_from_relative_paths is called to create the code bundle.

    Tests that:
    - When app.include is specified and serialization_context.code_bundle is None
    - build_code_bundle_from_relative_paths is called with the include files
    - The resulting code_bundle is set on the serialization_context
    """
    app_env = AppEnvironment(
        name="test-app-include-only",
        image=Image.from_base("python:3.11"),
        include=["app.py", "utils.py"],
    )
    app_env._app_filename = str(temp_app_directory / "app.py")

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=None,
        root_dir=temp_app_directory,
    )

    mock_bundle = CodeBundle(
        computed_version="abc123",
        tgz="s3://bucket/code.tgz",
        files=["app.py", "utils.py"],
    )

    with (
        patch("flyte.app._deploy.build_code_bundle_from_relative_paths", new_callable=AsyncMock) as mock_build_bundle,
        patch("flyte.app._runtime.translate_app_env_to_idl") as mock_translate,
        patch("flyte.app._deploy.ensure_client"),
    ):
        mock_build_bundle.return_value = mock_bundle
        mock_translate.aio = AsyncMock(return_value=MagicMock())

        # Call _deploy_app with dryrun=True to avoid actual deployment
        await _deploy_app(app_env, ctx, dryrun=True)

        # Verify build_code_bundle_from_relative_paths was called
        mock_build_bundle.assert_called_once()
        call_args = mock_build_bundle.call_args

        # Verify the files passed are the include files
        assert call_args[0][0] == ("app.py", "utils.py")
        # Verify the from_dir is the app's parent directory
        assert call_args[1]["from_dir"] == temp_app_directory

        # Verify the code_bundle was set on the serialization_context
        assert ctx.code_bundle == mock_bundle


@pytest.mark.asyncio
async def test_deploy_app_include_with_preexisting_tgz_bundle_merges_files(temp_app_directory):
    """
    GOAL: Verify that when both `include` and a pre-existing tgz code bundle with files
    are specified, the files are merged without duplication.

    Tests that:
    - Pre-existing code bundle files and include files are merged
    - Duplicate files are not included twice
    - build_code_bundle_from_relative_paths is called with merged file list
    """
    app_env = AppEnvironment(
        name="test-app-merge",
        image=Image.from_base("python:3.11"),
        include=["app.py", "config.yaml", "utils.py"],  # utils.py overlaps with preexisting
    )
    app_env._app_filename = str(temp_app_directory / "app.py")

    # Pre-existing code bundle with some files already included
    preexisting_bundle = CodeBundle(
        computed_version="preexisting123",
        tgz="s3://bucket/preexisting.tgz",
        files=["utils.py", "shared.py"],  # utils.py also in include list
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=preexisting_bundle,
        root_dir=temp_app_directory,
    )

    mock_new_bundle = CodeBundle(
        computed_version="merged123",
        tgz="s3://bucket/merged.tgz",
        files=["utils.py", "shared.py", "app.py", "config.yaml"],
    )

    with (
        patch("flyte.app._deploy.build_code_bundle_from_relative_paths", new_callable=AsyncMock) as mock_build_bundle,
        patch("flyte.app._runtime.translate_app_env_to_idl") as mock_translate,
        patch("flyte.app._deploy.ensure_client"),
    ):
        mock_build_bundle.return_value = mock_new_bundle
        mock_translate.aio = AsyncMock(return_value=MagicMock())

        await _deploy_app(app_env, ctx, dryrun=True)

        # Verify build_code_bundle_from_relative_paths was called
        mock_build_bundle.assert_called_once()
        call_args = mock_build_bundle.call_args

        # The merged files should be: preexisting files + include files (excluding duplicates)
        # utils.py from preexisting, shared.py from preexisting, app.py from include, config.yaml from include
        # utils.py should NOT appear twice
        merged_files = call_args[0][0]
        assert "utils.py" in merged_files
        assert "shared.py" in merged_files
        assert "app.py" in merged_files
        assert "config.yaml" in merged_files
        # Verify no duplicates
        assert merged_files.count("utils.py") == 1


@pytest.mark.asyncio
async def test_deploy_app_pkl_bundle_does_not_build_code_bundle(temp_app_directory):
    """
    GOAL: Verify that when a pickle (pkl) code bundle is specified, 
    build_code_bundle_from_relative_paths is NOT called, even if include is specified.

    Tests that:
    - When serialization_context.code_bundle.pkl is set (pickle bundle)
    - build_code_bundle_from_relative_paths is NOT called
    - The pickle bundle flow assumes the server function contains all needed code
    """

    def mock_server():
        """Mock server function for pkl bundle."""
        pass

    app_env = AppEnvironment(
        name="test-app-pkl",
        image=Image.from_base("python:3.11"),
        include=["app.py", "utils.py"],  # Include is specified but should be ignored for pkl
    )
    app_env._app_filename = str(temp_app_directory / "app.py")
    app_env._server = mock_server  # pkl bundles require a server function

    # Create a pickle code bundle
    pkl_bundle = CodeBundle(
        computed_version="pkl123",
        pkl="s3://bucket/code.pkl",  # pkl is set, not tgz
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
        patch("flyte.app._deploy.build_code_bundle_from_relative_paths", new_callable=AsyncMock) as mock_build_bundle,
        patch("flyte.app._runtime.translate_app_env_to_idl") as mock_translate,
        patch("flyte.app._deploy.ensure_client"),
    ):
        mock_translate.aio = AsyncMock(return_value=MagicMock())

        await _deploy_app(app_env, ctx, dryrun=True)

        # Verify build_code_bundle_from_relative_paths was NOT called for pkl bundle
        mock_build_bundle.assert_not_called()

        # Verify the original pkl bundle is still intact
        assert ctx.code_bundle == pkl_bundle
        assert ctx.code_bundle.pkl == "s3://bucket/code.pkl"


@pytest.mark.asyncio
async def test_deploy_app_no_include_does_not_build_code_bundle(temp_app_directory):
    """
    GOAL: Verify that when AppEnvironment has no `include` specified,
    build_code_bundle_from_relative_paths is NOT called.

    Tests that:
    - When app.include is empty/None
    - build_code_bundle_from_relative_paths is NOT called
    - The serialization_context.code_bundle remains as-is
    """
    app_env = AppEnvironment(
        name="test-app-no-include",
        image=Image.from_base("python:3.11"),
        include=[],  # Empty include list
    )
    app_env._app_filename = str(temp_app_directory / "app.py")

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=None,
        root_dir=temp_app_directory,
    )

    with (
        patch("flyte.app._deploy.build_code_bundle_from_relative_paths", new_callable=AsyncMock) as mock_build_bundle,
        patch("flyte.app._runtime.translate_app_env_to_idl") as mock_translate,
        patch("flyte.app._deploy.ensure_client"),
    ):
        mock_translate.aio = AsyncMock(return_value=MagicMock())

        await _deploy_app(app_env, ctx, dryrun=True)

        # Verify build_code_bundle_from_relative_paths was NOT called
        mock_build_bundle.assert_not_called()

        # Verify code_bundle remains None
        assert ctx.code_bundle is None


@pytest.mark.asyncio
async def test_deploy_app_include_with_preexisting_bundle_no_files(temp_app_directory):
    """
    GOAL: Verify that when include is specified and pre-existing code bundle has no files,
    only the include files are used.

    Tests that:
    - When pre-existing code bundle has files=None or files=[]
    - Only the include files are passed to build_code_bundle_from_relative_paths
    """
    app_env = AppEnvironment(
        name="test-app-include-no-preexisting-files",
        image=Image.from_base("python:3.11"),
        include=["app.py", "utils.py"],
    )
    app_env._app_filename = str(temp_app_directory / "app.py")

    # Pre-existing bundle with no files (files=None)
    preexisting_bundle = CodeBundle(
        computed_version="preexisting123",
        tgz="s3://bucket/preexisting.tgz",
        files=None,  # No files in pre-existing bundle
    )

    ctx = SerializationContext(
        org="test-org",
        project="test-project",
        domain="test-domain",
        version="v1.0.0",
        code_bundle=preexisting_bundle,
        root_dir=temp_app_directory,
    )

    mock_new_bundle = CodeBundle(
        computed_version="new123",
        tgz="s3://bucket/new.tgz",
        files=["app.py", "utils.py"],
    )

    with (
        patch("flyte.app._deploy.build_code_bundle_from_relative_paths", new_callable=AsyncMock) as mock_build_bundle,
        patch("flyte.app._runtime.translate_app_env_to_idl") as mock_translate,
        patch("flyte.app._deploy.ensure_client"),
    ):
        mock_build_bundle.return_value = mock_new_bundle
        mock_translate.aio = AsyncMock(return_value=MagicMock())

        await _deploy_app(app_env, ctx, dryrun=True)

        # Verify build_code_bundle_from_relative_paths was called
        mock_build_bundle.assert_called_once()
        call_args = mock_build_bundle.call_args

        # Only include files should be passed (no preexisting files to merge)
        assert call_args[0][0] == ("app.py", "utils.py")
