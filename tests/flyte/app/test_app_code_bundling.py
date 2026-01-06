"""
Unit tests for app deployment code bundling.

These tests verify that code bundles are consistent across multiple deploy/serve calls
when AppEnvironment has include files specified.
"""

import pathlib
import tempfile

import pytest

from flyte._code_bundle.bundle import build_code_bundle_from_relative_paths
from flyte._image import Image
from flyte.app import AppEnvironment


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
