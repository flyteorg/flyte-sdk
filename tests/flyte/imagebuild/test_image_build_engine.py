import json

import mock
import pytest
from flyteidl2.common import phase_pb2

from flyte._build import ImageBuild
from flyte._image import Image
from flyte._internal.imagebuild.image_builder import (
    DockerAPIImageChecker,
    ImageBuildEngine,
    LocalDockerCommandImageChecker,
    PersistentCacheImageChecker,
)


@mock.patch("flyte._internal.imagebuild.image_builder.DockerAPIImageChecker.image_exists")
@mock.patch("flyte._internal.imagebuild.image_builder.LocalDockerCommandImageChecker.image_exists")
@mock.patch("flyte._internal.imagebuild.image_builder.PersistentCacheImageChecker.image_exists")
@pytest.mark.asyncio
async def test_cached(mock_checker_cache, mock_checker_cli, mock_checker_api):
    # Simulate that the image exists via persistent cache
    mock_checker_cache.return_value = True

    img = Image.from_debian_base()
    await ImageBuildEngine.image_exists(img)
    await ImageBuildEngine.image_exists(img)

    # The persistent cache checker should be called once, and its result cached by alru_cache
    mock_checker_cache.assert_called_once()
    # All other checkers should not be called
    mock_checker_cli.assert_not_called()
    mock_checker_api.assert_not_called()


def test_persistent_cache_write_and_read(tmp_path, monkeypatch):
    """PersistentCacheImageChecker reads back what _write_image_cache wrote."""
    import flyte._internal.imagebuild.image_builder as ib
    from flyte._persistence._db import LocalDB

    monkeypatch.setattr(LocalDB, "_get_db_path", staticmethod(lambda: str(tmp_path / "cache.db")))
    monkeypatch.setattr(LocalDB, "_initialized", False)
    monkeypatch.setattr(LocalDB, "_conn_sync", None)
    monkeypatch.setattr(LocalDB, "_conn", None)
    LocalDB.initialize_sync()

    try:
        # Initially nothing cached — PersistentCacheImageChecker raises LookupError on miss
        import asyncio

        with pytest.raises(LookupError):
            asyncio.get_event_loop().run_until_complete(
                PersistentCacheImageChecker.image_exists("myrepo", "v1.0", ("linux/amd64",))
            )

        # Write to cache
        ib._write_image_cache("myrepo", "v1.0", ("linux/amd64",), "myrepo:v1.0")

        # Now it should be found
        result = asyncio.get_event_loop().run_until_complete(
            PersistentCacheImageChecker.image_exists("myrepo", "v1.0", ("linux/amd64",))
        )
        assert result == "myrepo:v1.0"

        # Different arch should NOT be found
        with pytest.raises(LookupError):
            asyncio.get_event_loop().run_until_complete(
                PersistentCacheImageChecker.image_exists("myrepo", "v1.0", ("linux/arm64",))
            )
    finally:
        LocalDB.close_sync()


@mock.patch("flyte._internal.imagebuild.image_builder.ImageBuildEngine._get_builder")
@mock.patch("flyte._internal.imagebuild.image_builder.ImageBuildEngine.image_exists", new_callable=mock.AsyncMock)
@pytest.mark.asyncio
async def test_build_skips_when_image_exists(mock_image_exists, mock_get_builder):
    """When image already exists and force=False, build_image should not be called."""
    ImageBuildEngine.build.cache_clear()
    mock_image_exists.return_value = "docker.io/test-image:v1.0"
    mock_builder = mock.AsyncMock()
    mock_get_builder.return_value = mock_builder

    img = Image.from_debian_base()
    result = await ImageBuildEngine.build(image=img)
    assert isinstance(result, ImageBuild)
    assert result.uri == "docker.io/test-image:v1.0"
    # Builder should NOT have been called since image exists
    mock_builder.build_image.assert_not_called()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_check():
    assert await DockerAPIImageChecker.image_exists("alpine", "3.9", {"linux/amd64"})


@pytest.mark.integration
@pytest.mark.asyncio
async def test_local_docker():
    await LocalDockerCommandImageChecker.image_exists("ghcr.io/flyteorg/flyte", "91793d843c8385ae386eeb41b54572a9")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_all_image_checkers():
    img = Image.from_debian_base()
    await ImageBuildEngine.image_exists(img)


@mock.patch("flyte._internal.imagebuild.image_builder.ImageBuildEngine._get_builder")
@mock.patch("flyte._internal.imagebuild.image_builder.ImageBuildEngine.image_exists", new_callable=mock.AsyncMock)
@pytest.mark.asyncio
async def test_force_bypasses_existence_check_and_rebuilds(mock_image_exists, mock_get_builder):
    """When force=True, build_image should be called without checking image_exists."""
    ImageBuildEngine.build.cache_clear()
    mock_builder = mock.AsyncMock()
    mock_builder.build_image.return_value = ImageBuild(uri="docker.io/test-image:v1.0", remote_run=None)
    mock_get_builder.return_value = mock_builder

    img = Image.from_debian_base()
    result = await ImageBuildEngine.build(image=img, force=True)
    assert isinstance(result, ImageBuild)
    # image_exists should NOT have been called
    mock_image_exists.assert_not_called()
    # The builder should have been called
    mock_builder.build_image.assert_called_once()
    # force=True should be passed through to build_image
    assert mock_builder.build_image.call_args.kwargs.get("force") is True


@mock.patch("flyte._internal.imagebuild.remote_builder.remote")
@mock.patch("flyte._internal.imagebuild.remote_builder.flyte")
@mock.patch("flyte._internal.imagebuild.remote_builder._validate_configuration", new_callable=mock.AsyncMock)
@mock.patch("flyte._internal.imagebuild.remote_builder._get_fully_qualified_image_name")
@mock.patch("flyte._internal.imagebuild.remote_builder._get_build_secrets_from_image")
@mock.patch("flyte._initialize.get_init_config")
@pytest.mark.asyncio
async def test_remote_builder_default_does_not_overwrite_cache(
    mock_get_init_config, mock_get_secrets, mock_get_fqin, mock_validate, mock_flyte_module, mock_remote
):
    """When force=False (default), RemoteImageBuilder should pass overwrite_cache=False."""
    from flyte._internal.imagebuild.remote_builder import RemoteImageBuilder

    mock_validate.return_value = ("spec_url", "context_url")
    mock_get_secrets.return_value = None
    mock_get_fqin.return_value = "registry/image:tag"

    mock_cfg = mock.MagicMock()
    mock_cfg.project = "test"
    mock_cfg.domain = "development"
    mock_get_init_config.return_value = mock_cfg

    mock_run = mock.AsyncMock()
    mock_run.url = "http://test"
    mock_run.wait.aio = mock.AsyncMock()
    mock_run_details = mock.AsyncMock()
    mock_run_details.action_details.raw_phase = phase_pb2.ACTION_PHASE_SUCCEEDED
    mock_run_details.outputs = mock.AsyncMock(return_value=mock.MagicMock())
    mock_run.details.aio = mock.AsyncMock(return_value=mock_run_details)

    mock_runner = mock.MagicMock()
    mock_runner.run.aio = mock.AsyncMock(return_value=mock_run)
    mock_flyte_module.with_runcontext.return_value = mock_runner

    mock_task = mock.MagicMock()
    mock_task.override.aio = mock.AsyncMock(return_value=mock_task)
    mock_remote.Task.get.return_value = mock_task

    builder = RemoteImageBuilder()
    img = Image.from_debian_base()
    await builder.build_image(img)

    mock_flyte_module.with_runcontext.assert_called_once()
    call_kwargs = mock_flyte_module.with_runcontext.call_args.kwargs
    assert call_kwargs.get("overwrite_cache") is False


@mock.patch("flyte._internal.imagebuild.remote_builder.remote")
@mock.patch("flyte._internal.imagebuild.remote_builder.flyte")
@mock.patch("flyte._internal.imagebuild.remote_builder._validate_configuration", new_callable=mock.AsyncMock)
@mock.patch("flyte._internal.imagebuild.remote_builder._get_fully_qualified_image_name")
@mock.patch("flyte._internal.imagebuild.remote_builder._get_build_secrets_from_image")
@mock.patch("flyte._initialize.get_init_config")
@pytest.mark.asyncio
async def test_remote_builder_force_sets_overwrite_cache(
    mock_get_init_config, mock_get_secrets, mock_get_fqin, mock_validate, mock_flyte_module, mock_remote
):
    """When force=True, RemoteImageBuilder should pass overwrite_cache=True to with_runcontext."""
    from flyte._internal.imagebuild.remote_builder import RemoteImageBuilder

    mock_validate.return_value = ("spec_url", "context_url")
    mock_get_secrets.return_value = None
    mock_get_fqin.return_value = "registry/image:tag"

    mock_cfg = mock.MagicMock()
    mock_cfg.project = "test"
    mock_cfg.domain = "development"
    mock_get_init_config.return_value = mock_cfg

    mock_run = mock.AsyncMock()
    mock_run.url = "http://test"
    mock_run.wait.aio = mock.AsyncMock()
    mock_run_details = mock.AsyncMock()
    mock_run_details.action_details.raw_phase = phase_pb2.ACTION_PHASE_SUCCEEDED
    mock_run_details.outputs = mock.AsyncMock(return_value=mock.MagicMock())
    mock_run.details.aio = mock.AsyncMock(return_value=mock_run_details)

    mock_runner = mock.MagicMock()
    mock_runner.run.aio = mock.AsyncMock(return_value=mock_run)
    mock_flyte_module.with_runcontext.return_value = mock_runner

    mock_task = mock.MagicMock()
    mock_task.override.aio = mock.AsyncMock(return_value=mock_task)
    mock_remote.Task.get.return_value = mock_task

    builder = RemoteImageBuilder()
    img = Image.from_debian_base()
    await builder.build_image(img, force=True)

    mock_flyte_module.with_runcontext.assert_called_once()
    call_kwargs = mock_flyte_module.with_runcontext.call_args.kwargs
    assert call_kwargs.get("overwrite_cache") is True


def _make_mock_process(returncode, stdout_bytes, stderr_bytes):
    proc = mock.AsyncMock()
    proc.communicate.return_value = (stdout_bytes, stderr_bytes)
    proc.returncode = returncode
    return proc


@mock.patch("asyncio.create_subprocess_exec")
@pytest.mark.asyncio
async def test_local_docker_checker_multiplatform_found(mock_exec):
    """Image with matching multi-platform manifest list is found."""
    manifest = {
        "manifests": [
            {"platform": {"os": "linux", "architecture": "amd64"}, "digest": "sha256:aaa"},
            {"platform": {"os": "linux", "architecture": "arm64"}, "digest": "sha256:bbb"},
        ]
    }
    mock_exec.return_value = _make_mock_process(0, json.dumps(manifest).encode(), b"")
    result = await LocalDockerCommandImageChecker.image_exists("localhost:30000/flyte", "abc123")
    assert result == "localhost:30000/flyte:abc123"


@mock.patch("asyncio.create_subprocess_exec")
@pytest.mark.asyncio
async def test_local_docker_checker_filters_attestation_manifests(mock_exec):
    """Attestation manifests (no platform.os) are filtered out."""
    manifest = {
        "manifests": [
            {"platform": {"os": "linux", "architecture": "amd64"}, "digest": "sha256:aaa"},
            {
                "digest": "sha256:ccc",
                "annotations": {"vnd.docker.reference.type": "attestation-manifest"},
            },
        ]
    }
    mock_exec.return_value = _make_mock_process(0, json.dumps(manifest).encode(), b"")
    result = await LocalDockerCommandImageChecker.image_exists("registry/img", "tag1")
    assert result == "registry/img:tag1"


@mock.patch("asyncio.create_subprocess_exec")
@pytest.mark.asyncio
async def test_local_docker_checker_single_platform_manifest(mock_exec):
    """Single-platform image without a manifests key is treated as found."""
    manifest = {"schemaVersion": 2, "config": {"digest": "sha256:abc"}}
    mock_exec.return_value = _make_mock_process(0, json.dumps(manifest).encode(), b"")
    result = await LocalDockerCommandImageChecker.image_exists("registry/img", "tag1")
    assert result == "registry/img:tag1"


@mock.patch("asyncio.create_subprocess_exec")
@pytest.mark.asyncio
async def test_local_docker_checker_not_found(mock_exec):
    """When stderr contains 'no such manifest', returns None."""
    mock_exec.return_value = _make_mock_process(1, b"", b"no such manifest: registry/img:tag1")
    result = await LocalDockerCommandImageChecker.image_exists("registry/img", "tag1")
    assert result is None


@mock.patch("asyncio.create_subprocess_exec")
@pytest.mark.asyncio
async def test_local_docker_checker_manifest_unknown(mock_exec):
    """When stderr contains 'manifest unknown', returns None."""
    mock_exec.return_value = _make_mock_process(1, b"", b"manifest unknown")
    result = await LocalDockerCommandImageChecker.image_exists("registry/img", "tag1")
    assert result is None


@mock.patch("asyncio.create_subprocess_exec")
@pytest.mark.asyncio
async def test_local_docker_checker_arch_mismatch(mock_exec):
    """Image exists but doesn't have the requested architecture."""
    manifest = {
        "manifests": [
            {"platform": {"os": "linux", "architecture": "arm64"}, "digest": "sha256:bbb"},
        ]
    }
    mock_exec.return_value = _make_mock_process(0, json.dumps(manifest).encode(), b"")
    result = await LocalDockerCommandImageChecker.image_exists("registry/img", "tag1", arch=("linux/amd64",))
    assert result is None


@mock.patch("asyncio.create_subprocess_exec")
@pytest.mark.asyncio
async def test_local_docker_checker_unexpected_error_raises(mock_exec):
    """Non-zero exit with unexpected stderr raises RuntimeError."""
    mock_exec.return_value = _make_mock_process(1, b"", b"connection refused")
    with pytest.raises(RuntimeError, match="Failed to run docker buildx imagetools inspect"):
        await LocalDockerCommandImageChecker.image_exists("registry/img", "tag1")
