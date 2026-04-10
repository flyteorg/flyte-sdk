from unittest import mock
from unittest.mock import patch

import pytest

from flyte._image import Image
from flyte._internal.imagebuild.image_builder import ImageBuildEngine, ImageCache, RunIdentifierData


@pytest.mark.asyncio
async def test_exists():
    """
    Test the ImageBuilder class.
    """

    ubuntu = Image.from_base("ghcr.io/unionai-oss:ubuntu")
    assert await ImageBuildEngine.image_exists(ubuntu) == "ghcr.io/unionai-oss:ubuntu"


@pytest.mark.asyncio
async def test_image_exists_skips_check_for_unmodified_default_image():
    """A released unmodified default image should short-circuit image_exists without
    hitting any backend checker."""
    ImageBuildEngine.image_exists.cache_clear()
    with patch("flyte._version.__version__", "1.2.3"):
        image = Image.from_debian_base(python_version=(3, 12))
    assert image._is_cloned is False

    # Patch the backend checkers so any attempt to hit them would fail the test.
    with (
        mock.patch(
            "flyte._internal.imagebuild.image_builder.DockerAPIImageChecker.image_exists",
            new_callable=mock.AsyncMock,
        ) as docker_api_check,
        mock.patch(
            "flyte._internal.imagebuild.image_builder.LocalDockerCommandImageChecker.image_exists",
            new_callable=mock.AsyncMock,
        ) as local_docker_check,
    ):
        result = await ImageBuildEngine.image_exists(image)
    assert result == image.uri
    docker_api_check.assert_not_called()
    local_docker_check.assert_not_called()


@pytest.mark.asyncio
async def test_image_exists_checks_backend_for_modified_default_image():
    """Adding a layer to the default image flips _is_cloned=True and the existence check
    should no longer short-circuit."""
    ImageBuildEngine.image_exists.cache_clear()
    with patch("flyte._version.__version__", "1.2.3"):
        image = Image.from_debian_base(
            python_version=(3, 12),
            registry="my-registry.example.com",
            name="my-image",
        ).with_pip_packages("requests")
    assert image._is_cloned is True

    with mock.patch(
        "flyte._internal.imagebuild.image_builder.LocalDockerCommandImageChecker.image_exists",
        new_callable=mock.AsyncMock,
        return_value="checked-uri",
    ) as local_docker_check:
        await ImageBuildEngine.image_exists(image)
    assert local_docker_check.called


@pytest.mark.asyncio
async def test_image_exists_skips_check_for_from_base():
    """from_base points at an existing image URI; image_exists should short-circuit."""
    ImageBuildEngine.image_exists.cache_clear()
    image = Image.from_base("ghcr.io/example/my-image:latest")
    assert image._is_cloned is False

    with (
        mock.patch(
            "flyte._internal.imagebuild.image_builder.DockerAPIImageChecker.image_exists",
            new_callable=mock.AsyncMock,
        ) as docker_api_check,
        mock.patch(
            "flyte._internal.imagebuild.image_builder.LocalDockerCommandImageChecker.image_exists",
            new_callable=mock.AsyncMock,
        ) as local_docker_check,
    ):
        result = await ImageBuildEngine.image_exists(image)
    assert result == image.uri
    docker_api_check.assert_not_called()
    local_docker_check.assert_not_called()


def test_image_cache_build_run_ids_roundtrip():
    """build_run_ids survives to_transport → from_transport serialization."""
    run_id = RunIdentifierData(org="my-org", project="my-project", domain="development", name="abc123")
    cache = ImageCache(
        image_lookup={"my-env": "registry/my-image:sha256abc"},
        build_run_ids={"my-env": run_id},
    )

    restored = ImageCache.from_transport(cache.to_transport)

    assert restored.image_lookup == cache.image_lookup
    assert restored.build_run_ids["my-env"] == run_id


def test_image_cache_build_run_ids_defaults_empty():
    """ImageCache with no build_run_ids defaults to empty dict."""
    cache = ImageCache(image_lookup={"my-env": "registry/my-image:sha256abc"})

    assert cache.build_run_ids == {}


def test_image_cache_old_serialized_form_still_deserializes():
    """An ImageCache serialized before build_run_ids existed can still be loaded."""
    old_cache = ImageCache(image_lookup={"my-env": "registry/my-image:sha256abc"})

    # Simulate an old serialized form that has no build_run_ids key
    restored = ImageCache.from_transport(old_cache.to_transport)

    assert restored.image_lookup == old_cache.image_lookup
    assert restored.build_run_ids == {}
