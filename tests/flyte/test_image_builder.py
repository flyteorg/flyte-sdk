import pytest

from flyte._image import Image
from flyte._internal.imagebuild.image_builder import ImageBuildEngine, ImageCache


@pytest.mark.asyncio
async def test_exists():
    """
    Test the ImageBuilder class.
    """

    ubuntu = Image.from_base("ghcr.io/unionai-oss:ubuntu")
    assert await ImageBuildEngine.image_exists(ubuntu) == "ghcr.io/unionai-oss:ubuntu"


def test_image_cache_build_run_urls_roundtrip():
    """build_run_urls survives to_transport → from_transport serialization."""
    cache = ImageCache(
        image_lookup={"my-env": "registry/my-image:sha256abc"},
        build_run_urls={"my-env": "https://console.union.ai/runs/abc123"},
    )

    restored = ImageCache.from_transport(cache.to_transport)

    assert restored.image_lookup == cache.image_lookup
    assert restored.build_run_urls == {"my-env": "https://console.union.ai/runs/abc123"}


def test_image_cache_build_run_urls_defaults_empty():
    """ImageCache with no build_run_urls defaults to empty dict."""
    cache = ImageCache(image_lookup={"my-env": "registry/my-image:sha256abc"})

    assert cache.build_run_urls == {}


def test_image_cache_old_serialized_form_still_deserializes():
    """An ImageCache serialized before build_run_urls existed can still be loaded."""
    old_cache = ImageCache(image_lookup={"my-env": "registry/my-image:sha256abc"})

    # Simulate an old serialized form that has no build_run_urls key
    restored = ImageCache.from_transport(old_cache.to_transport)

    assert restored.image_lookup == old_cache.image_lookup
    assert restored.build_run_urls == {}
