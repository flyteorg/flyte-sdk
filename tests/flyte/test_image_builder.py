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
