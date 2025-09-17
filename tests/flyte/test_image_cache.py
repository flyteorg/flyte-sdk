from flyte._image import Image
from flyte._internal.imagebuild.image_builder import ImageCache


def test_image_cache_serialization_round_trip():
    original_data = {
        "image_lookup": {
            "auto": {"3.10": Image.from_debian_base().uri},
            "abcdev": {"3.10": "cr.flyte.org/img2:latest"},
        }
    }

    # Create the original ImageCache object
    cache = ImageCache(**original_data)

    # Serialize to transport format
    serialized = cache.to_transport

    # Deserialize back into an ImageCache object
    # This should also save the serialized form into the object for downstream tasks to get it.
    restored_cache = ImageCache.from_transport(serialized)

    # Check that the deserialized data matches the original
    assert restored_cache.image_lookup == original_data["image_lookup"]
    assert restored_cache.image_lookup["abcdev"]["3.10"] == "cr.flyte.org/img2:latest"
    assert restored_cache.serialized_form


def test_image_cache_deserialize():
    test_data = ImageCache(
        image_lookup={
            "auto": {"3.12": "registry.example.com/auto:latest"},
            "test_id": {"3.11": "registry.example.com/test:latest"},
        }
    )
    serialized = test_data.to_transport

    restored = ImageCache.from_transport(serialized)
    assert "auto" in restored.image_lookup
    assert isinstance(restored.image_lookup["auto"], dict)
    assert "3.12" in restored.image_lookup["auto"]
