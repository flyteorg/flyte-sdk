from flyte._image import Image
from flyte._internal.imagebuild.image_builder import ImageCache


def test_image_cache_serialization_round_trip():
    original_data = {
        "image_lookup": {
            "auto": Image.from_debian_base().uri,
            "abcdev": "cr.flyte.org/img2:latest",
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
    assert restored_cache.image_lookup["abcdev"] == "cr.flyte.org/img2:latest"
    assert restored_cache.serialized_form


def test_image_cache_deserialize():
    serialized = (
        "H4sIAAAAAAAC/wXBQQ6EIAwAwL/07kK6miqfMZUWJbp0Y0QPxr87c0P+8azjZrbWP4QbuB4GAeYl7p9s7sqbNFpE+b"
        "Td1ZKtYJDed8qClNKkSPSdfGx1YIyUPArB87zeIf0fWQAAAA=="
    )
    restored = ImageCache.from_transport(serialized)
    assert "auto" in restored.image_lookup
