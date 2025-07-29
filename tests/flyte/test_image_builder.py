import pytest

from flyte._image import Image
from flyte._internal.imagebuild.image_builder import ImageBuildEngine


@pytest.mark.asyncio
async def test_exists():
    """
    Test the ImageBuilder class.
    """

    ubuntu = Image.from_base("ghcr.io/unionai-oss:ubuntu")
    assert await ImageBuildEngine.image_exists(ubuntu) == "ghcr.io/unionai-oss:ubuntu"
