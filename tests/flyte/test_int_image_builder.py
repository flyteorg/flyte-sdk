import pytest

from flyte._image import Image
from flyte._internal.imagebuild.image_builder import ImageBuildEngine


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_build():
    default_image = Image.from_debian_base()
    await ImageBuildEngine.build(default_image, force=True)


# Can't figure out how to run this locally... getting github auth error.
@pytest.mark.skip
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_build_copied():
    default_image = Image.from_debian_base(registry="ghcr.io/flyteorg", name="flyte-example")
    await ImageBuildEngine.build(default_image, force=True)


def test_real_build_copiedfsaf():
    default_image = Image.from_debian_base(registry="ghcr.io/flyteorg", name="flyte-example")
    print(default_image)
