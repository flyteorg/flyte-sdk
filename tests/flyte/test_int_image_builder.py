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


# This may fail in CI so marking this special
@pytest.mark.editable
def test_editable():
    from flyte._image import Image

    assert Image._is_editable_install()


def test_real_build_copiedfsaf():
    default_image = Image.from_debian_base(registry="ghcr.io/flyteorg", name="flyte-example")
    print(default_image)
