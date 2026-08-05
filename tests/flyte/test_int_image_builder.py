import pytest

from flyte._build import ImageBuild
from flyte._image import Image
from flyte._internal.imagebuild.image_builder import ImageBuildEngine


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_build():
    default_image = Image.from_debian_base()
    result = await ImageBuildEngine.build(default_image, force=True)
    assert isinstance(result, ImageBuild)
    assert result.uri is not None
    # Local builder doesn't create a remote run
    assert result.remote_run is None


# Can't figure out how to run this locally... getting github auth error.
@pytest.mark.skip
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_build_copied():
    default_image = Image.from_debian_base(registry="ghcr.io/flyteorg", name="flyte-example")
    result = await ImageBuildEngine.build(default_image, force=True)
    assert isinstance(result, ImageBuild)
    assert result.uri is not None


def test_real_build_copiedfsaf():
    default_image = Image.from_debian_base(registry="ghcr.io/flyteorg", name="flyte-example")
    print(default_image)
