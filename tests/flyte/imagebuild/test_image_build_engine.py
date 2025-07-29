import mock
import pytest

from flyte._image import Image
from flyte._internal.imagebuild.image_builder import (
    DockerAPIImageChecker,
    ImageBuildEngine,
    LocalDockerCommandImageChecker,
)


@mock.patch("flyte._internal.imagebuild.image_builder.DockerAPIImageChecker.image_exists")
@mock.patch("flyte._internal.imagebuild.image_builder.LocalDockerCommandImageChecker.image_exists")
@pytest.mark.asyncio
async def test_cached(mock_checker_cli, mock_checker_api):
    # Simulate that the image exists locally
    mock_checker_cli.return_value = True

    img = Image.from_debian_base()
    await ImageBuildEngine.image_exists(img)
    await ImageBuildEngine.image_exists(img)

    # The local checker should be called once, and its result cached
    mock_checker_cli.assert_called_once()
    # The API checker should not be called at all
    mock_checker_api.assert_not_called()


@pytest.mark.skip("/home/runner/work/unionv2/flyte/dist does not exist")
@mock.patch("flyte._internal.imagebuild.image_builder.ImageBuildEngine._get_builder")
@mock.patch("flyte._internal.imagebuild.image_builder.ImageBuildEngine.image_exists", new_callable=mock.AsyncMock)
@pytest.mark.asyncio
async def test_cached_2(mock_image_exists, mock_get_builder):
    mock_image_exists.return_value = False
    mock_builder = mock.AsyncMock()
    mock_builder.build_image.return_value = "docker.io/test-image:v1.0"
    mock_get_builder.return_value = mock_builder

    img = Image.from_debian_base()
    await ImageBuildEngine.build(image=img)
    await ImageBuildEngine.build(image=img)
    mock_builder.build_image.assert_called_once()


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
