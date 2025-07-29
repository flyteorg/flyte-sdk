from pathlib import Path

import pytest

from flyte._image import Image
from flyte._internal.imagebuild.docker_builder import DockerImageBuilder


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basic_image():
    img = Image.from_uv_debian("localhost:30000", "test_image").with_pip_packages("requests")

    builder = DockerImageBuilder()

    await builder.build_image(img, dry_run=False)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_image_folders_commands():
    img = (
        Image.from_uv_debian("localhost:30000", "img_with_more")
        .with_pip_packages("requests")
        .with_source_folder(Path("."), "/root/data/stuff")
        .with_source_folder(Path("/Users/ytong/temp/fasts/yt_dbg/scratchpad"), "/root/data/stuff")
        .with_commands(["echo hello world", "echo hello world again"])
        .with_source_file(Path("/Users/ytong/c"))
    )

    builder = DockerImageBuilder()
    await builder.build_image(img, dry_run=False)


@pytest.mark.skip("TemporaryDirectory.__init__() got an unexpected keyword argument 'delete")
@pytest.mark.asyncio
async def test_doesnt_work_yet():
    default_image = Image.from_debian_base()
    builder = DockerImageBuilder()
    await builder.build_image(default_image, dry_run=False)
