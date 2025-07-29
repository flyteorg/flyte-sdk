import asyncio

from flyte import Image
from flyte._internal.imagebuild.image_builder import ImageBuildEngine


async def build_auto():
    # Keep in mind depending on the python environment, auto() will be different.
    default_image = Image.from_debian_base()
    await ImageBuildEngine.build(default_image, force=True)


if __name__ == "__main__":
    asyncio.run(build_auto())
