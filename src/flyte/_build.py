from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from flyte.syncify import syncify

from ._image import Image

if TYPE_CHECKING:
    from flyte import remote


@dataclass
class ImageBuild:
    """
    Result of an image build operation.

    Attributes:
        uri: The fully qualified image URI. None if the build was started asynchronously
            and hasn't completed yet.
        remote_run: The Run object that kicked off an image build job when using the remote
            builder. None when using the local builder.
    """

    uri: str | None
    remote_run: Optional["remote.Run"]


@syncify
async def build(
    image: Image,
    dry_run: bool = False,
    force: bool = False,
    wait: bool = True,
) -> ImageBuild:
    """
    Build an image. The existing async context will be used.

    Args:
        image: The image(s) to build.
        dry_run: Tell the builder to not actually build. Different builders will have different behaviors.
        force: Skip the existence check. Normally if the image already exists we won't build it.
        wait: Wait for the build to finish. If wait is False, the function will return immediately and the build will
            run in the background.
    Returns:
        An ImageBuild object containing the image URI and optionally the remote run that kicked off the build.

    Example:
    ```
    import flyte
    image = flyte.Image("example_image")
    if __name__ == "__main__":
        result = asyncio.run(flyte.build.aio(image))
        print(result.uri)
    ```

    :param image: The image(s) to build.
    :param dry_run: Tell the builder to not actually build. Different builders will have different behaviors.
    :param force: Skip the existence check. Normally if the image already exists we won't build it.
    :param wait: Wait for the build to finish. If wait is False, the function will return immediately and the build will
        run in the background.
    :return: An ImageBuild object with the image URI and remote run (if applicable).
    """
    from flyte._internal.imagebuild.image_builder import ImageBuildEngine

    return await ImageBuildEngine.build(image, dry_run=dry_run, force=force, wait=wait)
