from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from flyte.syncify import syncify

from ._image import Image

if TYPE_CHECKING:
    from flyte import remote
    from flyte._internal.imagebuild.image_builder import RunIdentifierData


@dataclass
class ImageBuild:
    """
    Result of an image build operation.

    Attributes:
        uri: The fully qualified image URI. None if the build was started asynchronously
            and hasn't completed yet.
        remote_run: Live handle to the build run this process launched with the remote
            builder — wait on it or read its URL. None when no build was launched (local
            builder, or the image already existed). For the run's identifier, use build_run.
        build_run: Identifier of the remote build run that built (or, with wait=False, is
            building) the image — the canonical answer to "which run built this image". Set
            both when this process launches a remote build and when the registry existence
            check learns it from the image service on a cache hit. None for locally built
            images and for backends that don't report it.
    """

    uri: str | None
    remote_run: Optional["remote.Run"]
    build_run: Optional["RunIdentifierData"] = None


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
        force: Skip the existence check and force a rebuild. When using the remote builder, this also
            sets overwrite_cache=True on the build run.
        wait: Wait for the build to finish. If wait is False, the function will return immediately and the build will
            run in the background.
    Returns:
        An ImageBuild object containing the image URI and optionally the remote run that kicked off the build.

    Example:
    ```python
    import flyte
    image = flyte.Image("example_image")
    if __name__ == "__main__":
        result = asyncio.run(flyte.build.aio(image))
        print(result.uri)
    ```
    """
    from flyte._internal.imagebuild.image_builder import ImageBuildEngine

    return await ImageBuildEngine.build(image, dry_run=dry_run, force=force, wait=wait)
