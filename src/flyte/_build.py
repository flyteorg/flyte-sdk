from __future__ import annotations

from flyte.syncify import syncify

from ._image import Image


@syncify
async def build(
    image: Image,
    dry_run: bool = False,
    force: bool = False,
    wait: bool = True,
) -> str:
    """
    Build an image. The existing async context will be used.

    Args:
        image: The image(s) to build.
        dry_run: Tell the builder to not actually build. Different builders will have different behaviors.
        force: Skip the existence check. Normally if the image already exists we won't build it.
        wait: Wait for the build to finish. If wait is False, the function will return immediately and the build will
            run in the background.
    Returns:
        The image URI. If wait is False when using the remote image builder, the function will return the build image task URL.

    Example:
    ```
    import flyte
    image = flyte.Image("example_image")
    if __name__ == "__main__":
        asyncio.run(flyte.build.aio(image))
    ```

    :param image: The image(s) to build.
    :param dry_run: Tell the builder to not actually build. Different builders will have different behaviors.
    :param force: Skip the existence check. Normally if the image already exists we won't build it.
    :param wait: Wait for the build to finish. If wait is False, the function will return immediately and the build will
        run in the background.
    :return: The image URI.
    """
    from flyte._internal.imagebuild.image_builder import ImageBuildEngine

    return await ImageBuildEngine.build(image, dry_run=dry_run, force=force, wait=wait)
