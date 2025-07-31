import argparse
import asyncio

from flyte import Image
from flyte._internal.imagebuild.image_builder import ImageBuildEngine


async def build_auto(registry: str | None = None, name: str | None = None):
    """
    Build the SDK default Debian-based image, optionally overriding
    the container registry and image name.

    Args:
        registry: e.g. "ghcr.io/my-org" or "123456789012.dkr.ecr.us-west-2.amazonaws.com".
        name:     e.g. "my-flyte-image".
    """
    # can remove this and only specify one in the future
    assert (registry and name) or (not registry and not name)
    default_image = Image.from_debian_base(registry=registry, name=name)

    await ImageBuildEngine.build(default_image, force=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the default Flyte image.")
    parser.add_argument("--registry", help="Docker registry to push to")
    parser.add_argument("--name", help="Custom image name (without tag)")

    args = parser.parse_args()
    asyncio.run(build_auto(registry=args.registry, name=args.name))
