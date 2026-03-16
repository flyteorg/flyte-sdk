"""Re-tag and push an existing image to a target registry from CI.

Use this script when you want to take a pre-built image (e.g. the Flyte default
image) and publish it under your own registry, name, and tag — without rebuilding
it from scratch.

For building images defined as :class:`flyte.Environment` objects in Python, use
the ``flyte build`` CLI command instead::

    flyte build my_workflow.py

Usage::

    # Re-tag an existing image and push to ECR
    python examples/image/ci_build_image.py \
        --from ghcr.io/flyteorg/flyte:py3.12-v2.0.0b56 \
        --to 123456789.dkr.ecr.us-west-2.amazonaws.com/myorg/myimage:v1.0.0

    # Use the remote builder
    python examples/image/ci_build_image.py \
        --from ghcr.io/flyteorg/flyte:py3.12-v2.0.0b56 \
        --to 123456789.dkr.ecr.us-west-2.amazonaws.com/myorg/myimage:v1.0.0 \
        --builder remote

    # Force push even if the target already exists
    python examples/image/ci_build_image.py \
        --from ghcr.io/flyteorg/flyte:py3.12-v2.0.0b56 \
        --to 123456789.dkr.ecr.us-west-2.amazonaws.com/myorg/myimage:v1.0.0 \
        --force
"""

import argparse
import asyncio

import flyte
from flyte import Image
from flyte.extend import ImageBuildEngine


def parse_target(target: str) -> tuple[str, str, str]:
    """Parse a target image string into (registry, name, tag).

    Example:
        >>> parse_target("123456789.dkr.ecr.us-west-2.amazonaws.com/myorg/myimage:v1.0.0")
        ('123456789.dkr.ecr.us-west-2.amazonaws.com/myorg', 'myimage', 'v1.0.0')
    """
    if ":" not in target:
        raise ValueError(f"Target '{target}' must contain a tag (e.g., myregistry/myimage:v1.0)")
    image_path, tag = target.rsplit(":", 1)
    if "/" not in image_path:
        raise ValueError(f"Target '{target}' must contain a registry (e.g., myregistry/myimage:v1.0)")
    registry, name = image_path.rsplit("/", 1)
    return registry, name, tag


async def build_and_push(from_image: str, to_target: str, builder: str = "local", force: bool = False) -> str:
    """Re-tag a pre-built image and push it to a target registry/name:tag."""
    registry, name, tag = parse_target(to_target)
    image = Image.from_base(from_image).clone(registry=registry, name=name, tag=tag)
    result = await ImageBuildEngine.build(image, builder=builder, force=force)
    return result.uri


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and push an image to a target registry/name:tag.")
    parser.add_argument("--from", dest="from_image", required=True, help="Source image URI")
    parser.add_argument("--to", dest="to_target", required=True, help="Target image as registry/name:tag")
    parser.add_argument("--builder", choices=["local", "remote"], default="local", help="Image builder to use")
    parser.add_argument("--force", action="store_true", help="Skip existence check, always rebuild")

    args = parser.parse_args()

    flyte.init_from_config()
    uri = asyncio.run(build_and_push(args.from_image, args.to_target, args.builder, args.force))
    print(uri)
