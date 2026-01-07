import argparse
import asyncio

import flyte
from flyte.extend import ImageBuildEngine


async def build_flyte_vllm_image(registry: str | None = None, name: str | None = None, builder: str | None = "local"):
    """
    Build the SDK default Debian-based image, optionally overriding
    the container registry name.

    Args:
        registry: e.g. "ghcr.io/my-org" or "123456789012.dkr.ecr.us-west-2.amazonaws.com".
        builder:  e.g. "local" or "remote".
    """
    import flyteplugins.vllm

    default_image = flyteplugins.vllm.DEFAULT_VLLM_IMAGE.clone(registry=registry, name=name)
    await ImageBuildEngine.build(default_image, builder=builder)


async def build_flyte_sglang_image(registry: str | None = None, name: str | None = None, builder: str | None = "local"):
    """
    Build the SDK default Debian-based image, optionally overriding
    the container registry name.
    """
    import flyteplugins.sglang

    default_image = flyteplugins.sglang.DEFAULT_SGLANG_IMAGE.clone(registry=registry, name=name)
    await ImageBuildEngine.build(default_image, builder=builder)


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="Build the default Flyte image.")
    parser.add_argument("--registry", help="Docker registry to push to")
    parser.add_argument("--type", choices=["vllm", "sglang"], help="Type of image to build")
    parser.add_argument("--builder", choices=["local", "remote"], default="local", help="Image builder to use")

    args = parser.parse_args()

    if os.getenv("GITHUB_ACTIONS", "") == "true":
        flyte.init(
            endpoint=os.getenv("FLYTE_ENDPOINT", "dns:///playground.canary.unionai.cloud"),
            auth_type="ClientSecret",
            client_id="flyte-sdk-ci",
            client_credentials_secret=os.getenv("FLYTE_SDK_CI_TOKEN"),
            insecure=False,
            image_builder="remote",
            project=os.getenv("FLYTE_PROJECT", "flyte-sdk"),
            domain=os.getenv("FLYTE_DOMAIN", "development"),
        )
        builder = "remote"
    else:
        flyte.init_from_config()
        builder = args.builder

    if args.type == "vllm":
        print("Building vllm image...")
        asyncio.run(build_flyte_vllm_image(registry=args.registry, builder=builder))
    elif args.type == "sglang":
        print("Building sglang image...")
        asyncio.run(build_flyte_sglang_image(registry=args.registry, builder=builder))
    else:
        raise ValueError(f"Invalid type: {args.type}. Valid types are: [vllm, sglang].")
