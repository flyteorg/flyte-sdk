import argparse
import asyncio
from tqdm.asyncio import tqdm
import flyte
from flyte.extend import ImageBuildEngine


async def build_flyte_vllm_image(registry: str | None = None, name: str | None = None, builder: str | None = "local"):
    """
    Build the SDK default Debian-based image, optionally overriding
    the container registry and image name.

    Args:
        registry: e.g. "ghcr.io/my-org" or "123456789012.dkr.ecr.us-west-2.amazonaws.com".
        name:     e.g. "my-flyte-image".
        builder:  e.g. "local" or "remote".
    """
    import flyteplugins.vllm

    default_image = flyteplugins.vllm.DEFAULT_VLLM_IMAGE.clone(registry=registry, name=name)
    await ImageBuildEngine.build(default_image, builder=builder)


async def build_flyte_sglang_image(registry: str | None = None, name: str | None = None, builder: str | None = "local"):
    """
    Build the SDK default Debian-based image, optionally overriding
    the container registry and image name.
    """
    import flyteplugins.sglang

    default_image = flyteplugins.sglang.DEFAULT_SGLANG_IMAGE.clone(registry=registry, name=name)
    await ImageBuildEngine.build(default_image, builder=builder)


async def build_all(registry: str | None = None, name: str | None = None, builder: str | None = "local"):
    await tqdm.gather(
        build_flyte_vllm_image(registry=registry, name=name, builder=builder),
        build_flyte_sglang_image(registry=registry, name=name, builder=builder),
        desc="Building plugin images"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the default Flyte image.")
    parser.add_argument("--registry", help="Docker registry to push to")
    parser.add_argument("--name", help="Custom image name (without tag)")
    parser.add_argument("--type", choices=["vllm", "sglang", "all"], help="Type of image to build")
    parser.add_argument("--builder", choices=["local", "remote"], default="local", help="Image builder to use")

    args = parser.parse_args()
    # can remove this and only specify one in the future
    assert (args.registry and args.name) or (not args.registry and not args.name)

    flyte.init_from_config()
    if args.type == "vllm":
        print("Building vllm image...")
        asyncio.run(build_flyte_vllm_image(registry=args.registry, name=args.name, builder=args.builder))
    elif args.type == "sglang":
        print("Building sglang image...")
        asyncio.run(build_flyte_sglang_image(registry=args.registry, name=args.name, builder=args.builder))
    else:
        print("Building all images...")
        asyncio.run(build_all(registry=args.registry, name=args.name, builder=args.builder))
