import argparse
import asyncio
from pathlib import Path

import flyte
from flyte import Image
from flyte.extend import ImageBuildEngine


async def build_flyte_image(registry: str | None = None, name: str | None = None, builder: str | None = "local"):
    """
    Build the SDK default Debian-based image, optionally overriding
    the container registry and image name.

    Args:
        registry: e.g. "ghcr.io/my-org" or "123456789012.dkr.ecr.us-west-2.amazonaws.com".
        name:     e.g. "my-flyte-image".
        builder:  e.g. "local" or "remote".
    """
    default_image = Image.from_debian_base(registry=registry, name=name)
    await ImageBuildEngine.build(default_image, builder=builder)


async def build_flyte_connector_image(
    registry: str | None = None, name: str | None = None, builder: str | None = "local"
):
    """
    Build the SDK default connector image, optionally overriding
    the container registry and image name.

    Args:
        registry: e.g. "ghcr.io/my-org" or "123456789012.dkr.ecr.us-west-2.amazonaws.com".
        name:     e.g. "my-connector".
        builder:  e.g. "local" or "remote".
    """
    from flyte._image import _detect_python_version
    from flyte._version import __version__

    if name is None:
        name = "flyte-connectors"

    if "dev" in __version__:
        default_image = (
            Image.from_debian_base(registry=registry, name=name)
            .with_env_vars({"SETUPTOOLS_SCM_PRETEND_VERSION": "9.9.9"})
            .with_uv_project(
                pyproject_file=(Path(__file__).parent.parent / "plugins/connectors/pyproject.toml"),
                pre=True,
                extra_args="--all-extras",
                project_install_mode="install_project",
            )
            .with_local_v2()
        )
    else:
        default_image = Image.from_debian_base(registry=registry, name=name).with_pip_packages(
            "flyteplugins-connectors", pre=True, extra_args="--all-extras"
        )
    suffix = __version__.replace("+", "-")
    python_version = _detect_python_version()
    tag = f"py{python_version[0]}.{python_version[1]}-{suffix}"
    object.__setattr__(default_image, "_tag", tag)
    await ImageBuildEngine.build(default_image, builder=builder)


async def build_all(registry: str | None = None, name: str | None = None, builder: str | None = "local"):
    await asyncio.gather(
        build_flyte_image(registry=registry, name=name, builder=builder),
        build_flyte_connector_image(registry=registry, name=name, builder=builder),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the default Flyte image.")
    parser.add_argument("--registry", help="Docker registry to push to")
    parser.add_argument("--name", help="Custom image name (without tag)")
    parser.add_argument("--type", choices=["flyte", "connector", "all"], help="Type of image to build")
    parser.add_argument("--builder", choices=["local", "remote"], default="local", help="Image builder to use")

    args = parser.parse_args()
    # can remove this and only specify one in the future
    assert (args.registry and args.name) or (not args.registry and not args.name)

    flyte.init_from_config()
    if args.type == "flyte":
        print("Building flyte image...")
        asyncio.run(build_flyte_image(registry=args.registry, name=args.name, builder=args.builder))
    elif args.type == "connector":
        print("Building connector image...")
        asyncio.run(build_flyte_connector_image(registry=args.registry, name=args.name, builder=args.builder))
    else:
        print("Building all images...")
        asyncio.run(build_all(registry=args.registry, name=args.name, builder=args.builder))
