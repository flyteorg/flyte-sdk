from __future__ import annotations

import asyncio
import json
import re
import typing
from importlib.metadata import entry_points
from typing import ClassVar, Dict, Optional, Tuple

from async_lru import alru_cache
from pydantic import BaseModel
from typing_extensions import Protocol

from flyte._image import Architecture, Image, Layer, PipPackages, PythonWheels, UVScript
from flyte._initialize import _get_init_config
from flyte._logging import logger

from .heavy_deps import HEAVY_DEPENDENCIES


class ImageBuilder(Protocol):
    async def build_image(self, image: Image, dry_run: bool) -> str: ...

    def get_checkers(self) -> Optional[typing.List[typing.Type[ImageChecker]]]:
        """
        Returns ImageCheckers that can be used to check if the image exists in the registry.
        If None, then use the default checkers.
        """
        return None


class ImageChecker(Protocol):
    @classmethod
    async def image_exists(
        cls, repository: str, tag: str, arch: Tuple[Architecture, ...] = ("linux/amd64",)
    ) -> Optional[str]: ...


class DockerAPIImageChecker(ImageChecker):
    """
    Unfortunately only works for docker hub as there's no way to get a public token for ghcr.io. See SO:
    https://stackoverflow.com/questions/57316115/get-manifest-of-a-public-docker-image-hosted-on-docker-hub-using-the-docker-regi
    The token used here seems to be short-lived (<1 second), so copy pasting doesn't even work.
    """

    @classmethod
    async def image_exists(
        cls, repository: str, tag: str, arch: Tuple[Architecture, ...] = ("linux/amd64",)
    ) -> Optional[str]:
        import httpx

        if "/" not in repository:
            repository = f"library/{repository}"

        auth_url = "https://auth.docker.io/token"
        service = "registry.docker.io"
        scope = f"repository:{repository}:pull"

        async with httpx.AsyncClient() as client:
            # Get auth token
            auth_response = await client.get(auth_url, params={"service": service, "scope": scope})
            if auth_response.status_code != 200:
                raise Exception(f"Failed to get auth token: {auth_response.status_code}")

            token = auth_response.json()["token"]

            # ghcr.io/union-oss/flyte:latest
            manifest_url = f"https://registry-1.docker.io/v2/{repository}/manifests/{tag}"
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": (
                    "application/vnd.docker.distribution.manifest.v2+json,"
                    "application/vnd.docker.distribution.manifest.list.v2+json"
                ),
            }

            manifest_response = await client.get(manifest_url, headers=headers)
            if manifest_response.status_code != 200:
                logger.warning(f"Image not found: {repository}:{tag} (HTTP {manifest_response.status_code})")
                return None

            manifest_list = manifest_response.json()["manifests"]
            architectures = [f"{m['platform']['os']}/{m['platform']['architecture']}" for m in manifest_list]

            if set(arch).issubset(set(architectures)):
                logger.debug(f"Image {repository}:{tag} found with arch {architectures}")
                return f"{repository}:{tag}"
            else:
                logger.debug(f"Image {repository}:{tag} has {architectures}, but missing {arch}")
                return None


class LocalDockerCommandImageChecker(ImageChecker):
    command_name: ClassVar[str] = "docker"

    @classmethod
    async def image_exists(
        cls, repository: str, tag: str, arch: Tuple[Architecture, ...] = ("linux/amd64",)
    ) -> Optional[str]:
        # Check if the image exists locally by running the docker inspect command
        process = await asyncio.create_subprocess_exec(
            cls.command_name,
            "manifest",
            "inspect",
            f"{repository}:{tag}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if (stderr and "manifest unknown") or "no such manifest" in stderr.decode():
            logger.debug(f"Image {repository}:{tag} not found using the docker command.")
            return None

        if process.returncode != 0:
            raise RuntimeError(f"Failed to run docker image inspect {repository}:{tag}")

        inspect_data = json.loads(stdout.decode())
        if "manifests" not in inspect_data:
            raise RuntimeError(f"Invalid data returned from docker image inspect for {repository}:{tag}")
        manifest_list = inspect_data["manifests"]
        architectures = [f"{x['platform']['os']}/{x['platform']['architecture']}" for x in manifest_list]
        if set(architectures) >= set(arch):
            logger.debug(f"Image {repository}:{tag} found for architecture(s) {arch}, has {architectures}")
            return f"{repository}:{tag}"

        # Otherwise write a message and return false to trigger build
        logger.debug(f"Image {repository}:{tag} not found for architecture(s) {arch}, only has {architectures}")
        return None


class LocalPodmanCommandImageChecker(LocalDockerCommandImageChecker):
    command_name: ClassVar[str] = "podman"


class ImageBuildEngine:
    """
    ImageBuildEngine contains a list of builders that can be used to build an ImageSpec.
    """

    ImageBuilderType = typing.Literal["local", "remote"]

    @staticmethod
    def _optimize_image_layers(image: Image) -> Image:
        """
        Optimize pip layers by extracting heavy dependencies to the top for better caching.
        Heavy packages from each layer are extracted to separate layers (preserving per-layer arguments),
        and all heavy layers are placed at the top.  Original layers with light packages follow.
        PythonWheels layers with package_name 'flyte' are moved to the very end.
        """
        from flyte._utils import parse_uv_script_file

        # Step 1: Collect heavy and original layers separately
        heavy_layers: list[PipPackages] = []
        original_layers: list[Layer] = []
        flyte_wheel_layers: list[PythonWheels] = []

        for layer in image._layers:
            if isinstance(layer, PipPackages):
                assert layer.packages is not None

                heavy_pkgs: list[str] = []
                light_pkgs: list[str] = []

                # Split packages
                for pkg in layer.packages:
                    pkg_name = re.split(r"[<>=~!\[]", pkg, 1)[0].strip()
                    if pkg_name in HEAVY_DEPENDENCIES:
                        heavy_pkgs.append(pkg)
                    else:
                        light_pkgs.append(pkg)

                # Create heavy layer with original arguments (if any heavy packages)
                if heavy_pkgs:
                    heavy_layer = PipPackages(
                        packages=tuple(heavy_pkgs),
                        index_url=layer.index_url,
                        extra_index_urls=layer.extra_index_urls,
                        pre=layer.pre,
                        extra_args=layer.extra_args,
                        secret_mounts=layer.secret_mounts,
                    )
                    heavy_layers.append(heavy_layer)
                    logger.debug(f"Extracted {len(heavy_pkgs)} heavy package(s): {', '.join(heavy_pkgs)}")

                # Create light layer with original arguments (if any light packages)
                if light_pkgs:
                    light_layer = PipPackages(
                        packages=tuple(light_pkgs),
                        index_url=layer.index_url,
                        extra_index_urls=layer.extra_index_urls,
                        pre=layer.pre,
                        extra_args=layer.extra_args,
                        secret_mounts=layer.secret_mounts,
                    )
                    original_layers.append(light_layer)

            elif isinstance(layer, UVScript):
                # Parse UV scripts and extract dependencies
                metadata = parse_uv_script_file(layer.script)

                if metadata.dependencies:
                    uv_heavy_pkgs: list[str] = []
                    uv_light_pkgs: list[str] = []

                    for pkg in metadata.dependencies:
                        pkg_name = re.split(r"[<>=~!\[]", pkg, 1)[0].strip()
                        if pkg_name in HEAVY_DEPENDENCIES:
                            uv_heavy_pkgs.append(pkg)
                        else:
                            uv_light_pkgs.append(pkg)

                    # Create heavy pip layer from UV (if any heavy packages)
                    if uv_heavy_pkgs:
                        heavy_pip_layer = PipPackages(
                            packages=tuple(uv_heavy_pkgs),
                            index_url=layer.index_url,
                            extra_index_urls=layer.extra_index_urls,
                            pre=layer.pre,
                            extra_args=layer.extra_args,
                            secret_mounts=layer.secret_mounts,
                        )
                        heavy_layers.append(heavy_pip_layer)
                        logger.debug(
                            f"Extracted {len(uv_heavy_pkgs)} heavy package(s) from UV:  {', '.join(uv_heavy_pkgs)}"
                        )

                    # Create light pip layer from UV (if any light packages)
                    if uv_light_pkgs:
                        light_pip_layer = PipPackages(
                            packages=tuple(uv_light_pkgs),
                            index_url=layer.index_url,
                            extra_index_urls=layer.extra_index_urls,
                            pre=layer.pre,
                            extra_args=layer.extra_args,
                            secret_mounts=layer.secret_mounts,
                        )
                        original_layers.append(light_pip_layer)

                # Keep the UVScript layer in original_layers section
                original_layers.append(layer)

            elif isinstance(layer, PythonWheels):
                # Check if this is a flyte wheel - if so, move to end
                if layer.package_name == "flyte":
                    flyte_wheel_layers.append(layer)
                    logger.debug(f"Moving flyte wheel layer to end: {layer}")
                else:
                    # Keep other wheels with original_layers
                    original_layers.append(layer)

            else:
                # All other layers (apt, env, etc.) go with light layers
                original_layers.append(layer)

        # If no heavy packages found, return original image
        if not heavy_layers:
            logger.debug("No heavy packages found, skipping optimization")
            return image

        logger.info(f"Created {len(heavy_layers)} heavy layer(s) at top")

        # Final layer order: all heavy layers at top, then original layers, then flyte wheels at end
        final_layers = [*heavy_layers, *original_layers, *flyte_wheel_layers]

        if flyte_wheel_layers:
            logger.debug(f"Moved {len(flyte_wheel_layers)} flyte wheel layer(s) to end")

        return Image._new(
            base_image=image.base_image,
            dockerfile=image.dockerfile,
            registry=image.registry,
            name=image.name,
            platform=image.platform,
            python_version=image.python_version,
            _layers=tuple(final_layers),
            _image_registry_secret=image._image_registry_secret,
            _ref_name=image._ref_name,
        )

    @staticmethod
    @alru_cache
    async def image_exists(image: Image) -> Optional[str]:
        if image.base_image is not None and not image._layers:
            logger.debug(f"Image {image} has a base image: {image.base_image} and no layers. Skip existence check.")
            return image.uri
        assert image.name is not None, f"Image name is not set for {image}"

        tag = image._final_tag

        if tag == "latest":
            logger.debug(f"Image {image} has tag 'latest', skip existence check, always build")
            return image.uri

        builder = None
        cfg = _get_init_config()
        if cfg and cfg.image_builder:
            builder = cfg.image_builder
        image_builder = ImageBuildEngine._get_builder(builder)
        image_checker = image_builder.get_checkers()
        if image_checker is None:
            logger.info(f"No image checkers found for builder `{image_builder}`, assuming it exists")
            return image.uri
        for checker in image_checker:
            try:
                repository = image.registry + "/" + image.name if image.registry else image.name
                image_uri = await checker.image_exists(repository, tag, tuple(image.platform))
                if image_uri:
                    logger.debug(f"Image {image_uri} in registry")
                return image_uri
            except Exception as e:
                logger.debug(f"Error checking image existence with {checker.__name__}: {e}")
                continue

        # If all checkers fail, then assume the image exists. This is current flytekit behavior
        logger.info(f"All checkers failed to check existence of {image.uri}, assuming it does exists")
        return image.uri

    @classmethod
    @alru_cache
    async def build(
        cls,
        image: Image,
        builder: ImageBuildEngine.ImageBuilderType | None = None,
        dry_run: bool = False,
        force: bool = False,
        optimize_layers: bool = True,
    ) -> str:
        """
        Build the image. Images to be tagged with latest will always be built. Otherwise, this engine will check the
        registry to see if the manifest exists.

        :param image:
        :param builder:
        :param dry_run: Tell the builder to not actually build. Different builders will have different behaviors.
        :param force: Skip the existence check. Normally if the image already exists we won't build it.
        :param optimize_layers: If True, consolidate pip packages by category (default: True)
        :return:
        """

        # Always trigger a build if this is a dry run since builder shouldn't really do anything, or a force.
        image_uri = (await cls.image_exists(image)) or image.uri
        if force or dry_run or not await cls.image_exists(image):
            logger.info(f"Image {image_uri} does not exist in registry or force/dry-run, building...")

            # Validate the image before building
            image.validate()

            if optimize_layers:
                logger.debug("Optimizing image layers by consolidating pip packages...")
                image = ImageBuildEngine._optimize_image_layers(image)  # Call the optimizer
                logger.debug("=" * 60)
                logger.debug("Final layer order after optimization:")
                for i, layer in enumerate(image._layers):
                    logger.debug(f"  Layer {i}: {type(layer).__name__} - {layer}")
                logger.debug("=" * 60)

            # If a builder is not specified, use the first registered builder
            cfg = _get_init_config()
            if cfg and cfg.image_builder:
                builder = builder or cfg.image_builder
            img_builder = ImageBuildEngine._get_builder(builder)
            logger.debug(f"Using `{img_builder}` image builder to build image.")

            result = await img_builder.build_image(image, dry_run=dry_run)
            return result
        else:
            logger.info(f"Image {image_uri} already exists in registry. Skipping build.")
            return image_uri

    @classmethod
    def _get_builder(cls, builder: ImageBuildEngine.ImageBuilderType | None = "local") -> ImageBuilder:
        if builder is None:
            builder = "local"
        if builder == "remote":
            from flyte._internal.imagebuild.remote_builder import RemoteImageBuilder

            return RemoteImageBuilder()
        elif builder == "local":
            from flyte._internal.imagebuild.docker_builder import DockerImageBuilder

            return DockerImageBuilder()
        else:
            return cls._load_custom_type_transformers(builder)

    @classmethod
    def _load_custom_type_transformers(cls, name: str) -> ImageBuilder:
        plugins = entry_points(group="flyte.plugins.image_builders")
        for ep in plugins:
            if ep.name != name:
                continue
            try:
                logger.info(f"Loading image builder: {ep.name}")
                builder = ep.load()
                if callable(builder):
                    return builder()
                return builder
            except Exception as e:
                raise RuntimeError(f"Failed to load image builder {ep.name} with error: {e}")
        raise ValueError(
            f"Unknown image builder type: {name}. Available builders:"
            f" {[ep.name for ep in plugins] + ['local', 'remote']}"
        )


class ImageCache(BaseModel):
    image_lookup: Dict[str, str]
    serialized_form: str | None = None

    @property
    def to_transport(self) -> str:
        """
        :return: returns the serialization context as a base64encoded, gzip compressed, json string
        """
        # This is so that downstream tasks continue to have the same image lookup abilities
        import base64
        import gzip
        from io import BytesIO

        if self.serialized_form:
            return self.serialized_form
        json_str = self.model_dump_json(exclude={"serialized_form"})
        buf = BytesIO()
        with gzip.GzipFile(mode="wb", fileobj=buf, mtime=0) as f:
            f.write(json_str.encode("utf-8"))
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @classmethod
    def from_transport(cls, s: str) -> ImageCache:
        import base64
        import gzip

        compressed_val = base64.b64decode(s.encode("utf-8"))
        json_str = gzip.decompress(compressed_val).decode("utf-8")
        val = cls.model_validate_json(json_str)
        val.serialized_form = s
        return val

    def repr(self) -> typing.List[typing.List[Tuple[str, str]]]:
        """
        Returns a detailed representation of the deployed environments.
        """
        tuples = []
        for k, v in self.image_lookup.items():
            tuples.append(
                [
                    ("Name", k),
                    ("image", v),
                ]
            )
        return tuples
