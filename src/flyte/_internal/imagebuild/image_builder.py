from __future__ import annotations

import asyncio
import hashlib
import json
import random
import sqlite3
import time
import typing
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, ClassVar, Dict, Optional, Tuple

from async_lru import alru_cache
from pydantic import BaseModel
from typing_extensions import Protocol

from flyte._image import Architecture, Image
from flyte._initialize import _get_init_config
from flyte._logging import logger
from flyte._persistence._db import LocalDB
from flyte._status import status

_IMAGE_CACHE_TTL_DAYS = 30

if TYPE_CHECKING:
    from flyte._build import ImageBuild


class ImageBuilder(Protocol):
    async def build_image(
        self, image: Image, dry_run: bool, wait: bool = True, force: bool = False
    ) -> "ImageBuild": ...

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
    ) -> Optional[str]:
        """
        Check whether an image exists in a registry or cache.

        Returns the image URI if found, or None if the image definitively does not exist.
        Raise an exception if existence cannot be determined (e.g. cache miss, network failure)
        so the next checker in the chain gets a chance.
        """
        ...


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


def _cache_key(repository: str, tag: str, arch: Tuple[str, ...]) -> str:
    """Return a stable cache key for an image, scoped to the current endpoint/project/domain."""
    from flyte._persistence._db import _cache_scope

    raw = f"{_cache_scope()}:{repository}:{tag}:{','.join(sorted(arch))}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _read_image_cache(repository: str, tag: str, arch: Tuple[str, ...]) -> Optional[str]:
    """Look up a previously verified image URI by repository, tag, and arch. Returns image_uri or None."""
    try:
        conn = LocalDB.get_sync()
        cutoff = time.time() - _IMAGE_CACHE_TTL_DAYS * 86400
        row = conn.execute(
            "SELECT image_uri FROM image_cache WHERE key = ? AND created_at > ?",
            (_cache_key(repository, tag, arch), cutoff),
        ).fetchone()
        # Prune expired entries ~5% of the time to avoid doing it on every read
        if random.random() < 0.05:
            with LocalDB._write_lock:
                conn.execute("DELETE FROM image_cache WHERE created_at <= ?", (cutoff,))
                conn.commit()
        if row:
            return row[0]
    except (OSError, sqlite3.Error) as e:
        logger.debug(f"Failed to read image cache: {e}")
    return None


def _write_image_cache(repository: str, tag: str, arch: Tuple[str, ...], image_uri: str) -> None:
    """Persist a verified image URI to the SQLite cache."""
    try:
        conn = LocalDB.get_sync()
        with LocalDB._write_lock:
            conn.execute(
                "INSERT OR REPLACE INTO image_cache (key, image_uri, created_at) VALUES (?, ?, ?)",
                (_cache_key(repository, tag, arch), image_uri, time.time()),
            )
            conn.commit()
    except (OSError, sqlite3.Error) as e:
        logger.debug(f"Failed to write image cache: {e}")


class PersistentCacheImageChecker(ImageChecker):
    """Check if image was previously verified and cached in SQLite (~0ms)."""

    @classmethod
    async def image_exists(
        cls, repository: str, tag: str, arch: Tuple[Architecture, ...] = ("linux/amd64",)
    ) -> Optional[str]:
        uri = _read_image_cache(repository, tag, arch)
        if uri:
            logger.debug(f"Image {uri} found in persistent cache")
            return uri
        # Cache miss — raise so the next checker in the chain gets a chance.
        # Returning None would mean "image definitely doesn't exist".
        raise LookupError(f"Image {repository}:{tag} not found in persistent cache")


class LocalDockerCommandImageChecker(ImageChecker):
    command_name: ClassVar[str] = "docker"

    @classmethod
    async def image_exists(
        cls, repository: str, tag: str, arch: Tuple[Architecture, ...] = ("linux/amd64",)
    ) -> Optional[str]:
        # Use buildx imagetools inspect which works with both OCI and Docker manifest formats,
        # including local/insecure registries that docker manifest inspect doesn't handle well.
        process = await asyncio.create_subprocess_exec(
            cls.command_name,
            "buildx",
            "imagetools",
            "inspect",
            f"{repository}:{tag}",
            "--raw",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            stderr_str = stderr.decode() if stderr else ""
            if "manifest unknown" in stderr_str or "no such manifest" in stderr_str or "not found" in stderr_str:
                logger.debug(f"Image {repository}:{tag} not found using the docker command.")
                return None
            raise RuntimeError(f"Failed to run docker buildx imagetools inspect {repository}:{tag}: {stderr_str}")

        inspect_data = json.loads(stdout.decode())
        if "manifests" not in inspect_data:
            # Single-platform image without a manifest list — image exists
            logger.debug(f"Image {repository}:{tag} found (single-platform manifest)")
            return f"{repository}:{tag}"
        manifest_list = [m for m in inspect_data["manifests"] if "platform" in m and "os" in m.get("platform", {})]
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
    @alru_cache
    async def image_exists(image: Image) -> Optional[str]:
        if not image._is_cloned:
            # Unmodified flyte default images are built and pushed as part of flyte releases,
            # so they are guaranteed to exist — skip the existence check.
            return image.uri
        if image.name is None:
            logger.debug(f"Image {image} has no name. Skip existence check.")
            return image.uri

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
            status.info(f"No image checkers found for builder `{image_builder}`, assuming it exists")
            return image.uri
        for checker in image_checker:
            try:
                repository = image.registry + "/" + image.name if image.registry else image.name
                image_uri = await checker.image_exists(repository, tag, tuple(image.platform))
                if image_uri:
                    logger.debug(f"Image {image_uri} in registry")
                    # Persist to disk so future process invocations skip network checks
                    if checker is not PersistentCacheImageChecker:
                        _write_image_cache(repository, tag, tuple(image.platform), image_uri)
                    return image_uri
                # Checker ran successfully and returned None — image not found
                return None
            except Exception as e:
                logger.debug(f"Error checking image existence with {checker.__name__}: {e}")
                continue

        # All checkers raised exceptions (e.g. network failures) — assume image exists
        status.info(f"All checkers failed to check existence of {image.uri}, assuming it exists")
        return image.uri

    @classmethod
    @alru_cache
    async def build(
        cls,
        image: Image,
        builder: ImageBuildEngine.ImageBuilderType | None = None,
        dry_run: bool = False,
        force: bool = False,
        wait: bool = True,
    ) -> "ImageBuild":
        """
        Build the image. Images to be tagged with latest will always be built. Otherwise, this engine will check the
        registry to see if the manifest exists.

        :param image:
        :param builder:
        :param dry_run: Tell the builder to not actually build. Different builders will have different behaviors.
        :param force: Skip the existence check and force a rebuild. When using the remote builder, this
            also sets overwrite_cache=True on the build run.
        :param wait: Wait for the build to finish. If wait is False when using the remote image builder, the function
            will return the build image task URL.
        :return: An ImageBuild object with the image URI and remote run (if applicable).
        """
        from flyte._build import ImageBuild

        # Skip the existence check when force or dry_run is set.
        image_uri: str | None
        if force or dry_run:
            image_uri = image.uri
            status.step(f"Building image {image_uri}...")
        elif image_uri := await cls.image_exists(image):
            status.info(f"Image {image_uri} already exists, skipping build")
            return ImageBuild(uri=image_uri, remote_run=None)
        else:
            image_uri = image.uri
            status.step(f"Image {image_uri} not found, building...")

        # Validate the image before building
        image.validate()

        # If a builder is not specified, use the first registered builder
        cfg = _get_init_config()
        if cfg and cfg.image_builder:
            builder = builder or cfg.image_builder
        img_builder = ImageBuildEngine._get_builder(builder)
        logger.debug(f"Using `{img_builder}` image builder to build image.")

        result = await img_builder.build_image(image, dry_run=dry_run, wait=wait, force=force)
        return result

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
            return cls._load_custom_image_builders(builder)

    @classmethod
    def _load_custom_image_builders(cls, name: str) -> ImageBuilder:
        plugins = entry_points(group="flyte.plugins.image_builders")
        for ep in plugins:
            if ep.name != name:
                continue
            try:
                status.info(f"Loading image builder: {ep.name}")
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


class RunIdentifierData(BaseModel):
    org: str
    project: str
    domain: str
    name: str


class ImageCache(BaseModel):
    image_lookup: Dict[str, str]
    build_run_ids: Dict[str, RunIdentifierData] = {}
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
