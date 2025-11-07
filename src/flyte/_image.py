from __future__ import annotations

import hashlib
import sys
import typing
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Dict, List, Literal, Optional, Tuple, TypeVar, Union

import rich.repr
from packaging.version import Version

if TYPE_CHECKING:
    from flyte import Secret, SecretRequest

# Supported Python versions
PYTHON_3_10 = (3, 10)
PYTHON_3_11 = (3, 11)
PYTHON_3_12 = (3, 12)
PYTHON_3_13 = (3, 13)

# 0 is a file, 1 is a directory
CopyConfigType = Literal[0, 1]

T = TypeVar("T")


def _ensure_tuple(val: Union[T, List[T], Tuple[T, ...]]) -> Tuple[T] | Tuple[T, ...]:
    """
    Ensure that the input is a tuple. If it is a string, convert it to a tuple with one element.
    If it is a list, convert it to a tuple.
    """
    if isinstance(val, list):
        return tuple(val)
    elif isinstance(val, tuple):
        return val
    else:
        return (val,)


@rich.repr.auto
@dataclass(frozen=True, repr=True, kw_only=True)
class Layer:
    """
    This is an abstract representation of Container Image Layers, which can be used to create
     layered images programmatically.
    """

    @abstractmethod
    def update_hash(self, hasher: hashlib._Hash):
        """
        This method should be implemented by subclasses to provide a hash representation of the layer.

        :param hasher: The hash object to update with the layer's data.
        """

    def validate(self):
        """
        Raise any validation errors for the layer
        :return:
        """


@rich.repr.auto
@dataclass(kw_only=True, frozen=True, repr=True)
class PipOption:
    index_url: Optional[str] = None
    extra_index_urls: Optional[Tuple[str] | Tuple[str, ...] | List[str]] = None
    pre: bool = False
    extra_args: Optional[str] = None
    secret_mounts: Optional[Tuple[str | Secret, ...]] = None

    def get_pip_install_args(self) -> List[str]:
        pip_install_args = []
        if self.index_url:
            pip_install_args.append(f"--index-url {self.index_url}")

        if self.extra_index_urls:
            pip_install_args.extend([f"--extra-index-url {url}" for url in self.extra_index_urls])

        if self.pre:
            pip_install_args.append("--pre")

        if self.extra_args:
            pip_install_args.append(self.extra_args)
        return pip_install_args

    def update_hash(self, hasher: hashlib._Hash):
        """
        Update the hash with the PipOption
        """
        hash_input = ""
        if self.index_url:
            hash_input += self.index_url
        if self.extra_index_urls:
            for url in self.extra_index_urls:
                hash_input += url
        if self.pre:
            hash_input += str(self.pre)
        if self.extra_args:
            hash_input += self.extra_args
        if self.secret_mounts:
            for secret_mount in self.secret_mounts:
                hash_input += str(secret_mount)

        hasher.update(hash_input.encode("utf-8"))


@rich.repr.auto
@dataclass(kw_only=True, frozen=True, repr=True)
class PipPackages(PipOption, Layer):
    packages: Optional[Tuple[str, ...]] = None

    def update_hash(self, hasher: hashlib._Hash):
        """
        Update the hash with the pip packages
        """
        super().update_hash(hasher)
        hash_input = ""
        if self.packages:
            for package in self.packages:
                hash_input += package

        hasher.update(hash_input.encode("utf-8"))


@rich.repr.auto
@dataclass(kw_only=True, frozen=True, repr=True)
class PythonWheels(PipOption, Layer):
    wheel_dir: Path
    wheel_dir_name: str = field(init=False)
    package_name: str

    def __post_init__(self):
        object.__setattr__(self, "wheel_dir_name", self.wheel_dir.name)

    def update_hash(self, hasher: hashlib._Hash):
        super().update_hash(hasher)
        from ._utils import filehash_update

        # Iterate through all the wheel files in the directory and update the hash
        for wheel_file in self.wheel_dir.glob("*.whl"):
            if not wheel_file.is_file():
                # Skip if it's not a file (e.g., directory or symlink)
                continue
            filehash_update(wheel_file, hasher)


@rich.repr.auto
@dataclass(kw_only=True, frozen=True, repr=True)
class Requirements(PipPackages):
    file: Path

    def update_hash(self, hasher: hashlib._Hash):
        from ._utils import filehash_update

        super().update_hash(hasher)
        filehash_update(self.file, hasher)


@rich.repr.auto
@dataclass(frozen=True, repr=True)
class UVProject(PipOption, Layer):
    pyproject: Path
    uvlock: Path
    project_install_mode: typing.Literal["dependencies_only", "install_project"] = "dependencies_only"

    def validate(self):
        if not self.pyproject.exists():
            raise FileNotFoundError(f"pyproject.toml file {self.pyproject.resolve()} does not exist")
        if not self.pyproject.is_file():
            raise ValueError(f"Pyproject file {self.pyproject.resolve()} is not a file")
        if not self.uvlock.exists():
            raise ValueError(f"UVLock file {self.uvlock.resolve()} does not exist")
        super().validate()

    def update_hash(self, hasher: hashlib._Hash):
        from ._utils import filehash_update, update_hasher_for_source

        super().update_hash(hasher)
        if self.project_install_mode == "dependencies_only":
            filehash_update(self.uvlock, hasher)
            filehash_update(self.pyproject, hasher)
        else:
            update_hasher_for_source(self.pyproject.parent, hasher)


@rich.repr.auto
@dataclass(frozen=True, repr=True)
class PoetryProject(Layer):
    """
    Poetry does not use pip options, so the PoetryProject class do not inherits PipOption class
    """

    pyproject: Path
    poetry_lock: Path
    extra_args: Optional[str] = None
    project_install_mode: typing.Literal["dependencies_only", "install_project"] = "dependencies_only"
    secret_mounts: Optional[Tuple[str | Secret, ...]] = None

    def validate(self):
        if not self.pyproject.exists():
            raise FileNotFoundError(f"pyproject.toml file {self.pyproject} does not exist")
        if not self.pyproject.is_file():
            raise ValueError(f"Pyproject file {self.pyproject} is not a file")
        if not self.poetry_lock.exists():
            raise ValueError(f"poetry.lock file {self.poetry_lock} does not exist")
        super().validate()

    def update_hash(self, hasher: hashlib._Hash):
        from ._utils import filehash_update, update_hasher_for_source

        hash_input = ""
        if self.extra_args:
            hash_input += self.extra_args
        if self.secret_mounts:
            for secret_mount in self.secret_mounts:
                hash_input += str(secret_mount)
        hasher.update(hash_input.encode("utf-8"))

        if self.project_install_mode == "dependencies_only":
            filehash_update(self.poetry_lock, hasher)
            filehash_update(self.pyproject, hasher)
        else:
            update_hasher_for_source(self.pyproject.parent, hasher)


@rich.repr.auto
@dataclass(frozen=True, repr=True)
class UVScript(PipOption, Layer):
    script: Path
    script_name: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "script_name", self.script.name)

    def validate(self):
        if not self.script.exists():
            raise FileNotFoundError(f"UV script {self.script} does not exist")
        if not self.script.is_file():
            raise ValueError(f"UV script {self.script} is not a file")
        if not self.script.suffix == ".py":
            raise ValueError(f"UV script {self.script} must have a .py extension")
        super().validate()

    def update_hash(self, hasher: hashlib._Hash):
        from ._utils import parse_uv_script_file

        header = parse_uv_script_file(self.script)
        h_tuple = _ensure_tuple(header)
        if h_tuple:
            hasher.update(h_tuple.__str__().encode("utf-8"))
        super().update_hash(hasher)


@rich.repr.auto
@dataclass(frozen=True, repr=True)
class AptPackages(Layer):
    packages: Tuple[str, ...]
    secret_mounts: Optional[Tuple[str | Secret, ...]] = None

    def update_hash(self, hasher: hashlib._Hash):
        hash_input = "".join(self.packages)

        if self.secret_mounts:
            for secret_mount in self.secret_mounts:
                hash_input += str(secret_mount)
        hasher.update(hash_input.encode("utf-8"))


@rich.repr.auto
@dataclass(frozen=True, repr=True)
class Commands(Layer):
    commands: Tuple[str, ...]
    secret_mounts: Optional[Tuple[str | Secret, ...]] = None

    def update_hash(self, hasher: hashlib._Hash):
        hash_input = "".join(self.commands)

        if self.secret_mounts:
            for secret_mount in self.secret_mounts:
                hash_input += str(secret_mount)
        hasher.update(hash_input.encode("utf-8"))


@rich.repr.auto
@dataclass(frozen=True, repr=True)
class WorkDir(Layer):
    workdir: str

    def update_hash(self, hasher: hashlib._Hash):
        hasher.update(self.workdir.encode("utf-8"))


@rich.repr.auto
@dataclass(frozen=True, repr=True)
class DockerIgnore(Layer):
    path: str

    def update_hash(self, hasher: hashlib._Hash):
        hasher.update(self.path.encode("utf-8"))


@rich.repr.auto
@dataclass(frozen=True, repr=True)
class CopyConfig(Layer):
    path_type: CopyConfigType
    src: Path
    dst: str

    def __post_init__(self):
        if self.path_type not in (0, 1):
            raise ValueError(f"Invalid path_type {self.path_type}, must be 0 (file) or 1 (directory)")

    def validate(self):
        if not self.src.exists():
            raise ValueError(f"Source folder {self.src.absolute()} does not exist")
        if not self.src.is_dir() and self.path_type == 1:
            raise ValueError(f"Source folder {self.src.absolute()} is not a directory")
        if not self.src.is_file() and self.path_type == 0:
            raise ValueError(f"Source file {self.src.absolute()} is not a file")

    def update_hash(self, hasher: hashlib._Hash):
        from ._utils import update_hasher_for_source

        update_hasher_for_source(self.src, hasher)
        if self.dst:
            hasher.update(self.dst.encode("utf-8"))


@rich.repr.auto
@dataclass(frozen=True, repr=True)
class _DockerLines(Layer):
    """
    This is an internal class and should only be used by the default images. It is not supported by most
    builders so please don't use it.
    """

    lines: Tuple[str, ...]

    def update_hash(self, hasher: hashlib._Hash):
        hasher.update("".join(self.lines).encode("utf-8"))


@rich.repr.auto
@dataclass(frozen=True, repr=True)
class Env(Layer):
    """
    This is an internal class and should only be used by the default images. It is not supported by most
    builders so please don't use it.
    """

    env_vars: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    def update_hash(self, hasher: hashlib._Hash):
        txt = [f"{k}={v}" for k, v in self.env_vars]
        hasher.update(" ".join(txt).encode("utf-8"))

    @classmethod
    def from_dict(cls, envs: Dict[str, str]) -> Env:
        return cls(env_vars=tuple((k, v) for k, v in envs.items()))


Architecture = Literal["linux/amd64", "linux/arm64"]

_BASE_REGISTRY = "ghcr.io/flyteorg"
_DEFAULT_IMAGE_NAME = "flyte"


def _detect_python_version() -> Tuple[int, int]:
    """
    Detect the current Python version.
    :return: Tuple of major and minor version
    """
    return sys.version_info.major, sys.version_info.minor


@dataclass(frozen=True, repr=True, eq=True)
class Image:
    """
    This is a representation of Container Images, which can be used to create layered images programmatically.

    Use by first calling one of the base constructor methods. These all begin with `from` or `default_`
    The image can then be amended with additional layers using the various `with_*` methods.

    Invariant for this class: The construction of Image objects must be doable everywhere. That is, if a
      user has a custom image that is not accessible, calling .with_source_file on a file that doesn't exist, the
      instantiation of the object itself must still go through. Further, the .identifier property of the image must
      also still go through. This is because it may have been already built somewhere else.
      Use validate() functions to check each layer for actual errors. These are invoked at actual
      build time. See self.id for more information
    """

    # These are base properties of an image
    base_image: Optional[str] = field(default=None)
    dockerfile: Optional[Path] = field(default=None)
    registry: Optional[str] = field(default=None)
    name: Optional[str] = field(default=None)
    platform: Tuple[Architecture, ...] = field(default=("linux/amd64",))
    python_version: Tuple[int, int] = field(default_factory=_detect_python_version)
    # Refer to the image_refs (name:image-uri) set in CLI or config
    _ref_name: Optional[str] = field(default=None)

    # Layers to be added to the image. In init, because frozen, but users shouldn't access, so underscore.
    _layers: Tuple[Layer, ...] = field(default_factory=tuple)

    # Only settable internally.
    _tag: Optional[str] = field(default=None, init=False)

    _DEFAULT_IMAGE_PREFIXES: ClassVar = {
        PYTHON_3_10: "py3.10-",
        PYTHON_3_11: "py3.11-",
        PYTHON_3_12: "py3.12-",
        PYTHON_3_13: "py3.13-",
    }

    # class-level token not included in __init__
    _token: ClassVar[object] = object()

    # Underscore cuz we may rename in the future, don't expose for now,
    _image_registry_secret: Optional[Secret] = None

    # check for the guard that we put in place
    def __post_init__(self):
        if object.__getattribute__(self, "__dict__").pop("_guard", None) is not Image._token:
            raise TypeError(
                "Direct instantiation of Image not allowed, please use one of the various from_...() methods instead"
            )

    # Private constructor for internal use only
    @classmethod
    def _new(cls, **kwargs) -> Image:
        # call the normal __init__, injecting a private keyword that users won't know
        obj = cls.__new__(cls)  # allocate
        object.__setattr__(obj, "_guard", cls._token)  # set guard to prevent direct construction
        cls.__init__(obj, **kwargs)  # run dataclass generated __init__
        return obj

    def validate(self):
        for layer in self._layers:
            layer.validate()

    @classmethod
    def _get_default_image_for(
        cls,
        python_version: Tuple[int, int],
        flyte_version: Optional[str] = None,
        install_flyte: bool = True,
        platform: Optional[Tuple[Architecture, ...]] = None,
    ) -> Image:
        # Would love a way to move this outside of this class (but still needs to be accessible via Image.auto())
        # this default image definition may need to be updated once there is a released pypi version
        from flyte._version import __version__

        dev_mode = (__version__ and "dev" in __version__) and not flyte_version and install_flyte
        if install_flyte is False:
            preset_tag = f"py{python_version[0]}.{python_version[1]}"
        else:
            if flyte_version is None:
                flyte_version = __version__.replace("+", "-")
            suffix = flyte_version if flyte_version.startswith("v") else f"v{flyte_version}"
            preset_tag = f"py{python_version[0]}.{python_version[1]}-{suffix}"
        image = Image._new(
            base_image=f"python:{python_version[0]}.{python_version[1]}-slim-bookworm",
            registry=_BASE_REGISTRY,
            name=_DEFAULT_IMAGE_NAME,
            python_version=python_version,
            platform=("linux/amd64", "linux/arm64") if platform is None else platform,
        )
        labels_and_user = _DockerLines(
            (
                "LABEL org.opencontainers.image.authors='Union.AI <info@union.ai>'",
                "LABEL org.opencontainers.image.source=https://github.com/flyteorg/flyte",
                "RUN useradd --create-home --shell /bin/bash flytekit &&"
                " chown -R flytekit /root && chown -R flytekit /home",
                "WORKDIR /root",
            )
        )
        image = image.clone(addl_layer=labels_and_user)
        image = image.with_env_vars(
            {
                "VIRTUAL_ENV": "/opt/venv",
                "PATH": "/opt/venv/bin:$PATH",
                "PYTHONPATH": "/root",
                "UV_LINK_MODE": "copy",
            }
        )
        image = image.with_apt_packages("build-essential", "ca-certificates")

        if install_flyte:
            if dev_mode:
                image = image.with_local_v2()
            else:
                flyte_version = typing.cast(str, flyte_version)
                if Version(flyte_version).is_devrelease or Version(flyte_version).is_prerelease:
                    image = image.with_pip_packages(f"flyte=={flyte_version}", pre=True)
                else:
                    image = image.with_pip_packages(f"flyte=={flyte_version}")
        if not dev_mode:
            object.__setattr__(image, "_tag", preset_tag)

        return image

    @classmethod
    def from_debian_base(
        cls,
        python_version: Optional[Tuple[int, int]] = None,
        flyte_version: Optional[str] = None,
        install_flyte: bool = True,
        registry: Optional[str] = None,
        registry_secret: Optional[str | Secret] = None,
        name: Optional[str] = None,
        platform: Optional[Tuple[Architecture, ...]] = None,
    ) -> Image:
        """
        Use this method to start using the default base image, built from this library's base Dockerfile
        Default images are multi-arch amd/arm64

        :param python_version: If not specified, will use the current Python version
        :param flyte_version: Union version to use
        :param install_flyte: If True, will install the flyte library in the image
        :param registry: Registry to use for the image
        :param registry_secret: Secret to use to pull/push the private image.
        :param name: Name of the image if you want to override the default name
        :param platform: Platform to use for the image, default is linux/amd64, use tuple for multiple values
            Example: ("linux/amd64", "linux/arm64")

        :return: Image
        """
        if python_version is None:
            python_version = _detect_python_version()

        base_image = cls._get_default_image_for(
            python_version=python_version,
            flyte_version=flyte_version,
            install_flyte=install_flyte,
            platform=platform,
        )

        if registry or name:
            return base_image.clone(registry=registry, name=name, registry_secret=registry_secret)

        return base_image

    @classmethod
    def from_base(cls, image_uri: str) -> Image:
        """
        Use this method to start with a pre-built base image. This image must already exist in the registry of course.

        :param image_uri: The full URI of the image, in the format <registry>/<name>:<tag>
        :return:
        """
        img = cls._new(base_image=image_uri)
        return img

    @classmethod
    def from_ref_name(cls, name: str) -> Image:
        # NOTE: set image name as _ref_name to enable adding additional layers.
        # See: https://github.com/flyteorg/flyte-sdk/blob/14de802701aab7b8615ffb99c650a36305ef01f7/src/flyte/_image.py#L642
        img = cls._new(name=name, _ref_name=name)
        return img

    @classmethod
    def from_uv_script(
        cls,
        script: Path | str,
        *,
        name: str,
        registry: str | None = None,
        registry_secret: Optional[str | Secret] = None,
        python_version: Optional[Tuple[int, int]] = None,
        index_url: Optional[str] = None,
        extra_index_urls: Union[str, List[str], Tuple[str, ...], None] = None,
        pre: bool = False,
        extra_args: Optional[str] = None,
        platform: Optional[Tuple[Architecture, ...]] = None,
        secret_mounts: Optional[SecretRequest] = None,
    ) -> Image:
        """
        Use this method to create a new image with the specified uv script.
        It uses the header of the script to determine the python version, dependencies to install.
        The script must be a valid uv script, otherwise an error will be raised.

        Usually the header of the script will look like this:
        Example:
        ```python
        #!/usr/bin/env -S uv run --script
        # /// script
        # requires-python = ">=3.12"
        # dependencies = ["httpx"]
        # ///
        ```

        For more information on the uv script format, see the documentation:
        [UV: Declaring script dependencies](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies)

        :param name: name of the image
        :param registry: registry to use for the image
        :param registry_secret: Secret to use to pull/push the private image.
        :param python_version: Python version to use for the image, if not specified, will use the current Python
        version
        :param script: path to the uv script
        :param platform: architecture to use for the image, default is linux/amd64, use tuple for multiple values
        :param python_version: Python version for the image, if not specified, will use the current Python version
        :param index_url: index url to use for pip install, default is None
        :param extra_index_urls: extra index urls to use for pip install, default is None
        :param pre: whether to allow pre-release versions, default is False
        :param extra_args: extra arguments to pass to pip install, default is None
        :param secret_mounts: Secret mounts to use for the image, default is None.

        :return: Image

        Args:
            secret_mounts:
        """
        ll = UVScript(
            script=Path(script),
            index_url=index_url,
            extra_index_urls=_ensure_tuple(extra_index_urls) if extra_index_urls else None,
            pre=pre,
            extra_args=extra_args,
            secret_mounts=_ensure_tuple(secret_mounts) if secret_mounts else None,
        )

        img = cls.from_debian_base(
            registry=registry,
            registry_secret=registry_secret,
            name=name,
            python_version=python_version,
            platform=platform,
        )

        return img.clone(addl_layer=ll)

    def clone(
        self,
        registry: Optional[str] = None,
        registry_secret: Optional[str | Secret] = None,
        name: Optional[str] = None,
        base_image: Optional[str] = None,
        python_version: Optional[Tuple[int, int]] = None,
        addl_layer: Optional[Layer] = None,
    ) -> Image:
        """
        Use this method to clone the current image and change the registry and name

        :param registry: Registry to use for the image
        :param registry_secret: Secret to use to pull/push the private image.
        :param name: Name of the image
        :param python_version: Python version for the image, if not specified, will use the current Python version
        :param addl_layer: Additional layer to add to the image. This will be added to the end of the layers.
        :return:
        """
        from flyte import Secret

        if addl_layer and self.dockerfile:
            # We don't know how to inspect dockerfiles to know what kind it is (OS, python version, uv vs poetry, etc)
            # so there's no guarantee any of the layering logic will work.
            raise ValueError(
                "Flyte current cannot add additional layers to a Dockerfile-based Image."
                " Please amend the dockerfile directly."
            )
        registry = registry if registry else self.registry
        name = name if name else self.name
        registry_secret = registry_secret if registry_secret else self._image_registry_secret
        base_image = base_image if base_image else self.base_image
        if addl_layer and (not name):
            raise ValueError(
                f"Cannot add additional layer {addl_layer} to an image without name. Please first clone()."
            )
        new_layers = (*self._layers, addl_layer) if addl_layer else self._layers
        img = Image._new(
            base_image=base_image,
            dockerfile=self.dockerfile,
            registry=registry,
            name=name,
            platform=self.platform,
            python_version=python_version or self.python_version,
            _layers=new_layers,
            _image_registry_secret=Secret(key=registry_secret) if isinstance(registry_secret, str) else registry_secret,
            _ref_name=self._ref_name,
        )

        return img

    @classmethod
    def from_dockerfile(
        cls, file: Path, registry: str, name: str, platform: Union[Architecture, Tuple[Architecture, ...], None] = None
    ) -> Image:
        """
        Use this method to create a new image with the specified dockerfile. Note you cannot use additional layers
        after this, as the system doesn't attempt to parse/understand the Dockerfile, and what kind of setup it has
        (python version, uv vs poetry, etc), so please put all logic into the dockerfile itself.

        Also since Python sees paths as from the calling directory, please use Path objects with absolute paths. The
        context for the builder will be the directory where the dockerfile is located.

        :param file: path to the dockerfile
        :param name: name of the image
        :param registry: registry to use for the image
        :param platform: architecture to use for the image, default is linux/amd64, use tuple for multiple values
            Example: ("linux/amd64", "linux/arm64")

        :return:
        """
        platform = _ensure_tuple(platform) if platform else None
        kwargs = {
            "dockerfile": file,
            "registry": registry,
            "name": name,
        }
        if platform:
            kwargs["platform"] = platform
        img = cls._new(**kwargs)

        return img

    def _get_hash_digest(self) -> str:
        """
        Returns the hash digest of the image, which is a combination of all the layers and properties of the image
        """
        import hashlib

        from ._utils import filehash_update

        hasher = hashlib.md5()
        if self.base_image:
            hasher.update(self.base_image.encode("utf-8"))
        if self.dockerfile:
            # Note the location of the dockerfile shouldn't matter, only the contents
            filehash_update(self.dockerfile, hasher)
        if self._layers:
            for layer in self._layers:
                layer.update_hash(hasher)
        return hasher.hexdigest()

    @property
    def _final_tag(self) -> str:
        t = self._tag if self._tag else self._get_hash_digest()
        return t or "latest"

    @cached_property
    def uri(self) -> str:
        """
        Returns the URI of the image in the format <registry>/<name>:<tag>
        """
        if self.registry and self.name:
            tag = self._final_tag
            return f"{self.registry}/{self.name}:{tag}"
        elif self.name:
            return f"{self.name}:{self._final_tag}"
        elif self.base_image:
            return self.base_image

        raise ValueError("Image is not fully defined. Please set registry, name and tag.")

    def with_workdir(self, workdir: str) -> Image:
        """
        Use this method to create a new image with the specified working directory
        This will override any existing working directory

        :param workdir: working directory to use
        :return:
        """
        new_image = self.clone(addl_layer=WorkDir(workdir=workdir))
        return new_image

    def with_requirements(
        self,
        file: str | Path,
        secret_mounts: Optional[SecretRequest] = None,
    ) -> Image:
        """
        Use this method to create a new image with the specified requirements file layered on top of the current image
        Cannot be used in conjunction with conda

        :param file: path to the requirements file, must be a .txt file
        :param secret_mounts: list of secret to mount for the build process.
        :return:
        """
        if isinstance(file, str):
            file = Path(file)
        if file.suffix != ".txt":
            raise ValueError(f"Requirements file {file} must have a .txt extension")
        new_image = self.clone(
            addl_layer=Requirements(file=file, secret_mounts=_ensure_tuple(secret_mounts) if secret_mounts else None)
        )
        return new_image

    def with_pip_packages(
        self,
        *packages: str,
        index_url: Optional[str] = None,
        extra_index_urls: Union[str, List[str], Tuple[str, ...], None] = None,
        pre: bool = False,
        extra_args: Optional[str] = None,
        secret_mounts: Optional[SecretRequest] = None,
    ) -> Image:
        """
        Use this method to create a new image with the specified pip packages layered on top of the current image
        Cannot be used in conjunction with conda

        Example:
        ```python
        @flyte.task(image=(flyte.Image.from_debian_base().with_pip_packages("requests", "numpy")))
        def my_task(x: int) -> int:
            import numpy as np
            return np.sum([x, 1])
        ```

        To mount secrets during the build process to download private packages, you can use the `secret_mounts`.
        In the below example, "GITHUB_PAT" will be mounted as env var "GITHUB_PAT",
         and "apt-secret" will be mounted at /etc/apt/apt-secret.
        Example:
        ```python
        private_package = "git+https://$GITHUB_PAT@github.com/flyteorg/flytex.git@2e20a2acebfc3877d84af643fdd768edea41d533"
        @flyte.task(
            image=(
                flyte.Image.from_debian_base()
                .with_pip_packages("private_package", secret_mounts=[Secret(key="GITHUB_PAT")])
                .with_apt_packages("git", secret_mounts=[Secret(key="apt-secret", mount="/etc/apt/apt-secret")])
        )
        def my_task(x: int) -> int:
            import numpy as np
            return np.sum([x, 1])
        ```

        :param packages: list of pip packages to install, follows pip install syntax
        :param index_url: index url to use for pip install, default is None
        :param extra_index_urls: extra index urls to use for pip install, default is None
        :param pre: whether to allow pre-release versions, default is False
        :param extra_args: extra arguments to pass to pip install, default is None
        :param extra_args: extra arguments to pass to pip install, default is None
        :param secret_mounts: list of secret to mount for the build process.
        :return: Image
        """
        new_packages: Optional[Tuple] = packages or None
        new_extra_index_urls: Optional[Tuple] = _ensure_tuple(extra_index_urls) if extra_index_urls else None

        ll = PipPackages(
            packages=new_packages,
            index_url=index_url,
            extra_index_urls=new_extra_index_urls,
            pre=pre,
            extra_args=extra_args,
            secret_mounts=_ensure_tuple(secret_mounts) if secret_mounts else None,
        )
        new_image = self.clone(addl_layer=ll)
        return new_image

    def with_env_vars(self, env_vars: Dict[str, str]) -> Image:
        """
        Use this method to create a new image with the specified environment variables layered on top of
        the current image. Cannot be used in conjunction with conda

        :param env_vars: dictionary of environment variables to set
        :return: Image
        """
        new_image = self.clone(addl_layer=Env.from_dict(env_vars))
        return new_image

    def with_source_folder(self, src: Path, dst: str = ".", copy_contents_only: bool = False) -> Image:
        """
        Use this method to create a new image with the specified local directory layered on top of the current image.
        If dest is not specified, it will be copied to the working directory of the image

        :param src: root folder of the source code from the build context to be copied
        :param dst: destination folder in the image
        :param copy_contents_only: If True, will copy the contents of the source folder to the destination folder,
            instead of the folder itself. Default is False.
        :return: Image
        """
        if not copy_contents_only:
            dst = str("./" + src.name) if dst == "." else dst
        new_image = self.clone(addl_layer=CopyConfig(path_type=1, src=src, dst=dst))
        return new_image

    def with_source_file(self, src: Path, dst: str = ".") -> Image:
        """
        Use this method to create a new image with the specified local file layered on top of the current image.
        If dest is not specified, it will be copied to the working directory of the image

        :param src: root folder of the source code from the build context to be copied
        :param dst: destination folder in the image
        :return: Image
        """
        new_image = self.clone(addl_layer=CopyConfig(path_type=0, src=src, dst=dst))
        return new_image

    def with_dockerignore(self, path: Path) -> Image:
        new_image = self.clone(addl_layer=DockerIgnore(path=str(path)))
        return new_image

    def with_uv_project(
        self,
        pyproject_file: str | Path,
        uvlock: Path | None = None,
        index_url: Optional[str] = None,
        extra_index_urls: Union[List[str], Tuple[str, ...], None] = None,
        pre: bool = False,
        extra_args: Optional[str] = None,
        secret_mounts: Optional[SecretRequest] = None,
        project_install_mode: typing.Literal["dependencies_only", "install_project"] = "dependencies_only",
    ) -> Image:
        """
        Use this method to create a new image with the specified uv.lock file layered on top of the current image
        Must have a corresponding pyproject.toml file in the same directory
        Cannot be used in conjunction with conda

        By default, this method copies the pyproject.toml and uv.lock files into the image.

        If `project_install_mode` is "install_project", it will also copy directory
         where the pyproject.toml file is located into the image.

        :param pyproject_file: path to the pyproject.toml file, needs to have a corresponding uv.lock file
        :param uvlock: path to the uv.lock file, if not specified, will use the default uv.lock file in the same
        directory as the pyproject.toml file. (pyproject.parent / uv.lock)
        :param index_url: index url to use for pip install, default is None
        :param extra_index_urls: extra index urls to use for pip install, default is None
        :param pre: whether to allow pre-release versions, default is False
        :param extra_args: extra arguments to pass to pip install, default is None
        :param secret_mounts: list of secret mounts to use for the build process.
        :param project_install_mode: whether to install the project as a package or
         only dependencies, default is "dependencies_only"
        :return: Image
        """
        if isinstance(pyproject_file, str):
            pyproject_file = Path(pyproject_file)
        new_image = self.clone(
            addl_layer=UVProject(
                pyproject=pyproject_file,
                uvlock=uvlock or (pyproject_file.parent / "uv.lock"),
                index_url=index_url,
                extra_index_urls=extra_index_urls,
                pre=pre,
                extra_args=extra_args,
                secret_mounts=_ensure_tuple(secret_mounts) if secret_mounts else None,
                project_install_mode=project_install_mode,
            )
        )
        return new_image

    def with_poetry_project(
        self,
        pyproject_file: str | Path,
        poetry_lock: Path | None = None,
        extra_args: Optional[str] = None,
        secret_mounts: Optional[SecretRequest] = None,
        project_install_mode: typing.Literal["dependencies_only", "install_project"] = "dependencies_only",
    ):
        """
        Use this method to create a new image with the specified pyproject.toml layered on top of the current image.
        Must have a corresponding pyproject.toml file in the same directory.
        Cannot be used in conjunction with conda.

        By default, this method copies the entire project into the image,
        including files such as pyproject.toml, poetry.lock, and the src/ directory.

        If you prefer not to install the current project, you can pass through `extra_args`
        `--no-root`. In this case, the image builder will only copy pyproject.toml and poetry.lock
        into the image.

        :param pyproject_file: Path to the pyproject.toml file. A poetry.lock file must exist in the same directory
            unless `poetry_lock` is explicitly provided.
        :param poetry_lock: Path to the poetry.lock file. If not specified, the default is the file named
            'poetry.lock' in the same directory as `pyproject_file` (pyproject.parent / "poetry.lock").
        :param extra_args: Extra arguments to pass through to the package installer/resolver, default is None.
        :param secret_mounts: Secrets to make available during dependency resolution/build (e.g., private indexes).
        :param project_install_mode: whether to install the project as a package or
         only dependencies, default is "dependencies_only"
        :return: Image
        """
        if isinstance(pyproject_file, str):
            pyproject_file = Path(pyproject_file)
        new_image = self.clone(
            addl_layer=PoetryProject(
                pyproject=pyproject_file,
                poetry_lock=poetry_lock or (pyproject_file.parent / "poetry.lock"),
                extra_args=extra_args,
                secret_mounts=_ensure_tuple(secret_mounts) if secret_mounts else None,
                project_install_mode=project_install_mode,
            )
        )
        return new_image

    def with_apt_packages(self, *packages: str, secret_mounts: Optional[SecretRequest] = None) -> Image:
        """
        Use this method to create a new image with the specified apt packages layered on top of the current image

        :param packages: list of apt packages to install
        :param secret_mounts: list of secret mounts to use for the build process.
        :return: Image
        """
        new_image = self.clone(
            addl_layer=AptPackages(
                packages=packages,
                secret_mounts=_ensure_tuple(secret_mounts) if secret_mounts else None,
            )
        )
        return new_image

    def with_commands(self, commands: List[str], secret_mounts: Optional[SecretRequest] = None) -> Image:
        """
        Use this method to create a new image with the specified commands layered on top of the current image
        Be sure not to use RUN in your command.

        :param commands: list of commands to run
        :param secret_mounts: list of secret mounts to use for the build process.
        :return: Image
        """
        new_commands: Tuple = _ensure_tuple(commands)
        new_image = self.clone(
            addl_layer=Commands(
                commands=new_commands, secret_mounts=_ensure_tuple(secret_mounts) if secret_mounts else None
            )
        )
        return new_image

    def with_local_v2(self) -> Image:
        """
        Use this method to create a new image with the local v2 builder
        This will override any existing builder

        :return: Image
        """
        dist_folder = Path(__file__).parent.parent.parent / "dist"
        # Manually declare the PythonWheel so we can set the hashing
        # used to compute the identifier. Can remove if we ever decide to expose the lambda in with_ commands
        with_dist = self.clone(addl_layer=PythonWheels(wheel_dir=dist_folder, package_name="flyte", pre=True))

        return with_dist

    def __img_str__(self) -> str:
        """
        For the current image only, print all the details if they are not None
        """
        details = []
        if self.base_image:
            details.append(f"Base Image: {self.base_image}")
        elif self.dockerfile:
            details.append(f"Dockerfile: {self.dockerfile}")
        if self.registry:
            details.append(f"Registry: {self.registry}")
        if self.name:
            details.append(f"Name: {self.name}")
        if self.platform:
            details.append(f"Platform: {self.platform}")

        if self.__getattribute__("_layers"):
            for layer in self._layers:
                details.append(f"Layer: {layer}")

        return "\n".join(details)
