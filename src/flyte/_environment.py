from __future__ import annotations

import inspect
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import rich.repr

from ._image import Image
from ._pod import PodTemplate
from ._resources import Resources
from ._secret import Secret, SecretRequest

# Global registry to track all Environment instances in load order
_ENVIRONMENT_REGISTRY: List[Environment] = []


def list_loaded_environments() -> List[Environment]:
    """
    Return a list of all Environment objects in the order they were loaded.
    This is useful for deploying environments in the order they were defined.
    """
    return _ENVIRONMENT_REGISTRY


def is_snake_or_kebab_with_numbers(s: str) -> bool:
    return re.fullmatch(r"^[a-z0-9]+([_-][a-z0-9]+)*$", s) is not None


@rich.repr.auto
@dataclass(init=True, repr=True)
class Environment:
    """
    Base class for execution environments, shared by `TaskEnvironment` and
    `AppEnvironment`. Defines common infrastructure settings such as container
    image, compute resources, secrets, and deployment dependencies.

    You typically don't instantiate `Environment` directly — use
    `TaskEnvironment` for tasks or `AppEnvironment` for long-running apps.

    :param name: Name of the environment (required). Must be snake_case or
        kebab-case.
    :param image: Docker image for the environment. Can be a string (image URI),
        an `Image` object, or `"auto"` to use the default image.
    :param resources: Compute resources (CPU, memory, GPU, disk) via a
        `Resources` object.
    :param env_vars: Environment variables as `dict[str, str]`.
    :param secrets: Secrets to inject into the environment.
    :param pod_template: Kubernetes pod template as a string reference to a
        named template or a `PodTemplate` object.
    :param description: Human-readable description (max 255 characters).
    :param interruptible: Whether the environment can be scheduled on
        spot/preemptible instances.
    :param depends_on: List of other environments to deploy alongside this one.
    :param include: Extra files to bundle with the environment's code (e.g., HTML
        templates, config files, non-Python assets). Paths may be relative (resolved
        against the directory of the file where the environment is instantiated),
        absolute, directories (recursively included), or glob patterns. Files
        listed here are bundled **in addition to** the default ``copy_style``
        discovery (``loaded_modules`` or ``all``), not in place of it.
    """

    name: str
    depends_on: List[Environment] = field(default_factory=list)
    pod_template: Optional[Union[str, PodTemplate]] = None
    description: Optional[str] = None
    secrets: Optional[SecretRequest] = None
    env_vars: Optional[Dict[str, str]] = None
    resources: Optional[Resources] = None
    interruptible: bool = False
    image: Union[str, Image, Literal["auto"], None] = "auto"
    include: Tuple[str, ...] = field(default_factory=tuple)

    # Absolute path of the user file where this environment was instantiated.
    # Populated in __post_init__. Used to anchor relative `include` paths.
    _declaring_file: Optional[str] = field(init=False, default=None, repr=False)

    def _validate_name(self):
        if not is_snake_or_kebab_with_numbers(self.name):
            raise ValueError(f"Environment name '{self.name}' must be in snake_case or kebab-case format.")

    def _get_declaring_file(self) -> Optional[str]:
        """
        Return the absolute path of the user file that instantiated this environment.

        Walks the call stack to skip flyte SDK internals. Used to anchor relative
        `include` paths. Returns ``None`` only if no user file is discoverable
        (shouldn't happen in normal usage).
        """

        def is_user_file(filename: str) -> bool:
            if filename in ("<string>", "<stdin>"):
                return False
            if not os.path.exists(filename):
                return False
            abs_path = os.path.abspath(filename)
            return ("site-packages/flyte" not in abs_path and "/flyte/" not in abs_path) or "/examples/" in abs_path

        frame = inspect.currentframe()
        while frame is not None:
            filename = frame.f_code.co_filename
            if is_user_file(filename):
                return os.path.abspath(filename)
            frame = frame.f_back

        stack = inspect.stack()
        for frame_info in stack:
            filename = frame_info.filename
            if is_user_file(filename):
                return os.path.abspath(filename)

        import sys

        main = sys.modules.get("__main__")
        main_file = getattr(main, "__file__", None)
        if main_file and os.path.exists(main_file):
            return os.path.abspath(main_file)

        return None

    def __post_init__(self):
        if self.image and not isinstance(self.image, (Image, str)):
            raise TypeError(f"Expected image to be of type str or Image, got {type(self.image)}")
        if self.secrets and not isinstance(self.secrets, (str, Secret, List)):
            raise TypeError(f"Expected secrets to be of type SecretRequest, got {type(self.secrets)}")
        for dep in self.depends_on:
            if not isinstance(dep, Environment):
                raise TypeError(f"Expected depends_on to be of type List[Environment], got {type(dep)}")
        if self.resources is not None and not isinstance(self.resources, Resources):
            raise TypeError(f"Expected resources to be of type Resources, got {type(self.resources)}")
        if self.env_vars is not None and not isinstance(self.env_vars, dict):
            raise TypeError(f"Expected env_vars to be of type Dict[str, str], got {type(self.env_vars)}")
        if self.pod_template is not None and not isinstance(self.pod_template, (str, PodTemplate)):
            raise TypeError(f"Expected pod_template to be of type str or PodTemplate, got {type(self.pod_template)}")
        if self.description is not None and len(self.description) > 255:
            from flyte._utils.description_parser import parse_description

            self.description = parse_description(self.description, 255)
        # Normalize `include` to a tuple of strings so the dataclass stays hashable.
        if self.include and not isinstance(self.include, tuple):
            if isinstance(self.include, str):
                raise TypeError(f"Expected include to be a sequence of str paths, got a bare str ({self.include!r}).")
            try:
                self.include = tuple(self.include)
            except TypeError as exc:
                raise TypeError(f"Expected include to be a sequence of str paths, got {type(self.include)}") from exc
        for inc in self.include:
            if not isinstance(inc, str):
                raise TypeError(f"include entries must be str, got {type(inc)}")
        self._validate_name()
        self._declaring_file = self._get_declaring_file()
        # Automatically register this environment instance in load order
        _ENVIRONMENT_REGISTRY.append(self)

    def add_dependency(self, *env: Environment):
        """
        Add one or more environment dependencies so they are deployed together.

        When you deploy this environment, any environments added via
        `add_dependency` will also be deployed. This is an alternative to
        passing `depends_on=[...]` at construction time, useful when the
        dependency is defined after the environment is created.

        Duplicate dependencies are silently ignored. An environment cannot
        depend on itself.

        :param env: One or more `Environment` instances to add as dependencies.
        """
        for e in env:
            if not isinstance(e, Environment):
                raise TypeError(f"Expected Environment, got {type(e)}")
            if e.name == self.name:
                raise ValueError("Cannot add self as a dependency")
            if e in self.depends_on:
                continue
        self.depends_on.extend(env)

    def clone_with(
        self,
        name: str,
        image: Optional[Union[str, Image, Literal["auto"]]] = None,
        resources: Optional[Resources] = None,
        env_vars: Optional[Dict[str, str]] = None,
        secrets: Optional[SecretRequest] = None,
        depends_on: Optional[List[Environment]] = None,
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> Environment:
        raise NotImplementedError

    def _get_kwargs(self) -> Dict[str, Any]:
        """
        Get the keyword arguments for the environment.
        """
        kwargs: Dict[str, Any] = {
            "depends_on": self.depends_on,
            "image": self.image,
        }
        if self.resources is not None:
            kwargs["resources"] = self.resources
        if self.secrets is not None:
            kwargs["secrets"] = self.secrets
        if self.env_vars is not None:
            kwargs["env_vars"] = self.env_vars
        if self.pod_template is not None:
            kwargs["pod_template"] = self.pod_template
        if self.description is not None:
            kwargs["description"] = self.description
        if self.include:
            kwargs["include"] = self.include
        return kwargs
