from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

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
    :param name: Name of the environment
    :param image: Docker image to use for the environment. If set to "auto", will use the default image.
    :param resources: Resources to allocate for the environment.
    :param env_vars: Environment variables to set for the environment.
    :param secrets: Secrets to inject into the environment.
    :param pod_template: Pod template to use for the environment.
    :param description: Description of the environment.
    :param interruptible: Whether the environment is interruptible and can be scheduled on spot/preemptible instances
    :param depends_on: Environment dependencies to hint, so when you deploy the environment, the dependencies are
        also deployed. This is useful when you have a set of environments that depend on each other.
    """

    name: str
    depends_on: List[Environment] = field(default_factory=list)
    pod_template: Optional[Union[str, PodTemplate]] = None
    description: Optional[str] = None
    secrets: Optional[SecretRequest] = None
    env_vars: Optional[Dict[str, str]] = None
    resources: Optional[Resources] = None
    interruptible: bool = False
    image: Union[str, Image, Literal["auto"]] = "auto"

    def _validate_name(self):
        if not is_snake_or_kebab_with_numbers(self.name):
            raise ValueError(f"Environment name '{self.name}' must be in snake_case or kebab-case format.")

    def __post_init__(self):
        if not isinstance(self.image, (Image, str)):
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
        self._validate_name()
        # Automatically register this environment instance in load order
        _ENVIRONMENT_REGISTRY.append(self)

    def add_dependency(self, *env: Environment):
        """
        Add a dependency to the environment.
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
        return kwargs
