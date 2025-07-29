from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import rich.repr

from flyte._secret import SecretRequest

from ._image import Image
from ._resources import Resources

if TYPE_CHECKING:
    from kubernetes.client import V1PodTemplate


def is_snake_or_kebab_with_numbers(s: str) -> bool:
    return re.fullmatch(r"^[a-z0-9]+([_-][a-z0-9]+)*$", s) is not None


@rich.repr.auto
@dataclass(init=True, repr=True)
class Environment:
    """
    :param name: Name of the environment
    :param image: Docker image to use for the environment. If set to "auto", will use the default image.
    :param resources: Resources to allocate for the environment.
    :param env: Environment variables to set for the environment.
    :param secrets: Secrets to inject into the environment.
    :param depends_on: Environment dependencies to hint, so when you deploy the environment, the dependencies are
        also deployed. This is useful when you have a set of environments that depend on each other.
    """

    name: str
    depends_on: List[Environment] = field(default_factory=list)
    pod_template: Optional[Union[str, "V1PodTemplate"]] = None
    description: Optional[str] = None
    secrets: Optional[SecretRequest] = None
    env: Optional[Dict[str, str]] = None
    resources: Optional[Resources] = None
    image: Union[str, Image, Literal["auto"]] = "auto"

    def __post_init__(self):
        if not is_snake_or_kebab_with_numbers(self.name):
            raise ValueError(f"Environment name '{self.name}' must be in snake_case or kebab-case format.")

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
        env: Optional[Dict[str, str]] = None,
        secrets: Optional[SecretRequest] = None,
        depends_on: Optional[List[Environment]] = None,
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
        if self.env is not None:
            kwargs["env"] = self.env
        if self.pod_template is not None:
            kwargs["pod_template"] = self.pod_template
        if self.description is not None:
            kwargs["description"] = self.description
        return kwargs
