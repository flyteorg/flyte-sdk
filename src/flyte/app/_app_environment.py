from dataclasses import dataclass, field
from typing import Optional, Union, List, Any

import rich.repr

from flyte import Environment
from flyte.app._input import Input
from flyte.app._types import Scaling, Domain, Link


@rich.repr.auto
@dataclass(init=True, repr=True)
class AppEnvironment(Environment):
    """
    :param name: Name of the environment
    :param image: Docker image to use for the environment. If set to "auto", will use the default image.
    :param resources: Resources to allocate for the environment.
    :param env_vars: Environment variables to set for the environment.
    :param secrets: Secrets to inject into the environment.
    :param depends_on: Environment dependencies to hint, so when you deploy the environment, the dependencies are
        also deployed. This is useful when you have a set of environments that depend on each other.
    """

    type: Optional[str] = None
    framework_app: Optional[Any] = None
    docs: Optional[str] = None
    args: Optional[Union[List[str], str]] = None
    command: Optional[Union[List[str], str]] = None
    requires_auth: bool = True
    scaling: Scaling = field(default_factory=Scaling)
    domain: Domain | None = field(default_factory=Domain)
    # Integration
    links: List[Link] = field(default_factory=list)

    # Code
    include: List[str] = field(default_factory=list)
    inputs: List[Input] = field(default_factory=list)

    # queue / cluster_pool
    cluster_pool: str = "default"

    # config: Optional[AppConfigProtocol] = None

    def __post_init__(self):
        super().__post_init__()
        if self.args is not None and not isinstance(self.args, (list, str)):
            raise TypeError(f"Expected args to be of type List[str] or str, got {type(self.args)}")
        if self.command is not None and not isinstance(self.command, (list, str)):
            raise TypeError(f"Expected command to be of type List[str] or str, got {type(self.command)}")
        if not isinstance(self.scaling, Scaling):
            raise TypeError(f"Expected scaling to be of type Scaling, got {type(self.scaling)}")
        if not isinstance(self.domain, (Domain, type(None))):
            raise TypeError(f"Expected domain to be of type Domain or None, got {type(self.domain)}")
        for link in self.links:
            if not isinstance(link, Link):
                raise TypeError(f"Expected links to be of type List[Link], got {type(link)}")
