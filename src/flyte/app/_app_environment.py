from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional, Union, List, Any, Literal as L

import rich.repr

from flyte import Environment
from flyte.app._app_environment_bk import Input, AppConfigProtocol, Link
from flyte.app._app import App


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

    @dataclass
    class Port:
        port: int
        name: Optional[str] = None

    port: Optional[Union[int, Port]] = None
    args: Optional[Union[List[str], str]] = None
    min_replicas: int = 0
    max_replicas: int = 1
    scaledown_after: Optional[Union[int, timedelta]] = None
    scaling_metric: Optional[Union[ScalingMetric.Concurrency, ScalingMetric.RequestRate]] = None
    include: List[str] = field(default_factory=list)
    inputs: List[Input] = field(default_factory=list)
    env: dict = field(default_factory=dict)
    cluster_pool: str = "default"
    requires_auth: bool = True
    type: Optional[str] = None
    description: Optional[str] = None
    framework_app: Optional[Any] = None
    config: Optional[AppConfigProtocol] = None
    subdomain: Optional[str] = None
    custom_domain: Optional[str] = None
    links: List[Link] = field(default_factory=list)
    shared_memory: Optional[Union[L[True], str]] = None

    @property
    def app(self) -> "App":
        """
        Return the app in the environment.
        """
        return App(
            name=self.name,
            resources=self.resources,
            env=self.env_vars,
            secrets=self.secrets,
            dependencies=[app_env.app for app_env in self.depends_on if isinstance(app_env, AppEnvironment)]
            if self.depends_on
            else None,
            image=self.pod_template or self.image,
            port=self.port,
            args=self.args,
            min_replicas=self.min_replicas,
            max_replicas=self.max_replicas,
            scaledown_after=self.scaledown_after,
            scaling_metric=self.scaling_metric,
            include=self.include,
            inputs=self.inputs,
            cluster_pool=self.cluster_pool,
            requires_auth=self.requires_auth,
            type=self.type,
            description=self.description,
            framework_app=self.framework_app,
            config=self.config,
            subdomain=self.subdomain,
            custom_domain=self.custom_domain,
            links=self.links,
            shared_memory=self.shared_memory,
        )
