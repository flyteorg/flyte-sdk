from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union, List, Any, Literal, Dict

from flyte import Image, PodTemplate, Resources, SecretRequest
from flyte.app._app_environment_bk import Input, AppConfigProtocol, URLQuery
from flyte.app._types import Scaling, Domain, Link


@dataclass
class App:
    """
    App specification.

    :param name: The name of the application.
    :param type: Type of app.
    :param framework_app: Object for serving framework. When this is set, all user defined files are uploaded.
        - For FastAPI, args is set to `uvicorn module_name:app_name --port port`. For more control
          you can set `args` directly.
    :param image: The container image to use for the application.
    :param resources: Compute resource requests and limits for the application.
    :param interruptible: Whether the app can be interrupted.
    :param docs: Documentation for the app.
    :param env_vars: Environment variables to set for the application.
    :param secrets: Secrets that are requested for the application.
    :param pod_template: Pod template to use for the application.
    :param args: Entrypoint to start application.
    :param command: Command to start application.
    :param requires_auth: Whether the public URL requires authentication.
    :param scaling: Scaling configuration for the application.
    :param domain: Domain configuration for the application.
    :param links: Links to external URLs or relative paths.
    :param dependencies: List of apps that this app depends on.
    :param include: Files to include for your application.
    :param inputs: Inputs for the application.
    :param cluster_pool: The target cluster_pool where the app should be deployed.
    :param config: App configuration protocol.
    """

    name: str
    type: Optional[str] = None
    framework_app: Optional[Any] = None
    image: Union[str, Image, Literal["auto"]] = "auto"
    resources: Optional[Resources] = None
    interruptible: bool = False
    docs: Optional[str] = None
    env_vars: Optional[Dict[str, str]] = None
    secrets: Optional[SecretRequest] = None
    pod_template: Optional[Union[str, PodTemplate]] = None
    args: Optional[Union[List[str], str]] = None
    command: Optional[Union[List[str], str]] = None
    requires_auth: bool = True
    scaling: Scaling = field(default_factory=Scaling)
    domain: Domain | None = field(default_factory=Domain)
    # Integration
    links: List[Link] = field(default_factory=list)

    # App Dependencies
    dependencies: List[App] = field(default_factory=list)

    # Code
    include: List[str] = field(default_factory=list)
    inputs: List[Input] = field(default_factory=list)

    # queue / cluster_pool
    cluster_pool: str = "default"

    config: Optional[AppConfigProtocol] = None

    def __post_init__(self):
        if not isinstance(self.image, (Image, str)):
            raise TypeError(f"Expected image to be of type str or Image, got {type(self.image)}")
        if self.secrets and not isinstance(self.secrets, (str, SecretRequest, List)):
            raise TypeError(f"Expected secrets to be of type SecretRequest, got {type(self.secrets)}")
        if self.resources is not None and not isinstance(self.resources, Resources):
            raise TypeError(f"Expected resources to be of type Resources, got {type(self.resources)}")
        if self.env_vars is not None and not isinstance(self.env_vars, dict):
            raise TypeError(f"Expected env_vars to be of type Dict[str, str], got {type(self.env_vars)}")
        if self.pod_template is not None and not isinstance(self.pod_template, (str, PodTemplate)):
            raise TypeError(f"Expected pod_template to be of type str or PodTemplate, got {type(self.pod_template)}")
        if self.args is not None and not isinstance(self.args, (list, str)):
            raise TypeError(f"Expected args to be of type List[str] or str, got {type(self.args)}")
        if self.command is not None and not isinstance(self.command, (list, str)):
            raise TypeError(f"Expected command to be of type List[str] or str, got {type(self.command)}")
        if not isinstance(self.scaling, Scaling):
            raise TypeError(f"Expected scaling to be of type Scaling, got {type(self.scaling)}")
        if not isinstance(self.domain, (Domain, type(None))):
            raise TypeError(f"Expected domain to be of type Domain or None, got {type(self.domain)}")
        for app in self.dependencies:
            if not isinstance(app, App):
                raise TypeError(f"Expected dependencies to be of type List[App], got {type(app)}")
        for link in self.links:
            if not isinstance(link, Link):
                raise TypeError(f"Expected links to be of type List[Link], got {type(link)}")

    def query_endpoint(self, *, public: bool = False) -> URLQuery:
        """
        Query for endpoint.

        :param public: Whether to return the public or internal endpoint.
        :returns: Object representing a URL query.
        """
        return URLQuery(name=self.name, public=public)

    @property
    def endpoint(self) -> str:
        """
        Return endpoint for App.
        """
        raise NotImplementedError
