from flyte.app._app_environment import AppEnvironment
from flyte.app._connector_environment import ConnectorEnvironment
from flyte.app._parameter import AppEndpoint, Parameter, RunOutput, get_parameter
from flyte.app._types import Domain, Link, Port, Scaling

__all__ = [
    "AppEndpoint",
    "AppEnvironment",
    "ConnectorEnvironment",
    "Domain",
    "Link",
    "Parameter",
    "Port",
    "RunOutput",
    "Scaling",
    "get_parameter",
]


def register_app_deployer():
    from flyte import _deployer as deployer
    from flyte.app._deploy import _deploy_app_env

    deployer.register_deployer(AppEnvironment, _deploy_app_env)


register_app_deployer()
