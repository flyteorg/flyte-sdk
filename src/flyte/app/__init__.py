from flyte.app._app_environment import AppEnvironment
from flyte.app._input import Input
from flyte.app._types import Domain, Link, Port, Scaling

__all__ = [
    "AppEnvironment",
    "Domain",
    "Input",
    "Link",
    "Port",
    "Scaling",
]


def register_app_deployer():
    from flyte import _deployer as deployer
    from flyte.app._deploy import _deploy_app_env

    deployer.register_deployer(AppEnvironment, _deploy_app_env)


register_app_deployer()
