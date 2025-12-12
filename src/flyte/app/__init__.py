from flyte.app._app_environment import AppEnvironment
from flyte.app._input import AppEndpoint, Input, RunOutput, get_input
from flyte.app._types import Domain, Link, Port, Scaling

__all__ = ["AppEndpoint", "AppEnvironment", "Domain", "Input", "Link", "Port", "RunOutput", "Scaling", "get_input"]


def register_app_deployer():
    from flyte import _deployer as deployer
    from flyte.app._deploy import _deploy_app_env

    deployer.register_deployer(AppEnvironment, _deploy_app_env)


register_app_deployer()
