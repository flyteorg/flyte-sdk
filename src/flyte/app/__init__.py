from flyte.app._app_environment import AppEnvironment

__all__ = [
    "AppEnvironment",
]


def register_app_deployer():
    from flyte import _deployer as deployer
    from flyte.app._deploy import _deploy_app_env

    deployer.register_deployer(AppEnvironment, _deploy_app_env)


register_app_deployer()
