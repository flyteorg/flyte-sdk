import importlib
from pathlib import Path

from flyte._internal.resolvers._app_env_module import extract_app_env_module
from flyte._internal.resolvers.common import Resolver
from flyte.app._app_environment import AppEnvironment


class AppEnvResolver(Resolver):
    """
    Please see the notes in the TaskResolverMixin as it describes this default behavior.
    """

    @property
    def import_path(self) -> str:
        return "flyte._internal.resolvers.app_env.AppEnvResolver"

    def load_app_env(self, loader_args: str) -> AppEnvironment:
        module_name, app_var_name = loader_args.split(":")
        app_env_module = importlib.import_module(name=module_name)  # type: ignore
        app_env_def = getattr(app_env_module, app_var_name)
        return app_env_def

    def loader_args(self, app_env: AppEnvironment, root_dir: Path) -> str:  # type:ignore
        app_var_name, module_name = extract_app_env_module(app_env, root_dir)
        return f"{module_name}:{app_var_name}"
