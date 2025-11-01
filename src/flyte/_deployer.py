from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Tuple, Type

import rich.repr

from flyte.models import SerializationContext

from ._deploy import _deploy_task_env
from ._environment import Environment
from ._task_environment import TaskEnvironment


@rich.repr.auto
@dataclass
class DeploymentContext:
    """
    Context for deployment operations.
    """

    environment: Environment
    serialization_context: SerializationContext
    dryrun: bool = False


class DeployedEnvironment(Protocol):
    """
    Protocol for deployed environment representations.
    """

    def get_name(self) -> str:
        """
        Returns the name of the deployed environment.
        Returns:
        """
        ...

    def env_repr(self) -> List[Tuple[str, ...]]:
        """
        Returns a detailed representation of the deployed environment.
        Returns:
        """
        ...

    def table_repr(self) -> List[List[Tuple[str, ...]]]:
        """
        Returns a detailed representation of the deployed entities in the environment, useful for tabular display.
        Returns:

        """
        ...

    def summary_repr(self) -> str:
        """
        Returns a summary representation of the deployed environment.
        Returns:
        """
        ...


class Deployer(Protocol):
    """
    Protocol for deployment callables.
    """

    async def __call__(self, context: DeploymentContext) -> DeployedEnvironment:
        """
        Deploy the environment described in the context.

        Args:
            context: Deployment context containing environment, serialization context, and dryrun flag

        Returns:
            Deployment result
        """
        ...


_ENVTYPE_REGISTRY: Dict[Type[Environment], Deployer] = {
    TaskEnvironment: _deploy_task_env,
}


def register_deployer(env_type: Type[Environment], deployer: Deployer) -> None:
    """
    Register a deployer for a specific environment type.

    Args:
        env_type: Type of environment this deployer handles
        deployer: Deployment callable that conforms to the Deployer protocol
    """
    _ENVTYPE_REGISTRY[env_type] = deployer


def get_deployer(env_type: Type[Environment | TaskEnvironment]) -> Deployer:
    """
    Get the registered deployer for an environment type.

    Args:
        env_type: Type of environment to get deployer for

    Returns:
        Deployer for the environment type, defaults to task environment deployer
    """
    for tpe, v in _ENVTYPE_REGISTRY.items():
        if issubclass(env_type, tpe):
            return v
    raise ValueError(f"No deployer registered for environment type {env_type}")
