"""Hydra plugin registration for FlyteLauncher.

Hydra requires launcher plugin targets to live under the ``hydra_plugins``
namespace. The implementation lives in ``flyteplugins.hydra._launcher``; this
module exposes a thin Hydra-discoverable subclass and registers its config.
"""

from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore

from flyteplugins.hydra._launcher import FlyteLauncher as _FlyteLauncher


class FlyteLauncher(_FlyteLauncher):
    """Hydra-discoverable wrapper for the Flyte launcher implementation."""


@dataclass
class FlyteLauncherConfig:
    """Configuration schema for the Flyte Hydra launcher.

    Select with ``hydra/launcher=flyte`` on the CLI or in ``defaults``.
    """

    _target_: str = "hydra_plugins.hydra_flyte_launcher.FlyteLauncher"
    mode: str = "remote"  # "remote" | "local"
    wait: bool = True
    wait_max_workers: Optional[int] = 32


ConfigStore.instance().store(
    group="hydra/launcher",
    name="flyte",
    node=FlyteLauncherConfig,
)
