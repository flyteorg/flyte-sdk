"""Fixture plugin config dataclasses used by `flyte run python-script --plugin-config` tests."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DummyNodeConfig:
    group_name: str
    replicas: int
    min_replicas: Optional[int] = None


@dataclass
class DummyPluginConfig:
    nodes: List[DummyNodeConfig]
    enabled: bool = False
    runtime_env: Optional[Dict[str, str]] = None
