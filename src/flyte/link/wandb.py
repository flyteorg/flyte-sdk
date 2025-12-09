from dataclasses import dataclass
from typing import Dict

from flyte.link import Link


@dataclass
class Wandb(Link):
    project: str
    entity: str
    id: str
    host: str = "https://wandb.ai"

    def __post_init__(self):
        if self.id is None:
            self.link_type = "wandb-execution-id"
        else:
            self.link_type = "wandb-custom-id"

    def get_config(self) -> Dict[str, str]:
        return {
            "link_type": self.link_type,
            "project": self.project,
            "entity": self.entity,
            "id": self.id,
            "host": self.host,
            "port": self.port,
        }
