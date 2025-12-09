from dataclasses import dataclass
from typing import Dict, Optional

from flyte.link import Link


@dataclass
class Wandb(Link):
    project: str
    entity: str
    id: str
    host: Optional[str] = "https://wandb.ai"
    port: Optional[str] = None

    def get_config(self) -> Dict[str, str]:
        if id is None:
            link_type = "wandb-execution-id"
        else:
            link_type = "wandb-custom-id"

        return {
            "link_type": link_type,
            "project": self.project,
            "entity": self.entity,
            "id": self.id,
            "host": self.host,
            "port": self.port,
        }
