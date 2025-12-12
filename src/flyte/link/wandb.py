from dataclasses import dataclass
from typing import Dict, Optional

from flyte.link import Link


@dataclass
class Wandb(Link):
    project: str
    entity: str
    id: Optional[str] = None
    host: str = "https://wandb.ai"

    def __post_init__(self):
        if self.id is None:
            self.link_type = "wandb-execution-id"
        else:
            self.link_type = "wandb-custom-id"

    def get_config(
        self,
        run_name: str,
        project: str,
        domain: str,
        context: Dict[str, str],
        parent_action_name: str,
        action_name: str,
    ) -> Dict[str, str]:
        return {
            "link_type": self.link_type,
            "project": self.project,
            "entity": self.entity,
            "id": self.id if self.id else "",
            "host": self.host,
        }
