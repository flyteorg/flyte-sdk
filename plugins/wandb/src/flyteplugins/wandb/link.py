from dataclasses import dataclass
from typing import Dict, Optional

from flyte.link import Link

from .context import get_wandb_context


@dataclass
class Wandb(Link):
    """
    Wandb link that dynamically pulls configuration from context.

    This link automatically populates project, entity, and id from the
    wandb context set via with_runcontext() or wandb() context manager.
    """

    project: Optional[str] = None
    entity: Optional[str] = None
    id: Optional[str] = None
    host: str = "https://wandb.ai"

    def __post_init__(self):
        # Link type is always execution-id since we resolve at runtime
        self.link_type = "wandb-execution-id"

    def get_config(self) -> Dict[str, str]:
        """
        Get link configuration, pulling from wandb context if values not set.

        This is called during task serialization, after with_runcontext() has
        been called, so we can access the custom_context.
        """
        # Try to get from context if not explicitly set
        config = get_wandb_context()

        project = self.project
        entity = self.entity
        id_val = self.id

        # Fallback to context values if not explicitly set
        if config:
            if project is None:
                project = config.project
            if entity is None:
                entity = config.entity
            if id_val is None:
                id_val = config.id

        return {
            "link_type": self.link_type,
            "project": project or "",
            "entity": entity or "",
            "id": id_val or "",
            "host": self.host,
        }
