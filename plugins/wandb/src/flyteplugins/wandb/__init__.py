from flyte.models import TaskContext

from .context import get_wandb_context, wandb_config
from .decorator import wandb_init
from .link import Wandb

__all__ = ["wandb_config", "get_wandb_context", "wandb_init", "Wandb"]

__version__ = "0.1.0"


# Add wandb_run property to TaskContext
def _wandb_run_property(self):
    """
    Get the current wandb run if within a @wandb_init decorated function.
    Returns None if called from a child function without @wandb_init.
    """
    if not self.data:
        return None

    # Check if current action matches the action that has @wandb_init
    current_action = self.action.name
    wandb_action = self.data.get("_wandb_init_action")

    if current_action != wandb_action:
        return None  # Called from different action without @wandb_init

    # Return the run object directly
    return self.data.get("_wandb_run")


# Monkey-patch the property onto TaskContext
TaskContext.wandb_run = property(_wandb_run_property)
