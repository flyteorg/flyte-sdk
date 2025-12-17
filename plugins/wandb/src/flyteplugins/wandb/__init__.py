from flyte.models import TaskContext

from .context import (
    get_wandb_context,
    get_wandb_sweep_context,
    wandb_config,
    wandb_sweep_config,
)
from .decorator import wandb_init, wandb_sweep

__all__ = [
    "wandb_config",
    "get_wandb_context",
    "wandb_init",
    "wandb_sweep_config",
    "get_wandb_sweep_context",
    "wandb_sweep",
    "Wandb",
]

__version__ = "0.1.0"


# Add wandb_run property to TaskContext
def _wandb_run_property(self):
    """
    Get the current wandb run if within a @wandb_init decorated task/trace.
    Returns None otherwise.
    """
    if not self.data or not self.custom_context:
        return None

    # Check if current action matches the action that has @wandb_init
    # (action name is stored in custom_context which is shared between tasks)
    current_action = self.action.name
    wandb_action = self.custom_context.get("_wandb_init_action")

    if current_action != wandb_action:
        return None  # Called without @wandb_init decorator

    # Return the run object from task-local data
    return self.data.get("_wandb_run")


# Add wandb_sweep_id property to TaskContext
def _wandb_sweep_id_property(self):
    """
    Get the current wandb sweep_id if within a @wandb_sweep decorated task.
    Returns None otherwise.
    """
    if not self.custom_context:
        return None

    # Return the sweep_id
    return self.custom_context.get("_wandb_sweep_id")


# Monkey-patch the properties onto TaskContext
TaskContext.wandb_run = property(_wandb_run_property)
TaskContext.wandb_sweep_id = property(_wandb_sweep_id_property)
