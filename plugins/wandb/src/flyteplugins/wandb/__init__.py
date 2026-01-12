import logging

from flyte.models import TaskContext

logger = logging.getLogger(__name__)

from .context import (
    get_wandb_context,
    get_wandb_sweep_context,
    wandb_config,
    wandb_sweep_config,
)
from .decorator import wandb_init, wandb_sweep
from .link import Wandb, WandbSweep

__all__ = [
    "Wandb",
    "WandbSweep",
    "get_wandb_context",
    "get_wandb_sweep_context",
    "wandb_config",
    "wandb_init",
    "wandb_sweep",
    "wandb_sweep_config",
]

__version__ = "0.1.0"


# Add wandb_run property to TaskContext
def _wandb_run_property(self):
    """
    Get the current wandb run if within a @wandb_init decorated task or trace.

    The run is initialized when the @wandb_init context manager is entered.
    This property provides convenient access to the initialized run object.
    """
    if not self.data:
        return None

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
