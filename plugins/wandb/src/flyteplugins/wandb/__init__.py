import logging

import wandb

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
    "wandb_config",
    "get_wandb_context",
    "wandb_init",
    "wandb_sweep_config",
    "get_wandb_sweep_context",
    "wandb_sweep",
    "Wandb",
    "WandbSweep",
]

__version__ = "0.1.0"


# Add wandb_run property to TaskContext
def _wandb_run_property(self):
    """
    Get the current wandb run if within a @wandb_init decorated task or trace.

    Uses lazy initialization, i.e., the run is created when you access this property `flyte.ctx().wandb_run`.
    """
    if not self.data or not self.custom_context:
        return None

    # Check if run is already initialized for this action - is available for traces only
    run = self.data.get("_wandb_run")
    if run:
        # This is a trace and parent's run is already initialized
        return run

    current_action = self.action.name

    # Check if we have init kwargs
    init_kwargs_data = self.data.get("_wandb_init_kwargs")
    if not init_kwargs_data:
        logger.debug(f"No init kwargs found for action '{current_action}'")
        return None

    new_run = init_kwargs_data["new_run"]
    init_kwargs = init_kwargs_data["init_kwargs"].copy()
    saved_run_id = init_kwargs_data["saved_run_id"]

    # Determine if we should reuse parent's run or create new
    should_reuse = False
    if new_run == False:
        should_reuse = True
    elif new_run == "auto":
        # Auto: reuse if parent exists, otherwise create new
        should_reuse = bool(saved_run_id)
    # else: new_run == True, create new (should_reuse stays False)

    # Determine run ID using the current action name
    if "id" not in init_kwargs or init_kwargs["id"] is None:
        if should_reuse:
            # Reuse parent's run ID
            if not saved_run_id:
                raise RuntimeError("Cannot reuse parent run: no parent run ID found")
            init_kwargs["id"] = saved_run_id
        else:
            # Create new run ID
            init_kwargs["id"] = f"{self.action.run_name}-{current_action}"
            if "reinit" not in init_kwargs:
                init_kwargs["reinit"] = "create_new"

    # Configure shared mode settings
    is_primary = not should_reuse

    existing_settings = init_kwargs.get("settings", {})
    shared_config = {
        "mode": "shared",
        "x_primary": is_primary,
    }
    if not is_primary:
        shared_config["x_update_finish_state"] = False

    # Merge and create Settings object
    init_kwargs["settings"] = wandb.Settings(**{**existing_settings, **shared_config})

    # Initialize wandb
    run = wandb.init(**init_kwargs)

    # Store run ID, project, and entity in custom_context (shared with child tasks and accessible to links)
    self.custom_context["_wandb_run_id"] = run.id
    self.custom_context["_wandb_project"] = run.project
    self.custom_context["_wandb_entity"] = run.entity

    # Store run object in ctx.data (task-local only and accessible to traces)
    self.data["_wandb_run"] = run

    return run


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
