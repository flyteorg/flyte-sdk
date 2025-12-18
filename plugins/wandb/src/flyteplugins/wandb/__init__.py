from flyte.models import TaskContext

from .context import (
    get_wandb_context,
    get_wandb_sweep_context,
    wandb_config,
    wandb_sweep_config,
)
from .decorator import wandb, wandb_init, wandb_sweep

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

    Uses lazy initialization: if the run hasn't been initialized yet, this will
    call wandb.init() on first access. This ensures ctx.action.name is read after
    all decorators (including @flyte.trace) have set up their contexts.

    Returns None otherwise.
    """
    if not self.data or not self.custom_context:
        return None

    # Check if run is already initialized
    run = self.data.get("_wandb_run")
    if run:
        # Verify current action matches the action that has @wandb_init
        current_action = self.action.name
        wandb_action = self.custom_context.get("_wandb_init_action")
        if current_action == wandb_action:
            return run
        return None

    # Check if we have init kwargs for lazy initialization
    init_kwargs_data = self.data.get("_wandb_init_kwargs")
    if not init_kwargs_data:
        return None

    new_run = init_kwargs_data["new_run"]
    init_kwargs = init_kwargs_data["init_kwargs"].copy()
    saved_run_id = init_kwargs_data["saved_run_id"]

    current_action = self.action.name

    # Mark which action has @wandb_init
    self.custom_context["_wandb_init_action"] = current_action

    # Determine run ID using the current action name
    if "id" not in init_kwargs or init_kwargs["id"] is None:
        if new_run or not saved_run_id:
            # Create new run ID with current action name
            init_kwargs["id"] = f"{self.action.run_name}-{current_action}"
            if "reinit" not in init_kwargs:
                init_kwargs["reinit"] = "create_new"
        else:
            if not saved_run_id:
                raise RuntimeError("Expected saved_run_id when reusing parent's run ID")
            # Reuse parent's run ID
            init_kwargs["id"] = saved_run_id

    # Configure shared mode settings
    is_primary = new_run or not saved_run_id
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

    # Store run ID in custom_context (shared with child tasks)
    self.custom_context["_wandb_run_id"] = run.id

    # Store run object in ctx.data (task-local only)
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
