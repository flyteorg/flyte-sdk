from dataclasses import dataclass
from typing import Dict, Optional

from flyte import Link


@dataclass
class Wandb(Link):
    host: str = "https://wandb.ai"
    project: Optional[str] = None
    entity: Optional[str] = None
    new_run: bool | str = "auto"
    name: str = "Weights & Biases"

    def get_link(
        self,
        run_name: str,
        project: str,
        domain: str,
        context: Dict[str, str],
        parent_action_name: str,
        action_name: str,
        pod_name: str,
    ) -> str:
        # Get project and entity from decorator values or runtime context
        wandb_project = self.project
        wandb_entity = self.entity
        wandb_run_id = None

        if context:
            # Try to get from context if not provided at decoration time
            # These are stored when wandb.init() is called
            if not wandb_project:
                wandb_project = context.get("_wandb_project")
            if not wandb_entity:
                wandb_entity = context.get("_wandb_entity")

            # Get parent's run ID if available
            parent_run_id = context.get("_wandb_run_id")
        else:
            parent_run_id = None

        # If we don't have project/entity, we can't create a valid link
        if not wandb_project or not wandb_entity:
            return self.host

        # Determine run ID based on new_run setting
        if self.new_run == True:
            # Always create new run - generate ID for this task
            wandb_run_id = f"{run_name}-{action_name}"
        elif self.new_run == False:
            # Always reuse parent's run
            if parent_run_id:
                wandb_run_id = parent_run_id
            else:
                # Can't generate link without parent run ID
                return f"{self.host}/{wandb_entity}/{wandb_project}"
        else:  # new_run == "auto"
            # Use parent's run if available, otherwise create new
            if parent_run_id:
                wandb_run_id = parent_run_id
            else:
                wandb_run_id = f"{run_name}-{action_name}"

        return f"{self.host}/{wandb_entity}/{wandb_project}/runs/{wandb_run_id}"


@dataclass
class WandbSweep(Link):
    host: str = "https://wandb.ai"
    project: Optional[str] = None
    entity: Optional[str] = None
    name: str = "Weights & Biases Sweep"

    def get_link(
        self,
        run_name: str,
        project: str,
        domain: str,
        context: Dict[str, str],
        parent_action_name: str,
        action_name: str,
        pod_name: str,
    ) -> str:
        # Get project and entity from decorator values or runtime context
        wandb_project = self.project
        wandb_entity = self.entity

        sweep_id = None
        if context:
            # Try to get from context if not provided at decoration time
            # These are stored when wandb.init() is called
            if not wandb_project:
                wandb_project = context.get("_wandb_project")
            if not wandb_entity:
                wandb_entity = context.get("_wandb_entity")

            # Try to get sweep_id from context
            sweep_id = context.get("_wandb_sweep_id")

        # If we don't have project/entity, return base URL
        if not wandb_project or not wandb_entity:
            return self.host

        # If we have a sweep_id, return the specific sweep URL
        if sweep_id:
            return f"{self.host}/{wandb_entity}/{wandb_project}/sweeps/{sweep_id}"

        # Otherwise, return the sweeps list URL as a fallback
        return f"{self.host}/{wandb_entity}/{wandb_project}/sweeps"
