from dataclasses import dataclass
from typing import Dict, Literal, Optional

from flyte import Link

RunMode = Literal["auto", "new", "shared"]


@dataclass
class Wandb(Link):
    host: str = "https://wandb.ai"
    project: Optional[str] = None
    entity: Optional[str] = None
    run_mode: RunMode = "auto"
    id: Optional[str] = None
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
        **kwargs,
    ) -> str:
        # Get project and entity from decorator values or context
        wandb_project = self.project
        wandb_entity = self.entity
        wandb_run_id = None
        user_provided_id = self.id  # Prioritize ID provided at link creation time

        if context:
            # Try to get from context if not provided at decoration time
            if not wandb_project:
                wandb_project = context.get("wandb_project")
            if not wandb_entity:
                wandb_entity = context.get("wandb_entity")

            # Get parent's run ID if available (set by parent task)
            parent_run_id = context.get("_wandb_run_id")

            # Check if user provided a custom run ID in wandb_config (lower priority than self.id)
            if not user_provided_id:
                user_provided_id = context.get("wandb_id")
        else:
            parent_run_id = None

        # If we don't have project/entity, we can't create a valid link
        if not wandb_project or not wandb_entity:
            return self.host

        # Determine run ID based on run_mode setting
        if self.run_mode == "new":
            # Always create new run - use user-provided ID if available, otherwise generate
            wandb_run_id = user_provided_id or f"{run_name}-{action_name}"
        elif self.run_mode == "shared":
            # Always reuse parent's run
            if parent_run_id:
                wandb_run_id = parent_run_id
            else:
                # Can't generate link without parent run ID
                return f"{self.host}/{wandb_entity}/{wandb_project}"
        else:  # run_mode == "auto"
            # Use parent's run if available, otherwise create new
            if parent_run_id:
                wandb_run_id = parent_run_id
            else:
                wandb_run_id = user_provided_id or f"{run_name}-{action_name}"

        return f"{self.host}/{wandb_entity}/{wandb_project}/runs/{wandb_run_id}"


@dataclass
class WandbSweep(Link):
    host: str = "https://wandb.ai"
    project: Optional[str] = None
    entity: Optional[str] = None
    id: Optional[str] = None
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
        **kwargs,
    ) -> str:
        # Get project and entity from decorator values or context
        wandb_project = self.project
        wandb_entity = self.entity
        sweep_id = self.id  # Prioritize ID provided at link creation time

        if context:
            # Try to get from context config if not provided at decoration time
            if not wandb_project:
                wandb_project = context.get("wandb_project")
            if not wandb_entity:
                wandb_entity = context.get("wandb_entity")

            # Try to get the sweep_id from context if not provided at link creation
            # Child tasks inherit this from the parent that created the sweep
            if not sweep_id:
                sweep_id = context.get("_wandb_sweep_id")

        # If we don't have project/entity, return base URL
        if not wandb_project or not wandb_entity:
            return self.host

        # If we have a sweep_id, link to specific sweep
        if sweep_id:
            return f"{self.host}/{wandb_entity}/{wandb_project}/sweeps/{sweep_id}"

        # No sweep_id: link to the project's sweeps list page
        return f"{self.host}/{wandb_entity}/{wandb_project}/sweeps"
