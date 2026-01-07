from dataclasses import dataclass
from typing import Dict, Optional

from flyte import Link


@dataclass
class Wandb(Link):
    project: str
    entity: str
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
        # Try to get the actual run ID from context (handles user-provided IDs)
        wandb_run_id = None
        if context:
            wandb_run_id = context.get("_wandb_run_id")

        # Fallback: construct the run ID dynamically using Flyte's run_name and action_name
        # This matches the pattern used in the wandb_init decorator
        if not wandb_run_id:
            wandb_run_id = f"{run_name}-{action_name}"

        return f"https://wandb.ai/{self.entity}/{self.project}/runs/{wandb_run_id}"


@dataclass
class WandbSweep(Link):
    project: str
    entity: str
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
        # Try to get sweep_id from context
        sweep_id = None
        if context:
            sweep_id = context.get("_wandb_sweep_id")

        # If we have a sweep_id, return the specific sweep URL
        if sweep_id:
            return f"https://wandb.ai/{self.entity}/{self.project}/sweeps/{sweep_id}"

        # Otherwise, return the sweeps list URL as a fallback
        return f"https://wandb.ai/{self.entity}/{self.project}/sweeps"

