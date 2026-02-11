from dataclasses import dataclass
from typing import Dict, Optional

from flyte import Link

from ._context import RankScope, RunMode


@dataclass
class Wandb(Link):
    """
    Generates a Weights & Biases run link.

    Args:
        host: Base W&B host URL
        project: W&B project name (overrides context config if provided)
        entity: W&B entity/team name (overrides context config if provided)
        run_mode: Determines the link behavior:
            - "auto" (default): Use parent's run if available, otherwise create new
            - "new": Always creates a new wandb run with a unique ID
            - "shared": Always shares the parent's run ID
            In distributed training context (single-node):
            - "auto" (default): Only rank 0 logs.
            - "shared": All ranks log to a single shared W&B run.
            - "new": Each rank gets its own W&B run (grouped in W&B UI).
            Multi-node: behavior depends on `rank_scope`.
        rank_scope: Flyte-specific rank scope - "global" or "worker".
            Controls which ranks log in distributed training.
            run_mode="auto":
            - "global" (default): Only global rank 0 logs (1 run total).
            - "worker": Local rank 0 of each worker logs (1 run per worker).
            run_mode="shared":
            - "global": All ranks log to a single shared W&B run.
            - "worker": Ranks per worker log to a single shared W&B run (1 run per worker).
            run_mode="new":
            - "global": Each rank gets its own W&B run (1 run total).
            - "worker": Each rank gets its own W&B run grouped per worker -> N runs.
        id: Optional W&B run ID (overrides context config if provided)
        name: Link name in the Flyte UI
    """

    host: str = "https://wandb.ai"
    project: Optional[str] = None
    entity: Optional[str] = None
    run_mode: RunMode = "auto"
    rank_scope: RankScope = "global"
    id: Optional[str] = None
    name: str = "Weights & Biases"
    # Internal: set by @wandb_init for distributed training tasks
    _is_distributed: bool = False
    # Internal: worker index for multi-node distributed training (set by @wandb_init)
    _worker_index: Optional[int] = None

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
        run_mode = self.run_mode  # Defaults to "auto"

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

        # Distributed training links - derived from decorator-time info (plugin_config)
        # _is_distributed and _worker_index are set by @wandb_init based on Elastic config
        is_multi_node = self._worker_index is not None
        rank_scope = self.rank_scope  # Defaults to "global"

        if self._is_distributed:
            base_id = user_provided_id or f"{run_name}-{action_name}"

            is_worker_scope = is_multi_node and rank_scope == "worker"
            suffix = f"-worker-{self._worker_index}" if is_worker_scope else ""
            target_id = f"{base_id}{suffix}"

            if run_mode == "new":
                path = "groups"
            else:  # "auto" or "shared"
                path = "runs"

            return f"{self.host}/{wandb_entity}/{wandb_project}/{path}/{target_id}"

        # Non-distributed: link to specific run
        # Determine run ID based on run_mode setting
        if run_mode == "new":
            # Always create new run - use user-provided ID if available, otherwise generate
            wandb_run_id = user_provided_id or f"{run_name}-{action_name}"
        elif run_mode == "shared":
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
    """
    Generates a Weights & Biases Sweep link.

    Args:
        host: Base W&B host URL
        project: W&B project name (overrides context config if provided)
        entity: W&B entity/team name (overrides context config if provided)
        id: Optional W&B sweep ID (overrides context config if provided)
        name: Link name in the Flyte UI
    """

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
