import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from flyte.extend import AsyncFunctionTaskTemplate, TaskPluginRegistry


@dataclass(kw_only=True)
class WandbConfig(object):
    """
    Wandb configuration for tracking experiments.

    Attributes:
        project (str): the name of the project where you're sending the new run. (Required)
        wandb_api_key (Optional[str]): API key for Weights & Biases.
        api_host (str, optional): URL to your API Host, The default is "https://api.wandb.ai".
        entity (str, optional):
        id (str, optional): A unique id for this wandb run.
        **init_kwargs (dict): The rest of the arguments are passed directly to `wandb.init`.
        Please see [the `wandb.init` docs](https://docs.wandb.ai/ref/python/init) for details.
    """

    project: str
    wandb_api_key: Optional[str] = None
    api_host: Optional[str] = None
    entity: Optional[str] = None
    id: Optional[str] = None
    init_kwargs: Optional[Dict[str, Any]] = None


@dataclass(kw_only=True)
class WandbTracker(AsyncFunctionTaskTemplate):
    """
    Actual Plugin that track
    Attributes:
        plugin_config (Wandb): Configuration for Weights & Biases tracking.
        task_type (str): Type of the task, It should be "wandb".
    """

    plugin_config: WandbConfig
    task_type: str = "wandb"

    async def execute(self, *args, **kwargs) -> Any:
        import wandb

        if self.plugin_config.init_kwargs is None:
            self.plugin_config.init_kwargs = {}

        if not self.plugin_config:
            raise ValueError("wandb configuration is not provided.")

        wandb.login(key=self.plugin_config.wandb_api_key, host=self.plugin_config.api_host)

        # Initialize wandb with the provided configuration
        run = wandb.init(
            project=self.plugin_config.project,
            entity=self.plugin_config.entity,
            id=self.plugin_config.id,
            **self.plugin_config.init_kwargs,
        )

        result = await super().execute(*args, **kwargs)

        execution_url = os.getenv("FLYTE_EXECUTION_URL")
        if execution_url is not None:
            notes = [f"[Execution URL]({execution_url})"]
            if run.notes:
                notes.append(run.notes)
            run.notes = os.linesep.join(notes)

        run.finish()
        print(f":) Wandb run finished: {run.name} ({run.id})")

        return result


TaskPluginRegistry.register(WandbConfig, WandbTracker)
