"""
CLI commands for storing artifacts from remote registries.
"""

import typing
from pathlib import Path

import rich_click as click
from rich.console import Console

from flyte.cli._common import CommandBase
from flyte._resources import Accelerators
from flyte.remote import Run
from flyte.syncify import syncify

# Get all valid accelerator choices from the Accelerators literal type
ACCELERATOR_CHOICES = list(typing.get_args(Accelerators))


@click.group(name="store")
def store():
    """
    Store artifacts from remote registries.

    These commands help you download and store artifacts like HuggingFace models
    to your Flyte storage for faster access during task execution.
    """


@store.command(name="hf-model", cls=CommandBase)
@click.argument("repo", type=str)
@click.option(
    "--artifact-name",
    type=str,
    required=False,
    default=None,
    help=(
        "Artifact name to use for the stored model. Must only contain alphanumeric characters, "
        "underscores, and hyphens. If not provided, the repo name will be used (replacing '.' with '-')."
    ),
)
@click.option(
    "--architecture",
    type=str,
    help="Model architecture, as given in HuggingFace config.json. For non-transformer models use XGBoost, Custom, etc.",
)
@click.option(
    "--task",
    default="auto",
    type=str,
    help=(
        "Model task, e.g., 'generate', 'classify', 'embed', 'score', etc. "
        "Refer to vLLM docs. 'auto' will try to discover this automatically."
    ),
)
@click.option(
    "--modality",
    type=str,
    multiple=True,
    default=("text",),
    help="Modalities supported by the model, e.g., 'text', 'image', 'audio', 'video'. Can be specified multiple times.",
)
@click.option(
    "--format",
    "serial_format",
    type=str,
    help="Model serialization format, e.g., safetensors, onnx, torchscript, joblib, etc.",
)
@click.option(
    "--model-type",
    type=str,
    help=(
        "Model type, e.g., 'transformer', 'xgboost', 'custom', etc. "
        "For HuggingFace models, this is auto-determined from config.json['model_type']."
    ),
)
@click.option(
    "--short-description",
    type=str,
    help="Short description of the model.",
)
@click.option(
    "--force",
    type=int,
    default=0,
    help="Force store of the model. Increment value (--force=1, --force=2, ...) to force a new store.",
)
@click.option(
    "--wait",
    is_flag=True,
    help="Wait for the model to be stored before returning.",
)
@click.option(
    "--hf-token-key",
    type=str,
    default="HF_TOKEN",
    help=(
        "Name of the Flyte secret containing your HuggingFace token. "
        "Note: This is not the HuggingFace token itself, but the name of the "
        "secret in the Flyte secret store."
    ),
    show_default=True,
)
@click.option(
    "--cpu",
    type=str,
    help="CPU request for the store task (e.g., '2', '4').",
)
@click.option(
    "--mem",
    type=str,
    help="Memory request for the store task (e.g., '16Gi', '64Gi').",
)
@click.option(
    "--ephemeral-storage",
    type=str,
    help="Ephemeral storage request for the store task (e.g., '100Gi', '500Gi').",
)
@click.option(
    "--accelerator",
    type=click.Choice(ACCELERATOR_CHOICES),
    default=None,
    help="The accelerator to use for downloading and (optionally) sharding the model. Format: '{type}:{quantity}' (e.g., 'A100:8', 'L4:1').",
)
@click.option(
    "--shard-config",
    type=click.Path(exists=True, path_type=Path),
    help=(
        "Path to a YAML file containing sharding configuration. "
        "The file should have 'engine' (currently only 'vllm') and 'args' keys."
    ),
)
@click.pass_obj
def hf_model(
    cfg,
    repo: str,
    artifact_name: str | None,
    architecture: str | None,
    task: str,
    modality: tuple[str, ...],
    serial_format: str | None,
    model_type: str | None,
    short_description: str | None,
    force: int,
    wait: bool,
    hf_token_key: str,
    cpu: str | None,
    mem: str | None,
    ephemeral_storage: str | None,
    accelerator: Accelerators | None,
    shard_config: Path | None,
    project: str | None,
    domain: str | None,
):
    """
    Store a HuggingFace model to Flyte storage.

    Downloads a model from the HuggingFace Hub and stores it in your configured
    Flyte storage backend. This is useful for:

    - Pre-storing large models before running inference tasks
    - Sharding models for tensor-parallel inference
    - Avoiding repeated downloads during development

    **Basic Usage:**

    ```bash
    $ flyte store hf-model meta-llama/Llama-2-7b-hf --hf-token-key HF_TOKEN
    ```

    **With Sharding:**

    Create a shard config file (shard_config.yaml):

    ```yaml
    engine: vllm
    args:
      tensor_parallel_size: 8
      dtype: auto
      trust_remote_code: true
    ```

    Then run:

    ```bash
    $ flyte store hf-model meta-llama/Llama-2-70b-hf \\
        --shard-config shard_config.yaml \\
        --accelerator A100:8 \\
        --hf-token-key HF_TOKEN
    ```

    **Wait for Completion:**

    ```bash
    $ flyte store hf-model meta-llama/Llama-2-7b-hf --wait
    ```
    """
    import yaml

    from flyte.store import ShardConfig, VLLMShardArgs, hf_model as store_hf_model
    from flyte.cli._run import initialize_config

    # Initialize flyte config
    cfg = initialize_config(
        cfg.ctx,
        project or cfg.config.task.project,
        domain or cfg.config.task.domain,
    )

    # Parse shard config if provided
    parsed_shard_config = None
    if shard_config is not None:
        with shard_config.open() as f:
            shard_config_dict = yaml.safe_load(f)
            args_dict = shard_config_dict.get("args", {})
            parsed_shard_config = ShardConfig(
                engine=shard_config_dict.get("engine", "vllm"),
                args=VLLMShardArgs(**args_dict),
            )

    console = Console()

    with console.status("[bold green]Starting model store task..."):
        run = store_hf_model(
            repo=repo,
            artifact_name=artifact_name,
            architecture=architecture,
            task=task,
            modality=modality,
            serial_format=serial_format,
            model_type=model_type,
            short_description=short_description,
            shard_config=parsed_shard_config,
            hf_token_key=hf_token_key,
            cpu=cpu,
            mem=mem,
            ephemeral_storage=ephemeral_storage,
            accelerator=accelerator,
            force=force,
        )

    url = run.url
    console.print(
        f"🔄 Started background process to store model from HuggingFace repo [bold]{repo}[/bold].\n"
        f"   Check the console for status at [link={url}]{url}[/link]"
    )

    if wait:
        run.wait()
        try:
            model_path = _get_output_path(run)
            console.print(f"\n✅ Model stored successfully!")
            console.print(f"Remote path: [cyan]{model_path}[/cyan]")
        except Exception as e:
            console.print(f"\n❌ Model store failed!")
            console.print(f"Error: {e}")


@syncify
async def _get_output_path(run: Run) -> str:
    """Get the output path of the run.
    
    NOTE: This is a workaround to get the output path of the run, since the type
    engine can't handle re-hydrating nested Pydantic models in interactive mode.
    """
    from flyte._initialize import get_client
    from flyteidl2.workflow import run_service_pb2

    resp = await get_client().run_service.GetActionData(
        request=run_service_pb2.GetActionDataRequest(
            action_id=(await run.details.aio()).action_details.action_id,
        )
    )
    return resp.outputs.literals[0].value.scalar.blob.uri
