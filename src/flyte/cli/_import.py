"""
CLI commands for importing artifacts from remote registries.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import rich_click as click
from rich.console import Console

from flyte.cli._common import CommandBase


@click.group(name="import")
def import_cmd():
    """
    Import artifacts from remote registries.

    These commands help you download and import artifacts like HuggingFace models
    to your Flyte storage for faster access during task execution.
    """


ACCELERATOR_CHOICES = [
    "nvidia-l4",
    "nvidia-l4-vws",
    "nvidia-l40s",
    "nvidia-a100",
    "nvidia-a100-80gb",
    "nvidia-a10g",
    "nvidia-tesla-k80",
    "nvidia-tesla-m60",
    "nvidia-tesla-p4",
    "nvidia-tesla-p100",
    "nvidia-tesla-t4",
    "nvidia-tesla-v100",
]


@import_cmd.command(name="hf-model", cls=CommandBase)
@click.argument("repo", type=str)
@click.option(
    "--artifact-name",
    type=str,
    required=False,
    default=None,
    help=(
        "Artifact name to use for the imported model. Must only contain alphanumeric characters, "
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
    help="Force import of the model. Increment value (--force=1, --force=2, ...) to force a new import.",
)
@click.option(
    "--wait",
    is_flag=True,
    help="Wait for the model to be imported before returning.",
)
@click.option(
    "--hf-token-key",
    type=str,
    default="HF_TOKEN",
    help="Name of the Flyte secret containing your HuggingFace token.",
    show_default=True,
)
@click.option(
    "--cpu",
    type=str,
    help="CPU request for the import task (e.g., '2', '4').",
)
@click.option(
    "--gpu",
    type=str,
    help="GPU request for the import task (e.g., '1', '8').",
)
@click.option(
    "--mem",
    type=str,
    help="Memory request for the import task (e.g., '16Gi', '64Gi').",
)
@click.option(
    "--ephemeral-storage",
    type=str,
    help="Ephemeral storage request for the import task (e.g., '100Gi', '500Gi').",
)
@click.option(
    "--accelerator",
    type=click.Choice(ACCELERATOR_CHOICES),
    default=None,
    help="The accelerator to use for downloading and (optionally) sharding the model.",
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
    artifact_name: Optional[str],
    architecture: Optional[str],
    task: str,
    modality: Tuple[str, ...],
    serial_format: Optional[str],
    model_type: Optional[str],
    short_description: Optional[str],
    force: int,
    wait: bool,
    hf_token_key: str,
    cpu: Optional[str],
    gpu: Optional[str],
    mem: Optional[str],
    ephemeral_storage: Optional[str],
    accelerator: Optional[str],
    shard_config: Optional[Path],
    project: Optional[str] = None,
    domain: Optional[str] = None,
):
    """
    Import a HuggingFace model to Flyte storage.

    Downloads a model from the HuggingFace Hub and imports it to your configured
    Flyte storage backend. This is useful for:

    - Pre-importing large models before running inference tasks
    - Sharding models for tensor-parallel inference
    - Avoiding repeated downloads during development

    **Basic Usage:**

    ```bash
    $ flyte import hf-model meta-llama/Llama-2-7b-hf --hf-token-key HF_TOKEN
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
    $ flyte import hf-model meta-llama/Llama-2-70b-hf \\
        --shard-config shard_config.yaml \\
        --gpu 8 \\
        --accelerator nvidia-a100 \\
        --hf-token-key HF_TOKEN
    ```

    **Wait for Completion:**

    ```bash
    $ flyte import hf-model meta-llama/Llama-2-7b-hf --wait
    ```
    """
    import yaml

    from flyte.imports import HuggingFaceModelInfo, ShardConfig, VLLMShardArgs, hf_model as import_hf_model
    from flyte.cli._common import initialize_config

    # Initialize flyte config
    initialize_config(cfg.ctx, project or "", domain or "")

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

    with console.status("[bold green]Starting model import task..."):
        run = import_hf_model(
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
            gpu=gpu,
            mem=mem,
            ephemeral_storage=ephemeral_storage,
            accelerator=accelerator,
            project=project,
            domain=domain,
            wait=False,  # We handle waiting ourselves for better UX
            force=force,
        )

    url = run.url
    console.print(
        f"🔄 Started background process to import model from HuggingFace repo [bold]{repo}[/bold].\n"
        f"   Check the console for status at [link={url}]{url}[/link]"
    )

    if wait:
        with console.status("[bold green]Waiting for model to be imported...", spinner="dots"):
            run.wait()

        outputs = run.outputs()
        console.print(f"\n✅ Model imported successfully!")
        console.print(f"   Path: [cyan]{outputs.path}[/cyan]")
