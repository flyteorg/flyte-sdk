"""
This example demonstrates using HuggingFace Storage Buckets as the
`raw_data_path` for storing Files and Dirs. The HF token is read automatically from the
HF_TOKEN environment variable by huggingface_hub's fsspec integration.

For remote execution, inject HF_TOKEN via flyte.Secret so the task pod can
authenticate with HuggingFace Hub.

HF storage buckets work out of the box with Flyte since they support a
[filesystem integration](https://huggingface.co/blog/storage-buckets#filesystem-integration
) that's fsspec-compatible. Flyte defaults to the fsspec protocol when obstore doesn't support
the protocol we need.

## Prerequisites

1. HuggingFace account + token: https://huggingface.co/settings/tokens
2. A writable HF bucket
3. Install: pip install huggingface_hub
4. Set environment variables:
   ```
   export HF_TOKEN=...
   ```
"""

import tempfile

import flyte
from flyte.io import File

env = flyte.TaskEnvironment(
    name="hf-raw-data-example",
    image=flyte.Image.from_debian_base(name="hf-raw-data-example").with_pip_packages("huggingface_hub"),
    secrets=[flyte.Secret(key="hf-token", as_env_var="HF_TOKEN")],
)

USERNAME = "<USERNAME>"
BUCKET = "<BUCKET_NAME>"


@env.task
def write_greeting(name: str) -> File:
    print(flyte.ctx().raw_data_path)
    """Write a greeting to a file stored on HuggingFace Hub via raw_data_path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(f"Hello, {name}! This file is stored on HuggingFace Hub.")
        tmp_path = f.name
    return File.from_local_sync(tmp_path)


@env.task
def read_and_transform(file: File) -> str:
    """Read a file from HuggingFace Hub and transform its content."""
    local_path = file.download_sync()
    with open(local_path, "r") as f:
        text = f.read()
    return text.upper()


@env.task
def pipeline(name: str) -> str:
    """A simple pipeline: write a file to HF Hub, then read it back and transform."""
    file = write_greeting(name)
    result = read_and_transform(file)
    return result


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.with_runcontext(raw_data_path=f"hf://buckets/{USERNAME}/{BUCKET}/raw-data/").run(
        pipeline, name="HuggingFace"
    )

    print(f"Run URL: {run.url}")
    run.wait()

    result = run.outputs()[0]
    print(f"Result: {result}")
